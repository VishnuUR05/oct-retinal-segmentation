import os
import sys
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pathlib import Path
import segmentation_models_pytorch as smp

base_dir = r"D:\AIT Major Project\oct rertinal segmentation\OCT5k"
sys.path.insert(0, os.path.join(base_dir, "src"))
from dataset import get_validation_augmentation

# ---------------------------------------------------------
# CONSTANTS & CONFIG
# ---------------------------------------------------------
B_NAMES = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
B_CLASSES = [1, 2, 3, 4, 5]
CENTRAL_START = 128
CENTRAL_END = 384  # exclusive, meaning 128 through 383 (256 columns)
WIDTH = 512

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def get_dataset_root():
    return Path(base_dir) / "OCT5k dataset"

def extract_raw_boundaries(pred_mask_2d):
    """Extract RAW boundary positions (Class c-1 to c). Missing = np.nan."""
    H, W = pred_mask_2d.shape
    raw_boundaries = {b: np.full(W, np.nan) for b in B_NAMES}
    
    for x in range(W):
        col = pred_mask_2d[:, x]
        for c, b_name in zip(B_CLASSES, B_NAMES):
            idx = np.where((col[1:] == c) & (col[:-1] == c-1))[0]
            if len(idx) > 0:
                raw_boundaries[b_name][x] = float(idx[0] + 1)
    return raw_boundaries

def validate_boundaries(raw_boundaries):
    """Check ordering: ILM < OPL < IS-OS < IBRPE < OBRPE on valid columns."""
    ilm = raw_boundaries['ILM']
    opl = raw_boundaries['OPL']
    isos = raw_boundaries['IS-OS']
    ibrpe = raw_boundaries['IBRPE']
    obrpe = raw_boundaries['OBRPE']
    
    valid_mask = ~(np.isnan(ilm) | np.isnan(opl) | np.isnan(isos) | np.isnan(ibrpe) | np.isnan(obrpe))
    if valid_mask.sum() > 0:
        v_ilm, v_opl, v_isos, v_ibrpe, v_obrpe = ilm[valid_mask], opl[valid_mask], isos[valid_mask], ibrpe[valid_mask], obrpe[valid_mask]
        violations = np.sum(~((v_ilm <= v_opl) & (v_opl <= v_isos) & (v_isos <= v_ibrpe) & (v_ibrpe <= v_obrpe)))
        return violations == 0, violations
    return True, 0

def interpolate_boundaries(raw_boundaries):
    """Interpolate sparse NaNs. Entirely missing boundaries remain NaN."""
    interp_boundaries = {}
    stats = {}
    
    for b_name, arr in raw_boundaries.items():
        s = pd.Series(arr)
        n_missing = s.isna().sum()
        pct_missing = (n_missing / WIDTH) * 100.0
        
        qc = "VALID"
        if pct_missing > 50:
            qc = "INVALID"
        elif pct_missing > 20:
            qc = "UNCERTAIN"
            
        s_interp = s.interpolate(method='linear', limit_direction='both').values
        
        # If all NaN, keep it NaN! No zero fallback.
        if np.isnan(s_interp).all():
            s_interp = np.full(WIDTH, np.nan)
            
        interp_boundaries[b_name] = s_interp
        stats[b_name] = {
            'nan_percent': pct_missing,
            'interpolated_count': n_missing if not np.isnan(s_interp).all() else 0,
            'remaining_invalid': np.isnan(s_interp).sum(),
            'qc_status': qc
        }
    
    return interp_boundaries, stats

def calculate_thickness(interp_boundaries):
    thick = {
        'ILM_OPL': interp_boundaries['OPL'] - interp_boundaries['ILM'],
        'OPL_ISOS': interp_boundaries['IS-OS'] - interp_boundaries['OPL'],
        'ISOS_IBRPE': interp_boundaries['IBRPE'] - interp_boundaries['IS-OS'],
        'IBRPE_OBRPE': interp_boundaries['OBRPE'] - interp_boundaries['IBRPE'],
        'Total_Retinal': interp_boundaries['OBRPE'] - interp_boundaries['ILM']
    }
    # Ensure no negative thickness from interpolation artifacts
    for k in thick:
        thick[k][thick[k] < 0] = np.nan
    return thick

def calculate_statistics(thick_dict, prefix=""):
    stats = {}
    for layer, arr in thick_dict.items():
        if np.isnan(arr).all():
            stats[f'{prefix}{layer.lower()}_mean_px'] = np.nan
            stats[f'{prefix}{layer.lower()}_median_px'] = np.nan
            stats[f'{prefix}{layer.lower()}_std_px'] = np.nan
        else:
            stats[f'{prefix}{layer.lower()}_mean_px'] = np.nanmean(arr)
            stats[f'{prefix}{layer.lower()}_median_px'] = np.nanmedian(arr)
            stats[f'{prefix}{layer.lower()}_std_px'] = np.nanstd(arr)
    return stats

def validate_against_ground_truth(pred_boundaries, gt_df):
    errors = {}
    for b in B_NAMES:
        p = pred_boundaries[b]
        g = gt_df[b].values
        valid_mask = ~(np.isnan(p) | np.isnan(g))
        if valid_mask.sum() > 0:
            diff = np.abs(p[valid_mask] - g[valid_mask])
            errors[f'{b}_mae'] = np.mean(diff)
            errors[f'{b}_med_ae'] = np.median(diff)
            errors[f'{b}_95th_ae'] = np.percentile(diff, 95)
        else:
            errors[f'{b}_mae'] = np.nan
            errors[f'{b}_med_ae'] = np.nan
            errors[f'{b}_95th_ae'] = np.nan
    return errors

def generate_visualization(img_rgb, pred_mask, boundaries, gt_df, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    axs[1].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=5)
    axs[1].set_title("Predicted Segmentation")
    axs[1].axis('off')
    
    axs[2].imshow(img_rgb)
    colors = {'ILM': 'r', 'OPL': 'g', 'IS-OS': 'b', 'IBRPE': 'c', 'OBRPE': 'm'}
    x = np.arange(WIDTH)
    for b in B_NAMES:
        axs[2].plot(x, boundaries[b], color=colors[b], label=b, linewidth=1)
    axs[2].set_title("Predicted Boundaries")
    axs[2].legend(fontsize='small', loc='upper right')
    axs[2].axis('off')
    
    axs[3].imshow(img_rgb)
    for b in B_NAMES:
        axs[3].plot(x, gt_df[b], color=colors[b], linestyle='--', label=f'GT {b}', linewidth=1)
    axs[3].set_title("Ground Truth Boundaries")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("OCT STRUCTURAL FEATURE EXTRACTION — PHASE 2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_root = get_dataset_root()
    df_test = pd.read_csv(os.path.join(base_dir, "splits", "test.csv"))
    
    out_dir = Path(base_dir) / "outputs" / "structural_features"
    coords_dir = out_dir / "boundary_coordinates"
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=6)
    ckpt_path = os.path.join(base_dir, "outputs", "checkpoints", "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    transform = get_validation_augmentation()
    
    bscan_features = []
    error_records = []
    
    # Pick some samples for visualization
    random.seed(42)
    vis_samples = set(df_test.groupby('category').sample(n=2, random_state=42)['sample_id'])
    
    print(f"Processing {len(df_test)} test B-scans...")
    
    for idx, row in df_test.iterrows():
        sample_id = row['sample_id']
        e2e_id = row['e2e_group_id']
        cat = row['category']
        
        img_path = dataset_root / str(row['image_path']).replace('\\', '/')
        gt_path = dataset_root / str(row['boundary_path']).replace('\\', '/')
        
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_df = pd.read_csv(str(gt_path))
        
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        transformed = transform(image=img_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
        preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
        # 1. Raw Boundaries
        raw_b = extract_raw_boundaries(preds)
        
        # 2. QC & Interpolation
        interp_b, b_stats = interpolate_boundaries(raw_b)
        valid_order, order_violations = validate_boundaries(interp_b)
        
        overall_qc = "VALID"
        if any(v['qc_status'] == "INVALID" for v in b_stats.values()) or not valid_order:
            overall_qc = "INVALID"
        elif any(v['qc_status'] == "UNCERTAIN" for v in b_stats.values()):
            overall_qc = "UNCERTAIN"
            
        # 3. Thickness
        thick_global = calculate_thickness(interp_b)
        thick_central = {k: v[CENTRAL_START:CENTRAL_END] for k, v in thick_global.items()}
        
        # 4. Features
        feat_global = calculate_statistics(thick_global, prefix="")
        feat_central = calculate_statistics(thick_central, prefix="central_")
        
        # 5. Compile B-scan Row
        row_feat = {
            'sample_id': sample_id,
            'e2e_group_id': e2e_id,
            'category': cat,
            'overall_qc_status': overall_qc,
            'order_violations': order_violations
        }
        for b in B_NAMES:
            row_feat[f'{b}_qc_status'] = b_stats[b]['qc_status']
            row_feat[f'{b}_nan_percent'] = b_stats[b]['nan_percent']
            row_feat[f'{b}_interpolated_count'] = b_stats[b]['interpolated_count']
        
        row_feat.update(feat_global)
        row_feat.update(feat_central)
        bscan_features.append(row_feat)
        
        # 6. Errors vs GT
        err = validate_against_ground_truth(interp_b, gt_df)
        err['sample_id'] = sample_id
        error_records.append(err)
        
        # 7. Save Coordinates
        df_coords = pd.DataFrame(interp_b)
        df_coords.insert(0, 'x', np.arange(WIDTH))
        safe_name = str(sample_id).replace('/', '_').replace('\\', '_')
        df_coords.to_csv(coords_dir / f"{safe_name}.csv", index=False)
        
        # 8. Vis
        if sample_id in vis_samples:
            generate_visualization(img_rgb, preds, interp_b, gt_df, vis_dir / f"{safe_name}_vis.png")
            
    # Save B-scan features
    df_bscan = pd.DataFrame(bscan_features)
    df_bscan.to_csv(out_dir / "bscan_features.csv", index=False)
    
    # E2E Aggregation
    e2e_features = []
    for e2e_id, group in df_bscan.groupby('e2e_group_id'):
        stats = {'e2e_group_id': e2e_id, 'number_of_bscans': len(group)}
        stats['valid_bscan_count'] = (group['overall_qc_status'] == 'VALID').sum()
        stats['uncertain_bscan_count'] = (group['overall_qc_status'] == 'UNCERTAIN').sum()
        stats['invalid_bscan_count'] = (group['overall_qc_status'] == 'INVALID').sum()
        
        num_cols = [c for c in group.columns if c.endswith('_px')]
        for c in num_cols:
            stats[f'{c}_mean'] = group[c].mean()
            stats[f'{c}_median'] = group[c].median()
            stats[f'{c}_std'] = group[c].std()
            
        e2e_features.append(stats)
        
    df_e2e = pd.DataFrame(e2e_features)
    df_e2e.to_csv(out_dir / "e2e_features.csv", index=False)
    
    # Quality Report
    df_err = pd.DataFrame(error_records)
    with open(out_dir / "structural_feature_extraction_report.txt", "w", encoding="utf-8") as f:
        f.write("OCT STRUCTURAL FEATURE EXTRACTION REPORT\n")
        f.write("========================================\n\n")
        
        f.write("Note: Physical thickness is not reported because the available dataset does not provide verified optical pixel calibration.\n\n")
        
        f.write("1. PROCESSING SUMMARY\n")
        f.write(f"Number of B-scans processed: {len(df_bscan)}\n")
        f.write(f"Valid B-scans: {(df_bscan['overall_qc_status'] == 'VALID').sum()}\n")
        f.write(f"Uncertain B-scans: {(df_bscan['overall_qc_status'] == 'UNCERTAIN').sum()}\n")
        f.write(f"Invalid B-scans: {(df_bscan['overall_qc_status'] == 'INVALID').sum()}\n")
        f.write(f"E2E groups processed: {len(df_e2e)}\n\n")
        
        f.write("2. BOUNDARY QC SUMMARY\n")
        for b in B_NAMES:
            mean_nan = df_bscan[f'{b}_nan_percent'].mean()
            mean_interp = df_bscan[f'{b}_interpolated_count'].mean()
            f.write(f"{b} - Mean Missing: {mean_nan:.2f}%, Mean Interpolated Cols: {mean_interp:.2f}\n")
            
        f.write("\n3. BOUNDARY METRICS vs GROUND TRUTH (Pixels)\n")
        for b in B_NAMES:
            mae = df_err[f'{b}_mae'].mean()
            med_ae = df_err[f'{b}_med_ae'].mean()
            p95_ae = df_err[f'{b}_95th_ae'].mean()
            f.write(f"{b} - MAE: {mae:.2f}, Median AE: {med_ae:.2f}, 95th Pctl AE: {p95_ae:.2f}\n")
            
        f.write("\n4. THICKNESS FEATURE STATISTICS (Global Pixels)\n")
        layers = ['ilm_opl', 'opl_isos', 'isos_ibrpe', 'ibrpe_obrpe', 'total_retinal']
        for layer in layers:
            m = df_bscan[f'{layer}_mean_px'].mean()
            f.write(f"{layer.upper()} Global Mean Thickness: {m:.2f} px\n")
            
        f.write("\n5. CENTRAL-REGION STATISTICS (Central 50% Pixels)\n")
        for layer in layers:
            cm = df_bscan[f'central_{layer}_mean_px'].mean()
            f.write(f"{layer.upper()} Central Mean Thickness: {cm:.2f} px\n")
            
        f.write("\n6. SCIENTIFIC LIMITATIONS\n")
        f.write("- No micrometer conversions are possible without dataset optical metadata.\n")
        f.write("- Structural features are NOT disease classifications.\n")
        f.write("- Uncertain boundary locations propagate into thickness noise.\n")

if __name__ == "__main__":
    main()
