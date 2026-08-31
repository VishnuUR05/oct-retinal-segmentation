import os
import sys
import yaml
import torch
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from collections import defaultdict

# Append root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import OCTDataset, get_validation_augmentation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate_ground_truth(df_gt):
    if len(df_gt) != 512:
        return False, f"Expected 512 columns, got {len(df_gt)}"
    
    # Check anatomical ordering
    ilm = df_gt['ILM'].values
    opl = df_gt['OPL'].values
    isos = df_gt['IS-OS'].values
    ibrpe = df_gt['IBRPE'].values
    obrpe = df_gt['OBRPE'].values
    
    violations = np.sum(~((ilm <= opl) & (opl <= isos) & (isos <= ibrpe) & (ibrpe <= obrpe)))
    if violations > 0:
        return False, f"{violations} columns violate anatomical ordering"
        
    return True, "Valid"

def extract_raw_boundaries(pred_mask_2d):
    """
    Extracts RAW boundary positions only where strictly valid anatomical transitions exist.
    Missing transitions are explicitly marked as np.nan.
    """
    H, W = pred_mask_2d.shape
    b_names = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
    b_classes = [1, 2, 3, 4, 5]
    
    raw_boundaries = {b: np.full(W, np.nan) for b in b_names}
    
    for x in range(W):
        col = pred_mask_2d[:, x]
        for c, b_name in zip(b_classes, b_names):
            # Look for exact transition from (c-1) to c
            idx = np.where((col[1:] == c) & (col[:-1] == c-1))[0]
            if len(idx) > 0:
                # Store the y-coordinate of the first interface
                raw_boundaries[b_name][x] = float(idx[0] + 1)
                
    return raw_boundaries

def interpolate_boundaries(raw_boundaries):
    """
    Uses Pandas linear interpolation to bridge NaNs for structural calculations.
    Returns the interpolated dictionary and the number of interpolated pixels per boundary.
    """
    interp_boundaries = {}
    interp_counts = {}
    for b_name, arr in raw_boundaries.items():
        s = pd.Series(arr)
        n_missing = s.isna().sum()
        s_interp = s.interpolate(method='linear', limit_direction='both').values
        
        # If the entire array is NaN (model completely missed a class for the whole image)
        # we have to fallback to zeros just to prevent code crashing, but this is a severe error.
        if np.isnan(s_interp).all():
            s_interp = np.zeros_like(arr)
            
        interp_boundaries[b_name] = s_interp
        interp_counts[b_name] = n_missing
    return interp_boundaries, interp_counts

def calculate_boundary_errors(raw_pred, gt):
    """Calculates MAE, Median AE, RMSE, Max Error ignoring NaNs."""
    metrics = {}
    for b in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
        p = raw_pred[b]
        g = gt[b].values
        
        valid_mask = ~np.isnan(p)
        valid_count = valid_mask.sum()
        total_count = len(p)
        
        if valid_count == 0:
            metrics[b] = {
                'mae': np.nan, 'median_ae': np.nan, 'rmse': np.nan, 'max_ae': np.nan,
                'valid_pct': 0.0, 'missing_pct': 100.0, 'valid_count': 0
            }
            continue
            
        diff = np.abs(p[valid_mask] - g[valid_mask])
        metrics[b] = {
            'mae': float(np.mean(diff)),
            'median_ae': float(np.median(diff)),
            'rmse': float(np.sqrt(np.mean(diff**2))),
            'max_ae': float(np.max(diff)),
            'valid_pct': (valid_count / total_count) * 100.0,
            'missing_pct': ((total_count - valid_count) / total_count) * 100.0,
            'valid_count': valid_count
        }
    return metrics

def calculate_thickness(b_top, b_bottom):
    """Calculates thickness in pixels and returns summary stats."""
    thick = b_bottom - b_top
    # Ensure non-negative thickness mathematically
    thick = np.maximum(thick, 0)
    return {
        'mean': float(np.mean(thick)),
        'median': float(np.median(thick)),
        'min': float(np.min(thick)),
        'max': float(np.max(thick)),
        'std': float(np.std(thick))
    }

def calculate_structural_features(interp_boundaries):
    features = {}
    layers = [
        ('ILM_OPL', 'ILM', 'OPL'),
        ('OPL_ISOS', 'OPL', 'IS-OS'),
        ('ISOS_IBRPE', 'IS-OS', 'IBRPE'),
        ('IBRPE_OBRPE', 'IBRPE', 'OBRPE'),
        ('Total_Retinal', 'ILM', 'OBRPE')
    ]
    
    for name, top, bottom in layers:
        # Full width
        stats_full = calculate_thickness(interp_boundaries[top], interp_boundaries[bottom])
        for k, v in stats_full.items():
            features[f"{name}_{k}_pixels"] = v
            
        # Central window (128 to 383 inclusive)
        stats_central = calculate_thickness(interp_boundaries[top][128:384], interp_boundaries[bottom][128:384])
        for k, v in stats_central.items():
            features[f"central_{name}_{k}_pixels"] = v
            
    return features

def plot_visualizations(image, gt, raw_pred, interp_pred, save_path):
    # Denormalize image
    mean_val = np.array([0.485, 0.456, 0.406])
    std_val = np.array([0.229, 0.224, 0.225])
    image = (image * std_val + mean_val) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    x_axis = np.arange(512)
    colors = {'ILM': 'r', 'OPL': 'g', 'IS-OS': 'b', 'IBRPE': 'c', 'OBRPE': 'm'}
    
    # 1. Original
    axs[0].imshow(image)
    axs[0].set_title("Original OCT")
    axs[0].axis('off')
    
    # 2. GT Boundaries Overlaid
    axs[1].imshow(image)
    for b in colors:
        axs[1].plot(x_axis, gt[b].values, color=colors[b], label=b, linestyle='-', linewidth=2)
    axs[1].set_title("Ground-Truth Boundaries")
    axs[1].legend(loc='upper right', fontsize='small')
    axs[1].axis('off')
    
    # 3. Pred Boundaries Overlaid (Interpolated line + Raw scatter)
    axs[2].imshow(image)
    for b in colors:
        axs[2].plot(x_axis, interp_pred[b], color=colors[b], label=f"{b} (Interp)", linestyle='-', linewidth=1.5, alpha=0.7)
        axs[2].scatter(x_axis, raw_pred[b], color=colors[b], s=2, label=f"{b} (Raw)")
    axs[2].set_title("Predicted Boundaries (Raw & Interp)")
    axs[2].axis('off')
    
    # 4. Error plot (Raw Absolute Error)
    for b in colors:
        err = np.abs(raw_pred[b] - gt[b].values)
        axs[3].plot(x_axis, err, color=colors[b], label=f"{b} Error", alpha=0.8)
    axs[3].set_title("Raw Boundary Error (Pixels)")
    axs[3].set_xlabel("X-coordinate")
    axs[3].set_ylabel("Absolute Error (pixels)")
    axs[3].set_ylim(0, 50) # Cap at 50 for visualization scale
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def main():
    print("="*60)
    print("OCT5k BOUNDARY EXTRACTION & STRUCTURAL FEATURES")
    print("="*60)
    
    config_path = "configs/baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_path = os.path.join(config['paths']['outputs_dir'], "checkpoints", "best_model.pth")
    test_csv_path = config['paths']['test_csv']
    df_test = pd.read_csv(test_csv_path)
    
    if len(df_test) != 296:
        print(f"ERROR: Expected 296 test images, found {len(df_test)}")
        sys.exit(1)
        
    # Create Output Directories
    out_root = os.path.join(config['paths']['outputs_dir'], "boundary_analysis")
    dirs = {
        'predicted': os.path.join(out_root, "predicted_boundaries"),
        'vis': os.path.join(out_root, "visualizations"),
        'reports': os.path.join(out_root, "reports")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # Dataset and Model
    test_dataset = OCTDataset(test_csv_path, transform=get_validation_augmentation())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    ).to(device)
    
    print("Loading best checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Tracking Lists
    metrics_records = []
    thickness_records = []
    
    vis_indices = set(random.sample(range(len(df_test)), 10))
    
    print("Starting Boundary Extraction & Evaluation...")
    idx = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            
            if config['training']['mixed_precision'] and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for b in range(images.size(0)):
                row_meta = df_test.iloc[idx]
                sample_id = row_meta['sample_id']
                e2e_id = row_meta['e2e_group_id']
                cat = row_meta['category']
                
                # Load Ground Truth
                dataset_root = get_dataset_root()
                df_gt = pd.read_csv(str(dataset_root / str(row_meta['boundary_path'])))
                is_valid, msg = validate_ground_truth(df_gt)
                if not is_valid:
                    print(f"\nWARNING: GT invalid for {sample_id} ({msg})")
                
                # 1. Raw Boundaries
                raw_pred = extract_raw_boundaries(preds[b])
                
                # 2. Interpolated Boundaries
                interp_pred, interp_counts = interpolate_boundaries(raw_pred)
                
                # Save predicted CSV
                df_pred = pd.DataFrame(interp_pred)
                df_pred.insert(0, 'x', np.arange(512))
                df_pred.to_csv(os.path.join(dirs['predicted'], f"{sample_id}.csv"), index=False)
                
                # 3. Calculate Boundary MAE (using RAW)
                err_metrics = calculate_boundary_errors(raw_pred, df_gt)
                
                record_met = {'sample_id': sample_id, 'e2e_group_id': e2e_id, 'category': cat}
                for bn, m in err_metrics.items():
                    record_met[f'{bn}_MAE'] = m['mae']
                    record_met[f'{bn}_Median_AE'] = m['median_ae']
                    record_met[f'{bn}_RMSE'] = m['rmse']
                    record_met[f'{bn}_Max_AE'] = m['max_ae']
                    record_met[f'{bn}_Valid_Pct'] = m['valid_pct']
                    record_met[f'{bn}_Missing_Pct'] = m['missing_pct']
                metrics_records.append(record_met)
                
                # 4. Calculate Structural Features (using INTERPOLATED)
                # First for Predictions
                feat_pred = calculate_structural_features(interp_pred)
                # Then for Ground Truth (convert to numpy arrays for element-wise math)
                feat_gt = calculate_structural_features({col: df_gt[col].values for col in df_gt.columns})
                
                record_thick = {'sample_id': sample_id, 'e2e_group_id': e2e_id, 'category': cat}
                for k, v in feat_pred.items():
                    record_thick[f'PRED_{k}'] = v
                for k, v in feat_gt.items():
                    record_thick[f'GT_{k}'] = v
                thickness_records.append(record_thick)
                
                # 5. Visualizations
                if idx in vis_indices:
                    img_np = images[b].cpu().permute(1, 2, 0).numpy()
                    save_path = os.path.join(dirs['vis'], f"{sample_id}_boundary_vis.png")
                    plot_visualizations(img_np, df_gt, raw_pred, interp_pred, save_path)
                    
                idx += 1
                
    # Save CSVs
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(os.path.join(dirs['reports'], "boundary_metrics.csv"), index=False)
    
    df_thickness = pd.DataFrame(thickness_records)
    df_thickness.to_csv(os.path.join(dirs['reports'], "retinal_structural_features.csv"), index=False)
    
    # -----------------------------------------------------
    # E2E Aggregation & Final Report Generation
    # -----------------------------------------------------
    report_path = os.path.join(dirs['reports'], "boundary_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("OCT5k BOUNDARY & STRUCTURAL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Global Image-Level Stats
        f.write("--- GLOBAL IMAGE-LEVEL RAW BOUNDARY METRICS (across 296 B-scans) ---\n")
        for b in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
            mae = df_metrics[f'{b}_MAE'].mean()
            val_pct = df_metrics[f'{b}_Valid_Pct'].mean()
            f.write(f"{b:6s} | Valid: {val_pct:6.2f}% | MAE: {mae:6.2f} px\n")
            
        f.write("\n--- E2E GROUP LEVEL METRICS ---\n")
        e2e_group_metrics = []
        for e2e, group_df in df_metrics.groupby('e2e_group_id'):
            # Mean across the images in this group
            img_count = len(group_df)
            group_maes = {b: group_df[f'{b}_MAE'].mean() for b in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']}
            
            f.write(f"\nGroup: {e2e} (Images: {img_count})\n")
            for b, m in group_maes.items():
                f.write(f"  {b:6s} MAE: {m:.2f} px\n")
            
            e2e_group_metrics.append(group_maes)
            
        # Macro-Averages (Mean of the 9 group means)
        f.write("\n--- E2E MACRO-AVERAGE (Mean of Means) ---\n")
        df_e2e = pd.DataFrame(e2e_group_metrics)
        for b in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
            macro_mae = df_e2e[b].mean()
            f.write(f"{b:6s} Macro-MAE: {macro_mae:.2f} px\n")
            
    print(f"\nSuccessfully processed {idx} test images.")
    print(f"Results saved to {dirs['reports']}")
    print(f"Visualizations saved to {dirs['vis']}")

if __name__ == "__main__":
    main()
