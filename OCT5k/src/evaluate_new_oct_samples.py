import os
import sys
import glob
import time
import json
import yaml
import argparse
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

def get_validation_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def extract_raw_boundaries(pred_mask_2d):
    H, W = pred_mask_2d.shape
    b_names = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
    b_classes = [1, 2, 3, 4, 5]
    
    raw_boundaries = {b: np.full(W, np.nan) for b in b_names}
    
    for x in range(W):
        col = pred_mask_2d[:, x]
        for c, b_name in zip(b_classes, b_names):
            idx = np.where((col[1:] == c) & (col[:-1] == c-1))[0]
            if len(idx) > 0:
                raw_boundaries[b_name][x] = float(idx[0] + 1)
    return raw_boundaries

def interpolate_boundaries(raw_boundaries):
    interp_boundaries = {}
    for b_name, arr in raw_boundaries.items():
        s = pd.Series(arr)
        s_interp = s.interpolate(method='linear', limit_direction='both').values
        if np.isnan(s_interp).all():
            s_interp = np.zeros_like(arr)
        interp_boundaries[b_name] = s_interp
    return interp_boundaries

def calculate_thickness(b_top, b_bottom):
    thick = b_bottom - b_top
    thick = np.maximum(thick, 0)
    return float(np.mean(thick))

def main():
    parser = argparse.ArgumentParser(description="OCT Domain Generalization Evaluation")
    parser.add_argument("--input-root", required=True, type=str, help="Root folder of the new dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_root):
        print(f"ERROR: Input root {args.input_root} does not exist.")
        sys.exit(1)
        
    out_dir = "outputs/new_oct_evaluation"
    rep_dir = os.path.join(out_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    
    categories = ['DME', 'CNV', 'Drusen', 'Normal']
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # 1. Collect Files
    cat_files = {c: [] for c in categories}
    
    print("Scanning input directory...")
    for root, _, files in os.walk(args.input_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                path = os.path.join(root, file)
                path_lower = path.lower()
                
                # Assign to category
                for c in categories:
                    if c.lower() in path_lower:
                        cat_files[c].append(path)
                        break

    # Select deterministic 50
    selected_files = {}
    for c in categories:
        cat_files[c].sort()
        selected_files[c] = cat_files[c][:50]
        print(f"{c}: Found {len(cat_files[c])}, evaluating {len(selected_files[c])}")
        
    total_eval = sum(len(v) for v in selected_files.values())
    if total_eval == 0:
        print("No valid images found.")
        sys.exit(0)
        
    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n--- MODEL SETUP ---")
    print(f"Device: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    config_path = "configs/baseline.yaml"
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Total images evaluating: {total_eval}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=6
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    transform = get_validation_augmentation()
    
    # 3. Processing
    image_results = []
    
    b_colors = {'ILM': 'r', 'OPL': 'g', 'IS-OS': 'b', 'IBRPE': 'c', 'OBRPE': 'm'}
    mask_colors = np.array([
        [0, 0, 0],       
        [255, 0, 0],     
        [0, 255, 0],     
        [0, 0, 255],     
        [255, 255, 0],   
        [255, 0, 255]    
    ], dtype=np.uint8)
    
    # Storage for contact sheets
    contact_sheet_images = {c: [] for c in categories}
    
    print("\nStarting Evaluation...")
    for c in categories:
        cat_out = os.path.join(out_dir, c)
        os.makedirs(cat_out, exist_ok=True)
        
        for file_path in selected_files[c]:
            start_time = time.time()
            filename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(filename)[0]
            
            # Default result structure
            res = {
                'category': c,
                'filename': filename,
                'full_path': file_path,
                'original_width': np.nan,
                'original_height': np.nan,
                'inference_time_seconds': np.nan,
                'predicted_classes': "",
                'ILM_valid_percent': np.nan,
                'OPL_valid_percent': np.nan,
                'ISOS_valid_percent': np.nan,
                'IBRPE_valid_percent': np.nan,
                'OBRPE_valid_percent': np.nan,
                'ILM_OPL_mean_pixels': np.nan,
                'OPL_ISOS_mean_pixels': np.nan,
                'ISOS_IBRPE_mean_pixels': np.nan,
                'IBRPE_OBRPE_mean_pixels': np.nan,
                'Total_Retinal_mean_pixels': np.nan,
                'central_ILM_OPL_mean_pixels': np.nan,
                'central_OPL_ISOS_mean_pixels': np.nan,
                'central_ISOS_IBRPE_mean_pixels': np.nan,
                'central_IBRPE_OBRPE_mean_pixels': np.nan,
                'central_Total_Retinal_mean_pixels': np.nan,
                'quality_flag': False,
                'quality_reason': ""
            }
            
            # Load image
            img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                res['quality_flag'] = True
                res['quality_reason'] = "Image load failed"
                image_results.append(res)
                continue
                
            orig_h, orig_w = img_gray.shape
            res['original_height'] = orig_h
            res['original_width'] = orig_w
            
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            if orig_h != 512 or orig_w != 512:
                img_rgb = cv2.resize(img_rgb, (512, 512))
                
            # Predict
            augmented = transform(image=img_rgb)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = model(img_tensor)
                else:
                    logits = model(img_tensor)
                preds = torch.argmax(logits, dim=1)
                pred_mask = preds[0].cpu().numpy()
                
            inf_time = time.time() - start_time
            res['inference_time_seconds'] = inf_time
            
            unique_classes = np.unique(pred_mask).tolist()
            res['predicted_classes'] = str(unique_classes)
            
            if len(unique_classes) < 6:
                res['quality_flag'] = True
                res['quality_reason'] += f"Missing classes (found {len(unique_classes)}); "
                
            # Boundaries
            raw_boundaries = extract_raw_boundaries(pred_mask)
            interp_boundaries = interpolate_boundaries(raw_boundaries)
            
            for b in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
                valid_pct = (~np.isnan(raw_boundaries[b])).sum() / 512.0 * 100.0
                res[f"{b.replace('-','')}_valid_percent"] = valid_pct
                if valid_pct < 80.0:
                    res['quality_flag'] = True
                    res['quality_reason'] += f"Low {b} validity ({valid_pct:.1f}%); "
                    
            # Thickness Features
            layers = [
                ('ILM_OPL', 'ILM', 'OPL'),
                ('OPL_ISOS', 'OPL', 'IS-OS'),
                ('ISOS_IBRPE', 'IS-OS', 'IBRPE'),
                ('IBRPE_OBRPE', 'IBRPE', 'OBRPE'),
                ('Total_Retinal', 'ILM', 'OBRPE')
            ]
            
            for name, top, bottom in layers:
                val = calculate_thickness(interp_boundaries[top], interp_boundaries[bottom])
                res[f"{name}_mean_pixels"] = val
                
                c_val = calculate_thickness(interp_boundaries[top][128:384], interp_boundaries[bottom][128:384])
                res[f"central_{name}_mean_pixels"] = c_val
                
                if val < 0 or np.isnan(val) or np.isinf(val):
                    res['quality_flag'] = True
                    res['quality_reason'] += f"Invalid {name} thickness; "
                    
            if res['Total_Retinal_mean_pixels'] < 10.0:
                res['quality_flag'] = True
                res['quality_reason'] += "Suspiciously small total thickness; "
                
            # Visualizations
            mask_rgb = mask_colors[pred_mask]
            
            # 1. Original
            cv2.imwrite(os.path.join(cat_out, f"{name_no_ext}_orig.png"), img_rgb)
            # 2. Mask
            cv2.imwrite(os.path.join(cat_out, f"{name_no_ext}_mask.png"), mask_rgb)
            
            # Overlay Boundary
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_rgb)
            x_axis = np.arange(512)
            for b in b_colors:
                ax.plot(x_axis, interp_boundaries[b], color=b_colors[b], linewidth=1)
            ax.axis('off')
            plt.tight_layout()
            bnd_path = os.path.join(cat_out, f"{name_no_ext}_boundaries.png")
            plt.savefig(bnd_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            # Store overlay for contact sheet
            cs_img = cv2.imread(bnd_path)
            cs_img = cv2.cvtColor(cs_img, cv2.COLOR_BGR2RGB)
            contact_sheet_images[c].append(cs_img)
            
            image_results.append(res)
            
    # Master CSV
    df_results = pd.DataFrame(image_results)
    df_results.to_csv(os.path.join(rep_dir, "image_results.csv"), index=False)
    
    # Category Summary
    cat_summ = []
    for c in categories:
        df_c = df_results[df_results['category'] == c]
        if len(df_c) == 0: continue
        
        summ = {
            'category': c,
            'number_of_images': len(df_c),
            'mean_inference_time': df_c['inference_time_seconds'].mean(),
            'mean_ILM_OPL': df_c['ILM_OPL_mean_pixels'].mean(),
            'mean_OPL_ISOS': df_c['OPL_ISOS_mean_pixels'].mean(),
            'mean_ISOS_IBRPE': df_c['ISOS_IBRPE_mean_pixels'].mean(),
            'mean_IBRPE_OBRPE': df_c['IBRPE_OBRPE_mean_pixels'].mean(),
            'mean_total_retinal_thickness': df_c['Total_Retinal_mean_pixels'].mean(),
            'central_mean_total_retinal': df_c['central_Total_Retinal_mean_pixels'].mean(),
            'mean_ILM_valid_percent': df_c['ILM_valid_percent'].mean(),
            'mean_OPL_valid_percent': df_c['OPL_valid_percent'].mean(),
            'mean_ISOS_valid_percent': df_c['ISOS_valid_percent'].mean(),
            'mean_IBRPE_valid_percent': df_c['IBRPE_valid_percent'].mean(),
            'mean_OBRPE_valid_percent': df_c['OBRPE_valid_percent'].mean(),
            'suspicious_percentage': (df_c['quality_flag'].sum() / len(df_c)) * 100.0
        }
        cat_summ.append(summ)
        
    df_summ = pd.DataFrame(cat_summ)
    df_summ.to_csv(os.path.join(rep_dir, "category_summary.csv"), index=False)
    
    # Contact Sheets
    for c in categories:
        imgs = contact_sheet_images[c]
        if not imgs: continue
        n = len(imgs)
        cols = 10
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for i, ax in enumerate(axes):
            if i < n:
                ax.imshow(imgs[i])
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(rep_dir, f"{c}_contact_sheet.png"), dpi=150)
        plt.close(fig)
        
    # Text Report
    report_path = os.path.join(rep_dir, "domain_generalization_report.txt")
    with open(report_path, "w") as f:
        f.write("--------------------------------\n")
        f.write("NEW OCT DOMAIN EVALUATION\n")
        f.write("--------------------------------\n\n")
        
        f.write("Total images evaluated\n\n")
        for c in categories:
            df_c = df_results[df_results['category'] == c]
            f.write(f"{c}:\n{len(df_c)} evaluated\n\n")
            
        f.write("--------------------------------\n")
        f.write("CATEGORY STRUCTURAL SUMMARY\n")
        f.write("--------------------------------\n\n")
        
        for c in categories:
            f.write(f"{c}:\n")
            df_c = df_summ[df_summ['category'] == c]
            if len(df_c) == 0:
                f.write("No images processed.\n\n")
                continue
            row = df_c.iloc[0]
            f.write(f"  mean total retinal thickness: {row['mean_total_retinal_thickness']:.2f} px\n")
            f.write(f"  mean layer thicknesses: ILM-OPL {row['mean_ILM_OPL']:.2f}, OPL-ISOS {row['mean_OPL_ISOS']:.2f}, ISOS-IBRPE {row['mean_ISOS_IBRPE']:.2f}, IBRPE-OBRPE {row['mean_IBRPE_OBRPE']:.2f}\n")
            f.write(f"  boundary validity: ILM {row['mean_ILM_valid_percent']:.1f}%, OPL {row['mean_OPL_valid_percent']:.1f}%, ISOS {row['mean_ISOS_valid_percent']:.1f}%, IBRPE {row['mean_IBRPE_valid_percent']:.1f}%, OBRPE {row['mean_OBRPE_valid_percent']:.1f}%\n")
            f.write(f"  suspicious prediction percentage: {row['suspicious_percentage']:.1f}%\n\n")
            
        f.write("--------------------------------\n")
        f.write("QUALITY ANALYSIS\n")
        f.write("--------------------------------\n\n")
        
        df_susp = df_results[df_results['quality_flag'] == True]
        f.write(f"Number of suspicious images: {len(df_susp)}\n\n")
        if len(df_susp) > 0:
            for _, r in df_susp.iterrows():
                f.write(f"{r['category']} / {r['filename']}: {r['quality_reason']}\n")
        f.write("\n")
        
        f.write("--------------------------------\n")
        f.write("IMPORTANT INTERPRETATION\n")
        f.write("--------------------------------\n\n")
        f.write("1. This is an inference-only experiment.\n")
        f.write("2. The U-Net was NOT retrained.\n")
        f.write("3. Disease/category labels were NOT supplied to the model.\n")
        f.write("4. No disease classification was performed.\n")
        f.write("5. No Dice/IoU can be calculated because these new images have no pixel-level ground-truth segmentation masks.\n")
        f.write("6. The purpose is to determine whether the OCT5k-trained segmentation model generalizes to these new OCT image sources.\n")
        f.write("7. If segmentation quality is visually poor across a category, we will consider domain adaptation/fine-tuning later.\n")
        f.write("8. Do NOT claim that the model detects DME, CNV, Drusen, Alzheimer's disease, dementia, or any other disease.\n")

    print("\nEvaluation complete.")
    print(f"Results, reports, and contact sheets saved to: {out_dir}")

if __name__ == "__main__":
    main()
