import os
import sys
import json
import yaml
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        stats_full = calculate_thickness(interp_boundaries[top], interp_boundaries[bottom])
        for k, v in stats_full.items():
            features[f"{name}_{k}_pixels"] = v
            
        stats_central = calculate_thickness(interp_boundaries[top][128:384], interp_boundaries[bottom][128:384])
        for k, v in stats_central.items():
            features[f"central_{name}_{k}_pixels"] = v
            
    return features

def main():
    parser = argparse.ArgumentParser(description="Standalone OCT Inference")
    parser.add_argument("--image", required=True, type=str, help="Path to input OCT PNG image")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"ERROR: Image not found at {args.image}")
        sys.exit(1)
        
    config_path = "configs/baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    out_dir = "outputs/demo"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load and Preprocess Image
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("ERROR: Could not read image.")
        sys.exit(1)
        
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    print(f"Input image: {args.image}")
    print(f"Image shape: {img_rgb.shape}")
    
    transform = get_validation_augmentation()
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(device)  # [1, 3, 512, 512]
    
    # Load Model
    print("\nModel:\nU-Net + ResNet34")
    print(f"Device:\n{device.type.upper()}")
    
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    ).to(device)
    
    checkpoint_path = os.path.join(config['paths']['outputs_dir'], "checkpoints", "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Inference
    with torch.no_grad():
        if config['training']['mixed_precision'] and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(img_tensor)
        else:
            logits = model(img_tensor)
            
        preds = torch.argmax(logits, dim=1) # [1, 512, 512]
        pred_mask = preds[0].cpu().numpy()
        
    print(f"\nPrediction shape:\n{list(logits.shape)}")
    unique_classes = np.unique(pred_mask).tolist()
    print(f"\nPredicted classes:\n{unique_classes}")
    
    # Boundaries
    raw_boundaries = extract_raw_boundaries(pred_mask)
    interp_boundaries = interpolate_boundaries(raw_boundaries)
    
    print("\nBoundary extraction:")
    for b in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
        valid_pct = (~np.isnan(raw_boundaries[b])).sum() / 512.0 * 100.0
        print(f"{b}: {valid_pct:.2f}%")
        
    # Features
    features = calculate_structural_features(interp_boundaries)
    print("\nStructural features:")
    for k, v in features.items():
        if 'mean' in k:
            print(f"{k}: {v:.2f} px")
            
    # Save Outputs
    # 1. Original
    cv2.imwrite(os.path.join(out_dir, "original_oct.png"), img_gray)
    
    # 2. Raw Predicted Mask
    cv2.imwrite(os.path.join(out_dir, "predicted_mask.png"), pred_mask.astype(np.uint8))
    
    # 3. Predicted Boundary CSV (Interpolated so it's a complete profile)
    df_pred = pd.DataFrame(interp_boundaries)
    df_pred.insert(0, 'x', np.arange(512))
    df_pred.to_csv(os.path.join(out_dir, "predicted_boundaries.csv"), index=False)
    
    # 4. JSON Report
    with open(os.path.join(out_dir, "structural_features.json"), 'w') as f:
        json.dump(features, f, indent=4)
        
    # Visualizations
    b_colors = {'ILM': 'r', 'OPL': 'g', 'IS-OS': 'b', 'IBRPE': 'c', 'OBRPE': 'm'}
    x_axis = np.arange(512)
    
    # Mask coloring
    mask_colors = [
        [0, 0, 0],       # 0
        [255, 0, 0],     # 1
        [0, 255, 0],     # 2
        [0, 0, 255],     # 3
        [255, 255, 0],   # 4
        [255, 0, 255]    # 5
    ]
    mask_rgb = np.zeros_like(img_rgb)
    for c in range(6):
        mask_rgb[pred_mask == c] = mask_colors[c]
        
    # 5. Boundary Overlay
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)
    for b in b_colors:
        ax.plot(x_axis, interp_boundaries[b], color=b_colors[b], label=b, linewidth=1.5)
        ax.scatter(x_axis, raw_boundaries[b], color=b_colors[b], s=1)
    ax.set_title("Predicted Retinal Boundaries")
    ax.legend(loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boundary_overlay.png"))
    plt.close(fig)
    
    # 6. Final Visualization (Original + Mask + Boundaries)
    overlay_mask = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original OCT")
    axs[0].axis('off')
    
    axs[1].imshow(overlay_mask)
    axs[1].set_title("Segmentation Mask")
    axs[1].axis('off')
    
    axs[2].imshow(img_rgb)
    for b in b_colors:
        axs[2].plot(x_axis, interp_boundaries[b], color=b_colors[b], linewidth=1.5)
    axs[2].set_title("Predicted Boundaries")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_prediction.png"))
    plt.close(fig)
    
    print(f"\nAll outputs saved to {out_dir}/")

if __name__ == "__main__":
    main()
