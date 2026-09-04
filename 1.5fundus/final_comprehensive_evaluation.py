import os
import sys
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
vessel_module_src = os.path.join(repo_root, "fundus", "vessel_module", "src")
if vessel_module_src not in sys.path:
    sys.path.append(vessel_module_src)

from config import *
from data_loading import FIVESPatchDataset
from model_unet import ResNet34UNet
from biomarkers import extract_all_biomarkers

def calc_metrics(pred, gt):
    eps = 1e-6
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()
    
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    
    return {
        'dice': dice, 'iou': iou, 'precision': precision, 
        'recall': recall, 'specificity': specificity, 'accuracy': accuracy
    }

def save_qualitative_result(img_tensor, mask_tensor, pred_tensor, save_path, title_prefix=""):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_unnorm = img_tensor.cpu() * std + mean
    img_unnorm = torch.clamp(img_unnorm, 0, 1).permute(1, 2, 0).numpy()
    img_rgb = (img_unnorm * 255).astype(np.uint8)
    
    gt_mask = mask_tensor.squeeze().cpu().numpy()
    pred_prob = pred_tensor.squeeze().cpu().numpy()
    pred_mask = (pred_prob > 0.5).astype(np.float32)
    
    overlay = img_rgb.copy()
    green_layer = np.zeros_like(overlay)
    green_layer[:, :, 1] = 255
    
    alpha = 0.5
    overlay_mask = pred_mask > 0
    overlay[overlay_mask] = cv2.addWeighted(overlay[overlay_mask], 1 - alpha, green_layer[overlay_mask], alpha, 0)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"{title_prefix} - Original")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(pred_prob, cmap='inferno')
    axes[2].set_title("Probability Map")
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def evaluate():
    output_dir = os.path.join(repo_root, "fundus", "outputs", "fives_vessel_project", "final_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet34UNet(num_classes=1)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_dataset = FIVESPatchDataset(VAL_IMG_DIR, VAL_MASK_DIR, is_train=False)
    # Use batch_size=1 so we can easily track per-patch stats and biomarkers
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    
    per_patch_results = []
    threshold_results = {t: {'dice':[], 'iou':[], 'precision':[], 'recall':[]} for t in thresholds}
    biomarker_results = []
    
    saved_images_data = [] # Store tuple (idx, img, gt, prob, dice)
    
    print("Running comprehensive evaluation...")
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(val_loader)):
            img_tensor = image.to(device)
            mask_tensor = mask.to(device)
            
            logits = model(img_tensor)
            prob_map = torch.sigmoid(logits)
            
            gt_np = mask_tensor.squeeze().cpu().numpy()
            prob_np = prob_map.squeeze().cpu().numpy()
            
            # 1. Threshold analysis
            for t in thresholds:
                pred_t = prob_np > t
                mets = calc_metrics(pred_t, gt_np)
                threshold_results[t]['dice'].append(mets['dice'])
                threshold_results[t]['iou'].append(mets['iou'])
                threshold_results[t]['precision'].append(mets['precision'])
                threshold_results[t]['recall'].append(mets['recall'])
                
            # 2. Main performance at 0.50
            pred_050 = prob_np > 0.50
            main_mets = calc_metrics(pred_050, gt_np)
            main_mets['patch_idx'] = i
            per_patch_results.append(main_mets)
            
            # Save for qualitative sorting
            # We don't want to keep ALL tensors in RAM, but keeping 1920 (3x512x512) is ~1.5GB. We will subsample.
            # Only keep a random subset or skip saving if it's too much memory.
            # Actually, to find exact best/worst, we must do it AFTER computing all dice scores.
            # So let's just save indices and re-run inference on the chosen ones later!
            
            # 3. Biomarker extraction (only every 20th patch to save time, because skeletonization is slow)
            if i % 20 == 0:
                # Need FOV mask (for patches we assume full patch is FOV for simplicity, or we can just pass ones)
                fov = np.ones_like(gt_np, dtype=np.uint8) * 255
                
                gt_mask_uint8 = (gt_np * 255).astype(np.uint8)
                pred_mask_uint8 = (pred_050 * 255).astype(np.uint8)
                
                try:
                    bm_gt = extract_all_biomarkers(gt_mask_uint8, fov)
                    bm_pred = extract_all_biomarkers(pred_mask_uint8, fov)
                    
                    biomarker_results.append({
                        'patch_idx': i,
                        'dice': main_mets['dice'],
                        'gt_density': bm_gt['vessel_density'],
                        'pred_density': bm_pred['vessel_density'],
                        'gt_length': bm_gt['total_length'],
                        'pred_length': bm_pred['total_length'],
                        'gt_width_mean': bm_gt['mean_width'],
                        'pred_width_mean': bm_pred['mean_width'],
                        'gt_tortuosity': bm_gt['mean_tortuosity'],
                        'pred_tortuosity': bm_pred['mean_tortuosity'],
                        'gt_branches': bm_gt['branch_points'],
                        'pred_branches': bm_pred['branch_points']
                    })
                except Exception as e:
                    pass
                    
    # Process main metrics
    df_patch = pd.DataFrame(per_patch_results)
    df_patch.to_csv(os.path.join(output_dir, "per_patch_metrics.csv"), index=False)
    
    stats = df_patch.describe().T
    stats.to_csv(os.path.join(output_dir, "overall_metrics_summary.csv"))
    
    # Process thresholds
    thresh_summary = []
    for t in thresholds:
        thresh_summary.append({
            'threshold': t,
            'mean_dice': np.mean(threshold_results[t]['dice']),
            'mean_iou': np.mean(threshold_results[t]['iou']),
            'mean_precision': np.mean(threshold_results[t]['precision']),
            'mean_recall': np.mean(threshold_results[t]['recall'])
        })
    df_thresh = pd.DataFrame(thresh_summary)
    df_thresh.to_csv(os.path.join(output_dir, "threshold_analysis.csv"), index=False)
    
    # Process biomarkers
    df_bio = pd.DataFrame(biomarker_results)
    df_bio.to_csv(os.path.join(output_dir, "biomarker_validation.csv"), index=False)
    
    # Qualitative visual generation
    df_patch = df_patch.sort_values(by='dice', ascending=False)
    best_indices = df_patch.head(5)['patch_idx'].tolist()
    worst_indices = df_patch.tail(5)['patch_idx'].tolist()
    
    # Average indices (median)
    median_dice = df_patch['dice'].median()
    df_patch['dist_to_med'] = (df_patch['dice'] - median_dice).abs()
    avg_indices = df_patch.sort_values(by='dist_to_med').head(5)['patch_idx'].tolist()
    
    target_indices = best_indices + avg_indices + worst_indices
    
    print("Generating qualitative visuals...")
    for i in target_indices:
        image, mask = val_dataset[i]
        img_tensor = image.unsqueeze(0).to(device)
        mask_tensor = mask.unsqueeze(0).to(device)
        logits = model(img_tensor)
        prob_map = torch.sigmoid(logits)
        
        if i in best_indices:
            prefix = "BEST"
        elif i in worst_indices:
            prefix = "WORST"
        else:
            prefix = "AVERAGE"
            
        save_path = os.path.join(output_dir, f"qualitative_{prefix}_{i}.png")
        save_qualitative_result(image, mask, prob_map, save_path, title_prefix=f"{prefix} (Dice: {df_patch[df_patch['patch_idx']==i]['dice'].values[0]:.4f})")

    print("Evaluation completed successfully.")

if __name__ == "__main__":
    evaluate()
