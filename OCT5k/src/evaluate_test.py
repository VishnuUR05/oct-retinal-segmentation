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
from src.metrics import calculate_dice_iou

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_overlay(image, true_mask, pred_mask, save_path):
    # image: (C, H, W) normalized float, we need to denormalize and HWC
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Denormalize
    image = (image * std + mean) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Colors for 6 classes
    colors = [
        [0, 0, 0],       # 0: BG above ILM
        [255, 0, 0],     # 1: ILM-OPL (Red)
        [0, 255, 0],     # 2: OPL-IS/OS (Green)
        [0, 0, 255],     # 3: IS/OS-IBRPE (Blue)
        [255, 255, 0],   # 4: IBRPE-OBRPE (Yellow)
        [255, 0, 255]    # 5: BG below OBRPE (Magenta)
    ]
    
    true_colored = np.zeros_like(image)
    pred_colored = np.zeros_like(image)
    
    for c in range(6):
        true_colored[true_mask == c] = colors[c]
        pred_colored[pred_mask == c] = colors[c]
        
    # Overlays
    overlay_pred = cv2.addWeighted(image, 0.7, pred_colored, 0.3, 0)
    
    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis("off")
    
    axs[1].imshow(true_colored)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")
    
    axs[2].imshow(pred_colored)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")
    
    axs[3].imshow(overlay_pred)
    axs[3].set_title("Prediction Overlay")
    axs[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    print("="*50)
    print("OCT5k FINAL TEST EVALUATION")
    print("="*50)
    
    config_path = "configs/baseline.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Checkpoints and outputs
    checkpoint_path = os.path.join(config['paths']['outputs_dir'], "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    test_csv_path = config['paths']['test_csv']
    df_test = pd.read_csv(test_csv_path)
    
    # Validations
    if len(df_test) != 296:
        print(f"ERROR: Expected 296 test images, found {len(df_test)}")
        sys.exit(1)
        
    unique_e2e = df_test['e2e_group_id'].nunique()
    if unique_e2e != 9:
        print(f"ERROR: Expected 9 unique E2E groups, found {unique_e2e}")
        sys.exit(1)
        
    print(f"Verified: {len(df_test)} test images across {unique_e2e} E2E groups.")
    
    reports_dir = os.path.join(config['paths']['outputs_dir'], "reports")
    preds_dir = os.path.join(config['paths']['outputs_dir'], "test_predictions")
    vis_dir = os.path.join(config['paths']['outputs_dir'], "test_visualizations")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Dataset and Loader (shuffle MUST be False for correct index matching)
    test_dataset = OCTDataset(test_csv_path, transform=get_validation_augmentation())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Model
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None, # Not needed for inference
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    ).to(device)
    
    print("Loading best checkpoint weights...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Metrics tracking
    all_dice = []
    all_iou = []
    e2e_metrics = defaultdict(lambda: {'dice': [], 'iou': []})
    
    # Select 10 random indices reproducibly for visualization
    vis_indices = set(random.sample(range(len(df_test)), 10))
    
    print("Starting inference...")
    idx = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Inference
            if config['training']['mixed_precision'] and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(images)
            else:
                logits = model(images)
                
            preds = torch.argmax(logits, dim=1) # (B, H, W)
            
            # Batch metrics processing
            for b in range(images.size(0)):
                # Metrics per image
                img_dice, img_iou = calculate_dice_iou(preds[b:b+1], masks[b:b+1], num_classes=6)
                all_dice.append(img_dice)
                all_iou.append(img_iou)
                
                # E2E group metrics tracking
                e2e_id = df_test.iloc[idx]['e2e_group_id']
                e2e_metrics[e2e_id]['dice'].append(img_dice)
                e2e_metrics[e2e_id]['iou'].append(img_iou)
                
                # Save Raw Pred Mask
                sample_id = df_test.iloc[idx]['sample_id']
                pred_np = preds[b].cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(preds_dir, f"{sample_id}_pred.png"), pred_np)
                
                # Plot visualization if selected
                if idx in vis_indices:
                    # Move image to CPU and convert back from C,H,W to H,W,C
                    img_np = images[b].cpu().permute(1, 2, 0).numpy()
                    mask_np = masks[b].cpu().numpy()
                    save_path = os.path.join(vis_dir, f"{sample_id}_vis.png")
                    create_overlay(img_np, mask_np, pred_np, save_path)
                    
                idx += 1
                
    # Aggregate final metrics
    all_dice = np.array(all_dice) # (N, 6)
    all_iou = np.array(all_iou)   # (N, 6)
    
    mean_dice_per_class = np.mean(all_dice, axis=0)
    mean_iou_per_class = np.mean(all_iou, axis=0)
    
    total_mean_dice = np.mean(mean_dice_per_class)
    fg_mean_dice = np.mean(mean_dice_per_class[1:6])
    total_mean_iou = np.mean(mean_iou_per_class)
    
    # Save Report (.txt)
    report_path = os.path.join(reports_dir, "test_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("OCT5k FINAL TEST EVALUATION REPORT\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total Test Images: {len(df_test)}\n")
        f.write(f"Total E2E Groups:  {unique_e2e}\n\n")
        f.write("--- GLOBAL METRICS ---\n")
        f.write(f"Mean Dice (All classes): {total_mean_dice:.4f}\n")
        f.write(f"Mean Foreground Dice (1-5): {fg_mean_dice:.4f}\n")
        f.write(f"Mean IoU (All classes):  {total_mean_iou:.4f}\n\n")
        
        f.write("--- PER-CLASS METRICS ---\n")
        for c in range(6):
            f.write(f"Class {c} | Dice: {mean_dice_per_class[c]:.4f} | IoU: {mean_iou_per_class[c]:.4f}\n")
            
        f.write("\n--- E2E GROUP METRICS (Mean Dice over 6 classes) ---\n")
        for e2e, mets in e2e_metrics.items():
            grp_dice = np.mean(mets['dice'])
            f.write(f"Group {e2e}: {grp_dice:.4f}\n")
            
    # Save CSV
    csv_path = os.path.join(reports_dir, "test_metrics.csv")
    df_metrics = pd.DataFrame(all_dice, columns=[f"Dice_C{i}" for i in range(6)])
    df_iou = pd.DataFrame(all_iou, columns=[f"IoU_C{i}" for i in range(6)])
    df_out = pd.concat([df_test[['sample_id', 'e2e_group_id']], df_metrics, df_iou], axis=1)
    df_out.to_csv(csv_path, index=False)
    
    # Terminal Print
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Number of test images:     {len(df_test)}")
    print(f"Number of test E2E groups: {unique_e2e}")
    if device.type == 'cuda':
        print(f"GPU used:                  {torch.cuda.get_device_name(0)}")
    print("-" * 40)
    print(f"Mean Dice:                 {total_mean_dice:.4f}")
    print(f"Foreground Dice:           {fg_mean_dice:.4f}")
    print(f"Mean IoU:                  {total_mean_iou:.4f}")
    print("-" * 40)
    print(f"Per-class Dice: {[round(x,4) for x in mean_dice_per_class]}")
    print(f"Per-class IoU:  {[round(x,4) for x in mean_iou_per_class]}")
    print("="*40)
    print(f"\nSaved test_metrics.csv to: {csv_path}")
    print(f"Saved report to:           {report_path}")
    print(f"Saved {len(vis_indices)} visualizations to:  {vis_dir}")

if __name__ == "__main__":
    main()
