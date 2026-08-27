import os
import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import pandas as pd
from oct_dataset import OCTDataset, get_training_augmentation, get_validation_augmentation
from metrics import calculate_dice_iou, extract_boundaries
import matplotlib.pyplot as plt
import numpy as np

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=6,
    )
    return model

def create_dry_run_visualization(image, gt_mask, pred_mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img = image[0].cpu().numpy().transpose(1, 2, 0)
    # Unnormalize
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    gt = gt_mask[0].cpu().numpy()
    pred = pred_mask[0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    
    axes[1].imshow(gt, cmap='nipy_spectral', vmin=0, vmax=5)
    axes[1].set_title("Ground Truth Mask")
    
    axes[2].imshow(pred, cmap='nipy_spectral', vmin=0, vmax=5)
    axes[2].set_title("Predicted Mask (Random)")
    
    plt.savefig(os.path.join(out_dir, 'dry_run_visualization.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help="Execute dry run only")
    args = parser.parse_args()
    
    dataset_root = os.path.abspath('.')
    splits_dir = os.path.join(dataset_root, 'splits')
    out_dir = os.path.join(dataset_root, 'outputs')
    
    # Setup GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"====================================")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    print(f"PyTorch Version: {torch.__version__}")
    print(f"====================================\n")
    
    batch_size = 4  # RTX 3050 safe value for 512x512
    
    train_dataset = OCTDataset(os.path.join(splits_dir, 'train.csv'), transform=get_training_augmentation())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = build_model().to(device)
    
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    if args.dry_run:
        print(">>> STARTING DRY RUN <<<")
        
        # 1. Load one batch
        images, masks = next(iter(train_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        print("\n--- Tensor Information ---")
        print(f"Images Shape: {images.shape} | Dtype: {images.dtype}")
        print(f"Images Range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"Masks Shape: {masks.shape} | Dtype: {masks.dtype}")
        
        unique_vals = torch.unique(masks)
        print(f"Masks Unique Values: {unique_vals.tolist()}")
        
        # 2. Forward Pass
        print("\n--- Forward Pass ---")
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(images)
                l_ce = ce_loss(logits, masks)
                l_dice = dice_loss(logits, masks)
                total_loss = l_ce + l_dice
        else:
            logits = model(images)
            l_ce = ce_loss(logits, masks)
            l_dice = dice_loss(logits, masks)
            total_loss = l_ce + l_dice
            
        print(f"Logits Shape: {logits.shape}")
        
        print("\n--- Loss Calculation ---")
        print(f"Cross Entropy Loss: {l_ce.item():.4f}")
        print(f"Dice Loss: {l_dice.item():.4f}")
        print(f"Total Combined Loss: {total_loss.item():.4f}")
        
        # 4. Backward Pass (Gradients Check)
        print("\n--- Backward Pass ---")
        optimizer.zero_grad()
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
            
        print("Gradients computed successfully.")
        
        # 5. Metrics
        print("\n--- Metrics Check ---")
        preds = torch.argmax(logits, dim=1)
        dice_class, iou_class = calculate_dice_iou(preds, masks, num_classes=6)
        print(f"Mean Dice: {np.mean(dice_class):.4f}")
        print(f"Mean IoU: {np.mean(iou_class):.4f}")
        
        # 6. Memory Check
        if device.type == 'cuda':
            print("\n--- GPU Memory Check ---")
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            print(f"Allocated: {allocated:.2f} MB")
            print(f"Reserved:  {reserved:.2f} MB")
            
        # 7. Validation of output saving
        create_dry_run_visualization(images, masks, preds, out_dir)
        print(f"\nDry run visualization saved to {os.path.join(out_dir, 'dry_run_visualization.png')}")
        
        print("\n>>> DRY RUN COMPLETED SUCCESSFULLY. STOPPING. <<<")
        sys.exit(0)

if __name__ == '__main__':
    main()
