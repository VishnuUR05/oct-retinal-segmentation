import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np

# Append root to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import OCTDataset, get_training_augmentation
from src.metrics import calculate_dice_iou

def main():
    print("="*40)
    print(">>> DRY RUN STARTING <<<")
    print("="*40)
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Selected: {device}")
    
    # 1. Load one batch
    batch_size = 4
    train_dataset = OCTDataset("splits/train.csv", transform=get_training_augmentation())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("\n[Loading Batch...]")
    images, masks = next(iter(train_loader))
    
    # 2. Print Tensor info
    print("\n--- Tensor Information ---")
    print(f"Image Tensor Shape:  {images.shape}")
    print(f"Mask Tensor Shape:   {masks.shape}")
    print(f"Image Dtype:         {images.dtype}")
    print(f"Mask Dtype:          {masks.dtype}")
    print(f"Image Range:         [{images.min().item():.3f}, {images.max().item():.3f}]")
    
    unique_vals = torch.unique(masks)
    print(f"Unique Mask Values:  {unique_vals.tolist()}")
    
    # Move to CUDA
    images = images.to(device)
    masks = masks.to(device)
    
    # 3. Load Model
    print("\n[Loading Model...]")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=6,
    ).to(device)
    
    # Setup Loss & Optimizer
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 5. Forward Pass
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
    
    # 6. Loss Calculation
    print("\n--- Loss Calculation ---")
    print(f"Cross-Entropy Loss: {l_ce.item():.4f}")
    print(f"Dice Loss:          {l_dice.item():.4f}")
    print(f"Total Loss:         {total_loss.item():.4f}")
    
    # 7. Metrics
    print("\n--- Metrics ---")
    preds = torch.argmax(logits, dim=1)
    dice_class, iou_class = calculate_dice_iou(preds, masks, num_classes=6)
    print(f"Mean Dice: {np.mean(dice_class):.4f}")
    print(f"Mean IoU:  {np.mean(iou_class):.4f}")
    
    # 9. Backward Pass
    print("\n--- Backward Pass ---")
    optimizer.zero_grad()
    if scaler:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()
        
    # 10. Check Gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_grads}")
    
    # 11. Memory Check
    if device.type == 'cuda':
        print("\n--- GPU Memory Check ---")
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"Memory Allocated: {allocated:.2f} MB")
        print(f"Memory Reserved:  {reserved:.2f} MB")
        
    print("\n>>> DRY RUN COMPLETED SUCCESSFULLY. NO WEIGHTS SAVED. <<<")
    sys.exit(0)

if __name__ == '__main__':
    main()
