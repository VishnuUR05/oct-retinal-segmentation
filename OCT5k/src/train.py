import os
import sys
import time
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Append root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import OCTDataset, get_training_augmentation, get_validation_augmentation
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

def main():
    parser = argparse.ArgumentParser(description="OCT5k Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set seed for reproducibility
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    print(f"Reproducibility Seed Set: {seed}")
    
    # Setup Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device Selected: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.2f} GB")
        
    # Create output dirs
    outputs_dir = config['paths']['outputs_dir']
    checkpoints_dir = os.path.join(outputs_dir, "checkpoints")
    reports_dir = os.path.join(outputs_dir, "reports")
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save a copy of the run config
    with open(os.path.join(reports_dir, "run_config.yaml"), 'w') as f:
        yaml.dump(config, f)
        
    # 2. Datasets & Loaders
    print("\nLoading datasets...")
    train_csv = config['paths']['train_csv']
    val_csv = config['paths']['val_csv']
    batch_size = config['training']['batch_size']
    
    # ONLY load train and val. Test is STRICTLY excluded.
    train_dataset = OCTDataset(train_csv, transform=get_training_augmentation())
    val_dataset = OCTDataset(val_csv, transform=get_validation_augmentation())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train Dataset: {len(train_dataset)} images")
    print(f"Validation Dataset: {len(val_dataset)} images")
    
    # 3. Model
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=config['model']['encoder_weights'],
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    ).to(device)
    
    # 4. Losses, Optimizer, Scheduler, AMP
    ce_loss = nn.CrossEntropyLoss()
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    scheduler_patience = config['training'].get('scheduler_patience', 5)
    scheduler_factor = config['training'].get('scheduler_factor', 0.5)
    
    # Scheduler strictly tied to "max" foreground Dice later
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, verbose=True
    )
    
    scaler = torch.amp.GradScaler('cuda') if config['training']['mixed_precision'] and device.type == 'cuda' else None
    
    # 5. Training State
    start_epoch = 0
    epochs = config['training']['epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    best_val_foreground_dice = -1.0
    epochs_no_improve = 0
    
    history_file = os.path.join(reports_dir, "training_history.csv")
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,train_dice,val_dice,train_fg_dice,val_fg_dice,mean_iou,learning_rate,epoch_time_s\n")
            
    # 6. Training Loop
    print("\nStarting Training...")
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_dice_scores = []
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    l_ce = ce_loss(logits, masks)
                    l_dice = dice_loss_fn(logits, masks)
                    loss = l_ce + l_dice
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                l_ce = ce_loss(logits, masks)
                l_dice = dice_loss_fn(logits, masks)
                loss = l_ce + l_dice
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() * images.size(0)
            
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                b_dice, _ = calculate_dice_iou(preds, masks, num_classes=config['model']['classes'])
                train_dice_scores.append(b_dice)
                
        train_loss = train_loss / len(train_dataset)
        train_dice_scores = np.mean(np.array(train_dice_scores), axis=0) # shape: (6,)
        
        train_mean_dice = np.mean(train_dice_scores)
        # Class 0 is background, classes 1-5 are foreground boundaries
        train_fg_dice = np.mean(train_dice_scores[1:6])
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        val_iou_scores = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(images)
                        l_ce = ce_loss(logits, masks)
                        l_dice = dice_loss_fn(logits, masks)
                        loss = l_ce + l_dice
                else:
                    logits = model(images)
                    l_ce = ce_loss(logits, masks)
                    l_dice = dice_loss_fn(logits, masks)
                    loss = l_ce + l_dice
                    
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(logits, dim=1)
                b_dice, b_iou = calculate_dice_iou(preds, masks, num_classes=config['model']['classes'])
                val_dice_scores.append(b_dice)
                val_iou_scores.append(b_iou)
                
        val_loss = val_loss / len(val_dataset)
        val_dice_scores = np.mean(np.array(val_dice_scores), axis=0)
        val_iou_scores = np.mean(np.array(val_iou_scores), axis=0)
        
        val_mean_dice = np.mean(val_dice_scores)
        val_fg_dice = np.mean(val_dice_scores[1:6])
        val_mean_iou = np.mean(val_iou_scores)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time:.0f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Mean Dice: {train_mean_dice:.4f} | Train FG Dice: {train_fg_dice:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Mean Dice:   {val_mean_dice:.4f} | Val FG Dice:   {val_fg_dice:.4f} | Val Mean IoU: {val_mean_iou:.4f}")
        print(f"  Val Per-Class Dice: {[round(d, 4) for d in val_dice_scores]}")
        print(f"  Val Per-Class IoU:  {[round(i, 4) for i in val_iou_scores]}")
        
        # Save to history
        with open(history_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{train_mean_dice:.6f},{val_mean_dice:.6f},{train_fg_dice:.6f},{val_fg_dice:.6f},{val_mean_iou:.6f},{current_lr},{epoch_time:.2f}\n")
            
        # Step scheduler based on FOREGROUND DICE
        scheduler.step(val_fg_dice)
        
        # Save Checkpoints
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_fg_dice': val_fg_dice,
        }, os.path.join(checkpoints_dir, "last_model.pth"))
        
        if val_fg_dice > best_val_foreground_dice:
            print(f"  *** Val FG Dice improved from {best_val_foreground_dice:.4f} to {val_fg_dice:.4f}. Saving best_model.pth ***")
            best_val_foreground_dice = val_fg_dice
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Early stopping patience: {epochs_no_improve}/{early_stopping_patience}")
            
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered! No improvement in FG Dice for {early_stopping_patience} epochs.")
            break

if __name__ == "__main__":
    main()
