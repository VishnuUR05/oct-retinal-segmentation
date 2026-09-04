import os
import sys
import time
import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from data_loading import FIVESPatchDataset
from model_unet import ResNet34UNet
from losses import BCEDiceLoss
from metrics import BinaryMetrics
from utils import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to run")
    args = parser.parse_args()
    
    print("============================================================")
    print("Vessel Segmentation Training Initialization")
    print("============================================================")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {mem:.2f} GB")
    
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    csv_path = os.path.join(LOG_DIR, "training_history.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'learning_rate', 'epoch_time'])

    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(VAL_IMG_DIR):
        print("ERROR: Dataset paths do not exist. Please check config.py.")
        sys.exit(1)
        
    train_dataset = FIVESPatchDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, is_train=True)
    val_dataset = FIVESPatchDataset(VAL_IMG_DIR, VAL_MASK_DIR, is_train=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No images found in dataset directories.")
        sys.exit(1)
        
    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {args.epochs}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    device = torch.device(DEVICE)
    model = ResNet34UNet(num_classes=NUM_CLASSES)
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = BCEDiceLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    metrics = BinaryMetrics(threshold=THRESHOLD)
    
    best_dice = 0.0
    patience_counter = 0
    
    print("============================================================")
    print("Starting Training")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for images, masks in train_pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        metrics.reset()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = criterion(logits, masks)
                    
                val_loss += loss.item()
                metrics.update(logits, masks)
                val_pbar.set_postfix(loss=loss.item())
                
        val_loss /= len(val_loader)
        res = metrics.compute()
        val_dice = res['dice']
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_loss': val_loss
        }
        
        last_path = os.path.join(CHECKPOINT_DIR, "last_model.pth")
        torch.save(checkpoint, last_path)
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(checkpoint, best_path)
            best_str = f"{best_dice:.4f} (NEW BEST)"
        else:
            patience_counter += 1
            best_str = f"{best_dice:.4f}"
            
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_dice, lr, epoch_time])
            
        print("\n============================================================")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Training: Completed")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation: Completed")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {lr:.6f}")
        print(f"Best Dice: {best_str}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("============================================================\n")
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

if __name__ == '__main__':
    main()
