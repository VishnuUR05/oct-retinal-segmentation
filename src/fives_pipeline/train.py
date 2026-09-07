import os
import sys
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# Support running both as a package and as a standalone script
try:
    from src.fives_pipeline.dataset import FIVESDataset
    from src.fives_pipeline.preprocessing import FIVESTransform
    from src.fives_pipeline.model import UNet
    from src.fives_pipeline.metrics import BCEDiceLoss, calculate_metrics
except ImportError:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir.parent.parent))
    from src.fives_pipeline.dataset import FIVESDataset
    from src.fives_pipeline.preprocessing import FIVESTransform
    from src.fives_pipeline.model import UNet
    from src.fives_pipeline.metrics import BCEDiceLoss, calculate_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_fives(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    val_split: float = 0.2,
    output_dir: str = "outputs/fives_pipeline",
    use_clahe: bool = False,
    num_workers: int = 0,
    smoke_test: bool = False,
    max_samples: int = None,
    resume: str = None,
    device_name: str = None
):
    set_seed(42)
    
    # Device setup
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device: {device.type.upper()}")
    if device.type == "cpu":
        print(f"[*] CPU mode active. Using {torch.get_num_threads()} CPU threads for processing.")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    train_transform = FIVESTransform(
        size=(image_size, image_size),
        augment=True,
        use_clahe=use_clahe
    )
    
    # Load dataset
    print(f"[*] Loading training data from: {data_root}")
    full_dataset = FIVESDataset(root_dir=data_root, split="train", transform=train_transform)
    total_samples = len(full_dataset)
    print(f"[*] Total training samples found in dataset: {total_samples}")
    
    # Optional sample limiting
    if max_samples and max_samples < total_samples:
        indices = list(range(max_samples))
        full_dataset = Subset(full_dataset, indices)
        total_samples = max_samples
        print(f"[*] Limiting training run to {total_samples} samples.")
        
    # Split into train and validation sets
    val_size = max(1, int(val_split * total_samples))
    train_size = total_samples - val_size
    
    train_subset, val_subset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[*] Train set: {len(train_subset)} samples | Val set: {len(val_subset)} samples")
    total_batches = len(train_subset) // batch_size + (1 if len(train_subset) % batch_size != 0 else 0)
    print(f"[*] Batches per epoch: {total_batches}")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(f"[*] Initialized U-Net with {model.count_parameters():,} trainable parameters.")
    
    if resume and Path(resume).exists():
        print(f"[*] Resuming weights from checkpoint: {resume}")
        model.load_state_dict(torch.load(resume, map_location=device))
    
    # Loss & Optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_dice = 0.0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_specificity": []
    }
    
    actual_epochs = 1 if smoke_test else epochs
    print(f"\n{'='*65}")
    print(f"[*] Training started ({actual_epochs} Epochs) {'[SMOKE TEST MODE]' if smoke_test else ''}")
    print(f"{'='*65}\n")
    
    for epoch in range(1, actual_epochs + 1):
        # ---------------- Training Phase ----------------
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch:02d}/{actual_epochs:02d}] Train", 
            unit="batch",
            dynamic_ncols=True
        )
        
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
            
            # Live progress bar update
            avg_loss = running_train_loss / train_batches
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})
            
            if smoke_test and batch_idx >= 1:
                break
                
        epoch_train_loss = running_train_loss / max(1, train_batches)
        
        # ---------------- Validation Phase ----------------
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "accuracy": 0.0}
        
        val_pbar = tqdm(
            val_loader, 
            desc=f"Epoch [{epoch:02d}/{actual_epochs:02d}] Val  ", 
            unit="batch",
            dynamic_ncols=True
        )
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_pbar):
                images = images.to(device)
                masks = masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                running_val_loss += loss.item()
                val_batches += 1
                
                batch_metrics = calculate_metrics(logits, masks)
                for k in metric_sums:
                    metric_sums[k] += batch_metrics[k]
                    
                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "dice": f"{batch_metrics['dice']:.4f}"})
                
                if smoke_test and batch_idx >= 1:
                    break
                    
        epoch_val_loss = running_val_loss / max(1, val_batches)
        val_metrics = {k: v / max(1, val_batches) for k, v in metric_sums.items()}
        
        scheduler.step()
        
        # Record history
        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_specificity"].append(val_metrics["specificity"])
        
        print(f"\n>> Summary Epoch [{epoch:02d}/{actual_epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | Spec: {val_metrics['specificity']:.4f}\n")
              
        # Checkpoint saving
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_path = output_path / "best_fives_unet.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  [+] Saved new best model (Dice: {best_dice:.4f}) -> {best_path.name}")
            
        latest_path = output_path / "latest_fives_unet.pth"
        torch.save(model.state_dict(), latest_path)
        
    # Save training metrics history
    history_file = output_path / "fives_training_metrics.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"\n[*] Training complete.")
    print(f"[*] Checkpoints and metrics saved in: {output_path.resolve()}")
    return history

def parse_args():
    default_dataset = r"C:\AIT MAJOR PROJECT\new fundus dataset800\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
    parser = argparse.ArgumentParser(description="Train U-Net on FIVES Retinal Vessel Segmentation Dataset")
    parser.add_argument("--data-root", type=str, default=default_dataset, help="Path to FIVES dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=512, help="Input resolution (resized)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set split ratio")
    parser.add_argument("--output-dir", type=str, default="outputs/fives_pipeline", help="Directory to save checkpoints")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker processes")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast 1-epoch smoke test")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit total training samples (useful for fast CPU runs)")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing weights checkpoint to resume training from")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_fives(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        val_split=args.val_split,
        output_dir=args.output_dir,
        use_clahe=args.clahe,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
        max_samples=args.max_samples,
        resume=args.resume,
        device_name=args.device
    )
