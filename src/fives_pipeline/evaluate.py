import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Support running both as a package and as a standalone script
try:
    from src.fives_pipeline.dataset import FIVESDataset
    from src.fives_pipeline.preprocessing import FIVESTransform
    from src.fives_pipeline.model import UNet
    from src.fives_pipeline.metrics import calculate_metrics
except ImportError:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir.parent.parent))
    from src.fives_pipeline.dataset import FIVESDataset
    from src.fives_pipeline.preprocessing import FIVESTransform
    from src.fives_pipeline.model import UNet
    from src.fives_pipeline.metrics import calculate_metrics

def save_comparison_image(image_tensor, gt_mask_tensor, pred_mask_tensor, out_path, mean, std):
    """
    Saves a side-by-side composite: [Original Image | Ground Truth | Prediction]
    """
    # De-normalize image
    img = image_tensor.cpu().clone().numpy().transpose(1, 2, 0)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * std[c]) + mean[c]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    # Format GT mask
    gt = (gt_mask_tensor.cpu().squeeze().numpy() * 255.0).astype(np.uint8)
    gt_rgb = np.stack([gt, gt, gt], axis=-1)
    
    # Format Pred mask
    pred = (pred_mask_tensor.cpu().squeeze().numpy() * 255.0).astype(np.uint8)
    pred_rgb = np.stack([pred, pred, pred], axis=-1)
    
    # Combine horizontally
    combined = np.concatenate([img, gt_rgb, pred_rgb], axis=1)
    Image.fromarray(combined).save(out_path)

def evaluate_fives(
    data_root: str,
    checkpoint_path: str,
    batch_size: int = 4,
    image_size: int = 512,
    output_dir: str = "outputs/fives_pipeline",
    save_samples: int = 5,
    threshold: float = 0.5,
    device_name: str = None
):
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation device: {device.type.upper()}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = FIVESTransform(
        size=(image_size, image_size),
        augment=False,
        use_clahe=False,
        mean=mean,
        std=std
    )
    
    print(f"[*] Loading test data from: {data_root}")
    test_dataset = FIVESDataset(root_dir=data_root, split="test", transform=transform)
    print(f"[*] Total test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path.resolve()}")
        
    print(f"[*] Loading model weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "accuracy": 0.0}
    total_batches = 0
    saved_count = 0
    
    samples_dir = output_path / "test_samples"
    if save_samples > 0:
        samples_dir.mkdir(parents=True, exist_ok=True)

    print("\n[*] Evaluating on Test Set...")
    test_pbar = tqdm(test_loader, desc="Evaluating Test Set", unit="batch", dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > threshold).float()
            
            batch_metrics = calculate_metrics(logits, masks, threshold=threshold)
            for k in metric_sums:
                metric_sums[k] += batch_metrics[k]
            total_batches += 1
            
            test_pbar.set_postfix({"batch_dice": f"{batch_metrics['dice']:.4f}"})
            
            # Save visual samples if requested
            if saved_count < save_samples:
                for b in range(images.size(0)):
                    if saved_count >= save_samples:
                        break
                    sample_file = samples_dir / f"test_sample_{saved_count+1:03d}.png"
                    save_comparison_image(
                        image_tensor=images[b],
                        gt_mask_tensor=masks[b],
                        pred_mask_tensor=preds_bin[b],
                        out_path=sample_file,
                        mean=mean,
                        std=std
                    )
                    saved_count += 1

    final_metrics = {k: v / max(1, total_batches) for k, v in metric_sums.items()}
    
    print("\n" + "=" * 50)
    print("           FIVES TEST EVALUATION RESULTS          ")
    print("=" * 50)
    print(f"  Dice Similarity (F1) : {final_metrics['dice']:.4f}")
    print(f"  Jaccard Index (IoU)  : {final_metrics['iou']:.4f}")
    print(f"  Precision (PPV)      : {final_metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity) : {final_metrics['recall']:.4f}")
    print(f"  Specificity (TNR)    : {final_metrics['specificity']:.4f}")
    print(f"  Overall Accuracy     : {final_metrics['accuracy']:.4f}")
    print("=" * 50)
    
    # Save test results to JSON
    report_file = output_path / "fives_test_metrics.json"
    with open(report_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"[*] Results saved to: {report_file.resolve()}")
    if saved_count > 0:
        print(f"[*] {saved_count} sample visual predictions saved in: {samples_dir.resolve()}")
        
    return final_metrics

def parse_args():
    default_dataset = r"C:\AIT MAJOR PROJECT\new fundus dataset800\FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
    default_model = "outputs/fives_pipeline/best_fives_unet.pth"
    parser = argparse.ArgumentParser(description="Evaluate U-Net on FIVES Vessel Segmentation Test Set")
    parser.add_argument("--data-root", type=str, default=default_dataset, help="Path to FIVES dataset directory")
    parser.add_argument("--checkpoint", type=str, default=default_model, help="Path to model weights (.pth)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--image-size", type=int, default=512, help="Input resolution")
    parser.add_argument("--output-dir", type=str, default="outputs/fives_pipeline", help="Directory for results")
    parser.add_argument("--save-samples", type=int, default=5, help="Number of qualitative sample images to save")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_fives(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_dir=args.output_dir,
        save_samples=args.save_samples,
        threshold=args.threshold,
        device_name=args.device
    )
