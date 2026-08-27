import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np

from oct_dataset import OCTDataset, get_validation_augmentation

def calculate_metrics(pred, target, num_classes=10):
    """
    Calculate Dice score and IoU for multi-class segmentation.
    Args:
        pred: torch.Tensor of shape (B, H, W) containing predicted class indices
        target: torch.Tensor of shape (B, H, W) containing ground truth class indices
    """
    dice_scores = []
    iou_scores = []
    
    for cls in range(1, num_classes):  # Skip background (0)
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            continue
            
        iou = intersection / union
        dice = (2. * intersection) / (pred_cls.sum() + target_cls.sum()).float()
        
        iou_scores.append(iou.item())
        dice_scores.append(dice.item())
        
    return np.mean(dice_scores) if dice_scores else 0.0, np.mean(iou_scores) if iou_scores else 0.0

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = args.data_path
    if not os.path.exists(base_dir):
        print(f"Error: Data path {base_dir} does not exist.")
        return

    val_dataset = OCTDataset(base_path=base_dir, split="val", transform=get_validation_augmentation())
    
    if len(val_dataset) == 0:
        print("Cannot evaluate without data. Please ensure images are downloaded.")
        return

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = args.num_classes

    # Load Model
    print(f"Loading {args.model} with encoder {args.encoder} from {args.model_path}...")
    if args.model.lower() == 'unet':
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=None,     
            in_channels=3,                  
            classes=num_classes,                      
        )
    else:
        raise ValueError(f"Model {args.model} is not supported.")
        
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Error: Model weights not found at {args.model_path}")
        return
        
    model = model.to(device)
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Evaluating")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            if args.amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)
                
            # Convert logits to class indices
            preds = torch.argmax(outputs, dim=1)
            
            dice, iou = calculate_metrics(preds, masks, num_classes)
            total_dice += dice
            total_iou += iou
            
            loop.set_postfix(dice=f"{dice:.4f}", iou=f"{iou:.4f}")
            
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    print("\nEvaluation Complete!")
    print(f"Mean Dice Score: {avg_dice:.4f}")
    print(f"Mean IoU Score:  {avg_iou:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Model for OCT5k")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--model", type=str, default="unet", help="Model architecture")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Model encoder backbone")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--data_path", type=str, default=".", help="Base path to dataset")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to saved model weights")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of segmentation classes")
    
    args = parser.parse_args()
    
    if args.data_path == ".":
        args.data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    evaluate(args)
