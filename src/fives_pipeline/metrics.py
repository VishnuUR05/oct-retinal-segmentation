import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Composite Loss Function: BCE With Logits + Soft Dice Loss.
    Addresses severe foreground-background imbalance in retinal vessel segmentation.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, preds, targets):
        if preds.shape != targets.shape:
            targets = targets.view_as(preds)
            
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets)

        probs = torch.sigmoid(preds)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.eps) / (probs_flat.sum() + targets_flat.sum() + self.eps)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """
    Calculates segmentation evaluation metrics:
    - Dice Similarity Coefficient (F1-Score)
    - Intersection over Union (IoU / Jaccard Index)
    - Precision (Positive Predictive Value)
    - Recall / Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate)
    - Overall Accuracy
    """
    with torch.no_grad():
        if preds.shape != targets.shape:
            targets = targets.view_as(preds)

        probs = torch.sigmoid(preds)
        preds_bin = (probs > threshold).float()

        preds_flat = preds_bin.view(-1)
        targets_flat = targets.view(-1)

        tp = (preds_flat * targets_flat).sum().item()
        fp = (preds_flat * (1.0 - targets_flat)).sum().item()
        fn = ((1.0 - preds_flat) * targets_flat).sum().item()
        tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy
        }
