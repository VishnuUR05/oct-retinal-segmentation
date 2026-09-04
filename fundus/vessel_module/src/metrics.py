import torch

class BinaryMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.eps = 1e-6

    def update(self, logits, targets):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()

    def compute(self):
        dice = (2 * self.tp + self.eps) / (2 * self.tp + self.fp + self.fn + self.eps)
        iou = (self.tp + self.eps) / (self.tp + self.fp + self.fn + self.eps)
        precision = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)
        specificity = (self.tn + self.eps) / (self.tn + self.fp + self.eps)
        
        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'specificity': specificity
        }
        
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
