import torch
import numpy as np

def calculate_dice_iou(pred, target, num_classes=6):
    """
    Calculates Dice and IoU for each class.
    pred: (B, H, W) - integer predictions
    target: (B, H, W) - integer ground truth
    """
    dice_per_class = []
    iou_per_class = []
    
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        
        if target_c.sum() == 0 and pred_c.sum() == 0:
            # If class is not present in target or prediction, ignore or return 1.
            dice = torch.tensor(1.0, device=pred.device)
            iou = torch.tensor(1.0, device=pred.device)
        elif target_c.sum() == 0 and pred_c.sum() > 0:
            dice = torch.tensor(0.0, device=pred.device)
            iou = torch.tensor(0.0, device=pred.device)
        else:
            dice = (2.0 * intersection) / (pred_c.sum() + target_c.sum() + 1e-8)
            iou = intersection / (union + 1e-8)
            
        dice_per_class.append(dice.item())
        iou_per_class.append(iou.item())
        
    return dice_per_class, iou_per_class

def extract_boundaries(mask_2d, num_classes=6):
    """
    Extracts the y-coordinate of the transition between classes.
    mask_2d: (H, W) numpy array
    Returns a dictionary of boundaries.
    """
    H, W = mask_2d.shape
    boundaries = {
        'ILM': np.zeros(W, dtype=np.int32),     # 0 to 1
        'OPL': np.zeros(W, dtype=np.int32),     # 1 to 2
        'IS-OS': np.zeros(W, dtype=np.int32),   # 2 to 3
        'IBRPE': np.zeros(W, dtype=np.int32),   # 3 to 4
        'OBRPE': np.zeros(W, dtype=np.int32)    # 4 to 5
    }
    
    # Transition pairs (from_class, to_class) mapping to boundary name
    transitions = {
        (0, 1): 'ILM',
        (1, 2): 'OPL',
        (2, 3): 'IS-OS',
        (3, 4): 'IBRPE',
        (4, 5): 'OBRPE'
    }
    
    for x in range(W):
        col = mask_2d[:, x]
        # Find transition points where col[y] != col[y-1]
        for y in range(1, H):
            if col[y] != col[y-1]:
                trans = (col[y-1], col[y])
                if trans in transitions:
                    # Record the y position of the transition
                    boundaries[transitions[trans]][x] = y
                    
    # Simple fallback: if a column missed a transition, fill it with previous x's value
    for b_name in boundaries:
        arr = boundaries[b_name]
        # If no transition was found, arr[x] remains 0. Forward/backward fill 0s.
        non_zero = np.where(arr > 0)[0]
        if len(non_zero) > 0:
            for x in range(W):
                if arr[x] == 0:
                    # Find closest non_zero
                    closest = non_zero[np.argmin(np.abs(non_zero - x))]
                    arr[x] = arr[closest]
                    
    return boundaries
