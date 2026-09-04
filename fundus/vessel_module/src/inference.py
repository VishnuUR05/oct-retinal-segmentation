import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

def predict_full_image_tiled(image: Image.Image, model, device, patch_size=512, stride=256):
    """
    Performs tiled inference with overlapping patches and averages the predictions.
    Reconstructs the full resolution probability map.
    """
    model.eval()
    # Must normalize exactly as in training (data_loading.py)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = transforms.ToTensor()(image).unsqueeze(0) # 1, C, H, W
    img_tensor = torch.stack([normalize(img_tensor[0])]).to(device)
    
    _, _, H, W = img_tensor.shape
    
    # Calculate padding if needed
    pad_bottom = max(0, patch_size - (H % patch_size)) if H % patch_size != 0 else 0
    pad_right = max(0, patch_size - (W % patch_size)) if W % patch_size != 0 else 0
    
    if H < patch_size:
        pad_bottom = patch_size - H
    if W < patch_size:
        pad_right = patch_size - W
        
    if pad_bottom > 0 or pad_right > 0:
         img_tensor = F.pad(img_tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
         
    _, _, pad_H, pad_W = img_tensor.shape

    prob_map = torch.zeros((1, 1, pad_H, pad_W), device=device)
    count_map = torch.zeros((1, 1, pad_H, pad_W), device=device)
    
    # Generate tiles
    y_starts = list(range(0, pad_H - patch_size + 1, stride))
    if len(y_starts) == 0 or y_starts[-1] + patch_size < pad_H:
        y_starts.append(pad_H - patch_size)
        
    x_starts = list(range(0, pad_W - patch_size + 1, stride))
    if len(x_starts) == 0 or x_starts[-1] + patch_size < pad_W:
        x_starts.append(pad_W - patch_size)

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
                out = model(patch)
                prob = torch.sigmoid(out)
                
                prob_map[:, :, y:y+patch_size, x:x+patch_size] += prob
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1.0
                
    prob_map /= count_map
    
    # Crop back to original size
    prob_map = prob_map[:, :, 0:H, 0:W]
    
    return prob_map.squeeze().cpu().numpy()
