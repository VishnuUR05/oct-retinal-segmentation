import os
import yaml
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import streamlit as st

@st.cache_resource
def load_model():
    config_path = "configs/baseline.yaml"
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    return model, device

def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def extract_raw_boundaries(pred_mask_2d):
    H, W = pred_mask_2d.shape
    b_names = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
    b_classes = [1, 2, 3, 4, 5]
    
    raw_boundaries = {b: np.full(W, np.nan) for b in b_names}
    
    for x in range(W):
        col = pred_mask_2d[:, x]
        for c, b_name in zip(b_classes, b_names):
            idx = np.where((col[1:] == c) & (col[:-1] == c-1))[0]
            if len(idx) > 0:
                raw_boundaries[b_name][x] = float(idx[0] + 1)
    return raw_boundaries

def run_inference(image_path, model, device, transform):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    orig_h, orig_w = img_gray.shape
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    if orig_h != 512 or orig_w != 512:
        img_rgb = cv2.resize(img_rgb, (512, 512))
        
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(img_tensor)
        else:
            logits = model(img_tensor)
            
        preds = torch.argmax(logits, dim=1)
        pred_mask = preds[0].cpu().numpy()
        
    raw_boundaries = extract_raw_boundaries(pred_mask)
    return img_rgb, raw_boundaries
