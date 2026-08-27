import streamlit as st
import os
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp
from src.dataset import get_validation_augmentation
from src.extract_boundaries import (
    extract_raw_boundaries,
    interpolate_boundaries,
    calculate_structural_features
)

st.set_page_config(page_title="OCT5k Structural Analysis Demo", layout="wide")

st.title("OCT Segmentation & Retinal Structure Demo")
st.markdown("""
*Disclaimer: This is a research demonstration strictly for quantitative retinal structural feature extraction. 
This application does not make any claims regarding Alzheimer's disease or dementia prediction.*
""")

@st.cache_resource
def load_model():
    config_path = "configs/baseline.yaml"
    if not os.path.exists(config_path):
        return None, None, "baseline.yaml config not found!"
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = os.path.join(config['paths']['outputs_dir'], "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        return None, None, f"Checkpoint not found at {checkpoint_path}"
        
    model = smp.Unet(
        encoder_name=config['model']['encoder'],
        encoder_weights=None,
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes']
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    return model, device, None

model, device, err_msg = load_model()

if err_msg:
    st.error(f"Model Loading Error: {err_msg}")
    st.stop()

st.sidebar.header("Model Information")
st.sidebar.write("**Model:** U-Net + ResNet34")
st.sidebar.write("**Classes:** 6")
st.sidebar.write(f"**Device:** {device.type.upper()}")

uploaded_file = st.file_uploader("Upload an OCT Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Read image securely into memory
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            st.error("Invalid image format.")
            st.stop()
            
        if img.shape[0] != 512 or img.shape[1] != 512:
            st.warning(f"Image dimensions are {img.shape}. The model expects 512x512. Resizing now.")
            img = cv2.resize(img, (512, 512))
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()
        
    # Preprocessing
    transform = get_validation_augmentation()
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        # Avoid autocast on CPU if CUDA is unavailable
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(img_tensor)
        else:
            logits = model(img_tensor)
            
        preds = torch.argmax(logits, dim=1)
        pred_mask = preds[0].cpu().numpy()
        
    # Boundary Extraction
    raw_boundaries = extract_raw_boundaries(pred_mask)
    interp_boundaries, _ = interpolate_boundaries(raw_boundaries)
    features = calculate_structural_features(interp_boundaries)
    
    # Visualization Colors
    mask_colors = np.array([
        [0, 0, 0],       # 0: BG above ILM
        [255, 0, 0],     # 1: ILM-OPL (Red)
        [0, 255, 0],     # 2: OPL-IS-OS (Green)
        [0, 0, 255],     # 3: IS-OS-IBRPE (Blue)
        [255, 255, 0],   # 4: IBRPE-OBRPE (Yellow)
        [255, 0, 255]    # 5: BG below OBRPE (Magenta)
    ], dtype=np.uint8)
    
    b_colors = {'ILM': 'red', 'OPL': 'green', 'IS-OS': 'blue', 'IBRPE': 'cyan', 'OBRPE': 'magenta'}
    
    # Generate RGB mask
    mask_rgb = mask_colors[pred_mask]
    overlay_mask = cv2.addWeighted(img_rgb, 0.7, mask_rgb, 0.3, 0)
    
    # Render UI
    st.subheader("Inference Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original OCT", use_container_width=True)
    with col2:
        st.image(overlay_mask, caption="6-Class Segmentation Mask", use_container_width=True)
    with col3:
        # Boundary plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_rgb)
        x_axis = np.arange(512)
        for b in b_colors:
            ax.plot(x_axis, interp_boundaries[b], color=b_colors[b], label=b, linewidth=2)
        ax.axis('off')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)
        
    st.subheader("Quantitative Retinal Structural Features (Pixels)")
    
    # Prepare Table
    metrics_display = [
        {"Feature": "ILM → OPL Thickness", "Global Mean": features.get('ILM_OPL_mean_pixels', 0), "Central Window Mean": features.get('central_ILM_OPL_mean_pixels', 0)},
        {"Feature": "OPL → IS-OS Thickness", "Global Mean": features.get('OPL_ISOS_mean_pixels', 0), "Central Window Mean": features.get('central_OPL_ISOS_mean_pixels', 0)},
        {"Feature": "IS-OS → IBRPE Thickness", "Global Mean": features.get('ISOS_IBRPE_mean_pixels', 0), "Central Window Mean": features.get('central_ISOS_IBRPE_mean_pixels', 0)},
        {"Feature": "IBRPE → OBRPE Thickness", "Global Mean": features.get('IBRPE_OBRPE_mean_pixels', 0), "Central Window Mean": features.get('central_IBRPE_OBRPE_mean_pixels', 0)},
        {"Feature": "Total Retinal Thickness", "Global Mean": features.get('Total_Retinal_mean_pixels', 0), "Central Window Mean": features.get('central_Total_Retinal_mean_pixels', 0)}
    ]
    df_metrics = pd.DataFrame(metrics_display)
    # Format properly
    df_metrics['Global Mean'] = df_metrics['Global Mean'].apply(lambda x: f"{x:.2f}")
    df_metrics['Central Window Mean'] = df_metrics['Central Window Mean'].apply(lambda x: f"{x:.2f}")
    
    st.table(df_metrics)
