import os
import time
import numpy as np
import pandas as pd
import cv2
import yaml
import torch
import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Enforce no gradients globally for safety
torch.set_grad_enabled(False)

st.set_page_config(
    page_title="OCT Segmentation & Structural Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CONSTANTS & CACHED RESOURCES
# ---------------------------------------------------------
BOUNDARY_NAMES = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
BOUNDARY_CLASSES = [1, 2, 3, 4, 5]
BOUNDARY_COLORS = {
    'ILM': (255, 0, 0),      # Red
    'OPL': (0, 255, 0),      # Green
    'IS-OS': (0, 0, 255),    # Blue
    'IBRPE': (0, 255, 255),  # Cyan
    'OBRPE': (255, 0, 255)   # Magenta
}
MASK_COLORS = np.array([
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: ILM-OPL
    [0, 255, 0],     # 2: OPL-ISOS
    [0, 0, 255],     # 3: ISOS-IBRPE
    [255, 255, 0],   # 4: IBRPE-OBRPE
    [255, 0, 255]    # 5: Below OBRPE
], dtype=np.uint8)

@st.cache_resource
def load_oct_model():
    # Enforce Read-Only Checkpoint Load
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at {checkpoint_path}")
        st.stop()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Strictly define architecture (U-Net + ResNet34, classes=6)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=6
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval() # Ensure evaluation mode
    return model, device

@st.cache_resource
def get_inference_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ---------------------------------------------------------
# CORE PIPELINE LOGIC
# ---------------------------------------------------------
def extract_boundaries(pred_mask_2d):
    """ Extracts pixel transition boundaries from 6-class mask and linearly interpolates """
    H, W = pred_mask_2d.shape
    raw_boundaries = {b: np.full(W, np.nan) for b in BOUNDARY_NAMES}
    
    for x in range(W):
        col = pred_mask_2d[:, x]
        for c, b_name in zip(BOUNDARY_CLASSES, BOUNDARY_NAMES):
            idx = np.where((col[1:] == c) & (col[:-1] == c-1))[0]
            if len(idx) > 0:
                raw_boundaries[b_name][x] = float(idx[0] + 1)
                
    # Linear Interpolation
    interp_boundaries = {}
    for b_name, arr in raw_boundaries.items():
        s = pd.Series(arr)
        # linear interp to handle small gaps
        s_interp = s.interpolate(method='linear', limit_direction='both').values
        if np.isnan(s_interp).all():
            s_interp = np.zeros_like(arr)
        interp_boundaries[b_name] = s_interp
        
    return interp_boundaries

def process_image(img_bytes, model, device, transform):
    # 1. Decode Image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("Invalid image provided")
        
    orig_h, orig_w = img_gray.shape
    
    # 2. Resize and RGB Convert
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if orig_h != 512 or orig_w != 512:
        img_rgb = cv2.resize(img_rgb, (512, 512))
        
    # 3. Preprocess
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # 4. Inference
    with torch.no_grad(): # Enforce no gradients
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(img_tensor)
        else:
            logits = model(img_tensor)
            
        preds = torch.argmax(logits, dim=1)
        pred_mask = preds[0].cpu().numpy()
        
    # 5. Boundary Extraction
    boundaries = extract_boundaries(pred_mask)
    
    return img_rgb, pred_mask, boundaries

def calculate_thickness_features(boundaries):
    features = {}
    layers = [
        ('ILM → OPL', 'ILM', 'OPL'),
        ('OPL → IS-OS', 'OPL', 'IS-OS'),
        ('IS-OS → IBRPE', 'IS-OS', 'IBRPE'),
        ('IBRPE → OBRPE', 'IBRPE', 'OBRPE'),
        ('Total Retinal Thickness', 'ILM', 'OBRPE')
    ]
    
    for display_name, top, bottom in layers:
        thick = np.maximum(boundaries[bottom] - boundaries[top], 0)
        mean_global = np.mean(thick)
        mean_central = np.mean(thick[128:384])
        
        features[display_name] = {
            'Global Mean (pixels)': round(mean_global, 2),
            'Central Window Mean (pixels)': round(mean_central, 2)
        }
        
    return pd.DataFrame(features).T

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------
def draw_ui():
    model, device = load_oct_model()
    transform = get_inference_transform()
    
    # SIDEBAR
    st.sidebar.title("Model Information")
    st.sidebar.markdown(f"""
    **Model:** U-Net  
    **Encoder:** ResNet34  
    **Input:** 512×512 OCT image  
    **Output:** 6-class retinal-layer segmentation  
    **Inference Device:** {device.type.upper()}
    """)
    if device.type == 'cuda':
        st.sidebar.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
    with st.sidebar.expander("Model Performance on OCT5k Held-Out Test Set", expanded=True):
        st.markdown("""
        **Mean Dice:** 0.9433  
        **Foreground Dice:** 0.9327  
        **Mean IoU:** 0.9042  
        
        **Class Breakdown:**
        - **Class 0:** 0.9958
        - **Class 1:** 0.9652
        - **Class 2:** 0.9540
        - **Class 3:** 0.8640
        - **Class 4:** 0.8828
        - **Class 5:** 0.9978
        
        *Dice measures overlap between predicted and ground-truth segmentation.*  
        *IoU (Intersection over Union) measures the intersection divided by the union of prediction and ground truth.*  
        *Note: These are spatial overlap metrics, not classification accuracy.*
        """)
        
    # MAIN PAGE
    st.title("OCT Retinal Layer Segmentation & Structural Feature Extraction")
    st.subheader("U-Net + ResNet34")
    
    st.info("Upload an OCT image to perform semantic segmentation and extract predicted retinal structural features. This demonstration analyzes anatomical structure and does not provide Alzheimer's/dementia diagnosis or risk prediction.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload OCT Image (PNG/JPG/JPEG)", type=['png', 'jpg', 'jpeg'])
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("Use Sample OCT")
        
    img_bytes = None
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
    elif use_sample:
        try:
            df_test = pd.read_csv("splits/test.csv")
            sample_path = df_test.iloc[0]['image_path']
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    img_bytes = f.read()
                st.success(f"Loaded sample image from Test Set")
            else:
                st.error("Sample image not found on disk.")
        except Exception as e:
            st.error(f"Could not load sample image: {e}")
            
    if img_bytes:
        with st.spinner("Processing image..."):
            try:
                start_time = time.time()
                img_rgb, pred_mask, boundaries = process_image(img_bytes, model, device, transform)
                inf_time = time.time() - start_time
                
                st.success(f"Inference completed in {inf_time:.2f} seconds.")
                
                # Visualizations
                st.markdown("### 1. Segmentation Mask")
                v_col1, v_col2, v_col3 = st.columns(3)
                
                mask_color_img = MASK_COLORS[pred_mask]
                mask_overlay = cv2.addWeighted(img_rgb, 0.6, mask_color_img, 0.4, 0)
                
                with v_col1:
                    st.image(img_rgb, caption="Original OCT", use_container_width=True)
                with v_col2:
                    st.image(mask_color_img, caption="6-Class Segmentation", use_container_width=True)
                with v_col3:
                    st.image(mask_overlay, caption="Segmentation Overlay", use_container_width=True)
                    
                st.markdown("### 2. Extracted Boundaries")
                
                bnd_img = img_rgb.copy()
                for b_name in BOUNDARY_NAMES:
                    pts = np.array([[[x, int(y)]] for x, y in enumerate(boundaries[b_name]) if not np.isnan(y)], dtype=np.int32)
                    if len(pts) > 1:
                        cv2.polylines(bnd_img, [pts], False, BOUNDARY_COLORS[b_name], 2)
                        
                b_col1, b_col2 = st.columns([1, 2])
                with b_col1:
                    st.markdown("""
                    **Legend:**
                    - <span style='color:red'>ILM (Red)</span>
                    - <span style='color:green'>OPL (Green)</span>
                    - <span style='color:blue'>IS-OS (Blue)</span>
                    - <span style='color:cyan'>IBRPE (Cyan)</span>
                    - <span style='color:magenta'>OBRPE (Magenta)</span>
                    """, unsafe_allow_html=True)
                with b_col2:
                    st.image(bnd_img, caption="Boundary Overlay", use_container_width=True)
                
                st.markdown("### 3. Predicted Retinal Structural Features")
                df_features = calculate_thickness_features(boundaries)
                st.dataframe(df_features, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during processing: {e}")

if __name__ == '__main__':
    draw_ui()
