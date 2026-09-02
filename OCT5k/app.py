import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import sys

base_dir = r"D:\AIT Major Project\oct rertinal segmentation\OCT5k"
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dataset import get_validation_augmentation
from src.extract_boundaries import extract_raw_boundaries, interpolate_boundaries, calculate_structural_features

# Enforce no gradients globally for safety
torch.set_grad_enabled(False)

st.set_page_config(
    page_title="OCT Retinal Layer Segmentation — Inference Demo", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CONSTANTS & CACHED RESOURCES
# ---------------------------------------------------------
BOUNDARY_NAMES = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
BOUNDARY_COLORS = {
    'ILM': (255, 0, 0),      # Red
    'OPL': (0, 255, 0),      # Green
    'IS-OS': (0, 0, 255),    # Blue
    'IBRPE': (0, 255, 255),  # Cyan
    'OBRPE': (255, 0, 255)   # Magenta
}
MASK_COLORS = np.array([
    [0, 0, 0],       # 0: Background / above ILM
    [255, 0, 0],     # 1: ILM-OPL
    [0, 255, 0],     # 2: OPL-IS-OS
    [0, 0, 255],     # 3: IS-OS-IBRPE
    [255, 255, 0],   # 4: IBRPE-OBRPE
    [255, 0, 255]    # 5: Below OBRPE
], dtype=np.uint8)

@st.cache_resource
def load_oct_model():
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        st.stop()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Strictly define architecture
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=6
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()
        
    model.eval() # Ensure evaluation mode
    return model, device

# ---------------------------------------------------------
# CORE PIPELINE LOGIC
# ---------------------------------------------------------
def check_quality(boundaries):
    """ Quality control based on anatomical ordering """
    try:
        ilm = boundaries['ILM']
        opl = boundaries['OPL']
        isos = boundaries['IS-OS']
        ibrpe = boundaries['IBRPE']
        obrpe = boundaries['OBRPE']
        
        valid_mask = ~(np.isnan(ilm) | np.isnan(opl) | np.isnan(isos) | np.isnan(ibrpe) | np.isnan(obrpe))
        if valid_mask.sum() == 0:
            return "INVALID"
            
        v_ilm, v_opl, v_isos, v_ibrpe, v_obrpe = ilm[valid_mask], opl[valid_mask], isos[valid_mask], ibrpe[valid_mask], obrpe[valid_mask]
        violations = np.sum(~((v_ilm <= v_opl) & (v_opl <= v_isos) & (v_isos <= v_ibrpe) & (v_ibrpe <= v_obrpe)))
        
        if violations == 0:
            return "VALID"
        elif violations < len(v_ilm) * 0.1: # Less than 10% violations
            return "UNCERTAIN"
        else:
            return "INVALID"
    except:
        return "INVALID"

def process_image(img_bytes, model, device, transform):
    # 1. Decode Image securely
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("Invalid image format provided")
        
    orig_h, orig_w = img_gray.shape
    
    # 2. Resize and RGB Convert
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if orig_h != 512 or orig_w != 512:
        img_rgb = cv2.resize(img_rgb, (512, 512))
        
    # 3. Preprocess matching the training pipeline exactly
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # 4. Inference
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(img_tensor)
        else:
            logits = model(img_tensor)
            
        preds = torch.argmax(logits, dim=1)
        pred_mask = preds[0].cpu().numpy()
        
    # 5. Boundary Extraction using validated project functions
    raw_boundaries = extract_raw_boundaries(pred_mask)
    interp_boundaries, _ = interpolate_boundaries(raw_boundaries)
    
    # 6. Quality Control
    qc_status = check_quality(interp_boundaries)
    
    return img_rgb, pred_mask, interp_boundaries, qc_status

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------
def draw_ui():
    model, device = load_oct_model()
    transform = get_validation_augmentation()
    
    st.title("OCT Retinal Layer Segmentation — Inference Demo")
    st.info("Demonstrating the segmentation model only. This is an inference visualization task for retinal-layer segmentation.")
    
    # SIDEBAR
    st.sidebar.header("Image Source")
    img_source = st.sidebar.radio("Select source:", ["Upload Image", "Select Existing Dataset Image"])
    
    img_bytes = None
    
    if img_source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload an OCT Image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
        if uploaded_file is not None:
            img_bytes = uploaded_file.read()
    else:
        st.sidebar.markdown("---")
        try:
            df_test = pd.read_csv("splits/test.csv")
            if not df_test.empty:
                # Provide searchable dropdown
                img_paths = df_test['image_path'].tolist()
                selected_path = st.sidebar.selectbox("Browse Test Set:", img_paths)
                
                dataset_root = "OCT5k dataset"
                full_path = os.path.join(dataset_root, selected_path)
                
                if os.path.exists(full_path):
                    if st.sidebar.button("Load Selected Image"):
                        with open(full_path, "rb") as f:
                            img_bytes = f.read()
                else:
                    st.sidebar.error(f"Dataset image not found locally at {full_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading test split: {e}")
            
    st.sidebar.markdown("---")
    st.sidebar.write("**Model:** U-Net + ResNet34")
    st.sidebar.write("**Input:** 3 channels, 512x512")
    st.sidebar.write("**Output:** 6 classes")
    st.sidebar.write(f"**Inference Device:** {device.type.upper()}")
            
    if img_bytes:
        with st.spinner("Processing image..."):
            try:
                img_rgb, pred_mask, boundaries, qc_status = process_image(img_bytes, model, device, transform)
                
                st.markdown("### Inference Results")
                
                # Checkbox for views
                st.markdown("**Visualization Options**")
                cols_opts = st.columns(4)
                show_orig = cols_opts[0].checkbox("Original", value=True)
                show_mask = cols_opts[1].checkbox("Mask", value=True)
                show_overlay = cols_opts[2].checkbox("Overlay", value=True)
                show_bounds = cols_opts[3].checkbox("Boundaries", value=True)
                
                # Render logic
                display_cols = []
                if show_orig: display_cols.append("Original OCT")
                if show_mask: display_cols.append("Predicted Mask")
                if show_overlay: display_cols.append("Segmentation Overlay")
                if show_bounds: display_cols.append("Retinal Boundaries")
                
                if display_cols:
                    viz_cols = st.columns(len(display_cols))
                    
                    idx = 0
                    if show_orig:
                        with viz_cols[idx]:
                            st.image(img_rgb, caption="Original OCT", use_container_width=True)
                        idx += 1
                        
                    if show_mask:
                        mask_color_img = MASK_COLORS[pred_mask]
                        with viz_cols[idx]:
                            st.image(mask_color_img, caption="Predicted Mask", use_container_width=True)
                        idx += 1
                        
                    if show_overlay:
                        mask_color_img = MASK_COLORS[pred_mask]
                        mask_overlay = cv2.addWeighted(img_rgb, 0.6, mask_color_img, 0.4, 0)
                        with viz_cols[idx]:
                            st.image(mask_overlay, caption="Original + Mask", use_container_width=True)
                        idx += 1
                        
                    if show_bounds:
                        bnd_img = img_rgb.copy()
                        for b_name in BOUNDARY_NAMES:
                            pts = np.array([[[x, int(y)]] for x, y in enumerate(boundaries[b_name]) if not np.isnan(y)], dtype=np.int32)
                            if len(pts) > 1:
                                cv2.polylines(bnd_img, [pts], False, BOUNDARY_COLORS[b_name], 2)
                        with viz_cols[idx]:
                            st.image(bnd_img, caption="Retinal Boundaries", use_container_width=True)
                        idx += 1
                
                st.markdown("---")
                col_qc, col_feat = st.columns([1, 2])
                
                with col_qc:
                    st.markdown("### Quality Control")
                    if qc_status == "VALID":
                        st.success("Status: **VALID** (Anatomical order preserved)")
                    elif qc_status == "UNCERTAIN":
                        st.warning("Status: **UNCERTAIN** (Minor ordering violations detected)")
                    else:
                        st.error("Status: **INVALID** (Significant anatomical ordering violations)")
                        
                    st.markdown("### Segmentation Classes")
                    st.markdown("""
                    - **Class 0:** Background / above ILM (Black)
                    - **Class 1:** ILM–OPL (Red)
                    - **Class 2:** OPL–IS-OS (Green)
                    - **Class 3:** IS-OS–IBRPE (Blue)
                    - **Class 4:** IBRPE–OBRPE (Yellow)
                    - **Class 5:** Below OBRPE (Magenta)
                    """)
                    
                with col_feat:
                    st.markdown("### Structural Features")
                    st.markdown("*Note: Image-domain thickness (pixels). Do not assume physical micrometer conversion.*")
                    
                    # Compute using core project function
                    raw_features = calculate_structural_features(boundaries)
                    
                    # Reformat into the UI DataFrame
                    features_df_data = []
                    layers_map = {
                        'ILM_OPL': 'ILM -> OPL thickness',
                        'OPL_ISOS': 'OPL -> IS-OS thickness',
                        'ISOS_IBRPE': 'IS-OS -> IBRPE thickness',
                        'IBRPE_OBRPE': 'IBRPE -> OBRPE thickness',
                        'Total_Retinal': 'Total ILM -> OBRPE thickness'
                    }
                    
                    for layer_key, display_name in layers_map.items():
                        mean_global = raw_features.get(f"{layer_key}_mean_pixels", np.nan)
                        mean_central = raw_features.get(f"central_{layer_key}_mean_pixels", np.nan)
                        
                        features_df_data.append({
                            'Feature': display_name,
                            'Global Mean (pixels)': round(mean_global, 2) if not np.isnan(mean_global) else np.nan,
                            'Central Window Mean (pixels)': round(mean_central, 2) if not np.isnan(mean_central) else np.nan
                        })
                        
                    df_features = pd.DataFrame(features_df_data).set_index('Feature')
                    st.dataframe(df_features, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during processing: {e}")

if __name__ == '__main__':
    draw_ui()
