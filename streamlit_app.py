import os
import sys
import time
import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import streamlit as st

# Set page config with modern wide layout
st.set_page_config(
    page_title="Retinal AI | Dual-Model Analysis Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium medical dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 40%, #7209B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #8D99AE;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #00B4D8;
        margin-top: 4px;
    }
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #A0AAB2;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-primary { background: rgba(0, 180, 216, 0.2); color: #00B4D8; border: 1px solid rgba(0, 180, 216, 0.3); }
    .badge-purple  { background: rgba(114, 9, 183, 0.2); color: #C77DFF; border: 1px solid rgba(114, 9, 183, 0.3); }
</style>
""", unsafe_allow_html=True)

# Add repository paths
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

possible_roots = [
    APP_DIR,
    Path(r"C:\AIT MAJOR PROJECT\new fundus dataset800"),
    Path(r"C:\AIT MAJOR PROJECT\FUNDUS DTASET\oct-retinal-segmentation")
]

DATASET_ROOT = None
for r in possible_roots:
    candidate = r / "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"
    if candidate.exists():
        DATASET_ROOT = candidate
        break

DEFAULT_CKPT_UNET = None
for r in possible_roots:
    candidate = r / "outputs" / "fives_pipeline" / "best_fives_unet.pth"
    if candidate.exists():
        DEFAULT_CKPT_UNET = candidate
        break

DEFAULT_CKPT_RESNET = None
for r in possible_roots:
    candidate = r / "fundus" / "vessel_module" / "checkpoints" / "best_model.pth"
    if candidate.exists():
        DEFAULT_CKPT_RESNET = candidate
        break

# Import Model 1 (Custom U-Net)
from src.fives_pipeline.model import UNet
from src.fives_pipeline.preprocessing import FIVESTransform

# Import Model 2 & Friend's Pipeline Modules
sys.path.insert(0, str(APP_DIR / "fundus" / "vessel_module" / "src"))
for r in possible_roots:
    p = r / "fundus" / "vessel_module" / "src"
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_unet import ResNet34UNet
from inference import predict_full_image_tiled
from postprocessing import create_fov_mask as friend_create_fov_mask
from skimage import morphology
from biomarkers import extract_all_biomarkers

# ----------------- Model Loaders with Cache -----------------
@st.cache_resource(show_spinner="Loading Model 1 (Custom U-Net) & Model 2 (ResNet34-UNet)...")
def load_both_models(path_u: str, path_r: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model 1
    m1 = UNet(in_channels=3, out_channels=1)
    if os.path.exists(path_u):
        sd1 = torch.load(path_u, map_location=device)
        if isinstance(sd1, dict) and "model_state_dict" in sd1: sd1 = sd1["model_state_dict"]
        elif isinstance(sd1, dict) and "state_dict" in sd1: sd1 = sd1["state_dict"]
        m1.load_state_dict(sd1)
        m1.to(device)
        m1.eval()
    else:
        m1 = None

    # Model 2
    m2 = ResNet34UNet(num_classes=1)
    if os.path.exists(path_r):
        sd2 = torch.load(path_r, map_location=device)
        if isinstance(sd2, dict) and "model_state_dict" in sd2: sd2 = sd2["model_state_dict"]
        elif isinstance(sd2, dict) and "state_dict" in sd2: sd2 = sd2["state_dict"]
        m2.load_state_dict(sd2)
        m2.to(device)
        m2.eval()
    else:
        m2 = None

    return m1, m2, device

# ----------------- Model 1 Helper Functions -----------------
def model1_fov_mask(image_np: np.ndarray, threshold: int = 15) -> np.ndarray:
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if image_np.ndim == 3 else image_np
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fov_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return (fov_mask > 0).astype(np.uint8)

def model1_clean_speckles(binary_mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8))
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1
    return cleaned

def model1_color_overlay(base_img: Image.Image, vessel_mask: np.ndarray, color_hex: str = "#00FF66", alpha: float = 0.65) -> Image.Image:
    base_np = np.array(base_img).copy()
    h, w = base_np.shape[:2]
    if vessel_mask.shape != (h, w):
        vessel_mask = cv2.resize(vessel_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    hex_clean = color_hex.lstrip('#')
    c_rgb = np.array([int(hex_clean[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
    overlay = base_np.copy().astype(np.float32)
    mask_indices = vessel_mask > 0
    overlay[mask_indices] = (1.0 - alpha) * overlay[mask_indices] + alpha * c_rgb
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

def model1_biomarkers(vessel_mask: np.ndarray, fov_mask: np.ndarray = None):
    total_vessel_pixels = int(np.sum(vessel_mask > 0))
    total_retina_pixels = int(np.sum(fov_mask > 0)) if (fov_mask is not None and np.sum(fov_mask) > 0) else vessel_mask.size
    vessel_density_pct = (total_vessel_pixels / max(1, total_retina_pixels)) * 100.0
    contours, _ = cv2.findContours(vessel_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    h, w = vessel_mask.shape
    cy, cx = h // 2, w // 2
    q1 = int(np.sum(vessel_mask[:cy, cx:]))
    q2 = int(np.sum(vessel_mask[:cy, :cx]))
    q3 = int(np.sum(vessel_mask[cy:, :cx]))
    q4 = int(np.sum(vessel_mask[cy:, cx:]))
    return {
        "vessel_density": vessel_density_pct,
        "vessel_pixels": total_vessel_pixels,
        "perimeter": perimeter,
        "quadrants": [q1, q2, q3, q4]
    }

# ----------------- Main App Header -----------------
st.markdown('<div class="main-title">🔬 Dual-Model Retinal AI Analysis Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Single Image Input ➔ Two Distinct Deep Learning Models Executing Their Respective Purpose-Built Outputs</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Model Checkpoints")
path_u_input = st.sidebar.text_input("Your Model: Custom U-Net (.pth)", value=str(DEFAULT_CKPT_UNET) if DEFAULT_CKPT_UNET else "")
path_r_input = st.sidebar.text_input("Friend's Model: ResNet34-UNet (.pth)", value=str(DEFAULT_CKPT_RESNET) if DEFAULT_CKPT_RESNET else "")

model_u, model_r, device = load_both_models(path_u_input, path_r_input)

sc1, sc2 = st.sidebar.columns(2)
with sc1:
    if model_u: st.markdown('<span class="badge badge-success">● Your Model Ready</span>', unsafe_allow_html=True)
    else: st.markdown('<span class="badge badge-purple">✕ Your Model Missing</span>', unsafe_allow_html=True)
with sc2:
    if model_r: st.markdown('<span class="badge badge-primary">● Friend Model Ready</span>', unsafe_allow_html=True)
    else: st.markdown('<span class="badge badge-purple">✕ Friend Model Missing</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Execution Settings")
resolution_u = st.sidebar.select_slider("Your Model Resolution", options=[256, 512], value=256)
threshold_u = st.sidebar.slider("Your Model Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
threshold_r = st.sidebar.slider("Friend's Model Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Image Source")
input_mode = st.sidebar.radio("Input Mode", ["🖼️ Sample Test Gallery", "📁 Upload Custom Image"])

test_images_dir = (DATASET_ROOT / "test" / "Original") if DATASET_ROOT else None
test_masks_dir = (DATASET_ROOT / "test" / "Ground truth") if DATASET_ROOT else None

active_image = None
ground_truth_image = None
sample_name = "Uploaded Fundus Image"

if input_mode == "🖼️ Sample Test Gallery":
    if test_images_dir and test_images_dir.exists():
        sample_files = sorted([f for f in test_images_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        if sample_files:
            selected_file = st.sidebar.selectbox("Choose a FIVES Test Image", options=[f.name for f in sample_files[:30]])
            img_path = test_images_dir / selected_file
            active_image = Image.open(img_path).convert("RGB")
            sample_name = selected_file
            gt_path = test_masks_dir / selected_file
            if gt_path.exists():
                ground_truth_image = Image.open(gt_path).convert("L")
else:
    uploaded_file = st.file_uploader("Upload a Single Retinal Fundus Photograph (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg", "tif"])
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file).convert("RGB")
        sample_name = uploaded_file.name

# ----------------- Execution -----------------
if active_image is not None and model_u is not None and model_r is not None:
    with st.spinner("Processing single image through both models simultaneously..."):
        # =========================================================================
        # 1. EXECUTE YOUR MODEL (Custom U-Net Global Context)
        # =========================================================================
        t0 = time.time()
        t_u = FIVESTransform(size=(resolution_u, resolution_u), augment=False, use_clahe=True)
        img_t_u, _ = t_u(active_image, np.zeros((resolution_u, resolution_u), dtype=np.float32))
        img_t_u = img_t_u.unsqueeze(0).to(device)
        with torch.no_grad():
            logits_u = model_u(img_t_u)
            probs_u = torch.sigmoid(logits_u).squeeze().cpu().numpy()
        t_elapsed_u = time.time() - t0

        vessel_u = (probs_u >= threshold_u).astype(np.uint8)
        orig_resized_u = np.array(active_image.resize((resolution_u, resolution_u)))
        fov_u = model1_fov_mask(orig_resized_u)
        vessel_u = model1_clean_speckles(vessel_u * fov_u, min_size=20)

        orig_w, orig_h = active_image.size
        mask_u_full = cv2.resize(vessel_u, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay_u = model1_color_overlay(active_image, mask_u_full, color_hex="#00FF66", alpha=0.65)
        bm_u = model1_biomarkers(vessel_u, fov_mask=fov_u)

        # =========================================================================
        # 2. EXECUTE YOUR FRIEND'S MODEL (ResNet34-UNet Patch-Tiled & Morphology)
        # =========================================================================
        t0 = time.time()
        image_np_r = np.array(active_image)
        # Run friend's tiled inference module
        prob_map_r = predict_full_image_tiled(active_image, model_r, device, patch_size=512, stride=256)
        raw_bin_r = prob_map_r > threshold_r
        fov_mask_r = friend_create_fov_mask(image_np_r)
        mask_after_fov_r = cv2.bitwise_and((raw_bin_r.astype(np.uint8)*255), fov_mask_r)
        clean_mask_r = (morphology.remove_small_objects(mask_after_fov_r > 0, min_size=100) * 255).astype(np.uint8)
        
        bm_friend = extract_all_biomarkers(clean_mask_r, fov_mask_r)
        skeleton_mask_r = bm_friend.pop("skeleton_mask")
        
        overlay_r = image_np_r.copy()
        overlay_r[clean_mask_r > 0] = [0, 255, 0] # Friend's original overlay style
        t_elapsed_r = time.time() - t0

    # ----------------- Render Both Respective Model Outputs -----------------
    st.info(f"Analyzed Image: **{sample_name}** ({orig_w}×{orig_h}) | Single Image successfully processed through both independent pipelines.")

    tab_my_model, tab_friend_model, tab_side_by_side = st.tabs([
        "🔬 YOUR MODEL: FIVES Clinical Segmentation & Quadrant Analytics",
        "🧠 FRIEND'S MODEL: ResNet34-UNet & Microvascular Biomarkers",
        "🏆 SIDE-BY-SIDE OVERVIEW (Both Models Compared)"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: YOUR MODEL (Exact Output from your previous screenshot!)
    # -------------------------------------------------------------------------
    with tab_my_model:
        st.markdown("### 🔬 Your Custom U-Net Pipeline (FIVES Global Segmentation)")
        
        # 4 Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">🩸 Vessel Area Density (VAD)</div>
                <div class="stat-value">{bm_u['vessel_density']:.2f}%</div>
                <div style="font-size:0.75rem; color:#8D99AE; margin-top:2px;">Standard Range: 9.0% – 14.5%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">📏 Vascular Footprint</div>
                <div class="stat-value">{bm_u['vessel_pixels']:,} px</div>
                <div style="font-size:0.75rem; color:#8D99AE; margin-top:2px;">Total segmented vessel pixels</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">⚡ Inference Speed</div>
                <div class="stat-value">{t_elapsed_u*1000:.0f} ms</div>
                <div style="font-size:0.75rem; color:#8D99AE; margin-top:2px;">Resolution: {resolution_u}×{resolution_u} ({device.type.upper()})</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">🎯 Model Confidence</div>
                <div class="stat-value">{np.mean(probs_u[vessel_u > 0])*100:.1f}%</div>
                <div style="font-size:0.75rem; color:#8D99AE; margin-top:2px;">Average probability on vessel tree</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Multi-Panel Visuals
        st.markdown("#### Multi-Panel Segmentation Visuals")
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            st.image(active_image, caption="1. Original Fundus Photograph", use_column_width=True)
        with col_u2:
            st.image(Image.fromarray(mask_u_full * 255), caption=f"2. U-Net Predicted Vessel Mask (T={threshold_u})", use_column_width=True)
        with col_u3:
            st.image(overlay_u, caption="3. High-Visibility Vascular Overlay", use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Quadrants & Clinical Interpretation
        st.markdown("#### Retinal Microvascular Morphometry & Quadrants")
        q_col1, q_col2 = st.columns([1, 1])
        with q_col1:
            st.markdown("##### Clinical Interpretation Guide")
            vad = bm_u['vessel_density']
            if 8.0 <= vad <= 16.0:
                st.success("✅ **Normal Vessel Density Range:** Vasculature appears consistent with healthy perfusion patterns.")
            elif vad < 8.0:
                st.warning("⚠️ **Low Vessel Density detected:** Potential indicator of non-perfusion areas, capillary dropout, or advanced diabetic retinopathy.")
            else:
                st.info("ℹ️ **Elevated Vascular Caliber / Density:** May indicate venous dilation or neovascularization.")
            st.write(f"• **Vessel Perimeter Complexity:** `{bm_u['perimeter']:,.1f}`")
            st.write(f"• **Binarization Threshold:** `{threshold_u:.2f}`")

        with q_col2:
            st.markdown("##### Vascular Quadrant Distribution")
            quad_labels = ["Superior-Temporal", "Superior-Nasal", "Inferior-Nasal", "Inferior-Temporal"]
            total_q = max(1, sum(bm_u['quadrants']))
            for label, count in zip(quad_labels, bm_u['quadrants']):
                pct = (count / total_q) * 100.0
                st.write(f"**{label}:** {pct:.1f}% ({count:,} px)")
                st.progress(float(pct / 100.0))

    # -------------------------------------------------------------------------
    # TAB 2: FRIEND'S MODEL (Exact Output from Friend's original app.py!)
    # -------------------------------------------------------------------------
    with tab_friend_model:
        st.markdown("### 🧠 Friend's ResNet34-UNet Pipeline (Tiled Inference & Microvascular Features)")
        st.markdown(f"**Inference Latency:** `{t_elapsed_r:.2f} seconds` (High-Resolution Patch-Tiled Execution)")

        # Friend's Visualizations (5 visual outputs exactly from app.py)
        st.markdown("#### Image Visualizations")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("**1. Original Fundus Image**")
            st.image(active_image, use_column_width=True)
            st.markdown("**4. Skeletonized Vessel Network**")
            st.image(skeleton_mask_r, use_column_width=True)
        with f_col2:
            st.markdown("**2. Vessel Probability Map**")
            prob_display = (prob_map_r * 255).astype(np.uint8)
            st.image(prob_display, use_column_width=True)
            st.markdown("**5. Vessel Overlay on Original Image**")
            st.image(overlay_r, use_column_width=True)
        with f_col3:
            st.markdown("**3. Clean Binary Vessel Mask**")
            st.image(clean_mask_r, use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Friend's Vascular Feature Table (Exact table from app.py)
        st.markdown("#### Image-Derived Vascular Features")
        st.warning("These are image-derived vascular features at original image resolution from the ResNet-34 patch pipeline.")
        
        biomarker_data = [
            {"Biomarker": "Vessel Density", "Value": f"{bm_friend['vessel_density_percent']:.4f}", "Unit": "%"},
            {"Biomarker": "Total Vessel Length", "Value": f"{bm_friend['total_vessel_length_pixels']:,}", "Unit": "pixels"},
            {"Biomarker": "Mean Vessel Width", "Value": f"{bm_friend['mean_vessel_width_pixels']:.4f}", "Unit": "pixels"},
            {"Biomarker": "Median Vessel Width", "Value": f"{bm_friend['median_vessel_width_pixels']:.4f}", "Unit": "pixels"},
            {"Biomarker": "Mean Tortuosity", "Value": f"{bm_friend['mean_tortuosity']:.4f}", "Unit": "ratio"},
            {"Biomarker": "Branch Points", "Value": f"{bm_friend['branch_point_count']:,}", "Unit": "count"},
            {"Biomarker": "Endpoints", "Value": f"{bm_friend['endpoint_count']:,}", "Unit": "count"},
        ]
        st.table(pd.DataFrame(biomarker_data))

    # -------------------------------------------------------------------------
    # TAB 3: SIDE-BY-SIDE OVERVIEW
    # -------------------------------------------------------------------------
    with tab_side_by_side:
        st.markdown("### 🏆 Side-by-Side Architectural Comparison")
        st.markdown("Comparing the outputs generated by both models on this exact fundus scan:")
        
        comp_c1, comp_c2, comp_c3 = st.columns(3)
        with comp_c1:
            st.markdown("#### Original Fundus Input")
            st.image(active_image, use_column_width=True)
            if ground_truth_image:
                st.markdown("#### Ground Truth Reference")
                st.image(ground_truth_image, use_column_width=True)
        with comp_c2:
            st.markdown("#### Your Model: Custom U-Net")
            st.image(Image.fromarray(mask_u_full * 255), caption=f"Binary Mask (VAD: {bm_u['vessel_density']:.2f}%)", use_column_width=True)
            st.image(overlay_u, caption="Neon Green Overlay", use_column_width=True)
        with comp_c3:
            st.markdown("#### Friend's Model: ResNet34-UNet")
            st.image(clean_mask_r, caption=f"Binary Mask (VAD: {bm_friend['vessel_density_percent']:.2f}%)", use_column_width=True)
            st.image(skeleton_mask_r, caption=f"Skeleton Network ({bm_friend['branch_point_count']} Branch Points)", use_column_width=True)

        st.markdown("---")
        st.markdown("#### Summary Comparison Table")
        comparison_table = [
            {"Metric / Feature": "Architecture", "Your Model": "Custom U-Net (31M Params)", "Friend's Model": "ResNet34-UNet (21M Params)"},
            {"Metric / Feature": "Inference Paradigm", "Your Model": "Global Full-Context (1 Shot)", "Friend's Model": "Sliding-Window 512 Tiling"},
            {"Metric / Feature": "Vessel Area Density (VAD)", "Your Model": f"{bm_u['vessel_density']:.2f}%", "Friend's Model": f"{bm_friend['vessel_density_percent']:.2f}%"},
            {"Metric / Feature": "Inference Latency", "Your Model": f"{t_elapsed_u*1000:.0f} ms", "Friend's Model": f"{t_elapsed_r*1000:.0f} ms"},
            {"Metric / Feature": "Specialized Output", "Your Model": "Vascular Quadrants Perfusion & Clinical Classification", "Friend's Model": "Centerline Skeleton, Tortuosity, Branch & Endpoint Counts"}
        ]
        st.table(pd.DataFrame(comparison_table))

elif active_image is None:
    st.info("👈 Upload a fundus image or pick one from the **Sample Test Gallery** in the sidebar to execute both models.")
