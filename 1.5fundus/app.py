import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import pandas as pd
import cv2
import importlib

# Add vessel module source path to import inference, postprocessing and biomarkers
sys.path.append(r"F:\Ait Major Project\fundus\vessel_module\src")

import torch
import model_unet
import inference
import postprocessing
import biomarkers

# Force reload modules in case Streamlit cached old buggy versions!
importlib.reload(inference)
importlib.reload(postprocessing)
importlib.reload(biomarkers)

from inference import predict_full_image_tiled
from postprocessing import postprocess_vessel_mask, create_fov_mask
from biomarkers import extract_all_biomarkers
from data_loading import FIVESPatchDataset
from config import VAL_IMG_DIR, VAL_MASK_DIR

from image_validation import validate_fundus_image, check_image_quality

st.set_page_config(page_title="Fundus Vessel Segmentation", layout="wide")

@st.cache_resource
def load_vessel_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_unet.ResNet34UNet(num_classes=1)
    checkpoint_path = r"F:\Ait Major Project\fundus\vessel_module\checkpoints\best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_vessel_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Fundus Analysis", "Model Validation"])

st.sidebar.title("Model Settings")
st.sidebar.info(f"Inference module loaded from: {inference.__file__}")

if app_mode == "Fundus Analysis":
    st.title("👁️ Fundus Analysis Mode")
    st.markdown("Upload a retinal fundus image to segment vessels and extract vascular features.")
    st.info("Note: Objective segmentation metrics such as Dice and IoU require a reference ground-truth annotation. For new images without an expert vessel mask, these metrics cannot be calculated automatically.")
    with st.sidebar.expander("Advanced / Developer Options"):
        enable_diagnostics = st.checkbox("Enable segmentation diagnostics", value=False)
    
    uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["jpg", "png", "jpeg", "tif"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        is_valid_fundus, error_msg = validate_fundus_image(image)
        if not is_valid_fundus:
            st.error(error_msg)
            st.stop()
            
        st.success("Image uploaded successfully and passed fundus validation!")
        
        is_good_quality, warning_msg = check_image_quality(image)
        if not is_good_quality:
            st.warning(warning_msg)
            
        with st.spinner("Running high-resolution tiled inference..."):
            pixel_tracking = {}
            prob_map = predict_full_image_tiled(image, model, device, patch_size=512, stride=256)
            
            raw_binary_mask = prob_map > 0.5
            pixel_tracking['After Threshold (0.5)'] = np.sum(raw_binary_mask)
            
            fov_mask = create_fov_mask(image_np)
            pixel_tracking['FOV area'] = np.sum(fov_mask > 0)
            
            mask_after_fov = cv2.bitwise_and((raw_binary_mask.astype(np.uint8)*255), fov_mask)
            pixel_tracking['After FOV intersection'] = np.sum(mask_after_fov > 0)
            
            from skimage import morphology
            mask_after_small_obj = morphology.remove_small_objects(mask_after_fov > 0, min_size=100)
            pixel_tracking['After remove_small_objects'] = np.sum(mask_after_small_obj)
            
            clean_mask = (mask_after_small_obj * 255).astype(np.uint8)
            pixel_tracking['Final mask pixels'] = np.sum(clean_mask > 0)
            
            bm = extract_all_biomarkers(clean_mask, fov_mask)
            skeleton_mask = bm.pop("skeleton_mask")
            pixel_tracking['Skeleton pixels'] = np.sum(skeleton_mask > 0)
            
        if enable_diagnostics:
            st.subheader("Segmentation Diagnostics - DEBUG MODE")
            st.code(f"Inference module path: {inference.__file__}")
            
            prob_flat = prob_map.flatten()
            p_vals = np.percentile(prob_flat, [50, 75, 90, 95, 99, 99.5, 99.9])
            
            st.markdown("### Exact Numerical Diagnostics")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown("**Probability Map Stats**")
                st.write(f"Min: {prob_flat.min():.4f}")
                st.write(f"Max: {prob_flat.max():.4f}")
                st.write(f"Mean: {prob_flat.mean():.4f}")
                st.write(f"Std: {prob_flat.std():.4f}")
            with col_s2:
                st.markdown("**Probability Percentiles**")
                st.write(f"50th: {p_vals[0]:.4f}")
                st.write(f"75th: {p_vals[1]:.4f}")
                st.write(f"90th: {p_vals[2]:.4f}")
                st.write(f"95th: {p_vals[3]:.4f}")
            with col_s3:
                st.write(f"99th: {p_vals[4]:.4f}")
                st.write(f"99.5th: {p_vals[5]:.4f}")
                st.write(f"99.9th: {p_vals[6]:.4f}")
                
            st.markdown("### Thresholds Analysis")
            thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
            thresh_data = []
            fov_area = pixel_tracking['FOV area']
            for t in thresholds:
                px = np.sum(prob_flat > t)
                pct = (px / fov_area * 100) if fov_area > 0 else 0
                thresh_data.append({"Threshold": t, "Vessel Pixels": px, "% of FOV": pct})
            st.table(pd.DataFrame(thresh_data))
            
            st.markdown("### Pipeline Pixel Tracking (Where do they disappear?)")
            st.json(pixel_tracking)
            
            st.markdown("### Diagnostic Views")
            diag_cols = st.columns(4)
            
            with diag_cols[0]:
                st.markdown("**1. Original Image**")
                st.image(image, use_container_width=True)
                st.markdown("**5. Binary Mask (Before Post-processing)**")
                st.image((raw_binary_mask.astype(np.uint8)*255), use_container_width=True)
                st.markdown("**9. Final Vessel Mask**")
                st.image(clean_mask, use_container_width=True)
                
            with diag_cols[1]:
                st.markdown("**2. Raw Probability Map**")
                prob_display = (prob_map * 255).astype(np.uint8)
                st.image(prob_display, use_container_width=True)
                st.markdown("**6. FOV Mask**")
                st.image(fov_mask, use_container_width=True)
                st.markdown("**10. Skeleton**")
                st.image(skeleton_mask, use_container_width=True)
                
            with diag_cols[2]:
                st.markdown("**3. Probability Heatmap**")
                heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.image(heatmap, use_container_width=True)
                st.markdown("**7. Mask After FOV processing**")
                st.image(mask_after_fov, use_container_width=True)
                st.markdown("**11. Vessel Overlay**")
                overlay = image_np.copy()
                overlay[clean_mask > 0] = [0, 255, 0]
                st.image(overlay, use_container_width=True)
                
            with diag_cols[3]:
                st.markdown("**4. Histogram**")
                st.bar_chart(np.histogram(prob_flat, bins=50, range=(0.0, 1.0))[0])
                st.markdown("**8. Mask After small-object removal**")
                st.image((mask_after_small_obj*255).astype(np.uint8), use_container_width=True)
                
        else:
            st.subheader("Image Visualizations")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**1. Original Fundus Image**")
                st.image(image, use_container_width=True)
                st.markdown("**4. Skeletonized Vessel Network**")
                st.image(skeleton_mask, use_container_width=True)
                
            with col2:
                st.markdown("**2. Vessel Probability Map**")
                prob_display = (prob_map * 255).astype(np.uint8)
                st.image(prob_display, use_container_width=True)
                st.markdown("**5. Vessel Overlay on Original Image**")
                overlay = image_np.copy()
                overlay[clean_mask > 0] = [0, 255, 0]
                st.image(overlay, use_container_width=True)
                
            with col3:
                st.markdown("**3. Clean Binary Vessel Mask**")
                st.image(clean_mask, use_container_width=True)
            
        vessel_percentage = bm["vessel_density_percent"]
        if vessel_percentage < 0.1 or np.sum(clean_mask) < 500:
            st.error(f"Segmentation quality check failed: insufficient vessel pixels detected ({vessel_percentage}% density, {np.sum(clean_mask>0)} px). Biomarkers were not calculated.")
            st.stop()
            
        st.subheader("Image-Derived Vascular Features")
        st.warning("These are image-derived vascular features at the original image resolution. They are NOT direct Alzheimer's diagnostic biomarkers or physical measurements (e.g., mm) as physical calibration is unavailable.")
            
        biomarker_data = [
            {"Biomarker": "Vessel Density", "Value": bm["vessel_density_percent"], "Unit": "%"},
            {"Biomarker": "Total Vessel Length", "Value": bm["total_vessel_length_pixels"], "Unit": "pixels"},
            {"Biomarker": "Mean Vessel Width", "Value": bm["mean_vessel_width_pixels"], "Unit": "pixels"},
            {"Biomarker": "Median Vessel Width", "Value": bm["median_vessel_width_pixels"], "Unit": "pixels"},
            {"Biomarker": "Mean Tortuosity", "Value": bm["mean_tortuosity"], "Unit": "ratio"},
            {"Biomarker": "Branch Points", "Value": bm["branch_point_count"], "Unit": "count"},
            {"Biomarker": "Endpoints", "Value": bm["endpoint_count"], "Unit": "count"},
        ]
        
        df = pd.DataFrame(biomarker_data)
        st.table(df)

elif app_mode == "Model Validation":
    st.title("⚖️ Model Validation Mode")
    st.markdown("""
    **How do I know the model is correct?**
    - The FIVES ground-truth vessel mask is the expert reference annotation.
    - The AI prediction is compared directly with this reference pixel-by-pixel.
    - A higher Dice and IoU score indicates stronger agreement with the reference annotation.
    """)
    st.info("Objective metrics such as Dice and IoU require a reference ground-truth annotation. For a newly uploaded patient image without an expert vessel mask, true segmentation accuracy cannot be automatically calculated.")

    val_split_path = r"F:\Ait Major Project\fundus\outputs\fives_vessel_project\validation_split.csv"
    if not os.path.exists(val_split_path):
        st.error("Validation split file not found!")
        st.stop()
        
    val_df = pd.read_csv(val_split_path)
    # The CSV has columns: Image Path, Mask Path
    image_paths = val_df['Image Path'].tolist()
    mask_paths = val_df['Mask Path'].tolist()
    
    val_option = st.radio("Select Validation Option", ["Evaluate Single Sample", "Evaluate Complete Validation Set"])
    
    def calculate_metrics(tp, fp, fn, tn):
        eps = 1e-6
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)
        return dice, iou, precision, recall, specificity
    
    if val_option == "Evaluate Single Sample":
        st.subheader("Evaluate Single Sample")
        
        image_options = {os.path.basename(path): (path, mask_paths[i]) for i, path in enumerate(image_paths)}
        selected_filename = st.selectbox("Select FIVES Validation Sample", list(image_options.keys()))
        selected_img_path, selected_mask_path = image_options[selected_filename]
        selected_stem = os.path.splitext(selected_filename)[0]
        
        if st.button("Run Validation"):
            with st.spinner("Running full-resolution tiled inference..."):
                if not os.path.exists(selected_img_path):
                    st.error(f"Image not found at {selected_img_path}")
                    st.stop()
                if not os.path.exists(selected_mask_path):
                    st.error(f"Ground-truth mask not found at {selected_mask_path}")
                    st.stop()

                # Load images
                img_pil = Image.open(selected_img_path).convert("RGB")
                gt_mask_pil = Image.open(selected_mask_path).convert("L")
                img_np = np.array(img_pil)
                
                gt_mask_np = np.array(gt_mask_pil)
                gt_binary = gt_mask_np > 127
                
                # Inference
                prob_map = predict_full_image_tiled(img_pil, model, device, patch_size=512, stride=256)
                pred_binary = prob_map > 0.50
                
                # Metrics logic
                TP_mask = pred_binary & gt_binary
                FP_mask = pred_binary & (~gt_binary)
                FN_mask = (~pred_binary) & gt_binary
                TN_mask = (~pred_binary) & (~gt_binary)
                
                TP = np.sum(TP_mask)
                FP = np.sum(FP_mask)
                FN = np.sum(FN_mask)
                TN = np.sum(TN_mask)
                
                dice, iou, precision, recall, specificity = calculate_metrics(TP, FP, FN, TN)
                
                # Color-coded comparison map
                # GREEN: TP, RED: FN, BLUE: FP, BLACK: TN
                error_map = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
                error_map[TP_mask] = [0, 255, 0]
                error_map[FN_mask] = [255, 0, 0]
                error_map[FP_mask] = [0, 0, 255]
                
                # Layout
                st.markdown("### Metrics (Full Image Evaluation, Threshold 0.50)")
                metrics_df = pd.DataFrame([{
                    "Dice Score / F1": f"{dice:.4f}",
                    "IoU / Jaccard": f"{iou:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "Specificity": f"{specificity:.4f}"
                }])
                st.table(metrics_df)
                
                count_df = pd.DataFrame([{
                    "True Positives (Green)": f"{TP:,}",
                    "False Positives (Blue)": f"{FP:,}",
                    "False Negatives (Red)": f"{FN:,}",
                    "True Negatives (Black)": f"{TN:,}"
                }])
                st.table(count_df)
                
                st.markdown("### Visual Comparison")
                st.markdown("🟩 **Green**: True Positive | 🟥 **Red**: False Negative | 🟦 **Blue**: False Positive")
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                with col1:
                    st.markdown("**1. Original Fundus Image**")
                    st.image(img_pil, use_container_width=True)
                with col2:
                    st.markdown("**2. Reference Ground Truth Mask**")
                    st.image(gt_mask_pil, use_container_width=True)
                with col3:
                    st.markdown("**3. AI Predicted Vessel Mask**")
                    st.image((pred_binary*255).astype(np.uint8), use_container_width=True)
                with col4:
                    st.markdown("**4. Pixel-wise Comparison / Error Map**")
                    st.image(error_map, use_container_width=True)
                    
                # Save outputs securely to outputs dir
                save_dir = r"F:\Ait Major Project\fundus\outputs\fives_vessel_project\validation_ui_exports"
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"{selected_stem}_error_map.png"), cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR))
                st.success(f"Error map temporarily saved to project outputs directory.")

    elif val_option == "Evaluate Complete Validation Set":
        st.subheader("Evaluate Complete Patch Validation Set")
        st.info("This evaluation completely matches the official Phase 5 protocol using the strictly held-out 512x512 validation patches. The resulting cumulative metrics will be comparable to the formally reported baseline metrics.")
        
        if st.button("Start Full Validation Evaluation"):
            val_dataset = FIVESPatchDataset(VAL_IMG_DIR, VAL_MASK_DIR, is_train=False)
            total_samples = len(val_dataset)
            st.write(f"Evaluating {total_samples} held-out validation patches...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_TP = 0
            total_FP = 0
            total_FN = 0
            total_TN = 0
            per_sample_dice = []
            
            from torch.utils.data import DataLoader
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
            
            with torch.no_grad():
                samples_processed = 0
                for images, masks in val_loader:
                    img_tensor = images.to(device)
                    mask_tensor = masks.to(device)
                    
                    logits = model(img_tensor)
                    prob_map = torch.sigmoid(logits)
                    
                    for i in range(images.size(0)):
                        gt_np = mask_tensor[i].squeeze().cpu().numpy() > 0.5
                        pred_np = prob_map[i].squeeze().cpu().numpy() > 0.50
                        
                        tp = np.sum(pred_np & gt_np)
                        fp = np.sum(pred_np & (~gt_np))
                        fn = np.sum((~pred_np) & gt_np)
                        tn = np.sum((~pred_np) & (~gt_np))
                        
                        total_TP += tp
                        total_FP += fp
                        total_FN += fn
                        total_TN += tn
                        
                        eps = 1e-6
                        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                        per_sample_dice.append(dice)
                        
                        samples_processed += 1
                        
                    progress = int((samples_processed / total_samples) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {samples_processed} / {total_samples} samples...")
            
            # Cumulative
            c_dice, c_iou, c_prec, c_rec, c_spec = calculate_metrics(total_TP, total_FP, total_FN, total_TN)
            
            st.success("Validation complete!")
            
            st.markdown("### Cumulative Metrics (Calculated Live)")
            cum_df = pd.DataFrame([{
                "Cumulative Dice": f"{c_dice:.4f}",
                "Cumulative IoU": f"{c_iou:.4f}",
                "Cumulative Precision": f"{c_prec:.4f}",
                "Cumulative Recall": f"{c_rec:.4f}",
                "Cumulative Specificity": f"{c_spec:.4f}"
            }])
            st.table(cum_df)
            
            st.markdown("### Per-Sample Metrics (512x512 Patches)")
            per_sample_dice = np.array(per_sample_dice)
            per_df = pd.DataFrame([{
                "Mean Dice": f"{np.mean(per_sample_dice):.4f}",
                "Median Dice": f"{np.median(per_sample_dice):.4f}",
                "Std Deviation": f"{np.std(per_sample_dice):.4f}"
            }])
            st.table(per_df)
            
            st.markdown("### Cross-Check with Previous Phase 5 Reports")
            st.info("""
            **Previously reported Phase 5 values:**
            - Cumulative Dice: 0.9257
            - IoU: 0.8617
            - Precision: 0.9509
            - Recall: 0.9018
            - Specificity: 0.9963
            - Mean Dice: 0.8831
            - Median Dice: 0.9425
            
            **Discrepancy Analysis**: If the numbers above match perfectly, it confirms absolute consistency in the evaluation pipeline and zero data modification.
            """)
