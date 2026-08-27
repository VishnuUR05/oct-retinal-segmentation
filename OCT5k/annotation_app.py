import os
import datetime
import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from src.annotation.inference import load_model, get_transform, run_inference
from src.annotation.geometry import initialize_control_points, interpolate_boundary, generate_mask, check_invalid_crossings

st.set_page_config(page_title="OCT Retinal Annotation Tool", layout="wide")

# --- Constants & State Initialization ---
BOUNDARIES = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
B_COLORS = {'ILM': (255, 0, 0), 'OPL': (0, 255, 0), 'IS-OS': (0, 0, 255), 'IBRPE': (0, 255, 255), 'OBRPE': (255, 0, 255)}

if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'control_points' not in st.session_state:
    st.session_state.control_points = {b: [] for b in BOUNDARIES}
if 'boundary_validity' not in st.session_state:
    st.session_state.boundary_validity = {b: True for b in BOUNDARIES}
if 'last_click' not in st.session_state:
    st.session_state.last_click = None
if 'model_preds' not in st.session_state:
    st.session_state.model_preds = None
if 'unsaved_changes' not in st.session_state:
    st.session_state.unsaved_changes = False

# --- Data Loading ---
@st.cache_data
def load_manifest():
    manifest_path = "outputs/external_annotation/selected_images.csv"
    if not os.path.exists(manifest_path):
        return None
    return pd.read_csv(manifest_path)

@st.cache_data
def load_metadata():
    meta_path = "outputs/external_annotation/metadata.csv"
    if not os.path.exists(meta_path):
        return None
    return pd.read_csv(meta_path)

def save_metadata(df):
    meta_path = "outputs/external_annotation/metadata.csv"
    df.to_csv(meta_path, index=False)

manifest = load_manifest()
meta = load_metadata()

if manifest is None or meta is None:
    st.error("Manifest or metadata not found. Please ensure data_selection.py was run.")
    st.stop()

# --- Model Loading ---
model, device = load_model()
transform = get_transform()

# --- Helper Functions ---
def load_current_image():
    row = manifest.iloc[st.session_state.current_idx]
    img_rgb, raw_bnd = run_inference(row['image_path'], model, device, transform)
    st.session_state.model_preds = raw_bnd
    
    # Init points if empty (e.g. first load)
    if not any(st.session_state.control_points.values()):
        for b in BOUNDARIES:
            st.session_state.control_points[b] = initialize_control_points(raw_bnd[b], step=32)
            st.session_state.boundary_validity[b] = True
    return img_rgb, row

def reset_current_boundary(b):
    st.session_state.control_points[b] = initialize_control_points(st.session_state.model_preds[b], step=32)
    st.session_state.boundary_validity[b] = True
    st.session_state.unsaved_changes = True

def reset_all_boundaries():
    for b in BOUNDARIES:
        reset_current_boundary(b)

def save_annotation(row, img_rgb, dense_boundaries, mask, features):
    cat_dir = os.path.join("outputs", "external_annotation", row['category'])
    
    # Save Mask
    mask_name = f"{row['filename']}_mask.png"
    mask_path = os.path.join(cat_dir, "masks", mask_name)
    cv2.imwrite(mask_path, mask)
    
    # Save Boundaries CSV
    df_b = pd.DataFrame({'x': np.arange(512)})
    for b in BOUNDARIES:
        df_b[b] = dense_boundaries[b]
        df_b[f"{b.replace('-','')}_valid"] = st.session_state.boundary_validity[b]
    
    csv_name = f"{row['filename']}_boundaries.csv"
    csv_path = os.path.join(cat_dir, "boundaries", csv_name)
    df_b.to_csv(csv_path, index=False)
    
    # Update Metadata
    idx = meta.index[meta['image_name'] == row['filename']].tolist()[0]
    meta.at[idx, 'annotation_status'] = 'SAVED'
    meta.at[idx, 'saved_timestamp'] = datetime.datetime.now().isoformat()
    meta.at[idx, 'mask_path'] = mask_path
    meta.at[idx, 'boundary_path'] = csv_path
    
    val_str = ";".join([f"{b}:{st.session_state.boundary_validity[b]}" for b in BOUNDARIES])
    meta.at[idx, 'boundary_validity'] = val_str
    save_metadata(meta)
    
    # Save Features
    feat_path = "outputs/external_annotation/structural_features.csv"
    feat_row = {'image_name': row['filename'], 'category': row['category']}
    feat_row.update(features)
    
    if os.path.exists(feat_path):
        df_f = pd.read_csv(feat_path)
        # Drop existing row for this image if it exists
        df_f = df_f[df_f['image_name'] != row['filename']]
        df_f = pd.concat([df_f, pd.DataFrame([feat_row])], ignore_index=True)
    else:
        df_f = pd.DataFrame([feat_row])
    df_f.to_csv(feat_path, index=False)
    
    st.session_state.unsaved_changes = False
    st.success("Annotation Saved Successfully!")

# --- UI Layout ---
st.sidebar.title("Navigation")

cat_filter = st.sidebar.selectbox("Filter Category", ['All'] + list(manifest['category'].unique()))
if cat_filter != 'All':
    filtered_indices = manifest.index[manifest['category'] == cat_filter].tolist()
else:
    filtered_indices = manifest.index.tolist()

if st.session_state.current_idx not in filtered_indices:
    if len(filtered_indices) > 0:
        st.session_state.current_idx = filtered_indices[0]

# Previous / Next Logic
col1, col2 = st.sidebar.columns(2)
if col1.button("⬅️ Previous"):
    if st.session_state.unsaved_changes:
        st.sidebar.warning("You have unsaved changes!")
    else:
        curr_pos = filtered_indices.index(st.session_state.current_idx)
        if curr_pos > 0:
            st.session_state.current_idx = filtered_indices[curr_pos - 1]
            st.session_state.control_points = {b: [] for b in BOUNDARIES} # Force reload
            st.rerun()

if col2.button("Next ➡️"):
    if st.session_state.unsaved_changes:
        st.sidebar.warning("You have unsaved changes!")
    else:
        curr_pos = filtered_indices.index(st.session_state.current_idx)
        if curr_pos < len(filtered_indices) - 1:
            st.session_state.current_idx = filtered_indices[curr_pos + 1]
            st.session_state.control_points = {b: [] for b in BOUNDARIES} # Force reload
            st.rerun()

if st.sidebar.button("⏭️ Skip Image (Mark as SKIPPED)"):
    row = manifest.iloc[st.session_state.current_idx]
    idx = meta.index[meta['image_name'] == row['filename']].tolist()[0]
    meta.at[idx, 'annotation_status'] = 'SKIPPED'
    save_metadata(meta)
    st.session_state.unsaved_changes = False
    st.sidebar.success("Marked as Skipped.")

# Status Info
row = manifest.iloc[st.session_state.current_idx]
status = meta.loc[meta['image_name'] == row['filename'], 'annotation_status'].values[0]

st.sidebar.markdown("---")
st.sidebar.write(f"**Category:** {row['category']}")
st.sidebar.write(f"**Image:** {st.session_state.current_idx + 1} / {len(manifest)}")
st.sidebar.write(f"**Filename:** {row['filename']}")
st.sidebar.write(f"**Status:** {status}")
st.sidebar.markdown("---")
st.sidebar.write(f"**Device:** {device.type.upper()}")
if device.type == 'cuda':
    st.sidebar.write(f"**GPU:** {torch.cuda.get_device_name(0)}")

# Load Image & Run Inference
img_rgb, row = load_current_image()

# --- Editing Tools ---
st.subheader("Boundary Correction Tools")
tool_col1, tool_col2, tool_col3 = st.columns(3)

with tool_col1:
    active_b = st.radio("Active Boundary", BOUNDARIES)
    
with tool_col2:
    action_mode = st.radio("Click Action", ["Add Point", "Move Nearest", "Delete Nearest"])
    
with tool_col3:
    is_valid = st.checkbox("Boundary is Valid / Visible", value=st.session_state.boundary_validity[active_b])
    if is_valid != st.session_state.boundary_validity[active_b]:
        st.session_state.boundary_validity[active_b] = is_valid
        st.session_state.unsaved_changes = True
        st.rerun()
        
    if st.button("Reset Current Boundary"):
        reset_current_boundary(active_b)
        st.rerun()
    if st.button("Reset ALL Boundaries"):
        reset_all_boundaries()
        st.rerun()

# --- Dense Interpolation ---
dense_bnd = {}
for b in BOUNDARIES:
    if st.session_state.boundary_validity[b]:
        dense_bnd[b] = interpolate_boundary(st.session_state.control_points[b])
    else:
        dense_bnd[b] = np.full(512, np.nan)

# --- Drawing the Canvas ---
canvas_img = img_rgb.copy()
for b in BOUNDARIES:
    if not st.session_state.boundary_validity[b]:
        continue
        
    pts = np.array([[[x, int(y)]] for x, y in enumerate(dense_bnd[b]) if not np.isnan(y)], dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(canvas_img, [pts], False, B_COLORS[b], 1)
        
    # Draw control points
    for cp in st.session_state.control_points[b]:
        rad = 4 if b == active_b else 2
        cv2.circle(canvas_img, (cp['x'], int(cp['y'])), rad, B_COLORS[b], -1)

# --- Render Image and Handle Clicks ---
st.markdown("### Click Image to Edit Control Points")
click = streamlit_image_coordinates(canvas_img, key=f"canvas_{st.session_state.current_idx}")

if click is not None:
    click_id = f"{click['x']}_{click['y']}"
    if click_id != st.session_state.last_click:
        st.session_state.last_click = click_id
        st.session_state.unsaved_changes = True
        
        cx, cy = click['x'], click['y']
        pts_list = st.session_state.control_points[active_b]
        
        if action_mode == "Add Point":
            # Don't add if a point at exactly x already exists
            if not any(p['x'] == cx for p in pts_list):
                pts_list.append({'x': cx, 'y': cy})
                
        elif action_mode == "Move Nearest" and len(pts_list) > 0:
            # Find nearest x
            closest = min(pts_list, key=lambda p: abs(p['x'] - cx))
            # If within reasonable x-distance, move its y and maybe x
            closest['x'] = cx
            closest['y'] = cy
            
        elif action_mode == "Delete Nearest" and len(pts_list) > 0:
            closest = min(pts_list, key=lambda p: abs(p['x'] - cx))
            if abs(closest['x'] - cx) < 20: # deletion radius
                pts_list.remove(closest)
                
        st.rerun()

# --- Verification & Mask Generation ---
st.markdown("---")
st.subheader("Verification & Mask Generation")

mask = generate_mask(dense_bnd)
errors = check_invalid_crossings(dense_bnd)

if errors:
    for e in errors:
        st.error(f"ANATOMICAL ERROR: {e}. Please correct this before saving.")

v_col1, v_col2, v_col3 = st.columns(3)
with v_col1:
    st.image(img_rgb, caption="Original OCT", use_container_width=True)
with v_col2:
    # Colorize mask
    mask_colors = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]
    ], dtype=np.uint8)
    st.image(mask_colors[mask], caption="Generated 6-Class Mask", use_container_width=True)
with v_col3:
    overlay = cv2.addWeighted(img_rgb, 0.7, mask_colors[mask], 0.3, 0)
    for b in BOUNDARIES:
        if st.session_state.boundary_validity[b]:
            pts = np.array([[[x, int(y)]] for x, y in enumerate(dense_bnd[b]) if not np.isnan(y)], dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], False, B_COLORS[b], 1)
    st.image(overlay, caption="Final Overlay", use_container_width=True)

# --- Features & Save ---
def calc_all_features(bnd):
    features = {}
    layers = [('ILM_OPL', 'ILM', 'OPL'), ('OPL_ISOS', 'OPL', 'IS-OS'), ('ISOS_IBRPE', 'IS-OS', 'IBRPE'), ('IBRPE_OBRPE', 'IBRPE', 'OBRPE'), ('Total_Retinal', 'ILM', 'OBRPE')]
    
    for name, t, b in layers:
        thick = np.maximum(bnd[b] - bnd[t], 0)
        thick_clean = thick[~np.isnan(thick)]
        
        c_thick = thick[128:384]
        c_thick_clean = c_thick[~np.isnan(c_thick)]
        
        for pfx, arr in [("", thick_clean), ("central_", c_thick_clean)]:
            if len(arr) > 0:
                features[f"{pfx}{name}_mean"] = float(np.mean(arr))
                features[f"{pfx}{name}_median"] = float(np.median(arr))
                features[f"{pfx}{name}_min"] = float(np.min(arr))
                features[f"{pfx}{name}_max"] = float(np.max(arr))
                features[f"{pfx}{name}_std"] = float(np.std(arr))
            else:
                features[f"{pfx}{name}_mean"] = np.nan
    return features

features = calc_all_features(dense_bnd)

st.write("**Calculated Features (Mean px):**")
f_str = f"ILM→OPL: {features.get('ILM_OPL_mean', np.nan):.1f} | OPL→IS-OS: {features.get('OPL_ISOS_mean', np.nan):.1f} | Total: {features.get('Total_Retinal_mean', np.nan):.1f}"
st.info(f_str)

if st.button("💾 SAVE ANNOTATION", type="primary"):
    if errors:
        st.error("Cannot save with anatomical errors. Please fix crossing boundaries.")
    else:
        save_annotation(row, img_rgb, dense_bnd, mask, features)

if st.session_state.unsaved_changes:
    st.warning("⚠️ You have unsaved changes! Click 'Save Annotation' before pressing Next/Previous.")
