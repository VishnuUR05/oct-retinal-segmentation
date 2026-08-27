import os
import time
import json
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.annotation.inference import load_model, get_transform

# Configurable Parameters
CONFIG = {
    'w_img': 1.0,           # Weight for image gradient
    'w_unet': 1.0,          # Weight for U-Net probability
    'w_smooth': 0.5,        # Weight for vertical smoothness penalty
    'max_vertical_step': 5, # Max jump per x-column
    'search_window': 35,    # Pixels to search around U-Net prior
    'min_boundary_spacing': 2, # Minimum pixels between boundaries
    'confidence_threshold': 0.4, # Below this is marked as UNCERTAIN
    'polarity': {
        1: 1,  # ILM (Class 0->1): Dark to Bright -> Positive Gradient
        2: -1, # OPL (Class 1->2): Bright to Dark -> Negative Gradient
        3: 1,  # IS-OS (Class 2->3): Dark to Bright -> Positive Gradient
        4: 1,  # IBRPE (Class 3->4): Dark to Bright -> Positive Gradient
        5: -1  # OBRPE (Class 4->5): Bright to Dark -> Negative Gradient
    }
}
BOUNDARY_NAMES = {1: 'ILM', 2: 'OPL', 3: 'IS-OS', 4: 'IBRPE', 5: 'OBRPE'}

def optimize_boundary_dp(cost_matrix, max_step, smooth_weight, search_mask):
    """
    Dynamic Programming shortest-path for boundary tracing.
    """
    H, W = cost_matrix.shape
    dp = np.full((H, W), np.inf, dtype=np.float32)
    path = np.zeros((H, W), dtype=np.int32)
    
    # Initialize first column
    dp[:, 0] = np.where(search_mask[:, 0], cost_matrix[:, 0], np.inf)
    
    y_idx = np.arange(H)
    shifts = np.arange(-max_step, max_step + 1)
    
    for x in range(1, W):
        prev = dp[:, x-1]
        if np.all(np.isinf(prev)):
            continue
            
        stacked = np.full((len(shifts), H), np.inf, dtype=np.float32)
        for i, s in enumerate(shifts):
            if s < 0:
                stacked[i, -s:] = prev[:s]
            elif s > 0:
                stacked[i, :-s] = prev[s:]
            else:
                stacked[i, :] = prev
            stacked[i, :] += smooth_weight * (s ** 2)
            
        best_shift_idx = np.argmin(stacked, axis=0)
        best_prev_cost = stacked[best_shift_idx, y_idx]
        
        # Apply cost only within search mask
        valid_nodes = search_mask[:, x] & ~np.isinf(best_prev_cost)
        dp[valid_nodes, x] = cost_matrix[valid_nodes, x] + best_prev_cost[valid_nodes]
        
        # Record path
        path[:, x] = np.clip(y_idx + shifts[best_shift_idx], 0, H-1)

    # Backtrack
    best_final_y = np.argmin(dp[:, -1])
    if np.isinf(dp[best_final_y, -1]):
        return np.full(W, np.nan)
        
    optimal_path = np.zeros(W, dtype=np.int32)
    optimal_path[-1] = best_final_y
    for x in range(W - 1, 0, -1):
        optimal_path[x-1] = path[optimal_path[x], x]
        
    return optimal_path

def extract_raw_boundary(pred_mask, b_class):
    W = pred_mask.shape[1]
    raw = np.full(W, np.nan)
    for x in range(W):
        col = pred_mask[:, x]
        idx = np.where((col[1:] == b_class) & (col[:-1] == b_class - 1))[0]
        if len(idx) > 0:
            raw[x] = idx[0] + 1
    return raw

def evaluate_image(img_gray, unet_probs, config):
    H, W = img_gray.shape
    
    # Calculate global image gradient
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0).astype(np.float32)
    grad_y = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)
    
    raw_pred_mask = np.argmax(unet_probs, axis=0)
    
    results = {}
    prev_boundary = np.zeros(W) - config['min_boundary_spacing'] # For anatomical ordering
    
    for b_idx in range(1, 6):
        b_name = BOUNDARY_NAMES[b_idx]
        
        # 1. Raw U-Net Boundary
        raw_bnd = extract_raw_boundary(raw_pred_mask, b_idx)
        # Interpolate raw to ensure we have a center for search window
        s = pd.Series(raw_bnd)
        s_interp = s.interpolate(method='linear', limit_direction='both').values
        if np.isnan(s_interp).all():
            s_interp = np.full(W, H//2)
            
        # 2. Image Evidence
        polarity = config['polarity'][b_idx]
        grad_ev = np.maximum(grad_y, 0) if polarity == 1 else np.maximum(-grad_y, 0)
        g_max = grad_ev.max()
        if g_max > 0: grad_ev /= g_max
            
        # 3. U-Net Evidence
        p_top = np.roll(unet_probs[b_idx - 1], 1, axis=0)
        p_bot = np.roll(unet_probs[b_idx], -1, axis=0)
        unet_ev = p_top * p_bot
        u_max = unet_ev.max()
        if u_max > 0: unet_ev /= u_max
            
        # 4. Search Mask & Anatomical Constraints
        search_mask = np.zeros((H, W), dtype=bool)
        for x in range(W):
            center = int(s_interp[x])
            y_min = max(int(prev_boundary[x]) + config['min_boundary_spacing'], center - config['search_window'])
            y_max = min(H - 1, center + config['search_window'])
            if y_min <= y_max:
                search_mask[y_min:y_max+1, x] = True
                
        # 5. Ablation Study - DP Optimization
        cost_img = 1.0 - grad_ev
        cost_unet = 1.0 - unet_ev
        
        # DP: Image Only
        path_img_only = optimize_boundary_dp(cost_img, config['max_vertical_step'], config['w_smooth'], search_mask)
        # DP: U-Net Only
        path_unet_only = optimize_boundary_dp(cost_unet, config['max_vertical_step'], config['w_smooth'], search_mask)
        # DP: Combined
        total_cost = config['w_img'] * cost_img + config['w_unet'] * cost_unet
        path_combined = optimize_boundary_dp(total_cost, config['max_vertical_step'], config['w_smooth'], search_mask)
        
        # 6. Confidence & Uncertainty
        confidence_map = (unet_ev + grad_ev) / 2.0
        path_conf = np.zeros(W, dtype=np.float32)
        path_valid = np.zeros(W, dtype=bool)
        
        if not np.isnan(path_combined).all():
            for x in range(W):
                y = int(path_combined[x])
                conf = confidence_map[y, x]
                path_conf[x] = conf
                path_valid[x] = conf >= config['confidence_threshold']
                
        results[b_name] = {
            'raw': raw_bnd,
            'dp_img_only': path_img_only,
            'dp_unet_only': path_unet_only,
            'dp_combined': path_combined,
            'confidence': path_conf,
            'valid': path_valid,
            'confidence_map_full': confidence_map
        }
        
        # Update ordering constraint
        prev_boundary = np.where(~np.isnan(path_combined), path_combined, s_interp)

    return results

def plot_comparison(img_rgb, results, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 1. Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original OCT")
    
    # 2. Raw U-Net
    axes[1].imshow(img_rgb)
    axes[1].set_title("Raw U-Net Boundary")
    
    # 3. DP Combined Refined
    axes[2].imshow(img_rgb)
    axes[2].set_title("Image-Guided DP Refinement")
    
    # 4. Confidence Overlay (just showing IS-OS for example, or all)
    axes[3].imshow(img_rgb)
    axes[3].set_title("DP with Low Confidence Flagged")
    
    # 5. Ablation (Image vs Unet vs Combined for ILM)
    axes[4].imshow(img_rgb)
    axes[4].set_title("Ablation (ILM: Img=R, Unet=B, Comb=G)")
    
    colors = ['r', 'g', 'b', 'c', 'm']
    x_axis = np.arange(512)
    
    for i, b_name in enumerate(BOUNDARY_NAMES.values()):
        res = results[b_name]
        c = colors[i]
        
        # Raw
        axes[1].plot(x_axis, res['raw'], color=c, alpha=0.7, linewidth=1.5)
        # DP Combined
        axes[2].plot(x_axis, res['dp_combined'], color=c, alpha=0.9, linewidth=1.5)
        
        # Confidence flagged (Red dashed for uncertain, solid for valid)
        axes[3].plot(x_axis[res['valid']], res['dp_combined'][res['valid']], color=c, linestyle='-', marker='')
        axes[3].plot(x_axis[~res['valid']], res['dp_combined'][~res['valid']], color='r', linestyle='--', alpha=0.5)
        
    # Ablation just for ILM (Index 0 in colors, but we use R/B/G for comparison)
    res_ilm = results['ILM']
    axes[4].plot(x_axis, res_ilm['dp_img_only'], color='r', label='Img Only', alpha=0.7)
    axes[4].plot(x_axis, res_ilm['dp_unet_only'], color='b', label='UNet Only', alpha=0.7)
    axes[4].plot(x_axis, res_ilm['dp_combined'], color='g', label='Combined', alpha=0.9)
    axes[4].legend()

    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main():
    print("==================================================")
    print("AUTO-TRACE PROTOTYPE EVALUATION")
    print("==================================================")
    
    out_dir = "outputs/external_annotation/autotrace_evaluation"
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_path = "outputs/external_annotation/selected_images.csv"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Run data_selection.py first.")
        return
        
    df_manifest = pd.read_csv(manifest_path)
    # Pick 5 per category
    pilot_df = df_manifest.groupby('category').head(5)
    
    model, device = load_model()
    transform = get_transform()
    
    metrics_log = []
    
    for idx, row in pilot_df.iterrows():
        print(f"Processing: {row['category']} / {row['filename']}")
        
        img_gray = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = img_gray.shape
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        if orig_h != 512 or orig_w != 512:
            img_rgb = cv2.resize(img_rgb, (512, 512))
            img_gray = cv2.resize(img_gray, (512, 512))
            
        aug = transform(image=img_rgb)
        img_tensor = aug['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(img_tensor)
            else:
                logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        results = evaluate_image(img_gray, probs, CONFIG)
        
        plot_path = os.path.join(out_dir, f"{row['filename']}_comparison.png")
        plot_comparison(img_rgb, results, plot_path)
        
        # Calculate Metrics
        for b_name in BOUNDARY_NAMES.values():
            res = results[b_name]
            raw = res['raw']
            dp = res['dp_combined']
            conf = res['confidence']
            valid = res['valid']
            
            # Interpolate raw for fair displacement metric
            s = pd.Series(raw)
            s_raw = s.interpolate(method='linear', limit_direction='both').values
            
            disp = np.abs(dp - s_raw)
            disp_clean = disp[~np.isnan(disp)]
            
            vertical_jumps = np.abs(np.diff(dp[~np.isnan(dp)])) if len(dp[~np.isnan(dp)]) > 1 else [0]
            
            metrics_log.append({
                'category': row['category'],
                'filename': row['filename'],
                'boundary': b_name,
                'mean_displacement_px': float(np.mean(disp_clean)) if len(disp_clean) > 0 else np.nan,
                'max_displacement_px': float(np.max(disp_clean)) if len(disp_clean) > 0 else np.nan,
                'percent_changed': float(np.sum(disp_clean > 0) / 512 * 100) if len(disp_clean) > 0 else 0,
                'mean_confidence': float(np.mean(conf)),
                'percent_low_confidence': float(np.sum(~valid) / 512 * 100),
                'max_vertical_jump': float(np.max(vertical_jumps))
            })
            
    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(os.path.join(out_dir, "autotrace_metrics.csv"), index=False)
    
    print("\nPrototype Evaluation Complete.")
    print(f"Results saved to: {out_dir}")
    print("Review the contact sheets and metrics before integrating into the UI.")

if __name__ == "__main__":
    main()
