import io
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2

# Ensure Matplotlib uses a non-interactive backend
plt.switch_backend('Agg')

BOUNDARY_NAMES = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
BOUNDARY_COLORS = {
    'ILM': 'red',
    'OPL': 'green',
    'IS-OS': 'blue',
    'IBRPE': 'cyan',
    'OBRPE': 'magenta'
}

def create_pdf_report(
    img_rgb, 
    pred_mask, 
    boundaries, 
    features, 
    qc_status, 
    confidence_map, 
    confidence_stats,
    gradcam_heatmap,
    gradcam_target_name,
    fishy_warnings,
    overall_status,
    filename_meta="OCT_Image",
    dataset_meta="Unknown"
):
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        
        # --------------------------------------------------
        # PAGE 1: OCT SUMMARY
        # --------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        
        y_pos = 0.95
        plt.figtext(0.5, y_pos, "OCT Retinal Structural Analysis", ha='center', fontsize=20, weight='bold')
        y_pos -= 0.03
        plt.figtext(0.5, y_pos, "AI-Assisted Retinal Layer Segmentation", ha='center', fontsize=14, color='gray')
        
        y_pos -= 0.06
        plt.figtext(0.1, y_pos, f"Image ID: {filename_meta}", fontsize=11, weight='bold')
        y_pos -= 0.02
        plt.figtext(0.1, y_pos, f"Dataset Category: {dataset_meta}", fontsize=11)
        y_pos -= 0.02
        plt.figtext(0.1, y_pos, f"Date of Analysis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
        
        # Images section
        y_pos -= 0.40 # Leave room for images
        ax1 = fig.add_axes([0.1, y_pos, 0.38, 0.35])
        ax1.imshow(img_rgb)
        ax1.set_title("ORIGINAL OCT", fontsize=10, weight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_axes([0.52, y_pos, 0.38, 0.35])
        ax2.imshow(img_rgb)
        ax2.imshow(pred_mask, cmap='nipy_spectral', alpha=0.4)
        ax2.set_title("PREDICTED RETINAL SEGMENTATION", fontsize=10, weight='bold')
        ax2.axis('off')
        
        y_pos -= 0.08
        plt.figtext(0.1, y_pos, "OVERALL ANALYSIS STATUS", fontsize=14, weight='bold')
        
        status_clean = "GOOD" if overall_status == "PASS" else "REVIEW REQUIRED"
        status_color = "green" if status_clean == "GOOD" else "darkorange"
        
        y_pos -= 0.04
        plt.figtext(0.1, y_pos, status_clean, fontsize=16, weight='bold', color=status_color)
        
        if status_clean == "GOOD":
            status_desc = "Segmentation quality is acceptable with preserved retinal boundary ordering and low missing-boundary rates."
        else:
            status_desc = "Segmentation review is recommended due to potential technical anomalies or boundary discontinuities."
            
        y_pos -= 0.03
        plt.figtext(0.1, y_pos, status_desc, fontsize=11)
        
        y_pos -= 0.08
        plt.figtext(0.1, y_pos, "KEY STRUCTURAL FINDINGS", fontsize=14, weight='bold')
        
        y_pos -= 0.03
        findings = []
        
        gm_total = features.get("Total_Retinal_mean_pixels", np.nan)
        cm_total = features.get("central_Total_Retinal_mean_pixels", np.nan)
        if not np.isnan(gm_total) and not np.isnan(cm_total):
            if cm_total > gm_total + 1.0:
                findings.append("The central retinal region shows greater total segmented thickness than the global image average.")
            elif cm_total < gm_total - 1.0:
                findings.append("The central retinal region shows reduced total segmented thickness compared to the global image average.")
                
        cm_opl = features.get("central_OPL_ISOS_mean_pixels", np.nan)
        gm_opl = features.get("OPL_ISOS_mean_pixels", np.nan)
        if not np.isnan(gm_opl) and not np.isnan(cm_opl) and cm_opl > gm_opl + 1.0:
            findings.append("The OPL–IS-OS band is relatively thicker in the central analysis region.")
            
        if qc_status == "VALID":
            findings.append("Retinal boundaries remain anatomically ordered across the analyzed region.")
            
        if not fishy_warnings:
            findings.append("No major segmentation-quality anomaly was detected.")
            
        for finding in findings:
            plt.figtext(0.12, y_pos, f"• {finding}", fontsize=11)
            y_pos -= 0.025
            
        y_pos -= 0.04
        plt.figtext(0.1, y_pos, "AREAS REQUIRING TECHNICAL REVIEW", fontsize=12, weight='bold', color='darkred')
        y_pos -= 0.03
        if not fishy_warnings:
            plt.figtext(0.12, y_pos, "No major technical segmentation concerns identified.", fontsize=11)
        else:
            for i, w in enumerate(fishy_warnings[:3]):
                plt.figtext(0.12, y_pos, f"• {w['issue']}: {w['value']}", fontsize=11)
                y_pos -= 0.025
                
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------------
        # PAGE 2: RETINAL STRUCTURAL ANALYSIS
        # --------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        
        y_pos = 0.95
        plt.figtext(0.5, y_pos, "Retinal Structural Analysis", ha='center', fontsize=18, weight='bold')
        
        y_pos -= 0.06
        plt.figtext(0.1, y_pos, "Measurements are image-domain pixel measurements and are not calibrated to micrometers.", 
                    fontsize=10, style='italic', color='dimgray')
        
        y_pos -= 0.04
        # TABLE
        layers = ['ILM_OPL', 'OPL_ISOS', 'ISOS_IBRPE', 'IBRPE_OBRPE', 'Total_Retinal']
        display_names = ['ILM–OPL', 'OPL–IS-OS', 'IS-OS–IBRPE', 'IBRPE–OBRPE', 'Total ILM–OBRPE']
        
        table_text = f"{'Feature':<20} | {'Global (px)':<12} | {'Central (px)':<12} | {'Observation':<30}\n"
        table_text += "-"*85 + "\n"
        
        for k_raw, d_name in zip(layers, display_names):
            gm = features.get(f"{k_raw}_mean_pixels", np.nan)
            cm = features.get(f"central_{k_raw}_mean_pixels", np.nan)
            
            obs = "-"
            if not np.isnan(gm) and not np.isnan(cm):
                diff = cm - gm
                if diff > 1.0: obs = "Thicker centrally"
                elif diff < -1.0: obs = "Thinner centrally"
                else: obs = "Uniform"
                
            gm_str = f"{gm:.1f}" if not np.isnan(gm) else "N/A"
            cm_str = f"{cm:.1f}" if not np.isnan(cm) else "N/A"
            table_text += f"{d_name:<20} | {gm_str:<12} | {cm_str:<12} | {obs:<30}\n"
            
        plt.figtext(0.1, y_pos, table_text, fontsize=10, fontfamily='monospace', va='top')
        y_pos -= 0.15
        
        plt.figtext(0.1, y_pos, "STRUCTURAL INTERPRETATION", fontsize=12, weight='bold')
        y_pos -= 0.03
        interp = (
            "The analysis provides a quantitative description of retinal microstructure based on the\n"
            "segmented retinal boundaries. Differences between central and global measurements describe\n"
            "the spatial distribution of the segmented retinal bands.\n\n"
            "These measurements may be useful as candidate structural features in retinal and\n"
            "neurodegeneration research."
        )
        plt.figtext(0.1, y_pos, interp, fontsize=11, va='top')
        
        y_pos -= 0.13
        plt.figtext(0.1, y_pos, "BOUNDARY QUALITY", fontsize=12, weight='bold')
        y_pos -= 0.03
        
        bq_text = f"{'Boundary':<15} | {'Status':<15}\n"
        bq_text += "-"*35 + "\n"
        for b_name in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
            missing = np.isnan(boundaries[b_name]).mean()
            status = "GOOD" if missing < 0.05 else "REVIEW"
            bq_text += f"{b_name:<15} | {status:<15}\n"
            
        plt.figtext(0.1, y_pos, bq_text, fontsize=10, fontfamily='monospace', va='top')
        
        y_pos -= 0.15
        plt.figtext(0.1, y_pos, f"AI ATTRIBUTION MAP (Target: {gradcam_target_name})", fontsize=12, weight='bold')
        
        y_pos -= 0.32
        if gradcam_heatmap is not None:
            heatmap_resized = cv2.resize(gradcam_heatmap, (512, 512))
            
            ax3 = fig.add_axes([0.1, y_pos, 0.38, 0.30])
            ax3.imshow(heatmap_resized, cmap='jet')
            ax3.axis('off')
            
            ax4 = fig.add_axes([0.52, y_pos, 0.38, 0.30])
            ax4.imshow(img_rgb)
            ax4.imshow(heatmap_resized, cmap='jet', alpha=0.4)
            ax4.axis('off')
            
            y_pos -= 0.05
            txt = (
                "The highlighted regions represent image areas that contributed relatively strongly to the selected\n"
                "retinal-layer segmentation output. This visualization describes model attribution and does not\n"
                "establish disease or causality."
            )
            plt.figtext(0.1, y_pos, txt, fontsize=10, va='top', style='italic')
        else:
            plt.figtext(0.1, y_pos, "Grad-CAM visualization unavailable.", fontsize=11, va='top')
            
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------------
        # PAGE 3: CLINICAL CONTEXT
        # --------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        
        y_pos = 0.95
        plt.figtext(0.5, y_pos, "Clinical & Research Context", ha='center', fontsize=18, weight='bold')
        
        y_pos -= 0.08
        plt.figtext(0.1, y_pos, "What can this analysis tell us?", fontsize=14, weight='bold')
        y_pos -= 0.04
        plt.figtext(0.12, y_pos, "• Retinal layer structure", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• Relative image-domain thickness", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• Boundary continuity", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• Segmentation quality", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• Regions receiving stronger model attribution", fontsize=11)
        
        y_pos -= 0.08
        plt.figtext(0.1, y_pos, "What can this analysis NOT tell us?", fontsize=14, weight='bold', color='darkred')
        y_pos -= 0.04
        plt.figtext(0.12, y_pos, "• It does not diagnose Alzheimer's disease.", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• It does not diagnose dementia.", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• It does not establish AMD or DME diagnosis.", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• It does not provide a clinical disease probability.", fontsize=11)
        y_pos -= 0.025
        plt.figtext(0.12, y_pos, "• Pixel measurements are not calibrated anatomical micrometer measurements.", fontsize=11)
        
        y_pos -= 0.08
        plt.figtext(0.1, y_pos, "POTENTIAL CLINICAL RELEVANCE", fontsize=12, weight='bold')
        y_pos -= 0.03
        clin_rel = (
            "Retinal OCT provides non-invasive information about retinal microstructure. Retinal structural\n"
            "measures, including nerve-fibre and ganglion-cell-related measurements in appropriately defined\n"
            "OCT protocols, have been investigated as candidate biomarkers in Alzheimer's and other\n"
            "neurodegenerative disorders. However, published findings are heterogeneous and retinal imaging\n"
            "is not currently sufficient as a standalone Alzheimer's diagnostic test."
        )
        plt.figtext(0.1, y_pos, clin_rel, fontsize=11, va='top')
        
        y_pos -= 0.15
        
        # Current Model Interpretation Box
        # We can draw a simple rectangle using axes, or just bold text
        # Using a light gray background for a 'box' effect
        ax_box = fig.add_axes([0.1, y_pos - 0.08, 0.8, 0.1])
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.set_facecolor('#f0f0f0')
        ax_box.text(0.02, 0.8, "Current interpretation", fontsize=12, weight='bold')
        ax_box.text(0.02, 0.5, 
                    "The present model successfully provides an image-based retinal structural profile and\n"
                    "segmentation-quality assessment. Disease-level Alzheimer's inference is not available\n"
                    "from the current model.", fontsize=11, va='center')
        
        y_pos -= 0.14
        
        plt.figtext(0.1, y_pos, "CLINICAL REVIEW", fontsize=12, weight='bold')
        y_pos -= 0.03
        clin_rev = (
            "Any clinical interpretation should be performed by a qualified eye-care or medical\n"
            "professional in conjunction with the complete clinical history and appropriate\n"
            "diagnostic examinations."
        )
        plt.figtext(0.1, y_pos, clin_rev, fontsize=11, va='top')
        
        y_pos -= 0.20
        plt.figtext(0.5, y_pos, 
                    "Research use only. This AI-assisted analysis is not a standalone clinical\n"
                    "diagnostic tool and does not establish Alzheimer's disease or dementia.", 
                    ha='center', fontsize=12, weight='bold', color='dimgray')
        
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_buffer.getvalue()
