import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def analyze_data_quality(df_feat):
    # 1. Data Quality
    quality_report = []
    
    # Check NaN, Infs, Negatives for thickness columns (PRED and GT)
    thick_cols = [c for c in df_feat.columns if ('PRED_' in c or 'GT_' in c) and ('mean_pixels' in c or 'median_pixels' in c or 'min_pixels' in c or 'max_pixels' in c)]
    
    total_samples = len(df_feat)
    invalid_samples = set()
    
    for col in thick_cols:
        nans = df_feat[col].isna()
        infs = np.isinf(df_feat[col])
        negs = df_feat[col] < 0
        
        invalid_idx = df_feat[nans | infs | negs].index
        invalid_samples.update(invalid_idx)
        
        if nans.sum() > 0 or infs.sum() > 0 or negs.sum() > 0:
            quality_report.append(f"{col}: NaNs={nans.sum()}, Infs={infs.sum()}, Negs={negs.sum()}")
            
    # Also check Min/Max bounds of mean thickness
    for col in [c for c in thick_cols if 'mean_pixels' in c]:
        v_min = df_feat[col].min()
        v_max = df_feat[col].max()
        quality_report.append(f"{col} | Range: [{v_min:.2f}, {v_max:.2f}]")
        
    num_invalid = len(invalid_samples)
    num_valid = total_samples - num_invalid
    
    quality_summary = {
        'total_samples': total_samples,
        'valid_samples': num_valid,
        'invalid_samples': num_invalid,
        'invalid_pct': (num_invalid / total_samples) * 100.0,
        'report': "\n".join(quality_report)
    }
    return quality_summary

def main():
    print("="*60)
    print("OCT5k STRUCTURAL FEATURE QUALITY ANALYSIS")
    print("="*60)
    
    reports_in_dir = "outputs/boundary_analysis/reports"
    feat_csv = os.path.join(reports_in_dir, "retinal_structural_features.csv")
    met_csv = os.path.join(reports_in_dir, "boundary_metrics.csv")
    
    if not os.path.exists(feat_csv):
        print(f"File not found: {feat_csv}")
        sys.exit(1)
        
    df_feat = pd.read_csv(feat_csv)
    df_met = pd.read_csv(met_csv)
    
    out_dir = "outputs/boundary_analysis/feature_analysis"
    ensure_dir(out_dir)
    
    # 1. QUALITY ANALYSIS
    q_summary = analyze_data_quality(df_feat)
    
    # 2. PRED VS GT AGREEMENT (Image Level)
    layers = ['ILM_OPL', 'OPL_ISOS', 'ISOS_IBRPE', 'IBRPE_OBRPE', 'Total_Retinal']
    agreement_report = []
    
    for layer in layers:
        pred_col = f"PRED_{layer}_mean_pixels"
        gt_col = f"GT_{layer}_mean_pixels"
        
        pred_mean = df_feat[pred_col].mean()
        gt_mean = df_feat[gt_col].mean()
        abs_diff = np.abs(df_feat[pred_col] - df_feat[gt_col])
        mae = abs_diff.mean()
        
        agreement_report.append({
            'Layer': layer,
            'PRED_Mean': pred_mean,
            'GT_Mean': gt_mean,
            'MAE': mae
        })
    df_agreement = pd.DataFrame(agreement_report)
    
    # 3. CENTRAL VS GLOBAL
    cg_report = []
    for layer in layers:
        glob_col = f"PRED_{layer}_mean_pixels"
        cent_col = f"PRED_central_{layer}_mean_pixels"
        
        glob_mean = df_feat[glob_col].mean()
        cent_mean = df_feat[cent_col].mean()
        diff = cent_mean - glob_mean
        
        cg_report.append({
            'Layer': layer,
            'Global_Mean': glob_mean,
            'Central_Mean': cent_mean,
            'Diff_Central_Minus_Global': diff
        })
    df_cg = pd.DataFrame(cg_report)
    
    # 4. E2E GROUP LEVEL ANALYSIS
    e2e_cols = ['e2e_group_id', 'category'] + [f"PRED_{l}_mean_pixels" for l in layers] + [f"PRED_central_{l}_mean_pixels" for l in layers]
    df_e2e = df_feat.groupby(['e2e_group_id', 'category']).agg(
        b_scans=('sample_id', 'count'),
        **{c: (c, 'mean') for c in e2e_cols if c not in ['e2e_group_id', 'category']},
        **{f"{c}_std": (c, 'std') for c in e2e_cols if c not in ['e2e_group_id', 'category']}
    ).reset_index()
    
    e2e_out_csv = os.path.join(out_dir, "e2e_structural_features.csv")
    df_e2e.to_csv(e2e_out_csv, index=False)
    
    # 5 & 6. CATEGORY LEVEL EXPLORATION & STATS
    # Consolidate category names (e.g., 'AMD Part1' -> 'AMD')
    df_e2e['base_category'] = df_e2e['category'].apply(lambda x: x.split()[0])
    
    cat_summary = df_e2e.groupby('base_category').agg(
        num_e2e_groups=('e2e_group_id', 'count'),
        total_images=('b_scans', 'sum'),
        Total_Retinal_Mean=('PRED_Total_Retinal_mean_pixels', 'mean'),
        ILM_OPL_Mean=('PRED_ILM_OPL_mean_pixels', 'mean'),
        OPL_ISOS_Mean=('PRED_OPL_ISOS_mean_pixels', 'mean')
    ).reset_index()
    
    cat_out_csv = os.path.join(out_dir, "category_structural_summary.csv")
    cat_summary.to_csv(cat_out_csv, index=False)
    
    # Kruskal-Wallis test across categories for Total Retinal Thickness
    cats = df_e2e['base_category'].unique()
    stat_report = "--- CATEGORY STATISTICAL EXPLORATION (E2E-Level) ---\n"
    if len(cats) >= 2:
        samples = [df_e2e[df_e2e['base_category'] == c]['PRED_Total_Retinal_mean_pixels'].values for c in cats]
        # Only run if all groups have at least 1 sample
        if all(len(s) > 0 for s in samples):
            stat, pval = stats.kruskal(*samples)
            stat_report += f"Kruskal-Wallis on Total Retinal Thickness across {list(cats)}:\n"
            stat_report += f"H-statistic: {stat:.4f}, p-value: {pval:.4e}\n"
            if pval < 0.05:
                stat_report += "Result: Significant difference found between categories.\n"
            else:
                stat_report += "Result: No significant difference found at alpha=0.05.\n"
        else:
            stat_report += "Not enough samples per category for statistical testing.\n"
    
    # Discover Most Stable vs Most Variable Feature
    # Using Coefficient of Variation (CV) = std / mean
    cv_dict = {}
    for layer in layers:
        col = f"PRED_{layer}_mean_pixels"
        mean_val = df_feat[col].mean()
        std_val = df_feat[col].std()
        if mean_val > 0:
            cv_dict[layer] = std_val / mean_val
            
    most_stable = min(cv_dict, key=cv_dict.get)
    most_variable = max(cv_dict, key=cv_dict.get)
    
    # 7. VISUALIZATIONS
    vis_dir = os.path.join(out_dir, "visualizations")
    ensure_dir(vis_dir)
    
    sns.set_theme(style="whitegrid")
    
    # A. Distribution PRED vs GT
    for layer in layers:
        plt.figure(figsize=(8,5))
        sns.kdeplot(df_feat[f"PRED_{layer}_mean_pixels"], label='Predicted', fill=True, alpha=0.5)
        sns.kdeplot(df_feat[f"GT_{layer}_mean_pixels"], label='Ground Truth', fill=True, alpha=0.5)
        plt.title(f"Distribution: {layer} Thickness (Pixels)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"dist_{layer}.png"))
        plt.close()
        
    # B. Scatter PRED vs GT
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df_feat, x="GT_Total_Retinal_mean_pixels", y="PRED_Total_Retinal_mean_pixels", hue='category', alpha=0.7)
    # y=x line
    min_val = min(df_feat["GT_Total_Retinal_mean_pixels"].min(), df_feat["PRED_Total_Retinal_mean_pixels"].min())
    max_val = max(df_feat["GT_Total_Retinal_mean_pixels"].max(), df_feat["PRED_Total_Retinal_mean_pixels"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.title("Predicted vs Ground Truth Total Retinal Thickness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "scatter_total_thickness.png"))
    plt.close()
    
    # C. Central vs Global
    plt.figure(figsize=(10,6))
    df_cg_melt = df_cg.melt(id_vars='Layer', value_vars=['Global_Mean', 'Central_Mean'], var_name='Region', value_name='Thickness (Pixels)')
    sns.barplot(data=df_cg_melt, x='Layer', y='Thickness (Pixels)', hue='Region')
    plt.title("Global vs Central Retinal Structural Thickness")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "bar_central_vs_global.png"))
    plt.close()
    
    # D. Category Level Boxplots (E2E Level)
    for layer in ['Total_Retinal', 'ILM_OPL', 'OPL_ISOS', 'ISOS_IBRPE', 'IBRPE_OBRPE']:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=df_e2e, x='base_category', y=f"PRED_{layer}_mean_pixels")
        sns.swarmplot(data=df_e2e, x='base_category', y=f"PRED_{layer}_mean_pixels", color=".25")
        plt.title(f"{layer} Thickness by Category (E2E Group Level)")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"box_category_{layer}.png"))
        plt.close()
        
    # 8. WRITE FINAL REPORT
    report_path = os.path.join(out_dir, "feature_quality_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("OCT5k RETINAL STRUCTURAL FEATURE QUALITY REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Samples (B-scans): {q_summary['total_samples']}\n")
        f.write(f"Valid Samples:           {q_summary['valid_samples']}\n")
        f.write(f"Invalid Samples:         {q_summary['invalid_samples']} ({q_summary['invalid_pct']:.2f}%)\n\n")
        
        f.write("--- DATA QUALITY WARNINGS ---\n")
        if q_summary['report']:
            f.write(q_summary['report'] + "\n\n")
        else:
            f.write("No NaNs, Infs, or Negatives found.\n\n")
            
        f.write("--- PREDICTED VS GROUND-TRUTH AGREEMENT ---\n")
        f.write(df_agreement.to_string(index=False) + "\n\n")
        
        f.write("--- CENTRAL VS GLOBAL STRUCTURE ---\n")
        f.write(df_cg.to_string(index=False) + "\n\n")
        
        f.write("--- FEATURE STABILITY (Predicted) ---\n")
        f.write(f"Most Stable Feature:   {most_stable} (CV: {cv_dict[most_stable]:.4f})\n")
        f.write(f"Most Variable Feature: {most_variable} (CV: {cv_dict[most_variable]:.4f})\n\n")
        
        f.write(stat_report)
        f.write("\nNOTE: These are strictly exploratory comparisons of 'quantitative retinal structural features'.\n")
        f.write("No claims regarding Alzheimer's biomarkers are made at this stage.\n")

    print(f"\nAnalysis complete. Valid samples: {q_summary['valid_samples']}/{q_summary['total_samples']}")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
