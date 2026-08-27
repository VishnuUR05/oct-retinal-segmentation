import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def analyze_sample(sample_id, img_path, mask_path, csv_path, out_dir):
    print(f"\n{'='*50}\nAnalyzing: {sample_id}\n{'='*50}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Types of violations
    v_ilm_opl = df['ILM'] > df['OPL']
    v_opl_isos = df['OPL'] > df['IS-OS']
    v_isos_ibrpe = df['IS-OS'] > df['IBRPE']
    v_ibrpe_obrpe = df['IBRPE'] > df['OBRPE']
    
    any_violation = v_ilm_opl | v_opl_isos | v_isos_ibrpe | v_ibrpe_obrpe
    violating_x = np.where(any_violation)[0]
    
    print(f"Total Violations: {len(violating_x)}")
    print(f"  ILM > OPL: {v_ilm_opl.sum()}")
    print(f"  OPL > IS-OS: {v_opl_isos.sum()}")
    print(f"  IS-OS > IBRPE: {v_isos_ibrpe.sum()}")
    print(f"  IBRPE > OBRPE: {v_ibrpe_obrpe.sum()}")
    
    print("\nFirst 20 Violating X-Columns:")
    print("  x | ILM | OPL | IS-OS | IBRPE | OBRPE")
    print("---------------------------------------")
    for i, x in enumerate(violating_x[:20]):
        r = df.iloc[x]
        print(f"{x:3d} | {r['ILM']:3d} | {r['OPL']:3d} | {r['IS-OS']:5d} | {r['IBRPE']:5d} | {r['OBRPE']:5d}")
        
    # Get contiguous ranges
    def get_ranges(indices):
        if len(indices) == 0: return []
        ranges = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                ranges.append((start, prev))
                start = idx
                prev = idx
        ranges.append((start, prev))
        return ranges
        
    ranges = get_ranges(violating_x)
    print("\nViolation Ranges (x):")
    for s, e in ranges:
        if s == e:
            print(f"  {s}")
        else:
            print(f"  {s}-{e}")
            
    # Load images
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Map mask classes back to colors for visualization
    colors = [
        [0, 0, 0],       # 0
        [255, 0, 0],     # 1
        [0, 255, 0],     # 2
        [0, 0, 255],     # 3
        [255, 255, 0],   # 4
        [255, 0, 255]    # 5
    ]
    mask_colored = np.zeros_like(img_rgb)
    for c in range(6):
        mask_colored[mask == c] = colors[c]
        
    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    x_axis = np.arange(512)
    b_colors = {'ILM': 'r', 'OPL': 'g', 'IS-OS': 'b', 'IBRPE': 'c', 'OBRPE': 'm'}
    
    axs[0].imshow(img_rgb)
    axs[0].set_title(f"Original OCT: {sample_id}")
    axs[0].axis('off')
    
    axs[1].imshow(mask_colored)
    axs[1].set_title("Grading_1 Segmentation Mask")
    axs[1].axis('off')
    
    axs[2].imshow(img_rgb)
    for b in b_colors:
        axs[2].plot(x_axis, df[b], color=b_colors[b], label=b)
    axs[2].set_title("Ground Truth Boundaries")
    axs[2].legend()
    axs[2].axis('off')
    
    axs[3].imshow(img_rgb)
    for b in b_colors:
        axs[3].plot(x_axis, df[b], color=b_colors[b])
    for x in violating_x:
        axs[3].axvline(x, color='yellow', alpha=0.3)
    axs[3].set_title("Violations Highlighted (Yellow)")
    axs[3].axis('off')
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_id}_gt_violation.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved diagnostic plot to: {out_path}")
    
    return len(violating_x)

def main():
    test_csv = "splits/test.csv"
    df = pd.read_csv(test_csv)
    
    out_dir = "outputs/boundary_analysis/diagnostics"
    
    # Target 1
    sample_1_id = "DME (7).E2E_Image 3"
    row1 = df[df['sample_id'] == sample_1_id].iloc[0]
    
    analyze_sample(
        sample_1_id,
        row1['image_path'],
        row1['mask_path'],
        row1['boundary_path'],
        out_dir
    )
    
    # Target 2: Find DME (35).E2E sample with most violations
    print("\nScanning DME (35).E2E for worst sample...")
    dme35 = df[df['e2e_group_id'] == "DME (35).E2E"]
    max_viol = -1
    worst_row = None
    
    for _, row in dme35.iterrows():
        csv = pd.read_csv(row['boundary_path'])
        v_ilm_opl = csv['ILM'] > csv['OPL']
        v_opl_isos = csv['OPL'] > csv['IS-OS']
        v_isos_ibrpe = csv['IS-OS'] > csv['IBRPE']
        v_ibrpe_obrpe = csv['IBRPE'] > csv['OBRPE']
        v_any = v_ilm_opl | v_opl_isos | v_isos_ibrpe | v_ibrpe_obrpe
        v_count = v_any.sum()
        
        if v_count > max_viol:
            max_viol = v_count
            worst_row = row
            
    print(f"Worst sample in DME (35).E2E is {worst_row['sample_id']} with {max_viol} violations.")
    
    analyze_sample(
        worst_row['sample_id'],
        worst_row['image_path'],
        worst_row['mask_path'],
        worst_row['boundary_path'],
        out_dir
    )

if __name__ == "__main__":
    main()
