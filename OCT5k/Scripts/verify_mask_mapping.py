import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def get_mapping(mask, df):
    # df has columns: x, ILM, OPL, IS-OS, IBRPE, OBRPE
    mapping_counts = {
        'Above ILM': [],
        'ILM to OPL': [],
        'OPL to IS-OS': [],
        'IS-OS to IBRPE': [],
        'IBRPE to OBRPE': [],
        'Below OBRPE': []
    }
    
    for i, row in df.iterrows():
        x = int(row['x'])
        if x >= mask.shape[1]: continue
        
        ilm = int(row['ILM'])
        opl = int(row['OPL'])
        isos = int(row['IS-OS'])
        ibrpe = int(row['IBRPE'])
        obrpe = int(row['OBRPE'])
        
        # Gather all pixels in the column for each region
        if ilm > 0:
            mapping_counts['Above ILM'].extend(mask[0:ilm, x])
        if opl > ilm:
            mapping_counts['ILM to OPL'].extend(mask[ilm:opl, x])
        if isos > opl:
            mapping_counts['OPL to IS-OS'].extend(mask[opl:isos, x])
        if ibrpe > isos:
            mapping_counts['IS-OS to IBRPE'].extend(mask[isos:ibrpe, x])
        if obrpe > ibrpe:
            mapping_counts['IBRPE to OBRPE'].extend(mask[ibrpe:obrpe, x])
        if obrpe < mask.shape[0]:
            mapping_counts['Below OBRPE'].extend(mask[obrpe:, x])
            
    final_mapping = {}
    for region, vals in mapping_counts.items():
        if len(vals) == 0:
            final_mapping[region] = -1
            continue
        vals = np.array(vals)
        unique, counts = np.unique(vals, return_counts=True)
        predominant_val = unique[np.argmax(counts)]
        final_mapping[region] = int(predominant_val)
        
    return final_mapping

def analyze_sample(img_path, mask_path, csv_path, visualize=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    df = pd.read_csv(csv_path)
    
    mapping = get_mapping(mask, df)
    
    if visualize:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original OCT")
        plt.imshow(img, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Segmentation Mask")
        plt.imshow(mask, cmap='nipy_spectral')
        
        plt.subplot(1, 3, 3)
        plt.title("Boundaries overlaid on Mask")
        plt.imshow(mask, cmap='nipy_spectral')
        plt.plot(df['x'], df['ILM'], label='ILM', color='red')
        plt.plot(df['x'], df['OPL'], label='OPL', color='orange')
        plt.plot(df['x'], df['IS-OS'], label='IS-OS', color='yellow')
        plt.plot(df['x'], df['IBRPE'], label='IBRPE', color='green')
        plt.plot(df['x'], df['OBRPE'], label='OBRPE', color='cyan')
        plt.legend(loc='upper right', fontsize='small')
        
        # Add text mapping
        textstr = '\n'.join([f"{k}: {v}" for k, v in mapping.items()])
        plt.gcf().text(0.02, 0.5, textstr, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        out_dir = os.path.dirname(os.path.dirname(img_path)) # Just somewhere to save, wait, let's use fixed outputs dir
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(img_path)))))
        out_file = os.path.join(root, "outputs", "verified_mask_layer_mapping.png")
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        
    return mapping

if __name__ == "__main__":
    dataset_root = os.path.abspath(".")
    print(f"Dataset root: {dataset_root}")
    
    # Target image paths
    img_path = os.path.join(dataset_root, "Images", "Images_Manual", "AMD Part1", "AMD (1).E2E", "2- 25- 2017 9- 10- 42 PM", "Image 10.png")
    mask_path = os.path.join(dataset_root, "Masks", "Masks_Manual", "Grading_3", "AMD Part1", "AMD (1).E2E", "2- 25- 2017 9- 10- 42 PM", "Image 10.png")
    csv_path = os.path.join(dataset_root, "Boundaries", "Boundaries_Manual", "Grading_3", "AMD Part1", "AMD (1).E2E", "2- 25- 2017 9- 10- 42 PM", "Image 10.csv")
    
    print(f"Analyzing Target Sample:\nIMG: {img_path}\nMASK: {mask_path}\nCSV: {csv_path}\n")
    
    target_mapping = analyze_sample(img_path, mask_path, csv_path, visualize=True)
    
    print("TARGET SAMPLE MAPPING:")
    for k, v in target_mapping.items():
        print(f"Region: {k:<15} | Mask Value: {v}")
        
    # Pick 10 random samples
    print("\n-------------------------")
    print("Running random consistency check on 10 samples across all gradings...")
    
    import glob
    all_csvs = glob.glob(os.path.join(dataset_root, "Boundaries", "Boundaries_Manual", "*", "*", "*", "*", "*.csv"))
    
    random.seed(42)
    sample_csvs = random.sample(all_csvs, min(10, len(all_csvs)))
    
    consistent = True
    for csv_f in sample_csvs:
        # Reconstruct paths
        parts = csv_f.split(os.sep)
        grading = parts[-5] # Grading_X
        cat = parts[-4]
        e2e = parts[-3]
        date = parts[-2]
        base = parts[-1].replace(".csv", ".png")
        
        m_path = os.path.join(dataset_root, "Masks", "Masks_Manual", grading, cat, e2e, date, base)
        i_path = os.path.join(dataset_root, "Images", "Images_Manual", cat, e2e, date, base)
        
        if os.path.exists(m_path) and os.path.exists(i_path):
            mapping = analyze_sample(i_path, m_path, csv_f, visualize=False)
            if mapping != target_mapping:
                print(f"INCONSISTENCY FOUND in {csv_f}: {mapping}")
                consistent = False
        else:
            print(f"Missing file for {csv_f}")
            
    if consistent:
        print("\nSUCCESS! The mapping is 100% consistent across different images, categories, and gradings.")
    else:
        print("\nWARNING! Inconsistencies found in mapping.")
        
    print("\nFINAL DERIVED MAPPING:")
    # Invert mapping
    inv_map = {v: k for k, v in target_mapping.items()}
    for val in range(6):
        if val in inv_map:
            print(f"{val} = {inv_map[val]}")
        else:
            print(f"{val} = Unknown")
