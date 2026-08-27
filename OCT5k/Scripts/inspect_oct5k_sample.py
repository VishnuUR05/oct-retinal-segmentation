import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def main():
    report = []
    def log(msg):
        print(msg)
        report.append(str(msg))

    # 1. FIND DATASET ROOT
    log("1. FIND DATASET ROOT")
    log("--------------------------------")
    dataset_root = None
    # Search current directory and parents
    curr = os.path.abspath(".")
    for _ in range(3):
        if os.path.isdir(os.path.join(curr, "Masks")) and \
           os.path.isdir(os.path.join(curr, "Boundaries")):
            dataset_root = curr
            break
        curr = os.path.dirname(curr)

    if not dataset_root:
        log("ERROR: Could not find dataset root containing Images, Masks, Boundaries.")
        return

    log(f"Found Dataset Root: {dataset_root}")

    # Create outputs directory
    out_dir = os.path.join(dataset_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 2. FIND VALID IMAGE-MASK PAIRS
    log("\n2. FIND VALID IMAGE-MASK PAIRS")
    log("--------------------------------")
    
    images_manual_dir = os.path.join(dataset_root, "Images", "Images_Manual")
    masks_manual_dir = os.path.join(dataset_root, "Masks", "Masks_Manual")
    
    # FALLBACK LOGIC for current situation
    if not os.path.exists(images_manual_dir) or len(os.listdir(images_manual_dir)) == 0:
        log("Images_Manual is empty/missing! Falling back to checking Images_Automatic & Masks_Automatic...")
        images_manual_dir = os.path.join(dataset_root, "Images", "Images_Automatic")
        masks_manual_dir = os.path.join(dataset_root, "Masks", "Masks_Automatic", "Grading")
        
    all_images = []
    if os.path.isdir(images_manual_dir):
        for root, _, files in os.walk(images_manual_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
                    all_images.append(os.path.join(root, f))
    
    log(f"Total images found in {images_manual_dir}: {len(all_images)}")
    
    available_gradings = []
    if os.path.isdir(masks_manual_dir):
        if "Masks_Automatic" in masks_manual_dir:
            available_gradings = [""]
        else:
            available_gradings = [d for d in os.listdir(masks_manual_dir) if os.path.isdir(os.path.join(masks_manual_dir, d))]
    else:
        log("No available gradings found.")

    valid_pairs = []
    total_masks = 0

    for img_path in all_images:
        rel_path = os.path.relpath(img_path, images_manual_dir)
        has_any_mask = False
        img_masks = {}
        for grading in available_gradings:
            mask_dir = os.path.join(masks_manual_dir, grading) if grading else masks_manual_dir
            expected_mask_base = os.path.splitext(rel_path)[0]
            mask_path = os.path.join(mask_dir, expected_mask_base + ".png")
            
            if os.path.exists(mask_path):
                grading_key = grading if grading else "Automatic"
                img_masks[grading_key] = mask_path
                has_any_mask = True
                total_masks += 1
                
        if has_any_mask:
            valid_pairs.append({
                'image': img_path,
                'masks': img_masks,
                'rel_path': rel_path
            })

    log(f"Number of masks found matching images: {total_masks}")
    log(f"Number of valid image-mask pairs: {len(valid_pairs)}")
    log(f"Number of missing pairs (no mask): {len(all_images) - len(valid_pairs)}")

    if not valid_pairs:
        log("No valid pairs found to inspect. Exiting.")
        with open(os.path.join(out_dir, "oct5k_inspection_report.txt"), "w") as f:
            f.write("\n".join(report))
        return

    # 3. SELECT ONE REAL SAMPLE
    log("\n3. SELECT ONE REAL SAMPLE")
    log("--------------------------------")
    
    sample = valid_pairs[0]
    for p in valid_pairs:
        if len(p['masks']) >= 3:
            sample = p
            break
        elif "Grading_3" in p['masks']:
            sample = p
            
    img_path = sample['image']
    grading_used = list(sample['masks'].keys())[0]
    if "Grading_3" in sample['masks']:
        grading_used = "Grading_3"
    mask_path = sample['masks'][grading_used]
    
    parts = sample['rel_path'].replace('\\', '/').split('/')
    category = parts[0] if len(parts) > 0 else "Unknown"

    log(f"IMAGE:\n{img_path}")
    log(f"MASK:\n{mask_path}")
    log(f"GRADING:\n{grading_used}")
    log(f"CATEGORY:\n{category}")

    # 4. LOAD THE IMAGE
    log("\n4. LOAD THE IMAGE")
    log("--------------------------------")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        log("Failed to load image!")
        return
        
    log(f"Image shape: {img.shape}")
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    log(f"Width: {width}")
    log(f"Height: {height}")
    log(f"Number of channels: {channels}")
    log(f"Datatype: {img.dtype}")
    log(f"Minimum pixel value: {img.min()}")
    log(f"Maximum pixel value: {img.max()}")

    # 5. LOAD THE MASK
    log("\n5. LOAD THE MASK")
    log("--------------------------------")
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        log("Failed to load mask!")
        return
        
    log(f"Mask shape: {mask.shape}")
    log(f"Datatype: {mask.dtype}")
    log(f"Minimum value: {mask.min()}")
    log(f"Maximum value: {mask.max()}")
    
    unique_vals, counts = np.unique(mask, return_counts=True)
    log("ALL UNIQUE PIXEL VALUES:")
    total_pixels = mask.size
    for v, c in zip(unique_vals, counts):
        pct = (c / total_pixels) * 100
        log(f"Value: {v} -> Count: {c} ({pct:.2f}%)")

    # 6. VISUALIZE IMAGE AND MASK
    log("\n6. VISUALIZE IMAGE AND MASK")
    log("--------------------------------")
    
    if channels == 1:
        img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    if len(mask.shape) == 3:
        mask_disp = mask[:,:,0]
    else:
        mask_disp = mask
        
    colored_mask = plt.get_cmap('jet')(mask_disp / (mask_disp.max() if mask_disp.max() > 0 else 1))
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(img_disp, 0.7, colored_mask, 0.3, 0)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_disp)
    plt.title("Original OCT Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_disp, cmap='nipy_spectral')
    plt.title("Raw Segmentation Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Transparent Overlay")
    plt.axis('off')
    
    viz1_path = os.path.join(out_dir, "oct5k_sample_inspection.png")
    plt.tight_layout()
    plt.savefig(viz1_path)
    plt.close()
    log(f"Saved visualization to: {viz1_path}")

    # 7. INSPECT THE BOUNDARY CSV
    log("\n7. INSPECT THE BOUNDARY CSV")
    log("--------------------------------")
    boundaries_manual_dir = os.path.join(dataset_root, "Boundaries", "Boundaries_Manual")
    
    csv_path = None
    expected_csv_base = os.path.splitext(sample['rel_path'])[0]
    
    for grading in ["Grading_3", "Grading_2", "Grading_1"]:
        test_path = os.path.join(boundaries_manual_dir, grading, expected_csv_base + ".csv")
        if os.path.exists(test_path):
            csv_path = test_path
            break
            
    if csv_path:
        log(f"Exact CSV path: {csv_path}")
        log(f"CSV filename: {os.path.basename(csv_path)}")
        try:
            df = pd.read_csv(csv_path)
            log(f"Column names: {list(df.columns)}")
            log(f"Number of rows: {len(df)}")
            log(f"First 10 rows:\n{df.head(10)}")
            log("Datatype of each column:")
            for col in df.columns:
                log(f"- {col}: {df[col].dtype}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    log(f"  Min: {df[col].min()}, Max: {df[col].max()}")
                    
            if 'X' in df.columns.str.upper() or 'Y' in df.columns.str.upper():
                log("Conclusion: CSV contains x/y coordinates.")
            elif len(df.columns) >= 5 and len(df) == width:
                log("Conclusion: CSV seems to contain y-coordinates for each layer per x-column (width of image).")
            else:
                log("Conclusion: CSV format is unusual. Requires verification.")
                
        except Exception as e:
            log(f"Error reading CSV: {e}")
            df = None
    else:
        log("No boundary CSV found for this image.")
        df = None

    # 8. VISUALIZE THE BOUNDARIES
    log("\n8. VISUALIZE THE BOUNDARIES")
    log("--------------------------------")
    if df is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(img_disp)
        
        columns = df.columns
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.plot(range(len(df)), df[col], label=col, linewidth=1.5)
                
        plt.title("Boundary Overlay")
        plt.legend(loc='upper right', fontsize='small')
        plt.axis('off')
        
        viz2_path = os.path.join(out_dir, "oct5k_boundary_inspection.png")
        plt.tight_layout()
        plt.savefig(viz2_path)
        plt.close()
        log(f"Saved boundary visualization to: {viz2_path}")
        
        explicit_layers = ['ILM', 'OPL', 'IS', 'OS', 'IBRPE', 'OBRPE']
        has_explicit = any(any(layer.lower() in col.lower() for layer in explicit_layers) for col in columns)
        if has_explicit:
            log(f"Layer-to-column mapping found: {list(columns)}")
        else:
            log("Layer-to-column mapping requires verification.")

    # 9. COMPARE MASK AND BOUNDARY INFORMATION
    log("\n9. COMPARE MASK AND BOUNDARY INFORMATION")
    log("--------------------------------")
    has_filled_regions = len(unique_vals) > 2
    log(f"Mask contains filled segmentation regions: {has_filled_regions}")
    
    has_boundary_coords = df is not None
    log(f"Boundary CSV contains boundary coordinates: {has_boundary_coords}")
    
    if has_filled_regions and has_boundary_coords:
        log("Both represent the same anatomical structures (requires visual confirmation from plots).")
        log("The mask and boundary correspond to the same image.")
    else:
        log("Cannot establish full correspondence from files.")

    # 10. CHECK ALL THREE MANUAL GRADINGS
    log("\n10. CHECK ALL THREE MANUAL GRADINGS")
    log("--------------------------------")
    all_grading_masks = {}
    expected_mask_base = os.path.splitext(sample['rel_path'])[0]
    for g in ["Grading_1", "Grading_2", "Grading_3"]:
        m_path = os.path.join(masks_manual_dir, g, expected_mask_base + ".png")
        if os.path.exists(m_path):
            all_grading_masks[g] = m_path
            
    log(f"Annotations exist in: {list(all_grading_masks.keys())}")
    
    if len(all_grading_masks) == 3:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_disp)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        for i, (g, m_path) in enumerate(all_grading_masks.items()):
            g_mask = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
            g_uvals = np.unique(g_mask)
            log(f"{g} unique pixel values: {g_uvals}")
            log(f"{g} dimensions: {g_mask.shape}")
            
            if len(g_mask.shape) == 3:
                g_mask_disp = g_mask[:,:,0]
            else:
                g_mask_disp = g_mask
                
            axes[i+1].imshow(g_mask_disp, cmap='nipy_spectral')
            axes[i+1].set_title(g)
            axes[i+1].axis('off')
            
        viz3_path = os.path.join(out_dir, "oct5k_three_gradings.png")
        plt.tight_layout()
        plt.savefig(viz3_path)
        plt.close()
        log(f"Saved three gradings visualization to: {viz3_path}")
    else:
        log("Not all 3 gradings exist for this image.")

    # 11. CHECK THE DATASET CLASS COUNT
    log("\n11. CHECK THE DATASET CLASS COUNT")
    log("--------------------------------")
    log(f"Number of unique mask labels in sample: {len(unique_vals)}")
    log(f"Background included: {0 in unique_vals}")
    
    possible_classes = len(unique_vals)
    log(f"Possible number of segmentation classes: {possible_classes}")
    
    log("Does the training script 'num_classes = 10' appear correct?")
    if possible_classes <= 6:
        log("WARNING: The sample only contains <= 6 unique values (including background).")
        log("If the README states 5 layers, num_classes should likely be 6 (5 layers + 1 background).")
        log("num_classes=10 seems potentially INCORRECT or expects different encodings.")
    else:
        log("num_classes=10 might be correct, depending on other masks.")

    # 12. GENERATE FINAL REPORT
    log("\n12. GENERATE A FINAL REPORT")
    log("--------------------------------")
    report_path = os.path.join(out_dir, "oct5k_inspection_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
        
    log(f"Successfully generated final report at: {report_path}")

if __name__ == "__main__":
    main()
