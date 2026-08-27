import os
import glob
import random
import pandas as pd

def select_images():
    input_root = r"F:\Ait Major Project\dataset\archive\OCT2017\test"
    output_dir = r"outputs\external_annotation"
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['DME', 'CNV', 'Drusen', 'Normal']
    extensions = ['.jpeg', '.jpg', '.png', '.tif', '.tiff', '.bmp']
    
    random.seed(42)
    
    selected_data = []
    
    for c in categories:
        cat_dir = os.path.join(input_root, c)
        if not os.path.exists(cat_dir):
            cat_dir = os.path.join(input_root, c.upper()) # Fallback for case
        if not os.path.exists(cat_dir):
            print(f"WARNING: Directory for {c} not found at {cat_dir}.")
            continue
            
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(cat_dir, f"*{ext}")))
            
        all_files.sort() # Ensure deterministic before shuffle
        random.shuffle(all_files)
        
        selected = all_files[:20]
        
        for i, filepath in enumerate(selected):
            selected_data.append({
                'category': c,
                'image_path': filepath,
                'filename': os.path.basename(filepath),
                'selected_index': i + 1
            })
            
    df = pd.DataFrame(selected_data)
    out_csv = os.path.join(output_dir, "selected_images.csv")
    df.to_csv(out_csv, index=False)
    print(f"Selected {len(df)} images and saved manifest to {out_csv}")
    
    # Initialize metadata.csv
    meta_csv = os.path.join(output_dir, "metadata.csv")
    if not os.path.exists(meta_csv):
        df_meta = df[['category', 'filename', 'image_path']].copy()
        df_meta = df_meta.rename(columns={'image_path': 'original_path', 'filename': 'image_name'})
        df_meta['annotation_status'] = 'NOT_STARTED'
        df_meta['saved_timestamp'] = ''
        df_meta['annotator'] = ''
        df_meta['boundary_validity'] = ''
        df_meta['uncertain_columns'] = ''
        df_meta['mask_path'] = ''
        df_meta['boundary_path'] = ''
        df_meta.to_csv(meta_csv, index=False)
        print(f"Initialized metadata.csv at {meta_csv}")
        
    # Create category dirs
    for c in categories:
        os.makedirs(os.path.join(output_dir, c, "boundaries"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, c, "masks"), exist_ok=True)
        
if __name__ == "__main__":
    select_images()
