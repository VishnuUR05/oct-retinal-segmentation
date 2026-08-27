import os
import pandas as pd
import sys

def verify_splits():
    dataset_root = os.path.abspath('.')
    splits_dir = os.path.join(dataset_root, 'splits')
    
    train_csv = os.path.join(splits_dir, 'train.csv')
    val_csv = os.path.join(splits_dir, 'val.csv')
    test_csv = os.path.join(splits_dir, 'test.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
        print("ERROR: Missing split CSV files.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    # 1. E2E groups in multiple splits
    e2e_train = set(df_train['e2e_group_id'])
    e2e_val = set(df_val['e2e_group_id'])
    e2e_test = set(df_test['e2e_group_id'])
    
    intersect_tv = e2e_train.intersection(e2e_val)
    intersect_tt = e2e_train.intersection(e2e_test)
    intersect_vt = e2e_val.intersection(e2e_test)
    
    failed = False
    if intersect_tv:
        print(f"FAILED: E2E groups overlap between Train and Val: {intersect_tv}")
        failed = True
    if intersect_tt:
        print(f"FAILED: E2E groups overlap between Train and Test: {intersect_tt}")
        failed = True
    if intersect_vt:
        print(f"FAILED: E2E groups overlap between Val and Test: {intersect_vt}")
        failed = True
        
    # 2. Image in multiple splits
    img_train = set(df_train['image_path'])
    img_val = set(df_val['image_path'])
    img_test = set(df_test['image_path'])
    
    img_tv = img_train.intersection(img_val)
    img_tt = img_train.intersection(img_test)
    img_vt = img_val.intersection(img_test)
    
    if img_tv or img_tt or img_vt:
        print("FAILED: Images overlap between splits!")
        failed = True
        
    # 3. Mask in multiple splits
    mask_train = set(df_train['mask_path'])
    mask_val = set(df_val['mask_path'])
    mask_test = set(df_test['mask_path'])
    
    mask_tv = mask_train.intersection(mask_val)
    mask_tt = mask_train.intersection(mask_test)
    mask_vt = mask_val.intersection(mask_test)
    
    if mask_tv or mask_tt or mask_vt:
        print("FAILED: Masks overlap between splits!")
        failed = True
        
    # 4. Boundary CSV in multiple splits
    bnd_train = set(df_train['boundary_path'])
    bnd_val = set(df_val['boundary_path'])
    bnd_test = set(df_test['boundary_path'])
    
    bnd_tv = bnd_train.intersection(bnd_val)
    bnd_tt = bnd_train.intersection(bnd_test)
    bnd_vt = bnd_val.intersection(bnd_test)
    
    if bnd_tv or bnd_tt or bnd_vt:
        print("FAILED: Boundary CSVs overlap between splits!")
        failed = True
        
    # Check total image coverage
    # Assuming 1672 images total
    total_imgs = len(img_train) + len(img_val) + len(img_test)
    if total_imgs != 1672:
        print(f"FAILED: Expected 1672 images, but found {total_imgs} in the splits.")
        failed = True
        
    # Verify that the paths actually exist for at least one sample to ensure correctness
    sample_img = df_train.iloc[0]['image_path']
    if not os.path.exists(sample_img):
        print(f"FAILED: Paths in CSV are invalid. Cannot find {sample_img}")
        failed = True
        
    if failed:
        sys.exit(1)
    else:
        print("=========================================")
        print("INTEGRITY VERIFICATION PASSED SUCCESSFULLY")
        print("=========================================")
        print(f"Train Groups: {len(e2e_train)}, Images: {len(df_train)}")
        print(f"Val Groups:   {len(e2e_val)}, Images: {len(df_val)}")
        print(f"Test Groups:  {len(e2e_test)}, Images: {len(df_test)}")
        print("No overlaps detected across images, masks, boundaries, or E2E volumes.")
        
if __name__ == "__main__":
    verify_splits()
