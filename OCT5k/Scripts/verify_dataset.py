import os
import glob

def verify_dataset(base_path):
    print(f"Verifying dataset at {base_path}")
    
    images_dir = os.path.join(base_path, 'Images', 'Images_Manual')
    masks_dir = os.path.join(base_path, 'Masks', 'Masks_Manual', 'Grading_3')
    
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory missing: {images_dir}")
        print("Please ensure the dataset is properly downloaded and the 'Images' folder is placed correctly.")
        return False
        
    if not os.path.exists(masks_dir):
        print(f"ERROR: Masks directory missing: {masks_dir}")
        return False
        
    images = glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
    masks = glob.glob(os.path.join(masks_dir, '**', '*.png'), recursive=True)
    
    print(f"Found {len(images)} images.")
    print(f"Found {len(masks)} masks.")
    
    if len(images) == 0:
        print("ERROR: No images found. The dataset images are missing.")
        return False
        
    if len(masks) == 0:
        print("ERROR: No masks found.")
        return False
        
    print("Dataset verification complete.")
    return True

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    verify_dataset(base_dir)
