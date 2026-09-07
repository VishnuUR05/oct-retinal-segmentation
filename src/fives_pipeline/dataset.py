import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

class FIVESDataset(Dataset):
    """
    Dataset class for FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation.
    
    Expected folder structure:
    root_dir/
      train/
        Original/     (RGB fundus images)
        Ground truth/ (Binary vessel masks)
      test/
        Original/
        Ground truth/
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.split = split.lower().strip()
        self.transform = transform
        
        # Locate split folder
        split_dir = self._find_dir(self.root_dir, self.split)
        if split_dir is None:
            # Check if root_dir is already the split directory
            if self.root_dir.name.lower() == self.split:
                split_dir = self.root_dir
            else:
                raise FileNotFoundError(
                    f"Could not find split folder '{self.split}' in '{self.root_dir}'. "
                    f"Available folders: {[p.name for p in self.root_dir.iterdir() if p.is_dir()]}"
                )
                
        # Locate 'Original' and 'Ground truth' folders
        self.images_dir = self._find_dir(split_dir, "original")
        self.masks_dir = self._find_dir(split_dir, "ground truth")
        
        if self.images_dir is None:
            raise FileNotFoundError(f"Could not find 'Original' images directory inside '{split_dir}'.")
        if self.masks_dir is None:
            raise FileNotFoundError(f"Could not find 'Ground truth' masks directory inside '{split_dir}'.")
            
        self.samples = self._load_paired_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No matching image-mask pairs found in '{split_dir}'.")

    @staticmethod
    def _find_dir(parent_path: Path, target_name: str) -> Path:
        """Finds subdirectory matching target_name (case-insensitive, ignoring spaces)."""
        target_norm = target_name.lower().replace(" ", "")
        for child in parent_path.iterdir():
            if child.is_dir():
                if child.name.lower().replace(" ", "") == target_norm:
                    return child
        return None

    def _load_paired_samples(self):
        """Indexes all valid image files and matches them with their corresponding masks."""
        # Index all masks by filename and stem
        mask_files = {}
        for f in self.masks_dir.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                mask_files[f.name.lower()] = f
                mask_files[f.stem.lower()] = f

        samples = []
        for img_path in sorted(self.images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            
            # Look for exact matching mask
            mask_path = mask_files.get(img_path.name.lower())
            if mask_path is None:
                mask_path = mask_files.get(img_path.stem.lower())
                
            if mask_path is not None and mask_path.exists():
                samples.append((img_path, mask_path))
            else:
                print(f"[Warning] Missing ground truth mask for image: {img_path.name}")
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load image (RGB)
        image = Image.open(img_path).convert("RGB")
        
        # Load mask (Grayscale 0-255)
        mask_img = Image.open(mask_path).convert("L")
        mask_np = (np.array(mask_img, dtype=np.float32) > 127.0).astype(np.float32)

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask_np)
        else:
            image_tensor = TF.to_tensor(image)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Shape (1, H, W)
            
        return image_tensor, mask_tensor
