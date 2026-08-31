import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


import yaml
from pathlib import Path

def get_dataset_root():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(base_dir, "configs", "baseline.yaml")
    with open(baseline_path, 'r') as f:
        config = yaml.safe_load(f)
    
    local_path = os.path.join(base_dir, "configs", "local.yaml")
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            local_config = yaml.safe_load(f)
            if local_config and 'paths' in local_config and 'dataset_root' in local_config['paths']:
                return Path(local_config['paths']['dataset_root'])
    
    return Path(config['paths'].get('dataset_root', '.'))

class OCTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.dataset_root = get_dataset_root()
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(self.dataset_root / str(row['image_path']))
        mask_path = str(self.dataset_root / str(row['mask_path']))
        
        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image {img_path}")
            
        # Convert grayscale to 3 channels for ResNet34
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Load integer mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot load mask {mask_path}")
            
        # Ensure mask is strictly 0-5
        mask = np.clip(mask, 0, 5).astype(np.int64)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback if no transform (usually shouldn't happen)
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            mask = torch.from_numpy(mask).long()
            
        # Ensure mask is Long tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.to(torch.long)
        
        return image, mask

def get_training_augmentation():
    train_transform = [
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(test_transform)
