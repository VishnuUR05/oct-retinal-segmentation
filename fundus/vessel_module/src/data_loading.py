import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import numpy as np

class FIVESPatchDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks), "Mismatch in image/mask count"
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()
        
        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                img = torch.rot90(img, k, [1, 2])
                mask = torch.rot90(mask, k, [1, 2])
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = TF.adjust_brightness(img, factor)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = TF.adjust_contrast(img, factor)
                
        img = TF.normalize(img, self.mean, self.std)
        return img, mask
