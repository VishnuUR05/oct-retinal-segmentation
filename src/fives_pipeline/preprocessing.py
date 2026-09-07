import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

class FIVESTransform:
    """
    Preprocessing and augmentation pipeline for FIVES vessel segmentation.
    
    Performs:
    1. Synchronized resizing to specified target resolution (default 512x512)
    2. Optional CLAHE contrast enhancement on resized Luminance channel
    3. Synchronized augmentations (H-flip, V-flip, 90-degree rotations)
    4. ImageNet standardization for image and binary float tensor formatting for mask
    """
    def __init__(self, size=(512, 512), augment=False, use_clahe=False, mean=None, std=None):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
            
        self.augment = augment
        self.use_clahe = use_clahe
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def _apply_clahe(self, image: Image.Image) -> Image.Image:
        """Applies CLAHE on the L-channel in LAB color space to enhance vessel contrast."""
        try:
            import cv2
            img_np = np.array(image)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(enhanced_rgb)
        except ImportError:
            # Fallback if OpenCV is not installed
            return image

    def __call__(self, image: Image.Image, mask_np: np.ndarray):
        # 1. Resize image (bilinear) and mask (nearest) first for high CPU performance
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        
        # 2. CLAHE contrast enhancement on the resized image (25x faster on CPU)
        if self.use_clahe:
            image = self._apply_clahe(image)
        
        mask_tensor = torch.from_numpy(mask_np)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, C, H, W)
            
        mask_tensor = F.interpolate(mask_tensor, size=self.size, mode="nearest").squeeze(0)  # (1, H, W)

        # 3. Synchronized Augmentations
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask_tensor = TF.hflip(mask_tensor)

            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask_tensor = TF.vflip(mask_tensor)

            # Random 90-degree Rotations
            rot_choice = random.choice([0, 90, 180, 270])
            if rot_choice != 0:
                image = TF.rotate(image, rot_choice)
                mask_tensor = TF.rotate(mask_tensor, rot_choice)

        # 4. Standardize Image to Tensor
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)

        # 5. Ensure Mask is binary float [0.0, 1.0]
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
