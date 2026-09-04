import numpy as np
import cv2
from skimage import morphology

def postprocess_vessel_mask(prob_map: np.ndarray, threshold: float = 0.5, min_size: int = 100) -> np.ndarray:
    """
    Converts probability map to clean binary mask.
    1. Thresholding
    2. Remove small objects (noise)
    """
    # 1. Thresholding
    binary_mask = prob_map > threshold
    
    # 2. Remove small objects
    # morphology.remove_small_objects expects boolean array
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
    
    # Convert to uint8 (0, 255)
    return (cleaned_mask * 255).astype(np.uint8)

def create_fov_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Derives the valid retinal Field of View (FOV) mask from the original RGB image.
    Fundus images typically have a large black background.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Thresholding to separate the FoV from the dark background
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Fill small holes in the FOV mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume largest contour is the FOV
        largest_contour = max(contours, key=cv2.contourArea)
        fov_mask = np.zeros_like(mask)
        cv2.drawContours(fov_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return fov_mask
    
    return mask
