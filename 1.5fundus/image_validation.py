import numpy as np
import cv2
from PIL import Image

def validate_fundus_image(image: Image.Image) -> tuple[bool, str]:
    """
    Validates if the image is likely a retinal fundus photograph.
    Returns (is_valid, error_message)
    """
    # Convert to numpy array in RGB format (OpenCV uses BGR by default, but we'll stick to RGB for logic)
    img_array = np.array(image.convert("RGB"))
    height, width, _ = img_array.shape

    # 1. Image Dimensions Check
    if height < 100 or width < 100:
        return False, "Invalid Input: Image resolution is too low to be a valid fundus image."

    # 2. Extract foreground (Retinal Field of View)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Thresholding to separate the FoV from the dark background
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Calculate foreground area
    foreground_area = np.sum(mask > 0)
    total_area = height * width
    foreground_ratio = foreground_area / total_area

    if foreground_ratio < 0.1:
        return False, "Invalid Input: The uploaded image does not appear to be a retinal fundus photograph. No clear retinal field detected."

    # 3. Color Profile Check
    # Fundus images are heavily dominated by red/orange hues.
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Calculate means only within the foreground mask
    mean_r = np.mean(r[mask > 0]) if foreground_area > 0 else np.mean(r)
    mean_g = np.mean(g[mask > 0]) if foreground_area > 0 else np.mean(g)
    mean_b = np.mean(b[mask > 0]) if foreground_area > 0 else np.mean(b)

    # In a typical fundus image, Red is the dominant color, and Blue is usually the weakest
    if not (mean_r > mean_g and mean_r > mean_b):
        return False, "Invalid Input: The uploaded image does not appear to be a retinal fundus photograph. The color profile does not match a typical fundus image (lacks red dominance)."
        
    # Additional strictness: Red should be significantly higher than Blue
    if mean_r < mean_b * 1.2:
        return False, "Invalid Input: The uploaded image does not appear to be a retinal fundus photograph. Colors are too balanced or blue-dominant."

    # 4. Circular/Elliptical Retinal Field Check (if there is a significant background)
    if foreground_ratio < 0.95:
        # Check circularity of the largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                # A perfect circle has circularity 1. Fundus FOVs are usually > 0.4
                if circularity < 0.3:
                    return False, "Invalid Input: The uploaded image does not appear to be a retinal fundus photograph. The field of view is not sufficiently circular/elliptical."

    return True, ""


def check_image_quality(image: Image.Image) -> tuple[bool, str]:
    """
    Checks the quality of a likely fundus image (sharpness, brightness, contrast).
    Returns (is_sufficient_quality, warning_message)
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # 1. Minimum Resolution
    if height < 256 or width < 256:
        return False, "Warning: The retinal image quality appears insufficient for reliable vessel analysis. Resolution is too low."

    # Create mask for valid retinal area
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    valid_pixels = gray[mask > 0]

    if len(valid_pixels) == 0:
         return False, "Warning: The retinal image quality appears insufficient for reliable vessel analysis. The image is entirely dark."

    # 2. Brightness
    mean_intensity = np.mean(valid_pixels)
    if mean_intensity < 30:
        return False, "Warning: The retinal image quality appears insufficient for reliable vessel analysis. It is excessively dark."
    if mean_intensity > 220:
        return False, "Warning: The retinal image quality appears insufficient for reliable vessel analysis. It is excessively overexposed."

    # 3. Contrast
    std_intensity = np.std(valid_pixels)
    if std_intensity < 15:
        return False, "Warning: The retinal image quality appears insufficient for reliable vessel analysis. The contrast is very low."

    # 4. Blur / Sharpness
    # Calculate Variance of Laplacian within the mask
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian[mask > 0]) if np.any(mask > 0) else np.var(laplacian)
    
    # 50 is a common threshold for blur detection, though it can vary based on resolution.
    if laplacian_var < 15:
         return False, "Warning: The retinal image quality appears insufficient for reliable vessel analysis. The image appears blurry or out of focus."

    return True, ""
