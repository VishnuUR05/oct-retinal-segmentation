import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from scipy.spatial.distance import euclidean

def calculate_vessel_density(binary_mask: np.ndarray, fov_mask: np.ndarray = None) -> float:
    """Calculate Vessel Density (%)"""
    if fov_mask is None:
        fov_pixels = binary_mask.size
    else:
        fov_pixels = np.sum(fov_mask > 0)
        
    if fov_pixels == 0:
        return 0.0
        
    vessel_pixels = np.sum(binary_mask > 0)
    return (vessel_pixels / fov_pixels) * 100.0

def calculate_total_vessel_length(skeleton_mask: np.ndarray) -> int:
    """Calculate Total Vessel Length (pixels)"""
    return int(np.sum(skeleton_mask > 0))

def calculate_vessel_width(binary_mask: np.ndarray, skeleton_mask: np.ndarray) -> tuple[float, float]:
    """
    Calculate Mean and Median Vessel Width (pixels)
    Using Euclidean Distance Transform on the binary mask, sampled at the skeleton.
    """
    # EDT gives distance from vessel pixel to the nearest background pixel (radius)
    # Binary mask: vessels are 255 (or True). EDT works by treating 0 as background.
    dist_transform = distance_transform_edt(binary_mask > 0)
    
    # Sample the distance transform exactly at the skeleton pixels
    radius_values = dist_transform[skeleton_mask > 0]
    
    if len(radius_values) == 0:
        return 0.0, 0.0
        
    # Width is roughly 2 * radius
    widths = radius_values * 2.0
    
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))
    
    return mean_width, median_width

def find_branch_and_endpoints(skeleton_mask: np.ndarray) -> tuple[int, int]:
    """
    Counts branch points and endpoints using a 3x3 neighborhood on the skeleton.
    """
    # Ensure skeleton is binary 0/1
    skel = (skeleton_mask > 0).astype(np.uint8)
    
    # Kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    
    # Only consider neighbors for pixels that are actually part of the skeleton
    neighbor_count = neighbor_count * skel
    
    # Endpoints have exactly 1 neighbor
    endpoints = (neighbor_count == 1)
    # Branch points have > 2 neighbors
    branchpoints = (neighbor_count > 2)
    
    # Branch points can sometimes cluster (e.g. 2 adjacent branch pixels)
    # Label and count unique branch point clusters
    bp_labeled = label(branchpoints)
    num_branch_points = bp_labeled.max()
    
    num_endpoints = int(np.sum(endpoints))
    
    return num_branch_points, num_endpoints

def calculate_tortuosity(skeleton_mask: np.ndarray) -> tuple[float, float]:
    """
    Calculate Mean and Median Tortuosity.
    Removes branch points to isolate vessel segments, then computes:
    Tortuosity = Arc Length / Chord Length
    """
    skel = (skeleton_mask > 0).astype(np.uint8)
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel, -1, kernel) * skel
    branchpoints = (neighbor_count > 2)
    
    # Remove branch points from skeleton to isolate segments
    segments_mask = skel & (~branchpoints)
    
    # Label individual segments
    labeled_segments = label(segments_mask)
    props = regionprops(labeled_segments)
    
    tortuosities = []
    
    for prop in props:
        # Segment length is arc length (number of pixels)
        arc_length = prop.area
        
        # We need at least a few pixels to meaningfully calculate tortuosity
        if arc_length < 10:
            continue
            
        # Find endpoints of this segment
        coords = prop.coords
        
        # A simple approximation of chord length is the max distance between any two points in the segment.
        # This works well for simple curved segments.
        # Alternatively, if we just find the two points that are furthest apart:
        # To save computation on large segments, we can compute distance from the centroid to all points, 
        # find the furthest point A, then find the furthest point from A which is B.
        # Then chord_length is distance(A, B).
        if len(coords) < 2:
             continue
             
        point_a = coords[0]
        max_dist_a = 0
        for pt in coords:
             d = euclidean(coords[0], pt)
             if d > max_dist_a:
                 max_dist_a = d
                 point_a = pt
                 
        point_b = point_a
        max_dist_b = 0
        for pt in coords:
             d = euclidean(point_a, pt)
             if d > max_dist_b:
                 max_dist_b = d
                 point_b = pt
                 
        chord_length = max_dist_b
        
        if chord_length > 0:
            tortuosity = arc_length / chord_length
            if tortuosity >= 1.0: # Mathematically it should be >= 1
                tortuosities.append(tortuosity)
                
    if len(tortuosities) == 0:
        return 0.0, 0.0
        
    return float(np.mean(tortuosities)), float(np.median(tortuosities))

def extract_all_biomarkers(binary_mask: np.ndarray, fov_mask: np.ndarray = None) -> dict:
    """
    Orchestrates extraction of all biomarkers from a clean binary mask.
    """
    if np.sum(binary_mask) == 0:
        return {
            "vessel_density_percent": 0.0,
            "total_vessel_length_pixels": 0,
            "mean_vessel_width_pixels": 0.0,
            "median_vessel_width_pixels": 0.0,
            "mean_tortuosity": 0.0,
            "median_tortuosity": 0.0,
            "branch_point_count": 0,
            "endpoint_count": 0,
            "skeleton_mask": np.zeros_like(binary_mask)
        }
    
    # Skeletonize
    # morphology.skeletonize expects boolean mask
    skeleton_mask_bool = morphology.skeletonize(binary_mask > 0)
    skeleton_mask = (skeleton_mask_bool * 255).astype(np.uint8)
    
    density = calculate_vessel_density(binary_mask, fov_mask)
    length = calculate_total_vessel_length(skeleton_mask)
    mean_width, median_width = calculate_vessel_width(binary_mask, skeleton_mask)
    branches, endpoints = find_branch_and_endpoints(skeleton_mask)
    mean_tortuosity, median_tortuosity = calculate_tortuosity(skeleton_mask)
    
    return {
        "vessel_density_percent": round(density, 4),
        "total_vessel_length_pixels": int(length),
        "mean_vessel_width_pixels": round(mean_width, 4),
        "median_vessel_width_pixels": round(median_width, 4),
        "mean_tortuosity": round(mean_tortuosity, 4),
        "median_tortuosity": round(median_tortuosity, 4),
        "branch_point_count": int(branches),
        "endpoint_count": int(endpoints),
        "skeleton_mask": skeleton_mask
    }
