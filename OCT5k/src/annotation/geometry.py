import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def initialize_control_points(dense_boundary, step=32):
    """
    Downsamples a dense 512-length boundary into sparse control points.
    Returns list of dicts: [{'x': x, 'y': y}, ...]
    """
    points = []
    # Always include 0 and 511
    x_coords = list(range(0, 512, step))
    if 511 not in x_coords:
        x_coords.append(511)
        
    for x in x_coords:
        y = dense_boundary[x]
        if not np.isnan(y):
            points.append({'x': int(x), 'y': float(y)})
            
    return points

def interpolate_boundary(control_points, method='linear', length=512):
    """
    Interpolates control points to a dense array.
    """
    if not control_points:
        return np.full(length, np.nan)
        
    # Sort points by x
    pts = sorted(control_points, key=lambda p: p['x'])
    
    x_vals = [p['x'] for p in pts]
    y_vals = [p['y'] for p in pts]
    
    # If only one point, flat line
    if len(x_vals) == 1:
        return np.full(length, y_vals[0])
        
    f = interp1d(x_vals, y_vals, kind=method, bounds_error=False, fill_value='extrapolate')
    x_new = np.arange(length)
    y_new = f(x_new)
    
    # Clip to image bounds just in case extrapolation goes wild
    y_new = np.clip(y_new, 0, length - 1)
    
    return y_new

def generate_mask(boundaries, length=512):
    """
    Generates a 512x512 6-class integer mask from 5 boundary arrays.
    Classes:
    0: above ILM
    1: ILM-OPL
    2: OPL-ISOS
    3: ISOS-IBRPE
    4: IBRPE-OBRPE
    5: below OBRPE
    """
    mask = np.zeros((length, length), dtype=np.uint8)
    
    b_names = ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']
    # If any boundary is missing entirely, we treat it as NaN which makes inequalities False.
    # To handle overlapping or missing boundaries securely, we process row by row.
    
    for x in range(length):
        y_ilm = boundaries['ILM'][x]
        y_opl = boundaries['OPL'][x]
        y_isos = boundaries['IS-OS'][x]
        y_ibrpe = boundaries['IBRPE'][x]
        y_obrpe = boundaries['OBRPE'][x]
        
        for y in range(length):
            # Class 0: y < ILM
            if not np.isnan(y_ilm) and y < y_ilm:
                mask[y, x] = 0
            # Class 1: ILM <= y < OPL
            elif not np.isnan(y_ilm) and not np.isnan(y_opl) and y_ilm <= y < y_opl:
                mask[y, x] = 1
            # Class 2: OPL <= y < IS-OS
            elif not np.isnan(y_opl) and not np.isnan(y_isos) and y_opl <= y < y_isos:
                mask[y, x] = 2
            # Class 3: IS-OS <= y < IBRPE
            elif not np.isnan(y_isos) and not np.isnan(y_ibrpe) and y_isos <= y < y_ibrpe:
                mask[y, x] = 3
            # Class 4: IBRPE <= y < OBRPE
            elif not np.isnan(y_ibrpe) and not np.isnan(y_obrpe) and y_ibrpe <= y < y_obrpe:
                mask[y, x] = 4
            # Class 5: y >= OBRPE
            elif not np.isnan(y_obrpe) and y >= y_obrpe:
                mask[y, x] = 5
            else:
                # If a boundary is missing/NaN, fallback heuristic:
                # We can't perfectly assign it. Assign 0 for now.
                mask[y, x] = 0
                
    return mask

def check_invalid_crossings(boundaries, length=512):
    """
    Returns a list of errors if boundaries cross invalidly.
    """
    errors = []
    
    for x in range(length):
        y_ilm = boundaries['ILM'][x]
        y_opl = boundaries['OPL'][x]
        y_isos = boundaries['IS-OS'][x]
        y_ibrpe = boundaries['IBRPE'][x]
        y_obrpe = boundaries['OBRPE'][x]
        
        if not np.isnan(y_ilm) and not np.isnan(y_opl) and y_ilm > y_opl:
            errors.append(f"ILM > OPL at x={x}")
        if not np.isnan(y_opl) and not np.isnan(y_isos) and y_opl > y_isos:
            errors.append(f"OPL > IS-OS at x={x}")
        if not np.isnan(y_isos) and not np.isnan(y_ibrpe) and y_isos > y_ibrpe:
            errors.append(f"IS-OS > IBRPE at x={x}")
        if not np.isnan(y_ibrpe) and not np.isnan(y_obrpe) and y_ibrpe > y_obrpe:
            errors.append(f"IBRPE > OBRPE at x={x}")
            
    # Deduplicate while preserving order roughly (just return first 5)
    return list(dict.fromkeys(errors))[:5]
