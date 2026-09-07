import numpy as np

def generate_fishy_findings(boundaries, qc_status, confidence_stats, features):
    """
    Analyzes technical segmentation and QC anomalies ("Fishy Findings").
    Does NOT diagnose disease.
    """
    warnings = []
    
    # A. Missing Boundaries
    missing_flags = False
    for b_name in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
        arr = boundaries[b_name]
        pct_missing = float(np.isnan(arr).mean() * 100)
        if pct_missing > 5.0: # Moderate heuristic
            missing_flags = True
            severity = "HIGH" if pct_missing > 20.0 else "MODERATE"
            warnings.append({
                'issue': f'Missing {b_name} Boundary',
                'value': f'{pct_missing:.1f}% missing',
                'rule': f'Missing > 5%',
                'reason': f'High interpolation required, reducing reliability of adjacent structural features.',
                'severity': severity
            })

    # B. Anatomical Ordering Violations
    if qc_status == "INVALID":
        warnings.append({
            'issue': 'Severe Anatomical Ordering Violation',
            'value': qc_status,
            'rule': 'Project QC logic (significant boundary inversions)',
            'reason': 'Model predicted layers out of expected vertical sequence.',
            'severity': 'HIGH'
        })
    elif qc_status == "UNCERTAIN":
        warnings.append({
            'issue': 'Minor Anatomical Ordering Violation',
            'value': qc_status,
            'rule': 'Project QC logic (<10% boundary inversions)',
            'reason': 'Minor local crossing of predicted boundaries.',
            'severity': 'MODERATE'
        })
        
    # C/D. Extreme Thickness Anomalies (Technical heuristics only)
    # Using project pixel domain. A typical retinal thickness is 100-300 pixels for 512x512.
    # An individual layer > 200 is highly suspicious or < 2 is suspicious.
    for k, v in features.items():
        if 'mean_pixels' in k and 'central' not in k:
            val = float(v)
            if not np.isnan(val):
                if val < 1.0:
                    warnings.append({
                        'issue': f'Suspiciously Thin Layer ({k})',
                        'value': f'{val:.1f} px',
                        'rule': '< 1.0 px',
                        'reason': 'Predicted layer collapses to sub-pixel width on average.',
                        'severity': 'MODERATE'
                    })
                elif 'Total' not in k and val > 150.0:
                    warnings.append({
                        'issue': f'Suspiciously Thick Individual Layer ({k})',
                        'value': f'{val:.1f} px',
                        'rule': '> 150.0 px',
                        'reason': 'Single layer occupies extremely large portion of B-scan height.',
                        'severity': 'MODERATE'
                    })
                    
    # E. Confidence Proxy
    if confidence_stats['low_prob_pct'] > 15.0:
        severity = "HIGH" if confidence_stats['low_prob_pct'] > 30.0 else "MODERATE"
        warnings.append({
            'issue': 'Low Model Softmax Confidence',
            'value': f"{confidence_stats['low_prob_pct']:.1f}% pixels < 0.6 prob",
            'rule': '> 15% pixels below 0.6',
            'reason': 'Model exhibits widespread uncertainty in its predicted classes.',
            'severity': severity
        })
        
    # F. Discontinuous Boundaries
    # Check for vertical jumps > 20 pixels between adjacent x-columns
    for b_name in ['ILM', 'OPL', 'IS-OS', 'IBRPE', 'OBRPE']:
        arr = boundaries[b_name]
        valid_arr = arr[~np.isnan(arr)]
        if len(valid_arr) > 1:
            jumps = np.abs(np.diff(valid_arr))
            if np.max(jumps) > 30:
                warnings.append({
                    'issue': f'Discontinuous Boundary Jump ({b_name})',
                    'value': f'Max jump {np.max(jumps):.1f} px',
                    'rule': 'Adjacent column jump > 30 px',
                    'reason': 'Abrupt, unnatural vertical step in predicted boundary line.',
                    'severity': 'MODERATE'
                })

    # Summary Assessment
    if any(w['severity'] == 'HIGH' for w in warnings):
        overall_status = "WARNING"
    elif len(warnings) > 0:
        overall_status = "REVIEW"
    else:
        overall_status = "PASS"
        
    return warnings, overall_status
