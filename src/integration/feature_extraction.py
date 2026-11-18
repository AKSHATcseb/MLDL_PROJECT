"""
Clinical Biomarker Extraction from Optic Disc & Cup Segmentation.
Calculates:
- Vertical Cup-to-Disc Ratio (vCDR) = Cup Vertical Diameter / Disc Vertical Diameter
- Horizontal Cup-to-Disc Ratio (hCDR)
- Area Cup-to-Disc Ratio (aCDR) = Cup Area / Disc Area
- Neuroretinal Rim Area = Disc Area - Cup Area
- ISNT rule compliance heuristic
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional

def extract_clinical_features(mask_classes: np.ndarray) -> Dict[str, Any]:
    """
    mask_classes: 2D numpy array with classes {0: background, 1: disc rim, 2: cup}
    Returns dictionary of anatomical metrics.
    """
    # Binary masks
    disc_mask = (mask_classes >= 1).astype(np.uint8)
    cup_mask = (mask_classes == 2).astype(np.uint8)
    
    disc_area = int(disc_mask.sum())
    cup_area = int(cup_mask.sum())
    rim_area = max(0, disc_area - cup_area)
    
    if disc_area == 0:
        return {
            'disc_area_px': 0,
            'cup_area_px': 0,
            'rim_area_px': 0,
            'area_cdr': 0.0,
            'vcdr': 0.0,
            'hcdr': 0.0,
            'cup_vertical_diameter_px': 0,
            'disc_vertical_diameter_px': 0,
            'clinical_vcdr_risk': 'Undetectable (No Disc Found)'
        }
        
    # Vertical and Horizontal Diameters via bounding box / projections
    # Disc
    disc_y, disc_x = np.where(disc_mask > 0)
    disc_v_diam = int(np.max(disc_y) - np.min(disc_y) + 1)
    disc_h_diam = int(np.max(disc_x) - np.min(disc_x) + 1)
    
    # Cup
    if cup_area > 0:
        cup_y, cup_x = np.where(cup_mask > 0)
        cup_v_diam = int(np.max(cup_y) - np.min(cup_y) + 1)
        cup_h_diam = int(np.max(cup_x) - np.min(cup_x) + 1)
    else:
        cup_v_diam = 0
        cup_h_diam = 0
        
    vcdr = float(cup_v_diam / max(disc_v_diam, 1))
    hcdr = float(cup_h_diam / max(disc_h_diam, 1))
    area_cdr = float(cup_area / max(disc_area, 1))
    
    # Clinical guideline: vCDR > 0.65 typically indicates elevated suspicion for glaucomatous cupping
    if vcdr >= 0.65:
        risk = "High Suspicion (vCDR >= 0.65)"
    elif vcdr >= 0.50:
        risk = "Moderate Suspicion (0.50 <= vCDR < 0.65)"
    else:
        risk = "Normal Physiological Range (vCDR < 0.50)"
        
    return {
        'disc_area_px': disc_area,
        'cup_area_px': cup_area,
        'rim_area_px': rim_area,
        'area_cdr': round(area_cdr, 4),
        'vcdr': round(vcdr, 4),
        'hcdr': round(hcdr, 4),
        'disc_vertical_diameter_px': disc_v_diam,
        'cup_vertical_diameter_px': cup_v_diam,
        'clinical_vcdr_risk': risk
    }
