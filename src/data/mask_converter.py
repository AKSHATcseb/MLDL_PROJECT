"""
Mask Converter for Glaucoma Segmentation.
Standardizes heterogeneous segmentation masks from REFUGE and DRISHTI-GS into a unified canonical 3-class target:
0 = Background
1 = Optic Disc Rim (Disc excluding Cup)
2 = Optic Cup (Innermost structure)
"""

import os
import cv2
import numpy as np

def convert_refuge_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    REFUGE Challenge Mask Encoding:
    0   = Optic Cup -> Class 2
    128 = Optic Disc -> Class 1 (Rim)
    255 = Background -> Class 0
    """
    if len(raw_mask.shape) == 3:
        raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
    
    target = np.zeros(raw_mask.shape, dtype=np.uint8)
    
    # Tolerant thresholding for potential JPEG compression/quantization artifacts
    # Cup is around 0
    cup_mask = (raw_mask < 64)
    # Disc is around 128
    disc_mask = (raw_mask >= 64) & (raw_mask < 192)
    
    # Assign classes
    target[disc_mask] = 1   # Optic Disc Rim
    target[cup_mask] = 2    # Optic Cup
    return target

def convert_drishti_softmaps(cup_softmap: np.ndarray, od_softmap: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    DRISHTI-GS SoftMap consensus:
    Values: 0, 64, 128, 191, 255 (4 experts).
    Majority consensus: >= 128 (at least 2 of 4 experts).
    """
    if len(cup_softmap.shape) == 3:
        cup_softmap = cv2.cvtColor(cup_softmap, cv2.COLOR_BGR2GRAY)
    if len(od_softmap.shape) == 3:
        od_softmap = cv2.cvtColor(od_softmap, cv2.COLOR_BGR2GRAY)
        
    binary_cup = (cup_softmap >= threshold)
    binary_od = (od_softmap >= threshold)
    
    # Anatomical consistency: Disc must enclose Cup
    # Union disc with cup so cup is guaranteed inside disc
    binary_od = binary_od | binary_cup
    
    target = np.zeros(od_softmap.shape, dtype=np.uint8)
    target[binary_od] = 1 # Disc rim
    target[binary_cup] = 2 # Optic cup
    return target

def create_mask_overlay(image_rgb: np.ndarray, mask_classes: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Creates a visual color overlay of the 3-class segmentation:
    Class 0 (Background): Unaltered
    Class 1 (Optic Disc): Blue/Cyan tint (RGB: 0, 140, 255)
    Class 2 (Optic Cup): Red/Orange tint (RGB: 255, 60, 60)
    """
    overlay = image_rgb.copy()
    h, w = mask_classes.shape[:2]
    if overlay.shape[:2] != (h, w):
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
        
    color_mask = np.zeros_like(overlay)
    # Color Disc Rim: Blue-Cyan
    color_mask[mask_classes == 1] = [0, 165, 255] # BGR or RGB depending on caller
    # Color Cup: Bright Red/Coral
    color_mask[mask_classes == 2] = [255, 50, 50]
    
    has_foreground = (mask_classes > 0)
    blended = overlay.copy()
    blended[has_foreground] = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)[has_foreground]
    
    # Draw contour outlines for clear boundary delineation
    contours_disc, _ = cv2.findContours((mask_classes >= 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_cup, _ = cv2.findContours((mask_classes == 2).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(blended, contours_disc, -1, (0, 255, 255), 2) # Yellow line for disc boundary
    cv2.drawContours(blended, contours_cup, -1, (0, 0, 255), 2)    # Red line for cup boundary
    
    return blended
