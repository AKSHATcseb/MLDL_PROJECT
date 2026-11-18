"""
Image Preprocessing utilities for Retinal Fundus Photographs.
Handles loading, RGB conversion, letterbox aspect-ratio preserving padding/resizing, and normalization.
"""

import os
import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Union

def load_fundus_image(image_path: str) -> np.ndarray:
    """Loads image and converts to RGB uint8 array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Corrupted or unreadable image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def resize_with_aspect_ratio(
    image: np.ndarray, 
    target_size: Tuple[int, int] = (384, 384), 
    mask: Optional[np.ndarray] = None,
    interpolation_img: int = cv2.INTER_LINEAR,
    interpolation_mask: int = cv2.INTER_NEAREST
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Resizes image (and optional mask) to target_size (height, width).
    Uses letterboxing with zero-padding to strictly preserve retinal anatomical aspect ratio.
    """
    target_h, target_w = target_size
    h, w = image.shape[:2]
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation_img)
    
    # Pad borders
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    if mask is not None:
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=interpolation_mask)
        padded_mask = cv2.copyMakeBorder(
            resized_mask, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        return padded_img, padded_mask
        
    return padded_img

def normalize_tensor(
    image_tensor: torch.Tensor, 
    mean=(0.485, 0.456, 0.406), 
    std=(0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Normalizes a float32 tensor of shape (C, H, W) with range [0, 1]."""
    mean_t = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    return (image_tensor - mean_t) / std_t
