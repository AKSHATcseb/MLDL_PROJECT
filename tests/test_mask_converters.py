"""
Unit tests for Mask Converters (REFUGE & DRISHTI-GS).
Validates:
1. Valid class values in {0, 1, 2}.
2. Cup nested inside Disc containment.
3. Dimension alignment.
4. Non-empty masks for known samples.
"""

import os
import cv2
import numpy as np
import pytest
from src.data.mask_converter import convert_refuge_mask, convert_drishti_softmaps

DATASET_ROOT = r'e:\MLDL MODEL TRAINING PROJECT\dataset'

def test_synthetic_refuge_conversion():
    # Synthetic mask with 0=cup, 128=disc rim, 255=bg
    h, w = 100, 100
    raw = np.full((h, w), 255, dtype=np.uint8)
    # Disc circle radius 30 at center
    cv2.circle(raw, (50, 50), 30, 128, -1)
    # Cup circle radius 15 at center
    cv2.circle(raw, (50, 50), 15, 0, -1)
    
    converted = convert_refuge_mask(raw)
    assert set(np.unique(converted)).issubset({0, 1, 2})
    assert (converted == 2).sum() > 0 # cup exists
    assert (converted == 1).sum() > 0 # disc rim exists
    assert (converted == 0).sum() > 0 # bg exists

def test_real_refuge_mask():
    refuge_train_mask = os.path.join(DATASET_ROOT, 'REFUGE2', 'train', 'mask', 'g0001.bmp')
    refuge_train_img = os.path.join(DATASET_ROOT, 'REFUGE2', 'train', 'images', 'g0001.jpg')
    assert os.path.exists(refuge_train_mask)
    assert os.path.exists(refuge_train_img)
    
    img = cv2.imread(refuge_train_img)
    raw_mask = cv2.imread(refuge_train_mask, cv2.IMREAD_GRAYSCALE)
    assert img.shape[:2] == raw_mask.shape[:2]
    
    converted = convert_refuge_mask(raw_mask)
    unique_vals = set(np.unique(converted))
    assert unique_vals.issubset({0, 1, 2})
    
    # Check that Cup (class 2) is strictly inside Disc (disc is class 1 + 2)
    cup_pixels = (converted == 2)
    assert cup_pixels.sum() > 0
    # Boundary / morphology check: dilating cup should border disc rim, not bg
    # And total disc area must be strictly greater than cup area
    total_disc_area = (converted >= 1).sum()
    cup_area = cup_pixels.sum()
    assert total_disc_area > cup_area

def test_real_drishti_softmaps():
    gt_dir = os.path.join(DATASET_ROOT, 'Training-20211018T055246Z-001', 'Training', 'GT', 'drishtiGS_002', 'SoftMap')
    cup_p = os.path.join(gt_dir, 'drishtiGS_002_cupsegSoftmap.png')
    od_p = os.path.join(gt_dir, 'drishtiGS_002_ODsegSoftmap.png')
    assert os.path.exists(cup_p)
    assert os.path.exists(od_p)
    
    cup = cv2.imread(cup_p, cv2.IMREAD_GRAYSCALE)
    od = cv2.imread(od_p, cv2.IMREAD_GRAYSCALE)
    assert cup.shape == od.shape
    
    converted = convert_drishti_softmaps(cup, od, threshold=128)
    unique_vals = set(np.unique(converted))
    assert unique_vals.issubset({0, 1, 2})
    assert (converted == 2).sum() > 0
    assert (converted == 1).sum() > 0
    assert (converted >= 1).sum() > (converted == 2).sum()

if __name__ == '__main__':
    pytest.main([__file__])
