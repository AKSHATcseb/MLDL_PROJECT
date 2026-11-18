"""
Data Augmentation pipelines using Albumentations for Retinal Images.
Guarantees:
- Geometric transformations applied synchronously to image and segmentation mask.
- Nearest-neighbor interpolation strictly enforced for discrete mask classes.
- Zero augmentation on validation and test partitions.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

def get_segmentation_training_transforms(image_size: Tuple[int, int] = (384, 384)) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(h, w, interpolation=1), # cv2.INTER_LINEAR for image; albumentations automatically applies INTER_NEAREST to mask
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, 
            scale_limit=0.08, 
            rotate_limit=15, 
            border_mode=0, # BORDER_CONSTANT (black)
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_segmentation_eval_transforms(image_size: Tuple[int, int] = (384, 384)) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(h, w, interpolation=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_classification_training_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(h, w, interpolation=1),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_classification_eval_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    h, w = image_size
    return A.Compose([
        A.Resize(h, w, interpolation=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
