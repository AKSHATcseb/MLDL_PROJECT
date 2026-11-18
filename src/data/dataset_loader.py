"""
PyTorch Datasets & DataLoaders for Glaucoma Segmentation and Classification.
Provides:
- GlaucomaSegmentationDataset
- GlaucomaClassificationDataset
- Loader creation utilities with memory-conscious streaming
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.mask_converter import convert_refuge_mask, convert_drishti_softmaps
from src.data.augmentation import (
    get_segmentation_training_transforms, 
    get_segmentation_eval_transforms,
    get_classification_training_transforms,
    get_classification_eval_transforms
)

class GlaucomaSegmentationDataset(Dataset):
    """
    Segmentation Dataset for Optic Disc and Cup:
    Classes: 0=Background, 1=Optic Disc Rim, 2=Optic Cup.
    """
    def __init__(self, manifest_csv: str, transform: Optional[Callable] = None, is_train: bool = False, image_size=(384, 384)):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform if transform is not None else (
            get_segmentation_training_transforms(image_size) if is_train else get_segmentation_eval_transforms(image_size)
        )
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = row['image_path']
        dataset_name = row['source_dataset']
        mask_path = row['mask_path']
        
        # Load image
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Load and convert mask
        if dataset_name == 'REFUGE':
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                raise ValueError(f"Failed to load REFUGE mask: {mask_path}")
            canonical_mask = convert_refuge_mask(raw_mask)
        elif dataset_name == 'DRISHTI-GS':
            sample_id = row['patient_id']
            cup_soft_p = os.path.join(mask_path, f'{sample_id}_cupsegSoftmap.png')
            od_soft_p = os.path.join(mask_path, f'{sample_id}_ODsegSoftmap.png')
            cup_soft = cv2.imread(cup_soft_p, cv2.IMREAD_GRAYSCALE)
            od_soft = cv2.imread(od_soft_p, cv2.IMREAD_GRAYSCALE)
            if cup_soft is None or od_soft is None:
                raise ValueError(f"Failed to load DRISHTI softmaps for {sample_id}")
            canonical_mask = convert_drishti_softmaps(cup_soft, od_soft, threshold=128)
        else:
            raise ValueError(f"Unsupported segmentation dataset: {dataset_name}")
            
        # Apply albumentations transformation
        augmented = self.transform(image=rgb, mask=canonical_mask)
        image_tensor = augmented['image'] # Shape: (3, H, W), float32 normalized
        mask_tensor = augmented['mask'].long() # Shape: (H, W), int64 class indices
        
        meta = {
            'image_id': row['image_id'],
            'dataset': dataset_name,
            'image_path': img_path,
            'original_shape': rgb.shape[:2]
        }
        
        return image_tensor, mask_tensor, meta

class GlaucomaClassificationDataset(Dataset):
    """
    Binary Classification Dataset for Glaucoma:
    Label: 0=Normal, 1=Glaucoma.
    """
    def __init__(self, manifest_csv: str, transform: Optional[Callable] = None, is_train: bool = False, image_size=(224, 224)):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform if transform is not None else (
            get_classification_training_transforms(image_size) if is_train else get_classification_eval_transforms(image_size)
        )
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = float(row['label'])
        
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=rgb)
        image_tensor = augmented['image'] # (3, H, W)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        meta = {
            'image_id': row['image_id'],
            'dataset': row['source_dataset'],
            'image_path': img_path
        }
        return image_tensor, label_tensor, meta

def get_segmentation_loaders(
    train_csv: str, 
    val_csv: str, 
    batch_size: int = 4, 
    image_size: Tuple[int, int] = (384, 384),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    train_ds = GlaucomaSegmentationDataset(train_csv, is_train=True, image_size=image_size)
    val_ds = GlaucomaSegmentationDataset(val_csv, is_train=False, image_size=image_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader

def get_classification_loaders(
    train_csv: str, 
    val_csv: str, 
    batch_size: int = 16, 
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    train_ds = GlaucomaClassificationDataset(train_csv, is_train=True, image_size=image_size)
    val_ds = GlaucomaClassificationDataset(val_csv, is_train=False, image_size=image_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader
