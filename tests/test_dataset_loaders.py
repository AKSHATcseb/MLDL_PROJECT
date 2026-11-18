"""
Unit tests for GlaucomaSegmentationDataset and GlaucomaClassificationDataset.
"""

import os
import sys
import torch
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import (
    GlaucomaSegmentationDataset,
    GlaucomaClassificationDataset,
    get_segmentation_loaders,
    get_classification_loaders
)

REFUGE_TRAIN_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation', 'refuge_train_320.csv')
REFUGE_VAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation', 'refuge_val_80.csv')
ACRIMA_TRAIN_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', 'acrima_train.csv')
ACRIMA_VAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', 'acrima_val.csv')

def test_segmentation_dataset():
    assert os.path.exists(REFUGE_TRAIN_CSV)
    ds = GlaucomaSegmentationDataset(REFUGE_TRAIN_CSV, is_train=True, image_size=(256, 256))
    assert len(ds) == 320
    
    img, mask, meta = ds[0]
    assert img.shape == (3, 256, 256)
    assert mask.shape == (256, 256)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert set(mask.unique().tolist()).issubset({0, 1, 2})

def test_segmentation_dataloader():
    train_loader, val_loader = get_segmentation_loaders(
        REFUGE_TRAIN_CSV, REFUGE_VAL_CSV, batch_size=2, image_size=(256, 256), num_workers=0
    )
    imgs, masks, metas = next(iter(train_loader))
    assert imgs.shape == (2, 3, 256, 256)
    assert masks.shape == (2, 256, 256)

def test_classification_dataset():
    assert os.path.exists(ACRIMA_TRAIN_CSV)
    ds = GlaucomaClassificationDataset(ACRIMA_TRAIN_CSV, is_train=True, image_size=(224, 224))
    assert len(ds) > 0
    img, label, meta = ds[0]
    assert img.shape == (3, 224, 224)
    assert label.item() in (0.0, 1.0)

def test_classification_dataloader():
    train_loader, val_loader = get_classification_loaders(
        ACRIMA_TRAIN_CSV, ACRIMA_VAL_CSV, batch_size=4, image_size=(224, 224), num_workers=0
    )
    imgs, labels, metas = next(iter(train_loader))
    assert imgs.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)

if __name__ == '__main__':
    pytest.main([__file__])
