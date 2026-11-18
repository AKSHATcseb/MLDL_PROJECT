"""
Unit tests for Segmentation Architectures (ResNet34 U-Net & Attention U-Net).
Verifies:
1. Output dimensions match input dimensions.
2. Forward and backward pass.
3. Loss calculation and gradients.
4. Metric evaluation functions.
"""

import os
import sys
import torch
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation.unet import get_unet_model
from src.segmentation.attention_unet import get_attention_unet_model
from src.segmentation.losses import CombinedLoss, SoftDiceLoss
from src.evaluation.segmentation_metrics import evaluate_segmentation_batch

def test_unet_forward_backward():
    model = get_unet_model(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3, 128, 128)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_attention_unet_forward_backward():
    model = get_attention_unet_model(num_classes=3, base_channels=16)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3, 128, 128)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_combined_loss():
    loss_fn = CombinedLoss(num_classes=3)
    logits = torch.randn(2, 3, 64, 64)
    targets = torch.randint(0, 3, (2, 64, 64)).long()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0

def test_evaluation_metrics():
    # Perfect prediction test
    targets = torch.zeros((2, 64, 64), dtype=torch.long)
    targets[:, 20:40, 20:40] = 1 # Disc rim
    targets[:, 25:35, 25:35] = 2 # Cup
    
    preds = targets.clone()
    metrics = evaluate_segmentation_batch(preds, targets)
    assert metrics['disc_dice'] > 0.99
    assert metrics['cup_dice'] > 0.99
    assert metrics['disc_iou'] > 0.99

if __name__ == '__main__':
    pytest.main([__file__])
