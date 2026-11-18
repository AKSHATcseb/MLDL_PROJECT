"""
Segmentation Evaluation Metrics.
Calculates per-sample and aggregate:
- Optic Disc (OD = class 1 | class 2) Dice, IoU, Precision, Recall
- Optic Cup (OC = class 2) Dice, IoU, Precision, Recall
- Optic Disc Rim (class 1) Dice, IoU
"""

import numpy as np
import torch
from typing import Dict, Any, Union

def calculate_binary_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    pred_b = pred_binary.astype(bool)
    gt_b = gt_binary.astype(bool)
    
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
    tn = np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum()
    
    intersection = tp
    union = tp + fp + fn
    
    dice = (2.0 * intersection + smooth) / (pred_b.sum() + gt_b.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth) # Sensitivity
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity)
    }

def evaluate_segmentation_batch(
    preds: Union[torch.Tensor, np.ndarray], 
    targets: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    preds: (B, H, W) integer classes {0, 1, 2}
    targets: (B, H, W) integer classes {0, 1, 2}
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    batch_size = preds.shape[0]
    metrics_accum = {
        'disc_dice': [], 'disc_iou': [], 'disc_precision': [], 'disc_recall': [],
        'cup_dice': [], 'cup_iou': [], 'cup_precision': [], 'cup_recall': []
    }
    
    for i in range(batch_size):
        p = preds[i]
        t = targets[i]
        
        # Whole Optic Disc is Rim (1) + Cup (2)
        disc_p = (p >= 1)
        disc_t = (t >= 1)
        disc_m = calculate_binary_metrics(disc_p, disc_t)
        
        # Optic Cup is Class 2
        cup_p = (p == 2)
        cup_t = (t == 2)
        cup_m = calculate_binary_metrics(cup_p, cup_t)
        
        metrics_accum['disc_dice'].append(disc_m['dice'])
        metrics_accum['disc_iou'].append(disc_m['iou'])
        metrics_accum['disc_precision'].append(disc_m['precision'])
        metrics_accum['disc_recall'].append(disc_m['recall'])
        
        metrics_accum['cup_dice'].append(cup_m['dice'])
        metrics_accum['cup_iou'].append(cup_m['iou'])
        metrics_accum['cup_precision'].append(cup_m['precision'])
        metrics_accum['cup_recall'].append(cup_m['recall'])
        
    summary = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
    summary['mean_dice'] = float((summary['disc_dice'] + summary['cup_dice']) / 2.0)
    summary['mean_iou'] = float((summary['disc_iou'] + summary['cup_iou']) / 2.0)
    return summary
