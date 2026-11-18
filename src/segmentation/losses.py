"""
Loss functions for Multi-Class Retinal Image Segmentation.
Hybrid Loss = CrossEntropyLoss + SoftDiceLoss.
SoftDiceLoss accounts for class imbalance by weighting foreground classes (Disc Rim, Cup).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) integer class indices
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Determine classes to include in loss
        start_c = 1 if self.ignore_background else 0
        total_dice_loss = 0.0
        active_classes = self.num_classes - start_c
        
        for c in range(start_c, self.num_classes):
            p = probs[:, c, :, :]
            t = targets_one_hot[:, c, :, :]
            intersection = torch.sum(p * t, dim=(1, 2))
            union = torch.sum(p, dim=(1, 2)) + torch.sum(t, dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_dice_loss += (1.0 - dice).mean()
            
        return total_dice_loss / max(active_classes, 1)

class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int = 3, ce_weight: float = 0.5, dice_weight: float = 0.5, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weight_t = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight_t)
        self.dice = SoftDiceLoss(num_classes=num_classes, ignore_background=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_l = self.ce(logits, targets)
        dice_l = self.dice(logits, targets)
        return self.ce_weight * ce_l + self.dice_weight * dice_l
