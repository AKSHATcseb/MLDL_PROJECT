"""
U-Net Segmentation Architecture with Pretrained ResNet34 Encoder.
Output: 3-class logits tensor (0: Background, 1: Optic Disc Rim, 2: Optic Cup).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, encoder_weights: str = "imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None # Return raw logits for CrossEntropyLoss / SoftDiceLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def get_unet_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = "imagenet" if pretrained else None
    return ResNet34UNet(in_channels=3, num_classes=num_classes, encoder_weights=weights)
