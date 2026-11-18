"""
Classification Model Architectures for Glaucoma Detection.
Implements:
1. EfficientNet-B0
2. ResNet50
3. DenseNet121
With customizable classification head, transfer learning support, and layer extraction for Grad-CAM.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class GlaucomaEfficientNet(nn.Module):
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1) # Output shape (B,)

    def get_target_layer(self):
        # Final conv block for Grad-CAM
        return self.backbone.features[-1]

class GlaucomaResNet50(nn.Module):
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def get_target_layer(self):
        # Final conv block for Grad-CAM
        return self.backbone.layer4[-1]

class GlaucomaDenseNet121(nn.Module):
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def get_target_layer(self):
        # Final dense block for Grad-CAM
        return self.backbone.features.denseblock4[-1]

def get_classification_model(arch: str = 'efficientnet_b0', pretrained: bool = True) -> nn.Module:
    arch = arch.lower()
    if arch == 'efficientnet_b0':
        return GlaucomaEfficientNet(pretrained=pretrained)
    elif arch == 'resnet50':
        return GlaucomaResNet50(pretrained=pretrained)
    elif arch == 'densenet121':
        return GlaucomaDenseNet121(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
