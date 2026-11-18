"""
Attention U-Net for Retinal Image Segmentation.
Implements Additive Attention Gates on skip connections (Oktay et al., 2018)
to focus feature maps on the optic disc and optic cup regions while suppressing peripheral artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        f_g: gating signal channels (deeper layer)
        f_l: skip connection channels (shallower layer)
        f_int: intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Handle spatial dimension alignment if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32):
        super().__init__()
        nb = base_channels # 32
        
        # Encoder
        self.conv1 = ConvBlock(in_channels, nb)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = ConvBlock(nb, nb * 2) # 64
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = ConvBlock(nb * 2, nb * 4) # 128
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = ConvBlock(nb * 4, nb * 8) # 256
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(nb * 8, nb * 16) # 512
        
        # Decoder with Attention Gates
        self.up5 = UpConvBlock(nb * 16, nb * 8)
        self.ag5 = AttentionGate(f_g=nb * 8, f_l=nb * 8, f_int=nb * 4)
        self.conv5 = ConvBlock(nb * 16, nb * 8)
        
        self.up4 = UpConvBlock(nb * 8, nb * 4)
        self.ag4 = AttentionGate(f_g=nb * 4, f_l=nb * 4, f_int=nb * 2)
        self.conv4_dec = ConvBlock(nb * 8, nb * 4)
        
        self.up3 = UpConvBlock(nb * 4, nb * 2)
        self.ag3 = AttentionGate(f_g=nb * 2, f_l=nb * 2, f_int=nb)
        self.conv3_dec = ConvBlock(nb * 4, nb * 2)
        
        self.up2 = UpConvBlock(nb * 2, nb)
        self.ag2 = AttentionGate(f_g=nb, f_l=nb, f_int=nb // 2)
        self.conv2_dec = ConvBlock(nb * 2, nb)
        
        # Final segmentation head
        self.out_conv = nn.Conv2d(nb, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool1(e1))
        e3 = self.conv3(self.pool2(e2))
        e4 = self.conv4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder stage 1
        d5 = self.up5(b)
        x4 = self.ag5(g=d5, x=e4)
        d5 = torch.cat([x4, d5], dim=1)
        d5 = self.conv5(d5)
        
        # Decoder stage 2
        d4 = self.up4(d5)
        x3 = self.ag4(g=d4, x=e3)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.conv4_dec(d4)
        
        # Decoder stage 3
        d3 = self.up3(d4)
        x2 = self.ag3(g=d3, x=e2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv3_dec(d3)
        
        # Decoder stage 4
        d2 = self.up2(d3)
        x1 = self.ag2(g=d2, x=e1)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.conv2_dec(d2)
        
        out = self.out_conv(d2)
        return out

def get_attention_unet_model(num_classes: int = 3, base_channels: int = 32) -> nn.Module:
    return AttentionUNet(in_channels=3, num_classes=num_classes, base_channels=base_channels)
