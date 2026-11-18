"""
Explainable AI (XAI) Suite: Grad-CAM and Grad-CAM++ for Glaucoma Classification.
Computes activation-gradient heatmaps from the final convolutional feature layers.
Provides overlay rendering, threshold masking, and anatomical plausibility validation.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Generates Grad-CAM heatmap.
        input_tensor: shape (1, 3, H, W)
        Returns:
            cam: 2D numpy array [H, W] normalized in [0, 1]
            score: sigmoid glaucoma probability
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor) # Single logit
        prob = torch.sigmoid(output).item()
        
        # Compute gradient with respect to output logit
        output.backward(retain_graph=True)
        
        # Gradients: (1, C, H_feat, W_feat)
        # Global average pooling on gradients -> weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Discard negative gradients
        
        # Interpolate to input image size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize in [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, prob

class GradCAMPlusPlus(GradCAM):
    def generate_cam(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Grad-CAM++ with higher-order gradient weighting."""
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        prob = torch.sigmoid(output).item()
        
        # Score exp
        score = torch.exp(output)
        score.backward(retain_graph=True)
        
        grads = self.gradients
        acts = self.activations
        
        # Grad-CAM++ alpha weighting
        grads_2 = grads.pow(2)
        grads_3 = grads.pow(3)
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)
        eps = 1e-7
        
        alpha = grads_2 / (2.0 * grads_2 + sum_acts * grads_3 + eps)
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, prob

def create_heatmap_overlay(
    image_rgb: np.ndarray, 
    cam_normalized: np.ndarray, 
    colormap: int = cv2.COLORMAP_JET, 
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays normalized [0, 1] CAM on RGB uint8 image.
    Returns:
        colored_heatmap: (H, W, 3) RGB
        overlay: (H, W, 3) RGB blended image
    """
    h, w = image_rgb.shape[:2]
    if cam_normalized.shape != (h, w):
        cam_normalized = cv2.resize(cam_normalized, (w, h), interpolation=cv2.INTER_LINEAR)
        
    cam_uint8 = np.uint8(255 * cam_normalized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
    return heatmap_rgb, overlay
