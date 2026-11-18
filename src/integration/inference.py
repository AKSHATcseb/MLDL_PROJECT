"""
Unified End-to-End Glaucoma Inference Engine.
Combines:
1. Retinal Fundus Image Preprocessing
2. Deep Semantic Segmentation (ResNet34 U-Net)
3. Anatomical Feature & vCDR Extraction
4. Glaucoma Deep Classification (EfficientNet-B0)
5. Explainable AI Heatmap & Overlay Generation (Grad-CAM)
"""

import os
import sys
import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation.unet import get_unet_model
from src.classification.models import get_classification_model
from src.explainability.gradcam import GradCAM, create_heatmap_overlay
from src.data.mask_converter import create_mask_overlay
from src.integration.feature_extraction import extract_clinical_features

class GlaucomaInferenceEngine:
    def __init__(
        self, 
        seg_ckpt_path: str = None, 
        cls_ckpt_path: str = None, 
        device: str = None
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Default checkpoint paths
        if seg_ckpt_path is None:
            seg_ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentation', 'unet', 'unet_best.pth')
        if cls_ckpt_path is None:
            cls_ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'classification', 'efficientnet_b0_acrima', 'efficientnet_b0_acrima_best.pth')
            
        print(f"Loading Segmentation Model from: {seg_ckpt_path}")
        self.seg_model = get_unet_model(num_classes=3, pretrained=False)
        seg_ckpt = torch.load(seg_ckpt_path, map_location=self.device, weights_only=False)
        self.seg_model.load_state_dict(seg_ckpt['model_state'])
        self.seg_model = self.seg_model.to(self.device).eval()
        
        print(f"Loading Classification Model from: {cls_ckpt_path}")
        self.cls_model = get_classification_model('efficientnet_b0', pretrained=False)
        cls_ckpt = torch.load(cls_ckpt_path, map_location=self.device, weights_only=False)
        self.cls_model.load_state_dict(cls_ckpt['model_state'])
        self.cls_model = self.cls_model.to(self.device).eval()
        self.cls_threshold = cls_ckpt.get('best_threshold', 0.5)
        
        # Setup Grad-CAM
        self.gradcam = GradCAM(self.cls_model, self.cls_model.get_target_layer())
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, img_rgb: np.ndarray, target_size: tuple) -> torch.Tensor:
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        normalized = (resized / 255.0 - self.mean) / self.std
        tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def predict(self, input_image: np.ndarray) -> dict:
        """
        Accepts RGB image as uint8 numpy array.
        Returns comprehensive analysis dictionary.
        """
        orig_h, orig_w = input_image.shape[:2]
        
        # 1. Segmentation Pass (256x256 or 384x384)
        seg_tensor = self.preprocess_image(input_image, (256, 256)).to(self.device)
        with torch.no_grad():
            seg_logits = self.seg_model(seg_tensor)
            pred_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()
            
        # Resize predicted mask back to original image dimensions (Nearest Neighbor)
        pred_mask_orig = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Extract Clinical Biomarkers
        features = extract_clinical_features(pred_mask_orig)
        
        # 3. Mask Overlays
        mask_overlay = create_mask_overlay(input_image, pred_mask_orig)
        disc_binary = ((pred_mask_orig >= 1) * 255).astype(np.uint8)
        cup_binary = ((pred_mask_orig == 2) * 255).astype(np.uint8)
        
        # 4. Classification & Grad-CAM Pass (224x224)
        cls_tensor = self.preprocess_image(input_image, (224, 224)).to(self.device)
        cam, prob = self.gradcam.generate_cam(cls_tensor)
        
        pred_label = "Glaucoma" if prob >= self.cls_threshold else "Normal / Non-Glaucoma"
        heatmap_rgb, cam_overlay = create_heatmap_overlay(input_image, cam)
        
        return {
            'original_image': input_image,
            'predicted_mask': pred_mask_orig,
            'mask_overlay': mask_overlay,
            'optic_disc_mask': disc_binary,
            'optic_cup_mask': cup_binary,
            'glaucoma_probability': float(prob),
            'prediction_label': pred_label,
            'operating_threshold': float(self.cls_threshold),
            'gradcam_heatmap': heatmap_rgb,
            'gradcam_overlay': cam_overlay,
            'clinical_features': features,
            'disclaimer': "This application is an AI-based research prototype. It is not a substitute for professional ophthalmological examination or medical diagnosis."
        }
