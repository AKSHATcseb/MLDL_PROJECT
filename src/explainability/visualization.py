"""
Visualization and Comparative Evaluation for Explainable AI (Grad-CAM vs Grad-CAM++).
Generates multi-panel figures for test samples:
[Original Fundus | Grad-CAM Heatmap | Grad-CAM Overlay | Grad-CAM++ Overlay]
Assesses anatomical localization against the Optic Nerve Head.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification.models import get_classification_model
from src.explainability.gradcam import GradCAM, GradCAMPlusPlus, create_heatmap_overlay
from src.data.dataset_loader import get_classification_loaders

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'explainability')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_explainability_pipeline(model_arch: str = 'efficientnet_b0', dataset: str = 'acrima', num_samples: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Generating XAI Visualizations for {model_arch.upper()} on {dataset.upper()} ===")
    
    # 1. Load Model Checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'classification', f"{model_arch}_{dataset}", f"{model_arch}_{dataset}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_classification_model(model_arch, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    
    threshold = ckpt.get('best_threshold', 0.5)
    target_layer = model.get_target_layer()
    
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    
    test_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', f"{dataset}_test.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', f"{dataset}_val.csv")
        
    _, test_loader = get_classification_loaders(test_csv, test_csv, batch_size=1, image_size=(224, 224), num_workers=0)
    
    saved_count = 0
    records = []
    
    for i, (images, labels, metas) in enumerate(test_loader):
        if saved_count >= num_samples:
            break
            
        img_id = metas['image_id'][0]
        gt_label = int(labels[0].item())
        img_tensor = images.to(device)
        
        # De-normalize original image for visualization
        img_cpu = images[0].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = np.clip(img_cpu * std + mean, 0, 1)
        img_uint8 = (img_rgb * 255).astype(np.uint8)
        
        # Generate Grad-CAM
        cam, prob = gradcam.generate_cam(img_tensor)
        heatmap_rgb, cam_overlay = create_heatmap_overlay(img_uint8, cam)
        
        # Generate Grad-CAM++
        cam_pp, _ = gradcam_pp.generate_cam(img_tensor)
        _, cam_pp_overlay = create_heatmap_overlay(img_uint8, cam_pp)
        
        pred_label = 1 if prob >= threshold else 0
        status = "CORRECT" if pred_label == gt_label else "INCORRECT"
        
        # Plot 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        
        axes[0].imshow(img_uint8)
        axes[0].set_title(f"Original Fundus ({img_id})\nGT: {'Glaucoma' if gt_label==1 else 'Normal'}")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_rgb)
        axes[1].set_title(f"Grad-CAM Heatmap\nScore: {prob:.4f} [{status}]")
        axes[1].axis('off')
        
        axes[2].imshow(cam_overlay)
        axes[2].set_title("Grad-CAM Overlay")
        axes[2].axis('off')
        
        axes[3].imshow(cam_pp_overlay)
        axes[3].set_title("Grad-CAM++ Overlay")
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"xai_{model_arch}_{dataset}_sample_{saved_count+1:02d}_{status.lower()}_{img_id}.png")
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        
        records.append({
            'sample_idx': saved_count + 1,
            'image_id': img_id,
            'ground_truth': gt_label,
            'prediction': pred_label,
            'probability': prob,
            'threshold': threshold,
            'status': status,
            'saved_figure': os.path.basename(save_path)
        })
        saved_count += 1
        
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, f"xai_evaluation_{model_arch}_{dataset}.csv"), index=False)
    print(f"XAI visualization complete. Saved {saved_count} visual reports to {OUTPUT_DIR}")

if __name__ == '__main__':
    run_explainability_pipeline(model_arch='efficientnet_b0', dataset='acrima', num_samples=10)
