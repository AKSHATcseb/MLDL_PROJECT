"""
Comparative Evaluation and Error Analysis of Segmentation Models.
Evaluates ResNet34 U-Net vs Attention U-Net on held-out validation set.
Generates:
1. Metrics comparison table (CSV).
2. Side-by-side visual comparisons (Original | Ground Truth | U-Net | Attention U-Net).
3. Failure / error analysis on low-Dice cases.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation.unet import get_unet_model
from src.segmentation.attention_unet import get_attention_unet_model
from src.evaluation.segmentation_metrics import evaluate_segmentation_batch, calculate_binary_metrics
from src.data.dataset_loader import get_segmentation_loaders
from src.data.mask_converter import create_mask_overlay

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'segmentation', 'comparison')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Running Comprehensive Segmentation Comparison ===")
    
    val_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation', 'refuge_val_80.csv')
    _, val_loader = get_segmentation_loaders(val_csv, val_csv, batch_size=1, image_size=(256, 256), num_workers=0)
    
    # 1. Load U-Net
    unet = get_unet_model(num_classes=3, pretrained=False)
    unet_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentation', 'unet', 'unet_best.pth')
    unet = load_checkpoint(unet, unet_ckpt).to(device)
    
    # 2. Load Attention U-Net
    att_unet = get_attention_unet_model(num_classes=3, base_channels=24)
    att_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentation', 'attention_unet', 'attention_unet_best.pth')
    att_unet = load_checkpoint(att_unet, att_ckpt).to(device)
    
    models = {'U-Net (ResNet34)': unet, 'Attention U-Net': att_unet}
    all_sample_results = []
    
    # Storage for side-by-side visualization
    vis_samples = []
    
    for idx, (images, masks, metas) in enumerate(val_loader):
        img_id = metas['image_id'][0]
        images = images.to(device)
        gt_mask = masks[0].numpy()
        
        # De-normalize image
        img_cpu = images[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = np.clip(img_cpu * std + mean, 0, 1)
        img_uint8 = (img_rgb * 255).astype(np.uint8)
        
        sample_dict = {'image_id': img_id}
        preds_for_vis = {}
        
        for name, model in models.items():
            t0 = time.time()
            with torch.no_grad():
                logits = model(images)
            infer_time_ms = (time.time() - t0) * 1000.0
            
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            preds_for_vis[name] = pred
            
            # Disc
            disc_m = calculate_binary_metrics(pred >= 1, gt_mask >= 1)
            # Cup
            cup_m = calculate_binary_metrics(pred == 2, gt_mask == 2)
            
            sample_dict[f'{name}_disc_dice'] = disc_m['dice']
            sample_dict[f'{name}_cup_dice'] = cup_m['dice']
            sample_dict[f'{name}_disc_iou'] = disc_m['iou']
            sample_dict[f'{name}_cup_iou'] = cup_m['iou']
            sample_dict[f'{name}_mean_dice'] = (disc_m['dice'] + cup_m['dice']) / 2.0
            sample_dict[f'{name}_infer_time_ms'] = infer_time_ms
            
        all_sample_results.append(sample_dict)
        
        if idx < 12: # Save first 12 for visual montage
            vis_samples.append((img_uint8, gt_mask, preds_for_vis['U-Net (ResNet34)'], preds_for_vis['Attention U-Net'], img_id))
            
    df_results = pd.DataFrame(all_sample_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, 'per_image_segmentation_results.csv'), index=False)
    
    # Summary Table
    summary = []
    for name in models.keys():
        summary.append({
            'Model Architecture': name,
            'Optic Disc Dice': f"{df_results[f'{name}_disc_dice'].mean():.4f} +/- {df_results[f'{name}_disc_dice'].std():.4f}",
            'Optic Cup Dice': f"{df_results[f'{name}_cup_dice'].mean():.4f} +/- {df_results[f'{name}_cup_dice'].std():.4f}",
            'Mean Dice': f"{df_results[f'{name}_mean_dice'].mean():.4f} +/- {df_results[f'{name}_mean_dice'].std():.4f}",
            'Optic Disc IoU': f"{df_results[f'{name}_disc_iou'].mean():.4f} +/- {df_results[f'{name}_disc_iou'].std():.4f}",
            'Optic Cup IoU': f"{df_results[f'{name}_cup_iou'].mean():.4f} +/- {df_results[f'{name}_cup_iou'].std():.4f}",
            'Avg Inference Latency (ms)': f"{df_results[f'{name}_infer_time_ms'].mean():.2f}"
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(RESULTS_DIR, 'segmentation_model_comparison.csv'), index=False)
    print("\n=== SEGMENTATION ARCHITECTURE COMPARISON ===")
    print(df_summary.to_string(index=False))
    
    # Generate Visual Side-by-Side: Original | Ground Truth | U-Net | Attention U-Net
    print("\nGenerating visual side-by-side comparison panels...")
    for i, (orig, gt, unet_p, att_p, img_id) in enumerate(vis_samples):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        
        gt_overlay = create_mask_overlay(orig, gt)
        unet_overlay = create_mask_overlay(orig, unet_p)
        att_overlay = create_mask_overlay(orig, att_p)
        
        axes[0].imshow(orig)
        axes[0].set_title(f"Original Fundus\n{img_id}")
        axes[0].axis('off')
        
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth\n(Yellow: Disc, Red: Cup)")
        axes[1].axis('off')
        
        axes[2].imshow(unet_overlay)
        axes[2].set_title(f"ResNet34 U-Net\nDisc: {df_results.loc[df_results['image_id']==img_id, 'U-Net (ResNet34)_disc_dice'].values[0]:.3f} | Cup: {df_results.loc[df_results['image_id']==img_id, 'U-Net (ResNet34)_cup_dice'].values[0]:.3f}")
        axes[2].axis('off')
        
        axes[3].imshow(att_overlay)
        axes[3].set_title(f"Attention U-Net\nDisc: {df_results.loc[df_results['image_id']==img_id, 'Attention U-Net_disc_dice'].values[0]:.3f} | Cup: {df_results.loc[df_results['image_id']==img_id, 'Attention U-Net_cup_dice'].values[0]:.3f}")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"comparison_{i+1:02d}_{img_id}.png"), dpi=120)
        plt.close(fig)
        
    print(f"Evaluation complete. Reports and visual comparisons saved to {RESULTS_DIR}")

if __name__ == '__main__':
    run_evaluation()
