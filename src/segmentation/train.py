"""
Training Script for Optic Disc & Cup Semantic Segmentation.
Supports:
- Pretrained ResNet34 U-Net
- Attention U-Net
- Hybrid CrossEntropy + SoftDice loss
- Smoke-test mode for fast verification
- Full checkpointing, metrics tracking, and visualization saving.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation.unet import get_unet_model
from src.segmentation.attention_unet import get_attention_unet_model
from src.segmentation.losses import CombinedLoss
from src.evaluation.segmentation_metrics import evaluate_segmentation_batch
from src.data.dataset_loader import get_segmentation_loaders
from src.data.mask_converter import create_mask_overlay

def train_one_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for i, (images, masks, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)

def evaluate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metric_accum = {
        'disc_dice': [], 'cup_dice': [], 'mean_dice': [],
        'disc_iou': [], 'cup_iou': [], 'mean_iou': []
    }
    
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            batch_metrics = evaluate_segmentation_batch(preds, masks)
            
            for k in metric_accum:
                metric_accum[k].append(batch_metrics[k])
            num_batches += 1
            
    avg_loss = total_loss / max(num_batches, 1)
    summary_metrics = {k: float(np.mean(v)) for k, v in metric_accum.items()}
    summary_metrics['val_loss'] = avg_loss
    return summary_metrics

def save_visual_predictions(model, loader, device, save_dir, epoch, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    samples_saved = 0
    
    with torch.no_grad():
        for images, masks, metas in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            images_cpu = images.cpu().permute(0, 2, 3, 1).numpy()
            
            # De-normalize images
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            images_cpu = np.clip(images_cpu * std + mean, 0, 1)
            
            for b in range(images.shape[0]):
                if samples_saved >= num_samples:
                    return
                img = (images_cpu[b] * 255).astype(np.uint8)
                gt = masks[b]
                pred = preds[b]
                
                gt_overlay = create_mask_overlay(img, gt)
                pred_overlay = create_mask_overlay(img, pred)
                
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img)
                axes[0].set_title(f"Image: {metas['image_id'][b]}")
                axes[0].axis('off')
                
                axes[1].imshow(gt, cmap='nipy_spectral', vmin=0, vmax=2)
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis('off')
                
                axes[2].imshow(pred, cmap='nipy_spectral', vmin=0, vmax=2)
                axes[2].set_title("Predicted Mask")
                axes[2].axis('off')
                
                axes[3].imshow(pred_overlay)
                axes[3].set_title("Prediction Overlay")
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"val_pred_ep{epoch}_{samples_saved+1}.png"), dpi=100)
                plt.close(fig)
                samples_saved += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='unet', choices=['unet', 'attention_unet'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--smoke_test', action='store_true', help='Run 1 epoch smoke test on small subset')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Segmentation Training [{args.arch.upper()}] on {device} ===")
    
    train_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation', 'refuge_train_320.csv')
    val_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation', 'refuge_val_80.csv')
    
    train_loader, val_loader = get_segmentation_loaders(
        train_csv, val_csv, batch_size=args.batch_size, image_size=(args.img_size, args.img_size), num_workers=0
    )
    
    if args.arch == 'unet':
        model = get_unet_model(num_classes=3, pretrained=True)
    else:
        model = get_attention_unet_model(num_classes=3, base_channels=24)
        
    model = model.to(device)
    loss_fn = CombinedLoss(num_classes=3, ce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'segmentation', args.arch)
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'segmentation', args.arch)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    epochs = 1 if args.smoke_test else args.epochs
    max_batches = 3 if args.smoke_test else None
    
    best_val_dice = -1.0
    history = []
    
    start_time = time.time()
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches)
        val_metrics = evaluate(model, val_loader, loss_fn, device, max_batches)
        scheduler.step()
        
        ep_duration = time.time() - ep_start
        val_loss = val_metrics['val_loss']
        disc_dice = val_metrics['disc_dice']
        cup_dice = val_metrics['cup_dice']
        mean_dice = val_metrics['mean_dice']
        
        record = {
            'epoch': ep,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'disc_dice': disc_dice,
            'cup_dice': cup_dice,
            'mean_dice': mean_dice,
            'disc_iou': val_metrics['disc_iou'],
            'cup_iou': val_metrics['cup_iou'],
            'mean_iou': val_metrics['mean_iou'],
            'lr': optimizer.param_groups[0]['lr'],
            'duration_s': ep_duration
        }
        history.append(record)
        print(f"Epoch [{ep:02d}/{epochs:02d}] ({ep_duration:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Disc Dice: {disc_dice:.4f} | Cup Dice: {cup_dice:.4f} | Mean Dice: {mean_dice:.4f}")
        
        # Save checkpoints
        latest_path = os.path.join(checkpoint_dir, f"{args.arch}_latest.pth")
        torch.save({'epoch': ep, 'model_state': model.state_dict(), 'config': vars(args), 'metrics': record}, latest_path)
        
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            best_path = os.path.join(checkpoint_dir, f"{args.arch}_best.pth")
            torch.save({'epoch': ep, 'model_state': model.state_dict(), 'config': vars(args), 'metrics': record}, best_path)
            print(f"  --> [*] New Best Checkpoint Saved! Mean Dice: {best_val_dice:.4f}")
            
    total_time = time.time() - start_time
    print(f"Completed in {total_time:.1f}s. Best Val Mean Dice: {best_val_dice:.4f}")
    
    # Save metrics CSV and visual predictions
    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(results_dir, "training_metrics.csv")
    df_hist.to_csv(csv_path, index=False)
    
    if not args.smoke_test:
        save_visual_predictions(model, val_loader, device, os.path.join(results_dir, "predictions"), epoch=epochs)
        
        # Plot curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df_hist['epoch'], df_hist['train_loss'], label='Train Loss', color='tab:blue')
        ax1.plot(df_hist['epoch'], df_hist['val_loss'], label='Val Loss', color='tab:orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Combined Loss')
        ax1.set_title(f"{args.arch.upper()} - Loss Curve")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(df_hist['epoch'], df_hist['disc_dice'], label='Disc Dice', color='tab:cyan')
        ax2.plot(df_hist['epoch'], df_hist['cup_dice'], label='Cup Dice', color='tab:red')
        ax2.plot(df_hist['epoch'], df_hist['mean_dice'], label='Mean Dice', color='tab:green', linestyle='--')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Coefficient')
        ax2.set_title(f"{args.arch.upper()} - Validation Dice Progression")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "learning_curves.png"), dpi=120)
        plt.close(fig)

if __name__ == '__main__':
    main()
