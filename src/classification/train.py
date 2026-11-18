"""
Training and Evaluation Pipeline for Glaucoma Classification.
Supports:
- Architectures: EfficientNet-B0, ResNet50, DenseNet121
- Datasets: ACRIMA, REFUGE, ORIGA, RIM-ONE DL
- Two-stage Transfer Learning with positive-class imbalance weighting
- Validation threshold optimization (Youden's Index)
- ROC, PR, Confusion Matrix, and Loss curve generation
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_curve, precision_recall_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification.models import get_classification_model
from src.evaluation.classification_metrics import evaluate_classification, find_optimal_threshold
from src.data.dataset_loader import get_classification_loaders

def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    num_neg = (df['label'] == 0).sum()
    num_pos = (df['label'] == 1).sum()
    weight = float(num_neg / max(num_pos, 1))
    return torch.tensor([weight], dtype=torch.float32)

def train_epoch(model, loader, optimizer, loss_fn, device, max_batches=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for i, (images, labels, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)

def evaluate_loader(model, loader, loss_fn, device, threshold=0.5, max_batches=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())
            num_batches += 1
            
    avg_loss = total_loss / max(num_batches, 1)
    metrics = evaluate_classification(np.array(all_targets), np.array(all_probs), threshold=threshold)
    metrics['val_loss'] = avg_loss
    return metrics

def plot_classification_artifacts(results_dir: str, metrics: dict, df_history: pd.DataFrame, model_name: str, ds_name: str):
    os.makedirs(results_dir, exist_ok=True)
    y_true = np.array(metrics['y_true'])
    y_probs = np.array(metrics['y_probs'])
    thresh = metrics['threshold']
    
    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal (0)', 'Glaucoma (1)'],
                yticklabels=['Normal (0)', 'Glaucoma (1)'], ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Ground Truth')
    ax.set_title(f"Confusion Matrix: {model_name} on {ds_name}\nThreshold = {thresh:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=120)
    plt.close(fig)
    
    # 2. ROC & PR Curves
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        prec, rec, _ = precision_recall_curve(y_true, y_probs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.3f})", color='tab:blue', lw=2)
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_xlabel('False Positive Rate (1 - Specificity)')
        ax1.set_ylabel('True Positive Rate (Sensitivity)')
        ax1.set_title(f"ROC Curve: {model_name}")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(rec, prec, label=f"PR (AUC = {metrics['pr_auc']:.3f})", color='tab:green', lw=2)
        ax2.set_xlabel('Recall / Sensitivity')
        ax2.set_ylabel('Precision')
        ax2.set_title(f"Precision-Recall Curve: {model_name}")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'roc_pr_curves.png'), dpi=120)
        plt.close(fig)
        
    # 3. Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(df_history['epoch'], df_history['train_loss'], label='Train Loss', color='tab:blue')
    ax1.plot(df_history['epoch'], df_history['val_loss'], label='Val Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(df_history['epoch'], df_history['val_roc_auc'], label='Val ROC-AUC', color='tab:purple')
    ax2.plot(df_history['epoch'], df_history['val_f1'], label='Val F1', color='tab:green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title("Validation AUC & F1 Progression")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'learning_curves.png'), dpi=120)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet50', 'densenet121'])
    parser.add_argument('--dataset', type=str, default='acrima', choices=['acrima', 'origa', 'refuge'])
    parser.add_argument('--stage1_epochs', type=int, default=2)
    parser.add_argument('--stage2_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"=== CLASSIFICATION: {args.arch.upper()} on {args.dataset.upper()} ({device}) ===")
    print(f"=======================================================")
    
    train_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', f"{args.dataset}_train.csv")
    val_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', f"{args.dataset}_val.csv")
    test_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', f"{args.dataset}_test.csv")
    if not os.path.exists(test_csv):
        test_csv = val_csv # Fallback if dedicated test set is val set (e.g. REFUGE)
        
    df_train = pd.read_csv(train_csv)
    pos_weight = compute_pos_weight(df_train).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Computed positive class weight: {pos_weight.item():.3f} (for imbalance mitigation)")
    
    train_loader, val_loader = get_classification_loaders(
        train_csv, val_csv, batch_size=args.batch_size, image_size=(224, 224), num_workers=0
    )
    _, test_loader = get_classification_loaders(
        train_csv, test_csv, batch_size=args.batch_size, image_size=(224, 224), num_workers=0
    )
    
    model = get_classification_model(args.arch, pretrained=True).to(device)
    
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'classification', f"{args.arch}_{args.dataset}")
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'classification', f"{args.arch}_{args.dataset}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    total_epochs = 1 if args.smoke_test else (args.stage1_epochs + args.stage2_epochs)
    max_batches = 3 if args.smoke_test else None
    
    # Stage 1: Freeze backbone
    if not args.smoke_test and args.stage1_epochs > 0:
        print("\n--- Stage 1: Training Classification Head with Frozen Backbone ---")
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Unfreeze head
        head = model.backbone.classifier if hasattr(model.backbone, 'classifier') else model.backbone.fc
        for param in head.parameters():
            param.requires_grad = True
            
        optimizer = AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
        for ep in range(1, args.stage1_epochs + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, max_batches)
            val_m = evaluate_loader(model, val_loader, loss_fn, device, max_batches=max_batches)
            print(f"Stage 1 Ep [{ep}/{args.stage1_epochs}] ({time.time()-t0:.1f}s) | Train Loss: {tr_loss:.4f} | Val Loss: {val_m['val_loss']:.4f} | Val AUC: {val_m['roc_auc']:.4f}")

    # Stage 2: Fine-tune end-to-end
    print("\n--- Stage 2: End-to-End Fine-Tuning ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = AdamW(model.parameters(), lr=args.backbone_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_auc = -1.0
    best_thresh = 0.5
    history = []
    
    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, max_batches)
        val_m = evaluate_loader(model, val_loader, loss_fn, device, max_batches=max_batches)
        scheduler.step(val_m['roc_auc'])
        
        ep_time = time.time() - t0
        print(f"Epoch [{ep:02d}/{total_epochs:02d}] ({ep_time:.1f}s) | Train Loss: {tr_loss:.4f} | Val Loss: {val_m['val_loss']:.4f} | Val AUC: {val_m['roc_auc']:.4f} | Val F1: {val_m['f1_score']:.4f}")
        
        history.append({
            'epoch': ep,
            'train_loss': tr_loss,
            'val_loss': val_m['val_loss'],
            'val_roc_auc': val_m['roc_auc'],
            'val_pr_auc': val_m['pr_auc'],
            'val_accuracy': val_m['accuracy'],
            'val_sensitivity': val_m['sensitivity'],
            'val_specificity': val_m['specificity'],
            'val_f1': val_m['f1_score']
        })
        
        if val_m['roc_auc'] > best_val_auc:
            best_val_auc = val_m['roc_auc']
            # Find optimal threshold using validation set
            best_thresh = find_optimal_threshold(np.array(val_m['y_true']), np.array(val_m['y_probs']))
            best_path = os.path.join(ckpt_dir, f"{args.arch}_{args.dataset}_best.pth")
            torch.save({
                'epoch': ep,
                'model_state': model.state_dict(),
                'val_metrics': val_m,
                'best_threshold': best_thresh,
                'config': vars(args)
            }, best_path)
            print(f"  --> [*] Best Checkpoint Saved! Val AUC: {best_val_auc:.4f} | Optimal Val Threshold: {best_thresh:.3f}")
            
    # Load best checkpoint and evaluate on Held-Out Test Set
    if not args.smoke_test:
        best_ckpt = torch.load(os.path.join(ckpt_dir, f"{args.arch}_{args.dataset}_best.pth"), map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state'])
        optimal_thresh = best_ckpt['best_threshold']
        
        print(f"\n=== Final Evaluation on Held-Out TEST Set (Threshold = {optimal_thresh:.3f}) ===")
        test_metrics = evaluate_loader(model, test_loader, loss_fn, device, threshold=optimal_thresh)
        
        print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"Sensitivity: {test_metrics['sensitivity']:.4f} (Glaucoma Recall)")
        print(f"Specificity: {test_metrics['specificity']:.4f}")
        print(f"Precision:   {test_metrics['precision']:.4f}")
        print(f"F1-Score:    {test_metrics['f1_score']:.4f}")
        print(f"ROC-AUC:     {test_metrics['roc_auc']:.4f}")
        print(f"PR-AUC:      {test_metrics['pr_auc']:.4f}")
        print(f"Confusion Matrix: TN={test_metrics['confusion_matrix']['tn']}, FP={test_metrics['confusion_matrix']['fp']}, FN={test_metrics['confusion_matrix']['fn']}, TP={test_metrics['confusion_matrix']['tp']}")
        
        # Save artifacts
        df_hist = pd.DataFrame(history)
        df_hist.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
        
        test_summary = {k: v for k, v in test_metrics.items() if k not in ['y_true', 'y_probs']}
        pd.DataFrame([test_summary]).to_csv(os.path.join(results_dir, 'test_evaluation_metrics.csv'), index=False)
        
        plot_classification_artifacts(results_dir, test_metrics, df_hist, args.arch, args.dataset)
        print(f"Results and plots saved to {results_dir}")

if __name__ == '__main__':
    main()
