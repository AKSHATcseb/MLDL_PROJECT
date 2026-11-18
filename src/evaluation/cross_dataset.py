"""
Cross-Dataset Generalization & Domain Shift Study (Part 12).
Evaluates models trained on Source datasets across independent, external Target datasets:
- ACRIMA -> REFUGE
- ACRIMA -> ORIGA
- REFUGE -> ACRIMA
- REFUGE -> ORIGA
Generates cross-dataset generalization matrix and quantitative domain shift analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification.models import get_classification_model
from src.evaluation.classification_metrics import evaluate_classification
from src.data.dataset_loader import get_classification_loaders

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'comparisons')
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_cross_dataset(model_ckpt_path: str, target_csv: str, target_name: str, device: torch.device):
    ckpt = torch.load(model_ckpt_path, map_location=device, weights_only=False)
    arch = ckpt['config']['arch']
    source_ds = ckpt['config']['dataset']
    threshold = ckpt.get('best_threshold', 0.5)
    
    model = get_classification_model(arch, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    
    _, target_loader = get_classification_loaders(
        target_csv, target_csv, batch_size=16, image_size=(224, 224), num_workers=0
    )
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, _ in target_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.numpy())
            
    metrics = evaluate_classification(np.array(all_targets), np.array(all_probs), threshold=threshold)
    return {
        'Source Dataset (Trained)': source_ds.upper(),
        'Target Dataset (Tested)': target_name.upper(),
        'Architecture': arch,
        'ROC-AUC': round(metrics['roc_auc'], 4),
        'Sensitivity': round(metrics['sensitivity'], 4),
        'Specificity': round(metrics['specificity'], 4),
        'Accuracy': round(metrics['accuracy'], 4),
        'F1-Score': round(metrics['f1_score'], 4),
        'Operating Threshold': round(threshold, 3)
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Running Systematic Cross-Dataset Generalization Experiments ===")
    
    acrima_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'classification', 'efficientnet_b0_acrima', 'efficientnet_b0_acrima_best.pth')
    refuge_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints', 'classification', 'efficientnet_b0_refuge', 'efficientnet_b0_refuge_best.pth')
    
    refuge_test_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', 'refuge_val.csv')
    acrima_test_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', 'acrima_test.csv')
    origa_test_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'classification', 'origa_test.csv')
    
    experiments = [
        # In-domain baselines
        (acrima_ckpt, acrima_test_csv, 'ACRIMA (In-Domain)'),
        (refuge_ckpt, refuge_test_csv, 'REFUGE (In-Domain)'),
        # Cross-dataset transfers
        (acrima_ckpt, refuge_test_csv, 'REFUGE (External)'),
        (acrima_ckpt, origa_test_csv, 'ORIGA (External)'),
        (refuge_ckpt, acrima_test_csv, 'ACRIMA (External)'),
        (refuge_ckpt, origa_test_csv, 'ORIGA (External)'),
    ]
    
    results = []
    for ckpt_path, test_csv, target_name in experiments:
        print(f"Evaluating {os.path.basename(os.path.dirname(ckpt_path))} on {target_name}...")
        rec = evaluate_cross_dataset(ckpt_path, test_csv, target_name, device)
        results.append(rec)
        
    df_matrix = pd.DataFrame(results)
    csv_out = os.path.join(RESULTS_DIR, 'cross_dataset_generalization_matrix.csv')
    df_matrix.to_csv(csv_out, index=False)
    
    print("\n=== CROSS-DATASET GENERALIZATION MATRIX ===")
    print(df_matrix.to_string(index=False))

if __name__ == '__main__':
    main()
