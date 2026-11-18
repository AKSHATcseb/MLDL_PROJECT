"""
Feature Fusion Research Experiment (Section 11.2).
Investigates multi-modal feature combinations:
- Model A: Image-only deep learning (EfficientNet-B0)
- Model B: Segmentation-derived features only (vCDR, Area CDR, Disc Area, Cup Area) via Logistic Regression
- Model C: Multi-modal fusion (Deep Image Logit + Segmentation Geometric Features)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.integration.inference import GlaucomaInferenceEngine
from src.data.dataset_loader import get_segmentation_loaders

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'comparisons')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_fusion_experiment():
    print("=== Running Feature Fusion Comparative Experiment ===")
    engine = GlaucomaInferenceEngine()
    
    # Evaluate on REFUGE validation set where both segmentation and classification labels exist
    val_csv = os.path.join(PROJECT_ROOT, 'data', 'processed', 'segmentation', 'refuge_val_80.csv')
    df_val = pd.read_csv(val_csv)
    
    y_true = []
    prob_img = []
    vcdrs = []
    area_cdrs = []
    
    for idx, row in df_val.iterrows():
        img_p = row['image_path']
        label = int(row['label'])
        import cv2
        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        
        res = engine.predict(img)
        y_true.append(label)
        prob_img.append(res['glaucoma_probability'])
        vcdrs.append(res['clinical_features']['vcdr'])
        area_cdrs.append(res['clinical_features']['area_cdr'])
        
    y_true = np.array(y_true)
    prob_img = np.array(prob_img)
    X_seg = np.column_stack([vcdrs, area_cdrs])
    
    # Model A: Image-only
    auc_a = roc_auc_score(y_true, prob_img)
    acc_a = accuracy_score(y_true, (prob_img >= 0.5).astype(int))
    f1_a = f1_score(y_true, (prob_img >= 0.5).astype(int), zero_division=0)
    
    # Model B: Segmentation-only features (Logistic Regression)
    clf_b = LogisticRegression()
    clf_b.fit(X_seg, y_true)
    prob_b = clf_b.predict_proba(X_seg)[:, 1]
    auc_b = roc_auc_score(y_true, prob_b)
    acc_b = accuracy_score(y_true, (prob_b >= 0.5).astype(int))
    f1_b = f1_score(y_true, (prob_b >= 0.5).astype(int), zero_division=0)
    
    # Model C: Multi-modal fusion
    X_fused = np.column_stack([prob_img, vcdrs, area_cdrs])
    clf_c = LogisticRegression()
    clf_c.fit(X_fused, y_true)
    prob_c = clf_c.predict_proba(X_fused)[:, 1]
    auc_c = roc_auc_score(y_true, prob_c)
    acc_c = accuracy_score(y_true, (prob_c >= 0.5).astype(int))
    f1_c = f1_score(y_true, (prob_c >= 0.5).astype(int), zero_division=0)
    
    summary = [
        {'Model Pipeline': 'Model A (Image-Only Deep Learning)', 'ROC-AUC': round(auc_a, 4), 'Accuracy': round(acc_a, 4), 'F1-Score': round(f1_a, 4)},
        {'Model Pipeline': 'Model B (Segmentation Features: vCDR + Area Ratio)', 'ROC-AUC': round(auc_b, 4), 'Accuracy': round(acc_b, 4), 'F1-Score': round(f1_b, 4)},
        {'Model Pipeline': 'Model C (Multi-Modal Feature Fusion: Image + vCDR)', 'ROC-AUC': round(auc_c, 4), 'Accuracy': round(acc_c, 4), 'F1-Score': round(f1_c, 4)}
    ]
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(os.path.join(RESULTS_DIR, 'feature_fusion_comparison.csv'), index=False)
    print("\n=== FEATURE FUSION EXPERIMENTAL RESULTS ===")
    print(df_sum.to_string(index=False))

if __name__ == '__main__':
    run_fusion_experiment()
