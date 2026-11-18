"""
Classification Metrics Evaluation Suite.
Calculates Sensitivity, Specificity, Precision, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
Finds optimal operating threshold from validation set using Youden's Index.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple

def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Finds optimal operating threshold maximizing Youden's Index (Sens + Spec - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # J = tpr - fpr = Sensitivity + Specificity - 1
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    # Constrain within realistic range
    return max(min(best_thresh, 0.95), 0.05)

def evaluate_classification(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates comprehensive binary classification metrics.
    y_true: binary labels {0, 1}
    y_probs: continuous probabilities in [0, 1]
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)
    
    # Check for single-class cases
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_probs))
        pr_auc = float(average_precision_score(y_true, y_probs))
    else:
        roc_auc = 0.5
        pr_auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        'threshold': float(threshold),
        'accuracy': acc,
        'sensitivity': rec,
        'specificity': spec,
        'precision': prec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'y_true': y_true.tolist(),
        'y_probs': y_probs.tolist()
    }
