import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, accuracy_score, cohen_kappa_score, 
    matthews_corrcoef, brier_score_loss, confusion_matrix
)
import json

def compute_all_metrics(y_true, y_pred, y_probs, num_classes=5):
    metrics = {}
    
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    try:
        metrics['macro_roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except:
        metrics['macro_roc_auc'] = None
        
    metrics['per_class'] = {}
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        metrics['per_class'][str(c)] = {
            'tpr': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        }
        
    return metrics
