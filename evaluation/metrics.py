"""Strict, JSON-safe classification metrics for experimental image models.

Do not interpret these metrics as clinical validation.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score,
)


def compute_all_metrics(y_true, y_pred, y_probs, num_classes=5):
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    probs = np.asarray(y_probs, dtype=float)

    if true.ndim != 1 or predicted.ndim != 1 or len(true) == 0 or len(true) != len(predicted):
        raise ValueError("Labels and predictions must be non-empty, equally sized 1D arrays")
    if probs.shape != (len(true), num_classes):
        raise ValueError("Probability matrix must have (samples, num_classes) shape")
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("Each probability row must sum to one")
    if np.any((true < 0) | (true >= num_classes)) or np.any((predicted < 0) | (predicted >= num_classes)):
        raise ValueError("Class indices are outside the configured class range")

    result = {
        "sample_count": int(len(true)),
        "overall_accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=list(range(num_classes)), average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true, predicted, labels=list(range(num_classes)), average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, labels=list(range(num_classes)), average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(true, predicted)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "macro_roc_auc": None,
        "per_class": {},
    }

    # Multi-class ROC AUC requires every configured class in ground truth.
    # A subset of observed labels must not be scored as a full cohort.
    if len(np.unique(true)) == num_classes:
        result["macro_roc_auc"] = float(roc_auc_score(true, probs, labels=list(range(num_classes)), multi_class="ovr", average="macro"))

    cm = confusion_matrix(true, predicted, labels=list(range(num_classes)))
    for cls in range(num_classes):
        tp = int(cm[cls, cls])
        fn = int(cm[cls, :].sum() - tp)
        fp = int(cm[:, cls].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        def ratio(n, d):
            return float(n / d) if d else None
        result["per_class"][str(cls)] = {
            "support": int(tp + fn),
            "tpr": ratio(tp, tp + fn),
            "fpr": ratio(fp, fp + tn),
            "fnr": ratio(fn, fn + tp),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
        }
    return result
