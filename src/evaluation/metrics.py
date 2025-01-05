"""
Metrics calculation utilities. Includes a combined metric = (Accuracy + Macro-F1)/2.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)
from dataclasses import dataclass
from typing import List, Dict, Optional

def compute_classification_metrics(eval_pred):
    """
    Used by HF Trainer (or WeightedLossTrainer). Returns accuracy, macro-F1, 
    per-class F1, confusion matrix, and 'eval_combined_metric'.
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(predictions, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    combined = 0.5 * acc + 0.5 * macro_f1
    per_class_f1 = f1_score(labels, preds, average=None)
    conf_matrix = confusion_matrix(labels, preds)

    return {
        'eval_accuracy': acc,
        'eval_macro_f1': macro_f1,
        'eval_combined_metric': combined,
        'eval_negative_f1': per_class_f1[0],
        'eval_neutral_f1': per_class_f1[1],
        'eval_positive_f1': per_class_f1[2],
        'eval_confusion_matrix': conf_matrix.tolist()
    }

@dataclass
class MetricsResult:
    accuracy: float
    macro_f1: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    auroc: Dict[str, float]
    confusion_matrix: np.ndarray
    brier_score: float
    calibration_error: float

class MetricsCalculator:
    """
    Advanced metrics for offline analysis (precision, recall, AUROC, etc.).
    """
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.label_map = {label: idx for idx, label in enumerate(labels)}

    def compute_all_metrics(self,
                            true_labels: List[str],
                            predicted_labels: List[str],
                            probabilities: Optional[np.ndarray] = None
                            ) -> MetricsResult:
        y_true = np.array([self.label_map[l] for l in true_labels])
        y_pred = np.array([self.label_map[l] for l in predicted_labels])
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        precision, recall, auroc = {}, {}, {}
        for label in self.labels:
            idx = self.label_map[label]
            y_true_bin = (y_true == idx).astype(int)
            y_pred_bin = (y_pred == idx).astype(int)
            precision[label] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall[label] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            if probabilities is not None:
                try:
                    auroc[label] = roc_auc_score(y_true_bin, probabilities[:, idx])
                except ValueError:
                    auroc[label] = 0.5
            else:
                auroc[label] = 0.0

        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(self.labels)))
        brier, calibration_error = 0.0, 0.0
        if probabilities is not None:
            for i in range(len(self.labels)):
                y_true_i = (y_true == i).astype(int)
                brier += brier_score_loss(y_true_i, probabilities[:, i])
            brier /= len(self.labels)
            pred_probs = probabilities[np.arange(len(y_true)), y_pred]
            correctness = (y_true == y_pred).astype(float)
            calibration_error = np.mean(np.abs(pred_probs - correctness))

        return MetricsResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            precision=precision,
            recall=recall,
            auroc=auroc,
            confusion_matrix=conf_matrix,
            brier_score=brier,
            calibration_error=calibration_error
        ) 