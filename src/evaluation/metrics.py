from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, brier_score_loss,
    confusion_matrix
)
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    accuracy: float
    macro_f1: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    auroc: Dict[str, float]
    brier_score: float
    confusion_matrix: np.ndarray
    calibration_error: float

class MetricsCalculator:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
    
    def compute_all_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        y_prob: np.ndarray
    ) -> EvaluationMetrics:
        """Compute all evaluation metrics"""
        # Convert string labels to indices
        y_true_idx = [self.label2idx[y] for y in y_true]
        y_pred_idx = [self.label2idx[y] for y in y_pred]
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_idx, 
            y_pred_idx, 
            average='macro'
        )
        
        # Per-class metrics
        precision_dict = {}
        recall_dict = {}
        auroc_dict = {}
        
        # Compute per-class metrics
        for idx, label in enumerate(self.labels):
            true_binary = (np.array(y_true_idx) == idx).astype(int)
            pred_probs = y_prob[:, idx]
            
            precision_dict[label] = precision[idx]
            recall_dict[label] = recall[idx]
            auroc_dict[label] = roc_auc_score(true_binary, pred_probs)
        
        # Compute calibration error
        calibration_error = self._compute_calibration_error(y_true_idx, y_prob)
        
        return EvaluationMetrics(
            accuracy=accuracy_score(y_true_idx, y_pred_idx),
            macro_f1=f1,
            precision=precision_dict,
            recall=recall_dict,
            auroc=auroc_dict,
            brier_score=brier_score_loss(y_true_idx, y_prob),
            confusion_matrix=confusion_matrix(y_true_idx, y_pred_idx),
            calibration_error=calibration_error
        )
    
    def _compute_calibration_error(
        self,
        y_true: List[int],
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in bin
            in_bin = np.logical_and(
                y_prob.max(axis=1) > bin_lower,
                y_prob.max(axis=1) <= bin_upper
            )
            
            if np.sum(in_bin) > 0:
                pred_in_bin = y_prob[in_bin].argmax(axis=1)
                true_in_bin = np.array(y_true)[in_bin]
                
                # Compute average confidence and accuracy in bin
                accuracy_in_bin = np.mean(pred_in_bin == true_in_bin)
                confidence_in_bin = np.mean(y_prob[in_bin].max(axis=1))
                
                calibration_error += np.abs(accuracy_in_bin - confidence_in_bin) * (np.sum(in_bin) / len(y_true))
                
        return calibration_error 