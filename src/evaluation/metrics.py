"""Metrics calculation utilities for model evaluation."""
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)
from dataclasses import dataclass

# Add the function that LoRATrainer expects
def compute_classification_metrics(eval_pred):
    """
    Compute metrics for model evaluation during training.
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Get predicted labels
    pred_labels = np.argmax(predictions, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, pred_labels)
    macro_f1 = f1_score(labels, pred_labels, average='macro')
    
    # Per-class F1 scores
    label_names = ["negative", "neutral", "positive"]
    per_class_f1 = f1_score(labels, pred_labels, average=None)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, pred_labels)
    
    return {
        'eval_accuracy': accuracy,
        'eval_macro_f1': macro_f1,
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
    """Calculate various metrics for model evaluation."""
    
    def __init__(self, labels: List[str]):
        """Initialize with list of possible labels."""
        self.labels = labels
        self.label_map = {label: idx for idx, label in enumerate(labels)}
    
    def compute_all_metrics(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        probabilities: Optional[np.ndarray] = None
    ) -> MetricsResult:
        """Compute all evaluation metrics."""
        # Convert string labels to indices
        y_true = np.array([self.label_map[label] for label in true_labels])
        y_pred = np.array([self.label_map[label] for label in predicted_labels])
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        precision = {}
        recall = {}
        auroc = {}
        
        for label in self.labels:
            idx = self.label_map[label]
            y_true_binary = (y_true == idx).astype(int)
            y_pred_binary = (y_pred == idx).astype(int)
            
            precision[label] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall[label] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            
            if probabilities is not None:
                try:
                    auroc[label] = roc_auc_score(y_true_binary, probabilities[:, idx])
                except ValueError:
                    auroc[label] = 0.5  # Default for cases with single class
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(self.labels)))
        
        # Calibration metrics
        if probabilities is not None:
            brier = np.mean([
                brier_score_loss(y_true == i, probabilities[:, i])
                for i in range(len(self.labels))
            ])
            # Simple calibration error estimation
            pred_probs = probabilities[np.arange(len(y_true)), y_pred]
            calibration_error = np.mean(np.abs(pred_probs - (y_true == y_pred).astype(float)))
        else:
            brier = 0.0
            calibration_error = 0.0
        
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