from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
            eval_pred.predictions: Model predictions
            eval_pred.label_ids: True labels
    
    Returns:
        Dictionary containing various metrics
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Get predicted labels
    preds = np.argmax(predictions, axis=-1)
    
    # Calculate basic metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2]  # negative, neutral, positive
    )
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(labels, preds, labels=[0, 1, 2])
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        # Per-class metrics
        'negative_precision': per_class_precision[0],
        'negative_recall': per_class_recall[0],
        'negative_f1': per_class_f1[0],
        'neutral_precision': per_class_precision[1],
        'neutral_recall': per_class_recall[1],
        'neutral_f1': per_class_f1[1],
        'positive_precision': per_class_precision[2],
        'positive_recall': per_class_recall[2],
        'positive_f1': per_class_f1[2],
        # Store confusion matrix as list for JSON serialization
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics 