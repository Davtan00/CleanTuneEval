from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, f1_score
import logging
from datasets import Dataset

logger = logging.getLogger(__name__)

def compute_classification_metrics(pred):
    """
    Compute comprehensive classification metrics
    
    Args:
        pred: EvalPrediction object containing predictions and labels
            pred.predictions: Model predictions
            pred.label_ids: True labels
    
    Returns:
        Dictionary containing various metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'macro_precision': precision_score(labels, preds, average='macro', zero_division=1),
        'macro_recall': recall_score(labels, preds, average='macro'),
        'macro_f1': f1_score(labels, preds, average='macro'),
        'confusion_matrix': confusion_matrix(labels, preds).tolist()
    }
    
    # Per-class metrics
    for i, class_name in enumerate(['negative', 'neutral', 'positive']):
        metrics[f'{class_name}_precision'] = precision_score(labels, preds, average=None, zero_division=1)[i]
        metrics[f'{class_name}_recall'] = recall_score(labels, preds, average=None)[i]
        metrics[f'{class_name}_f1'] = f1_score(labels, preds, average=None)[i]
    
    return metrics 

def compute_baseline_metrics(dataset: Dataset) -> Dict[str, float]:
    """Compute metrics for majority class baseline"""
    majority_label = max(set(dataset['labels']), key=dataset['labels'].count)
    predictions = [majority_label] * len(dataset)
    
    return {
        'baseline_accuracy': accuracy_score(dataset['labels'], predictions),
        'baseline_f1': f1_score(dataset['labels'], predictions, average='macro')
    } 