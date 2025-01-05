from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .metrics import MetricsCalculator
from ..models.adaptation import ModelAdapter
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class ModelEvaluator:
    def __init__(
        self,
        hardware_config: HardwareConfig,
        labels: List[str] = ["negative", "neutral", "positive"]
    ):
        logger.info("Initializing ModelEvaluator")
        self.hardware_config = hardware_config
        self.metrics_calculator = MetricsCalculator(labels)
        logger.info(f"Metrics calculator initialized with labels: {labels}")
        
    def evaluate_model(
        self,
        model: ModelAdapter,
        test_data: Dict,
        domain: str,
        synthetic_ratio: Optional[float] = None
    ) -> Dict:
        """
        Evaluate model performance on test data
        """
        predictions, probabilities = model.predict(test_data['texts'])
        true_labels = test_data['labels']
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            true_labels,
            predictions,
            probabilities
        )
        
        # Prepare evaluation results
        results = {
            'domain': domain,
            'synthetic_ratio': synthetic_ratio,
            'metrics': {
                'accuracy': metrics.accuracy,
                'macro_f1': metrics.macro_f1,
                'per_class_metrics': {
                    label: {
                        'precision': metrics.precision[label],
                        'recall': metrics.recall[label],
                        'auroc': metrics.auroc[label]
                    }
                    for label in self.metrics_calculator.labels
                },
                'calibration': {
                    'brier_score': metrics.brier_score,
                    'calibration_error': metrics.calibration_error
                }
            },
            'confusion_matrix': metrics.confusion_matrix.tolist()
        }
        
        return results

class CrossDomainEvaluator:
    def __init__(
        self, 
        hardware_config: HardwareConfig,
        labels: List[str] = ["negative", "neutral", "positive"]  # Default labels
    ):
        logger.info("Initializing CrossDomainEvaluator")
        self.hardware_config = hardware_config
        self.evaluator = ModelEvaluator(hardware_config, labels)
        logger.info(f"Initialized with labels: {labels}")
    
    def run_cross_domain_evaluation(
        self,
        model: ModelAdapter,
        domain_datasets: Dict[str, Dict],
        synthetic_ratios: Optional[List[float]] = None
    ) -> Dict:
        """
        Run evaluation across multiple domains and synthetic ratios
        """
        results = []
        
        for domain, dataset in domain_datasets.items():
            # Evaluate with full real data
            domain_result = self.evaluator.evaluate_model(
                model,
                dataset,
                domain
            )
            results.append(domain_result)
            
            # If synthetic ratios provided, evaluate with mixed data
            if synthetic_ratios:
                for ratio in synthetic_ratios:
                    mixed_dataset = self._create_mixed_dataset(dataset, ratio)
                    mixed_result = self.evaluator.evaluate_model(
                        model,
                        mixed_dataset,
                        domain,
                        synthetic_ratio=ratio
                    )
                    results.append(mixed_result)
        
        return {
            'cross_domain_results': results,
            'summary': self._create_summary(results)
        }
    
    def _create_mixed_dataset(self, dataset: Dict, synthetic_ratio: float) -> Dict:
        """Create a mixed dataset with both real and synthetic data"""
        # TODO: Implement this, evaluate final if we want binary classification or 3 way
        # Depending also on which domain we care about, for example technology, healthcare, and so on.

        pass
    
    def _create_summary(self, results: List[Dict]) -> Dict:
        """Create a summary of cross-domain evaluation results"""
        df = pd.DataFrame(results)
        
        return {
            'average_metrics': {
                'accuracy': df['metrics'].apply(lambda x: x['accuracy']).mean(),
                'macro_f1': df['metrics'].apply(lambda x: x['macro_f1']).mean(),
            },
            'per_domain_performance': df.groupby('domain')['metrics'].apply(
                lambda x: x.apply(lambda y: y['macro_f1']).mean()
            ).to_dict()
        } 