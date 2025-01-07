import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from tabulate import tabulate
import warnings
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    """Utility class for analyzing model comparison results."""
    
    def __init__(self, comparison_file: str):
        """
        Initialize analyzer with path to comparison file.
        
        Args:
            comparison_file (str): Path to model comparison JSON file
        """
        self.storage_dir = Path("src/evaluation/storage")
        self.results_path = self.storage_dir / comparison_file
        self.results = self._load_comparison_results()
        self.metrics_of_interest = [
            'eval_accuracy', 'eval_precision', 'eval_recall', 
            'eval_f1', 'eval_matthews_correlation'
        ]

    def _load_comparison_results(self) -> Dict:
        """Load and validate comparison results."""
        try:
            with open(self.results_path) as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Comparison file not found: {self.results_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.results_path}")

    def detect_anomalous_models(self, 
                              threshold_std: float = 2.0,
                              min_metric_threshold: float = 0.1,
                              min_mcc_threshold: float = 0.2) -> List[Dict]:
        """
        Detect models with suspicious performance patterns.
        
        Args:
            threshold_std (float): Standard deviation threshold for anomaly detection
            min_metric_threshold (float): Minimum acceptable metric value
            min_mcc_threshold (float): Minimum acceptable Matthews Correlation Coefficient
            
        Returns:
            List[Dict]: List of anomalous models and their issues
        """
        anomalies = []
        results_df = pd.DataFrame(self.results['results'])
        
        for model in self.results['results']:
            issues = []
            metrics = {k: v for k, v in model.items() if k in self.metrics_of_interest}
            
            # Check for suspicious MCC patterns
            mcc = model.get('eval_matthews_correlation', 0)
            accuracy = model.get('eval_accuracy', 0)
            
            # Detect potential constant predictions
            if abs(mcc) < min_mcc_threshold:
                issues.append(
                    f"Very low Matthews Correlation ({mcc:.3f}) despite accuracy of {accuracy:.3f} "
                    "- model might be making constant predictions"
                )
            
            # Detect suspiciously high accuracy with low MCC
            if accuracy > 0.8 and mcc < 0.4:
                issues.append(
                    f"High accuracy ({accuracy:.3f}) but low Matthews Correlation ({mcc:.3f}) "
                    "- might be exploiting dataset imbalance"
                )
            
            # Original checks
            if model['eval_accuracy'] < min_metric_threshold:
                issues.append(f"Very low accuracy: {model['eval_accuracy']:.3f}")
            
            # Check for suspicious metric patterns
            metric_values = np.array(list(metrics.values()))
            metric_std = np.std(metric_values)
            metric_mean = np.mean(metric_values)
            
            # Detect outliers
            for metric, value in metrics.items():
                z_score = abs(value - metric_mean) / metric_std
                if z_score > threshold_std:
                    issues.append(f"Suspicious {metric}: {value:.3f} (z-score: {z_score:.2f})")
            
            # Check for perfect or near-perfect scores
            if model['eval_accuracy'] > 0.95:  # Lowered from 0.99 to be more sensitive
                issues.append(f"Suspiciously high accuracy ({model['eval_accuracy']:.3f}) - check for dataset bias")
            
            if issues:
                anomalies.append({
                    'model_name': model['model_name'],
                    'issues': issues
                })
        
        return anomalies

    def get_dataset_info(self) -> str:
        """Format dataset information into a readable string."""
        dataset_info = self.results['metadata']['dataset_info']
        return (
            f"Dataset Summary:\n"
            f"- Path: {dataset_info['path']}\n"
            f"- Samples: {dataset_info['num_samples']}\n"
            f"- Distribution:\n"
            f"  - Positive: {dataset_info['label_distribution']['positive']}\n"
            f"  - Neutral: {dataset_info['label_distribution']['neutral']}\n"
            f"  - Negative: {dataset_info['label_distribution']['negative']}\n"
        )

    def generate_research_table(self, 
                              metrics: List[str] = None,
                              format_type: str = 'grid',
                              save_to_file: bool = True,
                              decimal_places: int = 3) -> str:
        """
        Generate a research-friendly table of model comparisons.
        
        Args:
            metrics (List[str]): Metrics to include in table
            format_type (str): Output format ('latex', 'github', 'grid')
            save_to_file (bool): Whether to save the table to a file
            decimal_places (int): Number of decimal places for numeric values
            
        Returns:
            str: Formatted table string
        """
        if metrics is None:
            metrics = [
                'eval_accuracy', 
                'eval_f1', 
                'eval_matthews_correlation',
                'eval_combined_metric',
                'eval_samples_per_second'
            ]
        
        # Extract relevant data
        table_data = []
        headers = ['Model'] + [m.replace('eval_', '').replace('_', ' ').title() for m in metrics]
        
        for result in self.results['results']:
            row = [result['model_name']]
            for metric in metrics:
                value = result.get(metric, 0)
                # Ensure consistent decimal places for all numeric values
                row.append(f"{float(value):.{decimal_places}f}" if isinstance(value, (int, float)) else value)
            table_data.append(row)
        
        # Sort by combined metric if available, otherwise by accuracy
        sort_metric_idx = (metrics.index('eval_combined_metric') + 1 
                          if 'eval_combined_metric' in metrics else 1)
        table_data.sort(key=lambda x: float(x[sort_metric_idx]), reverse=True)
        
        table_str = tabulate(table_data, headers=headers, tablefmt=format_type)
        
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("src/evaluation/results") / f"model_comparison_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save dataset info
            dataset_info = self.results['metadata']['dataset_info']
            dataset_info_path = output_dir / "dataset_info.txt"
            with open(dataset_info_path, 'w') as f:
                f.write(f"Dataset Information:\n")
                f.write(f"Path: {dataset_info['path']}\n")
                f.write(f"Hash: {dataset_info['hash']}\n")
                f.write(f"Number of samples: {dataset_info['num_samples']}\n")
                f.write("\nLabel Distribution:\n")
                for label, count in dataset_info['label_distribution'].items():
                    f.write(f"{label}: {count}\n")
                
                # Add original comparison file reference
                f.write(f"\nOriginal comparison file: {self.results_path.name}\n")
            
            # Save the table
            output_path = output_dir / f"comparison_table.{format_type}"
            with open(output_path, 'w') as f:
                f.write(table_str)
                logger.info(f"Saved analysis results to: {output_dir}")
        
        return table_str

def analyze_comparison_file(file_name: str, formats: List[str] = None):
    """
    Analyze a model comparison file and print findings.
    
    Args:
        file_name (str): Name of the comparison file
        formats (List[str]): List of output formats to generate
    """
    if formats is None:
        formats = ['grid']  # Default to just grid format
        
    try:
        analyzer = ModelAnalyzer(file_name)
        
        # Print dataset information
        print("\n=== Dataset Information ===")
        print(analyzer.get_dataset_info())
        
        # Check for anomalous models
        print("\n=== Detecting Suspicious Model Performances ===")
        anomalies = analyzer.detect_anomalous_models()
        if anomalies:
            for anomaly in anomalies:
                print(f"\nModel: {anomaly['model_name']}")
                for issue in anomaly['issues']:
                    print(f"  - {issue}")
        else:
            print("No suspicious model performances detected.")
        
        # Generate and save tables in requested formats
        for fmt in formats:
            print(f"\n=== Model Performance Comparison ({fmt.upper()}) ===")
            table = analyzer.generate_research_table(format_type=fmt, save_to_file=True)
            print(table)
        
    except Exception as e:
        logger.error(f"Error analyzing comparison file: {str(e)}")
        raise

if __name__ == "__main__":

    analyze_comparison_file("model_comparison_20250107_124451.json") 