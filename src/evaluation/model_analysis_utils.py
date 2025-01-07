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
                                min_mcc_threshold: float = 0.2,
                                high_accuracy_threshold: float = 0.8,
                                suspicious_accuracy_threshold: float = 0.95,
                                max_allowed_precision_recall_gap: float = 0.5
                                ) -> List[Dict]:
        """
        Detect models with suspicious performance patterns, adding extra checks
        to catch questionable cases more thoroughly.

        Args:
            threshold_std (float): Standard deviation threshold for anomaly detection
            min_metric_threshold (float): Minimum acceptable metric value for accuracy/F1
            min_mcc_threshold (float): Minimum acceptable Matthews Correlation Coefficient
            high_accuracy_threshold (float): Accuracy threshold for checking mismatch with MCC
            suspicious_accuracy_threshold (float): Accuracy threshold for potential overfitting
            max_allowed_precision_recall_gap (float): Maximum acceptable gap between precision and recall

        Returns:
            List[Dict]: List of anomalous models and their issues
        """
        anomalies = []
        results_df = pd.DataFrame(self.results['results'])

        # Gather distribution info to detect near-constant predictions
        # (e.g., if accuracy ~ largest class proportion but MCC/F1 are very low)
        distribution = self.results['metadata']['dataset_info']['label_distribution']
        total_samples = self.results['metadata']['dataset_info']['num_samples']
        largest_class_count = max(distribution.values())
        largest_class_proportion = largest_class_count / total_samples

        for model in self.results['results']:
            issues = []
            metrics = {k: v for k, v in model.items() if k in self.metrics_of_interest}
            
            mcc = model.get('eval_matthews_correlation', 0)
            accuracy = model.get('eval_accuracy', 0)
            f1_score = model.get('eval_f1', 0)
            precision = model.get('eval_precision', 0)
            recall = model.get('eval_recall', 0)
            
            # 1. Very low MCC check
            if abs(mcc) < min_mcc_threshold:
                issues.append(
                    f"Very low MCC ({mcc:.3f}) with accuracy {accuracy:.3f} - could be near-constant predictions"
                )
            
            # 2. High accuracy but low MCC check
            if accuracy > high_accuracy_threshold and mcc < 0.4:
                issues.append(
                    f"Accuracy {accuracy:.3f} vs. MCC {mcc:.3f} mismatch - model may be exploiting imbalance"
                )
            
            # 3. Very low accuracy check
            if accuracy < min_metric_threshold:
                issues.append(f"Very low accuracy: {accuracy:.3f}")
            
            # 4. Very low F1 check
            if f1_score < min_metric_threshold:
                issues.append(f"Very low F1 score: {f1_score:.3f}")
            
            # 5. Suspicious outlier detection by standard deviation
            metric_values = np.array(list(metrics.values()))
            if len(metric_values) >= 2:  # Only valid if we have at least two metrics
                metric_std = np.std(metric_values)
                metric_mean = np.mean(metric_values)
                # Avoid division by zero if std is extremely small
                if metric_std > 0:
                    for metric, value in metrics.items():
                        z_score = abs(value - metric_mean) / metric_std
                        if z_score > threshold_std:
                            issues.append(
                                f"Suspicious {metric}={value:.3f} (z-score={z_score:.2f} from mean={metric_mean:.3f})"
                            )
            
            # 6. Check for suspiciously high accuracy (potential overfitting)
            if accuracy > suspicious_accuracy_threshold:
                issues.append(
                    f"Suspiciously high accuracy ({accuracy:.3f}) - check for data leakage or overfitting"
                )
            
            # 7. Check for near-constant prediction:
            #    If accuracy is close to largest_class_proportion, but MCC and F1 are very low
            if (abs(accuracy - largest_class_proportion) < 0.05
                and mcc < 0.1
                and f1_score < 0.2):
                issues.append(
                    "Accuracy close to majority class proportion with low MCC/F1 - possible constant predictions"
                )
            
            # 8. Check for large gap between precision and recall
            if abs(precision - recall) > max_allowed_precision_recall_gap:
                issues.append(
                    f"Large gap between precision ({precision:.3f}) and recall ({recall:.3f})"
                )
            
            if issues:
                anomalies.append({'model_name': model['model_name'], 'issues': issues})
        
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
                                decimal_places: int = 3) -> Tuple[str, Path]:
        """
        Generate a research-friendly table of model comparisons.
        
        Args:
            metrics (List[str]): Metrics to include in table
            format_type (str): Output format ('latex', 'github', 'grid')
            save_to_file (bool): Whether to save the table to a file
            decimal_places (int): Number of decimal places for numeric values
            
        Returns:
            Tuple[str, Path]: Formatted table string and output directory path
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
                if isinstance(value, (int, float)):
                    row.append(f"{float(value):.{decimal_places}f}")
                else:
                    row.append(str(value))
            table_data.append(row)
        
        # Sort by combined metric if available, otherwise by accuracy
        if 'eval_combined_metric' in metrics:
            sort_metric_idx = metrics.index('eval_combined_metric') + 1
        else:
            sort_metric_idx = 1  # Fallback to the first metric in the list after model name
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
                f.write("Dataset Information:\n")
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
        
        return table_str, output_dir

    def save_anomalies_report(self, anomalies: List[Dict], output_dir: Path) -> None:
        """
        Save anomalies report to a markdown file.
        
        Args:
            anomalies (List[Dict]): List of detected anomalies
            output_dir (Path): Directory to save the report
        """
        if not anomalies:
            return
        
        report_path = output_dir / "comparison_anomalies.md"
        
        with open(report_path, 'w') as f:
            f.write("# Model Performance Anomalies Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for anomaly in anomalies:
                f.write(f"## {anomaly['model_name']}\n\n")
                for issue in anomaly['issues']:
                    f.write(f"- {issue}\n")
                f.write("\n")

def analyze_comparison_file(file_name: str, formats: List[str] = None):
    """
    Analyze a model comparison file and print findings.
    
    Args:
        file_name (str): Name of the comparison file
        formats (List[str]): List of output formats to generate
    """
    if formats is None:
        formats = ['grid']
        
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
        output_dir = None
        for fmt in formats:
            print(f"\n=== Model Performance Comparison ({fmt.upper()}) ===")
            table, output_dir = analyzer.generate_research_table(format_type=fmt, save_to_file=True)
            print(table)
        
        # Save anomalies report if anomalies were detected and we have an output directory
        if anomalies and output_dir:
            analyzer.save_anomalies_report(anomalies, output_dir)
            logger.info(f"Saved anomalies report to: {output_dir / 'comparison_anomalies.md'}")
        
    except Exception as e:
        logger.error(f"Error analyzing comparison file: {str(e)}")
        raise
