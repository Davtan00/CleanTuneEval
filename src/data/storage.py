from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List
from ..config.logging_config import setup_logging
from .utils import generate_dataset_id

logger = setup_logging()

class DataStorage:
    """
    Responsible for saving and loading data files:
      - /raw: Original input files
      - /processed: Filtered & enumerated data
      - /metrics: Summaries of the filtering process
    """
    def __init__(self, base_path: str = "src/data/storage"):
        self.base_path = Path(base_path)
        self._ensure_directories()

    def _ensure_directories(self):
        """
        Creates raw, processed, and metrics directories if missing.
        """
        directories = ['raw', 'processed', 'metrics']
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")

    def save_processed_data(self, data: Dict, domain: str, custom_tag: Optional[str] = None) -> Dict:
        """
        Save final processed data and its metrics in two JSON files:
          processed/<dataset_id>.json and metrics/<dataset_id>_metrics.json
        Returns a dict with references to both saved paths.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = generate_dataset_id(
            domain=domain,
            data_size=len(data['generated_data']),
            custom_tag=custom_tag,
            timestamp=timestamp
        )

        # Add source info
        data['source_info'] = {
            'original_file': Path(data.get('input_file', 'unknown')).name,
            'processing_timestamp': timestamp
        }

        # 1) Write processed data to /processed
        processed_path = self.base_path / "processed" / f"{dataset_id}.json"
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data: {processed_path}")

        # 2) Write metrics to /metrics
        metrics_data = {
            "dataset_id": dataset_id,
            "domain": domain,
            "timestamp": timestamp,
            "summary": data.get('summary', {})
        }
        metrics_path = self.base_path / "metrics" / f"{dataset_id}_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metrics: {metrics_path}")

        return {
            "dataset_id": dataset_id,
            "data_path": str(processed_path),
            "metrics_path": str(metrics_path)
        }

    def load_raw_data(self, domain: str, filename: str) -> Dict:
        """
        Load raw data from /raw. Typically called if we store original unfiltered data here.
        """
        raw_path = self.base_path / "raw" / filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")

        with open(raw_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_processed_data(self, dataset_id: str) -> Dict:
        """
        Load previously processed data by dataset_id from /processed.
        """
        processed_path = self.base_path / "processed" / f"{dataset_id}.json"
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_path}")

        with open(processed_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_metrics_history(self, domain: str) -> List[Dict]:
        """
        Retrieve all metrics files for a given domain from /metrics.
        Useful for analyzing how data evolved over multiple runs.
        """
        metrics_files = list((self.base_path / "metrics").glob(f"{domain}_*_metrics.json"))
        metrics_history = []

        for file_path in sorted(metrics_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                metrics_history.append(json.load(f))

        return metrics_history