from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List
from ..config.logging_config import setup_logging
from .utils import generate_dataset_id

logger = setup_logging()

class DataStorage:
    def __init__(self, base_path: str = "src/data/storage"):
        # Store data within src/data/storage
        self.base_path = Path(base_path)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['raw', 'processed', 'metrics']
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
    
    def save_processed_data(self, data: Dict, domain: str, custom_tag: Optional[str] = None) -> Dict:
        """
        Save processed data and metrics with consistent return structure
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = generate_dataset_id(
            domain=domain,
            data_size=len(data['generated_data']),
            custom_tag=custom_tag,
            timestamp=timestamp
        )
        
        # Add source file information to the data
        data['source_info'] = {
            'original_file': Path(data.get('input_file', 'unknown')).name,
            'processing_timestamp': timestamp
        }
        
        # Save processed data
        processed_path = self.base_path / "processed" / f"{dataset_id}.json"
        with open(processed_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved processed data to: {processed_path}")
        
        # Save metrics separately
        metrics_path = self.base_path / "metrics" / f"{dataset_id}_metrics.json"
        metrics_data = {
            "dataset_id": dataset_id,
            "domain": domain,
            "timestamp": timestamp,
            "summary": data['summary']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_path}")
        
        return {
            "dataset_id": dataset_id,
            "data_path": str(processed_path),
            "metrics_path": str(metrics_path)
        }
    
    def load_raw_data(self, domain: str, filename: str) -> Dict:
        """Load raw input data"""
        raw_path = self.base_path / "raw" / filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        
        with open(raw_path) as f:
            return json.load(f)
    
    def load_processed_data(self, dataset_id: str) -> Dict:
        """Load processed data by dataset ID"""
        processed_path = self.base_path / "processed" / f"{dataset_id}.json"
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        
        with open(processed_path) as f:
            return json.load(f)
    
    def get_metrics_history(self, domain: str) -> List[Dict]:
        """Get historical metrics for a domain"""
        metrics_files = list((self.base_path / "metrics").glob(f"{domain}_*_metrics.json"))
        metrics_history = []
        
        for metrics_file in sorted(metrics_files):
            with open(metrics_file) as f:
                metrics_history.append(json.load(f))
        
        return metrics_history