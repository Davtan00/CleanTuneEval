from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, List
from ..config.logging_config import setup_logging

logger = setup_logging()

class DataStorage:
    def __init__(self, base_path: str = "src/data/storage"):
        # Store data within src/data/storage
        self.base_path = Path(base_path)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['raw', 'metrics']
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
    
    def save_processed_data(self, data: Dict, domain: str) -> Dict:
        """
        Save raw processed data and metrics
        Now focused on storing raw data and metrics, while dataset handling is done by DatasetManager
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw processed data
        raw_path = self.base_path / "raw" / f"{domain}_{timestamp}.json"
        with open(raw_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved raw processed data to: {raw_path}")
        
        # Save metrics separately
        metrics_path = self.base_path / "metrics" / f"{domain}_{timestamp}_metrics.json"
        metrics_data = {
            "domain": domain,
            "timestamp": timestamp,
            "summary": data['summary']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_path}")
        
        return {
            "raw_path": str(raw_path),
            "metrics_path": str(metrics_path)
        }
    
    def load_raw_data(self, domain: str, timestamp: Optional[str] = None) -> Dict:
        """Load raw processed data"""
        if timestamp:
            raw_path = self.base_path / "raw" / f"{domain}_{timestamp}.json"
        else:
            # Get latest file for domain
            raw_files = list((self.base_path / "raw").glob(f"{domain}_*.json"))
            if not raw_files:
                raise FileNotFoundError(f"No raw data found for domain: {domain}")
            raw_path = sorted(raw_files)[-1]
        
        with open(raw_path) as f:
            return json.load(f)
    
    def get_metrics_history(self, domain: str) -> List[Dict]:
        """Get historical metrics for a domain"""
        metrics_files = list((self.base_path / "metrics").glob(f"{domain}_*_metrics.json"))
        metrics_history = []
        
        for metrics_file in sorted(metrics_files):
            with open(metrics_file) as f:
                metrics_history.append(json.load(f))
        
        return metrics_history 