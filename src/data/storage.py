from pathlib import Path
import json
import jsonlines
from datasets import Dataset, DatasetDict
from typing import Dict, Optional
from ..config.logging_config import setup_logging

logger = setup_logging()

class DataStorage:
    def __init__(self, base_path: str = "processed_data"):
        self.base_path = Path(base_path)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['huggingface', 'jsonl', 'metrics']
        for dir_name in directories:
            dir_path = self.base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
    
    def save_processed_data(
        self, 
        data: Dict, 
        domain: str,
        format: str = "huggingface"
    ) -> Dict:
        """
        Save processed data in specified format
        format options: "huggingface", "jsonl"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "huggingface":
            return self._save_as_huggingface(data, domain, timestamp)
        elif format == "jsonl":
            return self._save_as_jsonl(data, domain, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_as_huggingface(self, data: Dict, domain: str, timestamp: str) -> Dict:
        """Save data in Hugging Face Dataset format"""
        logger.info("Converting to Hugging Face Dataset format...")
        
        # Prepare the data
        processed_reviews = data['generated_data']
        
        # Create Dataset object
        dataset = Dataset.from_dict({
            'text': [r['clean_text'] for r in processed_reviews],
            'sentiment': [r['sentiment'] for r in processed_reviews],
            'id': [r['id'] for r in processed_reviews]
        })
        
        # Split into train/val/test (80/10/10)
        splits = dataset.train_test_split(
            test_size=0.2,
            shuffle=True,
            seed=42
        )
        test_val = splits['test'].train_test_split(
            test_size=0.5,
            shuffle=True,
            seed=42
        )
        
        dataset_dict = DatasetDict({
            'train': splits['train'],
            'validation': test_val['train'],
            'test': test_val['test']
        })
        
        # Save the dataset
        save_path = self.base_path / "huggingface" / f"{domain}_{timestamp}"
        dataset_dict.save_to_disk(save_path)
        logger.info(f"Saved Hugging Face dataset to: {save_path}")
        
        # Save metrics separately
        metrics_path = self.base_path / "metrics" / f"{domain}_{timestamp}_metrics.json"
        metrics_data = {
            "domain": domain,
            "timestamp": timestamp,
            "summary": data['summary'],
            "splits": {
                "train_size": len(splits['train']),
                "validation_size": len(test_val['train']),
                "test_size": len(test_val['test'])
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return {
            "dataset_path": str(save_path),
            "metrics_path": str(metrics_path)
        }
    
    def _save_as_jsonl(self, data: Dict, domain: str, timestamp: str) -> Dict:
        """Save data in JSONL format"""
        jsonl_path = self.base_path / "jsonl" / f"{domain}_{timestamp}.jsonl"
        
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for review in data['generated_data']:
                writer.write({
                    'text': review['clean_text'],
                    'sentiment': review['sentiment'],
                    'id': review['id']
                })
        
        logger.info(f"Saved JSONL file to: {jsonl_path}")
        
        return {
            "jsonl_path": str(jsonl_path)
        }
    
    def load_dataset(
        self, 
        domain: str, 
        timestamp: Optional[str] = None,
        format: str = "huggingface"
    ) -> Dataset:
        """Load processed dataset"""
        if timestamp is None:
            # Get latest file
            if format == "huggingface":
                datasets = list((self.base_path / "huggingface").glob(f"{domain}_*"))
                if not datasets:
                    raise FileNotFoundError(f"No dataset found for domain: {domain}")
                dataset_path = sorted(datasets)[-1]
            else:
                jsonl_files = list((self.base_path / "jsonl").glob(f"{domain}_*.jsonl"))
                if not jsonl_files:
                    raise FileNotFoundError(f"No JSONL found for domain: {domain}")
                dataset_path = sorted(jsonl_files)[-1]
        else:
            if format == "huggingface":
                dataset_path = self.base_path / "huggingface" / f"{domain}_{timestamp}"
            else:
                dataset_path = self.base_path / "jsonl" / f"{domain}_{timestamp}.jsonl"
        
        if format == "huggingface":
            return DatasetDict.load_from_disk(dataset_path)
        else:
            return Dataset.from_json(str(dataset_path)) 