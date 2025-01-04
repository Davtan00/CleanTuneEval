from typing import Dict, Optional
from .processor import DataProcessor
from .storage import DataStorage
from .dataset_manager import DatasetManager
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class DataPipeline:
    def __init__(self):
        self.hardware_config = HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
        # Initialize storage and dataset manager with correct paths
        self.storage = DataStorage(base_path="src/data/storage")
        self.dataset_manager = DatasetManager(base_path="src/data/datasets")
        logger.info("Initialized DataPipeline with all components")
    
    def process_synthetic_data(self, data: Dict, custom_tag: Optional[str] = None) -> Dict:
        """
        Main entry point for processing synthetic data
        """
        try:
            # Process the data
            logger.info(f"Processing synthetic data for domain: {data['domain']}")
            processed_data = self.processor.process_batch(data)
            
            # Store processed data and metrics
            storage_paths = self.storage.save_processed_data(
                processed_data,
                domain=data['domain'],
                custom_tag=custom_tag
            )
            
            # Create dataset splits for model training
            logger.info("Creating dataset splits for model training")
            dataset_splits = self.dataset_manager.create_dataset(
                data=processed_data,
                dataset_id=storage_paths['dataset_id']
            )
            
            return {
                'status': 'success',
                'data': processed_data,
                'storage': storage_paths,
                'dataset_info': {
                    'id': storage_paths['dataset_id'],
                    'path': str(self.dataset_manager.base_path / storage_paths['dataset_id']),
                    'splits': {
                        split: len(dataset) for split, dataset in dataset_splits.items()
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }