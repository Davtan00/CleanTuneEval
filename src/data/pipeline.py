from typing import Dict, Optional
from .processor import DataProcessor
from .storage import DataStorage
from .dataset_manager import DatasetManager
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class DataPipeline:
    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        self.hardware_config = hw_config or HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
        # Initialize storage and dataset manager with correct paths
        self.storage = DataStorage(base_path="src/data/storage")
        self.dataset_manager = DatasetManager(base_path="src/data/datasets")
        logger.info("Initialized DataPipeline with all components")
    
    def process_synthetic_data(self, data: Dict, custom_tag: Optional[str] = None, batch_size: int = 1000) -> Dict:
        """
        Main entry point for processing synthetic data
        """
        try:
            # Process the data
            logger.info(f"Processing synthetic data for domain: {data['domain']}")
            process_result = self.processor.process_batch(data, batch_size=batch_size)
            
            # Check for processing errors
            if process_result['status'] != 'success':
                logger.error(f"Processing failed: {process_result.get('message', 'Unknown error')}")
                return process_result
            
            # Get the processed data from the result
            processed_data = process_result['data']
            
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
                },
                'performance': process_result.get('performance', {})
            }
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }