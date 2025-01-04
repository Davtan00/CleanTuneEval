from typing import Dict
from .processor import DataProcessor
from .storage import DataStorage
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class DataPipeline:
    def __init__(self):
        self.hardware_config = HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
        self.storage = DataStorage()
    
    def process_synthetic_data(self, data: Dict) -> Dict:
        """
        Main entry point for processing synthetic data
        """
        try:
            # Process the data
            processed_data = self.processor.process_batch(data)
            
            # Store the processed data
            storage_paths = self.storage.save_processed_data(
                processed_data,
                domain=data['domain']
            )
            
            return {
                'status': 'success',
                'data': processed_data,
                'storage': storage_paths
            }
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 