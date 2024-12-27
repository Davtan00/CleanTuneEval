from typing import Dict
from .processor import DataProcessor
from ..config.environment import HardwareConfig

class DataPipeline:
    def __init__(self):
        self.hardware_config = HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
    
    def process_synthetic_data(self, data: Dict) -> Dict:
        """
        Main entry point for processing synthetic data
        """
        try:
            processed_data = self.processor.process_batch(data)
            return {
                'status': 'success',
                'data': processed_data
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            } 