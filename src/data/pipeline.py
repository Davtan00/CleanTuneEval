from typing import Dict, Optional, List
from .processor import DataProcessor
from .storage import DataStorage
from .dataset_manager import DatasetManager
from .validators import ValidationMetrics
from .utils import generate_dataset_id
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging
import time
from datetime import datetime

logger = setup_logging()

class DataPipeline:
    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        self.hardware_config = hw_config or HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
        # Initialize storage and dataset manager with correct paths
        self.storage = DataStorage(base_path="src/data/storage")
        self.dataset_manager = DatasetManager(base_path="src/data/datasets")
        logger.info("Initialized DataPipeline with all components")
    
    def process_synthetic_data(
        self,
        data: Dict,
        custom_tag: Optional[str] = None,
        batch_size: int = 1000
    ) -> Dict:
        """Process synthetic data through the pipeline with proper metrics tracking"""
        try:
            domain = data['domain']
            start_time = time.time()
            logger.info(f"Processing synthetic data for domain: {domain}")
            
            metrics = ValidationMetrics()
            processed_data = []
            
            initial_count = len(data['generated_data'])
            metrics.total_processed = initial_count
            
            # Process reviews in batches
            for i in range(0, initial_count, batch_size):
                batch = data['generated_data'][i:i + batch_size]
                batch_result = self.processor.process_batch(batch, domain)
                
                # Extract batch metrics and update pipeline metrics
                if isinstance(batch_result, dict) and 'summary' in batch_result:
                    filtering_summary = batch_result['summary']['filtering_summary']
                    metrics.length_filtered += filtering_summary['length_filtered']
                    metrics.duplicates_removed += filtering_summary['duplicates_removed']
                    metrics.exact_duplicates += filtering_summary['exact_duplicates']
                    metrics.near_duplicates += filtering_summary['near_duplicates']
                    
                    # Get processed reviews from correct location in batch_result
                    if 'generated_data' in batch_result:
                        processed_data.extend(batch_result['generated_data'])
                
                logger.debug(f"Batch {i//batch_size + 1}: "
                            f"Processed {len(batch)}, "
                            f"kept {len(batch_result.get('generated_data', []))}")
            
            # Re-enumerate IDs unconditionally since we know processor filtered the data
            final_count = len(processed_data)
            for idx, review in enumerate(processed_data, start=1):
                review['id'] = idx
            
            # Calculate final metrics
            metrics.total_removed = (
                metrics.length_filtered
                + metrics.duplicates_removed
            )
            
            dataset_id = generate_dataset_id(
                domain=domain,
                data_size=final_count,
                custom_tag=custom_tag
            )
            
            # Update data structure with processed reviews and metrics
            data['generated_data'] = processed_data
            data['summary'] = {
                'filtering_summary': metrics.to_dict(),
                'sentiment_distribution': self._calculate_sentiment_distribution(processed_data)
            }
            
            # Save processed data
            storage_info = self.storage.save_processed_data(
                data=data,
                domain=domain,
                custom_tag=custom_tag
            )
            
            # Create dataset splits
            dataset = self.dataset_manager.create_dataset(data, dataset_id)
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'data': {
                    'generated_data': processed_data,
                    'summary': data['summary']
                },
                'dataset_info': {
                    'id': dataset_id,
                    'path': str(self.dataset_manager.base_path / dataset_id),
                    'splits': list(dataset.keys())
                },
                'storage': storage_info,
                'performance': {
                    'processing_time': processing_time,
                    'avg_time_per_review': processing_time / initial_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _calculate_sentiment_distribution(self, processed_data: List[Dict]) -> Dict[str, int]:
        """Calculate the distribution of sentiments in processed data."""
        distribution = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        for review in processed_data:
            sentiment = review.get('sentiment')
            if sentiment in distribution:
                distribution[sentiment] += 1
                
        return distribution