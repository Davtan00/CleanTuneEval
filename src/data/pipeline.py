from typing import Dict, Optional, List
from .processor import DataProcessor
from .storage import DataStorage
from .dataset_manager import DatasetManager
from .validators import ValidationMetrics
from .utils import generate_dataset_id
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging
import time

logger = setup_logging()

class DataPipeline:
    """
    Orchestrates data processing in a step-by-step fashion:
      1) Splits raw data into batches and processes them via DataProcessor.
      2) Accumulates metrics.
      3) Re-enumerates IDs once filtering is done.
      4) Saves processed data to disk.
      5) Creates dataset splits for Hugging Face.
    """
    def __init__(self, hw_config: Optional[HardwareConfig] = None):
        self.hardware_config = hw_config or HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
        self.storage = DataStorage(base_path="src/data/storage")
        self.dataset_manager = DatasetManager(base_path="src/data/datasets")
        logger.info("Initialized DataPipeline with all components")

    def process_synthetic_data(
        self,
        data: Dict,
        custom_tag: Optional[str] = None,
        batch_size: int = 1000
    ) -> Dict:
        """
        Entrypoint for processing synthetic data:
          - data: Dict with at least {'domain', 'generated_data'}.
          - custom_tag: Additional dataset labeling.
          - batch_size: Number of reviews to process in a single batch.
        """
        start_time = time.time()

        try:
            domain = data['domain']
            logger.info(f"Processing synthetic data for domain: {domain}")

            # 1) Initialize metrics and retrieve input count
            initial_count = len(data['generated_data'])
            metrics = ValidationMetrics(total_processed=initial_count)

            # 2) Process all reviews in batches
            processed_data = self._process_in_batches(
                data['generated_data'],
                domain,
                metrics,
                batch_size
            )

            # 3) Re-enumerate final IDs
            final_count = len(processed_data)
            self._re_enumerate_ids(processed_data)

            # 4) Build final summary, including sentiment distribution
            data['generated_data'] = processed_data
            data['summary'] = self._build_final_summary(processed_data, metrics)

            # 5) Save results to disk
            storage_info = self.storage.save_processed_data(
                data=data,
                domain=domain,
                custom_tag=custom_tag
            )

            # 6) Create HF dataset
            dataset_id = generate_dataset_id(
                domain=domain,
                data_size=final_count,
                custom_tag=custom_tag
            )
            dataset_splits = self.dataset_manager.create_dataset(data, dataset_id)

            # 7) Calculate performance stats
            processing_time = time.time() - start_time
            performance_data = {
                'processing_time': processing_time,
                'avg_time_per_review': processing_time / initial_count if initial_count > 0 else 0.0
            }

            return {
                'status': 'success',
                'data': {
                    'generated_data': processed_data,
                    'summary': data['summary']
                },
                'dataset_info': {
                    'id': dataset_id,
                    'path': str(self.dataset_manager.base_path / dataset_id),
                    'splits': list(dataset_splits.keys())
                },
                'storage': storage_info,
                'performance': performance_data
            }

        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _process_in_batches(
        self,
        all_reviews: List[Dict],
        domain: str,
        metrics: ValidationMetrics,
        batch_size: int
    ) -> List[Dict]:
        """
        Splits the input reviews into batches and processes each via DataProcessor,
        updating pipeline-level metrics as we go.
        """
        processed_data = []
        initial_count = len(all_reviews)

        for start_idx in range(0, initial_count, batch_size):
            batch = all_reviews[start_idx:start_idx + batch_size]
            batch_output = self.processor.process_batch(batch, domain)

            # The processor returns a dict with 'generated_data' & 'summary'
            if 'generated_data' in batch_output:
                filtered = [
                    r for r in batch_output['generated_data']
                    if not r.get('is_removed', False)
                ]
                processed_data.extend(filtered)

            # Accumulate sub-metrics
            if 'summary' in batch_output and 'filtering_summary' in batch_output['summary']:
                fs = batch_output['summary']['filtering_summary']
                metrics.length_filtered += fs.get('length_filtered', 0)
                metrics.duplicates_removed += fs.get('duplicates_removed', 0)
                metrics.exact_duplicates += fs.get('exact_duplicates', 0)
                metrics.near_duplicates += fs.get('near_duplicates', 0)

            logger.debug(f"Batch processed: start_idx={start_idx}, size={len(batch)}")

        # finalize the total_removed
        metrics.finalize()
        return processed_data

    def _re_enumerate_ids(self, reviews: List[Dict]) -> None:
        """
        Overwrites each review's 'id' with a new index from 1..N, preserving order.
        """
        for idx, review in enumerate(reviews, start=1):
            review['id'] = idx

    def _build_final_summary(
        self,
        processed_data: List[Dict],
        metrics: ValidationMetrics
    ) -> Dict:
        """
        Constructs a summary object with filtering counts and sentiment distribution.
        """
        # Sentiment distribution
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        for r in processed_data:
            sentiment = r.get('sentiment')
            if sentiment in distribution:
                distribution[sentiment] += 1

        return {
            'filtering_summary': metrics.to_dict(),
            'sentiment_distribution': distribution
        }
