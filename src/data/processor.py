from typing import Dict, List, Optional
import pandas as pd
import re
from tqdm import tqdm
from .validators import TextValidator, ValidationMetrics
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging
import time

logger = setup_logging()

class DataProcessor:
    """
    Handles text cleaning, basic filtering (length), and duplicate detection.
    Returns results that include processed reviews and a filtering summary.
    """
    def __init__(self, hardware_config: HardwareConfig):
        self.validator = TextValidator(hardware_config)
        self.min_words = 5
        self.max_words = 150
        logger.info(f"Initialized DataProcessor with word limits: {self.min_words}-{self.max_words}")
        
    def process_batch(self, batch: List[Dict], domain: str) -> Dict:
        """
        High-level public method for batch processing. Returns a dict containing:
          {
            'generated_data': [...],
            'domain': domain,
            'summary': {
                'filtering_summary': {
                    'length_filtered': ...,
                    'duplicates_removed': ...,
                    'exact_duplicates': ...,
                    'near_duplicates': ...,
                    'total_removed': ...
                },
                'total_accepted': ...,
                'sentiment_distribution': {...},
            }
          }
        """
        start_time = time.time()
        initial_size = len(batch)

        logger.info(f"Processing batch of {initial_size} reviews for domain: {domain}")
        if initial_size == 0:
            # Return an empty structure if nothing to process
            return self._empty_result(domain)

        # Convert to DataFrame
        df = pd.DataFrame(batch, columns=['id', 'text', 'sentiment'])
        
        # 1) Clean text and filter by length
        df['clean_text'] = df['text'].apply(self._basic_clean)
        df['word_count'] = df['clean_text'].str.split().str.len()
        length_mask = (df['word_count'] >= self.min_words) & (df['word_count'] <= self.max_words)
        length_filtered_count = (~length_mask).sum()  # how many are out of range
        df = df[length_mask]

        # 2) Duplicate detection
        duplicate_stats = {'exact': 0, 'similar': 0}
        is_duplicate = self._detect_batch_duplicates(df['clean_text'].tolist(), duplicate_stats)
        df['is_duplicate'] = is_duplicate
        df = df[~df['is_duplicate']]

        # 3) Handle case where all are removed
        if len(df) == 0:
            return {
                'domain': domain,
                'generated_data': [],
                'summary': {
                    'filtering_summary': {
                        'length_filtered': int(length_filtered_count),
                        'duplicates_removed': sum(duplicate_stats.values()),
                        'exact_duplicates': duplicate_stats['exact'],
                        'near_duplicates': duplicate_stats['similar'],
                        'total_removed': initial_size
                    },
                    'total_accepted': 0,
                    'sentiment_distribution': {}
                }
            }

        # 4) Build final data
        sentiment_dist = df['sentiment'].value_counts().to_dict()
        sentiment_dist = {k: int(v) for k, v in sentiment_dist.items()}

        final_reviews = df[['id', 'clean_text', 'sentiment']].to_dict('records')
        
        # Return an explicit structure
        return {
            'domain': domain,
            'generated_data': final_reviews,
            'summary': {
                'filtering_summary': {
                    'length_filtered': int(length_filtered_count),
                    'duplicates_removed': sum(duplicate_stats.values()),
                    'exact_duplicates': duplicate_stats['exact'],
                    'near_duplicates': duplicate_stats['similar'],
                    'total_removed': int(length_filtered_count + sum(duplicate_stats.values()))
                },
                'total_accepted': len(final_reviews),
                'sentiment_distribution': sentiment_dist
            }
        }

    def _detect_batch_duplicates(
        self,
        texts: List[str],
        duplicate_stats: Dict[str, int],
        chunk_size: int = 100
    ) -> List[bool]:
        """
        Helper function to run the validator's detect_duplicates in chunks.
        Updates the passed-in 'duplicate_stats' for exact or similar duplicates.
        Returns a boolean list indicating whether each text is a duplicate.
        """
        is_duplicate = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            results = self.validator.detect_duplicates(chunk)
            for (dup_bool, dup_type) in results:
                is_duplicate.append(dup_bool)
                if dup_type == 'exact':
                    duplicate_stats['exact'] += 1
                elif dup_type == 'similar':
                    duplicate_stats['similar'] += 1
        return is_duplicate

    def _basic_clean(self, text: str) -> str:
        """
        Enhanced text cleaning:
          - Remove extra spaces
          - Remove special chars (except .,!?-)
          - Standardize punctuation
          - Simple capitalization
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        # Standardize punctuation spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        # Fix multiple punctuation
        text = re.sub(r'([.,!?]){2,}', r'\1', text)
        # Ensure basic capitalization
        text = '. '.join(s.capitalize() for s in text.split('. '))
        return text

    def _empty_result(self, domain: str) -> Dict:
        """
        Returns a blank structure for empty input batches.
        """
        return {
            'domain': domain,
            'generated_data': [],
            'summary': {
                'filtering_summary': {
                    'length_filtered': 0,
                    'duplicates_removed': 0,
                    'exact_duplicates': 0,
                    'near_duplicates': 0,
                    'total_removed': 0
                },
                'total_accepted': 0,
                'sentiment_distribution': {}
            }
        }
