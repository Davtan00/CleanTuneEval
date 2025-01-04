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
    def __init__(self, hardware_config: HardwareConfig):
        self.validator = TextValidator(hardware_config)
        self.min_words = 5
        self.max_words = 150
        logger.info(f"Initialized DataProcessor with word limits: {self.min_words}-{self.max_words}")
        
    def process_batch(self, data: Dict, batch_size: int = 1000) -> Dict:
        """Process a batch of synthetic reviews with proper structure for pipeline."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing batch for domain: {data['domain']}")
            
            # Create DataFrame with required columns
            df = pd.DataFrame(data['generated_data'])[['id', 'text', 'sentiment']]
            initial_size = len(df)
            
            # Basic cleaning and filtering
            df['clean_text'] = df['text'].apply(self._basic_clean)
            df['word_count'] = df['clean_text'].str.split().str.len()
            
            # Length filtering
            length_mask = (df['word_count'] >= self.min_words) & (df['word_count'] <= self.max_words)
            length_filtered = int((~length_mask).sum())  # Convert to Python int otherwise JSON serialization ERROR
            df = df[length_mask]
            
            # Process duplicates in batches
            duplicate_stats = {'exact': 0, 'similar': 0}
            is_duplicate = []
            
            for i in range(0, len(df), batch_size):
                batch = df['clean_text'].iloc[i:i + batch_size].tolist()
                batch_results = self.validator.detect_duplicates(batch)
                is_duplicate.extend([r[0] for r in batch_results])
                
                # Update stats
                for _, dup_type in batch_results:
                    if dup_type:
                        duplicate_stats[dup_type] += 1
            
            df['is_duplicate'] = is_duplicate
            df = df[~df['is_duplicate']]
            
            if len(df) == 0:
                logger.error("All reviews were filtered out")
                return {
                    'status': 'error',
                    'message': 'All reviews were filtered out',
                    'summary': {
                        'total_processed': int(initial_size),
                        'filtering_summary': {
                            'length_filtered': length_filtered,
                            'duplicates_removed': sum(duplicate_stats.values()),
                            'exact_duplicates': duplicate_stats['exact'],
                            'near_duplicates': duplicate_stats['similar'],
                            'total_removed': int(initial_size)
                        }
                    }
                }
            
            # Convert numeric values to Python native types
            sentiment_dist = df['sentiment'].value_counts().to_dict()
            sentiment_dist = {k: int(v) for k, v in sentiment_dist.items()}
            
            # Prepare final output structure
            processed_data = {
                'generated_data': df[['id', 'clean_text', 'sentiment']].to_dict('records'),
                'domain': data['domain'],
                'summary': {
                    'total_processed': int(initial_size),
                    'total_accepted': int(len(df)),
                    'sentiment_distribution': sentiment_dist,
                    'filtering_summary': {
                        'length_filtered': length_filtered,
                        'duplicates_removed': sum(duplicate_stats.values()),
                        'exact_duplicates': duplicate_stats['exact'],
                        'near_duplicates': duplicate_stats['similar'],
                        'total_removed': int(initial_size - len(df))
                    }
                }
            }
            
            return {
                'status': 'success',
                'data': processed_data,
                'performance': {
                    'processing_time': float(time.time() - start_time),
                    'avg_time_per_review': float((time.time() - start_time) / initial_size)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_batch: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _basic_clean(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        
        # Standardize punctuation spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Fix multiple punctuation
        text = re.sub(r'([.,!?]){2,}', r'\1', text)
        
        # Ensure proper capitalization
        text = '. '.join(s.capitalize() for s in text.split('. '))
        
        return text
    
    def _compute_metrics(self, text: str) -> ValidationMetrics:
        """Compute quality metrics for a single text"""
        vocab_richness = self.validator.compute_vocabulary_richness(text)
        
        # More sophisticated metrics could be added here
        return ValidationMetrics(
            perplexity=0.0,  # Placeholder for now
            similarity_score=0.0,  # Will be computed in batch
            vocabulary_richness=vocab_richness,
            is_outlier=False  # Will be updated in batch
        ) 