from typing import Dict, List, Optional
import pandas as pd
import re
from .validators import TextValidator, ValidationMetrics
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class DataProcessor:
    def __init__(self, hardware_config: HardwareConfig):
        self.validator = TextValidator(hardware_config)
        self.min_words = 5
        self.max_words = 150
        logger.info(f"Initialized DataProcessor with word limits: {self.min_words}-{self.max_words}")
        
    def process_batch(self, data: Dict) -> Dict:
        """Process a batch of synthetic reviews"""
        logger.info(f"Processing batch for domain: {data['domain']}")
        logger.info(f"Initial data size: {len(data['generated_data'])} reviews")
        
        df = pd.DataFrame(data['generated_data'])
        
        # Basic cleaning
        df['clean_text'] = df['text'].apply(self._basic_clean)
        initial_size = len(df)
        
        # Length filtering
        df['word_count'] = df['clean_text'].str.split().str.len()
        df = df[(df['word_count'] >= self.min_words) & 
                (df['word_count'] <= self.max_words)]
        length_filtered = initial_size - len(df)
        logger.info(f"Removed {length_filtered} reviews due to length constraints")
        
        # Compute quality metrics
        logger.info("Computing quality metrics...")
        df['metrics'] = df['clean_text'].apply(self._compute_metrics)
        
        # Detect duplicates
        logger.info("Detecting duplicates...")
        duplicates = self.validator.detect_duplicates(df['clean_text'].tolist())
        df['is_duplicate'] = duplicates
        duplicate_count = sum(duplicates)
        logger.info(f"Found {duplicate_count} duplicate reviews")
        
        # Filter out low quality samples
        df = df[
            (~df['is_duplicate']) & 
            (df['metrics'].apply(lambda x: not x.is_outlier)) &
            (df['metrics'].apply(lambda x: x.vocabulary_richness > 0.3))
        ]
        
        final_size = len(df)
        logger.info(f"Final dataset size: {final_size} reviews")
        
        # Prepare output format
        processed_data = {
            'domain': data['domain'],
            'generated_data': df[['id', 'clean_text', 'sentiment']].to_dict('records'),
            'summary': {
                'total_analyzed': len(data['generated_data']),
                'total_accepted': len(df),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
                'quality_metrics': {
                    'avg_vocabulary_richness': df['metrics'].apply(lambda x: x.vocabulary_richness).mean(),
                    'duplicate_rate': duplicate_count / initial_size,
                    'acceptance_rate': final_size / initial_size,
                    'avg_word_count': df['word_count'].mean()
                },
                'filtering_summary': {
                    'length_filtered': length_filtered,
                    'duplicates_removed': duplicate_count,
                    'total_removed': initial_size - final_size
                }
            }
        }
        
        return processed_data
    
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