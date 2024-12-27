from typing import Dict, List, Optional
import pandas as pd
from .validators import TextValidator, ValidationMetrics

class DataProcessor:
    def __init__(self, hardware_config: HardwareConfig):
        self.validator = TextValidator(hardware_config)
        self.min_words = 3
        self.max_words = 100
        
    def process_batch(self, data: Dict) -> Dict:
        """Process a batch of synthetic reviews"""
        df = pd.DataFrame(data['generated_data'])
        
        # Basic cleaning
        df['clean_text'] = df['text'].apply(self._basic_clean)
        
        # Length filtering
        df['word_count'] = df['clean_text'].str.split().str.len()
        df = df[(df['word_count'] >= self.min_words) & 
                (df['word_count'] <= self.max_words)]
        
        # Compute quality metrics
        df['metrics'] = df['clean_text'].apply(self._compute_metrics)
        
        # Detect duplicates
        duplicates = self.validator.detect_duplicates(df['clean_text'].tolist())
        df['is_duplicate'] = duplicates
        
        # Filter out low quality samples
        df = df[
            (~df['is_duplicate']) & 
            (df['metrics'].apply(lambda x: not x.is_outlier)) &
            (df['metrics'].apply(lambda x: x.vocabulary_richness > 0.4))
        ]
        
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
                    'duplicate_rate': sum(duplicates) / len(duplicates)
                }
            }
        }
        
        return processed_data
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Add more cleaning as needed
        return text
    
    def _compute_metrics(self, text: str) -> ValidationMetrics:
        """Compute quality metrics for a single text"""
        vocab_richness = self.validator.compute_vocabulary_richness(text)
        # More metrics can be added here
        return ValidationMetrics(
            perplexity=0.0,  # Placeholder for now
            similarity_score=0.0,  # Will be computed in batch
            vocabulary_richness=vocab_richness,
            is_outlier=False  # Will be updated in batch
        ) 