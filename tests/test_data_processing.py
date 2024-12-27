import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.environment import HardwareConfig
from src.data.pipeline import DataPipeline
from src.config.logging_config import setup_logging

logger = setup_logging()

def test_process_movie_reviews():
    """Test the data processing pipeline with movie reviews"""
    logger.info("Loading movie review test data...")
    
    try:
        with open('tests/data/movie.json', 'r') as f:
            movie_data = json.load(f)
            logger.info(f"Loaded {len(movie_data['generated_data'])} reviews")
    except FileNotFoundError:
        logger.error("Movie review test data not found!")
        return
    
    logger.info("Initializing data pipeline...")
    pipeline = DataPipeline()
    
    logger.info("Processing movie reviews...")
    result = pipeline.process_synthetic_data(movie_data)
    
    if result['status'] == 'success':
        data = result['data']
        logger.success("\nProcessing Results:")
        logger.info(f"Total reviews analyzed: {data['summary']['total_analyzed']}")
        logger.info(f"Total reviews accepted: {data['summary']['total_accepted']}")
        
        logger.info("\nSentiment Distribution:")
        for sentiment, count in data['summary']['sentiment_distribution'].items():
            logger.info(f"{sentiment}: {count}")
        
        logger.info("\nQuality Metrics:")
        logger.info(f"Average vocabulary richness: {data['summary']['quality_metrics']['avg_vocabulary_richness']:.3f}")
        logger.info(f"Duplicate rate: {data['summary']['quality_metrics']['duplicate_rate']:.3f}")
    else:
        logger.error(f"Processing failed: {result['message']}")
    
    return result

def test_data_validation():
    """Test specific validation features"""
    with open('tests/data/movie.json', 'r') as f:
        movie_data = json.load(f)
    
    pipeline = DataPipeline()
    result = pipeline.process_synthetic_data(movie_data)
    
    # Validate structure
    assert 'summary' in result['data'], "Missing summary in results"
    assert 'quality_metrics' in result['data']['summary'], "Missing quality metrics"
    
    # Validate metrics
    metrics = result['data']['summary']['quality_metrics']
    assert 0 <= metrics['duplicate_rate'] <= 1, "Invalid duplicate rate"
    assert 0 <= metrics['avg_vocabulary_richness'] <= 1, "Invalid vocabulary richness"

if __name__ == "__main__":
    test_process_movie_reviews()
    test_data_validation() 