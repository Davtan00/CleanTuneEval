import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.environment import HardwareConfig
from src.data.pipeline import DataPipeline
from src.data.processor import DataProcessor
from src.config.logging_config import setup_logging
from src.data.storage import DataStorage
from src.data.dataset_manager import DatasetManager
from tests.config import (
    TEST_STORAGE_DIR,
    TEST_DATASETS_DIR,
    setup_test_environment,
    cleanup_test_environment
)

logger = setup_logging()

class TestDataPipeline(DataPipeline):
    """Test-specific data pipeline that uses test directories."""
    def __init__(self):
        self.hardware_config = HardwareConfig.detect_hardware()
        self.processor = DataProcessor(self.hardware_config)
        # Initialize storage and dataset manager with test paths
        self.storage = DataStorage(base_path=str(TEST_STORAGE_DIR))
        self.dataset_manager = DatasetManager(base_path=str(TEST_DATASETS_DIR))
        logger.info("Initialized TestDataPipeline with test directories")

def setup_module():
    """Set up test environment before any tests run."""
    setup_test_environment()

def teardown_module():
    """Clean up test environment after all tests complete."""
    cleanup_test_environment()

def test_process_movie_reviews():
    """Test the data processing pipeline with movie reviews"""
    logger.info("Loading movie review test data...")
    
    try:
        with open('tests/data/movie.json', 'r') as f:
            movie_data = json.load(f)
            logger.info(f"Loaded {len(movie_data['generated_data'])} reviews")
    except FileNotFoundError:
        logger.error("Movie review test data not found!")
        assert False, "Movie review test data not found!"
    
    logger.info("Initializing data pipeline...")
    pipeline = TestDataPipeline()
    
    logger.info("Processing movie reviews...")
    result = pipeline.process_synthetic_data(movie_data)
    
    assert result['status'] == 'success', "Data processing failed"
    data = result['data']
    
    # Log processing results
    logger.success("\nProcessing Results:")
    logger.info(f"Total reviews analyzed: {data['summary']['total_analyzed']}")
    logger.info(f"Total reviews accepted: {data['summary']['total_accepted']}")
    
    logger.info("\nSentiment Distribution:")
    for sentiment, count in data['summary']['sentiment_distribution'].items():
        logger.info(f"{sentiment}: {count}")
    
    logger.info("\nQuality Metrics:")
    logger.info(f"Average vocabulary richness: {data['summary']['quality_metrics']['avg_vocabulary_richness']:.3f}")
    logger.info(f"Duplicate rate: {data['summary']['quality_metrics']['duplicate_rate']:.3f}")
    
    logger.info("\nSentiment Distribution:")
    for sentiment, count in data['summary']['sentiment_distribution'].items():
        logger.info(f"{sentiment}: {count}")
    
    logger.info("\nQuality Metrics:")
    logger.info(f"Average vocabulary richness: {data['summary']['quality_metrics']['avg_vocabulary_richness']:.3f}")
    logger.info(f"Duplicate rate: {data['summary']['quality_metrics']['duplicate_rate']:.3f}")

    # Verify test data is stored in test directories
    assert 'test_data/storage' in result['storage']['raw_path'], "Data not stored in test directory"
    assert 'test_data/datasets' in result['dataset_info']['path'], "Dataset not stored in test directory"

def test_data_validation():
    """Test specific validation features"""
    with open('tests/data/movie.json', 'r') as f:
        movie_data = json.load(f)
    
    pipeline = TestDataPipeline()
    result = pipeline.process_synthetic_data(movie_data)
    
    # Validate structure
    assert 'summary' in result['data'], "Missing summary in results"
    assert 'quality_metrics' in result['data']['summary'], "Missing quality metrics"
    
    # Validate metrics
    metrics = result['data']['summary']['quality_metrics']
    assert 0 <= metrics['duplicate_rate'] <= 1, "Invalid duplicate rate"
    assert 0 <= metrics['avg_vocabulary_richness'] <= 1, "Invalid vocabulary richness"

def test_process_tech_reviews():
    """Test processing of technology domain reviews"""
    logger.info("Loading technology review test data...")
    
    try:
        with open('tests/data/tech_reviews.json', 'r') as f:
            tech_data = json.load(f)
            logger.info(f"Loaded {len(tech_data['generated_data'])} technology reviews")
    except FileNotFoundError:
        logger.error("Technology review test data not found!")
        assert False, "Technology review test data not found!"
    
    pipeline = TestDataPipeline()
    result = pipeline.process_synthetic_data(tech_data)
    
    assert result['status'] == 'success', "Data processing failed"
    data = result['data']
    
    # Log processing results
    logger.success("\nProcessing Results:")
    logger.info(f"Total reviews analyzed: {data['summary']['total_analyzed']}")
    logger.info(f"Total reviews accepted: {data['summary']['total_accepted']}")
    logger.info(f"Acceptance rate: {data['summary']['quality_metrics']['acceptance_rate']:.2%}")
    
    logger.info("\nSentiment Distribution:")
    for sentiment, count in data['summary']['sentiment_distribution'].items():
        logger.info(f"{sentiment}: {count}")
    
    logger.info("\nQuality Metrics:")
    metrics = data['summary']['quality_metrics']
    logger.info(f"Average vocabulary richness: {metrics['avg_vocabulary_richness']:.3f}")
    logger.info(f"Duplicate rate: {metrics['duplicate_rate']:.3f}")
    logger.info(f"Average word count: {metrics['avg_word_count']:.1f}")
    
    logger.info("\nFiltering Summary:")
    filtering = data['summary']['filtering_summary']
    logger.info(f"Length filtered: {filtering['length_filtered']}")
    logger.info(f"Duplicates removed: {filtering['duplicates_removed']}")
    logger.info(f"Total removed: {filtering['total_removed']}")
    
    # Verify test data is stored in test directories
    assert 'test_data/storage' in result['storage']['raw_path'], "Data not stored in test directory"
    assert 'test_data/datasets' in result['dataset_info']['path'], "Dataset not stored in test directory"

if __name__ == "__main__":
    test_process_movie_reviews()
    test_data_validation()
    test_process_tech_reviews() 