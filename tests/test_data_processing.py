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
from datasets import load_from_disk

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
    """Test the complete data processing pipeline with movie reviews"""
    hw_config = HardwareConfig.detect_hardware()
    pipeline = DataPipeline(hw_config)
    
    try:
        # Test data with more samples to allow proper splitting
        test_data = {
            "domain": "movies",
            "generated_data": [
                {
                    "text": f"This is test review number {i}. It contains enough words to pass length check.",
                    "sentiment": "positive" if i % 3 == 0 else "negative" if i % 3 == 1 else "neutral",
                    "id": i
                }
                for i in range(30)  # Generate 30 test reviews to ensure proper splitting
            ]
        }
        
        # Process the data
        result = pipeline.process_synthetic_data(test_data, custom_tag="test")
        
        # Check if processing was successful
        assert result['status'] == 'success', "Processing failed"
        data = result['data']
        
        logger.info("\nProcessing Results:")
        logger.info(f"Total reviews processed: {data['summary']['total_processed']}")
        logger.info(f"Reviews accepted: {data['summary']['total_accepted']}")
        logger.info("\nFiltering Summary:")
        logger.info(f"Length filtered: {data['summary']['filtering_summary']['length_filtered']}")
        logger.info(f"Duplicates removed: {data['summary']['filtering_summary']['duplicates_removed']}")
        logger.info(f"Total removed: {data['summary']['filtering_summary']['total_removed']}")
        
        # Verify dataset creation
        dataset_path = Path(result['dataset_info']['path'])
        assert dataset_path.exists(), "Dataset was not created"
        
        # Load and verify dataset
        dataset = load_from_disk(dataset_path)
        assert 'train' in dataset, "Missing train split"
        assert 'validation' in dataset, "Missing validation split"
        assert 'test' in dataset, "Missing test split"
        
        logger.info("Test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def test_data_validation():
    """Test data validation functionality"""
    # Your existing validation test code here
    pass

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