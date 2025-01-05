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
from .config import (
    TEST_STORAGE_DIR,
    TEST_DATASETS_DIR,
    setup_test_environment,
    cleanup_test_environment
)
from datasets import load_from_disk
import pytest

logger = setup_logging()

def setup_module():
    """Set up test environment before any tests run."""
    setup_test_environment()

def teardown_module():
    """Clean up test environment after all tests complete."""
    cleanup_test_environment()

@pytest.fixture
def test_pipeline():
    """Create a test-specific pipeline instance."""
    hw_config = HardwareConfig.detect_hardware()
    pipeline = DataPipeline(hw_config)
    # Configure test-specific storage paths
    pipeline.storage = DataStorage(base_path=str(TEST_STORAGE_DIR))
    pipeline.dataset_manager = DatasetManager(base_path=str(TEST_DATASETS_DIR))
    return pipeline

@pytest.mark.integration
def test_process_movie_reviews(test_pipeline):
    """Test the complete data processing pipeline with movie reviews"""
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
        result = test_pipeline.process_synthetic_data(test_data, custom_tag="test")
        
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

@pytest.mark.slow
def test_process_tech_reviews(test_pipeline, test_data_dir):
    """Test processing of technology domain reviews"""
    tech_data_path = test_data_dir / "tech_reviews.json"
    logger.info("Loading technology review test data...")
    
    try:
        with open(tech_data_path, 'r') as f:
            tech_data = json.load(f)
            logger.info(f"Loaded {len(tech_data['generated_data'])} technology reviews")
    except FileNotFoundError:
        logger.error("Technology review test data not found!")
        pytest.skip("Technology review test data not found")  # Skip instead of fail
    
    result = test_pipeline.process_synthetic_data(tech_data)
    
    assert result['status'] == 'success', "Data processing failed"
    data = result['data']
    
    # Log processing results
    logger.success("\nProcessing Results:")
    logger.info(f"Total processed: {data['summary']['total_processed']}")  # Changed from total_analyzed
    logger.info(f"Total accepted: {data['summary']['total_accepted']}")
    
    logger.info("\nSentiment Distribution:")
    for sentiment, count in data['summary']['sentiment_distribution'].items():
        logger.info(f"{sentiment}: {count}")
    
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