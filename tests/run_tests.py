import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logging_config import setup_logging
from test_data_processing import test_process_movie_reviews, test_data_validation
from test_model_adaptation import test_model_initialization, test_model_adaptation
from test_api_endpoints import test_process_data_endpoint, test_hardware_info_endpoint

logger = setup_logging()

def run_all_tests():
    """Run all tests and print results"""
    logger.info("="*50)
    logger.info("Starting test suite execution")
    logger.info("="*50)
    
    try:
        # Data Processing Tests
        logger.info("\n[1/3] Running Data Processing Tests")
        logger.info("-"*30)
        test_process_movie_reviews()
        test_data_validation()
        logger.success("Data Processing Tests completed")
        
        # Model Adaptation Tests
        logger.info("\n[2/3] Running Model Adaptation Tests")
        logger.info("-"*30)
        test_model_initialization()
        logger.success("Model Adaptation Tests completed")
        
        # API Endpoint Tests
        logger.info("\n[3/3] Running API Endpoint Tests")
        logger.info("-"*30)
        test_process_data_endpoint()
        test_hardware_info_endpoint()
        logger.success("API Endpoint Tests completed")
        
        logger.success("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {str(e)}")
        logger.exception("Detailed error trace:")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests() 