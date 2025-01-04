import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.config.logging_config import setup_logging

logger = setup_logging()
client = TestClient(app)

def test_process_data_endpoint():
    """Test the process-data endpoint"""
    logger.info("Testing process-data endpoint")
    
    # Test data with sufficient length and variety
    test_data = {
        "domain": "movie_reviews",
        "generated_data": [
            {
                "text": f"This is a detailed movie review number {i}. The plot was interesting and the acting was convincing. " 
                       f"The cinematography was well done and the soundtrack enhanced the experience. Overall, it's worth watching.",
                "sentiment": "positive" if i % 3 == 0 else "negative" if i % 3 == 1 else "neutral",
                "id": i
            }
            for i in range(100)
        ]
    }
    
    response = client.post("/process-data", json=test_data)
    logger.info(f"Response status code: {response.status_code}")
    
    result = response.json()
    assert result['status'] == 'success', f"Processing failed: {result.get('message', 'Unknown error')}"
    
    if result['status'] == 'success':
        logger.info("\nAPI Processing Results:")
        logger.info(f"Total processed: {result['data']['summary']['total_processed']}")
        logger.info(f"Reviews accepted: {result['data']['summary']['total_accepted']}")
        logger.info(f"Dataset ID: {result['dataset_info']['id']}")
    else:
        logger.error(f"Processing failed: {result.get('message', 'Unknown error')}")
    
    return True

def test_hardware_info_endpoint():
    """Test the hardware info endpoint"""
    logger.info("Testing hardware-info endpoint")
    response = client.get("/hardware-info")
    
    assert response.status_code == 200
    info = response.json()
    logger.info("Hardware Info from API:")
    logger.info(f"Device: {info['device']}")
    logger.info(f"Cores: {info['cores']}") 