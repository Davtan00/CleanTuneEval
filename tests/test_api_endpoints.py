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
    """Test the data processing endpoint"""
    logger.info("Testing process-data endpoint")
    
    # Load test data
    with open('tests/data/movie.json', 'r') as f:
        test_data = json.load(f)
    
    response = client.post("/process-data", json=test_data)
    
    logger.info(f"Response status code: {response.status_code}")
    if response.status_code != 200:
        logger.error(f"Response body: {response.text}")
    
    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'success'
    
    logger.info("API Processing Results:")
    logger.info(f"Total processed: {result['data']['summary']['total_analyzed']}")
    logger.info(f"Accepted: {result['data']['summary']['total_accepted']}")

def test_hardware_info_endpoint():
    """Test the hardware info endpoint"""
    logger.info("Testing hardware-info endpoint")
    response = client.get("/hardware-info")
    
    assert response.status_code == 200
    info = response.json()
    logger.info("Hardware Info from API:")
    logger.info(f"Device: {info['device']}")
    logger.info(f"Cores: {info['cores']}") 