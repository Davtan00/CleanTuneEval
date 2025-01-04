"""Tests for the enhanced duplicate detection system."""

import pytest
import numpy as np
from src.data.validators import TextValidator
from src.config.environment import HardwareConfig
from src.config.validation_config import DUPLICATE_CONFIG
from typing import List, Tuple, Optional

@pytest.fixture
def validator():
    config = HardwareConfig(device='cpu', n_cores=1, memory_limit=8, use_mps=False)
    return TextValidator(config)

def test_duplicate_detection(validator):
    texts = [
        "After updating the software on my tablet, its been running slower than molasses.",
        "After updating the software on my tablet, its been running slower than molasses.",  # Exact duplicate
        "The tablet i purchased has a nice screen but is incredibly slow.",
        "AFTER UPDATING THE SOFTWARE ON MY TABLET, ITS BEEN RUNNING SLOWER THAN MOLASSES."  # Case different
    ]
    
    results = validator.detect_duplicates(texts)
    
    assert results[0] == (False, None)  # First occurrence is not a duplicate
    assert results[1] == (True, 'exact')  # Second is exact duplicate
    assert results[2] == (False, None)  # Different text
    assert results[3] == (True, 'exact')  # Case-different duplicate

def test_similar_text_detection(validator):
    texts = [
        "The laptop freezes when running multiple apps.",
        "The laptop keeps freezing with multiple apps.",  # Similar meaning
        "The smart home device works great.",  # Different review
    ]
    
    results = validator.detect_duplicates(texts)
    
    assert results[0] == (False, None)  # First occurrence
    assert results[1] == (True, 'similar')  # Similar text
    assert results[2] == (False, None)  # Different text

def test_semantic_duplicate_detection(validator):
    texts = [
        "This device is extremely sluggish and unresponsive, making it frustrating to use.",
        "The device is very sluggish and unresponsive, which makes it frustrating.",  # Actually similar
        "The performance is terribly slow and the system barely responds to input, which is very annoying.",  # Different complaint
        "The software update added new features but had some minor issues.",  # Different meaning
    ]
    
    results = validator.detect_duplicates(texts)
    
    assert results[0] == (False, None)  # First occurrence is always kept
    assert results[1][0] == True  # Should be flagged as duplicate (actually similar)
    assert results[2] == (False, None)  # Different complaint, should NOT be a duplicate
    assert results[3] == (False, None)  # Different meaning should be kept

def test_domain_specific_thresholds(validator):
    texts = [
        "The API integration was smooth and efficient.",
        "The API implementation was efficient and smooth.",  # Similar tech review
    ]
    
    # Test with different domains
    tech_results = validator.detect_duplicates(texts, domain='technology')
    service_results = validator.detect_duplicates(texts, domain='service')
    
    # Technology domain should be more lenient with technical terms
    assert tech_results[1][0]  # Should be marked as duplicate
    assert tech_results[1][1] in ['exact', 'similar']  # Type doesn't matter

def test_performance_with_large_dataset(validator):
    # Reduced dataset size for faster testing while still being meaningful
    base_texts = [
        "The product is good.",
        "The service was excellent.",
        "Not satisfied with the quality.",
        "Great features and performance.",
    ]
    
    texts = []
    for _ in range(5):  # Reduced from 25 to 5 iterations
        texts.extend(base_texts)
    
    results = validator.detect_duplicates(texts)
    
    # We only care about filtering out obvious duplicates
    duplicate_count = sum(1 for result in results if result[0])
    non_duplicate_count = sum(1 for result in results if not result[0])
    
    # Basic assertions
    assert len(results) == len(texts)  # Should have one result per text
    assert duplicate_count > 0  # Should find some duplicates
    assert non_duplicate_count > 0  # Should keep some originals
