"""Tests for the enhanced duplicate detection system."""

import pytest
import numpy as np
from src.data.validators import TextValidator
from src.config.environment import HardwareConfig
from src.config.validation_config import DUPLICATE_CONFIG

@pytest.fixture
def validator():
    config = HardwareConfig(device='cpu', n_cores=1, memory_limit=8, use_mps=False)
    return TextValidator(config)

def test_exact_duplicate_detection(validator):
    texts = [
        "This is a test review.",
        "This is a test review.",  # Exact duplicate
        "This is a different review.",
        "THIS IS A TEST REVIEW."  # Case different
    ]
    
    results = validator.detect_duplicates(texts)
    
    # Check results
    assert results[0] == (False, None)  # First occurrence is not a duplicate
    assert results[1] == (True, 'exact')  # Second is exact duplicate
    assert results[2] == (False, None)  # Different text
    assert results[3] == (True, 'exact')  # Case-insensitive match

def test_ngram_duplicate_detection(validator):
    texts = [
        "The product has great features and good quality.",
        "This product has good features and great quality.",  # Similar but rearranged
        "A completely different review about something else.",
    ]
    
    results = validator.detect_duplicates(texts)
    
    # Check results
    assert results[0] == (False, None)  # First occurrence
    assert results[1] == (True, 'ngram')  # Similar text detected by n-grams
    assert results[2] == (False, None)  # Different text

def test_semantic_duplicate_detection(validator):
    texts = [
        "The smartphone has excellent performance.",
        "The mobile phone performs exceptionally well.",  # Semantically similar
        "The weather is nice today.",  # Different meaning
    ]
    
    results = validator.detect_duplicates(texts)
    
    # Check results
    assert results[0] == (False, None)  # First occurrence
    assert results[1] == (True, 'semantic')  # Semantically similar
    assert results[2] == (False, None)  # Different meaning

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
    assert tech_results[1][1] in ['ngram', 'semantic']  # Either by n-gram or semantic similarity

def test_performance_with_large_dataset(validator):
    # Generate a large dataset
    base_texts = [
        "The product is good.",
        "The service was excellent.",
        "Not satisfied with the quality.",
        "Great features and performance.",
    ]
    
    # Create variations and duplicates
    texts = []
    for _ in range(25):  # Creates 100 reviews
        texts.extend(base_texts)
    
    results = validator.detect_duplicates(texts)
    
    # Check basic expectations
    assert len(results) == len(texts)
    assert any(r[0] for r in results)  # Should find some duplicates
    
    # Count duplicate types
    duplicate_counts = {
        'exact': sum(1 for r in results if r[1] == 'exact'),
        'ngram': sum(1 for r in results if r[1] == 'ngram'),
        'semantic': sum(1 for r in results if r[1] == 'semantic')
    }
    
    # We should find mostly exact duplicates in this case
    assert duplicate_counts['exact'] > duplicate_counts['ngram']
    assert duplicate_counts['exact'] > duplicate_counts['semantic']
