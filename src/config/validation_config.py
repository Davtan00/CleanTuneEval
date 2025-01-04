"""
Configuration settings for text validation and quality assessment.
"""
from typing import Dict, Any

# Duplicate detection configuration
DUPLICATE_CONFIG = {
    'exact_match': {
        'enabled': True,
        'normalize_case': True,
        'strip_punctuation': True
    },
    'ngram': {
        'enabled': True,
        'n': 3,
        'threshold': 0.4  # Even more lenient n-gram threshold
    },
    'semantic': {
        'enabled': True,
        'model_name': 'all-MiniLM-L6-v2',
        'threshold': 0.8,  # Stricter semantic threshold
        'min_length_ratio': 0.6
    }
}

# Quality metrics configuration
QUALITY_CONFIG = {
    'text_length': {
        'min_words': 10,
        'max_words': 200,
        'min_sentences': 2,
        'max_sentences': 20
    },
    'readability': {
        'min_flesch_score': 60,
        'enabled': True
    },
    'vocabulary': {
        'min_unique_ratio': 0.7,
        'max_repeated_phrases': 2
    }
}

# Domain-specific configurations
DOMAIN_CONFIG = {
    'technology': {
        'duplicate_threshold_modifier': 1.0,
        'min_technical_terms': 1
    },
    'product': {
        'duplicate_threshold_modifier': 0.95,
        'required_aspects': ['quality', 'functionality', 'value']
    },
    'service': {
        'duplicate_threshold_modifier': 0.9,
        'required_aspects': ['service_quality', 'timeliness']
    }
}

def get_domain_config(domain: str) -> Dict[str, Any]:
    """Get domain-specific configuration with fallback to default values."""
    default_config = {
        'duplicate_threshold_modifier': 1.0,
        'required_aspects': []
    }
    return DOMAIN_CONFIG.get(domain, default_config)
