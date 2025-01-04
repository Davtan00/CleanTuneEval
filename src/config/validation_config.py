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
    'similarity': {
        'enabled': True,
        'stage1': {
            'enabled': True,
            'token_ngram_size': 2,  # Use word bigrams instead of character n-grams
            'threshold': 0.3  # Relaxed threshold for first stage
        },
        'stage2': {
            'enabled': True,
            'model_name': 'all-mpnet-base-v2',  # Stronger model for semantic similarity
            'threshold': 0.8  # Stricter threshold for semantic similarity
        },
        'text_preprocessing': {
            'remove_stopwords': True,
            'normalize_contractions': True,
            'strip_special_chars': True
        }
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
DOMAIN_CONFIGS = {
    'general': {
        'duplicate_threshold_modifier': 1.0
    },
    'technology': {
        'duplicate_threshold_modifier': 0.9  # More lenient for tech reviews
    },
    'service': {
        'duplicate_threshold_modifier': 1.1  # Stricter for service reviews
    }
}

def get_domain_config(domain: str) -> dict:
    """Get domain-specific configuration."""
    return DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS['general'])
