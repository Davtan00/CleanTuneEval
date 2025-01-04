from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class SentimentDistributionAnalyzer:
    """Analyze and compute sentiment distributions and weights based on domain research"""
    
    def __init__(self, research_data_path: str = "src/data/research/domain_distributions.json"):
        self.research_data_path = Path(research_data_path)
        self.distributions = self._load_research_distributions()
        
    def _load_research_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Load researched sentiment distributions for different domains
        Format:
        {
            "healthcare": {
                "distribution": [0.15, 0.45, 0.40],  # [neg, neu, pos]
                "source": "Smith et al. 2023, Analysis of Patient Feedback",
                "sample_size": 50000,
                "confidence": 0.95
            },
            ...
        }
        """
        if not self.research_data_path.exists():
            logger.warning("No research distribution data found. Using balanced weights.")
            return {}
            
        with open(self.research_data_path) as f:
            return json.load(f)
    
    def analyze_domain_distribution(self, 
                                  domain: str,
                                  actual_labels: List[int]) -> Dict[str, Any]:
        """Compare actual distribution with research expectations"""
        actual_dist = np.bincount(actual_labels, minlength=3) / len(actual_labels)
        
        if domain in self.distributions:
            expected_dist = self.distributions[domain]["distribution"]
            divergence = {
                "actual": actual_dist.tolist(),
                "expected": expected_dist,
                "difference": (actual_dist - np.array(expected_dist)).tolist(),
                "source": self.distributions[domain]["source"]
            }
        else:
            divergence = {
                "actual": actual_dist.tolist(),
                "warning": "No research distribution available for comparison"
            }
            
        return divergence
    
    def get_domain_weights(self, 
                          domain: str,
                          actual_labels: Optional[List[int]] = None) -> np.ndarray:
        """
        Calculate weights based on research distributions and actual data
        """
        if domain not in self.distributions:
            logger.warning(f"No research distribution for {domain}. Using balanced weights.")
            return np.ones(3)
            
        expected_dist = np.array(self.distributions[domain]["distribution"])
        
        # Inverse frequency weighting with smoothing
        weights = 1 / (expected_dist + 1e-5)
        
        # Normalize weights
        weights = weights / weights.sum() * 3
        
        if actual_labels is not None:
            # Log comparison with actual distribution
            analysis = self.analyze_domain_distribution(domain, actual_labels)
            logger.info(f"Domain distribution analysis: {analysis}")
            
        return weights 