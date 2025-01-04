from typing import List, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

class TextValidator:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
        
        # Stricter thresholds
        self.exact_threshold = 0.999
        self.filter_threshold = 0.85   # Increased from 0.65 to be more strict
        
        self.domain_adjustments = {
            'technology': -0.05,
            'service': 0.0,
        }

    def _should_filter(self, text1: str, text2: str, domain: str = None) -> Tuple[bool, Optional[str]]:
        """
        Returns (should_filter, filter_type) with stricter similarity requirements
        """
        # Normalize texts
        text1 = ' '.join(text1.lower().strip().split())
        text2 = ' '.join(text2.lower().strip().split())
        
        # Exact match check first
        if text1 == text2:
            return True, 'exact'
            
        # Get embeddings and calculate similarity
        try:
            embeddings = self.model.encode([text1, text2], 
                                         convert_to_tensor=True,
                                         normalize_embeddings=True)
            
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            
            # Add length ratio check to avoid matching very different length texts
            len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
            
            # Adjust threshold based on domain
            threshold = self.filter_threshold
            if domain:
                threshold += self.domain_adjustments.get(domain, 0)
            
            # Only consider similar if both similarity and length ratio are high enough    
            if similarity >= threshold and len_ratio > 0.7:
                return True, 'similar'
                
        except Exception as e:
            print(f"Warning: Error during similarity calculation: {e}")
            
        return False, None

    def detect_duplicates(self, texts: List[str], domain: str = None) -> List[Tuple[bool, Optional[str]]]:
        """
        Returns a list indicating which texts should be filtered out.
        Now properly distinguishes between exact and similar matches.
        """
        results = []
        for i, text in enumerate(texts):
            should_filter = False
            filter_type = None
            
            # Check against all previous texts
            for j in range(i):
                should_filter, detected_type = self._should_filter(texts[j], text, domain)
                if should_filter:
                    filter_type = detected_type
                    break
                    
            results.append((should_filter, filter_type))
        
        return results