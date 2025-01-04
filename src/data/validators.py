from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
import re
from collections import defaultdict
from nltk.util import ngrams
from ..config.environment import HardwareConfig
from ..config.validation_config import DUPLICATE_CONFIG, get_domain_config

@dataclass
class ValidationMetrics:
    perplexity: float
    similarity_score: float
    vocabulary_richness: float
    is_outlier: bool
    duplicate_type: str = None  # 'exact', 'ngram', 'semantic', or None

class TextValidator:
    def __init__(self, hardware_config: HardwareConfig):
        self.config = DUPLICATE_CONFIG
        self.hardware_config = hardware_config
        
        # Initialize SentenceBERT if semantic similarity is enabled
        if self.config['semantic']['enabled']:
            self.sentence_model = SentenceTransformer(self.config['semantic']['model_name'])
            if hardware_config.use_mps:
                self.sentence_model = self.sentence_model.to(hardware_config.device)
        
        self.outlier_detector = IsolationForest(contamination=0.1)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.config['exact_match']['normalize_case']:
            text = text.lower()
        if self.config['exact_match']['strip_punctuation']:
            text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())
    
    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Generate n-grams from text."""
        tokens = self._normalize_text(text).split()
        return set(''.join(gram) for gram in ngrams(tokens, n))
    
    def _calculate_ngram_similarity(self, text1: str, text2: str) -> float:
        """Calculate n-gram based similarity between two texts."""
        n = self.config['ngram']['n']
        ngrams1 = self._get_ngrams(text1, n)
        ngrams2 = self._get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union)
    
    def _calculate_semantic_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate semantic similarity matrix using SentenceBERT embeddings."""
        return np.inner(embeddings, embeddings)
    
    def detect_duplicates(self, texts: List[str], domain: str = 'general') -> List[Tuple[bool, str]]:
        """Enhanced duplicate detection using multiple methods.
        
        Args:
            texts: List of review texts to check
            domain: Domain of the reviews for domain-specific thresholds
            
        Returns:
            List of tuples (is_duplicate, duplicate_type)
        """
        # Get domain-specific configuration
        domain_config = get_domain_config(domain)
        threshold_modifier = domain_config['duplicate_threshold_modifier']
        
        results = []
        normalized_texts = [self._normalize_text(text) for text in texts]
        
        # Step 1: Exact matching
        if self.config['exact_match']['enabled']:
            seen_texts = set()
            for text in normalized_texts:
                if text in seen_texts:
                    results.append((True, 'exact'))
                else:
                    seen_texts.add(text)
                    results.append((False, None))
        else:
            results = [(False, None) for _ in texts]
        
        # Step 2: N-gram similarity (only for non-exact duplicates)
        if self.config['ngram']['enabled']:
            ngram_threshold = self.config['ngram']['threshold'] * threshold_modifier
            for i in range(len(texts)):
                if not results[i][0]:  # Only if not already marked as duplicate
                    for j in range(i):
                        if not results[j][0]:  # Only compare with non-duplicates
                            sim = self._calculate_ngram_similarity(texts[i], texts[j])
                            if sim > ngram_threshold:
                                results[i] = (True, 'ngram')
                                break
        
        # Step 3: Semantic similarity (only for remaining non-duplicates)
        if self.config['semantic']['enabled']:
            semantic_threshold = self.config['semantic']['threshold'] * threshold_modifier
            non_dup_indices = [i for i, (is_dup, _) in enumerate(results) if not is_dup]
            
            if non_dup_indices:
                non_dup_texts = [texts[i] for i in non_dup_indices]
                embeddings = self.sentence_model.encode(non_dup_texts)
                similarity_matrix = self._calculate_semantic_similarity(embeddings)
                
                for idx, i in enumerate(non_dup_indices):
                    for j_idx, j in enumerate(non_dup_indices[:idx]):
                        if similarity_matrix[idx, j_idx] > semantic_threshold:
                            # Check length ratio
                            len_ratio = (len(texts[i]) / len(texts[j]) 
                                       if len(texts[j]) > len(texts[i])
                                       else len(texts[j]) / len(texts[i]))
                            
                            if len_ratio > self.config['semantic']['min_length_ratio']:
                                results[i] = (True, 'semantic')
                                break
        
        return results
    
    def compute_vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary richness score."""
        words = self._normalize_text(text).split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0