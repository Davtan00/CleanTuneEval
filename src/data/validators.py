from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from ..config.environment import HardwareConfig

@dataclass
class ValidationMetrics:
    perplexity: float
    similarity_score: float
    vocabulary_richness: float
    is_outlier: bool

class TextValidator:
    def __init__(self, hardware_config: HardwareConfig):
        # Initialize SentenceBERT for similarity detection
        # Legit just copy what we had done  in the review quality backend repo, should be on davtan00 private
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        if hardware_config.use_mps:
            self.sentence_model = self.sentence_model.to(hardware_config.device)
        
        self.outlier_detector = IsolationForest(contamination=0.1)
        
    def compute_vocabulary_richness(self, text: str) -> float:
        words = text.split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0
    
    def detect_duplicates(self, texts: List[str], threshold: float = 0.95) -> List[bool]:
        """Detect near-duplicate reviews using semantic similarity.
        
        Args:
            texts: List of review texts to check
            threshold: Similarity threshold (default: 0.95). Higher means more lenient.
                      Only extremely similar reviews will be marked as duplicates.
        """
        embeddings = self.sentence_model.encode(texts)
        similarity_matrix = np.inner(embeddings, embeddings)
        
        # Consider both semantic similarity and length ratio
        duplicates = []
        for i, (row, text1) in enumerate(zip(similarity_matrix, texts)):
            is_duplicate = False
            for j, (sim, text2) in enumerate(zip(row, texts)):
                if i != j and sim > threshold:
                    # Additional check: length ratio
                    len_ratio = len(text1) / len(text2) if len(text2) > len(text1) else len(text2) / len(text1)
                    if len_ratio > 0.8:  # Only mark as duplicate if lengths are also similar
                        is_duplicate = True
                        break
            duplicates.append(is_duplicate)
        return duplicates