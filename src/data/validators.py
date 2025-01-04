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
    
    def detect_duplicates(self, texts: List[str], threshold: float = 0.85) -> List[bool]:
        embeddings = self.sentence_model.encode(texts)
        similarity_matrix = np.inner(embeddings, embeddings)
        # Mark as duplicate if similarity exceeds threshold (excluding self-similarity)
        duplicates = []
        for i, row in enumerate(similarity_matrix):
            is_duplicate = any(sim > threshold and j != i for j, sim in enumerate(row))
            duplicates.append(is_duplicate)
        return duplicates 