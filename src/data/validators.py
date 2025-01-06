from typing import List, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Dict, Any
import torch
import logging
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TextValidator:
    def __init__(self, config):
        self.config = config
        logger.info("Initializing TextValidator with SentenceTransformer...")
        
        try:
            # Initialize SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
            
            # Handle device placement
            if self.config.use_mps and torch.backends.mps.is_available():
                logger.info("Using MPS acceleration...")
                self.model.to('mps')
            elif torch.cuda.is_available() and not self.config.force_cpu:
                logger.info("Using CUDA acceleration...")
                self.model.to('cuda')
            else:
                logger.info("Using CPU processing...")
                self.model.to('cpu')
                
            logger.info("SentenceTransformer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        # Thresholds remain the same
        self.exact_threshold = 0.999
        self.similarity_threshold = 0.92

    def detect_duplicates(self, texts: List[str]) -> List[Tuple[bool, Optional[str]]]:
        """Detect duplicates with enhanced error handling and logging."""
        try:
            if not texts:
                logger.warning("Empty text list provided to detect_duplicates")
                return []
                
            logger.info(f"Processing batch of {len(texts)} texts")
            results = []
            
            # First pass: exact matches (fast)
            normalized_texts = [' '.join(text.lower().strip().split()) for text in texts]
            seen_texts = {}
            
            # Process texts in smaller chunks for similarity detection
            chunk_size = 100  # Process 100 texts at a time
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                chunk_results = []
                
                try:
                    # Get embeddings for entire chunk at once
                    embeddings = self.model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
                    
                    # Process each text in chunk
                    for j, text in enumerate(chunk):
                        should_filter = False
                        filter_type = None
                        
                        # Check exact duplicates
                        norm_text = normalized_texts[i + j]
                        if norm_text in seen_texts:
                            should_filter, filter_type = True, 'exact'
                        else:
                            seen_texts[norm_text] = i + j
                            
                            # Check semantic similarity only if not exact duplicate
                            if not should_filter:
                                # Compare with previous texts in chunk
                                for k in range(j):
                                    similarity = float(torch.dot(embeddings[j], embeddings[k]))
                                    if similarity >= self.similarity_threshold:
                                        should_filter, filter_type = True, 'similar'
                                        break
                                        
                        chunk_results.append((should_filter, filter_type))
                        
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    chunk_results.extend([(False, None)] * len(chunk))
                    
                results.extend(chunk_results)
                logger.debug(f"Processed chunk {i//chunk_size + 1}, found {sum(1 for r in chunk_results if r[0])} duplicates")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in detect_duplicates: {str(e)}")
            return [(False, None)] * len(texts)

    def compute_vocabulary_richness(self, text: str) -> float:
        """Compute vocabulary richness score for a text."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Calculate unique words ratio
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words

    def check_cross_split_similarity(
        self,
        splits: DatasetDict,
        threshold: float = 0.98
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Check for near-duplicates across splits"""
        results = {}
        split_pairs = [('train', 'validation'), ('train', 'test'), ('validation', 'test')]
        
        for split1, split2 in split_pairs:
            texts1 = splits[split1]['text']
            texts2 = splits[split2]['text']
            
            # Use existing embeddings functionality
            embeddings1 = self.model.encode(texts1, convert_to_tensor=True, normalize_embeddings=True)
            embeddings2 = self.model.encode(texts2, convert_to_tensor=True, normalize_embeddings=True)
            
            similarities = []
            for i, emb1 in enumerate(embeddings1):
                for j, emb2 in enumerate(embeddings2):
                    sim = float(torch.dot(emb1, emb2))
                    if sim > threshold:
                        similarities.append((texts1[i], texts2[j], sim))
            
            results[f"{split1}-{split2}"] = similarities
        
        return results

    def validate_text_complexity(self, text: str) -> Dict[str, float]:
        """Validate text complexity using multiple metrics"""
        # Existing vocabulary richness
        vocab_richness = self.compute_vocabulary_richness(text)
        
        # Add sentence structure variety
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        sentence_length_std = np.std([len(s.split()) for s in sentences if s.strip()])
        
        return {
            'vocabulary_richness': vocab_richness,
            'avg_sentence_length': avg_sentence_length,
            'sentence_length_variation': sentence_length_std
        }

@dataclass
class ValidationMetrics:
    """Metrics collected during the validation process."""
    total_processed: int = 0
    length_filtered: int = 0
    duplicates_removed: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    total_removed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        self.finalize()  # Ensure metrics are finalized before conversion, encountered way too many weird exceptions
        return {
            'total_processed': self.total_processed,
            'length_filtered': self.length_filtered,
            'duplicates_removed': self.duplicates_removed,
            'exact_duplicates': self.exact_duplicates,
            'near_duplicates': self.near_duplicates,
            'total_removed': self.total_removed
        }
        
    def finalize(self) -> None:
        """Calculate final metrics."""
        self.total_removed = (
            self.length_filtered
            + self.duplicates_removed
        )