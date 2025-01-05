from typing import List, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Dict, Any
import torch
import logging
from datasets import DatasetDict

logger = logging.getLogger(__name__)

class TextValidator:
    def __init__(self, config):
        self.config = config
        logger.info("Initializing TextValidator with Sentence Transformer...")
        try:
            self.model = SentenceTransformer('paraphrase-mpnet-base-v2')
            if self.config.use_mps:
                logger.info("Attempting to use MPS acceleration...")
                self.model = self.model.to('mps')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        # Simplified thresholds without domain specifics
        self.exact_threshold = 0.999
        self.similarity_threshold = 0.92  # Single threshold for all duplicates

    def _should_filter(self, text1: str, text2: str) -> Tuple[bool, Optional[str]]:
        """Simplified duplicate detection without domain specifics"""
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
            len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
            
            if similarity >= self.similarity_threshold and len_ratio > 0.7:
                return True, 'similar'
                
        except Exception as e:
            logger.error(f"Error during similarity calculation: {e}")
            
        return False, None

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
                    # Get embeddings for chunk
                    embeddings = self.model.encode(
                        chunk,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    
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
                    logger.error(f"Error processing chunk {i//chunk_size}: {str(e)}")
                    # Return safe results for this chunk
                    chunk_results.extend([(False, None)] * len(chunk))
                    
                results.extend(chunk_results)
                logger.debug(f"Processed chunk {i//chunk_size + 1}, found {sum(1 for r in chunk_results if r[0])} duplicates")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in detect_duplicates: {str(e)}")
            # Return safe default results
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
    # Add these fields for quality metrics
    perplexity: float = 0.0
    similarity_score: float = 0.0
    vocabulary_richness: float = 0.0
    is_outlier: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'total_processed': self.total_processed,
            'length_filtered': self.length_filtered,
            'duplicates_removed': self.duplicates_removed,
            'exact_duplicates': self.exact_duplicates,
            'near_duplicates': self.near_duplicates,
            'total_removed': self.total_removed
        }    
    def update_duplicate_counts(self, is_duplicate: bool, duplicate_type: str = None) -> None:
        """Update duplicate-related counters."""
        if is_duplicate:
            self.duplicates_removed += 1
            if duplicate_type == 'exact':
                self.exact_duplicates += 1
            elif duplicate_type == 'similar':
                self.near_duplicates += 1
                
    def finalize(self) -> None:
        """Calculate final metrics."""
        self.total_removed = self.length_filtered + self.duplicates_removed
