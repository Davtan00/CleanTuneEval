from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import torch
import logging
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TextValidator:
    """
    Handles text-based validation, including duplicate detection
    and optional complexity checks.
    Used by DataProcessor for real-time filtering.
    """
    def __init__(self, config):
        self.config = config
        logger.info("Initializing TextValidator with SentenceTransformer...")

        try:
            # Load sentence-transformers model
            self.model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

            # Decide device usage (CPU, CUDA, or MPS)
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

        # Duplicate detection thresholds
        self.exact_threshold = 0.999
        self.similarity_threshold = 0.92

    def detect_duplicates(self, texts: List[str]) -> List[Tuple[bool, Optional[str]]]:
        """
        Detect duplicates within a batch of texts.
        Returns a list of (bool, str) pairs:
          - bool: whether the text should be filtered out
          - str: 'exact' or 'similar' if we found a duplicate
        Uses a combination of exact matching & semantic similarity.
        """
        try:
            if not texts:
                logger.warning("Empty text list provided to detect_duplicates")
                return []

            logger.info(f"Running duplicate detection on {len(texts)} texts")

            results = []
            normalized_texts = [' '.join(txt.lower().strip().split()) for txt in texts]
            seen_texts = {}
            chunk_size = 100  # user-specified chunk size for embeddings

            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                chunk_results = []

                try:
                    # Compute embeddings for the entire chunk at once
                    embeddings = self.model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)

                    for j, text in enumerate(chunk):
                        should_filter = False
                        filter_type = None
                        norm_text = normalized_texts[i + j]

                        # 1) Exact check
                        if norm_text in seen_texts:
                            should_filter, filter_type = True, 'exact'
                        else:
                            seen_texts[norm_text] = i + j
                            # 2) Semantic similarity check (only if not exact)
                            if not should_filter:
                                for k in range(j):
                                    similarity = float(torch.dot(embeddings[j], embeddings[k]))
                                    if similarity >= self.similarity_threshold:
                                        should_filter, filter_type = True, 'similar'
                                        break

                        chunk_results.append((should_filter, filter_type))

                except Exception as e:
                    logger.error(f"Error in chunk duplicate detection: {str(e)}")
                    # If chunk fails, assume no duplicates for this portion
                    chunk_results.extend([(False, None)] * len(chunk))

                results.extend(chunk_results)
                logger.debug(f"Processed chunk {i // chunk_size + 1}: found {sum(1 for r in chunk_results if r[0])} duplicates")

            return results

        except Exception as e:
            logger.error(f"Error in detect_duplicates: {str(e)}")
            return [(False, None)] * len(texts)

    def compute_vocabulary_richness(self, text: str) -> float:
        """
        Calculates the ratio of unique words to total words.
        """
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / total_words

    def check_cross_split_similarity(
        self,
        splits: DatasetDict,
        threshold: float = 0.98
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Example of a higher-level check to detect near-duplicates across dataset splits.
        Provided as a utility if needed.
        """
        results = {}
        split_pairs = [('train', 'validation'), ('train', 'test'), ('validation', 'test')]

        for split1, split2 in split_pairs:
            texts1 = splits[split1]['text']
            texts2 = splits[split2]['text']

            emb1 = self.model.encode(texts1, convert_to_tensor=True, normalize_embeddings=True)
            emb2 = self.model.encode(texts2, convert_to_tensor=True, normalize_embeddings=True)

            similarities = []
            for i, e1 in enumerate(emb1):
                for j, e2 in enumerate(emb2):
                    sim = float(torch.dot(e1, e2))
                    if sim > threshold:
                        similarities.append((texts1[i], texts2[j], sim))

            results[f"{split1}-{split2}"] = similarities

        return results

    def validate_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Example method returning additional metrics, such as average sentence length.
        Currently used by DataProcessor only if needed.
        """
        vocab_richness = self.compute_vocabulary_richness(text)
        sentences = text.split('.')
        lengths = [len(s.strip().split()) for s in sentences if s.strip()]

        if not lengths:
            return {
                'vocabulary_richness': vocab_richness,
                'avg_sentence_length': 0.0,
                'sentence_length_variation': 0.0
            }

        avg_len = float(np.mean(lengths))
        std_dev = float(np.std(lengths))

        return {
            'vocabulary_richness': vocab_richness,
            'avg_sentence_length': avg_len,
            'sentence_length_variation': std_dev
        }

@dataclass
class ValidationMetrics:
    """
    Holds counters for length-filtered, duplicates, etc.
    The pipeline finalizes 'total_removed' based on these values.
    """
    total_processed: int = 0
    length_filtered: int = 0
    duplicates_removed: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    total_removed: int = 0

    def finalize(self) -> None:
        """
        Recomputes 'total_removed' in case length or duplicates changed.
        """
        self.total_removed = self.length_filtered + self.duplicates_removed

    def to_dict(self) -> Dict[str, Any]:
        """
        Provides a dictionary representation with final metrics updated.
        """
        self.finalize()
        return {
            'total_processed': self.total_processed,
            'length_filtered': self.length_filtered,
            'duplicates_removed': self.duplicates_removed,
            'exact_duplicates': self.exact_duplicates,
            'near_duplicates': self.near_duplicates,
            'total_removed': self.total_removed
        }
