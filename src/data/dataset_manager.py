from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..config.logging_config import setup_logging
import numpy as np

logger = setup_logging()

class DatasetManager:
    """
    Creates and manages Hugging Face datasets from processed data.
    Expects that input data has already been filtered and enumerated in the pipeline.
    Still does a local re-enumeration (id=0..N-1) for the final HF Dataset structure.
    """
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DatasetManager at {self.base_path}")

    def create_dataset(
        self,
        data: Dict,
        dataset_id: str,
        split_ratios: Dict[str, float] = {"train": 0.7, "validation": 0.15, "test": 0.15},
        format_type: str = "sentiment_classification"
    ) -> DatasetDict:
        """
        Main entry for creating an HF dataset.
          - data: Dict with 'generated_data' (already enumerated by pipeline).
          - dataset_id: Unique ID for saving under self.base_path / dataset_id.
          - split_ratios: Ratios for train/val/test.
          - format_type: "sentiment_classification" or other supported type.
        
        Steps:
          1) Convert to a DataFrame and drop any rows marked is_removed (if they exist).
          2) Shuffle data for randomness.
          3) Re-enumerate IDs from 0..N-1, but preserve 'original_id' from pipeline.
          4) Create stratified splits.
          5) Save the dataset to disk and verify.
        """
        logger.info(f"Creating dataset with ID: {dataset_id}")

        # Convert to DataFrame
        reviews_df = pd.DataFrame(data['generated_data'])

        # Filter out any 'is_removed' if present
        if 'is_removed' in reviews_df.columns:
            original_count = len(reviews_df)
            reviews_df = reviews_df[~reviews_df['is_removed']].copy()
            logger.info(f"Filtered out {original_count - len(reviews_df)} removed reviews")

        # Reset index and store pipeline IDs as 'original_id', then create a new local HF ID (0..N-1)
        reviews_df = reviews_df.reset_index(drop=True)
        if 'id' in reviews_df.columns:
            reviews_df['original_id'] = reviews_df['id']
        else:
            reviews_df['original_id'] = range(len(reviews_df))

        reviews_df['id'] = range(len(reviews_df))  # local HF ID

        # Verify re-enumeration
        max_id = reviews_df['id'].max()
        total_reviews = len(reviews_df)
        assert max_id + 1 == total_reviews, \
            f"ID mismatch after enumeration: max_id={max_id}, total_reviews={total_reviews}"

        logger.info(f"Re-enumerated {total_reviews} reviews. "
                    f"Example: original_id={reviews_df['original_id'].iloc[:5].tolist()} -> "
                    f"new_id={reviews_df['id'].iloc[:5].tolist()}")

        # Shuffle before splitting
        reviews_df = reviews_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Format data for HF Dataset
        if format_type == "sentiment_classification":
            dataset_dict = {
                'text': reviews_df['clean_text'].tolist(),
                'labels': reviews_df['sentiment'].tolist(),
                'id': reviews_df['id'].tolist(),
                'original_id': reviews_df['original_id'].tolist()
            }
        elif format_type == "sequence_classification":
            dataset_dict = {
                'input_text': reviews_df['clean_text'].tolist(),
                'label': reviews_df['sentiment'].tolist(),
                'id': reviews_df['id'].tolist(),
                'original_id': reviews_df['original_id'].tolist()
            }
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        dataset = Dataset.from_dict(dataset_dict)

        # Create stratified splits (train, validation, test)
        splits = self._create_stratified_splits(dataset, split_ratios)

        # Save dataset to disk
        save_path = self.base_path / dataset_id
        splits.save_to_disk(save_path)
        logger.info(f"Saved dataset to {save_path}")

        # Optional checks
        self._verify_splits(splits)
        self._verify_label_distribution(splits)

        return splits

    def _create_stratified_splits(
        self,
        dataset: Dataset,
        split_ratios: Dict[str, float]
    ) -> DatasetDict:
        """
        Creates splits while maintaining label distribution if possible.
        Uses scikit-learn's train_test_split with 'stratify'.
        """
        # Extract labels
        labels = dataset['labels']

        # Split into train/test
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=split_ratios['test'],
            stratify=labels,
            random_state=42
        )

        # Further split train into train/val if needed
        if 'validation' in split_ratios:
            train_labels = [labels[i] for i in train_idx]
            val_size = split_ratios['validation'] / (1 - split_ratios['test'])
            train_final_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_size,
                stratify=train_labels,
                random_state=42
            )

            return DatasetDict({
                'train': dataset.select(train_final_idx),
                'validation': dataset.select(val_idx),
                'test': dataset.select(test_idx)
            })

        # If no validation split is provided
        return DatasetDict({
            'train': dataset.select(train_idx),
            'test': dataset.select(test_idx)
        })

    def load_dataset(
        self,
        dataset_id: str,
        model_type: str = "sentiment_classification"
    ) -> DatasetDict:
        """
        Load an already created dataset from disk.
        If 'model_type' is different, reformat as needed.
        """
        dataset_path = self.base_path / dataset_id
        if not dataset_path.exists():
            raise FileNotFoundError(f"No dataset found with ID: {dataset_id}")

        dataset = DatasetDict.load_from_disk(dataset_path)

        # If needed, reformat for specific model usage
        if model_type != "sentiment_classification":
            dataset = self._reformat_for_model(dataset, model_type)

        return dataset

    def _reformat_for_model(
        self,
        dataset: DatasetDict,
        model_type: str
    ) -> DatasetDict:
        """
        Adjust dataset columns for certain model architectures.
        Example:
          - roberta => rename 'text' to 'input_text'
          - bert => keep default
        """
        if model_type == "roberta":
            return DatasetDict({
                split: dataset[split].rename_column('text', 'input_text')
                for split in dataset.keys()
            })
        elif model_type == "bert":
            return dataset
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _verify_splits(self, splits: DatasetDict) -> bool:
        """
        Ensures there are no overlapping IDs between splits.
        Warnings are logged if overlaps exist.
        """
        train_ids = set(splits['train']['id'])
        val_ids = set(splits['validation']['id'])
        test_ids = set(splits['test']['id'])

        overlaps = {
            'train-val': len(train_ids.intersection(val_ids)),
            'train-test': len(train_ids.intersection(test_ids)),
            'val-test': len(val_ids.intersection(test_ids))
        }

        for split_pair, overlap in overlaps.items():
            if overlap > 0:
                logger.warning(f"Found {overlap} overlapping IDs between {split_pair}, which is unusual.")
                return False
        return True

    def _verify_label_distribution(self, splits: DatasetDict) -> None:
        """
        Logs label distribution for each split. 
        Good for double-checking the stratification is working as expected.
        """
        for split_name, split in splits.items():
            labels = split['labels']
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique, counts))
            percentages = {k: v / len(labels) * 100 for k, v in dist.items()}
            logger.info(f"{split_name} distribution: {percentages}")