from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..config.logging_config import setup_logging
import numpy as np

logger = setup_logging()

class DatasetManager:
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
        Create a dataset with flexible formatting for different model types
        """
        logger.info(f"Creating dataset with ID: {dataset_id}")
        
        reviews_df = pd.DataFrame(data['generated_data'])
        
        # Shuffle before creating dataset to break sentiment blocks
        reviews_df = reviews_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Format data according to task type
        if format_type == "sentiment_classification":
            dataset_dict = {
                'text': reviews_df['clean_text'].tolist(),
                'labels': reviews_df['sentiment'].tolist(),
                'id': reviews_df['id'].tolist()
            }
        elif format_type == "sequence_classification":
            dataset_dict = {
                'input_text': reviews_df['clean_text'].tolist(),
                'label': reviews_df['sentiment'].tolist(),
                'id': reviews_df['id'].tolist()
            }
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        # Create dataset with stratified splits
        dataset = Dataset.from_dict(dataset_dict)
        splits = self._create_stratified_splits(dataset, split_ratios)
        
        # Save dataset
        save_path = self.base_path / dataset_id
        splits.save_to_disk(save_path)
        logger.info(f"Saved dataset to {save_path}")
        
        # Call verification once here
        self._verify_label_distribution(splits)
        
        return splits

    def _create_stratified_splits(
        self,
        dataset: Dataset,
        split_ratios: Dict[str, float]
    ) -> DatasetDict:
        """Create stratified splits maintaining sentiment distribution"""
        # Get labels for stratification
        labels = dataset['labels']
        
        # Create stratified splits
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=split_ratios['test'],
            stratify=labels,
            random_state=42
        )
        
        if 'validation' in split_ratios:
            # Further split train into train/val
            train_labels = [labels[i] for i in train_idx]
            train_final_idx, val_idx = train_test_split(
                train_idx,
                test_size=split_ratios['validation']/(1-split_ratios['test']),
                stratify=train_labels,
                random_state=42
            )
            
            return DatasetDict({
                'train': dataset.select(train_final_idx),
                'validation': dataset.select(val_idx),
                'test': dataset.select(test_idx)
            })
        else:
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
        Load dataset with optional reformatting for specific models
        """
        dataset_path = self.base_path / dataset_id
        if not dataset_path.exists():
            raise FileNotFoundError(f"No dataset found with ID: {dataset_id}")
        
        dataset = DatasetDict.load_from_disk(dataset_path)
        
        # Reformat for specific model types if needed
        if model_type != "sentiment_classification":
            dataset = self._reformat_for_model(dataset, model_type)
        
        return dataset

    def _reformat_for_model(
        self,
        dataset: DatasetDict,
        model_type: str
    ) -> DatasetDict:
        """
        Reformat dataset for different model types
        So far we have roberta and bert
        TODO: add other models
        """
        if model_type == "roberta":
            # Format specifically for RoBERTa
            return DatasetDict({
                split: dataset[split].rename_column('text', 'input_text')
                for split in dataset.keys()
            })
        elif model_type == "bert":
            # Format specifically for BERT
            return dataset  # BERT uses the default format
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 

    def _verify_splits(self, splits: DatasetDict) -> bool:
        """Verify no overlap between splits"""
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
                logger.warning(f"Found {overlap} overlapping IDs between {split_pair}")
                return False
        return True 

    def _verify_label_distribution(self, splits: DatasetDict) -> None:
        """Log label distribution for each split"""
        for split_name, split in splits.items():
            labels = split['labels']
            unique, counts = np.unique(labels, return_counts=True)
            dist = dict(zip(unique, counts))
            percentages = {k: v/len(labels)*100 for k, v in dist.items()}
            logger.info(f"{split_name} distribution: {percentages}")
        
    def verify_dataset_structure(self, dataset_path: str):
        """Verify dataset structure after creation"""
        dataset = load_from_disk(dataset_path)
        for split in dataset.keys():
            logger.info(f"Split {split}:")
            logger.info(f"- Size: {len(dataset[split])}")
            logger.info(f"- Features: {dataset[split].features}")
            logger.info(f"- First few IDs: {dataset[split]['id'][:5]}")
        
        