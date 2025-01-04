from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..config.logging_config import setup_logging

logger = setup_logging()

class DatasetManager:
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DatasetManager at {self.base_path}")

    def create_dataset(
        self,
        data: Dict,
        domain: str,
        split_ratios: Dict[str, float] = {"train": 0.7, "validation": 0.15, "test": 0.15}, # TODO: make this configurable, perhaps also try the split of the previous repo
        format_type: str = "sentiment_classification"
    ) -> DatasetDict:
        """
        Create a dataset with flexible formatting for different model types
        """
        logger.info(f"Creating dataset for domain: {domain}")
        
        reviews_df = pd.DataFrame(data['generated_data'])
        
        # Format data according to task type
        # Perhaps we should have had "label" instead of "sentiment" from the start
        if format_type == "sentiment_classification":
            dataset_dict = {
                'text': reviews_df['clean_text'].tolist(),
                'labels': reviews_df['sentiment'].tolist(),
                'id': reviews_df['id'].tolist()
            }
        elif format_type == "sequence_classification":
            # Different format for other classification models
            dataset_dict = {
                'input_text': reviews_df['clean_text'].tolist(),
                'label': reviews_df['sentiment'].tolist(),
                'id': reviews_df['id'].tolist()
            }
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        # Create initial dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Create splits
        splits = self._create_splits(dataset, split_ratios)
        
        # Save dataset
        save_path = self.base_path / domain
        splits.save_to_disk(save_path)
        logger.info(f"Saved dataset to {save_path}")
        
        return splits

    def _create_splits(
        self,
        dataset: Dataset,
        split_ratios: Dict[str, float]
    ) -> DatasetDict:
        """
        Create train/validation/test splits with flexible ratios
        """
        # Validate ratios
        total = sum(split_ratios.values())
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1, got {total}")

        # Create initial train/test split
        test_size = split_ratios['test']
        train_val = dataset.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=42
        )

        # If we have a validation split
        if 'validation' in split_ratios:
            # Calculate validation size relative to remaining data
            remaining_data_ratio = 1 - test_size
            val_size = split_ratios['validation'] / remaining_data_ratio
            
            # Split training data into train/validation
            train_val_split = train_val['train'].train_test_split(
                test_size=val_size,
                shuffle=True,
                seed=42
            )
            
            splits = DatasetDict({
                'train': train_val_split['train'],
                'validation': train_val_split['test'],
                'test': train_val['test']
            })
        else:
            splits = DatasetDict({
                'train': train_val['train'],
                'test': train_val['test']
            })

        logger.info(f"Created splits with sizes: {splits}")
        return splits

    def load_dataset(
        self,
        domain: str,
        model_type: str = "sentiment_classification"
    ) -> DatasetDict:
        """
        Load dataset with optional reformatting for specific models
        """
        dataset_path = self.base_path / domain
        if not dataset_path.exists():
            raise FileNotFoundError(f"No dataset found for domain: {domain}")
        
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