from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path
from ..config.logging_config import setup_logging

logger = setup_logging()

class SimpleEvaluator:
    """
    Simple evaluator specifically for comparing a LoRA-tuned model with its base model.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 
                               'cuda' if torch.cuda.is_available() else 
                               'cpu')
        logger.info(f"Initialized SimpleEvaluator using device: {self.device}")

    def evaluate_pair(self, 
                     lora_model_path: str,
                     test_dataset_path: str) -> pd.DataFrame:
        """
        Compare LoRA-tuned model with its base model
        
        Args:
            lora_model_path: Path to LoRA model directory
            test_dataset_path: Path to HuggingFace dataset
        """
        lora_path = Path(lora_model_path)
        
        # Load adapter config to get base model
        with open(lora_path / "adapter_config.json", "r") as f:
            import json
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        
        logger.info(f"Comparing models:")
        logger.info(f"- LoRA model: {lora_path}")
        logger.info(f"- Base model: {base_model_name}")
        
        # Evaluate both models
        results = []
        
        # 1. Evaluate base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3
        ).to(self.device)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_metrics = self._evaluate_single_model(
            base_model, base_tokenizer, test_dataset_path
        )
        results.append({
            'model_type': 'base',
            'model_name': base_model_name,
            **base_metrics
        })
        
        # 2. Evaluate LoRA model
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        lora_tokenizer = AutoTokenizer.from_pretrained(lora_path)
        lora_metrics = self._evaluate_single_model(
            lora_model, lora_tokenizer, test_dataset_path
        )
        results.append({
            'model_type': 'lora',
            'model_name': lora_path.name,
            **lora_metrics
        })
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Log comparison
        logger.info("\nModel Comparison Results:")
        logger.info("="*50)
        logger.info(df.to_string())
        logger.info("="*50)
        
        return df
    
    def _evaluate_single_model(self, 
                             model: torch.nn.Module,
                             tokenizer: AutoTokenizer,
                             dataset_path: str) -> Dict:
        """Internal method to evaluate a single model"""
        model = model.to(self.device)
        model.eval()
        
        # Load test dataset
        dataset = load_from_disk(dataset_path)
        test_data = dataset['test']
        
        # Tokenize
        encoded_dataset = test_data.map(
            lambda examples: tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None
            ),
            batched=True,
            remove_columns=['text', 'id']
        )
        
        encoded_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        # Evaluate
        from torch.utils.data import DataLoader
        dataloader = DataLoader(encoded_dataset, batch_size=8)
        
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions.extend(outputs.logits.argmax(-1).cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        return {
            'accuracy': accuracy_score(labels, predictions),
            'macro_f1': f1_score(labels, predictions, average='macro'),
            'per_class_f1': f1_score(labels, predictions, average=None).tolist(),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }


# Usage example:
"""
evaluator = SimpleEvaluator()
results = evaluator.evaluate_pair(
    lora_model_path="src/models/storage/deberta-v3-base/lora/three_way/20250105_115544",
    test_dataset_path="src/data/datasets/ecommerce_7k_20250105_015001_REAL"
)
""" 