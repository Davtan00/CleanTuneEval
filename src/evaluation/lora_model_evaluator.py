import os
import json
from pathlib import Path
import logging
import torch
from datasets import load_from_disk
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, Any, Optional
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from src.config.environment import HardwareConfig
from src.models.deberta_trainer import DebertaTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoraModelEvaluator:
    def __init__(
        self,
        base_model_name: str,
        checkpoint_path: str,
        dataset_path: str,
        hardware_config: Optional[HardwareConfig] = None
    ):
        """
        Initialize LoRA model evaluator.
        
        Args:
            base_model_name: Name of base model from HuggingFace (e.g. "microsoft/deberta-v3-base")
            checkpoint_path: Path to checkpoint directory containing LoRA adapter
            dataset_path: Path to evaluation dataset
            hardware_config: Optional HardwareConfig object
        """
        self.base_model_name = base_model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_path = dataset_path
        self.hardware_config = hardware_config or HardwareConfig(force_cpu=False)
        self.device = self.hardware_config.device
        
        # Load dataset
        self.dataset = load_from_disk(dataset_path)
        self.test_dataset = self.dataset["test"]
        
        # Standard label mapping
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize tokenizer and models
        self._setup_tokenizer()
        self._setup_model()

    def _setup_tokenizer(self):
        """Initialize and configure tokenizer."""
        logger.info(f"Loading tokenizer from {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        # Ensure proper padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _setup_model(self):
        """Load base model and apply LoRA adapter."""
        logger.info(f"Loading base model from {self.base_model_name}")
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,
            label2id=self.label2id,
            id2label=self.id2label,
            trust_remote_code=True
        )
        
        # Load and apply LoRA adapter
        logger.info(f"Loading LoRA adapter from {self.checkpoint_path}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.checkpoint_path,
            is_trainable=False
        )
        
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_dataset(self):
        """Preprocess test dataset with tokenization."""
        def tokenize_fn(example):
            # Basic tokenization without tensor conversion
            tokenized = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors=None  # Let set_format handle torch conversion
            )
            # Convert label to integer
            label_id = self.label2id[example["labels"]]
            tokenized["labels"] = label_id
            return tokenized

        # Process dataset
        processed_dataset = self.test_dataset.map(
            tokenize_fn,
            remove_columns=self.test_dataset.column_names,  # Remove all original columns
            desc="Tokenizing test dataset"
        )

        # Set format for PyTorch
        processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return processed_dataset

    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics including combined score."""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        
        # Calculate metrics
        precision = np.mean([
            np.mean(predictions[labels == i] == i) 
            for i in range(3)
        ])
        recall = np.mean([
            np.mean(labels[predictions == i] == i) 
            for i in range(3)
        ])
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = np.mean(predictions == labels)
        
        # Additional metrics
        balanced_acc = balanced_accuracy_score(labels, predictions)
        matthews = matthews_corrcoef(labels, predictions)
        
        # Combined metric (matching TaskSpec.MD)
        combined_metric = 0.7 * f1 + 0.3 * accuracy
        
        return {
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
            "eval_accuracy": accuracy,
            "eval_balanced_accuracy": balanced_acc,
            "eval_matthews_correlation": matthews,
            "eval_combined_metric": combined_metric
        }

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation and return metrics with metadata."""
        processed_dataset = self._preprocess_dataset()
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="eval_tmp_dir",
                per_device_eval_batch_size=16,
                do_train=False,
                do_eval=True,
                evaluation_strategy="no"
            ),
            eval_dataset=processed_dataset,
            compute_metrics=self._compute_metrics
        )
        
        # Run evaluation
        metrics = trainer.evaluate()
        
        # Add metadata
        evaluation_info = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "base_model": self.base_model_name,
                "lora_adapter_path": str(self.checkpoint_path),
                "dataset_path": self.dataset_path,
                "dataset_hash": hashlib.sha256(
                    str(self.test_dataset[:1000]).encode()
                ).hexdigest(),
                "hardware_info": {
                    "device": str(self.device),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
                }
            },
            "metrics": metrics
        }
        
        return evaluation_info

def main():
    """Example usage of LoraModelEvaluator."""
    # Example paths - adjust as needed
    base_model = "microsoft/deberta-v3-base"
    checkpoint_path = "src/models/storage/deberta-v3-base/lora/three_way/checkpoint-2004"
    dataset_path = "src/data/datasets/technology_7k_20250105_152324"
    
    evaluator = LoraModelEvaluator(
        base_model_name=base_model,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path
    )
    
    results = evaluator.evaluate()
    
    # Save results
    output_dir = Path("src/evaluation/results/lora_evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"lora_evaluation_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    # Print key metrics
    print("\n=== Evaluation Results ===")
    print(f"Combined Metric: {results['metrics']['eval_combined_metric']:.4f}")
    print(f"F1 Score: {results['metrics']['eval_f1']:.4f}")
    print(f"Balanced Accuracy: {results['metrics']['eval_balanced_accuracy']:.4f}")

if __name__ == "__main__":
    main()