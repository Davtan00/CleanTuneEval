import os
import json
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from typing import Dict, Optional

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebertaTrainer:
    def __init__(
        self,
        dataset_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        lora_config: Optional[Dict] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        
        # Extract dataset ID from the directory name itself, not its parent
        # APPLY THIS TO ALL FILES THAT USE THIS LOGIC
        self.dataset_id = self.dataset_path.name
        logger.info(f"Extracted dataset ID: {self.dataset_id}")
        
        # Create a timestamp-based output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("src/models/storage/deberta-v3-base/lora/three_way") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load any stored dataset metrics (optional)
        self.dataset_metrics = self._load_dataset_metrics()
        
        # Default or custom LoRA config
        self.lora_config = lora_config or LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )

    def _load_dataset_metrics(self) -> Dict:
        """Load the metrics from Part A processing for reference."""
        metrics_path = Path("src/data/storage/metrics") / f"{self.dataset_id}_metrics.json"
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"No metrics found at {metrics_path}")
            return {}

    def _setup_training_args(self) -> TrainingArguments:
        dataset = load_from_disk(self.dataset_path)
        train_size = len(dataset['train'])
        logger.info(f"Training set size: {train_size} samples")
        
        # Adjust batch size based on environment
        batch_size = 16 if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 8
        
        return TrainingArguments(
            output_dir=str(self.output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            warmup_ratio=0.06,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            # Disable wandb reporting
            report_to="none"
        )

    def _initialize_tokenizer(self):
        """Initialize tokenizer specifically for DeBERTa v3."""
        logger.info(f"Initializing tokenizer for {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=512,
                use_fast=True,
                trust_remote_code=True
            )
            if not any(name in str(type(tokenizer)).lower() for name in ['deberta', 'sentencepiece']):
                raise ValueError("Loaded tokenizer is not DeBERTa/SentencePiece-based")
            return tokenizer
        except Exception as first_error:
            logger.warning(f"Failed to load fast tokenizer: {first_error}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    trust_remote_code=True
                )
                return tokenizer
            except Exception as e:
                logger.error("Failed to initialize tokenizer with both attempts")
                raise RuntimeError(
                    f"DeBERTa tokenizer initialization failed.\nOriginal error: {str(e)}"
                )

    def _preprocess_data(self, dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
        """Tokenize text and set torch format."""
        def tokenize_function(example):
            return tokenizer(
                example["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=128
            )
        for split in dataset_dict.keys():
            dataset_dict[split] = dataset_dict[split].map(tokenize_function, batched=True)
            dataset_dict[split].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"]
            )
        return dataset_dict

    def _setup_model(self):
        """Load base DeBERTa model and inject LoRA adapters."""
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        lora_model = get_peft_model(base_model, self.lora_config)
        return lora_model

    def _compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """Compute accuracy, precision, recall, and macro-F1."""
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def train(self) -> Dict:
        """Execute the full training pipeline."""
        try:
            # 1. Load dataset
            dataset_dict = load_from_disk(self.dataset_path)
            logger.info(f"Loaded dataset splits: {dataset_dict.keys()}")
            
            # 2. Initialize tokenizer
            tokenizer = self._initialize_tokenizer()
            
            # 3. Preprocess data
            dataset_dict = self._preprocess_data(dataset_dict, tokenizer)
            
            # 4. Setup model
            model = self._setup_model()
            
            # 5. Prepare trainer
            trainer_args = self._setup_training_args()
            trainer = Trainer(
                model=model,
                args=trainer_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                compute_metrics=self._compute_metrics
            )
            
            # 6. Train
            train_result = trainer.train()
            
            # 7. Evaluate on test set
            test_metrics = trainer.evaluate(dataset_dict["test"])
            
            # 8. Save training metadata
            self._save_training_results(train_result, test_metrics)
            
            return {
                "status": "success",
                "test_metrics": test_metrics,
                "model_path": str(self.output_dir)
            }
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _save_training_results(self, train_result, test_metrics):
        """Save training metadata and final metrics."""
        metadata = {
            "dataset_info": {
                "id": self.dataset_id,
                "metrics": self.dataset_metrics
            },
            "training_config": {
                "model_name": self.model_name,
                "lora_config": self.lora_config.__dict__ if hasattr(self.lora_config, '__dict__') else self.lora_config,
                "device": str(self._get_device())
            },
            "results": {
                "train": train_result.metrics,
                "test": test_metrics
            }
        }
        
        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
