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
        
        # Extract dataset ID from path for model storage
        self.dataset_id = self.dataset_path.parent.name
        
        # Setup model storage path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("src/models/storage/deberta-v3-base/lora/three_way") / timestamp
        
        # Load dataset metrics to inform training
        self.dataset_metrics = self._load_dataset_metrics()
        
        # Configure LoRA
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
        """Configure training based on dataset characteristics."""
        # Calculate steps based on dataset size
        dataset = load_from_disk(self.dataset_path)
        train_size = len(dataset['train'])
        
        # Adjust batch size based on available memory
        batch_size = 16 if torch.cuda.is_available() or hasattr(torch.backends, 'mps') else 8
        
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
            save_total_limit=2
        )

    def _initialize_tokenizer(self):
        """Initialize tokenizer specifically for DeBERTa v3."""
        logger.info(f"Initializing tokenizer for {self.model_name}")
        
        try:
            # First attempt: Standard initialization with specific settings
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=512,
                use_fast=True,
                trust_remote_code=True  # Important for DeBERTa v3
            )
            # Verify we got a DeBERTa tokenizer
            if not any(name in str(type(tokenizer)).lower() for name in ['deberta', 'sentencepiece']):
                raise ValueError("Loaded tokenizer is not DeBERTa/SentencePiece-based")
            return tokenizer
        
        except Exception as first_error:
            logger.warning(f"Failed to load fast tokenizer: {first_error}")
            
            try:
                # Second attempt: Try slow tokenizer with minimal settings
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    trust_remote_code=True
                )
                return tokenizer
            
            except Exception as e:
                logger.error("Failed to initialize tokenizer with both attempts")
                raise RuntimeError(
                    "DeBERTa tokenizer initialization failed. Please ensure:\n"
                    "1. Your model path is correct (microsoft/deberta-v3-base)\n"
                    "2. You have a working internet connection\n"
                    "3. Your local cache is not corrupted\n"
                    f"Original error: {str(e)}"
                )

    def train(self) -> Dict:
        """Execute the full training pipeline."""
        try:
            # 1. Load and validate dataset
            dataset_dict = load_from_disk(self.dataset_path)
            logger.info(f"Loaded dataset splits: {dataset_dict.keys()}")
            
            # 2. Initialize tokenizer with robust fallback
            tokenizer = self._initialize_tokenizer()
            
            # 3. Preprocess data
            dataset_dict = self._preprocess_data(dataset_dict, tokenizer)
            
            # 4. Initialize model
            model = self._setup_model()
            
            # 5. Train
            trainer = Trainer(
                model=model,
                args=self._setup_training_args(),
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                compute_metrics=self._compute_metrics
            )
            
            train_result = trainer.train()
            
            # 6. Evaluate on test set
            test_metrics = trainer.evaluate(dataset_dict["test"])
            
            # 7. Save results and metadata
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
        """Save training metadata and metrics."""
        metadata = {
            "dataset_info": {
                "id": self.dataset_id,
                "metrics": self.dataset_metrics
            },
            "training_config": {
                "model_name": self.model_name,
                "lora_config": self.lora_config.__dict__,
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

    