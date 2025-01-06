# File: /Users/davidtanner/Documents/GitHub/CleanTuneEval/src/models/deberta_trainer.py

import os
import json
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from typing import Dict, Any

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .config.training_utils import TrainingConfigurator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebertaTrainer:
    """
    Expects a dictionary for LoRA config (lora_config) and another for
    training config (training_config). These must be fully specified
    in JSON files, with no CLI override.

    Now also takes an optional `hardware_config`, which must be passed
    if we need platform-specific optimizer settings.
    """

    def __init__(
        self,
        dataset_path: str,
        lora_config: Dict,
        training_config: Dict,
        hardware_config=None
    ):
        self.dataset_path = Path(dataset_path)

        # The training config must include a model_name_or_path
        if "model_name_or_path" not in training_config:
            raise ValueError("training_config.json must contain 'model_name_or_path'")

        self.model_name = training_config["model_name_or_path"]
        self.dataset_id = self.dataset_path.name
        logger.info(f"Extracted dataset ID: {self.dataset_id}")

        # Create an output directory named with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("src/models/storage/deberta-v3-base/lora/three_way") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_metrics = self._load_dataset_metrics()

        # Store config details
        self.lora_config_dict = lora_config
        self.training_config = training_config
        if hardware_config is None:
            raise ValueError(
                "hardware_config is required. Initialize HardwareConfig "
                "from environment.py before creating DebertaTrainer."
            )
        self.hardware_config = hardware_config
        self.device = torch.device(hardware_config.device)
        logger.info(f"Using device from hardware_config: {self.device}")

        # Convert LoRA dict to actual config object
        self.lora_config = self._create_lora_config(self.lora_config_dict)

        # Hard-coded label mapping for 3-way sentiment
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}

    def _load_dataset_metrics(self) -> Dict:
        """
        Load any previously computed metrics for the dataset, if available.
        """
        metrics_path = Path("src/data/storage/metrics") / f"{self.dataset_id}_metrics.json"
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"No precomputed metrics found at {metrics_path}")
            return {}

    def _create_lora_config(self, config_dict: Dict) -> LoraConfig:
        """
        Convert the provided dictionary into a LoraConfig object.
        Raises an error if any required keys for LoRA are missing.
        """
        try:
            return LoraConfig(**config_dict)
        except Exception as e:
            logger.error(f"Invalid LoRA config dictionary: {config_dict}")
            raise ValueError(f"Failed to create LoraConfig: {e}")

    def _initialize_tokenizer(self):
        logger.info(f"Initializing tokenizer for {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=512,
                use_fast=True,
                trust_remote_code=True
            )
            # Validate correct tokenizer type
            if not any(name in str(type(tokenizer)).lower() for name in ["deberta", "sentencepiece"]):
                raise ValueError("The loaded tokenizer is not DeBERTa/SentencePiece-based.")
            return tokenizer
        except Exception as fast_error:
            logger.warning(f"Failed to load fast tokenizer: {fast_error}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    trust_remote_code=True
                )
                return tokenizer
            except Exception as e:
                logger.error("Failed to load both fast and non-fast tokenizer.")
                raise RuntimeError(f"DeBERTa tokenizer initialization failed: {e}")

    def _preprocess_dataset(self, dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
        """
        Tokenize text and map label strings to integer IDs.
        """
        def process(example):
            tokenized = tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
            label_str = example["labels"]
            logger.debug(f"Processing label: {label_str}")

            if label_str not in self.label2id:
                raise ValueError(
                    f"Unexpected label '{label_str}'. "
                    f"Must be one of: {list(self.label2id.keys())}."
                )
            tokenized["labels"] = self.label2id[label_str]
            return tokenized

        for split_name in dataset_dict.keys():
            logger.info(f"Processing {split_name} split...")
            dataset_dict[split_name] = dataset_dict[split_name].map(process, batched=False)
            dataset_dict[split_name].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"]
            )
            print(dataset_dict["validation"].column_names)

            unique_labels = dataset_dict[split_name]["labels"].unique()
            logger.info(f"Unique labels in {split_name}: {unique_labels}")
            
        return dataset_dict

    def _create_model(self):
        """
        Load the base DeBERTa model and apply LoRA adapters with explicit gradient setup.
        """
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        # Move to device before LoRA adaptation
        base_model = base_model.to(self.device)
        
        # Ensure gradients are enabled
        for param in base_model.parameters():
            param.requires_grad_(True)
        
        model = get_peft_model(base_model, self.lora_config)
        
        # Double-check gradients after LoRA
        trainable_params = 0
        all_param = 0
        for param in model.parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        logger.info(
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param}"
        )
        return model

    def _setup_training_args(self) -> TrainingArguments:
        """
        Build TrainingArguments with platform-specific optimizations,
        using the hardware_config if provided.
        """
        dataset = load_from_disk(self.dataset_path)
        train_size = len(dataset["train"])
        logger.info(f"Training set size: {train_size} samples")

        # Use hardware_config directly - it's required in __init__
        device_type = self.hardware_config.optimizer_type

        optimizer_config = TrainingConfigurator.get_optimizer_config(device_type)
        
        args_dict = {
            "output_dir": str(self.output_dir),
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "report_to": "none",
            "load_best_model_at_end": True,
            **optimizer_config
        }

        user_params = {
            k: v for k, v in self.training_config.items() 
            if k not in ["model_name_or_path"]
        }
        args_dict.update(user_params)

        return TrainingArguments(**args_dict)

    def _compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute combined_metric = 0.7 * macro-F1 + 0.3 * accuracy
        along with basic metrics (accuracy, precision, recall, f1).
        The HF Trainer will see them as "eval_accuracy", "eval_f1", etc.
        but we reference "eval_combined_metric" in training_config.json.
        """
        logger.info("Inside _compute_metrics, about to compute accuracy, f1, etc.")
        try:
            if isinstance(p.predictions, tuple):
                logits = p.predictions[0]
            else:
                logits = p.predictions

            preds = np.argmax(logits, axis=1)
            labels = p.label_ids

            accuracy_val = accuracy_score(labels, preds)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            # Our new combined metric
            combined_metric = 0.7 * f1_macro + 0.3 * accuracy_val

            metrics = {
                "accuracy": float(accuracy_val),
                "precision": float(precision_macro),
                "recall": float(recall_macro),
                "f1": float(f1_macro),
                "combined_metric": float(combined_metric)
            }

            logger.debug(f"Computed metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            raise

    def train(self) -> Dict:
        """
        The end-to-end training process:
        1. Load dataset and tokenizer
        2. Preprocess
        3. Build LoRA model
        4. Instantiate Trainer w/ a combined metric
        5. Train and evaluate on test set
        """
        try:
            dataset_dict = load_from_disk(self.dataset_path)
            logger.info(f"Loaded dataset splits: {list(dataset_dict.keys())}")

            tokenizer = self._initialize_tokenizer()
            dataset_dict = self._preprocess_dataset(dataset_dict, tokenizer)

            model = self._create_model()
            training_args = self._setup_training_args()

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            # Train
            train_result = trainer.train()

            # Evaluate on test
            test_metrics = trainer.evaluate(dataset_dict["test"])
            self._save_results(train_result, test_metrics)

            return {
                "status": "success",
                "test_metrics": test_metrics,
                "model_path": str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}

    def _save_results(self, train_result, test_metrics):
        """
        Save relevant metadata and results. 
        """
        def make_json_serializable(obj):
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj

        # Convert training metrics
        train_metrics = getattr(train_result, "metrics", {})
        if callable(train_metrics):
            train_metrics = train_metrics()

        meta = {
            "dataset_info": {
                "id": self.dataset_id,
                "metrics": self.dataset_metrics
            },
            "training_config": {
                "model_name_or_path": self.model_name,
                "lora_config": self.lora_config_dict,
                "device": str(self.device)
            },
            "results": {
                "train_metrics": {k: make_json_serializable(v) for k, v in train_metrics.items()},
                "test_metrics": {k: make_json_serializable(v) for k, v in test_metrics.items()}
            }
        }
        out_path = self.output_dir / "training_metadata.json"
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2, default=make_json_serializable)
