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
        from pathlib import Path
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
        """Simplified preprocessing with explicit tensor handling"""

        def process(example):
            # Basic tokenization without tensor conversion
            tokenized = tokenizer(
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

        # Process all splits
        processed = dataset_dict.map(
            process,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing datasets"
        )

        # Set format for PyTorch
        processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return processed

    def _create_model(self):
        """
        Simplified model creation with explicit gradient setup
        """
        try:
            # 1. Initialize base model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3,  # Hardcoded for 3-way sentiment
                trust_remote_code=True
            )

            # 2. Enable gradients explicitly (in case any layers were off)
            for param in model.parameters():
                param.requires_grad = True

            # 3. Create and apply LoRA config
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=self.lora_config.target_modules
            )
            model = get_peft_model(model, peft_config)

            # 4. Move to device after LoRA application
            model = model.to(self.device)

            # 5. Double-check trainable parameters
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

            # 6. Ensure training mode
            model.train()
            return model

        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            raise

    def _setup_training_args(self) -> TrainingArguments:
        """
        Merge user-supplied training_config.json with the device-specific
        defaults from TrainingConfigurator (if you wish to use them).
        """

        # Start with user JSON config
        user_conf = self.training_config.copy()

        # Pull in device-specific defaults (cuda/mps/cpu)
        device_config = TrainingConfigurator.get_optimizer_config(self.hardware_config.optimizer_type)

        # Merge device_config into user_conf, but user_conf keys take priority
        for k, v in device_config.items():
            if k not in user_conf:
                user_conf[k] = v

        # Build a dictionary of TrainingArguments from the merged config
        # (some keys in user_conf won't be recognized by TrainingArguments, so filter them)
        known_args = {
            # Must-have fields
            "output_dir": str(self.output_dir),
            "evaluation_strategy": user_conf["evaluation_strategy"],
            "save_strategy": user_conf["save_strategy"],
            "load_best_model_at_end": user_conf["load_best_model_at_end"],
            "metric_for_best_model": user_conf["metric_for_best_model"],
            "greater_is_better": user_conf["greater_is_better"],
            "logging_steps": user_conf["logging_steps"],
            # Typical hyperparams
            "num_train_epochs": user_conf["num_train_epochs"],
            "per_device_train_batch_size": user_conf["per_device_train_batch_size"],
            "per_device_eval_batch_size": user_conf["per_device_eval_batch_size"],
            "learning_rate": user_conf["learning_rate"],
            "weight_decay": user_conf["weight_decay"],
            # Additional optional fields from device_config or user_conf
            "warmup_ratio": user_conf.get("warmup_ratio", 0.0),
            "optim": user_conf.get("optim", "adamw_torch"),
            "fp16": user_conf.get("fp16", False),
            "gradient_checkpointing": user_conf.get("gradient_checkpointing", False),
            "gradient_accumulation_steps": user_conf.get("gradient_accumulation_steps", 1),
            "dataloader_pin_memory": user_conf.get("dataloader_pin_memory", True),
        }

        return TrainingArguments(**known_args)

    def _compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute combined_metric = 0.7 * macro-F1 + 0.3 * accuracy
        along with basic metrics (accuracy, precision, recall, f1).
        """
        logger.info("Inside _compute_metrics, about to compute accuracy, f1, etc.")
        try:
            # Unpack logits
            if isinstance(p.predictions, tuple):
                logits = p.predictions[0]
            else:
                logits = p.predictions

            preds = np.argmax(logits, axis=1)
            labels = p.label_ids

            # Basic metrics
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
            # 1. Load dataset
            dataset_dict = load_from_disk(self.dataset_path)
            logger.info(f"Loaded dataset splits: {list(dataset_dict.keys())}")

            # 2. Initialize tokenizer, preprocess
            tokenizer = self._initialize_tokenizer()
            dataset_dict = self._preprocess_dataset(dataset_dict, tokenizer)

            # 3. Create the model (base + LoRA)
            model = self._create_model()

            # 4. Setup training args and create Trainer
            training_args = self._setup_training_args()
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            # 5. Train
            train_result = trainer.train()

            # Evaluate on test set
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
