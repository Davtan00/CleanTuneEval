import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import DataLoader
from .model_factory import ModelFactory
from ..evaluation.metrics import compute_classification_metrics
from .weights import SentimentDistributionAnalyzer  # Domain-based weights
logger = logging.getLogger(__name__)

class WeightedLossTrainer(Trainer):
    """
    HF Trainer subclass applying class-weighted CrossEntropyLoss if 'class_weights' is set.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation with optional class weights.
        Ignores additional kwargs passed by the trainer.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class LoRATrainer:
    """
    Orchestrates LoRA fine-tuning with domain-based weighting, dynamic hyperparams,
    multi-metric evaluation, and training metadata logging.
    """
    def __init__(self, 
                 model_factory: ModelFactory,
                 config_dir: str = "src/models/config"):
        """
        Initialize trainer with mandatory configuration files.
        
        Args:
            model_factory: ModelFactory instance
            config_dir: Directory containing configuration files
        
        Raises:
            FileNotFoundError: If either config file is missing
            ValueError: If configs are invalid
        """
        self.model_factory = model_factory
        self.device = self.model_factory.get_device()
        self.class_weights = None
        
        # Load mandatory configurations
        lora_config_path = Path(config_dir) / "lora_config.json"
        training_config_path = Path(config_dir) / "training_config.json"
        
        if not lora_config_path.exists():
            raise FileNotFoundError(f"LoRA config not found at {lora_config_path}")
        if not training_config_path.exists():
            raise FileNotFoundError(f"Training config not found at {training_config_path}")
            
        # Load and validate configs
        try:
            with open(lora_config_path) as f:
                self.lora_config = json.load(f)
            with open(training_config_path) as f:
                self.training_config = json.load(f)
                
            # Validate required fields
            self._validate_configs()
            
            logger.info("Successfully loaded LoRA and training configurations")
            logger.debug(f"LoRA config: {self.lora_config}")
            logger.debug(f"Training config: {self.training_config}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config files: {str(e)}")
            
    def _validate_configs(self):
        """Ensure all required configuration fields are present."""
        required_lora_fields = {"r", "lora_alpha", "lora_dropout", "bias", "target_modules"}
        required_training_fields = {
            "model_name_or_path", "num_train_epochs", "per_device_train_batch_size",
            "learning_rate", "weight_decay"
        }
        
        missing_lora = required_lora_fields - set(self.lora_config.keys())
        missing_training = required_training_fields - set(self.training_config.keys())
        
        if missing_lora:
            raise ValueError(f"Missing required LoRA config fields: {missing_lora}")
        if missing_training:
            raise ValueError(f"Missing required training config fields: {missing_training}")

    def setup_training_args(self, output_dir: str, dataset_size: int, domain: Optional[str] = None) -> TrainingArguments:
        """
        Set up training arguments using values from training_config.json.
        Falls back to calculated defaults only if values are missing.
        
        Args:
            output_dir: Directory to save model outputs
            dataset_size: Size of training dataset
            domain: Optional domain for weight calculation
        """
        # Get values from training config
        batch_size = self.training_config.get("per_device_train_batch_size")
        learning_rate = self.training_config.get("learning_rate")
        num_epochs = self.training_config.get("num_train_epochs")
        weight_decay = self.training_config.get("weight_decay", 0.01)
        
        # Only calculate defaults if not specified in config
        if not batch_size:
            if dataset_size < 10000:
                batch_size = 32
            elif dataset_size < 50000:
                batch_size = 24
            else:
                batch_size = 16
            logger.info(f"Batch size not specified in config, using calculated value: {batch_size}")
        
        # Adjust batch size for MPS
        if self.device.type == "mps":
            batch_size = min(batch_size, 24)
            logger.info(f"Adjusted batch size for MPS: {batch_size}")

        # Handle domain-based weighting
        if domain and getattr(self.model_factory.hardware, 'use_research_weights', False):
            analyzer = SentimentDistributionAnalyzer()
            w = analyzer.get_domain_weights(domain)
            logger.info(f"Using domain-based weights for '{domain}': {w}")
            self.class_weights = torch.tensor(w).to(self.device)

        # Determine optimizer settings based on hardware
        optimizer_kwargs = {}
        if self.model_factory.hardware.optimizer_type == "cuda":
            try:
                import bitsandbytes as bnb
                optimizer_kwargs = {
                    "optim": "adamw_bnb_8bit",
                    "fp16": True
                }
                logger.info("Using 8-bit AdamW optimizer with CUDA")
            except ImportError:
                optimizer_kwargs = {
                    "optim": "adamw_torch",
                    "fp16": True
                }
                logger.warning("Bitsandbytes not available, falling back to torch AdamW with CUDA")
        elif self.model_factory.hardware.optimizer_type == "mps":
            optimizer_kwargs = {
                "optim": "adamw_torch",
                "fp16": False  # MPS doesn't support fp16 training
            }
            logger.info("Using torch AdamW optimizer with MPS")
        else:
            optimizer_kwargs = {
                "optim": "adamw_torch",
                "fp16": False
            }
            logger.info("Using torch AdamW optimizer with CPU")

        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_ratio=self.training_config.get("warmup_ratio", 0.1),
            evaluation_strategy=self.training_config.get("evaluation_strategy", "epoch"),
            save_strategy=self.training_config.get("save_strategy", "epoch"),
            load_best_model_at_end=self.training_config.get("load_best_model_at_end", True),
            metric_for_best_model=self.training_config.get("metric_for_best_model", "eval_combined_metric"),
            greater_is_better=self.training_config.get("greater_is_better", True),
            logging_dir="logs",
            logging_steps=self.training_config.get("logging_steps", 50),
            dataloader_num_workers=0,
            dataloader_pin_memory=False if self.device.type == "mps" else True,
            report_to="none",
            **optimizer_kwargs
        )

    def train(self, 
              train_dataset,
              eval_dataset,
              output_dir: str,
              model_name: str = "microsoft/deberta-v3-base") -> Dict[str, Any]:
        logger.info(f"Training on device: {self.device}, model={model_name}")

        # Get dataset path correctly
        try:
            # Extract the actual dataset name from the cache path
            cache_path = train_dataset.cache_files[0]['filename']
            dataset_name = cache_path.split('/')[-4]  # Get the dataset folder name
            dataset_path = str(Path("src/data/datasets") / dataset_name)
            logger.info(f"Dataset path detected: {dataset_path}")
            logger.info(f"Dataset name extracted: {dataset_name}")
        except (AttributeError, IndexError, KeyError) as e:
            logger.warning(f"Could not detect dataset path from dataset object: {e}")
            dataset_path = None

        domain = self._detect_domain(dataset_path)

        # Training args
        training_args = self.setup_training_args(output_dir, len(train_dataset), domain=domain)

        # Convert textual labels (neg, neu, pos) to 0,1,2
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        train_dataset = train_dataset.map(lambda ex: self._convert_label(ex, label_map))
        eval_dataset = eval_dataset.map(lambda ex: self._convert_label(ex, label_map))
        self._log_label_distribution(train_dataset)

        dataset_info = self._create_dataset_info(train_dataset, eval_dataset)
        model, tokenizer = self.model_factory.create_model()
        model.config.custom_dataset_info = dataset_info

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

        # Preprocess
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing train dataset",
            remove_columns=['text', 'id']
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing val dataset",
            remove_columns=['text', 'id']
        )

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        trainer = WeightedLossTrainer(
            class_weights=self.class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_classification_metrics,
            callbacks=[DetailedLossCallback()]
        )

        try:
            train_result = trainer.train()
            
            # Fix: Check if train_result is a tuple or TrainOutput object
            metrics = {}
            if hasattr(train_result, 'metrics'):
                metrics = train_result.metrics
            elif isinstance(train_result, tuple) and len(train_result) > 0:
                metrics = train_result[0]
            
            # Save model before trying to access metrics
            try:
                trainer.save_model()
                logger.info(f"Model saved to {output_dir}")
            except Exception as save_error:
                logger.error(f"Failed to save model: {save_error}")
            
            # Save metadata with safe defaults
            training_metadata = {
                "dataset_info": dataset_info,
                "training_metrics": metrics,
                "model_config": {
                    "base_model": model_name,
                    "training_method": "lora",
                    "hardware_used": str(self.device)
                }
            }
            
            try:
                metadata_path = Path(output_dir) / "training_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(training_metadata, f, indent=2)
                logger.info(f"Saved training metadata to {metadata_path}")
            except Exception as meta_error:
                logger.error(f"Failed to save metadata: {meta_error}")

            # Evaluate with proper error handling
            try:
                eval_metrics = trainer.evaluate()
                if isinstance(eval_metrics, dict):
                    metrics.update(eval_metrics)
            except Exception as eval_error:
                logger.error(f"Evaluation failed: {eval_error}")
            
            logger.info("Training completed.")
            self._log_metrics(metrics)
            
            return {
                "status": "success",
                "metrics": metrics,
                "model_path": str(output_dir)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            return {
                "status": "error",
                "message": str(e),
                "type": str(type(e))
            }

    @staticmethod
    def _convert_label(example: Dict[str, Any], mapping: Dict[str, int]) -> Dict[str, Any]:
        example["labels"] = mapping[example["labels"]]
        return example

    def _detect_domain(self, dataset_path: str) -> Optional[str]:
        if not dataset_path:
            logger.warning("No dataset_path; cannot detect domain.")
            return None
        
        try:
            # Construct path to metrics file using dataset name
            dataset_name = Path(dataset_path).name
            metrics_file = Path("src/data/storage/metrics") / f"{dataset_name}_metrics.json"
            
            logger.debug(f"Looking for metrics file at: {metrics_file}")
            
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        data = json.load(f)
                        domain = data.get("domain")
                        if domain:
                            logger.info(f"Domain from dataset metrics: {domain}")
                            return domain
                except Exception as exc:
                    logger.warning(f"Could not load domain from metrics file: {exc}")
            else:
                logger.warning(f"Metrics file not found at: {metrics_file}")
        except Exception as e:
            logger.error(f"Error detecting domain: {e}")
        return None

    def _log_label_distribution(self, dataset) -> None:
        train_labels = [ex["labels"] for ex in dataset]
        counts = np.bincount(train_labels)
        total = len(train_labels)
        logger.info("Label distribution:")
        for idx, c in enumerate(counts):
            logger.info(f"  Label {idx}: {c} ({(c/total)*100:.2f}%)")

    def _create_dataset_info(self, train_dataset, eval_dataset) -> Dict[str, Any]:
        path = getattr(train_dataset, "dataset_info", {}).get("path", "unknown")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "dataset_path": path,
            "dataset_name": Path(path).name if path != "unknown" else "unknown",
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "training_date": now_str
        }

    def _log_metrics(self, metrics: Dict[str, Any]):
        logger.info("Final Metrics:")
        for k, v in metrics.items():
            if "confusion_matrix" in k:
                logger.info(f"{k}:")
                for row in v:
                    logger.info(f"  {row}")
            elif isinstance(v, float) or isinstance(v, int):
                logger.info(f"{k}: {v:.4f}")
            else:
                logger.info(f"{k}: {v}")

    @staticmethod
    def get_training_info(model_path: str) -> Dict[str, Any]:
        try:
            mp = Path(model_path) / "training_metadata.json"
            if mp.exists():
                with open(mp) as f:
                    return json.load(f)
            config = AutoConfig.from_pretrained(model_path)
            if hasattr(config, "custom_dataset_info"):
                return {"dataset_info": config.custom_dataset_info}
            return {"error": "No training information found"}
        except Exception as e:
            return {"error": f"Failed to load training info: {str(e)}"}

class DetailedLossCallback(TrainerCallback):
    """
    Callback to log training/eval loss in real time.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if 'loss' in logs:
                self.train_losses.append((step, logs['loss']))
                logger.info(f"Step {step} - Train loss: {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                self.eval_losses.append((step, logs['eval_loss']))
                logger.info(f"Step {step} - Eval loss: {logs['eval_loss']:.4f}")
            if 'loss' in logs and 'eval_loss' in logs:
                diff = abs(logs['loss'] - logs['eval_loss'])
                logger.info(f"Loss difference (train-eval): {diff:.4f}") 