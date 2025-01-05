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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights) if self.class_weights else torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class LoRATrainer:
    """
    Orchestrates LoRA fine-tuning with domain-based weighting, dynamic hyperparams,
    multi-metric evaluation, and training metadata logging.
    """
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
        self.device = self.model_factory.get_device()
        self.class_weights = None

    def setup_training_args(self, output_dir: str, dataset_size: int, domain: Optional[str] = None) -> TrainingArguments:
        """
        Dynamically set hyperparams (batch size, epochs) based on dataset_size.
        Domain-based weighting if hardware config says 'use_research_weights'.
        """
        use_fp16 = (self.device.type == "cuda")
        if dataset_size < 10000:
            batch_size, grad_accum, num_epochs, eval_steps = 32, 2, 5, 100
        elif dataset_size < 50000:
            batch_size, grad_accum, num_epochs, eval_steps = 24, 4, 3, 200
        else:
            batch_size, grad_accum, num_epochs, eval_steps = 16, 8, 2, 500
        if self.device.type == "mps":
            batch_size = min(batch_size, 24)

        # Domain-based weighting
        if domain and getattr(self.model_factory.hardware, 'use_research_weights', False):
            analyzer = SentimentDistributionAnalyzer()
            w = analyzer.get_domain_weights(domain)
            logger.info(f"Using domain-based weights for '{domain}': {w}")
            self.class_weights = torch.tensor(w).to(self.device)

        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_combined_metric",
            greater_is_better=True,
            fp16=use_fp16,
            logging_dir="logs",
            logging_steps=eval_steps,
            dataloader_num_workers=0,
            dataloader_pin_memory=False if self.device.type == "mps" else True,
            report_to="none"
        )

    def train(self, train_dataset, eval_dataset,
              output_dir: str = "./results",
              model_name: str = "microsoft/deberta-v3-base") -> Dict[str, Any]:
        logger.info(f"Training on device: {self.device}, model={model_name}")

        # Attempt domain detection from dataset info
        dataset_path = getattr(train_dataset, "dataset_info", {}).get("path", "")
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
            metrics = train_result.metrics
            trainer.save_model()

            # Save metadata
            training_metadata = {
                "dataset_info": dataset_info,
                "training_metrics": metrics,
                "model_config": {
                    "base_model": model_name,
                    "training_method": "lora",
                    "hardware_used": str(self.device)
                }
            }
            metadata_path = Path(output_dir) / "training_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(training_metadata, f, indent=2)

            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            logger.info("Training completed.")
            self._log_metrics(metrics)
            return {"status": "success", "metrics": metrics, "model_path": output_dir}
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _convert_label(example: Dict[str, Any], mapping: Dict[str, int]) -> Dict[str, Any]:
        example["labels"] = mapping[example["labels"]]
        return example

    def _detect_domain(self, dataset_path: str) -> Optional[str]:
        if not dataset_path:
            logger.warning("No dataset_path; cannot detect domain.")
            return None
        metrics_file = Path(dataset_path).parent.parent / "metrics" / f"{Path(dataset_path).name}_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    domain = json.load(f).get("domain")
                    if domain:
                        logger.info(f"Domain from dataset metrics: {domain}")
                    return domain
            except Exception as exc:
                logger.warning(f"Could not load domain: {exc}")
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