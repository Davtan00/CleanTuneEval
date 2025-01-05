from transformers import Trainer, TrainingArguments
from .model_factory import ModelFactory
import torch
from typing import Dict, Any, Optional
import logging
from ..evaluation.metrics import compute_classification_metrics
from torch.utils.data import DataLoader, Sampler
import numpy as np
from .weights import SentimentDistributionAnalyzer
from transformers.trainer_callback import TrainerCallback
from pathlib import Path
import json
from datetime import datetime
from transformers import AutoConfig

logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
        self.device = self.model_factory.get_device()
        
    def setup_training_args(self, output_dir: str, dataset_size: int, domain: str = None) -> TrainingArguments:
        """Dynamically configure training arguments based on dataset size and domain"""
        use_fp16 = self.device.type == "cuda"
        
        if dataset_size < 10000:
            batch_size = 32
            grad_accum = 2
            num_epochs = 5
            eval_steps = 100
        elif dataset_size < 50000:
            batch_size = 24
            grad_accum = 4
            num_epochs = 3
            eval_steps = 200
        else:
            batch_size = 16
            grad_accum = 8
            num_epochs = 2
            eval_steps = 500
        
        # Adjust for available memory
        if self.device.type == "mps":
            batch_size = min(batch_size, 24)
        
        # Store weights for use in custom trainer
        self.class_weights = None
        if domain and hasattr(self.model_factory.hardware, 'use_research_weights') and self.model_factory.hardware.use_research_weights:
            analyzer = SentimentDistributionAnalyzer()
            weights = analyzer.get_domain_weights(domain)
            logger.info(f"Using research-based weights for {domain} domain: {weights}")
            self.class_weights = torch.tensor(weights).to(self.device)
        else:
            logger.info("No domain-specific weights applied")
        
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
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=use_fp16,
            logging_dir="logs",
            logging_steps=eval_steps,
            dataloader_num_workers=0,
            dataloader_pin_memory=False if self.device.type == "mps" else True,
            report_to="none"
        )
        
    def train(self, 
              train_dataset, 
              eval_dataset, 
              output_dir: str = "./results",
              model_name: str = "microsoft/deberta-v3-base") -> Dict[str, Any]:
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Using model: {model_name}")
        
        # Extract domain from dataset path
        dataset_path = getattr(train_dataset, "dataset_info", {}).get("path", "")
        domain = None
        
        # Try to get domain from metrics file
        if dataset_path:
            metrics_file = Path(dataset_path).parent.parent / "metrics" / f"{Path(dataset_path).name}_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics_data = json.load(f)
                        domain = metrics_data.get("domain")
                        logger.info(f"Detected domain from dataset metrics: {domain}")
                except Exception as e:
                    logger.warning(f"Could not load domain from metrics file: {e}")
        
        if not domain:
            logger.warning("No domain detected! Training will proceed without class weights.")
        
        # Pass domain to setup_training_args
        training_args = self.setup_training_args(output_dir, len(train_dataset), domain=domain)
        
        # Get dataset size for dynamic configuration
        dataset_size = len(train_dataset)
        
        # Add verification of label distribution
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        def convert_labels(example):
            example['labels'] = label_mapping[example['labels']]
            return example
        
        # Verify label distribution before training
        train_dataset = train_dataset.map(convert_labels)
        eval_dataset = eval_dataset.map(convert_labels)
        
        # Add distribution check with better formatting
        train_labels = [example['labels'] for example in train_dataset]
        label_counts = np.bincount(train_labels)
        label_percentages = label_counts / len(train_labels) * 100

        logger.info("Label distribution after conversion:")
        for label, (count, percentage) in enumerate(zip(label_counts, label_percentages)):
            logger.info(f"Label {label}: {count} samples ({percentage:.2f}%)")
        
        # Get dataset info before training
        dataset_info = {
            "dataset_path": getattr(train_dataset, "dataset_info", {}).get("path", "unknown"),
            "dataset_name": Path(getattr(train_dataset, "dataset_info", {}).get("path", "unknown")).name,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "label_distribution": {
                str(label): int(count) for label, count in zip(range(len(label_counts)), label_counts)
            },
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Basic model setup
        model, tokenizer = self.model_factory.create_model()
        
        # Store dataset info in model config
        model.config.custom_dataset_info = dataset_info
        
        # Standard preprocessing
        def preprocess_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None
            )
        
        # Process datasets
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing train dataset",
            remove_columns=['text', 'id']
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing validation dataset",
            remove_columns=['text', 'id']
        )
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Add callback for loss monitoring
        class DetailedLossCallback(TrainerCallback):
            def __init__(self):
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
                        
                    # Log loss difference if both available
                    if 'loss' in logs and 'eval_loss' in logs:
                        diff = abs(logs['loss'] - logs['eval_loss'])
                        logger.info(f"Loss difference: {diff:.4f}")
        
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
            
            # Save model with updated config
            trainer.save_model()
            
            # Also save a separate JSON with more detailed info
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
            
            logger.info(f"Training metadata saved to: {metadata_path}")
            
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            
            logger.info("Training completed successfully")
            logger.info(f"Model saved at: {output_dir}")
            self._log_metrics(metrics)
            
            return {
                "status": "success",
                "metrics": metrics,
                "model_path": output_dir
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Helper method to log metrics in a consistent format"""
        logger.info("Final Metrics:")
        for name, value in metrics.items():
            if name == 'confusion_matrix':
                logger.info(f"Confusion Matrix:")
                for row in value:
                    logger.info(f"    {row}")
            elif isinstance(value, (int, float)):
                logger.info(f"{name}: {value:.4f}")
            else:
                logger.info(f"{name}: {value}") 

    @staticmethod
    def get_training_info(model_path: str) -> Dict[str, Any]:
        """
        Retrieve training information for a saved model.
        
        Args:
            model_path: Path to the saved model directory
        
        Returns:
            Dict containing dataset and training information
        """
        try:
            # Try to load the detailed metadata file first
            metadata_path = Path(model_path) / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    return json.load(f)
            
            # Fallback to config-only info
            config = AutoConfig.from_pretrained(model_path)
            if hasattr(config, "custom_dataset_info"):
                return {"dataset_info": config.custom_dataset_info}
            
            return {"error": "No training information found"}
            
        except Exception as e:
            return {"error": f"Failed to load training info: {str(e)}"} 

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss 