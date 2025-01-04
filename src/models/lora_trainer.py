from transformers import Trainer, TrainingArguments
from peft import get_peft_model
import torch
from typing import Dict, Any, Optional
from .lora_config import create_lora_config, LoRAParameters
from .model_factory import ModelFactory
import logging
from ..evaluation.metrics import compute_classification_metrics
from torch.utils.data import DataLoader, Sampler
import numpy as np
from .weights import SentimentDistributionAnalyzer

logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
        self.device = self.model_factory.get_device()
        
    def setup_training_args(self, output_dir: str, dataset_size: int) -> TrainingArguments:
        """
        Dynamically configure training arguments based on dataset size
        
        Args:
            output_dir: Directory to save model
            dataset_size: Number of training examples
        """
        use_fp16 = self.device.type == "cuda"
        
        # Dynamic parameter calculation
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
            batch_size = min(batch_size, 24)  # Conservative for M4 Pro
        
        logger.info(f"Training configuration for {dataset_size} examples:")
        logger.info(f"- Batch size: {batch_size}")
        logger.info(f"- Gradient accumulation: {grad_accum}")
        logger.info(f"- Epochs: {num_epochs}")
        
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
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
            report_to="none",
        )
        
    def train(self, 
              train_dataset, 
              eval_dataset, 
              lora_params: Optional[LoRAParameters] = None,
              output_dir: str = "./results",
              model_name: str = "microsoft/deberta-v3-base") -> Dict[str, Any]:
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Using model: {model_name}")
        
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
        
        # Add distribution check
        train_labels = [example['labels'] for example in train_dataset]
        logger.info("Label distribution after conversion:")
        logger.info(f"Label counts: {np.bincount(train_labels)}")
        logger.info(f"Label percentages: {np.bincount(train_labels) / len(train_labels) * 100}%")
        
        # Basic model setup
        model, tokenizer = self.model_factory.create_model()
        model = get_peft_model(model, create_lora_config(lora_params))
        
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
        
        trainer = Trainer(
            model=model,
            args=self.setup_training_args(output_dir, dataset_size),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_classification_metrics
        )
        
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.save_model()
            
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