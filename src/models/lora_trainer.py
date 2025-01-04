from transformers import Trainer, TrainingArguments
from peft import get_peft_model
import torch
from typing import Dict, Any, Optional
from .lora_config import create_lora_config, LoRAParameters
from .model_factory import ModelFactory
import logging
from ..evaluation.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
        self.device = self.model_factory.get_device()
        
    def setup_training_args(self, output_dir: str) -> TrainingArguments:
        # Determine hardware-specific settings
        use_fp16 = False
        if self.device.type == "cuda":
            use_fp16 = True
        elif self.device.type == "mps":
            # MPS doesn't support FP16 training yet
            use_fp16 = False
            
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=32,  # Increased for M4 Pro
            gradient_accumulation_steps=2,
            num_train_epochs=5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",  # Changed to F1 score
            greater_is_better=True,  # Changed because F1 should be maximized
            fp16=use_fp16,
            logging_dir="logs",
            logging_steps=100,
            dataloader_num_workers=0,
            dataloader_pin_memory=False if self.device.type == "mps" else True,
            report_to="none",
        )
        
    def train(self, 
              train_dataset, 
              eval_dataset, 
              lora_params: Optional[LoRAParameters] = None,
              output_dir: str = "./results") -> Dict[str, Any]:
        
        logger.info(f"Training on device: {self.device}")
        
        # Create label mapping
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        def convert_labels(example):
            example['labels'] = label_mapping[example['labels']]
            return example
        
        # Convert string labels to integers
        logger.info("Converting labels to integers...")
        train_dataset = train_dataset.map(convert_labels)
        eval_dataset = eval_dataset.map(convert_labels)
        
        model, tokenizer = self.model_factory.create_model()
        lora_config = create_lora_config(lora_params)
        
        model = get_peft_model(model, lora_config)
        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
        
        # Add preprocessing function for the datasets
        def preprocess_function(examples):
            # Tokenize the texts
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None
            )
            
            # Important: Keep the labels in the dataset
            tokenized['labels'] = examples['labels']
            return tokenized
        
        # Preprocess the datasets
        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing train dataset",
            remove_columns=['text', 'id']  # Only remove text and id, keep labels
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing validation dataset",
            remove_columns=['text', 'id']  # Only remove text and id, keep labels
        )
        
        # Set the format of our datasets to PyTorch tensors
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        training_args = self.setup_training_args(output_dir)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_classification_metrics  # Add metrics computation
        )
        
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.save_model()
            
            # Get detailed evaluation metrics
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            
            # Log detailed metrics
            logger.info("Training completed successfully")
            logger.info(f"Model saved at: {output_dir}")
            logger.info("Final Metrics:")
            for metric_name, value in metrics.items():
                if metric_name != 'confusion_matrix':  # Don't log the confusion matrix
                    logger.info(f"{metric_name}: {value:.4f}")
            
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