from transformers import Trainer, TrainingArguments
from peft import get_peft_model
import torch
from typing import Dict, Any, Optional
from .lora_config import create_lora_config, LoRAParameters
from .model_factory import ModelFactory
import logging

logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory
        self.device = self.model_factory.get_device()
        
    def setup_training_args(self, output_dir: str) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=2,
            num_train_epochs=5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            use_mps_device=self.device.type == "mps",
            fp16=self.device.type != "cpu",
            logging_dir="logs",
            logging_steps=100,
        )
        
    def train(self, 
              train_dataset, 
              eval_dataset, 
              lora_params: Optional[LoRAParameters] = None,
              output_dir: str = "./results") -> Dict[str, Any]:
        
        model, tokenizer = self.model_factory.create_model()
        lora_config = create_lora_config(lora_params)
        
        model = get_peft_model(model, lora_config)
        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
        
        training_args = self.setup_training_args(output_dir)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.save_model()
            
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            
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