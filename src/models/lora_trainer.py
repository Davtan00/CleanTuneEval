from typing import Dict, Optional
import torch
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from .lora_config import LoRAConfig
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class LoRATrainer:
    def __init__(
        self, 
        base_model, 
        tokenizer, 
        hardware_config: HardwareConfig,
        lora_config: Optional[LoRAConfig] = None
    ):
        logger.info(f"Initializing LoRA trainer with device: {hardware_config.device}")
        self.hardware_config = hardware_config
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.lora_config = lora_config or LoRAConfig.get_optimal_config(hardware_config)
        
        # Configure for Apple Silicon if available
        self.device = hardware_config.device
        self.use_mixed_precision = hardware_config.use_mps
        logger.info(f"Mixed precision training: {self.use_mixed_precision}")
        
    def prepare_model(self):
        """Prepare the model with LoRA configuration"""
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Initialize PEFT model
        model = get_peft_model(self.base_model, peft_config)
        
        if self.use_mixed_precision:
            model = model.half()  
        
        return model.to(self.device)
    
    def train(
        self, 
        train_dataset, 
        eval_dataset=None, 
        output_dir="./lora_output",
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4
    ):
        """Train the LoRA model"""
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-5,
            fp16=self.use_mixed_precision,
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="wandb"  # Enable W&B logging
        )
        
        model = self.prepare_model()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        
        return {
            'train_loss': train_result.training_loss,
            'model_path': output_dir
        } 