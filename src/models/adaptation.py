from typing import Dict, Optional
from .lora_trainer import LoRATrainer
from .model_factory import ModelFactory
from ..config.environment import HardwareConfig

class ModelAdaptation:
    def __init__(self, hardware_config: HardwareConfig):
        self.hardware_config = hardware_config
        self.model_factory = ModelFactory()
        
    def adapt_model(
        self,
        base_model_name: str,
        train_data: Dict,
        eval_data: Optional[Dict] = None,
        custom_lora_config: Optional[Dict] = None
    ):
        """
        Main entry point for model adaptation
        """
        try:
            # Initialize base model
            model, tokenizer = self.model_factory.create_model(
                base_model_name, 
                self.hardware_config
            )
            
            # Create LoRA trainer
            lora_config = LoRAConfig(**custom_lora_config) if custom_lora_config else None
            trainer = LoRATrainer(
                model,
                tokenizer,
                self.hardware_config,
                lora_config
            )
            
            # Train the model
            result = trainer.train(
                train_data,
                eval_data,
                output_dir=f"./adapted_models/{base_model_name}_lora"
            )
            
            return {
                'status': 'success',
                'training_results': result
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            } 