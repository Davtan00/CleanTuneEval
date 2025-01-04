from typing import Dict, Any, Optional
from .model_factory import ModelFactory
from .lora_trainer import LoRATrainer
from .lora_config import LoRAParameters
import logging

logger = logging.getLogger(__name__)

class ModelAdapter:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.trainer = LoRATrainer(self.model_factory)
        
    def adapt_model(self,
                   train_dataset,
                   eval_dataset,
                   output_dir: str = "./results",
                   lora_params: Optional[LoRAParameters] = None) -> Dict[str, Any]:
        
        try:
            result = self.trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                lora_params=lora_params,
                output_dir=output_dir
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Model adaptation failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            } 