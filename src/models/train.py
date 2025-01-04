from datasets import load_from_disk
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Literal
from .adaptation import ModelAdapter
from .lora_config import LoRAParameters
from ..config.logging_config import setup_logging

logger = setup_logging()

class ModelTrainer:
    def __init__(self, 
                 base_model: str = "microsoft/deberta-v3-base",
                 tuning_method: str = "lora",
                 classification_type: Literal["binary", "three_way"] = "three_way"):
        """
        Initialize trainer with flexible model configuration
        
        Args:
            base_model: Base model identifier (e.g., "microsoft/deberta-v3-base", "roberta-base")
            tuning_method: Fine-tuning method (e.g., "lora", "full", "prefix")
            classification_type: Type of sentiment classification
        """
        self.base_path = Path(__file__).parent.parent
        self.base_model = base_model
        self.tuning_method = tuning_method
        self.classification_type = classification_type
        
        # Create structured output path: models/storage/{model_name}/{tuning_method}/{classification_type}
        model_name = base_model.split('/')[-1]
        self.model_save_dir = (self.base_path / "models/storage" 
                             / model_name 
                             / tuning_method 
                             / classification_type)
        
        self.adapter = ModelAdapter()
        
    def train(self, 
              dataset_path: str,
              lora_params: Optional[LoRAParameters] = None) -> Dict[str, Any]:
        """
        Train model using specified configuration
        """
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            dataset = load_from_disk(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return {"status": "error", "message": f"Dataset loading failed: {str(e)}"}
            
        logger.info(f"Dataset loaded with {len(dataset['train'])} training samples")
        
        if lora_params is None:
            lora_params = LoRAParameters(
                r=16,                     
                lora_alpha=32,            
                lora_dropout=0.1,         
                task_type="SEQ_CLS"       
            )
        
        # Create output directory if it doesn't exist
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        logger.info(f"Starting training with configuration:")
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Tuning method: {self.tuning_method}")
        logger.info(f"Classification: {self.classification_type}")
        
        result = self.adapter.adapt_model(
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            output_dir=str(self.model_save_dir),
            lora_params=lora_params
        )
        
        if result["status"] == "success":
            logger.info(f"Training completed successfully")
            logger.info(f"Model saved at: {self.model_save_dir}")
            logger.info("Metrics:")
            for metric_name, value in result["metrics"].items():
                logger.info(f"{metric_name}: {value}")
        
        return result

def train_model(dataset_path: str,
                base_model: str = "microsoft/deberta-v3-base",
                tuning_method: str = "lora",
                classification_type: str = "three_way"):
    """
    Convenience function for training models
    """
    trainer = ModelTrainer(
        base_model=base_model,
        tuning_method=tuning_method,
        classification_type=classification_type
    )
    return trainer.train(dataset_path)

if __name__ == "__main__":
    # Example usage
    dataset_path = "src/data/datasets/technology_7k_20250104_194358_adjusted_ccc"
    train_model(dataset_path) 