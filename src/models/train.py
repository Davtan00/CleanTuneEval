import os
from datasets import load_from_disk
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Literal
from datetime import datetime
from .model_factory import ModelFactory
from .lora_trainer import LoRATrainer
from ..config.logging_config import setup_logging

# Set tokenizer parallelism explicitly
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logging()

class ModelTrainer:
    def __init__(self, 
                 base_model: str = "microsoft/deberta-v3-base",
                 tuning_method: str = "lora",
                 classification_type: str = "three_way",
                 use_research_weights: bool = False):
        self.base_model = base_model
        self.tuning_method = tuning_method
        self.classification_type = classification_type
        self.use_research_weights = use_research_weights
        
        # Initialize model adapter based on tuning method
        if tuning_method == "lora":
            model_factory = ModelFactory(model_name=base_model)
            self.adapter = LoRATrainer(model_factory)
        else:
            raise ValueError(f"Unsupported tuning method: {tuning_method}")
            
        # Setup model save directory
        self.model_save_dir = Path("src/models/storage") / \
                             Path(base_model).name / \
                             tuning_method / \
                             classification_type / \
                             datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def train(self, dataset_path: str) -> Dict[str, Any]:
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
        
        # Train model
        logger.info(f"Starting training with configuration:")
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Tuning method: {self.tuning_method}")
        logger.info(f"Classification: {self.classification_type}")
        
        result = self.adapter.train(
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            output_dir=str(self.model_save_dir),
            model_name=self.base_model
        )
        
        return result

def validate_dataset_splits(dataset):
    """Check for potential data quality issues"""
    train_texts = set(dataset['train']['text'])
    val_texts = set(dataset['validation']['text'])
    
    # Check overlap
    overlap = train_texts.intersection(val_texts)
    overlap_percentage = len(overlap) / len(val_texts) * 100
    
    logger.info("\nDataset Validation:")
    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Val samples: {len(val_texts)}")
    logger.info(f"Overlap: {len(overlap)} samples ({overlap_percentage:.2f}%)")
    
    # Sample check
    if len(overlap) > 0:
        logger.warning("Example overlapping texts:")
        for text in list(overlap)[:3]:
            logger.warning(f"- {text[:100]}...")
            
    return overlap_percentage < 1  # Flag if overlap > 1%

def train_model(dataset_path: str,
                base_model: str = "microsoft/deberta-v3-base",
                tuning_method: str = "lora",
                classification_type: str = "three_way",
                use_research_weights: bool = False):
    """
    Convenience function for training models
    """
    trainer = ModelTrainer(
        base_model=base_model,
        tuning_method=tuning_method,
        classification_type=classification_type,
        use_research_weights=use_research_weights
    )
    result = trainer.train(dataset_path)
    
    # Add completion summary
    if result["status"] == "success":
        logger.info("\n" + "="*50)
        logger.info("Training Completed Successfully!")
        logger.info("="*50)
        logger.info(f"Model saved at: {trainer.model_save_dir}")
        logger.info("\nFinal Metrics:")
        logger.info(f"Accuracy: {result['metrics']['eval_accuracy']:.4f}")
        logger.info(f"Macro F1: {result['metrics']['eval_macro_f1']:.4f}")
        logger.info("\nPer-class F1 Scores:")
        logger.info(f"Negative: {result['metrics']['eval_negative_f1']:.4f}")
        logger.info(f"Neutral:  {result['metrics']['eval_neutral_f1']:.4f}")
        logger.info(f"Positive: {result['metrics']['eval_positive_f1']:.4f}")
        logger.info("\nConfusion Matrix:")
        for row in result['metrics']['eval_confusion_matrix']:
            logger.info(str(row))
        logger.info("="*50)
    else:
        logger.error(f"Training failed: {result['message']}")

    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model with LoRA adaptation')
    parser.add_argument('--dataset', type=str, 
                       help='Path to the dataset')
    parser.add_argument('--use-research-weights', action='store_true',
                       help='Use domain-specific research weights during training')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information about the dataset')
    
    args = parser.parse_args()
    
    if args.debug:
        dataset = load_from_disk(args.dataset)
        print("\nDataset structure:")
        print(dataset)
        print("\nFirst example:")
        print(dataset['train'][0])
        print("\nFeatures:")
        print(dataset['train'].features)
    
    print("\nStarting model training...")
    train_model(args.dataset) 