import argparse
import logging
from pathlib import Path
from typing import Optional

from src.models.deberta_trainer import DebertaTrainer
from src.config.logging_config import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeBERTa model using LoRA on sentiment analysis data"
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HuggingFace dataset directory (e.g., src/data/datasets/technology_7k_*)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Base model to use (default: microsoft/deberta-v3-base)"
    )
    
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Rank for LoRA adaptation (if not specified, uses trainer defaults)"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Alpha scaling for LoRA (if not specified, uses trainer defaults)"
    )
    
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="Dropout for LoRA layers (if not specified, uses trainer defaults)"
    )
    
    return parser.parse_args()

def validate_dataset_path(dataset_path: str) -> Optional[Path]:
    """Validate that the dataset path exists and has the expected structure."""
    path = Path(dataset_path)
    
    if not path.exists():
        logger.error(f"Dataset path does not exist: {path}")
        return None
        
    # Check for required files/directories
    required_items = ["dataset_dict.json", "train", "validation", "test"]
    missing_items = [item for item in required_items if not (path / item).exists()]
    
    if missing_items:
        logger.error(f"Dataset at {path} is missing required items: {missing_items}")
        return None
        
    return path

def verify_environment():
    """Verify that all required packages are installed with correct versions."""
    try:
        import pkg_resources
        
        requirements = {
            'transformers': '4.47.1',
            'sentencepiece': '0.1.99',
            'tokenizers': '0.21.0'
        }
        
        for package, min_version in requirements.items():
            installed = pkg_resources.get_distribution(package).version
            if pkg_resources.parse_version(installed) < pkg_resources.parse_version(min_version):
                logger.warning(
                    f"{package} version {installed} is installed, but {min_version} or higher "
                    "is recommended for DeBERTa v3"
                )
        return True
        
    except Exception as e:
        logger.error(f"Environment verification failed: {e}")
        return False

def main():
    args = parse_args()
    
    # Verify environment first
    if not verify_environment():
        logger.error("Environment verification failed. Please check your dependencies.")
        return 1
    
    # Validate dataset path
    dataset_path = validate_dataset_path(args.dataset_path)
    if dataset_path is None:
        return 1
    
    # Only create lora_config if ALL LoRA parameters are explicitly specified
    lora_config = None
    if all(v is not None for v in [args.lora_rank, args.lora_alpha, args.lora_dropout]):
        lora_config = {
            "r": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "SEQ_CLS"
        }
        logger.info("Using custom LoRA configuration")
    else:
        if any(v is not None for v in [args.lora_rank, args.lora_alpha, args.lora_dropout]):
            logger.warning("Partial LoRA configuration provided. Using trainer defaults instead.")
        else:
            logger.info("No LoRA configuration provided. Using trainer defaults.")
    
    try:
        # Initialize trainer
        logger.info(f"Initializing DeBERTa trainer with model: {args.model_name}")
        trainer = DebertaTrainer(
            dataset_path=str(dataset_path),
            model_name=args.model_name,
            lora_config=lora_config  # Will be None if not all params specified
        )
        
        # Train model
        logger.info("Starting training...")
        result = trainer.train()
        
        # Check training result
        if result["status"] == "success":
            logger.info("Training completed successfully!")
            logger.info(f"Model saved to: {result['model_path']}")
            logger.info("Test metrics:")
            for metric, value in result["test_metrics"].items():
                logger.info(f"  {metric}: {value:.4f}")
            return 0
        else:
            logger.error(f"Training failed: {result['message']}")
            return 1
            
    except Exception as e:
        logger.exception("Unexpected error during training")
        return 1

if __name__ == "__main__":
    exit(main()) 