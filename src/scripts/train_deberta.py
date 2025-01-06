import argparse
import logging
from pathlib import Path
from typing import Optional
import os
from datasets import load_from_disk
from src.models.deberta_trainer import DebertaTrainer
from src.config.logging_config import setup_logging
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeBERTa model using LoRA on sentiment analysis data"
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HuggingFace dataset directory"
    )
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
    
    # Existing LoRA overrides
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    
    # Extended: specify config files
    parser.add_argument("--config-lora", type=str, default=None,
                        help="Path to a custom lora_config.json")
    parser.add_argument("--config-training", type=str, default=None,
                        help="Path to a custom training_config.json")

    # Basic training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    return parser.parse_args()

def load_lora_config(config_path: str) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Provided LoRA config path does not exist: {path}")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def load_training_config(config_path: str) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Provided training config path does not exist: {path}")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def validate_dataset_path(dataset_path: str) -> Optional[Path]:
    path = Path(dataset_path)
    if not path.exists():
        logger.error(f"Dataset path does not exist: {path}")
        return None
        
    required_items = ["dataset_dict.json", "train", "validation", "test"]
    missing_items = [item for item in required_items if not (path / item).exists()]
    if missing_items:
        logger.error(f"Dataset at {path} is missing required items: {missing_items}")
        return None
    
    dataset = load_from_disk(str(path))
    required_columns = ["text", "labels", "id", "original_id"]
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            logger.error(f"Dataset is missing the '{split_name}' split.")
            return None
        split_cols = dataset[split_name].column_names
        for col in required_columns:
            if col not in split_cols:
                logger.error(
                    f"Dataset split '{split_name}' lacks '{col}'. Found columns: {split_cols}"
                )
                return None
    return path

def verify_environment():
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
                    f"{package} version {installed} is installed, but {min_version} or higher is recommended."
                )
        return True
    except Exception as e:
        logger.error(f"Environment verification failed: {e}")
        return False

def main():
    args = parse_args()
    
    if not verify_environment():
        logger.error("Environment verification failed. Please check your dependencies.")
        return 1
    
    dataset_path = validate_dataset_path(args.dataset_path)
    if dataset_path is None:
        return 1

    # Load config files if specified
    lora_config_dict = load_lora_config(args.config_lora)
    training_config_dict = load_training_config(args.config_training)

    # Merge partial CLI overrides into the LoRA config
    if args.lora_rank is not None:
        lora_config_dict["r"] = args.lora_rank
    if args.lora_alpha is not None:
        lora_config_dict["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        lora_config_dict["lora_dropout"] = args.lora_dropout
    if lora_config_dict:
        # ensure essential fields
        lora_config_dict.setdefault("bias", "none")
        lora_config_dict.setdefault("task_type", "SEQ_CLS")

    # Merge CLI overrides into training config
    if args.epochs is not None:
        training_config_dict["num_train_epochs"] = args.epochs
    if args.learning_rate is not None:
        training_config_dict["learning_rate"] = args.learning_rate

    logger.info(f"Initializing DeBERTa trainer with model: {args.model_name}")
    try:
        trainer = DebertaTrainer(
            dataset_path=str(dataset_path),
            model_name=args.model_name,
            lora_config=lora_config_dict if lora_config_dict else None,
            training_config=training_config_dict if training_config_dict else None
        )
        
        logger.info("Starting training...")
        result = trainer.train()
        
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
