import argparse
import logging
import os
import json
from pathlib import Path
from typing import Optional
from datasets import load_from_disk

from src.models.deberta_trainer import DebertaTrainer
from src.config.logging_config import setup_logging
from src.config.environment import HardwareConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeBERTa model with LoRA on sentiment analysis data. "
                    "Config files can be specified or will default to standard paths."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the HuggingFace dataset directory (e.g., src/data/datasets/technology_7k_*)"
    )
    parser.add_argument(
        "--lora-config-file",
        type=str,
        default="src/models/config/lora_config.json",
        help="Path to the LoRA config JSON (defaults to src/models/config/lora_config.json)"
    )
    parser.add_argument(
        "--training-config-file",
        type=str,
        default="src/models/config/training_config.json",
        help="Path to the training config JSON (defaults to src/models/config/training_config.json)"
    )
    args = parser.parse_args()

    # Add warning messages for default config paths
    if args.lora_config_file == "src/models/config/lora_config.json":
        logger.warning("⚠️  Using default LoRA config path: src/models/config/lora_config.json")
        logger.warning("   To use a custom config, specify --lora-config-file")
    
    if args.training_config_file == "src/models/config/training_config.json":
        logger.warning("⚠️  Using default training config path: src/models/config/training_config.json")
        logger.warning("   To use a custom config, specify --training-config-file")

    return args

def validate_dataset_path(dataset_path: str) -> Optional[Path]:
    """
    Verify that the dataset path exists and has the expected splits and columns.
    Requires ['text', 'labels', 'id', 'original_id'] in each of train/validation/test.
    """
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
                    f"Dataset split '{split_name}' lacks required column '{col}'. "
                    f"Found columns: {split_cols}"
                )
                return None

    return path

def verify_environment() -> bool:
    """Check essential packages and versions to ensure compatibility."""
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
                    f"{package} version {installed} is installed, "
                    f"but {min_version} or higher is recommended."
                )
        return True
    except Exception as e:
        logger.error(f"Environment verification failed: {e}")
        return False

def load_config_or_fail(config_path: str, config_type: str) -> dict:
    """
    Load a JSON config from the provided path.
    If not found or invalid, log an error and exit immediately.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"{config_type} not found at: {config_path}")
        exit(1)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {config_type} from {config_path}: {e}")
        exit(1)

def main():
    args = parse_args()

    # 1. Environment check
    if not verify_environment():
        logger.error("Environment verification failed. Terminating.")
        return 1
    
    hardware_config = HardwareConfig()
    #hardware_config.check_mps_limitations()

    # 2. Dataset validation
    dataset_path = validate_dataset_path(args.dataset_path)
    if dataset_path is None:
        logger.error("Dataset path validation failed. Terminating.")
        return 1

    # 3. Load configs
    lora_config = load_config_or_fail(args.lora_config_file, "LoRA config")
    training_config = load_config_or_fail(args.training_config_file, "Training config")

    # 4. Initialize trainer
    try:
        trainer = DebertaTrainer(
            dataset_path=str(dataset_path),
            lora_config=lora_config,
            training_config=training_config,
            hardware_config=hardware_config
        )
    except Exception as e:
        logger.exception(f"Failed to initialize trainer: {e}")
        return 1

    # 5. Train
    logger.info("Starting training...")
    try:
        result = trainer.train()
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        return 1

    # 6. Check training result
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

if __name__ == "__main__":
    exit(main())
