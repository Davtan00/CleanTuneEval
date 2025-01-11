import argparse
import json
from pathlib import Path
import logging
from src.evaluation.lora_model_evaluator import LoraModelEvaluator
from src.config.environment import HardwareConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find checkpoint with best combined_metric from trainer_state.json."""
    # Get all checkpoint directories
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    
    if not checkpoints:
        raise ValueError(f"No checkpoint directories found in {checkpoint_dir}")
    
    best_checkpoint = None
    best_metric = -float('inf')
    
    for checkpoint_dir in checkpoints:
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.exists():
            logger.warning(f"No trainer_state.json found in {checkpoint_dir}")
            continue
            
        with open(state_path) as f:
            state = json.load(f)
            
        # Get the best metric from the log history
        log_history = state.get('log_history', [])
        for log in log_history:
            if 'eval_combined_metric' in log:
                metric = log['eval_combined_metric']
                if metric > best_metric:
                    best_metric = metric
                    best_checkpoint = checkpoint_dir
    
    if not best_checkpoint:
        raise ValueError(f"No valid checkpoints with metrics found in {checkpoint_dir}")
        
    logger.info(f"Found best checkpoint: {best_checkpoint}")
    logger.info(f"Best combined metric: {best_metric:.4f}")
    
    return best_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-tuned DeBERTa model")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="src/models/storage/deberta-v3-base/lora/three_way/20250107_004251",
        help="Directory containing LoRA checkpoints"
    )
    parser.add_argument(
        "--specific-checkpoint",
        type=str,
        help="Specific checkpoint to evaluate (e.g., checkpoint-522). If not provided, will use best checkpoint."
    )
    parser.add_argument(
        "--dataset-path",#ecommerce_7k_20250106_234227
        type=str,
        default="src/data/datasets/ecommerce_7k_20250106_234227",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Base model name from HuggingFace"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Determine checkpoint path
    if args.specific_checkpoint:
        checkpoint_path = checkpoint_dir / args.specific_checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Specific checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
    
    # Initialize hardware config
    hardware_config = HardwareConfig(force_cpu=args.force_cpu)
    
    # Run evaluation
    evaluator = LoraModelEvaluator(
        base_model_name=args.base_model,
        checkpoint_path=str(checkpoint_path),
        dataset_path=args.dataset_path,
        hardware_config=hardware_config
    )
    
    results = evaluator.evaluate()
    
    # Save results
    output_dir = Path("src/evaluation/results/lora_evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = checkpoint_path.name
    dataset_name = Path(args.dataset_path).name
    output_path = output_dir / f"evaluation_{dataset_name}_{checkpoint_name}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Model: {args.base_model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Device: {hardware_config.device}")
    print("\nMetrics:")
    print(f"Combined Metric: {results['metrics']['eval_combined_metric']:.4f}")
    print(f"F1 Score: {results['metrics']['eval_f1']:.4f}")
    print(f"Balanced Accuracy: {results['metrics']['eval_balanced_accuracy']:.4f}")

if __name__ == "__main__":
    main() 