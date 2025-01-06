import torch
from transformers import TrainingArguments
import logging
from typing import Dict, Any
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pytorch_native_features():
    """Test PyTorch's native features on MPS."""
    if not torch.backends.mps.is_available():
        logger.info("MPS not available on this system")
        return
    
    results: Dict[str, Any] = {}
    
    # Basic PyTorch Operations
    try:
        device = torch.device('mps')
        # Test tensor operations
        x = torch.randn(2, 2, device=device)
        y = torch.randn(2, 2, device=device)
        _ = x @ y  # Matrix multiplication
        results["basic_operations"] = "✓ Supported"
        
        # Test FP16
        x_half = x.half()
        results["fp16_support"] = "✓ Supported"
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64, 128]
        max_successful = 0
        for bs in batch_sizes:
            try:
                x = torch.randn(bs, 512, 768, device=device)
                del x
                max_successful = bs
            except RuntimeError:
                break
        results["max_batch_size"] = f"Tested up to {max_successful}"
        
    except Exception as e:
        results["pytorch_native"] = f"✗ Error: {str(e)}"
    
    logger.info("\nPyTorch Native Features:")
    for feature, result in results.items():
        logger.info(f"- {feature}: {result}")

def test_training_features():
    """Test training-specific features and optimizations."""
    if not torch.backends.mps.is_available():
        logger.info("MPS not available on this system")
        return
    
    results = {}
    
    # Test 1: PyTorch Native Training Features
    try:
        # Test native optimizers
        from torch.optim import AdamW, SGD
        model = torch.nn.Linear(10, 2).to('mps')
        
        optimizers = {
            "AdamW": AdamW(model.parameters(), lr=1e-5),
            "SGD": SGD(model.parameters(), lr=1e-5)
        }
        results["native_optimizers"] = "✓ Supported: " + ", ".join(optimizers.keys())
        
        # Test gradient scaling (important for mixed precision)
        scaler = torch.cuda.amp.GradScaler()
        results["gradient_scaling"] = "✓ Supported"
    except Exception as e:
        results["native_training"] = f"✗ Error: {str(e)}"

    # Test 2: HuggingFace Integration
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        model.to('mps')
        results["hf_models"] = "✓ Supported"
    except Exception as e:
        results["hf_models"] = f"✗ Error: {str(e)}"

    # Test 3: LoRA Features
    try:
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        results["lora_basic"] = "✓ Supported"
    except Exception as e:
        results["lora_basic"] = f"✗ Error: {str(e)}"

    # Test 4: Advanced Optimization Libraries
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
        results["8bit_optimization"] = "✓ Supported"
    except Exception as e:
        results["8bit_optimization"] = f"✗ Not supported: {str(e)}"

    try:
        import optuna
        results["optuna_support"] = "✓ Supported (CPU-based hyperparameter optimization)"
    except ImportError:
        results["optuna_support"] = "✗ Not installed"

    # Test 5: Optimum Integration
    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        results["optimum_bettertransformer"] = "✓ Supported"
    except Exception as e:
        results["optimum_bettertransformer"] = f"✗ Error: {str(e)}"

    logger.info("\nTraining and Optimization Features:")
    for feature, result in results.items():
        logger.info(f"- {feature}: {result}")

def test_inference_features():
    """Test inference-specific features and optimizations."""
    if not torch.backends.mps.is_available():
        logger.info("MPS not available on this system")
        return
    
    results = {}
    
    # Test 1: Basic Inference
    try:
        model = torch.nn.Linear(10, 2).to('mps')
        x = torch.randn(1, 10).to('mps')
        with torch.no_grad():
            _ = model(x)
        results["basic_inference"] = "✓ Supported"
    except Exception as e:
        results["basic_inference"] = f"✗ Error: {str(e)}"

    # Test 2: Batch Inference
    try:
        x = torch.randn(32, 10).to('mps')
        with torch.no_grad():
            _ = model(x)
        results["batch_inference"] = "✓ Supported"
    except Exception as e:
        results["batch_inference"] = f"✗ Error: {str(e)}"

    logger.info("\nInference Features:")
    for feature, result in results.items():
        logger.info(f"- {feature}: {result}")

if __name__ == "__main__":
    logger.info("Testing MPS Compatibility...")
    test_pytorch_native_features()
    test_training_features()
    test_inference_features() 