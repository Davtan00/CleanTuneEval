from typing import Dict, Any
import logging
from functools import wraps
import warnings

logger = logging.getLogger(__name__)

def warn_cuda_only(func):
    """Decorator to warn when CUDA-specific features are used."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import bitsandbytes
            warnings.warn(
                "This feature uses bitsandbytes which is CUDA-only. "
                "Ensure you have MPS/CPU fallback implemented.",
                UserWarning
            )
        except ImportError:
            pass
        return func(*args, **kwargs)
    return wrapper

class TrainingConfigurator:
    """Handles platform-specific training configurations."""
    
    @staticmethod
    def get_optimizer_config(device_type: str) -> Dict[str, Any]:
        """
        Get platform-specific optimizer and training configuration.
        """
        base_config = {
            "per_device_train_batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1
        }
        
        configs = {
            "cuda": TrainingConfigurator._get_cuda_config,
            "mps": TrainingConfigurator._get_mps_config,
            "cpu": TrainingConfigurator._get_cpu_config
        }
        
        config_func = configs.get(device_type, TrainingConfigurator._get_cpu_config)
        return config_func(base_config)
    
    @staticmethod
    @warn_cuda_only
    def _get_cuda_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import bitsandbytes as bnb
            return {
                **base_config,
                "optim": "adamw_bnb_8bit",
                "fp16": True,
                "dataloader_pin_memory": True
            }
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")
            return {
                **base_config,
                "optim": "adamw_torch",
                "fp16": True,
                "dataloader_pin_memory": True
            }
    
    @staticmethod
    def _get_mps_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **base_config,
            "optim": "adamw_torch",
            "fp16": False,
            "dataloader_pin_memory": False,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 2
        }
    
    @staticmethod
    def _get_cpu_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **base_config,
            "optim": "adamw_torch",
            "fp16": False,
            "dataloader_pin_memory": True
        } 