from dataclasses import dataclass
from typing import Optional, List
import torch
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

@dataclass
class LoRAConfig:
    r: int = 8  # LoRA rank
    alpha: int = 16  # LoRA scaling
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    
    @classmethod
    def get_optimal_config(cls, hardware_config: HardwareConfig):
        """Returns optimal LoRA config based on hardware"""
        logger.info(f"Configuring LoRA for hardware: {hardware_config.device}")
        
        if hardware_config.memory_limit >= 48:  # Our M4 Pro case
            logger.info("Using high-memory configuration")
            return cls(
                r=32,  # Larger rank for more capacity
                alpha=64,
                dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                task_type="CAUSAL_LM"
            )
        # Fallback for lower memory systems
        logger.info("Using standard memory configuration")
        return cls() 