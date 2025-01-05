import torch
import platform
from dataclasses import dataclass
from typing import Optional, Literal
from ..config.logging_config import setup_logging

logger = setup_logging()

@dataclass
class HardwareConfig:
    device: str
    n_cores: int
    memory_limit: int  # in GB
    use_mps: bool = False
    preserve_distribution: bool = False  # Controls dataset filtering
    use_research_weights: bool = False   # Controls training weights
    optimizer_type: Literal["cuda", "cpu", "mps"] = "cpu"
    
    def __init__(
        self,
        device: str = 'cpu',
        n_cores: int = 14,
        memory_limit: int = 48,
        use_mps: bool = False,
        preserve_distribution: bool = False
    ):
        self.device = device
        self.n_cores = n_cores
        self.memory_limit = memory_limit
        self.use_mps = use_mps
        self.preserve_distribution = preserve_distribution
        
        # Determine optimizer type based on hardware
        if torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                self.optimizer_type = "cuda"
            except ImportError:
                logger.warning("CUDA available but bitsandbytes not installed. Falling back to CPU optimizer.")
                self.optimizer_type = "cpu"
        elif self.use_mps:
            self.optimizer_type = "mps"
        else:
            self.optimizer_type = "cpu"
        
        logger.info(f"Hardware config initialized:")
        logger.info(f"- Device: {device}")
        logger.info(f"- Cores: {n_cores}")
        logger.info(f"- Memory limit: {memory_limit}GB")
        logger.info(f"- MPS enabled: {use_mps}")
        logger.info(f"- Preserve distribution: {preserve_distribution}")
        logger.info(f"- Optimizer type: {self.optimizer_type}")
    
    @classmethod
    def detect_hardware(cls) -> 'HardwareConfig':
        if platform.processor() == 'arm':  # Apple Silicon
            use_mps = torch.backends.mps.is_available()
            device = 'mps' if use_mps else 'cpu'
            return cls(
                device=device,
                n_cores=14,  # M4 Pro specific
                memory_limit=48,  # GB
                use_mps=use_mps
            )
        elif torch.cuda.is_available():
            return cls(
                device='cuda',
                n_cores=torch.cuda.device_count() * 2,
                memory_limit=torch.cuda.get_device_properties(0).total_memory // (1024**3),
                use_mps=False
            )
        return cls(device='cpu', n_cores=1, memory_limit=8) 