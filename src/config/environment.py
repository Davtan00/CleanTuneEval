import torch
import platform
from dataclasses import dataclass
from typing import Literal
from ..config.logging_config import setup_logging

logger = setup_logging()

@dataclass
class HardwareConfig:
    device: str
    n_cores: int
    memory_limit: int  # in GB
    use_mps: bool = False
    preserve_distribution: bool = False
    use_research_weights: bool = False
    optimizer_type: Literal["cuda", "cpu", "mps"] = "cpu"
    
    def __init__(
        self,
        preserve_distribution: bool = False,
        force_cpu: bool = False
    ):
        """
        Initialize hardware config with automatic detection.
        
        Args:
            preserve_distribution: Whether to preserve dataset distribution
            force_cpu: Force CPU usage even if accelerators are available
        """
        self.preserve_distribution = preserve_distribution
        
        if force_cpu:
            self._setup_cpu()
            return
            
        # Try MPS (Apple Silicon) first
        if platform.processor() == 'arm' and torch.backends.mps.is_available():
            self._setup_mps()
        # Then try CUDA
        elif torch.cuda.is_available():
            self._setup_cuda()
        # Fallback to CPU
        else:
            self._setup_cpu()
            
        self._log_config()
    
    def _setup_mps(self):
        """Configure for Apple Silicon MPS"""
        self.device = 'mps'
        self.use_mps = True
        self.optimizer_type = 'mps'
        self.n_cores = 14  # M4 Pro specific
        self.memory_limit = 48  # GB
        logger.info("Using Apple Silicon MPS acceleration")
    
    def _setup_cuda(self):
        """Configure for NVIDIA CUDA"""
        try:
            import bitsandbytes as bnb
            self.device = 'cuda'
            self.optimizer_type = 'cuda'
            self.use_mps = False
            self.n_cores = torch.cuda.device_count() * 2
            self.memory_limit = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            logger.info("Using NVIDIA CUDA acceleration")
        except ImportError:
            logger.warning("CUDA available but bitsandbytes not installed. Falling back to CPU.")
            self._setup_cpu()
    
    def _setup_cpu(self):
        """Configure for CPU"""
        import multiprocessing
        self.device = 'cpu'
        self.use_mps = False
        self.optimizer_type = 'cpu'
        self.n_cores = multiprocessing.cpu_count()
        self.memory_limit = 16  # Conservative default
        logger.info("Using CPU processing")
    
    def _log_config(self):
        """Log the current configuration"""
        logger.info("Hardware configuration:")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Cores: {self.n_cores}")
        logger.info(f"- Memory limit: {self.memory_limit}GB")
        logger.info(f"- MPS enabled: {self.use_mps}")
        logger.info(f"- Optimizer type: {self.optimizer_type}")
        logger.info(f"- Preserve distribution: {self.preserve_distribution}") 