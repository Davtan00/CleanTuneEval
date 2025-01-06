import torch
import platform
from dataclasses import dataclass
from typing import Literal
from ..config.logging_config import setup_logging

logger = setup_logging()

@dataclass
class HardwareConfig:
    device: str = "cpu"
    n_cores: int = 1
    memory_limit: int = 16
    use_mps: bool = False
    optimizer_type: Literal["cuda", "cpu", "mps"] = "cpu"
    preserve_distribution: bool = False
    use_research_weights: bool = False
    
    def __init__(
        self,
        preserve_distribution: bool = False,
        force_cpu: bool = False
    ):
        self.preserve_distribution = preserve_distribution
        
        if force_cpu:
            self._setup_cpu()
        else:
            if platform.processor() == 'arm' and torch.backends.mps.is_available():
                self._setup_mps()
            elif torch.cuda.is_available():
                self._setup_cuda()
            else:
                self._setup_cpu()
        
        self._log_config()
    
    def _setup_mps(self):
        self.device = 'mps'
        self.use_mps = True
        self.optimizer_type = 'mps'
        self.n_cores = 14
        self.memory_limit = 48
        logger.info("Using Apple Silicon MPS acceleration")
    
    def _setup_cuda(self):
        try:
            import bitsandbytes
            if torch.cuda.device_count() == 0:
                raise RuntimeError("No actual CUDA device found.")
            self.device = 'cuda'
            self.optimizer_type = 'cuda'
            self.use_mps = False
            self.n_cores = torch.cuda.device_count() * 2
            self.memory_limit = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            logger.info("Using NVIDIA CUDA acceleration with bitsandbytes")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to set up CUDA/bitsandbytes: {e}. Falling back to CPU.")
            self._setup_cpu()
    
    def _setup_cpu(self):
        import multiprocessing
        self.device = 'cpu'
        self.use_mps = False
        self.optimizer_type = 'cpu'
        self.n_cores = multiprocessing.cpu_count()
        self.memory_limit = 16
        logger.info("Using CPU processing")
    
    def _log_config(self):
        logger.info("Hardware configuration:")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Cores: {self.n_cores}")
        logger.info(f"- Memory limit: {self.memory_limit}GB")
        logger.info(f"- MPS enabled: {self.use_mps}")
        logger.info(f"- Optimizer type: {self.optimizer_type}")
        logger.info(f"- Preserve distribution: {self.preserve_distribution}")
    
    def test_mps_compatibility(self):
        """Test MPS compatibility for various operations."""
        if not self.use_mps:
            logger.info("MPS not enabled, skipping compatibility test")
            return
        
        import torch
        device = torch.device('mps')
        test_results = {}
        
        operations = {
            'basic_tensor': lambda: torch.ones(2, 2).to(device),
            'matmul': lambda: torch.matmul(torch.ones(2, 2).to(device), torch.ones(2, 2).to(device)),
            'conv2d': lambda: torch.nn.functional.conv2d(
                torch.ones(1, 1, 5, 5).to(device),
                torch.ones(1, 1, 3, 3).to(device)
            ),
            'attention': lambda: torch.nn.functional.scaled_dot_product_attention(
                torch.ones(2, 4, 8).to(device),
                torch.ones(2, 4, 8).to(device),
                torch.ones(2, 4, 8).to(device)
            ),
            'quantization': lambda: torch.quantize_per_tensor(
                torch.ones(2, 2),
                scale=1.0,
                zero_point=0,
                dtype=torch.qint8
            )
        }
        
        for name, op in operations.items():
            try:
                _ = op()
                test_results[name] = "✓ Supported"
            except Exception as e:
                test_results[name] = f"✗ Not supported: {str(e)}"
        
        logger.info("MPS Compatibility Test Results:")
        for op, result in test_results.items():
            logger.info(f"- {op}: {result}")
    
    def check_mps_limitations(self):
        """Check and log MPS limitations for the current setup."""
        if not self.use_mps:
            return
        
        limitations = {
            "8-bit Quantization": "Limited support, falls back to CPU",
            "Sparse Operations": "Limited support",
            "Complex Attention": "Some operations may fall back to CPU",
            "Memory Management": "Different from CUDA, may require explicit management",
            "Debugging Tools": "Limited compared to CUDA",
        }
        
        logger.info("MPS Known Limitations:")
        for feature, limitation in limitations.items():
            logger.info(f"- {feature}: {limitation}")