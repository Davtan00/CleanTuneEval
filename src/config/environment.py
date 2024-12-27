import torch
import platform
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareConfig:
    device: str
    n_cores: int
    memory_limit: int  # in GB
    use_mps: bool = False
    
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
        return cls(device='cpu', n_cores=1, memory_limit=8) 