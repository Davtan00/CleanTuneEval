from dataclasses import dataclass
from typing import Optional
from peft import LoraConfig

@dataclass
class LoRAParameters:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "SEQ_CLS"
    
def create_lora_config(params: Optional[LoRAParameters] = None) -> LoraConfig:
    if params is None:
        params = LoRAParameters()
        
    return LoraConfig(
        r=params.r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        bias=params.bias,
        task_type=params.task_type
    ) 