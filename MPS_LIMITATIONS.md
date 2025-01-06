# PyTorch MPS (Metal Performance Shaders) Limitations

This document outlines verified limitations affecting cross-platform ML development with PyTorch's MPS backend.

## Library Compatibility Status

### ❌ Incompatible Libraries
1. **Bitsandbytes**
   - Not compatible with MPS backend
   - Error: "compiled without GPU support"
   - Impact: Cannot use:
     - 8-bit quantization
     - 8-bit optimizers
     - GPU quantization features
   - Source: Verified through testing and [bitsandbytes documentation](https://github.com/TimDettmers/bitsandbytes)

### ✅ Compatible Libraries/Features
1. **PyTorch Core (2.5.1)**
   - Basic tensor operations
   - Matrix multiplication
   - Convolution operations
   - FP16 support
   - DataLoader functionality
   - Gradient checkpointing

2. **Hugging Face Transformers**
   - Basic model operations
   - LoRA (through PEFT)
   - Standard optimizers (AdamW)

## Development Implications

### Cross-Platform Code Requirements
When developing code that needs to work on both CUDA and MPS:

1. **Quantization**
   ```python
   # Need separate paths for CUDA vs MPS
   if device == "cuda":
       # Can use bitsandbytes quantization
   elif device == "mps":
       # Must use standard PyTorch or fall back to CPU
   ```

2. **Optimizers**
   ```python
   # CUDA can use specialized optimizers
   if device == "cuda":
       from bitsandbytes.optim import AdamW8bit
   # MPS must use standard optimizers
   elif device == "mps":
       from torch.optim import AdamW
   ```

## Optimization Strategies

### Optimizer Selection
```python
def get_optimizer_config(device_type: str) -> Dict[str, Any]:
    """
    Get platform-specific optimizer configuration.
    """
    if device_type == "cuda":
        try:
            import bitsandbytes as bnb
            return {
                "optim": "adamw_bnb_8bit",
                "fp16": True,
                "dataloader_pin_memory": True
            }
        except ImportError:
            return {
                "optim": "adamw_torch",
                "fp16": True,
                "dataloader_pin_memory": True
            }
    elif device_type == "mps":
        return {
            "optim": "adamw_torch",
            "fp16": False,  # MPS doesn't support fp16 training
            "dataloader_pin_memory": False,
            # MPS-specific optimizations
            "gradient_checkpointing": True,  # Verified working on MPS
            "gradient_accumulation_steps": 2  # Compensate for smaller batch sizes
        }
    else:  # CPU
        return {
            "optim": "adamw_torch",
            "fp16": False,
            "dataloader_pin_memory": True
        }
```

### Platform-Specific Training Configurations
- CUDA:
  - ✅ 8-bit quantization (via bitsandbytes)
  - ✅ FP16 mixed precision
  - ✅ Large batch sizes
  - ✅ Memory pinning

- MPS:
  - ✅ Gradient checkpointing
  - ✅ Gradient accumulation
  - ✅ Standard optimizers
  - ❌ No 8-bit quantization
  - ❌ No FP16 training
  - ❌ No memory pinning

## Official Documentation References
From [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html):
- Some operations may fall back to CPU silently
- Not all PyTorch operations are implemented for MPS

## Testing
Run compatibility tests before implementing new features:
```bash
python test_mps_limitations.py
```

---

**Note**: This document focuses on limitations that affect cross-platform development decisions. Last updated: January 2025