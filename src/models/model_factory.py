from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from ..config.environment import HardwareConfig
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    def __init__(self):
        self.hardware = HardwareConfig.detect_hardware()
        
    def get_device(self):
        if self.hardware.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
        
    def create_model(self, model_name="microsoft/deberta-v3-base", num_labels=3):
        """
        Initialize model with optimal settings for the detected hardware
        """
        device = self.get_device()
        logger.info(f"Initializing {model_name} on {device}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16 if device != torch.device("cpu") else torch.float32
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model.to(device), tokenizer 