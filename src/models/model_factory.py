from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from ..config.environment import HardwareConfig
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        self.hardware = HardwareConfig.detect_hardware()
        self.model_name = model_name
        self.device = self.get_device()
        logger.info(f"Initialized ModelFactory with {model_name} for {self.device}")
        
    def get_device(self):
        if self.hardware.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
        
    def create_model(self):
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=3,  # Explicitly set for 3-class classification
            problem_type="single_label_classification"
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)
        
        logger.info(f"Model config: {config}")
        return model, AutoTokenizer.from_pretrained(self.model_name) 