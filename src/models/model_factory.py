from transformers import AutoModel, AutoTokenizer
import torch
from ..config.environment import HardwareConfig
from ..config.logging_config import setup_logging

logger = setup_logging()

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, hardware_config: HardwareConfig):
        """
        Initialize model with optimal settings for Apple Silicon
        """
        logger.info(f"Creating model: {model_name}")
        logger.info(f"Hardware config: device={hardware_config.device}, mps={hardware_config.use_mps}")
        
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if hardware_config.use_mps:
            logger.info("Enabling Metal Performance Shaders")
            # Enable Metal Performance Shaders
            model = model.to(hardware_config.device)
            # Enable mixed precision for better performance
            model = model.half()  # Use FP16
            logger.info("Model converted to FP16 for better performance")
            
        return model, tokenizer 