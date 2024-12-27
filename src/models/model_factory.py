from transformers import AutoModel, AutoTokenizer
import torch

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, hardware_config: HardwareConfig):
        """
        Initialize model with optimal settings for Apple Silicon
        """
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if hardware_config.use_mps:
            model = model.to(hardware_config.device)
            model = model.half()  
            
        return model, tokenizer 