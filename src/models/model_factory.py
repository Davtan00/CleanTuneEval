from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from ..config.environment import HardwareConfig
import logging
from peft import get_peft_model, LoraConfig
import warnings

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="The sentencepiece tokenizer")

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
        # Create base model with proper classification setup
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=3,
            problem_type="single_label_classification",
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2}
        )
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=8,                     # Rank
            lora_alpha=32,           # Scaling
            lora_dropout=0.05,       # Dropout probability
            bias="none",
            task_type="SEQ_CLS",     # Sequence Classification
            target_modules=["query_proj", "key_proj"],  # DeBERTa attention layers
            inference_mode=False,
        )
        
        # Create PEFT model with LoRA
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()  # Log the trainable parameters
        
        logger.info(f"Model config: {config}")
        logger.info(f"LoRA config: {lora_config}")
        
        return model, AutoTokenizer.from_pretrained(self.model_name) 