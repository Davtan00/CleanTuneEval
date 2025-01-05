import logging
import warnings
import json
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig
from ..config.environment import HardwareConfig  

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer")

class ModelFactory:
    """
    Creates a HF model for sequence classification, applies LoRA (PEFT),
    and sets device (CPU, CUDA, or MPS).
    """

    def __init__(self,
                 model_name: str = "microsoft/deberta-v3-base",
                 lora_config_path: str = ""):
        """
        Args:
            model_name: Base model identifier.
            lora_config_path: Path to a LoRA JSON config (optional).
        """
        self.hardware = HardwareConfig.detect_hardware()
        self.model_name = model_name
        self.lora_config_path = lora_config_path
        self.device = self.get_device()
        logger.info(f"Initialized ModelFactory with '{model_name}' on device: {self.device}")

    def get_device(self) -> torch.device:
        if self.hardware.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def create_model(self):
        """
        Create a base classification model (3 labels) and wrap with LoRA.
        """
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

        # If a JSON config is given, load LoRA params from file; else use defaults
        if self.lora_config_path and Path(self.lora_config_path).exists():
            with open(self.lora_config_path, "r") as f:
                lora_params = json.load(f)
            logger.info(f"Loaded LoRA config from {self.lora_config_path}")
            lora_config = LoraConfig(
                r=lora_params["r"],
                lora_alpha=lora_params["lora_alpha"],
                lora_dropout=lora_params["lora_dropout"],
                bias=lora_params["bias"],
                task_type="SEQ_CLS",
                target_modules=lora_params["target_modules"]
            )
        else:
            # Fallback defaults
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=["query_proj", "key_proj"],
            )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer 