import pytest
from src.models.model_factory import ModelFactory
from src.models.adaptation import ModelAdapter
from src.models.lora_config import LoRAParameters
import torch

def test_model_factory_device_selection():
    factory = ModelFactory()
    device = factory.get_device()
    assert isinstance(device, torch.device)
    
def test_lora_config_creation():
    params = LoRAParameters(r=8, lora_alpha=16)
    config = create_lora_config(params)
    assert config.r == 8
    assert config.lora_alpha == 16

@pytest.mark.integration
def test_model_adaptation(sample_dataset):
    adapter = ModelAdapter()
    result = adapter.adapt_model(
        train_dataset=sample_dataset["train"],
        eval_dataset=sample_dataset["validation"]
    )
    assert result["status"] == "success"
    assert "metrics" in result 