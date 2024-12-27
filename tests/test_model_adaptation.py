import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.environment import HardwareConfig
from src.models.adaptation import ModelAdaptation

def test_model_initialization():
    """Test model initialization with hardware config"""
    hardware_config = HardwareConfig.detect_hardware()
    adapter = ModelAdaptation(hardware_config)
    
    print("\nHardware Configuration:")
    print(f"Device: {hardware_config.device}")
    print(f"Cores: {hardware_config.n_cores}")
    print(f"Memory: {hardware_config.memory_limit}GB")
    print(f"MPS Available: {hardware_config.use_mps}")
    
    return adapter

def test_model_adaptation():
    """Test model adaptation with sample data"""
    adapter = test_model_initialization()
    
    # Load test data (you would need to create this)
    with open('tests/data/training_sample.json', 'r') as f:
        train_data = json.load(f)
    
    result = adapter.adapt_model(
        base_model_name="roberta-base",
        train_data=train_data,
        eval_data=None
    )
    
    assert result['status'] == 'success', "Model adaptation failed"
    print("\nModel Adaptation Results:")
    print(f"Training Loss: {result['training_results']['train_loss']:.4f}")
    print(f"Model saved at: {result['training_results']['model_path']}") 