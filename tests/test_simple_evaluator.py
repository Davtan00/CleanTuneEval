import pytest
from unittest.mock import Mock, patch, mock_open
import torch
import pandas as pd
from pathlib import Path
from src.evaluation.simple_evaluator import SimpleEvaluator
from tests.config import TEST_DATA_DIR
import json

@pytest.fixture
def test_dataset():
    """Create a minimal test dataset structure"""
    dataset_path = TEST_DATA_DIR / "datasets" / "test_eval_dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create a mock dataset that behaves like a proper Dataset
    mock_dataset = Mock()
    
    # Basic Dataset interface
    mock_dataset.__len__ = Mock(return_value=10)
    mock_dataset.map = Mock(return_value=mock_dataset)
    mock_dataset.set_format = Mock(return_value=None)
    
    # Make dataset subscriptable
    def getitem(idx):
        return {
            'input_ids': torch.ones(5),
            'attention_mask': torch.ones(5),
            'labels': torch.tensor(1)
        }
    mock_dataset.__getitem__ = Mock(side_effect=getitem)
    
    # Add iterator behavior
    mock_dataset.__iter__ = Mock(return_value=iter([
        getitem(i) for i in range(10)
    ]))
    
    return {
        'test': mock_dataset
    }

@pytest.fixture
def mock_model_outputs():
    """Mock model outputs with realistic sentiment predictions"""
    return torch.tensor([
        [0.7, 0.2, 0.1],  # Confident negative
        [0.1, 0.8, 0.1],  # Confident neutral
        [0.1, 0.2, 0.7],  # Confident positive
    ])

def test_evaluator_initialization():
    """Test basic initialization and device selection"""
    evaluator = SimpleEvaluator()
    assert evaluator.device in ['cpu', 'cuda', 'mps']
    assert hasattr(evaluator, 'evaluate_pair')

@patch('src.evaluation.simple_evaluator.AutoTokenizer')
@patch('src.evaluation.simple_evaluator.PeftModel')
@patch('src.evaluation.simple_evaluator.AutoModelForSequenceClassification')
@patch('src.evaluation.simple_evaluator.load_from_disk')
def test_model_loading(mock_load_dataset, mock_auto_model, mock_peft, mock_tokenizer, test_dataset):
    """Test model loading functionality"""
    # Setup all mocks
    mock_load_dataset.return_value = test_dataset
    
    # Create a mock model that handles all required methods
    mock_model = Mock()
    mock_model.eval = Mock(return_value=None)
    mock_model.to = Mock(return_value=mock_model)
    
    # Make sure model returns predictions for exactly 10 samples (matching dataset)
    def model_forward(**kwargs):
        batch_size = len(kwargs.get('input_ids', []))
        return Mock(logits=torch.tensor([[0.1, 0.8, 0.1]] * batch_size))
    mock_model.__call__ = Mock(side_effect=model_forward)
    
    # Setup model and tokenizer mocks
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_peft.from_pretrained.return_value = mock_model
    mock_tokenizer.from_pretrained.return_value = Mock()
    
    adapter_config = json.dumps({"base_model_name_or_path": "microsoft/deberta-v3-base"})
    with patch('builtins.open', mock_open(read_data=adapter_config)):
        evaluator = SimpleEvaluator()
        evaluator.evaluate_pair(
            lora_model_path="fake/path",
            test_dataset_path="fake/dataset"
        )
        
        # Verify correct model loading sequence
        mock_auto_model.from_pretrained.assert_called_once()
        mock_peft.from_pretrained.assert_called_once()

@patch('src.evaluation.simple_evaluator.load_from_disk')
@patch('src.evaluation.simple_evaluator.AutoModelForSequenceClassification')
@patch('src.evaluation.simple_evaluator.PeftModel')
@patch('src.evaluation.simple_evaluator.AutoTokenizer')
def test_evaluation_results_structure(mock_peft, mock_auto_model, mock_load_dataset, 
                                    mock_tokenizer, test_dataset, mock_model_outputs):
    """Test that evaluation results have the correct structure"""
    mock_load_dataset.return_value = test_dataset
    
    # Create a mock model that handles all required methods
    mock_model = Mock()
    mock_model.eval = Mock(return_value=None)
    mock_model.to = Mock(return_value=mock_model)
    mock_model.return_value = Mock(logits=mock_model_outputs)
    
    # Setup model and tokenizer mocks
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_peft.from_pretrained.return_value = mock_model
    mock_tokenizer.from_pretrained.return_value = Mock()
    
    adapter_config = json.dumps({"base_model_name_or_path": "microsoft/deberta-v3-base"})
    with patch('builtins.open', mock_open(read_data=adapter_config)):
        evaluator = SimpleEvaluator()
        results = evaluator.evaluate_pair(
            lora_model_path="fake/path",
            test_dataset_path="fake/dataset"
        )
        
        # Check DataFrame structure
        assert isinstance(results, pd.DataFrame)
        assert set(results.columns) >= {
            'model_type', 'model_name', 'accuracy', 
            'macro_f1', 'per_class_f1', 'confusion_matrix'
        }
        assert len(results) == 2  # Base and LoRA models

@pytest.mark.parametrize("device", ['cpu', 'cuda', 'mps'])
def test_device_handling(device):
    """Test proper device handling"""
    with patch('torch.cuda.is_available', return_value=(device == 'cuda')), \
         patch('torch.backends.mps.is_available', return_value=(device == 'mps')):
        evaluator = SimpleEvaluator()
        assert evaluator.device == device

def test_error_handling():
    """Test proper error handling for missing files/models"""
    evaluator = SimpleEvaluator()
    
    with pytest.raises(FileNotFoundError):
        evaluator.evaluate_pair(
            lora_model_path="nonexistent/path",
            test_dataset_path="nonexistent/dataset"
        ) 