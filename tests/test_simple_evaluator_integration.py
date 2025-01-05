import pytest
from src.evaluation.simple_evaluator import SimpleEvaluator
import pandas as pd

def skip_if_no_integration(request):
    """Helper function to check if integration tests should be run"""
    return not request.config.getoption("--run-integration")

@pytest.mark.integration
@pytest.mark.skipif(skip_if_no_integration, reason="Only run when explicitly requested")
def test_real_model_evaluation():
    evaluator = SimpleEvaluator()
    results = evaluator.evaluate_pair(
        lora_model_path="src/models/storage/test_model",  # CI/CD specific path
        test_dataset_path="src/data/datasets/test_dataset"  # CI/CD specific path
    )
    assert isinstance(results, pd.DataFrame)
    assert results['accuracy'].mean() > 0 