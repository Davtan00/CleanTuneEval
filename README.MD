Example of how to interact with the server:

# Process data
response = requests.post(
    "http://localhost:8000/process-data",
    json={
        "data": {
            "domain": "restaurant",
            "generated_data": [...]
        }
    }
)

# Adapt model
response = requests.post(
    "http://localhost:8000/adapt-model",
    json={
        "base_model_name": "roberta-large",
        "train_data": {...},
        "eval_data": {...}
    }
)

# Evaluate model
response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "model_id": "adapted_model_1",
        "test_data": {...},
        "domain": "restaurant"
    }
)

python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"



# Dataset Structure Example
{
    'train': Dataset({
        features: {
            'text': Value(dtype='string', id=None),  # Clean, processed review text
            'labels': ClassLabel(names=['negative', 'neutral', 'positive'], id=None),  # Mapped to [0, 1, 2]
            'id': Value(dtype='int64', id=None)  # Original review ID
        },
        num_rows: 7000  # Example size (70% of cleaned data)
    }),
    'validation': Dataset({
        # Same structure as train
        num_rows: 1500  # Example size (15% of cleaned data)
    }),
    'test': Dataset({
        # Same structure as train
        num_rows: 1500  # Example size (15% of cleaned data)
    })
}

# Example single entry
{
    'text': 'The latest smartphone offers excellent performance with its processor. The design is sleek and modern.',
    'labels': 2,  # 2 for positive, 1 for neutral, 0 for negative
    'id': 42
}

## USAGE : 
from datasets import load_from_disk

dataset = load_from_disk("src/data/datasets/technology_10k_20250105_v1")