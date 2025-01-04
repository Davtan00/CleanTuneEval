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

