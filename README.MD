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

## Model Training Usage

```python
from src.models.train import train_model

# Basic usage
train_model("path/to/dataset")

# Advanced configuration
train_model(
    dataset_path="path/to/dataset",
    base_model="microsoft/deberta-v3-base",
    tuning_method="lora",
    classification_type="three_way"
)
```

## Command Line Usage
```bash
# Train with default settings
python -m src.models.train

# Train with debug information
python -m src.models.train --debug

# Train with custom dataset
python -m src.models.train --dataset path/to/dataset
```

## Model Training and Evaluation

```python
# Train with comprehensive metrics
train_model(
    dataset_path="path/to/dataset",
    base_model="microsoft/deberta-v3-base",
    tuning_method="lora",
    classification_type="three_way"
)

# List archived models
from src.models.utils import list_archived_models

archived_models = list_archived_models()
for model in archived_models:
    print(f"Model from {model['archived_date']}")
    print(f"F1 Score: {model['metric_value']:.4f}")

# Archive older models
from src.models.utils import archive_models

archive_models(
    "src/models/storage/deberta-v3-base/lora/three_way",
    keep_best=3,
    metric="macro_f1"
)
```

## Model Storage Structure
```
src/models/
├── storage/                   # Active models
│   └── deberta-v3-base/
│       └── lora/
│           └── three_way/
│               └── 20250104_211234/
└── archived_models/          # Archived models
     └── 20250104_201234__macro_f1_0.8123/
         ├── archive_metadata.json
         └── model/
```

## Training Pipeline Phases

The training pipeline follows a progressive enhancement approach:

```bash
# Phase 1: Basic Training 
python -m src.models.train --dataset path/to/dataset

# Phase 2: With Data Balancing Current)
python -m src.models.train --dataset path/to/dataset --balance-data

# Phase 3: With Focal Loss
python -m src.models.train --dataset path/to/dataset --loss-fn focal

# Phase 4: With Domain Adaptation
python -m src.models.train --dataset path/to/dataset --domain technology
```

Each phase is triggered only when specific performance metrics indicate its necessity.

# CleanTuneEval

## Hardware Support & Installation

This project supports multiple hardware configurations:

### CUDA (NVIDIA GPUs)
```bash
# Install CUDA requirements
pip install -r requirements/requirements-cuda.txt
```

### Apple Silicon (M1/M2/M3)
```bash
# Install MPS requirements
pip install -r requirements/requirements-mps.txt
```

### CPU Only
```bash
# Install CPU requirements
pip install -r requirements/requirements-cpu.txt
```

## Hardware-Specific Notes

### CUDA
- Supports 8-bit quantization via bitsandbytes
- Enables FP16 training
- Recommended for large-scale training

### Apple Silicon (MPS)
- Uses native Metal acceleration
- FP16 training disabled (MPS limitation)
- Optimized batch sizes for M-series chips

### CPU
- Fallback option for all platforms
- Standard FP32 training
- Limited performance compared to GPU/MPS

 pip install pip-tools   
pip-compile requirements.in --upgrade --output-file requirements.txt  



#"src/data/datasets/technology_18k_20250106_095813" detailed synthetic balanced dataset
#"src/data/datasets/healthcare_46k_20250107_131545" Low quality scraped healthcare dataset

```
{  "domain": "healthcare",
  "generated_data": [
    {
      "text": "Worst experience ever! Waited forever for my appointment and the doc seemed rushed. I felt like just another number. No one was really friendly either, which made it worse. Overall super disappointed.",
      "sentiment": "negative",
      "id": 1
    },
    {
      "text": "This new clinic on Main St is clean and modern. The staff were nice but I didn't feel like they listened to my concerns about my pain. However, the nurse was very professional and explained things well.",
      "sentiment": "neutral",
      "id": 2
    },
  ],
  "summary": {
    "total_generated": 9978,
    "sentiment_distribution": {
      "positive": 3012,
      "negative": 3297,
      "neutral": 3669
    }
  } 
}
``` 

## Example Commands for the paper relevant things

1. **Augment + Train** in a single pass:
   ```bash
   python generate_challenging_synthetic_data.py
   ```
2. **Only Augment**, no training:
   ```bash
   python generate_challenging_synthetic_data.py --only_augment
   ```
3. **Only Train**, dataset is preprocessed at `my_dataset_path`:
   ```bash
   python generate_challenging_synthetic_data.py --only_train \
          --dataset_path my_dataset_path
   ```
4. **Evaluate** multiple models on a test JSON:
   ```bash
   python quick_eval.py
   ```