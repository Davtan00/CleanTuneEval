import os
import json
from pathlib import Path
import logging
import torch
from datasets import load_from_disk, DatasetDict
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from src.models.deberta_trainer import DebertaTrainer  
from src.config.environment import HardwareConfig
from datetime import datetime
import hashlib
import platform
from typing import Dict, Any
import torch.cuda
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import warnings
import time
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def ensure_model_downloaded(model_name: str) -> Path:
    """Safely download and verify model files."""
    cache_dir = Path("model_cache") / model_name.replace("/", "_")
    
    try:
        # Download with verification
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=HF_TOKEN,
            ignore_patterns=["*.msgpack", "*.h5", "*.safetensors"],
            local_files_only=False,
            revision="main"
        )
        return cache_dir
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        raise

def evaluate_single_model(model_name_or_path, tokenizer_name_or_path, test_dataset, compute_metrics_fn, label2id, device):
    """Evaluate a single Hugging Face model, using known metrics from DebertaTrainer."""
    
    # Ensure model is downloaded first
    try:
        model_path = ensure_model_downloaded(model_name_or_path)
        tokenizer_path = ensure_model_downloaded(tokenizer_name_or_path)
    except Exception as e:
        logger.error(f"Failed to prepare model/tokenizer: {str(e)}")
        raise

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    eval_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # Ensure tokenizer has a valid pad token
    if eval_tokenizer.pad_token_id is None:
        if eval_tokenizer.eos_token_id is not None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
            eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id
        else:
            eval_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            eval_tokenizer.pad_token_id = eval_tokenizer.convert_tokens_to_ids('[PAD]')

    # Preprocess test split for this model
    def tokenize_fn(example):
        tok = eval_tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors=None
        )
        # Convert label from string to ID
        tok["labels"] = label2id[example["labels"]]
        return tok

    model_test = test_dataset.map(
        tokenize_fn,
        remove_columns=test_dataset.column_names,
        desc=f"Tokenizing test for {model_name_or_path}"
    )
    model_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info(f"Loading model from: {model_path}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,
                label2id=label2id,
                trust_remote_code=True,
                token=HF_TOKEN,
                revision="main",
                timeout=60
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to load model after {max_retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
            time.sleep(2 ** attempt)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = eval_tokenizer.pad_token_id

    # If additional tokens were added, resize the model’s embeddings
    if len(eval_tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(eval_tokenizer))

    model.to(device)

    eval_args = TrainingArguments(
        output_dir="eval_tmp_dir",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        evaluation_strategy="no"
    )

    def enhanced_compute_metrics(eval_pred):
        base_metrics = compute_metrics_fn(eval_pred)
        
        # Get raw predictions and labels
        if isinstance(eval_pred.predictions, tuple):
            logits = eval_pred.predictions[0]
        else:
            logits = eval_pred.predictions
        
        preds = np.argmax(logits, axis=1)
        labels = eval_pred.label_ids
        
        # Add balanced accuracy
        base_metrics['eval_balanced_accuracy'] = balanced_accuracy_score(labels, preds)
        
        # Add Matthews Correlation Coefficient
        base_metrics['eval_matthews_correlation'] = matthews_corrcoef(labels, preds)
        
        return base_metrics

    # Update trainer to use enhanced metrics
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=model_test,
        compute_metrics=enhanced_compute_metrics
    )
    metrics = trainer.evaluate()
    return metrics

def get_system_info() -> Dict[str, Any]:
    """Collect system information for reproducibility."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU/MPS",
        "timestamp": datetime.now().isoformat()
    }

def calculate_dataset_hash(dataset) -> str:
    """Calculate a hash of the dataset for verification."""
    # Convert first 1000 examples to string for hashing
    sample = str(dataset[:1000]).encode('utf-8')
    return hashlib.sha256(sample).hexdigest()
#Standard metrics given by sklearn
def calculate_additional_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate additional evaluation metrics with class imbalance handling."""
    additional = {}
    
    # Balanced accuracy-based efficiency score
    if all(k in metrics for k in ['eval_samples_per_second', 'eval_accuracy']):
        balanced_acc = metrics.get('eval_balanced_accuracy', metrics['eval_accuracy'])
        additional['efficiency_score'] = (
            metrics['eval_samples_per_second'] * balanced_acc * 
            (1 - abs(metrics['eval_precision'] - metrics['eval_recall']))
        )
    
    # Matthews Correlation Coefficient based reliability
    if 'eval_matthews_correlation' in metrics:
        additional['reliability_score'] = (1 + metrics['eval_matthews_correlation']) / 2
    
    # Balanced F1 score
    if all(k in metrics for k in ['eval_precision', 'eval_recall']):
        if (metrics['eval_precision'] + metrics['eval_recall']) > 0:
            additional['balanced_score'] = (
                2 * (metrics['eval_precision'] * metrics['eval_recall']) /
                (metrics['eval_precision'] + metrics['eval_recall'])
            )
        else:
            additional['balanced_score'] = 0.0
    
    return additional

def main():
    # 1. Load the dataset from disk
    dataset_path = "src/data/datasets/technology_18k_20250106_095813"
    dataset_dict = load_from_disk(dataset_path)
    test_dataset = dataset_dict["test"]

    # 2. Create a minimal DebertaTrainer to reuse the compute_metrics function
    hardware_config = HardwareConfig(force_cpu=False)
    dummy_lora = {}
    dummy_training = {"model_name_or_path": "microsoft/deberta-v3-base"}

    temp_trainer = DebertaTrainer(
        dataset_path=dataset_path,
        lora_config=dummy_lora,
        training_config=dummy_training,
        hardware_config=hardware_config
    )

    compute_metrics_fn = temp_trainer._compute_metrics
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    device = hardware_config.device

    # 3. Define the models we want to evaluate
    models_to_compare = {
        "deberta-v3-base": {
            "model_path": "microsoft/deberta-v3-base",
            "tokenizer_path": "microsoft/deberta-v3-base"
        },
        "bert-base-uncased": {
            "model_path": "bert-base-uncased",
            "tokenizer_path": "bert-base-uncased"
        },
        "distilbert-base-uncased": {
            "model_path": "distilbert-base-uncased",
            "tokenizer_path": "distilbert-base-uncased"
        },
        "roberta-base": {
            "model_path": "roberta-base",
            "tokenizer_path": "roberta-base"
        },
        "bert-base-chinese": {
            "model_path": "bert-base-chinese",
            "tokenizer_path": "bert-base-chinese"
        },
        "phi-2": {
           "model_path": "microsoft/phi-2",
           "tokenizer_path": "microsoft/phi-2"
        },
        "xlm-roberta-base": {
            "model_path": "xlm-roberta-base",
            "tokenizer_path": "xlm-roberta-base"
        },
        "albert-base-v2": {
            "model_path": "albert-base-v2",
            "tokenizer_path": "albert-base-v2"
        },
        "cardiffnlp/twitter-roberta-base-sentiment-latest": {
            "model_path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "tokenizer_path": "cardiffnlp/twitter-roberta-base-sentiment-latest"
        },
        "deberta_v3_large": {
            "model_path": "microsoft/deberta-v3-large",
            "tokenizer_path": "microsoft/deberta-v3-large"
        },
        "mobilebert": {
            "model_path": "google/mobilebert-uncased",
            "tokenizer_path": "google/mobilebert-uncased"
        },
        "tiny-bert": {
            "model_path": "prajjwal1/bert-tiny",
            "tokenizer_path": "prajjwal1/bert-tiny"
        }
    }
    

    # 4. Evaluate each model
    results = []
    for name, paths in models_to_compare.items():
        metrics = evaluate_single_model(
            model_name_or_path=paths["model_path"],
            tokenizer_name_or_path=paths["tokenizer_path"],
            test_dataset=test_dataset,
            compute_metrics_fn=compute_metrics_fn,
            label2id=label2id,
            device=device
        )
        results.append({"model_name": name, **metrics})

    # Create evaluation metadata
    evaluation_metadata = {
        "system_info": get_system_info(),
        "dataset_info": {
            "path": dataset_path,
            "hash": calculate_dataset_hash(test_dataset),
            "num_samples": len(test_dataset),
            "label_distribution": test_dataset.to_pandas()['labels'].value_counts().to_dict()
        },
        "models_evaluated": list(models_to_compare.keys()),
        "evaluation_parameters": {
            "max_length": 128,
            "batch_size": 16,
            "label_mapping": label2id
        }
    }

    # Process results with additional metrics
    processed_results = []
    for result in results:
        additional_metrics = calculate_additional_metrics(result)
        processed_results.append({**result, **additional_metrics})

    # Prepare final output
    final_output = {
        "metadata": evaluation_metadata,
        "results": processed_results
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("src/evaluation/storage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"model_comparison_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Saved detailed comparison to {output_path}")
    
    # Print summary of best models by different metrics
    print("\n=== Best Models by Metric ===")
    metrics_to_compare = ["combined_metric", "efficiency_score", "reliability_score"]
    for metric in metrics_to_compare:
        sorted_models = sorted(processed_results, key=lambda x: x.get(f"eval_{metric}", 0), reverse=True)
        best = sorted_models[0]
        print(f"\nBest by {metric}:")
        print(f"Model: {best['model_name']}")
        print(f"Score: {best.get(f'eval_{metric}', 0):.4f}")

    # Detect dataset imbalance
    label_dist = test_dataset.to_pandas()['labels'].value_counts()
    imbalance_ratio = label_dist.max() / label_dist.min()
    if imbalance_ratio > 10:
        warnings.warn(
            f"Severe class imbalance detected (ratio {imbalance_ratio:.1f}:1). "
            "Metrics are adjusted to account for imbalance.",
            UserWarning
        )

if __name__ == "__main__":
    main()
