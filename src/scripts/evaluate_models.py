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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_single_model(model_name_or_path, tokenizer_name_or_path, test_dataset, compute_metrics_fn, label2id, device):
    """
    Loads and evaluates a classification model on the given test dataset.
    'model_name_or_path' might be a checkpoint directory or huggingface model name.
    'tokenizer_name_or_path' is the corresponding tokenizer reference.
    """
    logger.info(f"Loading tokenizer for: {tokenizer_name_or_path}")
    eval_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    # Preprocess test split just for this model (so tokenization matches the model)
    # E.G foreign language models are not gonna have a good time here
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

    # We'll map over test_dataset only, removing columns for format consistency (lazy should do it differently later)
    model_test = test_dataset.map(
        tokenize_fn,
        remove_columns=test_dataset.column_names,
        desc=f"Tokenizing test for {model_name_or_path}"
    )
    model_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info(f"Loading model for: {model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=3,  # 3-way sentiment, IF OPPONENT MODELS DONT SUPPORT 3-WAY SENT DONT EVEN BOTHER
        label2id=label2id,
        trust_remote_code=True
    ).to(device)

    eval_args = TrainingArguments(
        output_dir="eval_tmp_dir",
        per_device_eval_batch_size=16,
        # We'll do inference-only for now, later more
        do_train=False,
        do_eval=True,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=model_test,
        compute_metrics=compute_metrics_fn
    )
    metrics = trainer.evaluate()
    return metrics

def main():
    # 1. Load the same dataset from disk
    # ecommerce_7k_20250106_234227 === scraped ecommerce reviews(fairly new HIGH AMOUNT OF POSITIVE REVIEWS)
    dataset_path = "src/data/datasets/ecommerce_7k_20250106_234227"
    dataset_dict = load_from_disk(dataset_path)
    test_dataset = dataset_dict["test"]

    # 2. For now we create a minimal DebertaTrainer instance to reuse the same metric function. 
    #    Metric function should obviously become its own file later.
    #
    #    The 'dummy_lora' and 'dummy_training' dictionaries are simply placeholders: 
    #    they let us initialize DebertaTrainer *without* messing up your real 
    #    training or LoRA code. We only want to borrow _compute_metrics and 
    #    label mappings, not actually train here.
    hardware_config = HardwareConfig(force_cpu=False)  # Use MPS if available, or CPU, etc.
    dummy_lora = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["query_proj", "value_proj"]
    }
    dummy_training = {"model_name_or_path": "microsoft/deberta-v3-base"}

    temp_trainer = DebertaTrainer(
        dataset_path=dataset_path,
        lora_config=dummy_lora,
        training_config=dummy_training,
        hardware_config=hardware_config
    )

    # Reusing perhaps we could code it better
    compute_metrics_fn = temp_trainer._compute_metrics
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    device = hardware_config.device

    # 3. Define the models we want to compare always also researching correct tokenizers
    #    Below we add 'cardiffnlp/twitter-roberta-base-sentiment-latest' as an example
    #    of an existing HF model that does 3-class sentiment classification. 
    #    It's trained on tweets, but still has negative/neutral/positive labels.
    models_to_compare = {
        "base_deberta": {
            "model_path": "microsoft/deberta-v3-base",
            "tokenizer_path": "microsoft/deberta-v3-base"
        },
        "lora_deberta": { # 20250106_224355 == 18k research based distribution, FULL SYNTHETIC
            "model_path": "src/models/storage/deberta-v3-base/lora/three_way/20250106_224355",
            "tokenizer_path": "microsoft/deberta-v3-base"
        },
        "bert": {
            "model_path": "bert-base-uncased",
            "tokenizer_path": "bert-base-uncased"
        },
        "distilbert": {
            "model_path": "distilbert-base-uncased",
            "tokenizer_path": "distilbert-base-uncased"
        },
        "roberta": {
            "model_path": "roberta-base",
            "tokenizer_path": "roberta-base"
        },
        "chinese_bert": {  # This should always provide bad results in English
            "model_path": "bert-base-chinese",
            "tokenizer_path": "bert-base-chinese"
        },
        # Example of a well-established 3-class sentiment model
        "twitter_roberta_3class": {
            "model_path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "tokenizer_path": "cardiffnlp/twitter-roberta-base-sentiment-latest"
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

    # 5. Print or store the ranking
    print("=== Comparison Results ===")
    for entry in results:
        print(entry)

    # Save to JSON or perhaps csv?
    with open("model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved comparison to model_comparison.json")

if __name__ == "__main__":
    main()