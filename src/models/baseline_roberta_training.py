import torch
from datasets import load_from_disk
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report
import logging
from pathlib import Path
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device() -> str:
    """Configure the optimal device for training."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA - Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon MPS")
    else:
        device = "cpu"
        logger.info("Using CPU - No GPU detected")
    return device

def load_and_prepare_data(dataset_path: str):
    """Load and prepare the dataset for training."""
    try:
        dataset = load_from_disk(dataset_path)
        logger.info(f"Dataset loaded from {dataset_path}")
        
        # Map string labels to integers
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        def map_labels(example):
            example['labels'] = label_mapping[example['labels']]
            return example
            
        for split in dataset.keys():
            dataset[split] = dataset[split].map(map_labels)
            
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def compute_metrics(pred) -> Dict[str, float]:
    """Calculate various metrics for model evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'recall_weighted': recall_score(labels, preds, average='weighted'),
        'neutral_recall': recall_score(labels, preds, labels=[1], average='weighted')
    }
    
    # Composite score as per requirements
    metrics['composite_score'] = (
        0.3 * metrics['f1_weighted'] + 
        0.3 * metrics['recall_weighted'] + 
        0.2 * metrics['accuracy'] + 
        0.2 * metrics['neutral_recall']
    )
    
    return metrics

def get_training_args(device: str, output_dir: str) -> TrainingArguments:
    """Configure training arguments based on device and requirements."""
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="composite_score",
        greater_is_better=True,
        fp16=device == "cuda",  # Enable mixed precision for CUDA
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100
    )

def main():
    # Setup paths and device
    dataset_path = 'src/data/datasets/healthcare_19k_20250107_151329'
    output_dir = Path("results/roberta_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = setup_device()
    
    # Load and prepare data
    dataset = load_and_prepare_data(dataset_path)
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-large',
        num_labels=3
    )
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    tokenized_datasets = {}
    for split in dataset.keys():
        # First tokenize the texts
        tokenized_datasets[split] = dataset[split].map(
            tokenize_function, 
            batched=True,
            remove_columns=[col for col in dataset[split].column_names if col not in ['labels']]  
        )
        # Set the format to PyTorch tensors
        tokenized_datasets[split].set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']  
        )
    
    # Setup training arguments and trainer
    training_args = get_training_args(device, str(output_dir))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train and evaluate
    try:
        trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(tokenized_datasets['test'])
        logger.info(f"Test set results: {test_results}")
        
        # Save model
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

if __name__ == "__main__":
    main()