import torch
import logging
from pathlib import Path
from typing import Dict

from datasets import load_from_disk
from sklearn.metrics import f1_score, recall_score, accuracy_score
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from peft import LoraConfig, get_peft_model, TaskType

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

def compute_metrics(pred) -> Dict[str, float]:
    """Calculate various metrics for model evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    neutral_recall = recall_score(labels, preds, labels=[1], average='weighted')
    
    composite_score = (0.3 * f1) + (0.3 * recall) + (0.2 * accuracy) + (0.2 * neutral_recall)
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'recall_weighted': recall,
        'neutral_recall': neutral_recall,
        'composite_score': composite_score
    }

def main():
    # ------------------------
    # 1. SETUP
    # ------------------------
    dataset_path = 'src/data/datasets/healthcare_19k_20250107_151329'  
    output_dir = Path("results/roberta_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device()

    # ------------------------
    # 2. LOAD DATA
    # ------------------------
    dataset = load_and_prepare_data(dataset_path)

    # ------------------------
    # 3. TOKENIZER & BASE MODEL
    # ------------------------
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    base_model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)

    # ------------------------
    # 4. LORA CONFIG & WRAPPING
    # ------------------------

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  
        inference_mode=False,
        r=16,               
        lora_alpha=32,     
        lora_dropout=0.1
    )
    
    # Wrap base model with LoRA
    model = get_peft_model(base_model, peft_config)
    logger.info("LoRA parameters have been added to the base model.")

    # ------------------------
    # 5. DATA TOKENIZATION
    # ------------------------
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    tokenized_datasets = {}
    for split in dataset.keys():
        tokenized_datasets[split] = dataset[split].map(
            tokenize_function, 
            batched=True,
            # Keep only 'labels' so it passes correctly to the Trainer
            remove_columns=[col for col in dataset[split].column_names if col not in ['labels']]
        )
        tokenized_datasets[split].set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']
    test_dataset = tokenized_datasets['test']

    # ------------------------
    # 6. TRAINING ARGUMENTS
    # ------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        remove_unused_columns=False,  # Ensure 'labels' aren't dropped
        fp16=(device == "cuda"),
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100
    )

    # ------------------------
    # 7. TRAINER INITIALIZATION
    # ------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # ------------------------
    # 8. TRAIN & EVALUATE
    # ------------------------
    trainer.train()
    results = trainer.evaluate(test_dataset)
    logger.info(f"Test set results: {results}")

    # ------------------------
    # 9. SAVE FINAL
    # ------------------------
    trainer.save_model(str(output_dir / "final_lora_model"))
    tokenizer.save_pretrained(str(output_dir / "final_lora_model"))

if __name__ == "__main__":
    main()
