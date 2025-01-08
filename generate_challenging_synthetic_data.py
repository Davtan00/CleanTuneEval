import json
import random
import os
from typing import Dict, Any, List
from datasets import Dataset, DatasetDict
import torch

# Transformers / Trainer / LoRA
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType

# For metrics
from sklearn.metrics import accuracy_score, f1_score

##############################################################################
# USER-ADJUSTABLE PARAMETERS
##############################################################################

# Ratio of data to keep labeled vs. unlabeled for local re-labeling
LABELED_RATIO = 0.7    # Keep 70% labeled from JSON
UNLABELED_RATIO = 0.3  # Convert 30% to unlabeled

# Train-Validation-Test split proportions
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Local model for re-labeling unlabeled data (no OpenAI usage)
LOCAL_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Synonym replacements for text augmentation (healthcare oriented )
SYNONYMS_MAP = {
    "patient": ["client", "individual", "case"],
    "treatment": ["therapy", "procedure", "intervention"],
    "clinic": ["medical center", "outpatient facility"],
    "medicine": ["drug", "pharmaceutical", "medication"],
    "symptoms": ["indicators", "manifestations", "clinical signs"]
}

def load_ambiguous_phrases() -> List[str]:
    """
    Load ambiguous phrases from the domain config JSON file.
    Returns a list of ambiguous phrases suitable for healthcare domain.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "domain_config/healthcare/ambiguous.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ambiguous_phrases", [])
    except FileNotFoundError:
        print(f"Warning: Could not find ambiguous phrases config at {config_path}")
        return [
            "though the final outcome is still unclear,",
            "but side effects can vary widely,",
            "and some doctors question its overall effectiveness,"
        ]

# Initialize ambiguous phrases from config
AMBIGUOUS_PHRASES = load_ambiguous_phrases()

def read_json_reviews(json_path: str) -> List[Dict[str, Any]]:
    """
    Read the JSON file (syn_reviews.json) in the same directory.  
    Example structure:
      {
        "generated_data": [
           {
             "text": "...",
             "sentiment": "...",
             "id": ...
           },
           ...
        ]
      }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["generated_data"]


def partition_reviews(all_reviews: List[Dict[str, Any]]):
    """
    Partition data into:
      1) labeled set (original sentiment from JSON)
      2) unlabeled set (remove 'sentiment')
    """
    random.shuffle(all_reviews)
    total = len(all_reviews)
    labeled_count = int(total * LABELED_RATIO)
    
    labeled_reviews = all_reviews[:labeled_count]
    unlabeled_reviews = all_reviews[labeled_count:]
    
    # Remove sentiment from unlabeled
    for r in unlabeled_reviews:
        if "sentiment" in r:
            del r["sentiment"]
    return labeled_reviews, unlabeled_reviews


def local_relabel(unlabeled_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use a local sentiment model (pipeline) to assign 'positive','negative','neutral' to unlabeled reviews.
    We'll do a simple mapping:
       'POSITIVE' -> 'positive'
       'NEGATIVE' -> 'negative'
    If the model can't decide, default to 'neutral'
    """
    if not unlabeled_reviews:
        return unlabeled_reviews  # nothing to re-label

    clf = pipeline("sentiment-analysis", model=LOCAL_SENTIMENT_MODEL)
    relabeled = []
    batch_size = 16

    for i in range(0, len(unlabeled_reviews), batch_size):
        batch = unlabeled_reviews[i:i+batch_size]
        texts = [b["text"] for b in batch]
        
        preds = clf(texts)
        for review, pred in zip(batch, preds):
            label = pred["label"].upper()  # "POSITIVE" or "NEGATIVE"
            if "POS" in label:
                sentiment = "positive"
            elif "NEG" in label:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            review["sentiment"] = sentiment
            relabeled.append(review)
    
    return relabeled


def replace_synonyms(text: str) -> str:
    words = text.split()
    for i in range(len(words)):
        w_clean = words[i].lower().strip(",.!?")
        if w_clean in SYNONYMS_MAP:
            synonyms = SYNONYMS_MAP[w_clean]
            words[i] = random.choice(synonyms)
    return " ".join(words)


def inject_ambiguity(text: str) -> str:
    """
    Randomly insert domain-specific ambiguous phrases to mimic uncertain or hedged language.
    Phrases are loaded from domain_config/healthcare/ambiguous.json
    """
    words = text.split()
    idx = random.randint(0, len(words))
    words.insert(idx, random.choice(AMBIGUOUS_PHRASES))
    return " ".join(words)


def augment_review_text(text: str) -> str:
    """
    Combine augmentation steps:
     1) Replace synonyms
     2) Inject ambiguous phrase
    """
    text = replace_synonyms(text)
    text = inject_ambiguity(text)
    return text


def build_augmented_dataset(labeled: List[Dict[str, Any]], unlabeled: List[Dict[str, Any]]) -> Dataset:
    """
    Merge labeled + re-labeled data. For each review, keep original + augmented to diversify.
    """
    combined = labeled + unlabeled
    augmented_entries = []
    for r in combined:
        orig_text = r["text"]
        orig_sent = r["sentiment"]
        orig_id = r["id"]
        
        # Keep original example
        augmented_entries.append({
            "text": orig_text,
            "sentiment": orig_sent,
            "id": f"{orig_id}_orig"
        })
        
        # Augmented version
        aug_text = augment_review_text(orig_text)
        augmented_entries.append({
            "text": aug_text,
            "sentiment": orig_sent,
            "id": f"{orig_id}_aug"
        })
    
    return Dataset.from_list(augmented_entries)


def split_dataset(dataset: Dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42) -> DatasetDict:
    """
    Split the dataset into train/val/test with given fractions.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "Train/Val/Test must sum to 1."
    total = len(dataset)
    dataset = dataset.shuffle(seed=seed)
    
    train_size = int(total * train_frac)
    val_size = int(total * val_frac)
    
    train_ds = dataset.select(range(0, train_size))
    val_ds = dataset.select(range(train_size, train_size+val_size))
    test_ds = dataset.select(range(train_size+val_size, total))
    
    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


def map_sentiment_to_label(sent: str) -> int:
    """
    Map string sentiments to integer labels for classification.
    negative=0, neutral=1, positive=2
    """
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    return mapping[sent]


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1w = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1w}


def create_lora_model(base_model: RobertaForSequenceClassification):
    """
    Use recommended LoRA parameters for roberta-large, focusing on attention modules.
     r=16, alpha=32, dropout=0.05, target_modules for 'query' & 'value'.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],  
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    lora_model = get_peft_model(base_model, lora_config)
    return lora_model


def main():
    random.seed(42)
    
    # Load original data
    json_path = os.path.join(os.path.dirname(__file__), "syn_reviews.json")
    all_reviews = read_json_reviews(json_path)
    
    print("\n=== ORIGINAL REVIEW EXAMPLE ===")
    print(all_reviews[0]["text"])
    
    # After synonym replacement
    print("\n=== AFTER SYNONYM REPLACEMENT ===")
    print(replace_synonyms(all_reviews[0]["text"]))
    
    # After ambiguity injection
    print("\n=== AFTER AMBIGUITY INJECTION ===")
    print(inject_ambiguity(all_reviews[0]["text"]))
    
    # Full augmentation
    print("\n=== AFTER FULL AUGMENTATION ===")
    print(augment_review_text(all_reviews[0]["text"]))

    # 2) Partition into labeled & unlabeled
    labeled_reviews, unlabeled_reviews = partition_reviews(all_reviews)

    # 3) Re-label the unlabeled portion with a local model
    unlabeled_reviews = local_relabel(unlabeled_reviews)

    # 4) Build augmented dataset
    dataset = build_augmented_dataset(labeled_reviews, unlabeled_reviews)

    # 5) Split into train/val/test
    dset_dict = split_dataset(dataset, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    # 6) Convert sentiment strings to numeric labels
    def label_map_fn(example):
        example["labels"] = map_sentiment_to_label(example["sentiment"])
        return example

    dset_dict = dset_dict.map(label_map_fn)

    # 7) Tokenize
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    def token_map_fn(example):
        return tokenize_function(example, tokenizer)

    tokenized_dsets = dset_dict.map(token_map_fn, batched=True)
    tokenized_dsets = tokenized_dsets.remove_columns(["text", "sentiment", "id"])
    tokenized_dsets.set_format("torch")

    # Add dataset saving
    output_dir = "lora_roberta_output"
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dsets.save_to_disk(os.path.join(output_dir, "processed_dataset"))
    
    base_model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=3)

    lora_model = create_lora_model(base_model)

    training_args = TrainingArguments(
        output_dir="lora_roberta_output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir="lora_logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dsets["train"],
        eval_dataset=tokenized_dsets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate(tokenized_dsets["test"])
    print("Test Results:", results)

    
    trainer.save_model("lora_roberta_output/final_model")
    tokenizer.save_pretrained("lora_roberta_output/final_model")


if __name__ == "__main__":
    main()
