import os
import json
import torch
import random
import numpy as np
from typing import Dict, Any
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import torch.nn.functional as F  # for softmax

###############################################################################
# SEED HELPER
###############################################################################
def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    Note: Full determinism can still depend on hardware and library versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

###############################################################################
# 1) HELPER: Load JSON and Create a Dataset
###############################################################################
def load_test_dataset(json_path: str):
    """
    Expects a JSON with structure:
      {
        "generated_data": [
           {
             "id": ...,
             "text": "...",
             "sentiment": "positive"/"negative"/"neutral"
           },
           ...
        ]
      }
    Returns a Hugging Face Dataset with "text" and integer "label".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We'll assume data["generated_data"] holds all examples
    raw_examples = data["generated_data"]

    # Map sentiment to integer label
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    dataset_entries = []
    for ex in raw_examples:
        text = ex["text"]
        sent_str = ex["sentiment"].lower().strip()
        # default to "neutral" if unknown
        label_id = label_map[sent_str] if sent_str in label_map else 1
        dataset_entries.append({"text": text, "label": label_id})

    hf_dataset = Dataset.from_list(dataset_entries)
    return hf_dataset

###############################################################################
# 2) EVALUATION FUNCTION
###############################################################################
def evaluate_model(
    model, 
    tokenizer, 
    dataset, 
    batch_size=16, 
    aggregator_type=None
):
    """
    Runs inference on 'dataset' (with "text", "label") in batches 
    and returns accuracy & weighted F1.

    'aggregator_type' can be:
      - None / "3-class": No aggregator. Model natively outputs 3 classes (neg/neu/pos).
      - "nlp5_simple": For 'nlptown/bert-base-multilingual-uncased-sentiment' (5 classes),
        we do a simpler aggregator:
            1 star => negative
            5 star => positive
            else   => neutral
      - "tabular5": For tabularisai (5 classes: 0=Very Neg,1=Neg,2=Neu,3=Pos,4=Very Pos)
        negative = 0 or 1
        neutral  = 2
        positive = 3 or 4
    """
    model.eval()
    all_labels = []
    all_preds = []

    data_list = list(dataset)
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(data_list), batch_size), desc="Evaluating"):
        batch = data_list[i : i + batch_size]
        texts = [x["text"] for x in batch]
        labels = [x["label"] for x in batch]
        all_labels.extend(labels)

        # Tokenize
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        if aggregator_type is None or aggregator_type == "3-class":
            # Normal scenario: model has 3-class output
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        elif aggregator_type == "nlp5_simple":
            # This model outputs 5 classes (stars). We do a CRUDE aggregator:
            #   class=0 => 1 star => negative
            #   class=4 => 5 star => positive
            #   everything else => neutral
            pred_5 = torch.argmax(logits, dim=-1)  # 0..4
            # Convert to our 3-class scheme
            final_preds = []
            for p in pred_5.cpu().tolist():
                if p == 0: 
                    final_preds.append(0)  # negative
                elif p == 4:
                    final_preds.append(2)  # positive
                else:
                    final_preds.append(1)  # neutral
            preds = final_preds

        elif aggregator_type == "tabular5":
            # tabularisai: 5 classes (0=VeryNeg,1=Neg,2=Neu,3=Pos,4=VeryPos)
            # Lump 0,1 => negative, 2 => neutral, 3,4 => positive
            pred_5 = torch.argmax(logits, dim=-1)  # 0..4
            final_preds = []
            for p in pred_5.cpu().tolist():
                if p in [0,1]:
                    final_preds.append(0) # negative
                elif p == 2:
                    final_preds.append(1) # neutral
                else:
                    final_preds.append(2) # positive
            preds = final_preds

        else:
            # fallback: no aggregator
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        all_preds.extend(preds)

    acc = accuracy_score(all_labels, all_preds)
    f1w = f1_score(all_labels, all_preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1w}

###############################################################################
# 3) MAIN SCRIPT
###############################################################################
def main():
    # A) Set the seed
    set_seed(42)

    # B) Load the dataset
    json_path = "src/data/storage/processed/healthcare_46k_20250107_131545.json"
    print(f"Loading test data from: {json_path}")
    test_dataset = load_test_dataset(json_path)
    print("Original test dataset size:", len(test_dataset))

    # Sample smaller
    desired_size = 10000
    if len(test_dataset) > desired_size:
        test_dataset = test_dataset.shuffle(seed=42).select(range(desired_size))
        print(f"Reduced test dataset to {desired_size} samples")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # C) Models
    models_to_evaluate = [
        # Good LoRA
        {
            "name": "LoRA Roberta (synthetic-qual)",
            "base_ckpt": "roberta-large",
            "lora_ckpt": "lora_roberta_output/final_model",
            "aggregator": None  # 3 classes natively
        },
        # Bad LoRA
        {
            "name": "LoRA Roberta",
            "base_ckpt": "roberta-large",
            "lora_ckpt": "lora_roberta_scraped/final_model",
            "aggregator": None
        },
        # Base Roberta
        {
            "name": "Base Roberta (no tuning)",
            "base_ckpt": "roberta-large",
            "lora_ckpt": None,
            "aggregator": None
        },
        # Twitter RoBERTa (3 label)
        {
            "name": "Twitter-RoBERTa (3-label)",
            "base_ckpt": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "lora_ckpt": None,
            "aggregator": "3-class"  # It has 3 classes natively
        },
        # NLPBertTown (5 label) => Use CRUDE aggregator
        {
            "name": "NLPBertTown (5-class, crude aggregator)",
            "base_ckpt": "nlptown/bert-base-multilingual-uncased-sentiment",
            "lora_ckpt": None,
            "aggregator": "nlp5_simple"
        },
        # TabularisAI (5 label => VeryNeg,Neg,Neu,Pos,VeryPos)
        {
            "name": "TabularisAI (5-class aggregator)",
            "base_ckpt": "tabularisai/multilingual-sentiment-analysis",
            "lora_ckpt": None,
            "aggregator": "tabular5"
        },
    ]

    results_data = []

    # D) Evaluate
    for entry in models_to_evaluate:
        model_name = entry["name"]
        base_ckpt = entry["base_ckpt"]
        lora_ckpt = entry["lora_ckpt"]
        aggregator_type = entry["aggregator"]

        print(f"\n=== Evaluating: {model_name} ===")

        # If aggregator_type indicates 5-class, let it load default # classes
        if aggregator_type in ["nlp5_simple", "tabular5"]:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_ckpt
            )
        else:
            # Force 3 if needed, ignoring mismatch so it doesn't throw errors
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_ckpt,
                num_labels=3,
            )

        tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
        base_model.to(device)

        if lora_ckpt:
            # Merge with LoRA
            model = PeftModel.from_pretrained(base_model, lora_ckpt)
            model.to(device)
        else:
            model = base_model

        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            batch_size=16,
            aggregator_type=aggregator_type
        )
        print(f"Results for {model_name}:", metrics)

        results_data.append({
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"]
        })

        # Cleanup
        del model
        del base_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # E) Generate LaTeX table
    latex_table_path = "benchmark_results.tex"
    latex_table = create_latex_table(results_data)
    with open(latex_table_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to: {latex_table_path}")
    print("\nDone!")

def create_latex_table(results_data):
    """
    Takes a list of dicts: [{"model_name":..., "accuracy":..., "f1_weighted":...}],
    returns a LaTeX table
    """
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{lcc}\n"
        "\\hline\n"
        "Model & Accuracy & F1 (Weighted) \\\\\n"
        "\\hline\n"
    )

    rows = []
    for r in results_data:
        row = f"{r['model_name']} & {r['accuracy']:.3f} & {r['f1_weighted']:.3f} \\\\"
        rows.append(row)

    footer = (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\caption{Benchmark results on reduced scraped reviews.}\n"
        "\\label{tab:benchmark_results}\n"
        "\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer

if __name__ == "__main__":
    main()
