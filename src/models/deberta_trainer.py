import os
import json
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from typing import Dict

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebertaTrainer:
    """
    Expects a dictionary for LoRA config (lora_config) and another for training config (training_config).
    These must be fully specified in JSON files, with no CLI override. 
    """

    def __init__(
        self,
        dataset_path: str,
        lora_config: Dict,
        training_config: Dict
    ):
        self.dataset_path = Path(dataset_path)

        # The training config must include a model_name_or_path for the base model
        if "model_name_or_path" not in training_config:
            raise ValueError("training_config.json must contain 'model_name_or_path'")

        self.model_name = training_config["model_name_or_path"]
        self.dataset_id = self.dataset_path.name
        logger.info(f"Extracted dataset ID: {self.dataset_id}")

        # Create an output directory named with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("src/models/storage/deberta-v3-base/lora/three_way") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_metrics = self._load_dataset_metrics()

        # Store config details
        self.lora_config_dict = lora_config
        self.training_config = training_config

        # Convert LoRA dict to actual config object
        self.lora_config = self._create_lora_config(self.lora_config_dict)

        # Hard-coded label mapping
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}

    def _load_dataset_metrics(self) -> Dict:
        """
        Load any previously computed metrics for the dataset, if available.
        """
        metrics_path = Path("src/data/storage/metrics") / f"{self.dataset_id}_metrics.json"
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"No precomputed metrics found at {metrics_path}")
            return {}

    def _create_lora_config(self, config_dict: Dict) -> LoraConfig:
        """
        Convert the provided dictionary into a LoraConfig object.
        Throws an error if any required keys for LoRA are missing.
        """
        try:
            return LoraConfig(**config_dict)
        except Exception as e:
            logger.error(f"Invalid LoRA config dictionary: {config_dict}")
            raise ValueError(f"Failed to create LoraConfig: {e}")

    def _initialize_tokenizer(self):
        logger.info(f"Initializing tokenizer: {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=512,
                use_fast=True,
                trust_remote_code=True
            )
            # Validate correct tokenizer type
            if not any(name in str(type(tokenizer)).lower() for name in ["deberta", "sentencepiece"]):
                raise ValueError("The loaded tokenizer is not DeBERTa/SentencePiece-based.")
            return tokenizer
        except Exception as fast_error:
            logger.warning(f"Failed to load fast tokenizer: {fast_error}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=False,
                    trust_remote_code=True
                )
                return tokenizer
            except Exception as e:
                logger.error("Failed to load both fast and non-fast tokenizer.")
                raise RuntimeError(f"DeBERTa tokenizer initialization failed: {e}")

    def _preprocess_dataset(self, dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
        """
        Tokenize text and map label strings to integer IDs.
        """
        def process(example):
            tokenized = tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
            label_str = example["labels"]
            if label_str not in self.label2id:
                raise ValueError(
                    f"Unexpected label '{label_str}'. "
                    f"Must be one of: {list(self.label2id.keys())}."
                )
            tokenized["labels"] = self.label2id[label_str]
            return tokenized

        for split_name in dataset_dict.keys():
            dataset_dict[split_name] = dataset_dict[split_name].map(process, batched=False)
            dataset_dict[split_name].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels"]
            )
        return dataset_dict

    def _create_model(self):
        """
        Load the base DeBERTa model and apply LoRA adapters.
        """
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        model = get_peft_model(base_model, self.lora_config)
        return model

    def _setup_training_args(self) -> TrainingArguments:
        """
        Construct TrainingArguments from the user-provided training_config.
        Filters out non-training arguments like model_name_or_path.
        """
        dataset = load_from_disk(self.dataset_path)
        train_size = len(dataset["train"])
        logger.info(f"Training set size: {train_size} samples")

        # Some default values can be set here; they can also be overridden by JSON.
        args_dict = {
            "output_dir": str(self.output_dir),
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "report_to": "none",
            "load_best_model_at_end": True,
        }

        # Filter out non-training config parameters and merge in user-provided config
        training_params = {k: v for k, v in self.training_config.items() 
                         if k not in ["model_name_or_path"]}  # exclude model path
        args_dict.update(training_params)

        # Ensure mandatory fields
        required_keys = ["num_train_epochs", "learning_rate"]
        for k in required_keys:
            if k not in args_dict:
                raise ValueError(f"training_config.json must include '{k}'")

        return TrainingArguments(**args_dict)

    def _compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Standard classification metrics: accuracy, precision, recall, F1 (macro).
        """
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def train(self) -> Dict:
        """
        Executes the entire training process: load, tokenize, train, evaluate.
        Returns a dictionary with status and results.
        """
        try:
            dataset_dict = load_from_disk(self.dataset_path)
            logger.info(f"Loaded dataset splits: {list(dataset_dict.keys())}")

            tokenizer = self._initialize_tokenizer()
            dataset_dict = self._preprocess_dataset(dataset_dict, tokenizer)

            model = self._create_model()
            training_args = self._setup_training_args()

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            train_result = trainer.train()
            test_metrics = trainer.evaluate(dataset_dict["test"])

            self._save_results(train_result, test_metrics)

            return {
                "status": "success",
                "test_metrics": test_metrics,
                "model_path": str(self.output_dir)
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}

    def _save_results(self, train_result, test_metrics):
        """
        Save training and evaluation metadata to a JSON file.
        Handles non-serializable objects by converting them to serializable types.
        """
        def make_json_serializable(obj):
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj

        # Convert metrics to serializable format
        train_metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
        if callable(train_metrics):
            train_metrics = train_metrics()

        meta = {
            "dataset_info": {
                "id": self.dataset_id,
                "metrics": self.dataset_metrics
            },
            "training_config": {
                "model_name_or_path": self.model_name,
                "lora_config": self.lora_config_dict,
                "device": str(self._get_device())
            },
            "results": {
                "train_metrics": {k: make_json_serializable(v) for k, v in train_metrics.items()},
                "test_metrics": {k: make_json_serializable(v) for k, v in test_metrics.items()}
            }
        }
        out_path = self.output_dir / "training_metadata.json"
        with open(out_path, 'w') as f:
            json.dump(meta, f, indent=2, default=make_json_serializable)

    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
