from pathlib import Path
import torch
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import logging
from tqdm import tqdm

from .metrics import compute_classification_metrics
from ..config.environment import HardwareConfig

logger = logging.getLogger(__name__)

class ModelComparisonEvaluator:
    """Evaluates and compares multiple models on sentiment analysis."""
    
    def __init__(
        self,
        hardware_config: HardwareConfig,
        competitor_models: Dict[str, str]
    ):
        self.device = torch.device(hardware_config.device)
        self.competitor_models = competitor_models
        logger.info(f"Initialized evaluator on device: {self.device}")
    
    def evaluate_all(
        self,
        dataset_path: str,
        our_model_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate our model (if provided) and competitor models.
        Returns rankings and detailed metrics.
        """
        results = {
            "dataset_info": self._get_dataset_info(dataset_path),
            "models": {},
            "rankings": {}
        }
        
        # First evaluate our model if provided
        if our_model_path:
            our_results = self._evaluate_our_model(our_model_path, dataset_path)
            results["models"]["our_model"] = {
                "name": Path(our_model_path).name,
                "metrics": our_results
            }
            results["rankings"]["Our LoRA Model"] = {
                "combined_metric": our_results["eval_combined_metric"],
                "accuracy": our_results["eval_accuracy"],
                "macro_f1": our_results["eval_macro_f1"]
            }
        
        # Then evaluate competitors
        for model_id, model_name in tqdm(self.competitor_models.items(), desc="Evaluating models"):
            try:
                metrics = self._evaluate_competitor(model_id, dataset_path)
                results["models"][model_id] = {
                    "name": model_name,
                    "metrics": metrics
                }
                results["rankings"][model_name] = {
                    "combined_metric": metrics["eval_combined_metric"],
                    "accuracy": metrics["eval_accuracy"],
                    "macro_f1": metrics["eval_macro_f1"]
                }
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        return results
    
    def _get_dataset_info(self, dataset_path: str) -> Dict:
        """Get basic dataset information."""
        dataset = load_from_disk(dataset_path)
        return {
            "name": Path(dataset_path).name,
            "num_samples": len(dataset["test"]),
            "label_distribution": dataset["test"]["labels"].value_counts().to_dict()
        }
    
    def _evaluate_our_model(self, model_path: str, dataset_path: str) -> Dict:
        """Evaluate our LoRA-tuned model."""
        from peft import PeftModel, PeftConfig
        
        # Load base model and apply LoRA
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=3
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        return self._evaluate_single_model(model, tokenizer, dataset_path)
    
    def _evaluate_competitor(self, model_id: str, dataset_path: str) -> Dict:
        """Evaluate a competitor model."""
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return self._evaluate_single_model(model, tokenizer, dataset_path)
    
    def _evaluate_single_model(self, model, tokenizer, dataset_path: str) -> Dict:
        """Evaluate a single model and return metrics."""
        model = model.to(self.device)
        model.eval()
        
        dataset = load_from_disk(dataset_path)
        test_data = dataset["test"]
        
        # Tokenize
        encoded = test_data.map(
            lambda x: tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors=None
            ),
            batched=True,
            remove_columns=test_data.column_names
        )
        
        # Prepare data
        encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataloader = torch.utils.data.DataLoader(encoded, batch_size=16)
        
        # Get predictions
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                all_preds.append(outputs.logits)
                all_labels.append(batch["labels"])
        
        # Compute metrics
        predictions = torch.cat(all_preds).cpu()
        labels = torch.cat(all_labels).cpu()
        
        return compute_classification_metrics(
            type("EvalPred", (), {"predictions": predictions, "label_ids": labels})()
        ) 