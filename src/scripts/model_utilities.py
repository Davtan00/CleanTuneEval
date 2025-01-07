import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging
import os
from typing import Dict, Any
import psutil

logging.set_verbosity_error()  # Suppress warnings

class ModelUtility:
    def __init__(self, models: Dict[str, Dict[str, str]]):
        """
        Initialize the utility with a dictionary of models.

        Args:
            models (dict): Dictionary containing model and tokenizer paths
        """
        self.models = models
        self.results = {}
        self.device = self._get_available_device()
    
    def _get_available_device(self) -> str:
        """Determine the best available device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _check_model_requirements(self, model) -> Dict[str, Any]:
        """Check model memory and compute requirements."""
        try:
            param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # Size in MB
            return {
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "model_size_mb": param_size,
                "minimum_ram_gb": param_size * 3 / 1024,  # Rough estimate of RAM needed
                "device_compatibility": {
                    "cpu": True,
                    "cuda": torch.cuda.is_available(),
                    "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                }
            }
        except Exception as e:
            return {"error": f"Failed to check requirements: {str(e)}"}

    def load_models(self):
        """Load and analyze all models specified in the models dictionary."""
        available_ram = psutil.virtual_memory().total / (1024**3)  # GB
        
        for name, paths in self.models.items():
            model_path = paths["model_path"]
            tokenizer_path = paths["tokenizer_path"]

            try:
                # Load model and tokenizer
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True
                )

                # Get model requirements
                requirements = self._check_model_requirements(model)
                
                self.results[name] = {
                    "model_load_success": True,
                    "tokenizer_load_success": True,
                    "original_num_labels": model.config.num_labels,
                    "can_be_fine_tuned": True,
                    "requirements": requirements,
                    "hardware_compatibility": {
                        "meets_memory_requirements": requirements["minimum_ram_gb"] < available_ram,
                        "supported_devices": requirements["device_compatibility"]
                    },
                    "model_type": model.config.model_type,
                    "vocab_size": len(tokenizer)
                }

            except Exception as e:
                self.results[name] = {
                    "model_load_success": False,
                    "tokenizer_load_success": False,
                    "error": str(e)
                }

        return self.results

    def refine_failed_models(self):
        """
        Attempt to identify and refine issues with failed models.
        """
        for name, result in self.results.items():
            if not result.get("model_load_success"):
                model_path = self.models[name]["model_path"]
                tokenizer_path = self.models[name]["tokenizer_path"]
                
                try:
                    # Check if model exists
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    self.results[name]["model_load_success"] = True
                except OSError:
                    self.results[name]["error"] = "Model not found on Hugging Face."

                try:
                    # Check if tokenizer exists
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    self.results[name]["tokenizer_load_success"] = True
                except OSError:
                    self.results[name]["error"] = "Tokenizer not found or incorrect."

                continue

            if result.get("supports_sentiment_classification") is False:
                try:
                    # Identify the number of labels
                    model = AutoModelForSequenceClassification.from_pretrained(self.models[name]["model_path"])
                    self.results[name]["num_labels"] = model.config.num_labels

                    if model.config.num_labels > 3:
                        self.results[name]["error"] = f"Model supports {model.config.num_labels} labels, not 3."

                except Exception as e:
                    self.results[name]["error"] = "Error identifying number of labels: " + str(e)

        return self.results

    def get_results(self):
        """
        Get the results of the model loading process.

        Returns:
            dict: The results dictionary with information about each model.
        """
        return self.results

def print_model_analysis(results: Dict[str, Any]) -> None:
    """Print a formatted analysis of model requirements and compatibility."""
    print("\n=== Model Analysis Report ===\n")
    
    # Sort models by size
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get('requirements', {}).get('model_size_mb', 0)
    )
    
    for name, info in sorted_models:
        if not info.get('model_load_success'):
            print(f"❌ {name}: Failed to load - {info.get('error', 'Unknown error')}")
            continue
            
        req = info.get('requirements', {})
        print(f"\n🔹 {name}")
        print("  └─ Model Type:", info.get('model_type', 'Unknown'))
        print(f"  └─ Size: {req.get('model_size_mb', 0):.1f}MB")
        print(f"  └─ RAM Required: {req.get('minimum_ram_gb', 0):.1f}GB")
        print(f"  └─ Parameters: {req.get('parameter_count', 0):,}")
        print(f"  └─ Labels: {info.get('original_num_labels', 0)}")
        print("  └─ Hardware Support:")
        devices = info.get('hardware_compatibility', {}).get('supported_devices', {})
        for device, supported in devices.items():
            status = "✓" if supported else "✗"
            print(f"     └─ {device.upper()}: {status}")

# Rename and make standalone script or implement it as an ACTUAL util func
if __name__ == "__main__":
    models_to_check = {
        # Existing base models
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
        "xlm-roberta-base": {
            "model_path": "xlm-roberta-base",
            "tokenizer_path": "xlm-roberta-base"
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
        },
        
        # Verified 3-label sentiment models
        "bertweet-base-sentiment": {
            "model_path": "finiteautomata/bertweet-base-sentiment-analysis",
            "tokenizer_path": "finiteautomata/bertweet-base-sentiment-analysis"
        },
        "bert-multilingual-sentiment": {
            "model_path": "nlptown/bert-base-multilingual-uncased-sentiment",
            "tokenizer_path": "nlptown/bert-base-multilingual-uncased-sentiment"
        },
        "twitter-xlm-roberta-sentiment": {
            "model_path": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "tokenizer_path": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        },
        "vinai-bertweet-sentiment": {
            "model_path": "vinai/bertweet-base",
            "tokenizer_path": "vinai/bertweet-base"
        },
        "siebert-sentiment-roberta": {
            "model_path": "siebert/sentiment-roberta-large-english",
            "tokenizer_path": "siebert/sentiment-roberta-large-english"
        },
        
        # Additional verified 3-label models
        "pysentimiento-roberta": {
            "model_path": "pysentimiento/robertuito-sentiment-analysis",
            "tokenizer_path": "pysentimiento/robertuito-sentiment-analysis"
        },
        "pysentimiento-bertweet-pt": {
            "model_path": "pysentimiento/bertweet-pt-sentiment",
            "tokenizer_path": "pysentimiento/bertweet-pt-sentiment"
        },
        "tabularisai-multilingual": {
            "model_path": "tabularisai/multilingual-sentiment-analysis",
            "tokenizer_path": "tabularisai/multilingual-sentiment-analysis"
        },
        "distilbert-student": {
            "model_path": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            "tokenizer_path": "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        },
        
        # Emotion model that can be mapped to sentiment
        "emotion-english-distilroberta": {
            "model_path": "j-hartmann/emotion-english-distilroberta-base",
            "tokenizer_path": "j-hartmann/emotion-english-distilroberta-base"
        }
    }

    utility = ModelUtility(models_to_check)
    results = utility.load_models()
    print_model_analysis(results)
