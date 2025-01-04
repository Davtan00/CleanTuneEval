from pathlib import Path
import json
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def archive_models(base_dir: str, 
                  keep_best: int = 3, 
                  metric: str = "macro_f1",
                  archive_dir: str = None):
    """
    Archive models instead of deleting them, keeping only N best in active directory
    
    Args:
        base_dir: Base directory containing model versions
        keep_best: Number of best models to keep in active directory
        metric: Metric used for comparison
        archive_dir: Optional custom archive directory
    """
    base_path = Path(base_dir)
    
    # Setup archive directory
    if archive_dir is None:
        archive_path = base_path.parent.parent.parent / "archived_models"
    else:
        archive_path = Path(archive_dir)
    
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all model runs and their metrics
    model_runs = []
    for timestamp_dir in base_path.glob("*"):
        if not timestamp_dir.is_dir():
            continue
            
        metrics_file = timestamp_dir / "metrics.json"
        config_file = timestamp_dir / "training_config.json"
        
        if not metrics_file.exists():
            continue
            
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Get or create config info
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            
        model_runs.append({
            "path": timestamp_dir,
            "timestamp": timestamp_dir.name,
            "metric_value": metrics.get(f"eval_{metric}", 0),
            "metrics": metrics,
            "config": config
        })
    
    # Sort by metric value (descending)
    model_runs.sort(key=lambda x: x["metric_value"], reverse=True)
    
    # Archive models beyond keep_best
    archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for run in model_runs[keep_best:]:
        # Create archive subdirectory with original timestamp and metric value
        archive_name = f"{run['timestamp']}__{metric}_{run['metric_value']:.4f}"
        archive_model_path = archive_path / archive_name
        
        # Save metadata before moving
        metadata = {
            "original_path": str(run["path"]),
            "archived_date": archive_timestamp,
            "metric_used": metric,
            "metric_value": run["metric_value"],
            "metrics": run["metrics"],
            "training_config": run["config"]
        }
        
        # Create archive directory and save metadata
        archive_model_path.mkdir(parents=True, exist_ok=True)
        with open(archive_model_path / "archive_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Move model to archive
        logger.info(f"Archiving model run {run['timestamp']} with {metric}={run['metric_value']:.4f}")
        shutil.move(str(run["path"]), str(archive_model_path / "model"))

def list_archived_models(archive_dir: str = None):
    """List all archived models with their metrics"""
    if archive_dir is None:
        archive_path = Path(__file__).parent.parent / "models/archived_models"
    else:
        archive_path = Path(archive_dir)
        
    if not archive_path.exists():
        logger.info("No archived models found")
        return []
        
    archived_models = []
    for model_dir in archive_path.glob("*"):
        if not model_dir.is_dir():
            continue
            
        metadata_file = model_dir / "archive_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                archived_models.append(metadata)
    
    return archived_models 