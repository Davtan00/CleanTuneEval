from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import uvicorn
from pydantic import BaseModel

from ..config.environment import HardwareConfig
from ..data.pipeline import DataPipeline
from ..models.adaptation import ModelAdapter
from ..evaluation.evaluator import CrossDomainEvaluator
from ..config.logging_config import setup_logging

logger = setup_logging()

app = FastAPI(title="Seminar CleanTuneEval")

# Configure CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing API components")
hardware_config = HardwareConfig.detect_hardware()
data_pipeline = DataPipeline()
model_adapter = ModelAdapter()
evaluator = CrossDomainEvaluator(
    hardware_config=hardware_config,
    labels=["negative", "neutral", "positive"]
)

# API endpoints
@app.post("/process-data")
async def process_data(request: dict):
    """Process and validate synthetic review data"""
    logger.info("Received request to process data")
    try:
        result = data_pipeline.process_synthetic_data(request)
        return result
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hardware-info")
async def get_hardware_info():
    """Get current hardware configuration"""
    return {
        "device": hardware_config.device,
        "cores": hardware_config.n_cores,
        "memory": hardware_config.memory_limit,
        "mps_available": hardware_config.use_mps
    } 