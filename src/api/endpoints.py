from fastapi import APIRouter, HTTPException
from typing import Dict
import asyncio
from .main import (
    ProcessDataRequest, 
    AdaptModelRequest, 
    EvaluationRequest,
    data_pipeline,
    model_adapter,
    evaluator
)

router = APIRouter()

@router.post("/process-data")
async def process_data(request: ProcessDataRequest):
    """
    Process and validate synthetic review data
    """
    try:
        result = data_pipeline.process_synthetic_data(request.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/adapt-model")
async def adapt_model(request: AdaptModelRequest):
    """
    Adapt model using LoRA
    """
    try:
        result = model_adapter.adapt_model(
            base_model_name=request.base_model_name,
            train_data=request.train_data,
            eval_data=request.eval_data,
            custom_lora_config=request.lora_config
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate model performance
    """
    try:
        result = evaluator.evaluate_model(
            model=model_adapter,
            test_data=request.test_data,
            domain=request.domain
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hardware-info")
async def get_hardware_info():
    """
    Get current hardware configuration
    """
    from ..config.environment import HardwareConfig
    config = HardwareConfig.detect_hardware()
    return {
        "device": config.device,
        "cores": config.n_cores,
        "memory": config.memory_limit,
        "mps_available": config.use_mps
    } 