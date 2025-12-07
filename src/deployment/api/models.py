"""Pydantic models for API request/response validation."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import pandas as pd


class PredictionRequest(BaseModel):
    """Request model for single prediction.
    
    Can accept either:
    - model_uri: MLflow model URI
    - features: Feature dictionary or DataFrame-like structure
    """
    
    model_uri: str = Field(..., description="MLflow model URI (e.g., 'runs:/run_id/model')")
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    return_proba: bool = Field(False, description="Return prediction probabilities")
    
    @validator('model_uri')
    def validate_model_uri(cls, v):
        """Validate model URI format."""
        if not v or not isinstance(v, str):
            raise ValueError("model_uri must be a non-empty string")
        # Basic validation: should start with runs:/, models:/, or file://
        if not (v.startswith('runs:/') or v.startswith('models:/') or v.startswith('file://')):
            raise ValueError(
                "model_uri must start with 'runs:/', 'models:/', or 'file://'"
            )
        return v
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "model_uri": "runs:/abc123def456/model",
                "features": {
                    "bar_id": 12345,
                    "bidPrice": 1.1000,
                    "askPrice": 1.1005,
                    "spread": 0.0005,
                    "return_1": 0.001,
                    "return_5": 0.002
                },
                "return_proba": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    model_uri: str = Field(..., description="MLflow model URI")
    features_list: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    return_proba: bool = Field(False, description="Return prediction probabilities")
    
    @validator('model_uri')
    def validate_model_uri(cls, v):
        """Validate model URI format."""
        if not v or not isinstance(v, str):
            raise ValueError("model_uri must be a non-empty string")
        if not (v.startswith('runs:/') or v.startswith('models:/') or v.startswith('file://')):
            raise ValueError(
                "model_uri must start with 'runs:/', 'models:/', or 'file://'"
            )
        return v
    
    @validator('features_list')
    def validate_features_list(cls, v):
        """Validate features list is not empty."""
        if not v or len(v) == 0:
            raise ValueError("features_list must contain at least one feature dictionary")
        return v
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "model_uri": "runs:/abc123def456/model",
                "features_list": [
                    {"bar_id": 12345, "bidPrice": 1.1000, "askPrice": 1.1005},
                    {"bar_id": 12346, "bidPrice": 1.1001, "askPrice": 1.1006}
                ],
                "return_proba": False
            }
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    prediction: int = Field(..., description="Predicted label (-1, 0, or 1)")
    probability: Optional[float] = Field(None, description="Prediction probability (if requested)")
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Class probabilities (if requested)"
    )
    confidence: str = Field(..., description="Confidence level: 'high', 'medium', or 'low'")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "probabilities": {"-1": 0.05, "0": 0.10, "1": 0.85},
                "confidence": "high"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[int] = Field(..., description="List of predicted labels")
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None,
        description="List of class probabilities (if requested)"
    )
    count: int = Field(..., description="Number of predictions")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "predictions": [1, -1, 0],
                "probabilities": [
                    {"-1": 0.05, "0": 0.10, "1": 0.85},
                    {"-1": 0.80, "0": 0.15, "1": 0.05},
                    {"-1": 0.30, "0": 0.40, "1": 0.30}
                ],
                "count": 3
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    mlflow_connected: bool = Field(..., description="MLflow connection status")
    gpu_available: bool = Field(..., description="GPU availability")
    cached_models: int = Field(..., description="Number of cached models")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "mlflow_connected": True,
                "gpu_available": False,
                "cached_models": 2
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    
    model_uri: str
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None


class ModelsListResponse(BaseModel):
    """Response model for available models list."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    count: int = Field(..., description="Number of models")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "models": [
                    {
                        "model_uri": "runs:/abc123/model",
                        "run_id": "abc123",
                        "experiment_id": "1",
                        "metrics": {"accuracy": 0.85, "sharpe_ratio": 1.5},
                        "params": {"n_estimators": 200}
                    }
                ],
                "count": 1
            }
        }

