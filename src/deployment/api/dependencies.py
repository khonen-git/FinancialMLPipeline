"""Dependencies for FastAPI endpoints."""

from typing import Annotated
from fastapi import Depends
from src.deployment.model_loader import ModelLoader
import os


def get_model_loader() -> ModelLoader:
    """Get ModelLoader instance.
    
    Returns:
        ModelLoader instance configured with MLflow tracking URI from environment
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    return ModelLoader(mlflow_tracking_uri=tracking_uri)


# Dependency for endpoints
ModelLoaderDep = Annotated[ModelLoader, Depends(get_model_loader)]

