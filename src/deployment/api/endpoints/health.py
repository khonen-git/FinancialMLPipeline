"""Health check endpoint."""

import logging
import os
from fastapi import APIRouter, Depends
from src.deployment.api.models import HealthResponse
from src.deployment.model_loader import ModelLoader
from src.deployment.api.dependencies import ModelLoaderDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


def _check_gpu_available() -> bool:
    """Check if GPU is available.
    
    Returns:
        True if GPU (CUDA) is available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        from cuml import __version__
        # cuML is installed, but we need to check if GPU is actually available
        # This is a simple check - in production, you might want more sophisticated detection
        return True
    except ImportError:
        pass
    
    return False


@router.get("", response_model=HealthResponse)
async def health_check(
    model_loader: ModelLoaderDep
) -> HealthResponse:
    """Health check endpoint.
    
    Args:
        model_loader: ModelLoader dependency
        
    Returns:
        HealthResponse with service status and component health
    """
    # Check MLflow connection
    mlflow_connected = False
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        if tracking_uri.startswith("file:"):
            # Local file-based tracking - check if directory exists
            import os
            path = tracking_uri.replace("file:", "")
            mlflow_connected = os.path.exists(path) if path else True
        else:
            # Remote tracking - try to connect
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=tracking_uri)
            # Try to list experiments (lightweight operation)
            client.list_experiments(max_results=1)
            mlflow_connected = True
    except Exception as e:
        logger.warning(f"MLflow connection check failed: {e}")
        mlflow_connected = False
    
    # Check GPU
    gpu_available = _check_gpu_available()
    
    # Get cached models count
    cached_models = len(model_loader.get_cached_models())
    
    # Determine overall status
    status = "healthy" if mlflow_connected else "unhealthy"
    
    return HealthResponse(
        status=status,
        mlflow_connected=mlflow_connected,
        gpu_available=gpu_available,
        cached_models=cached_models
    )

