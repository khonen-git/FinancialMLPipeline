"""Models listing endpoint."""

import logging
import os
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from src.deployment.api.models import ModelsListResponse, ModelInfo
from src.deployment.model_loader import ModelLoader
from src.deployment.api.dependencies import ModelLoaderDep
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsListResponse)
async def list_models(
    model_loader: ModelLoaderDep,
    experiment_name: str = "financial_ml",
    limit: int = 100
) -> ModelsListResponse:
    """List available models from MLflow.
    
    Args:
        model_loader: ModelLoader dependency
        experiment_name: MLflow experiment name to search
        limit: Maximum number of runs to return
        
    Returns:
        ModelsListResponse with list of available models
        
    Raises:
        HTTPException: If MLflow connection fails
    """
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        client = MlflowClient(tracking_uri=tracking_uri)
        
        # Get experiment
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return ModelsListResponse(models=[], count=0)
            experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"Failed to get experiment: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_name}' not found: {str(e)}"
            )
        
        # Search runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        models = []
        for run in runs:
            # Check if run has a model artifact
            artifacts = client.list_artifacts(run.info.run_id)
            model_uri = None
            
            for artifact in artifacts:
                if artifact.path == "model" or artifact.path.endswith("/model"):
                    model_uri = f"runs:/{run.info.run_id}/model"
                    break
            
            if model_uri:
                # Get metrics and params
                metrics = {k: float(v) for k, v in run.data.metrics.items()}
                params = dict(run.data.params)
                tags = dict(run.data.tags)
                
                model_info = ModelInfo(
                    model_uri=model_uri,
                    run_id=run.info.run_id,
                    experiment_id=experiment_id,
                    metrics=metrics,
                    params=params,
                    tags=tags
                )
                models.append(model_info)
        
        return ModelsListResponse(models=models, count=len(models))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )

