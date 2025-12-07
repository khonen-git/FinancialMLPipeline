"""Batch prediction endpoint."""

import logging
from typing import Optional
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from src.deployment.api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse
)
from src.deployment.model_loader import ModelLoader
from src.deployment.api.dependencies import ModelLoaderDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict/batch", tags=["prediction"])


@router.post("", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    model_loader: ModelLoaderDep
) -> BatchPredictionResponse:
    """Make batch predictions.
    
    Args:
        request: Batch prediction request with model_uri and features_list
        model_loader: ModelLoader dependency
        
    Returns:
        BatchPredictionResponse with list of predictions
        
    Raises:
        HTTPException: If model cannot be loaded or prediction fails
    """
    try:
        # Load model
        model = model_loader.load_from_mlflow(
            request.model_uri,
            backend="cpu"
        )
        
        # Convert features list to DataFrame
        features_df = pd.DataFrame(request.features_list)
        
        # Make predictions
        if request.return_proba:
            predictions, probabilities = model_loader.predict(
                model,
                features_df,
                return_proba=True
            )
            
            # Convert probabilities to list of dicts
            proba_list = None
            if probabilities is not None:
                # Get class names from model if available
                if hasattr(model, 'classes_'):
                    class_names = [str(int(cls)) for cls in model.classes_]
                else:
                    class_names = ["-1", "0", "1"]
                
                proba_list = [
                    {
                        class_names[i]: float(prob)
                        for i, prob in enumerate(probs)
                    }
                    for probs in probabilities
                ]
        else:
            predictions = model_loader.predict(model, features_df, return_proba=False)
            proba_list = None
        
        # Convert numpy array to list of integers
        if isinstance(predictions, np.ndarray):
            # Convert to Python list first, then to integers
            predictions_list = [int(pred) for pred in predictions.tolist()]
        else:
            predictions_list = [int(pred) for pred in predictions]
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            probabilities=proba_list,
            count=len(predictions_list)
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

