"""Prediction endpoint for single predictions."""

import logging
from typing import Union, Optional
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from src.deployment.api.models import (
    PredictionRequest,
    PredictionResponse
)
from src.deployment.model_loader import ModelLoader
from src.deployment.api.dependencies import ModelLoaderDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


def _calculate_confidence(probability: Optional[float], probabilities: Optional[dict]) -> str:
    """Calculate confidence level from probability.
    
    Args:
        probability: Single probability value (max class)
        probabilities: Dictionary of class probabilities
        
    Returns:
        Confidence level: 'high', 'medium', or 'low'
    """
    if probabilities:
        max_prob = max(probabilities.values())
        if max_prob >= 0.8:
            return "high"
        elif max_prob >= 0.6:
            return "medium"
        else:
            return "low"
    elif probability:
        if probability >= 0.8:
            return "high"
        elif probability >= 0.6:
            return "medium"
        else:
            return "low"
    return "low"


@router.post("", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_loader: ModelLoaderDep
) -> PredictionResponse:
    """Make a single prediction.
    
    Args:
        request: Prediction request with model_uri and features
        model_loader: ModelLoader dependency
        
    Returns:
        PredictionResponse with prediction and optional probabilities
        
    Raises:
        HTTPException: If model cannot be loaded or prediction fails
    """
    try:
        # Load model
        model = model_loader.load_from_mlflow(
            request.model_uri,
            backend="cpu"  # Could be determined from model metadata
        )
        
        # Convert features dict to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        if request.return_proba:
            predictions, probabilities = model_loader.predict(
                model,
                features_df,
                return_proba=True
            )
            
            # Convert probabilities to dict if available
            proba_dict = None
            if probabilities is not None:
                # Assuming classes are -1, 0, 1 (or 0, 1, 2)
                # Get class names from model if available
                if hasattr(model, 'classes_'):
                    class_names = [str(int(cls)) for cls in model.classes_]
                else:
                    # Default assumption: 3 classes
                    class_names = ["-1", "0", "1"]
                
                proba_dict = {
                    class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities[0])
                }
                
                # Get max probability
                max_prob = float(max(probabilities[0]))
            else:
                max_prob = None
        else:
            predictions = model_loader.predict(model, features_df, return_proba=False)
            proba_dict = None
            max_prob = None
        
        prediction = int(predictions[0])
        
        # Calculate confidence
        confidence = _calculate_confidence(max_prob, proba_dict)
        
        return PredictionResponse(
            prediction=prediction,
            probability=max_prob,
            probabilities=proba_dict,
            confidence=confidence
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

