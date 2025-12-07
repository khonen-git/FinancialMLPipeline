"""Model loader for MLflow models.

Loads trained models from MLflow tracking server or local files.
Supports both CPU (sklearn) and GPU (cuML) models.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Union
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and caches models from MLflow or local files.
    
    Supports:
    - Loading models from MLflow tracking server
    - Loading models from local pickle files
    - In-memory caching for performance
    - Support for both CPU and GPU models
    """
    
    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        """Initialize model loader.
        
        Args:
            mlflow_tracking_uri: MLflow tracking URI. If None, uses default
                or environment variable MLFLOW_TRACKING_URI.
        """
        self.cache: Dict[str, Any] = {}
        
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        else:
            # Use default or environment variable
            self.mlflow_client = MlflowClient()
        
        logger.info(f"ModelLoader initialized with tracking URI: {mlflow.get_tracking_uri()}")
    
    def load_from_mlflow(
        self,
        model_uri: str,
        backend: str = "cpu"
    ) -> Any:
        """Load model from MLflow.
        
        Args:
            model_uri: MLflow model URI (e.g., "runs:/run_id/model" or
                "models:/model_name/version")
            backend: Model backend ("cpu" or "gpu")
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model_uri format is invalid
            FileNotFoundError: If model cannot be loaded
        """
        cache_key = f"{model_uri}:{backend}"
        
        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Loading model from cache: {cache_key}")
            return self.cache[cache_key]
        
        logger.info(f"Loading model from MLflow: {model_uri} (backend: {backend})")
        
        try:
            # Try to load as MLflow model (sklearn flavor)
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully from MLflow (sklearn flavor)")
        except Exception as e1:
            logger.warning(f"Failed to load as sklearn model: {e1}")
            try:
                # Try pyfunc flavor (generic)
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info("Model loaded successfully from MLflow (pyfunc flavor)")
            except Exception as e2:
                logger.error(f"Failed to load model from MLflow: {e2}")
                raise FileNotFoundError(
                    f"Cannot load model from {model_uri}. "
                    f"Tried sklearn ({e1}) and pyfunc ({e2}) flavors."
                )
        
        # Cache the model
        self.cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")
        
        return model
    
    def load_from_file(self, file_path: Union[str, Path]) -> Any:
        """Load model from local pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be unpickled
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        cache_key = f"file:{file_path}"
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Loading model from cache: {cache_key}")
            return self.cache[cache_key]
        
        logger.info(f"Loading model from file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully from file")
        except Exception as e:
            logger.error(f"Failed to load model from file: {e}")
            raise ValueError(f"Cannot unpickle model from {file_path}: {e}")
        
        # Cache the model
        self.cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")
        
        return model
    
    def predict(
        self,
        model: Any,
        features: pd.DataFrame,
        return_proba: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Make predictions using loaded model.
        
        Args:
            model: Loaded model object (must have predict/predict_proba methods)
            features: Feature DataFrame
            return_proba: If True, also return probabilities
            
        Returns:
            Predictions array, or tuple (predictions, probabilities) if return_proba=True
            
        Raises:
            AttributeError: If model doesn't have predict method
        """
        if not hasattr(model, 'predict'):
            raise AttributeError("Model must have a 'predict' method")
        
        predictions = model.predict(features)
        
        if return_proba:
            if not hasattr(model, 'predict_proba'):
                logger.warning("Model doesn't have predict_proba, returning None for probabilities")
                probabilities = None
            else:
                probabilities = model.predict_proba(features)
            return predictions, probabilities
        
        return predictions
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def get_cached_models(self) -> list[str]:
        """Get list of cached model keys.
        
        Returns:
            List of cache keys
        """
        return list(self.cache.keys())

