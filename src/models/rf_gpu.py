"""Random Forest model (GPU - cuML).

GPU-accelerated Random Forest using RAPIDS cuML.
Requires CUDA-capable GPU and cuML installation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import cuML, fallback gracefully if not available
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.preprocessing import LabelEncoder as cuLabelEncoder
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    logger.warning("cuML not available. GPU Random Forest will not work. Install with: conda install -c rapidsai cuml")


class RandomForestGPU:
    """Random Forest model using RAPIDS cuML (GPU-accelerated)."""
    
    def __init__(self, config: dict):
        """Initialize GPU Random Forest model.
        
        Args:
            config: Model configuration
            
        Raises:
            ImportError: If cuML is not available
            ValueError: If required config is missing
        """
        if not CUML_AVAILABLE:
            raise ImportError(
                "cuML is not available. Please install RAPIDS cuML:\n"
                "conda install -c rapidsai cuml cudatoolkit=11.8"
            )
        
        self.config = config
        if 'params' not in config:
            raise ValueError("Missing required config: params")
        params = config['params']
        
        # Require essential parameters
        if 'n_estimators' not in params:
            raise ValueError("Missing required config: params.n_estimators")
        if 'max_depth' not in params:
            raise ValueError("Missing required config: params.max_depth")
        if 'min_samples_leaf' not in params:
            raise ValueError("Missing required config: params.min_samples_leaf")
        
        # cuML RandomForest parameters (similar to sklearn but GPU-accelerated)
        self.model = cuRF(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            n_streams=params.get('n_streams', 4),  # cuML-specific: number of CUDA streams
            max_features=params.get('max_features', 'sqrt'),  # cuML supports this
            bootstrap=params.get('bootstrap', True),
            random_state=params.get('random_state', 42),
            handle=params.get('handle', None)  # cuML handle for memory management
        )
        
        # cuML doesn't have built-in calibration like sklearn
        # We'll use sklearn's calibration on GPU predictions if needed
        self.use_calibration = config.get('calibration', {}).get('enabled', False)
        if self.use_calibration:
            logger.warning("Calibration not directly supported by cuML. Will use sklearn CalibratedClassifierCV on GPU predictions.")
            from sklearn.calibration import CalibratedClassifierCV
            self.calibrated_model = None
            self.calibration_method = config.get('calibration', {}).get('method', 'isotonic')
        else:
            self.calibrated_model = None
            self.calibration_method = None
        
        # Label encoder for cuML (cuML requires numeric labels)
        self.label_encoder = cuLabelEncoder()
        self.label_mapping = {}  # Store mapping for inverse transform
    
    def _prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """Convert pandas DataFrame to numpy array for cuML.
        
        cuML works with numpy arrays or cuDF DataFrames.
        For now, we'll use numpy arrays.
        
        Args:
            X: Pandas DataFrame
            
        Returns:
            numpy array
        """
        # Convert to numpy (cuML can work with numpy arrays)
        # For better performance with large datasets, consider converting to cuDF
        return X.values.astype(np.float32)  # cuML prefers float32
    
    def _prepare_labels(self, y: pd.Series) -> np.ndarray:
        """Prepare labels for cuML (must be numeric).
        
        Args:
            y: Pandas Series with labels
            
        Returns:
            numpy array with numeric labels
        """
        # cuML requires numeric labels (0, 1, 2, ...)
        # Map -1, 0, 1 to 0, 1, 2
        unique_labels = sorted(y.unique())
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping
        
        # Encode labels
        y_encoded = y.map(label_mapping).values.astype(np.int32)
        return y_encoded
    
    def _decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode numeric labels back to original format.
        
        Args:
            y_encoded: Encoded labels
            
        Returns:
            Decoded labels
        """
        return np.array([self.label_mapping.get(int(label), label) for label in y_encoded])
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_X: Optional[pd.DataFrame] = None,
        calibration_y: Optional[pd.Series] = None
    ) -> 'RandomForestGPU':
        """Fit the GPU Random Forest model.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target labels (pandas Series)
            calibration_X: Optional calibration set features (not used by cuML directly)
            calibration_y: Optional calibration set labels
            
        Returns:
            self
        """
        logger.info(f"Training GPU Random Forest on {len(X)} samples with {len(X.columns)} features")
        
        # Prepare data for cuML
        X_array = self._prepare_data(X)
        y_encoded = self._prepare_labels(y)
        
        # Fit model
        self.model.fit(X_array, y_encoded)
        
        logger.info("GPU Random Forest trained successfully")
        
        # Calibration if enabled (using sklearn on GPU predictions)
        if self.use_calibration:
            if calibration_X is not None and calibration_y is not None:
                logger.info("Calibrating model probabilities (using sklearn on GPU predictions)")
                # Get uncalibrated probabilities from GPU model
                cal_X_array = self._prepare_data(calibration_X)
                cal_y_encoded = self._prepare_labels(calibration_y)
                
                # Get probabilities from GPU model
                proba = self.model.predict_proba(cal_X_array)
                
                # Use sklearn calibration
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.base import BaseEstimator, ClassifierMixin
                
                # Create a wrapper to use GPU model with sklearn calibration
                class GPUModelWrapper(BaseEstimator, ClassifierMixin):
                    def __init__(self, gpu_model):
                        self.gpu_model = gpu_model
                    
                    def predict_proba(self, X):
                        return self.gpu_model.predict_proba(X)
                    
                    def predict(self, X):
                        return self.gpu_model.predict(X)
                
                wrapper = GPUModelWrapper(self.model)
                self.calibrated_model = CalibratedClassifierCV(
                    wrapper,
                    method=self.calibration_method,
                    cv='prefit'
                )
                # For calibration, we need to use the encoded labels
                self.calibrated_model.fit(cal_X_array, cal_y_encoded)
            else:
                logger.warning("Calibration enabled but no calibration data provided")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            
        Returns:
            Predicted labels (original format, e.g., -1, 0, 1)
        """
        X_array = self._prepare_data(X)
        
        if self.calibrated_model is not None:
            predictions_encoded = self.calibrated_model.predict(X_array)
        else:
            predictions_encoded = self.model.predict(X_array)
        
        # Decode labels back to original format
        return self._decode_labels(predictions_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            
        Returns:
            Probability array (n_samples, n_classes)
        """
        X_array = self._prepare_data(X)
        
        if self.calibrated_model is not None:
            proba = self.calibrated_model.predict_proba(X_array)
        else:
            proba = self.model.predict_proba(X_array)
        
        # Convert to numpy array (cuML returns cupy arrays)
        if hasattr(proba, 'get'):  # cupy array
            proba = proba.get()
        
        return proba.astype(np.float64)
    
    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importances.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        importances = self.model.feature_importances_
        
        # Convert cupy array to numpy if needed
        if hasattr(importances, 'get'):
            importances = importances.get()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

