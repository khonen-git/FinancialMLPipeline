"""Random Forest model (CPU - sklearn)."""

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class RandomForestCPU:
    """Random Forest model using sklearn."""
    
    def __init__(self, config: dict):
        """Initialize Random Forest model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        params = config.get('params', {})
        
        self.model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5),
            n_jobs=params.get('n_jobs', -1),
            class_weight=params.get('class_weight', 'balanced'),
            random_state=42
        )
        
        self.calibrated_model = None
        self.use_calibration = config.get('calibration', {}).get('enabled', False)
        self.calibration_method = config.get('calibration', {}).get('method', 'isotonic')
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibration_X: pd.DataFrame = None,
        calibration_y: pd.Series = None
    ) -> 'RandomForestCPU':
        """Fit the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target labels
            calibration_X: Optional calibration set features
            calibration_y: Optional calibration set labels
            
        Returns:
            self
        """
        logger.info(f"Training Random Forest on {len(X)} samples")
        
        self.model.fit(X, y)
        
        logger.info(f"Random Forest trained. OOB score: {getattr(self.model, 'oob_score_', 'N/A')}")
        
        # Calibration if enabled
        if self.use_calibration:
            if calibration_X is not None and calibration_y is not None:
                logger.info("Calibrating model probabilities")
                self.calibrated_model = CalibratedClassifierCV(
                    self.model,
                    method=self.calibration_method,
                    cv='prefit'
                )
                self.calibrated_model.fit(calibration_X, calibration_y)
            else:
                logger.warning("Calibration enabled but no calibration data provided")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array (n_samples, n_classes)
        """
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Get feature importances.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        importances = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

