"""Unit tests for Random Forest CPU model."""

import pytest
import pandas as pd
import numpy as np
from src.models.rf_cpu import RandomForestCPU


@pytest.mark.unit
class TestRandomForestCPU:
    """Test RandomForestCPU model."""
    
    def test_rf_cpu_init(self):
        """Test RandomForestCPU initialization."""
        config = {
            'params': {
                'n_estimators': 10,
                'max_depth': 5,
                'min_samples_leaf': 5,
                'n_jobs': 1,
                'class_weight': 'balanced'
            },
            'calibration': {
                'enabled': False
            }
        }
        
        model = RandomForestCPU(config)
        
        assert model is not None
        assert model.model is not None
        assert model.use_calibration == False
    
    def test_rf_cpu_init_missing_params(self):
        """Test error when params are missing."""
        config = {
            'calibration': {'enabled': False}
        }
        
        with pytest.raises(ValueError, match="Missing required config: params"):
            RandomForestCPU(config)
    
    def test_rf_cpu_fit_predict(self):
        """Test fit and predict."""
        config = {
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'min_samples_leaf': 2,
                'n_jobs': 1,
                'class_weight': 'balanced'
            },
            'calibration': {'enabled': False}
        }
        
        model = RandomForestCPU(config)
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([-1, 1], 100))
        
        # Fit
        model.fit(X, y)
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert len(probabilities) == len(X)
        assert all(p in [-1, 1] for p in predictions)

