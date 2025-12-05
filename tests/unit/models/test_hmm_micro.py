"""Unit tests for Micro HMM model."""

import pytest
import pandas as pd
import numpy as np
from src.models.hmm_micro import MicroHMM


@pytest.mark.unit
class TestMicroHMM:
    """Test MicroHMM model."""
    
    def test_hmm_micro_init(self):
        """Test MicroHMM initialization."""
        config = {
            'n_states': 3,
            'covariance_type': 'full',
            'n_init': 10,
            'max_iter': 100
        }
        
        hmm = MicroHMM(config)
        
        assert hmm is not None
        assert hmm.n_states == 3
    
    def test_hmm_micro_fit_predict(self, sample_bars):
        """Test fit and predict."""
        config = {
            'n_states': 3,
            'covariance_type': 'full',
            'n_init': 10,
            'max_iter': 100
        }
        
        hmm = MicroHMM(config)
        
        # Create features DataFrame (HMM expects features, not raw bars)
        features = pd.DataFrame({
            'of_imbalance': np.random.randn(len(sample_bars)),
            'spread': np.random.uniform(0.00001, 0.0001, len(sample_bars)),
            'spread_change': np.random.randn(len(sample_bars)),
            'tick_direction': np.random.choice([-1, 0, 1], len(sample_bars))
        }, index=sample_bars.index)
        
        # Fit
        hmm.fit(features)
        
        # Predict
        predictions = hmm.predict(features)
        
        assert len(predictions) == len(features)
        assert all(p in range(hmm.n_states) for p in predictions)

