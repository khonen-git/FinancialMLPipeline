"""Unit tests for Macro HMM model."""

import pytest
import pandas as pd
import numpy as np
from src.models.hmm_macro import MacroHMM


@pytest.mark.unit
class TestMacroHMM:
    """Test MacroHMM model."""
    
    def test_hmm_macro_init(self):
        """Test MacroHMM initialization."""
        config = {
            'n_states': 3,
            'covariance_type': 'full',
            'n_init': 10,
            'max_iter': 100
        }
        
        hmm = MacroHMM(config)
        
        assert hmm is not None
        assert hmm.n_states == 3
    
    def test_hmm_macro_fit_predict(self, sample_bars):
        """Test fit and predict."""
        config = {
            'n_states': 3,
            'covariance_type': 'full',
            'n_init': 10,
            'max_iter': 100
        }
        
        hmm = MacroHMM(config)
        
        # Create features DataFrame (HMM expects features, not raw bars)
        features = pd.DataFrame({
            'ret_long': np.random.randn(len(sample_bars)),
            'vol_long': np.random.uniform(0.0001, 0.001, len(sample_bars)),
            'trend_slope': np.random.randn(len(sample_bars)),
            'trend_strength': np.random.uniform(0, 1, len(sample_bars))
        }, index=sample_bars.index)
        
        # Fit
        hmm.fit(features)
        
        # Predict
        predictions = hmm.predict(features)
        
        assert len(predictions) == len(features)
        assert all(p in range(hmm.n_states) for p in predictions)

