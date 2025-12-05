"""Unit tests for HMM features."""

import pytest
import pandas as pd
from src.features.hmm_features import create_macro_hmm_features, create_micro_hmm_features
from tests.conftest import sample_bars


@pytest.mark.unit
class TestHMMFeatures:
    """Test HMM feature engineering."""
    
    def test_create_macro_hmm_features(self, sample_bars):
        """Test macro HMM feature creation."""
        config = {
            'window': 50
        }
        
        features = create_macro_hmm_features(sample_bars, config)
        
        assert len(features) == len(sample_bars)
        assert not features.empty
    
    def test_create_micro_hmm_features(self, sample_bars, sample_ticks):
        """Test micro HMM feature creation."""
        config = {
            'window': 20
        }
        
        features = create_micro_hmm_features(sample_bars, sample_ticks, config)
        
        assert len(features) == len(sample_bars)
        assert not features.empty

