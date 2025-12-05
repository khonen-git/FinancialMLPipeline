"""Unit tests for MA slope features."""

import pytest
import pandas as pd
from src.features.ma_slopes import create_ma_slope_features, create_ma_cross_features
from tests.conftest import sample_bars


@pytest.mark.unit
class TestMASlopeFeatures:
    """Test MA slope feature engineering."""
    
    def test_create_ma_slope_features(self, sample_bars):
        """Test MA slope feature creation."""
        periods = [5, 10, 20]
        
        features = create_ma_slope_features(sample_bars, periods)
        
        assert len(features) == len(sample_bars)
        assert not features.empty
        
        # Should have features for each period
        for period in periods:
            assert f'ma_{period}' in features.columns or f'ma_{period}_slope' in features.columns
    
    def test_create_ma_cross_features(self, sample_bars):
        """Test MA cross feature creation."""
        features = create_ma_cross_features(sample_bars)
        
        assert len(features) == len(sample_bars)
        assert not features.empty

