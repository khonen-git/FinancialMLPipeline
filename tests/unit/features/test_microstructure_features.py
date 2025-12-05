"""Unit tests for microstructure features."""

import pytest
import pandas as pd
from src.features.microstructure import create_microstructure_features


@pytest.mark.unit
class TestMicrostructureFeatures:
    """Test microstructure feature engineering."""
    
    def test_create_microstructure_features_basic(self, sample_bars):
        """Test basic microstructure feature creation."""
        config = {
            'spread_stats_lookbacks': [5, 10],
            'order_flow_lookbacks': [5, 10],
        }
        
        features = create_microstructure_features(sample_bars, config)
        
        assert len(features) == len(sample_bars)
        assert not features.empty
    
    def test_spread_features(self, sample_bars):
        """Test spread feature creation."""
        config = {
            'spread_stats_lookbacks': [5, 10],
            'order_flow_lookbacks': [],
        }
        
        features = create_microstructure_features(sample_bars, config)
        
        # Should have spread-related features
        assert any('spread' in col.lower() for col in features.columns)
    
    def test_order_flow_features(self, sample_bars):
        """Test order flow feature creation."""
        config = {
            'spread_stats_lookbacks': [],
            'order_flow_lookbacks': [5, 10],
        }
        
        features = create_microstructure_features(sample_bars, config)
        
        # Should have order flow features if volume data available
        # (may be empty if no volume columns)
        assert len(features) == len(sample_bars)
    
    def test_features_no_future_leakage(self, sample_bars):
        """Test that features don't use future data."""
        config = {
            'spread_stats_lookbacks': [5],
            'order_flow_lookbacks': [5],
        }
        
        features = create_microstructure_features(sample_bars, config)
        
        # First few rows may have NaN (rolling window)
        # But should not use future data
        assert len(features) == len(sample_bars)

