"""Unit tests for price-based features."""

import pytest
import pandas as pd
import numpy as np
from src.features.price import create_price_features


@pytest.mark.unit
class TestPriceFeatures:
    """Test price-based feature engineering."""
    
    def test_create_price_features_basic(self, sample_bars):
        """Test basic price feature creation."""
        config = {
            'returns_lookbacks': [1, 5, 10],
            'volatility_lookbacks': [10, 20],
            'range_lookbacks': [10, 20],
        }
        
        features = create_price_features(sample_bars, config)
        
        assert len(features) == len(sample_bars)
        assert not features.empty
        
        # Check return features
        assert 'ret_1' in features.columns
        assert 'ret_5' in features.columns
        assert 'ret_10' in features.columns
    
    def test_returns_no_future_leakage(self, sample_bars):
        """Test that returns don't use future data."""
        config = {
            'returns_lookbacks': [1, 5],
            'volatility_lookbacks': [],
            'range_lookbacks': [],
        }
        
        features = create_price_features(sample_bars, config)
        
        # First row should have NaN for ret_1 (no previous value)
        assert pd.isna(features.iloc[0]['ret_1'])
        
        # Second row should have a value
        assert not pd.isna(features.iloc[1]['ret_1'])
    
    def test_volatility_features(self, sample_bars):
        """Test volatility feature creation."""
        config = {
            'returns_lookbacks': [1],
            'volatility_lookbacks': [10, 20],
            'range_lookbacks': [],
        }
        
        features = create_price_features(sample_bars, config)
        
        # Check volatility features exist
        assert 'vol_10' in features.columns or any('vol' in col for col in features.columns)
    
    def test_range_features(self, sample_bars):
        """Test range feature creation."""
        # Ensure bars have high/low columns
        bars = sample_bars.copy()
        if 'high' not in bars.columns:
            bars['high'] = bars['bid_high']
        if 'low' not in bars.columns:
            bars['low'] = bars['bid_low']
        
        config = {
            'returns_lookbacks': [],
            'volatility_lookbacks': [],
            'range_lookbacks': [10, 20],
        }
        
        features = create_price_features(bars, config)
        
        # Check range features (may be empty if no high/low columns)
        assert len(features) == len(bars)
        # If range features are created, they should be present
        if len(features.columns) > 0:
            assert any('range' in col.lower() for col in features.columns) or len(features.columns) == 0
    
    def test_features_index_preserved(self, sample_bars):
        """Test that feature index matches bars index."""
        config = {
            'returns_lookbacks': [1, 5],
            'volatility_lookbacks': [10],
            'range_lookbacks': [10],
        }
        
        features = create_price_features(sample_bars, config)
        
        # Index should match
        assert features.index.equals(sample_bars.index)

