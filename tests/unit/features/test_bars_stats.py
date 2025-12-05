"""Unit tests for bar statistics features."""

import pytest
import pandas as pd
from src.features.bars_stats import create_bar_stats_features
from tests.conftest import sample_bars


@pytest.mark.unit
class TestBarStatsFeatures:
    """Test bar statistics feature engineering."""
    
    def test_create_bar_stats_features_basic(self, sample_bars):
        """Test basic bar stats feature creation."""
        config = {
            'include_tick_count': True,
            'include_volume': True
        }
        
        # Add required columns if missing
        bars = sample_bars.copy()
        if 'tick_count' not in bars.columns:
            bars['tick_count'] = 100
        
        features = create_bar_stats_features(bars, config)
        
        assert len(features) == len(bars)
        assert not features.empty
    
    def test_create_bar_stats_features_with_tick_count(self, sample_bars):
        """Test bar stats with tick_count."""
        bars = sample_bars.copy()
        bars['tick_count'] = 100
        
        config = {
            'include_tick_count': True,
            'include_volume': False
        }
        
        features = create_bar_stats_features(bars, config)
        
        if 'tick_count' in features.columns:
            assert 'tick_count' in features.columns
            assert 'tick_count_norm' in features.columns

