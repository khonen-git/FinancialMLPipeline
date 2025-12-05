"""Unit tests for feature engineer module."""

import pytest
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from src.pipeline.feature_engineer import engineer_features
from tests.conftest import sample_bars


@pytest.mark.unit
class TestFeatureEngineer:
    """Test feature engineering module."""
    
    def test_engineer_features_basic(self, sample_bars):
        """Test basic feature engineering."""
        cfg = OmegaConf.create({
            'features': {
                'returns_lookbacks': [1, 5],
                'volatility_lookbacks': [10],
                'range_lookbacks': [10],
                'spread_stats_lookbacks': [5],
                'order_flow_lookbacks': [5],
            },
            'experiment': {
                'audit_data_leakage': False
            }
        })
        
        features = engineer_features(sample_bars, cfg)
        
        assert len(features) == len(sample_bars)
        assert len(features.columns) > 0
    
    def test_engineer_features_all_types(self, sample_bars):
        """Test feature engineering with all feature types."""
        cfg = OmegaConf.create({
            'features': {
                'returns_lookbacks': [1, 5, 10],
                'volatility_lookbacks': [10, 20],
                'range_lookbacks': [10, 20],
                'spread_stats_lookbacks': [5, 10],
                'order_flow_lookbacks': [5, 10],
            },
            'experiment': {
                'audit_data_leakage': False
            }
        })
        
        features = engineer_features(sample_bars, cfg)
        
        # Should have features from all modules
        assert len(features.columns) > 0
        assert len(features) == len(sample_bars)

