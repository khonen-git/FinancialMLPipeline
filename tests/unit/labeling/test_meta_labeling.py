"""Unit tests for meta-labeling."""

import pytest
import pandas as pd
import numpy as np
from src.labeling.meta_labeling import create_meta_labels, add_meta_label_features


@pytest.mark.unit
class TestMetaLabeling:
    """Test meta-labeling functionality."""
    
    def test_create_meta_labels_basic(self):
        """Test basic meta-label creation."""
        labels = pd.DataFrame({
            'label': [1, -1, 0, 1, -1],
            'pnl': [0.001, -0.001, 0.0005, 0.002, -0.0005]
        })
        
        meta_labels = create_meta_labels(labels)
        
        assert len(meta_labels) == len(labels)
        assert all(m in [0, 1] for m in meta_labels)
        # PnL > 0 should have meta-label = 1
        assert meta_labels.iloc[0] == 1  # pnl = 0.001 > 0
        assert meta_labels.iloc[1] == 0  # pnl = -0.001 < 0
    
    def test_create_meta_labels_missing_pnl(self):
        """Test error when pnl column is missing."""
        labels = pd.DataFrame({
            'label': [1, -1, 0]
            # Missing pnl column
        })
        
        with pytest.raises(ValueError, match="must contain 'pnl' column"):
            create_meta_labels(labels)
    
    def test_add_meta_label_features(self):
        """Test adding meta-label features."""
        features = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        base_model_proba = pd.Series([0.6, 0.7, 0.5])
        regime_features = pd.DataFrame({'regime': [0, 1, 0]})
        volatility_features = pd.DataFrame({'vol': [0.01, 0.02, 0.01]})
        spread_features = pd.DataFrame({'spread': [0.0001, 0.0002, 0.0001]})
        
        combined = add_meta_label_features(
            features, base_model_proba, regime_features,
            volatility_features, spread_features
        )
        
        assert len(combined) == len(features)
        assert len(combined.columns) > len(features.columns)

