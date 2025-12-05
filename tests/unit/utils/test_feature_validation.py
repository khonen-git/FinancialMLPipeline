"""Unit tests for feature validation."""

import pytest
import pandas as pd
import numpy as np
from src.utils.feature_validation import validate_features, validate_feature_consistency


@pytest.mark.unit
class TestFeatureValidation:
    """Test feature validation functionality."""
    
    def test_validate_features_basic(self):
        """Test basic feature validation."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        
        results = validate_features(X, name="test_features")
        
        assert results['is_valid'] == True
        assert results['n_samples'] == 5
        assert results['n_features'] == 3
    
    def test_validate_features_with_nan(self):
        """Test validation with NaN values."""
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        results = validate_features(X, name="test_features", strict=False)
        
        assert results['n_nan'] > 0
        assert 'feature1' in results['nan_features']
    
    def test_validate_features_constant(self):
        """Test validation with constant features."""
        X = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1],  # Constant
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        results = validate_features(X, name="test_features", strict=False)
        
        assert 'feature1' in results['constant_features']
    
    def test_validate_features_empty(self):
        """Test validation with empty DataFrame."""
        X = pd.DataFrame()
        
        results = validate_features(X, name="test_features", strict=False)
        
        assert results['is_valid'] == False
        assert len(results['errors']) > 0
    
    def test_validate_feature_consistency(self):
        """Test feature consistency validation."""
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        X_test = pd.DataFrame({
            'feature1': [4, 5, 6],
            'feature2': [0.4, 0.5, 0.6]
        })
        
        results = validate_feature_consistency(X_train, X_test)
        
        assert results['is_consistent'] == True
        # Check for missing or extra features
        missing = results.get('missing_features', [])
        extra = results.get('extra_features', [])
        assert len(missing) == 0
        assert len(extra) == 0

