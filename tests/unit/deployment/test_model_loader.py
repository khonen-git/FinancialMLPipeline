"""Unit tests for model loader."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pickle
from unittest.mock import Mock, patch, MagicMock
from src.deployment.model_loader import ModelLoader


# Simple pickleable model class for testing
class SimpleModel:
    """Simple model class for testing pickle functionality."""
    def predict(self, X):
        return np.array([1])


class TestModelLoader:
    """Test ModelLoader class."""
    
    def test_init_default(self):
        """Test ModelLoader initialization with default tracking URI."""
        loader = ModelLoader()
        assert loader is not None
        assert loader.cache == {}
    
    def test_init_custom_tracking_uri(self):
        """Test ModelLoader initialization with custom tracking URI."""
        loader = ModelLoader(mlflow_tracking_uri="file:./test_mlruns")
        assert loader is not None
    
    def test_load_from_file_exists(self):
        """Test loading model from existing pickle file."""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            model = SimpleModel()
            pickle.dump(model, f)
            file_path = f.name
        
        try:
            loader = ModelLoader()
            loaded_model = loader.load_from_file(file_path)
            
            assert loaded_model is not None
            # Cache key format is "file:<path>"
            cache_key = f"file:{file_path}"
            assert cache_key in loader.cache
            # Verify it's the same type
            assert hasattr(loaded_model, 'predict')
        finally:
            Path(file_path).unlink()
    
    def test_load_from_file_not_found(self):
        """Test loading model from non-existent file raises FileNotFoundError."""
        loader = ModelLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("nonexistent.pkl")
    
    @patch('src.deployment.model_loader.mlflow')
    def test_load_from_mlflow_success(self, mock_mlflow):
        """Test loading model from MLflow."""
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1]))
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        loader = ModelLoader()
        model = loader.load_from_mlflow("runs:/test_run/model")
        
        assert model is not None
        assert "runs:/test_run/model:cpu" in loader.cache
    
    @patch('src.deployment.model_loader.mlflow')
    def test_load_from_mlflow_cached(self, mock_mlflow):
        """Test that cached models are returned without reloading."""
        mock_model = Mock()
        mock_mlflow.sklearn.load_model.return_value = mock_model
        
        loader = ModelLoader()
        
        # First load
        model1 = loader.load_from_mlflow("runs:/test_run/model")
        
        # Second load should use cache
        model2 = loader.load_from_mlflow("runs:/test_run/model")
        
        # Should only call mlflow.load_model once
        assert mock_mlflow.sklearn.load_model.call_count == 1
        assert model1 is model2
    
    def test_predict_with_proba(self):
        """Test prediction with probabilities."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1]))
        mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
        
        loader = ModelLoader()
        features = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        
        predictions, probabilities = loader.predict(mock_model, features, return_proba=True)
        
        assert len(predictions) == 1
        assert predictions[0] == 1
        assert probabilities is not None
        assert len(probabilities) == 1
    
    def test_predict_without_proba(self):
        """Test prediction without probabilities."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, -1]))
        
        loader = ModelLoader()
        features = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [2.0, 3.0]})
        
        predictions = loader.predict(mock_model, features, return_proba=False)
        
        assert len(predictions) == 2
        assert predictions[0] == 1
        assert predictions[1] == -1
    
    def test_predict_no_predict_method(self):
        """Test that model without predict method raises AttributeError."""
        mock_model = Mock(spec=[])  # No predict method
        
        loader = ModelLoader()
        features = pd.DataFrame({"feature1": [1.0]})
        
        with pytest.raises(AttributeError, match="predict"):
            loader.predict(mock_model, features)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        loader = ModelLoader()
        loader.cache["test_key"] = Mock()
        
        assert len(loader.cache) == 1
        loader.clear_cache()
        assert len(loader.cache) == 0
    
    def test_get_cached_models(self):
        """Test getting list of cached model keys."""
        loader = ModelLoader()
        loader.cache["key1"] = Mock()
        loader.cache["key2"] = Mock()
        
        cached = loader.get_cached_models()
        assert len(cached) == 2
        assert "key1" in cached
        assert "key2" in cached

