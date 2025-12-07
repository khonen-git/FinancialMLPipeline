"""Integration tests for inference API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from src.deployment.api.main import app
from src.deployment.api.dependencies import get_model_loader


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = Mock()
    model.predict = Mock(return_value=np.array([1]))
    model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
    model.classes_ = np.array([-1, 0, 1])
    return model


@pytest.fixture
def mock_model_loader(mock_model):
    """Create mock model loader."""
    loader = Mock()
    loader.load_from_mlflow = Mock(return_value=mock_model)
    loader.predict = Mock(return_value=(np.array([1]), np.array([[0.1, 0.2, 0.7]])))
    loader.get_cached_models = Mock(return_value=["model1", "model2"])
    return loader


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @patch('src.deployment.api.endpoints.health._check_gpu_available')
    @patch('src.deployment.api.endpoints.health.os.getenv')
    def test_health_check_success(self, mock_getenv, mock_gpu):
        """Test successful health check."""
        mock_getenv.return_value = "file:./mlruns"
        mock_gpu.return_value = False
        
        client = TestClient(app)
        
        # Mock model loader dependency
        with patch('src.deployment.api.dependencies.get_model_loader') as mock_get_loader:
            mock_loader = Mock()
            mock_loader.get_cached_models = Mock(return_value=["model1"])
            mock_get_loader.return_value = mock_loader
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "mlflow_connected" in data
            assert "gpu_available" in data
            assert "cached_models" in data


class TestPredictEndpoint:
    """Test prediction endpoint."""
    
    def test_predict_success(self, mock_model_loader, mock_model):
        """Test successful prediction."""
        # Setup mock to return model and predictions correctly
        mock_model_loader.load_from_mlflow.return_value = mock_model
        mock_model_loader.predict.return_value = (np.array([1]), np.array([[0.1, 0.2, 0.7]]))
        
        # Override dependency
        app.dependency_overrides[get_model_loader] = lambda: mock_model_loader
        
        try:
            client = TestClient(app)
            
            request_data = {
                "model_uri": "runs:/test_run/model",
                "features": {
                    "bar_id": 12345,
                    "bidPrice": 1.1000,
                    "askPrice": 1.1005
                },
                "return_proba": True
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert data["prediction"] == 1
        finally:
            # Clean up override
            app.dependency_overrides.clear()
    
    def test_predict_invalid_model_uri(self):
        """Test prediction with invalid model URI."""
        mock_loader = Mock()
        mock_loader.load_from_mlflow = Mock(side_effect=FileNotFoundError("Model not found"))
        
        # Override dependency
        app.dependency_overrides[get_model_loader] = lambda: mock_loader
        
        try:
            client = TestClient(app)
            
            request_data = {
                "model_uri": "runs:/invalid/model",
                "features": {"bar_id": 12345}
            }
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()
    
    def test_predict_invalid_request(self):
        """Test prediction with invalid request format."""
        client = TestClient(app)
        
        # Missing required fields
        response = client.post("/predict", json={})
        
        assert response.status_code == 422  # Validation error


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_success(self, mock_model_loader, mock_model):
        """Test successful batch prediction."""
        # Setup mock - return numpy array that can be converted
        predictions_array = np.array([1, -1], dtype=np.int32)
        mock_model_loader.load_from_mlflow.return_value = mock_model
        
        # Mock predict to return just array when return_proba=False
        def mock_predict(model, features, return_proba=False):
            if return_proba:
                return (predictions_array, None)
            else:
                return predictions_array
        
        mock_model_loader.predict = mock_predict
        
        # Override dependency
        app.dependency_overrides[get_model_loader] = lambda: mock_model_loader
        
        try:
            client = TestClient(app)
            
            request_data = {
                "model_uri": "runs:/test_run/model",
                "features_list": [
                    {"bar_id": 12345, "bidPrice": 1.1000},
                    {"bar_id": 12346, "bidPrice": 1.1001}
                ],
                "return_proba": False
            }
            
            response = client.post("/predict/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "count" in data
            assert data["count"] == 2
            assert len(data["predictions"]) == 2
        finally:
            app.dependency_overrides.clear()


class TestModelsEndpoint:
    """Test models listing endpoint."""
    
    @patch('src.deployment.api.endpoints.models.MlflowClient')
    @patch('src.deployment.api.dependencies.get_model_loader')
    def test_list_models_success(self, mock_get_loader, mock_mlflow_client):
        """Test successful model listing."""
        # Mock MLflow client
        mock_client = Mock()
        mock_experiment = Mock()
        mock_experiment.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        mock_run = Mock()
        mock_run.info.run_id = "test_run"
        mock_run.data.metrics = {"accuracy": 0.85}
        mock_run.data.params = {"n_estimators": 200}
        mock_run.data.tags = {}
        mock_client.search_runs.return_value = [mock_run]
        
        mock_artifact = Mock()
        mock_artifact.path = "model"
        mock_client.list_artifacts.return_value = [mock_artifact]
        
        mock_mlflow_client.return_value = mock_client
        mock_get_loader.return_value = Mock()
        
        client = TestClient(app)
        
        response = client.get("/models?experiment_name=financial_ml")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self):
        """Test root endpoint returns API info."""
        client = TestClient(app)
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

