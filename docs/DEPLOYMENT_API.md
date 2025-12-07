# API Deployment Guide

This document describes how to deploy and use the inference API for the Financial ML Pipeline.

---

## 1. Overview

The inference API is a FastAPI-based REST service that provides endpoints for:
- Single predictions (`POST /predict`)
- Batch predictions (`POST /predict/batch`)
- Health checks (`GET /health`)
- Model listing (`GET /models`)

The API loads models from MLflow tracking server and serves predictions in real-time.

---

## 2. Quick Start

### 2.1 Local Development

Run the API locally:

```bash
# Install dependencies
pip install -e .

# Run API server
python scripts/run_api.py

# Or with custom settings
python scripts/run_api.py --host 0.0.0.0 --port 8000 --mlflow-tracking-uri file:./mlruns
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2.2 Docker Compose (On-Premise)

Deploy with Docker Compose:

```bash
# Build and start services
cd docker
docker-compose up -d

# View logs
docker-compose logs -f inference-api

# Stop services
docker-compose down
```

This will start:
- MLflow tracking server on port 5000
- Inference API on port 8000

---

## 3. API Endpoints

### 3.1 Single Prediction

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "model_uri": "runs:/abc123def456/model",
  "features": {
    "bar_id": 12345,
    "bidPrice": 1.1000,
    "askPrice": 1.1005,
    "spread": 0.0005,
    "return_1": 0.001,
    "return_5": 0.002
  },
  "return_proba": true
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "probabilities": {
    "-1": 0.05,
    "0": 0.10,
    "1": 0.85
  },
  "confidence": "high"
}
```

### 3.2 Batch Prediction

**Endpoint**: `POST /predict/batch`

**Request**:
```json
{
  "model_uri": "runs:/abc123def456/model",
  "features_list": [
    {"bar_id": 12345, "bidPrice": 1.1000, "askPrice": 1.1005},
    {"bar_id": 12346, "bidPrice": 1.1001, "askPrice": 1.1006}
  ],
  "return_proba": false
}
```

**Response**:
```json
{
  "predictions": [1, -1],
  "probabilities": null,
  "count": 2
}
```

### 3.3 Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "mlflow_connected": true,
  "gpu_available": false,
  "cached_models": 2
}
```

### 3.4 List Models

**Endpoint**: `GET /models?experiment_name=financial_ml&limit=100`

**Response**:
```json
{
  "models": [
    {
      "model_uri": "runs:/abc123/model",
      "run_id": "abc123",
      "experiment_id": "1",
      "metrics": {
        "accuracy": 0.85,
        "sharpe_ratio": 1.5
      },
      "params": {
        "n_estimators": 200
      },
      "tags": {
        "asset": "EURUSD"
      }
    }
  ],
  "count": 1
}
```

---

## 4. Model URI Formats

The API supports multiple MLflow model URI formats:

- **Run URI**: `runs:/<run_id>/model`
  - Example: `runs:/abc123def456/model`
  - Loads model from a specific MLflow run

- **Model Registry URI**: `models:/<model_name>/<version>`
  - Example: `models:/eurusd_rf/1`
  - Loads model from MLflow Model Registry

- **File URI**: `file:///path/to/model.pkl`
  - Example: `file:///app/models/model.pkl`
  - Loads model from local file (for development/testing)

---

## 5. Configuration

### 5.1 Environment Variables

Set these environment variables before starting the API:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="financial_ml"
```

### 5.2 Configuration File

Edit `configs/deployment/api.yaml`:

```yaml
deployment:
  api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    mlflow:
      tracking_uri: "${MLFLOW_TRACKING_URI:file:./mlruns}"
      experiment_name: "financial_ml"
```

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md#17-deployment--api-configuration) for complete configuration options.

---

## 6. Docker Deployment

### 6.1 Build Inference Image

```bash
docker build -f docker/Dockerfile.inference -t financial-ml-api:latest .
```

### 6.2 Run Container

```bash
docker run -d \
  --name financial-ml-api \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -v $(pwd)/mlruns:/app/mlruns:ro \
  financial-ml-api:latest
```

### 6.3 Docker Compose

Use the provided `docker/docker-compose.yml`:

```bash
cd docker
docker-compose up -d
```

This starts:
- MLflow server (port 5000)
- Inference API (port 8000)

---

## 7. Model Loading

### 7.1 From MLflow

Models are automatically loaded from MLflow when requested:

1. First request to a model URI triggers loading
2. Model is cached in memory for subsequent requests
3. Cache is cleared on API restart

### 7.2 Supported Model Types

- **scikit-learn models**: Loaded via `mlflow.sklearn.load_model()`
- **Generic models**: Loaded via `mlflow.pyfunc.load_model()`
- **Pickle files**: Loaded via `pickle.load()` (for local files)

### 7.3 Model Caching

- Models are cached in memory after first load
- Cache size is configurable (`deployment.api.cache.max_models`)
- Cache is cleared on API restart

---

## 8. Feature Validation

The API expects features in the same format as used during training.

**Required features** (example):
- `bar_id`: Bar identifier
- `bidPrice`: Bid price
- `askPrice`: Ask price
- `spread`: Spread (askPrice - bidPrice)
- Price features: `return_1`, `return_5`, etc.
- Microstructure features: `order_flow_imbalance`, etc.

**Note**: The exact feature set depends on your training configuration. Check your model's feature requirements in MLflow.

---

## 9. Error Handling

The API returns standard HTTP status codes:

- **200 OK**: Successful prediction
- **404 Not Found**: Model not found
- **422 Unprocessable Entity**: Invalid request format
- **500 Internal Server Error**: Prediction failed

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## 10. Performance Considerations

### 10.1 Model Loading

- First request to a model URI may be slow (model loading)
- Subsequent requests use cached model (fast)
- Consider pre-loading models in production

### 10.2 Batch Predictions

- Batch predictions are more efficient than multiple single predictions
- Use batch endpoint for processing multiple samples

### 10.3 GPU Support

- GPU models (cuML) are supported if GPU is available
- Set `backend="gpu"` when loading models (future enhancement)

---

## 11. Security

### 11.1 CORS Configuration

In production, restrict CORS origins:

```yaml
deployment:
  api:
    cors:
      allow_origins: ["https://yourdomain.com"]
      allow_credentials: true
```

### 11.2 Authentication

Currently, the API has no authentication. For production:

- Add API key authentication
- Use OAuth2/JWT tokens
- Implement rate limiting

### 11.3 Network Security

- Use HTTPS in production
- Restrict API access via firewall
- Use reverse proxy (nginx, Traefik) for SSL termination

---

## 12. Monitoring

### 12.1 Health Checks

Use the `/health` endpoint for:
- Load balancer health checks
- Container orchestration (Kubernetes liveness/readiness)
- Monitoring systems

### 12.2 Logging

API logs are written to stdout/stderr. In production:
- Use structured logging (JSON)
- Forward logs to centralized system (ELK, Loki)
- Set appropriate log levels

### 12.3 Metrics

Consider adding:
- Request rate (requests/second)
- Latency (p50, p95, p99)
- Error rate
- Model cache hit rate

---

## 13. Troubleshooting

### 13.1 Model Not Found

**Error**: `404 Model not found`

**Solutions**:
- Verify model URI is correct
- Check MLflow tracking URI is accessible
- Ensure model was logged to MLflow during training

### 13.2 Feature Mismatch

**Error**: `500 Prediction failed` (feature shape mismatch)

**Solutions**:
- Verify features match training feature set
- Check feature order and names
- Ensure all required features are present

### 13.3 MLflow Connection Issues

**Error**: `mlflow_connected: false` in health check

**Solutions**:
- Verify MLflow server is running
- Check `MLFLOW_TRACKING_URI` environment variable
- Test MLflow connection: `mlflow ui --backend-store-uri <uri>`

---

## 14. Example Usage

### 14.1 Python Client

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "model_uri": "runs:/abc123/model",
        "features": {
            "bar_id": 12345,
            "bidPrice": 1.1000,
            "askPrice": 1.1005,
            "spread": 0.0005
        },
        "return_proba": True
    }
)
result = response.json()
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']}")
```

### 14.2 cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_uri": "runs:/abc123/model",
    "features": {
      "bar_id": 12345,
      "bidPrice": 1.1000,
      "askPrice": 1.1005
    }
  }'
```

---

## 15. Next Steps

- Add authentication/authorization
- Implement rate limiting
- Add request/response logging
- Set up monitoring and alerting
- Optimize model loading and caching
- Add support for model versioning

---

**See also**:
- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md#17-deployment--api-configuration) - Configuration reference
- [MLFLOW.md](MLFLOW.md) - MLflow integration guide
- [ARCH_INFRA.md](ARCH_INFRA.md#10-deployment-strategy) - Infrastructure overview

