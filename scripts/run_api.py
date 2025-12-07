"""Script to run the inference API server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000
    python scripts/run_api.py --mlflow-tracking-uri http://localhost:5000
"""

import argparse
import logging
import uvicorn
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Run the FastAPI inference server."""
    parser = argparse.ArgumentParser(description="Run Financial ML Pipeline Inference API")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: from environment or file:./mlruns)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        import os
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    
    logger.info(f"Starting Inference API on {args.host}:{args.port}")
    logger.info(f"MLflow tracking URI: {args.mlflow_tracking_uri or 'from environment'}")
    
    # Run uvicorn
    uvicorn.run(
        "src.deployment.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()

