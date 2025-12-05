"""Logging configuration for the pipeline.

Uses standard Python logging module as per CODING_STANDARDS.md.
Logs are saved to MLflow artifacts as per ARCH_INFRA.md.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import io


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    mlflow_log_handler: Optional[io.StringIO] = None,
) -> Optional[io.StringIO]:
    """Setup logging configuration for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs (deprecated, use MLflow instead)
        format_string: Custom format string for log messages
        mlflow_log_handler: Optional StringIO handler for MLflow logging
        
    Returns:
        StringIO handler if created, None otherwise. This should be logged to MLflow artifacts.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Create StringIO handler for MLflow if not provided
    if mlflow_log_handler is None:
        mlflow_log_handler = io.StringIO()
    
    # Add StringIO handler for MLflow artifacts
    mlflow_handler = logging.StreamHandler(mlflow_log_handler)
    mlflow_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(mlflow_handler)
    
    # Legacy file handler (deprecated, but kept for backward compatibility)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )
    
    # Suppress noisy loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return mlflow_log_handler


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

