"""FastAPI application main file."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.deployment.api.endpoints import predict, batch_predict, health, models

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Financial ML Pipeline Inference API",
    description="REST API for serving MLflow-trained trading models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(batch_predict.router)
app.include_router(health.router)
app.include_router(models.router)


@app.get("/")
async def root():
    """Root endpoint.
    
    Returns:
        Welcome message and API information
    """
    return {
        "message": "Financial ML Pipeline Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Financial ML Pipeline Inference API")
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down Financial ML Pipeline Inference API")

