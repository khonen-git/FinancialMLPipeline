"""Model factory for creating CPU or GPU models based on configuration."""

import logging
from typing import Union

logger = logging.getLogger(__name__)

# Try to import both models
try:
    from src.models.rf_cpu import RandomForestCPU
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False
    logger.warning("RandomForestCPU not available")

try:
    from src.models.rf_gpu import RandomForestGPU, CUML_AVAILABLE
    GPU_AVAILABLE = CUML_AVAILABLE if 'CUML_AVAILABLE' in globals() else False
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("RandomForestGPU not available (cuML not installed)")


def create_random_forest(config: dict) -> Union[RandomForestCPU, RandomForestGPU]:
    """Create Random Forest model (CPU or GPU) based on configuration.
    
    Args:
        config: Model configuration with 'backend' key:
            - 'cpu' or 'sklearn' -> CPU (scikit-learn)
            - 'gpu' or 'cuml' -> GPU (RAPIDS cuML)
        
    Returns:
        RandomForestCPU or RandomForestGPU instance
        
    Raises:
        ValueError: If backend is not supported or not available
    """
    backend = config.get('backend', 'cpu').lower()
    
    # Support both conventions: cpu/sklearn and gpu/cuml
    if backend in ['cpu', 'sklearn']:
        if not CPU_AVAILABLE:
            raise ImportError("RandomForestCPU not available")
        logger.info("Creating CPU Random Forest (scikit-learn)")
        return RandomForestCPU(config)
    
    elif backend in ['gpu', 'cuml']:
        if not GPU_AVAILABLE:
            raise ImportError(
                "RandomForestGPU not available. cuML is required.\n"
                "Install with: conda install -c rapidsai cuml cudatoolkit=11.8"
            )
        logger.info("Creating GPU Random Forest (RAPIDS cuML)")
        return RandomForestGPU(config)
    
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Must be 'cpu'/'sklearn' or 'gpu'/'cuml'. "
            f"Available backends: {get_available_backends()}"
        )


def get_available_backends() -> list[str]:
    """Get list of available backends.
    
    Returns:
        List of available backend names
    """
    backends = []
    if CPU_AVAILABLE:
        backends.append('cpu')
    if GPU_AVAILABLE:
        backends.append('gpu')
    return backends

