"""Model implementations."""

from src.models.rf_cpu import RandomForestCPU
from src.models.model_factory import create_random_forest, get_available_backends

# Try to import GPU model (optional)
try:
    from src.models.rf_gpu import RandomForestGPU
    __all__ = ['RandomForestCPU', 'RandomForestGPU', 'create_random_forest', 'get_available_backends']
except ImportError:
    __all__ = ['RandomForestCPU', 'create_random_forest', 'get_available_backends']
