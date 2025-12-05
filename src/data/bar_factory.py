"""Factory for creating bar builders with different backends."""

import logging
from typing import Union

from .bars import BarBuilder
from .bars_optimized import BarBuilderOptimized

logger = logging.getLogger(__name__)

# Try to import optional backends
_polars_available = False
_cudf_available = False

try:
    from .bars_polars import BarBuilderPolars
    _polars_available = True
except ImportError:
    BarBuilderPolars = None

try:
    from .bars_cudf import BarBuilderCuDF
    _cudf_available = True
except ImportError:
    BarBuilderCuDF = None


def create_bar_builder(config: dict, backend: str = 'pandas') -> Union[BarBuilder, BarBuilderOptimized, BarBuilderPolars, BarBuilderCuDF]:
    """Create a bar builder with the specified backend.
    
    Args:
        config: Bar configuration
        backend: Backend to use ('pandas', 'optimized', 'polars', 'cudf')
    
    Returns:
        Bar builder instance
    
    Raises:
        ValueError: If backend is not supported
        ImportError: If requested backend is not available
    """
    backend = backend.lower()
    
    if backend == 'pandas':
        return BarBuilder(config)
    elif backend == 'optimized':
        return BarBuilderOptimized(config)
    elif backend == 'polars':
        if not _polars_available:
            raise ImportError(
                "Polars backend requested but polars is not installed. "
                "Install with: pip install polars"
            )
        return BarBuilderPolars(config)
    elif backend == 'cudf' or backend == 'gpu':
        if not _cudf_available:
            raise ImportError(
                "cuDF backend requested but cuDF is not installed. "
                "Install with: conda install -c rapidsai cudf"
            )
        return BarBuilderCuDF(config)
    else:
        raise ValueError(f"Unsupported bar builder backend: {backend}")


def get_available_backends() -> list[str]:
    """Get list of available bar builder backends."""
    backends = ['pandas', 'optimized']
    
    if _polars_available:
        backends.append('polars')
    
    if _cudf_available:
        backends.append('cudf')
    
    return backends

