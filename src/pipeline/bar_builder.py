"""Bar construction module.

Handles:
- Session calendar initialization
- Bar construction from ticks
"""

import logging
from typing import Tuple
from omegaconf import DictConfig
import pandas as pd

from src.labeling.session_calendar import SessionCalendar
from src.data.bar_factory import create_bar_builder, get_available_backends

logger = logging.getLogger(__name__)


def build_bars(ticks: pd.DataFrame, cfg: DictConfig) -> Tuple[pd.DataFrame, SessionCalendar]:
    """Build bars from ticks and initialize session calendar.
    
    Args:
        ticks: Cleaned tick DataFrame with timestamp index
        cfg: Hydra configuration with session and data.bars sections
        
    Returns:
        Tuple of (bars DataFrame, SessionCalendar instance)
    """
    # Step 3: Session calendar
    logger.info("Step 3: Initializing session calendar")
    calendar = SessionCalendar(cfg.session)
    
    # Step 4: Bar construction
    logger.info("Step 4: Bar construction")
    # Get backend from config (default: 'optimized' for performance)
    bar_backend = cfg.data.bars.get('backend', 'optimized')
    available_backends = get_available_backends()
    logger.info(f"Available bar backends: {available_backends}, using: {bar_backend}")
    
    bar_builder = create_bar_builder(cfg.data.bars, backend=bar_backend)
    bars = bar_builder.build_bars(ticks)
    logger.info(f"Built {len(bars)} bars")
    
    return bars, calendar

