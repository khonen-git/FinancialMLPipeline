"""Custom Backtrader data feed for bid/ask bars."""

import logging
import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)


class PandasDataBidAsk(bt.feeds.PandasData):
    """Custom data feed with bid/ask OHLC.
    
    Expected columns:
    - bid_open, bid_high, bid_low, bid_close
    - ask_open, ask_high, ask_low, ask_close
    - volume (optional)
    """
    
    lines = ('ask',)  # Add 'ask' line for ask_close
    
    params = (
        ('datetime', None),
        ('open', 'bid_open'),
        ('high', 'bid_high'),
        ('low', 'bid_low'),
        ('close', 'bid_close'),
        ('volume', 'volume'),
        ('openinterest', None),
        ('ask', 'ask_close'),  # Map ask line to ask_close column
    )


def create_backtrader_feed(bars: pd.DataFrame) -> PandasDataBidAsk:
    """Create Backtrader data feed from bars DataFrame.
    
    Args:
        bars: DataFrame with bid/ask OHLC
        
    Returns:
        PandasDataBidAsk feed
    """
    logger.info(f"Creating Backtrader feed with {len(bars)} bars")
    
    # Ensure required columns
    required = ['bid_open', 'bid_high', 'bid_low', 'bid_close', 'ask_close']
    missing = [col for col in required if col not in bars.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add volume if not present
    if 'volume' not in bars.columns:
        bars['volume'] = 0
    
    # Create feed
    data = PandasDataBidAsk(dataname=bars)
    
    return data

