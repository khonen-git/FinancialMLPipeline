"""Unit tests for backtest data feed."""

import pytest
import pandas as pd
import numpy as np
from src.backtest.data_feed import PandasDataBidAsk, create_backtrader_feed
from tests.conftest import sample_bars


@pytest.mark.unit
class TestBacktestDataFeed:
    """Test backtest data feed functionality."""
    
    def test_create_backtrader_feed(self, sample_bars):
        """Test creating Backtrader feed from bars."""
        feed = create_backtrader_feed(sample_bars)
        
        assert feed is not None
        assert isinstance(feed, PandasDataBidAsk)
    
    def test_pandas_data_bid_ask_init(self, sample_bars):
        """Test PandasDataBidAsk initialization."""
        feed = PandasDataBidAsk(dataname=sample_bars)
        
        assert feed is not None
        assert feed.p.dataname is not None

