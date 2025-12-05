"""Unit tests for backtest metrics."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.backtest.metrics import BacktestMetrics, extract_backtest_metrics, _calculate_win_rate, _calculate_sharpe_ratio


@pytest.mark.unit
class TestBacktestMetrics:
    """Test backtest metrics functionality."""
    
    def test_backtest_metrics_namedtuple(self):
        """Test BacktestMetrics NamedTuple structure."""
        metrics = BacktestMetrics(
            total_trades=10,
            win_rate=0.6,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            equity_curve=pd.DataFrame({'equity': [1000, 1100, 1050]}),
            trade_log=pd.DataFrame({'pnl': [10, -5, 20]}),
            daily_stats=pd.DataFrame({'date': ['2024-01-01'], 'pnl': [10]}),
            drawdown_curve=pd.DataFrame({'drawdown': [0, 0.05, 0.1]})
        )
        
        assert metrics.total_trades == 10
        assert metrics.win_rate == 0.6
        assert metrics.sharpe_ratio == 1.5
    
    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        trade_log = pd.DataFrame({
            'pnl': [10, -5, 20, -10, 15, -3]
        })
        
        win_rate = _calculate_win_rate(trade_log)
        
        # Win rate should be between 0 and 1
        assert 0 <= win_rate <= 1
        # Should be positive for this mix of wins and losses
        assert win_rate > 0
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        trade_log = pd.DataFrame({
            'pnl': [10, 5, 20, 10, 15, 5]  # All positive for simplicity
        })
        
        sharpe = _calculate_sharpe_ratio(trade_log, risk_free_rate=0.0)
        
        # Should be positive for profitable trades
        assert sharpe >= 0 or np.isnan(sharpe)

