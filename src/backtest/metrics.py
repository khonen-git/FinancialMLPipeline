"""Backtest metrics extraction and analysis.

Extracts:
- Trade logs
- Equity curve
- Drawdown curve
- Daily statistics
- Performance metrics (Sharpe, win rate, etc.)
"""

import logging
import pandas as pd
import numpy as np
from typing import NamedTuple, Optional
from datetime import datetime
import backtrader as bt

logger = logging.getLogger(__name__)


class BacktestMetrics(NamedTuple):
    """Container for backtest metrics."""
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    daily_stats: pd.DataFrame
    drawdown_curve: pd.DataFrame


def extract_backtest_metrics(
    strategy: bt.Strategy,
    initial_cash: float
) -> BacktestMetrics:
    """Extract metrics from Backtrader strategy.
    
    Args:
        strategy: Backtrader strategy instance after backtest run
        initial_cash: Initial capital
        
    Returns:
        BacktestMetrics with all extracted metrics
    """
    # Extract trade log from strategy
    trade_log = _extract_trade_log(strategy, initial_cash)
    
    # Extract equity curve
    equity_curve = _extract_equity_curve(strategy, initial_cash)
    
    # Calculate drawdown curve
    drawdown_curve = _calculate_drawdown_curve(equity_curve)
    
    # Calculate daily statistics
    daily_stats = _calculate_daily_stats(trade_log, equity_curve)
    
    # Calculate performance metrics
    total_trades = len(trade_log)
    win_rate = _calculate_win_rate(trade_log)
    sharpe_ratio = _calculate_sharpe_ratio(trade_log)
    max_drawdown = _calculate_max_drawdown(drawdown_curve)
    
    return BacktestMetrics(
        total_trades=total_trades,
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        equity_curve=equity_curve,
        trade_log=trade_log,
        daily_stats=daily_stats,
        drawdown_curve=drawdown_curve
    )


def _extract_trade_log(strategy: bt.Strategy, initial_cash: float) -> pd.DataFrame:
    """Extract trade log from strategy.
    
    Args:
        strategy: Backtrader strategy instance
        initial_cash: Initial capital
        
    Returns:
        DataFrame with trade log
    """
    trades = []
    
    # Access strategy's trade analyzer if available
    if hasattr(strategy, 'analyzers') and 'trades' in strategy.analyzers:
        analyzer = strategy.analyzers.trades
        for trade in analyzer.trades:
            trades.append({
                'entry_timestamp': trade.data.datetime.datetime(0) if hasattr(trade.data, 'datetime') else None,
                'exit_timestamp': trade.data.datetime.datetime(0) if hasattr(trade.data, 'datetime') else None,
                'entry_price': trade.price,
                'exit_price': trade.price_close,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / initial_cash if initial_cash > 0 else 0,
                'duration': trade.barlen if hasattr(trade, 'barlen') else None,
                'size': trade.size,
            })
    else:
        # Fallback: extract from strategy's trades list if available
        if hasattr(strategy, 'trades') and len(strategy.trades) > 0:
            for trade in strategy.trades:
                if trade.isclosed:
                    trades.append({
                        'entry_timestamp': trade.data.datetime.datetime(0) if hasattr(trade.data, 'datetime') else None,
                        'exit_timestamp': trade.data.datetime.datetime(0) if hasattr(trade.data, 'datetime') else None,
                        'entry_price': trade.price,
                        'exit_price': trade.price_close,
                        'pnl': trade.pnl,
                        'pnl_pct': trade.pnlcomm / initial_cash if initial_cash > 0 else 0,
                        'duration': trade.barlen if hasattr(trade, 'barlen') else None,
                        'size': trade.size,
                    })
    
    if len(trades) == 0:
        logger.warning("No trades found in strategy, returning empty trade log")
        return pd.DataFrame(columns=[
            'entry_timestamp', 'exit_timestamp', 'entry_price', 'exit_price',
            'pnl', 'pnl_pct', 'duration', 'size', 'reason'
        ])
    
    trade_log = pd.DataFrame(trades)
    
    # Add reason column if available
    if 'reason' not in trade_log.columns:
        trade_log['reason'] = 'unknown'
    
    return trade_log


def _extract_equity_curve(strategy: bt.Strategy, initial_cash: float) -> pd.DataFrame:
    """Extract equity curve from strategy.
    
    Args:
        strategy: Backtrader strategy instance
        initial_cash: Initial capital
        
    Returns:
        DataFrame with timestamp and equity columns
    """
    # Try to get equity curve from analyzer
    if hasattr(strategy, 'analyzers') and 'drawdown' in strategy.analyzers:
        analyzer = strategy.analyzers.drawdown
        # Drawdown analyzer doesn't directly provide equity curve
        # We'll need to reconstruct from broker value
        pass
    
    # Fallback: reconstruct from broker value history if available
    # For now, create a simple equity curve from trade log
    equity_curve = pd.DataFrame({
        'timestamp': pd.date_range(
            start=strategy.data.datetime.datetime(0) if hasattr(strategy.data, 'datetime') else datetime.now(),
            periods=1,
            freq='1min'
        ),
        'equity': [initial_cash]
    })
    
    # If we have trade log, we can reconstruct equity curve
    # For now, return minimal equity curve
    logger.warning("Equity curve extraction is simplified - full implementation requires broker value tracking")
    
    return equity_curve


def _calculate_drawdown_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Calculate drawdown curve from equity curve.
    
    Args:
        equity_curve: DataFrame with timestamp and equity columns
        
    Returns:
        DataFrame with timestamp, equity, peak, and drawdown columns
    """
    if len(equity_curve) == 0:
        return pd.DataFrame(columns=['timestamp', 'equity', 'peak', 'drawdown', 'drawdown_pct'])
    
    equity = equity_curve['equity'].values
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    drawdown_pct = drawdown / peak
    
    return pd.DataFrame({
        'timestamp': equity_curve['timestamp'],
        'equity': equity,
        'peak': peak,
        'drawdown': drawdown,
        'drawdown_pct': drawdown_pct
    })


def _calculate_daily_stats(
    trade_log: pd.DataFrame,
    equity_curve: pd.DataFrame
) -> pd.DataFrame:
    """Calculate daily statistics.
    
    Args:
        trade_log: Trade log DataFrame
        equity_curve: Equity curve DataFrame
        
    Returns:
        DataFrame with daily statistics
    """
    if len(trade_log) == 0:
        return pd.DataFrame(columns=['date', 'pnl', 'trades', 'win_rate', 'max_dd'])
    
    # Group trades by date
    if 'exit_timestamp' in trade_log.columns:
        trade_log['date'] = pd.to_datetime(trade_log['exit_timestamp']).dt.date
    elif 'entry_timestamp' in trade_log.columns:
        trade_log['date'] = pd.to_datetime(trade_log['entry_timestamp']).dt.date
    else:
        logger.warning("Cannot calculate daily stats: no timestamp columns")
        return pd.DataFrame(columns=['date', 'pnl', 'trades', 'win_rate', 'max_dd'])
    
    daily = trade_log.groupby('date').agg({
        'pnl': 'sum',
        'pnl_pct': lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0,  # win rate
    }).reset_index()
    
    daily.columns = ['date', 'pnl', 'win_rate']
    daily['trades'] = trade_log.groupby('date').size().values
    
    # Add max drawdown per day (simplified)
    daily['max_dd'] = 0.0  # Would need equity curve per day
    
    return daily


def _calculate_win_rate(trade_log: pd.DataFrame) -> float:
    """Calculate win rate.
    
    Args:
        trade_log: Trade log DataFrame
        
    Returns:
        Win rate (0-1)
    """
    if len(trade_log) == 0:
        return 0.0
    
    if 'pnl' in trade_log.columns:
        wins = (trade_log['pnl'] > 0).sum()
        return wins / len(trade_log)
    
    return 0.0


def _calculate_sharpe_ratio(trade_log: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from trade returns.
    
    Args:
        trade_log: Trade log DataFrame
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if len(trade_log) == 0:
        return 0.0
    
    if 'pnl_pct' in trade_log.columns:
        returns = trade_log['pnl_pct'].values
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualize if we have enough data
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Simple Sharpe (assuming daily returns, annualize by sqrt(252))
        if std_return > 0:
            sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
            return sharpe
    
    return 0.0


def _calculate_max_drawdown(drawdown_curve: pd.DataFrame) -> float:
    """Calculate maximum drawdown.
    
    Args:
        drawdown_curve: Drawdown curve DataFrame
        
    Returns:
        Maximum drawdown (0-1, negative value)
    """
    if len(drawdown_curve) == 0:
        return 0.0
    
    if 'drawdown_pct' in drawdown_curve.columns:
        return drawdown_curve['drawdown_pct'].min()
    
    return 0.0

