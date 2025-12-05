"""Extended performance metrics for benchmarking.

Calculates additional metrics beyond basic Sharpe ratio:
- Sortino Ratio (Sharpe with downside deviation)
- Calmar Ratio (return / max drawdown)
- Profit Factor (gross profit / gross loss)
- Expectancy (average profit per trade)
- Recovery Factor (net profit / max drawdown)
- Ulcer Index (drawdown-based risk metric)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio (Sharpe with downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Downside deviation: only negative returns
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        # No downside, return high value
        return np.inf if excess_returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    
    # Annualize
    sortino = (excess_returns.mean() * periods_per_year) / (downside_std * np.sqrt(periods_per_year))
    
    return sortino


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    n_periods: int = 252
) -> float:
    """Calculate Calmar ratio (return / max drawdown).
    
    Args:
        total_return: Total return (not annualized)
        max_drawdown: Maximum drawdown (absolute value, e.g., 0.1 for 10%)
        n_periods: Number of periods for annualization
        
    Returns:
        Calmar ratio (annualized)
    """
    if max_drawdown == 0:
        return np.inf if total_return > 0 else 0.0
    
    # Annualize return
    if n_periods > 0:
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
    else:
        annualized_return = total_return
    
    calmar = annualized_return / abs(max_drawdown)
    
    return calmar


def calculate_profit_factor(trade_log: pd.DataFrame) -> float:
    """Calculate profit factor (gross profit / gross loss).
    
    Args:
        trade_log: DataFrame with 'pnl' column
        
    Returns:
        Profit factor (inf if no losses, 0 if no profits)
    """
    if len(trade_log) == 0 or 'pnl' not in trade_log.columns:
        return 0.0
    
    gross_profit = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(trade_log: pd.DataFrame) -> float:
    """Calculate expectancy (average profit per trade).
    
    Args:
        trade_log: DataFrame with 'pnl' column
        
    Returns:
        Expectancy (average PnL per trade)
    """
    if len(trade_log) == 0 or 'pnl' not in trade_log.columns:
        return 0.0
    
    return trade_log['pnl'].mean()


def calculate_recovery_factor(
    net_profit: float,
    max_drawdown: float
) -> float:
    """Calculate recovery factor (net profit / max drawdown).
    
    Args:
        net_profit: Net profit (total PnL)
        max_drawdown: Maximum drawdown (absolute value)
        
    Returns:
        Recovery factor
    """
    if max_drawdown == 0:
        return np.inf if net_profit > 0 else 0.0
    
    return net_profit / abs(max_drawdown)


def calculate_ulcer_index(equity_curve: pd.Series) -> float:
    """Calculate Ulcer Index (drawdown-based risk metric).
    
    The Ulcer Index measures the depth and duration of drawdowns.
    Lower is better.
    
    Args:
        equity_curve: Series of equity values (cumulative)
        
    Returns:
        Ulcer Index
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown percentage
    drawdown_pct = ((equity_curve - running_max) / running_max) * 100
    
    # Square the drawdowns and take mean, then square root
    ulcer = np.sqrt((drawdown_pct ** 2).mean())
    
    return ulcer


def calculate_extended_metrics(
    trade_log: pd.DataFrame,
    equity_curve: pd.Series,
    returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """Calculate all extended metrics.
    
    Args:
        trade_log: DataFrame with 'pnl' column
        equity_curve: Series of equity values
        returns: Optional series of returns (if None, calculated from equity_curve)
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary with all extended metrics
    """
    metrics = {}
    
    # Calculate returns if not provided
    if returns is None and len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
    elif returns is None:
        returns = pd.Series(dtype=float)
    
    # Basic metrics
    if len(trade_log) > 0 and 'pnl' in trade_log.columns:
        net_profit = trade_log['pnl'].sum()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0
    else:
        net_profit = 0.0
        total_return = 0.0
    
    # Calculate max drawdown
    if len(equity_curve) > 0:
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0
    
    # Extended metrics
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    n_periods = len(returns) if len(returns) > 0 else 252
    metrics['calmar_ratio'] = calculate_calmar_ratio(total_return, max_drawdown, n_periods)
    metrics['profit_factor'] = calculate_profit_factor(trade_log)
    metrics['expectancy'] = calculate_expectancy(trade_log)
    metrics['recovery_factor'] = calculate_recovery_factor(net_profit, max_drawdown)
    metrics['ulcer_index'] = calculate_ulcer_index(equity_curve)
    
    return metrics

