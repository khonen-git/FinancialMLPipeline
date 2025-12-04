"""Backtesting runner using Backtrader.

Orchestrates:
- Data feed creation
- Strategy initialization
- Backtrader engine execution
- Metrics extraction
- MLflow logging
"""

import logging
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import mlflow

from src.backtest.data_feed import create_backtrader_feed, PandasDataBidAsk
from src.backtest.backtrader_strategy import SessionAwareStrategy
from src.backtest.metrics import extract_backtest_metrics, BacktestMetrics
from src.labeling.session_calendar import SessionCalendar

logger = logging.getLogger(__name__)


def run_backtest(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    session_calendar: SessionCalendar,
    config: dict,
    labeling_config: dict,
    assets_config: dict
) -> Dict:
    """Run backtest using Backtrader.
    
    Args:
        bars: DataFrame with bid/ask OHLC data (indexed by timestamp)
        predictions: DataFrame with model predictions, must contain:
            - 'prediction': directional signal (+1 for long, -1 for short, 0 for no trade)
            - 'probability': model probability (optional)
            - 'meta_decision': meta-model decision (1=take, 0=skip, optional)
            Index must match bars.index
        session_calendar: SessionCalendar instance for session-aware logic
        config: Backtest configuration from Hydra
        labeling_config: Triple barrier labeling configuration (for TP/SL)
        assets_config: Assets configuration (for tick_size)
        
    Returns:
        Dictionary with backtest results:
            - 'total_trades': int
            - 'win_rate': float
            - 'sharpe_ratio': float
            - 'max_drawdown': float
            - 'total_pnl': float
            - 'equity_curve': pd.DataFrame
            - 'trade_log': pd.DataFrame
            - 'daily_stats': pd.DataFrame
    """
    logger.info("=" * 80)
    logger.info("Starting Backtrader backtest")
    logger.info("=" * 80)
    
    # Validate inputs
    if len(bars) == 0:
        raise ValueError("Empty bars DataFrame")
    if len(predictions) == 0:
        raise ValueError("Empty predictions DataFrame")
    
    # Align predictions with bars
    common_idx = bars.index.intersection(predictions.index)
    if len(common_idx) == 0:
        raise ValueError("No common timestamps between bars and predictions")
    
    logger.info(f"Backtest data: {len(bars)} bars, {len(predictions)} predictions, {len(common_idx)} aligned")
    
    # Merge bars with predictions
    backtest_data = bars.copy()
    for col in ['prediction', 'probability', 'meta_decision']:
        if col in predictions.columns:
            backtest_data[col] = predictions[col]
    
    # Keep only aligned rows
    backtest_data = backtest_data.loc[common_idx].sort_index()
    
    # Create Backtrader data feed
    data_feed = create_backtrader_feed(backtest_data)
    
    # Initialize Cerebro (Backtrader engine)
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    
    # Get TP/SL from labeling config
    if 'tp_ticks' not in labeling_config:
        raise ValueError("Missing required config: labeling.triple_barrier.tp_ticks")
    if 'sl_ticks' not in labeling_config:
        raise ValueError("Missing required config: labeling.triple_barrier.sl_ticks")
    if 'tick_size' not in assets_config:
        raise ValueError("Missing required config: assets.tick_size")
    
    tp_ticks = labeling_config['tp_ticks']
    sl_ticks = labeling_config['sl_ticks']
    tick_size = assets_config['tick_size']
    
    # Strategy parameters
    strategy_params = {
        'session_calendar': session_calendar,
        'tp_ticks': tp_ticks,
        'sl_ticks': sl_ticks,
        'tick_size': tick_size,
        'use_meta_model': config.get('meta_model', {}).get('enabled', False),
        'meta_threshold': config.get('meta_model', {}).get('threshold', 0.5),
        'position_size': config.get('sizing', {}).get('size', 1.0),
    }
    
    # Add strategy
    strategy = cerebro.addstrategy(SessionAwareStrategy, **strategy_params)
    
    # Set signal data on strategy (after strategy is created)
    # We'll need to access the strategy instance after cerebro.run()
    # For now, we'll pass signal data through a custom attribute
    # This is a workaround - Backtrader doesn't easily allow passing data to strategy
    # We'll set it via the strategy's set_signal_data method after creation
    
    # Set initial capital
    initial_cash = config.get('capital', {}).get('initial', 10000.0)
    cerebro.broker.setcash(initial_cash)
    
    # Commission (raw spread account, typically 0)
    commission_value = config.get('commission', {}).get('value', 0.0)
    if commission_value > 0:
        commission_type = config.get('commission', {}).get('type', 'per_lot')
        if commission_type == 'per_lot':
            cerebro.broker.setcommission(commission=commission_value)
        else:
            logger.warning(f"Unknown commission type: {commission_type}, ignoring")
    
    # Slippage (optional)
    slippage_config = config.get('slippage', {})
    if slippage_config.get('enabled', False):
        slippage_ticks = slippage_config.get('fixed_slippage_ticks', 0.0)
        if slippage_ticks > 0:
            slippage_price = slippage_ticks * tick_size
            cerebro.broker.set_slippage_fixed(slippage_price)
            logger.info(f"Slippage enabled: {slippage_ticks} ticks = {slippage_price:.5f}")
    
    # Run backtest
    logger.info(f"Running backtest with initial cash: {initial_cash}")
    logger.info(f"TP: {tp_ticks} ticks, SL: {sl_ticks} ticks")
    
    # Set signal data on strategy before running
    # We need to access the strategy instance, but Backtrader creates it during run()
    # Workaround: store signal data in a way the strategy can access it
    # We'll use a class variable or pass through params
    # Actually, we can set it after strategy creation via a custom method
    # For now, we'll pass predictions as a DataFrame that strategy can look up
    
    # Store predictions in a way strategy can access
    # We'll add it as a custom attribute to the data feed
    if hasattr(data_feed, 'p'):
        # Backtrader stores params in p attribute
        data_feed.p.predictions = backtest_data[['prediction', 'probability', 'meta_decision']].copy()
    
    try:
        results = cerebro.run()
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}", exc_info=True)
        raise
    
    # Extract final value
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash
    
    logger.info(f"Backtest complete. Final value: {final_value:.2f}, Return: {total_return:.2%}")
    
    # Extract metrics from strategy
    strategy_instance = results[0][0]  # Get strategy instance from results
    
    # Set signal data on strategy instance (for future use)
    if hasattr(strategy_instance, 'set_signal_data'):
        signal_df = backtest_data[['prediction', 'probability', 'meta_decision']].copy()
        strategy_instance.set_signal_data(signal_df)
    
    metrics = extract_backtest_metrics(strategy_instance, initial_cash)
    
    # Build results dictionary
    backtest_results = {
        'total_trades': metrics.total_trades,
        'win_rate': metrics.win_rate,
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown': metrics.max_drawdown,
        'total_pnl': final_value - initial_cash,
        'total_return': total_return,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'equity_curve': metrics.equity_curve,
        'trade_log': metrics.trade_log,
        'daily_stats': metrics.daily_stats,
        'drawdown_curve': metrics.drawdown_curve,
    }
    
    logger.info(f"Backtest metrics:")
    logger.info(f"  Total trades: {metrics.total_trades}")
    logger.info(f"  Win rate: {metrics.win_rate:.2%}")
    logger.info(f"  Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    logger.info(f"  Max drawdown: {metrics.max_drawdown:.2%}")
    logger.info(f"  Total PnL: {backtest_results['total_pnl']:.2f}")
    
    return backtest_results

