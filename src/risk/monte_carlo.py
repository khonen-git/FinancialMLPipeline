"""Monte Carlo simulation for risk analysis.

Simulates trading outcomes to compute:
- Probability of ruin
- Expected maximum drawdown
- Profit targets under prop firm constraints
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def run_monte_carlo_simulation(
    trade_outcomes: pd.DataFrame,
    config: dict,
    n_simulations: int = 10000
) -> Dict:
    """Run Monte Carlo simulation on trade outcomes.
    
    Args:
        trade_outcomes: DataFrame with 'pnl' column (in account currency)
        config: Risk configuration
        n_simulations: Number of simulation runs
        
    Returns:
        Dictionary with simulation results
    """
    logger.info(f"Running {n_simulations} Monte Carlo simulations")
    
    # Extract parameters
    if 'initial_capital' not in config:
        raise ValueError("Missing required config: initial_capital")
    if 'max_drawdown_pct' not in config:
        raise ValueError("Missing required config: max_drawdown_pct")
    if 'profit_target_pct' not in config:
        raise ValueError("Missing required config: profit_target_pct")
    if 'max_trades' not in config:
        raise ValueError("Missing required config: max_trades")
    
    initial_capital = config['initial_capital']
    max_drawdown_pct = config['max_drawdown_pct']
    profit_target_pct = config['profit_target_pct']
    max_trades_per_sim = config['max_trades']
    
    # Trade PnL distribution
    trade_pnls = trade_outcomes['pnl'].values
    
    if len(trade_pnls) == 0:
        logger.warning("No trades for Monte Carlo simulation")
        return {
            'prob_ruin': 0.0,
            'prob_profit_target': 0.0,
            'mean_final_capital': initial_capital,
            'mean_max_dd': 0.0
        }
    
    # Run simulations
    ruin_count = 0
    profit_target_count = 0
    final_capitals = []
    max_drawdowns = []
    
    for sim_idx in range(n_simulations):
        capital = initial_capital
        peak_capital = initial_capital
        max_dd = 0.0
        
        # Sample trades with replacement
        sampled_pnls = np.random.choice(trade_pnls, size=max_trades_per_sim, replace=True)
        
        for pnl in sampled_pnls:
            capital += pnl
            
            # Update peak and drawdown
            if capital > peak_capital:
                peak_capital = capital
            
            dd = (peak_capital - capital) / peak_capital
            if dd > max_dd:
                max_dd = dd
            
            # Check ruin
            if capital <= initial_capital * (1 - max_drawdown_pct):
                ruin_count += 1
                break
            
            # Check profit target
            if capital >= initial_capital * (1 + profit_target_pct):
                profit_target_count += 1
                break
        
        final_capitals.append(capital)
        max_drawdowns.append(max_dd)
    
    # Compute statistics
    prob_ruin = ruin_count / n_simulations
    prob_profit_target = profit_target_count / n_simulations
    mean_final_capital = np.mean(final_capitals)
    mean_max_dd = np.mean(max_drawdowns)
    
    logger.info(
        f"MC Results: P(ruin)={prob_ruin:.2%}, "
        f"P(profit target)={prob_profit_target:.2%}, "
        f"Mean final capital={mean_final_capital:.2f}"
    )
    
    return {
        'prob_ruin': prob_ruin,
        'prob_profit_target': prob_profit_target,
        'mean_final_capital': mean_final_capital,
        'mean_max_dd': mean_max_dd,
        'final_capitals': final_capitals,
        'max_drawdowns': max_drawdowns
    }


def analyze_prop_firm_constraints(
    mc_results: Dict,
    config: dict
) -> Dict:
    """Analyze Monte Carlo results under prop firm constraints.
    
    Args:
        mc_results: Monte Carlo simulation results
        config: Risk configuration
        
    Returns:
        Dictionary with prop firm analysis
    """
    logger.info("Analyzing prop firm constraints")
    
    if 'initial_capital' not in config:
        raise ValueError("Missing required config: initial_capital")
    if 'max_daily_loss_pct' not in config:
        raise ValueError("Missing required config: max_daily_loss_pct")
    if 'max_total_loss_pct' not in config:
        raise ValueError("Missing required config: max_total_loss_pct")
    
    initial_capital = config['initial_capital']
    max_daily_loss_pct = config['max_daily_loss_pct']
    max_total_loss_pct = config['max_total_loss_pct']
    
    # Simplified analysis (could be extended)
    prob_ruin = mc_results['prob_ruin']
    prob_profit = mc_results['prob_profit_target']
    
    # Risk-reward ratio
    if prob_ruin > 0:
        risk_reward = prob_profit / prob_ruin
    else:
        risk_reward = np.inf
    
    return {
        'prob_ruin': prob_ruin,
        'prob_profit_target': prob_profit,
        'risk_reward_ratio': risk_reward,
        'max_drawdown_limit': max_total_loss_pct,
        'is_viable': prob_ruin < 0.05 and prob_profit > 0.3
    }

