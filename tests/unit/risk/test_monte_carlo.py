"""Unit tests for Monte Carlo risk analysis."""

import pytest
import pandas as pd
import numpy as np
from src.risk.monte_carlo import run_monte_carlo_simulation, analyze_prop_firm_constraints


@pytest.mark.unit
class TestMonteCarlo:
    """Test Monte Carlo simulation."""
    
    def test_monte_carlo_simulation_basic(self):
        """Test basic Monte Carlo simulation."""
        trade_outcomes = pd.DataFrame({
            'pnl': [10, -5, 20, -10, 15, -3, 8, -2]
        })
        
        config = {
            'initial_capital': 10000.0,
            'max_drawdown_pct': 0.1,
            'profit_target_pct': 0.05,
            'max_trades': 100
        }
        
        results = run_monte_carlo_simulation(trade_outcomes, config, n_simulations=100)
        
        assert 'prob_ruin' in results
        assert 'prob_profit_target' in results
        assert 'mean_final_capital' in results
        assert 0 <= results['prob_ruin'] <= 1
        assert 0 <= results['prob_profit_target'] <= 1
    
    def test_monte_carlo_missing_config(self):
        """Test error when config is missing required keys."""
        trade_outcomes = pd.DataFrame({'pnl': [10, -5]})
        config = {
            'initial_capital': 10000.0
            # Missing other required keys
        }
        
        with pytest.raises(ValueError, match="Missing required config"):
            run_monte_carlo_simulation(trade_outcomes, config)
    
    def test_monte_carlo_empty_trades(self):
        """Test Monte Carlo with empty trade outcomes."""
        trade_outcomes = pd.DataFrame({'pnl': []})
        
        config = {
            'initial_capital': 10000.0,
            'max_drawdown_pct': 0.1,
            'profit_target_pct': 0.05,
            'max_trades': 100
        }
        
        results = run_monte_carlo_simulation(trade_outcomes, config, n_simulations=10)
        
        assert results['prob_ruin'] == 0.0
        assert results['mean_final_capital'] == config['initial_capital']

