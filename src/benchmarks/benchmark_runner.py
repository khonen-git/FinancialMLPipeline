"""Benchmark runner for comparing strategies.

Orchestrates running baseline strategies and comparing them to ML models.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import mlflow

from src.backtest.runner import run_backtest
from src.benchmarks.metrics import calculate_extended_metrics
from src.benchmarks.statistical_tests import compare_strategies
from src.labeling.session_calendar import SessionCalendar

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run and compare multiple strategies."""
    
    def __init__(
        self,
        bars: pd.DataFrame,
        session_calendar: SessionCalendar,
        config: dict,
        labeling_config: dict,
        assets_config: dict
    ):
        """Initialize benchmark runner.
        
        Args:
            bars: DataFrame with OHLCV bars
            session_calendar: SessionCalendar instance
            config: Full pipeline configuration
            labeling_config: Triple barrier labeling configuration
            assets_config: Assets configuration
        """
        self.bars = bars
        self.session_calendar = session_calendar
        self.config = config
        self.labeling_config = labeling_config
        self.assets_config = assets_config
    
    def run_benchmark(
        self,
        strategy,
        name: str,
        use_extended_metrics: bool = True
    ) -> Dict:
        """Run a single benchmark strategy.
        
        Args:
            strategy: Strategy object with generate_signals() method
            name: Strategy name
            use_extended_metrics: Whether to calculate extended metrics
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running benchmark: {name}")
        
        # Generate signals
        signals = strategy.generate_signals(self.bars)
        
        # Run backtest
        results = run_backtest(
            bars=self.bars,
            predictions=signals,
            session_calendar=self.session_calendar,
            config=self.config,
            labeling_config=self.labeling_config,
            assets_config=self.assets_config
        )
        
        # Add extended metrics if requested
        if use_extended_metrics:
            trade_log = results.get('trade_log', pd.DataFrame())
            equity_curve = results.get('equity_curve', pd.Series())
            
            if len(trade_log) > 0 and len(equity_curve) > 0:
                extended = calculate_extended_metrics(
                    trade_log=trade_log,
                    equity_curve=equity_curve['equity'] if isinstance(equity_curve, pd.DataFrame) else equity_curve,
                    risk_free_rate=self.config.get('risk', {}).get('risk_free_rate', 0.0),
                    periods_per_year=252
                )
                results.update(extended)
        
        results['strategy_name'] = name
        return results
    
    def compare_strategies(
        self,
        strategies: Dict[str, object],
        use_extended_metrics: bool = True
    ) -> pd.DataFrame:
        """Compare multiple strategies.
        
        Args:
            strategies: Dictionary mapping strategy names to strategy objects
            use_extended_metrics: Whether to calculate extended metrics
        
        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing {len(strategies)} strategies")
        
        all_results = []
        
        for name, strategy in strategies.items():
            try:
                results = self.run_benchmark(strategy, name, use_extended_metrics)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error running benchmark {name}: {e}")
                continue
        
        if len(all_results) == 0:
            logger.warning("No successful benchmark runs")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = {
            'strategy': [r.get('strategy_name', 'unknown') for r in all_results],
            'sharpe_ratio': [r.get('sharpe_ratio', 0.0) for r in all_results],
            'total_return': [r.get('total_return', 0.0) for r in all_results],
            'max_drawdown': [r.get('max_drawdown', 0.0) for r in all_results],
            'win_rate': [r.get('win_rate', 0.0) for r in all_results],
            'total_trades': [r.get('total_trades', 0) for r in all_results],
        }
        
        # Add extended metrics if available
        if use_extended_metrics:
            extended_cols = ['sortino_ratio', 'calmar_ratio', 'profit_factor', 
                           'expectancy', 'recovery_factor', 'ulcer_index']
            for col in extended_cols:
                comparison_data[col] = [r.get(col, np.nan) for r in all_results]
        
        comparison = pd.DataFrame(comparison_data)
        
        return comparison
    
    def compare_model_to_baselines(
        self,
        model_results: Dict,
        baseline_strategies: Dict[str, object],
        use_statistical_tests: bool = True,
        significance_level: float = 0.05
    ) -> Dict:
        """Compare ML model results to baseline strategies.
        
        Args:
            model_results: Results from ML model backtest
            baseline_strategies: Dictionary of baseline strategies
            use_statistical_tests: Whether to perform statistical tests
            significance_level: Significance level for tests
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing model to baseline strategies")
        
        # Run baseline strategies
        baseline_results = {}
        for name, strategy in baseline_strategies.items():
            try:
                results = self.run_benchmark(strategy, name)
                baseline_results[name] = results
            except Exception as e:
                logger.error(f"Error running baseline {name}: {e}")
                continue
        
        # Extract returns for statistical tests
        comparison_summary = {
            'model_results': model_results,
            'baseline_results': baseline_results,
            'statistical_tests': {}
        }
        
        if use_statistical_tests and 'trade_log' in model_results:
            model_trade_log = model_results['trade_log']
            if 'pnl' in model_trade_log.columns:
                model_returns = model_trade_log['pnl'].values
                
                for baseline_name, baseline_result in baseline_results.items():
                    if 'trade_log' in baseline_result and 'pnl' in baseline_result['trade_log'].columns:
                        baseline_returns = baseline_result['trade_log']['pnl'].values
                        
                        # Perform statistical tests
                        test_results = compare_strategies(
                            model_returns=model_returns,
                            baseline_returns=baseline_returns,
                            significance_level=significance_level
                        )
                        comparison_summary['statistical_tests'][baseline_name] = test_results
        
        return comparison_summary

