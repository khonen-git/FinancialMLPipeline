"""Benchmarking framework for comparing strategies."""

from src.benchmarks.baselines import (
    BuyAndHold,
    RandomStrategy,
    MovingAverageCrossover,
    RSIStrategy
)
from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.benchmarks.metrics import (
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_profit_factor,
    calculate_expectancy,
    calculate_recovery_factor,
    calculate_ulcer_index,
    calculate_extended_metrics
)
from src.benchmarks.statistical_tests import (
    t_test_returns,
    mann_whitney_test,
    bootstrap_confidence_interval,
    compare_strategies
)

__all__ = [
    'BuyAndHold',
    'RandomStrategy',
    'MovingAverageCrossover',
    'RSIStrategy',
    'BenchmarkRunner',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_profit_factor',
    'calculate_expectancy',
    'calculate_recovery_factor',
    'calculate_ulcer_index',
    'calculate_extended_metrics',
    't_test_returns',
    'mann_whitney_test',
    'bootstrap_confidence_interval',
    'compare_strategies',
]

