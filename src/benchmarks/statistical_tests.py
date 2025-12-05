"""Statistical significance tests for benchmarking.

Compares model performance to baseline strategies using:
- T-test for returns
- Mann-Whitney U test (non-parametric)
- Bootstrap confidence intervals
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats

logger = logging.getLogger(__name__)


def t_test_returns(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """Perform t-test to compare returns.
    
    Args:
        model_returns: Array of model returns
        baseline_returns: Array of baseline returns
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Dictionary with test results
    """
    if len(model_returns) == 0 or len(baseline_returns) == 0:
        return {
            't_statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'model_mean': 0.0,
            'baseline_mean': 0.0,
            'difference': 0.0
        }
    
    # Perform t-test
    if alternative == 'two-sided':
        t_stat, p_value = stats.ttest_ind(model_returns, baseline_returns)
    elif alternative == 'greater':
        t_stat, p_value = stats.ttest_ind(model_returns, baseline_returns, alternative='greater')
    elif alternative == 'less':
        t_stat, p_value = stats.ttest_ind(model_returns, baseline_returns, alternative='less')
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
    
    model_mean = np.mean(model_returns)
    baseline_mean = np.mean(baseline_returns)
    difference = model_mean - baseline_mean
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'model_mean': model_mean,
        'baseline_mean': baseline_mean,
        'difference': difference,
        'alternative': alternative
    }


def mann_whitney_test(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """Perform Mann-Whitney U test (non-parametric).
    
    Args:
        model_returns: Array of model returns
        baseline_returns: Array of baseline returns
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Dictionary with test results
    """
    if len(model_returns) == 0 or len(baseline_returns) == 0:
        return {
            'u_statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'model_median': 0.0,
            'baseline_median': 0.0
        }
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(
        model_returns,
        baseline_returns,
        alternative=alternative
    )
    
    model_median = np.median(model_returns)
    baseline_median = np.median(baseline_returns)
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'model_median': model_median,
        'baseline_median': baseline_median,
        'alternative': alternative
    }


def bootstrap_confidence_interval(
    returns: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate bootstrap confidence interval.
    
    Args:
        returns: Array of returns
        statistic_func: Function to calculate statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Dictionary with confidence interval bounds
    """
    if len(returns) == 0:
        return {
            'lower': np.nan,
            'upper': np.nan,
            'mean': 0.0,
            'confidence_level': confidence_level
        }
    
    # Bootstrap sampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    mean_stat = statistic_func(returns)
    
    return {
        'lower': lower,
        'upper': upper,
        'mean': mean_stat,
        'confidence_level': confidence_level
    }


def compare_strategies(
    model_returns: np.ndarray,
    baseline_returns: np.ndarray,
    significance_level: float = 0.05,
    use_bootstrap: bool = True
) -> Dict[str, any]:
    """Compare model vs baseline using multiple statistical tests.
    
    Args:
        model_returns: Array of model returns
        baseline_returns: Array of baseline returns
        significance_level: Significance level (default: 0.05)
        use_bootstrap: Whether to include bootstrap CI
    
    Returns:
        Dictionary with all test results
    """
    results = {}
    
    # T-test
    t_test_results = t_test_returns(model_returns, baseline_returns, alternative='two-sided')
    results['t_test'] = t_test_results
    results['t_test_significant'] = t_test_results['p_value'] < significance_level
    
    # Mann-Whitney U test
    mw_test_results = mann_whitney_test(model_returns, baseline_returns, alternative='two-sided')
    results['mann_whitney'] = mw_test_results
    results['mann_whitney_significant'] = mw_test_results['p_value'] < significance_level
    
    # Bootstrap confidence intervals
    if use_bootstrap:
        model_ci = bootstrap_confidence_interval(model_returns)
        baseline_ci = bootstrap_confidence_interval(baseline_returns)
        results['model_bootstrap_ci'] = model_ci
        results['baseline_bootstrap_ci'] = baseline_ci
        
        # Check if intervals overlap
        results['ci_overlap'] = not (
            model_ci['upper'] < baseline_ci['lower'] or
            baseline_ci['upper'] < model_ci['lower']
        )
    
    # Summary
    results['model_better'] = (
        t_test_results['difference'] > 0 and
        results['t_test_significant']
    )
    
    return results

