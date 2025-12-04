"""MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) computation.

MFE = Maximum profit during the trade horizon
MAE = Maximum loss during the trade horizon

⚠️ WARNING: MFE/MAE uses FUTURE data (looks ahead in the horizon).
DO NOT use MFE/MAE values as features for ML models - this causes data leakage!

These functions should ONLY be used to:
- Select optimal TP/SL based on quantile distribution (historical analysis)

The quantiles computed from MFE/MAE can be used to set TP/SL parameters,
but the individual MFE/MAE values should NEVER be used as model features.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation, fallback to numpy if not available
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available, using numpy vectorized operations instead")


def _compute_mfe_mae_vectorized(
    entry_price: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    mfe_values: np.ndarray,
    mae_values: np.ndarray,
    n: int,
    horizon_bars: int
):
    """Optimized MFE/MAE computation using numpy vectorization.
    
    This function is called by compute_mfe_mae() and should not be called directly.
    Uses numba JIT if available, otherwise falls back to numpy vectorized operations.
    """
    # Vectorized approach: use numpy sliding window view
    # This is much faster than Python loops
    for i in range(n - horizon_bars):
        start_price = entry_price[i]
        start_idx = i + 1
        end_idx = min(start_idx + horizon_bars, n)
        
        if end_idx > start_idx:
            # Use numpy for max/min (much faster than Python loops)
            mfe = np.max(future_high[start_idx:end_idx]) - start_price
            mae = start_price - np.min(future_low[start_idx:end_idx])
            
            mfe_values[i] = mfe
            mae_values[i] = mae


# If numba is available, create a JIT-compiled version
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _compute_mfe_mae_numba(
        entry_price: np.ndarray,
        future_high: np.ndarray,
        future_low: np.ndarray,
        mfe_values: np.ndarray,
        mae_values: np.ndarray,
        n: int,
        horizon_bars: int
    ):
        """Numba-optimized MFE/MAE computation (faster than numpy for large datasets)."""
        for i in range(n - horizon_bars):
            start_price = entry_price[i]
            start_idx = i + 1
            end_idx = min(start_idx + horizon_bars, n)
            
            if end_idx > start_idx:
                # MFE: maximum profit (highest bid - ask entry)
                max_high = future_high[start_idx]
                for j in range(start_idx + 1, end_idx):
                    if future_high[j] > max_high:
                        max_high = future_high[j]
                mfe = max_high - start_price
                
                # MAE: maximum loss (ask entry - lowest bid)
                min_low = future_low[start_idx]
                for j in range(start_idx + 1, end_idx):
                    if future_low[j] < min_low:
                        min_low = future_low[j]
                mae = start_price - min_low
                
                mfe_values[i] = mfe
                mae_values[i] = mae
    
    # Override the function with numba version
    _compute_mfe_mae_vectorized = _compute_mfe_mae_numba


def compute_mfe_mae(
    bars: pd.DataFrame,
    horizon_bars: int = 32,
    quantile: float = 0.5
) -> pd.DataFrame:
    """Compute MFE and MAE for each bar over a fixed horizon.
    
    ⚠️ WARNING: This function uses FUTURE data (looks ahead).
    DO NOT use the returned DataFrame as features for ML models!
    This function should ONLY be used to compute quantiles for TP/SL selection.
    
    MFE (Maximum Favorable Excursion): Maximum profit during horizon
    MAE (Maximum Adverse Excursion): Maximum loss during horizon
    
    Args:
        bars: DataFrame with bid/ask OHLC data
        horizon_bars: Fixed horizon to compute excursions
        quantile: Quantile to extract for TP/SL suggestion (0.5 = median)
        
    Returns:
        DataFrame with MFE/MAE values (for quantile analysis only, NOT for features)
    """
    features = pd.DataFrame(index=bars.index)
    
    # Use ask for entry (long only), bid for exit
    if 'ask_close' in bars.columns and 'bid_high' in bars.columns and 'bid_low' in bars.columns:
        entry_price = bars['ask_close'].values
        future_high = bars['bid_high'].values
        future_low = bars['bid_low'].values
    else:
        # Fallback to close/high/low if bid/ask not available
        entry_price = bars['close'].values
        future_high = bars['high'].values
        future_low = bars['low'].values
    
    n = len(bars)
    mfe_values = np.full(n, np.nan)
    mae_values = np.full(n, np.nan)
    
    # Convert to numpy arrays (already arrays from .values, but ensure they're contiguous)
    entry_price_arr = np.ascontiguousarray(entry_price, dtype=np.float64)
    future_high_arr = np.ascontiguousarray(future_high, dtype=np.float64)
    future_low_arr = np.ascontiguousarray(future_low, dtype=np.float64)
    
    # Compute MFE/MAE using optimized function (numba if available, numpy otherwise)
    _compute_mfe_mae_vectorized(
        entry_price_arr, future_high_arr, future_low_arr,
        mfe_values, mae_values, n, horizon_bars
    )
    
    features['mfe'] = mfe_values
    features['mae'] = mae_values
    
    # Note: We do NOT create rolling statistics or ratios here
    # because these would be used as features, which causes data leakage.
    # This function is ONLY for computing quantiles to suggest TP/SL.
    
    logger.info(f"Computed MFE/MAE for {len(features)} bars (for TP/SL quantile analysis only)")
    
    # Log quantile statistics (useful for TP/SL selection)
    valid_mfe = features['mfe'].dropna()
    valid_mae = features['mae'].dropna()
    
    if len(valid_mfe) > 0:
        mfe_quantile = valid_mfe.quantile(quantile)
        mae_quantile = valid_mae.quantile(quantile)
        
        # For EURUSD: 1 pip = 0.0001, 1 tick = 0.00001
        # So to convert to ticks, divide by 0.00001 (or divide by 0.0001 and multiply by 10)
        tick_size_actual = 0.00001  # Fractional pip for EURUSD
        
        logger.info(f"MFE quantile {quantile:.1f}: {mfe_quantile:.5f} ({mfe_quantile/tick_size_actual:.1f} ticks, {mfe_quantile/0.0001:.1f} pips)")
        logger.info(f"MAE quantile {quantile:.1f}: {mae_quantile:.5f} ({mae_quantile/tick_size_actual:.1f} ticks, {mae_quantile/0.0001:.1f} pips)")
        logger.info(f"Suggested TP (MFE q{quantile}): {mfe_quantile/tick_size_actual:.0f} ticks = {mfe_quantile/0.0001:.1f} pips")
        logger.info(f"Suggested SL (MAE q{quantile}): {mae_quantile/tick_size_actual:.0f} ticks = {mae_quantile/0.0001:.1f} pips")
    
    return features


def suggest_tp_sl_from_mfe_mae(
    bars: pd.DataFrame,
    horizon_bars: int = 32,
    quantile: float = 0.5,
    tick_size: float = 0.0001
) -> dict:
    """Suggest TP/SL based on MFE/MAE quantile distribution.
    
    This implements the methodology from your previous project:
    - Compute MFE/MAE over fixed horizon
    - Select quantile (e.g. 0.5 = median)
    - Use as TP/SL distances
    
    Args:
        bars: DataFrame with OHLC data
        horizon_bars: Fixed horizon for excursion computation
        quantile: Quantile to extract (0.5 = median, 0.75 = 75th percentile)
        tick_size: Tick size for the instrument
        
    Returns:
        Dictionary with suggested TP/SL in ticks and price units
    """
    features = compute_mfe_mae(bars, horizon_bars=horizon_bars, quantile=quantile)
    
    valid_mfe = features['mfe'].dropna()
    valid_mae = features['mae'].dropna()
    
    if len(valid_mfe) == 0 or len(valid_mae) == 0:
        logger.warning("Insufficient data for MFE/MAE analysis")
        return {
            'tp_ticks': 50,
            'sl_ticks': 50,
            'tp_price': 50 * tick_size,
            'sl_price': 50 * tick_size
        }
    
    mfe_quantile = valid_mfe.quantile(quantile)
    mae_quantile = valid_mae.quantile(quantile)
    
    # CRITICAL: tick_size parameter might be pip_size (0.0001), not true tick size (0.00001)
    # For EURUSD: 1 pip = 10 ticks
    # To be safe, calculate in both units
    tp_price_units = mfe_quantile
    sl_price_units = mae_quantile
    
    # If tick_size is actually pip_size (0.0001), multiply by 10 to get ticks
    if tick_size >= 0.0001:
        # This is pip_size, not tick_size
        tp_ticks = int(mfe_quantile / (tick_size / 10))
        sl_ticks = int(mae_quantile / (tick_size / 10))
    else:
        # This is true tick_size (0.00001)
        tp_ticks = int(mfe_quantile / tick_size)
        sl_ticks = int(mae_quantile / tick_size)
    
    # Ensure minimum values
    tp_ticks = max(tp_ticks, 10)
    sl_ticks = max(sl_ticks, 10)
    
    logger.info(f"MFE/MAE analysis complete:")
    logger.info(f"  Horizon: {horizon_bars} bars")
    logger.info(f"  Quantile: {quantile}")
    logger.info(f"  MFE q{quantile}: {mfe_quantile:.5f} → TP: {tp_ticks} ticks ({tp_ticks/10:.1f} pips)")
    logger.info(f"  MAE q{quantile}: {mae_quantile:.5f} → SL: {sl_ticks} ticks ({sl_ticks/10:.1f} pips)")
    logger.info(f"  Risk/Reward ratio: {tp_ticks/sl_ticks:.2f}")
    
    return {
        'tp_ticks': tp_ticks,
        'sl_ticks': sl_ticks,
        'tp_price': tp_ticks * tick_size,
        'sl_price': sl_ticks * tick_size,
        'mfe_quantile': mfe_quantile,
        'mae_quantile': mae_quantile,
        'horizon_bars': horizon_bars
    }

