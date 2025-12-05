"""Baseline trading strategies for benchmarking.

These strategies serve as reference points to evaluate whether ML models
provide genuine value over simple approaches.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class BuyAndHold:
    """Buy-and-hold baseline strategy.
    
    The simplest strategy: buy at the start, hold until the end.
    Serves as a baseline for directional strategies.
    """
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate buy-and-hold signals.
        
        Args:
            bars: DataFrame with OHLCV bars
            
        Returns:
            DataFrame with 'prediction', 'probability', and 'meta_decision' columns
        """
        signals = pd.DataFrame(index=bars.index)
        signals['prediction'] = 1  # Always long
        signals['probability'] = 1.0
        signals['meta_decision'] = 1
        return signals


class RandomStrategy:
    """Random baseline strategy.
    
    Generates random entry signals (50/50 long/short or long-only).
    Serves as a null hypothesis test (model should beat random).
    """
    
    def __init__(self, seed: int = 42, long_only: bool = True):
        """Initialize random strategy.
        
        Args:
            seed: Random seed for reproducibility
            long_only: If True, only generate long signals (0 or 1).
                      If False, generate long/short signals (-1, 0, 1).
        """
        self.rng = np.random.default_rng(seed)
        self.long_only = long_only
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate random signals.
        
        Args:
            bars: DataFrame with OHLCV bars
            
        Returns:
            DataFrame with 'prediction', 'probability', and 'meta_decision' columns
        """
        signals = pd.DataFrame(index=bars.index)
        
        if self.long_only:
            # Random long signals (50% probability)
            signals['prediction'] = self.rng.choice([0, 1], size=len(bars), p=[0.5, 0.5])
        else:
            # Random long/short signals
            signals['prediction'] = self.rng.choice([-1, 0, 1], size=len(bars), p=[1/3, 1/3, 1/3])
        
        signals['probability'] = 0.5
        signals['meta_decision'] = signals['prediction']
        return signals


class MovingAverageCrossover:
    """Moving average crossover baseline strategy.
    
    Simple technical indicator: buy when short MA crosses above long MA.
    Serves as a baseline for trend-following strategies.
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 50):
        """Initialize MA crossover strategy.
        
        Args:
            short_period: Short moving average period
            long_period: Long moving average period
        """
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals.
        
        Args:
            bars: DataFrame with OHLCV bars (must have 'bid_close' or 'close' column)
            
        Returns:
            DataFrame with 'prediction', 'probability', and 'meta_decision' columns
        """
        # Get price column
        if 'bid_close' in bars.columns:
            close = bars['bid_close']
        elif 'close' in bars.columns:
            close = bars['close']
        else:
            raise ValueError("bars must have 'bid_close' or 'close' column")
        
        # Calculate moving averages
        ma_short = close.rolling(window=self.short_period, min_periods=self.short_period).mean()
        ma_long = close.rolling(window=self.long_period, min_periods=self.long_period).mean()
        
        # Generate signals: buy when short MA crosses above long MA
        signals = pd.DataFrame(index=bars.index)
        signals['prediction'] = (ma_short > ma_long).astype(int)
        signals['probability'] = 0.5  # Placeholder
        signals['meta_decision'] = signals['prediction']
        
        return signals


class RSIStrategy:
    """RSI-based baseline strategy.
    
    Relative Strength Index strategy: buy when RSI < oversold, sell when RSI > overbought.
    Serves as a baseline for mean-reversion strategies.
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """Initialize RSI strategy.
        
        Args:
            period: RSI calculation period
            oversold: RSI threshold for oversold (buy signal)
            overbought: RSI threshold for overbought (sell signal)
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator.
        
        Args:
            prices: Price series
            
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period, min_periods=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period, min_periods=self.period).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based signals.
        
        Args:
            bars: DataFrame with OHLCV bars (must have 'bid_close' or 'close' column)
            
        Returns:
            DataFrame with 'prediction', 'probability', and 'meta_decision' columns
        """
        # Get price column
        if 'bid_close' in bars.columns:
            close = bars['bid_close']
        elif 'close' in bars.columns:
            close = bars['close']
        else:
            raise ValueError("bars must have 'bid_close' or 'close' column")
        
        # Calculate RSI
        rsi = self.calculate_rsi(close)
        
        # Generate signals: buy when RSI < oversold
        signals = pd.DataFrame(index=bars.index)
        signals['prediction'] = (rsi < self.oversold).astype(int)
        signals['probability'] = 0.5  # Placeholder
        signals['meta_decision'] = signals['prediction']
        
        return signals

