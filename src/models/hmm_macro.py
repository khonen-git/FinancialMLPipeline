"""Macro HMM for slow market regime detection.

Detects regimes like:
- Trending up/down
- High/low volatility
- Range-bound vs directional
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class MacroHMM:
    """Macro Hidden Markov Model for market regimes."""
    
    def __init__(self, config: dict):
        """Initialize macro HMM.
        
        Args:
            config: HMM macro configuration
        """
        self.config = config
        self.n_states = config.get('n_states', 3)
        self.covariance_type = config.get('covariance_type', 'full')
        self.n_init = config.get('n_init', 5)
        self.max_iter = config.get('max_iter', 500)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = config.get('features', [
            'ret_long', 'vol_long', 'trend_slope', 'trend_strength'
        ])
    
    def fit(self, features: pd.DataFrame) -> 'MacroHMM':
        """Fit HMM on features.
        
        Args:
            features: DataFrame with macro features
            
        Returns:
            self
        """
        logger.info(f"Fitting macro HMM with {self.n_states} states")
        
        # Select and scale features
        X = features[self.feature_names].dropna()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.max_iter,
            random_state=42
        )
        
        self.model.fit(X_scaled)
        
        logger.info(f"Macro HMM fitted. Converged: {self.model.monitor_.converged}")
        
        return self
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict regime states.
        
        Args:
            features: DataFrame with macro features
            
        Returns:
            Series with regime states (integers)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X = features[self.feature_names].copy()
        
        # Handle NaN
        valid_mask = X.notna().all(axis=1)
        
        regimes = pd.Series(-1, index=features.index, dtype=int)
        
        if valid_mask.sum() > 0:
            X_scaled = self.scaler.transform(X[valid_mask])
            regimes[valid_mask] = self.model.predict(X_scaled)
        
        return regimes
    
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict regime probabilities.
        
        Args:
            features: DataFrame with macro features
            
        Returns:
            DataFrame with probability for each state
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        X = features[self.feature_names].copy()
        valid_mask = X.notna().all(axis=1)
        
        proba = pd.DataFrame(
            0.0,
            index=features.index,
            columns=[f'macro_state_{i}_proba' for i in range(self.n_states)]
        )
        
        if valid_mask.sum() > 0:
            X_scaled = self.scaler.transform(X[valid_mask])
            proba.loc[valid_mask, :] = self.model.predict_proba(X_scaled)
        
        return proba

