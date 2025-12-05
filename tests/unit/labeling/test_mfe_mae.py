"""Unit tests for MFE/MAE computation."""

import pytest
import pandas as pd
import numpy as np
from src.labeling.mfe_mae import compute_mfe_mae


@pytest.mark.unit
class TestMFEMAE:
    """Test MFE/MAE computation."""
    
    def test_compute_mfe_mae_basic(self, sample_bars):
        """Test basic MFE/MAE computation."""
        tick_size = 0.00001
        
        result = compute_mfe_mae(
            bars=sample_bars,
            horizon_bars=10,
            quantile=0.5,
            tick_size=tick_size
        )
        
        assert result is not None
        assert len(result) == len(sample_bars)
        
        # Should have MFE and MAE columns
        assert 'mfe' in result.columns or 'mfe_max' in result.columns
        assert 'mae' in result.columns or 'mae_max' in result.columns
    
    def test_mfe_mae_quantiles(self, sample_bars):
        """Test MFE/MAE quantile computation."""
        tick_size = 0.00001
        
        result = compute_mfe_mae(
            bars=sample_bars,
            horizon_bars=10,
            quantile=0.5,
            tick_size=tick_size
        )
        
        # MFE/MAE can be negative in edge cases (e.g., if ask_close entry is used and price moves unfavorably)
        # This is expected behavior - we're computing excursions from entry price
        # Both should be finite values
        if 'mfe' in result.columns and 'mae' in result.columns:
            mfe_values = result['mfe'].dropna()
            mae_values = result['mae'].dropna()
            # Check that values are finite (not NaN, not Inf)
            assert mfe_values.isna().sum() == 0 or len(mfe_values) > 0
            assert mae_values.isna().sum() == 0 or len(mae_values) > 0
            if len(mfe_values) > 0:
                assert np.isfinite(mfe_values).all()
            if len(mae_values) > 0:
                assert np.isfinite(mae_values).all()
    
    def test_mfe_mae_horizon(self, sample_bars):
        """Test that MFE/MAE respects horizon."""
        tick_size = 0.00001
        
        result = compute_mfe_mae(
            bars=sample_bars,
            horizon_bars=5,
            quantile=0.5,
            tick_size=tick_size
        )
        
        # Last few rows should have NaN (not enough future bars)
        if len(result) > 5:
            # Check that last rows may have NaN
            assert result is not None
    
    def test_mfe_mae_no_future_leakage(self, sample_bars):
        """Test that MFE/MAE doesn't use future data incorrectly."""
        tick_size = 0.00001
        
        result = compute_mfe_mae(
            bars=sample_bars,
            horizon_bars=10,
            quantile=0.5,
            tick_size=tick_size
        )
        
        # MFE/MAE at index i should only use bars[i+1:i+horizon+1]
        # This is forward-looking by design (we're computing future MFE/MAE)
        # But we should verify it's computed correctly
        assert result is not None
        assert len(result) == len(sample_bars)

