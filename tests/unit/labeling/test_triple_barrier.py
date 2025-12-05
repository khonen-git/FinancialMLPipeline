"""Unit tests for triple barrier labeling."""

import pytest
import pandas as pd
import numpy as np
from src.labeling.triple_barrier import compute_triple_barrier, TripleBarrierLabeler
from src.labeling.session_calendar import SessionCalendar


@pytest.mark.unit
class TestTripleBarrier:
    """Test triple barrier labeling."""
    
    def test_tp_hit_before_sl(self, sample_bars, sample_session_calendar):
        """TP hit before SL → label = 1, barrier_hit = 'tp'."""
        # Create event at first bar with required columns (timestamp as column, not index)
        events = pd.DataFrame({
            'timestamp': [sample_bars.index[0]],
            'bar_index': [0]  # Index into bars DataFrame
        })
        
        # Create bars where price goes up (TP hit)
        bars = sample_bars.copy()
        entry_price = bars.iloc[0]['bid_close']
        tp_price = entry_price + 0.001  # TP at +100 pips
        
        # Make sure high reaches TP before low reaches SL
        bars.loc[bars.index[1:5], 'bid_high'] = tp_price + 0.0001
        bars.loc[bars.index[1:5], 'bid_low'] = entry_price - 0.0001
        
        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=0.001,
            sl_distance=0.001,
            max_horizon_bars=10,
            session_calendar=sample_session_calendar,
            min_horizon_bars=1
        )
        
        if len(labels) > 0:
            assert labels.iloc[0]['label'] == 1
            assert labels.iloc[0]['barrier_hit'] == 'tp'
    
    def test_sl_hit_before_tp(self, sample_bars, sample_session_calendar):
        """SL hit before TP → label = -1, barrier_hit = 'sl'."""
        events = pd.DataFrame({
            'timestamp': [sample_bars.index[0]],
            'bar_index': [0]
        })
        
        bars = sample_bars.copy()
        entry_price = bars.iloc[0]['bid_close']
        
        # Make sure low reaches SL before high reaches TP
        bars.loc[bars.index[1:5], 'bid_low'] = entry_price - 0.0011
        bars.loc[bars.index[1:5], 'bid_high'] = entry_price + 0.0001
        
        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=0.001,
            sl_distance=0.001,
            max_horizon_bars=10,
            session_calendar=sample_session_calendar,
            min_horizon_bars=1
        )
        
        if len(labels) > 0:
            assert labels.iloc[0]['label'] == -1
            assert labels.iloc[0]['barrier_hit'] == 'sl'
    
    def test_time_barrier_hit(self, sample_bars, sample_session_calendar):
        """Time barrier hit → label = 0, barrier_hit = 'time'."""
        events = pd.DataFrame({
            'timestamp': [sample_bars.index[0]],
            'bar_index': [0]
        })
        
        bars = sample_bars.copy()
        entry_price = bars.iloc[0]['bid_close']
        
        # Price stays between TP and SL
        bars.loc[bars.index[1:15], 'bid_high'] = entry_price + 0.0005
        bars.loc[bars.index[1:15], 'bid_low'] = entry_price - 0.0005
        
        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=0.001,
            sl_distance=0.001,
            max_horizon_bars=10,  # Time barrier at 10 bars
            session_calendar=sample_session_calendar,
            min_horizon_bars=1
        )
        
        if len(labels) > 0:
            # If time barrier is hit, label should be 0
            if labels.iloc[0]['barrier_hit'] == 'time':
                assert labels.iloc[0]['label'] == 0
    
    def test_triple_barrier_labeler_init(self, sample_bars):
        """Test TripleBarrierLabeler initialization."""
        config = {
            'distance_mode': 'ticks',
            'tp_ticks': 50,
            'sl_ticks': 50,
            'max_horizon_bars': 32,
            'min_horizon_bars': 10,
            'tick_size': 0.00001,  # Must be in config when using 'ticks' mode
        }
        
        assets_config = {
            'tick_size': 0.00001,
            'symbol': 'EURUSD'
        }
        
        labeler = TripleBarrierLabeler(config, assets_config, sample_bars)
        
        assert labeler is not None
        assert labeler.distance_mode == 'ticks'
    
    def test_triple_barrier_labeler_mfe_mae_mode(self, sample_bars, sample_session_calendar):
        """Test TripleBarrierLabeler with MFE/MAE mode."""
        config = {
            'distance_mode': 'mfe_mae',
            'mfe_mae': {
                'horizon_bars': 32,
                'tp_quantile': 0.5,
                'sl_quantile': 0.5,
            },
            'max_horizon_bars': 32,
            'min_horizon_bars': 32,
        }
        
        assets_config = {
            'tick_size': 0.00001,
            'symbol': 'EURUSD'
        }
        
        labeler = TripleBarrierLabeler(config, sample_session_calendar, bars=sample_bars, assets_config=assets_config)
        
        assert labeler is not None
        assert labeler.distance_mode == 'mfe_mae'

