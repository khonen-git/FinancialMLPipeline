"""Unit tests for bar builder module."""

import pytest
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from src.pipeline.bar_builder import build_bars
from tests.conftest import sample_ticks


@pytest.mark.unit
class TestBarBuilderModule:
    """Test bar builder module functionality."""
    
    def test_build_bars_basic(self, sample_ticks):
        """Test basic bar building."""
        cfg = OmegaConf.create({
            'session': {
                'timezone': 'UTC',
                'session_start': '00:00',
                'session_end': '21:55',
                'friday_end': '20:00',
                'weekend_trading': False,
            },
            'data': {
                'bars': {
                    'type': 'tick',
                    'threshold': 100
                }
            }
        })
        
        bars, calendar = build_bars(sample_ticks, cfg)
        
        assert len(bars) > 0
        assert 'bid_close' in bars.columns
        assert calendar is not None
        assert calendar.session_start is not None
    
    def test_build_bars_volume(self, sample_ticks):
        """Test volume bar building."""
        cfg = OmegaConf.create({
            'session': {
                'timezone': 'UTC',
                'session_start': '00:00',
                'session_end': '21:55',
                'friday_end': '20:00',
                'weekend_trading': False,
            },
            'data': {
                'bars': {
                    'type': 'volume',
                    'threshold': 500
                }
            }
        })
        
        bars, calendar = build_bars(sample_ticks, cfg)
        
        assert len(bars) > 0
        assert calendar is not None

