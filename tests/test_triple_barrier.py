"""Unit tests for Triple Barrier labeling."""

import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling.triple_barrier import compute_triple_barrier, TripleBarrierLabeler
from src.labeling.session_calendar import SessionCalendar


class TestComputeTripleBarrier(unittest.TestCase):
    """Tests de la fonction de base compute_triple_barrier (long-only, bid/ask)."""

    def setUp(self):
        session_config = {
            'session_start': "00:00",
            'session_end': "21:55",
            'friday_end': "20:00",
            'weekend_trading': False,
        }
        self.calendar = SessionCalendar(session_config)

    def _make_bars(self, dates, bid_high, bid_low, bid_close=None):
        if bid_close is None:
            bid_close = [bh for bh in bid_high]

        return pd.DataFrame(
            {
                "timestamp": dates,
                "bid_open": 1.0700,
                "bid_high": bid_high,
                "bid_low": bid_low,
                "bid_close": bid_close,
                "ask_open": 1.0705,
                "ask_high": 1.0715,
                "ask_low": 1.0700,
                "ask_close": 1.0705,
            }
        ).set_index("timestamp")

    def test_tp_hit_before_sl(self):
        """TP touché avant SL → label = 1, barrier_hit = 'tp'."""
        dates = pd.date_range('2024-01-09 10:00', periods=10, freq='5min', tz='UTC')

        bars = self._make_bars(
            dates,
            bid_high=[1.0700, 1.0716] + [1.0716] * 8,  # TP atteint dès la 2e barre
            bid_low=[1.0695] * 10,
        )

        events = pd.DataFrame(
            {"timestamp": [dates[0]], "bar_index": [0]}
        )

        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=0.001,   # 10 ticks
            sl_distance=0.001,
            max_horizon_bars=5,
            session_calendar=self.calendar,
            min_horizon_bars=1,
            avg_bar_duration_sec=5 * 60,
        )

        self.assertEqual(len(labels), 1)
        row = labels.iloc[0]
        self.assertEqual(row["label"], 1)
        self.assertEqual(row["barrier_hit"], "tp")
        self.assertGreater(row["pnl"], 0)

    def test_sl_hit_before_tp(self):
        """SL touché avant TP → label = -1, barrier_hit = 'sl'."""
        dates = pd.date_range('2024-01-09 10:00', periods=10, freq='5min', tz='UTC')

        bars = self._make_bars(
            dates,
            bid_high=[1.0700] * 10,
            bid_low=[1.0690] + [1.0690] * 9,  # SL atteint dès la 2e barre
        )

        events = pd.DataFrame(
            {"timestamp": [dates[0]], "bar_index": [0]}
        )

        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=0.001,
            sl_distance=0.001,
            max_horizon_bars=5,
            session_calendar=self.calendar,
            min_horizon_bars=1,
            avg_bar_duration_sec=5 * 60,
        )

        self.assertEqual(len(labels), 1)
        row = labels.iloc[0]
        self.assertEqual(row["label"], -1)
        self.assertEqual(row["barrier_hit"], "sl")
        self.assertLess(row["pnl"], 0)

    def test_time_barrier_only(self):
        """Aucune barrière touchée → label = 0, barrier_hit = 'time'."""
        dates = pd.date_range('2024-01-09 10:00', periods=10, freq='5min', tz='UTC')

        bars = self._make_bars(
            dates,
            bid_high=[1.0701] * 10,
            bid_low=[1.0699] * 10,
            bid_close=[1.0700] * 10,
        )

        events = pd.DataFrame(
            {"timestamp": [dates[0]], "bar_index": [0]}
        )

        labels = compute_triple_barrier(
            events=events,
            prices=bars,
            tp_distance=0.005,
            sl_distance=0.005,
            max_horizon_bars=5,
            session_calendar=self.calendar,
            min_horizon_bars=1,
            avg_bar_duration_sec=5 * 60,
        )

        self.assertEqual(len(labels), 1)
        row = labels.iloc[0]
        self.assertEqual(row["label"], 0)
        self.assertEqual(row["barrier_hit"], "time")


class TestTripleBarrierLabeler(unittest.TestCase):
    """Tests de la classe TripleBarrierLabeler (API publique)."""

    def setUp(self):
        self.config = {
            'tp_ticks': 10,
            'sl_ticks': 10,
            'max_horizon_bars': 50,
            'min_horizon_bars': 5,
            'distance_mode': 'ticks',
            'tick_size': 0.0001,
        }

        session_config = {
            'session_start': "00:00",
            'session_end': "21:55",
            'friday_end': "20:00",
            'weekend_trading': False,
        }
        self.calendar = SessionCalendar(session_config)
        self.labeler = TripleBarrierLabeler(self.config, self.calendar)

    def test_initialization_distances(self):
        """Vérifie que les distances TP/SL en prix sont correctement dérivées des ticks."""
        self.assertAlmostEqual(self.labeler.tp_distance, 10 * 0.0001)
        self.assertAlmostEqual(self.labeler.sl_distance, 10 * 0.0001)

    def test_label_dataset_basic_flow(self):
        """label_dataset doit retourner des labels alignés et non vides."""
        dates = pd.date_range('2024-01-09 10:00', periods=30, freq='5min', tz='UTC')

        bars = pd.DataFrame(
            {
                "timestamp": dates,
                "bid_open": 1.0700,
                "bid_high": [1.0715] * 30,  # assez haut pour toucher le TP
                "bid_low": [1.0690] * 30,
                "bid_close": [1.0710] * 30,
                "ask_open": 1.0705,
                "ask_high": 1.0716,
                "ask_low": 1.0700,
                "ask_close": 1.0705,
            }
        ).set_index("timestamp")

        # On utilise toutes les barres comme événements potentiels
        event_indices = bars.index

        labels = self.labeler.label_dataset(bars, event_indices, avg_bar_duration_sec=5 * 60)

        self.assertGreater(len(labels), 0)
        self.assertIn("label", labels.columns)
        # Tous les labels doivent être dans {-1, 0, 1}
        self.assertTrue(set(labels["label"].unique()).issubset({-1, 0, 1}))


if __name__ == '__main__':
    unittest.main()

