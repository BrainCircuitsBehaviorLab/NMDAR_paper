from __future__ import annotations

import sys
import unittest
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process.two_afc_delay import TwoAFCDelayAdapter


class TestTwoAfcDelayGlm(unittest.TestCase):
    def test_default_emission_cols_expand_one_hot_delay_families(self) -> None:
        adapter = TwoAFCDelayAdapter()
        df = pl.DataFrame(
            {
                "subject": ["N1", "N1", "N1", "N1"],
                "session": ["s1", "s1", "s2", "s2"],
                "delays": [0.1, 1.0, 3.0, 1.0],
            }
        )

        default_cols = adapter.default_emission_cols(df)

        self.assertEqual(
            default_cols,
            [
                "bias_0",
                "bias_1",
                "choice_lag_01",
                "choice_lag_02",
                "choice_lag_03",
                "choice_lag_04",
                "choice_lag_05",
                "choice_lag_06",
                "choice_lag_07",
                "choice_lag_08",
                "choice_lag_09",
                "choice_lag_10",
                "choice_lag_11",
                "choice_lag_12",
                "choice_lag_13",
                "choice_lag_14",
                "choice_lag_15",
                "stim",
                "delay_0p1",
                "delay_1",
                "delay_3",
                "stim_x_delay_hot_0p1",
                "stim_x_delay_hot_1",
                "stim_x_delay_hot_3",
            ],
        )

    def test_stim_x_delay_hot_cols_are_inferred_from_raw_delay_levels(self) -> None:
        adapter = TwoAFCDelayAdapter()
        df = pl.DataFrame({"delays": [0.1, 1.0, 3.0]})

        self.assertEqual(
            adapter.stim_x_delay_hot_cols(df),
            ["stim_x_delay_hot_0p1", "stim_x_delay_hot_1", "stim_x_delay_hot_3"],
        )


if __name__ == "__main__":
    unittest.main()
