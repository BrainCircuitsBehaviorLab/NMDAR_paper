from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process.nuo_auditory import NuoAuditoryAdapter


class TestNuoAuditoryGlm(unittest.TestCase):
    def test_dynamic_emission_cols_use_max_sessions_per_subject(self) -> None:
        adapter = NuoAuditoryAdapter()
        df = pl.DataFrame(
            {
                "subject": ["A", "A", "B", "B", "B", "B"],
                "session": ["s1", "s2", "t1", "t2", "t3", "t4"],
                "response": [0, 1, 0, 1, 0, 1],
                "last_choice": [0, 1, 0, 1, 0, 1],
                "difficulty": ["easy"] * 6,
                "correct_side": ["left"] * 6,
                "total_evidence_strength": [0.1] * 6,
            }
        )

        self.assertEqual(
            adapter._dynamic_emission_cols(df),
            [
                "stim_bin_00",
                "stim_bin_01",
                "stim_bin_02",
                "stim_bin_03",
                "stim_bin_04",
                "stim_bin_05",
                "stim_bin_06",
                "stim_bin_07",
                "stim_bin_08",
                "difficulty_easy",
                "difficulty_medium",
                "difficulty_hard",
                "bias_0",
                "bias_1",
                "bias_2",
                "bias_3",
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
            ],
        )

    def test_build_design_matrices_drops_unavailable_bias_hot_cols(self) -> None:
        adapter = NuoAuditoryAdapter()
        feature_df = pl.DataFrame(
            {
                "response": [0, 1],
                "bias": [1.0, 1.0],
                "bias_0": [1.0, 1.0],
                "stim_vals": [0.5, -0.5],
            }
        )

        y, X, U, names = adapter.build_design_matrices(
            feature_df,
            emission_cols=["bias", "bias_0", "bias_1", "stim_vals"],
            transition_cols=[],
        )

        self.assertEqual(y.shape, (2,))
        self.assertEqual(X.shape, (2, 3))
        self.assertEqual(U.shape, (2, 0))
        self.assertEqual(names["X_cols"], ["bias", "bias_0", "stim_vals"])

    def test_prepare_weight_family_plot_uses_task_defined_difficulty_levels(self) -> None:
        adapter = NuoAuditoryAdapter()
        weights_df = pl.DataFrame(
            {
                "subject": ["A", "A", "A", "A"],
                "weight_row_idx": [0, 1, 0, 1],
                "feature": [
                    "difficulty_easy",
                    "difficulty_easy",
                    "difficulty_hard",
                    "difficulty_hard",
                ],
                "weight": [0.75, 9.0, -0.25, 7.0],
            }
        )

        prepared = adapter.prepare_weight_family_plot(weights_df, "difficulty_hot")

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.x_order, ("easy", "hard"))
        data = prepared.data.sort_values("x_label").reset_index(drop=True)
        self.assertEqual(data["x_label"].tolist(), ["easy", "hard"])
        np.testing.assert_allclose(data["weight"].to_numpy(), [0.75, -0.25])


if __name__ == "__main__":
    unittest.main()
