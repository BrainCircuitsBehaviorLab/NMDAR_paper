from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process.two_afc import TwoAFCAdapter


def _subject_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject": ["911", "911", "911"],
            "Session": ["s1", "s1", "s1"],
            "Trial": [1, 2, 3],
            "Experiment": ["2AFC_2", "2AFC_2", "2AFC_2"],
            "ILD": [2, -2, 4],
            "Choice": [1, 0, 1],
            "Hit": [1, 0, 1],
            "Side": [1, 0, 1],
        }
    )


class TestTwoAfcGlm(unittest.TestCase):
    def test_load_subject_skips_stim_param_when_not_requested(self) -> None:
        adapter = TwoAFCAdapter()
        df_sub = _subject_df()

        with (
            patch("process.two_afc.load_subject_choice_half_life", return_value=None),
            patch("process.two_afc._max_subject_sessions", return_value=1),
            patch(
                "glmhmmt.cli.alexis_functions.get_action_trace",
                return_value=(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
            ),
            patch(
                "process.two_afc._build_stim_param",
                side_effect=AssertionError("stim_param should not be built"),
            ),
        ):
            y, X, U, names = adapter.load_subject(
                df_sub,
                emission_cols=["bias", "stim_vals"],
                transition_cols=[],
            )

        self.assertEqual(y.shape, (3,))
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(names["X_cols"], ["bias", "stim_vals"])
        self.assertEqual(names["U_cols"], [])

    def test_load_subject_builds_stim_param_when_requested(self) -> None:
        adapter = TwoAFCAdapter()
        df_sub = _subject_df()
        stim_param = np.array([0.25, -0.25, 0.5], dtype=np.float32)

        with (
            patch("process.two_afc.load_subject_choice_half_life", return_value=None),
            patch("process.two_afc._max_subject_sessions", return_value=1),
            patch(
                "glmhmmt.cli.alexis_functions.get_action_trace",
                return_value=(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
            ),
            patch("process.two_afc._build_stim_param", return_value=stim_param) as build_stim_param,
        ):
            y, X, U, names = adapter.load_subject(
                df_sub,
                emission_cols=["bias", "stim_param"],
                transition_cols=[],
            )

        build_stim_param.assert_called_once()
        self.assertEqual(y.shape, (3,))
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(names["X_cols"], ["bias", "stim_param"])
        np.testing.assert_allclose(np.asarray(X[:, 1]), stim_param)

    def test_load_subject_drops_unavailable_bias_hot_cols(self) -> None:
        adapter = TwoAFCAdapter()
        df_sub = _subject_df()

        with (
            patch("process.two_afc.load_subject_choice_half_life", return_value=None),
            patch(
                "glmhmmt.cli.alexis_functions.get_action_trace",
                return_value=(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
            ),
        ):
            y, X, U, names = adapter.load_subject(
                df_sub,
                emission_cols=["bias", "bias_0", "bias_1", "stim_vals"],
                transition_cols=[],
            )

        self.assertEqual(y.shape, (3,))
        self.assertEqual(X.shape, (3, 3))
        self.assertEqual(U.shape, (3, 0))
        self.assertEqual(names["X_cols"], ["bias", "bias_0", "stim_vals"])


if __name__ == "__main__":
    unittest.main()
