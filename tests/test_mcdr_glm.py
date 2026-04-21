from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process import MCDR as mcdr


class TestMcdrGlm(unittest.TestCase):
    def test_build_emission_groups_keeps_bulk_stim_and_choice_lag_families(self) -> None:
        available_cols = [
            "bias",
            "bias_0",
            "bias_1",
            "biasL",
            "biasC",
            "biasR",
            "SLxD",
            "SCxD",
            "SRxD",
            "stim_param",
            "stim1L",
            "stim1C",
            "stim1R",
            "stim2L",
            "stim2C",
            "stim2R",
            "A_L",
            "A_C",
            "A_R",
            "choice_lag_01L",
            "choice_lag_01C",
            "choice_lag_01R",
            "choice_lag_02L",
            "choice_lag_02C",
            "choice_lag_02R",
        ]

        groups = mcdr._build_emission_groups(available_cols)
        groups_by_key = {group["key"]: group for group in groups}

        self.assertTrue(groups_by_key["bias_hot"]["hide_members"])
        self.assertEqual(
            groups_by_key["bias_side"]["members"],
            {"L": "biasL", "C": "biasC", "R": "biasR"},
        )
        self.assertEqual(
            groups_by_key["SxD"]["members"],
            {"L": "SLxD", "C": "SCxD", "R": "SRxD"},
        )
        self.assertTrue(groups_by_key["stim_one_hot"]["hide_members"])
        self.assertEqual(
            groups_by_key["stim_one_hot"]["toggle_members"],
            [
                "stim1L",
                "stim1C",
                "stim1R",
                "stim2L",
                "stim2C",
                "stim2R",
            ],
        )
        self.assertTrue(groups_by_key["choice_lag"]["hide_members"])
        self.assertEqual(
            groups_by_key["choice_lag"]["toggle_members"],
            [
                "choice_lag_01L",
                "choice_lag_01R",
                "choice_lag_02L",
                "choice_lag_02R",
            ],
        )
        self.assertEqual(
            groups_by_key["choice_lag"]["exclude_members"],
            [
                "choice_lag_01C",
                "choice_lag_02C",
            ],
        )
        self.assertNotIn("stim1", groups_by_key)
        self.assertNotIn("stim2", groups_by_key)
        self.assertNotIn("choice_lag_01", groups_by_key)

    def test_choice_lag_half_life_falls_back_to_matching_fit_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir) / "results"
            fit_dir = results_dir / "fits" / "MCDR" / "glm" / "choice-lag-fit"
            fit_dir.mkdir(parents=True)
            (fit_dir / "config.json").write_text(
                json.dumps(
                    {
                        "task": "MCDR",
                        "tau": 37,
                        "emission_cols": [
                            "choice_lag_01L",
                            "choice_lag_01C",
                            "choice_lag_01R",
                            "choice_lag_02L",
                        ],
                    }
                )
            )

            half_life = mcdr._resolve_choice_action_half_life(
                subject="A83",
                default_half_life=11.0,
                results_dir=results_dir,
            )

        self.assertEqual(half_life, 37.0)


if __name__ == "__main__":
    unittest.main()
