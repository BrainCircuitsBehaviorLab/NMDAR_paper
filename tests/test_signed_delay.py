import pandas as pd
import numpy as np

from src.process.common import (
    attach_signed_delay_columns,
    closed_loop_autocorrelogram_x,
    infer_autocorrelogram_choice_history_values,
)
from src.process.two_adc import _stimulus_to_signed_side


def test_attach_signed_delay_columns_maps_binary_stimulus_to_left_right_signs():
    df = pd.DataFrame(
        {
            "stimulus": [0, 1, 0, 1],
            "delay": [0.1, 0.1, 10.0, 10.0],
        }
    )

    out = attach_signed_delay_columns(df)

    assert out["_signed_delay"].tolist() == [-0.1, 0.1, -10.0, 10.0]
    assert list(out["_signed_delay_cat"].cat.categories) == [
        "-0.1",
        "-10",
        "10",
        "0.1",
    ]


def test_attach_signed_delay_columns_preserves_signed_stimulus_encoding():
    df = pd.DataFrame(
        {
            "stim": [-1, 1],
            "delays": [3.0, 3.0],
        }
    )

    out = attach_signed_delay_columns(df)

    assert out["_signed_delay"].tolist() == [-3.0, 3.0]
    assert list(out["_signed_delay_cat"].cat.categories) == ["-3", "3"]


def test_stimulus_to_signed_side_maps_tiffany_binary_stimulus():
    side = _stimulus_to_signed_side(pd.Series([0, 1, 0, 1]))

    assert side.tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_autocorrelogram_history_infers_aggregate_choice_lag_orientation():
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=float)
    base_x = np.asarray([[0.0], [1.0], [-1.0], [1.0], [-1.0], [1.0]], dtype=float)
    sessions = np.asarray(["s1"] * len(y))
    x_cols = ["choice_lag_param"]

    history_values = infer_autocorrelogram_choice_history_values(
        y,
        base_x,
        sessions,
        x_cols,
    )

    assert history_values == {0: 1.0, 1: -1.0}

    x_trial = closed_loop_autocorrelogram_x(
        np.asarray([0.0], dtype=float),
        trial_idx=2,
        choices=np.asarray([0, 1, np.nan], dtype=float),
        starts=np.asarray([0, 0, 0]),
        x_cols=x_cols,
        lag_param_weights={"choice_lag_param": {1: 0.7}},
        choice_history_values=history_values,
    )

    assert x_trial.tolist() == [-0.7]
