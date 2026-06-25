import pandas as pd

from src.process.common import attach_signed_delay_columns
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
