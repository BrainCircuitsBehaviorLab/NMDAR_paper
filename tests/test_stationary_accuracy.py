import numpy as np
import pandas as pd

from src.process.common import add_stationary_accuracy_band


def test_stationary_accuracy_rolls_full_stimulus_condition_probabilities():
    session_data = pd.DataFrame(
        {
            "stimulus": [0, 0, 0, 0, 1, 1, 1, 1],
            "level": [1, 1, 2, 2, 1, 1, 2, 2],
            "correct": [1, 1, 1, 0, 1, 0, 0, 0],
            "accuracy_window_n": [1, 2, 2, 2, 2, 2, 2, 2],
        }
    )

    result = add_stationary_accuracy_band(session_data)

    np.testing.assert_allclose(
        result["stationary_accuracy_trial_fraction"],
        [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        result["stationary_accuracy_fraction"],
        [1.0, 1.0, 0.75, 0.5, 0.5, 0.5, 0.25, 0.0],
    )
    np.testing.assert_array_equal(
        result["stationary_accuracy_n"],
        [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    )


def test_stationary_accuracy_ci_is_analytical_binomial_interval():
    session_data = pd.DataFrame(
        {
            "stimulus": [0, 0, 0, 0],
            "level": [2, 2, 2, 2],
            "correct": [1, 1, 1, 0],
            "accuracy_window_n": [1, 2, 3, 4],
        }
    )

    result = add_stationary_accuracy_band(session_data)

    expected_pc = 0.75
    expected_half_width = 1.96 * np.sqrt(expected_pc * (1.0 - expected_pc) / 4)
    np.testing.assert_allclose(result["stationary_accuracy_fraction"], expected_pc)
    np.testing.assert_allclose(
        result["stationary_accuracy_low"].iloc[-1],
        max(0.0, expected_pc - expected_half_width),
    )
    np.testing.assert_allclose(
        result["stationary_accuracy_high"].iloc[-1],
        min(1.0, expected_pc + expected_half_width),
    )
