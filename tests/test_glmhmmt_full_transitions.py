import numpy as np

from src.process.common import (
    autocorrelogram_transition_matrices,
    glmhmmt_transition_weights_df,
)


class _View:
    K = 2
    state_name_by_idx = {0: "Engaged", 1: "Disengaged"}


def test_glmhmmt_transition_weights_df_keeps_source_destination_axis():
    weights = np.arange(8, dtype=float).reshape(2, 2, 2)
    df = glmhmmt_transition_weights_df(
        {
            "subject_a": {
                "transition_weights": weights,
                "U_cols": np.array(["cumulative_reward", "filtered_reward"], dtype=object),
            }
        },
        {"subject_a": _View()},
    )

    assert len(df) == 8
    row = df[
        (df["source_state_idx"] == 1)
        & (df["destination_state_idx"] == 0)
        & (df["feature"] == "filtered_reward")
    ].iloc[0]

    assert row["transition_label"] == "Disengaged -> Engaged"
    assert row["weight"] == weights[1, 0, 1]


def test_glmhmmt_transition_weights_df_restores_self_baseline_transitions():
    weights = np.array([[[1.0]], [[2.0]]])
    df = glmhmmt_transition_weights_df(
        {
            "subject_a": {
                "transition_weights": weights,
                "U_cols": np.array(["cumulative_reward"], dtype=object),
            }
        },
        {"subject_a": _View()},
    )

    assert len(df) == 4
    matrix = (
        df.pivot(
            index="source_state_idx",
            columns="destination_state_idx",
            values="weight",
        )
        .sort_index()
        .sort_index(axis=1)
        .to_numpy()
    )
    np.testing.assert_allclose(matrix, np.array([[0.0, 1.0], [2.0, 0.0]]))


def test_autocorrelogram_transition_matrices_use_full_source_destination_weights():
    arrays = {
        "transition_bias": np.zeros((2, 2), dtype=float),
        "transition_weights": np.array(
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
            ]
        ),
        "U": np.array([[0.0], [2.0], [3.0]], dtype=float),
    }

    matrices = autocorrelogram_transition_matrices(arrays, K=2, T=3)

    expected_logits = np.array([[0.0, 2.0], [4.0, 6.0]])
    expected = np.exp(expected_logits - expected_logits.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(matrices[0], expected)


def test_autocorrelogram_transition_matrices_restore_self_baseline_parameters():
    arrays = {
        "transition_bias": np.array([[-1.0], [-2.0]], dtype=float),
        "transition_weights": np.array([[[0.5]], [[-0.25]]], dtype=float),
        "U": np.array([[0.0], [2.0], [3.0]], dtype=float),
    }

    matrices = autocorrelogram_transition_matrices(arrays, K=2, T=3)

    expected_logits = np.array([[0.0, 0.0], [-2.5, 0.0]])
    expected = np.exp(expected_logits - expected_logits.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(matrices[0], expected)
