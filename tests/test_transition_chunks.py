import numpy as np
import polars as pl

from src.process.common import (
    build_outcome_streak_plot_data,
    build_transition_chunk_plot_data,
)


def test_transition_chunk_repeat_probability_is_subject_balanced():
    df = pl.DataFrame(
        {
            "subject": ["high_repeat"] * 5 + ["low_repeat"] * 2,
            "session": [1] * 7,
            "trial_idx": [0, 1, 2, 3, 4, 0, 1],
            "response": [0, 0, 0, 0, 0, 0, 1],
            "stimulus": [0, 0, 0, 0, 0, 0, 1],
        }
    )

    _, _, _, repeat_probabilities = build_transition_chunk_plot_data(
        {"task": df},
        ("task",),
        task_labels={"task": "Task"},
        task_order=("Task",),
    )

    p_repeat = repeat_probabilities.loc[
        repeat_probabilities["sequence"] == "Choices",
        "p_repeat",
    ].iloc[0]

    assert p_repeat == 0.5


def test_transition_chunk_null_uses_subject_balanced_repeat_probability():
    df = pl.DataFrame(
        {
            "subject": ["high_repeat"] * 5 + ["low_repeat"] * 2,
            "session": [1] * 7,
            "trial_idx": [0, 1, 2, 3, 4, 0, 1],
            "response": [0, 0, 0, 0, 0, 0, 1],
            "stimulus": [0, 0, 0, 0, 0, 0, 1],
        }
    )

    _, plot_data, _, _ = build_transition_chunk_plot_data(
        {"task": df},
        ("task",),
        stat="probability",
        task_labels={"task": "Task"},
        task_order=("Task",),
        max_chunk_length=3,
    )

    null_repeating = plot_data[
        (plot_data["transition"] == "repeating")
        & (plot_data["source"] == "Independent choices")
    ].sort_values("chunk_length")

    assert np.allclose(null_repeating["weight"].to_numpy(), [0.5, 0.25, 0.125])


def test_outcome_streak_null_uses_subject_balanced_accuracy():
    df = pl.DataFrame(
        {
            "subject": ["high_accuracy"] * 5 + ["low_accuracy"] * 2,
            "session": [1] * 7,
            "trial_idx": [0, 1, 2, 3, 4, 0, 1],
            "performance": [1, 1, 1, 1, 1, 1, 0],
        }
    )

    _, plot_data, _, outcome_probabilities = build_outcome_streak_plot_data(
        {"task": df},
        ("task",),
        stat="probability",
        task_labels={"task": "Task"},
        task_order=("Task",),
        max_chunk_length=3,
    )

    p_correct = outcome_probabilities.loc[
        outcome_probabilities["task_label"] == "Task",
        "p_correct",
    ].iloc[0]
    null_correct = plot_data[
        (plot_data["outcome"] == "Correct")
        & (plot_data["source"] == "Independent choices")
    ].sort_values("chunk_length")
    null_incorrect = plot_data[
        (plot_data["outcome"] == "Incorrect")
        & (plot_data["source"] == "Independent choices")
    ].sort_values("chunk_length")

    assert p_correct == 0.75
    assert np.allclose(null_correct["weight"].to_numpy(), [0.25, 0.1875, 0.140625])
    assert np.allclose(null_incorrect["weight"].to_numpy(), [0.75, 0.1875, 0.046875])
