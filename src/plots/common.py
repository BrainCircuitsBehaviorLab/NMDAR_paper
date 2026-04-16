from __future__ import annotations

from glmhmmt.plots import (
    custom_boxplot,
    plot_transition_matrix as _plot_transition_matrix,
    plot_transition_matrix_by_subject as _plot_transition_matrix_by_subject,
    plot_weights_boxplot as _plot_weights_boxplot,
)
from glmhmmt.postprocess import (
    build_transition_matrix_by_subject_payload,
    build_transition_matrix_payload,
    build_weights_boxplot_payload,
)


def plot_weights_boxplot(
    weights,
    feature_names=None,
    state_labels=None,
    state_colors=None,
    figsize=None,
    title: str = "GLM-HMM weights (across subjects)",
    connect_subjects: bool = True,
    show_ttests: bool = True,
    subject_line_color: str = "#7A7A7A",
    subject_line_alpha: float = 0.15,
    subject_line_width: float = 1.0,
):
    return _plot_weights_boxplot(
        **build_weights_boxplot_payload(
            weights,
            feature_names=feature_names,
            state_labels=state_labels,
            state_colors=state_colors,
        ),
        figsize=figsize,
        title=title,
        connect_subjects=connect_subjects,
        show_ttests=show_ttests,
        subject_line_color=subject_line_color,
        subject_line_alpha=subject_line_alpha,
        subject_line_width=subject_line_width,
    )


def plot_transition_matrix(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
):
    return _plot_transition_matrix(
        **build_transition_matrix_payload(
            arrays_store=arrays_store,
            state_labels=state_labels,
            K=K,
            subjects=subjects,
        )
    )


def plot_transition_matrix_by_subject(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
):
    return _plot_transition_matrix_by_subject(
        **build_transition_matrix_by_subject_payload(
            arrays_store=arrays_store,
            state_labels=state_labels,
            K=K,
            subjects=subjects,
        )
    )


__all__ = [
    "custom_boxplot",
    "plot_transition_matrix",
    "plot_transition_matrix_by_subject",
    "plot_weights_boxplot",
]
