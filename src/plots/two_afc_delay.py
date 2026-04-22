"""
two_afc.py
──────────
Plotting utilities for 2-AFC (binary) GLM-HMM results.

This is the task-owned 2AFC plotting module exposed via
``TaskAdapter.get_plots()``. Its public API mirrors the task plot interface
used by the analysis notebooks.

High-level functions:
  - plot_emission_weights
  - plot_posterior_probs
  - plot_state_accuracy
  - plot_session_trajectories
  - plot_state_occupancy
  - plot_state_dwell_times
  - plot_psychometric_all        (≡ plot_categorical_performance_all)
  - plot_psychometric_by_state   (≡ plot_categorical_performance_by_state)
  - plot_regressor_psychometric_by_state
  - plot_trans_mat               (already homologous)
  - plot_trans_mat_boxplots      (already homologous)
  - plot_model_comparison
  - plot_model_comparison_diffs
  - norm_ll

Low-level primitives are kept for direct use:
  - remap_states
  - plot_weights / plot_weights_per_contrast / plot_weights_boxplot
  - plot_occupancy / plot_occupancy_boxplot
  - plot_ll
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from src.process.two_afc_delay import EMISSION_REGRESSOR_LABELS
from glmhmmt.plots import (
    plot_weights_boxplot as _plot_weights_boxplot_simple,
)
from glmhmmt.postprocess import (
    build_transition_matrix_by_subject_payload,
    build_transition_matrix_payload,
    build_weights_boxplot_payload,
)
from glmhmmt.model_plots import (
    norm_ll as _norm_ll,
    plot_binary_emission_weights as _plot_binary_emission_weights,
    plot_binary_emission_weights_by_subject as _plot_binary_emission_weights_by_subject,
    plot_binary_emission_weights_summary as _plot_binary_emission_weights_summary,
    plot_binary_emission_weights_summary_boxplot as _plot_binary_emission_weights_summary_boxplot,
    plot_binary_emission_weights_summary_lineplot as _plot_binary_emission_weights_summary_lineplot,
    plot_lapse_rates_boxplot as _plot_lapse_rates_boxplot,
    plot_ll as _plot_ll,
    plot_model_comparison as _plot_model_comparison,
    plot_model_comparison_diffs as _plot_model_comparison_diffs,
    plot_occupancy as _plot_occupancy,
    plot_occupancy_boxplot as _plot_occupancy_boxplot,
    plot_posterior_probs as _plot_posterior_probs,
    plot_change_triggered_posteriors_by_subject as _plot_change_triggered_posteriors_by_subject,
    plot_change_triggered_posteriors_summary as _plot_change_triggered_posteriors_summary,
    plot_session_deepdive as _plot_session_deepdive,
    plot_session_trajectories as _plot_session_trajectories,
    plot_state_accuracy as _plot_state_accuracy,
    plot_state_dwell_times as _plot_state_dwell_times,
    plot_state_dwell_times_by_subject as _plot_state_dwell_times_by_subject,
    plot_state_dwell_times_summary as _plot_state_dwell_times_summary,
    plot_state_occupancy as _plot_state_occupancy,
    plot_state_occupancy_overall as _plot_state_occupancy_overall,
    plot_state_occupancy_overall_by_subject as _plot_state_occupancy_overall_by_subject,
    plot_state_occupancy_overall_boxplot as _plot_state_occupancy_overall_boxplot,
    plot_state_occupancy_overall_summary as _plot_state_occupancy_overall_summary,
    plot_state_posterior_count_kde as _plot_state_posterior_count_kde,
    plot_state_session_occupancy as _plot_state_session_occupancy,
    plot_state_session_occupancy_by_subject as _plot_state_session_occupancy_by_subject,
    plot_state_session_occupancy_summary as _plot_state_session_occupancy_summary,
    plot_state_switches as _plot_state_switches,
    plot_state_switches_by_subject as _plot_state_switches_by_subject,
    plot_state_switches_summary as _plot_state_switches_summary,
    plot_trans_mat as _plot_trans_mat,
    plot_trans_mat_boxplots as _plot_trans_mat_boxplots,
    plot_transition_matrix as _plot_transition_matrix_simple,
    plot_transition_matrix_by_subject as _plot_transition_matrix_by_subject_simple,
    plot_transition_weights,
)
from glmhmmt.views import get_state_color, get_state_palette

# ── state colour palette ──────────────────────────────────────────────────────
sns.set_style("ticks")

_SESSION_COL = "session"
_SORT_COL = "trial_idx"
_EMISSION_WEIGHT_SIGN = 1.0

def _state_colors(K: int) -> List[str]:
    return get_state_palette(K)[:K]


def _default_labels(K: int, C: int = 2) -> List[str]:
    """Auto-generate state labels like ['Disengaged','Engaged'] for K=2."""
    if K == 1:
        return ["State 0"]
    if K == 2:
        return ["Disengaged", "Engaged"]
    if K == 3:
        return ["Engaged", "Biased L", "Biased R"]
    return [f"State {k}" for k in range(K)]


# ─────────────────────────────────────────────────────────────────────────────
# State remapping
# ─────────────────────────────────────────────────────────────────────────────


def remap_states(
    weights: np.ndarray,
    trans_mat: np.ndarray,
    smoothed_probs: np.ndarray,
    stim_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Re-order states so the most stimulus-sensitive is last ('Engaged').

    For K=2: [disengaged, engaged]
    For K=3: [engaged, biased-left, biased-right]

    Args:
        weights:        (K, C-1, M) emission weight array.
        trans_mat:      (K, K) transition matrix.
        smoothed_probs: (T, K) posterior state probabilities.
        stim_idx:       Feature column index used to rank engagement.

    Returns:
        Remapped (weights, trans_mat, smoothed_probs, remap_indices).
    """
    K = weights.shape[0]
    stim_w = weights[:, 0, stim_idx]
    engaged = int(np.argmax(np.abs(stim_w)))

    if K == 2:
        order = [1 - engaged, engaged]
    elif K == 3:
        others = [k for k in range(K) if k != engaged]
        bias_w = weights[:, 0, :]
        biased_left = others[int(np.argmin([bias_w[k, 0] for k in others]))]
        biased_right = others[int(np.argmax([bias_w[k, 0] for k in others]))]
        order = [engaged, biased_left, biased_right]
    else:
        order = list(range(K))

    o = np.array(order)
    return weights[o], trans_mat[np.ix_(o, o)], smoothed_probs[:, o], order


# ─────────────────────────────────────────────────────────────────────────────
# Low-level weight plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_weights(
    weights: np.ndarray,
    feature_names: Sequence[str],
    state_labels: Optional[Sequence[str]] = None,
    state_colors: Optional[Sequence[str]] = None,
    title: str = "GLM-HMM weights",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar chart of emission weights per state.

    For C-1=1 (binary) each state has one row W[k,0,:].
    Multiple contrasts are averaged.

    Args:
        weights:       (K, C-1, M) or (K, M) weight array.
        feature_names: Names of the M features.
        state_labels:  Per-state labels.
        title:         Figure title.
        figsize:       Figure size.
        ax:            Optional existing Axes.

    Returns:
        matplotlib Figure.
    """
    W = np.asarray(weights)
    if W.ndim == 2:
        W = W[:, None, :]
    K, C_m1, M = W.shape
    labels = list(state_labels) if state_labels else _default_labels(K, C_m1 + 1)
    colors = list(state_colors) if state_colors is not None else _state_colors(K)
    x = np.arange(M)
    width = 0.8 / K

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(5, 0.7 * M), 3.5))
    else:
        fig = ax.figure

    for k in range(K):
        w_k = W[k].mean(axis=0)
        offset = (k - (K - 1) / 2) * width
        ax.bar(x + offset, w_k, width, label=labels[k], color=colors[k], alpha=0.85)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_weights_per_contrast(
    weights: np.ndarray,
    feature_names: Sequence[str],
    contrast_names: Optional[Sequence[str]] = None,
    state_labels: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """One subplot per contrast (row of W), all states overlaid."""
    W = np.asarray(weights)
    if W.ndim == 2:
        W = W[:, None, :]
    K, C_m1, M = W.shape
    labels = list(state_labels) if state_labels else _default_labels(K, C_m1 + 1)
    cnames = list(contrast_names) if contrast_names else [f"Contrast {c}" for c in range(C_m1)]
    colors = _state_colors(K)
    x = np.arange(M)
    bar_w = 0.8 / K

    fig, axes = plt.subplots(1, C_m1, figsize=figsize or (max(5, 0.7 * M) * C_m1, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for c, ax in enumerate(axes):
        for k in range(K):
            offset = (k - (K - 1) / 2) * bar_w
            ax.bar(x + offset, W[k, c], bar_w, label=labels[k], color=colors[k], alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
        ax.set_title(cnames[c])
        sns.despine(ax=ax)
    axes[0].set_ylabel("Weight")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_weights_boxplot(
    all_weights: np.ndarray,
    feature_names: Sequence[str],
    state_labels: Optional[Sequence[str]] = None,
    state_colors: Optional[Sequence[str]] = None,
    title: str = "GLM-HMM weights (across subjects)",
    figsize: Optional[Tuple[float, float]] = None,
    connect_subjects: bool = True,
    show_ttests: bool = True,
) -> plt.Figure:
    return _plot_weights_boxplot_simple(
        **build_weights_boxplot_payload(
            all_weights,
            feature_names=feature_names,
            state_labels=state_labels,
            state_colors=state_colors,
        ),
        title=title,
        figsize=figsize,
        connect_subjects=connect_subjects,
        show_ttests=show_ttests,
    )


plot_trans_mat = _plot_trans_mat
plot_trans_mat_boxplots = _plot_trans_mat_boxplots
plot_occupancy = _plot_occupancy
plot_occupancy_boxplot = _plot_occupancy_boxplot
norm_ll = _norm_ll
plot_ll = _plot_ll
plot_model_comparison = _plot_model_comparison
plot_model_comparison_diffs = _plot_model_comparison_diffs


def plot_transition_matrix_by_subject(
    matrices,
    subject_ids,
    tick_labels_by_subject,
    **kwargs,
):
    return _plot_transition_matrix_by_subject_simple(
        matrices=matrices,
        subject_ids=subject_ids,
        tick_labels_by_subject=tick_labels_by_subject,
        **kwargs,
    )


def plot_transition_matrix(
    matrix,
    tick_labels,
    **kwargs,
):
    return _plot_transition_matrix_simple(
        matrix=matrix,
        tick_labels=tick_labels,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric helpers  (2AFC equivalent of categorical performance panels)
# ─────────────────────────────────────────────────────────────────────────────

_LABELED_ILDS = {-8, 8}


def _legacy_square_panel_size(n_cols: int = 2) -> tuple[float, float]:
    """Return the legacy A4-derived square panel size used in older plots."""
    a4_size = np.array((8.27, 11.69), dtype=float)
    margins = 2.0
    usable = a4_size - margins
    panel_w = float(usable[0] / float(n_cols))
    return panel_w, panel_w


# ─────────────────────────────────────────────────────────────────────────────
# GLM grid evaluation  (smooth sigmoid for psychometric plots)
# ─────────────────────────────────────────────────────────────────────────────


def _feature_label(feature_name: str) -> str:
    return EMISSION_REGRESSOR_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def _format_feature_labels(feature_names: Sequence[str]) -> list[str]:
    return [_feature_label(name) for name in feature_names]


def _reorder_two_afc_emission_features(
    weights: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Put stimulus, |bias|, and action-trace features first for 2AFC emission plots."""
    feat_names = list(feature_names)
    if not feat_names:
        return np.asarray(weights), feat_names

    def _group(idx: int, name: str) -> tuple[int, int]:
        lname = name.lower()
        if lname.startswith("stim"):
            return (0, idx)
        if lname == "bias":
            return (1, idx)
        if lname.startswith("at_"):
            return (2, idx)
        return (3, idx)

    order = [idx for idx, _ in sorted(enumerate(feat_names), key=lambda item: _group(item[0], item[1]))]
    W = np.take(np.asarray(weights), order, axis=-1).copy()
    ordered_names = [feat_names[idx] for idx in order]

    for idx, name in enumerate(ordered_names):
        if name.lower() == "bias":
            W[..., idx] = np.abs(W[..., idx])

    return W, ordered_names


def _two_afc_feature_order(feature_names: Sequence[str]) -> list[str]:
    return _reorder_two_afc_emission_features(
        np.zeros((1, 1, max(1, len(feature_names))), dtype=float),
        feature_names,
    )[1]


def _reorder_two_afc_emission_states(
    weights: np.ndarray,
    state_labels: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Move Disengaged to the front for 2AFC emission plot display order."""
    labels = list(state_labels)
    if not labels:
        return np.asarray(weights), labels

    disengaged = [idx for idx, label in enumerate(labels) if label.lower() == "disengaged"]
    remaining = [idx for idx in range(len(labels)) if idx not in disengaged]
    order = disengaged + remaining
    W = np.take(np.asarray(weights), order, axis=0)
    ordered_labels = [labels[idx] for idx in order]
    return W, ordered_labels


def _sparse_ild_labels(ilds: list) -> list:
    """Return tick labels that show only the extreme values and ±8; rest are empty."""
    lo, hi = min(ilds), max(ilds)
    labeled = _LABELED_ILDS | {lo, hi}
    labels: list[str] = []
    for v in ilds:
        if float(v) not in labeled:
            labels.append("")
            continue
        if float(v) == -20.0:
            labels.append("-70")
        elif float(v) == 20.0:
            labels.append("70")
        else:
            labels.append(str(int(v)))
    return labels


def _resolve_ild_ticks(
    ilds: Sequence,
    tick_ilds: Optional[Sequence[float]] = None,
) -> list[float]:
    vals = tick_ilds if tick_ilds is not None else ilds
    ticks = sorted({float(v) for v in vals if pd.notna(v)})
    if ticks:
        return ticks
    return sorted({float(v) for v in ilds if pd.notna(v)})


def _apply_ild_axis_ticks(ax: plt.Axes, xticks: Sequence[float]) -> None:
    xticks = np.asarray(xticks, dtype=float)
    ax.set_xticks(xticks, labels=_sparse_ild_labels(list(xticks)))
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        direction="out",
        length=7,
        width=1.1,
        color="#111827",
        labelcolor="#111827",
        pad=4,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["bottom"].set_color("#111827")


def _style_legacy_psych_axis(ax: plt.Axes, xticks: Sequence[float]) -> None:
    """Match the legacy categorical psychometric axis styling."""
    _apply_ild_axis_ticks(ax, xticks)
    ax.axhline(0.5, color="tab:gray", ls="--", lw=1.6)
    ax.axvline(0.0, color="tab:gray", ls="--", lw=1.6)
    ticks = np.asarray(xticks, dtype=float)
    if ticks.size >= 2:
        ax.set_xlim(float(ticks[0]), float(ticks[-1]))
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(r"$p(\mathrm{right})$")


def _require_plot_col(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        raise KeyError(f"Missing required plotting column {col!r}.")
    return col


def _psych_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    ild_col: str = "ILD",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    title: str = "",
    xlabel: str = "ILD (dB)",
    ylabel: Optional[str] = None,
    color: str = "k",
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    tick_ilds: Optional[Sequence[float]] = None,
) -> None:
    """Draw a pooled psychometric curve from a process-prepared payload."""
    if df.empty:
        ax.set_title(title)
        return

    choice_col = _require_plot_col(df, choice_col)
    payload = prepare_psych_panel_payload(
        df,
        x_col=ild_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        tick_values=tick_ilds,
    )
    if payload is None:
        ax.set_title(title)
        return

    subj_agg = payload["subject_summary"]
    xticks = payload["ticks"]

    if background_style == "data":
        for subj, grp in subj_agg.groupby(subj_col):
            grp_ilds = [i for i in payload["x"] if i in grp[ild_col].values]
            xi = np.array(grp_ilds, dtype=float)
            yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
            ax.plot(xi, yi, "-o", color=color, alpha=0.12, lw=1, ms=3, zorder=2)
    elif background_style == "model" and subject_curves is not None:
        for curve in subject_curves.values():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.12, lw=1.2, zorder=2)

    if smooth_curve is not None:
        ild_g, p_g = smooth_curve
        x0, x1 = float(xticks[0]), float(xticks[-1])
        clip = (ild_g >= x0) & (ild_g <= x1)
        ax.plot(ild_g[clip], p_g[clip], "-", color="black", lw=2, label="Model", zorder=6)
    else:
        ax.plot(payload["x"], payload["model_mean"], "-", color="black", lw=2, label="Model", zorder=6)

    ax.errorbar(
        payload["x"],
        payload["data_mean"],
        yerr=payload["data_sem"],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1,
        ms=5,
        label="Data",
        zorder=5,
    )

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    _apply_ild_axis_ticks(ax, xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(xlabel)


def _psych_state_panel(
    ax: plt.Axes,
    df_state: pd.DataFrame,
    ild_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    color: str,
    label: str,
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_subject_traces: bool = True,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    weight_col: Optional[str] = None,
    tick_ilds: Optional[Sequence[float]] = None,
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    model_line_mode: str = "smooth",
) -> Tuple:
    """Draw state-specific psychometric from a process-prepared payload."""
    if df_state.empty:
        return None, None

    choice_col = _require_plot_col(df_state, choice_col)
    payload = prepare_psych_state_panel_payload(
        df_state,
        x_col=ild_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        weight_col=weight_col,
        smooth_grid=smooth_curve[0] if smooth_curve is not None else None,
        tick_values=tick_ilds,
    )
    if payload is None:
        return None, None

    if show_subject_traces and background_style == "data":
        for subj, grp in payload["subject_summary"].groupby(subj_col):
            grp_ilds = [i for i in payload["x"] if i in grp[ild_col].values]
            xi = np.array(grp_ilds, dtype=float)
            yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
            ax.plot(xi, yi, "-o", color=color, alpha=0.14, lw=1.1, ms=4.0, zorder=2)
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for curve in subject_curves.values():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.14, lw=1.2, zorder=2)

    if show_data_smooth and payload["empirical_smooth"] is not None:
        x_emp, y_emp = payload["empirical_smooth"]
        ax.plot(x_emp, y_emp, "--", color=color, lw=1.9, alpha=0.95, zorder=4, label="_nolegend_")

    data_h = None
    if show_weighted_points:
        data_h = ax.errorbar(
            payload["x"],
            payload["data_mean"],
            yerr=payload["data_sem"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=0,
            ms=5.8,
            zorder=5,
            label=label,
        )

    if show_model_smooth and model_line_mode == "smooth" and smooth_curve is not None:
        ild_g, p_g = smooth_curve
        x0, x1 = float(payload["ticks"][0]), float(payload["ticks"][-1])
        clip = (ild_g >= x0) & (ild_g <= x1)
        (model_h,) = ax.plot(ild_g[clip], p_g[clip], "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    elif show_model_smooth:
        (model_h,) = ax.plot(payload["x"], payload["model_mean"], "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    else:
        model_h = None

    _style_legacy_psych_axis(ax, payload["ticks"])
    return data_h, model_h


def _regressor_state_panel(
    ax: plt.Axes,
    df_state: pd.DataFrame,
    feature_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    color: str,
    label: str,
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_subject_traces: bool = True,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    n_bins: int = 9,
    bin_edges: Optional[np.ndarray] = None,
    bin_centers: Optional[np.ndarray] = None,
    weight_col: Optional[str] = None,
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    model_line_mode: str = "smooth",
) -> Tuple:
    """Draw state-specific P(right) vs arbitrary regressor from a payload."""
    if df_state.empty:
        return None, None

    choice_col = _require_plot_col(df_state, choice_col)
    payload = prepare_regressor_state_panel_payload(
        df_state,
        feature_col=feature_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        n_bins=n_bins,
        weight_col=weight_col,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        smooth_grid=smooth_curve[0] if smooth_curve is not None else None,
    )
    if payload is None:
        return None, None

    if show_subject_traces and background_style == "data":
        for subj, grp in payload["subject_summary"].groupby(subj_col):
            grp = grp.sort_values("center")
            ax.plot(
                grp["center"].to_numpy(dtype=float),
                grp["data_mean"].to_numpy(dtype=float),
                "-o",
                color=color,
                alpha=0.15,
                lw=1.1,
                ms=4.0,
                zorder=2,
            )
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for curve in subject_curves.values():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.14, lw=1.2, zorder=2)

    if show_data_smooth and payload["empirical_smooth"] is not None:
        x_emp, y_emp = payload["empirical_smooth"]
        ax.plot(x_emp, y_emp, "--", color=color, lw=1.9, alpha=0.95, zorder=4, label="_nolegend_")

    data_h = None
    if show_weighted_points:
        data_h = ax.errorbar(
            payload["x"],
            payload["data_mean"],
            yerr=payload["data_sem"],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=0,
            ms=5.8,
            zorder=5,
            label=label,
        )
    if show_model_smooth and model_line_mode == "smooth" and smooth_curve is not None:
        feat_g, p_g = smooth_curve
        (model_h,) = ax.plot(feat_g, p_g, "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    elif show_model_smooth:
        (model_h,) = ax.plot(payload["x"], payload["model_mean"], "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    else:
        model_h = None

    ax.axhline(0.5, color="tab:gray", ls="--", lw=1.6)
    ax.axvline(0.0, color="tab:gray", ls="--", lw=1.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])
    ax.set_xlim([-1, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], labels=["-1", "0.5", "0", "0.5", "1"])
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        direction="out",
        length=7,
        width=1.1,
        color="#111827",
        labelcolor="#111827",
        pad=4,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.set_ylabel(r"$p(\mathrm{right})$")
    return data_h, model_h


# ─────────────────────────────────────────────────────────────────────────────
# High-level API used by the task plot facade
# ─────────────────────────────────────────────────────────────────────────────


def _weights_feature_order(weights_df) -> list[str]:
    features = weights_df["feature"].unique()
    if hasattr(features, "to_list"):
        features = features.to_list()
    return _two_afc_feature_order([str(feature) for feature in features])


def plot_emission_weights_by_subject(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    feature_order = _weights_feature_order(weights_df)
    return _plot_binary_emission_weights_by_subject(
        weights_df,
        K=K,
        weight_sign=_EMISSION_WEIGHT_SIGN,
        state_label_order=("Disengaged",),
        feature_order=feature_order,
        abs_features=("bias",),
        feature_labeler=_feature_label,
    )


def plot_emission_weights_summary(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    feature_order = _weights_feature_order(weights_df)
    return _plot_binary_emission_weights_summary(
        weights_df,
        K=K,
        weight_sign=_EMISSION_WEIGHT_SIGN,
        state_label_order=("Disengaged",),
        feature_order=feature_order,
        abs_features=("bias",),
        feature_labeler=_feature_label,
    )


def plot_emission_weights_summary_lineplot(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    feature_order = _weights_feature_order(weights_df)
    return _plot_binary_emission_weights_summary_lineplot(
        weights_df,
        K=K,
        weight_sign=_EMISSION_WEIGHT_SIGN,
        state_label_order=("Disengaged",),
        feature_order=feature_order,
        abs_features=("bias",),
        feature_labeler=_feature_label,
    )


def plot_emission_weights_summary_boxplot(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    feature_order = _weights_feature_order(weights_df)
    return _plot_binary_emission_weights_summary_boxplot(
        weights_df,
        K=K,
        weight_sign=_EMISSION_WEIGHT_SIGN,
        state_label_order=("Disengaged",),
        feature_order=feature_order,
        abs_features=("bias",),
        feature_labeler=_feature_label,
    )


def plot_lapse_rates_boxplot(
    views: dict,
    K: int,
) -> plt.Figure:
    return _plot_lapse_rates_boxplot(
        views,
        K,
        choice_labels=("Right", "Left"),
        title=f"Lapse rates  (K={K})",
    )


def plot_emission_weights(
    weights_df,
    *,
    K: int | None = None,
) -> Tuple[plt.Figure, plt.Figure]:
    feature_order = _weights_feature_order(weights_df)
    return _plot_binary_emission_weights(
        weights_df,
        K=K,
        weight_sign=_EMISSION_WEIGHT_SIGN,
        state_label_order=("Disengaged",),
        feature_order=feature_order,
        abs_features=("bias",),
        feature_labeler=_feature_label,
    )

def plot_posterior_probs(
    posterior_df,
    *,
    subject: str | None = None,
) -> plt.Figure:
    return _plot_posterior_probs(posterior_df, subject=subject)


def plot_state_accuracy(payload: dict) -> Tuple[plt.Figure, pd.DataFrame]:
    return _plot_state_accuracy(payload)


def plot_session_trajectories(payload: dict) -> plt.Figure:
    return _plot_session_trajectories(payload)


def plot_state_posterior_count_kde(payload: dict) -> plt.Figure:
    return _plot_state_posterior_count_kde(payload)


def plot_change_triggered_posteriors_summary(payload: dict) -> plt.Figure:
    return _plot_change_triggered_posteriors_summary(payload)


def plot_change_triggered_posteriors_by_subject(payload: dict) -> plt.Figure:
    return _plot_change_triggered_posteriors_by_subject(payload)


def plot_state_occupancy(payload: dict) -> plt.Figure:
    return _plot_state_occupancy(payload)


def plot_state_occupancy_overall(payload: dict) -> plt.Figure:
    return _plot_state_occupancy_overall(payload)


def plot_state_occupancy_overall_summary(payload: dict) -> plt.Figure:
    return _plot_state_occupancy_overall_summary(payload)


def plot_state_occupancy_overall_by_subject(payload: dict) -> plt.Figure:
    return _plot_state_occupancy_overall_by_subject(payload)


def plot_state_session_occupancy(payload: dict) -> plt.Figure:
    return _plot_state_session_occupancy(payload)


def plot_state_session_occupancy_summary(payload: dict) -> plt.Figure:
    return _plot_state_session_occupancy_summary(payload)


def plot_state_session_occupancy_by_subject(payload: dict) -> plt.Figure:
    return _plot_state_session_occupancy_by_subject(payload)


def plot_state_switches(payload: dict) -> plt.Figure:
    return _plot_state_switches(payload)


def plot_state_switches_summary(payload: dict) -> plt.Figure:
    return _plot_state_switches_summary(payload)


def plot_state_switches_by_subject(payload: dict) -> plt.Figure:
    return _plot_state_switches_by_subject(payload)


def plot_state_occupancy_overall_boxplot(payload: dict) -> plt.Figure:
    return _plot_state_occupancy_overall_boxplot(payload)


def plot_state_dwell_times_by_subject(payload: dict) -> plt.Figure:
    return _plot_state_dwell_times_by_subject(payload)


def plot_state_dwell_times_summary(payload: dict) -> plt.Figure:
    return _plot_state_dwell_times_summary(payload)


def plot_state_dwell_times(payload: dict) -> plt.Figure:
    return _plot_state_dwell_times(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Session deep-dive
# ─────────────────────────────────────────────────────────────────────────────


def plot_session_deepdive(
    payload: dict,
) -> plt.Figure:
    return _plot_session_deepdive(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric performance helpers
# ─────────────────────────────────────────────────────────────────────────────


def _plot_delay_accuracy_panel(
    ax: plt.Axes,
    df_pd: pd.DataFrame,
    *,
    title: str,
    color: str,
    delay_col: str = "delay",
    weight_col: str | None = None,
    ylabel: str = "Accuracy",
) -> None:
    summary, meta = process.prepare_delay_accuracy_summary(
        df_pd,
        delay_col=delay_col,
        weight_col=weight_col,
    )
    if summary.empty:
        ax.text(0.5, 0.5, "No valid delay data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.plot(summary["delay"], summary["model_acc"], color=color, lw=2.0, label="Model")
    ax.scatter(
        summary["delay"],
        summary["data_acc"],
        color=color,
        edgecolor=color,
        s=45,
        linewidth=1.5,
        zorder=3,
        label="Data",
    )
    ax.axhline(0.5, color="#888888", lw=0.8, ls="--", zorder=0)
    ax.set_title(title)
    ax.set_xlabel("Delay")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    if meta.get("xticks"):
        ax.set_xticks(meta["xticks"], labels=meta["x_tick_labels"])
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)


def plot_categorical_performance_all(
    df,
    model_name: str,
    ild_col: str = "delay",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    cond_col: str = "condition",
    exp_col: str = "experiment",
    views: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
) -> plt.Figure:
    """Plot task accuracy as a function of delay, ignoring stimulus sign."""
    del ild_col, choice_col, pred_col, subj_col, views, X_cols, ild_max, background_style
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()

    conds = sorted(df_pd[cond_col].dropna().unique()) if cond_col in df_pd.columns else []
    exps = sorted(df_pd[exp_col].dropna().unique()) if exp_col in df_pd.columns else []
    n_panels = 1 + len(conds) + len(exps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    axes = np.atleast_1d(axes)
    ax_idx = 0

    _plot_delay_accuracy_panel(
        axes[ax_idx],
        df_pd,
        title="a) Accuracy by delay",
        color="#2b7bba",
    )
    ax_idx += 1

    if conds:
        cond_colors = {"rest": "#444444", "saline": "#1f77b4", "drug": "#d62728"}
        for ci, cond in enumerate(conds):
            _plot_delay_accuracy_panel(
                axes[ax_idx],
                df_pd[df_pd[cond_col] == cond],
                title=f"{chr(ord('b') + ci)}) {cond}",
                color=cond_colors.get(cond, "k"),
            )
            ax_idx += 1

    if exps:
        exp_palette = sns.color_palette("Set2", len(exps))
        for ei, exp in enumerate(exps):
            _plot_delay_accuracy_panel(
                axes[ax_idx],
                df_pd[df_pd[exp_col] == exp],
                title=f"{chr(ord('b') + len(conds) + ei)}) {exp}",
                color=exp_palette[ei],
            )
            ax_idx += 1
    fig.tight_layout()
    return fig, None


def plot_categorical_performance_all_by_state(
    df,
    views: dict,
    model_name: str,
    ild_col: str = "delay",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    figure_dpi: float = 80.0,
    overlay_only: bool = False,
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
) -> plt.Figure:
    """Per-state accuracy by delay, ignoring stimulus sign."""
    del ild_col, choice_col, pred_col, X_cols, ild_max, background_style
    del show_weighted_points, show_data_smooth, show_model_smooth, model_line_mode
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().reset_index(drop=True)
    else:
        df_pd = df.reset_index(drop=True)

    K = next(iter(views.values())).K if views else 2

    # State assignment from trial_df (state_rank: 0=Engaged, 1=Disengaged, …)
    if "state_rank" in df_pd.columns:
        _arr = df_pd["state_rank"].to_numpy().astype(int)
    elif "_state_k" in df_pd.columns:
        _arr = df_pd["_state_k"].to_numpy().astype(int)
    else:
        raise ValueError("df must contain a 'state_rank' column (output of build_trial_df)")

    df_pd = df_pd.copy()
    df_pd["_state_k"] = _arr
    if state_assignment_mode == "weighted":
        df_pd = _attach_rank_posterior_cols(df_pd, views, subj_col=subj_col)
        df_pd = _attach_rank_state_model_cols(df_pd, views, subj_col=subj_col, base_col="pR_state")

    slbls = ranked_state_labels(views)

    panel_w = 4

    _include_overlay = K > 1
    if overlay_only:
        _include_overlay = True
    _n_panels = K + int(_include_overlay)
    if overlay_only:
        _n_panels = 1
    _figsize = (3, 3) if overlay_only else (panel_w * _n_panels, 4)
    fig, axes = plt.subplots(1, _n_panels, figsize=_figsize, sharey=True, dpi=figure_dpi)
    axes = np.atleast_1d(axes)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _plot_delay_accuracy_panel(
                _ax_overlay,
                _df_state,
                title="",
                color=color,
                weight_col=_weight_col,
            )
        _ax_overlay.set_xlabel("Delay")
        _ax_overlay.set_title("")
        _ax_overlay.legend(frameon=False, fontsize=8)

    if not overlay_only:
        for k, ax in enumerate(axes[int(_include_overlay) :]):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _plot_delay_accuracy_panel(
                ax,
                _df_state,
                title=lbl,
                color=color,
                weight_col=_weight_col,
            )
            ax.set_xlabel("Delay")
            if k == 0:
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylabel("")

    fig.suptitle(model_name, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, None


# Alias used by the analysis notebooks
plot_categorical_performance_by_state = plot_categorical_performance_all_by_state


def plot_regressor_psychometric_by_state(
    df,
    views: dict,
    model_name: str,
    feature_col: str = "at_choice",
    choice_col: str = "response",
    subj_col: str = "subject",
    X_cols: Optional[Sequence[str]] = None,
    feature_min: Optional[float] = None,
    feature_max: Optional[float] = None,
    background_style: str = "data",
    n_bins: int = 9,
    n_grid: int = 300,
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    figure_dpi: float = 80.0,
    overlay_only: bool = False,
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
) -> plt.Figure:
    """Per-state partial-dependence plot for any emission regressor.

    The x-axis is the chosen regressor (for example ``at_choice``) instead of
    ILD. Empirical points are pooled within quantile bins of that regressor,
    while the model line sweeps the same regressor over a dense grid and
    marginalises over the empirical distribution of the remaining features.
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().reset_index(drop=True)
    else:
        df_pd = df.reset_index(drop=True)

    if feature_col not in df_pd.columns:
        raise ValueError(f"df must contain the regressor column {feature_col!r}.")

    df_pd = df_pd.copy()
    df_pd[feature_col] = pd.to_numeric(df_pd[feature_col], errors="coerce")
    df_pd = df_pd.dropna(subset=[feature_col])
    if df_pd.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No valid {feature_col} data", ha="center", va="center")
        ax.axis("off")
        return fig, None

    if "state_rank" in df_pd.columns:
        _arr = df_pd["state_rank"].to_numpy().astype(int)
    elif "_state_k" in df_pd.columns:
        _arr = df_pd["_state_k"].to_numpy().astype(int)
    else:
        raise ValueError("df must contain a 'state_rank' column (output of build_trial_df)")
    df_pd["_state_k"] = _arr
    if state_assignment_mode == "weighted":
        df_pd = _attach_rank_posterior_cols(df_pd, views, subj_col=subj_col)
        df_pd = _attach_rank_state_model_cols(df_pd, views, subj_col=subj_col, base_col="pR_state")
    _global_bin_edges, _global_bin_centers = _quantile_bin_spec(
        df_pd[feature_col].to_numpy(dtype=float),
        n_bins=n_bins,
    )

    if feature_min is None:
        feature_min = float(np.nanmin(df_pd[feature_col].to_numpy(dtype=float)))
    if feature_max is None:
        feature_max = float(np.nanmax(df_pd[feature_col].to_numpy(dtype=float)))
    if not np.isfinite(feature_min) or not np.isfinite(feature_max):
        raise ValueError(f"Could not infer finite range for {feature_col!r}.")
    if feature_min == feature_max:
        feature_min -= 1e-6
        feature_max += 1e-6

    K = next(iter(views.values())).K if views else int(df_pd["_state_k"].max()) + 1

    slbls = ranked_state_labels(views)
    _as = rank_ordered_arrays_store(views)
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else list(_as.keys())

    _smooth_by_k: dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    _test_W = next((v.emission_weights for v in views.values()), None)
    _K_fit = int(np.asarray(_test_W).shape[0]) if _test_W is not None else 1
    _smooth_single = _mean_glm_feature_curve(
        _as,
        _all_subjects,
        X_cols,
        feature_name=feature_col,
        grid_min=feature_min,
        grid_max=feature_max,
        state_k=None,
        n_grid=n_grid,
    )
    for k in range(K):
        if _K_fit == 1:
            _smooth_by_k[k] = _smooth_single
        else:
            _smooth_by_k[k] = _mean_glm_feature_curve(
                _as,
                _all_subjects,
                X_cols,
                feature_name=feature_col,
                grid_min=feature_min,
                grid_max=feature_max,
                state_k=k,
                n_grid=n_grid,
            )
    _subject_curves_by_k = (
        {
            k: _subject_glm_feature_curves(
                _as,
                _all_subjects,
                X_cols,
                feature_name=feature_col,
                grid_min=feature_min,
                grid_max=feature_max,
                state_k=None if _K_fit == 1 else k,
                n_grid=n_grid,
            )
            for k in range(K)
        }
        if background_style == "model"
        else {}
    )

    _include_overlay = K > 1
    if overlay_only:
        _include_overlay = True
    _n_panels = K + int(_include_overlay)
    if overlay_only:
        _n_panels = 1
    _figsize = (3, 3) if overlay_only else (4 * _n_panels, 4)
    fig, axes = plt.subplots(1, _n_panels, figsize=_figsize, sharey=True, dpi=figure_dpi)
    axes = np.atleast_1d(axes)

    xlabel = _feature_label(feature_col)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _regressor_state_panel(
                _ax_overlay,
                _df_state,
                feature_col,
                choice_col,
                pred_col=f"_pR_state_rank_{k}" if state_assignment_mode == "weighted" else "p_pred",
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                show_subject_traces=False,
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                n_bins=n_bins,
                bin_edges=_global_bin_edges,
                bin_centers=_global_bin_centers,
                weight_col=_weight_col,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
            )
        _ax_overlay.set_xlabel(xlabel)
        _ax_overlay.set_ylabel(r"$p(\mathrm{right})$")
        _ax_overlay.set_title("")
        _ax_overlay.legend(frameon=False, fontsize=8)

    if not overlay_only:
        for k, ax in enumerate(axes[int(_include_overlay) :]):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _regressor_state_panel(
                ax,
                _df_state,
                feature_col,
                choice_col,
                pred_col=f"_pR_state_rank_{k}" if state_assignment_mode == "weighted" else "p_pred",
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                n_bins=n_bins,
                bin_edges=_global_bin_edges,
                bin_centers=_global_bin_centers,
                weight_col=_weight_col,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
            )
            ax.set_xlabel(xlabel)
            # ax.set_title(lbl)
            if k == 0:
                ax.set_ylabel("P(Right)")
            else:
                ax.set_ylabel("")

    # fig.suptitle(f"{model_name} — {_feature_label(feature_col)} psychometric", y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, None


from src.process.common import (
    attach_repeat_choice_evidence,
    attach_signed_delay_columns,
    attach_total_fitted_evidence,
    attach_rank_posterior_cols,
    attach_rank_state_model_cols,
    binned_feature_summary,
    display_regressor_name as _display_regressor_name,
    mean_weighted_empirical_curve,
    pick_choice_history_regressor,
    prepare_evidence_curve,
    add_choice_lag_summary_regressor,
    prepare_psych_panel_payload,
    prepare_psych_state_panel_payload,
    prepare_regressor_state_panel_payload,
    prepare_right_integration_maps,
    quantile_bin_spec,
    rank_ordered_arrays_store,
    REPEAT_EVIDENCE_TAIL_QUANTILES,
    ranked_state_labels,
    resolve_ild_max,
    to_pandas_df,
)
from src.process import two_afc_delay as process
from src.plots.common import (
    add_shared_figure_legend,
    centered_numeric_group_palette,
    make_single_panel_figure,
    plot_grouped_summary,
    plot_empirical_accuracy_curve,
    plot_integration_map_panels,
    plot_simple_summary,
)

display_regressor_name = _display_regressor_name
_resolve_ild_max = resolve_ild_max
_mean_glm_curve = process.mean_glm_ild_curve
_subject_glm_curves = process.subject_glm_ild_curves
_mean_glm_feature_curve = process.mean_glm_feature_curve
_subject_glm_feature_curves = process.subject_glm_feature_curves
_mean_weighted_empirical_curve = mean_weighted_empirical_curve
_quantile_bin_spec = quantile_bin_spec
_binned_feature_summary = lambda df, feature_col, choice_col, pred_col, subj_col, **kwargs: binned_feature_summary(
    df,
    feature_col=feature_col,
    choice_col=choice_col,
    pred_col=pred_col,
    subj_col=subj_col,
    **kwargs,
)
_attach_rank_posterior_cols = attach_rank_posterior_cols
_attach_rank_state_model_cols = attach_rank_state_model_cols

SIGNED_DELAY_ORDER = ["0L", "-1", "-3", "-10", "10", "3", "1", "0R"]
SIGNED_DELAY_LABELS = ["0", "-1", "-3", "-10", "10", "3", "1", "0"]


def plot_accuracy_by_delay(plot_df):
    df_pd = to_pandas_df(plot_df)
    
    return plot_empirical_accuracy_curve(
        df_pd,
        x_col="delays",
        invert_x=False,
        accuracy_col="hit",
        xlabel="Delay",
        title="2AFC delay",
        baseline=0.5,
        color="#1f77b4",
    )


def plot_right_by_regressor_simple(
    plot_df,
    *,
    regressor_col: str,
    title: str | None = None,
    xlabel: str | None = None,
    n_bins: int = 10,
):
    summary, meta = process.prepare_right_by_regressor_simple(
        plot_df,
        regressor_col=regressor_col,
        xlabel=xlabel,
        n_bins=n_bins,
    )
    return plot_simple_summary(summary, meta=meta, title=title)


def plot_binned_accuracy_figure(
    plot_df,
    *,
    regressor_col: str,
):
    panels, legend_title = process.prepare_binned_accuracy_figure(
        plot_df,
        regressor_col=regressor_col,
    )
    if not panels:
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels, strict=False):
        x_col = "_signed_delay_cat" if panel["meta"].get("categorical_x") else "delay"
        plot_grouped_summary(
            ax,
            panel["summary"],
            line_group_col="_reg_bin",
            x_col=x_col,
            meta=panel["meta"],
        )
        if ax.legend_ is not None:
            ax.legend_.remove()

    add_shared_figure_legend(fig, source_ax=axes[-1], title=legend_title)
    return fig


def plot_right_by_regressor(
    plot_df,
    *,
    regressor_col: str,
    title: str | None = None,
    xlabel: str | None = None,
    n_bins: int = 10,
):
    summary, meta = process.prepare_right_by_regressor(
        plot_df,
        regressor_col=regressor_col,
        xlabel=xlabel,
        n_bins=n_bins,
    )
    if summary is None or summary.empty:
        return None

    raw_line_order = meta.get("line_order") or []

    # map fake categorical delay ordering into a numeric palette ordering
    numeric_proxy = []
    for value in raw_line_order:
        if value == "0L":
            numeric_proxy.append(-999.0)
        elif value == "0R":
            numeric_proxy.append(999.0)
        else:
            numeric_proxy.append(float(value))

    numeric_palette = centered_numeric_group_palette(numeric_proxy)
    palette = {}
    for raw, proxy in zip(raw_line_order, numeric_proxy, strict=False):
        palette[raw] = numeric_palette[proxy]

    line_labels = meta.get("line_labels") or []
    label_map = dict(zip(raw_line_order, line_labels, strict=False)) if line_labels else {}

    fig, ax = make_single_panel_figure(extra_right_legend=True)
    plot_grouped_summary(
        ax,
        summary,
        line_group_col="_signed_delay_cat",
        x_col="x_center",
        meta=meta,
        label_map=label_map,
        palette=palette,
    )
    if title:
        ax.set_title(title)
    return fig


def plot_right_integration_map(
    plot_df,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    value_col: str | None = None,
    include_model: bool = True,
    bnd: float | None = None,
    dx: float | None = None,
    n_bins: int = 64,
    sigma: float | None = None,
    smooth: bool = True,
):
    _n_bins = n_bins
    _plot_df = plot_df
    _x_col = x_col
    _x_edges = None
    _xticks = None
    _x_tick_labels = None
    if x_col is None:
        _df_pd = attach_signed_delay_columns(to_pandas_df(plot_df))
        if "_signed_delay_cat" in _df_pd.columns and _df_pd["_signed_delay_cat"].notna().any():
            _x_order, _x_tick_labels = process.signed_delay_order_and_labels(_df_pd)
            _code_col = "_signed_delay_code"
            _code_map = {value: idx for idx, value in enumerate(_x_order)}
            _df_pd[_code_col] = _df_pd["_signed_delay_cat"].astype(str).map(_code_map)
            _df_pd = _df_pd[_df_pd[_code_col].notna()].copy()
            _plot_df = _df_pd
            _x_col = _code_col
            _x_edges = np.arange(-0.5, len(_x_order) + 0.5, 1.0)
            _xticks = list(range(len(_x_order)))

    panels, meta = prepare_right_integration_maps(
        _plot_df,
        response_mode=process.RESPONSE_MODE,
        pred_col=process.PRED_COL,
        x_col=_x_col,
        y_col=y_col,
        value_col=value_col,
        include_model=include_model,
        bnd=bnd,
        dx=dx,
        n_bins=_n_bins,
        sigma=sigma,
        fill_empty=smooth,
        default_sigma_dx=5.0,
        x_edges=_x_edges,
        xticks=_xticks,
        x_tick_labels=_x_tick_labels,
    )
    if _x_edges is not None:
        meta["xlabel"] = "Signed delay"
    return plot_integration_map_panels(
        panels,
        meta=meta,
        interpolation=None,
    )


def plot_accuracy_by_total_evidence(
    plot_df,
    *,
    adapter,
    views: dict,
):
    df_pd = attach_total_fitted_evidence(
        plot_df,
        adapter=adapter,
        views=views,
        is_mcdr=False,
    )
    if df_pd.empty or "_fitted_total_evidence" not in df_pd.columns:
        return None

    summary, meta = prepare_evidence_curve(
        df_pd,
        evidence_col="_fitted_total_evidence",
        data_col="correct_bool",
        model_col="_fitted_correct_prob",
        baseline=0.5,
        xlabel="Correct-vs-rest fitted evidence",
        ylabel="Accuracy",
    )
    return plot_simple_summary(summary, meta=meta)


def plot_repeat_by_repeat_evidence(
    plot_df,
    *,
    views: dict,
):
    df_pd = attach_repeat_choice_evidence(
        plot_df,
        views=views,
        is_mcdr=False,
    )
    if df_pd.empty:
        return None

    baseline = 1.0 / next(iter(views.values())).num_classes if views else 0.5
    summary, meta = prepare_evidence_curve(
        df_pd,
        evidence_col="_repeat_choice_evidence",
        data_col="_repeat_choice",
        model_col="_p_repeat_model",
        baseline=float(baseline),
        xlabel="Fitted evidence for repeating choice",
        ylabel="P(Repeat)",
        quantiles=REPEAT_EVIDENCE_TAIL_QUANTILES,
    )
    # Overlay theoretical pure logistic function (zero lapse)
    try:
        x = summary["x_center"].to_numpy(dtype=float)
        if x.size >= 2:
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            pad = max(1.0, (x_max - x_min) * 0.2)
            x_dense = np.linspace(x_min - pad, x_max + pad, 400)
            model_dense = 1.0 / (1.0 + np.exp(-x_dense))
        else:
            x_dense = None
            model_dense = None
        # Use model asymptotes (binned model endpoints) for lapse estimates
        left_model = float(summary["model_mean"].iloc[0])
        right_model = float(summary["model_mean"].iloc[-1])
        lapse_to_alternate = 1.0 - right_model
        lapse_to_repeat = left_model
        title = f"lapse to repeat: {lapse_to_repeat:.2f}, lapse to alternate: {lapse_to_alternate:.2f}"
    except Exception:
        x_dense = None
        model_dense = None
        title = None

    fig = plot_simple_summary(summary, meta=meta, title=title)
    if fig is not None and x_dense is not None and model_dense is not None:
        ax = fig.axes[0]
        ax.plot(x_dense, model_dense, color="black", linewidth=1.0, linestyle=(0, (3, 1)), alpha=0.9, zorder=1)
    return fig


__all__ = [
    "display_regressor_name",
    "pick_choice_history_regressor",
    "plot_accuracy_by_delay",
    "plot_accuracy_by_total_evidence",
    "plot_binned_accuracy_figure",
    "plot_repeat_by_repeat_evidence",
    "plot_right_integration_map",
    "plot_right_by_regressor",
    "plot_right_by_regressor_simple",
]
