"""
two_afc.py
──────────
Plotting utilities for 2-AFC (binary) GLM-HMM results.

This module keeps only 2AFC task-owned plotting helpers. General model
diagnostics live in ``glmhmmt.plots`` and should be imported from there.

Task-owned high-level functions:
  - plot_categorical_performance_all
  - plot_categorical_performance_by_state
  - plot_regressor_psychometric_by_state
Task-specific primitives kept for direct use:
  - remap_states
  - plot_weights / plot_weights_per_contrast
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from typing import List, Optional, Sequence, Tuple
from src.process.two_afc import EMISSION_REGRESSOR_LABELS
from glmhmmt.views import get_state_color, get_state_palette
from glmhmmt.plots.common import resolve_axes_grid

from src.process.common import (
    compute_rb_by_x,
    to_pandas_df,
)

# ── state colour palette ──────────────────────────────────────────────────────

def _state_colors(K: int) -> List[str]:
    return get_state_palette(K)[:K]


_FOUR_STATE_LABELS = {"Engaged L", "Engaged R", "Disengaged L", "Disengaged R"}


def _two_afc_state_color(label: str, rank: int, K: int) -> str:
    if K == 4 and label in _FOUR_STATE_LABELS:
        palette = get_state_palette(K)
        return palette[int(rank) % len(palette)]
    return get_state_color(label, rank, K=K)


def _local_state_order(view) -> list[int]:
    order = [int(k) for k in (getattr(view, "state_idx_order", []) or [])]
    K = int(getattr(view, "K", len(order)))
    if len(order) == K and len(set(order)) == K:
        return order
    return list(range(K))


def _two_afc_ranked_state_labels(views: dict) -> dict[int, str]:
    labels: dict[int, str] = {}
    for view in views.values():
        for rank, raw_idx in enumerate(_local_state_order(view)):
            labels.setdefault(rank, getattr(view, "state_name_by_idx", {}).get(raw_idx, f"State {rank}"))
    return labels


def _two_afc_rank_ordered_arrays_store(views: dict) -> dict:
    out = {}
    for subject, view in views.items():
        order = _local_state_order(view)
        payload = {
            "emission_weights": np.asarray(view.emission_weights)[order],
            "X_cols": view.feat_names,
            "X": view.X,
            "smoothed_probs": np.asarray(view.smoothed_probs)[:, order],
        }
        payload["lapse_rates"] = getattr(view, "lapse_rates", None)
        out[subject] = payload
    return out


def _two_afc_raw_by_rank_maps(views: dict) -> tuple[int, dict, dict]:
    if not views:
        return 0, {}, {}
    K = int(next(iter(views.values())).K)
    raw_by_rank_by_subj = {}
    rank_by_raw_by_subj = {}
    for subject, view in views.items():
        raw_by_rank = {rank: raw_idx for rank, raw_idx in enumerate(_local_state_order(view))}
        rank_by_raw = {raw_idx: rank for rank, raw_idx in raw_by_rank.items()}
        raw_by_rank_by_subj[subject] = raw_by_rank
        raw_by_rank_by_subj[str(subject)] = raw_by_rank
        rank_by_raw_by_subj[subject] = rank_by_raw
        rank_by_raw_by_subj[str(subject)] = rank_by_raw
    return K, raw_by_rank_by_subj, rank_by_raw_by_subj


def _attach_two_afc_ranked_cols(
    df: pd.DataFrame,
    views: dict,
    *,
    subj_col: str,
    base_col: str = "pR_state",
) -> pd.DataFrame:
    """Attach rank-local posterior/model columns without using package label ranks."""
    if df.empty or subj_col not in df.columns or not views:
        return df

    K, raw_by_rank_by_subj, rank_by_raw_by_subj = _two_afc_raw_by_rank_maps(views)
    out = df.copy()

    if "state_idx" in out.columns:
        local_state = np.full(len(out), np.nan, dtype=float)
        for subject, idx in out.groupby(subj_col, observed=True).groups.items():
            rank_by_raw = rank_by_raw_by_subj.get(subject) or rank_by_raw_by_subj.get(str(subject))
            if rank_by_raw is None:
                continue
            row_idx = np.asarray(idx, dtype=int)
            raw_values = pd.to_numeric(out.iloc[row_idx]["state_idx"], errors="coerce")
            local_state[row_idx] = raw_values.map(rank_by_raw).to_numpy(dtype=float)
        if np.isfinite(local_state).any():
            out["_state_k"] = local_state
    elif "state_rank" in out.columns:
        out["_state_k"] = pd.to_numeric(out["state_rank"], errors="coerce")

    for rank in range(K):
        posterior_dst = f"_p_state_rank_{rank}"
        model_dst = f"_{base_col}_rank_{rank}"
        posterior_vals = np.full(len(out), np.nan, dtype=float)
        model_vals = np.full(len(out), np.nan, dtype=float)
        wrote_posterior = False
        wrote_model = False

        for subject, idx in out.groupby(subj_col, observed=True).groups.items():
            raw_by_rank = raw_by_rank_by_subj.get(subject) or raw_by_rank_by_subj.get(str(subject))
            if raw_by_rank is None or rank not in raw_by_rank:
                continue
            raw_idx = raw_by_rank[rank]
            row_idx = np.asarray(idx, dtype=int)

            posterior_src = f"p_state_pred_{raw_idx}"
            if posterior_src in out.columns:
                posterior_vals[row_idx] = pd.to_numeric(
                    out.iloc[row_idx][posterior_src],
                    errors="coerce",
                ).to_numpy(dtype=float)
                wrote_posterior = True

            model_src = f"{base_col}_{raw_idx}"
            if model_src in out.columns:
                model_vals[row_idx] = pd.to_numeric(
                    out.iloc[row_idx][model_src],
                    errors="coerce",
                ).to_numpy(dtype=float)
                wrote_model = True

        if wrote_posterior:
            out[posterior_dst] = posterior_vals
        if wrote_model:
            out[model_dst] = model_vals

    return out


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
    **plot_kwargs,
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

    style = dict(plot_kwargs)
    ax = style.pop("ax", None)
    figsize_arg = style.pop("figsize", None)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_arg or (max(5, 0.7 * M), 3.5))
    fig = ax.figure

    for k in range(K):
        w_k = W[k].mean(axis=0)
        offset = (k - (K - 1) / 2) * width
        ax.bar(x + offset, w_k, width, label=labels[k], color=colors[k], alpha=0.85)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
    ax.set_ylabel("Weight")
    ax.legend(frameon=False)
    fig.tight_layout()
    apply_axis_style(ax, **style)
    return ax


def plot_weights_per_contrast(
    weights: np.ndarray,
    feature_names: Sequence[str],
    contrast_names: Optional[Sequence[str]] = None,
    state_labels: Optional[Sequence[str]] = None,
    **plot_kwargs,
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

    style = dict(plot_kwargs)
    axes_arg = style.pop("axes", None)
    figsize_arg = style.pop("figsize", None)
    fig, axes = resolve_axes(
        axes_arg,
        n_axes=C_m1,
        figsize=figsize_arg or (max(5, 0.7 * M) * C_m1, 3.5),
        sharey=True,
    )
    for c, ax in enumerate(axes):
        for k in range(K):
            offset = (k - (K - 1) / 2) * bar_w
            ax.bar(x + offset, W[k, c], bar_w, label=labels[k], color=colors[k], alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
    axes[0].set_ylabel("Weight")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    for ax in axes[:C_m1]:
        apply_axis_style(ax, **style)
    return fig, axes[:C_m1]


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric helpers  (2AFC equivalent of categorical performance panels)
# ─────────────────────────────────────────────────────────────────────────────

_LABELED_ILDS = {-8, 8}

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
        color="#111827",
        labelcolor="#111827",
        pad=4,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#111827")


def _style_legacy_psych_axis(ax: plt.Axes, xticks: Sequence[float]) -> None:
    """Match the legacy categorical psychometric axis styling."""
    _apply_ild_axis_ticks(ax, xticks)
    ax.axhline(0.5, color="tab:gray", ls="--")
    ax.axvline(0.0, color="tab:gray", ls="--")
    ticks = np.asarray(xticks, dtype=float)
    if ticks.size >= 2:
        ax.set_xlim(float(ticks[0]), float(ticks[-1]))
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])
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
            ax.plot(xi, yi, "-o", color=color, alpha=0.14, ms=4.0, zorder=2)
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for curve in subject_curves.values():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.14, zorder=2)

    if show_data_smooth and payload["empirical_smooth"] is not None:
        x_emp, y_emp = payload["empirical_smooth"]
        ax.plot(x_emp, y_emp, "--", color=color, alpha=0.95, zorder=4, label="_nolegend_")

    data_h = None
    if show_weighted_points:
        data_h = ax.errorbar(
            payload["x"],
            payload["data_mean"],
            yerr=payload["data_sem"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=0,
            ms=5.8,
            zorder=5,
            label=label,
        )

    if show_model_smooth and model_line_mode == "smooth" and smooth_curve is not None:
        ild_g, p_g = smooth_curve
        x0, x1 = float(payload["ticks"][0]), float(payload["ticks"][-1])
        clip = (ild_g >= x0) & (ild_g <= x1)
        (model_h,) = ax.plot(ild_g[clip], p_g[clip], "-", color=color, zorder=6, label="_nolegend_")
    elif show_model_smooth:
        (model_h,) = ax.plot(payload["x"], payload["model_mean"], "-", color=color, zorder=6, label="_nolegend_")
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


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric performance helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_categorical_performance_all(
    df,
    model_name: str,
    ild_col: str = "ILD",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    cond_col: str = "condition",
    exp_col: str = "experiment",
    views: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
    **plot_kwargs,
) -> plt.Figure:
    """Overall psychometric + by-condition + by-experiment panels.

    2AFC equivalent of plots.plot_categorical_performance_all.

    Panels
    ------
    a) Overall      - P(right) vs ILD, all trials pooled
    b) By condition - separate curves per rest / saline / drug
                      (skipped if 'condition' column absent)
    c) By experiment - separate curves per experiment batch

    Parameters
    ----------
    df         : Polars or pandas DataFrame with Trial-level predictions.
                 Must contain: ILD, response/Choice (0/1), p_pred, subject.
                 The default model overlay is ``p_pred`` because this figure
                 shows psychometrics, i.e. P(rightward choice), not accuracy.
    model_name : String for figure suptitle.
    ild_max    : Optional explicit normalisation scale. When omitted, the
                 maximum absolute ILD in ``df[ild_col]`` is used.

    Returns
    -------
    fig
    """
    style = dict(plot_kwargs)
    axes_arg = style.pop("axes", None)
    figsize_arg = style.pop("figsize", None)
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()
    ild_max = _resolve_ild_max(df_pd, ild_col, ild_max)
    ild_ticks = (
        sorted(pd.to_numeric(df_pd[ild_col], errors="coerce").dropna().unique()) if ild_col in df_pd.columns else []
    )

    has_cond = cond_col in df_pd.columns
    has_exp = exp_col in df_pd.columns
    n_panels = 1 + int(has_cond) + int(has_exp)

    # Pre-compute smooth GLM sigmoid averaged over all subjects
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else []

    _as = rank_ordered_arrays_store(views) if views is not None else None
    _smooth_all = _mean_glm_curve(_as, _all_subjects, X_cols, ild_max=ild_max) if _as is not None else None
    _subject_curves_all = (
        _subject_glm_curves(_as, _all_subjects, X_cols, ild_max=ild_max) if _as is not None and background_style == "model" else None
    )

    fig, axes = resolve_axes(
        axes_arg,
        n_axes=n_panels,
        figsize=figsize_arg or (4 * n_panels, 4),
        sharey=True,
    )
    ax_idx = 0

    # a) Overall
    _psych_panel(
        axes[ax_idx],
        df_pd,
        ild_col=ild_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        xlabel="ILD (dB)",
        ylabel="P(Right)",
        color="#2b7bba",
        smooth_curve=_smooth_all,
        background_style=background_style,
        subject_curves=_subject_curves_all,
        tick_ilds=ild_ticks,
    )
    ax_idx += 1

    # b) By condition
    if has_cond:
        conds = sorted(df_pd[cond_col].dropna().unique())
        cond_colors = {"rest": "#444444", "saline": "#1f77b4", "drug": "#d62728"}
        for cond in conds:
            _cond_subjs = list(df_pd[df_pd[cond_col] == cond][subj_col].unique())
            _smooth_cond = _mean_glm_curve(_as, _cond_subjs, X_cols, ild_max=ild_max) if _as is not None else None
            _subject_curves_cond = (
                _subject_glm_curves(_as, _cond_subjs, X_cols, ild_max=ild_max)
                if _as is not None and background_style == "model"
                else None
            )
            _psych_panel(
                axes[ax_idx],
                df_pd[df_pd[cond_col] == cond],
                ild_col=ild_col,
                choice_col=choice_col,
                pred_col=pred_col,
                subj_col=subj_col,
                xlabel="ILD (dB)",
                color=cond_colors.get(cond, "k"),
                smooth_curve=_smooth_cond,
                background_style=background_style,
                subject_curves=_subject_curves_cond,
                tick_ilds=ild_ticks,
            )
        ax_idx += 1

    # c) By experiment
    if has_exp:
        exps = sorted(df_pd[exp_col].dropna().unique())
        exp_palette = sns.color_palette("Set2", len(exps))
        for ei, exp in enumerate(exps):
            _exp_subjs = list(df_pd[df_pd[exp_col] == exp][subj_col].unique())
            _smooth_exp = _mean_glm_curve(_as, _exp_subjs, X_cols, ild_max=ild_max) if _as is not None else None
            _subject_curves_exp = (
                _subject_glm_curves(_as, _exp_subjs, X_cols, ild_max=ild_max)
                if _as is not None and background_style == "model"
                else None
            )
            _psych_panel(
                axes[ax_idx],
                df_pd[df_pd[exp_col] == exp],
                ild_col=ild_col,
                choice_col=choice_col,
                pred_col=pred_col,
                subj_col=subj_col,
                xlabel="ILD (dB)",
                color=exp_palette[ei],
                smooth_curve=_smooth_exp,
                background_style=background_style,
                subject_curves=_subject_curves_exp,
                tick_ilds=ild_ticks,
            )

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ax in axes[:n_panels]:
        apply_axis_style(ax, **style)
    return fig, axes[:n_panels]


def plot_categorical_performance_all_by_state(
    df,
    views: dict,
    model_name: str,
    ild_col: str = "ILD",
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
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Figure:
    """Per-state psychometric grid (K panels, one per state).

    2AFC equivalent of plots.plot_categorical_performance_by_state.

    Each state gets its own panel showing P(right) vs ILD; data (markers) and
    model prediction (lines) are drawn in the state's colour.

    Parameters
    ----------
    df     : Trial-level DataFrame (polars or pandas). Should contain
             ``state_idx`` from :func:`~glmhmmt.postprocess.build_trial_df`;
             if present, ranks are derived locally from the adapter's
             ``state_idx_order`` so task labels cannot collide with package
             label ranks.
    views  : {subj: SubjectFitView} as produced by build_views.
    model_name : string used as figure suptitle.
    ild_max : Optional explicit normalisation scale. When omitted, the
              maximum absolute ILD in ``df[ild_col]`` is used.

    Returns
    -------
    (fig, None)
    """
    style = dict(plot_kwargs)
    axes_arg = style.pop("axes", None)
    figsize_arg = style.pop("figsize", None)
    if ax is not None:
        if not overlay_only:
            raise ValueError("ax can only be used when overlay_only=True.")
        axes_arg = [ax]
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().reset_index(drop=True)
    else:
        df_pd = df.reset_index(drop=True)
    ild_max = _resolve_ild_max(df_pd, ild_col, ild_max)
    ild_ticks = (
        sorted(pd.to_numeric(df_pd[ild_col], errors="coerce").dropna().unique()) if ild_col in df_pd.columns else []
    )

    K = next(iter(views.values())).K if views else 2

    df_pd = df_pd.copy()
    if "state_idx" not in df_pd.columns and "state_rank" not in df_pd.columns and "_state_k" not in df_pd.columns:
        raise ValueError("df must contain a 'state_idx' or 'state_rank' column (output of build_trial_df)")
    df_pd = _attach_two_afc_ranked_cols(df_pd, views, subj_col=subj_col, base_col="pR_state")
    if "_state_k" not in df_pd.columns:
        if "state_rank" in df_pd.columns:
            df_pd["_state_k"] = pd.to_numeric(df_pd["state_rank"], errors="coerce")
        else:
            raise ValueError("Could not derive local state ranks from trial_df and views.")
    df_pd["_state_k"] = pd.to_numeric(df_pd["_state_k"], errors="coerce")
    if state_assignment_mode == "weighted":
        missing_cols = [
            col
            for k in range(K)
            for col in (f"_p_state_rank_{k}", f"_pR_state_rank_{k}")
            if col not in df_pd.columns
        ]
        if missing_cols:
            df_pd = _attach_rank_posterior_cols(df_pd, views, subj_col=subj_col)
            df_pd = _attach_rank_state_model_cols(df_pd, views, subj_col=subj_col, base_col="pR_state")

    # Resolve labels: {rank: label} merged across all subjects
    slbls = _two_afc_ranked_state_labels(views)
    _as = _two_afc_rank_ordered_arrays_store(views)

    panel_w = 4

    # ── K-panel grid ──────────────────────────────────────────────────────────
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else []

    # Pre-compute per-state smooth sigmoid curves from views
    _smooth_by_k: dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    _test_W = next((v.emission_weights for v in views.values()), None)
    _K_fit = int(np.asarray(_test_W).shape[0]) if _test_W is not None else 1
    _smooth_single = _mean_glm_curve(_as, _all_subjects, X_cols, ild_max=ild_max, state_k=None)
    for k in range(K):
        if _K_fit == 1:
            _smooth_by_k[k] = _smooth_single
        else:
            _smooth_by_k[k] = _mean_glm_curve(_as, _all_subjects, X_cols, ild_max=ild_max, state_k=k)
    _subject_curves_by_k = (
        {k: _subject_glm_curves(_as, _all_subjects, X_cols, ild_max=ild_max, state_k=None if _K_fit == 1 else k) for k in range(K)}
        if background_style == "model"
        else {}
    )

    _include_overlay = bool(overlay_only)
    _n_panels = K + int(_include_overlay)
    if overlay_only:
        _n_panels = 1
    _figsize = (3, 3) if overlay_only else (panel_w * _n_panels, 4)
    fig, axes = resolve_axes(
        axes_arg,
        n_axes=_n_panels,
        figsize=figsize_arg or _figsize,
        sharey=True,
        dpi=figure_dpi,
    )

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = _two_afc_state_color(lbl, k, K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _psych_state_panel(
                _ax_overlay,
                _df_state,
                ild_col,
                choice_col,
                pred_col=f"_pR_state_rank_{k}" if state_assignment_mode == "weighted" else pred_col,
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                show_subject_traces=False,
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                weight_col=_weight_col,
                tick_ilds=ild_ticks,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
            )
        _ax_overlay.axhline(0.5, color="gray", ls="--", alpha=0.5)
        _ax_overlay.axvline(0.0, color="gray", ls="--", alpha=0.5)
        _ax_overlay.set_ylim(0, 1)
        _ax_overlay.set_yticks([0, 0.5, 1])
        _ax_overlay.set_xlabel("Stimulus ILD (dB)")
        _ax_overlay.legend(frameon=False)

    if not overlay_only:
        for k, ax in enumerate(axes[int(_include_overlay) :]):
            lbl = slbls.get(k, f"State {k}")
            color = _two_afc_state_color(lbl, k, K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _psych_state_panel(
                ax,
                _df_state,
                ild_col,
                choice_col,
                pred_col=f"_pR_state_rank_{k}" if state_assignment_mode == "weighted" else pred_col,
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                weight_col=_weight_col,
                tick_ilds=ild_ticks,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
            )
            ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("ILD (dB)")
            if k == 0:
                ax.set_ylabel("P(Right)")
            else:
                ax.set_ylabel("")

    # fig.suptitle(model_name, y=1.02)
    fig.tight_layout()
    for ax in axes[:_n_panels]:
        apply_axis_style(ax, **style)
    return fig, axes[:_n_panels]


# Alias used by the analysis notebooks
def plot_categorical_performance_state_overlay(
    df,
    views: dict,
    model_name: str,
    ild_col: str = "ILD",
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
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Figure:
    """Single-panel state-overlay psychometric."""
    return plot_categorical_performance_all_by_state(
        df=df,
        views=views,
        model_name=model_name,
        ild_col=ild_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        X_cols=X_cols,
        ild_max=ild_max,
        background_style=background_style,
        show_weighted_points=show_weighted_points,
        show_data_smooth=show_data_smooth,
        show_model_smooth=show_model_smooth,
        figure_dpi=figure_dpi,
        overlay_only=True,
        model_line_mode=model_line_mode,
        state_assignment_mode=state_assignment_mode,
        ax=ax,
        **plot_kwargs,
    )


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
    **plot_kwargs,
) -> plt.Figure:
    """Per-state partial-dependence plot for any emission regressor.

    The x-axis is the chosen regressor (for example ``at_choice``) instead of
    ILD. Empirical points are pooled within quantile bins of that regressor,
    while the model line sweeps the same regressor over a dense grid and
    marginalises over the empirical distribution of the remaining features.
    """
    style = dict(plot_kwargs)
    axes_arg = style.pop("axes", None)
    figsize_arg = style.pop("figsize", None)
    ax_arg = style.pop("ax", None)
    if ax_arg is not None:
        axes_arg = [ax_arg]
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
        fig, ax = resolve_single_axis(
            ax=np.asarray(axes_arg, dtype=object).ravel()[0] if axes_arg is not None else None,
            figsize=figsize_arg or (3.0, 3.0),
        )
        ax.text(0.5, 0.5, f"No valid {feature_col} data", ha="center", va="center")
        ax.axis("off")
        apply_axis_style(ax, **style)
        return ax

    if "state_idx" not in df_pd.columns and "state_rank" not in df_pd.columns and "_state_k" not in df_pd.columns:
        raise ValueError("df must contain a 'state_idx' or 'state_rank' column (output of build_trial_df)")
    df_pd = _attach_two_afc_ranked_cols(df_pd, views, subj_col=subj_col, base_col="pR_state")
    if "_state_k" not in df_pd.columns:
        if "state_rank" in df_pd.columns:
            df_pd["_state_k"] = pd.to_numeric(df_pd["state_rank"], errors="coerce")
        else:
            raise ValueError("Could not derive local state ranks from trial_df and views.")
    df_pd["_state_k"] = pd.to_numeric(df_pd["_state_k"], errors="coerce")
    if state_assignment_mode == "weighted":
        K = next(iter(views.values())).K if views else int(df_pd["_state_k"].max()) + 1
        missing_cols = [
            col
            for k in range(K)
            for col in (f"_p_state_rank_{k}", f"_pR_state_rank_{k}")
            if col not in df_pd.columns
        ]
        if missing_cols:
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

    slbls = _two_afc_ranked_state_labels(views)
    _as = _two_afc_rank_ordered_arrays_store(views)
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

    _include_overlay = bool(overlay_only)
    _n_panels = K + int(_include_overlay)
    if overlay_only:
        _n_panels = 1
    _figsize = (3, 3) if overlay_only else (4 * _n_panels, 4)
    fig, axes = resolve_axes(
        axes_arg,
        n_axes=_n_panels,
        figsize=figsize_arg or _figsize,
        sharey=True,
        dpi=figure_dpi,
    )

    xlabel = _feature_label(feature_col)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = _two_afc_state_color(lbl, k, K)
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
        _ax_overlay.legend(frameon=False, fontsize=8)

    if not overlay_only:
        for k, ax in enumerate(axes[int(_include_overlay) :]):
            lbl = slbls.get(k, f"State {k}")
            color = _two_afc_state_color(lbl, k, K)
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
    fig.tight_layout()
    for ax in axes[:_n_panels]:
        apply_axis_style(ax, **style)
    return fig, axes[:_n_panels]


from src.process.common import (
    attach_repeat_choice_evidence,
    attach_total_fitted_evidence,
    attach_rank_posterior_cols,
    attach_rank_state_model_cols,
    binned_feature_summary,
    display_regressor_name,
    fit_lapse_logistic_by_group,
    fit_lapse_logistic_by_subject_group,
    format_lapse_logistic_fits,
    lapse_logistic_label,
    mean_weighted_empirical_curve,
    prepare_evidence_curve,
    add_choice_lag_summary_regressor,
    pick_choice_history_regressor as _pick_choice_history_regressor,
    prepare_psych_panel_payload,
    prepare_psych_state_panel_payload,
    prepare_regressor_state_panel_payload,
    prepare_right_integration_maps,
    quantile_bin_spec,
    rank_ordered_arrays_store,
    REPEAT_EVIDENCE_TAIL_QUANTILES,
    ranked_state_labels,
    resolve_ild_max,
)
from src.process import two_afc as process
from src.plots.common import (
    add_shared_figure_legend,
    apply_axis_style,
    centered_numeric_group_palette,
    fit_lapse_logistic_for_panel,
    make_single_panel_figure,
    prepare_binned_accuracy_total_evidence_panels,
    plot_lapse_fit_parameter_panels,
    plot_grouped_summary,
    plot_mean_over_data,
    plot_integration_map_panels,
    plot_simple_summary,
    plot_repeat_by_regressor_simple as _plot_repeat_by_regressor_simple,
    resolve_axes,
    resolve_single_axis,
)

pick_choice_history_regressor = _pick_choice_history_regressor
_resolve_ild_max = resolve_ild_max
_mean_glm_curve = process.mean_glm_ild_curve
_subject_glm_curves = process.subject_glm_ild_curves


def _attach_weighted_stimulus_evidence(
    plot_df,
    *,
    views: dict | None,
    output_col: str = "_weighted_stimulus_evidence",
) -> pd.DataFrame | None:
    df_pd = to_pandas_df(plot_df).copy()
    if "subject" not in df_pd.columns:
        return None

    view_by_subject = {str(subject): view for subject, view in (views or {}).items()}
    out = pd.Series(np.nan, index=df_pd.index, dtype=float)
    computed_any = False

    for subject, idx in df_pd.groupby("subject", observed=True).groups.items():
        view = view_by_subject.get(str(subject))
        if view is None:
            continue
        feat_names = [str(name) for name in (getattr(view, "feat_names", []) or [])]
        weights = np.asarray(getattr(view, "emission_weights", []), dtype=float)
        if weights.ndim != 3 or weights.shape[0] < 1 or weights.shape[1] < 1:
            continue

        weighted = np.zeros(len(idx), dtype=float)
        used_any = False
        for feat_idx, feat_name in enumerate(feat_names):
            if not (feat_name.startswith("stim_") and feat_name.removeprefix("stim_").isdigit()):
                continue
            if feat_name not in df_pd.columns or feat_idx >= weights.shape[2]:
                continue
            x = pd.to_numeric(df_pd.loc[idx, feat_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            weighted += x * float(weights[0, 0, feat_idx])
            used_any = True
        if used_any:
            out.loc[idx] = weighted
            computed_any = True

    if "stim_param" in df_pd.columns:
        fallback = pd.to_numeric(df_pd["stim_param"], errors="coerce")
        out = out.fillna(fallback) if computed_any else fallback
    elif not computed_any:
        return None

    df_pd[output_col] = out
    if "ILD" in df_pd.columns:
        ild = pd.to_numeric(df_pd["ILD"], errors="coerce")
        pooled = (
            pd.DataFrame({"ILD": ild, output_col: df_pd[output_col]})
            .dropna(subset=["ILD", output_col])
            .groupby("ILD", observed=True)[output_col]
            .median()
        )
        if not pooled.empty:
            df_pd[output_col] = ild.map(pooled)
    return df_pd
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

def plot_accuracy(plot_df, ax=None, figsize=(3.0, 3.0), color=None, title="2AFC", label=None):
    df_pd = (
        plot_df.to_pandas().copy()
        if hasattr(plot_df, "to_pandas")
        else pd.DataFrame(plot_df).copy()
    )

    df_pd["abs_ILD"] = pd.to_numeric(df_pd["ILD"], errors="coerce").abs()
    x_col = "abs_ILD"
    xlabel = "|ILD| (dB)"

    return plot_mean_over_data(
        df_pd,
        x_col=x_col,
        # x_order=[20, 8, 4, 2, 0],
        invert_x=True,
        x_tick_labels={0: "0", 2: "2", 4: "4", 8: "8", 20: "20"},
        y_col="Hit",
        xlabel=xlabel,
        title=title,
        baseline=0.5,
        color=color if color is not None else "tab:blue",
        ax=ax,
        figsize=figsize,
        label=label,
    )

def plot_rb(
    plot_df,
    ax=None,
    figsize=(3.0, 3.0),
    title="2AFC",
    color=None,
    show_baseline_ttest=False,
    label=None,
):
    df_pd = to_pandas_df(plot_df).copy()
    df_pd["abs_ILD"] = pd.to_numeric(df_pd["ILD"], errors="coerce").abs()

    rb_df = compute_rb_by_x(df_pd, "abs_ILD", "Choice")

    return plot_mean_over_data(
        rb_df,
        x_col="abs_ILD",
        y_col="rb",
        invert_x=True,
        x_tick_labels={0: "0", 2: "2", 4: "4", 8: "8", 20: "20"},
        xlabel="|ILD| (dB)",
        ylabel="Rep. bias",
        title=title,
        baseline=0.5,
        baseline_area=False,
        color=color if color is not None else "tab:blue",
        show_baseline_ttest=show_baseline_ttest,
        ax=ax,
        figsize=figsize,
        label=label,
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
    style = {}
    if xlabel is not None:
        style["xlabel"] = xlabel
    return plot_simple_summary(summary, meta=meta, **style)


def plot_binned_accuracy_figure(
    plot_df,
    *,
    regressor_col: str,
    x_axis: str | None = None,
    adapter=None,
    views: dict | None = None,
    figsize: tuple[float, float] | None = None,
    max_panels: int | None = None,
    legend: bool = True,
    fit_lapse_logistic: bool = False,
    show_lapses_in_legend: bool = True,
    print_lapse_fits: bool | None = None,
    lapse_max: float = 0.4,
    share_lapse_logistic_core: bool = False,
    fit_lapse_by_subject: bool = True,
    n_bins: int = 4,
    ax: plt.Axes | None = None,
    legend_ax: plt.Axes | None = None,
    **plot_kwargs,
):
    style = dict(plot_kwargs)
    axes_arg = style.pop("axes", None)
    figsize_arg = style.pop("figsize", None)
    if ax is not None:
        axes_arg = [ax]
    x_axis_key = str(x_axis).lower() if x_axis is not None else None
    if x_axis_key in {"total_evidence", "total evidence", "fitted_total_evidence", "evidence"}:
        panels, legend_title = prepare_binned_accuracy_total_evidence_panels(
            plot_df,
            regressor_col=regressor_col,
            adapter=adapter,
            views=views,
            is_mcdr=False,
            baseline=process.BASELINE,
            n_bins=int(n_bins),
        )
    elif x_axis_key in {
        "weighted_stimulus",
        "weighted stimulus",
        "weighted_stimulus_evidence",
        "weighted stimulus evidence",
        "stim_one_hot_weight",
        "stim one-hot weight",
    }:
        _weighted_df = _attach_weighted_stimulus_evidence(plot_df, views=views)
        if _weighted_df is None:
            panels, legend_title = None, None
        else:
            panels, legend_title = process.prepare_binned_accuracy_figure(
                _weighted_df,
                regressor_col=regressor_col,
                x_col="_weighted_stimulus_evidence",
                xlabel="Weighted stimulus evidence",
                n_bins=int(n_bins),
            )
    else:
        panels, legend_title = process.prepare_binned_accuracy_figure(
            plot_df,
            regressor_col=regressor_col,
            n_bins=int(n_bins),
        )
        if panels and x_axis_key in {"ild", "native"}:
            panels = panels[:1]
    if not panels:
        return None
    if max_panels is not None:
        panels = panels[:max_panels]

    extra_fit_axes = 2 if fit_lapse_logistic else 0
    resolved_figsize = (
        figsize_arg
        if figsize_arg is not None
        else (figsize if figsize is not None else (4 * (len(panels) + extra_fit_axes), 4))
    )
    n_axes = len(panels) + extra_fit_axes
    if ax is not None and n_axes != 1:
        raise ValueError("ax can only be used when plot_binned_accuracy_figure resolves to one axis.")
    if extra_fit_axes:
        fig, axes_grid, _ = resolve_axes_grid(
            axes=axes_arg,
            n_panels=n_axes,
            nrows=1,
            ncols=n_axes,
            figsize=resolved_figsize,
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        axes = axes_grid.ravel()
    else:
        fig, axes = resolve_axes(
            axes_arg,
            n_axes=n_axes,
            figsize=resolved_figsize,
            sharey=True,
        )
    panel_axes = axes[: len(panels)]
    diagnostic_axes = axes[len(panels) : len(panels) + extra_fit_axes]

    lapse_fit_reports: list[str] = []
    diagnostic_fits = {}
    diagnostic_meta = None
    diagnostic_line_order = None
    for ax, panel in zip(panel_axes, panels, strict=False):
        summary = panel["summary"]
        meta = panel["meta"]
        x_plot_col = meta.get("x_col", "ILD")
        fits = {}
        label_map = None
        if fit_lapse_logistic and summary is not None and not summary.empty:
            line_order = meta.get("line_order") or summary["_reg_bin"].dropna().unique().tolist()
            fits = fit_lapse_logistic_for_panel(
                panel,
                fit_lapse_by_subject=fit_lapse_by_subject,
                lapse_max=lapse_max,
                share_lapse_logistic_core=share_lapse_logistic_core,
                default_x_col="ILD",
            )
            if fits and show_lapses_in_legend:
                label_map = {
                    group_value: lapse_logistic_label(group_value, fits.get(group_value))
                    for group_value in line_order
                }
            elif fits and (print_lapse_fits is None or print_lapse_fits):
                title = panel.get("label") or meta.get("title") or "Binned psychometric lapse fits"
                lapse_fit_reports.append(format_lapse_logistic_fits(fits, title=title))
            if fits and not diagnostic_fits:
                diagnostic_fits = fits
                diagnostic_meta = meta
                diagnostic_line_order = line_order

        plot_grouped_summary(
            ax,
            summary,
            line_group_col="_reg_bin",
            x_col=x_plot_col,
            meta=meta,
            label_map=label_map,
            legend=legend,
        )
        if fits:
            line_order = meta.get("line_order") or list(fits)
            default_palette = sns.color_palette("viridis", len(line_order))
            for group_value, color in zip(line_order, default_palette, strict=False):
                fit = fits.get(group_value)
                if fit is None:
                    continue
                ax.plot(
                    fit.x_fit,
                    fit.y_fit,
                    "--",
                    color=color,
                    lw=1.5,
                    alpha=0.9,
                    label="_nolegend_",
                )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    if lapse_fit_reports:
        print("\n\n".join(report for report in lapse_fit_reports if report))
    if fit_lapse_logistic:
        plot_lapse_fit_parameter_panels(
            diagnostic_axes,
            diagnostic_fits,
            line_order=diagnostic_line_order,
            meta=diagnostic_meta or {},
            regressor_label=display_regressor_name(regressor_col),
        )
    add_shared_figure_legend(
        fig,
        source_ax=panel_axes[-1],
        title=legend_title,
        legend_ax=legend_ax,
        legend=legend,
    )
    _reserve_legend_space = legend and legend_ax is None
    # fig.tight_layout(rect=(0.0, 0.0, 0.92 if _reserve_legend_space else 1.0, 1.0))
    for ax in panel_axes:
        apply_axis_style(ax, **style)
    return fig, axes[: len(panels) + extra_fit_axes]


def plot_repeat_by_regressor_simple(
    plot_df,
    *,
    regressor_col: str,
    views: dict,
    title: str | None = None,
    xlabel: str | None = None,
    n_bins: int = 10,
    legend: bool = True,
    **plot_kwargs,
):
    _ = title
    return _plot_repeat_by_regressor_simple(
        plot_df,
        regressor_col=regressor_col,
        views=views,
        is_mcdr=False,
        baseline=process.BASELINE,
        xlabel=xlabel,
        n_bins=n_bins,
        legend=legend,
        **plot_kwargs,
    )


def plot_right_by_regressor(
    plot_df,
    *,
    regressor_col: str,
    title: str | None = None,
    xlabel: str | None = None,
    n_bins: int = 10,
    group_col: str | None = None,
    group_order: Sequence | None = None,
    group_labels: dict | None = None,
    palette: dict | None = None,
    legend: bool = True,
    legend_ax: plt.Axes | None = None,
    **plot_kwargs,
):
    summary, meta = process.prepare_right_by_regressor(
        plot_df,
        regressor_col=regressor_col,
        xlabel=xlabel,
        n_bins=n_bins,
        group_col=group_col,
        group_order=group_order,
    )
    if summary is None or summary.empty:
        return None

    line_group_col = meta.get("line_group_col", "ILD")
    line_order = meta.get("line_order") or []
    resolved_palette = palette if palette is not None else (
        centered_numeric_group_palette(line_order) if group_col is None else None
    )

    fig, ax = make_single_panel_figure(
        extra_right_legend=True,
        ax=plot_kwargs.get("ax"),
        figsize=plot_kwargs.get("figsize", (3.0, 3.0)),
    )
    plot_grouped_summary(
        ax,
        summary,
        line_group_col=line_group_col,
        x_col="x_center",
        meta=meta,
        label_map=group_labels,
        palette=resolved_palette,
        legend=False if legend_ax is not None else legend,
    )
    if legend_ax is not None:
        legend_ax.axis("off")
        add_shared_figure_legend(
            fig,
            source_ax=ax,
            title=meta.get("legend_title"),
            legend_ax=legend_ax,
            legend=legend,
        )
    apply_axis_style(ax, **({"xlabel": xlabel} if xlabel is not None else {}))
    return ax


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
    panel: str | None = None,
    ax: plt.Axes | None = None,
    **plot_kwargs,
):
    _n_bins = n_bins
    panels, meta = prepare_right_integration_maps(
        plot_df,
        response_mode=process.RESPONSE_MODE,
        pred_col=process.PRED_COL,
        x_col=x_col,
        y_col=y_col,
        value_col=value_col,
        include_model=include_model,
        bnd=bnd,
        dx=dx,
        n_bins=_n_bins,
        sigma=sigma,
        fill_empty=smooth,
        default_sigma_dx=5.0,
    )
    if panel is not None:
        panel_label = str(panel).casefold()
        panels = [
            panel_data
            for panel_data in panels
            if str(panel_data.get("label", "")).casefold() == panel_label
        ]
        if not panels:
            return None
    axes = [ax] if ax is not None else None
    return plot_integration_map_panels(
        panels,
        meta=meta,
        axes=axes,
        interpolation=None,
        **plot_kwargs,
    )


def plot_accuracy_by_total_evidence(
    plot_df,
    *,
    adapter,
    views: dict,
    group_col: str | None = None,
    group_order: Sequence | None = None,
    group_labels: dict | None = None,
    palette: dict | None = None,
    legend: bool = True,
    **plot_kwargs,
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
        group_col=group_col,
        group_order=group_order,
    )
    if group_col is None:
        return plot_simple_summary(summary, meta=meta, **plot_kwargs)

    fig, ax = make_single_panel_figure(
        ax=plot_kwargs.get("ax"),
        figsize=plot_kwargs.get("figsize", (3.0, 3.0)),
    )
    return plot_grouped_summary(
        ax,
        summary,
        line_group_col=group_col,
        x_col="x_center",
        label_map=group_labels,
        palette=palette,
        meta=meta,
    )


def plot_repeat_by_repeat_evidence(
    plot_df,
    *,
    views: dict,
    group_col: str | None = None,
    group_order: Sequence | None = None,
    group_labels: dict | None = None,
    palette: dict | None = None,
    legend: bool = True,
    **plot_kwargs,
):
    style = dict(plot_kwargs)
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
        group_col=group_col,
        group_order=group_order,
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

    if group_col is None:
        ax = plot_simple_summary(summary, meta=meta, legend=legend, **style)
    else:
        fig, ax = make_single_panel_figure(
            ax=style.get("ax"),
            figsize=style.get("figsize", (3.0, 3.0)),
        )
        ax = plot_grouped_summary(
            ax,
            summary,
            line_group_col=group_col,
            x_col="x_center",
            label_map=group_labels,
            palette=palette,
            meta=meta,
            legend=legend,
        )
    if ax is not None and x_dense is not None and model_dense is not None:
        ax.plot(x_dense, model_dense, color="black", linewidth=1.0, linestyle=(0, (3, 1)), alpha=0.9, zorder=1)
    return ax


__all__ = [
    "display_regressor_name",
    "pick_choice_history_regressor",
    "plot_accuracy_by_stimulus",
    "plot_accuracy_by_total_evidence",
    "plot_binned_accuracy_figure",
    "plot_repeat_by_repeat_evidence",
    "plot_repeat_by_regressor_simple",
    "plot_right_integration_map",
    "plot_right_by_regressor",
    "plot_right_by_regressor_simple",
]
