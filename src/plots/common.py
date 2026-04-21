from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def make_single_panel_figure(*, extra_right_legend: bool = False):
    fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
    ax.set_box_aspect(1)
    return fig, ax


def add_shared_figure_legend(
    fig,
    *,
    source_ax,
    title: str,
    bbox_x: float = 0.94,
) -> None:
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return
    legend = fig.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(bbox_x, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        labelspacing=0.35,
        handlelength=2.0,
    )


def centered_numeric_group_palette(group_order: list) -> dict:
    numeric_order = [float(val) for val in group_order]
    negatives = [val for val in numeric_order if val < 0]
    positives = [val for val in numeric_order if val > 0]
    has_zero = any(np.isclose(val, 0.0) for val in numeric_order)

    palette = {}
    if negatives:
        neg_colors = list(
            reversed(sns.color_palette("Blues", len(negatives) + 2)[1:-1])
        )
        for value, color in zip(sorted(negatives), neg_colors, strict=False):
            palette[value] = color
    if has_zero:
        palette[0.0] = (0.45, 0.45, 0.45)
    if positives:
        pos_colors = sns.color_palette("Reds", len(positives) + 2)[1:-1]
        for value, color in zip(sorted(positives), pos_colors, strict=False):
            palette[value] = color
    return palette


def _apply_axis_style(ax, *, meta, title: str | None = None):
    if title:
        ax.set_title(title)
    ax.set_xlabel(meta["xlabel"])
    ax.set_ylabel(meta["ylabel"])
    ax.set_ylim(0.0, 1.0)

    if meta["baseline"] is not None:
        ax.axhline(meta["baseline"], color="gray", lw=0.8, ls="--", alpha=0.5)

    sns.despine(ax=ax)


def plot_simple_summary(summary_df, *, meta, title: str | None = None):
    if summary_df is None or summary_df.empty:
        return None

    fig, ax = make_single_panel_figure()
    x = summary_df["x_center"].to_numpy(dtype=float)
    model_mean = summary_df["model_mean"].to_numpy(dtype=float)
    model_sem = summary_df["model_sem"].to_numpy(dtype=float)

    ax.plot(
        x,
        model_mean,
        color="black",
        linewidth=2.0,
        label="Model",
        zorder=3,
    )
    ax.fill_between(
        x,
        np.clip(model_mean - model_sem, 0.0, 1.0),
        np.clip(model_mean + model_sem, 0.0, 1.0),
        color="black",
        alpha=0.12,
        linewidth=0.0,
        zorder=2,
    )
    ax.errorbar(
        x,
        summary_df["data_mean"].to_numpy(dtype=float),
        yerr=summary_df["data_sem"].to_numpy(dtype=float),
        fmt="o",
        color="#2b7bba",
        ecolor="#2b7bba",
        elinewidth=1.0,
        capsize=3,
        label="Data",
        zorder=4,
    )
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    _apply_axis_style(ax, meta=meta, title=title)
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_grouped_summary(
    ax,
    summary_df,
    *,
    line_group_col: str,
    x_col: str,
    meta,
    label_map: dict | None = None,
    palette: dict | None = None,
):
    if summary_df is None or summary_df.empty:
        ax.set_axis_off()
        return

    line_order = meta.get("line_order") or list(
        summary_df[line_group_col].dropna().unique()
    )
    default_palette = sns.color_palette("viridis", len(line_order))

    for group_value, default_color in zip(line_order, default_palette, strict=False):
        sub = summary_df[summary_df[line_group_col] == group_value].copy()
        if sub.empty:
            continue

        color = (
            palette.get(group_value, default_color)
            if palette is not None
            else default_color
        )
        label = (
            label_map.get(group_value, group_value)
            if label_map is not None
            else group_value
        )

        if meta.get("categorical_x", False):
            xpos = np.arange(len(sub), dtype=float)
        else:
            xpos = sub[x_col].to_numpy(dtype=float)

        ax.plot(
            xpos,
            sub["mm"].to_numpy(dtype=float),
            "-",
            color=color,
            lw=2.0,
            label=str(label),
        )
        ax.errorbar(
            xpos,
            sub["md"].to_numpy(dtype=float),
            yerr=sub["sem"].to_numpy(dtype=float),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            ms=5,
            capsize=3,
            zorder=5,
        )

    if meta.get("categorical_x", False) and meta.get("x_order") is not None:
        ax.set_xticks(np.arange(len(meta["x_order"]), dtype=float))
        ax.set_xticklabels(meta["x_tick_labels"] or meta["x_order"])
    elif meta.get("xticks") is not None:
        ax.set_xticks(meta["xticks"], labels=meta.get("x_tick_labels"))

    _apply_axis_style(ax, meta=meta)

    legend_kwargs = {
        "title": meta.get("legend_title"),
        "frameon": False,
        "fontsize": 8,
    }
    if meta.get("legend_outside", False):
        legend_kwargs.update(
            {
                "loc": "upper left",
                "bbox_to_anchor": (1.0, 1.0),
                "borderaxespad": 0.0,
                "title_fontsize": 9,
                "labelspacing": 0.35,
                "handlelength": 2.0,
            }
        )
    ax.legend(**legend_kwargs)


__all__ = [
    "custom_boxplot",
    "plot_transition_matrix",
    "plot_transition_matrix_by_subject",
    "plot_weights_boxplot",
]
