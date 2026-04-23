from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd

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


def make_single_panel_figure(*, extra_right_legend: bool = False, figsize=(3.0, 3.0)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def plot_empirical_accuracy_curve(
    df_like,
    *,
    x_col: str,
    accuracy_col: str,
    subject_col: str = "subject",
    x_order: list | None = None,
    x_tick_labels: list | dict | None = None,
    xlabel: str,
    title: str,
    baseline: float,
    color: str = "#2b7bba",
    invert_x: bool = False,
    ax=None,
    figsize=(3.0, 3.0),
):
    if isinstance(df_like, pd.DataFrame):
        df = df_like.copy()
    elif hasattr(df_like, "to_pandas"):
        df = df_like.to_pandas().copy()
    else:
        df = pd.DataFrame(df_like).copy()

    if x_col not in df.columns:
        raise ValueError(f"Missing x column {x_col!r}.")
    if accuracy_col not in df.columns:
        raise ValueError(f"Missing accuracy column {accuracy_col!r}.")

    df["_accuracy"] = pd.to_numeric(df[accuracy_col], errors="coerce")
    df = df[df[x_col].notna() & df["_accuracy"].notna()].copy()
    if df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No valid accuracy data", ha="center", va="center")
        ax.axis("off")
        return fig

    if subject_col in df.columns:
        subject_summary = (
            df.groupby([subject_col, x_col], observed=True)["_accuracy"]
            .mean()
            .reset_index(name="subject_accuracy")
        )
        summary = (
            subject_summary.groupby(x_col, observed=True)["subject_accuracy"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
    else:
        summary = (
            df.groupby(x_col, observed=True)["_accuracy"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )

    if x_order is not None:
        summary = summary[summary[x_col].isin(x_order)].copy()
        if summary.empty:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            else:
                fig = ax.figure
            ax.text(0.5, 0.5, "No valid accuracy data", ha="center", va="center")
            ax.axis("off")
            return fig
        summary[x_col] = pd.Categorical(summary[x_col], categories=x_order, ordered=True)
        summary = summary.sort_values(x_col)
        x = np.arange(len(summary), dtype=float)
        if x_tick_labels is None:
            tick_labels = [str(value) for value in summary[x_col]]
        elif isinstance(x_tick_labels, dict):
            tick_labels = [x_tick_labels.get(value, str(value)) for value in summary[x_col]]
        else:
            label_map = dict(zip(x_order, x_tick_labels, strict=False))
            tick_labels = [label_map.get(value, str(value)) for value in summary[x_col]]
    else:
        summary["_x_numeric"] = pd.to_numeric(summary[x_col], errors="coerce")
        summary = summary.dropna(subset=["_x_numeric"]).sort_values("_x_numeric")
        if summary.empty:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            else:
                fig = ax.figure
            ax.text(0.5, 0.5, "No valid accuracy data", ha="center", va="center")
            ax.axis("off")
            return fig
        x = summary["_x_numeric"].to_numpy(dtype=float)
        tick_labels = []
        for val in x:
            if isinstance(x_tick_labels, dict) and val in x_tick_labels:
                tick_labels.append(x_tick_labels[val])
            elif isinstance(x_tick_labels, dict) and int(val) in x_tick_labels:
                tick_labels.append(x_tick_labels[int(val)])
            elif np.isclose(val, 0.1):
                tick_labels.append("0")
            else:
                tick_labels.append(f"{val:g}")

    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    ax.errorbar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["sem"].to_numpy(dtype=float),
        fmt="o-",
        color=color,
        ecolor=color,
        # elinewidth=1.0,
        # linewidth=2.0,
        # markersize=4,
        capsize=0,
    )
    
    ax.axhline(baseline, color="gray", ls="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, baseline, 1.0])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:.2f}".rstrip("0").rstrip("."))
    )
    ax.set_xticks(x, labels=tick_labels)
    ax.axhspan(0.0, baseline, color="gray", alpha=0.1, zorder=0)
    if invert_x:
        ax.invert_xaxis()
    
    sns.despine(ax=ax)
    return fig


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


def plot_integration_map_panels(
    panels: list[dict],
    *,
    meta: dict,
    contour_levels: tuple[float, ...] = (0.15, 0.3, 0.5, 0.7, 0.85),
    colours=None,
    cmap: str | None = None,
    interpolation: str | None = None,
    data_points_cutoff: float = 20.0,
):
    if not panels:
        return None

    _ = cmap, interpolation
    if colours is None:
        colours = np.array([[103, 169, 221], [241, 135, 34]], dtype=float) / 255.0
    else:
        colours = np.asarray(colours, dtype=float)
        if colours.max(initial=0.0) > 1.0:
            colours = colours / 255.0
    if colours.shape != (2, 3):
        raise ValueError("colours must be a 2-by-3 RGB array.")

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.4 * n_panels, 4.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels, strict=False):
        z = np.asarray(panel["map"], dtype=float)
        z_for_colour = np.nan_to_num(z, nan=0.0)
        rgb = colours[0] + z_for_colour[..., None] * (colours[1] - colours[0])
        intensity = np.minimum(
            np.nan_to_num(np.asarray(panel["n_datapoints"], dtype=float), nan=0.0)
            / float(data_points_cutoff),
            1.0,
        )
        rgb = 1.0 - (1.0 - rgb) * intensity[..., None]

        x_centers = np.asarray(panel["x_centers"], dtype=float)
        y_centers = np.asarray(panel["y_centers"], dtype=float)
        if x_centers.size == 0 or y_centers.size == 0:
            ax.set_axis_off()
            continue

        if x_centers.size == 1:
            x_extent = (float(panel["x_edges"][0]), float(panel["x_edges"][-1]))
        else:
            x_extent = (float(x_centers[0]), float(x_centers[-1]))
        if y_centers.size == 1:
            y_extent = (float(panel["y_edges"][0]), float(panel["y_edges"][-1]))
        else:
            y_extent = (float(y_centers[0]), float(y_centers[-1]))

        ax.imshow(
            np.transpose(rgb, (1, 0, 2)),
            extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        z_for_contour = np.nan_to_num(z, nan=0.0)
        finite = np.isfinite(z_for_contour)
        if finite.any():
            lo = float(np.nanmin(z_for_contour))
            hi = float(np.nanmax(z_for_contour))
            levels = list(contour_levels)
            mid_idx = int(round((len(levels) + 1) / 2.0)) - 1
            thick_level = levels[mid_idx] if 0 <= mid_idx < len(levels) else None
            thin_levels = [
                level
                for idx, level in enumerate(levels)
                if idx != mid_idx and lo < level < hi
            ]
            if thin_levels:
                ax.contour(
                    x_centers,
                    y_centers,
                    z_for_contour.T,
                    levels=thin_levels,
                    colors="black",
                    linewidths=0.5,
                )
            if thick_level is not None and lo < thick_level < hi:
                ax.contour(
                    x_centers,
                    y_centers,
                    z_for_contour.T,
                    levels=[thick_level],
                    colors="black",
                    linewidths=1.0,
                )
        ax.set_title(panel["label"])
        ax.set_xlabel(meta["xlabel"])
        if meta.get("xticks") is not None:
            ax.set_xticks(meta["xticks"], labels=meta.get("x_tick_labels"))
        ax.set_box_aspect(1)

    axes[0].set_ylabel(meta["ylabel"])
    return fig


__all__ = [
    "custom_boxplot",
    "plot_integration_map_panels",
    "plot_transition_matrix",
    "plot_transition_matrix_by_subject",
    "plot_weights_boxplot",
]
