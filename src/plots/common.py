from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from src.process.common import (
    PreparedWeightFamilyPlot,
    attach_quantile_bin_column,
    display_regressor_name,
    to_pandas_df,
)


def apply_axis_style(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    title: str | None = None,
    grid: bool = False,
):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if title is not None:
        ax.set_title(title)

    if grid:
        ax.grid(True)


def fig_size(n_cols=1, ratio=None):
    """
    Get figure size for A4 page with n_cols columns and specified ratio (width/height).
    :param n_cols: Number of columns (0 for full page)
    :param ratio: Width/height ratio (None for default)
    :return:
    """

    if ratio is None:
        default_figsize = np.array(plt.rcParams["figure.figsize"])
        default_ratio = default_figsize[0] / default_figsize[1]
        ratio = default_ratio  # 4:3

    # All measurements are in inches
    A4_size = np.array((8.27, 11.69))  # A4 measurements
    margins = 2  # On both dimension
    size = A4_size - margins  # Effective size after margins removal (2 per dimension)
    width = size[0]
    height = size[1]

    # Full page (minus margins)
    if n_cols == 0:
        # Full A4 minus margins
        figsize = (width, height)
        if ratio == 1:  # Square
            figsize = (size[0], size[0])
        return figsize

    else:
        fig_width = width / n_cols
        fig_height = fig_width / ratio
        figsize = (fig_width, fig_height)
        return figsize


def plot_prepared_weight_family(
    prepared: PreparedWeightFamilyPlot | None,
):
    if prepared is None:
        return None

    df = to_pandas_df(prepared.data)
    if df.empty:
        return None

    required = {"subject", "x_label", "weight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Prepared weight family data must contain 'subject', 'x_label', and 'weight'. "
            f"Missing: {sorted(missing)}."
        )

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["x_label"] = df["x_label"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])
    if df.empty:
        return None

    x_order = list(prepared.x_order) if prepared.x_order is not None else pd.unique(df["x_label"]).tolist()
    df = df[df["x_label"].isin(x_order)].copy()
    if df.empty:
        return None

    if prepared.plot_kind == "line":
        summary = (
            df.groupby("x_label", as_index=False, observed=False)["weight"]
            .mean()
        )
        summary["x_label"] = pd.Categorical(summary["x_label"], categories=x_order, ordered=True)
        summary = summary.sort_values("x_label")
        if summary.empty:
            return None

        positions = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(summary)), 4.0))
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(prepared.title)
        ax.set_xlabel(prepared.xlabel)
        ax.set_ylabel(prepared.ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(summary["x_label"].astype(str).tolist())
        fig.tight_layout()
        return fig

    subject_order = pd.unique(df["subject"]).tolist()
    per_feature_values: list[np.ndarray] = []
    subject_lines = np.full((len(subject_order), len(x_order)), np.nan, dtype=float)

    for feature_idx, x_label in enumerate(x_order):
        feature_df = df[df["x_label"] == x_label].copy()
        if feature_df.empty:
            per_feature_values.append(np.asarray([], dtype=float))
            continue
        by_subject = (
            feature_df.groupby("subject", observed=False)["weight"]
            .mean()
            .reindex(subject_order)
        )
        subject_lines[:, feature_idx] = by_subject.to_numpy(dtype=float)
        per_feature_values.append(by_subject.dropna().to_numpy(dtype=float))

    if not any(values.size for values in per_feature_values):
        return None

    fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(x_order)), 4.0))
    positions = np.arange(1, len(x_order) + 1)
    ax.boxplot(
        per_feature_values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "#d9e8f6", "edgecolor": "#356b9a", "linewidth": 1.0},
        whiskerprops={"color": "#356b9a", "linewidth": 1.0},
        capprops={"color": "#356b9a", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
    )
    for row in subject_lines:
        if np.isfinite(row).sum() >= 2:
            ax.plot(positions, row, color="#777777", alpha=0.18, linewidth=0.8, zorder=0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(prepared.title)
    ax.set_xlabel(prepared.xlabel)
    ax.set_ylabel(prepared.ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(x_order))
    fig.tight_layout()
    return fig


def make_single_panel_figure(
    *,
    extra_right_legend: bool = False,
    figsize=(3.0, 3.0),
    ax: plt.Axes | None = None,
    **style,
):
    _ = extra_right_legend
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    apply_axis_style(ax, **style)
    return fig, ax


def resolve_single_axis(
    *,
    ax: plt.Axes | None = None,
    figsize=(3.0, 3.0),
    constrained_layout: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained_layout)
    else:
        fig = ax.figure
    return fig, ax


def resolve_axes(
    axes: Sequence[plt.Axes] | None = None,
    *,
    n_axes: int,
    figsize,
    squeeze: bool = False,
    **subplots_kwargs,
):
    if axes is None:
        fig, axes = plt.subplots(
            1,
            n_axes,
            figsize=figsize,
            squeeze=squeeze,
            **subplots_kwargs,
        )
        axes = np.atleast_1d(axes).ravel()
        return fig, axes

    axes = np.asarray(axes, dtype=object).ravel()
    if len(axes) < n_axes:
        raise ValueError(f"Expected at least {n_axes} axes, got {len(axes)}.")
    return axes[0].figure, axes


def plot_mean_over_data(
    df_like,
    *,
    x_col: str,
    y_col: str,
    subject_col: str = "subject",
    x_order: list | None = None,
    x_tick_labels: list | dict | None = None,
    xlabel: str,
    ylabel: str = "Accuracy",
    title: str,
    baseline: float,
    baseline_area: bool = True,
    color: str = "#2b7bba",
    invert_x: bool = False,
    ax: plt.Axes | None = None,
    figsize=(3.0, 3.0),
    **style,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    if isinstance(df_like, pd.DataFrame):
        df = df_like.copy()
    elif hasattr(df_like, "to_pandas"):
        df = df_like.to_pandas().copy()
    else:
        df = pd.DataFrame(df_like).copy()

    if x_col not in df.columns:
        raise ValueError(f"Missing x column {x_col!r}.")
    if y_col not in df.columns:
        raise ValueError(f"Missing y column {y_col!r}.")

    df["_y"] = pd.to_numeric(df[y_col], errors="coerce")
    df = df[df[x_col].notna() & df["_y"].notna()].copy()

    if df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        apply_axis_style(ax, **style)
        return ax

    if subject_col in df.columns:
        subject_summary = (
            df.groupby([subject_col, x_col], observed=True)["_y"]
            .mean()
            .reset_index(name="subject_mean")
        )
        summary = (
            subject_summary.groupby(x_col, observed=True)["subject_mean"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
    else:
        summary = (
            df.groupby(x_col, observed=True)["_y"]
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
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.axis("off")
            apply_axis_style(ax, **style)
            return ax
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
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.axis("off")
            apply_axis_style(ax, **style)
            return ax
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
    ax.errorbar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["sem"].to_numpy(dtype=float),
        fmt="o-",
        color=color,
        ecolor=color,
        capsize=0,
    )

    ax.axhline(baseline, color="gray", ls="--")
    if baseline_area:
        ax.axhspan(0.0, baseline, color="gray", alpha=0.1, zorder=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, baseline, 1.0])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:.2f}".rstrip("0").rstrip("."))
    )
    ax.set_xticks(x, labels=tick_labels)

    if invert_x:
        ax.invert_xaxis()

    return fig


def add_shared_figure_legend(
    fig,
    *,
    source_ax,
    title: str | None = None,
    bbox_x: float = 0.94,
    legend: bool = True,
) -> None:
    if not legend:
        return
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
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


def _axes_from_plot_result(result):
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, plt.Axes):
                return item.figure, np.asarray([item], dtype=object)
            if isinstance(item, (list, tuple, np.ndarray)):
                axes = [ax for ax in np.asarray(item, dtype=object).ravel() if isinstance(ax, plt.Axes)]
                if axes:
                    return axes[0].figure, np.asarray(axes, dtype=object)
            if isinstance(item, plt.Figure):
                return item, np.asarray(item.axes, dtype=object)
    if isinstance(result, plt.Axes):
        return result.figure, np.asarray([result], dtype=object)
    if isinstance(result, plt.Figure):
        return result, np.asarray(result.axes, dtype=object)
    raise TypeError("Could not resolve matplotlib axes from plot result.")


def _axis_artist_snapshot(axes) -> dict[int, dict[str, set]]:
    return {
        id(ax): {
            "lines": set(ax.lines),
            "collections": set(ax.collections),
            "patches": set(ax.patches),
        }
        for ax in np.asarray(axes, dtype=object).ravel()
        if isinstance(ax, plt.Axes)
    }


def _style_axis_artists(ax, *, before: dict[str, set] | None, style: dict) -> None:
    color = style.get("color")
    linestyle = style.get("linestyle")
    linewidth = style.get("linewidth")
    alpha = style.get("alpha")
    marker = style.get("marker")

    new_lines = list(ax.lines) if before is None else [artist for artist in ax.lines if artist not in before["lines"]]
    for line in new_lines:
        if color is not None:
            line.set_color(color)
            line.set_markerfacecolor(color)
            line.set_markeredgecolor(color)
        if linestyle is not None and line.get_linestyle() not in {"None", "", " "}:
            line.set_linestyle(linestyle)
        if linewidth is not None:
            line.set_linewidth(linewidth)
        if alpha is not None:
            line.set_alpha(alpha)
        if marker is not None and line.get_marker() not in {None, "None", "", " "}:
            line.set_marker(marker)

    new_collections = (
        list(ax.collections)
        if before is None
        else [artist for artist in ax.collections if artist not in before["collections"]]
    )
    for collection in new_collections:
        if color is not None:
            try:
                collection.set_edgecolor(color)
                collection.set_facecolor(color)
            except Exception:
                pass
        if alpha is not None:
            collection.set_alpha(alpha)

    new_patches = list(ax.patches) if before is None else [artist for artist in ax.patches if artist not in before["patches"]]
    for patch in new_patches:
        if color is not None:
            patch.set_edgecolor(color)
            patch.set_facecolor(color)
        if alpha is not None:
            patch.set_alpha(alpha)


def overlay_plot_by_group(
    plot_fn,
    df_like,
    *,
    group_col: str,
    group_order: list | None = None,
    group_labels: dict | None = None,
    group_styles: dict | None = None,
    use_default_colors: bool = True,
    plot_kwargs: dict | None = None,
    axes_kwarg: str = "axes",
    legend_title: str | None = None,
    legend_loc: str = "upper right",
):
    """Call an existing axes-aware plot once per group and overlay the result.

    This keeps task plots unchanged: the wrapper filters the dataframe, reuses
    the axes from the first call, and styles only the artists added by each
    subsequent call.
    """
    df = to_pandas_df(df_like)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}.")
    df = df[df[group_col].notna()].copy()
    if df.empty:
        return None, []

    if group_order is None:
        group_order = list(pd.unique(df[group_col]))
    if group_labels is None:
        group_labels = {}

    default_colors = sns.color_palette("tab10", n_colors=max(1, len(group_order)))
    default_styles = {}
    for idx, value in enumerate(group_order):
        style = {"linestyle": "-"}
        if use_default_colors:
            style["color"] = default_colors[idx]
        default_styles[value] = style
    if group_styles is not None:
        for value, style in group_styles.items():
            default_styles.setdefault(value, {}).update(style)

    fig = None
    axes = None
    base_kwargs = dict(plot_kwargs or {})

    for group_value in group_order:
        sub = df[df[group_col] == group_value].copy()
        if sub.empty:
            continue

        kwargs = dict(base_kwargs)
        before = None
        if axes is not None:
            kwargs[axes_kwarg] = axes[0] if axes_kwarg == "ax" else axes
            before = _axis_artist_snapshot(axes)

        result = plot_fn(sub, **kwargs)
        if result is None:
            continue
        fig, axes = _axes_from_plot_result(result)
        style = default_styles.get(group_value, {})

        for ax in np.asarray(axes, dtype=object).ravel():
            if not isinstance(ax, plt.Axes):
                continue
            _style_axis_artists(
                ax,
                before=None if before is None else before.get(id(ax)),
                style=style,
            )
            if ax.legend_ is not None:
                ax.legend_.remove()

    if fig is None or axes is None:
        return None, []

    handles = []
    for group_value in group_order:
        style = default_styles.get(group_value, {})
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color", "black"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 2.0),
                marker=style.get("marker", None),
                label=group_labels.get(group_value, str(group_value)),
            )
        )
    fig.legend(handles=handles, title=legend_title, loc=legend_loc, frameon=False)
    fig.tight_layout()
    return fig, axes


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


def _apply_summary_axis_style(ax, *, meta, **style):
    ax.set_xlabel(style.get("xlabel", meta["xlabel"]))
    ax.set_ylabel(style.get("ylabel", meta["ylabel"]))
    ax.set_ylim(0.0, 1.0)

    if meta["baseline"] is not None:
        ax.axhline(meta["baseline"], color="gray", lw=0.8, ls="--", alpha=0.5)

    apply_axis_style(ax, **style)


def plot_simple_summary(
    summary_df,
    *,
    meta,
    ax: plt.Axes | None = None,
    figsize=(3.0, 3.0),
    legend: bool = True,
    **style,
):
    if summary_df is None or summary_df.empty:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

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

    _apply_summary_axis_style(ax, meta=meta, **style)
    if legend:
        ax.legend(frameon=False, fontsize=8)
    elif ax.legend_ is not None:
        ax.legend_.remove()
    return ax


def plot_grouped_summary(
    ax,
    summary_df,
    *,
    line_group_col: str,
    x_col: str,
    meta,
    label_map: dict | None = None,
    palette: dict | None = None,
    legend: bool = True,
    **style,
):
    if summary_df is None or summary_df.empty:
        ax.set_axis_off()
        apply_axis_style(ax, **style)
        return ax

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
            x_order = meta.get("x_order")
            if x_order is not None:
                x_pos_map = {str(value): idx for idx, value in enumerate(x_order)}
                xpos = sub[x_col].astype(str).map(x_pos_map).to_numpy(dtype=float)
            else:
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

    _apply_summary_axis_style(ax, meta=meta, **style)

    legend_kwargs = {
        "title": meta.get("legend_title"),
        "frameon": False,
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
    if legend:
        ax.legend(**legend_kwargs)
    elif ax.legend_ is not None:
        ax.legend_.remove()
    return ax


def _summarize_regressor_by_regressor_magnitude(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    subject_col: str,
    n_bins: int,
    use_abs_x: bool,
) -> pd.DataFrame | None:
    required = {x_axis, y_axis, subject_col}
    if not required.issubset(df.columns):
        return None

    df = df.copy()
    df[x_axis] = pd.to_numeric(df[x_axis], errors="coerce")
    df[y_axis] = pd.to_numeric(df[y_axis], errors="coerce")
    df["_x_magnitude"] = np.abs(df[x_axis]) if use_abs_x else df[x_axis]
    df = df[np.isfinite(df["_x_magnitude"]) & np.isfinite(df[y_axis])].copy()
    if df.empty:
        return None

    df, centers = attach_quantile_bin_column(
        df,
        value_col="_x_magnitude",
        bin_col="_x_bin",
        max_bins=n_bins,
        center_col="x_center",
        center_agg="median",
    )
    if df is None or centers.empty:
        return None

    subject_summary = (
        df.groupby([subject_col, "_x_bin"], observed=True)
        .agg(y_mean=(y_axis, "mean"))
        .reset_index()
        .merge(centers[["_x_bin", "x_center"]], on="_x_bin", how="left")
    )
    if subject_summary.empty:
        return None

    summary = (
        subject_summary.groupby("_x_bin", observed=True)
        .agg(
            x_center=("x_center", "mean"),
            y_mean=("y_mean", "mean"),
            y_std=("y_mean", "std"),
            n_subjects=(subject_col, "count"),
        )
        .reset_index()
        .sort_values("x_center")
    )
    summary["y_sem"] = summary["y_std"].fillna(0.0) / np.sqrt(summary["n_subjects"].clip(lower=1))
    return summary


def plot_regressor_net_impact(
    plot_df,
    *,
    x_axis: str,
    y_axis: str,
    subject_col: str = "subject",
    n_bins: int = 6,
    use_abs_x: bool = True,
    ax: plt.Axes | None = None,
    axes: Sequence[plt.Axes] | None = None,
    figsize=(3.4, 3.0),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = "#2b7bba",
    **style,
):
    """Plot one regressor's value across bins of another regressor's magnitude."""
    if axes is not None:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) == 0:
            raise ValueError("Expected at least one axis in `axes`.")
        ax = axes[0]
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize)

    df = to_pandas_df(plot_df)
    summary = _summarize_regressor_by_regressor_magnitude(
        df,
        x_axis=x_axis,
        y_axis=y_axis,
        subject_col=subject_col,
        n_bins=n_bins,
        use_abs_x=use_abs_x,
    )
    if summary is None or summary.empty:
        ax.text(0.5, 0.5, "No valid regressor data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        apply_axis_style(ax, **style)
        return ax

    x = summary["x_center"].to_numpy(dtype=float)
    y_mean = summary["y_mean"].to_numpy(dtype=float)
    y_sem = summary["y_sem"].to_numpy(dtype=float)

    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.errorbar(
        x,
        y_mean,
        yerr=y_sem,
        fmt="o-",
        color=color,
        ecolor=color,
        elinewidth=1.0,
        linewidth=2.0,
        markersize=5,
        capsize=3,
        zorder=3,
    )

    x_label = display_regressor_name(x_axis)
    y_label = display_regressor_name(y_axis)
    ax.set_xlabel(xlabel or (f"|{x_label}|" if use_abs_x else x_label))
    ax.set_ylabel(ylabel or y_label)
    if title is not None:
        ax.set_title(title)
    apply_axis_style(ax, **style)
    return ax


def plot_integration_map_panels(
    panels: list[dict],
    *,
    meta: dict,
    axes: Sequence[plt.Axes] | None = None,
    figsize=None,
    contour_levels: tuple[float, ...] = (0.15, 0.3, 0.5, 0.7, 0.85),
    colours=None,
    cmap: str | None = None,
    interpolation: str | None = None,
    data_points_cutoff: float = 20.0,
    **style,
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
    if axes is None:
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize or (4 * n_panels, 4.0),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes, dtype=object).ravel()
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        fig = axes[0].figure

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
        ax.set_xlabel(meta["xlabel"])
        if meta.get("xticks") is not None:
            ax.set_xticks(meta["xticks"], labels=meta.get("x_tick_labels"))
        # ax.set_box_aspect(1)
        apply_axis_style(ax, **style)

    axes[0].set_ylabel(meta["ylabel"])
    return fig, axes


__all__ = [
    "add_shared_figure_legend",
    "apply_axis_style",
    "centered_numeric_group_palette",
    "make_single_panel_figure",
    "plot_empirical_accuracy_curve",
    "plot_prepared_weight_family",
    "plot_grouped_summary",
    "plot_integration_map_panels",
    "plot_regressor_net_impact",
    "plot_simple_summary",
    "resolve_axes",
    "resolve_single_axis",
]
