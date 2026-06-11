from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from src.process.common import (
    PreparedWeightFamilyPlot,
    attach_repeat_choice_evidence,
    attach_quantile_bin_column,
    attach_total_fitted_evidence,
    display_regressor_name,
    fit_lapse_logistic_by_group,
    fit_lapse_logistic_by_subject_group,
    padded_numeric_limits,
    summarize_grouped_panel,
    summarize_simple_curve,
    to_pandas_df,
)
from glmhmmt.plots.common import (
    custom_boxplot,
    resolve_single_axis as resolve_glmhmmt_single_axis,
)


def _significance_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue) or pvalue >= 0.05:
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    return "*"


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
    figsize=(4.0, 4.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
    connect_subjects: bool = True,
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
        fig, ax, created_fig = resolve_glmhmmt_single_axis(ax=ax, figsize=figsize)
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(prepared.xlabel)
        ax.set_ylabel(prepared.ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(summary["x_label"].astype(str).tolist())
        if created_fig:
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

    fig, ax, created_fig = resolve_glmhmmt_single_axis(ax=ax, figsize=figsize)
    positions = np.arange(1, len(x_order) + 1)
    custom_boxplot(
        ax,
        per_feature_values,
        positions=positions,
        widths=0.55,
        median_colors="tab:blue",
        line_values=subject_lines if connect_subjects else None,
        line_color="#7A7A7A",
        line_alpha=0.15,
        line_linewidth=1.0,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(prepared.xlabel)
    ax.set_ylabel(prepared.ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(x_order))
    if created_fig:
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
    show_baseline_ttest: bool = False,
    ax: plt.Axes | None = None,
    figsize=fig_size(n_cols=3),
    label: str | None = None,
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
        label=label,
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

    if show_baseline_ttest:
        for xi, (_, row) in zip(x, summary.iterrows(), strict=False):
            if x_order is not None:
                values = subject_summary.loc[
                    subject_summary[x_col] == row[x_col], "subject_mean"
                ].dropna()
            else:
                values = subject_summary.loc[
                    pd.to_numeric(subject_summary[x_col], errors="coerce").eq(row["_x_numeric"]),
                    "subject_mean",
                ].dropna()
            if len(values) < 2:
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=baseline).pvalue)
            label = _significance_stars(pvalue)
            if not label:
                continue
            y = min(0.97, float(row["mean"]) + float(row["sem"]) + 0.05)
            ax.text(
                xi,
                y,
                label,
                ha="center",
                va="bottom",
                # fontsize=9,
                color="black",
            )

    if invert_x:
        # Only invert if not already inverted (track with a marker on the axis)
        if not getattr(ax, '_x_inverted_marker', False):
            ax.invert_xaxis()
            ax._x_inverted_marker = True

    # Add legend if labels are present
    if label is not None or ax.get_lines():
        ax.legend(frameon=False)

    return fig


COUNTERFACTUAL_PALETTE = {
    "Data": "#1f77b4",
    "Full fitted": "#111111",
    "Fixed bias": "#d55e00",
    "Fixed lapses": "#009e73",
}


def _plot_counterfactual_summary(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    order: Sequence[str],
) -> None:
    """Shared line/error plotting for the RB and lag-match simulation summaries."""
    required = {"scenario", x_col, mean_col, lo_col, hi_col}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return

    for scenario in order:
        sub = df[df["scenario"] == scenario].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(x_col)
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[mean_col].to_numpy(dtype=float)
        lo = sub[lo_col].to_numpy(dtype=float)
        hi = sub[hi_col].to_numpy(dtype=float)
        color = COUNTERFACTUAL_PALETTE[scenario]
        if scenario == "Data":
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                label=scenario,
                zorder=4,
            )
        else:
            ax.plot(x, y, lw=1.8, marker="o", ms=3, color=color, label=scenario)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0.0)


def plot_action_trace_counterfactual_rb(summary, meta, *, ax: plt.Axes | None = None, figsize=(4.6, 3.4)):
    """Plot repetition bias from empirical data and parameter-fixed simulations."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)
    df = to_pandas_df(summary)
    _plot_counterfactual_summary(
        ax,
        df,
        x_col="_counterfactual_rb_x",
        mean_col="rb_mean",
        lo_col="rb_lo",
        hi_col="rb_hi",
        order=("Data", "Full fitted", "Fixed bias", "Fixed lapses"),
    )
    ax.axhline(float(meta.get("baseline", 0.5)), color="0.5", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel(meta.get("xlabel", "Task variable"))
    ax.set_ylabel("Rep. bias")
    ax.set_ylim(0.0, 1.0)
    if meta.get("xticks") is not None:
        ax.set_xticks(meta["xticks"])
        if meta.get("x_tick_labels") is not None:
            ax.set_xticklabels(meta["x_tick_labels"])
    if meta.get("invert_x", False):
        ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("")
    return fig, ax


def plot_action_trace_counterfactual_lag_match(
    lag_summary,
    meta,
    *,
    ax: plt.Axes | None = None,
    figsize=(4.6, 3.2),
):
    """Plot p(response_t = response_{t-L}) from lag 1 to the selected max lag."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)
    df = to_pandas_df(lag_summary)
    _plot_counterfactual_summary(
        ax,
        df,
        x_col="lag",
        mean_col="lag_match_mean",
        lo_col="lag_match_lo",
        hi_col="lag_match_hi",
        order=("Data", "Full fitted", "Fixed bias", "Fixed lapses"),
    )
    ax.axhline(float(meta.get("baseline", 0.5)), color="0.5", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("History lag")
    ax.set_ylabel(r"$p(\hat r_t = r_{t-L})$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.5, float(meta.get("max_history_lag", 10)) + 0.5)
    ax.set_xticks(np.arange(1, int(meta.get("max_history_lag", 10)) + 1))
    ax.set_title("Lag-match parameter-fixed simulation")
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


def plot_action_trace_counterfactual_subject_scatter(
    subject_scatter,
    meta=None,
    *,
    ax: plt.Axes | None = None,
    figsize=(3.6, 3.6),
):
    """Plot one animal per point: empirical RB vs full-fitted simulated RB."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)
    df = to_pandas_df(subject_scatter)
    required = {"empirical_rb", "full_fitted_rb"}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "No valid subject data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    x = df["empirical_rb"].to_numpy(dtype=float)
    y = df["full_fitted_rb"].to_numpy(dtype=float)
    yerr = None
    if {"full_fitted_rb_lo", "full_fitted_rb_hi"}.issubset(df.columns):
        lo = df["full_fitted_rb_lo"].to_numpy(dtype=float)
        hi = df["full_fitted_rb_hi"].to_numpy(dtype=float)
        yerr = np.vstack([np.clip(y - lo, 0.0, None), np.clip(hi - y, 0.0, None)])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=".",
        ms=5,
        color=COUNTERFACTUAL_PALETTE["Full fitted"],
        ecolor=COUNTERFACTUAL_PALETTE["Full fitted"],
        alpha=0.8,
        capsize=2,
        linewidth=1.0,
    )
    ax.plot([0, 1], [0, 1], color="0.45", lw=0.9, ls="--", zorder=0)
    ax.axhline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental RB")
    ax.set_ylabel("Full fitted RB")
    ax.set_title("Full model by animal")
    if len(df) > 1 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.2f}", ha="left", va="top", transform=ax.transAxes)
    return fig, ax


def plot_action_trace_parameter_fixed_rb(*args, **kwargs):
    """Alias for the parameter-fixed repetition-bias plot."""
    return plot_action_trace_counterfactual_rb(*args, **kwargs)


def plot_action_trace_parameter_fixed_lag_match(*args, **kwargs):
    """Alias for the parameter-fixed lag-match plot."""
    return plot_action_trace_counterfactual_lag_match(*args, **kwargs)


def plot_action_trace_parameter_fixed_subject_scatter(*args, **kwargs):
    """Alias for the parameter-fixed full-model animal scatter."""
    return plot_action_trace_counterfactual_subject_scatter(*args, **kwargs)


def add_shared_figure_legend(
    fig,
    *,
    source_ax,
    title: str | None = None,
    legend_ax: plt.Axes | None = None,
    bbox_x: float = 0.94,
    legend: bool = True,
) -> None:
    if not legend:
        return
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return
    if legend_ax is not None:
        legend_ax.axis("off")
        legend_ax.legend(
            handles,
            labels,
            title=title,
            loc="center left",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            labelspacing=0.35,
            handlelength=2.0,
        )
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


def plot_repeat_by_regressor_simple(
    plot_df,
    *,
    regressor_col: str,
    views: dict,
    is_mcdr: bool,
    baseline: float | None = None,
    xlabel: str | None = None,
    n_bins: int = 10,
    legend: bool = True,
    **style,
):
    df_pd = attach_repeat_choice_evidence(
        plot_df,
        views=views,
        is_mcdr=is_mcdr,
    )
    required_cols = {regressor_col, "_repeat_choice", "_p_repeat_model", "subject"}
    if df_pd.empty or not required_cols.issubset(df_pd.columns):
        return None

    df_pd = df_pd.copy()
    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd["_repeat_choice"] = pd.to_numeric(df_pd["_repeat_choice"], errors="coerce")
    df_pd["_p_repeat_model"] = pd.to_numeric(df_pd["_p_repeat_model"], errors="coerce")
    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd["_repeat_choice"])
        & np.isfinite(df_pd["_p_repeat_model"])
    ].copy()
    if df_pd.empty:
        return None

    df_pd, bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=n_bins,
        quantiles=None,
    )
    if df_pd is None:
        return None

    summary = summarize_simple_curve(
        df_pd,
        subject_col="subject",
        reg_bin_col="_reg_bin",
        regressor_col=regressor_col,
        data_col="_repeat_choice",
        model_col="_p_repeat_model",
    )
    if summary.empty:
        return None

    if baseline is None:
        baseline = 1.0 / next(iter(views.values())).num_classes if views else 0.5
    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": r"$p(\mathrm{repeat})$",
        "baseline": float(baseline),
        "xlim": padded_numeric_limits(bin_centers["x_center"], absolute_pad=0.25),
    }
    return plot_simple_summary(summary, meta=meta, legend=legend, **style)


def prepare_binned_accuracy_total_evidence_panels(
    plot_df,
    *,
    regressor_col: str,
    adapter,
    views: dict,
    is_mcdr: bool,
    baseline: float,
    n_bins: int = 10,
) -> tuple[list[dict] | None, str | None]:
    if adapter is None or views is None:
        raise ValueError("x_axis='total_evidence' requires adapter=... and views=....")

    df_pd = attach_total_fitted_evidence(
        plot_df,
        adapter=adapter,
        views=views,
        is_mcdr=is_mcdr,
    )
    required_cols = {
        regressor_col,
        "correct_bool",
        "_fitted_correct_prob",
        "_fitted_total_evidence",
        "subject",
    }
    if df_pd.empty or not required_cols.issubset(df_pd.columns):
        return None, None

    df_pd = df_pd.copy()
    for col in [regressor_col, "correct_bool", "_fitted_correct_prob", "_fitted_total_evidence"]:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")
    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd["correct_bool"])
        & np.isfinite(df_pd["_fitted_correct_prob"])
        & np.isfinite(df_pd["_fitted_total_evidence"])
    ].copy()
    if df_pd.empty:
        return None, None

    df_pd, reg_bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=4,
        quantiles=None,
    )
    if df_pd is None:
        return None, None
    reg_bin_labels = reg_bin_centers["_reg_bin"].tolist()

    df_pd, evidence_bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col="_fitted_total_evidence",
        bin_col="_evidence_bin",
        max_bins=n_bins,
        quantiles=None,
    )
    if df_pd is None:
        return None, None

    summary = summarize_grouped_panel(
        df_pd,
        line_group_col="_reg_bin",
        x_col="_evidence_bin",
        subject_col="subject",
        data_col="correct_bool",
        model_col="_fitted_correct_prob",
        line_order=reg_bin_labels,
    )
    if summary.empty:
        return None, None
    summary = summary.merge(
        evidence_bin_centers[["_evidence_bin", "x_center"]],
        on="_evidence_bin",
        how="left",
    ).sort_values(["_reg_bin", "x_center"])

    subject_summary = (
        df_pd.groupby(["_reg_bin", "subject", "_evidence_bin"], observed=True)
        .agg(
            data_mean=("correct_bool", "mean"),
            model_mean=("_fitted_correct_prob", "mean"),
            n_trials=("correct_bool", "count"),
        )
        .reset_index()
        .merge(evidence_bin_centers[["_evidence_bin", "x_center"]], on="_evidence_bin", how="left")
    )

    meta = {
        "xlabel": "Correct-vs-rest fitted evidence",
        "ylabel": "Accuracy",
        "legend_title": display_regressor_name(regressor_col),
        "baseline": baseline,
        "line_order": reg_bin_labels,
        "line_x_centers": dict(zip(reg_bin_centers["_reg_bin"], reg_bin_centers["x_center"], strict=False)),
        "x_col": "x_center",
        "fit_x_col": "x_center",
    }
    return [{"summary": summary, "subject_summary": subject_summary, "meta": meta}], display_regressor_name(regressor_col)


def fit_lapse_logistic_for_panel(
    panel: dict,
    *,
    fit_lapse_by_subject: bool = True,
    lapse_max: float = 0.4,
    share_lapse_logistic_core: bool = False,
    default_x_col: str,
    fit_x_limits: tuple[float, float] | None = None,
):
    summary = panel.get("summary")
    meta = panel.get("meta", {})
    if summary is None or summary.empty:
        return {}

    x_fit_col = meta.get("fit_x_col", meta.get("x_col", default_x_col))
    if x_fit_col not in summary.columns:
        return {}
    if pd.to_numeric(summary[x_fit_col], errors="coerce").notna().sum() < 2:
        return {}
    if fit_x_limits is not None:
        x_min, x_max = (float(v) for v in fit_x_limits)
        x_values = pd.to_numeric(summary[x_fit_col], errors="coerce")
        summary = summary.loc[x_values.between(x_min, x_max, inclusive="both")].copy()
        if summary.empty or x_values.loc[summary.index].nunique() < 2:
            return {}

    line_order = meta.get("line_order") or summary["_reg_bin"].dropna().unique().tolist()
    subject_summary = panel.get("subject_summary")
    if (
        fit_lapse_by_subject
        and subject_summary is not None
        and not subject_summary.empty
        and x_fit_col in subject_summary.columns
    ):
        if fit_x_limits is not None:
            x_min, x_max = (float(v) for v in fit_x_limits)
            subject_x_values = pd.to_numeric(subject_summary[x_fit_col], errors="coerce")
            subject_summary = subject_summary.loc[
                subject_x_values.between(x_min, x_max, inclusive="both")
            ].copy()
            if subject_summary.empty:
                return {}
        subject_fits = fit_lapse_logistic_by_subject_group(
            subject_summary,
            subject_col="subject",
            line_group_col="_reg_bin",
            x_col=x_fit_col,
            y_col="data_mean",
            weight_col="n_trials",
            line_order=line_order,
            lapse_max=lapse_max,
            shared_core=share_lapse_logistic_core,
        )
        if subject_fits:
            return subject_fits

    return fit_lapse_logistic_by_group(
        summary,
        line_group_col="_reg_bin",
        x_col=x_fit_col,
        y_col="md",
        weight_col="nd",
        line_order=line_order,
        lapse_max=lapse_max,
        shared_core=share_lapse_logistic_core,
    )


def plot_lapse_fit_parameter_panels(
    axes,
    fits: dict,
    *,
    line_order,
    meta: dict,
    regressor_label: str,
):
    axes = np.atleast_1d(axes)
    if len(axes) < 2:
        return

    lapse_ax, bias_ax = axes[:2]
    groups = [group for group in list(line_order or fits.keys()) if group in fits]
    if not groups:
        for ax in (bias_ax, lapse_ax):
            ax.set_axis_off()
        return

    center_map = meta.get("line_x_centers") or {}
    x_values = []
    use_numeric_centers = bool(center_map)
    for idx, group in enumerate(groups):
        value = center_map.get(group, center_map.get(str(group), np.nan))
        if use_numeric_centers and np.isfinite(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]):
            x_values.append(float(value))
        else:
            use_numeric_centers = False
            x_values.append(float(idx))

    if not use_numeric_centers:
        x_values = np.arange(len(groups), dtype=float)
        for ax in (bias_ax, lapse_ax):
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(group) for group in groups], rotation=0)

    lapse_left = np.asarray([fits[group].lapse_left for group in groups], dtype=float)
    lapse_right = np.asarray([fits[group].lapse_right for group in groups], dtype=float)
    bias = np.asarray([fits[group].bias for group in groups], dtype=float)
    y_values = np.concatenate([lapse_left, lapse_right, bias, np.asarray([0.0])])
    y_values = y_values[np.isfinite(y_values)]
    shared_ylim = None
    if y_values.size:
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if np.isclose(y_min, y_max):
            pad = 0.1 if np.isclose(y_min, 0.0) else 0.1 * abs(y_min)
        else:
            pad = 0.05 * (y_max - y_min)
        shared_ylim = (y_min - pad, y_max + pad)

    lapse_ax.plot(x_values, lapse_left, "-o", color="#2b7bba", lw=1.5, ms=4, label="left")
    lapse_ax.plot(x_values, lapse_right, "-o", color="#c43c39", lw=1.5, ms=4, label="right")
    apply_axis_style(
        lapse_ax,
        xlabel=regressor_label,
        ylabel="Lapse",
        title="Lapses",
    )
    lapse_ax.legend(frameon=False, fontsize=8)

    bias_ax.plot(x_values, bias, "-o", color="black", lw=1.5, ms=4)
    bias_ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    apply_axis_style(
        bias_ax,
        xlabel=regressor_label,
        ylabel="Psychometric bias",
        title="Bias",
    )
    if shared_ylim is not None:
        lapse_ax.set_ylim(shared_ylim)
        bias_ax.set_ylim(shared_ylim)


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

    summary_df = summary_df.copy()
    rename_map = {
        "data_mean": "md",
        "model_mean": "mm",
        "data_sem": "sem",
    }
    for src, dst in rename_map.items():
        if dst not in summary_df.columns and src in summary_df.columns:
            summary_df[dst] = summary_df[src]

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
    cbar_ax: plt.Axes | None = None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
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
        ncols = n_panels + int(show_colorbar and cbar_ax is None)
        width_ratios = [1.0] * n_panels + ([0.08] if show_colorbar and cbar_ax is None else [])
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=figsize or (4 * n_panels + (0.35 if show_colorbar else 0.0), 4.0),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios},
        )
        axes = np.asarray(axes, dtype=object).ravel()
        if show_colorbar and cbar_ax is None:
            cbar_ax = axes[-1]
            axes = axes[:n_panels]
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
    if show_colorbar:
        color_map = LinearSegmentedColormap.from_list("integration_map", colours)
        scalar_mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=color_map)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            cax=cbar_ax,
            ax=None if cbar_ax is not None else axes,
        )
        colorbar.set_label(colorbar_label or meta.get("colorbar_label", r"$\mathit{p}(\mathrm{right})$"))
    return fig, axes

def plot_session_running_accuracy_repetition(
    prepared,
    *,
    ax: plt.Axes | None = None,
    figsize=(4.8, 3.0),
    accuracy_label: str = "Accuracy",
    repetition_label: str = "Repeating bias",
    **style,
):
    """Plot running accuracy and repetition probability across trials."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)

    trace = to_pandas_df(prepared.get("trace", pd.DataFrame()))
    required = {"trial_index", "running_accuracy", "running_repetition"}
    if trace.empty or not required.issubset(trace.columns):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        apply_axis_style(ax, **style)
        return fig, ax

    ax.plot(
        trace["trial_index"],
        trace["running_accuracy"],
        lw=1.8,
        color="#2b7bba",
        label=accuracy_label,
    )
    ax.plot(
        trace["trial_index"],
        trace["running_repetition"],
        lw=1.8,
        color="#c43c39",
        label=repetition_label,
    )

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Running probability")
    ax.legend(frameon=False, fontsize=8)

    window = prepared.get("meta", {}).get("running_window")
    if window is not None:
        ax.set_title(f"Running behavior, window={window}")

    apply_axis_style(ax, **style)
    return fig, ax


def plot_session_behavior_autocorrelogram(
    prepared,
    *,
    ax: plt.Axes | None = None,
    figsize=(4.0, 3.0),
    **style,
):
    """Plot autocorrelogram of outcome and repetition vectors."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)

    ac = to_pandas_df(prepared.get("autocorr", pd.DataFrame()))
    required = {"lag", "autocorr", "signal"}
    if ac.empty or not required.issubset(ac.columns):
        ax.text(0.5, 0.5, "No valid autocorrelation data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        apply_axis_style(ax, **style)
        return fig, ax

    palette = {
        "Outcome": "#2b7bba",
        "Repetition": "#c43c39",
    }

    for signal in ("Outcome", "Repetition"):
        sub = ac[ac["signal"] == signal].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("lag")
        ax.plot(
            sub["lag"],
            sub["autocorr"],
            lw=1.8,
            marker="o",
            ms=3,
            color=palette.get(signal, "black"),
            label=signal,
        )

    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Behavioral timescale")
    if ac["lag"].notna().any():
        ax.set_xlim(left=1.0, right=float(np.nanmax(ac["lag"])) if len(ac) else None)
    ax.legend(frameon=False, fontsize=8)

    apply_axis_style(ax, **style)
    return fig, ax


def plot_session_accuracy_repetition_timescale(
    prepared,
    *,
    axes: Sequence[plt.Axes] | None = None,
    figsize=(8.8, 3.0),
    running_style: dict | None = None,
    autocorr_style: dict | None = None,
):
    """Plot running behavior and behavioral autocorrelogram as a two-panel figure."""
    fig, axes = resolve_axes(
        axes=axes,
        n_axes=2,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    plot_session_running_accuracy_repetition(
        prepared,
        ax=axes[0],
        **(running_style or {}),
    )
    plot_session_behavior_autocorrelogram(
        prepared,
        ax=axes[1],
        **(autocorr_style or {}),
    )
    return fig, axes


def plot_corrected_behavior_autocorrelograms(
    prepared,
    *,
    axes: Sequence[plt.Axes] | None = None,
    figsize=(7.0, 3.0),
    model_autocorr=None,
    glm_autocorr=None,
    autocorr_col: str = "autocorr",
    sem_col: str | None = "autocorr_sem",
    model_autocorr_col: str | None = None,
    data_color: str = "#1f77b4",
    model_color: str = "black",
    data_label: str = "Data",
    model_label: str = "Fitted GLM",
    ylabel: str = "Corrected autocorrelation",
    signals: Sequence[str] = ("Outcome", "Repetition"),
    titles: dict[str, str] | None = None,
    **style,
):
    """Plot Tiffany-style autocorrelograms for outcomes and repetitions.

    Pass ``model_autocorr`` or ``glm_autocorr`` to overlay a fitted GLM simulation.
    ``autocorr_col``/``sem_col`` can be changed to plot raw autocorrelation or
    cross-session correction terms with the same style.
    """
    fig, axes = resolve_axes(
        axes=axes,
        n_axes=len(signals),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )

    data = to_pandas_df(prepared.get("autocorr", pd.DataFrame()))
    model = model_autocorr if model_autocorr is not None else glm_autocorr
    if model is None:
        model = prepared.get("model_autocorr")
    model = to_pandas_df(model) if model is not None else pd.DataFrame()

    default_titles = {
        "Outcome": "Choice outcomes",
        "Repetition": "Repeated responses",
    }
    if titles is not None:
        default_titles.update(titles)

    for ax, signal in zip(axes, signals, strict=False):
        title = default_titles.get(signal, str(signal))
        sub = (
            data[data["signal"] == signal].copy()
            if "signal" in data.columns
            else pd.DataFrame()
        )
        if sub.empty or not {"lag", autocorr_col}.issubset(sub.columns):
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        sub = sub.sort_values("lag")
        yerr = sub[sem_col].to_numpy(dtype=float) if sem_col is not None and sem_col in sub.columns else None
        ax.errorbar(
            sub["lag"].to_numpy(dtype=float),
            sub[autocorr_col].to_numpy(dtype=float),
            yerr=yerr,
            fmt="o",
            ms=4,
            color=data_color,
            ecolor=data_color,
            elinewidth=1.0,
            capsize=2,
            label=data_label,
            zorder=4,
        )

        model_sub = (
            model[model["signal"] == signal].copy()
            if "signal" in model.columns
            else pd.DataFrame()
        )
        _model_col = model_autocorr_col or autocorr_col
        if _model_col not in model_sub.columns and "autocorr" in model_sub.columns:
            _model_col = "autocorr"
        if not model_sub.empty and {"lag", _model_col}.issubset(model_sub.columns):
            model_sub = model_sub.sort_values("lag")
            ax.plot(
                model_sub["lag"].to_numpy(dtype=float),
                model_sub[_model_col].to_numpy(dtype=float),
                color=model_color,
                lw=1.8,
                label=model_label,
                zorder=3,
            )

        ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        # ax.set_xlim(0, 25.5)
        apply_axis_style(ax, **style)

    return fig, axes


def psychometric_repeat(
    plot_df,
    ax=None,
    figsize=fig_size(n_cols=3),
    title="",
    color="tab:blue",
    *,
    session_col,
    trial_col,
    choice_col,
    stimulus_col,
    subject_col="subject",
    delay_col=None,
    difficulty_col=None,
    is_mcdr=False,
):
    df_pd = plot_df.to_pandas().copy() if hasattr(plot_df, "to_pandas") else pd.DataFrame(plot_df).copy()

    required_cols = [subject_col, session_col, trial_col, choice_col, stimulus_col]
    missing_cols = [column for column in required_cols if column not in df_pd.columns]
    optional_cols = [column for column in [delay_col, difficulty_col] if column is not None]
    missing_cols.extend(column for column in optional_cols if column not in df_pd.columns)
    if missing_cols:
        raise KeyError(f"psychometric_repeat missing dataframe columns: {', '.join(missing_cols)}")

    sort_cols = [subject_col, session_col, trial_col]
    plot_df = df_pd.sort_values(sort_cols, kind="stable").copy()
    plot_df["choice"] = pd.to_numeric(plot_df[choice_col], errors="coerce")
    plot_df["previous_choice"] = plot_df.groupby(
        [subject_col, session_col],
        observed=True,
    )["choice"].shift(1)
    plot_df["_repeat"] = (plot_df["choice"] == plot_df["previous_choice"]).astype(float)

    if is_mcdr:
        baseline = 1.0 / 3.0
    else:
        choice_values = set(plot_df["choice"].dropna().unique().tolist())
        if choice_values.issubset({-1.0, 1.0}):
            plot_df["previous_choice_sign"] = plot_df["previous_choice"]
        elif choice_values.issubset({0.0, 1.0}):
            plot_df["previous_choice_sign"] = (2.0 * plot_df["previous_choice"]) - 1.0
        else:
            raise ValueError("psychometric_repeat expects binary choices unless is_mcdr=True.")
        baseline = 0.5

    signed_stimulus = pd.to_numeric(plot_df[stimulus_col], errors="coerce")
    x_order = None
    x_tick_labels = None
    if delay_col is not None:
        x_values = (
            pd.to_numeric(plot_df[delay_col], errors="coerce").abs()
            * np.sign(signed_stimulus)
            * plot_df["previous_choice_sign"]
        )
        x_order = ["neg_0.1", "neg_1", "neg_3", "neg_10", "pos_10", "pos_3", "pos_1", "pos_0.1"]
        x_tick_labels = ["-0", "-1", "-3", "-10", "10", "3", "1", "0"]

        delay_magnitude = pd.Series(x_values, index=plot_df.index).abs()
        delay_label = np.where(
            np.isclose(delay_magnitude, 0.1),
            "0.1",
            delay_magnitude.round().astype("Int64").astype(str),
        )
        delay_side = np.where(x_values < 0, "neg", "pos")
        x_values = pd.Series(delay_side, index=plot_df.index) + "_" + pd.Series(delay_label, index=plot_df.index)
        xlabel = "Delay x choice$_{-1}$"
    elif is_mcdr:
        if difficulty_col is None:
            raise ValueError("psychometric_repeat requires difficulty_col when is_mcdr=True.")
        difficulty_labels = plot_df[difficulty_col].astype(str).map(
            {
                "VG": "VG",
                "DS": "DS",
                "DM": "DM",
                "DL": "DL",
            }
        )
        current_target = pd.to_numeric(plot_df[stimulus_col], errors="coerce")
        side_labels = np.where(current_target == plot_df["previous_choice"], "pos", "neg")
        x_values = pd.Series(side_labels, index=plot_df.index) + "_" + difficulty_labels
        x_order = ["neg_VG", "neg_DS", "neg_DM", "neg_DL", "pos_DL", "pos_DM", "pos_DS", "pos_VG"]
        x_tick_labels = ["-VG", "-DS", "-DM", "-DL", "DL", "DM", "DS", "VG"]
        xlabel = "Difficulty. x choice$_{-1}$"
    else:
        x_values = signed_stimulus * plot_df["previous_choice_sign"]
        x_values = pd.Series(x_values, index=plot_df.index).mask(lambda values: np.isclose(values, 0.0), 0.0)
        xlabel = "Stim. x choice$_{-1}$"
        x_tick_labels = [-20, -8, "", "", 0, "", "", 8, 20]

    plot_df["_repeat_x"] = x_values
    plot_df = plot_df.dropna(subset=["_repeat_x", "_repeat", "previous_choice"]).copy()

    return plot_mean_over_data(
        plot_df,
        x_col="_repeat_x",
        y_col="_repeat",
        subject_col=subject_col,
        x_order=x_order,
        x_tick_labels=x_tick_labels,
        xlabel=xlabel,
        ylabel=r"$p(\mathrm{repeat})$",
        title=title,
        baseline=baseline,
        baseline_area=False,
        color=color,
        ax=ax,
        figsize=figsize,
    )


__all__ = [
    "add_shared_figure_legend",
    "apply_axis_style",
    "centered_numeric_group_palette",
    "make_single_panel_figure",
    "plot_empirical_accuracy_curve",
    "plot_action_trace_counterfactual_lag_match",
    "plot_action_trace_counterfactual_rb",
    "plot_action_trace_counterfactual_subject_scatter",
    "plot_action_trace_parameter_fixed_lag_match",
    "plot_action_trace_parameter_fixed_rb",
    "plot_action_trace_parameter_fixed_subject_scatter",
    "plot_corrected_behavior_autocorrelograms",
    "plot_session_accuracy_repetition_timescale",
    "plot_session_behavior_autocorrelogram",
    "plot_session_running_accuracy_repetition",
    "plot_prepared_weight_family",
    "plot_grouped_summary",
    "plot_integration_map_panels",
    "plot_regressor_net_impact",
    "plot_simple_summary",
    "resolve_axes",
    "resolve_single_axis",
]
