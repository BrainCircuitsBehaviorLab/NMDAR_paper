"""MCDR task-owned plots.

This module owns plots that depend on MCDR task semantics such as trial
difficulty, stimulus duration, delay duration, and side-stratified
performance. Shared model diagnostics are re-exported from
``glmhmmt.model_plots``.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import t

from glmhmmt.postprocess import (
    build_transition_matrix_by_subject_payload,
    build_transition_matrix_payload,
)
from glmhmmt.runtime import load_app_config
from glmhmmt.model_plots import (
    _state_color,
    plot_emission_weights as _plot_emission_weights_generic,
    plot_emission_weights_by_subject as _plot_emission_weights_by_subject_generic,
    plot_emission_weights_summary_boxplot as _plot_emission_weights_summary_boxplot_generic,
    plot_emission_weights_summary_lineplot as _plot_emission_weights_summary_lineplot_generic,
    plot_lapse_rates_boxplot as _plot_lapse_rates_boxplot,
    plot_posterior_probs,
    plot_change_triggered_posteriors_by_subject,
    plot_change_triggered_posteriors_summary,
    plot_session_deepdive,
    plot_session_trajectories,
    plot_state_accuracy,
    plot_state_dwell_times_by_subject,
    plot_state_dwell_times_summary,
    plot_state_dwell_times,
    plot_state_posterior_count_kde,
    plot_state_occupancy,
    plot_state_occupancy_overall,
    plot_state_occupancy_overall_by_subject,
    plot_state_occupancy_overall_boxplot,
    plot_state_occupancy_overall_summary,
    plot_state_session_occupancy,
    plot_state_session_occupancy_by_subject,
    plot_state_session_occupancy_summary,
    plot_state_switches,
    plot_state_switches_by_subject,
    plot_state_switches_summary,
    plot_tau_sweep,
    plot_transition_matrix as _plot_transition_matrix_simple,
    plot_transition_matrix_by_subject as _plot_transition_matrix_by_subject_simple,
    plot_transition_weights,
)

cfg = load_app_config()
CI_BAND_ERR_KWS = {"edgecolor": "none", "linewidth": 0}


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


def plot_emission_weights_by_subject(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    return _plot_emission_weights_by_subject_generic(
        K=K,
        weights_df=weights_df,
    )


def plot_emission_weights_summary(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    return _plot_emission_weights_generic(
        weights_df,
        K=K,
    )[1]


def plot_emission_weights_summary_lineplot(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    return _plot_emission_weights_summary_lineplot_generic(
        weights_df,
        K=K,
    )


def plot_emission_weights_summary_boxplot(
    weights_df,
    *,
    K: int | None = None,
) -> plt.Figure:
    return _plot_emission_weights_summary_boxplot_generic(
        weights_df,
        K=K,
    )


def plot_lapse_rates_boxplot(
    views: Optional[dict] = None,
    K: Optional[int] = None,
) -> plt.Figure:
    _ = K
    if not views:
        raise ValueError("views is required for lapse-rate plots.")
    return _plot_lapse_rates_boxplot(
        views,
        choice_labels=("Left", "Center", "Right"),
        title="Lapse rates",
    )


def plot_emission_weights(
    weights_df,
    *,
    K: int | None = None,
):
    return _plot_emission_weights_generic(
        weights_df,
        K=K,
    )


def truncate_colormap(cmap_name, minval=0.2, maxval=0.9, n=256):
    """Return a colormap truncated to a subrange."""
    cmap = cm.get_cmap(cmap_name, n)
    return colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def get_plot_path(subfolder: str, fname: str, model_name: str) -> Path:
    out_dir = Path("results") / "plots" / model_name / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def plot_cat_panel(ax, df, group_col, order, title, xlabel, ylabel=None, palette=None, labels=None):
    payload = process.prepare_cat_panel_payload(df, group_col=group_col, order=list(order))
    if payload is None:
        ax.set_visible(False)
        return

    cats = payload["cats"]
    md = payload["md"]
    sd = payload["sd"]
    mm = payload["mm"]
    sm = payload["sm"]

    ax.plot(np.arange(len(cats)), mm, "-", color="black", lw=2, label="Model")

    colors_used = palette if palette else ["black"] * len(cats)
    if payload["n_subjects"] > 1:
        ax.fill_between(np.arange(len(cats)), mm - sm, mm + sm, color="black", alpha=0.12)
        for i, (xpos, yval, err) in enumerate(zip(np.arange(len(cats)), md, sd)):
            ax.errorbar(xpos, yval, yerr=err, fmt="o", color=colors_used[i], ms=7, capsize=3)
    else:
        for i, (xpos, yval) in enumerate(zip(np.arange(len(cats)), md)):
            ax.errorbar(xpos, yval, fmt="o", color=colors_used[i], ms=7, capsize=3)

    ax.set_xticks(np.arange(len(cats)))
    if labels:
        label_map = dict(zip(order, labels))
        tick_labels = [label_map.get(c, c) for c in cats]
    else:
        tick_labels = cats
    ax.set_xticklabels(tick_labels)

    ax.set_ylim(0.2, 1.05)
    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15)
    ax.set_xlim(left=-0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plot_state_panel(ax, df_state, group_col, order, color):
    payload = process.prepare_state_panel_payload(df_state, group_col=group_col, order=list(order))
    if payload is None:
        return None, None

    xpos = payload["xpos"]
    md = payload["md"]
    sd = payload["sd"]
    mm = payload["mm"]
    sm = payload["sm"]
    n_subj = payload["n_subjects"]

    data_h = None
    for i, (x, y) in enumerate(zip(xpos, md)):
        eb = ax.errorbar(
            x,
            y,
            yerr=sd[i] if n_subj > 1 else None,
            fmt="o",
            color=color,
            ms=7,
            capsize=3,
            alpha=0.55,
            zorder=5,
            label="_nolegend_",
        )
        if data_h is None:
            data_h = eb

    (model_h,) = ax.plot(
        xpos,
        mm,
        "-",
        color=color,
        lw=2.2,
        alpha=0.95,
        zorder=6,
        label="_nolegend_",
    )
    if n_subj > 1:
        ax.fill_between(xpos, mm - sm, mm + sm, color=color, alpha=0.10, zorder=3)

    return data_h, model_h


def plot_categorical_performance_by_state(
    df,
    views: dict,
    model_name: str,
    background_style: str = "data",
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    figure_dpi: float = 80.0,
    overlay_only: bool = False,
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
):
    # Accepted for notebook API compatibility; MCDR keeps its current plot style.
    _ = (
        background_style,
        show_weighted_points,
        show_data_smooth,
        show_model_smooth,
        figure_dpi,
        overlay_only,
        model_line_mode,
        state_assignment_mode,
    )
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    if "state_rank" not in df.columns:
        raise ValueError("df must contain 'state_rank' (from build_trial_df).")

    K = next(iter(views.values())).K if views else int(df["state_rank"].max()) + 1

    state_labels = {}
    for v in views.values():
        for raw_idx, lbl in v.state_name_by_idx.items():
            rank = v.state_rank_by_idx[int(raw_idx)]
            state_labels.setdefault(rank, lbl)

    df = df.with_columns(pl.col("state_rank").cast(pl.Int64).alias("_state_k"))

    state_colors = {k: _state_color(state_labels.get(k, f"State {k}"), k) for k in range(K)}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    ax1, ax2, ax3 = axes
    panels = [
        (
            ax1,
            df,
            "ttype_c",
            cfg["plots"]["ttype"]["order"],
            "a) Trial difficulty",
            "Trial difficulty",
            cfg["plots"]["ttype"]["labels"],
        ),
        (
            ax2,
            df.filter(pl.col("ttype_c") == "DS"),
            "stimd_c",
            cfg["plots"]["stimd"]["order"],
            "b) Stim duration",
            "Stimulus type",
            cfg["plots"]["stimd"]["labels"],
        ),
        (
            ax3,
            df.filter(pl.col("stimd_c") == "SS"),
            "ttype_c",
            cfg["plots"]["delay"]["order"],
            "c) Delay duration",
            "Delay type",
            cfg["plots"]["delay"]["labels"],
        ),
    ]

    for ax, df_panel, gcol, order, title, xlabel, labels in panels:
        for k in range(K):
            df_k = df_panel.filter(pl.col("_state_k") == k)
            _plot_state_panel(ax, df_k, gcol, order, color=state_colors[k])

        cats = [c for c in order if df_panel.filter(pl.col(gcol) == c).height > 0]
        if labels:
            label_map = dict(zip(order, labels))
            tick_labels = [label_map.get(c, c) for c in cats]
        else:
            tick_labels = cats
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0.2, 1.05)
        ax.axhspan(0, 1 / 3, color="gray", alpha=0.15)
        ax.set_xlim(left=-0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ax is ax1:
            ax.set_ylabel("Accuracy")

    import matplotlib.lines as mlines

    legend_handles = []
    legend_labels = []
    for k in range(K):
        label = state_labels.get(k, f"State {k}")
        color = state_colors[k]
        legend_handles.append(mlines.Line2D([], [], marker="o", color=color, linestyle="None", ms=7, alpha=0.55))
        legend_labels.append(f"{label} data")
        legend_handles.append(mlines.Line2D([], [], color=color, lw=2.2, alpha=0.95))
        legend_labels.append(f"{label} model")

    ax3.legend(legend_handles, legend_labels, fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, axes


def plot_categorical_performance_all(
    df,
    model_name: str,
    views: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
):
    # Accepted for notebook API compatibility; MCDR keeps its current plot style.
    _ = (views, X_cols, ild_max, background_style)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    ax1, ax2, ax3 = axes
    df = process.prepare_categorical_performance_df(df)

    plot_cat_panel(
        ax1,
        df.clone(),
        "ttype_c",
        cfg["plots"]["ttype"]["order"],
        title="a) Trial difficulty",
        xlabel="Trial difficulty",
        ylabel="Accuracy",
        palette=cfg["plots"]["ttype"]["palette"],
        labels=cfg["plots"]["ttype"]["labels"],
    )
    plot_cat_panel(
        ax2,
        df.filter(pl.col("ttype_c") == "DS"),
        "stimd_c",
        cfg["plots"]["stimd"]["order"],
        title="b) Stim duration",
        xlabel="Stimulus type",
        palette=cfg["plots"]["stimd"]["palette"],
        labels=cfg["plots"]["stimd"]["labels"],
    )
    plot_cat_panel(
        ax3,
        df.filter(pl.col("stimd_c") == "SS"),
        "ttype_c",
        cfg["plots"]["delay"]["order"],
        title="c) Delay duration",
        xlabel="Delay type",
        palette=cfg["plots"]["delay"]["palette"],
        labels=cfg["plots"]["delay"]["labels"],
    )
    sns.despine()
    fig.tight_layout()
    return fig, axes


def plot_delay_or_stim_1d_on_ax(ax, df, subject, n_bins, which):
    """Plot delay or stimulus duration accuracy for one subject."""
    payload = process.prepare_delay_or_stim_1d_payload(df, subject=subject, n_bins=n_bins, which=which)
    if payload is None:
        title_suffix = "Delay" if which == "delay" else "Stimulus"
        ax.set_title(f"{subject} - {title_suffix}\n(no data)", fontsize=9)
        ax.axis("off")
        return False

    plot_df = payload["plot_df"]
    meta = payload["meta"]
    palette_data = (
        truncate_colormap("Purples_r", 0, 0.7)
        if meta["palette"] == "Purples_r"
        else truncate_colormap("Oranges", 0.3, 1.0)
    )

    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        err_kws=CI_BAND_ERR_KWS,
        ax=ax,
    )
    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Data"],
        x="center",
        y="acc",
        hue="center",
        palette=palette_data,
        marker="o",
        linewidth=0,
        errorbar=("ci", 95),
        err_style="bars",
        legend=False,
        ax=ax,
        zorder=10,
    )

    ax.axhspan(0, meta["band_floor"], color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel(meta["xlabel"], fontsize=12)
    ax.set_ylabel("Frac. correct responses", fontsize=12)
    ax.set_title(f"{subject}", fontsize=12)
    ax.tick_params(labelsize=12)
    sns.despine()
    return True


def plot_categorical_strat_by_side(
    df,
    subject,
    model_name,
    df_silent=None,
    cond_col="stimd_c",
    cond_order=("VG", "SL", "SM", "SS", "SIL"),
    cond_labels=("Visual", "Easy", "Medium", "Hard", "Silent"),
):
    payload = process.prepare_categorical_strat_by_side_payload(
        df,
        df_silent=df_silent,
        cond_col=cond_col,
        cond_order=cond_order,
        cond_labels=cond_labels,
    )
    g = payload["summary"]
    p_silent = payload["p_silent"]
    meta = payload["meta"]
    cond_order = meta["cond_order"]
    cond_labels = meta["cond_labels"]

    side_palette = {"L": "#e41a1c", "C": "#4daf4a", "R": "#377eb8"}
    fig, ax = plt.subplots(figsize=(4, 4))

    for side in ["L", "C", "R"]:
        sub = g[g["x_c"] == side].dropna(subset=["x_pos"]).sort_values("x_pos")
        if sub.empty:
            continue
        ax.plot(
            sub["x_pos"],
            sub["model_mean"],
            "-",
            lw=2,
            color=side_palette.get(side, "gray"),
            label=f"Model {side}",
            zorder=2,
        )
        ax.errorbar(
            sub["x_pos"],
            sub["data_mean"],
            yerr=sub["data_sem"],
            fmt="o",
            ms=5,
            capsize=3,
            color=side_palette.get(side, "gray"),
            linestyle="none",
            label=f"Data {side}",
            zorder=3,
        )
        if p_silent is not None:
            ax.plot(
                len(cond_order) - 1,
                p_silent[side],
                marker="D",
                ms=7,
                color=side_palette[side],
                linestyle="none",
                zorder=4,
            )

    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15, zorder=0)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_labels)
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Frac. correct responses")
    ax.set_xlabel("Trial difficulty")
    ax.set_title(f"{subject}")
    sns.despine()
    fig.tight_layout()
    return fig, ax


def plot_delay_binned_1d(df, model_name, subject=None, n_bins=7):
    payload = process.prepare_delay_binned_1d_payload(df, subject=subject, n_bins=n_bins)
    if payload is None:
        return None

    plot_delay = payload["plot_delay"]
    plot_stim = payload["plot_stim"]

    fig_delay, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(
        data=plot_delay[plot_delay["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        hue="ttype_c",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        err_kws=CI_BAND_ERR_KWS,
        ax=ax,
    )
    sns.lineplot(
        x="center",
        y="acc",
        hue="ttype_c",
        data=plot_delay[plot_delay["kind"] == "Data"],
        errorbar=("ci", 95),
        err_style="bars",
        marker="o",
        linewidth=0,
        ax=ax,
        zorder=10,
        legend=False,
    )
    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Delay (1D)")
    sns.despine()
    fig_delay.tight_layout()

    fig_stim, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(
        data=plot_stim[plot_stim["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        hue="stimd_c",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        err_kws=CI_BAND_ERR_KWS,
        ax=ax,
    )
    sns.lineplot(
        x="center",
        y="acc",
        hue="stimd_c",
        data=plot_stim[plot_stim["kind"] == "Data"],
        errorbar=("ci", 95),
        err_style="bars",
        marker="o",
        linewidth=0,
        ax=ax,
        zorder=10,
        legend=False,
    )
    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    ax.set_title(f"{title_subj} - Stimulus (1D)")
    sns.despine()
    fig_stim.tight_layout()

    return fig_delay, fig_stim



from src.process.common import (
    attach_repeat_choice_evidence,
    attach_total_fitted_evidence,
    display_regressor_name,
    pick_choice_history_regressor,
    prepare_evidence_curve,
    prepare_right_integration_maps,
    REPEAT_EVIDENCE_TAIL_QUANTILES,
    add_choice_lag_summary_regressor,
    compute_rb_by_x
)
from src.process import MCDR as process
from src.plots.common import (
    add_shared_figure_legend,
    make_single_panel_figure,
    plot_grouped_summary,
    plot_mean_over_data,
    plot_integration_map_panels,
    plot_simple_summary,
)


def plot_accuracy(df, ax=None, figsize=(3.0, 3.0), title="MCDR"):
    df_pd = df.to_pandas().copy() if hasattr(df, "to_pandas") else pd.DataFrame(df).copy()
    if "ttype_c" not in df_pd.columns and "ttype_n" in df_pd.columns:
        ttype_map = {float(key): value for key, value in cfg["encoding"]["ttype"].items()}
        df_pd["ttype_c"] = pd.to_numeric(df_pd["ttype_n"], errors="coerce").map(ttype_map)

    accuracy_col = "correct_bool" if "correct_bool" in df_pd.columns else "performance"
    return plot_mean_over_data(
        df_pd,
        x_col="ttype_c",
        y_col=accuracy_col,
        invert_x=False,
        x_order=cfg["plots"]["ttype"]["order"],
        # x_tick_labels=cfg["plots"]["ttype"]["labels"],
        x_tick_labels=['VG', 'Easy', 'Mid', 'Hard'],
        xlabel="Difficulty",
        title=title,
        baseline=1 / 3,
        color="tab:blue",
        ax=ax,
        figsize=figsize,
    )

def plot_rb(df, ax=None, figsize=(3.0, 3.0), title="MCDR"):
    df_pd = df.to_pandas().copy() if hasattr(df, "to_pandas") else pd.DataFrame(df).copy()
    if "ttype_c" not in df_pd.columns and "ttype_n" in df_pd.columns:
        ttype_map = {float(key): value for key, value in cfg["encoding"]["ttype"].items()}
        df_pd["ttype_c"] = pd.to_numeric(df_pd["ttype_n"], errors="coerce").map(ttype_map)

    rb_df = compute_rb_by_x(df_pd,x_col="ttype_c",choice_col="response",)

    return plot_mean_over_data(
        rb_df,
        x_col="ttype_c",
        y_col="rb",
        x_order=cfg["plots"]["ttype"]["order"],
        x_tick_labels=["VG", "Easy", "Mid", "Hard"],
        xlabel="Difficulty",
        ylabel="Repeating bias",
        title=title,
        baseline=1 / 3,
        baseline_area=False,
        color="tab:blue",
        ax=ax,
        figsize=figsize,
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
        cfg=cfg,
    )
    if not panels:
        return None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4), sharey=True)
    axes = np.atleast_1d(axes)

    x_cols = ["ttype_c", "stimd_c", "ttype_c"]

    for ax, panel, x_col in zip(axes, panels, x_cols, strict=False):
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
    fig.tight_layout()
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
        cfg=cfg,
        xlabel=xlabel,
        n_bins=n_bins,
    )
    if summary is None or summary.empty:
        return None

    label_map = dict(zip(cfg["plots"]["delay"]["order"], cfg["plots"]["delay"]["labels"]))

    fig, ax = make_single_panel_figure()
    plot_grouped_summary(
        ax,
        summary,
        line_group_col="ttype_c",
        x_col="x_center",
        meta=meta,
        label_map=label_map,
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
        is_mcdr=True,
    )
    if df_pd.empty or "_fitted_total_evidence" not in df_pd.columns:
        return None

    summary, meta = prepare_evidence_curve(
        df_pd,
        evidence_col="_fitted_total_evidence",
        data_col="correct_bool",
        model_col="_fitted_correct_prob",
        baseline=1.0 / 3.0,
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
        is_mcdr=True,
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
    "plot_accuracy_by_difficulty",
    "plot_accuracy_by_total_evidence",
    "plot_binned_accuracy_figure",
    "plot_repeat_by_repeat_evidence",
    "plot_right_integration_map",
    "plot_right_by_regressor",
    "plot_right_by_regressor_simple",
]
