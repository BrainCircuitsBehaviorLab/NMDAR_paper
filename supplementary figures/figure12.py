# /// script
# [tool.marimo.opengraph]
# title = "Figure 12"
# description = "Figure 12: Behavioral stability and reversibility across early and late drug/saline sessions."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    import os
    from scipy.stats import ttest_rel
    from statannotations.Annotator import Annotator

    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths
    from src.process.common import (
        label_condition_session_windows,
        prepare_session_rolling_accuracy,
    )
    from src.plots.common import BOXPLOT_STYLE, fig_size

    return (
        Annotator,
        BOXPLOT_STYLE,
        Line2D,
        Path,
        configure_paths,
        fig_size,
        get_adapter,
        label_condition_session_windows,
        mo,
        np,
        os,
        pl,
        plt,
        prepare_session_rolling_accuracy,
        sns,
        ttest_rel,
    )


@app.cell
def _(Line2D, Path, fig_size, plt, sns):
    # Set style
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(Path(__file__).resolve().parents[1] / "paper.mplstyle")
    rolling_figsize = fig_size(n_cols=2, ratio=1.5)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    session_window_palette = {
        ("Saline", "First"): "#bdbdbd",
        ("Saline", "Late (≥3)"): "#4d4d4d",
        ("Drug", "First"): "#f4a3c4",
        ("Drug", "Late (≥3)"): "#b2186b",
    }
    session_window_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markeredgewidth=0,
            label=f"{condition} {window.split()[0].lower()}",
        )
        for (condition, window), color in session_window_palette.items()
    ]
    condition_order = ["Saline", "Drug"]
    condition_palette = {
        "Saline": "tab:gray",
        "Drug": "tab:pink",
    }
    return (
        condition_order,
        condition_palette,
        rolling_figsize,
        session_window_handles,
        session_window_palette,
    )


@app.cell
def _():
    mount_figure = True
    return (mount_figure,)


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                [
                    "rolling_accuracy_mean_2ADC",
                    "rolling_accuracy_mean_2ADC",
                    "rolling_accuracy_mean_2AFC",
                    "rolling_accuracy_mean_2AFC",
                    "rolling_accuracy_mean_MCDR",
                    "rolling_accuracy_mean_MCDR",
                ],
                [
                    "session_accuracy_2ADC",
                    "session_accuracy_2ADC",
                    "session_accuracy_2AFC",
                    "session_accuracy_2AFC",
                    "session_accuracy_MCDR",
                    "session_accuracy_MCDR",
                ],
                [
                    "whole_session_variance_subject_2ADC",
                    "whole_session_variance_subject_2ADC",
                    "whole_session_variance_subject_2AFC",
                    "whole_session_variance_subject_2AFC",
                    "whole_session_variance_subject_MCDR",
                    "whole_session_variance_subject_MCDR",
                ],
            ],
            figsize=fig_size(1, 1),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell
def _(Path, configure_paths, os):
    # Set paths
    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")

    data_path = Path(__file__).parents[1] / "data/processed"
    print(data_path)

    path_panels = Path(__file__).resolve().parent / "panels12"
    print(path_panels)
    os.makedirs(path_panels, exist_ok=True)
    return data_path, path_panels


@app.cell
def _(get_adapter):
    # Get adapters
    MCDR = get_adapter("MCDR")
    two_afc = get_adapter("2AFC")
    return MCDR, two_afc


@app.cell
def _(MCDR, data_path, pl, two_afc):
    # Import data
    df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "df_alexis_drug_combined.parquet"))  # With drug
    df_2AFC_delay = pl.read_parquet(data_path / "tiffany.parquet")
    df_MCDR = MCDR.subject_filter(pl.read_parquet(data_path / "MCDR_all.parquet"))
    return df_2AFC, df_2AFC_delay, df_MCDR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Accuracy across drug sessions
    """)
    return


@app.cell
def _(df_2AFC, df_2AFC_delay, df_MCDR, label_condition_session_windows, pl):
    df_2ADC_session_windows = label_condition_session_windows(
        df_2AFC_delay.filter(pl.col("drug").is_in(["Saline", "NR2B"])),
        subject_col="subject",
        session_col="session",
        condition_col="drug",
        late_sessions=None,
        late_from_session=3,
        late_label="Late (≥3)",
    )
    df_2AFC_session_windows = label_condition_session_windows(
        df_2AFC,
        subject_col="subject",
        session_col="Session",
        condition_col="Drug",
        order_col="Date",
        late_sessions=None,
        late_from_session=3,
        late_label="Late (≥3)",
    )
    df_MCDR_session_windows = label_condition_session_windows(
        df_MCDR.filter(pl.col("drug").is_in(["saline", "drug"])),
        subject_col="subject",
        session_col="session",
        condition_col="drug",
        late_sessions=None,
        late_from_session=3,
        late_label="Late (≥3)",
    )

    summary_2ADC = (
        df_2ADC_session_windows.filter(pl.col("session_window").is_not_null())
        .select("subject", "drug", "session", "session_window")
        .unique()
        .with_columns(
            pl.lit("2ADC").alias("Task"),
            pl.col("drug").replace({"NR2B": "Drug"}).alias("Condition"),
        )
        .group_by("Task", "Condition", "session_window")
        .agg(
            pl.col("subject").n_unique().alias("Animals"),
            pl.len().alias("Sessions"),
        )
    )
    summary_2AFC = (
        df_2AFC_session_windows.filter(pl.col("session_window").is_not_null())
        .select("subject", "Drug", "Session", "session_window")
        .unique()
        .with_columns(
            pl.lit("2AFC").alias("Task"),
            pl.when(pl.col("Drug") == 0)
            .then(pl.lit("Saline"))
            .otherwise(pl.lit("Drug"))
            .alias("Condition"),
        )
        .group_by("Task", "Condition", "session_window")
        .agg(
            pl.col("subject").n_unique().alias("Animals"),
            pl.len().alias("Sessions"),
        )
    )
    summary_MCDR = (
        df_MCDR_session_windows.filter(pl.col("session_window").is_not_null())
        .select("subject", "drug", "session", "session_window")
        .unique()
        .with_columns(
            pl.lit("MCDR").alias("Task"),
            pl.col("drug").replace({"saline": "Saline", "drug": "Drug"}).alias("Condition"),
        )
        .group_by("Task", "Condition", "session_window")
        .agg(
            pl.col("subject").n_unique().alias("Animals"),
            pl.len().alias("Sessions"),
        )
    )
    session_window_summary = pl.concat(
        [summary_2ADC, summary_2AFC, summary_MCDR],
        how="vertical_relaxed",
    ).sort("Task", "Condition", "session_window")
    return (
        df_2ADC_session_windows,
        df_2AFC_session_windows,
        df_MCDR_session_windows,
        session_window_summary,
    )


@app.cell
def _(session_window_summary):
    session_window_summary
    return


@app.cell
def _(
    df_2ADC_session_windows,
    df_2AFC_session_windows,
    df_MCDR_session_windows,
    pl,
    prepare_session_rolling_accuracy,
    ttest_rel,
):
    rolling_2ADC = prepare_session_rolling_accuracy(
        df_2ADC_session_windows.with_columns(
            pl.col("drug").replace({"NR2B": "Drug"}).alias("condition_label")
        ),
        subject_col="subject",
        session_col="session",
        trial_col="trial",
        accuracy_col="hit",
        condition_col="condition_label",
        rolling_window=20,
    )
    rolling_2AFC = prepare_session_rolling_accuracy(
        df_2AFC_session_windows.with_columns(
            pl.when(pl.col("Drug") == 0)
            .then(pl.lit("Saline"))
            .otherwise(pl.lit("Drug"))
            .alias("condition_label")
        ),
        subject_col="subject",
        session_col="Session",
        trial_col="Trial",
        accuracy_col="Hit",
        condition_col="condition_label",
        rolling_window=20,
    )
    rolling_MCDR = prepare_session_rolling_accuracy(
        df_MCDR_session_windows.with_columns(
            pl.col("drug")
            .replace({"saline": "Saline", "drug": "Drug"})
            .alias("condition_label")
        ),
        subject_col="subject",
        session_col="session",
        trial_col="trial",
        accuracy_col="performance",
        condition_col="condition_label",
        rolling_window=20,
    )

    rolling_summaries = {
        "2ADC": rolling_2ADC["summary"],
        "2AFC": rolling_2AFC["summary"],
        "MCDR": rolling_MCDR["summary"],
    }

    session_accuracy_by_number = pl.concat(
        [
            df_2ADC_session_windows.group_by(
                "subject", "drug", "session", "condition_session_number"
            )
            .agg(pl.col("hit").cast(pl.Float64).mean().alias("accuracy"))
            .with_columns(
                pl.lit("2ADC").alias("Task"),
                pl.col("drug").replace({"NR2B": "Drug"}).alias("Condition"),
            ),
            df_2AFC_session_windows.group_by(
                "subject", "Drug", "Session", "condition_session_number"
            )
            .agg(pl.col("Hit").cast(pl.Float64).mean().alias("accuracy"))
            .with_columns(
                pl.lit("2AFC").alias("Task"),
                pl.when(pl.col("Drug") == 0)
                .then(pl.lit("Saline"))
                .otherwise(pl.lit("Drug"))
                .alias("Condition"),
            )
            .rename({"Session": "session"}),
            df_MCDR_session_windows.group_by(
                "subject", "drug", "session", "condition_session_number"
            )
            .agg(pl.col("performance").cast(pl.Float64).mean().alias("accuracy"))
            .with_columns(
                pl.lit("MCDR").alias("Task"),
                pl.col("drug")
                .replace({"saline": "Saline", "drug": "Drug"})
                .alias("Condition"),
            ),
        ],
        how="diagonal_relaxed",
    ).select(
        "Task",
        "subject",
        "Condition",
        "session",
        "condition_session_number",
        "accuracy",
    ).with_columns(
        pl.col("subject")
        .n_unique()
        .over("Task", "Condition", "condition_session_number")
        .alias("n_subjects")
    ).filter(
        pl.col("n_subjects") >= 2
    ).to_pandas()

    variance_subject_dfs = {}
    for _task, _rolling in {
        "2ADC": rolling_2ADC,
        "2AFC": rolling_2AFC,
        "MCDR": rolling_MCDR,
    }.items():
        variance_subject_dfs[_task] = (
            _rolling["session_traces"]
            .groupby(["subject", "condition", "session"], observed=True)[
                "rolling_accuracy"
            ]
            .var()
            .groupby(["subject", "condition"], observed=True)
            .mean()
            .rename("mean_session_variance")
            .reset_index()
        )

    variance_pairs = [("Saline", "Drug")]
    variance_pvalues = {}
    for _task, _variance_df in variance_subject_dfs.items():
        _paired = (
            _variance_df.pivot(
                index="subject",
                columns="condition",
                values="mean_session_variance",
            )
            .reindex(columns=["Saline", "Drug"])
            .dropna()
        )
        variance_pvalues[_task] = [
            ttest_rel(_paired["Saline"], _paired["Drug"]).pvalue
        ]

    rolling_session_counts = pl.concat(
        [
            pl.from_pandas(rolling_2ADC["session_counts"]).with_columns(
                pl.lit("2ADC").alias("Task")
            ),
            pl.from_pandas(rolling_2AFC["session_counts"]).with_columns(
                pl.lit("2AFC").alias("Task")
            ),
            pl.from_pandas(rolling_MCDR["session_counts"]).with_columns(
                pl.lit("MCDR").alias("Task")
            ),
        ],
        how="vertical_relaxed",
    ).select(
        "Task",
        "condition",
        "session_window",
        "included_subjects",
        "included_sessions",
    )
    return (
        rolling_session_counts,
        rolling_summaries,
        session_accuracy_by_number,
        variance_pairs,
        variance_pvalues,
        variance_subject_dfs,
    )


@app.cell
def _(rolling_session_counts):
    rolling_session_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2ADC rolling accuracy

    Sessions are averaged within subject first. Line: across-subject mean. Shaded
    band: one across-subject standard deviation.
    """)
    return


@app.cell
def _(
    axd,
    mount_figure,
    np,
    path_panels,
    plt,
    rolling_figsize,
    rolling_summaries,
    session_window_handles,
    session_window_palette,
    sns,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    rolling_accuracy_mean_2ADC = plt.gca() if not mount_figure else axd["rolling_accuracy_mean_2ADC"]
    rolling_accuracy_mean_2ADC.clear()
    for (_condition, _window), _color in session_window_palette.items():
        _group = rolling_summaries["2ADC"].loc[
            (rolling_summaries["2ADC"]["condition"] == _condition)
            & (rolling_summaries["2ADC"]["session_window"] == _window)
        ]
        _x = _group["session_progress"].to_numpy(dtype=float)
        _mean = _group["mean_accuracy"].to_numpy(dtype=float)
        _std = _group["std_accuracy"].fillna(0).to_numpy(dtype=float)
        rolling_accuracy_mean_2ADC.plot(_x, _mean, color=_color)
        rolling_accuracy_mean_2ADC.fill_between(
            _x,
            np.clip(_mean - _std, 0, 1),
            np.clip(_mean + _std, 0, 1),
            color=_color,
            alpha=0.12,
            edgecolor="none",
        )
    rolling_accuracy_mean_2ADC.axhline(0.5, color="gray", ls="--")
    rolling_accuracy_mean_2ADC.set(
        xlabel="Session progress (%)",
        ylabel="Rolling accuracy",
        xlim=(0, 100),
        ylim=(0, 1),
    )
    rolling_accuracy_mean_2ADC.legend(
        handles=session_window_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=rolling_accuracy_mean_2ADC)
    if not mount_figure:
        rolling_accuracy_mean_2ADC.figure.savefig(
            path_panels / "rolling_accuracy_mean_2ADC.svg"
        )
        rolling_accuracy_mean_2ADC.figure.savefig(
            path_panels / "rolling_accuracy_mean_2ADC.png"
        )
    rolling_accuracy_mean_2ADC
    return


@app.cell
def _(
    axd,
    condition_order,
    condition_palette,
    mount_figure,
    path_panels,
    plt,
    rolling_figsize,
    session_accuracy_by_number,
    sns,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    session_accuracy_2ADC = plt.gca() if not mount_figure else axd["session_accuracy_2ADC"]
    session_accuracy_2ADC.clear()
    sns.lineplot(
        data=session_accuracy_by_number.query("Task == '2ADC'"),
        x="condition_session_number",
        y="accuracy",
        hue="Condition",
        hue_order=condition_order,
        palette=condition_palette,
        marker="o",
        markeredgewidth=0,
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        ax=session_accuracy_2ADC,
    )
    session_accuracy_2ADC.set(
        xlabel="Session #",
        ylabel="Mean accuracy",
        ylim=(0.5, 1),
    )
    session_accuracy_2ADC.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=session_accuracy_2ADC)
    if not mount_figure:
        session_accuracy_2ADC.figure.savefig(
            path_panels / "session_accuracy_2ADC.svg"
        )
        session_accuracy_2ADC.figure.savefig(
            path_panels / "session_accuracy_2ADC.png"
        )
    session_accuracy_2ADC
    return


@app.cell
def _(
    Annotator,
    BOXPLOT_STYLE,
    axd,
    condition_order,
    condition_palette,
    mount_figure,
    path_panels,
    plt,
    rolling_figsize,
    sns,
    variance_pairs,
    variance_pvalues,
    variance_subject_dfs,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    whole_session_variance_subject_2ADC = plt.gca() if not mount_figure else axd["whole_session_variance_subject_2ADC"]
    whole_session_variance_subject_2ADC.clear()
    sns.boxplot(
        data=variance_subject_dfs["2ADC"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
        hue="condition",
        hue_order=condition_order,
        palette=condition_palette,
        dodge=False,
        width=0.6,
        legend=False,
        ax=whole_session_variance_subject_2ADC,
        **BOXPLOT_STYLE,
    )
    sns.stripplot(
        data=variance_subject_dfs["2ADC"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
        hue="condition",
        hue_order=condition_order,
        palette=condition_palette,
        alpha=0.65,
        dodge=False,
        jitter=0.14,
        legend=False,
        size=3,
        zorder=3,
        ax=whole_session_variance_subject_2ADC,
    )
    whole_session_variance_subject_2ADC.set(
        xlabel="",
        ylabel="Mean within-session variance",
        ylim=(0, None),
    )
    Annotator(
        whole_session_variance_subject_2ADC,
        variance_pairs,
        data=variance_subject_dfs["2ADC"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
    ).configure(
        text_format="star",
        loc="inside",
        verbose=0,
    ).set_pvalues(
        variance_pvalues["2ADC"]
    ).annotate()
    sns.despine(ax=whole_session_variance_subject_2ADC)
    if not mount_figure:
        whole_session_variance_subject_2ADC.figure.savefig(
            path_panels / "mean_session_variance_subject_2ADC.svg"
        )
        whole_session_variance_subject_2ADC.figure.savefig(
            path_panels / "mean_session_variance_subject_2ADC.png"
        )
    whole_session_variance_subject_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2AFC rolling accuracy

    Sessions are averaged within subject first. Line: across-subject mean. Shaded
    band: one across-subject standard deviation.
    """)
    return


@app.cell
def _(
    axd,
    mount_figure,
    np,
    path_panels,
    plt,
    rolling_figsize,
    rolling_summaries,
    session_window_handles,
    session_window_palette,
    sns,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    rolling_accuracy_mean_2AFC = plt.gca() if not mount_figure else axd["rolling_accuracy_mean_2AFC"]
    rolling_accuracy_mean_2AFC.clear()
    for (_condition, _window), _color in session_window_palette.items():
        _group = rolling_summaries["2AFC"].loc[
            (rolling_summaries["2AFC"]["condition"] == _condition)
            & (rolling_summaries["2AFC"]["session_window"] == _window)
        ]
        _x = _group["session_progress"].to_numpy(dtype=float)
        _mean = _group["mean_accuracy"].to_numpy(dtype=float)
        _std = _group["std_accuracy"].fillna(0).to_numpy(dtype=float)
        rolling_accuracy_mean_2AFC.plot(_x, _mean, color=_color)
        rolling_accuracy_mean_2AFC.fill_between(
            _x,
            np.clip(_mean - _std, 0, 1),
            np.clip(_mean + _std, 0, 1),
            color=_color,
            alpha=0.12,
            edgecolor="none",
        )
    rolling_accuracy_mean_2AFC.axhline(0.5, color="gray", ls="--")
    rolling_accuracy_mean_2AFC.set(
        xlabel="Session progress (%)",
        ylabel="Rolling accuracy",
        xlim=(0, 100),
        ylim=(0, 1),
    )
    rolling_accuracy_mean_2AFC.legend(
        handles=session_window_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=rolling_accuracy_mean_2AFC)
    if not mount_figure:
        rolling_accuracy_mean_2AFC.figure.savefig(
            path_panels / "rolling_accuracy_mean_2AFC.svg"
        )
        rolling_accuracy_mean_2AFC.figure.savefig(
            path_panels / "rolling_accuracy_mean_2AFC.png"
        )
    rolling_accuracy_mean_2AFC
    return


@app.cell
def _(
    axd,
    condition_order,
    condition_palette,
    mount_figure,
    path_panels,
    plt,
    rolling_figsize,
    session_accuracy_by_number,
    sns,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    session_accuracy_2AFC = plt.gca() if not mount_figure else axd["session_accuracy_2AFC"]
    session_accuracy_2AFC.clear()
    sns.lineplot(
        data=session_accuracy_by_number.query("Task == '2AFC'"),
        x="condition_session_number",
        y="accuracy",
        hue="Condition",
        hue_order=condition_order,
        palette=condition_palette,
        marker="o",
        markeredgewidth=0,
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        ax=session_accuracy_2AFC,
    )
    session_accuracy_2AFC.set(
        xlabel="Session #",
        ylabel="Mean accuracy",
        ylim=(0.5, 1),
    )
    session_accuracy_2AFC.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=session_accuracy_2AFC)
    if not mount_figure:
        session_accuracy_2AFC.figure.savefig(
            path_panels / "session_accuracy_2AFC.svg"
        )
        session_accuracy_2AFC.figure.savefig(
            path_panels / "session_accuracy_2AFC.png"
        )
    session_accuracy_2AFC
    return


@app.cell
def _(
    Annotator,
    BOXPLOT_STYLE,
    axd,
    condition_order,
    condition_palette,
    mount_figure,
    path_panels,
    plt,
    rolling_figsize,
    sns,
    variance_pairs,
    variance_pvalues,
    variance_subject_dfs,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    whole_session_variance_subject_2AFC = plt.gca() if not mount_figure else axd["whole_session_variance_subject_2AFC"]
    whole_session_variance_subject_2AFC.clear()
    sns.boxplot(
        data=variance_subject_dfs["2AFC"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
        hue="condition",
        hue_order=condition_order,
        palette=condition_palette,
        dodge=False,
        width=0.6,
        legend=False,
        ax=whole_session_variance_subject_2AFC,
        **BOXPLOT_STYLE,
    )
    sns.stripplot(
        data=variance_subject_dfs["2AFC"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
        hue="condition",
        hue_order=condition_order,
        palette=condition_palette,
        alpha=0.65,
        dodge=False,
        jitter=0.14,
        legend=False,
        size=3,
        zorder=3,
        ax=whole_session_variance_subject_2AFC,
    )
    whole_session_variance_subject_2AFC.set(
        xlabel="",
        ylabel="Mean within-session variance",
        ylim=(0, None),
    )
    Annotator(
        whole_session_variance_subject_2AFC,
        variance_pairs,
        data=variance_subject_dfs["2AFC"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
    ).configure(
        text_format="star",
        loc="inside",
        verbose=0,
    ).set_pvalues(
        variance_pvalues["2AFC"]
    ).annotate()
    sns.despine(ax=whole_session_variance_subject_2AFC)
    if not mount_figure:
        whole_session_variance_subject_2AFC.figure.savefig(
            path_panels / "mean_session_variance_subject_2AFC.svg"
        )
        whole_session_variance_subject_2AFC.figure.savefig(
            path_panels / "mean_session_variance_subject_2AFC.png"
        )
    whole_session_variance_subject_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MCDR rolling accuracy

    Sessions are averaged within subject first. Line: across-subject mean. Shaded
    band: one across-subject standard deviation.
    """)
    return


@app.cell
def _(
    axd,
    mount_figure,
    np,
    path_panels,
    plt,
    rolling_figsize,
    rolling_summaries,
    session_window_handles,
    session_window_palette,
    sns,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    rolling_accuracy_mean_MCDR = plt.gca() if not mount_figure else axd["rolling_accuracy_mean_MCDR"]
    rolling_accuracy_mean_MCDR.clear()
    for (_condition, _window), _color in session_window_palette.items():
        _group = rolling_summaries["MCDR"].loc[
            (rolling_summaries["MCDR"]["condition"] == _condition)
            & (rolling_summaries["MCDR"]["session_window"] == _window)
        ]
        _x = _group["session_progress"].to_numpy(dtype=float)
        _mean = _group["mean_accuracy"].to_numpy(dtype=float)
        _std = _group["std_accuracy"].fillna(0).to_numpy(dtype=float)
        rolling_accuracy_mean_MCDR.plot(_x, _mean, color=_color)
        rolling_accuracy_mean_MCDR.fill_between(
            _x,
            np.clip(_mean - _std, 0, 1),
            np.clip(_mean + _std, 0, 1),
            color=_color,
            alpha=0.12,
            edgecolor="none",
        )
    rolling_accuracy_mean_MCDR.axhline(1 / 3, color="gray", ls="--")
    rolling_accuracy_mean_MCDR.set(
        xlabel="Session progress (%)",
        ylabel="Rolling accuracy",
        xlim=(0, 100),
        ylim=(0, 1),
    )
    rolling_accuracy_mean_MCDR.legend(
        handles=session_window_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=rolling_accuracy_mean_MCDR)
    if not mount_figure:
        rolling_accuracy_mean_MCDR.figure.savefig(
            path_panels / "rolling_accuracy_mean_MCDR.svg"
        )
        rolling_accuracy_mean_MCDR.figure.savefig(
            path_panels / "rolling_accuracy_mean_MCDR.png"
        )
    rolling_accuracy_mean_MCDR
    return


@app.cell
def _(
    axd,
    condition_order,
    condition_palette,
    mount_figure,
    path_panels,
    plt,
    rolling_figsize,
    session_accuracy_by_number,
    sns,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    session_accuracy_MCDR = plt.gca() if not mount_figure else axd["session_accuracy_MCDR"]
    session_accuracy_MCDR.clear()
    sns.lineplot(
        data=session_accuracy_by_number.query("Task == 'MCDR'"),
        x="condition_session_number",
        y="accuracy",
        hue="Condition",
        hue_order=condition_order,
        palette=condition_palette,
        marker="o",
        markeredgewidth=0,
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        ax=session_accuracy_MCDR,
    )
    session_accuracy_MCDR.set(
        xlabel="Session #",
        ylabel="Mean accuracy",
        ylim=(0.3, 0.8),
    )
    session_accuracy_MCDR.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    sns.despine(ax=session_accuracy_MCDR)
    if not mount_figure:
        session_accuracy_MCDR.figure.savefig(
            path_panels / "session_accuracy_MCDR.svg"
        )
        session_accuracy_MCDR.figure.savefig(
            path_panels / "session_accuracy_MCDR.png"
        )
    session_accuracy_MCDR
    return


@app.cell
def _(
    Annotator,
    BOXPLOT_STYLE,
    axd,
    condition_order,
    condition_palette,
    mount_figure,
    path_panels,
    plt,
    rolling_figsize,
    sns,
    variance_pairs,
    variance_pvalues,
    variance_subject_dfs,
):
    plt.figure(figsize=rolling_figsize, constrained_layout=True)
    whole_session_variance_subject_MCDR = plt.gca() if not mount_figure else axd["whole_session_variance_subject_MCDR"]
    whole_session_variance_subject_MCDR.clear()
    sns.boxplot(
        data=variance_subject_dfs["MCDR"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
        hue="condition",
        hue_order=condition_order,
        palette=condition_palette,
        dodge=False,
        width=0.6,
        legend=False,
        ax=whole_session_variance_subject_MCDR,
        **BOXPLOT_STYLE,
    )
    sns.stripplot(
        data=variance_subject_dfs["MCDR"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
        hue="condition",
        hue_order=condition_order,
        palette=condition_palette,
        alpha=0.65,
        dodge=False,
        jitter=0.14,
        legend=False,
        size=3,
        zorder=3,
        ax=whole_session_variance_subject_MCDR,
    )
    whole_session_variance_subject_MCDR.set(
        xlabel="",
        ylabel="Mean within-session variance",
        ylim=(0, None),
    )
    Annotator(
        whole_session_variance_subject_MCDR,
        variance_pairs,
        data=variance_subject_dfs["MCDR"],
        x="condition",
        y="mean_session_variance",
        order=condition_order,
    ).configure(
        text_format="star",
        loc="inside",
        verbose=0,
    ).set_pvalues(
        variance_pvalues["MCDR"]
    ).annotate()
    sns.despine(ax=whole_session_variance_subject_MCDR)
    if not mount_figure:
        whole_session_variance_subject_MCDR.figure.savefig(
            path_panels / "mean_session_variance_subject_MCDR.svg"
        )
        whole_session_variance_subject_MCDR.figure.savefig(
            path_panels / "mean_session_variance_subject_MCDR.png"
        )
    whole_session_variance_subject_MCDR
    return


@app.cell
def _(axd, fig, mount_figure, path_panels):
    if mount_figure:
        for _name, _ax in axd.items():
            _legend = _ax.get_legend()
            if _legend is not None and _name not in {
                "rolling_accuracy_mean_2ADC",
                "session_accuracy_2ADC",
            }:
                _legend.remove()

        for _name, _ax in axd.items():
            _ax.set_ylabel("")
            _ax.set_title("")

        axd["rolling_accuracy_mean_2ADC"].set_title("2ADC")
        axd["rolling_accuracy_mean_2AFC"].set_title("2AFC")
        axd["rolling_accuracy_mean_MCDR"].set_title("MCDR")

        axd["rolling_accuracy_mean_2ADC"].set_ylabel("Rolling accuracy")
        axd["session_accuracy_2ADC"].set_ylabel("Mean accuracy")
        axd["whole_session_variance_subject_2ADC"].set_ylabel(
            "Mean within-session variance"
        )

        fig.align_labels()
        fig.savefig(path_panels / "supplementary_figure12.svg")
        fig.savefig(path_panels / "supplementary_figure12.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure12.pdf")
    fig
    return


if __name__ == "__main__":
    app.run()
