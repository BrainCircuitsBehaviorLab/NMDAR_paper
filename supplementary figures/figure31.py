# /// script
# [tool.marimo.opengraph]
# title = "Supplementary Figure 4"
# description = "GLM-relative model comparisons for 2ADC and 2AFC."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3.1

    ## Description

    Model comparison wrt number of previous choices used and number of states in the GLMHMM model.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import math
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_1samp, ttest_rel
    from src.plots.common import fig_size

    from glmhmmt.runtime import configure_paths, get_runtime_paths

    return (
        Path,
        configure_paths,
        fig_size,
        get_runtime_paths,
        math,
        mo,
        os,
        pl,
        plt,
        sns,
        ttest_1samp,
        ttest_rel,
    )


@app.cell
def _():
    mount_figure = True
    return (mount_figure,)


@app.cell
def _(Path, configure_paths, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    path_panels = ROOT / "supplementary figures" / "panels31"
    for panel_format in ("svg", "png"):
        os.makedirs(path_panels / panel_format, exist_ok=True)
    return ROOT, path_panels, paths


@app.cell
def _(ROOT, plt, sns):
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    freeze_model_order = ["Free", "Both0", "Stim0", "Hist0"]
    family_palette = {"free": "black", "frozen": "black"}
    return family_palette, freeze_model_order


@app.cell
def _(paths):
    TASK_CONFIGS = {
        "2ADC": {
            "task": "2AFC_delay",
            "score_column": "ll_per_trial",
            "lag_prefix": "2adc_bias-stim-choice_lags_",
            "freeze_models": [
                ("Free", "2adc_bias-stim-choice-lag-param_free"),
                ("Both0", "2adc_bias-stimD0-choice-lag-paramE0"),
                ("Stim0", "2adc_bias-stimD0-choice-lag-param_free"),
                ("Hist0", "2adc_bias-stim_free-choice-lag-paramE0"),
            ],
            "state_free_dir": paths.RESULTS / "fits" / "model_comparison" / "2ADC",
            "state_frozen_dir": paths.RESULTS / "fits" / "model_comparison" / "2ADC" / "param_half_frozen",
        },
        "2AFC": {
            "task": "2AFC",
            "score_column": "test_ll_per_trial_mean",
            "lag_prefix": "2afc_bias-stim-choice_lags_",
            "freeze_models": [
                ("Free", "2afc_bias-stim-choice-lag-param_free"),
                ("Both0", "2afc_bias-stimD0-choice-lag-paramE0"),
                ("Stim0", "2afc_bias-stimD0-choice-lag-param_free"),
                ("Hist0", "2afc_bias-stim_free-choice-lag-paramE0"),
            ],
            "state_free_dir": paths.RESULTS / "fits" / "model_comparison" / "2AFC",
            "state_frozen_dir": paths.RESULTS / "fits" / "model_comparison" / "2AFC" / "param_half_frozen",
        },
    }
    for _config in TASK_CONFIGS.values():
        _config["glm_dir"] = paths.RESULTS / "fits" / _config["task"] / "glm" / "one hot"
        _config["freeze_root"] = paths.RESULTS / "fits" / _config["task"] / "glmhmm"
        _config["ashwood_dir"] = _config["freeze_root"] / "ashwood"
    return (TASK_CONFIGS,)


@app.cell
def _(math, pl, ttest_1samp, ttest_rel):
    metric_schema = {
        "subject": pl.Utf8,
        "K": pl.Int64,
        "ll_per_trial": pl.Float64,
        "test_ll_per_trial_mean": pl.Float64,
        "bic": pl.Float64,
    }

    def read_metric_directory(directory, pattern):
        frames = [pl.read_parquet(path) for path in sorted(directory.glob(pattern))]
        if not frames:
            return pl.DataFrame(schema=metric_schema)

        metrics = pl.concat(frames, how="diagonal_relaxed")
        if "K" not in metrics.columns:
            metrics = metrics.with_columns(
                pl.col("k").cast(pl.Int64).alias("K")
                if "k" in metrics.columns
                else pl.lit(1, dtype=pl.Int64).alias("K")
            )
        for column, dtype in metric_schema.items():
            if column not in metrics.columns:
                metrics = metrics.with_columns(pl.lit(None, dtype=dtype).alias(column))
        return metrics.select(list(metric_schema)).with_columns(
            pl.col("subject").cast(pl.Utf8),
            pl.col("K").cast(pl.Int64),
        )

    def clean_plot_edges(axis):
        for line in axis.lines:
            line.set_markeredgewidth(0)
            line.set_markeredgecolor("none")
        for collection in axis.collections:
            collection.set_edgecolor("none")

    def significance_stars(pvalue):
        if not math.isfinite(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_one_sample_zero_annotations(axis, dataframe, *, x, y, order):
        finite_values = dataframe[y].dropna()
        if finite_values.empty:
            return
        value_range = float(finite_values.max() - finite_values.min())
        padding = max(value_range * 0.08, 1e-6)
        text_y = float(finite_values.max()) + padding
        axis.set_ylim(top=text_y + padding)
        for x_index, x_value in enumerate(order):
            values = dataframe.loc[dataframe[x] == x_value, y].dropna()
            if len(values) < 2 or float(values.std()) == 0:
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=0.0).pvalue)
            label = significance_stars(pvalue)
            if label:
                axis.text(x_index, text_y, label, ha="center", va="bottom")

    def add_model_pair_annotations(axis, dataframe, *, y, order, pairs):
        finite_values = dataframe[y].dropna()
        if finite_values.empty:
            return
        value_range = float(finite_values.max() - finite_values.min())
        padding = max(value_range * 0.08, 1e-6)
        positions = {label: index for index, label in enumerate(order)}
        tested_pairs = []
        for left, right in pairs:
            paired = (
                dataframe.loc[dataframe["model"].isin([left, right]), ["subject", "model", y]]
                .pivot_table(index="subject", columns="model", values=y, aggfunc="first")
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[left], paired[right]).pvalue)
            if math.isfinite(pvalue):
                tested_pairs.append((left, right, significance_stars(pvalue) or "ns"))
        base_y = float(finite_values.max()) + padding
        axis.set_ylim(top=base_y + padding * (len(tested_pairs) + 1))
        for pair_index, (left, right, label) in enumerate(tested_pairs):
            left_position = positions[left]
            right_position = positions[right]
            line_y = base_y + pair_index * padding
            cap = padding * 0.25
            axis.plot(
                [left_position, right_position],
                [line_y + cap, line_y + cap],
                color="0.35",
                linewidth=0.7,
            )
            axis.text(
                (left_position + right_position) / 2,
                line_y + cap * 1.2,
                label,
                ha="center",
                va="bottom",
            )

    def add_numeric_pair_annotations(axis, dataframe, *, y, pairs):
        finite_values = dataframe[y].dropna()
        if finite_values.empty:
            return
        value_range = float(finite_values.max() - finite_values.min())
        padding = max(value_range * 0.08, 1e-6)
        tested_pairs = []
        for left, right in pairs:
            paired = (
                dataframe.loc[dataframe["plot_order"].isin([left, right]), ["subject", "plot_order", y]]
                .pivot_table(index="subject", columns="plot_order", values=y, aggfunc="first")
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[left], paired[right]).pvalue)
            if math.isfinite(pvalue):
                tested_pairs.append((left, right, significance_stars(pvalue) or "ns"))
        base_y = float(finite_values.max()) + padding
        axis.set_ylim(top=base_y + padding * (len(tested_pairs) + 1))
        for pair_index, (left, right, label) in enumerate(tested_pairs):
            line_y = base_y + pair_index * padding
            cap = padding * 0.25
            axis.plot(
                [left, right],
                [line_y + cap, line_y + cap],
                color="0.35",
                linewidth=0.7,
            )
            axis.text((left + right) / 2, line_y + cap * 1.2, label, ha="center", va="bottom")

    def add_state_comparison_annotations(axis, dataframe, *, y):
        finite_values = dataframe[y].dropna()
        if finite_values.empty:
            return
        value_range = float(finite_values.max() - finite_values.min())
        padding = max(value_range * 0.08, 1e-6)
        state_counts = sorted(dataframe["K"].unique())
        pairs = (
            ([(1, 2)] if 1 in state_counts and 2 in state_counts else []) 
            + [(2,3)]
            # + [(2, state_count) for state_count in state_counts if state_count > 2]
        )
        tested_pairs = []
        for left, right in pairs:
            paired = (
                dataframe.loc[dataframe["K"].isin([left, right]), ["subject", "K", y]]
                .pivot_table(index="subject", columns="K", values=y, aggfunc="first")
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[right], paired[left]).pvalue)
            if math.isfinite(pvalue):
                tested_pairs.append((left, right, significance_stars(pvalue) or "ns"))
        base_y = float(finite_values.max()) + padding
        axis.set_ylim(top=base_y + padding * (len(tested_pairs) + 1))
        for pair_index, (left, right, label) in enumerate(tested_pairs):
            line_y = base_y + pair_index * padding
            cap = padding * 0.25
            axis.plot(
                [left, right],
                [line_y + cap, line_y + cap],
                color="0.35",
                linewidth=0.7,
            )
            axis.text((left + right) / 2, line_y + cap * 1.2, label, ha="center", va="bottom")

    def plot_ashwood_point(axis, dataframe, metric):
        values = dataframe[metric].dropna()
        if values.empty:
            return
        sem = values.std() / len(values) ** 0.5
        axis.errorbar(
            3,
            values.mean(),
            yerr=sem,
            fmt="o",
            markersize=4,
            markerfacecolor="tab:orange",
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor="black",
            elinewidth=1,
            capsize=0,
            linestyle="none",
            zorder=6,
        )

    def align_zero_fraction(axes, fraction):
        for axis in axes:
            bottom, top = axis.get_ylim()
            scale = max(
                -bottom / fraction if bottom < 0 else 0,
                top / (1 - fraction) if top > 0 else 0,
                1e-6,
            )
            axis.set_ylim(-fraction * scale, (1 - fraction) * scale)

    def zoom_zero_centered(axis, dataframe, metric, quantile=0.95):
        values = dataframe[metric].dropna().abs()
        if values.empty:
            axis.set_ylim(-1, 1)
            return
        annotation_top = max(float(axis.get_ylim()[1]), 0)
        limit = max(
            float(values.quantile(quantile)) * 1.1,
            annotation_top * 1.05,
            1,
        )
        axis.set_ylim(-limit, limit)

    return (
        add_model_pair_annotations,
        add_numeric_pair_annotations,
        add_one_sample_zero_annotations,
        add_state_comparison_annotations,
        align_zero_fraction,
        clean_plot_edges,
        plot_ashwood_point,
        read_metric_directory,
        zoom_zero_centered,
    )


@app.cell
def _(TASK_CONFIGS, math, pl, read_metric_directory):
    ashwood_plot_dfs = {}
    freeze_plot_dfs = {}
    lag_plot_dfs = {}
    state_plot_dfs = {"free": {}, "frozen": {}}
    count_rows = []

    for task_label, _config in TASK_CONFIGS.items():
        score_column = _config["score_column"]
        glm_metrics = read_metric_directory(_config["glm_dir"], "*_glm_metrics.parquet")
        glm_baseline = glm_metrics.select(
            "subject",
            pl.col(score_column).alias("glm_ll"),
            pl.col("bic").alias("glm_bic"),
        ).drop_nulls(["glm_ll", "glm_bic"])

        lag_frames = []
        for n_regressors in range(1, 11):
            lag_metrics = read_metric_directory(
                _config["freeze_root"] / f'{_config["lag_prefix"]}{n_regressors:02d}',
                "*_metrics.parquet",
            )
            lag_frames.append(
                lag_metrics.with_columns(
                    pl.lit(n_regressors, dtype=pl.Int64).alias("n_regressors"),
                    pl.lit(n_regressors, dtype=pl.Int64).alias("plot_order"),
                    pl.lit(str(n_regressors)).alias("regressor_label"),
                )
            )

        full_model_metrics = read_metric_directory(
            _config["freeze_root"] / _config["freeze_models"][0][1],
            "*_metrics.parquet",
        ).with_columns(
            pl.lit(15, dtype=pl.Int64).alias("n_regressors"),
            pl.lit(12, dtype=pl.Int64).alias("plot_order"),
            pl.lit("15").alias("regressor_label"),
        )
        lag_metrics = pl.concat([*lag_frames, full_model_metrics], how="diagonal_relaxed")
        lag_deltas = (
            lag_metrics
            .join(glm_baseline, on="subject", how="inner")
            .with_columns(
                ((pl.col(score_column) - pl.col("glm_ll")) / math.log(2)).alias("delta_ll_vs_glm")
            )
            .select(
                "subject",
                "n_regressors",
                "plot_order",
                "regressor_label",
                "delta_ll_vs_glm",
            )
        )
        glm_zero = glm_baseline.select(
            "subject",
            pl.lit(0, dtype=pl.Int64).alias("n_regressors"),
            pl.lit(0, dtype=pl.Int64).alias("plot_order"),
            pl.lit("GLM").alias("regressor_label"),
            pl.lit(0.0).alias("delta_ll_vs_glm"),
        )
        lag_with_glm = pl.concat([glm_zero, lag_deltas]).sort(["plot_order", "subject"])
        lag_plot_dfs[task_label] = lag_with_glm.to_pandas()
        count_rows.append(
            {
                "task": task_label,
                "comparison": "action-trace regressors",
                "subjects": lag_deltas.get_column("subject").n_unique(),
                "rows": lag_deltas.height,
            }
        )

        ashwood_metrics = read_metric_directory(
            _config["ashwood_dir"],
            "*_K3_glmhmm_metrics.parquet",
        )
        ashwood_deltas = (
            ashwood_metrics
            .join(glm_baseline, on="subject", how="inner")
            .with_columns(
                ((pl.col(score_column) - pl.col("glm_ll")) / math.log(2)).alias("delta_ll_vs_glm"),
                (pl.col("bic") - pl.col("glm_bic")).alias("delta_bic_vs_glm"),
            )
            .select("subject", "K", "delta_ll_vs_glm", "delta_bic_vs_glm")
        )
        ashwood_plot_dfs[task_label] = ashwood_deltas.to_pandas()
        count_rows.append(
            {
                "task": task_label,
                "comparison": "Ashwood K=3",
                "subjects": ashwood_deltas.get_column("subject").n_unique(),
                "rows": ashwood_deltas.height,
            }
        )

        freeze_frames = []
        for model_order, (model_label, model_id) in enumerate(_config["freeze_models"]):
            model_metrics = read_metric_directory(
                _config["freeze_root"] / model_id,
                "*_metrics.parquet",
            )
            freeze_frames.append(
                model_metrics.with_columns(
                    pl.lit(model_label).alias("model"),
                    pl.lit(model_order, dtype=pl.Int64).alias("model_order"),
                )
            )

        freeze_metrics = pl.concat(freeze_frames, how="diagonal_relaxed")
        freeze_deltas = (
            freeze_metrics
            .join(glm_baseline, on="subject", how="inner")
            .with_columns(
                ((pl.col(score_column) - pl.col("glm_ll")) / math.log(2)).alias("delta_ll_vs_glm"),
                (pl.col("bic") - pl.col("glm_bic")).alias("delta_bic_vs_glm"),
                pl.lit(task_label).alias("task"),
            )
            .select(
                "task",
                "subject",
                "model",
                "model_order",
                "delta_ll_vs_glm",
                "delta_bic_vs_glm",
            )
            .sort(["model_order", "subject"])
        )
        freeze_plot_dfs[task_label] = freeze_deltas.to_pandas()
        count_rows.append(
            {
                "task": task_label,
                "comparison": "emission constraints",
                "subjects": freeze_deltas.get_column("subject").n_unique(),
                "rows": freeze_deltas.height,
            }
        )

        for family, state_dir in {
            "free": _config["state_free_dir"],
            "frozen": _config["state_frozen_dir"],
        }.items():
            state_metrics = read_metric_directory(state_dir, "*_glmhmm_metrics.parquet").filter(
                pl.col("K").is_between(2, 7)
            )
            state_deltas = (
                state_metrics
                .join(glm_baseline, on="subject", how="inner")
                .with_columns(
                    ((pl.col(score_column) - pl.col("glm_ll")) / math.log(2)).alias("delta_ll_vs_glm"),
                    (pl.col("bic") - pl.col("glm_bic")).alias("delta_bic_vs_glm"),
                    pl.lit(task_label).alias("task"),
                    pl.lit(family).alias("family"),
                )
                .select(
                    "task",
                    "family",
                    "subject",
                    "K",
                    "delta_ll_vs_glm",
                    "delta_bic_vs_glm",
                )
            )
            glm_zero = glm_baseline.select(
                pl.lit(task_label).alias("task"),
                pl.lit(family).alias("family"),
                "subject",
                pl.lit(1, dtype=pl.Int64).alias("K"),
                pl.lit(0.0).alias("delta_ll_vs_glm"),
                pl.lit(0.0).alias("delta_bic_vs_glm"),
            )
            state_with_glm = pl.concat([glm_zero, state_deltas]).sort(["K", "subject"])
            state_plot_dfs[family][task_label] = state_with_glm.to_pandas()
            count_rows.append(
                {
                    "task": task_label,
                    "comparison": f"state count: {family}",
                    "subjects": state_deltas.get_column("subject").n_unique(),
                    "rows": state_deltas.height,
                }
            )

    data_counts = pl.DataFrame(count_rows)
    return (
        ashwood_plot_dfs,
        data_counts,
        freeze_plot_dfs,
        lag_plot_dfs,
        state_plot_dfs,
    )


@app.cell
def _(data_counts):
    data_counts
    return


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            # Active scheme: the LL-only comparisons occupy separate rows,
            # followed by LL + BIC state-count rows.
            [
                ["lag_ll_2ADC", "lag_ll_2ADC", "lag_ll_2AFC", "lag_ll_2AFC"],
                # ["freeze_ll_2ADC", "freeze_ll_2ADC", "freeze_ll_2AFC", "freeze_ll_2AFC"],
                ["state_free_ll_2ADC", "state_free_bic_2ADC", "state_free_ll_2AFC", "state_free_bic_2AFC"],
                # ["state_frozen_ll_2ADC", "state_frozen_bic_2ADC", "state_frozen_ll_2AFC", "state_frozen_bic_2AFC"],
            ],
            # Full ΔLL + ΔBIC alternative:
            # [
            #     ["freeze_ll_2ADC", "freeze_bic_2ADC", "freeze_ll_2AFC", "freeze_bic_2AFC"],
            #     ["state_free_ll_2ADC", "state_free_bic_2ADC", "state_free_ll_2AFC", "state_free_bic_2AFC"],
            #     ["state_frozen_ll_2ADC", "state_frozen_bic_2ADC", "state_frozen_ll_2AFC", "state_frozen_bic_2AFC"],
            # ],
            # ΔBIC-only alternative:
            # [
            #     ["freeze_bic_2ADC", "freeze_bic_2AFC"],
            #     ["state_free_bic_2ADC", "state_free_bic_2AFC"],
            #     ["state_frozen_bic_2ADC", "state_frozen_bic_2AFC"],
            # ],
            figsize=fig_size(1),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of action-trace regressors
    """)
    return


@app.cell
def _(
    add_numeric_pair_annotations,
    axd,
    clean_plot_edges,
    lag_plot_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    lag_ll_2ADC = plt.gca() if not mount_figure else axd["lag_ll_2ADC"]
    lag_ll_2ADC.clear()
    _plot_df = lag_plot_dfs["2ADC"]
    sns.lineplot(data=_plot_df, x="plot_order", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=lag_ll_2ADC)
    sns.lineplot(data=_plot_df, x="plot_order", y="delta_ll_vs_glm", errorbar=("se", 1), color="black", linewidth=1, marker="o", markersize=4, markeredgewidth=0, markeredgecolor="none", sort=True, ax=lag_ll_2ADC)
    lag_ll_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_numeric_pair_annotations(
        lag_ll_2ADC,
        _plot_df,
        y="delta_ll_vs_glm",
        pairs=[(0, 1), (9, 10), (10, 12)],
    )
    lag_ll_2ADC.set(
        xlabel="Number of previous choices used",
        ylabel="$\Delta$ LL vs GLM\n with 15 previous choices\n (bits/trial)",
        xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        xticklabels=["GLM", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15"],
    )
    if _plot_df["plot_order"].max() == 0:
        lag_ll_2ADC.text(
            0.5,
            0.5,
            "No action-trace fits found",
            ha="center",
            va="center",
            color="0.4",
            transform=lag_ll_2ADC.transAxes,
        )
    lag_ll_2ADC.get_xticklabels()[0].set_ha("right")
    lag_ll_2ADC.get_xticklabels()[1].set_ha("left")
    clean_plot_edges(lag_ll_2ADC)
    sns.despine(ax=lag_ll_2ADC)
    if not mount_figure:
        lag_ll_2ADC.figure.savefig(path_panels / "svg" / "action_trace_regressors_delta_ll_2ADC.svg")
        lag_ll_2ADC.figure.savefig(path_panels / "png" / "action_trace_regressors_delta_ll_2ADC.png", dpi=300)
    lag_ll_2ADC
    return (lag_ll_2ADC,)


@app.cell
def _(
    add_numeric_pair_annotations,
    axd,
    clean_plot_edges,
    lag_plot_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    lag_ll_2AFC = plt.gca() if not mount_figure else axd["lag_ll_2AFC"]
    lag_ll_2AFC.clear()
    _plot_df = lag_plot_dfs["2AFC"]
    sns.lineplot(data=_plot_df, x="plot_order", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=lag_ll_2AFC)
    sns.lineplot(data=_plot_df, x="plot_order", y="delta_ll_vs_glm", errorbar=("se", 1), color="black", linewidth=1, marker="o", markersize=4, markeredgewidth=0, markeredgecolor="none", sort=True, ax=lag_ll_2AFC)
    lag_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_numeric_pair_annotations(
        lag_ll_2AFC,
        _plot_df,
        y="delta_ll_vs_glm",
        pairs=[(0, 1), (9, 10), (10, 12)],
    )
    lag_ll_2AFC.set(
        xlabel="Number of action-trace regressors",
        ylabel="",
        xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        xticklabels=["GLM", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15"],
    )
    lag_ll_2AFC.get_xticklabels()[0].set_ha("right")
    lag_ll_2AFC.get_xticklabels()[1].set_ha("left")
    clean_plot_edges(lag_ll_2AFC)
    sns.despine(ax=lag_ll_2AFC)
    if not mount_figure:
        lag_ll_2AFC.figure.savefig(path_panels / "svg" / "action_trace_regressors_delta_ll_2AFC.svg")
        lag_ll_2AFC.figure.savefig(path_panels / "png" / "action_trace_regressors_delta_ll_2AFC.png", dpi=300)
    lag_ll_2AFC
    return (lag_ll_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two-state emission constraints
    """)
    return


@app.cell
def _(
    add_model_pair_annotations,
    axd,
    clean_plot_edges,
    freeze_model_order,
    freeze_plot_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    freeze_ll_2ADC = plt.gca() if not mount_figure or "freeze_ll_2ADC" not in axd else axd["freeze_ll_2ADC"]
    freeze_ll_2ADC.clear()
    _plot_df = freeze_plot_dfs["2ADC"]
    sns.lineplot(data=_plot_df, x="model", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=False, ax=freeze_ll_2ADC)
    sns.lineplot(data=_plot_df, x="model", y="delta_ll_vs_glm", errorbar=("se", 1), color="black", linewidth=1, marker="o", markersize=4, markeredgewidth=0, markeredgecolor="none", sort=False, ax=freeze_ll_2ADC)
    freeze_ll_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_model_pair_annotations(
        freeze_ll_2ADC,
        _plot_df,
        y="delta_ll_vs_glm",
        order=freeze_model_order,
        pairs=[("Stim0", "Hist0"), ("Stim0", "Free"), ("Stim0", "Both0")],
    )
    # freeze_ll_2ADC.set(xlabel="", ylabel="Emission constraints\nbits/trial")
    freeze_ll_2ADC.set(xlabel="", ylabel="Emission constraints\nbits/trial")
    freeze_ll_2ADC.tick_params(axis="x", labelrotation=0)
    clean_plot_edges(freeze_ll_2ADC)
    sns.despine(ax=freeze_ll_2ADC)
    if not mount_figure:
        freeze_ll_2ADC.figure.savefig(path_panels / "svg" / "freeze_delta_ll_2ADC.svg")
        freeze_ll_2ADC.figure.savefig(path_panels / "png" / "freeze_delta_ll_2ADC.png", dpi=300)
    freeze_ll_2ADC
    return (freeze_ll_2ADC,)


@app.cell
def _(
    add_one_sample_zero_annotations,
    axd,
    clean_plot_edges,
    freeze_model_order,
    freeze_plot_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    freeze_bic_2ADC = plt.gca() if not mount_figure or "freeze_bic_2ADC" not in axd else axd["freeze_bic_2ADC"]
    freeze_bic_2ADC.clear()
    _plot_df = freeze_plot_dfs["2ADC"]
    sns.lineplot(data=_plot_df, x="model", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=False, ax=freeze_bic_2ADC)
    sns.lineplot(data=_plot_df, x="model", y="delta_bic_vs_glm", errorbar=("se", 1), color="black", linewidth=1, marker="o", markersize=4, markeredgewidth=0, markeredgecolor="none", sort=False, ax=freeze_bic_2ADC)
    freeze_bic_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_one_sample_zero_annotations(freeze_bic_2ADC, _plot_df, x="model", y="delta_bic_vs_glm", order=freeze_model_order)
    freeze_bic_2ADC.set(xlabel="", ylabel="BIC")
    freeze_bic_2ADC.tick_params(axis="x", labelrotation=30)
    clean_plot_edges(freeze_bic_2ADC)
    sns.despine(ax=freeze_bic_2ADC)
    if not mount_figure:
        freeze_bic_2ADC.figure.savefig(path_panels / "svg" / "freeze_delta_bic_2ADC.svg")
        freeze_bic_2ADC.figure.savefig(path_panels / "png" / "freeze_delta_bic_2ADC.png", dpi=300)
    freeze_bic_2ADC
    return


@app.cell
def _(
    add_model_pair_annotations,
    axd,
    clean_plot_edges,
    freeze_model_order,
    freeze_plot_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    freeze_ll_2AFC = plt.gca() if not mount_figure or "freeze_ll_2AFC" not in axd else axd["freeze_ll_2AFC"]
    freeze_ll_2AFC.clear()
    _plot_df = freeze_plot_dfs["2AFC"]
    sns.lineplot(data=_plot_df, x="model", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=False, ax=freeze_ll_2AFC)
    sns.lineplot(data=_plot_df, x="model", y="delta_ll_vs_glm", errorbar=("se", 1), color="black", linewidth=1, marker="o", markersize=4, markeredgewidth=0, markeredgecolor="none", sort=False, ax=freeze_ll_2AFC)
    freeze_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_model_pair_annotations(
        freeze_ll_2AFC,
        _plot_df,
        y="delta_ll_vs_glm",
        order=freeze_model_order,
        pairs=[("Stim0", "Hist0"), ("Stim0", "Free"), ("Stim0", "Both0")],
    )
    freeze_ll_2AFC.set(xlabel="", ylabel="bits/trial")
    freeze_ll_2AFC.tick_params(axis="x", labelrotation=0)
    clean_plot_edges(freeze_ll_2AFC)
    sns.despine(ax=freeze_ll_2AFC)
    if not mount_figure:
        freeze_ll_2AFC.figure.savefig(path_panels / "svg" / "freeze_delta_ll_2AFC.svg")
        freeze_ll_2AFC.figure.savefig(path_panels / "png" / "freeze_delta_ll_2AFC.png", dpi=300)
    freeze_ll_2AFC
    return (freeze_ll_2AFC,)


@app.cell
def _(
    add_one_sample_zero_annotations,
    axd,
    clean_plot_edges,
    freeze_model_order,
    freeze_plot_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    freeze_bic_2AFC = plt.gca() if not mount_figure or "freeze_bic_2AFC" not in axd else axd["freeze_bic_2AFC"]
    freeze_bic_2AFC.clear()
    _plot_df = freeze_plot_dfs["2AFC"]
    sns.lineplot(data=_plot_df, x="model", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=False, ax=freeze_bic_2AFC)
    sns.lineplot(data=_plot_df, x="model", y="delta_bic_vs_glm", errorbar=("se", 1), color="black", linewidth=1, marker="o", markersize=4, markeredgewidth=0, markeredgecolor="none", sort=False, ax=freeze_bic_2AFC)
    freeze_bic_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_one_sample_zero_annotations(freeze_bic_2AFC, _plot_df, x="model", y="delta_bic_vs_glm", order=freeze_model_order)
    freeze_bic_2AFC.set(xlabel="", ylabel="BIC")
    freeze_bic_2AFC.tick_params(axis="x", labelrotation=30)
    clean_plot_edges(freeze_bic_2AFC)
    sns.despine(ax=freeze_bic_2AFC)
    if not mount_figure:
        freeze_bic_2AFC.figure.savefig(path_panels / "svg" / "freeze_delta_bic_2AFC.svg")
        freeze_bic_2AFC.figure.savefig(path_panels / "png" / "freeze_delta_bic_2AFC.png", dpi=300)
    freeze_bic_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of states: free emissions
    """)
    return


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_free_ll_2ADC = plt.gca() if not mount_figure or "state_free_ll_2ADC" not in axd else axd["state_free_ll_2ADC"]
    state_free_ll_2ADC.clear()
    _plot_df = state_plot_dfs["free"]["2ADC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_free_ll_2ADC)
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", errorbar=("se", 1), color=family_palette["free"], marker="o", markersize=4, sort=True, ax=state_free_ll_2ADC)
    state_free_ll_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_free_ll_2ADC.set(xlabel="Number of states K", ylabel="Free emissions\nbits/trial", xticks=range(1, 8))
    add_state_comparison_annotations(state_free_ll_2ADC, _plot_df, y="delta_ll_vs_glm")
    clean_plot_edges(state_free_ll_2ADC)
    plot_ashwood_point(state_free_ll_2ADC, ashwood_plot_dfs["2ADC"], "delta_ll_vs_glm")
    sns.despine(ax=state_free_ll_2ADC)
    if not mount_figure:
        state_free_ll_2ADC.figure.savefig(path_panels / "svg" / "state_free_delta_ll_2ADC.svg")
        state_free_ll_2ADC.figure.savefig(path_panels / "png" / "state_free_delta_ll_2ADC.png", dpi=300)
    state_free_ll_2ADC
    return (state_free_ll_2ADC,)


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_free_bic_2ADC = plt.gca() if not mount_figure or "state_free_bic_2ADC" not in axd else axd["state_free_bic_2ADC"]
    state_free_bic_2ADC.clear()
    _plot_df = state_plot_dfs["free"]["2ADC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_free_bic_2ADC)
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", errorbar=("se", 1), color=family_palette["free"], marker="o", sort=True, ax=state_free_bic_2ADC)
    state_free_bic_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_free_bic_2ADC.set(xlabel="Number of states K", ylabel="BIC", xticks=range(1, 8))
    add_state_comparison_annotations(state_free_bic_2ADC, _plot_df, y="delta_bic_vs_glm")
    clean_plot_edges(state_free_bic_2ADC)
    plot_ashwood_point(state_free_bic_2ADC, ashwood_plot_dfs["2ADC"], "delta_bic_vs_glm")
    sns.despine(ax=state_free_bic_2ADC)
    if not mount_figure:
        state_free_bic_2ADC.figure.savefig(path_panels / "svg" / "state_free_delta_bic_2ADC.svg")
        state_free_bic_2ADC.figure.savefig(path_panels / "png" / "state_free_delta_bic_2ADC.png", dpi=300)
    state_free_bic_2ADC
    return (state_free_bic_2ADC,)


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_free_ll_2AFC = plt.gca() if not mount_figure or "state_free_ll_2AFC" not in axd else axd["state_free_ll_2AFC"]
    state_free_ll_2AFC.clear()
    _plot_df = state_plot_dfs["free"]["2AFC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_free_ll_2AFC)
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", errorbar=("se", 1), color=family_palette["free"], marker="o", markersize=4, sort=True, ax=state_free_ll_2AFC)
    state_free_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_free_ll_2AFC.set(xlabel="Number of states K", ylabel="bits/trial", xticks=range(1, 8))
    add_state_comparison_annotations(state_free_ll_2AFC, _plot_df, y="delta_ll_vs_glm")
    clean_plot_edges(state_free_ll_2AFC)
    plot_ashwood_point(state_free_ll_2AFC, ashwood_plot_dfs["2AFC"], "delta_ll_vs_glm")
    sns.despine(ax=state_free_ll_2AFC)
    if not mount_figure:
        state_free_ll_2AFC.figure.savefig(path_panels / "svg" / "state_free_delta_ll_2AFC.svg")
        state_free_ll_2AFC.figure.savefig(path_panels / "png" / "state_free_delta_ll_2AFC.png", dpi=300)
    state_free_ll_2AFC
    return (state_free_ll_2AFC,)


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_free_bic_2AFC = plt.gca() if not mount_figure or "state_free_bic_2AFC" not in axd else axd["state_free_bic_2AFC"]
    state_free_bic_2AFC.clear()
    _plot_df = state_plot_dfs["free"]["2AFC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_free_bic_2AFC)
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", errorbar=("se", 1), color=family_palette["free"], marker="o", markersize=4, sort=True, ax=state_free_bic_2AFC)
    state_free_bic_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_free_bic_2AFC.set(xlabel="Number of states K", ylabel="BIC", xticks=range(1, 8))
    add_state_comparison_annotations(state_free_bic_2AFC, _plot_df, y="delta_bic_vs_glm")
    clean_plot_edges(state_free_bic_2AFC)
    plot_ashwood_point(state_free_bic_2AFC, ashwood_plot_dfs["2AFC"], "delta_bic_vs_glm")
    sns.despine(ax=state_free_bic_2AFC)
    if not mount_figure:
        state_free_bic_2AFC.figure.savefig(path_panels / "svg" / "state_free_delta_bic_2AFC.svg")
        state_free_bic_2AFC.figure.savefig(path_panels / "png" / "state_free_delta_bic_2AFC.png", dpi=300)
    state_free_bic_2AFC
    return (state_free_bic_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of states: frozen emissions
    """)
    return


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_frozen_ll_2ADC = plt.gca() if not mount_figure or "state_frozen_ll_2ADC" not in axd else axd["state_frozen_ll_2ADC"]
    state_frozen_ll_2ADC.clear()
    _plot_df = state_plot_dfs["frozen"]["2ADC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_ll_2ADC)
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_ll_2ADC)
    state_frozen_ll_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_ll_2ADC.set(xlabel="Number of states K", ylabel="Frozen emissions\nbits/trial", xticks=range(1, 8), xlim=(0.8, 7.2))
    if _plot_df["K"].max() == 1:
        state_frozen_ll_2ADC.text(0.5, 0.5, "No frozen fits found", ha="center", va="center", transform=state_frozen_ll_2ADC.transAxes, color="0.4")
    add_state_comparison_annotations(state_frozen_ll_2ADC, _plot_df, y="delta_ll_vs_glm")
    clean_plot_edges(state_frozen_ll_2ADC)
    plot_ashwood_point(state_frozen_ll_2ADC, ashwood_plot_dfs["2ADC"], "delta_ll_vs_glm")
    sns.despine(ax=state_frozen_ll_2ADC)
    if not mount_figure:
        state_frozen_ll_2ADC.figure.savefig(path_panels / "svg" / "state_frozen_delta_ll_2ADC.svg")
        state_frozen_ll_2ADC.figure.savefig(path_panels / "png" / "state_frozen_delta_ll_2ADC.png", dpi=300)
    state_frozen_ll_2ADC
    return (state_frozen_ll_2ADC,)


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_frozen_bic_2ADC = plt.gca() if not mount_figure or "state_frozen_bic_2ADC" not in axd else axd["state_frozen_bic_2ADC"]
    state_frozen_bic_2ADC.clear()
    _plot_df = state_plot_dfs["frozen"]["2ADC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_bic_2ADC)
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_bic_2ADC)
    state_frozen_bic_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_bic_2ADC.set(xlabel="Number of states K", ylabel="BIC", xticks=range(1, 8), xlim=(0.8, 7.2))
    if _plot_df["K"].max() == 1:
        state_frozen_bic_2ADC.text(0.5, 0.5, "No frozen fits found", ha="center", va="center", transform=state_frozen_bic_2ADC.transAxes, color="0.4")
    add_state_comparison_annotations(state_frozen_bic_2ADC, _plot_df, y="delta_bic_vs_glm")
    clean_plot_edges(state_frozen_bic_2ADC)
    plot_ashwood_point(state_frozen_bic_2ADC, ashwood_plot_dfs["2ADC"], "delta_bic_vs_glm")
    sns.despine(ax=state_frozen_bic_2ADC)
    if not mount_figure:
        state_frozen_bic_2ADC.figure.savefig(path_panels / "svg" / "state_frozen_delta_bic_2ADC.svg")
        state_frozen_bic_2ADC.figure.savefig(path_panels / "png" / "state_frozen_delta_bic_2ADC.png", dpi=300)
    state_frozen_bic_2ADC
    return (state_frozen_bic_2ADC,)


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_frozen_ll_2AFC = plt.gca() if not mount_figure or "state_frozen_ll_2AFC" not in axd else axd["state_frozen_ll_2AFC"]
    state_frozen_ll_2AFC.clear()
    _plot_df = state_plot_dfs["frozen"]["2AFC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_ll_2AFC)
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_ll_2AFC)
    state_frozen_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_ll_2AFC.set(xlabel="Number of states K", ylabel="bits/trial", xticks=range(1, 8))
    add_state_comparison_annotations(state_frozen_ll_2AFC, _plot_df, y="delta_ll_vs_glm")
    clean_plot_edges(state_frozen_ll_2AFC)
    plot_ashwood_point(state_frozen_ll_2AFC, ashwood_plot_dfs["2AFC"], "delta_ll_vs_glm")
    sns.despine(ax=state_frozen_ll_2AFC)
    if not mount_figure:
        state_frozen_ll_2AFC.figure.savefig(path_panels / "svg" / "state_frozen_delta_ll_2AFC.svg")
        state_frozen_ll_2AFC.figure.savefig(path_panels / "png" / "state_frozen_delta_ll_2AFC.png", dpi=300)
    state_frozen_ll_2AFC
    return (state_frozen_ll_2AFC,)


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_plot_dfs,
    axd,
    clean_plot_edges,
    family_palette,
    mount_figure,
    path_panels,
    plot_ashwood_point,
    plt,
    sns,
    state_plot_dfs,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    state_frozen_bic_2AFC = plt.gca() if not mount_figure or "state_frozen_bic_2AFC" not in axd else axd["state_frozen_bic_2AFC"]
    state_frozen_bic_2AFC.clear()
    _plot_df = state_plot_dfs["frozen"]["2AFC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_bic_2AFC)
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_bic_2AFC)
    state_frozen_bic_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_bic_2AFC.set(xlabel="Number of states K", ylabel="BIC", xticks=range(1, 8))
    add_state_comparison_annotations(state_frozen_bic_2AFC, _plot_df, y="delta_bic_vs_glm")
    clean_plot_edges(state_frozen_bic_2AFC)
    plot_ashwood_point(state_frozen_bic_2AFC, ashwood_plot_dfs["2AFC"], "delta_bic_vs_glm")
    sns.despine(ax=state_frozen_bic_2AFC)
    if not mount_figure:
        state_frozen_bic_2AFC.figure.savefig(path_panels / "svg" / "state_frozen_delta_bic_2AFC.svg")
        state_frozen_bic_2AFC.figure.savefig(path_panels / "png" / "state_frozen_delta_bic_2AFC.png", dpi=300)
    state_frozen_bic_2AFC
    return (state_frozen_bic_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full figure
    """)
    return


@app.cell
def _(
    align_zero_fraction,
    axd,
    fig,
    freeze_ll_2ADC,
    freeze_ll_2AFC,
    lag_ll_2ADC,
    lag_ll_2AFC,
    mount_figure,
    path_panels,
    state_free_bic_2ADC,
    state_free_bic_2AFC,
    state_free_ll_2ADC,
    state_free_ll_2AFC,
    state_frozen_bic_2ADC,
    state_frozen_bic_2AFC,
    state_frozen_ll_2ADC,
    state_frozen_ll_2AFC,
    state_plot_dfs,
    zoom_zero_centered,
):
    if mount_figure:
        # Reset from artists before applying layout adjustments. This keeps the
        # limits and significance brackets fixed when this cell is rerun.
        for _axis in axd.values():
            _axis.relim()
            _axis.autoscale(enable=True, axis="y")
            _axis.margins(y=0.08)

        align_zero_fraction(
            [
                lag_ll_2ADC,
                freeze_ll_2ADC,
                state_free_ll_2ADC,
                state_frozen_ll_2ADC,
                lag_ll_2AFC,
                freeze_ll_2AFC,
                state_free_ll_2AFC,
                state_frozen_ll_2AFC,
            ],
            fraction=0.25,
        )

        zoom_zero_centered(
            state_free_bic_2ADC,
            state_plot_dfs["free"]["2ADC"],
            "delta_bic_vs_glm",
            quantile=0.9,
        )
        zoom_zero_centered(
            state_free_bic_2AFC,
            state_plot_dfs["free"]["2AFC"],
            "delta_bic_vs_glm",
            quantile=0.9,
        )
        zoom_zero_centered(
            state_frozen_bic_2ADC,
            state_plot_dfs["frozen"]["2ADC"],
            "delta_bic_vs_glm",
            quantile=0.9,
        )
        zoom_zero_centered(
            state_frozen_bic_2AFC,
            state_plot_dfs["frozen"]["2AFC"],
            "delta_bic_vs_glm",
            quantile=0.9,
        )

        for _axis in (
            lag_ll_2ADC,
            lag_ll_2AFC,
            freeze_ll_2ADC,
            freeze_ll_2AFC,
            state_free_ll_2ADC,
            state_free_bic_2ADC,
            state_free_ll_2AFC,
            state_free_bic_2AFC,
        ):
            _axis.set_xlabel("")

        for _axis in axd.values():
            _axis.set_title("")
        lag_ll_2ADC.set_title("2ADC")
        lag_ll_2AFC.set_title("2AFC")
        state_free_ll_2ADC.set_title("ΔLL vs GLM")
        state_free_bic_2ADC.set_title("ΔBIC vs GLM")
        state_free_ll_2AFC.set_title("ΔLL vs GLM")
        state_free_bic_2AFC.set_title("ΔBIC vs GLM")
        fig.align_labels()
        fig.savefig(path_panels / "supplementary_figure31.svg")
        fig.savefig(path_panels / "supplementary_figure31.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure31.pdf")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
