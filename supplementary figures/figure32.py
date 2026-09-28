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
    # Supplementary 3.2: Model comparison for frozen parameters

    ## Description

    We compare free and constrained GLM-HMM emission models in the 2ADC and 2AFC tasks. Model fit is evaluated relative to each task's GLM, while RT and nLicks AUCs (pooled across tasks) quantify how well behavior distinguishes Engaged from Disengaged states.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import math
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_1samp, ttest_rel
    from src.plots.common import BOXPLOT_STYLE, fig_size

    from glmhmmt.notebook_support.analysis_common import load_fit_bundle
    from glmhmmt.postprocess import build_trial_df
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views

    return (
        BOXPLOT_STYLE,
        Path,
        build_trial_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        load_fit_bundle,
        math,
        mo,
        np,
        os,
        pd,
        pl,
        plt,
        sns,
        ttest_1samp,
        ttest_rel,
    )


@app.cell
def _():
    mount_figure = True
    EXAMPLE_MODEL = "Stim0"
    EXAMPLE_SUBJECTS = {"2ADC": "C10b", "2AFC": "325"}
    return EXAMPLE_MODEL, EXAMPLE_SUBJECTS, mount_figure


@app.cell
def _():
    panel_names = {
        "freeze_ll_2ADC": "S4a",
        "freeze_ll_2AFC": "S4b",
        "auc_behavior": "S4c",
        "rt_2ADC": "S4d",
        "nlicks_2ADC": "S4e",
        "rt_2AFC": "S4f",
        "nlicks_2AFC": "S4g",
        "state_frozen_ll_2ADC": "S4h",
        "state_frozen_bic_2ADC": "S4i",
        "state_frozen_ll_2AFC": "S4j",
        "state_frozen_bic_2AFC": "S4k",
    }
    return (panel_names,)


@app.cell
def _(Path, configure_paths, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    path_panels = ROOT / "supplementary figures" / "panels32"
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
    auc_model_order = freeze_model_order
    auc_metric_order = ["RT", "nLicks"]
    state_order = ["Engaged", "Disengaged"]
    state_palette = {"Engaged": "tab:green", "Disengaged": "tab:gray"}
    metric_palette = dict(zip(auc_metric_order, sns.color_palette("Set2", 2)))
    family_palette = {"free": "black", "frozen": "black"}
    return (
        auc_metric_order,
        auc_model_order,
        family_palette,
        freeze_model_order,
        metric_palette,
        state_order,
        state_palette,
    )


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
def _(np, pd):
    def binary_engaged_target(labels):
        """Return Engaged targets and the valid binary-state rows."""
        label_text = pd.Series(labels, copy=False).astype(str).str.strip().str.lower()
        engaged = label_text.eq("engaged") | label_text.str.startswith("engaged ")
        disengaged = label_text.eq("disengaged") | label_text.str.startswith("disengaged ")
        return engaged.to_numpy(bool), (engaged | disengaged).to_numpy(bool)

    def correct_trial_mask(trial_df):
        """Return the correct-trial mask used for the nLicks AUC."""
        return (
            pd.to_numeric(trial_df["correct_bool"], errors="coerce")
            .eq(1)
            .fillna(False)
            .to_numpy(bool)
        )

    def roc_auc(target, score):
        """Return trapezoidal ROC AUC for binary targets and continuous scores."""
        target = np.asarray(target, dtype=bool)
        score = np.asarray(score, dtype=float)
        finite_score = np.isfinite(score)
        target = target[finite_score]
        score = score[finite_score]
        n_positive = int(target.sum())
        n_negative = int((~target).sum())
        if target.size == 0 or n_positive == 0 or n_negative == 0:
            return None

        score_order = np.argsort(-score, kind="mergesort")
        sorted_target = target[score_order]
        sorted_score = score[score_order]
        threshold_indices = np.r_[
            np.where(np.diff(sorted_score))[0],
            sorted_target.size - 1,
        ]
        true_positives = np.cumsum(sorted_target)[threshold_indices]
        false_positives = 1 + threshold_indices - true_positives
        true_positive_rate = np.r_[0.0, true_positives / n_positive]
        false_positive_rate = np.r_[0.0, false_positives / n_negative]
        return float(
            np.sum(
                np.diff(false_positive_rate)
                * (true_positive_rate[:-1] + true_positive_rate[1:])
                / 2.0
            )
        )

    return binary_engaged_target, correct_trial_mask, roc_auc


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
                pl.col("k").cast(pl.Int64).alias("K") if "k" in metrics.columns else pl.lit(1, dtype=pl.Int64).alias("K")
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

    def bonferroni_pvalue(pvalue, n_tests):
        return min(float(pvalue) * n_tests, 1.0)

    def add_grouped_model_pair_annotations(
        axis,
        dataframe,
        *,
        metric_order,
        model_order,
        metric_palette,
    ):
        metric_offsets = {
            metric: (index - (len(metric_order) - 1) / 2) * 0.8 / len(metric_order)
            for index, metric in enumerate(metric_order)
        }
        model_pairs = [
            ("Stim0", "Hist0"),
            ("Stim0", "Free"),
            ("Stim0", "Both0"),
        ]
        panel_tests = []
        for metric in metric_order:
            for left, right in model_pairs:
                paired = (
                    dataframe.loc[
                        dataframe["metric"].eq(metric)
                        & dataframe["model"].isin([left, right]),
                        ["subject", "model", "auc"],
                    ]
                    .pivot_table(
                        index="subject",
                        columns="model",
                        values="auc",
                        aggfunc="first",
                    )
                    .reindex(columns=[left, right])
                    .dropna()
                )
                if len(paired) < 2:
                    continue
                pvalue = float(ttest_rel(paired[left], paired[right]).pvalue)
                if math.isfinite(pvalue):
                    panel_tests.append((metric, left, right, pvalue))

        n_tests = len(panel_tests)
        bottom, top = axis.get_ylim()
        step = (top - bottom) * 0.065
        model_positions = {model: index for index, model in enumerate(model_order)}
        annotation_index = 0
        for metric in metric_order:
            significant_pairs = []
            for test_metric, left, right, pvalue in panel_tests:
                if test_metric != metric:
                    continue
                adjusted_pvalue = bonferroni_pvalue(pvalue, n_tests)
                label = significance_stars(adjusted_pvalue)
                significant_pairs.append((left, right, label or "ns"))

            for left, right, label in significant_pairs:
                line_y = top + annotation_index * step
                left_x = model_positions[left] + metric_offsets[metric]
                right_x = model_positions[right] + metric_offsets[metric]
                axis.plot(
                    [left_x, right_x],
                    [line_y, line_y],
                    color=metric_palette[metric],
                    linewidth=0.7,
                )
                axis.text(
                    (left_x + right_x) / 2,
                    line_y + step * 0.08,
                    label,
                    ha="center",
                    va="bottom",
                )
                annotation_index += 1
        axis.set_ylim(top=top + annotation_index * step)


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
            paired = dataframe.loc[dataframe["model"].isin([left, right]), ["subject", "model", y]].pivot_table(
                index="subject", columns="model", values=y, aggfunc="first"
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[left], paired[right]).pvalue)
            if math.isfinite(pvalue):
                tested_pairs.append((left, right, pvalue))
        n_tests = len(tested_pairs)
        base_y = float(finite_values.max()) + padding
        axis.set_ylim(top=base_y + padding * (len(tested_pairs) + 1))
        for pair_index, (left, right, pvalue) in enumerate(tested_pairs):
            adjusted_pvalue = bonferroni_pvalue(pvalue, n_tests)
            label = significance_stars(adjusted_pvalue) or "ns"
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
            paired = dataframe.loc[dataframe["plot_order"].isin([left, right]), ["subject", "plot_order", y]].pivot_table(
                index="subject", columns="plot_order", values=y, aggfunc="first"
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
            ([(1, 2)] if 1 in state_counts and 2 in state_counts else []) + [(2, 3)]
            # + [(2, state_count) for state_count in state_counts if state_count > 2]
        )
        tested_pairs = []
        for left, right in pairs:
            paired = dataframe.loc[dataframe["K"].isin([left, right]), ["subject", "K", y]].pivot_table(
                index="subject", columns="K", values=y, aggfunc="first"
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[right], paired[left]).pvalue)
            if math.isfinite(pvalue):
                tested_pairs.append((left, right, pvalue))
        n_tests = len(tested_pairs)
        base_y = float(finite_values.max()) + padding
        axis.set_ylim(top=base_y + padding * (len(tested_pairs) + 1))
        for pair_index, (left, right, pvalue) in enumerate(tested_pairs):
            adjusted_pvalue = bonferroni_pvalue(pvalue, n_tests)
            label = significance_stars(adjusted_pvalue) or "ns"
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
        add_grouped_model_pair_annotations,
        add_model_pair_annotations,
        add_one_sample_zero_annotations,
        add_state_comparison_annotations,
        clean_plot_edges,
        plot_ashwood_point,
        read_metric_directory,
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
    return ashwood_plot_dfs, data_counts, freeze_plot_dfs, state_plot_dfs


@app.cell
def _(
    TASK_CONFIGS,
    binary_engaged_target,
    build_trial_df,
    build_views,
    correct_trial_mask,
    get_adapter,
    load_fit_bundle,
    paths,
    pd,
    pl,
    roc_auc,
):
    behavior_auc_dfs = {}
    behavior_trial_dfs = {}
    metric_specs = [("RT", "RT"), ("nLicks", "nLicks")]

    for behavior_task_label, behavior_task_config in TASK_CONFIGS.items():
        behavior_task_name = behavior_task_config["task"]
        behavior_adapter = get_adapter(behavior_task_name)
        behavior_task_df = behavior_adapter.subject_filter(
            behavior_adapter.read_dataset()
        )
        behavior_subjects = behavior_task_df["subject"].unique().sort().to_list()
        behavior_trial_frames = []

        for behavior_model_order, (behavior_model_label, behavior_model_id) in enumerate(
            behavior_task_config["freeze_models"]
        ):
            behavior_model_adapter, _, _, behavior_views = load_fit_bundle(
                task_name=behavior_task_name,
                model_kind="glmhmm",
                alias=behavior_model_id,
                k=2,
                subjects=behavior_subjects,
                get_adapter=get_adapter,
                build_views=build_views,
                scoring_key=behavior_adapter.scoring_key,
                local_root=paths.RESULTS / "fits" / behavior_task_name / "glmhmm",
            )
            for behavior_subject, behavior_view in behavior_views.items():
                behavior_subject_df = (
                    behavior_task_df
                    .filter(pl.col("subject") == behavior_subject)
                    .sort(behavior_model_adapter.sort_col)
                    .filter(
                        pl.col(behavior_model_adapter.session_col)
                        .count()
                        .over(behavior_model_adapter.session_col)
                        >= 2
                    )
                )
                assert behavior_subject_df.height == behavior_view.T
                behavior_trial_frames.append(
                    build_trial_df(
                        behavior_view,
                        behavior_model_adapter,
                        behavior_subject_df,
                        behavior_model_adapter.behavioral_cols,
                    ).with_columns(
                        pl.lit(behavior_model_label).alias("model"),
                        pl.lit(behavior_model_order, dtype=pl.Int64).alias("model_order"),
                    )
                )

        behavior_trial_df = pl.concat(behavior_trial_frames, how="diagonal").to_pandas()
        behavior_trial_dfs[behavior_task_label] = behavior_trial_df
        behavior_auc_rows = []
        for (behavior_model_label, behavior_subject), behavior_subject_df in behavior_trial_df.groupby(
            ["model", "subject"], sort=False
        ):
            behavior_target, behavior_valid_labels = binary_engaged_target(
                behavior_subject_df["state_label"]
            )
            for behavior_metric, behavior_metric_label in metric_specs:
                behavior_score = pd.to_numeric(
                    behavior_subject_df[behavior_metric], errors="coerce"
                ).to_numpy(float)
                behavior_valid_trials = behavior_valid_labels.copy()
                if behavior_metric == "nLicks":
                    behavior_valid_trials &= correct_trial_mask(behavior_subject_df)
                else:
                    # Faster RT should predict the Engaged state.
                    behavior_score = -behavior_score
                behavior_auc = roc_auc(
                    behavior_target[behavior_valid_trials],
                    behavior_score[behavior_valid_trials],
                )
                if behavior_auc is not None:
                    behavior_auc_rows.append(
                        {
                            "task": behavior_task_label,
                            "subject": str(behavior_subject),
                            "model": behavior_model_label,
                            "metric": behavior_metric_label,
                            "auc": behavior_auc,
                        }
                    )

        behavior_auc_df = pd.DataFrame(behavior_auc_rows)
        behavior_auc_df["model"] = pd.Categorical(
            behavior_auc_df["model"],
            categories=[label for label, _ in behavior_task_config["freeze_models"]],
            ordered=True,
        )
        behavior_auc_dfs[behavior_task_label] = behavior_auc_df.sort_values(
            ["metric", "model", "subject"]
        )
    # Prefix animal IDs so paired tests cannot match animals across tasks.
    behavior_auc_dfs["combined"] = pd.concat(
        behavior_auc_dfs.values(), ignore_index=True,
    ).assign(subject=lambda df: df["task"] + ":" + df["subject"].astype(str))
    return behavior_auc_dfs, behavior_trial_dfs


@app.cell
def _(data_counts):
    data_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure layout
    """)
    return


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                ["freeze_ll_2ADC"] * 2 + ["freeze_ll_2AFC"] * 2,
                ["auc_behavior"] * 4,
                ["rt_2ADC", "nlicks_2ADC", "rt_2AFC", "nlicks_2AFC"],
            ],
            figsize=fig_size(1, 0.9), constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 1.3, 1]},
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two-state emission constraints
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
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
    freeze_ll_2ADC.set(xlabel="", ylabel="Emission constraints\n ΔLL vs GLM bits/trial")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
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
    ## Single-animal state histograms

    Examples use the same animal and model for RT and nLicks within each task.
    Histograms are normalized separately within each state, using common bins.
    The x-axis shows percentiles 5–95 pooled across states for each animal and measure;
    normalization still uses all trials, including the tails outside the displayed range.
    nLicks uses correct trials, as in the AUC analysis; RT uses all finite trials.
    Animals and the model can be changed in `EXAMPLE_SUBJECTS` and `EXAMPLE_MODEL`.
    """)
    return


@app.cell
def _(
    EXAMPLE_MODEL,
    EXAMPLE_SUBJECTS,
    auc_metric_order,
    behavior_trial_dfs,
    binary_engaged_target,
    correct_trial_mask,
    np,
    pd,
):
    example_dfs = {}
    example_bins = {}
    example_counts = []
    for example_task, example_subject in EXAMPLE_SUBJECTS.items():
        trials = behavior_trial_dfs[example_task]
        animal_trials = trials.loc[
            trials["subject"].astype(str).eq(example_subject) & trials["model"].eq(EXAMPLE_MODEL)
        ]
        engaged, valid_states = binary_engaged_target(animal_trials["state_label"])
        for example_metric in auc_metric_order:
            values = pd.to_numeric(animal_trials[example_metric], errors="coerce").to_numpy(float)
            valid_trials = valid_states & np.isfinite(values)
            if example_metric == "nLicks":
                valid_trials &= correct_trial_mask(animal_trials)
            histogram_df = pd.DataFrame({
                "value": values[valid_trials],
                "state": np.where(engaged[valid_trials], "Engaged", "Disengaged"),
            })
            assert histogram_df["state"].nunique() == 2, "Choose an animal with both states."
            example_dfs[example_task, example_metric] = histogram_df
            # Both states share bin edges; all finite observations are retained.
            example_bins[example_task, example_metric] = (
                np.arange(histogram_df["value"].min() - 0.5, histogram_df["value"].max() + 1.5)
                if example_metric == "nLicks" else np.arange(histogram_df["value"].min(), histogram_df["value"].max() + 0.025, 0.025)
            )
            for example_state, state_trials in histogram_df.groupby("state"):
                example_counts.append({
                    "task": example_task, "subject": example_subject, "model": EXAMPLE_MODEL,
                    "metric": example_metric, "state": example_state, "n_trials": len(state_trials),
                })
    example_counts_df = pd.DataFrame(example_counts)
    return example_bins, example_counts_df, example_dfs


@app.cell
def _(
    EXAMPLE_MODEL,
    EXAMPLE_SUBJECTS,
    axd,
    example_bins,
    example_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    rt_example_2ADC = axd["rt_2ADC"] if mount_figure else plt.figure(
        figsize=fig_size(3, 1), constrained_layout=True,
    ).gca()
    rt_example_2ADC.clear()
    sns.histplot(
        data=example_dfs["2ADC", "RT"], x="value", hue="state",
        hue_order=state_order, palette=state_palette, bins=example_bins["2ADC", "RT"],
        stat="probability", common_norm=False, element="step", fill=False,
        linewidth=1, legend=False, ax=rt_example_2ADC,
    )
    rt_example_2ADC.set(
        xlabel="RT (s)", ylabel="Probability",
        xlim=example_dfs["2ADC", "RT"]["value"].quantile([0.05, 0.95]),
        title=f"2ADC · {EXAMPLE_SUBJECTS['2ADC']}\n{EXAMPLE_MODEL}",
    )
    rt_example_2ADC.xaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=rt_example_2ADC)
    if not mount_figure:
        rt_example_2ADC.figure.savefig(path_panels / "svg" / "rt_example_2ADC.svg")
        rt_example_2ADC.figure.savefig(path_panels / "png" / "rt_example_2ADC.png", dpi=300)
    rt_example_2ADC
    return (rt_example_2ADC,)


@app.cell
def _(
    EXAMPLE_MODEL,
    EXAMPLE_SUBJECTS,
    axd,
    example_bins,
    example_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    nlicks_example_2ADC = axd["nlicks_2ADC"] if mount_figure else plt.figure(
        figsize=fig_size(3, 1), constrained_layout=True,
    ).gca()
    nlicks_example_2ADC.clear()
    sns.histplot(
        data=example_dfs["2ADC", "nLicks"], x="value", hue="state",
        hue_order=state_order, palette=state_palette, bins=example_bins["2ADC", "nLicks"],
        stat="probability", common_norm=False, element="step", fill=False,
        linewidth=1, legend=False, ax=nlicks_example_2ADC,
    )
    nlicks_example_2ADC.set(
        xlabel="nLicks", ylabel="",
        xlim=example_dfs["2ADC", "nLicks"]["value"].quantile([0.05, 0.95]),
        title=f"2ADC · {EXAMPLE_SUBJECTS['2ADC']}\n{EXAMPLE_MODEL}",
    )
    sns.despine(ax=nlicks_example_2ADC)
    if not mount_figure:
        nlicks_example_2ADC.figure.savefig(path_panels / "svg" / "nlicks_example_2ADC.svg")
        nlicks_example_2ADC.figure.savefig(path_panels / "png" / "nlicks_example_2ADC.png", dpi=300)
    nlicks_example_2ADC
    return (nlicks_example_2ADC,)


@app.cell
def _(
    EXAMPLE_MODEL,
    EXAMPLE_SUBJECTS,
    axd,
    example_bins,
    example_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    rt_example_2AFC = axd["rt_2AFC"] if mount_figure else plt.figure(
        figsize=fig_size(3, 1), constrained_layout=True,
    ).gca()
    rt_example_2AFC.clear()
    sns.histplot(
        data=example_dfs["2AFC", "RT"], x="value", hue="state",
        hue_order=state_order, palette=state_palette, bins=example_bins["2AFC", "RT"],
        stat="probability", common_norm=False, element="step", fill=False,
        linewidth=1, legend=False, ax=rt_example_2AFC,
    )
    rt_example_2AFC.set(
        xlabel="RT (s)", ylabel="",
        xlim=example_dfs["2AFC", "RT"]["value"].quantile([0.05, 0.95]),
        title=f"2AFC · {EXAMPLE_SUBJECTS['2AFC']}\n{EXAMPLE_MODEL}",
    )
    rt_example_2AFC.xaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=rt_example_2AFC)
    if not mount_figure:
        rt_example_2AFC.figure.savefig(path_panels / "svg" / "rt_example_2AFC.svg")
        rt_example_2AFC.figure.savefig(path_panels / "png" / "rt_example_2AFC.png", dpi=300)
    rt_example_2AFC
    return (rt_example_2AFC,)


@app.cell
def _(
    EXAMPLE_MODEL,
    EXAMPLE_SUBJECTS,
    axd,
    example_bins,
    example_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    nlicks_example_2AFC = axd["nlicks_2AFC"] if mount_figure else plt.figure(
        figsize=fig_size(3, 1), constrained_layout=True,
    ).gca()
    nlicks_example_2AFC.clear()
    sns.histplot(
        data=example_dfs["2AFC", "nLicks"], x="value", hue="state",
        hue_order=state_order, palette=state_palette, bins=example_bins["2AFC", "nLicks"],
        stat="probability", common_norm=False, element="step", fill=False,
        linewidth=1, legend=False, ax=nlicks_example_2AFC,
    )
    nlicks_example_2AFC.set(
        xlabel="nLicks", ylabel="",
        xlim=example_dfs["2AFC", "nLicks"]["value"].quantile([0.05, 0.95]),
        title=f"2AFC · {EXAMPLE_SUBJECTS['2AFC']}\n{EXAMPLE_MODEL}",
    )
    sns.despine(ax=nlicks_example_2AFC)
    if not mount_figure:
        nlicks_example_2AFC.figure.savefig(path_panels / "svg" / "nlicks_example_2AFC.svg")
        nlicks_example_2AFC.figure.savefig(path_panels / "png" / "nlicks_example_2AFC.png", dpi=300)
    nlicks_example_2AFC
    return (nlicks_example_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavioral AUC by emission constraint

    ### 2ADC + 2AFC (animal-level pooling)
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    add_grouped_model_pair_annotations,
    auc_metric_order,
    auc_model_order,
    axd,
    behavior_auc_dfs,
    fig_size,
    metric_palette,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    auc_behavior = axd["auc_behavior"] if mount_figure else plt.figure(
        figsize=fig_size(1, 2.5), constrained_layout=True,
    ).gca()
    auc_behavior.clear()
    sns.boxplot(
        data=behavior_auc_dfs["combined"], x="model", y="auc", hue="metric",
        order=auc_model_order, hue_order=auc_metric_order, palette=metric_palette,
        ax=auc_behavior, **BOXPLOT_STYLE,
    )
    auc_behavior.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    auc_behavior.set(xlabel="", ylabel="State AUC")
    add_grouped_model_pair_annotations(
        auc_behavior, behavior_auc_dfs["combined"],
        metric_order=auc_metric_order, model_order=auc_model_order,
        metric_palette=metric_palette,
    )
    auc_behavior.legend(title="", frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    sns.despine(ax=auc_behavior)
    if not mount_figure:
        auc_behavior.figure.savefig(path_panels / "svg" / "behavior_auc_combined.svg")
        auc_behavior.figure.savefig(path_panels / "png" / "behavior_auc_combined.png", dpi=300)
    auc_behavior
    return (auc_behavior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of states: frozen emissions

    ### 2ADC
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
    state_frozen_ll_2ADC.set(xlabel="Number of states K", ylabel="Frozen emissions\n$\\Delta$ LL vs GLM bits/trial", xticks=range(1, 8), xlim=(0.8, 7.2))
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
    state_frozen_bic_2ADC = plt.gca() if not mount_figure or "state_frozen_bic_2ADC" not in axd else axd["state_frozen_bic_2ADC"]
    state_frozen_bic_2ADC.clear()
    _plot_df = state_plot_dfs["frozen"]["2ADC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_bic_2ADC)
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_bic_2ADC)
    state_frozen_bic_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_bic_2ADC.set(xlabel="Number of states K", ylabel="ΔBIC vs GLM", xticks=range(1, 8), xlim=(0.8, 7.2))
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
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
    state_frozen_ll_2AFC = plt.gca() if not mount_figure or "state_frozen_ll_2AFC" not in axd else axd["state_frozen_ll_2AFC"]
    state_frozen_ll_2AFC.clear()
    _plot_df = state_plot_dfs["frozen"]["2AFC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_ll_2AFC)
    sns.lineplot(data=_plot_df, x="K", y="delta_ll_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_ll_2AFC)
    state_frozen_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_ll_2AFC.set(xlabel="Number of states K", ylabel="$\\Delta$ LL vs GLM bits/trial", xticks=range(1, 8))
    add_state_comparison_annotations(state_frozen_ll_2AFC, _plot_df, y="delta_ll_vs_glm")
    clean_plot_edges(state_frozen_ll_2AFC)
    plot_ashwood_point(state_frozen_ll_2AFC, ashwood_plot_dfs["2AFC"], "delta_ll_vs_glm")
    sns.despine(ax=state_frozen_ll_2AFC)
    if not mount_figure:
        state_frozen_ll_2AFC.figure.savefig(path_panels / "svg" / "state_frozen_delta_ll_2AFC.svg")
        state_frozen_ll_2AFC.figure.savefig(path_panels / "png" / "state_frozen_delta_ll_2AFC.png", dpi=300)
    state_frozen_ll_2AFC
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
    state_frozen_bic_2AFC = plt.gca() if not mount_figure or "state_frozen_bic_2AFC" not in axd else axd["state_frozen_bic_2AFC"]
    state_frozen_bic_2AFC.clear()
    _plot_df = state_plot_dfs["frozen"]["2AFC"]
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", units="subject", estimator=None, color="0.82", linewidth=0.6, marker="o", sort=True, ax=state_frozen_bic_2AFC)
    sns.lineplot(data=_plot_df, x="K", y="delta_bic_vs_glm", errorbar=("se", 1), color=family_palette["frozen"], marker="o", markersize=4, sort=True, ax=state_frozen_bic_2AFC)
    state_frozen_bic_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    state_frozen_bic_2AFC.set(xlabel="Number of states K", ylabel="ΔBIC vs GLM", xticks=range(1, 8))
    add_state_comparison_annotations(state_frozen_bic_2AFC, _plot_df, y="delta_bic_vs_glm")
    clean_plot_edges(state_frozen_bic_2AFC)
    plot_ashwood_point(state_frozen_bic_2AFC, ashwood_plot_dfs["2AFC"], "delta_bic_vs_glm")
    sns.despine(ax=state_frozen_bic_2AFC)
    if not mount_figure:
        state_frozen_bic_2AFC.figure.savefig(path_panels / "svg" / "state_frozen_delta_bic_2AFC.svg")
        state_frozen_bic_2AFC.figure.savefig(path_panels / "png" / "state_frozen_delta_bic_2AFC.png", dpi=300)
    state_frozen_bic_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final figure
    """)
    return


@app.cell
def _(
    auc_behavior,
    fig,
    freeze_ll_2ADC,
    freeze_ll_2AFC,
    mount_figure,
    nlicks_example_2ADC,
    nlicks_example_2AFC,
    path_panels,
    rt_example_2ADC,
    rt_example_2AFC,
    state_palette,
):
    mounted_axes = [
        freeze_ll_2ADC, freeze_ll_2AFC, auc_behavior,
        rt_example_2ADC, nlicks_example_2ADC, rt_example_2AFC, nlicks_example_2AFC,
    ]
    if mount_figure:
        freeze_ll_2ADC.set(xlabel="", title="2ADC")
        freeze_ll_2AFC.set(xlabel="", title="2AFC")
        for panel_label, axis in zip("abcdefg", mounted_axes):
            axis.text(-0.15 if axis in (freeze_ll_2ADC, freeze_ll_2AFC, auc_behavior) else -0.35,
                      1.12, panel_label, transform=axis.transAxes, weight="bold", va="top")
        for state, color in state_palette.items():
            rt_example_2ADC.plot([], [], color=color, label=state)
        rt_example_2ADC.legend(frameon=False, fontsize=6, loc="upper right")
        fig.savefig(path_panels / "supplementary_figure32.svg")
        fig.savefig(path_panels / "supplementary_figure32.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure32.pdf")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical tests
    """)
    return


@app.cell
def _(
    auc_metric_order,
    behavior_auc_dfs,
    freeze_plot_dfs,
    panel_names,
    state_plot_dfs,
    ttest_rel,
):
    tests = []

    _emission_pairs = [
        ("Stim0", "Hist0"),
        ("Stim0", "Free"),
        ("Stim0", "Both0"),
    ]
    for _task, _panel_key in {
        "2ADC": "freeze_ll_2ADC",
        "2AFC": "freeze_ll_2AFC",
    }.items():
        _test_df = freeze_plot_dfs[_task]
        for _left, _right in _emission_pairs:
            _paired = (
                _test_df.loc[
                    _test_df["model"].isin([_left, _right]),
                    ["subject", "model", "delta_ll_vs_glm"],
                ]
                .pivot_table(
                    index="subject",
                    columns="model",
                    values="delta_ll_vs_glm",
                    aggfunc="first",
                )
                .reindex(columns=[_left, _right])
                .dropna()
            )
            _result = ttest_rel(_paired[_left], _paired[_right])
            tests.append(
                {
                    "panel": panel_names[_panel_key],
                    "task": _task,
                    "measure": "ΔLL vs. GLM (bits/trial)",
                    "comparison": f"{_left} vs. {_right}",
                    "test": "Paired t-test",
                    "n_subjects": len(_paired),
                    "statistic": _result.statistic,
                    "degrees_of_freedom": _result.df,
                    "p_value_raw": _result.pvalue,
                    "alternative": "two-sided",
                }
            )

    _auc_pairs = [
        ("Stim0", "Hist0"),
        ("Stim0", "Free"),
        ("Stim0", "Both0"),
    ]
    for _task, _panel_key in {
        "combined": "auc_behavior",
    }.items():
        _test_df = behavior_auc_dfs[_task]
        for _metric in auc_metric_order:
            _metric_df = _test_df[_test_df["metric"].eq(_metric)]
            for _left, _right in _auc_pairs:
                _paired = (
                    _metric_df.loc[
                        _metric_df["model"].isin([_left, _right]),
                        ["subject", "model", "auc"],
                    ]
                    .pivot_table(
                        index="subject",
                        columns="model",
                        values="auc",
                        aggfunc="first",
                    )
                    .reindex(columns=[_left, _right])
                    .dropna()
                )
                _result = ttest_rel(_paired[_left], _paired[_right])
                tests.append(
                    {
                        "panel": panel_names[_panel_key],
                        "task": _task,
                        "measure": f"{_metric} state AUC",
                        "comparison": f"{_left} vs. {_right}",
                        "test": "Paired t-test",
                        "n_subjects": len(_paired),
                        "statistic": _result.statistic,
                        "degrees_of_freedom": _result.df,
                        "p_value_raw": _result.pvalue,
                        "alternative": "two-sided",
                    }
                )

    _state_metrics = [
        ("delta_ll_vs_glm", "state_frozen_ll", "ΔLL vs. GLM (bits/trial)"),
        ("delta_bic_vs_glm", "state_frozen_bic", "ΔBIC vs. GLM"),
    ]
    for _task in ("2ADC", "2AFC"):
        _test_df = state_plot_dfs["frozen"][_task]
        for _column, _panel_prefix, _measure in _state_metrics:
            _panel_key = f"{_panel_prefix}_{_task}"
            for _left, _right in ((1, 2), (2, 3)):
                _paired = (
                    _test_df.loc[
                        _test_df["K"].isin([_left, _right]),
                        ["subject", "K", _column],
                    ]
                    .pivot_table(
                        index="subject",
                        columns="K",
                        values=_column,
                        aggfunc="first",
                    )
                    .reindex(columns=[_left, _right])
                    .dropna()
                )
                _result = ttest_rel(_paired[_right], _paired[_left])
                tests.append(
                    {
                        "panel": panel_names[_panel_key],
                        "task": _task,
                        "measure": _measure,
                        "comparison": f"K={_right} vs. K={_left}",
                        "test": "Paired t-test",
                        "n_subjects": len(_paired),
                        "statistic": _result.statistic,
                        "degrees_of_freedom": _result.df,
                        "p_value_raw": _result.pvalue,
                        "alternative": "two-sided",
                    }
                )

    for _panel in {_test["panel"] for _test in tests}:
        _panel_tests = [_test for _test in tests if _test["panel"] == _panel]
        _n_tests = len(_panel_tests)
        for _test in _panel_tests:
            _test["p_value_adjusted"] = min(
                float(_test["p_value_raw"]) * _n_tests,
                1.0,
            )
            _test["correction"] = f"Bonferroni within panel (m={_n_tests})"
    return (tests,)


@app.cell
def _(mo, path_panels, pd, tests):
    tests_df = pd.DataFrame(tests)[
        [
            "panel",
            "task",
            "measure",
            "comparison",
            "test",
            "n_subjects",
            "statistic",
            "degrees_of_freedom",
            "p_value_raw",
            "p_value_adjusted",
            "alternative",
            "correction",
        ]
    ]

    tex_path = path_panels / "table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tests_latex = tests_df.to_latex(
        buf=tex_path,
        index=False,
        float_format="%.3g",
        escape=True,
    )

    tests_latex = tests_df.to_latex(
        index=False,
        float_format="%.3g",
        escape=True,
    )
    mo.vstack(
        [
            tests_df,
            mo.md(f"```latex\n{tests_latex}\n```"),
        ]
    )
    return


@app.cell
def _(example_counts_df, path_panels):
    example_counts_df.to_csv(path_panels / "example_trial_counts.csv", index=False)
    example_counts_df
    return


if __name__ == "__main__":
    app.run()
