# /// script
# [tool.marimo.opengraph]
# title = "Supplementary Figure 4.1"
# description = "Drug-regressor placement in two-state GLM-HMM-t models."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supplementary 4.1: Model comparison for the drug regressor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Description

    We compare three two-state GLM-HMM-t models for the 2ADC and 2AFC tasks: a model without a drug regressor, a model with the drug regressor only in transitions, and a model with drug effects in the emissions. Model fit is quantified by the change in held-out log-likelihood relative to the model without the drug regressor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports and settings
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_rel

    def fig_size(n_cols=1, ratio=None):
        """Return an A4-column figure size in inches."""
        ratio = ratio or plt.rcParams["figure.figsize"][0] / plt.rcParams["figure.figsize"][1]
        width = (210 - 50.8) / n_cols
        return width / 25.4, width / ratio / 25.4

    return Path, fig_size, math, mo, pd, pl, plt, sns, ttest_rel


@app.cell
def _(Path, plt, sns):
    ROOT = Path(__file__).resolve().parents[1]
    path_panels = ROOT / "supplementary figures" / "panels41"
    for panel_format in ("svg", "png"):
        (path_panels / panel_format).mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return ROOT, path_panels


@app.cell
def _(ROOT):
    mount_figure = True
    model_order = ["No drug", "Transitions", "Emissions"]
    panel_names = {"drug_ll_2ADC": "S4a", "drug_ll_2AFC": "S4b"}
    task_configs = {
        "2ADC": {
            "fit_root": ROOT / "results" / "fits" / "2ADC_DRUG" / "glmhmmt",
            "models": {
                "No drug": "base_param",
                "Transitions": "drug_transitions",
                "Emissions": "drug_emissions",
            },
        },
        "2AFC": {
            "fit_root": ROOT / "results" / "fits" / "2AFC_DRUG" / "glmhmmt",
            "models": {
                "No drug": "base_model",
                "Transitions": "drug_transitions",
                "Emissions": "drug_emissions",
            },
        },
    }
    return model_order, mount_figure, panel_names, task_configs


@app.cell
def _(math, model_order, pd, pl, task_configs):
    def read_model_metrics(directory, model_label, model_index):
        """Read one row of held-out fit metrics per subject for one model."""
        metrics = pl.concat(
            [pl.read_parquet(path) for path in sorted(directory.glob("*_K2_glmhmmt_metrics.parquet"))],
            how="diagonal_relaxed",
        )
        return metrics.select("subject", "test_ll_per_trial_mean").with_columns(
            pl.col("subject").cast(pl.Utf8),
            pl.lit(model_label).alias("model"),
            pl.lit(model_index, dtype=pl.Int64).alias("model_order"),
        )

    plot_dfs = {}
    count_rows = []
    for _task_label, _config in task_configs.items():
        model_metrics = pl.concat(
            [
                read_model_metrics(
                    _config["fit_root"] / model_id,
                    model_label,
                    model_index,
                )
                for model_index, (model_label, model_id) in enumerate(_config["models"].items())
            ],
            how="vertical",
        )
        no_drug_scores = (
            model_metrics
            .filter(pl.col("model") == "No drug")
            .select(
                "subject",
                pl.col("test_ll_per_trial_mean").alias("no_drug_ll"),
            )
        )
        comparison_df = (
            model_metrics
            .join(no_drug_scores, on="subject", how="inner")
            .with_columns(
                (
                    (pl.col("test_ll_per_trial_mean") - pl.col("no_drug_ll"))
                    / math.log(2)
                ).alias("delta_ll_vs_no_drug")
            )
            .sort(["model_order", "subject"])
        )
        plot_df = comparison_df.to_pandas()
        plot_df["model"] = pd.Categorical(
            plot_df["model"], categories=model_order, ordered=True
        )
        plot_dfs[_task_label] = plot_df
        count_rows.append(
            {
                "task": _task_label,
                "subjects": comparison_df.get_column("subject").n_unique(),
                "models": comparison_df.get_column("model").n_unique(),
            }
        )

    data_counts = pl.DataFrame(count_rows)
    return data_counts, plot_dfs


@app.cell
def _(data_counts):
    data_counts
    return


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [["drug_ll_2ADC", "drug_ll_2AFC"]],
            figsize=fig_size(1, 2),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell
def _(math, model_order, ttest_rel):
    model_pairs = [
        ("No drug", "Transitions"),
        ("No drug", "Emissions"),
        ("Transitions", "Emissions"),
    ]

    def paired_test(dataframe, left, right):
        """Return a paired t-test after aligning model scores by subject."""
        paired = (
            dataframe.loc[
                dataframe["model"].isin([left, right]),
                ["subject", "model", "delta_ll_vs_no_drug"],
            ]
            .pivot_table(
                index="subject",
                columns="model",
                values="delta_ll_vs_no_drug",
                aggfunc="first",
                observed=False,
            )
            .reindex(columns=[left, right])
            .dropna()
        )
        test = ttest_rel(paired[left], paired[right])
        return len(paired), float(test.statistic), float(test.pvalue)

    def significance_stars(pvalue):
        if not math.isfinite(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_pair_annotations(axis, dataframe):
        tests = []
        for left, right in model_pairs:
            n_subjects, statistic, pvalue = paired_test(dataframe, left, right)
            tests.append((left, right, n_subjects, statistic, min(pvalue * len(model_pairs), 1.0)))

        y_values = dataframe["delta_ll_vs_no_drug"].dropna()
        y_range = max(float(y_values.max() - y_values.min()), 0.01)
        y_start = float(y_values.max()) + 0.10 * y_range
        y_step = 0.15 * y_range
        annotation_index = 0
        for left, right, _, _, corrected_pvalue in tests:
            stars = significance_stars(corrected_pvalue)
            if not stars:
                continue
            left_x, right_x = model_order.index(left), model_order.index(right)
            y = y_start + annotation_index * y_step
            axis.plot(
                [left_x, left_x, right_x, right_x],
                [y, y + 0.03 * y_range, y + 0.03 * y_range, y],
                color="black",
                linewidth=0.7,
                clip_on=False,
            )
            axis.text(
                (left_x + right_x) / 2,
                y + 0.04 * y_range,
                stars,
                ha="center",
                va="bottom",
            )
            annotation_index += 1
        if annotation_index:
            axis.set_ylim(top=y_start + annotation_index * y_step)
        return tests

    def clean_plot_edges(axis):
        for line in axis.lines:
            line.set_markeredgewidth(0)
            line.set_markeredgecolor("none")
        for collection in axis.collections:
            collection.set_edgecolor("none")

    return add_pair_annotations, clean_plot_edges, model_pairs, paired_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drug-regressor placement
    """)
    return


@app.cell
def _(
    add_pair_annotations,
    axd,
    clean_plot_edges,
    model_order,
    mount_figure,
    path_panels,
    plot_dfs,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    drug_ll_2ADC = plt.gca() if not mount_figure else axd["drug_ll_2ADC"]
    drug_ll_2ADC.clear()
    _plot_df = plot_dfs["2ADC"]
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_no_drug",
        units="subject",
        estimator=None,
        color="0.82",
        linewidth=0.6,
        marker="o",
        sort=False,
        ax=drug_ll_2ADC,
    )
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_no_drug",
        errorbar=("se", 1),
        color="black",
        linewidth=1,
        marker="o",
        markersize=4,
        markeredgewidth=0,
        markeredgecolor="none",
        sort=False,
        ax=drug_ll_2ADC,
    )
    drug_ll_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_pair_annotations(drug_ll_2ADC, _plot_df)
    drug_ll_2ADC.set(
        xlabel="",
        ylabel=r"$\Delta$ held-out LL vs no drug (bits/trial)",
    )
    drug_ll_2ADC.set_xticks(range(len(model_order)), model_order)
    clean_plot_edges(drug_ll_2ADC)
    sns.despine(ax=drug_ll_2ADC)
    if not mount_figure:
        drug_ll_2ADC.figure.savefig(path_panels / "svg" / "drug_delta_ll_2ADC.svg")
        drug_ll_2ADC.figure.savefig(path_panels / "png" / "drug_delta_ll_2ADC.png", dpi=300)
    drug_ll_2ADC
    return (drug_ll_2ADC,)


@app.cell
def _(
    add_pair_annotations,
    axd,
    clean_plot_edges,
    model_order,
    mount_figure,
    path_panels,
    plot_dfs,
    plt,
    sns,
):
    plt.figure(figsize=(3, 3), constrained_layout=True)
    drug_ll_2AFC = plt.gca() if not mount_figure else axd["drug_ll_2AFC"]
    drug_ll_2AFC.clear()
    _plot_df = plot_dfs["2AFC"]
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_no_drug",
        units="subject",
        estimator=None,
        color="0.82",
        linewidth=0.6,
        marker="o",
        sort=False,
        ax=drug_ll_2AFC,
    )
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_no_drug",
        errorbar=("se", 1),
        color="black",
        linewidth=1,
        marker="o",
        markersize=4,
        markeredgewidth=0,
        markeredgecolor="none",
        sort=False,
        ax=drug_ll_2AFC,
    )
    drug_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_pair_annotations(drug_ll_2AFC, _plot_df)
    drug_ll_2AFC.set(xlabel="", ylabel="")
    drug_ll_2AFC.set_xticks(range(len(model_order)), model_order)
    clean_plot_edges(drug_ll_2AFC)
    sns.despine(ax=drug_ll_2AFC)
    if not mount_figure:
        drug_ll_2AFC.figure.savefig(path_panels / "svg" / "drug_delta_ll_2AFC.svg")
        drug_ll_2AFC.figure.savefig(path_panels / "png" / "drug_delta_ll_2AFC.png", dpi=300)
    drug_ll_2AFC
    return (drug_ll_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final figure
    """)
    return


@app.cell
def _(drug_ll_2ADC, drug_ll_2AFC, fig, mount_figure, path_panels):
    if mount_figure:
        drug_ll_2ADC.set_title("2ADC")
        drug_ll_2AFC.set_title("2AFC")
        fig.align_labels()
        fig.savefig(path_panels / "supplementary_figure41.svg")
        fig.savefig(path_panels / "supplementary_figure41.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure41.pdf")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical tests

    Paired two-sided t-tests compare held-out log-likelihood across subjects. P-values are Bonferroni-corrected over the three model comparisons within each panel.
    """)
    return


@app.cell
def _(model_pairs, paired_test, panel_names, pl, plot_dfs):
    test_rows = []
    for _task_label, _dataframe in plot_dfs.items():
        for left, right in model_pairs:
            n_subjects, statistic, pvalue = paired_test(_dataframe, left, right)
            test_rows.append(
                {
                    "panel": panel_names[f"drug_ll_{_task_label}"],
                    "task": _task_label,
                    "comparison": f"{left} vs {right}",
                    "n": n_subjects,
                    "t": statistic,
                    "p": pvalue,
                    "p_bonferroni": min(pvalue * len(model_pairs), 1.0),
                }
            )
    statistical_tests = pl.DataFrame(test_rows)
    statistical_tests
    return


if __name__ == "__main__":
    app.run()
