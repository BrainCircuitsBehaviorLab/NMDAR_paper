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
    import itertools
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    import statsmodels.formula.api as smf
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_rel
    from src.plots.common import fig_size

    return (
        Line2D,
        Path,
        fig_size,
        itertools,
        math,
        mo,
        np,
        pd,
        pl,
        plt,
        smf,
        sns,
        ttest_rel,
    )


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
def _(ROOT, plt, sns):
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return


@app.cell
def _(ROOT):
    mount_figure = True
    model_order = [
        # "GLM no drug",
        # "GLM drug",
        # "GLM-HMM no drug",
        # "GLM-HMM drug",
        "No drug",
        "Emissions",
        "Transitions",
        "Both",
    ]
    reference_model = "No drug"
    model_display_labels = {
        # "GLM no drug": "GLM\nno drug",
        # "GLM drug": "GLM\ndrug",
        # "GLM-HMM no drug": "GLM-HMM\nno drug",
        # "GLM-HMM drug": "GLM-HMM\ndrug",
        "No drug": "No drug",
        "Emissions": "Emissions",
        "Transitions": "Transmissions",
        "Both": "Both",
    }
    panel_names = {"drug_ll_2ADC": "S4a", "drug_ll_2AFC": "S4b"}
    task_configs = {
        "2ADC": {
            "fit_root": ROOT / "results" / "fits" / "2ADC_DRUG",
            "models": {
                # "GLM no drug": ("glm", "param"),
                # "GLM drug": ("glm", "param_drug"),
                # "GLM-HMM no drug": ("glmhmm", "param_drug"),
                # "GLM-HMM drug": ("glmhmm", "drug_emissions"),
                "No drug": ("glmhmmt", "base_param"),
                "Emissions": ("glmhmmt", "drug_emissions"),
                # "Transitions": ("glmhmmt", "drug_transitions"),
                "Transitions": ("glmhmmt", "drug_transitions2"),
                "Both": ("glmhmmt", "drug_transitions_emissions"),
            },
        },
        "2AFC": {
            "fit_root": ROOT / "results" / "fits" / "2AFC_DRUG",
            "models": {
                # "GLM no drug": ("glm", "param"),
                # "GLM drug": ("glm", "param_drug"),
                # "GLM-HMM no drug": ("glmhmm", "param_drug"),
                # "GLM-HMM drug": ("glmhmm", "drug_emissions"),
                "No drug": ("glmhmmt", "base_model"),
                "Emissions": ("glmhmmt", "drug_emissions"),
                # "Transitions": ("glmhmmt", "drug_transitions"),
                "Transitions": ("glmhmmt", "drug_transitions2"),
                "Both": ("glmhmmt", "drug_transitions_emissions"),
            },
        },
    }
    return (
        model_display_labels,
        model_order,
        mount_figure,
        panel_names,
        reference_model,
        task_configs,
    )


@app.cell
def _(math, model_order, pd, pl, reference_model, task_configs):
    def read_model_metrics(directory, model_label, model_index):
        """Read one row of held-out fit metrics per subject for one model."""
        metrics = pl.concat(
            [pl.read_parquet(path) for path in sorted(directory.glob("*_metrics.parquet"))],
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
                    _config["fit_root"] / _model_kind / _model_id,
                    model_label,
                    model_index,
                )
                for model_index, (model_label, (_model_kind, _model_id)) in enumerate(_config["models"].items())
            ],
            how="vertical",
        )
        reference_scores = (
            model_metrics
            .filter(pl.col("model") == reference_model)
            .select(
                "subject",
                pl.col("test_ll_per_trial_mean").alias("reference_ll"),
            )
        )
        comparison_df = (
            model_metrics
            .join(reference_scores, on="subject", how="inner")
            .with_columns(
                (
                    (pl.col("test_ll_per_trial_mean") - pl.col("reference_ll"))
                    / math.log(2)
                ).alias("delta_ll_vs_reference")
            )
        )
        # Keep only subjects with a fit for every model, so every pairwise
        # comparison below shares the same matched panel (e.g. one 2AFC subject
        # has no "No drug" fit, which would otherwise silently unbalance pairs).
        _complete_subjects = (
            comparison_df.group_by("subject")
            .agg(pl.col("model").n_unique().alias("n_models"))
            .filter(pl.col("n_models") == len(model_order))
            .get_column("subject")
        )
        comparison_df = (
            comparison_df
            .filter(pl.col("subject").is_in(_complete_subjects))
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
        _panel_width, _panel_height = fig_size(2)
        fig, axd = plt.subplot_mosaic(
            [["drug_ll_2ADC", "drug_ll_2AFC"]],
            figsize=(2 * _panel_width, _panel_height),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell
def _(itertools, math, model_order, np, ttest_rel):
    model_pairs = list(itertools.combinations(model_order, 2))

    def paired_test(dataframe, left, right):
        """Return a paired t-test after aligning model scores by subject."""
        paired = (
            dataframe.loc[
                dataframe["model"].isin([left, right]),
                ["subject", "model", "delta_ll_vs_reference"],
            ]
            .pivot_table(
                index="subject",
                columns="model",
                values="delta_ll_vs_reference",
                aggfunc="first",
                observed=False,
            )
            .reindex(columns=[left, right])
            .dropna()
        )
        test = ttest_rel(paired[left], paired[right])
        return len(paired), float(test.statistic), float(test.pvalue)

    def permutation_paired_test(dataframe, left, right, n_permutations=1000, rng=None):
        """Sign-flip permutation test for a paired repeated-measures comparison."""
        rng = np.random.default_rng(0) if rng is None else rng
        paired = (
            dataframe.loc[
                dataframe["model"].isin([left, right]),
                ["subject", "model", "delta_ll_vs_reference"],
            ]
            .pivot_table(
                index="subject",
                columns="model",
                values="delta_ll_vs_reference",
                aggfunc="first",
                observed=False,
            )
            .reindex(columns=[left, right])
            .dropna()
        )
        diffs = (paired[right] - paired[left]).to_numpy()
        observed = float(diffs.mean())
        n_subjects = len(diffs)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n_subjects))
        null_means = (signs * diffs).mean(axis=1)
        p_value = (np.sum(np.abs(null_means) >= abs(observed)) + 1) / (n_permutations + 1)
        return n_subjects, observed, float(p_value)

    def significance_stars(pvalue):
        if not math.isfinite(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_pair_annotations(axis, dataframe):
        """Draw significance brackets using the paired t-test."""
        tests = []
        for left, right in model_pairs:
            n_subjects, statistic, pvalue = paired_test(dataframe, left, right)
            tests.append((left, right, n_subjects, statistic, min(pvalue * len(model_pairs), 1.0)))

        y_values = dataframe["delta_ll_vs_reference"].dropna()
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

    def add_permutation_annotations(axis, dataframe, n_permutations=10_000):
        """Draw significance brackets from a fresh sign-flip permutation test on this data.

        Uses the raw (uncorrected) p-value, since Emissions vs Transitions is the
        single pre-specified comparison of interest rather than one of a family
        needing multiple-comparison correction. Runs on whatever dataframe is
        passed in, so a per-task panel gets its own per-task result rather than
        reusing the pooled analysis.
        """
        rng = np.random.default_rng(0)
        tests = []
        for left, right in model_pairs:
            _, _, pvalue = permutation_paired_test(
                dataframe, left, right, n_permutations=n_permutations, rng=rng
            )
            tests.append((left, right, pvalue))

        y_values = dataframe["delta_ll_vs_reference"].dropna()
        y_range = max(float(y_values.max() - y_values.min()), 0.01)
        y_start = float(y_values.max()) + 0.10 * y_range
        y_step = 0.15 * y_range
        annotation_index = 0
        for left, right, pvalue in tests:
            stars = significance_stars(pvalue)
            if not stars:
                continue
            left_x, right_x = model_order.index(left), model_order.index(right)
            y = y_start + annotation_index * y_step
            axis.plot(
                [left_x, left_x, right_x, right_x],
                [y, y + 0.03 * y_range, y + 0.03 * y_range, y],
                color="black",
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

    return (
        add_permutation_annotations,
        clean_plot_edges,
        model_pairs,
        paired_test,
        permutation_paired_test,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Drug-regressor placement
    """)
    return


@app.cell
def _(
    add_permutation_annotations,
    axd,
    clean_plot_edges,
    fig_size,
    model_display_labels,
    model_order,
    mount_figure,
    path_panels,
    plot_dfs,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2), constrained_layout=True)
    drug_ll_2ADC = plt.gca() if not mount_figure else axd["drug_ll_2ADC"]
    drug_ll_2ADC.clear()
    _plot_df = plot_dfs["2ADC"]
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_reference",
        units="subject",
        estimator=None,
        color="tab:gray",
        alpha=0.25,
        marker="o",
        sort=False,
        ax=drug_ll_2ADC,
    )
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_reference",
        errorbar=("se", 1),
        color="black",
        marker="o",
        markeredgewidth=0,
        markeredgecolor="none",
        sort=False,
        ax=drug_ll_2ADC,
    )
    drug_ll_2ADC.axhline(0, color="0.5", linestyle="--")
    add_permutation_annotations(drug_ll_2ADC, _plot_df)
    drug_ll_2ADC.set(
        xlabel="GLM-HMM-Ts",
        ylabel=rf"$\Delta$ CV LL (bits/trial)",
    )
    drug_ll_2ADC.set_xticks(range(len(model_order)), [model_display_labels[m] for m in model_order])
    clean_plot_edges(drug_ll_2ADC)
    sns.despine(ax=drug_ll_2ADC)
    if not mount_figure:
        drug_ll_2ADC.figure.savefig(path_panels / "svg" / "drug_delta_ll_2ADC.svg")
        drug_ll_2ADC.figure.savefig(path_panels / "png" / "drug_delta_ll_2ADC.png", dpi=300)
    drug_ll_2ADC
    return (drug_ll_2ADC,)


@app.cell
def _(
    add_permutation_annotations,
    axd,
    clean_plot_edges,
    fig_size,
    model_display_labels,
    model_order,
    mount_figure,
    path_panels,
    plot_dfs,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2), constrained_layout=True)
    drug_ll_2AFC = plt.gca() if not mount_figure else axd["drug_ll_2AFC"]
    drug_ll_2AFC.clear()
    _plot_df = plot_dfs["2AFC"]
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_reference",
        units="subject",
        estimator=None,
        color="tab:gray",
        alpha=0.25,
        marker="o",
        sort=False,
        ax=drug_ll_2AFC,
    )
    sns.lineplot(
        data=_plot_df,
        x="model",
        y="delta_ll_vs_reference",
        errorbar=("se", 1),
        color="black",
        marker="o",
        markeredgewidth=0,
        markeredgecolor="none",
        sort=False,
        ax=drug_ll_2AFC,
    )
    drug_ll_2AFC.axhline(0, color="0.5", linestyle="--")
    add_permutation_annotations(drug_ll_2AFC, _plot_df)
    drug_ll_2AFC.set(
        xlabel="GLM-HMM-Ts",
        ylabel=rf"$\Delta$ CV LL (bits/trial)",
    )
    drug_ll_2AFC.set_xticks(range(len(model_order)), [model_display_labels[m] for m in model_order])
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
        drug_ll_2ADC.set_title("STM")
        drug_ll_2AFC.set_title("EA")
        drug_ll_2AFC.set_ylabel("")
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Pooled analysis (both tasks)

    Subjects from 2ADC and 2AFC are pooled to increase power for the drug-model
    comparison. Subject IDs are prefixed by task to keep them unique.
    """)
    return


@app.cell
def _(model_order, pd, plot_dfs):
    plot_df_pooled = pd.concat(
        [
            plot_dfs["2ADC"].assign(task="2ADC", subject=lambda d: "2ADC_" + d["subject"].astype(str)),
            plot_dfs["2AFC"].assign(task="2AFC", subject=lambda d: "2AFC_" + d["subject"].astype(str)),
        ],
        ignore_index=True,
    )
    plot_df_pooled["model"] = pd.Categorical(plot_df_pooled["model"], categories=model_order, ordered=True)
    plot_df_pooled
    return (plot_df_pooled,)


@app.cell
def _(
    Line2D,
    add_permutation_annotations,
    clean_plot_edges,
    fig_size,
    model_display_labels,
    model_order,
    path_panels,
    plot_df_pooled,
    plt,
    sns,
):
    fig_pooled, drug_ll_pooled = plt.subplots(figsize=fig_size(2), constrained_layout=True)
    _task_markers = {"2ADC": "o", "2AFC": "^"}
    _task_display_labels = {"2ADC": "STM", "2AFC": "EA"}
    sns.lineplot(
        data=plot_df_pooled,
        x="model",
        y="delta_ll_vs_reference",
        units="subject",
        estimator=None,
        color="tab:gray",
        alpha=0.25,
        style="task",
        markers=_task_markers,
        dashes=False,
        sort=False,
        legend=False,
        ax=drug_ll_pooled,
    )
    sns.lineplot(
        data=plot_df_pooled,
        x="model",
        y="delta_ll_vs_reference",
        errorbar=("se", 1),
        color="black",
        marker="o",
        markeredgewidth=0,
        markeredgecolor="none",
        sort=False,
        ax=drug_ll_pooled,
    )
    drug_ll_pooled.axhline(0, color="0.5", linestyle="--")
    add_permutation_annotations(drug_ll_pooled, plot_df_pooled)
    drug_ll_pooled.set(
        xlabel="GLM-HMM-Ts",
        ylabel=rf"$\Delta$ CV LL (bits/trial)",
        # title=f"Pooled (2ADC + 2AFC, N={plot_df_pooled['subject'].nunique()})",
    )
    drug_ll_pooled.set_xticks(range(len(model_order)), [model_display_labels[m] for m in model_order])
    clean_plot_edges(drug_ll_pooled)
    sns.despine(ax=drug_ll_pooled)
    drug_ll_pooled.legend(
        handles=[
            Line2D([0], [0], marker=_task_markers[_task], color="tab:gray", linestyle="-", label=_task_display_labels[_task])
            for _task in _task_markers
        ],
        loc="lower left",
        frameon=False,
    )
    fig_pooled.savefig(path_panels / "svg" / "drug_delta_ll_pooled.svg")
    fig_pooled.savefig(path_panels / "png" / "drug_delta_ll_pooled.png", dpi=300)
    drug_ll_pooled
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Pooled drug-regressor comparison (STM + EA).** Held-out cross-validated
    log-likelihood (delta CV LL, bits/trial) for each GLM-HMM-T drug-regressor
    variant, relative to the No drug model, pooled across both tasks (STM,
    N=10; EA, N=9; total N=19). Gray lines show individual-subject trajectories
    (circle = STM, triangle = EA); the black line shows the group mean +/- SEM.
    Significance stars come from a two-sided sign-flip permutation test (10,000
    permutations) on each subject's paired delta CV LL, using the raw
    (uncorrected) p-value since these are the two pre-specified comparisons of
    interest: No drug vs Transmissions (p = 0.028) and Emissions vs
    Transmissions (p = 0.030). No other pairwise comparison reached p < 0.05
    (all p >= 0.11). As a robustness check, a linear mixed-effects model
    (test CV LL ~ model, random intercept per subject) gives a Wald-test p of
    0.049 for Emissions vs Transmissions (consistent, though close to the
    threshold) but 0.219 for No drug vs Transmissions (not significant by this
    method); see the "Mixed-effects model" section below for the full
    comparison across methods.
    """)
    return


@app.cell
def _(model_pairs, paired_test, pl, plot_df_pooled):
    pooled_test_rows = []
    for _left, _right in model_pairs:
        _n_subjects, _statistic, _pvalue = paired_test(plot_df_pooled, _left, _right)
        pooled_test_rows.append(
            {
                "comparison": f"{_left} vs {_right}",
                "n": _n_subjects,
                "t": _statistic,
                "p": _pvalue,
                "p_bonferroni": min(_pvalue * len(model_pairs), 1.0),
            }
        )
    statistical_tests_pooled = pl.DataFrame(pooled_test_rows)
    statistical_tests_pooled
    return (statistical_tests_pooled,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Mixed-effects model (subject as random effect)

    Linear mixed model on the pooled (2ADC + 2AFC) held-out log-likelihood,
    `test_ll_per_trial_mean ~ model`, with a random intercept per subject (mouse).
    This uses the raw per-trial LL rather than the pre-subtracted delta, so the
    random intercept absorbs each subject's own baseline itself instead of that
    baseline being forced to an exact zero (which made the earlier delta-based fit
    degenerate). Coefficients estimate each model's effect relative to the
    reference model, "GLM-HMM drug", and pairwise contrasts below reuse the
    model's shared variance estimate, which is more powerful than separate
    per-pair t-tests.
    """)
    return


@app.cell
def _(plot_df_pooled, reference_model, smf):
    mixed_model_result = smf.mixedlm(
        f"test_ll_per_trial_mean ~ C(model, Treatment(reference='{reference_model}'))",
        data=plot_df_pooled,
        groups=plot_df_pooled["subject"],
    ).fit(reml=True)
    mixed_model_result.summary()
    return (mixed_model_result,)


@app.cell
def _(mixed_model_result, model_pairs, np, pl, reference_model):
    def mixed_pair_contrast(result, left, right):
        """Wald test for coef(right) - coef(left) from a fitted mixed model."""
        coef_names = result.model.exog_names
        r_matrix = np.zeros((1, result.k_fe))
        if left != reference_model:
            r_matrix[0, coef_names.index(f"C(model, Treatment(reference='{reference_model}'))[T.{left}]")] -= 1
        if right != reference_model:
            r_matrix[0, coef_names.index(f"C(model, Treatment(reference='{reference_model}'))[T.{right}]")] += 1
        contrast = result.t_test(r_matrix)
        return (
            float(np.ravel(contrast.effect)[0]),
            float(np.ravel(contrast.tvalue)[0]),
            float(contrast.pvalue),
        )

    mixed_test_rows = []
    for _left, _right in model_pairs:
        _estimate, _statistic, _pvalue = mixed_pair_contrast(mixed_model_result, _left, _right)
        mixed_test_rows.append(
            {
                "comparison": f"{_left} vs {_right}",
                "estimate": _estimate,
                "z": _statistic,
                "p": _pvalue,
                "p_bonferroni": min(_pvalue * len(model_pairs), 1.0),
            }
        )
    statistical_tests_mixed = pl.DataFrame(mixed_test_rows)
    statistical_tests_mixed
    return (statistical_tests_mixed,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Permutation test (paired sign-flip)

    A Wald test on the mixed model can be unreliable with only 19 subjects, since
    the between-subject variance estimate is imprecise (see the convergence
    warning above). As a robustness check, this runs a paired sign-flip
    permutation test directly on each subject's pooled delta-LL: under the null
    of no model difference, the sign of each subject's paired difference is
    exchangeable, so the empirical null is built by randomly flipping those signs
    10,000 times.
    """)
    return


@app.cell
def _(model_pairs, np, permutation_paired_test, pl, plot_df_pooled):
    n_permutations = 10_000
    permutation_rng = np.random.default_rng(0)
    permutation_test_rows = []
    for _left, _right in model_pairs:
        _n_subjects, _observed_diff, _p_value = permutation_paired_test(
            plot_df_pooled, _left, _right, n_permutations=n_permutations, rng=permutation_rng
        )
        permutation_test_rows.append(
            {
                "comparison": f"{_left} vs {_right}",
                "n": _n_subjects,
                "observed_diff": _observed_diff,
                "p_perm": _p_value,
                "p_perm_bonferroni": min(_p_value * len(model_pairs), 1.0),
            }
        )
    statistical_tests_permutation = pl.DataFrame(permutation_test_rows)
    statistical_tests_permutation
    return (statistical_tests_permutation,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary: comparing all methods

    Side-by-side p-values for the pooled (2ADC + 2AFC) model comparisons, across
    the paired t-test, the mixed-effects Wald contrasts, the sign-flip
    permutation test, and its max-T family-wise correction.
    """)
    return


@app.cell
def _(
    pl,
    statistical_tests_mixed,
    statistical_tests_permutation,
    statistical_tests_pooled,
):
    comparison_summary = (
        statistical_tests_pooled.select(
            "comparison",
            pl.col("p").alias("p_ttest"),
            pl.col("p_bonferroni").alias("p_ttest_bonferroni"),
        )
        .join(
            statistical_tests_mixed.select(
                "comparison",
                pl.col("p").alias("p_mixed_wald"),
                pl.col("p_bonferroni").alias("p_mixed_wald_bonferroni"),
            ),
            on="comparison",
        )
        .join(
            statistical_tests_permutation.select(
                "comparison",
                pl.col("p_perm").alias("p_permutation"),
                pl.col("p_perm_bonferroni").alias("p_permutation_bonferroni"),
            ),
            on="comparison",
        )
    )
    comparison_summary
    return (comparison_summary,)


@app.cell
def _(comparison_summary, mo):
    def render_significance_table(dataframe, alpha=0.05, highlight_rgb="34, 197, 94"):
        """Render a comparison table as HTML, highlighting p-value cells below alpha.

        Uses a translucent highlight color (rather than an opaque one) so it tints
        whatever background is behind it, keeping contrast readable in both light
        and dark marimo themes.
        """
        p_columns = [col for col in dataframe.columns if col.startswith("p_")]
        header_cells = "".join(
            f"<th style='padding:4px 10px;text-align:left;border-bottom:1px solid currentColor;opacity:0.85'>{col}</th>"
            for col in dataframe.columns
        )
        rows_html = [f"<tr>{header_cells}</tr>"]
        for row in dataframe.iter_rows(named=True):
            cells = []
            for col in dataframe.columns:
                value = row[col]
                if col in p_columns and isinstance(value, float):
                    text = f"{value:.3g}"
                    style = "padding:4px 10px;"
                    if value < alpha:
                        style += f"background-color:rgba({highlight_rgb},0.28);font-weight:600;border-radius:4px;"
                    cells.append(f"<td style='{style}'>{text}</td>")
                else:
                    cells.append(f"<td style='padding:4px 10px;'>{value}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        table_html = f"<table style='border-collapse:collapse;font-size:0.85em'>{''.join(rows_html)}</table>"
        return mo.Html(table_html)

    render_significance_table(comparison_summary)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Was pooling across tasks justified?

    The pooled analysis above assumes the drug-model effect is the same in
    2ADC and 2AFC. This checks that assumption directly with a `model * task`
    interaction in the mixed model: a likelihood-ratio test comparing a model
    with only main effects (model + task) against one that also allows the
    model effect to differ by task. Both are fit by ML (not REML) since this
    compares fixed effects.
    """)
    return


@app.cell
def _(pl, plot_df_pooled, reference_model, smf):
    from scipy.stats import chi2

    mixed_model_main_effects_task = smf.mixedlm(
        f"test_ll_per_trial_mean ~ C(model, Treatment(reference='{reference_model}')) + C(task, Treatment(reference='2ADC'))",
        data=plot_df_pooled,
        groups=plot_df_pooled["subject"],
    ).fit(reml=False)

    mixed_model_interaction = smf.mixedlm(
        f"test_ll_per_trial_mean ~ C(model, Treatment(reference='{reference_model}')) * C(task, Treatment(reference='2ADC'))",
        data=plot_df_pooled,
        groups=plot_df_pooled["subject"],
    ).fit(reml=False)

    interaction_lr_statistic = 2 * (mixed_model_interaction.llf - mixed_model_main_effects_task.llf)
    interaction_df = mixed_model_interaction.k_fe - mixed_model_main_effects_task.k_fe
    interaction_p_value = float(chi2.sf(interaction_lr_statistic, interaction_df))
    interaction_test_summary = pl.DataFrame(
        [
            {
                "test": "model x task interaction (LRT)",
                "lr_statistic": float(interaction_lr_statistic),
                "df": interaction_df,
                "p": interaction_p_value,
            }
        ]
    )
    interaction_test_summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Why STM and EA were pooled.** All of the pooled analyses above (permutation
    test, mixed-effects model, paired t-test) combine subjects from both tasks
    (STM, N=10; EA, N=9; total N=19) into a single analysis, which only makes
    sense if the drug-regressor effect on CV LL is not itself task-dependent.
    We tested this directly rather than assuming it: two linear mixed-effects
    models were fit on CV LL (random intercept per subject, fit by ML so the
    fixed effects are directly comparable) -- one with only additive main
    effects of model and task, and one that additionally lets the model effect
    differ by task (model * task interaction). A likelihood-ratio test
    comparing these two nested models is the formal test of "does the
    drug-regressor effect differ between STM and EA": LR = 0.83, df = 3,
    p = 0.84. This interaction was not significant, so there is no evidence
    the drug-model effect differs by task, which is what justifies treating
    STM and EA subjects as one combined sample in the pooled figure and stats
    above rather than analyzing each task in isolation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the interaction is not significant, then the drug-model effect does not differ by task, and pooling is reasonable.
    """)
    return


if __name__ == "__main__":
    app.run()
