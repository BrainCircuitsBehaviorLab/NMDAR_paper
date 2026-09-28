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

    We compare four two-state GLM-HMM-t models for the 2ADC and 2AFC tasks: no drug regressor, drug effects in transitions, drug effects in emissions, and drug effects in both components. For each animal and model, the CV score is the sum of held-out log-likelihoods divided by the sum of held-out trial counts across the five folds. Differences from the no-drug model are divided by ln(2) to obtain bits per trial. Each panel includes only subjects with all four models; animals have equal weight in the group summary.

    Comparing Both with Transitions measures the additional predictive value of the emission terms; comparing Both with Emissions measures the additional value of the transition terms. These interpretations require identical shared regressors, shared parameter constraints, and test sessions across models. Transitions uses drug_transitions2.
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
    from itertools import combinations
    import math
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    import statsmodels.formula.api as smf
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_rel
    from statsmodels.formula.api import ols
    from glmhmmt.notebook_support.analysis_common import build_trial_and_weights_df, load_fit_arrays
    from glmhmmt.runtime import configure_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import two_adc, two_afc
    from src.process.common import glmhmmt_state_dwell_df
    from src.plots.common import BOXPLOT_STYLE

    def fig_size(n_cols=1, ratio=None):
        """Return an A4-column figure size in inches."""
        ratio = ratio or plt.rcParams["figure.figsize"][0] / plt.rcParams["figure.figsize"][1]
        width = (210 - 50.8) / n_cols
        return width / 25.4, width / ratio / 25.4

    return (
        BOXPLOT_STYLE,
        Path,
        build_trial_and_weights_df,
        build_views,
        combinations,
        configure_paths,
        fig_size,
        get_adapter,
        glmhmmt_state_dwell_df,
        json,
        load_fit_arrays,
        math,
        mo,
        ols,
        pd,
        pl,
        plt,
        sns,
        ttest_rel,
        two_adc,
        two_afc,
    )


@app.cell
def _(Path, configure_paths, plt, sns):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    path_panels = ROOT / "supplementary figures" / "panels41"
    for panel_format in ("svg", "png"):
        (path_panels / panel_format).mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    task_palette = {"2ADC": "tab:blue", "2AFC": "tab:orange"}
    return ROOT, path_panels, task_palette


@app.cell
def _(ROOT):
    mount_figure = True
    model_order = ["No drug", "Transitions", "Emissions", "Both"]
    model_tick_labels = model_order
    panel_names = {"drug_ll_2ADC": "S4a", "drug_ll_2AFC": "S4b", "drug_ll_pooled": "S4 pooled"}
    task_configs = {
        "2ADC": {
            "fit_root": ROOT / "results" / "fits" / "2ADC_DRUG" / "glmhmmt",
            "models": {
                "No drug": "base_param",
                "Transitions": "drug_transitions2",
                "Emissions": "drug_emissions",
                "Both": "drug_transitions_emissions",
            },
        },
        "2AFC": {
            "fit_root": ROOT / "results" / "fits" / "2AFC_DRUG" / "glmhmmt",
            "models": {
                "No drug": "base_model",
                "Transitions": "drug_transitions2",
                "Emissions": "drug_emissions",
                "Both": "drug_transitions_emissions",
            },
        },
    }
    return (
        model_order,
        model_tick_labels,
        mount_figure,
        panel_names,
        task_configs,
    )


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
        "Transitions": "Transitions",
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
        """Aggregate held-out LL over folds, weighting each fold by its trial count."""
        folds = pl.concat(
            [pl.read_parquet(path) for path in sorted(directory.glob("*_K2_glmhmmt_cv_repeats.parquet"))],
            how="diagonal_relaxed",
        )
        metrics = folds.group_by("subject").agg(
            pl.col("test_raw_ll").sum(),
            pl.col("test_T").sum(),
            pl.col("repeat_index").n_unique().alias("cv_folds"),
        )
        return metrics.with_columns(
            (pl.col("test_raw_ll") / pl.col("test_T")).alias("ll_cv"),
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
        complete_subjects = (
            model_metrics.group_by("subject")
            .agg(pl.col("model").n_unique().alias("n_models"))
            .filter(pl.col("n_models") == len(model_order))
            .select("subject")
        )
        model_metrics = model_metrics.join(complete_subjects, on="subject", how="semi")
        no_drug_scores = (
            model_metrics
            .filter(pl.col("model") == reference_model)
            .select(
                "subject",
                pl.col("ll_cv").alias("no_drug_ll"),
            )
        )
        comparison_df = (
            model_metrics
            .join(reference_scores, on="subject", how="inner")
            .with_columns(
                (
                    (pl.col("ll_cv") - pl.col("no_drug_ll"))
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dwell times by model and treatment

    Free = Both (drug_transitions_emissions); Transitions = drug_transitions2;
    Emissions = drug_emissions. As in Figure 4, dwell times
    are contiguous MAP-state runs within sessions, averaged across bouts for each
    animal and treatment. Boxes summarize animal means; grey lines pair saline
    and drug within animals. The y-axis is logarithmic. These panels are descriptive.
    An animal without bouts in a state/treatment has no dwell-time estimate for
    that group and contributes no paired line there.
    """)
    return


@app.cell
def _():
    dwell_models = {
        "Free": "Both",
        "Transitions": "Transitions",
        "Emissions": "Emissions",
    }
    dwell_model_order = list(dwell_models)
    dwell_model_ticks = ["Free", "Trans.", "Emiss."]
    treatment_order = ["Saline", "Drug"]
    treatment_palette = {"Saline": "tab:gray", "Drug": "tab:pink"}
    return (
        dwell_model_order,
        dwell_model_ticks,
        dwell_models,
        treatment_order,
        treatment_palette,
    )


@app.cell
def _(get_adapter, task_configs):
    adapters = {task: get_adapter(f"{task}_DRUG") for task in task_configs}
    dfs = {
        task: adapter.subject_filter(adapter.read_dataset())
        for task, adapter in adapters.items()
    }
    return adapters, dfs


@app.cell
def _(
    adapters,
    build_trial_and_weights_df,
    build_views,
    dfs,
    dwell_models,
    json,
    load_fit_arrays,
    plot_dfs,
    task_configs,
    two_adc,
    two_afc,
):
    dwell_trial_dfs = {}
    for _task, _task_config in task_configs.items():
        for _model, _comparison_label in dwell_models.items():
            _directory = _task_config["fit_root"] / _task_config["models"][_comparison_label]
            _config = json.loads((_directory / "config.json").read_text())
            _adapter = adapters[_task]
            for _key in ("state_scoring_feature", "state_scoring_rule", "state_split_feature", "state_split_rule"):
                if _key in _config:
                    setattr(_adapter, _key, _config[_key] or None)
            _subjects = sorted(plot_dfs[_task]["subject"].astype(str).unique())
            _arrays, _ = load_fit_arrays(
                out_dir=_directory,
                arrays_suffix="glmhmmt_arrays.npz",
                adapter=_adapter,
                df_all=dfs[_task],
                subjects=_subjects,
                emission_cols=_config["emission_cols"],
                transition_cols=_config["transition_cols"],
                k=2,
            )
            _views = build_views({s: _arrays[s] for s in _subjects}, _adapter, 2, _subjects)
            _trials, _ = build_trial_and_weights_df(
                dfs[_task], views=_views, adapter=_adapter, min_session_length=2,
            )
            _process = two_adc if _task == "2ADC" else two_afc
            dwell_trial_dfs[_task, _model] = _process.prepare_predictions_df(_trials)
    return (dwell_trial_dfs,)


@app.cell
def _(dwell_trial_dfs, glmhmmt_state_dwell_df, pd, pl, task_configs):
    _frames = {task: [] for task in task_configs}
    for (_task, _model), _trials in dwell_trial_dfs.items():
        _treatments = (
            _trials.select("subject", "session", "condition")
            .with_columns(
                pl.col("condition").str.to_lowercase().replace_strict(
                    {"saline": "Saline", "drug": "Drug"}, default=None,
                ).alias("treatment")
            )
            .drop("condition").drop_nulls().unique().to_pandas()
        )
        _dwell = glmhmmt_state_dwell_df(_trials).merge(
            _treatments, on=["subject", "session"], how="inner", validate="many_to_one",
        )
        _frames[_task].append(
            _dwell.groupby(["subject", "treatment", "state_label"], as_index=False, observed=True)
            .agg(mean_dwell_trials=("dwell_trials", "mean")).assign(model=_model)
        )
    treatment_dwell_dfs = {task: pd.concat(frames, ignore_index=True) for task, frames in _frames.items()}
    dwell_plot_dfs = {
        (task, state): frame.loc[frame["state_label"] == state]
        for task, frame in treatment_dwell_dfs.items()
        for state in ("Engaged", "Disengaged")
    }
    return dwell_plot_dfs, treatment_dwell_dfs


@app.cell
def _(pd, treatment_dwell_dfs):
    dwell_counts = (
        pd.concat(treatment_dwell_dfs, names=["task"])
        .groupby(["task", "model", "state_label", "treatment"])["subject"]
        .nunique().rename("n_animals").reset_index()
    )
    dwell_counts
    return


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        _panel_width, _panel_height = fig_size(2)
        fig, axd = plt.subplot_mosaic(
            [
                ["drug_ll_2ADC", "drug_ll_2ADC", "drug_ll_2AFC", "drug_ll_2AFC"],
                ["dwell_2ADC_engaged", "dwell_2ADC_disengaged", "dwell_2AFC_engaged", "dwell_2AFC_disengaged"],
            ],
            figsize=fig_size(1, 1.15),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell
def _(combinations, math, model_order, ttest_rel):
    model_pairs = list(combinations(model_order, 2))

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
    model_tick_labels,
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
    drug_ll_2ADC.set_xticks(range(len(model_order)), model_tick_labels)
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
    model_tick_labels,
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
    drug_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_pair_annotations(drug_ll_2AFC, _plot_df)
    drug_ll_2AFC.set(xlabel="", ylabel="")
    drug_ll_2AFC.set_xticks(range(len(model_order)), model_tick_labels)
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
    ### 2ADC — Engaged
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    axd,
    dwell_model_order,
    dwell_model_ticks,
    dwell_plot_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_2ADC_engaged = axd["dwell_2ADC_engaged"] if mount_figure else plt.gca()
    dwell_2ADC_engaged.clear()
    sns.boxplot(
        data=dwell_plot_dfs["2ADC", "Engaged"],
        x="model", y="mean_dwell_trials", hue="treatment",
        order=dwell_model_order, hue_order=treatment_order,
        palette=treatment_palette, ax=dwell_2ADC_engaged,
        legend=False, **BOXPLOT_STYLE,
    )
    for _index, _model in enumerate(dwell_model_order):
        _paired = (
            dwell_plot_dfs["2ADC", "Engaged"]
            .loc[dwell_plot_dfs["2ADC", "Engaged"]["model"] == _model]
            .pivot(index="subject", columns="treatment", values="mean_dwell_trials")
            .reindex(columns=treatment_order).dropna()
        )
        dwell_2ADC_engaged.plot(
            [_index - 0.2, _index + 0.2], _paired.to_numpy().T,
            color="0.75", linewidth=0.5, zorder=0,
        )
    dwell_2ADC_engaged.set(
        title="Engaged", xlabel="", ylabel="Dwell time (trials)", yscale="log",
    )
    dwell_2ADC_engaged.set_xticks(range(len(dwell_model_order)), dwell_model_ticks, fontsize=7)
    sns.despine(ax=dwell_2ADC_engaged)
    if not mount_figure:
        dwell_2ADC_engaged.figure.savefig(path_panels / "svg" / "dwell_2ADC_engaged.svg")
        dwell_2ADC_engaged.figure.savefig(path_panels / "png" / "dwell_2ADC_engaged.png", dpi=300)
    dwell_2ADC_engaged
    return (dwell_2ADC_engaged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC — Disengaged
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    axd,
    dwell_model_order,
    dwell_model_ticks,
    dwell_plot_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_2ADC_disengaged = axd["dwell_2ADC_disengaged"] if mount_figure else plt.gca()
    dwell_2ADC_disengaged.clear()
    sns.boxplot(
        data=dwell_plot_dfs["2ADC", "Disengaged"],
        x="model", y="mean_dwell_trials", hue="treatment",
        order=dwell_model_order, hue_order=treatment_order,
        palette=treatment_palette, ax=dwell_2ADC_disengaged,
        legend=False, **BOXPLOT_STYLE,
    )
    for _index, _model in enumerate(dwell_model_order):
        _paired = (
            dwell_plot_dfs["2ADC", "Disengaged"]
            .loc[dwell_plot_dfs["2ADC", "Disengaged"]["model"] == _model]
            .pivot(index="subject", columns="treatment", values="mean_dwell_trials")
            .reindex(columns=treatment_order).dropna()
        )
        dwell_2ADC_disengaged.plot(
            [_index - 0.2, _index + 0.2], _paired.to_numpy().T,
            color="0.75", linewidth=0.5, zorder=0,
        )
    dwell_2ADC_disengaged.set(
        title="Disengaged", xlabel="", ylabel="", yscale="log",
    )
    dwell_2ADC_disengaged.set_xticks(range(len(dwell_model_order)), dwell_model_ticks, fontsize=7)
    sns.despine(ax=dwell_2ADC_disengaged)
    if not mount_figure:
        dwell_2ADC_disengaged.figure.savefig(path_panels / "svg" / "dwell_2ADC_disengaged.svg")
        dwell_2ADC_disengaged.figure.savefig(path_panels / "png" / "dwell_2ADC_disengaged.png", dpi=300)
    dwell_2ADC_disengaged
    return (dwell_2ADC_disengaged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC — Engaged
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    axd,
    dwell_model_order,
    dwell_model_ticks,
    dwell_plot_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_2AFC_engaged = axd["dwell_2AFC_engaged"] if mount_figure else plt.gca()
    dwell_2AFC_engaged.clear()
    sns.boxplot(
        data=dwell_plot_dfs["2AFC", "Engaged"],
        x="model", y="mean_dwell_trials", hue="treatment",
        order=dwell_model_order, hue_order=treatment_order,
        palette=treatment_palette, ax=dwell_2AFC_engaged,
        legend=False, **BOXPLOT_STYLE,
    )
    for _index, _model in enumerate(dwell_model_order):
        _paired = (
            dwell_plot_dfs["2AFC", "Engaged"]
            .loc[dwell_plot_dfs["2AFC", "Engaged"]["model"] == _model]
            .pivot(index="subject", columns="treatment", values="mean_dwell_trials")
            .reindex(columns=treatment_order).dropna()
        )
        dwell_2AFC_engaged.plot(
            [_index - 0.2, _index + 0.2], _paired.to_numpy().T,
            color="0.75", linewidth=0.5, zorder=0,
        )
    dwell_2AFC_engaged.set(
        title="Engaged", xlabel="", ylabel="Dwell time (trials)", yscale="log",
    )
    dwell_2AFC_engaged.set_xticks(range(len(dwell_model_order)), dwell_model_ticks, fontsize=7)
    sns.despine(ax=dwell_2AFC_engaged)
    if not mount_figure:
        dwell_2AFC_engaged.figure.savefig(path_panels / "svg" / "dwell_2AFC_engaged.svg")
        dwell_2AFC_engaged.figure.savefig(path_panels / "png" / "dwell_2AFC_engaged.png", dpi=300)
    dwell_2AFC_engaged
    return (dwell_2AFC_engaged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC — Disengaged
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    axd,
    dwell_model_order,
    dwell_model_ticks,
    dwell_plot_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_2AFC_disengaged = axd["dwell_2AFC_disengaged"] if mount_figure else plt.gca()
    dwell_2AFC_disengaged.clear()
    sns.boxplot(
        data=dwell_plot_dfs["2AFC", "Disengaged"],
        x="model", y="mean_dwell_trials", hue="treatment",
        order=dwell_model_order, hue_order=treatment_order,
        palette=treatment_palette, ax=dwell_2AFC_disengaged,
        legend=True, **BOXPLOT_STYLE,
    )
    for _index, _model in enumerate(dwell_model_order):
        _paired = (
            dwell_plot_dfs["2AFC", "Disengaged"]
            .loc[dwell_plot_dfs["2AFC", "Disengaged"]["model"] == _model]
            .pivot(index="subject", columns="treatment", values="mean_dwell_trials")
            .reindex(columns=treatment_order).dropna()
        )
        dwell_2AFC_disengaged.plot(
            [_index - 0.2, _index + 0.2], _paired.to_numpy().T,
            color="0.75", linewidth=0.5, zorder=0,
        )
    dwell_2AFC_disengaged.set(
        title="Disengaged", xlabel="", ylabel="", yscale="log",
    )
    dwell_2AFC_disengaged.set_xticks(range(len(dwell_model_order)), dwell_model_ticks, fontsize=7)
    sns.despine(ax=dwell_2AFC_disengaged)
    if not mount_figure:
        dwell_2AFC_disengaged.figure.savefig(path_panels / "svg" / "dwell_2AFC_disengaged.svg")
        dwell_2AFC_disengaged.figure.savefig(path_panels / "png" / "dwell_2AFC_disengaged.png", dpi=300)
    dwell_2AFC_disengaged
    return (dwell_2AFC_disengaged,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final figure
    """)
    return


@app.cell
def _(
    drug_ll_2ADC,
    drug_ll_2AFC,
    dwell_2ADC_disengaged,
    dwell_2ADC_engaged,
    dwell_2AFC_disengaged,
    dwell_2AFC_engaged,
    fig,
    mount_figure,
    path_panels,
):
    if mount_figure:
        drug_ll_2ADC.set_title("2ADC")
        drug_ll_2AFC.set_title("2AFC")
        _dwell_axes = [dwell_2ADC_engaged, dwell_2ADC_disengaged, dwell_2AFC_engaged, dwell_2AFC_disengaged]
        _bottom = min(axis.get_ylim()[0] for axis in _dwell_axes)
        _top = max(axis.get_ylim()[1] for axis in _dwell_axes)
        for _axis in _dwell_axes:
            _axis.set_ylim(_bottom, _top)
        dwell_2AFC_disengaged.legend(frameon=False, title="", fontsize=7)
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

    Paired two-sided t-tests compare trial-weighted CV log-likelihood across subjects, using one score per animal and model. P-values are Bonferroni-corrected over the six model comparisons within each panel.
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
def _(
    fig_size,
    model_order,
    model_tick_labels,
    ols,
    panel_names,
    path_panels,
    pd,
    plot_dfs,
    plt,
    sns,
    task_palette,
):
    pooled_df = pd.concat(plot_dfs, names=["task"]).reset_index(level="task").reset_index(drop=True)
    pooled_df["animal"] = pooled_df["task"] + ":" + pooled_df["subject"]
    _paired = pooled_df.pivot(index=["task", "animal"], columns="model", values="delta_ll_vs_no_drug").dropna()
    _data = _paired.reset_index()
    # Centering task makes the intercept the mean contrast across animals.
    _data["task_centered"] = (_data["task"] == "2AFC").astype(float)
    _data["task_centered"] -= _data["task_centered"].mean()
    _pairs = [
        ("Transitions", "No drug"), ("Emissions", "No drug"), ("Both", "No drug"),
        ("Transitions", "Emissions"), ("Both", "Transitions"), ("Both", "Emissions"),
    ]
    _rows = []
    for _left, _right in _pairs:
        _data["delta"] = _data[_left] - _data[_right]
        _fit = ols("delta ~ task_centered", data=_data).fit(cov_type="HC3", use_t=True)
        _ci = _fit.conf_int().loc["Intercept"]
        _rows.append({
            "panel": panel_names["drug_ll_pooled"],
            "model_a": _left, "model_b": _right,
            "comparison": f"{_left} − {_right}",
            "test": "Task-adjusted OLS, HC3, two-sided t",
            "n_animals": len(_data),
            "n_2ADC": int((_data["task"] == "2ADC").sum()),
            "n_2AFC": int((_data["task"] == "2AFC").sum()),
            "mean_bits_per_trial": _fit.params["Intercept"],
            "ci95_low": _ci.iloc[0], "ci95_high": _ci.iloc[1],
            "t": _fit.tvalues["Intercept"], "df": _fit.df_resid,
            "p": _fit.pvalues["Intercept"],
            "p_bonferroni": min(_fit.pvalues["Intercept"] * len(_pairs), 1.0),
        })
    pooled_tests = pd.DataFrame(_rows)
    pooled_tests_latex = pooled_tests.drop(columns=["model_a", "model_b"]).to_latex(
        index=False, float_format="%.4g", escape=True,
    )
    pooled_tests.to_csv(path_panels / "pooled_model_comparisons.csv", index=False)
    (path_panels / "pooled_model_comparisons.tex").write_text(pooled_tests_latex)

    _summary = pooled_tests.loc[pooled_tests["model_b"] == "No drug"].set_index("model_a")
    _summary = _summary[["mean_bits_per_trial", "ci95_low", "ci95_high"]].reindex(model_order)
    _summary.loc["No drug"] = 0.0  # The reference has exactly zero difference from itself.
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    drug_ll_pooled = plt.gca()
    sns.lineplot(
        data=pooled_df.loc[pooled_df["animal"].isin(_data["animal"])],
        x="model", y="delta_ll_vs_no_drug", hue="task", palette=task_palette,
        units="animal", estimator=None, alpha=0.4, linewidth=0.7,
        marker="o", markersize=3, markeredgewidth=0, sort=False, ax=drug_ll_pooled,
    )
    drug_ll_pooled.errorbar(
        range(len(model_order)), _summary["mean_bits_per_trial"],
        yerr=[_summary["mean_bits_per_trial"] - _summary["ci95_low"],
              _summary["ci95_high"] - _summary["mean_bits_per_trial"]],
        color="black", marker="o", markersize=4, markeredgewidth=0,
        linewidth=1.2, capsize=3, label="Pooled mean (95% CI)",
    )
    drug_ll_pooled.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    drug_ll_pooled.set(
        xlabel="", ylabel=r"$\Delta$ held-out LL vs no drug (bits/trial)",
        title=f"Both tasks pooled (n = {len(_data)} animals)",
    )
    drug_ll_pooled.set_xticks(range(len(model_order)), model_tick_labels)
    drug_ll_pooled.legend(frameon=False, title="", fontsize=7)
    sns.despine(ax=drug_ll_pooled)
    drug_ll_pooled.figure.savefig(path_panels / "svg" / "drug_delta_ll_pooled.svg")
    drug_ll_pooled.figure.savefig(path_panels / "png" / "drug_delta_ll_pooled.png", dpi=300)
    drug_ll_pooled
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
