import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import math
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_rel

    _PROJECT_ROOT = next(
        (
            path
            for base in (Path.cwd(), Path(__file__).resolve())
            for path in (base, *base.parents)
            if (path / "config.toml").exists() and (path / "src").exists()
        ),
        Path.cwd(),
    )
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from glmhmmt.cli.fit_glmhmmt import main as fit_main
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from src.plots.common import fig_size

    configure_paths(config_path=_PROJECT_ROOT / "config.toml")
    paths = get_runtime_paths()
    project_root = _PROJECT_ROOT
    return (
        fig_size,
        fit_main,
        get_adapter,
        math,
        mo,
        paths,
        pl,
        plt,
        project_root,
        sns,
        ttest_rel,
    )


@app.cell
def _(plt, project_root, sns):
    plt.style.use(project_root / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    sns.set_theme(style="ticks", context="paper")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GLM-HMM-T state-count comparison

    Compare the `param half-pure` GLM-HMM-T with **2–7 latent states** using
    five-fold session-level cross-validation in both tasks.

    Fits are isolated from the rest of the project in
    `results/fits/model_comparison/2AFC` and
    `results/fits/model_comparison/2ADC`.

    The task-specific `glm/one hot` fit is the comparison baseline and is shown
    at K=1. Delta plots subtract each subject's GLM score.

    Delta-plot brackets show uncorrected, two-sided paired t tests comparing
    K=2 against the K=1 GLM and K=3–7 against K=2 (`ns`, `*`, `**`, `***`).
    """)
    return


@app.cell
def _(mo):
    TASK_CONFIGS = {
        "2AFC": {
            "task": "2AFC",
            "emission_cols": ["bias", "stim_param", "choice_lag_param"],
            "transition_cols": ["cumulative_reward"],
            "frozen_emissions": {"1": {"stim_param": 0.0}},
        },
        "2ADC": {
            "task": "2AFC_delay",
            "emission_cols": ["bias", "choice_lag_param", "stim_x_delay_param"],
            "transition_cols": ["cumulative_reward"],
            "frozen_emissions": {"1": {"stim_x_delay_param": 0.0}},
        },
    }
    ui_task = mo.ui.dropdown(
        options=list(TASK_CONFIGS),
        value="2AFC",
        label="Task",
    )
    return TASK_CONFIGS, ui_task


@app.cell
def _(TASK_CONFIGS, mo, ui_task):
    task_label = ui_task.value
    task_config = TASK_CONFIGS[task_label]
    ui_show_ashwood = mo.ui.checkbox(
        value=False,
        label="Show Ashwood K=3",
    )
    mo.hstack([ui_task, ui_show_ashwood])
    return task_config, task_label, ui_show_ashwood


@app.cell
def _(paths, project_root, task_config, task_label):
    K_VALUES = list(range(2, 8))
    TAU = 50
    model_label = "GLM-HMM-T param half-pure"
    fit_root = paths.RESULTS / "fits" / "model_comparison" / task_label
    glm_dir = paths.RESULTS / "fits" / task_config["task"] / "glm" / "one hot"
    ashwood_dir = paths.RESULTS / "fits" / task_config["task"] / "glmhmm" / "ashwood"
    panel_root = (
        project_root
        / "figures"
        / "panels_glmhmmt_state_comparison"
        / task_label
    )
    fit_dir = fit_root / "glmhmmt_param_half_pure"
    panel_dir = panel_root / "glmhmmt_param_half_pure"
    panel_dir.mkdir(parents=True, exist_ok=True)
    return (
        K_VALUES,
        TAU,
        ashwood_dir,
        fit_dir,
        glm_dir,
        model_label,
        panel_dir,
    )


@app.cell
def _(get_adapter, pl, task_config):
    adapter = get_adapter(task_config["task"])
    df_all = adapter.subject_filter(adapter.read_dataset())

    # Match the existing comparison notebook: CV needs at least five sessions.
    session_col = adapter.behavioral_cols["session"]
    df_all = df_all.filter(
        pl.col(session_col).n_unique().over("subject") >= 5
    )
    all_subjects = df_all["subject"].unique().sort().to_list()
    baseline_class_idx = int(adapter.baseline_class_idx)
    return all_subjects, baseline_class_idx


@app.cell
def _(K_VALUES, all_subjects, fit_dir, glm_dir, mo, model_label):
    ui_subjects = mo.ui.multiselect(
        options=all_subjects,
        value=all_subjects,
        label="Subjects",
    )
    ui_num_iters = mo.ui.number(
        start=1,
        stop=500,
        step=1,
        value=50,
        label="EM iterations",
    )
    ui_n_restarts = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="Restarts per fold",
    )
    ui_run = mo.ui.run_button(label=f"Fit {model_label}, K=2–7")
    mo.vstack(
        [
            mo.md(
                f"**Model:** `{model_label}`  ·  "
                f"**States:** `{K_VALUES[0]}–{K_VALUES[-1]}`  ·  "
                f"**Output:** `{fit_dir}`  ·  "
                f"**K=1 baseline:** `{glm_dir}`"
            ),
            ui_subjects,
            mo.hstack([ui_num_iters, ui_n_restarts, ui_run]),
        ]
    )
    return ui_n_restarts, ui_num_iters, ui_run, ui_subjects


@app.cell
def _(
    K_VALUES,
    TAU,
    baseline_class_idx,
    fit_dir,
    fit_main,
    mo,
    task_config,
    ui_n_restarts,
    ui_num_iters,
    ui_run,
    ui_subjects,
):
    if ui_run.value:
        _folds = 5
        _total = max(1, len(ui_subjects.value) * len(K_VALUES) * _folds)
        with mo.status.progress_bar(
            total=_total,
            title="Fitting GLM-HMM-T state-count comparison",
            subtitle=(
                f"{len(ui_subjects.value)} subjects × "
                f"{len(K_VALUES)} state counts × {_folds} folds"
            ),
            completion_title="GLM-HMM-T state-count fits complete",
        ) as _bar:
            def _on_progress(info: dict) -> None:
                """Advance the notebook progress bar after each CV fold."""
                if info.get("event") == "cv_repeat_complete":
                    _bar.update(
                        increment=1,
                        title=f"Fitting K={info.get('K')}",
                        subtitle=(
                            f"Subject {info.get('subject')} · fold "
                            f"{info.get('cv_repeat_index')}/"
                            f"{info.get('cv_repeat_total')}"
                        ),
                    )

            fit_main(
                subjects=list(ui_subjects.value),
                K_list=K_VALUES,
                num_iters=int(ui_num_iters.value),
                n_restarts=int(ui_n_restarts.value),
                out_dir=fit_dir,
                tau=TAU,
                emission_cols=list(task_config["emission_cols"]),
                transition_cols=list(task_config["transition_cols"]),
                frozen_emissions=task_config["frozen_emissions"],
                task=task_config["task"],
                cv_mode="balanced_session_holdout",
                cv_repeats=_folds,
                verbose=False,
                baseline_class_idx=baseline_class_idx,
                progress_callback=_on_progress,
            )
        _message = mo.md(f"Fits saved in `{fit_dir}`.")
    else:
        _message = mo.md("Press the fit button to create or refresh the selected fits.")
    _message
    fit_refresh = ui_run.value
    return (fit_refresh,)


@app.cell
def _(
    K_VALUES,
    ashwood_dir,
    fit_dir,
    fit_refresh,
    glm_dir,
    math,
    mo,
    pl,
    ui_show_ashwood,
    ui_subjects,
):
    del fit_refresh
    metric_schema = {
        "subject": pl.Utf8,
        "K": pl.Int64,
        "model_kind": pl.Utf8,
        "test_ll_per_trial_mean": pl.Float64,
        "cv_ll_bits_per_trial": pl.Float64,
        "bic": pl.Float64,
    }

    glm_frames = [
        pl.read_parquet(path)
        for path in sorted(glm_dir.glob("*_glm_metrics.parquet"))
    ]
    if glm_frames:
        glm_raw = pl.concat(glm_frames, how="diagonal")
        if "test_ll_per_trial_mean" not in glm_raw.columns:
            glm_raw = glm_raw.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("test_ll_per_trial_mean")
            )
        glm_metrics = (
            glm_raw
            .filter(pl.col("subject").is_in(ui_subjects.value))
            .with_columns(
                pl.lit(1, dtype=pl.Int64).alias("K"),
                pl.lit("glm").alias("model_kind"),
                (pl.col("test_ll_per_trial_mean") / math.log(2)).alias(
                    "cv_ll_bits_per_trial"
                ),
            )
            .select(list(metric_schema))
        )
    else:
        glm_metrics = pl.DataFrame(schema=metric_schema)

    glmhmmt_frames = [
        pl.read_parquet(path)
        for path in sorted(fit_dir.glob("*_glmhmmt_metrics.parquet"))
    ]
    if glmhmmt_frames:
        glmhmmt_metrics = (
            pl.concat(glmhmmt_frames, how="diagonal")
            .filter(
                pl.col("subject").is_in(ui_subjects.value)
                & pl.col("K").is_in(K_VALUES)
            )
            .with_columns(
                pl.lit("glmhmmt").alias("model_kind"),
                (pl.col("test_ll_per_trial_mean") / math.log(2)).alias(
                    "cv_ll_bits_per_trial"
                ),
            )
            .select(list(metric_schema))
        )
    else:
        glmhmmt_metrics = pl.DataFrame(schema=metric_schema)

    ashwood_frames = (
        [
            pl.read_parquet(path)
            for path in sorted(ashwood_dir.glob("*_K3_glmhmm_metrics.parquet"))
        ]
        if ui_show_ashwood.value
        else []
    )
    if ashwood_frames:
        ashwood_raw = pl.concat(ashwood_frames, how="diagonal")
        if "test_ll_per_trial_mean" not in ashwood_raw.columns:
            ashwood_raw = ashwood_raw.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("test_ll_per_trial_mean")
            )
        ashwood_metrics = (
            ashwood_raw
            .filter(pl.col("subject").is_in(ui_subjects.value))
            .with_columns(
                pl.lit(3, dtype=pl.Int64).alias("K"),
                pl.lit("ashwood").alias("model_kind"),
                (pl.col("test_ll_per_trial_mean") / math.log(2)).alias(
                    "cv_ll_bits_per_trial"
                ),
            )
            .select(list(metric_schema))
        )
    else:
        ashwood_metrics = pl.DataFrame(schema=metric_schema)

    metrics = pl.concat([glm_metrics, glmhmmt_metrics]).sort(["subject", "K"])
    fit_counts = (
        metrics
        .group_by(["model_kind", "K"])
        .agg(pl.col("subject").n_unique().alias("n_subjects"))
        .sort("K")
    )
    glm_has_cv = (
        glm_metrics
        .select(pl.col("cv_ll_bits_per_trial").is_not_null().any())
        .item()
        if not glm_metrics.is_empty()
        else False
    )
    status_messages = [fit_counts]
    if glm_metrics.is_empty():
        status_messages.append(mo.md(f"No GLM baseline fits found in `{glm_dir}`."))
    elif not glm_has_cv:
        status_messages.append(
            mo.md(
                "⚠️ The GLM baseline has no held-out LL. K=1 is included in "
                "the BIC plots, but excluded from CV-LL comparisons."
            )
        )
    if glmhmmt_metrics.is_empty():
        status_messages.append(mo.md(f"No GLM-HMM-T comparison fits found in `{fit_dir}`."))
    if ui_show_ashwood.value and ashwood_metrics.is_empty():
        status_messages.append(mo.md(f"No Ashwood K=3 fits found in `{ashwood_dir}`."))
    mo.vstack(status_messages)
    return ashwood_metrics, glm_metrics, glmhmmt_metrics, metrics


@app.cell
def _(metrics, pl):
    metric_summary = (
        metrics
        .group_by("K")
        .agg(
            pl.col("subject").n_unique().alias("n_subjects"),
            pl.mean("cv_ll_bits_per_trial").alias("cv_ll_mean"),
            pl.std("cv_ll_bits_per_trial").alias("cv_ll_sd"),
            pl.mean("bic").alias("bic_mean"),
            pl.std("bic").alias("bic_sd"),
        )
        .with_columns(
            (pl.col("cv_ll_sd") / pl.col("n_subjects").sqrt()).alias("cv_ll_sem"),
            (pl.col("bic_sd") / pl.col("n_subjects").sqrt()).alias("bic_sem"),
        )
        .sort("K")
    )
    metric_summary
    return


@app.cell
def _(ashwood_metrics, glm_metrics, glmhmmt_metrics, pl):
    # Pair each GLM-HMM-T fit with the same animal's task-specific one-hot GLM.
    glm_baseline = glm_metrics.select(
        "subject",
        pl.col("cv_ll_bits_per_trial").alias("glm_cv_ll"),
        pl.col("bic").alias("glm_bic"),
    )
    glmhmmt_deltas = (
        glmhmmt_metrics
        .join(glm_baseline, on="subject", how="inner")
        .with_columns(
            (pl.col("cv_ll_bits_per_trial") - pl.col("glm_cv_ll")).alias(
                "delta_cv_ll_vs_glm"
            ),
            (pl.col("bic") - pl.col("glm_bic")).alias("delta_bic_vs_glm"),
        )
        .select(["subject", "K", "delta_cv_ll_vs_glm", "delta_bic_vs_glm"])
    )
    glm_zero = glm_baseline.select(
        "subject",
        pl.lit(1, dtype=pl.Int64).alias("K"),
        pl.when(pl.col("glm_cv_ll").is_not_null())
        .then(0.0)
        .otherwise(None)
        .alias("delta_cv_ll_vs_glm"),
        pl.lit(0.0).alias("delta_bic_vs_glm"),
    )
    delta_metrics = pl.concat([glm_zero, glmhmmt_deltas]).sort(["subject", "K"])
    ashwood_delta_metrics = (
        ashwood_metrics
        .join(glm_baseline, on="subject", how="inner")
        .with_columns(
            (pl.col("cv_ll_bits_per_trial") - pl.col("glm_cv_ll")).alias(
                "delta_cv_ll_vs_glm"
            ),
            (pl.col("bic") - pl.col("glm_bic")).alias("delta_bic_vs_glm"),
        )
        .select(["subject", "K", "delta_cv_ll_vs_glm", "delta_bic_vs_glm"])
    )
    return ashwood_delta_metrics, delta_metrics


@app.function
def clean_lineplot_edges(ax):
    """Remove marker and error-bar outlines to match the project style."""
    for line in ax.lines:
        line.set_markeredgewidth(0)
        line.set_markeredgecolor("none")
    for collection in ax.collections:
        collection.set_edgecolor("none")
        collection.set_linewidth(0)


@app.function
def plot_ashwood_point(ax, metrics, metric):
    """Plot the Ashwood K=3 group mean and SEM as an orange point."""
    values = metrics[metric].drop_nulls()
    if values.is_empty():
        return
    sem = values.std() / len(values) ** 0.5
    ax.errorbar(
        3,
        values.mean(),
        yerr=sem,
        fmt="o",
        markersize=6,
        markerfacecolor="tab:orange",
        markeredgecolor="black",
        markeredgewidth=0.8,
        ecolor="black",
        elinewidth=1,
        capsize=0,
        linestyle="none",
        zorder=6,
    )


@app.cell
def _(math, ttest_rel):
    def add_state_comparison_annotations(
        ax,
        df,
        *,
        x: str,
        y: str,
        subject: str = "subject",
    ) -> None:
        """Draw paired-test brackets for K=1 vs 2 and K=2 vs each larger K."""
        finite_values = df[y].dropna()
        if finite_values.empty:
            return

        value_range = float(finite_values.max() - finite_values.min())
        padding = max(value_range * 0.08, 1e-6)
        state_counts = sorted(df[x].unique())
        pairs = (
            ([(1, 2)] if 1 in state_counts and 2 in state_counts else [])
            + [(2, state_count) for state_count in state_counts if state_count > 2]
        )
        tested_pairs = []
        for left, right in pairs:
            paired = (
                df.loc[df[x].isin([left, right]), [subject, x, y]]
                .pivot_table(index=subject, columns=x, values=y, aggfunc="first")
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[right], paired[left]).pvalue)
            if not math.isfinite(pvalue):
                continue
            label = (
                "***" if pvalue < 0.001 else
                "**" if pvalue < 0.01 else
                "*" if pvalue < 0.05 else
                "ns"
            )
            tested_pairs.append((left, right, label))

        base_y = float(finite_values.max()) + padding
        ax.set_ylim(top=base_y + padding * (len(tested_pairs) + 1))
        for pair_index, (left, right, label) in enumerate(tested_pairs):
            line_y = base_y + pair_index * padding
            cap = padding * 0.25
            ax.plot(
                [left, left, right, right],
                [line_y, line_y + cap, line_y + cap, line_y],
                color="0.35",
                linewidth=0.7,
            )
            ax.text(
                (left + right) / 2,
                line_y + cap * 1.2,
                label,
                ha="center",
                va="bottom",
            )

    return (add_state_comparison_annotations,)


@app.cell
def _(
    ashwood_metrics,
    fig_size,
    metrics,
    mo,
    model_label,
    panel_dir,
    plt,
    sns,
    task_label,
):
    _cv_metrics = metrics.drop_nulls(["cv_ll_bits_per_trial"])
    mo.stop(
        _cv_metrics.is_empty(),
        mo.md("No cross-validated LL values are available to plot."),
    )

    _plot_df = _cv_metrics.to_pandas()
    _fig, _ax = plt.subplots(figsize=fig_size(2,1), constrained_layout=True)
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="cv_ll_bits_per_trial",
        units="subject",
        estimator=None,
        color="0.82",
        linewidth=0.8,
        sort=True,
        ax=_ax,
    )
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="cv_ll_bits_per_trial",
        marker="o",
        markersize=4,
        markeredgewidth=0,
        markeredgecolor="none",
        errorbar=("se", 1),
        err_kws={"edgecolor": "none", "linewidth": 0},
        color="black",
        sort=True,
        ax=_ax,
    )
    _ax.set(
        xlabel="Number of states K",
        ylabel="CV test LL (bits/trial)",
        title=f"{task_label} {model_label}",
        xticks=sorted(_plot_df["K"].unique()),
    )
    clean_lineplot_edges(_ax)
    plot_ashwood_point(_ax, ashwood_metrics, "cv_ll_bits_per_trial")
    sns.despine(ax=_ax)
    _fig.savefig(panel_dir / "cv_ll_by_state.svg")
    _fig.savefig(panel_dir / "cv_ll_by_state.png", dpi=300)
    _fig
    return


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_delta_metrics,
    delta_metrics,
    fig_size,
    mo,
    model_label,
    panel_dir,
    plt,
    sns,
    task_label,
):
    _delta_cv_metrics = delta_metrics.drop_nulls(["delta_cv_ll_vs_glm"])
    mo.stop(
        _delta_cv_metrics.is_empty(),
        mo.md("No paired GLM baseline available for the ΔCV-LL plot."),
    )
    _plot_df = _delta_cv_metrics.to_pandas()
    _fig, _ax = plt.subplots(figsize=fig_size(2,1), constrained_layout=True)
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="delta_cv_ll_vs_glm",
        units="subject",
        estimator=None,
        color="0.7",
        linewidth=0.8,
        marker = "o",
        sort=True,
        ax=_ax,
    )
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="delta_cv_ll_vs_glm",
        marker="o",
        markeredgewidth=0,
        markeredgecolor="none",
        errorbar=("se", 1),
        err_kws={"edgecolor": "none", "linewidth": 0},
        color="black",
        sort=True,
        ax=_ax,
    )
    _ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--")
    _ax.set(
        xlabel="Number of states K",
        ylabel="ΔCV test LL vs GLM (bits/trial)",
        title=f"{task_label} {model_label}",
        xticks=sorted(_plot_df["K"].unique()),
    )
    add_state_comparison_annotations(
        _ax,
        _plot_df,
        x="K",
        y="delta_cv_ll_vs_glm",
    )
    clean_lineplot_edges(_ax)
    plot_ashwood_point(_ax, ashwood_delta_metrics, "delta_cv_ll_vs_glm")
    sns.despine(ax=_ax)
    _fig.savefig(panel_dir / "delta_cv_ll_vs_glm.svg")
    _fig.savefig(panel_dir / "delta_cv_ll_vs_glm.png", dpi=300)
    _fig
    return


@app.cell
def _(
    ashwood_metrics,
    fig_size,
    metrics,
    model_label,
    panel_dir,
    plt,
    sns,
    task_label,
):
    _plot_df = metrics.to_pandas()
    _fig, _ax = plt.subplots(figsize=fig_size(2,1), constrained_layout=True)
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="bic",
        units="subject",
        estimator=None,
        color="0.7",
        linewidth=0.8,
        marker = "o",
        sort=True,
        ax=_ax,
    )
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="bic",
        marker="o",
        markeredgewidth=0,
        markeredgecolor="none",
        errorbar=("se", 1),
        err_kws={"edgecolor": "none", "linewidth": 0},
        color="black",
        sort=True,
        ax=_ax,
    )
    _ax.set(
        xlabel="Number of states K",
        ylabel="BIC (lower is better)",
        title=f"{task_label} {model_label}",
        xticks=sorted(_plot_df["K"].unique()),
    )
    clean_lineplot_edges(_ax)
    plot_ashwood_point(_ax, ashwood_metrics, "bic")
    sns.despine(ax=_ax)
    _fig.savefig(panel_dir / "bic_by_state.svg")
    _fig.savefig(panel_dir / "bic_by_state.png", dpi=300)
    _fig
    return


@app.cell
def _(
    add_state_comparison_annotations,
    ashwood_delta_metrics,
    delta_metrics,
    fig_size,
    model_label,
    panel_dir,
    plt,
    sns,
    task_label,
):
    _plot_df = delta_metrics.to_pandas()
    _fig, _ax = plt.subplots(figsize=fig_size(2,1), constrained_layout=True)
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="delta_bic_vs_glm",
        units="subject",
        estimator=None,
        color="0.7",
        linewidth=0.8,
        marker = "o",
        sort=True,
        ax=_ax,
    )
    sns.lineplot(
        data=_plot_df,
        x="K",
        y="delta_bic_vs_glm",
        marker="o",
        markeredgewidth=0,
        markeredgecolor="none",
        errorbar=("se", 1),
        err_kws={"edgecolor": "none", "linewidth": 0},
        color="black",
        sort=True,
        ax=_ax,
    )
    _ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--")
    _ax.set(
        xlabel="Number of states K",
        ylabel="ΔBIC vs GLM (lower is better)",
        title=f"{task_label} {model_label}",
        xticks=sorted(_plot_df["K"].unique()),
    )
    add_state_comparison_annotations(
        _ax,
        _plot_df,
        x="K",
        y="delta_bic_vs_glm",
    )
    clean_lineplot_edges(_ax)
    plot_ashwood_point(_ax, ashwood_delta_metrics, "delta_bic_vs_glm")
    sns.despine(ax=_ax)
    _fig.savefig(panel_dir / "delta_bic_vs_glm.svg")
    _fig.savefig(panel_dir / "delta_bic_vs_glm.png", dpi=300)
    _fig
    return


if __name__ == "__main__":
    app.run()
