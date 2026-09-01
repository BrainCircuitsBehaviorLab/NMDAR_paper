# /// script
# [tool.marimo.opengraph]
# title = "Supplementary Figure 21"
# description = "MCDR GLM weights and behavioral autocorrelograms."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supplementary Figure 2.1

    MCDR stimulus and previous-choice weights, with outcome and repetition
    autocorrelograms.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.plots.emissions import (
        _fold_three_choice_raw_weights as fold_three_choice,
    )
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.plots.common import fig_size
    from src.process.common import (
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )

    return (
        Path,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        fold_three_choice,
        get_adapter,
        get_runtime_paths,
        load_fit_arrays,
        mo,
        pl,
        plt,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings and paths
    """)
    return


@app.cell
def _(Path, configure_paths, get_runtime_paths):
    project_path = Path(__file__).resolve().parents[1]
    configure_paths(config_path=project_path / "config.toml")
    paths = get_runtime_paths()

    path_panels = Path(__file__).resolve().parent / "panels21"
    path_panels.mkdir(parents=True, exist_ok=True)
    return path_panels, paths


@app.cell
def _(Path, plt, sns):
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(Path(__file__).resolve().parents[1] / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return


@app.cell
def _():
    boxplot_STYLE = dict(
        fill=False,
        boxprops={"color": "0.5"},
        whiskerprops={"color": "0.5"},
        medianprops={"linewidth": 3},
        showfliers=False,
        showcaps=False,
    )
    return (boxplot_STYLE,)


@app.cell
def _():
    mount_figure = True
    return (mount_figure,)


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                ["stimulus_weights", "previous_choice_weights"],
                ["outcome_autocorrelogram", "repetition_autocorrelogram"],
            ],
            figsize=fig_size(1, 1),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load MCDR data and GLM fit
    """)
    return


@app.cell
def _(get_adapter, pl):
    task_name = "MCDR"
    model_name = "one hot"
    adapter = get_adapter(task_name)
    behavior_df = adapter.subject_filter(adapter.read_dataset()).filter(
        pl.col("subject").str.contains("B")
    )
    subjects = behavior_df["subject"].unique().sort().to_list()
    return adapter, behavior_df, model_name, subjects, task_name


@app.cell
def _(
    adapter,
    behavior_df,
    build_trial_and_weights_df,
    build_views,
    load_fit_arrays,
    model_name,
    paths,
    subjects,
    task_name,
):
    fit_path = paths.RESULTS / "fits" / task_name / "glm" / model_name
    arrays, _ = load_fit_arrays(
        out_dir=fit_path,
        arrays_suffix="glm_arrays.npz",
        adapter=adapter,
        df_all=behavior_df,
        subjects=subjects,
        emission_cols=None,
    )
    views = build_views(arrays, adapter, 1, subjects)
    _, weight_df = build_trial_and_weights_df(
        behavior_df,
        views=views,
        adapter=adapter,
        min_session_length=1,
    )
    return views, weight_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GLM weights
    """)
    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    fold_three_choice,
    mount_figure,
    path_panels,
    pl,
    plt,
    sns,
    weight_df,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    stimulus_weights = (
        plt.gca() if not mount_figure else axd["stimulus_weights"]
    )
    stimulus_weights.clear()

    stimulus_weight_df = pl.from_pandas(fold_three_choice(weight_df)).filter(
        pl.col("feature").str.contains("stim")
    )
    stimulus_order = sorted(
        stimulus_weight_df["feature"].unique(),
        key=lambda feature: int(feature.replace("stim", "")),
    )
    sns.boxplot(
        data=stimulus_weight_df,
        x="feature",
        y="weight",
        order=stimulus_order,
        color="tab:gray",
        ax=stimulus_weights,
        **boxplot_STYLE,
    )
    stimulus_weights.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    stimulus_weights.set_title("Stimulus")
    stimulus_weights.set_ylabel("Weight")
    stimulus_weights.set_xlabel("Difficulty")
    stimulus_weights.set_xticks(range(len(stimulus_order)), [1, 2, 3, 4])
    if not mount_figure:
        stimulus_weights.figure.savefig(path_panels / "MCDR_stimulus_weights.svg")
        stimulus_weights.figure.savefig(
            path_panels / "MCDR_stimulus_weights.png", dpi=300
        )
    stimulus_weights
    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    fold_three_choice,
    mount_figure,
    path_panels,
    pl,
    plt,
    sns,
    weight_df,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    previous_choice_weights = (
        plt.gca() if not mount_figure else axd["previous_choice_weights"]
    )
    previous_choice_weights.clear()

    previous_choice_weight_df = pl.from_pandas(
        fold_three_choice(weight_df)
    ).filter(pl.col("feature").str.contains("choice_lag"))
    previous_choice_order = sorted(
        previous_choice_weight_df["feature"].unique(),
        key=lambda feature: int(feature.split("_")[-1]),
    )
    sns.boxplot(
        data=previous_choice_weight_df,
        x="feature",
        y="weight",
        order=previous_choice_order,
        color="tab:gray",
        ax=previous_choice_weights,
        **boxplot_STYLE,
    )
    previous_choice_weights.axhline(
        0, color="0.5", linestyle="--", linewidth=0.8
    )
    previous_choice_weights.set_title("Previous choices")
    previous_choice_weights.set_ylabel("Weight")
    previous_choice_weights.set_xlabel("Lag")
    previous_choice_weights.set_xticks(
        range(len(previous_choice_order)),
        [
            str(lag) if lag == 1 or lag % 5 == 0 else ""
            for lag in range(1, len(previous_choice_order) + 1)
        ],
    )
    if not mount_figure:
        previous_choice_weights.figure.savefig(
            path_panels / "MCDR_previous_choice_weights.svg"
        )
        previous_choice_weights.figure.savefig(
            path_panels / "MCDR_previous_choice_weights.png", dpi=300
        )
    previous_choice_weights
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Autocorrelograms
    """)
    return


@app.cell
def _(
    adapter,
    behavior_df,
    prepare_closed_loop_model_autocorrelograms,
    prepare_corrected_behavior_autocorrelograms,
    views,
):
    data_autocorr = prepare_corrected_behavior_autocorrelograms(
        behavior_df,
        subject_col="subject",
        session_col=adapter.behavioral_cols["session"],
        choice_col=adapter.behavioral_cols["response"],
        outcome_col=adapter.behavioral_cols["performance"],
        trial_index_col=adapter.behavioral_cols["trial"],
        max_lag=50,
        min_cross_pairs=20,
        max_cross_pairs=80,
        seed=0,
    )["autocorr"]
    glm_autocorr = prepare_closed_loop_model_autocorrelograms(
        behavior_df,
        views=views,
        adapter=adapter,
        n_simulations=1,
        max_lag=50,
        min_cross_pairs=20,
        max_cross_pairs=80,
        seed=1,
        progress_label="MCDR GLM closed-loop simulations",
    )["autocorr"]
    return data_autocorr, glm_autocorr


@app.cell
def _(
    axd,
    data_autocorr,
    fig_size,
    glm_autocorr,
    mount_figure,
    path_panels,
    plt,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    outcome_autocorrelogram = (
        plt.gca() if not mount_figure else axd["outcome_autocorrelogram"]
    )
    outcome_autocorrelogram.clear()
    data_outcome = data_autocorr[data_autocorr["signal"] == "Outcome"].sort_values(
        "lag"
    )
    glm_outcome = glm_autocorr[glm_autocorr["signal"] == "Outcome"].sort_values(
        "lag"
    )

    outcome_autocorrelogram.errorbar(
        data_outcome["lag"],
        data_outcome["autocorr"],
        yerr=data_outcome.get("autocorr_sem"),
        fmt="o",
        capsize=0,
        markersize=3,
        color="tab:blue",
        ecolor="tab:blue",
        label="Data",
        zorder=4,
    )
    outcome_autocorrelogram.plot(
        glm_outcome["lag"],
        glm_outcome["autocorr"],
        color="tab:gray",
        label="GLM",
        zorder=3,
    )
    outcome_autocorrelogram.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    outcome_autocorrelogram.set_xlabel("Trial lag")
    outcome_autocorrelogram.set_ylabel("Autocorrelation")
    outcome_autocorrelogram.set_title("Outcome")
    outcome_autocorrelogram.set_xlim(0, 20.5)
    outcome_autocorrelogram.set_xticks([1, 5, 10, 15, 20])
    outcome_autocorrelogram.set_ylim(top=0.05)
    outcome_autocorrelogram.legend(frameon=False)
    if not mount_figure:
        outcome_autocorrelogram.figure.savefig(
            path_panels / "MCDR_autocorrelogram_outcome.svg"
        )
        outcome_autocorrelogram.figure.savefig(
            path_panels / "MCDR_autocorrelogram_outcome.png", dpi=300
        )
    outcome_autocorrelogram
    return


@app.cell
def _(
    axd,
    data_autocorr,
    fig_size,
    glm_autocorr,
    mount_figure,
    path_panels,
    plt,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    repetition_autocorrelogram = (
        plt.gca() if not mount_figure else axd["repetition_autocorrelogram"]
    )
    repetition_autocorrelogram.clear()
    data_repetition = data_autocorr[
        data_autocorr["signal"] == "Repetition"
    ].sort_values("lag")
    glm_repetition = glm_autocorr[
        glm_autocorr["signal"] == "Repetition"
    ].sort_values("lag")

    repetition_autocorrelogram.errorbar(
        data_repetition["lag"],
        data_repetition["autocorr"],
        yerr=data_repetition.get("autocorr_sem"),
        fmt="o",
        capsize=0,
        markersize=3,
        color="tab:blue",
        ecolor="tab:blue",
        label="Data",
        zorder=4,
    )
    repetition_autocorrelogram.plot(
        glm_repetition["lag"],
        glm_repetition["autocorr"],
        color="tab:gray",
        label="GLM",
        zorder=3,
    )
    repetition_autocorrelogram.axhline(
        0, color="0.5", linestyle="--", linewidth=0.8
    )
    repetition_autocorrelogram.set_xlabel("Trial lag")
    repetition_autocorrelogram.set_ylabel("Autocorrelation")
    repetition_autocorrelogram.set_title("Repetition")
    repetition_autocorrelogram.set_xlim(0, 20.5)
    repetition_autocorrelogram.set_xticks([1, 5, 10, 15, 20])
    repetition_autocorrelogram.set_ylim(top=0.15)
    repetition_autocorrelogram.legend(frameon=False)
    if not mount_figure:
        repetition_autocorrelogram.figure.savefig(
            path_panels / "MCDR_autocorrelogram_repetition.svg"
        )
        repetition_autocorrelogram.figure.savefig(
            path_panels / "MCDR_autocorrelogram_repetition.png", dpi=300
        )
    repetition_autocorrelogram
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mounted figure
    """)
    return


@app.cell
def _(fig, mount_figure, path_panels):
    if mount_figure:
        fig.savefig(path_panels / "supplementary_figure21.svg")
        fig.savefig(path_panels / "supplementary_figure21.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure21.pdf")
    mounted_figure = fig if mount_figure else None
    mounted_figure
    return


if __name__ == "__main__":
    app.run()
