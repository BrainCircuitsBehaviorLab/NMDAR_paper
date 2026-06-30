# /// script
# [tool.marimo.opengraph]
# title = "Figure 2 - Nuo auditory"
# description = "Figure 2 GLM model predictions for the Nuo auditory task."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    # Imports
    import os
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import nuo_auditory as process_nuo_auditory
    from src.process.common import (
        add_choice_lag_summary_regressor,
        build_transition_chunk_plot_data,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )
    from src.plots.common import (
        animal_chunk_histogram,
        boxplot_STYLE,
        build_session_repetition_data,
        build_session_trial_outcomes_data,
        fig_size,
        plot_session_response_raster,
    )

    return (
        Path,
        add_choice_lag_summary_regressor,
        animal_chunk_histogram,
        boxplot_STYLE,
        build_session_repetition_data,
        build_session_trial_outcomes_data,
        build_transition_chunk_plot_data,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        load_fit_arrays,
        mo,
        np,
        os,
        pd,
        pl,
        plot_session_response_raster,
        plt,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
        process_nuo_auditory,
        re,
        sns,
    )


@app.cell
def _(process_nuo_auditory):
    def prepare_predictions_df(df):
        return process_nuo_auditory.prepare_predictions_df(df)

    return (prepare_predictions_df,)


@app.cell
def _(re):
    def feature_sort_key(feature: str):
        match = re.search(r"(\d+)$", str(feature))
        return (0, int(match.group(1))) if match else (1, str(feature))

    def nonempty_axis_message(ax, message: str):
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    return feature_sort_key, nonempty_axis_message


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings
    """)
    return


@app.cell
def _():
    mount_figure = False
    format = "png"
    return format, mount_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paths
    """)
    return


@app.cell
def _(Path, configure_paths, format, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]

    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    project_path = Path(__file__).resolve().parents[1]
    path_panels = project_path / "figures" / "panels2_nuo_auditory" / format
    os.makedirs(path_panels, exist_ok=True)

    print(project_path)
    print(path_panels)
    return path_panels, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Style
    """)
    return


@app.cell
def _(Path, plt, sns):
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(Path(__file__).resolve().parents[1] / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data and fits
    """)
    return


@app.cell
def _(get_adapter):
    TASK_NAME = "nuo_auditory"
    TASK_LABEL = "Nuo auditory"
    task_names = (TASK_NAME,)
    task_labels = {TASK_NAME: TASK_LABEL}
    task_order = (TASK_LABEL,)
    model_name = "one hot"

    adapters = {TASK_NAME: get_adapter(TASK_NAME)}
    plots_by_task = {TASK_NAME: adapters[TASK_NAME].get_plots()}
    dfs = {TASK_NAME: adapters[TASK_NAME].subject_filter(adapters[TASK_NAME].read_dataset())}
    subjects_by_task = {TASK_NAME: list(dfs[TASK_NAME]["subject"].unique().sort())}
    return (
        TASK_LABEL,
        TASK_NAME,
        adapters,
        dfs,
        model_name,
        plots_by_task,
        subjects_by_task,
        task_labels,
        task_names,
        task_order,
    )


@app.cell
def _(dfs):
    (dfs["nuo_auditory"].group_by("subject").len().rename({"len": "n_rows"}).sort("subject").mean())
    return


@app.cell
def _(mo, subjects_by_task):
    subjects_nuo = mo.ui.dropdown(options = subjects_by_task["nuo_auditory"])
    return (subjects_nuo,)


@app.cell
def _(subjects_nuo):
    subjects_nuo
    return


@app.cell
def _(
    adapters,
    add_choice_lag_summary_regressor,
    build_trial_and_weights_df,
    build_views,
    dfs,
    load_fit_arrays,
    model_name,
    paths,
    prepare_predictions_df,
    subjects_by_task,
    task_names,
):
    trial_dfs, weight_dfs, views, plot_dfs = dict(), dict(), dict(), dict()
    for _task in task_names:
        _adapter = adapters[_task]
        _df_all = dfs[_task]
        _subjects = subjects_by_task[_task]
        _out = paths.RESULTS / "fits" / _task / "glm" / model_name
        _arrays_store, _ = load_fit_arrays(
            out_dir=_out,
            arrays_suffix="glm_arrays.npz",
            adapter=_adapter,
            df_all=_df_all,
            subjects=_subjects,
            emission_cols=None,
        )
        views[_task] = build_views(_arrays_store, _adapter, 1, _subjects)
        trial_dfs[_task], weight_dfs[_task] = build_trial_and_weights_df(
            _df_all,
            views=views[_task],
            adapter=_adapter,
            min_session_length=1,
        )
        plot_dfs[_task] = prepare_predictions_df(trial_dfs[_task])

        _choice_lag_cols = []
        for _view in views[_task].values():
            for _feature in list(getattr(_view, "feat_names", []) or []):
                _feature = str(_feature)
                if _feature.startswith("choice_lag_") and _feature not in _choice_lag_cols:
                    _choice_lag_cols.append(_feature)
        plot_dfs[_task] = add_choice_lag_summary_regressor(
            plot_dfs[_task],
            choice_lag_cols=_choice_lag_cols,
        )

    # plot_dfs["nuo_auditory"] = plot_dfs["nuo_auditory"].filter(pl.col("subject") == subjects_nuo.value)
    return plot_dfs, views, weight_dfs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots
    """)
    return


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                ["single_session", "transition_chunks"],
                ["stim_weights", "choice_weights"],
                ["autocorr_outcome", "autocorr_repetition"],
            ],
            figsize=fig_size(1, 0.65),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Autocorrelograms
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(
    adapters,
    dfs,
    prepare_closed_loop_model_autocorrelograms,
    prepare_corrected_behavior_autocorrelograms,
    task_names,
    views,
):
    _max_lag = 50
    _min_cross_pairs = 20
    _max_cross_pairs = 80
    _n_simulations = 1

    autocorrelograms_by_task = {}
    for _task_idx, _task in enumerate(task_names):
        _adapter = adapters[_task]
        _df_model = dfs[_task]
        _trial_col = _adapter.behavioral_cols["trial"]
        _session_col = _adapter.behavioral_cols["session"]

        _data_autocorr = prepare_corrected_behavior_autocorrelograms(
            _df_model,
            subject_col="subject",
            session_col=_session_col,
            choice_col=_adapter.behavioral_cols["response"],
            outcome_col=_adapter.behavioral_cols["performance"],
            trial_index_col=_trial_col,
            max_lag=_max_lag,
            min_cross_pairs=_min_cross_pairs,
            max_cross_pairs=_max_cross_pairs,
            seed=0,
        )
        _glm_autocorr = prepare_closed_loop_model_autocorrelograms(
            _df_model,
            views=views[_task],
            adapter=_adapter,
            n_simulations=_n_simulations,
            max_lag=_max_lag,
            min_cross_pairs=_min_cross_pairs,
            max_cross_pairs=_max_cross_pairs,
            seed=1 + (100 * _task_idx),
            progress_label=f"{_task} GLM closed-loop simulations",
        )
        autocorrelograms_by_task[_task] = {
            "data": _data_autocorr,
            "glm": _glm_autocorr,
        }
    return (autocorrelograms_by_task,)


@app.cell
def _(
    TASK_NAME,
    autocorrelograms_by_task,
    axd,
    fig_size,
    format,
    mo,
    mount_figure,
    path_panels,
    plt,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    ax_autocorrelograms_outcome = (
        plt.gca() if not mount_figure else axd["autocorr_outcome"]
    )
    ax_autocorrelograms_outcome.clear()

    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    ax_autocorrelograms_repetition = (
        plt.gca() if not mount_figure else axd["autocorr_repetition"]
    )
    ax_autocorrelograms_repetition.clear()

    _data_ac = autocorrelograms_by_task[TASK_NAME]["data"]["autocorr"]
    _glm_ac = autocorrelograms_by_task[TASK_NAME]["glm"]["autocorr"]
    _colors = {
        "data": "tab:blue",
        "glm": "tab:gray",
    }
    for _signal, _ax in (
        ("Outcome", ax_autocorrelograms_outcome),
        ("Repetition", ax_autocorrelograms_repetition),
    ):
        _data_sub = _data_ac[_data_ac["signal"] == _signal].sort_values("lag")
        _ax.errorbar(
            _data_sub["lag"],
            _data_sub["autocorr"],
            yerr=_data_sub.get("autocorr_sem"),
            fmt="o",
            capsize=0,
            ms=3,
            color=_colors["data"],
            ecolor=_colors["data"],
            label="Data",
            zorder=4,
        )
        _sub = _glm_ac[_glm_ac["signal"] == _signal].sort_values("lag")
        if not _sub.empty:
            _ax.plot(_sub["lag"], _sub["autocorr"], color=_colors["glm"], label="GLM", zorder=3)

        _ax.axhline(0.0, color="0.5", ls="--")
        _ax.set_title("Outcomes" if _signal == "Outcome" else "Repetitions")
        _ax.set_xlabel("Lag")
        _ax.set_ylabel("Autocorrelation")
        _ax.legend(frameon=False)
        if not mount_figure:
            _ax.figure.savefig(
                (path_panels / f"{TASK_NAME}_autocorrelogram_{_signal.lower()}").with_suffix(
                    f".{format}"
                )
            )

    mo.hstack([ax_autocorrelograms_outcome, ax_autocorrelograms_repetition], justify="start", gap=1)
    return


@app.cell
def _(dfs):
    dfs["nuo_auditory"]["subject"].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GLM weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Previous choices
    """)
    return


@app.cell
def _(
    TASK_NAME,
    axd,
    boxplot_STYLE,
    feature_sort_key,
    fig_size,
    format,
    mount_figure,
    nonempty_axis_message,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    prev_choices = plt.gca() if not mount_figure else axd["choice_weights"]
    prev_choices.clear()

    _plot_df = (
        weight_dfs[TASK_NAME]
        .filter(pl.col("feature").str.starts_with("choice_lag_"))
        .to_pandas()
    )
    if _plot_df.empty:
        nonempty_axis_message(prev_choices, "No choice-lag weights")
    else:
        _order = sorted(_plot_df["feature"].unique(), key=feature_sort_key)
        sns.boxplot(
            data=_plot_df,
            x="feature",
            y="weight",
            order=_order,
            color="tab:gray",
            ax=prev_choices,
            **boxplot_STYLE,
        )
        prev_choices.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        prev_choices.set_ylabel("Weight")
        prev_choices.set_xlabel("Lag")
        prev_choices.set_xticklabels(
            [
                str(i) if i == 1 or i % 5 == 0 else ""
                for i in range(1, len(_order) + 1)
            ]
        )
    if not mount_figure:
        prev_choices.figure.savefig((path_panels / f"{TASK_NAME}_prev_choices").with_suffix(f".{format}"))
    prev_choices
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stimulus
    """)
    return


@app.cell
def _(
    TASK_NAME,
    axd,
    boxplot_STYLE,
    feature_sort_key,
    fig_size,
    format,
    mount_figure,
    nonempty_axis_message,
    path_panels,
    pl,
    plt,
    process_nuo_auditory,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    stim_weights = plt.gca() if not mount_figure else axd["stim_weights"]
    stim_weights.clear()

    _features = [str(_feature) for _feature in weight_dfs[TASK_NAME]["feature"].unique()]
    _stim_features = [_feature for _feature in _features if _feature.startswith("stim_bin_")]
    if not _stim_features:
        _stim_features = [_feature for _feature in _features if "stim" in _feature]

    _plot_df = (
        weight_dfs[TASK_NAME]
        .filter(pl.col("feature").is_in(_stim_features))
        .to_pandas()
    )
    if _plot_df.empty:
        nonempty_axis_message(stim_weights, "No stimulus weights")
    else:
        _order = sorted(_stim_features, key=feature_sort_key)
        sns.boxplot(
            data=_plot_df,
            x="feature",
            y="weight",
            order=_order,
            color="tab:gray",
            ax=stim_weights,
            **boxplot_STYLE,
        )
        stim_weights.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        stim_weights.set_ylabel("Weight")
        stim_weights.set_xlabel("Evidence bin" if all(f.startswith("stim_bin_") for f in _order) else "")
        if all(f.startswith("stim_bin_") for f in _order):
            _centers = process_nuo_auditory._stim_bin_centers()
            stim_weights.set_xticklabels([f"{_center:g}" for _center in _centers[: len(_order)]])
        else:
            stim_weights.set_xticklabels(_order, rotation=45, ha="right")
    if not mount_figure:
        stim_weights.figure.savefig((path_panels / f"{TASK_NAME}_stim").with_suffix(f".{format}"))
    stim_weights
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Psychometrics
    """)
    return


@app.cell
def _(
    TASK_NAME,
    fig_size,
    format,
    path_panels,
    plot_dfs,
    plots_by_task,
    views,
):
    _plots = plots_by_task[TASK_NAME]
    _result = _plots.plot_categorical_performance_all(
        plot_dfs[TASK_NAME],
        "glm",
        background_style="model",
        views=views[TASK_NAME],
    )
    fig_psychometric = _result[0] if isinstance(_result, tuple) else _result
    fig_psychometric.set_size_inches(fig_size(2, 1))
    fig_psychometric.savefig((path_panels / f"{TASK_NAME}_psychometric").with_suffix(f".{format}"))
    fig_psychometric
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example session repetition
    """)
    return


@app.cell(hide_code=True)
def _(TASK_NAME, mo, subjects_by_task):
    ui_single_session_subject = mo.ui.dropdown(
        options=subjects_by_task[TASK_NAME],
        value=subjects_by_task[TASK_NAME][0],
        label="Subject",
    )
    return (ui_single_session_subject,)


@app.cell(hide_code=True)
def _(TASK_NAME, mo, pl, plot_dfs, ui_single_session_subject):
    single_subject_df = plot_dfs[TASK_NAME].filter(
        pl.col("subject") == ui_single_session_subject.value
    )
    _sessions = list(single_subject_df["session"].unique().sort())
    ui_single_session_session = mo.ui.dropdown(
        options=_sessions,
        value=_sessions[0],
        label="Session",
    )
    return single_subject_df, ui_single_session_session


@app.cell(hide_code=True)
def _(mo, ui_single_session_session, ui_single_session_subject):
    mo.hstack([ui_single_session_subject, ui_single_session_session], justify="start")
    return


@app.cell
def _(
    TASK_NAME,
    adapters,
    build_session_trial_outcomes_data,
    single_subject_df,
    ui_single_session_session,
    ui_single_session_subject,
):
    session_trial_outcomes_data, session_trial_xlabel, _ = build_session_trial_outcomes_data(
        single_subject_df,
        task_name=TASK_NAME,
        subject=ui_single_session_subject.value,
        session=ui_single_session_session.value,
        adapter=adapters[TASK_NAME],
    )
    print(ui_single_session_subject.value)
    print(ui_single_session_session.value)
    return


@app.cell
def _(
    TASK_NAME,
    adapters,
    build_session_repetition_data,
    single_subject_df,
    ui_single_session_session,
    ui_single_session_subject,
):
    session_repetition_data = build_session_repetition_data(
        single_subject_df,
        subject=ui_single_session_subject.value,
        session=ui_single_session_session.value,
        adapter=adapters[TASK_NAME],
        window=20,
    )
    return (session_repetition_data,)


@app.cell
def _(plot_session_response_raster, session_repetition_data):
    fig_response_raster, _ = plot_session_response_raster(session_repetition_data)
    fig_response_raster
    return


@app.cell
def _(fig_size, plt, session_repetition_data):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    single_session = plt.gca()

    single_session.plot(
        "trial_x",
        "response_repeat_window_fraction",
        color="tab:brown",
        linewidth=1.5,
        label="Choice",
        data=session_repetition_data,
    )
    single_session.plot(
        "trial_x",
        "stimulus_repeat_window_fraction",
        color="tab:blue",
        linewidth=1.5,
        label="Stimulus",
        data=session_repetition_data,
    )
    single_session.set_xlabel("Trial")
    single_session.set_ylabel("Rep. fraction")
    single_session.set_ylim(0, 1)
    single_session.set_xlim(-0.5, len(session_repetition_data) - 0.5)
    single_session.legend(frameon=False, loc="lower right")
    return


@app.cell
def _():
    # _subject = subjects_by_task[TASK_NAME][0]
    # _session = list(
    #     plot_dfs[TASK_NAME]
    #     .filter(pl.col("subject") == _subject)["session"]
    #     .unique()
    #     .sort()
    # )[0]
    # _subject_df = plot_dfs[TASK_NAME].filter(
    #     pl.col("subject") == _subject,
    #     pl.col("session") == _session,
    # )
    # session_repetition_data_nuo = build_session_repetition_data(
    #     _subject_df,
    #     subject=_subject,
    #     session=_session,
    #     adapter=adapters[TASK_NAME],
    #     window=20,
    # )
    # plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    # single_session_nuo = plt.gca() if not mount_figure else axd["single_session"]
    # single_session_nuo.clear()

    # single_session_nuo.plot(
    #     "trial_x",
    #     "response_repeat_window_fraction",
    #     color="tab:brown",
    #     linewidth=1.5,
    #     label="Choice",
    #     data=session_repetition_data_nuo,
    # )
    # single_session_nuo.plot(
    #     "trial_x",
    #     "stimulus_repeat_window_fraction",
    #     color="tab:blue",
    #     linewidth=1.5,
    #     label="Stimulus",
    #     data=session_repetition_data_nuo,
    # )
    # single_session_nuo.set_xlabel("Trial")
    # single_session_nuo.set_ylabel("Rep. fraction")
    # single_session_nuo.set_ylim(0, 1)
    # single_session_nuo.set_xlim(-0.5, len(session_repetition_data_nuo) - 0.5)
    # single_session_nuo.legend(frameon=False, loc="upper right")
    # if not mount_figure:
    #     single_session_nuo.figure.savefig(
    #         (path_panels / f"{TASK_NAME}_single_session").with_suffix(f".{format}")
    #     )
    # single_session_nuo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histograms of repetition/alternation chunks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Settings
    """)
    return


@app.cell
def _():
    chunk_hist_stat = "count"
    chunk_hist_ylabel = {"count": "Count", "probability": "Frequency"}[chunk_hist_stat]
    transition_palette = {"repeating": "tab:brown", "alternating": "tab:purple"}
    return chunk_hist_stat, chunk_hist_ylabel, transition_palette


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Process data
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    build_transition_chunk_plot_data,
    chunk_hist_stat,
    plot_dfs,
    task_labels,
    task_names,
    task_order,
    transition_palette,
):
    transition_chunk_lengths_by_task, transition_chunk_plot_data, _transition_palette, transition_repeat_probabilities = (
        build_transition_chunk_plot_data(
            plot_dfs,
            task_names,
            stat=chunk_hist_stat,
            task_labels=task_labels,
            task_order=task_order,
            transition_palette=transition_palette,
        )
    )
    return (
        transition_chunk_lengths_by_task,
        transition_chunk_plot_data,
        transition_repeat_probabilities,
    )


@app.cell
def _(TASK_LABEL, transition_chunk_plot_data):
    transition_chunk_plot_data[
            transition_chunk_plot_data["task_label"] == TASK_LABEL
        ]
    return


@app.cell
def _(subjects_nuo):
    subjects_nuo
    return


@app.cell
def _(plot_dfs):
    plot_dfs["nuo_auditory"]["current_training_stage"].unique()
    return


@app.cell
def _(
    TASK_LABEL,
    TASK_NAME,
    axd,
    chunk_hist_ylabel,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    transition_chunk_plot_data,
    transition_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    consec_rep_nuo = plt.gca() if not mount_figure else axd["transition_chunks"]
    consec_rep_nuo.clear()

    sns.lineplot(
        data=transition_chunk_plot_data[
            transition_chunk_plot_data["task_label"] == TASK_LABEL
        ],
        x="chunk_length",
        y="weight",
        hue="transition",
        style="source",
        palette=transition_palette,
        dashes={"Data": "", "Independent choices": (2, 2)},
        markers=False,
        errorbar=None,
        ax=consec_rep_nuo,
    )
    consec_rep_nuo.set_xlim(0, 30)
    consec_rep_nuo.set_ylim(1, 1e3)
    # consec_rep_nuo.set_ylim(1e-6,1)
    consec_rep_nuo.set_yscale("log")
    consec_rep_nuo.set_xlabel("Consecutive choices")
    consec_rep_nuo.set_ylabel(chunk_hist_ylabel)
    _handles, _labels = consec_rep_nuo.get_legend_handles_labels()
    consec_rep_nuo.legend(
        [h for h, label in zip(_handles, _labels) if label not in ["transition", "source"]],
        [label for label in _labels if label not in ["transition", "source"]],
        frameon=False,
    )
    if not mount_figure:
        consec_rep_nuo.figure.savefig(
            (path_panels / f"{TASK_NAME}_choice_transition_chunks").with_suffix(f".{format}")
        )
    consec_rep_nuo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution tests
    """)
    return


@app.cell
def _(
    animal_chunk_histogram,
    np,
    pd,
    task_order,
    transition_chunk_lengths_by_task,
    transition_repeat_probabilities,
):
    from scipy.stats import chi2

    _max_test_chunk_length = 20

    def _geometric_chunk_probability(_chunk_lengths, _repeat_probability, _transition):
        _continue_probability = (
            _repeat_probability
            if _transition == "repeating"
            else 1.0 - _repeat_probability
        )
        return (1.0 - _continue_probability) * (_continue_probability ** (_chunk_lengths - 1))

    def _repeat_probability_for(_task_label):
        _matches = transition_repeat_probabilities.loc[
            (transition_repeat_probabilities["task_label"] == _task_label)
            & (transition_repeat_probabilities["sequence"] == "Choices"),
            "p_repeat",
        ]
        if _matches.empty:
            return None
        return float(_matches.iloc[0])

    _rows = []
    _x = np.arange(1, _max_test_chunk_length + 1)
    for _task_label in task_order:
        _repeat_probability = _repeat_probability_for(_task_label)
        if _repeat_probability is None:
            continue
        _data = transition_chunk_lengths_by_task[
            (transition_chunk_lengths_by_task["task_label"] == _task_label)
            & (transition_chunk_lengths_by_task["sequence"] == "Choices")
        ]
        if _data.empty:
            continue
        _hist_data = animal_chunk_histogram(
            _data,
            group_cols=["transition"],
            stat="count",
        )
        for _transition in ["repeating", "alternating"]:
            _transition_data = _hist_data[_hist_data["transition"] == _transition]
            if _transition_data.empty:
                continue
            _observed = (
                _transition_data
                .groupby("chunk_length", observed=True)["hist_weight"]
                .sum()
            )
            _observed_bins = _observed.reindex(_x, fill_value=0).to_numpy(dtype=float)
            _observed_tail = float(_observed[_observed.index > _max_test_chunk_length].sum())
            _observed_bins = np.r_[_observed_bins, _observed_tail]

            _continue_probability = (
                _repeat_probability
                if _transition == "repeating"
                else 1.0 - _repeat_probability
            )
            _expected_probabilities = np.r_[
                _geometric_chunk_probability(_x, _repeat_probability, _transition),
                _continue_probability ** _max_test_chunk_length,
            ]
            _expected_bins = _expected_probabilities * _observed_bins.sum()
            _valid_bins = _expected_bins > 0
            _chi_square = float(
                (((_observed_bins[_valid_bins] - _expected_bins[_valid_bins]) ** 2) / _expected_bins[_valid_bins]).sum()
            )
            _degrees_of_freedom = int(_valid_bins.sum() - 1)
            _p_value = float(chi2.sf(_chi_square, _degrees_of_freedom))
            _rows.append(
                {
                    "task": _task_label,
                    "transition": _transition,
                    "n_subjects": _transition_data["subject"].nunique(),
                    "mean_chunks_per_subject": _observed_bins.sum(),
                    "chi_square": _chi_square,
                    "df": _degrees_of_freedom,
                    "p_value": _p_value,
                }
            )

    transition_chunk_distribution_tests = pd.DataFrame(_rows)
    transition_chunk_distribution_tests
    return (chi2,)


@app.cell
def _(chi2, np, pd, task_order, transition_chunk_lengths_by_task):
    _max_test_chunk_length = 20

    def _geometric_chunk_probability(_chunk_lengths, _repeat_probability, _transition):
        _continue_probability = (
            _repeat_probability
            if _transition == "repeating"
            else 1.0 - _repeat_probability
        )
        return (1.0 - _continue_probability) * (_continue_probability ** (_chunk_lengths - 1))

    _rows = []
    _x = np.arange(1, _max_test_chunk_length + 1)
    for _task_label in task_order:
        _data = transition_chunk_lengths_by_task[
            (transition_chunk_lengths_by_task["task_label"] == _task_label)
            & (transition_chunk_lengths_by_task["sequence"] == "Choices")
        ]
        if _data.empty:
            continue

        for _subject, _subject_data in _data.groupby("subject", observed=True):
            _subject_transition_counts = (
                _subject_data
                .groupby("transition", observed=True)["chunk_length"]
                .sum()
            )
            _subject_total_transitions = float(_subject_transition_counts.sum())
            if _subject_total_transitions <= 0:
                continue

            _repeat_probability = (
                float(_subject_transition_counts.get("repeating", 0.0))
                / _subject_total_transitions
            )
            if _repeat_probability <= 0.0 or _repeat_probability >= 1.0:
                continue

            for _transition in ["repeating", "alternating"]:
                _transition_data = _subject_data[
                    _subject_data["transition"] == _transition
                ]
                if _transition_data.empty:
                    continue

                _observed = (
                    _transition_data
                    .groupby("chunk_length", observed=True)
                    .size()
                )
                _observed_bins = _observed.reindex(_x, fill_value=0).to_numpy(dtype=float)
                _observed_tail = float(_observed[_observed.index > _max_test_chunk_length].sum())
                _observed_bins = np.r_[_observed_bins, _observed_tail]
                _n_chunks = float(_observed_bins.sum())
                if _n_chunks <= 0:
                    continue

                _continue_probability = (
                    _repeat_probability
                    if _transition == "repeating"
                    else 1.0 - _repeat_probability
                )
                _expected_probabilities = np.r_[
                    _geometric_chunk_probability(_x, _repeat_probability, _transition),
                    _continue_probability ** _max_test_chunk_length,
                ]
                _expected_bins = _expected_probabilities * _n_chunks
                _valid_bins = _expected_bins > 0
                _degrees_of_freedom = int(_valid_bins.sum() - 1)
                if _degrees_of_freedom <= 0:
                    continue

                _chi_square = float(
                    (((_observed_bins[_valid_bins] - _expected_bins[_valid_bins]) ** 2) / _expected_bins[_valid_bins]).sum()
                )
                _p_value = float(chi2.sf(_chi_square, _degrees_of_freedom))
                _rows.append(
                    {
                        "task": _task_label,
                        "subject": _subject,
                        "transition": _transition,
                        "p_repeat_subject": _repeat_probability,
                        "n_chunks": _n_chunks,
                        "chi_square": _chi_square,
                        "df": _degrees_of_freedom,
                        "p_value": _p_value,
                    }
                )

    _test_columns = [
        "task",
        "subject",
        "transition",
        "p_repeat_subject",
        "n_chunks",
        "chi_square",
        "df",
        "p_value",
    ]
    transition_chunk_distribution_subject_tests = pd.DataFrame(
        _rows,
        columns=_test_columns,
    )
    transition_chunk_distribution_subject_summary = (
        transition_chunk_distribution_subject_tests
        .groupby(["task", "transition"], observed=True)
        .agg(
            n_subjects=("subject", "nunique"),
            median_chunks_per_subject=("n_chunks", "median"),
            median_chi_square=("chi_square", "median"),
            median_df=("df", "median"),
            median_p_value=("p_value", "median"),
            fraction_p_lt_0_05=("p_value", lambda values: float((values < 0.05).mean())),
        )
        .reset_index()
    )

    transition_chunk_distribution_subject_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Full figure
    """)
    return


@app.cell
def _(fig, format, mount_figure, path_panels):
    if mount_figure:
        fig.savefig((path_panels / "figure2_nuo_auditory").with_suffix(f".{format}"))
    fig
    return


if __name__ == "__main__":
    app.run()
