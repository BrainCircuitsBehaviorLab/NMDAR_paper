# /// script
# [tool.marimo.opengraph]
# title = "Figure 2" 
# description = " Figure 2: GLM model predictions."
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
    import base64
    import io
    import re
    from pathlib import Path
    import os

    import marimo as mo
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_1samp
    from statannotations.Annotator import Annotator

    # Custom package and plots
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays)
    from glmhmmt.plots.emissions import _fold_three_choice_raw_weights as fold_three_choice
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import (
        add_choice_lag_summary_regressor,
        add_fixed_accuracy_repetition_band,
        build_outcome_streak_plot_data,
        build_repetition_chunk_plot_data,
        build_transition_chunk_drug_plot_data,
        build_transition_chunk_plot_data,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms)
    from src.plots.common import (
        animal_chunk_histogram,
        boxplot_STYLE,
        build_repetition_variance_by_drug_task,
        build_session_repetition_data,
        build_session_trial_outcomes_data,
        fig_size,
        pick_existing_column,
        plot_session_repetition_running_count,
        plot_session_response_raster,
        plot_session_trial_outcomes,
        two_afc_session_repeat_alternate_accuracy as build_two_afc_session_repeat_alternate_accuracy,
        two_afc_transition_chunk_lengths as build_two_afc_transition_chunk_lengths)

    return (
        Annotator,
        Path,
        add_choice_lag_summary_regressor,
        add_fixed_accuracy_repetition_band,
        animal_chunk_histogram,
        boxplot_STYLE,
        build_outcome_streak_plot_data,
        build_repetition_chunk_plot_data,
        build_repetition_variance_by_drug_task,
        build_session_repetition_data,
        build_session_trial_outcomes_data,
        build_transition_chunk_drug_plot_data,
        build_transition_chunk_plot_data,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        fold_three_choice,
        get_adapter,
        get_runtime_paths,
        load_app_config,
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
        process_mcdr,
        process_two_adc,
        process_two_afc,
        sns,
        ttest_1samp,
    )


@app.cell
def _(load_app_config, process_mcdr, process_two_adc, process_two_afc):
    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    return (prepare_predictions_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings
    """)
    return


@app.cell
def _():
    mount_figure = False
    format = "svg"
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

    data_path = Path(__file__).parents[1] / "data/processed"
    print(data_path)

    project_path = Path(__file__).resolve().parents[1]
    print(project_path)

    path_panels = project_path / "figures" / "panels2" / format

    os.makedirs(path_panels, exist_ok=True)
    print(path_panels)
    return path_panels, paths, project_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Style
    """)
    return


@app.cell
def _(Path, plt, sns):
    sns.set_theme(style='ticks', context='notebook')
    plt.style.use(Path(__file__).resolve().parents[1] / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = 'none'
    plt.rcParams['savefig.bbox'] = 'standard'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data and fits
    """)
    return


@app.cell
def _(get_adapter, pl):
    task_names = ("2AFC_delay", "2AFC", "MCDR")
    model_name = "one hot"
    adapters = {_task_name: get_adapter(_task_name) for _task_name in task_names}
    plots_by_task = {
        _task_name: _adapter.get_plots()
        for _task_name, _adapter in adapters.items()
    }
    dfs = {
        _task_name: _adapter.subject_filter(_adapter.read_dataset())
        for _task_name, _adapter in adapters.items()
    }
    dfs["2AFC"] = dfs["2AFC"].filter(pl.col("subject") != "326")
    dfs["MCDR"] = dfs["MCDR"].filter(pl.col("subject").str.contains("B"))


    subjects_by_task = {
        _task_name: list(_df["subject"].unique())
        for _task_name, _df in dfs.items()
    }
    return (
        adapters,
        dfs,
        model_name,
        plots_by_task,
        subjects_by_task,
        task_names,
    )


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
        views[_task] = build_views(_arrays_store, _adapter, 1, subjects_by_task[_task])
        trial_dfs[_task], weight_dfs[_task] = build_trial_and_weights_df(_df_all, views=views[_task], adapter=_adapter, min_session_length=1)
        plot_dfs[_task] = prepare_predictions_df(_task, trial_dfs[_task])
        # plot_dfs[_task] = plot_dfs[_task].sort(["subject", "session", "trial_idx"]).filter(pl.len().over(["subject", "session"]) > 50, pl.cum_count("trial_idx").over(["subject", "session"]) <= pl.len().over(["subject", "session"]) - 50)
        _choice_lag_cols = []
        for _view in views[_task].values():
            for _feature in list(getattr(_view, "feat_names", []) or []):
                _feature = str(_feature)
                if _feature.startswith("choice_lag_") and _feature not in _choice_lag_cols:
                    _choice_lag_cols.append(_feature)
        # plot_dfs[_task] = add_choice_lag_summary_regressor(plot_dfs[_task], choice_lag_cols=_adapter.choice_lag_cols(trial_dfs[_task]))
        plot_dfs[_task] = add_choice_lag_summary_regressor(plot_dfs[_task], choice_lag_cols=_choice_lag_cols)
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
                ["a", "a", "b", "b"],
                # ["single_sess_acc_2ADC", "single_sess_acc_2ADC", "single_sess_acc_2AFC", "single_sess_acc_2AFC"],
                # ["_running_legend", "_running_legend", "_running_legend", "_running_legend"],
                ["hist_correct_2ADC", "hist_repeat_2ADC", "hist_correct_2AFC", "hist_repeat_2AFC"],
                # ["a", "a", "b", "b"],
                ["e", "f", "g", "h"],
                ["i", "k", "j", "l"]
                # ["i", "i", "j", "j"],
                # ["k", "k", "l", "l"],
            ],
            figsize=fig_size(1,0.8),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1.1, 1, 1, 1]},
        )
        # fig.set_constrained_layout_pads(
        #         w_pad=0.01,
        #         h_pad=0.01,
        #         wspace=0.01,
        #         hspace=0.07,
        #     )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Autocorrelograms
    """)
    return


@app.cell
def _(
    adapters,
    dfs,
    prepare_closed_loop_model_autocorrelograms,
    prepare_corrected_behavior_autocorrelograms,
    subjects_by_task,
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
        _subjects = subjects_by_task[_task]
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    autocorrelograms_by_task,
    axd,
    fig,
    fig_size,
    format,
    mo,
    mount_figure,
    path_panels,
    plt,
):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    ax_autocorrelograms_2ADC_outcome = plt.gca() if not mount_figure else axd["i"]
    ax_autocorrelograms_2ADC_outcome.clear()
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    ax_autocorrelograms_2ADC_repetition = plt.gca() if not mount_figure else axd["k"]
    ax_autocorrelograms_2ADC_repetition.clear()


    _data_ac = autocorrelograms_by_task["2AFC_delay"]["data"]["autocorr"]
    _glm_ac = autocorrelograms_by_task["2AFC_delay"]["glm"]["autocorr"]
    _colors = {
        "data": "tab:blue",
        "glm": "tab:gray",
    }
    for _signal, _ax in (
        ("Outcome", ax_autocorrelograms_2ADC_outcome),
        ("Repetition", ax_autocorrelograms_2ADC_repetition)):
        _fig = _ax.figure
        _data_sub = _data_ac[_data_ac["signal"] == _signal].sort_values("lag")
        _ax.errorbar(
            _data_sub["lag"],
            _data_sub["autocorr"],
            yerr=_data_sub.get("autocorr_sem"),
            fmt="o",
            capsize=0,
            ms = 3,
            color=_colors["data"],
            ecolor=_colors["data"],
            label="Data",
            zorder=4,
        )

        for _label, _model_ac, _color in (
            ("GLM", _glm_ac, _colors["glm"]),
        ):
            _sub = _model_ac[_model_ac["signal"] == _signal].sort_values("lag")
            if _sub.empty:
                continue
            _ax.plot(_sub["lag"], _sub["autocorr"], color=_color, label=_label, zorder=3)

        _ax.axhline(0.0, color="0.5", ls="--")
        _ax.set_title("Outcomes" if _signal == "Outcome" else "Repetitions")
        _ax.set_xlabel("Lag")
        _ax.set_ylabel("Autocorrelation")
        _ax.set_xlim(0,20)

        _major_pos = [i for i in range(20) if (i + 1) % 5 == 0 or (i + 1) == 1]

        _minor_pos = [i for i in range(20) if i not in _major_pos]
        _ax.set_xticks(_major_pos)
        _ax.set_xticklabels([str(i + 1) for i in _major_pos])
        _ax.set_xticks(_minor_pos, minor=True)
        _ax.tick_params(axis='x', which='major', length=6)
        _ax.tick_params(axis='x', which='minor', length=3)

        if _signal == "Repetition":
            _ax.set_ylim(top=0.15)
        else:
            _ax.set_ylim(top=0.05)
        _ax.legend(frameon=False)
        if not mount_figure:
            _fig.savefig((path_panels / f"2ADC_autocorrelogram_{_signal.lower()}").with_suffix(f".{format}"))


    if not mount_figure:
        _display = mo.hstack([ax_autocorrelograms_2ADC_outcome, ax_autocorrelograms_2ADC_repetition], justify="start", gap=1)
    else: 
        _display = fig
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    autocorrelograms_by_task,
    axd,
    fig,
    fig_size,
    format,
    mo,
    mount_figure,
    path_panels,
    plt,
):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    ax_autocorrelograms_2AFC_outcome = plt.gca() if not mount_figure else axd["j"]
    ax_autocorrelograms_2AFC_outcome.clear()
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    ax_autocorrelograms_2AFC_repetition = plt.gca() if not mount_figure else axd["l"]
    ax_autocorrelograms_2AFC_repetition.clear()

    _data_ac = autocorrelograms_by_task["2AFC"]["data"]["autocorr"]
    _glm_ac = autocorrelograms_by_task["2AFC"]["glm"]["autocorr"]
    _colors = {
        "data": "tab:blue",
        "glm": "tab:gray",
    }
    for _signal, _ax in (
        ("Outcome", ax_autocorrelograms_2AFC_outcome),
        ("Repetition", ax_autocorrelograms_2AFC_repetition)):
        _fig = _ax.figure
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
        for _label, _model_ac, _color in (
            ("GLM", _glm_ac, _colors["glm"]),
        ):
            _sub = _model_ac[_model_ac["signal"] == _signal].sort_values("lag")
            if _sub.empty:
                continue
            _ax.plot(_sub["lag"], _sub["autocorr"], color=_color, label=_label, zorder=3)

        _ax.axhline(0.0, color="0.5", ls="--")
        _ax.set_title("Outcomes" if _signal == "Outcome" else "Repetitions")
        _ax.set_xlabel("Lag")
        _ax.set_ylabel("Autocorrelation")
        _ax.set_xlim(0,20)

        _major_pos = [i for i in range(20) if (i + 1) % 5 == 0 or (i + 1) == 1]

        _minor_pos = [i for i in range(20) if i not in _major_pos]
        _ax.set_xticks(_major_pos)
        _ax.set_xticklabels([str(i + 1) for i in _major_pos])
        _ax.set_xticks(_minor_pos, minor=True)
        _ax.tick_params(axis='x', which='major', length=6)
        _ax.tick_params(axis='x', which='minor', length=3)

        if _signal == "Repetition":
            _ax.set_ylim(top=0.15)
        else:
            _ax.set_ylim(top=0.05)
        _ax.legend(frameon=False)
        if not mount_figure:
            _fig.savefig((path_panels / f"2AFC_autocorrelogram_{_signal.lower()}").with_suffix(f".{format}"))

    if not mount_figure:
        _display = mo.hstack([ax_autocorrelograms_2AFC_outcome, ax_autocorrelograms_2AFC_repetition], justify="start", gap=1)
    else: 
        _display = fig
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell
def _(autocorrelograms_by_task, fig_size, format, mo, path_panels, plt):
    fig_autocorrelograms_MCDR_outcome, ax_autocorrelograms_MCDR_outcome = plt.subplots(
        figsize=fig_size(2), constrained_layout=True)
    fig_autocorrelograms_MCDR_repetition, ax_autocorrelograms_MCDR_repetition = plt.subplots(
        figsize=fig_size(2), constrained_layout=True)

    _data_ac = autocorrelograms_by_task["MCDR"]["data"]["autocorr"]
    _glm_ac = autocorrelograms_by_task["MCDR"]["glm"]["autocorr"]
    _colors = {
        "data": "tab:blue",
        "glm": "tab:gray",
    }
    for _signal, _ax in (
        ("Outcome", ax_autocorrelograms_MCDR_outcome),
        ("Repetition", ax_autocorrelograms_MCDR_repetition)):
        _fig = fig_autocorrelograms_MCDR_outcome if _signal == "Outcome" else fig_autocorrelograms_MCDR_repetition
        _data_sub = _data_ac[_data_ac["signal"] == _signal].sort_values("lag")
        _ax.errorbar(
            _data_sub["lag"],
            _data_sub["autocorr"],
            yerr=_data_sub.get("autocorr_sem"),
            fmt="o",
            capsize=0,
            ms=3,        color=_colors["data"],
            ecolor=_colors["data"],
            label="Data",
            zorder=4,
        )
        for _label, _model_ac, _color in (
            ("GLM", _glm_ac, _colors["glm"]),
        ):
            _sub = _model_ac[_model_ac["signal"] == _signal].sort_values("lag")
            if _sub.empty:
                continue
            _ax.plot(_sub["lag"], _sub["autocorr"], color=_color, label=_label, zorder=3)

        _ax.axhline(0.0, color="0.5", ls="--")
        _ax.set_title("Outcomes" if _signal == "Outcome" else "Repetitions")
        _ax.set_xlabel("Lag")
        _ax.set_ylabel("Autocorrelation")
        if _signal == "Repetition":
            _ax.set_ylim(top=0.15)
        else:
            _ax.set_ylim(top=0.05)
        _ax.legend(frameon=False)
        _fig.savefig((path_panels / f"MCDR_autocorrelogram_{_signal.lower()}").with_suffix(f".{format}"))

    mo.hstack([fig_autocorrelograms_MCDR_outcome, fig_autocorrelograms_MCDR_repetition], justify="start", gap=1)
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
    ### 2ADC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Previous choices
    """)
    return


@app.cell
def _():
    import matplotlib.ticker as mticker

    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    prev_choices_2ADC = plt.gca() if not mount_figure else axd["f"]
    prev_choices_2ADC.clear()

    # Filter to just have lagged choices
    _plot_df = weight_dfs["2AFC_delay"].filter(pl.col("feature").str.contains("choice_lag")) 
    _order = sorted(_plot_df["feature"].unique(), key=lambda x: int(x.split("_")[-1]),)
    sns.boxplot(
        data=_plot_df,
        x="feature",
        y="weight",
        order = _order,
        color="tab:gray",
        ax=prev_choices_2ADC,
        **boxplot_STYLE,
    )
    prev_choices_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    # prev_choices_2ADC.set_title("2ADC")
    prev_choices_2ADC.set_ylabel("Weight")
    prev_choices_2ADC.set_xlabel("Lag")

    _major_pos = [i for i in range(len(_order)) if (i + 1) % 5 == 0 or (i + 1) == 1]

    _minor_pos = [i for i in range(len(_order)) if i not in _major_pos]
    prev_choices_2ADC.set_xticks(_major_pos)
    prev_choices_2ADC.set_xticklabels([str(i + 1) for i in _major_pos])
    prev_choices_2ADC.set_xticks(_minor_pos, minor=True)
    prev_choices_2ADC.tick_params(axis='x', which='major', length=6)
    prev_choices_2ADC.tick_params(axis='x', which='minor', length=3)


    prev_choices_2ADC.set_ylim(-0.1,0.85)

    if not mount_figure:
        prev_choices_2ADC.figure.savefig((path_panels / "2ADC_prev_choices").with_suffix(f".{format}"))
    prev_choices_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stimulus
    """)
    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    stim_2ADC = plt.gca() if not mount_figure else axd["e"]
    stim_2ADC.clear()

    # Filter to just have lagged choices
    _plot_df = weight_dfs["2AFC_delay"].filter(pl.col("feature").str.contains("stim")) 
    _order = sorted(
        _plot_df["feature"].unique(),
        key=lambda x: float(
            x.split("stim_x_delay_hot_")[-1].replace("m", "-").replace("p", ".").replace("h", ".")
        ),
        reverse=True,
    )
    sns.boxplot(
        data=_plot_df,
        x="feature",
        y="weight",
        order = _order,
        color="tab:gray",
        ax=stim_2ADC,
        **boxplot_STYLE,
    )
    stim_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    stim_2ADC.set_xlabel("")
    stim_2ADC.set_ylabel("Weight")
    stim_2ADC.set_xlabel("Delay")
    stim_2ADC.set_xticklabels([10 ,3,1, 0.1])
    stim_2ADC.set_ylim(-0.1,3.5)

    if not mount_figure:
        plt.savefig((path_panels / "2ADC_stim").with_suffix(f".{format}"))
    stim_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Previous choices
    """)
    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    prev_choices_2AFC = plt.gca() if not mount_figure else axd["h"]
    prev_choices_2AFC.clear()

    # Filter to just have lagged choices
    _plot_df = weight_dfs["2AFC"].filter(pl.col("feature").str.contains("choice_lag")) 
    _order = sorted(_plot_df["feature"].unique(), key=lambda x: int(x.split("_")[-1]),)
    sns.boxplot(
        data=_plot_df,
        x="feature",
        y="weight",
        order = _order,
        color="tab:gray",
        ax=prev_choices_2AFC,
        **boxplot_STYLE,
    )
    prev_choices_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    # prev_choices_2AFC.set_title("2AFC")
    prev_choices_2AFC.set_ylabel("Weight")
    prev_choices_2AFC.set_xlabel("Lag")

    _major_pos = [i for i in range(len(_order)) if (i + 1) % 5 == 0 or (i + 1) == 1]

    _minor_pos = [i for i in range(len(_order)) if i not in _major_pos]
    prev_choices_2AFC.set_xticks(_major_pos)
    prev_choices_2AFC.set_xticklabels([str(i + 1) for i in _major_pos])
    prev_choices_2AFC.set_xticks(_minor_pos, minor=True)
    prev_choices_2AFC.tick_params(axis='x', which='major', length=6)
    prev_choices_2AFC.tick_params(axis='x', which='minor', length=3)
    prev_choices_2AFC.set_ylim(-0.1,0.85)
    if not mount_figure:
        prev_choices_2AFC.figure.savefig((path_panels / "2AFC_prev_choices").with_suffix(f".{format}"))
    prev_choices_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stimulus
    """)
    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    stim_2AFC = plt.gca() if not mount_figure else axd["g"]
    stim_2AFC.clear()

    # Filter to just have stimulus
    _plot_df = weight_dfs["2AFC"].filter(pl.col("feature").str.contains("stim")) 
    _order = sorted(_plot_df["feature"].unique(), key=lambda x: int(x.split("_")[-1]),)
    sns.boxplot(
        data=_plot_df,
        x="feature",
        y="weight",
        order = _order,
        color="tab:gray",
        ax=stim_2AFC,
        **boxplot_STYLE,
    )
    stim_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    # stim_2AFC.set_title("2AFC")
    stim_2AFC.set_ylabel("Weight")
    stim_2AFC.set_xlabel("ILD")
    stim_2AFC.set_xticklabels([2,4,8, 70])
    stim_2AFC.set_ylim(-0.1,3.5)
    if not mount_figure:
        stim_2AFC.figure.savefig((path_panels / "2AFC_stim").with_suffix(f".{format}"))
    stim_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Previous choices
    """)
    return


@app.cell
def _(
    boxplot_STYLE,
    fig_size,
    fold_three_choice,
    format,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    prev_choices_MCDR = plt.gca()

    # Filter to just have lagged choices
    _plot_df = pl.from_pandas(fold_three_choice(weight_dfs["MCDR"])).filter(pl.col("feature").str.contains("choice_lag")) 
    _order = sorted(_plot_df["feature"].unique(), key=lambda x: int(x.split("_")[-1]),)
    sns.boxplot(
        data=_plot_df,
        x="feature",
        y="weight",
        order = _order,
        color="tab:gray",
        ax=prev_choices_MCDR,
        **boxplot_STYLE,
    )
    prev_choices_MCDR.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    # prev_choices_MCDR.set_title("MCDR")
    prev_choices_MCDR.set_ylabel("Weight")
    prev_choices_MCDR.set_xlabel("Lag")
    prev_choices_MCDR.set_xticklabels([str(i) if i == 1 or i % 5 == 0 else "" for i in range(1, len(_order) + 1)])
    plt.savefig((path_panels / "MCDR_prev_choices").with_suffix(f".{format}"))
    prev_choices_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Stimulus
    """)
    return


@app.cell
def _(
    boxplot_STYLE,
    fig_size,
    fold_three_choice,
    format,
    path_panels,
    pl,
    plt,
    sns,
    weight_dfs,
):
    plt.figure(figsize=fig_size(3), constrained_layout=True)
    stim_MCDR = plt.gca()

    # Filter to just have lagged choices
    _plot_df = pl.from_pandas(fold_three_choice(weight_dfs["MCDR"])).filter(pl.col("feature").str.contains("stim")) 
    _order = sorted(_plot_df["feature"].unique(), key=lambda x: int(x.replace("stim", "")),)
    sns.boxplot(
        data=_plot_df,
        x="feature",
        y="weight",
        order = _order,
        color="tab:gray",
        ax=stim_MCDR,
        **boxplot_STYLE,
    )
    stim_MCDR.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    # stim_MCDR.set_title("MCDR")
    stim_MCDR.set_ylabel("Weight")
    stim_MCDR.set_xlabel("Difficulty")
    stim_MCDR.set_xticklabels([1, 2, 3, 4])
    plt.savefig((path_panels / "MCDR_stim").with_suffix(f".{format}"))
    stim_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Psychometrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(fig_size, format, path_panels, plot_dfs, plots_by_task, views):
    _plots = plots_by_task["2AFC_delay"]
    _perf_kwargs = {"views": views["2AFC_delay"]}
    fig_psychometric_2ADC, _ = _plots.plot_categorical_performance_all(
        plot_dfs["2AFC_delay"],
        "glm",
        background_style="model",
        **_perf_kwargs,
        figsize=fig_size(2, 1),
    )
    fig_psychometric_2ADC.savefig((path_panels / "2ADC_psychometric").with_suffix(f".{format}"))
    fig_psychometric_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(fig_size, format, path_panels, plot_dfs, plots_by_task, views):
    _plots = plots_by_task["2AFC"]
    _perf_kwargs = {"views": views["2AFC"]}
    fig_psychometric_2AFC, _ = _plots.plot_categorical_performance_all(
        plot_dfs["2AFC"],
        "glm",
        background_style="model",
        **_perf_kwargs,
        figsize=fig_size(2, 1),
    )
    fig_psychometric_2AFC.savefig((path_panels / "2AFC_psychometric").with_suffix(f".{format}"))
    fig_psychometric_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell
def _(fig_size, format, path_panels, plot_dfs, plots_by_task):
    _plots = plots_by_task["MCDR"]
    fig_psychometric_MCDR, _ = _plots.plot_categorical_performance_all(
        plot_dfs["MCDR"],
        "glm",
        background_style="model",
        figsize=fig_size(3, 1),
    )
    for _fig, _stem in zip(
        fig_psychometric_MCDR,
        ("difficulty", "stimulus", "delay"),
        strict=False,
    ):
        _fig.savefig((path_panels / f"MCDR_psychometric_{_stem}").with_suffix(f".{format}"))
    fig_psychometric_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example session repetition
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exploratory for multiple tasks with UI
    """)
    return


@app.cell(hide_code=True)
def _(mo, task_names):
    ui_single_session_task = mo.ui.dropdown(
        options=list(task_names),
        value=task_names[0],    label="Task",
    )
    return (ui_single_session_task,)


@app.cell(hide_code=True)
def _(mo, subjects_by_task, ui_single_session_task):
    ui_single_session_subject = mo.ui.dropdown(
        options=subjects_by_task[ui_single_session_task.value],
        value=subjects_by_task[ui_single_session_task.value][0],
        label="Subject",
    )
    return (ui_single_session_subject,)


@app.cell(hide_code=True)
def _(mo, pl, plot_dfs, ui_single_session_subject, ui_single_session_task):
    single_subject_df = plot_dfs[ui_single_session_task.value].filter(pl.col("subject") == ui_single_session_subject.value)
    ui_single_session_session = mo.ui.dropdown(
        options=single_subject_df["session"].unique().sort(),
        value=single_subject_df["session"].unique()[0],
        label="Session",
    )
    return single_subject_df, ui_single_session_session


@app.cell(hide_code=True)
def _(
    mo,
    ui_single_session_session,
    ui_single_session_subject,
    ui_single_session_task,
):
    mo.hstack([ui_single_session_task, ui_single_session_subject, ui_single_session_session], justify="start")
    return


@app.cell
def _(
    adapters,
    build_session_trial_outcomes_data,
    single_subject_df,
    ui_single_session_session,
    ui_single_session_subject,
    ui_single_session_task,
):
    session_trial_outcomes_data, session_trial_xlabel, _ = build_session_trial_outcomes_data(
        single_subject_df,
        task_name=ui_single_session_task.value,
        subject=ui_single_session_subject.value,
        session=ui_single_session_session.value,
        adapter=adapters[ui_single_session_task.value],
    )
    print(ui_single_session_subject.value)
    print(ui_single_session_session.value)
    return


@app.cell
def _(
    adapters,
    add_fixed_accuracy_repetition_band,
    build_session_repetition_data,
    single_subject_df,
    ui_single_session_session,
    ui_single_session_subject,
    ui_single_session_task,
):
    session_repetition_data = build_session_repetition_data(
        single_subject_df,
        subject=ui_single_session_subject.value,
        session=ui_single_session_session.value,
        adapter=adapters[ui_single_session_task.value],
        window = 20,
    )
    session_repetition_data = add_fixed_accuracy_repetition_band(session_repetition_data)
    return (session_repetition_data,)


@app.cell
def _(plot_session_response_raster, session_repetition_data):
    fig_response_raster, _ = plot_session_response_raster(session_repetition_data)
    fig_response_raster
    return


@app.cell
def _(fig_size, plt, session_repetition_data):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    single_session_repetition = plt.gca()

    if {"fixed_accuracy_repeat_low", "fixed_accuracy_repeat_high"}.issubset(session_repetition_data.columns):
        single_session_repetition.fill_between(
            session_repetition_data["trial_x"].to_numpy(),
            session_repetition_data["fixed_accuracy_repeat_low"].to_numpy(),
            session_repetition_data["fixed_accuracy_repeat_high"].to_numpy(),
            color="tab:blue",
            alpha=0.1,
            linewidth=0,
        )
    single_session_repetition.plot(
        "trial_x",
        "response_repeat_window_fraction",
        color="tab:brown",
        linewidth=1.5,
        label="Choice rep.",
        data=session_repetition_data
    )
    single_session_repetition.plot(
        "trial_x",
        "stimulus_repeat_window_fraction",
        color="tab:blue",
        linewidth=1.5,
        label="Stim. rep.",
        data=session_repetition_data
    )
    single_session_repetition.plot(
        "trial_x",
        "accuracy_window_fraction",
        color="black",
        linewidth=1.5,
        label="Acc",
        data=session_repetition_data
    )
    single_session_repetition.set_xlabel("Trial")
    single_session_repetition.set_ylabel("Running fraction")
    single_session_repetition.set_ylim(0, 1)
    single_session_repetition.set_xlim(-0.5, len(session_repetition_data) - 0.5)
    single_session_repetition.legend(frameon=False, loc="lower right")
    return


@app.cell
def _(fig_size, plt, session_repetition_data):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    single_session_accuracy = plt.gca()

    single_session_accuracy.plot(
        "trial_x",
        "accuracy_window_fraction",
        color="black",
        linewidth=1.5,
        label="Accuracy",
        data=session_repetition_data
    )
    single_session_accuracy.set_xlabel("Trial")
    single_session_accuracy.set_ylabel("Running accuracy")
    single_session_accuracy.set_ylim(0, 1)
    single_session_accuracy.set_xlim(-0.5, len(session_repetition_data) - 0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    adapters,
    add_fixed_accuracy_repetition_band,
    build_session_repetition_data,
    pl,
    plot_dfs,
):
    _subject = "E11"
    _session = "E11_StageTraining_Ephys_V1_20210409-115620"
    _subject_df  = plot_dfs["2AFC_delay"].filter(pl.col("subject") == _subject, pl.col("session") == _session)
    session_repetition_data_2ADC = build_session_repetition_data(
        _subject_df,
        subject=_subject,
        session=_session,
        adapter=adapters["2AFC_delay"],
        window = 20,
    )
    session_repetition_data_2ADC = add_fixed_accuracy_repetition_band(session_repetition_data_2ADC)
    return (session_repetition_data_2ADC,)


@app.cell
def _(
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    session_repetition_data_2ADC,
):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    single_session_2ADC_repetition = plt.gca() if not mount_figure else axd["a"]
    single_session_2ADC_repetition.clear()

    single_session_2ADC_repetition.fill_between(
        session_repetition_data_2ADC["trial_x"].to_numpy(),
        session_repetition_data_2ADC["fixed_accuracy_repeat_low"].to_numpy(),
        session_repetition_data_2ADC["fixed_accuracy_repeat_high"].to_numpy(),
        color="tab:blue",
        alpha=0.1,
        linewidth=0,
    )
    # single_session_2ADC_repetition.fill_between(
    #     session_repetition_data_2ADC["trial_x"].to_numpy(),
    #     session_repetition_data_2ADC["fixed_accuracy_repeat_high"].to_numpy(),
    #     session_repetition_data_2ADC["response_repeat_window_fraction"].to_numpy(),
    #     where=session_repetition_data_2ADC["fixed_accuracy_choice_above"].to_numpy(),
    #     color="tab:brown",
    #     alpha=0.18,
    #     linewidth=0,
    # )
    single_session_2ADC_repetition.plot(
        "trial_x",
        "response_repeat_window_fraction",
        color="tab:brown",
        linewidth=1.5,
        label="Choice rep.",
        data=session_repetition_data_2ADC
    )
    single_session_2ADC_repetition.plot(
        "trial_x",
        "stimulus_repeat_window_fraction",
        color="tab:blue",
        linewidth=1.5,
        label="Stim. rep.",
        data=session_repetition_data_2ADC
    )
    single_session_2ADC_repetition.plot(
        "trial_x",
        "accuracy_window_fraction",
        color="black",
        linewidth=1.5,
        label="Acc",
        data=session_repetition_data_2ADC
    )
    # single_session_2ADC.set_title("2ADC")
    single_session_2ADC_repetition.set_xlabel("Trial")
    single_session_2ADC_repetition.set_ylabel("Running fraction")
    single_session_2ADC_repetition.set_ylim(0, 1)
    single_session_2ADC_repetition.set_xlim(-0.5, len(session_repetition_data_2ADC) - 0.5)
    single_session_2ADC_repetition.legend(frameon=False, loc="upper right")
    if not mount_figure:
        single_session_2ADC_repetition.figure.savefig(
            (path_panels / "2ADC_single_session_repetition").with_suffix(f".{format}")
        )
    single_session_2ADC_repetition
    return


@app.cell
def _(
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    session_repetition_data_2ADC,
):
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_2ADC_accuracy = plt.gca() if not mount_figure else axd["single_sess_acc_2ADC"]
    if mount_figure and "a_accuracy" in axd:
        single_session_2ADC_accuracy = axd["a_accuracy"]
    single_session_2ADC_accuracy.clear()

    single_session_2ADC_accuracy.plot(
        "trial_x",
        "accuracy_window_fraction",
        color="black",
        linewidth=1.5,
        label="Accuracy",
        data=session_repetition_data_2ADC
    )
    single_session_2ADC_accuracy.set_xlabel("Trial")
    single_session_2ADC_accuracy.set_ylabel("Running accuracy")
    single_session_2ADC_accuracy.set_ylim(0, 1)
    single_session_2ADC_accuracy.set_xlim(-0.5, len(session_repetition_data_2ADC) - 0.5)
    if not mount_figure or "a_accuracy" in axd:
        single_session_2ADC_accuracy.figure.savefig(
            (path_panels / "2ADC_single_session_accuracy").with_suffix(f".{format}")
        )
    single_session_2ADC_accuracy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    adapters,
    add_fixed_accuracy_repetition_band,
    build_session_repetition_data,
    pl,
    plot_dfs,
):
    _subject = "875"
    _session = "875_stage_training_v4_20230717-112113" 
    _subject_df  = plot_dfs["2AFC"].filter(pl.col("subject") == _subject, pl.col("session") == _session)
    session_repetition_data_2AFC = build_session_repetition_data(
        _subject_df,
        subject=_subject,
        session=_session,
        adapter=adapters["2AFC"],
        window = 20,
    )
    session_repetition_data_2AFC = add_fixed_accuracy_repetition_band(session_repetition_data_2AFC)
    return (session_repetition_data_2AFC,)


@app.cell
def _(
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    session_repetition_data_2AFC,
):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    single_session_2AFC_repetition = plt.gca() if not mount_figure else axd["b"]
    single_session_2AFC_repetition.clear()

    single_session_2AFC_repetition.fill_between(
        session_repetition_data_2AFC["trial_x"].to_numpy(),
        session_repetition_data_2AFC["fixed_accuracy_repeat_low"].to_numpy(),
        session_repetition_data_2AFC["fixed_accuracy_repeat_high"].to_numpy(),
        color="tab:blue",
        alpha=0.1,
        linewidth=0,
    )
    # single_session_2AFC_repetition.fill_between(
    #     session_repetition_data_2AFC["trial_x"].to_numpy(),
    #     session_repetition_data_2AFC["fixed_accuracy_repeat_high"].to_numpy(),
    #     session_repetition_data_2AFC["response_repeat_window_fraction"].to_numpy(),
    #     where=session_repetition_data_2AFC["fixed_accuracy_choice_above"].to_numpy(),
    #     color="tab:brown",
    #     alpha=0.18,
    #     linewidth=0,
    # )
    single_session_2AFC_repetition.plot(
        "trial_x",
        "response_repeat_window_fraction",
        color="tab:brown",
        linewidth=1.5,
        label="Choice Rep.",
        data=session_repetition_data_2AFC
    )
    single_session_2AFC_repetition.plot(
        "trial_x",
        "stimulus_repeat_window_fraction",
        color="tab:blue",
        linewidth=1.5,
        label="Stim. Rep.",
        data=session_repetition_data_2AFC
    )

    single_session_2AFC_repetition.plot(
        "trial_x",
        "accuracy_window_fraction",
        color="black",
        linewidth=1.5,
        label="Acc",
        data=session_repetition_data_2AFC
    )
    single_session_2AFC_repetition.set_title("")
    single_session_2AFC_repetition.set_xlabel("Trial")
    single_session_2AFC_repetition.set_ylabel("Running fraction")
    single_session_2AFC_repetition.set_ylim(0, 1)
    single_session_2AFC_repetition.set_xlim(-0.5, len(session_repetition_data_2AFC) - 0.5)
    single_session_2AFC_repetition.legend(frameon=False, loc="best")
    if not mount_figure:
        single_session_2AFC_repetition.figure.savefig(
            (path_panels / "2AFC_single_session_repetition").with_suffix(f".{format}")
        )
    single_session_2AFC_repetition
    return


@app.cell
def _(
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    session_repetition_data_2AFC,
):
    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    single_session_2AFC_accuracy = plt.gca() if not mount_figure else axd["single_sess_acc_2AFC"]
    if mount_figure and "b_accuracy" in axd:
        single_session_2AFC_accuracy = axd["b_accuracy"]
    single_session_2AFC_accuracy.clear()

    single_session_2AFC_accuracy.plot(
        "trial_x",
        "accuracy_window_fraction",
        color="black",
        linewidth=1.5,
        label="Accuracy",
        data=session_repetition_data_2AFC
    )
    single_session_2AFC_accuracy.set_xlabel("Trial")
    single_session_2AFC_accuracy.set_ylabel("Running accuracy")
    single_session_2AFC_accuracy.set_ylim(0, 1)
    single_session_2AFC_accuracy.set_xlim(-0.5, len(session_repetition_data_2AFC) - 0.5)
    if not mount_figure or "b_accuracy" in axd:
        single_session_2AFC_accuracy.figure.savefig(
            (path_panels / "2AFC_single_session_accuracy").with_suffix(f".{format}")
        )
    single_session_2AFC_accuracy
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fixed-accuracy band position
    """)
    return


@app.cell
def _(
    adapters,
    add_fixed_accuracy_repetition_band,
    autocorrelograms_by_task,
    build_session_repetition_data,
    pd,
    pl,
    plot_dfs,
    task_names,
):
    band_position_order = ["Below", "In", "Above"]
    band_position_source_order = ["Data", "GLM closed-loop"]
    _task_labels = {"2AFC_delay": "2ADC", "2AFC": "2AFC", "MCDR": "MCDR"}
    band_position_by_task = {}
    band_position_source_by_task = {}

    def _append_band_position_rows(_rows, _session_data, *, _subject, _session, _source):
        _session_data = add_fixed_accuracy_repetition_band(_session_data)
        _required = {
            "response_repeat_window_fraction",
            "fixed_accuracy_repeat_low",
            "fixed_accuracy_repeat_high",
        }
        if not _required.issubset(_session_data.columns):
            return

        _band_data = _session_data[list(_required)].dropna()
        if _band_data.empty:
            return

        _observed = _band_data["response_repeat_window_fraction"]
        _below = _observed < _band_data["fixed_accuracy_repeat_low"]
        _above = _observed > _band_data["fixed_accuracy_repeat_high"]
        _in = ~(_below | _above)
        for _position, _mask in [
            ("Below", _below),
            ("In", _in),
            ("Above", _above),
        ]:
            _rows.append(
                {
                    "subject": str(_subject),
                    "session": str(_session),
                    "position": _position,
                    "proportion": float(_mask.mean()),
                    "source": _source,
                }
            )

    def _glm_pdf_for_task(_task):
        _simulated_df = autocorrelograms_by_task.get(_task, {}).get("glm", {}).get(
            "simulated_df",
            pd.DataFrame(),
        )
        if hasattr(_simulated_df, "to_pandas"):
            return _simulated_df.to_pandas()
        return pd.DataFrame(_simulated_df)

    for _task in task_names:
        _session_rows = []
        _source_session_rows = []
        _glm_pdf = _glm_pdf_for_task(_task)
        _glm_trial_col = "trial_index" if "trial_index" in _glm_pdf.columns else "trial_idx"
        _session_index = (
            plot_dfs[_task]
            .select(["subject", "session"])
            .unique()
            .sort(["subject", "session"])
            .to_pandas()
        )
        for _subject, _session in _session_index.itertuples(index=False, name=None):
            try:
                _session_data = build_session_repetition_data(
                    plot_dfs[_task],
                    subject=_subject,
                    session=_session,
                    adapter=adapters[_task],
                    window=20,
                )
            except ValueError:
                continue

            _session_row_start = len(_session_rows)
            _append_band_position_rows(
                _session_rows,
                _session_data,
                _subject=_subject,
                _session=_session,
                _source="Data",
            )
            _data_source_rows = _session_rows[_session_row_start:]

            if not {"subject", "session", _glm_trial_col, "response"}.issubset(_glm_pdf.columns):
                continue
            _glm_session = (
                _glm_pdf[
                    (_glm_pdf["subject"].astype(str) == f"{_subject}__closed_loop_000")
                    & (_glm_pdf["session"].astype(str) == str(_session))
                ]
                .sort_values(_glm_trial_col)
                .copy()
            )
            if _glm_session.empty:
                continue

            _n_trials = min(len(_glm_session), len(_session_data))
            if _n_trials <= 0:
                continue
            _glm_session = _glm_session.iloc[:_n_trials].copy()
            if _glm_trial_col == "trial_index":
                _glm_session = _glm_session.rename(columns={"trial_index": "trial"})
            _glm_session["subject"] = str(_subject)
            _glm_session["session"] = str(_session)
            _glm_session["stimulus"] = _session_data["stimulus"].to_numpy()[:_n_trials]

            try:
                _glm_session_data = build_session_repetition_data(
                    pl.from_pandas(_glm_session),
                    subject=_subject,
                    session=_session,
                    adapter=adapters[_task],
                    window=20,
                )
            except ValueError:
                continue
            _glm_source_rows = []
            _append_band_position_rows(
                _glm_source_rows,
                _glm_session_data,
                _subject=_subject,
                _session=_session,
                _source="GLM closed-loop",
            )
            if _glm_source_rows:
                _source_session_rows.extend(_data_source_rows)
                _source_session_rows.extend(_glm_source_rows)

        if _session_rows:
            _subject_summary = (
                pd.DataFrame(_session_rows).drop(columns=["source"])
                .groupby(["subject", "position"], as_index=False, observed=True)["proportion"]
                .mean()
            )
            _subject_summary["position"] = pd.Categorical(
                _subject_summary["position"],
                categories=band_position_order,
                ordered=True,
            )
            _subject_summary = _subject_summary.sort_values(["position", "subject"])
            _subject_summary["task"] = _task
            _subject_summary["task_label"] = _task_labels.get(_task, _task)
        else:
            _subject_summary = pd.DataFrame(
                columns=["subject", "position", "proportion", "task", "task_label"]
            )
        band_position_by_task[_task] = _subject_summary

        if _source_session_rows:
            _source_subject_summary = (
                pd.DataFrame(_source_session_rows)
                .groupby(["subject", "source", "position"], as_index=False, observed=True)["proportion"]
                .mean()
            )
            _source_subject_summary["position"] = pd.Categorical(
                _source_subject_summary["position"],
                categories=band_position_order,
                ordered=True,
            )
            _source_subject_summary["source"] = pd.Categorical(
                _source_subject_summary["source"],
                categories=band_position_source_order,
                ordered=True,
            )
            _source_subject_summary = _source_subject_summary.sort_values(
                ["position", "source", "subject"]
            )
            _source_subject_summary["task"] = _task
            _source_subject_summary["task_label"] = _task_labels.get(_task, _task)
        else:
            _source_subject_summary = pd.DataFrame(
                columns=["subject", "source", "position", "proportion", "task", "task_label"]
            )
        band_position_source_by_task[_task] = _source_subject_summary
    return (
        band_position_by_task,
        band_position_order,
        band_position_source_by_task,
        band_position_source_order,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    band_position_by_task,
    band_position_order,
    boxplot_STYLE,
    fig_size,
    format,
    np,
    path_panels,
    pd,
    plt,
    sns,
    ttest_1samp,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    fixed_band_position_2ADC = plt.gca()

    _plot_df = band_position_by_task["2AFC_delay"]
    sns.boxplot(
        data=_plot_df,
        x="position",
        y="proportion",
        order=band_position_order,
        color="tab:blue",
        ax=fixed_band_position_2ADC,
        **boxplot_STYLE,
    )
    fixed_band_position_2ADC.set_xlabel("")
    fixed_band_position_2ADC.set_ylabel("Proportion of trials")
    fixed_band_position_2ADC.set_ylim(0, 1)
    for _x_idx, _position in enumerate(band_position_order):
        _values = pd.to_numeric(
            _plot_df.loc[_plot_df["position"] == _position, "proportion"],
            errors="coerce",
        ).dropna()
        if len(_values) < 2:
            continue
        _pvalue = float(ttest_1samp(_values.to_numpy(dtype=float), popmean=0.0).pvalue)
        if not np.isfinite(_pvalue) or _pvalue >= 0.05:
            continue
        _stars = "***" if _pvalue < 0.001 else "**" if _pvalue < 0.01 else "*"
        _text_y = min(0.96, max(0.06, float(_values.max()) + 0.04))
        fixed_band_position_2ADC.text(_x_idx, _text_y, _stars, ha="center", va="bottom")
    fixed_band_position_2ADC.figure.savefig(
        (path_panels / "2ADC_fixed_accuracy_band_position").with_suffix(f".{format}")
    )
    fixed_band_position_2ADC
    return


@app.cell
def _(
    Annotator,
    band_position_order,
    band_position_source_by_task,
    band_position_source_order,
    boxplot_STYLE,
    fig_size,
    format,
    path_panels,
    pd,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    fixed_band_position_2ADC_data_glm = plt.gca()

    _plot_df = band_position_source_by_task["2AFC_delay"]
    _source_palette = {"Data": "tab:blue", "GLM closed-loop": "tab:gray"}
    sns.boxplot(
        data=_plot_df,
        x="position",
        y="proportion",
        hue="source",
        order=band_position_order,
        hue_order=band_position_source_order,
        palette=_source_palette,
        ax=fixed_band_position_2ADC_data_glm,
        **boxplot_STYLE,
    )
    fixed_band_position_2ADC_data_glm.set_xlabel("")
    fixed_band_position_2ADC_data_glm.set_ylabel("Proportion of trials")
    fixed_band_position_2ADC_data_glm.set_ylim(0, 1)
    _paired_frames = []
    _available_pairs = []
    for _position in band_position_order:
        _sub = _plot_df[_plot_df["position"] == _position]
        _paired = _sub.pivot_table(
            values="proportion",
            index="subject",
            columns="source",
            aggfunc="first",
        )
        if not all(_source in _paired.columns for _source in band_position_source_order):
            continue
        _paired = _paired.dropna(subset=list(band_position_source_order))
        if len(_paired) < 2:
            continue
        _paired_subjects = set(_paired.index.astype(str))
        _paired_sub = _sub[_sub["subject"].astype(str).isin(_paired_subjects)].copy()
        _paired_frames.append(_paired_sub)
        _available_pairs.append(
            ((_position, band_position_source_order[0]), (_position, band_position_source_order[1]))
        )
    if _available_pairs and _paired_frames:
        _annotator = Annotator(
            fixed_band_position_2ADC_data_glm,
            _available_pairs,
            data=pd.concat(_paired_frames, ignore_index=True),
            x="position",
            y="proportion",
            hue="source",
            order=band_position_order,
            hue_order=band_position_source_order,
        )
        _annotator.configure(
            test="t-test_paired",
            text_format="star",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()
    _legend = fixed_band_position_2ADC_data_glm.get_legend()
    if _legend is not None:
        _legend.set_title(None)
        _legend.set_frame_on(False)
    fixed_band_position_2ADC_data_glm.figure.savefig(
        (path_panels / "2ADC_fixed_accuracy_band_position_data_glm").with_suffix(f".{format}")
    )
    fixed_band_position_2ADC_data_glm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    band_position_by_task,
    band_position_order,
    boxplot_STYLE,
    fig_size,
    format,
    np,
    path_panels,
    pd,
    plt,
    sns,
    ttest_1samp,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    fixed_band_position_2AFC = plt.gca()

    _plot_df = band_position_by_task["2AFC"]
    sns.boxplot(
        data=_plot_df,
        x="position",
        y="proportion",
        order=band_position_order,
        color="tab:blue",
        ax=fixed_band_position_2AFC,
        **boxplot_STYLE,
    )
    fixed_band_position_2AFC.set_xlabel("")
    fixed_band_position_2AFC.set_ylabel("Proportion of trials")
    fixed_band_position_2AFC.set_ylim(0, 1)
    for _x_idx, _position in enumerate(band_position_order):
        _values = pd.to_numeric(
            _plot_df.loc[_plot_df["position"] == _position, "proportion"],
            errors="coerce",
        ).dropna()
        if len(_values) < 2:
            continue
        _pvalue = float(ttest_1samp(_values.to_numpy(dtype=float), popmean=0.0).pvalue)
        if not np.isfinite(_pvalue) or _pvalue >= 0.05:
            continue
        _stars = "***" if _pvalue < 0.001 else "**" if _pvalue < 0.01 else "*"
        _text_y = min(0.96, max(0.06, float(_values.max()) + 0.04))
        fixed_band_position_2AFC.text(_x_idx, _text_y, _stars, ha="center", va="bottom")
    fixed_band_position_2AFC.figure.savefig(
        (path_panels / "2AFC_fixed_accuracy_band_position").with_suffix(f".{format}")
    )
    fixed_band_position_2AFC
    return


@app.cell
def _(
    Annotator,
    band_position_order,
    band_position_source_by_task,
    band_position_source_order,
    boxplot_STYLE,
    fig_size,
    format,
    path_panels,
    pd,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    fixed_band_position_2AFC_data_glm = plt.gca()

    _plot_df = band_position_source_by_task["2AFC"]
    _source_palette = {"Data": "tab:blue", "GLM closed-loop": "tab:gray"}
    sns.boxplot(
        data=_plot_df,
        x="position",
        y="proportion",
        hue="source",
        order=band_position_order,
        hue_order=band_position_source_order,
        palette=_source_palette,
        ax=fixed_band_position_2AFC_data_glm,
        **boxplot_STYLE,
    )
    fixed_band_position_2AFC_data_glm.set_xlabel("")
    fixed_band_position_2AFC_data_glm.set_ylabel("Proportion of trials")
    fixed_band_position_2AFC_data_glm.set_ylim(0, 1)
    _paired_frames = []
    _available_pairs = []
    for _position in band_position_order:
        _sub = _plot_df[_plot_df["position"] == _position]
        _paired = _sub.pivot_table(
            values="proportion",
            index="subject",
            columns="source",
            aggfunc="first",
        )
        _paired_subjects = set(_paired.index.astype(str))
        _paired_frames.append(_sub[_sub["subject"].astype(str).isin(_paired_subjects)].copy())
        _available_pairs.append(((_position, band_position_source_order[0]), (_position, band_position_source_order[1])))

    _annotator = Annotator(
        fixed_band_position_2AFC_data_glm,
        _available_pairs,
        data=pd.concat(_paired_frames, ignore_index=True),
        x="position",
        y="proportion",
        hue="source",
        order=band_position_order,
        hue_order=band_position_source_order,
    )
    _annotator.configure(
        test="t-test_paired",
        text_format="star",
        line_height=0,
        verbose=False,
    ).apply_and_annotate()
    _legend = fixed_band_position_2AFC_data_glm.get_legend()
    if _legend is not None:
        _legend.set_title(None)
        _legend.set_frame_on(False)
    fixed_band_position_2AFC_data_glm.figure.savefig((path_panels / "2ADC_fixed_accuracy_band_position_data_glm").with_suffix(f".{format}"))
    fixed_band_position_2AFC_data_glm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell
def _(
    band_position_by_task,
    band_position_order,
    boxplot_STYLE,
    fig_size,
    format,
    np,
    path_panels,
    pd,
    plt,
    sns,
    ttest_1samp,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    fixed_band_position_MCDR = plt.gca()

    _plot_df = band_position_by_task["MCDR"]
    sns.boxplot(
        data=_plot_df,
        x="position",
        y="proportion",
        order=band_position_order,
        color="tab:blue",
        ax=fixed_band_position_MCDR,
        **boxplot_STYLE,
    )
    fixed_band_position_MCDR.set_xlabel("")
    fixed_band_position_MCDR.set_ylabel("Proportion of trials")
    fixed_band_position_MCDR.set_ylim(0, 1)
    for _x_idx, _position in enumerate(band_position_order):
        _values = pd.to_numeric(
            _plot_df.loc[_plot_df["position"] == _position, "proportion"],
            errors="coerce",
        ).dropna()
        if len(_values) < 2:
            continue
        _pvalue = float(ttest_1samp(_values.to_numpy(dtype=float), popmean=0.0).pvalue)
        if not np.isfinite(_pvalue) or _pvalue >= 0.05:
            continue
        _stars = "***" if _pvalue < 0.001 else "**" if _pvalue < 0.01 else "*"
        _text_y = min(0.96, max(0.06, float(_values.max()) + 0.04))
        fixed_band_position_MCDR.text(_x_idx, _text_y, _stars, ha="center", va="bottom")
    fixed_band_position_MCDR.figure.savefig(
        (path_panels / "MCDR_fixed_accuracy_band_position").with_suffix(f".{format}")
    )
    fixed_band_position_MCDR
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
    chunk_hist_stat = "count"  # Use "probability" for relative frequencies.
    chunk_hist_ylabel = {"count": "Count", "probability": "Frequency"}[chunk_hist_stat]
    transition_palette = {"repeating": "tab:brown", "alternating": "tab:purple"}
    transition_drug_palette = transition_palette
    outcome_palette = {"Correct": "tab:green", "Incorrect": "tab:red"}
    outcome_source_style = {
        "Data": "",
        "Indep.": (2, 2),
        "GLM": (4, 1),
    }
    source_palette = {
        "Data": "tab:blue",
        "GLM": "tab:gray",
        "Indep.": "tab:blue",
    }
    source_style = {
        "Data": "",
        "Indep.": (2, 2),
        "GLM": "",
    }
    return (
        chunk_hist_stat,
        chunk_hist_ylabel,
        outcome_palette,
        outcome_source_style,
        source_palette,
        source_style,
        transition_drug_palette,
        transition_palette,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Process data
    """)
    return


@app.cell
def _(
    autocorrelograms_by_task,
    build_outcome_streak_plot_data,
    build_repetition_chunk_plot_data,
    build_transition_chunk_plot_data,
    chunk_hist_stat,
    outcome_palette,
    plot_dfs,
    task_names,
    transition_palette,
):
    transition_chunk_lengths_by_task, transition_chunk_plot_data, _transition_palette, transition_repeat_probabilities = (
        build_transition_chunk_plot_data(
            plot_dfs,
            task_names,
            stat=chunk_hist_stat,
            transition_palette=transition_palette,
        )
    )
    _glm_simulated_dfs = {
        _task: autocorrelograms_by_task[_task]["glm"]["simulated_df"]
        for _task in task_names
    }
    repetition_chunk_plot_data = build_repetition_chunk_plot_data(
        plot_dfs,
        task_names,
        glm_simulated_dfs=_glm_simulated_dfs,
        stat=chunk_hist_stat,
    )
    _outcome_streak_lengths_by_task, outcome_streak_plot_data, _outcome_palette, _outcome_streak_probabilities = (
        build_outcome_streak_plot_data(
            plot_dfs,
            task_names,
            glm_simulated_dfs=_glm_simulated_dfs,
            stat=chunk_hist_stat,
            outcome_palette=outcome_palette,
        )
    )
    for _plot_data in (
        transition_chunk_plot_data,
        repetition_chunk_plot_data,
        outcome_streak_plot_data,
    ):
        if "source" in _plot_data.columns:
            _plot_data["source"] = _plot_data["source"].replace(
                {"Independent choices": "Indep."}
            )
    return (
        outcome_streak_plot_data,
        repetition_chunk_plot_data,
        transition_chunk_lengths_by_task,
        transition_repeat_probabilities,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    axd,
    chunk_hist_ylabel,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    repetition_chunk_plot_data,
    sns,
    source_palette,
    source_style,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    consec_rep_2ADC = plt.gca() if not mount_figure else axd["hist_repeat_2ADC"]
    consec_rep_2ADC.clear()

    sns.lineplot(
        data=repetition_chunk_plot_data[repetition_chunk_plot_data["task_label"] == "2ADC"],
        x="chunk_length",
        y="weight",
        hue="source",
        style="source",
        palette=source_palette,
        dashes=source_style,
        markers=False,
        errorbar=None,
        ax=consec_rep_2ADC,
    )
    consec_rep_2ADC.set_xlim(0, 30)
    consec_rep_2ADC.set_ylim(1, 1e4)
    consec_rep_2ADC.set_yscale("log")
    # consec_rep_2ADC.set_title("2ADC")
    consec_rep_2ADC.set_xlabel("Consecutive repeated choices")
    consec_rep_2ADC.set_ylabel(chunk_hist_ylabel)
    _handles, _labels = consec_rep_2ADC.get_legend_handles_labels()
    consec_rep_2ADC.legend(
        [h for h, l in zip(_handles, _labels) if l not in ["transition", "source", "repeating"]],
        [l for l in _labels if l not in ["transition", "source", "repeating"]],
        frameon=False,
    )
    if not mount_figure:
        plt.savefig((path_panels / "2ADC_choice_transition_chunks").with_suffix(f".{format}"))
    consec_rep_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    axd,
    chunk_hist_ylabel,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    repetition_chunk_plot_data,
    sns,
    source_palette,
    source_style,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    consec_rep_2AFC = plt.gca() if not mount_figure else axd["hist_repeat_2AFC"]
    consec_rep_2AFC.clear()

    sns.lineplot(
        data=repetition_chunk_plot_data[repetition_chunk_plot_data["task_label"] == "2AFC"],
        x="chunk_length",
        y="weight",
         hue="source",
        style="source",
        palette=source_palette,
        dashes=source_style,
        markers=False,
        errorbar=None,
        ax=consec_rep_2AFC,
    )
    consec_rep_2AFC.set_xlim(0, 30)
    consec_rep_2AFC.set_ylim(1, 1e4)
    consec_rep_2AFC.set_yscale("log")
    # consec_rep_2AFC.set_title("2AFC")
    consec_rep_2AFC.set_xlabel("Consecutive repeated choices")
    consec_rep_2AFC.set_ylabel(chunk_hist_ylabel)
    _handles, _labels = consec_rep_2AFC.get_legend_handles_labels()

    consec_rep_2AFC.legend(
        [h for h, l in zip(_handles, _labels) if l not in ["transition", "source", "repeating"]],
        [l for l in _labels if l not in ["transition", "source", "repeating"]],
        frameon=False,
    )
    if not mount_figure:
        plt.savefig((path_panels / "2AFC_choice_transition_chunks").with_suffix(f".{format}"))
    consec_rep_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell
def _(
    chunk_hist_ylabel,
    fig_size,
    format,
    path_panels,
    plt,
    repetition_chunk_plot_data,
    sns,
    transition_palette,
):
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    consec_rep_MCDR = plt.gca()
    sns.lineplot(
        data=repetition_chunk_plot_data[repetition_chunk_plot_data["task_label"] == "MCDR"],
        x="chunk_length",
        y="weight",
        hue="transition",
        style="source",
        palette=transition_palette,
        dashes={"Data": "", "Indep.": (2, 2), "GLM": (4, 1)},
        markers=False,
        errorbar=None,
        ax=consec_rep_MCDR,
    )
    consec_rep_MCDR.set_xlim(0, 30)
    consec_rep_MCDR.set_ylim(1, 1e4)
    consec_rep_MCDR.set_yscale("log")
    consec_rep_MCDR.set_title("MCDR")
    consec_rep_MCDR.set_xlabel("Consecutive repeated choices")
    consec_rep_MCDR.set_ylabel(chunk_hist_ylabel)
    consec_rep_MCDR.legend(frameon=False)
    _handles, _labels = consec_rep_MCDR.get_legend_handles_labels()
    consec_rep_MCDR.legend(
        [h for h, l in zip(_handles, _labels) if l not in ["transition", "source", "repeating"]],
        [l for l in _labels if l not in ["transition", "source", "repeating"]],
        frameon=False,
    )
    plt.savefig((path_panels / "MCDR_choice_transition_chunks").with_suffix(f".{format}"))
    consec_rep_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histograms of correct/incorrect streaks
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
    axd,
    chunk_hist_ylabel,
    fig_size,
    format,
    mount_figure,
    outcome_palette,
    outcome_source_style,
    outcome_streak_plot_data,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    correct_streak_2ADC = plt.gca() if not mount_figure else axd["hist_correct_2ADC"]

    sns.lineplot(
        data=outcome_streak_plot_data[outcome_streak_plot_data["task_label"] == "2ADC"],
        x="chunk_length",
        y="weight",
        hue="outcome",
        hue_order=["Correct", "Incorrect"],
        style="source",
        style_order=["Data", "Indep.", "GLM"],
        palette=outcome_palette,
        dashes=outcome_source_style,
        markers=False,
        errorbar=None,
        ax=correct_streak_2ADC,
    )
    correct_streak_2ADC.set_xlim(0, 30)
    correct_streak_2ADC.set_ylim(1, 1e3)
    correct_streak_2ADC.set_yscale("log")
    correct_streak_2ADC.set_xlabel("Consecutive correct/incorrect trials")
    correct_streak_2ADC.set_ylabel(chunk_hist_ylabel)
    _legend = correct_streak_2ADC.get_legend()
    if _legend is not None:
        _legend.set_title(None)
        _legend.set_frame_on(False)
    correct_streak_2ADC.figure.savefig(
        (path_panels / "2ADC_correct_incorrect_streaks").with_suffix(f".{format}")
    )
    correct_streak_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    axd,
    chunk_hist_ylabel,
    fig_size,
    format,
    mount_figure,
    outcome_palette,
    outcome_source_style,
    outcome_streak_plot_data,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    correct_streak_2AFC = plt.gca() if not mount_figure else axd["hist_correct_2AFC"]

    sns.lineplot(
        data=outcome_streak_plot_data[outcome_streak_plot_data["task_label"] == "2AFC"],
        x="chunk_length",
        y="weight",
        hue="outcome",
        hue_order=["Correct", "Incorrect"],
        style="source",
        style_order=["Data", "Indep.", "GLM"],
        palette=outcome_palette,
        dashes=outcome_source_style,
        markers=False,
        errorbar=None,
        ax=correct_streak_2AFC,
    )
    correct_streak_2AFC.set_xlim(0, 30)
    correct_streak_2AFC.set_ylim(1, 1e3)
    correct_streak_2AFC.set_yscale("log")
    correct_streak_2AFC.set_xlabel("Consecutive correct/incorrect trials")
    correct_streak_2AFC.set_ylabel(chunk_hist_ylabel)
    _legend = correct_streak_2AFC.get_legend()
    if _legend is not None:
        _legend.set_title(None)
        _legend.set_frame_on(False)
    correct_streak_2AFC.figure.savefig(
        (path_panels / "2AFC_correct_incorrect_streaks").with_suffix(f".{format}")
    )
    correct_streak_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell
def _(
    chunk_hist_ylabel,
    fig_size,
    format,
    outcome_palette,
    outcome_source_style,
    outcome_streak_plot_data,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    correct_streak_MCDR = plt.gca()

    sns.lineplot(
        data=outcome_streak_plot_data[outcome_streak_plot_data["task_label"] == "MCDR"],
        x="chunk_length",
        y="weight",
        hue="outcome",
        hue_order=["Correct", "Incorrect"],
        style="source",
        style_order=["Data", "Indep.", "GLM"],
        palette=outcome_palette,
        dashes=outcome_source_style,
        markers=False,
        errorbar=None,
        ax=correct_streak_MCDR,
    )
    correct_streak_MCDR.set_xlim(0, 30)
    correct_streak_MCDR.set_ylim(1, 1e3)
    correct_streak_MCDR.set_yscale("log")
    correct_streak_MCDR.set_xlabel("Consecutive correct/incorrect trials")
    correct_streak_MCDR.set_ylabel(chunk_hist_ylabel)
    _legend = correct_streak_MCDR.get_legend()
    if _legend is not None:
        _legend.set_title(None)
        _legend.set_frame_on(False)
    correct_streak_MCDR.figure.savefig(
        (path_panels / "MCDR_correct_incorrect_streaks").with_suffix(f".{format}")
    )
    correct_streak_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histograms repetition alternation splitting Drug/Saline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Process data
    """)
    return


@app.cell
def _(
    build_transition_chunk_drug_plot_data,
    get_adapter,
    transition_drug_palette,
):
    transition_chunk_drug_plot_data, _transition_drug_palette = (
        build_transition_chunk_drug_plot_data(
            get_adapter,
            transition_palette=transition_drug_palette,
        )
    )
    if "source" in transition_chunk_drug_plot_data.columns:
        transition_chunk_drug_plot_data["source"] = transition_chunk_drug_plot_data[
            "source"
        ].replace({"Independent choices": "Indep."})
    return (transition_chunk_drug_plot_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2ADC
    """)
    return


@app.cell
def _(
    fig_size,
    format,
    path_panels,
    plt,
    sns,
    transition_chunk_drug_plot_data,
    transition_drug_palette,
):
    fig_consec_rep_drug_2ADC, (consec_rep_2ADC_saline, consec_rep_2ADC_Drug) = plt.subplots(
        1, 2, figsize=fig_size(1,2), constrained_layout=True, sharex=True, sharey=True
    )
    _data = transition_chunk_drug_plot_data[transition_chunk_drug_plot_data["task_label"] == "2ADC"]
    for _ax, _drug_label in zip(
        [consec_rep_2ADC_saline, consec_rep_2ADC_Drug],
        ["No drug", "Drug"],
        strict=False,
    ):
        sns.lineplot(
            data=_data[_data["drug_label"] == _drug_label],
            x="chunk_length",
            y="weight",
            hue="transition",
            style="source",
            palette=transition_drug_palette,
            dashes={"Data": "", "Indep.": (2, 2)},
            errorbar=None,
            ax=_ax,
        )
        _ax.set_title(f"2ADC {_drug_label}")
        _ax.set_xlim(0, 30)
        _ax.set_ylim(1e-4, 1)
        _ax.set_yscale("log")
        _ax.set_xlabel("Consecutive choices")
        _ax.legend(frameon=False)
        _handles, _labels = _ax.get_legend_handles_labels()
        _ax.legend(
            [h for h, l in zip(_handles, _labels) if l not in ["transition", "source"]],
            [l for l in _labels if l not in ["transition", "source"]],
            frameon=False,
        )
    consec_rep_2ADC_saline.set_ylabel("Frequency")
    consec_rep_2ADC_Drug.set_ylabel("")
    plt.savefig((path_panels / "2ADC_drug_choice_transition_chunks").with_suffix(f".{format}"))
    fig_consec_rep_drug_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2AFC
    """)
    return


@app.cell
def _(
    fig_size,
    format,
    path_panels,
    plt,
    sns,
    transition_chunk_drug_plot_data,
    transition_drug_palette,
):
    fig_consec_rep_drug_2AFC, (consec_rep_2AFC_saline, consec_rep_2AFC_Drug) = plt.subplots(
        1, 2, figsize=fig_size(1,2), constrained_layout=True, sharex=True, sharey=True
    )
    _data = transition_chunk_drug_plot_data[transition_chunk_drug_plot_data["task_label"] == "2AFC"]
    for _ax, _drug_label in zip(
        [consec_rep_2AFC_saline, consec_rep_2AFC_Drug],
        ["No drug", "Drug"],
        strict=False,
    ):
        sns.lineplot(
            data=_data[_data["drug_label"] == _drug_label],
            x="chunk_length",
            y="weight",
            hue="transition",
            style="source",
            palette=transition_drug_palette,
            dashes={"Data": "", "Indep.": (2, 2)},
            errorbar=None,
            ax=_ax,
        )
        _ax.set_title(f"2AFC {_drug_label}")
        _ax.set_xlim(0, 30)
        _ax.set_ylim(1e-4, 1)
        _ax.set_yscale("log")
        _ax.set_xlabel("Consecutive choices")
        _ax.legend(frameon=False)
        _handles, _labels = _ax.get_legend_handles_labels()
        _ax.legend(
            [h for h, l in zip(_handles, _labels) if l not in ["transition", "source"]],
            [l for l in _labels if l not in ["transition", "source"]],
            frameon=False,
        )

    consec_rep_2AFC_saline.set_ylabel("Frequency")
    consec_rep_2AFC_Drug.set_ylabel("")
    plt.savefig((path_panels / "2AFC_drug_choice_transition_chunks").with_suffix(f".{format}"))
    fig_consec_rep_drug_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MCDR
    """)
    return


@app.cell
def _(
    fig_size,
    format,
    path_panels,
    plt,
    sns,
    transition_chunk_drug_plot_data,
    transition_drug_palette,
):
    fig_consec_rep_drug_MCDR, (consec_rep_MCDR_saline, consec_rep_2ADC_drug) = plt.subplots(
        1, 2, figsize=fig_size(1,2), constrained_layout=True, sharex=True, sharey=True
    )
    _data = transition_chunk_drug_plot_data[transition_chunk_drug_plot_data["task_label"] == "MCDR"]
    for _ax, _drug_label in zip(
        [consec_rep_MCDR_saline, consec_rep_2ADC_drug],
        ["No drug", "Drug"],
        strict=False,
    ):
        sns.lineplot(
            data=_data[_data["drug_label"] == _drug_label],
            x="chunk_length",
            y="weight",
            hue="transition",
            style="source",
            palette=transition_drug_palette,
            dashes={"Data": "", "Indep.": (2, 2)},
            errorbar=None,
            ax=_ax,
        )
        _ax.set_title(f"MCDR {_drug_label}")
        _ax.set_xlim(0, 30)
        _ax.set_ylim(1e-4, 1)
        _ax.set_yscale("log")
        _ax.set_xlabel("Consecutive choices")
        _ax.legend(frameon=False)
        _handles, _labels = _ax.get_legend_handles_labels()
        _ax.legend(
            [h for h, l in zip(_handles, _labels) if l not in ["transition", "source"]],
            [l for l in _labels if l not in ["transition", "source"]],
            frameon=False,
        )
    consec_rep_MCDR_saline.set_ylabel("Frequency")
    consec_rep_2ADC_drug.set_ylabel("")
    plt.savefig((path_panels / "MCDR_drug_choice_transition_chunks").with_suffix(f".{format}"))
    fig_consec_rep_drug_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variance boxplots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Process data
    """)
    return


@app.cell
def _(build_repetition_variance_by_drug_task, get_adapter):
    repetition_variance_by_drug_task_long, stimulus_repeat_binomial_variance_by_task = (
        build_repetition_variance_by_drug_task(
            get_adapter,
            window=20,
        )
    )
    return (
        repetition_variance_by_drug_task_long,
        stimulus_repeat_binomial_variance_by_task,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    boxplot_STYLE,
    fig_size,
    format,
    path_panels,
    plt,
    repetition_variance_by_drug_task_long,
    sns,
    stimulus_repeat_binomial_variance_by_task,
):
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    ax_drug_repetition_variance_2ADC = plt.gca()
    sns.boxplot(
        data=repetition_variance_by_drug_task_long[
            repetition_variance_by_drug_task_long["task_label"] == "2ADC"
        ],
        x="signal",
        y="variance",
        hue="drug_label",
        order=["Repetition", "Stimulus"],
        hue_order=["Saline", "Drug"],
        palette={"Saline": "tab:gray", "Drug": "tab:pink"},
        ax=ax_drug_repetition_variance_2ADC,
        **boxplot_STYLE,
    )
    ax_drug_repetition_variance_2ADC.axhline(
        stimulus_repeat_binomial_variance_by_task["2ADC"],
        color="tab:blue",
        linestyle="--",
    )
    ax_drug_repetition_variance_2ADC.set_title("2ADC")
    ax_drug_repetition_variance_2ADC.set_xlabel("")
    ax_drug_repetition_variance_2ADC.set_ylabel("Variance of running fraction")
    ax_drug_repetition_variance_2ADC.legend(frameon=False)
    plt.savefig((path_panels / "2ADC_drug_repetition_variance").with_suffix(f".{format}"))
    ax_drug_repetition_variance_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    boxplot_STYLE,
    fig_size,
    format,
    path_panels,
    plt,
    repetition_variance_by_drug_task_long,
    sns,
    stimulus_repeat_binomial_variance_by_task,
):
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    ax_drug_repetition_variance_2AFC = plt.gca()
    sns.boxplot(
        data=repetition_variance_by_drug_task_long[
            repetition_variance_by_drug_task_long["task_label"] == "2AFC"
        ],
        x="signal",
        y="variance",
        hue="drug_label",
        order=["Repetition", "Stimulus"],
        hue_order=["Saline", "Drug"],
        palette={"Saline": "tab:gray", "Drug": "tab:pink"},
        ax=ax_drug_repetition_variance_2AFC,
        **boxplot_STYLE,
    )
    ax_drug_repetition_variance_2AFC.axhline(
        stimulus_repeat_binomial_variance_by_task["2AFC"],
        color="tab:blue",
        linestyle="--",
    )
    ax_drug_repetition_variance_2AFC.set_title("2AFC")
    ax_drug_repetition_variance_2AFC.set_xlabel("")
    ax_drug_repetition_variance_2AFC.set_ylabel("Variance of running fraction")
    ax_drug_repetition_variance_2AFC.legend(frameon=False)
    plt.savefig((path_panels / "2AFC_drug_repetition_variance").with_suffix(f".{format}"))
    ax_drug_repetition_variance_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MCDR
    """)
    return


@app.cell
def _(
    boxplot_STYLE,
    fig_size,
    format,
    path_panels,
    plt,
    repetition_variance_by_drug_task_long,
    sns,
    stimulus_repeat_binomial_variance_by_task,
):
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    ax_drug_repetition_variance_MCDR = plt.gca()
    sns.boxplot(
        data=repetition_variance_by_drug_task_long[
            repetition_variance_by_drug_task_long["task_label"] == "MCDR"
        ],
        x="signal",
        y="variance",
        hue="drug_label",
        order=["Repetition", "Stimulus"],
        hue_order=["Saline", "Drug"],
        palette={"Saline": "tab:gray", "Drug": "tab:pink"},
        ax=ax_drug_repetition_variance_MCDR,
        **boxplot_STYLE,
    )
    ax_drug_repetition_variance_MCDR.axhline(
        stimulus_repeat_binomial_variance_by_task["MCDR"],
        color="tab:blue",
        linestyle="--",
    )
    ax_drug_repetition_variance_MCDR.set_title("MCDR")
    ax_drug_repetition_variance_MCDR.set_xlabel("")
    ax_drug_repetition_variance_MCDR.set_ylabel("Variance of running fraction")
    ax_drug_repetition_variance_MCDR.legend(frameon=False)
    plt.savefig((path_panels / "MCDR_drug_repetition_variance").with_suffix(f".{format}"))
    ax_drug_repetition_variance_MCDR
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
    transition_chunk_lengths_by_task,
    transition_repeat_probabilities,
):
    from scipy.stats import chi2

    _task_order = ["2ADC", "2AFC", "MCDR"]
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
    for _task_label in _task_order:
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
def _(chi2, np, pd, transition_chunk_lengths_by_task):
    _task_order = ["2ADC", "2AFC", "MCDR"]
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
    for _task_label in _task_order:
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
                    (
                        (
                            (_observed_bins[_valid_bins] - _expected_bins[_valid_bins])
                            ** 2
                        )
                        / _expected_bins[_valid_bins]
                    ).sum()
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
def _(fig_size):
    fig_size(1)
    return


@app.cell
def _():
    ["a", "a", "b", "b"],
    ["hist_repeat_2ADC", "hist_repeat_2ADC", "hist_repeat_2AFC", "hist_repeat_2AFC"],
    ["hist_repeat_2ADC", "hist_repeat_2ADC", "hist_repeat_2AFC", "hist_repeat_2AFC"],
    # ["a", "a", "b", "b"],
    ["e", "m", "h", "q"],
    ["i", "i", "k", "k"],
    ["l", "l", "m", "m"],
    return


@app.cell
def _(axd, fig, format, mount_figure, project_path):
    if mount_figure:
        for _name, _ax in axd.items():
            _ax.set_title("")
            _ax.set_xlabel("")
            if not _name.startswith("_model_comparison_parent"):
                _ax.set_ylabel("")
            _legend = _ax.get_legend()
            if _legend is not None and not _ax in [axd["hist_repeat_2ADC"], axd["i"]]:
                _legend.remove()

        axd["a"].set_ylabel("Running fraction")
        axd["a"].set_title("2ADC")
        axd["a"].set_xlabel("Trial")

        axd["a"].legend(
            *axd["a"].get_legend_handles_labels(),
            handlelength=0.4,
            ncol=2,
            frameon=False,
        )

        axd["b"].set_title("2AFC")
        axd["b"].set_xlabel("Trial")

        # axd["single_sess_acc_2ADC"].set_ylabel("Running Accuracy")
        # axd["single_sess_acc_2ADC"].set_xlabel("Trial")

        # axd["single_sess_acc_2AFC"].set_xlabel("Trial")

        axd["hist_correct_2ADC"].set_xlabel("Corr. Streak length")
        axd["hist_correct_2ADC"].set_ylabel("Count")

        # axd["hist_repeat_2ADC"].set_ylabel("Count")
        axd["hist_repeat_2ADC"].set_xlabel("Rep. Streak length")
        axd["hist_repeat_2ADC"].legend(
            *axd["hist_repeat_2ADC"].get_legend_handles_labels(),
            handlelength=1,
            frameon=False,
        )


        # axd["hist_correct_2ADC"].legend(
        #     axd["hist_correct_2ADC"].get_legend_handles_labels()[:3],
        #     handlelength=1,
        #     frameon=False,
        #     title = ""
        # )

        axd["hist_repeat_2AFC"].set_xlabel("Rep. Streak length")

        axd["hist_correct_2AFC"].set_xlabel("Corr. Streak length")

        axd["e"].set_ylabel("Weight")
        axd["e"].set_xlabel("Delay")
        axd["e"].set_title("Stimulus")

        axd["f"].set_title("Prev. Choice")
        axd["f"].set_xlabel("Lag")

        axd["g"].set_xlabel("ILD")
        axd["g"].set_title("Stimulus")

        axd["h"].set_title("Prev. Choice")
        axd["h"].set_xlabel("Lag")

        axd["i"].set_ylabel("Autocorrelation")
        axd["i"].set_xlabel("Lag")
        axd["i"].set_title("Outcome")
        axd["i"].legend(
            *axd["i"].get_legend_handles_labels(),
            handlelength=0.4,
            ncol=2,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0, -0.06),
            columnspacing=0.6,
            handletextpad=0.5,
        )

        axd["j"].set_title("Outcome")
        axd["j"].set_xlabel("Lag")

        axd["k"].set_xlabel("Lag")
        axd["k"].set_title("Repetition")

        axd["l"].set_title("Repetition")
        axd["l"].set_xlabel("Lag")

        for _ax in [axd["hist_repeat_2ADC"], axd["hist_repeat_2AFC"]]:
            _ax.set_ylim(top=1e3)
        for _ax in [axd["i"], axd["j"], axd["k"], axd["l"]]:
            _ax.set_xlim(0, 20.5)

        fig.align_ylabels()
        fig.savefig((project_path / "figures" / "panels2" / "figure2").with_suffix(f".{format}"))

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
