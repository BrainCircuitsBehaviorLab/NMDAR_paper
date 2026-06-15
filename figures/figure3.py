import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import base64
    import io
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    # Custom package and plots
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.plots.emissions import _fold_three_choice_raw_weights as fold_three_choice
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import (
        add_choice_lag_summary_regressor,
        build_transition_chunk_drug_plot_data,
        build_transition_chunk_plot_data,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )
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
        two_afc_transition_chunk_lengths as build_two_afc_transition_chunk_lengths,
    )

    return (
        Path,
        add_choice_lag_summary_regressor,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        load_app_config,
        load_fit_arrays,
        mo,
        plt,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
        process_mcdr,
        process_two_adc,
        process_two_afc,
        sns,
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
    return (mount_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paths
    """)
    return


@app.cell
def _(Path, configure_paths, get_runtime_paths):
    ROOT = Path(__file__).resolve().parents[1]

    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    data_path = Path(__file__).parents[1] / "data/processed"
    print(data_path)

    project_path = Path(__file__).resolve().parents[1]
    print(project_path)

    format = "pdf"

    path_panels = project_path / "figures" / "panels3" / format
    import os
    os.makedirs(path_panels, exist_ok=True)
    print(path_panels)
    return (paths,)


@app.cell
def _(Path, plt, sns):
    sns.set_theme(style='ticks', context='notebook')
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = 'none'
    plt.rcParams['savefig.bbox'] = 'standard'
    return


@app.cell
def _(get_adapter):
    task_names = ("2AFC_delay", "2AFC", "MCDR")
    model_name = "param"
    adapters = {_task_name: get_adapter(_task_name) for _task_name in task_names}
    plots_by_task = {
        _task_name: _adapter.get_plots()
        for _task_name, _adapter in adapters.items()
    }
    dfs = {
        _task_name: _adapter.subject_filter(_adapter.read_dataset())
        for _task_name, _adapter in adapters.items()
    }

    subjects_by_task = {
        _task_name: list(_df["subject"].unique())
        for _task_name, _df in dfs.items()
    }
    return adapters, dfs, model_name, subjects_by_task, task_names


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
        _out = paths.RESULTS / "fits" / _task / "glmhmmt" / model_name
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
        _choice_lag_cols = []
        for _view in views[_task].values():
            for _feature in list(getattr(_view, "feat_names", []) or []):
                _feature = str(_feature)
                if _feature.startswith("choice_lag_") and _feature not in _choice_lag_cols:
                    _choice_lag_cols.append(_feature)
        # plot_dfs[_task] = add_choice_lag_summary_regressor(plot_dfs[_task], choice_lag_cols=_adapter.choice_lag_cols(trial_dfs[_task]))
        plot_dfs[_task] = add_choice_lag_summary_regressor(plot_dfs[_task], choice_lag_cols=_choice_lag_cols)
    return (views,)


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
                ["c", "c", "e", "e"],
                ["l", "m", "p", "q"],
                ["f", "f", "h", "h"],
                ["i", "i", "k", "k"],
            ],
            figsize=fig_size(1, 0.5),
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
                w_pad=0.01,
                h_pad=0.01,
                wspace=0.02,
                hspace=0.04,
            )
    else:
        fig, axd = None, {}
    return


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
    return


if __name__ == "__main__":
    app.run()
