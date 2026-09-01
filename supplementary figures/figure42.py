# /// script
# [tool.marimo.opengraph]
# title = "Supplementary Figure 5 — Treatment model checks"
# description = "Observed and GLM-HMM-T-predicted psychometric curves and repetition bias under saline and NMDAr blockade."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supplementary figure 4.2

    ## Description
    Psychometric curves and repetition bias in saline and drug sessions for the 2ADC and 2AFC tasks.
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
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import two_adc as process_two_adc
    from src.process import two_afc as process_two_afc
    from src.process.common import prepare_treatment_accuracy_repetition_curves
    from src.plots.common import fig_size

    return (
        Line2D,
        Path,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        json,
        load_fit_arrays,
        mo,
        np,
        os,
        pd,
        plt,
        prepare_treatment_accuracy_repetition_curves,
        process_two_adc,
        process_two_afc,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings
    """)
    return


@app.cell
def _():
    mount_figure = True
    format = "svg"
    MODEL_BY_TASK = {
        "2ADC_DRUG": "drug_transitions2",
        "2AFC_DRUG": "drug_transitions2",
    }
    task_names = tuple(MODEL_BY_TASK)
    return MODEL_BY_TASK, format, mount_figure, task_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paths
    """)
    return


@app.cell
def _(Path, configure_paths, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    path_panels = ROOT / "supplementary figures" / "panels42"
    for panel_format in ("svg", "png"):
        os.makedirs(path_panels / panel_format, exist_ok=True)
    return ROOT, path_panels, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Style
    """)
    return


@app.cell
def _(Line2D, ROOT, plt, sns):
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    task_labels = {
        "2ADC_DRUG": "2ADC",
        "2AFC_DRUG": "2AFC",
    }
    treatment_order = ["Saline", "Drug"]
    treatment_palette = {
        "Saline": "tab:gray",
        "Drug": "tab:pink",
    }
    check_legend_handles = [
        Line2D([0], [0], color=treatment_palette["Saline"], label="Saline"),
        Line2D([0], [0], color=treatment_palette["Drug"], label="Drug"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="None",
            markeredgewidth=0,
            label="Data",
        ),
        Line2D([0], [0], color="black", label="Model"),
    ]
    return (
        check_legend_handles,
        task_labels,
        treatment_order,
        treatment_palette,
    )


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                ["psychometric_2ADC", "repetition_bias_2ADC"],
                ["psychometric_2AFC", "repetition_bias_2AFC"],
            ],
            figsize=fig_size(1, 1.15),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data and fits
    """)
    return


@app.cell
def _(MODEL_BY_TASK, get_adapter):
    adapters = {
        task_name: get_adapter(task_name)
        for task_name in MODEL_BY_TASK
    }
    dfs = {
        task_name: adapter.subject_filter(adapter.read_dataset())
        for task_name, adapter in adapters.items()
    }
    return adapters, dfs


@app.cell
def _(MODEL_BY_TASK, adapters, json, paths):
    model_configs = {}
    for _task_name, _model_id in MODEL_BY_TASK.items():
        _model_dir = paths.RESULTS / "fits" / _task_name / "glmhmmt" / _model_id
        _config_path = _model_dir / "config.json"
        _config = json.loads(_config_path.read_text())
        _config["model_dir"] = str(_model_dir)
        _config["model_id"] = _model_id
        model_configs[_task_name] = _config

        _adapter = adapters[_task_name]
        for _key in (
            "state_scoring_feature",
            "state_scoring_rule",
            "state_split_feature",
            "state_split_rule",
        ):
            if _key in _config:
                setattr(_adapter, _key, _config[_key] or None)
    return (model_configs,)


@app.cell
def _(
    adapters,
    build_trial_and_weights_df,
    build_views,
    dfs,
    load_fit_arrays,
    model_configs,
    paths,
    process_two_adc,
    process_two_afc,
    task_names,
):
    plot_dfs = {}
    model_load_report = []

    for _task_name in task_names:
        _config = model_configs[_task_name]
        _adapter = adapters[_task_name]
        _model_id = _config["model_id"]
        _model_dir = paths.RESULTS / "fits" / _task_name / "glmhmmt" / _model_id
        _n_states = int((_config.get("K_list") or [2])[0])
        _subjects = [
            str(subject)
            for subject in (
                _config.get("subjects") or list(dfs[_task_name]["subject"].unique())
            )
        ]

        _arrays_store, _ = load_fit_arrays(
            out_dir=_model_dir,
            arrays_suffix="glmhmmt_arrays.npz",
            adapter=_adapter,
            df_all=dfs[_task_name],
            subjects=_subjects,
            emission_cols=_config.get("emission_cols") or None,
            transition_cols=_config.get("transition_cols") or None,
            k=_n_states,
        )
        _selected = [subject for subject in _subjects if subject in _arrays_store]
        _selected_arrays = {
            subject: _arrays_store[subject]
            for subject in _selected
        }
        _views = build_views(_selected_arrays, _adapter, _n_states, _selected)
        _trial_df, _ = build_trial_and_weights_df(
            dfs[_task_name],
            views=_views,
            adapter=_adapter,
            min_session_length=2,
        )
        if _task_name == "2ADC_DRUG":
            plot_dfs[_task_name] = process_two_adc.prepare_predictions_df(_trial_df)
        else:
            plot_dfs[_task_name] = process_two_afc.prepare_predictions_df(_trial_df)
        model_load_report.append(
            f"{_task_name}: loaded {len(_selected)} animals from glmhmmt/{_model_id}"
        )
    return model_load_report, plot_dfs


@app.cell
def _(mo, model_load_report):
    mo.md("\n".join(f"- {line}" for line in model_load_report))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare psychometric and repetition-bias curves
    """)
    return


@app.cell
def _(
    plot_dfs,
    prepare_treatment_accuracy_repetition_curves,
    task_names,
    treatment_order,
):
    treatment_curves = {}
    curve_meta = {}
    for _task_name in task_names:
        treatment_curves[_task_name], curve_meta[_task_name] = (
            prepare_treatment_accuracy_repetition_curves(
                plot_dfs[_task_name],
                task_name=_task_name,
                treatment_order=treatment_order,
            )
        )
    return curve_meta, treatment_curves


@app.cell
def _(np, pd, plot_dfs, task_names):
    treatment_psychometric_dfs = {}
    psychometric_limits = {}

    for _task_name in task_names:
        _trials = plot_dfs[_task_name].to_pandas().copy()
        _condition = _trials["condition"].astype("string").str.lower()
        _trials["treatment"] = _condition.map({"saline": "Saline", "drug": "Drug"})
        _trials = _trials.dropna(subset=["subject", "treatment"])
        _trials["subject"] = _trials["subject"].astype(str)

        _evidence_col = {
            "2ADC_DRUG": "stim_x_delay_param",
            "2AFC_DRUG": "stim_param",
        }[_task_name]
        _model_col = next(
            (
                _column
                for _column in ("p_model_right", "p_pred", "pR")
                if _column in _trials.columns
            ),
            None,
        )
        if _model_col is None:
            _trials["p_right_model"] = np.where(
                pd.to_numeric(_trials["state_idx"], errors="coerce").eq(0),
                pd.to_numeric(_trials["pR_state_0"], errors="coerce"),
                pd.to_numeric(_trials["pR_state_1"], errors="coerce"),
            )
        else:
            _trials["p_right_model"] = pd.to_numeric(_trials[_model_col], errors="coerce")

        _response = pd.to_numeric(_trials["response"], errors="coerce")
        _trials["p_right_data"] = np.where(
            _response.notna(),
            (_response > 0).astype(float),
            np.nan,
        )
        _trials["stimulus_evidence"] = pd.to_numeric(
            _trials[_evidence_col], errors="coerce"
        )
        _psychometric_trials = _trials.dropna(
            subset=[
                "subject",
                "treatment",
                "stimulus_evidence",
                "p_right_data",
                "p_right_model",
            ]
        ).copy()
        psychometric_limits[_task_name] = (
            float(_psychometric_trials["stimulus_evidence"].min()),
            float(_psychometric_trials["stimulus_evidence"].max()),
        )
        _psychometric_trials["_evidence_bin"] = pd.qcut(
            _psychometric_trials["stimulus_evidence"], q=9, duplicates="drop"
        )
        _psychometric_trials["stimulus_evidence"] = _psychometric_trials.groupby(
            "_evidence_bin", observed=True
        )["stimulus_evidence"].transform("mean")
        treatment_psychometric_dfs[_task_name] = (
            _psychometric_trials.groupby(
                ["subject", "treatment", "stimulus_evidence"],
                as_index=False,
                observed=True,
            )
            .agg(
                p_right_data=("p_right_data", "mean"),
                p_right_model=("p_right_model", "mean"),
            )
            .sort_values(["treatment", "stimulus_evidence", "subject"])
        )
    return psychometric_limits, treatment_psychometric_dfs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2ADC PC
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    psychometric_limits,
    sns,
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_psychometric_dfs,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    psychometric_2ADC = plt.gca() if not mount_figure else axd["psychometric_2ADC"]
    psychometric_2ADC.clear()
    sns.lineplot(
        data=treatment_psychometric_dfs["2ADC_DRUG"],
        x="stimulus_evidence",
        y="p_right_model",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        palette=treatment_palette,
        ax=psychometric_2ADC,
    )
    sns.lineplot(
        data=treatment_psychometric_dfs["2ADC_DRUG"],
        x="stimulus_evidence",
        y="p_right_data",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_style="bars",
        marker="o",
        markeredgewidth=0,
        linewidth=0,
        palette=treatment_palette,
        legend=False,
        ax=psychometric_2ADC,
    )
    psychometric_2ADC.set(
        title=task_labels["2ADC_DRUG"],
        xlabel="Stimulus evidence",
        ylabel=r"$p(\mathrm{right})$",
        xlim=psychometric_limits["2ADC_DRUG"],
        ylim=(0, 1),
    )
    psychometric_2ADC.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    psychometric_2ADC.legend(handles=check_legend_handles, frameon=False, ncol=2)
    if not mount_figure:
        psychometric_2ADC.figure.savefig(
            (path_panels / format / "psychometric_2ADC").with_suffix(f".{format}")
        )
        psychometric_2ADC.figure.savefig(
            path_panels / "png" / "psychometric_2ADC.png",
            dpi=300,
        )
    psychometric_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2ADC RB
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    curve_meta,
    fig_size,
    format,
    mount_figure,
    np,
    path_panels,
    plt,
    task_labels,
    treatment_curves,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    repetition_bias_2ADC = (
        plt.gca() if not mount_figure else axd["repetition_bias_2ADC"]
    )
    repetition_bias_2ADC.clear()
    _repetition_df = treatment_curves["2ADC_DRUG"]["repetition_bias"]
    for _treatment in treatment_order:
        _treatment_df = _repetition_df[_repetition_df["treatment"] == _treatment]
        _x = _treatment_df["x_value"].to_numpy(dtype=float)
        _model = _treatment_df["model_mean"].to_numpy(dtype=float)
        _model_sem = _treatment_df["model_sem"].to_numpy(dtype=float)
        _data = _treatment_df["data_mean"].to_numpy(dtype=float)
        _data_sem = _treatment_df["data_sem"].to_numpy(dtype=float)
        _color = treatment_palette[_treatment]
        repetition_bias_2ADC.plot(_x, _model, color=_color, linewidth=1.8)
        repetition_bias_2ADC.fill_between(
            _x,
            np.clip(_model - _model_sem, 0, 1),
            np.clip(_model + _model_sem, 0, 1),
            color=_color,
            alpha=0.18,
            linewidth=0,
        )
        repetition_bias_2ADC.errorbar(
            _x,
            _data,
            yerr=_data_sem,
            fmt="o",
            color=_color,
            markeredgewidth=0,
            capsize=2,
            zorder=3,
        )
    repetition_bias_2ADC.axhline(curve_meta["2ADC_DRUG"]["baseline"], color="0.6", linestyle="--", linewidth=0.8)
    repetition_bias_2ADC.set(
        title=task_labels["2ADC_DRUG"],
        xlabel=curve_meta["2ADC_DRUG"]["xlabel"],
        ylabel="Rep. bias",
        ylim=(0.45, 1),
    )
    if not mount_figure:
        repetition_bias_2ADC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        repetition_bias_2ADC.figure.savefig(
            (path_panels / format / "repetition_bias_2ADC").with_suffix(f".{format}")
        )
        repetition_bias_2ADC.figure.savefig(
            path_panels / "png" / "repetition_bias_2ADC.png",
            dpi=300,
        )
    repetition_bias_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2AFC PC
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    psychometric_limits,
    sns,
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_psychometric_dfs,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    psychometric_2AFC = plt.gca() if not mount_figure else axd["psychometric_2AFC"]
    psychometric_2AFC.clear()
    sns.lineplot(
        data=treatment_psychometric_dfs["2AFC_DRUG"],
        x="stimulus_evidence",
        y="p_right_model",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        palette=treatment_palette,
        ax=psychometric_2AFC,
    )
    sns.lineplot(
        data=treatment_psychometric_dfs["2AFC_DRUG"],
        x="stimulus_evidence",
        y="p_right_data",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_style="bars",
        marker="o",
        markeredgewidth=0,
        linewidth=0,
        palette=treatment_palette,
        legend=False,
        ax=psychometric_2AFC,
    )
    psychometric_2AFC.set(
        title=task_labels["2AFC_DRUG"],
        xlabel="Stimulus evidence",
        ylabel=r"$p(\mathrm{right})$",
        xlim=psychometric_limits["2AFC_DRUG"],
        ylim=(0, 1),
    )
    psychometric_2AFC.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    if not mount_figure:
        psychometric_2AFC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        psychometric_2AFC.figure.savefig(
            (path_panels / format / "psychometric_2AFC").with_suffix(f".{format}")
        )
        psychometric_2AFC.figure.savefig(
            path_panels / "png" / "psychometric_2AFC.png",
            dpi=300,
        )
    psychometric_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2AFC RB
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    curve_meta,
    fig_size,
    format,
    mount_figure,
    np,
    path_panels,
    plt,
    task_labels,
    treatment_curves,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    repetition_bias_2AFC = (
        plt.gca() if not mount_figure else axd["repetition_bias_2AFC"]
    )
    repetition_bias_2AFC.clear()
    _repetition_df = treatment_curves["2AFC_DRUG"]["repetition_bias"]
    for _treatment in treatment_order:
        _treatment_df = _repetition_df[_repetition_df["treatment"] == _treatment]
        _x = _treatment_df["x_value"].to_numpy(dtype=float)
        _model = _treatment_df["model_mean"].to_numpy(dtype=float)
        _model_sem = _treatment_df["model_sem"].to_numpy(dtype=float)
        _data = _treatment_df["data_mean"].to_numpy(dtype=float)
        _data_sem = _treatment_df["data_sem"].to_numpy(dtype=float)
        _color = treatment_palette[_treatment]
        repetition_bias_2AFC.plot(_x, _model, color=_color, linewidth=1.8)
        repetition_bias_2AFC.fill_between(
            _x,
            np.clip(_model - _model_sem, 0, 1),
            np.clip(_model + _model_sem, 0, 1),
            color=_color,
            alpha=0.18,
            linewidth=0,
        )
        repetition_bias_2AFC.errorbar(
            _x,
            _data,
            yerr=_data_sem,
            fmt="o",
            color=_color,
            markeredgewidth=0,
            capsize=2,
            zorder=3,
        )
    repetition_bias_2AFC.axhline(curve_meta["2AFC_DRUG"]["baseline"], color="0.6", linestyle="--", linewidth=0.8)
    repetition_bias_2AFC.set(
        title=task_labels["2AFC_DRUG"],
        xlabel=curve_meta["2AFC_DRUG"]["xlabel"],
        ylabel="Rep. bias",
        ylim=(0.45, 1),
    )
    repetition_bias_2AFC.set_xticks([0, 2, 4, 8, 20])
    if curve_meta["2AFC_DRUG"]["invert_x"]:
        repetition_bias_2AFC.invert_xaxis()
    if not mount_figure:
        repetition_bias_2AFC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        repetition_bias_2AFC.figure.savefig(
            (path_panels / format / "repetition_bias_2AFC").with_suffix(f".{format}")
        )
        repetition_bias_2AFC.figure.savefig(
            path_panels / "png" / "repetition_bias_2AFC.png",
            dpi=300,
        )
    repetition_bias_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final figure
    """)
    return


@app.cell
def _(axd, fig, mount_figure, path_panels):
    if mount_figure:
        axd["psychometric_2ADC"].set_xlabel("")
        axd["repetition_bias_2ADC"].set_xlabel("")
        axd["repetition_bias_2ADC"].set_ylabel("")
        axd["repetition_bias_2AFC"].set_ylabel("")
        fig.align_labels()
        fig.savefig(path_panels / "supplementary_figure42.svg")
        fig.savefig(path_panels / "supplementary_figure42.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure42.pdf")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
