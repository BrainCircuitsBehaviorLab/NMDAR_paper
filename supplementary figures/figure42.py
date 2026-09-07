# /// script
# [tool.marimo.opengraph]
# title = "Supplementary Figure 5 — Treatment model checks"
# description = "Observed and GLM-HMM-T-predicted psychometric curves and repetition bias, including engaged-state panels, under saline and NMDAr blockade."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Supplementary figure 4.2

    ## Description
    Psychometric curves and repetition bias in saline and drug sessions for the 2ADC and 2AFC tasks, including analyses restricted to the engaged state.
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

    from glmhmmt.glm import predict_glm_probs
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        select_subject_behavior_df,
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
        predict_glm_probs,
        prepare_treatment_accuracy_repetition_curves,
        process_two_adc,
        process_two_afc,
        select_subject_behavior_df,
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
        "2ADC_DRUG": "drug_transitions2_nocv",
        "2AFC_DRUG": "drug_transitions2_nocv",
    }
    GLM_MODEL_BY_TASK = {
        "2ADC_DRUG": "one hot",
        "2AFC_DRUG": "one hot",
    }
    task_names = tuple(MODEL_BY_TASK)
    return GLM_MODEL_BY_TASK, MODEL_BY_TASK, format, mount_figure, task_names


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
    task_palette = {
        "2ADC_DRUG": "tab:blue",
        "2AFC_DRUG": "tab:orange",
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
        task_palette,
        treatment_order,
        treatment_palette,
    )


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                [
                    "psychometric_2ADC",
                    "engaged_psychometric_2ADC",
                    "repetition_bias_2ADC",
                    "engaged_repetition_bias_2ADC",
                ],
                [
                    "psychometric_2AFC",
                    "engaged_psychometric_2AFC",
                    "repetition_bias_2AFC",
                    "engaged_repetition_bias_2AFC",
                ],
            ],
            figsize=fig_size(1, 2),
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
def _(
    GLM_MODEL_BY_TASK,
    adapters,
    build_trial_and_weights_df,
    build_views,
    dfs,
    json,
    model_configs,
    np,
    paths,
    predict_glm_probs,
    process_two_adc,
    process_two_afc,
    select_subject_behavior_df,
):
    glm_plot_dfs = {}
    for _task_name, _model_id in GLM_MODEL_BY_TASK.items():
        _model_dir = paths.RESULTS / "fits" / _task_name / "glm" / _model_id
        _config = json.loads((_model_dir / "config.json").read_text())
        _subjects = [str(subject) for subject in model_configs[_task_name]["subjects"]]
        _adapter = adapters[_task_name]
        _arrays_store = {}
        for _subject in _subjects:
            _arrays_path = _model_dir / f"{_subject}_glm_arrays.npz"
            if not _arrays_path.exists():
                continue
            with np.load(_arrays_path, allow_pickle=True) as _stored:
                _arrays = {key: _stored[key] for key in _stored.files}

            _subject_trials = select_subject_behavior_df(
                dfs[_task_name],
                subject=_subject,
                sort_col=_adapter.sort_col,
                session_col=_adapter.session_col,
                min_session_length=2,
            )
            _feature_df = _adapter.build_feature_df(
                _subject_trials,
                tau=float(_config.get("tau", 50.0)),
                emission_cols=_config["emission_cols"],
            )
            _y, _X, _, _names = _adapter.build_design_matrices(
                _feature_df,
                emission_cols=_config["emission_cols"],
            )
            _weights = np.asarray(_arrays["emission_weights"], dtype=float)
            _p_pred = predict_glm_probs(
                np.asarray(_X),
                _weights[0],
                y=np.asarray(_y),
                baseline_class_idx=int(np.asarray(_arrays["baseline_class_idx"])),
                num_classes=_adapter.num_classes,
                lapse_mode=str(np.asarray(_arrays.get("lapse_mode", "none"))),
                lapse_rates=np.asarray(_arrays.get("lapse_rates", []), dtype=float),
            )
            _n_trials = len(_y)
            _arrays.update(
                {
                    "X": np.asarray(_X),
                    "X_cols": np.asarray(_names["X_cols"], dtype=object),
                    "y": np.asarray(_y),
                    "p_pred": _p_pred,
                    "smoothed_probs": np.ones((_n_trials, 1), dtype=float),
                    "predictive_state_probs": np.ones((_n_trials, 1), dtype=float),
                }
            )
            _arrays_store[_subject] = _arrays

        _selected = [subject for subject in _subjects if subject in _arrays_store]
        _views = build_views(_arrays_store, _adapter, 1, _selected)
        _trial_df, _ = build_trial_and_weights_df(
            dfs[_task_name],
            views=_views,
            adapter=_adapter,
            min_session_length=2,
        )
        if _task_name == "2ADC_DRUG":
            glm_plot_dfs[_task_name] = process_two_adc.prepare_predictions_df(_trial_df)
        else:
            glm_plot_dfs[_task_name] = process_two_afc.prepare_predictions_df(_trial_df)
    return (glm_plot_dfs,)


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
def _(
    np,
    pd,
    plot_dfs,
    prepare_treatment_accuracy_repetition_curves,
    task_names,
    treatment_order,
):
    engaged_treatment_curves = {}
    for _task_name in task_names:
        _trials = plot_dfs[_task_name].to_pandas().copy()
        _state_idx = pd.to_numeric(_trials["state_idx"], errors="coerce")
        _trials["p_model_right"] = np.where(
            _state_idx.eq(0),
            pd.to_numeric(_trials["pR_state_0"], errors="coerce"),
            pd.to_numeric(_trials["pR_state_1"], errors="coerce"),
        )
        engaged_treatment_curves[_task_name], _ = (
            prepare_treatment_accuracy_repetition_curves(
                _trials,
                task_name=_task_name,
                treatment_order=treatment_order,
                state_label="Engaged",
            )
        )
    return (engaged_treatment_curves,)


@app.cell
def _(np, pd, plot_dfs, task_names):
    treatment_psychometric_dfs = {}
    engaged_psychometric_dfs = {}
    disengaged_psychometric_dfs = {}
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
        _trials["p_right_state"] = np.where(
            pd.to_numeric(_trials["state_idx"], errors="coerce").eq(0),
            pd.to_numeric(_trials["pR_state_0"], errors="coerce"),
            pd.to_numeric(_trials["pR_state_1"], errors="coerce"),
        )

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

        for _state_label, _state_dfs in {
            "Engaged": engaged_psychometric_dfs,
            "Disengaged": disengaged_psychometric_dfs,
        }.items():
            _state_trials = _trials[_trials["state_label"] == _state_label].dropna(
                subset=[
                    "subject",
                    "treatment",
                    "stimulus_evidence",
                    "p_right_data",
                    "p_right_state",
                ]
            ).copy()
            _state_trials["_evidence_bin"] = pd.qcut(
                _state_trials["stimulus_evidence"], q=9, duplicates="drop"
            )
            _state_trials["stimulus_evidence"] = _state_trials.groupby(
                "_evidence_bin", observed=True
            )["stimulus_evidence"].transform("mean")
            _state_dfs[_task_name] = (
                _state_trials.groupby(
                    ["subject", "treatment", "stimulus_evidence"],
                    as_index=False,
                    observed=True,
                )
                .agg(
                    p_right_data=("p_right_data", "mean"),
                    p_right_model=("p_right_state", "mean"),
                )
                .sort_values(["treatment", "stimulus_evidence", "subject"])
            )
    return (
        disengaged_psychometric_dfs,
        engaged_psychometric_dfs,
        psychometric_limits,
        treatment_psychometric_dfs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2ADC PC
    """)
    return


@app.cell
def _(
    axd,
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
    if psychometric_2ADC.get_legend() is not None:
        psychometric_2ADC.get_legend().remove()
    # psychometric_2ADC.legend(handles=check_legend_handles, frameon=False, ncol=2)
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
    ## 2ADC engaged PC
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    engaged_psychometric_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    psychometric_limits,
    sns,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    engaged_psychometric_2ADC = (
        plt.gca() if not mount_figure else axd["engaged_psychometric_2ADC"]
    )
    engaged_psychometric_2ADC.clear()
    sns.lineplot(
        data=engaged_psychometric_dfs["2ADC_DRUG"],
        x="stimulus_evidence",
        y="p_right_model",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        palette=treatment_palette,
        ax=engaged_psychometric_2ADC,
    )
    sns.lineplot(
        data=engaged_psychometric_dfs["2ADC_DRUG"],
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
        ax=engaged_psychometric_2ADC,
    )
    engaged_psychometric_2ADC.set(
        title="2ADC Engaged",
        xlabel="Stimulus evidence",
        ylabel=r"$p(\mathrm{right})$",
        xlim=psychometric_limits["2ADC_DRUG"],
        ylim=(0, 1),
    )
    engaged_psychometric_2ADC.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    if mount_figure and engaged_psychometric_2ADC.get_legend() is not None:
        engaged_psychometric_2ADC.get_legend().remove()
    if not mount_figure:
        engaged_psychometric_2ADC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        engaged_psychometric_2ADC.figure.savefig(
            (path_panels / format / "engaged_psychometric_2ADC").with_suffix(
                f".{format}"
            )
        )
        engaged_psychometric_2ADC.figure.savefig(
            path_panels / "png" / "engaged_psychometric_2ADC.png",
            dpi=300,
        )
    engaged_psychometric_2ADC
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
    ## 2ADC engaged RB
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    curve_meta,
    engaged_treatment_curves,
    fig_size,
    format,
    mount_figure,
    np,
    path_panels,
    plt,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    engaged_repetition_bias_2ADC = (
        plt.gca() if not mount_figure else axd["engaged_repetition_bias_2ADC"]
    )
    engaged_repetition_bias_2ADC.clear()
    _repetition_df = engaged_treatment_curves["2ADC_DRUG"]["repetition_bias"]
    for _treatment in treatment_order:
        _treatment_df = _repetition_df[_repetition_df["treatment"] == _treatment]
        _x = _treatment_df["x_value"].to_numpy(dtype=float)
        _model = _treatment_df["model_mean"].to_numpy(dtype=float)
        _model_sem = _treatment_df["model_sem"].to_numpy(dtype=float)
        _data = _treatment_df["data_mean"].to_numpy(dtype=float)
        _data_sem = _treatment_df["data_sem"].to_numpy(dtype=float)
        _color = treatment_palette[_treatment]
        engaged_repetition_bias_2ADC.plot(
            _x,
            _model,
            color=_color,
            linewidth=1.8,
        )
        engaged_repetition_bias_2ADC.fill_between(
            _x,
            np.clip(_model - _model_sem, 0, 1),
            np.clip(_model + _model_sem, 0, 1),
            color=_color,
            alpha=0.18,
            linewidth=0,
        )
        engaged_repetition_bias_2ADC.errorbar(
            _x,
            _data,
            yerr=_data_sem,
            fmt="o",
            color=_color,
            markeredgewidth=0,
            capsize=2,
            zorder=3,
        )
    engaged_repetition_bias_2ADC.axhline(
        curve_meta["2ADC_DRUG"]["baseline"],
        color="0.6",
        linestyle="--",
        linewidth=0.8,
    )
    engaged_repetition_bias_2ADC.set(
        title="2ADC Engaged",
        xlabel=curve_meta["2ADC_DRUG"]["xlabel"],
        ylabel="Rep. bias",
        ylim=(0.45, 0.8),
    )
    if not mount_figure:
        engaged_repetition_bias_2ADC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        engaged_repetition_bias_2ADC.figure.savefig(
            (path_panels / format / "engaged_repetition_bias_2ADC").with_suffix(
                f".{format}"
            )
        )
        engaged_repetition_bias_2ADC.figure.savefig(
            path_panels / "png" / "engaged_repetition_bias_2ADC.png",
            dpi=300,
        )
    engaged_repetition_bias_2ADC
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
    if mount_figure and psychometric_2AFC.get_legend() is not None:
        psychometric_2AFC.get_legend().remove()
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
    ## 2AFC engaged PC
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    engaged_psychometric_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    psychometric_limits,
    sns,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    engaged_psychometric_2AFC = (
        plt.gca() if not mount_figure else axd["engaged_psychometric_2AFC"]
    )
    engaged_psychometric_2AFC.clear()
    sns.lineplot(
        data=engaged_psychometric_dfs["2AFC_DRUG"],
        x="stimulus_evidence",
        y="p_right_model",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        palette=treatment_palette,
        ax=engaged_psychometric_2AFC,
    )
    sns.lineplot(
        data=engaged_psychometric_dfs["2AFC_DRUG"],
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
        ax=engaged_psychometric_2AFC,
    )
    engaged_psychometric_2AFC.set(
        title="2AFC Engaged",
        xlabel="Stimulus evidence",
        ylabel=r"$p(\mathrm{right})$",
        xlim=psychometric_limits["2AFC_DRUG"],
        ylim=(0, 1),
    )
    engaged_psychometric_2AFC.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    if mount_figure and engaged_psychometric_2AFC.get_legend() is not None:
        engaged_psychometric_2AFC.get_legend().remove()
    if not mount_figure:
        engaged_psychometric_2AFC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        engaged_psychometric_2AFC.figure.savefig(
            (path_panels / format / "engaged_psychometric_2AFC").with_suffix(
                f".{format}"
            )
        )
        engaged_psychometric_2AFC.figure.savefig(
            path_panels / "png" / "engaged_psychometric_2AFC.png",
            dpi=300,
        )
    engaged_psychometric_2AFC
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
    repetition_bias_2AFC.set_xticks([0, 8, 20])
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
    ## 2AFC engaged RB
    """)
    return


@app.cell
def _(
    axd,
    check_legend_handles,
    curve_meta,
    engaged_treatment_curves,
    fig_size,
    format,
    mount_figure,
    np,
    path_panels,
    plt,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    engaged_repetition_bias_2AFC = (
        plt.gca() if not mount_figure else axd["engaged_repetition_bias_2AFC"]
    )
    engaged_repetition_bias_2AFC.clear()
    _repetition_df = engaged_treatment_curves["2AFC_DRUG"]["repetition_bias"]
    for _treatment in treatment_order:
        _treatment_df = _repetition_df[_repetition_df["treatment"] == _treatment]
        _x = _treatment_df["x_value"].to_numpy(dtype=float)
        _model = _treatment_df["model_mean"].to_numpy(dtype=float)
        _model_sem = _treatment_df["model_sem"].to_numpy(dtype=float)
        _data = _treatment_df["data_mean"].to_numpy(dtype=float)
        _data_sem = _treatment_df["data_sem"].to_numpy(dtype=float)
        _color = treatment_palette[_treatment]
        engaged_repetition_bias_2AFC.plot(
            _x,
            _model,
            color=_color,
            linewidth=1.8,
        )
        engaged_repetition_bias_2AFC.fill_between(
            _x,
            np.clip(_model - _model_sem, 0, 1),
            np.clip(_model + _model_sem, 0, 1),
            color=_color,
            alpha=0.18,
            linewidth=0,
        )
        engaged_repetition_bias_2AFC.errorbar(
            _x,
            _data,
            yerr=_data_sem,
            fmt="o",
            color=_color,
            markeredgewidth=0,
            capsize=2,
            zorder=3,
        )
    engaged_repetition_bias_2AFC.axhline(
        curve_meta["2AFC_DRUG"]["baseline"],
        color="0.6",
        linestyle="--",
        linewidth=0.8,
    )
    engaged_repetition_bias_2AFC.set(
        title="2AFC Engaged",
        xlabel=curve_meta["2AFC_DRUG"]["xlabel"],
        ylabel="Rep. bias",
        ylim=(0.45, 0.8),
    )
    engaged_repetition_bias_2AFC.set_xticks([0, 8, 20])
    if curve_meta["2AFC_DRUG"]["invert_x"]:
        engaged_repetition_bias_2AFC.invert_xaxis()
    if not mount_figure:
        engaged_repetition_bias_2AFC.legend(
            handles=check_legend_handles,
            frameon=False,
            ncol=2,
        )
        engaged_repetition_bias_2AFC.figure.savefig(
            (path_panels / format / "engaged_repetition_bias_2AFC").with_suffix(
                f".{format}"
            )
        )
        engaged_repetition_bias_2AFC.figure.savefig(
            path_panels / "png" / "engaged_repetition_bias_2AFC.png",
            dpi=300,
        )
    engaged_repetition_bias_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Disengaged-state psychometrics

    Saline and drug psychometric curves restricted to trials assigned to the
    disengaged state. These panels are saved separately from the main figure.
    """)
    return


@app.cell
def _(fig_size, plt):
    fig_disengaged_psychometrics, axd_disengaged_psychometrics = (
        plt.subplot_mosaic(
            [["disengaged_psychometric_2ADC", "disengaged_psychometric_2AFC"]],
            figsize=fig_size(1, 2),
            constrained_layout=True,
            sharey=True,
        )
    )
    return axd_disengaged_psychometrics, fig_disengaged_psychometrics


@app.cell
def _(
    axd_disengaged_psychometrics,
    disengaged_psychometric_dfs,
    psychometric_limits,
    sns,
    treatment_order,
    treatment_palette,
):
    disengaged_psychometric_2ADC = axd_disengaged_psychometrics[
        "disengaged_psychometric_2ADC"
    ]
    disengaged_psychometric_2ADC.clear()
    sns.lineplot(
        data=disengaged_psychometric_dfs["2ADC_DRUG"],
        x="stimulus_evidence",
        y="p_right_model",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        palette=treatment_palette,
        ax=disengaged_psychometric_2ADC,
    )
    sns.lineplot(
        data=disengaged_psychometric_dfs["2ADC_DRUG"],
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
        ax=disengaged_psychometric_2ADC,
    )
    disengaged_psychometric_2ADC.set(
        title="2ADC Disengaged",
        xlabel="Stimulus evidence",
        ylabel=r"$p(\mathrm{right})$",
        xlim=psychometric_limits["2ADC_DRUG"],
        ylim=(0, 1),
    )
    disengaged_psychometric_2ADC.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    if disengaged_psychometric_2ADC.get_legend() is not None:
        disengaged_psychometric_2ADC.get_legend().remove()
    disengaged_psychometric_2ADC
    return


@app.cell
def _(
    axd_disengaged_psychometrics,
    disengaged_psychometric_dfs,
    psychometric_limits,
    sns,
    treatment_order,
    treatment_palette,
):
    disengaged_psychometric_2AFC = axd_disengaged_psychometrics[
        "disengaged_psychometric_2AFC"
    ]
    disengaged_psychometric_2AFC.clear()
    sns.lineplot(
        data=disengaged_psychometric_dfs["2AFC_DRUG"],
        x="stimulus_evidence",
        y="p_right_model",
        hue="treatment",
        hue_order=treatment_order,
        estimator="mean",
        errorbar="se",
        err_kws={"edgecolor": "none", "linewidth": 0},
        palette=treatment_palette,
        ax=disengaged_psychometric_2AFC,
    )
    sns.lineplot(
        data=disengaged_psychometric_dfs["2AFC_DRUG"],
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
        ax=disengaged_psychometric_2AFC,
    )
    disengaged_psychometric_2AFC.set(
        title="2AFC Disengaged",
        xlabel="Stimulus evidence",
        ylabel="",
        xlim=psychometric_limits["2AFC_DRUG"],
        ylim=(0, 1),
    )
    if disengaged_psychometric_2AFC.get_legend() is not None:
        disengaged_psychometric_2AFC.get_legend().remove()
    disengaged_psychometric_2AFC
    return


@app.cell
def _(fig_disengaged_psychometrics, path_panels):
    fig_disengaged_psychometrics.align_labels()
    fig_disengaged_psychometrics.savefig(
        path_panels / "disengaged_psychometrics.svg"
    )
    fig_disengaged_psychometrics.savefig(
        path_panels / "disengaged_psychometrics.png",
        dpi=300,
    )
    fig_disengaged_psychometrics.savefig(
        path_panels / "disengaged_psychometrics.pdf"
    )
    fig_disengaged_psychometrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pooled-treatment repetition bias preview

    Saline and drug trials are pooled within each animal. Task conditions are
    aligned from easier to harder because delay and |ILD| use different units.
    """)
    return


@app.cell
def _(
    Line2D,
    fig_size,
    glm_plot_dfs,
    np,
    path_panels,
    plot_dfs,
    plt,
    prepare_treatment_accuracy_repetition_curves,
    task_labels,
    task_names,
    task_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    pooled_repetition_bias_preview = plt.gca()

    for _task_name in task_names:
        _pooled_trials = plot_dfs[_task_name].to_pandas().copy()
        _pooled_trials["condition"] = "Pooled"
        _pooled_curves, _ = prepare_treatment_accuracy_repetition_curves(
            _pooled_trials,
            task_name=_task_name,
            treatment_order=("Pooled",),
        )
        _glm_pooled_trials = glm_plot_dfs[_task_name].to_pandas().copy()
        _glm_pooled_trials["condition"] = "Pooled"
        _glm_pooled_curves, _ = prepare_treatment_accuracy_repetition_curves(
            _glm_pooled_trials,
            task_name=_task_name,
            treatment_order=("Pooled",),
        )
        _task_curve = _pooled_curves["repetition_bias"].sort_values("x_value")
        _glm_task_curve = _glm_pooled_curves["repetition_bias"].sort_values("x_value")
        if _task_name == "2AFC_DRUG":
            _task_curve = _task_curve.iloc[::-1]
            _glm_task_curve = _glm_task_curve.iloc[::-1]
        _task_curve = _task_curve.reset_index(drop=True)
        _glm_task_curve = _glm_task_curve.reset_index(drop=True)
        _difficulty_rank = np.linspace(0.0, 1.0, len(_task_curve))
        _glm_difficulty_rank = np.linspace(0.0, 1.0, len(_glm_task_curve))
        _model_mean = _task_curve["model_mean"].to_numpy(dtype=float)
        _model_sem = _task_curve["model_sem"].to_numpy(dtype=float)
        _color = task_palette[_task_name]

        pooled_repetition_bias_preview.plot(
            _difficulty_rank,
            _model_mean,
            color=_color,
            linewidth=1.8,
        )
        pooled_repetition_bias_preview.fill_between(
            _difficulty_rank,
            np.clip(_model_mean - _model_sem, 0, 1),
            np.clip(_model_mean + _model_sem, 0, 1),
            color=_color,
            alpha=0.18,
            linewidth=0,
        )
        pooled_repetition_bias_preview.plot(
            _glm_difficulty_rank,
            _glm_task_curve["model_mean"].to_numpy(dtype=float),
            color=_color,
            linestyle="--",
            linewidth=1.8,
        )
        pooled_repetition_bias_preview.errorbar(
            _difficulty_rank,
            _task_curve["data_mean"].to_numpy(dtype=float),
            yerr=_task_curve["data_sem"].to_numpy(dtype=float),
            fmt="o",
            color=_color,
            markeredgewidth=0,
            capsize=2,
            zorder=3,
        )

    pooled_repetition_bias_preview.axhline(
        0.5,
        color="0.6",
        linestyle="--",
        linewidth=0.8,
    )
    pooled_repetition_bias_preview.set(
        title="Saline + drug pooled",
        xlabel="Task difficulty",
        ylabel="Rep. bias",
        xlim=(0, 1),
        ylim=(0.45, 1),
    )
    pooled_repetition_bias_preview.set_xticks([0, 1], ["Easier", "Harder"])
    pooled_repetition_bias_preview.legend(
        handles=[
            *[
                Line2D(
                    [0],
                    [0],
                    color=task_palette[_task_name],
                    label=task_labels[_task_name],
                )
                for _task_name in task_names
            ],
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markeredgewidth=0,
                label="Data",
            ),
            Line2D([0], [0], color="black", label="GLM-HMM-T"),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="GLM one-hot",
            ),
        ],
        frameon=False,
        ncol=3,
    )
    pooled_repetition_bias_preview.figure.savefig(
        path_panels / "svg" / "pooled_repetition_bias_preview.svg"
    )
    pooled_repetition_bias_preview.figure.savefig(
        path_panels / "png" / "pooled_repetition_bias_preview.png",
        dpi=300,
    )
    pooled_repetition_bias_preview
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
        axd["engaged_psychometric_2ADC"].set_xlabel("")
        axd["engaged_psychometric_2ADC"].set_ylabel("")
        axd["engaged_psychometric_2AFC"].set_ylabel("")
        axd["repetition_bias_2ADC"].set_xlabel("")
        axd["repetition_bias_2ADC"].set_ylabel("")
        axd["repetition_bias_2AFC"].set_ylabel("")
        axd["engaged_repetition_bias_2ADC"].set_xlabel("")
        axd["engaged_repetition_bias_2ADC"].set_ylabel("")
        axd["engaged_repetition_bias_2AFC"].set_ylabel("")
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
