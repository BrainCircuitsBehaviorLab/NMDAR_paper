# /// script
# [tool.marimo.opengraph]
# title = "Figure 3"
# description = "Figure 3: GLM-HMM-T states and state-dependent behavior."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Same as ``figure3.py`` but with 1 cell per plot for that iterates over tasks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    # Imports
    import json
    import os
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
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from statannotations.Annotator import Annotator
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import (
        add_choice_lag_summary_regressor,
        glmhmmt_change_triggered_posterior_df,
        glmhmmt_state_accuracy_df,
        glmhmmt_state_dwell_df,
        glmhmmt_state_metric_df,
        glmhmmt_state_occupancy_df,
        glmhmmt_state_psychometric_df,
        glmhmmt_state_switch_histogram_df,
        glmhmmt_state_switches_df,
        glmhmmt_state_trace_df,
        glmhmmt_transition_weights_df,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )
    from src.plots.common import boxplot_STYLE, fig_size

    return (
        Annotator,
        Path,
        add_choice_lag_summary_regressor,
        boxplot_STYLE,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        glmhmmt_change_triggered_posterior_df,
        glmhmmt_state_accuracy_df,
        glmhmmt_state_dwell_df,
        glmhmmt_state_metric_df,
        glmhmmt_state_occupancy_df,
        glmhmmt_state_psychometric_df,
        glmhmmt_state_switch_histogram_df,
        glmhmmt_state_switches_df,
        glmhmmt_state_trace_df,
        glmhmmt_transition_weights_df,
        json,
        load_app_config,
        load_fit_arrays,
        mo,
        np,
        os,
        pd,
        pl,
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
    format = "pdf"
    return format, mount_figure


@app.cell
def _():
    MODEL_BY_TASK = {
        "2AFC_delay": "param2_dif_pure",
        "2AFC": "param2_dif_pure",
        "MCDR": "param",
    }
    task_names = tuple(MODEL_BY_TASK)
    return MODEL_BY_TASK, task_names


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

    project_path = ROOT
    path_panels = project_path / "figures" / "panels3" / format
    os.makedirs(path_panels, exist_ok=True)
    print(project_path)
    print(path_panels)
    return ROOT, path_panels, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Style
    """)
    return


@app.cell
def _(ROOT, plt, sns):
    sns.set_theme(style="ticks", context="notebook")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    task_labels = {
        "2AFC_delay": "2ADC",
        "2AFC": "2AFC",
        "MCDR": "3CDR",
    }
    task_palette = {
        "2AFC_delay": "tab:green",
        "2AFC": "tab:blue",
        "MCDR": "tab:orange",
    }
    state_palette = {
        "Engaged": "tab:green",
        "Disengaged": "tab:gray",
        "State 0": "tab:blue",
        "State 1": "tab:gray",
        "State 2": "tab:orange",
        "State 3": "tab:green",
    }
    feature_labels = {
        "bias": "Bias",
        "bias_param": "Bias",
        "biasparam": "Bias",
        "stim": "Stimulus",
        "stim_param": "Stimulus",
        "stim_vals": "Stimulus",
        "stim_x_delay_param": "Stimulus x delay",
        "choice_lag_param": "A",
        "choice_lag_param_correct": "A",
        "prev_choice": "Prev. choice",
        "filtered_reward": "Reward trace",
        "filtered_choice": "Choice trace",
        "filtered_stim_side": "Stimulus trace",
        "prev_difficulty": "Previous difficulty",
        "cumulative_reward": "Cumulative reward",
        "trial_index": "Trial index",
        "Drug": "NMDAr",
        "drug_code": "NMDAr",
    }
    delay_order = [-0.1, -1, -3, -10, 10, 3, 1, 0.1]
    delay_mapping = {value: index for index, value in enumerate(delay_order)}
    return (
        delay_mapping,
        delay_order,
        feature_labels,
        state_palette,
        task_labels,
        task_palette,
    )


@app.cell
def _(feature_labels):
    def label_feature(feature):
        return feature_labels.get(str(feature), str(feature).replace("_", " "))

    def with_feature_labels(df):
        if df is None or df.empty or "feature" not in df.columns:
            return df
        out = df.copy()
        out["feature_label"] = out["feature"].map(label_feature)
        return out

    return (with_feature_labels,)


@app.cell
def _(Annotator, np, pd):
    state_order = ["Engaged", "Disengaged"]

    def add_paired_state_annotation(
        ax,
        df,
        *,
        x,
        y,
        order,
        hue="state_label",
        subject_col="subject",
        hue_order=state_order,
    ):
        if df is None or df.empty or len(hue_order) != 2:
            return
        if not {x, y, hue, subject_col}.issubset(df.columns):
            return

        paired_frames = []
        available_pairs = []
        for x_idx, x_value in enumerate(order):
            sub = df[df[x] == x_value]
            paired = sub.pivot_table(
                values=y,
                index=subject_col,
                columns=hue,
                aggfunc="first",
            )
            if not all(state in paired.columns for state in hue_order):
                continue
            paired = paired.dropna(subset=list(hue_order))
            if len(paired) < 2:
                continue
            paired_subjects = set(paired.index.astype(str))
            paired_sub = sub[sub[subject_col].astype(str).isin(paired_subjects)].copy()
            paired_frames.append(paired_sub)
            available_pairs.append(((x_value, hue_order[0]), (x_value, hue_order[1])))

        if not available_pairs or not paired_frames:
            return

        annotator = Annotator(
            ax,
            available_pairs,
            data=pd.concat(paired_frames, ignore_index=True),
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            # loc="outside",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()

    def add_subject_pair_lines(
        ax,
        df,
        *,
        x,
        y,
        order,
        hue="state_label",
        subject_col="subject",
        hue_order=state_order,
        offset=0.2,
    ):
        if df is None or df.empty or len(hue_order) != 2:
            return
        if not {x, y, hue, subject_col}.issubset(df.columns):
            return
        for x_idx, x_value in enumerate(order):
            sub = df[df[x] == x_value]
            paired = sub.pivot_table(
                values=y,
                index=subject_col,
                columns=hue,
                aggfunc="first",
            )
            if not all(state in paired.columns for state in hue_order):
                continue
            paired = paired.dropna(subset=list(hue_order))
            for _, row in paired.iterrows():
                ax.plot(
                    [x_idx - offset, x_idx + offset],
                    [row[hue_order[0]], row[hue_order[1]]],
                    color="0.75",
                    linewidth=0.5,
                    zorder=0,
                )

    def _stars(pvalue):
        if not np.isfinite(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_one_sample_zero_annotations(ax, df, *, x, y, order):
        if df is None or df.empty or not {x, y}.issubset(df.columns):
            return
        try:
            from scipy.stats import ttest_1samp
        except ImportError:
            return

        y_values = pd.to_numeric(df[y], errors="coerce")
        finite = y_values[np.isfinite(y_values)]
        if finite.empty:
            return
        y_min = float(finite.min())
        y_max = float(finite.max())
        pad = max((y_max - y_min) * 0.08, 0.05)
        text_y = y_max + pad
        ax.set_ylim(top=text_y + pad)

        for x_idx, x_value in enumerate(order):
            values = pd.to_numeric(df.loc[df[x] == x_value, y], errors="coerce").dropna()
            if len(values) < 2:
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=0.0).pvalue)
            stars = _stars(pvalue)
            if stars:
                ax.text(x_idx, text_y, stars, ha="center", va="bottom")

    return (
        add_one_sample_zero_annotations,
        add_paired_state_annotation,
        add_subject_pair_lines,
        state_order,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data And Fits
    """)
    return


@app.cell
def _(MODEL_BY_TASK, get_adapter, pl):
    adapters = {_task_name: get_adapter(_task_name) for _task_name in MODEL_BY_TASK}
    plots_by_task = {
        _task_name: _adapter.get_plots()
        for _task_name, _adapter in adapters.items()
    }
    dfs = {
        _task_name: _adapter.subject_filter(_adapter.read_dataset())
        for _task_name, _adapter in adapters.items()
    }
    if "2AFC" in dfs:
        dfs["2AFC"] = dfs["2AFC"].filter(pl.col("subject") != "326")
    if "MCDR" in dfs:
        dfs["MCDR"] = dfs["MCDR"].filter(pl.col("subject").str.contains("B"))
    return adapters, dfs


@app.cell
def _(MODEL_BY_TASK, adapters, dfs, json, paths):
    model_configs = {}
    for _task_name, _model_id in MODEL_BY_TASK.items():
        _model_dir = paths.RESULTS / "fits" / _task_name / "glmhmmt" / _model_id
        _config_path = _model_dir / "config.json"
        if _config_path.exists():
            _cfg = json.loads(_config_path.read_text())
        else:
            _cfg = {
                "task": _task_name,
                "model_id": _model_id,
                "subjects": list(dfs[_task_name]["subject"].unique()),
                "K_list": [2],
                "emission_cols": None,
                "transition_cols": None,
            }
        _cfg["model_dir"] = str(_model_dir)
        _cfg["model_id"] = _model_id
        model_configs[_task_name] = _cfg

        _adapter = adapters[_task_name]
        for _key in (
            "state_scoring_feature",
            "state_scoring_rule",
            "state_split_feature",
            "state_split_rule",
        ):
            if _key in _cfg:
                setattr(_adapter, _key, _cfg[_key] or None)
    return (model_configs,)


@app.cell
def _(
    adapters,
    add_choice_lag_summary_regressor,
    build_trial_and_weights_df,
    build_views,
    dfs,
    load_fit_arrays,
    model_configs,
    paths,
    prepare_predictions_df,
    task_names,
):
    arrays_by_task = {}
    views = {}
    trial_dfs = {}
    weight_dfs = {}
    plot_dfs = {}
    subjects_by_task = {}
    model_load_report = []

    for _task_name in task_names:
        _cfg = model_configs[_task_name]
        _adapter = adapters[_task_name]
        _df_all = dfs[_task_name]
        _model_id = _cfg["model_id"]
        _model_dir = paths.RESULTS / "fits" / _task_name / "glmhmmt" / _model_id
        _K = int((_cfg.get("K_list") or [2])[0])
        _subjects = [str(_subject) for _subject in (_cfg.get("subjects") or list(_df_all["subject"].unique()))]
        _emission_cols = _cfg.get("emission_cols") or None
        _transition_cols = _cfg.get("transition_cols") or None

        _arrays_store, _ = load_fit_arrays(
            out_dir=_model_dir,
            arrays_suffix="glmhmmt_arrays.npz",
            adapter=_adapter,
            df_all=_df_all,
            subjects=_subjects,
            emission_cols=_emission_cols,
            transition_cols=_transition_cols,
            k=_K,
        )
        _selected = [_subject for _subject in _subjects if _subject in _arrays_store]
        if not _selected:
            model_load_report.append(
                f"{_task_name}: no arrays found for glmhmmt/{_model_id}"
            )
            continue

        arrays_by_task[_task_name] = {subject: _arrays_store[subject] for subject in _selected}
        subjects_by_task[_task_name] = _selected
        views[_task_name] = build_views(arrays_by_task[_task_name], _adapter, _K, _selected)
        trial_dfs[_task_name], weight_dfs[_task_name] = build_trial_and_weights_df(
            _df_all,
            views=views[_task_name],
            adapter=_adapter,
            min_session_length=2,
        )
        plot_dfs[_task_name] = prepare_predictions_df(_task_name, trial_dfs[_task_name])

        _choice_lag_cols = []
        for _view in views[_task_name].values():
            for _feature in list(getattr(_view, "feat_names", []) or []):
                _feature = str(_feature)
                if _feature.startswith("choice_lag_") and _feature not in _choice_lag_cols:
                    _choice_lag_cols.append(_feature)
        plot_dfs[_task_name] = add_choice_lag_summary_regressor(
            plot_dfs[_task_name],
            choice_lag_cols=_choice_lag_cols,
        )
        model_load_report.append(f"{_task_name}: loaded {len(_selected)} subjects from glmhmmt/{_model_id}")

    active_task_names = tuple(views)
    return (
        active_task_names,
        arrays_by_task,
        model_load_report,
        plot_dfs,
        views,
        weight_dfs,
    )


@app.cell
def _(mo, model_load_report):
    mo.md("\n".join(f"- {_line}" for _line in model_load_report))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Plot Data
    """)
    return


@app.cell
def _(
    active_task_names,
    adapters,
    dfs,
    plot_dfs,
    prepare_closed_loop_model_autocorrelograms,
    prepare_corrected_behavior_autocorrelograms,
    views,
):
    autocorrelograms_by_task = {}
    for _task_idx, _task_name in enumerate(active_task_names):
        _df = plot_dfs[_task_name]
        _data_autocorr = prepare_corrected_behavior_autocorrelograms(
            _df,
            subject_col="subject",
            session_col="session",
            choice_col="response",
            outcome_col="performance",
            trial_index_col="trial_idx",
            max_lag=50,
            min_cross_pairs=20,
            max_cross_pairs=80,
            seed=10 + _task_idx,
        )
        _model_autocorr = prepare_closed_loop_model_autocorrelograms(
            dfs[_task_name],
            views=views[_task_name],
            adapter=adapters[_task_name],
            n_simulations=1,
            max_lag=50,
            min_cross_pairs=20,
            max_cross_pairs=80,
            seed=100 + _task_idx,
            progress_label=f"{_task_name} GLM-HMM-T closed-loop simulations",
        )
        autocorrelograms_by_task[_task_name] = {
            "data": _data_autocorr,
            "glmhmmt": _model_autocorr,
        }
    return (autocorrelograms_by_task,)


@app.cell
def _(
    active_task_names,
    arrays_by_task,
    delay_order,
    glmhmmt_change_triggered_posterior_df,
    glmhmmt_state_accuracy_df,
    glmhmmt_state_dwell_df,
    glmhmmt_state_metric_df,
    glmhmmt_state_occupancy_df,
    glmhmmt_state_psychometric_df,
    glmhmmt_state_switch_histogram_df,
    glmhmmt_state_switches_df,
    glmhmmt_state_trace_df,
    glmhmmt_transition_weights_df,
    pd,
    pl,
    plot_dfs,
    views,
    weight_dfs,
    with_feature_labels,
):
    emission_plot_dfs = {}
    transition_plot_dfs = {}
    psychometric_dfs = {}
    accuracy_dfs = {}
    occupancy_dfs = {}
    trace_dfs = {}
    metric_dfs = {}
    dwell_dfs = {}
    switch_session_dfs = {}
    switch_subject_dfs = {}
    switch_hist_dfs = {}
    change_posterior_dfs = {}
    psychometric_x_cols = {}
    psychometric_x_labels = {}

    for _task_name in active_task_names:
        emission_plot_dfs[_task_name] = with_feature_labels(weight_dfs[_task_name].to_pandas())
        transition_plot_dfs[_task_name] = with_feature_labels(
            glmhmmt_transition_weights_df(arrays_by_task[_task_name], views[_task_name])
        )
        _psychometric_source_df = plot_dfs[_task_name]
        if _task_name == "2AFC_delay":
            _psychometric_source_df = _psychometric_source_df.clone().with_columns(
                (
                    pl.col("delays").cast(pl.Float64)
                    * pl.col("stimulus").cast(pl.Float64)
                ).round(1).alias("signed_delay")
            )
            _x_col = "signed_delay"
        else:
            _x_col = {
                "2AFC": "ILD",
                "MCDR": "ttype_c",
            }[_task_name]
        psychometric_x_cols[_task_name] = _x_col
        psychometric_x_labels[_task_name] = {
            "2AFC": "ILD",
            "2AFC_delay": "Signed delay",
            "MCDR": "Trial type",
        }[_task_name]
        psychometric_dfs[_task_name] = glmhmmt_state_psychometric_df(
            _psychometric_source_df,
            x_col=_x_col,
            x_order=delay_order if _task_name == "2AFC_delay" else None,
        )
        accuracy_dfs[_task_name] = glmhmmt_state_accuracy_df(plot_dfs[_task_name])
        occupancy_dfs[_task_name] = glmhmmt_state_occupancy_df(plot_dfs[_task_name])
        trace_dfs[_task_name] = glmhmmt_state_trace_df(plot_dfs[_task_name])
        metric_dfs[_task_name] = glmhmmt_state_metric_df(
            plot_dfs[_task_name],
            metrics=("RT", "RT2", "nLicks"),
        )
        dwell_dfs[_task_name] = glmhmmt_state_dwell_df(plot_dfs[_task_name])
        switch_session_dfs[_task_name], switch_subject_dfs[_task_name] = glmhmmt_state_switches_df(plot_dfs[_task_name])
        switch_hist_dfs[_task_name] = glmhmmt_state_switch_histogram_df(switch_session_dfs[_task_name])
        change_posterior_dfs[_task_name] = glmhmmt_change_triggered_posterior_df(plot_dfs[_task_name])

    all_accuracy_df = pd.concat(accuracy_dfs.values(), names=["task"], keys=accuracy_dfs.keys()).reset_index(level=0) if accuracy_dfs else pd.DataFrame()
    all_occupancy_df = pd.concat(occupancy_dfs.values(), names=["task"], keys=occupancy_dfs.keys()).reset_index(level=0) if occupancy_dfs else pd.DataFrame()
    all_switch_subject_df = pd.concat(switch_subject_dfs.values(), names=["task"], keys=switch_subject_dfs.keys()).reset_index(level=0) if switch_subject_dfs else pd.DataFrame()
    all_switch_hist_df = pd.concat(switch_hist_dfs.values(), names=["task"], keys=switch_hist_dfs.keys()).reset_index(level=0) if switch_hist_dfs else pd.DataFrame()
    return (
        all_accuracy_df,
        all_occupancy_df,
        change_posterior_dfs,
        dwell_dfs,
        emission_plot_dfs,
        metric_dfs,
        psychometric_dfs,
        psychometric_x_labels,
        switch_hist_dfs,
        trace_dfs,
        transition_plot_dfs,
    )


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
                ["c", "c", "d", "d"],
                ["e", "f", "g", "h"],
                ["i", "i", "j", "j"],
                ["k", "k", "l", "l"],
            ],
            figsize=fig_size(1, 0.7),
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.04)
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
    active_task_names,
    autocorrelograms_by_task,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    task_labels,
):
    _figs = []
    for _task_name in active_task_names:
        _payload = autocorrelograms_by_task.get(_task_name, {})
        _data_ac = _payload.get("data", {}).get("autocorr")
        _model_ac = _payload.get("glmhmmt", {}).get("autocorr")
        if _data_ac is None or _data_ac.empty:
            continue

        _fig, _axes = plt.subplots(1, 2, figsize=fig_size(1, 2), constrained_layout=True)
        for _signal, _ax in zip(("Outcome", "Repetition"), _axes, strict=False):
            _data_sub = _data_ac[_data_ac["signal"] == _signal].sort_values("lag")
            if not _data_sub.empty:
                _ax.errorbar(
                    _data_sub["lag"],
                    _data_sub["autocorr"],
                    yerr=_data_sub.get("autocorr_sem"),
                    fmt="o",
                    capsize=0,
                    ms=3,
                    color="tab:blue",
                    ecolor="tab:blue",
                    label="Data",
                    zorder=4,
                )
            if _model_ac is not None and not _model_ac.empty:
                _model_sub = _model_ac[_model_ac["signal"] == _signal].sort_values("lag")
                _ax.plot(
                    _model_sub["lag"],
                    _model_sub["autocorr"],
                    color="tab:red",
                    label="GLM-HMM-T",
                    zorder=3,
                )
            _ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
            _ax.set_title(_signal)
            _ax.set_xlabel("Lag")
            _ax.set_ylabel("Autocorrelation")
            _ax.legend(frameon=False)
        _fig.suptitle(task_labels.get(_task_name, _task_name))
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_autocorrelograms").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No autocorrelogram data loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Emission Weights
    """)
    return


@app.cell
def _(
    active_task_names,
    add_paired_state_annotation,
    add_subject_pair_lines,
    boxplot_STYLE,
    emission_plot_dfs,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
    task_labels,
):
    _figs = []
    for _task_name in active_task_names:
        _df = emission_plot_dfs[_task_name]
        if _df.empty:
            continue
        _order = list(dict.fromkeys(_df["feature_label"]))
        _hue_order = [state for state in state_order if state in set(_df["state_label"])]
        _hue_order += [state for state in dict.fromkeys(_df["state_label"]) if state not in _hue_order]
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        sns.boxplot(
            data=_df,
            x="feature_label",
            y="weight",
            hue="state_label",
            order=_order,
            hue_order=_hue_order,
            gap = 0.2,
            palette=state_palette,
            ax=_ax,
            **boxplot_STYLE,
        )
        add_subject_pair_lines(_ax, _df, x="feature_label", y="weight", order=_order)
        add_paired_state_annotation(_ax, _df, x="feature_label", y="weight", order=_order)
        _ax.axhline(0, color="0.5", linestyle="--")
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel("")
        _ax.set_ylabel("Emission weight")
        _ax.tick_params(axis="x")
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_emission_weights").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No emission weights loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transition Weights
    """)
    return


@app.cell
def _(
    active_task_names,
    add_one_sample_zero_annotations,
    boxplot_STYLE,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    task_labels,
    transition_plot_dfs,
):
    _figs = []
    for _task_name in active_task_names:
        _df = transition_plot_dfs[_task_name]
        if _df.empty:
            continue
        _order = list(dict.fromkeys(_df["feature_label"]))
        _fig, _ax = plt.subplots(figsize=fig_size(3, 1), constrained_layout=True)
        sns.boxplot(
            data=_df,
            x="feature_label",
            y="weight",
            order=_order,
            color="tab:gray",
            ax=_ax,
            **boxplot_STYLE,
        )
        _ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        add_one_sample_zero_annotations(_ax, _df, x="feature_label", y="weight", order=_order)
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel("")
        _ax.set_ylabel("Transition weight")
        _ax.tick_params(axis="x", rotation=30)
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_transition_weights").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No transition weights loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Psychometrics By State
    """)
    return


@app.cell
def _(
    active_task_names,
    delay_mapping,
    delay_order,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    psychometric_dfs,
    psychometric_x_labels,
    sns,
    state_palette,
    task_labels,
):
    _figs = []
    for _task_name in active_task_names:
        _df = psychometric_dfs[_task_name]
        if _df.empty:
            continue
        _order = (
            list(_df["x_label"].cat.categories)
            if str(_df["x_label"].dtype) == "category"
            else list(dict.fromkeys(_df["x_label"]))
        )
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        if _task_name == "2AFC_delay":
            sns.lineplot(
                data=_df,
                x="x_position",
                y="p_right",
                hue="state_label",
                estimator="mean",
                errorbar="se",
                marker="o",
                markeredgewidth=0,
                err_kws={
                    "edgecolor": "none",
                    "linewidth": 0,
                },
                palette=state_palette,
                ax=_ax,
            )
            _ax.set_xticks(list(delay_mapping.values()))
            _ax.set_xticklabels([f"{value:g}" for value in delay_order])
        elif _task_name == "2AFC" and "x_numeric" in _df.columns:
            sns.lineplot(
                data=_df,
                x="x_numeric",
                y="p_right",
                hue="state_label",
                estimator="mean",
                errorbar="se",
                marker="o",
                markeredgewidth=0,
                err_kws={
                    "edgecolor": "none",
                    "linewidth": 0,
                },
                palette=state_palette,
                ax=_ax,
            )
            _xticks = sorted(_df["x_numeric"].dropna().unique())
            _xticks = [float(_tick) for _tick in _xticks if abs(float(_tick)) not in {2.0, 4.0}]
            _ax.set_xticks(_xticks)
            _ax.set_xticklabels([f"{_tick:g}" for _tick in _xticks])
        else:
            sns.pointplot(
                data=_df,
                x="x_label",
                y="p_right",
                hue="state_label",
                order=_order,
                errorbar="se",
                dodge=0.2,
                palette=state_palette,
                ax=_ax,
            )
        _ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel(psychometric_x_labels.get(_task_name, ""))
        _ax.set_ylabel(r"$p(\mathrm{right})$")
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_psychometric_by_state").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No psychometric data loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy And Occupancy By State
    """)
    return


@app.cell
def _(
    add_subject_pair_lines,
    all_accuracy_df,
    all_occupancy_df,
    boxplot_STYLE,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
    task_labels,
):
    _figs = []
    for _name, _df, _y, _ylabel in (
        ("accuracy_by_state", all_accuracy_df, "accuracy", "Accuracy"),
        ("occupancy_by_state", all_occupancy_df, "occupancy", "Occupancy"),
    ):
        if _df.empty:
            continue
        _plot_df = _df.copy()
        _plot_df["task_label"] = _plot_df["task"].map(task_labels).fillna(_plot_df["task"])
        _order = list(dict.fromkeys(_plot_df["task_label"]))
        _hue_order = [state for state in state_order if state in set(_plot_df["state_label"])]
        _hue_order += [state for state in dict.fromkeys(_plot_df["state_label"]) if state not in _hue_order]
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        sns.boxplot(
            data=_plot_df,
            x="task_label",
            y=_y,
            hue="state_label",
            order=_order,
            hue_order=_hue_order,
            palette=state_palette,
            ax=_ax,
            **boxplot_STYLE,
        )
        add_subject_pair_lines(_ax, _plot_df, x="task_label", y=_y, order=_order)
        # add_paired_state_annotation(_ax, _plot_df, x="task_label", y=_y, order=_order)
        _ax.set_xlabel("")
        _ax.set_ylabel(_ylabel)
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / f"glmhmmt_{_name}").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No accuracy or occupancy data loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean Traces
    """)
    return


@app.cell
def _(
    active_task_names,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
    trace_dfs,
):
    _figs = []
    for _task_name in active_task_names:
        _df = trace_dfs[_task_name]
        if _df.empty:
            continue
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        sns.lineplot(
            data=_df,
            x="trial_bin",
            y="p_state",
            hue="state_label",
            estimator="mean",
            errorbar="se",
            palette=state_palette,
            ax=_ax,
        )
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel("Normalized session time")
        _ax.set_ylabel("State posterior")
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_mean_state_traces").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No trace data loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RTs And Licks By State (2AFC)
    """)
    return


@app.cell
def _(
    add_paired_state_annotation,
    add_subject_pair_lines,
    boxplot_STYLE,
    fig_size,
    format,
    metric_dfs,
    mo,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    _df = metric_dfs.get("2AFC")
    if _df is None or _df.empty:
        _out = mo.md("No 2AFC RT or lick data loaded.")
    else:
        _order = list(dict.fromkeys(_df["metric"]))
        _hue_order = [state for state in state_order if state in set(_df["state_label"])]
        _hue_order += [state for state in dict.fromkeys(_df["state_label"]) if state not in _hue_order]
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        sns.boxplot(
            data=_df,
            x="metric",
            y="value",
            hue="state_label",
            order=_order,
            hue_order=_hue_order,
            palette=state_palette,
            ax=_ax,
            **boxplot_STYLE,
        )
        add_subject_pair_lines(_ax, _df, x="metric", y="value", order=_order)
        add_paired_state_annotation(_ax, _df, x="metric", y="value", order=_order)
        _ax.set_xlabel("")
        _ax.set_ylabel("Mean per subject")
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / "2AFC_glmhmmt_rt_licks_by_state").with_suffix(f".{format}"))
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cumulative Dwell-Time Probability
    """)
    return


@app.cell
def _(
    active_task_names,
    dwell_dfs,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    _figs = []
    for _task_name in active_task_names:
        _df = dwell_dfs[_task_name]
        if _df.empty:
            continue
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        sns.ecdfplot(
            data=_df,
            x="dwell_trials",
            hue="state_label",
            palette=state_palette,
            ax=_ax,
        )
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel("Dwell time (trials)")
        _ax.set_ylabel("Cumulative probability")
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_dwell_time_cdf").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No dwell-time data loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State Switch Histogram By Animal
    """)
    return


@app.cell
def _(
    active_task_names,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    switch_hist_dfs,
    task_labels,
    task_palette,
):
    _figs = []
    for _task_name in active_task_names:
        _plot_df = switch_hist_dfs.get(_task_name)
        if _plot_df is None or _plot_df.empty:
            continue
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        _color = task_palette.get(_task_name, "tab:gray")
        sns.histplot(
            data=_plot_df,
            x="n_switches",
            weights="switch_probability",
            discrete=True,
            stat="probability",
            shrink=0.85,
            color=_color,
            edgecolor="white",
            ax=_ax,
        )
        _summary = (
            _plot_df.groupby("n_switches", as_index=False, observed=True)
            .agg(
                switch_probability=("switch_probability", "mean"),
                sem=("switch_probability", "sem"),
            )
            .fillna({"sem": 0.0})
        )
        _ax.errorbar(
            _summary["n_switches"],
            _summary["switch_probability"],
            yerr=_summary["sem"],
            fmt="none",
            color="0.2",
            capsize=2,
            linewidth=0.8,
        )
        _ticks = sorted(_plot_df["n_switches"].dropna().astype(int).unique())
        _ax.set_xticks(_ticks)
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel("Mean switches per session")
        _ax.set_ylabel("Animal probability")
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_state_switch_histogram_by_animal").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No state-switch data loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posteriors Around A Change
    """)
    return


@app.cell
def _(
    active_task_names,
    change_posterior_dfs,
    fig_size,
    format,
    mo,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    _figs = []
    for _task_name in active_task_names:
        _df = change_posterior_dfs[_task_name]
        if _df.empty:
            continue
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1), constrained_layout=True)
        sns.lineplot(
            data=_df,
            x="lag",
            y="p_state",
            hue="state_label",
            estimator="mean",
            errorbar="se",
            palette=state_palette,
            ax=_ax,
        )
        _ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
        _ax.set_title(task_labels.get(_task_name, _task_name))
        _ax.set_xlabel("Trials from state change")
        _ax.set_ylabel("State posterior")
        _ax.legend(frameon=False, title="")
        _fig.savefig((path_panels / f"{_task_name}_glmhmmt_posteriors_around_change").with_suffix(f".{format}"))
        _figs.append(_fig)
    mo.hstack(_figs, justify="start", gap=1) if _figs else mo.md("No change-triggered posterior data loaded.")
    return


if __name__ == "__main__":
    app.run()
