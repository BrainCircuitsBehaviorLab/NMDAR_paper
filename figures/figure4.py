import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


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
    from scipy.stats import ttest_1samp

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        load_metrics_dir as load_metrics_dir_raw,
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
        build_trial_and_weights_df,
        build_views,
        configure_paths,
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


@app.cell
def _():
    mount_figure = False
    format = "pdf"
    return (format,)


@app.cell
def _():
    MODEL_BY_TASK = {
        "2AFC_delay": "param",
        "2AFC": "param2",
        # "MCDR": "param",
    }
    task_names = tuple(MODEL_BY_TASK)
    return MODEL_BY_TASK, task_names


@app.cell
def _(Path, configure_paths, format, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    project_path = ROOT
    path_panels = project_path / "figures" / "panels4" / format
    os.makedirs(path_panels, exist_ok=True)
    print(project_path)
    print(path_panels)
    return ROOT, paths


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

    drug_palette = {
        "drug" : "tab:pink",
        "saline" : "tab:gray",
        "rest": "tab:purple"
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
    return delay_order, feature_labels


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
def _(Annotator, np, pd, ttest_1samp):
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
        _model_dir = paths.RESULTS / "fits" / _task_name / "glmhmm" / _model_id
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
        _model_dir = paths.RESULTS / "fits" / _task_name / "glmhmm" / _model_id
        _K = int((_cfg.get("K_list") or [2])[0])
        _subjects = [str(_subject) for _subject in (_cfg.get("subjects") or list(_df_all["subject"].unique()))]
        _emission_cols = _cfg.get("emission_cols") or None
        _transition_cols = _cfg.get("transition_cols") or None

        _arrays_store, _ = load_fit_arrays(
            out_dir=_model_dir,
            arrays_suffix="glmhmm_arrays.npz",
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
