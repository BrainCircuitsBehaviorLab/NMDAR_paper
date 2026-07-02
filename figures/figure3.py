# /// script
# [tool.marimo.opengraph]
# title = "Figure 3 Split"
# description = "Figure 3 split by task: GLM-HMM-T states and state-dependent behavior."
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
    import json
    import os
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
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
        glmhmmt_state_switch_histogram_df,
        glmhmmt_state_switches_df,
        glmhmmt_state_trace_df,
        glmhmmt_transition_weights_df,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )
    from src.plots.common import boxplot_STYLE, build_session_repetition_data, fig_size

    return (
        Annotator,
        FuncFormatter,
        Path,
        add_choice_lag_summary_regressor,
        boxplot_STYLE,
        build_session_repetition_data,
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
        glmhmmt_state_switch_histogram_df,
        glmhmmt_state_switches_df,
        glmhmmt_state_trace_df,
        glmhmmt_transition_weights_df,
        json,
        load_app_config,
        load_fit_arrays,
        load_metrics_dir_raw,
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


@app.cell
def _():
    model_type = "glmhmmt"
    MODEL_BY_TASK = {
        "2AFC_delay": "param half-pure",
        "2AFC": "param half-pure",
        # "MCDR": "param",
    }
    task_names = tuple(MODEL_BY_TASK)
    return MODEL_BY_TASK, model_type, task_names


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
        "stim": "Stim.",
        "stim_param": "Stim.",
        "stim_vals": "Stimulus",
        "stim_x_delay_param": "Delay",
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
def _(Annotator, FuncFormatter, np, pd, ttest_1samp):
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

    def align_zero_to_axis_fraction(ax, *, zero_fraction=0.25):
        bottom, top = ax.get_ylim()
        lower_span = max(0 - min(bottom, 0), 0)
        upper_span = max(max(top, 0) - 0, 0)
        if lower_span == 0 and upper_span == 0:
            return
        total_span = max(
            lower_span / zero_fraction,
            upper_span / (1 - zero_fraction),
        )
        ax.set_ylim(-zero_fraction * total_span, (1 - zero_fraction) * total_span)

    def format_delta_bic_axis(ax):
        def _format_bic_tick(value, _position):
            if not np.isfinite(value):
                return ""
            if value == 0:
                return "0"
            return f"{value / 1000:g}"

        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_bic_tick))
        ax.yaxis.get_offset_text().set_visible(False)

    return (
        add_one_sample_zero_annotations,
        add_paired_state_annotation,
        add_subject_pair_lines,
        align_zero_to_axis_fraction,
        format_delta_bic_axis,
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
    model_type,
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
        _model_dir = paths.RESULTS / "fits" / _task_name / model_type / _model_id
        _K = int((_cfg.get("K_list") or [2])[0])
        _subjects = [str(_subject) for _subject in (_cfg.get("subjects") or list(_df_all["subject"].unique()))]
        _emission_cols = _cfg.get("emission_cols") or None
        _transition_cols = _cfg.get("transition_cols") or None

        _arrays_store, _ = load_fit_arrays(
            out_dir=_model_dir,
            arrays_suffix=f"{model_type}_arrays.npz",
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
    glmhmmt_state_switch_histogram_df,
    glmhmmt_state_switches_df,
    glmhmmt_state_trace_df,
    glmhmmt_transition_weights_df,
    np,
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
    choice_lag_psychometric_dfs = {}
    psychometric_x_cols = {}
    psychometric_x_labels = {}
    action_trace_psychometric_dfs = {}

    def prediction_col(df):
        return next((col for col in ("p_model_right", "p_pred", "pR") if col in df.columns), None)

    def state_psychometric_data_model_df(
        trial_df,
        *,
        x_col,
        x_order=None,
        response_col="response",
        state_col="state_label",
        subject_col="subject",
    ):
        df = trial_df.to_pandas().copy() if hasattr(trial_df, "to_pandas") else pd.DataFrame(trial_df).copy()
        empty = pd.DataFrame(
            columns=[
                subject_col,
                state_col,
                "source",
                "x_value",
                "x_label",
                "x_numeric",
                "x_position",
                "p_right",
                "n_trials",
            ]
        )
        model_col = prediction_col(df)
        required = {subject_col, state_col, x_col, response_col}
        if df.empty or not required.issubset(df.columns):
            return empty

        cols = [subject_col, state_col, x_col, response_col]
        if model_col is not None:
            cols.append(model_col)
        out = df[cols].copy()
        out["_response_right"] = (pd.to_numeric(out[response_col], errors="coerce") > 0).astype(float)
        if model_col is not None:
            out["_model_right"] = pd.to_numeric(out[model_col], errors="coerce")
        else:
            out["_model_right"] = np.nan
        out = out.dropna(subset=[subject_col, state_col, x_col, "_response_right"])
        if out.empty:
            return empty

        if x_order is None:
            unique_values = list(pd.unique(out[x_col]))
            numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")
            if numeric_values.notna().all():
                order = [
                    value
                    for _, value in sorted(
                        zip(numeric_values.astype(float), unique_values, strict=False),
                        key=lambda item: item[0],
                    )
                ]
            else:
                order = sorted(unique_values, key=lambda value: str(value))
        else:
            order = list(x_order)

        def format_x_label(value):
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                return f"{float(numeric):g}"
            return str(value)

        label_order = [format_x_label(value) for value in order]
        label_by_position = dict(enumerate(label_order))
        out["x_value"] = pd.Categorical(out[x_col], categories=order, ordered=True)
        out = out.dropna(subset=["x_value"])
        if out.empty:
            return empty

        out["x_position"] = out["x_value"].cat.codes
        out["x_label"] = pd.Categorical(
            out["x_position"].map(label_by_position),
            categories=label_order,
            ordered=True,
        )
        summary = (
            out.groupby([subject_col, state_col, "x_value", "x_position", "x_label"], as_index=False, observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=("_model_right", "mean"),
                n_trials=("_response_right", "size"),
            )
            .sort_values([state_col, "x_position", subject_col])
        )
        summary["x_numeric"] = pd.to_numeric(summary["x_label"].astype(str), errors="coerce")
        data_summary = (
            summary.assign(source="Data", p_right=summary["data_mean"])
            [[subject_col, state_col, "source", "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"]]
        )
        if model_col is None:
            return data_summary
        model_summary = (
            summary.assign(source="Model", p_right=summary["model_mean"])
            [[subject_col, state_col, "source", "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"]]
        )
        return pd.concat([data_summary, model_summary], ignore_index=True)

    def action_trace_psychometric_df(
        trial_df,
        *,
        x_col,
        x_order=None,
        action_cols=("choice_lag_param", "choice_lag_one_hot_sum", "choice_lag_param_correct", "filtered_choice"),
        n_bins=4,
    ):
        df = trial_df.to_pandas().copy() if hasattr(trial_df, "to_pandas") else pd.DataFrame(trial_df).copy()
        empty = pd.DataFrame(
            columns=[
                "subject",
                "source",
                "action_bin",
                "x_value",
                "x_label",
                "x_numeric",
                "x_position",
                "p_right",
                "n_trials",
            ]
        )
        action_col = next((col for col in action_cols if col in df.columns), None)
        model_col = prediction_col(df)
        if action_col is None or not {"subject", x_col, "response"}.issubset(df.columns):
            return empty

        cols = ["subject", x_col, "response", action_col]
        if model_col is not None:
            cols.append(model_col)
        out = df[cols].copy()
        out["_action_trace"] = pd.to_numeric(out[action_col], errors="coerce")
        out["_response_right"] = (pd.to_numeric(out["response"], errors="coerce") > 0).astype(float)
        if model_col is not None:
            out["_model_right"] = pd.to_numeric(out[model_col], errors="coerce")
        else:
            out["_model_right"] = np.nan
        out = out.dropna(subset=["subject", x_col, "_response_right", "_action_trace"])
        if out.empty or out["_action_trace"].nunique() < 2:
            return empty

        q_count = min(int(n_bins), int(out["_action_trace"].nunique()))
        action_cut = pd.qcut(out["_action_trace"], q=q_count, duplicates="drop")
        action_labels = [f"Q{idx + 1}" for idx in range(len(action_cut.cat.categories))]
        out["action_bin"] = pd.Categorical(
            action_cut.cat.rename_categories(action_labels).astype(str),
            categories=action_labels,
            ordered=True,
        )

        if x_order is None:
            unique_values = list(pd.unique(out[x_col]))
            numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")
            if numeric_values.notna().all():
                order = [
                    value
                    for _, value in sorted(
                        zip(numeric_values.astype(float), unique_values, strict=False),
                        key=lambda item: item[0],
                    )
                ]
            else:
                order = sorted(unique_values, key=lambda value: str(value))
        else:
            order = list(x_order)

        def format_x_label(value):
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                return f"{float(numeric):g}"
            return str(value)

        label_order = [format_x_label(value) for value in order]
        label_by_position = dict(enumerate(label_order))
        out["x_value"] = pd.Categorical(out[x_col], categories=order, ordered=True)
        out = out.dropna(subset=["x_value", "action_bin"])
        if out.empty:
            return empty

        out["x_position"] = out["x_value"].cat.codes
        out["x_label"] = pd.Categorical(
            out["x_position"].map(label_by_position),
            categories=label_order,
            ordered=True,
        )
        summary = (
            out.groupby(["subject", "action_bin", "x_value", "x_position", "x_label"], as_index=False, observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=("_model_right", "mean"),
                n_trials=("_response_right", "size"),
            )
            .sort_values(["action_bin", "x_position", "subject"])
        )
        summary["x_numeric"] = pd.to_numeric(summary["x_label"].astype(str), errors="coerce")
        data_summary = (
            summary.assign(source="Data", p_right=summary["data_mean"])
            [["subject", "source", "action_bin", "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"]]
        )
        if model_col is None:
            return data_summary
        model_summary = (
            summary.assign(source="Model", p_right=summary["model_mean"])
            [["subject", "source", "action_bin", "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"]]
        )
        return pd.concat([data_summary, model_summary], ignore_index=True)

    def choice_lag_psychometric_df(trial_df, *, n_bins=9):
        df = trial_df.to_pandas().copy() if hasattr(trial_df, "to_pandas") else pd.DataFrame(trial_df).copy()
        empty = pd.DataFrame(
            columns=[
                "subject",
                "state_label",
                "source",
                "x_value",
                "x_label",
                "x_numeric",
                "x_position",
                "p_right",
                "n_trials",
            ]
        )
        model_col = prediction_col(df)
        required = {"subject", "state_label", "choice_lag_param", "response"}
        if df.empty or not required.issubset(df.columns):
            return empty

        cols = ["subject", "state_label", "choice_lag_param", "response"]
        if model_col is not None:
            cols.append(model_col)
        out = df[cols].copy()
        out["_choice_lag"] = pd.to_numeric(out["choice_lag_param"], errors="coerce")
        out["_response_right"] = (pd.to_numeric(out["response"], errors="coerce") > 0).astype(float)
        if model_col is not None:
            out["_model_right"] = pd.to_numeric(out[model_col], errors="coerce")
        else:
            out["_model_right"] = np.nan
        out = out.dropna(subset=["subject", "state_label", "_choice_lag", "_response_right"])
        if out.empty:
            return empty

        out["x_value"] = pd.cut(out["_choice_lag"], bins=int(n_bins), duplicates="drop")
        out = out.dropna(subset=["x_value"])
        if out.empty:
            return empty

        bin_categories = out["x_value"].cat.categories
        bin_centers = [float(interval.mid) for interval in bin_categories]
        out["x_position"] = out["x_value"].cat.codes
        out["x_numeric"] = out["x_position"].map(dict(enumerate(bin_centers))).astype(float)
        out["x_label"] = out["x_numeric"].map(lambda value: f"{value:g}")

        summary = (
            out.groupby(["subject", "state_label", "x_value", "x_position", "x_numeric", "x_label"], as_index=False, observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=("_model_right", "mean"),
                n_trials=("_response_right", "size"),
            )
            .sort_values(["state_label", "x_position", "subject"])
        )
        data_summary = (
            summary.assign(source="Data", p_right=summary["data_mean"])
            [["subject", "state_label", "source", "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"]]
        )
        if model_col is None:
            return data_summary
        model_summary = (
            summary.assign(source="Model", p_right=summary["model_mean"])
            [["subject", "state_label", "source", "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"]]
        )
        return pd.concat([data_summary, model_summary], ignore_index=True)

    for _task_name in active_task_names:
        emission_plot_dfs[_task_name] = with_feature_labels(weight_dfs[_task_name].to_pandas())
        _transition_df = with_feature_labels(
            glmhmmt_transition_weights_df(arrays_by_task[_task_name], views[_task_name])
        )
        if "transition_label" in _transition_df.columns and _transition_df["transition_label"].nunique() > 1:
            _transition_df["transition_feature_label"] = (
                _transition_df["transition_label"].astype(str)
                + "\n"
                + _transition_df["feature_label"].astype(str)
            )
        else:
            _transition_df["transition_feature_label"] = _transition_df["feature_label"].astype(str)
        transition_plot_dfs[_task_name] = _transition_df
        _psychometric_source_df = plot_dfs[_task_name]
        if _task_name == "2AFC_delay":
            _psychometric_source_df = _psychometric_source_df.clone().with_columns(
                (
                    pl.col("delays").cast(pl.Float64)
                    * pl.when(pl.col("stimulus").cast(pl.Float64) == 0)
                    .then(pl.lit(-1.0))
                    .when(pl.col("stimulus").cast(pl.Float64) == 1)
                    .then(pl.lit(1.0))
                    .otherwise(pl.col("stimulus").cast(pl.Float64).sign())
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
        psychometric_dfs[_task_name] = state_psychometric_data_model_df(
            _psychometric_source_df,
            x_col=_x_col,
            x_order=delay_order if _task_name == "2AFC_delay" else None,
        )
        choice_lag_psychometric_dfs[_task_name] = choice_lag_psychometric_df(_psychometric_source_df)
        action_trace_psychometric_dfs[_task_name] = action_trace_psychometric_df(
            _psychometric_source_df,
            x_col=_x_col,
            x_order=delay_order if _task_name == "2AFC_delay" else None,
        )
        accuracy_dfs[_task_name] = glmhmmt_state_accuracy_df(plot_dfs[_task_name])
        occupancy_dfs[_task_name] = glmhmmt_state_occupancy_df(plot_dfs[_task_name])
        trace_dfs[_task_name] = glmhmmt_state_trace_df(plot_dfs[_task_name])
        metric_dfs[_task_name] = glmhmmt_state_metric_df(
            plot_dfs[_task_name],
            metrics=("RT", "RT2", "nLicks", "ILI"),
        )
        dwell_dfs[_task_name] = glmhmmt_state_dwell_df(plot_dfs[_task_name])
        switch_session_dfs[_task_name], switch_subject_dfs[_task_name] = glmhmmt_state_switches_df(plot_dfs[_task_name])
        switch_hist_dfs[_task_name] = glmhmmt_state_switch_histogram_df(switch_session_dfs[_task_name])
        change_posterior_dfs[_task_name] = glmhmmt_change_triggered_posterior_df(plot_dfs[_task_name])
    return (
        accuracy_dfs,
        change_posterior_dfs,
        choice_lag_psychometric_dfs,
        dwell_dfs,
        emission_plot_dfs,
        metric_dfs,
        occupancy_dfs,
        psychometric_dfs,
        psychometric_x_labels,
        switch_hist_dfs,
        trace_dfs,
        transition_plot_dfs,
    )


@app.cell
def _(
    accuracy_dfs,
    emission_plot_dfs,
    metric_dfs,
    model_type,
    occupancy_dfs,
    psychometric_dfs,
    state_order,
    switch_hist_dfs,
    task_labels,
    task_names,
    transition_plot_dfs,
):
    def ordered_values(df, column):
        return list(dict.fromkeys(df[column]))


    def ordered_feature_values(df, column):
        values = ordered_values(df, column)

        def feature_priority(value):
            label = str(value).lower()
            if label == "bias":
                return 0
            if "stim" in label:
                return 1
            if label == "a":
                return 2
            return 3

        return sorted(values, key=feature_priority)


    def ordered_states(df):
        values = set(df["state_label"])
        ordered = [state for state in state_order if state in values]
        ordered += [state for state in dict.fromkeys(df["state_label"]) if state not in ordered]
        return ordered


    emission_orders = {task: ordered_feature_values(emission_plot_dfs[task], "feature_label") for task in task_names}
    emission_hue_orders = {task: ordered_states(emission_plot_dfs[task]) for task in task_names}

    transition_orders = {}
    if model_type == "glmhmmt":  # Check if it is a model with transitions
        transition_orders = {task: ordered_values(transition_plot_dfs[task], "transition_feature_label") for task in task_names}

    psychometric_orders = {
        task: (
            list(psychometric_dfs[task]["x_label"].cat.categories)
            if str(psychometric_dfs[task]["x_label"].dtype) == "category"
            else ordered_values(psychometric_dfs[task], "x_label")
        )
        for task in task_names
    }
    accuracy_plot_dfs = {task: accuracy_dfs[task].assign(task_label=task_labels.get(task, task)) for task in task_names}
    occupancy_plot_dfs = {task: occupancy_dfs[task].assign(task_label=task_labels.get(task, task)) for task in task_names}
    occupancy_annotation_dfs = {
        task: (
            occupancy_plot_dfs[task]
            .groupby(["subject", "task_label", "state_label"], as_index=False, observed=True)
            .agg(occupancy=("occupancy", "mean"))
        )
        for task in task_names
    }
    task_label_orders = {task: [task_labels.get(task, task)] for task in task_names}
    accuracy_hue_orders = {task: ordered_states(accuracy_plot_dfs[task]) for task in task_names}
    occupancy_hue_orders = {task: ordered_states(occupancy_plot_dfs[task]) for task in task_names}
    metric_order = ordered_values(metric_dfs["2AFC"], "metric")
    metric_hue_order = ordered_states(metric_dfs["2AFC"])
    switch_hist_summary_dfs = {
        task: (
            switch_hist_dfs[task]
            .groupby("n_switches", as_index=False, observed=True)
            .agg(
                switch_probability=("switch_probability", "mean"),
                sem=("switch_probability", "sem"),
            )
            .fillna({"sem": 0.0})
        )
        for task in task_names
    }
    return (
        accuracy_hue_orders,
        accuracy_plot_dfs,
        emission_hue_orders,
        emission_orders,
        occupancy_annotation_dfs,
        occupancy_hue_orders,
        occupancy_plot_dfs,
        psychometric_orders,
        switch_hist_summary_dfs,
        task_label_orders,
        transition_orders,
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
        fig = plt.figure(figsize=fig_size(1), constrained_layout=True)

        gs = fig.add_gridspec(
            4, 4,
            width_ratios=[1, 1, 1, 1],
            height_ratios=[1, 1, 1, 1],
        )

        axd = {
            "emission_weights_2ADC": fig.add_subplot(gs[0, 0]),
            "transition_weights_2ADC": fig.add_subplot(gs[0, 1]),
            "emission_weights_2AFC": fig.add_subplot(gs[0, 2]),
            "transition_weights_2AFC": fig.add_subplot(gs[0, 3]),

            "psychometric_by_state_2ADC": fig.add_subplot(gs[1, 0]),
            "nlicks_by_state_2ADC": fig.add_subplot(gs[1, 1]),
            "psychometric_by_state_2AFC": fig.add_subplot(gs[1, 2]),
            "nlicks_by_state_2AFC": fig.add_subplot(gs[1, 3]),

            "autocorrelograms_2ADC_repetition": fig.add_subplot(gs[2, 0]),
            "autocorrelograms_2AFC_repetition": fig.add_subplot(gs[2, 2]),

            "single_session_2ADC": fig.add_subplot(gs[3, 0:2]),
            "single_session_2AFC": fig.add_subplot(gs[3, 2:4]),
        }

        # ---- parent axes for model comparison blocks ----
        parent_2ADC = fig.add_subplot(gs[2, 1])
        parent_2AFC = fig.add_subplot(gs[2, 3])

        for parent in [parent_2ADC, parent_2AFC]:
            parent.set_xticks([])
            parent.set_yticks([])
            parent.set_frame_on(False)
            parent.set_ylabel(r"$\Delta$ GLM-HMM", labelpad=2)

        # ---- actual inner axes ----
        axd["model_comparison_ll_2ADC"] = parent_2ADC.inset_axes([0.00, 0.0, 0.35, 1.0])
        axd["model_comparison_bic_2ADC"] = parent_2ADC.inset_axes([0.65, 0.0, 0.35, 1.0])

        axd["model_comparison_ll_2AFC"] = parent_2AFC.inset_axes([0.00, 0.0, 0.35, 1.0])
        axd["model_comparison_bic_2AFC"] = parent_2AFC.inset_axes([0.65, 0.0, 0.35, 1.0])

        # Optional: keep parents accessible
        axd["_model_comparison_parent_2ADC"] = parent_2ADC
        axd["_model_comparison_parent_2AFC"] = parent_2AFC

        fig.set_constrained_layout_pads(
            w_pad=0.005,
            h_pad=0.01,
            wspace=0.005,
            hspace=0.04,
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
):
    _data_autocorr = autocorrelograms_by_task["2AFC_delay"]["data"]["autocorr"]
    _model_autocorr = autocorrelograms_by_task["2AFC_delay"]["glmhmmt"]["autocorr"]

    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    autocorrelograms_2ADC_outcome = plt.gca() if not mount_figure else axd.get("autocorrelograms_2ADC_outcome", plt.gca())
    autocorrelograms_2ADC_outcome.clear()

    _data_sub = _data_autocorr[_data_autocorr["signal"] == "Outcome"].sort_values("lag")
    autocorrelograms_2ADC_outcome.errorbar(
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
    # autocorrelograms_2ADC_outcome.plot(
    #     _data_sub["lag"],
    #     _data_sub["autocorr"],
    #     color="tab:blue",
    #     label="Data",
    #     zorder=3,
    # )

    _model_sub = _model_autocorr[_model_autocorr["signal"] == "Outcome"].sort_values("lag")
    autocorrelograms_2ADC_outcome.plot(
        _model_sub["lag"],
        _model_sub["autocorr"],
        color="tab:red",
        label="GLM-HMM-T",
        zorder=3,
    )

    autocorrelograms_2ADC_outcome.axhline(0.0, color="0.5", linestyle="--")
    autocorrelograms_2ADC_outcome.set_title("Outcomes")
    autocorrelograms_2ADC_outcome.set_xlabel("Lag")
    autocorrelograms_2ADC_outcome.set_ylabel("Autocorrelation")
    autocorrelograms_2ADC_outcome.legend(frameon=False)

    if not mount_figure:
        autocorrelograms_2ADC_outcome.figure.savefig((path_panels / "2AFC_delay_autocorrelogram_outcome").with_suffix(f".{format}"))

    autocorrelograms_2ADC_outcome
    return


@app.cell
def _(
    autocorrelograms_by_task,
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
):
    _data_autocorr = autocorrelograms_by_task["2AFC_delay"]["data"]["autocorr"]
    _model_autocorr = autocorrelograms_by_task["2AFC_delay"]["glmhmmt"]["autocorr"]

    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    autocorrelograms_2ADC_repetition = plt.gca() if not mount_figure else axd.get("autocorrelograms_2ADC_repetition", plt.gca())
    autocorrelograms_2ADC_repetition.clear()

    _data_sub = _data_autocorr[_data_autocorr["signal"] == "Repetition"].sort_values("lag")
    autocorrelograms_2ADC_repetition.errorbar(
        _data_sub["lag"],
        _data_sub["autocorr"],
        yerr=_data_sub.get("autocorr_sem"),
        fmt="o",
        capsize=0,
        ms=1,
        color="tab:blue",
        ecolor="tab:blue",
        label="Data",
        zorder=4,
    )
    autocorrelograms_2ADC_repetition.plot(
        _data_sub["lag"],
        _data_sub["autocorr"],
        color="tab:blue",
        label="_nolegend_",
        zorder=3,
    )

    _model_sub = _model_autocorr[_model_autocorr["signal"] == "Repetition"].sort_values("lag")
    autocorrelograms_2ADC_repetition.plot(
        _model_sub["lag"],
        _model_sub["autocorr"],
        color="tab:red",
        label="GLM-HMM-T",
        zorder=3,
    )

    autocorrelograms_2ADC_repetition.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    autocorrelograms_2ADC_repetition.set_title("Repetition")
    autocorrelograms_2ADC_repetition.set_xlabel("Lag")
    autocorrelograms_2ADC_repetition.set_ylabel("Autocorrelation")
    autocorrelograms_2ADC_repetition.legend(frameon=False)

    if not mount_figure:
        autocorrelograms_2ADC_repetition.figure.savefig((path_panels / "2AFC_delay_autocorrelogram_repetition").with_suffix(f".{format}"))

    autocorrelograms_2ADC_repetition
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
):
    _data_autocorr = autocorrelograms_by_task["2AFC"]["data"]["autocorr"]
    _model_autocorr = autocorrelograms_by_task["2AFC"]["glmhmmt"]["autocorr"]

    plt.figure(figsize=fig_size(2, 2), constrained_layout=True)
    autocorrelograms_2AFC_outcome = plt.gca() if not mount_figure else axd.get("autocorrelograms_2AFC_outcome", plt.gca())
    autocorrelograms_2AFC_outcome.clear()

    _data_sub = _data_autocorr[_data_autocorr["signal"] == "Outcome"].sort_values("lag")
    autocorrelograms_2AFC_outcome.errorbar(
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

    _model_sub = _model_autocorr[_model_autocorr["signal"] == "Outcome"].sort_values("lag")
    autocorrelograms_2AFC_outcome.plot(
        _model_sub["lag"],
        _model_sub["autocorr"],
        color="tab:red",
        label="GLM-HMM-T",
        zorder=3,
    )

    autocorrelograms_2AFC_outcome.axhline(0.0, color="0.5", linestyle="--")
    autocorrelograms_2AFC_outcome.set_title("Outcomes")
    autocorrelograms_2AFC_outcome.set_xlabel("Lag")
    autocorrelograms_2AFC_outcome.set_ylabel("Autocorrelation")
    autocorrelograms_2AFC_outcome.legend(frameon=False)

    if not mount_figure:
        autocorrelograms_2AFC_outcome.figure.savefig((path_panels / "2AFC_autocorrelogram_outcome").with_suffix(f".{format}"))

    autocorrelograms_2AFC_outcome
    return


@app.cell
def _(
    autocorrelograms_by_task,
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
):
    _data_autocorr = autocorrelograms_by_task["2AFC"]["data"]["autocorr"]
    _model_autocorr = autocorrelograms_by_task["2AFC"]["glmhmmt"]["autocorr"]

    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    autocorrelograms_2AFC_repetition = plt.gca() if not mount_figure else axd.get("autocorrelograms_2AFC_repetition", plt.gca())
    autocorrelograms_2AFC_repetition.clear()

    _data_sub = _data_autocorr[_data_autocorr["signal"] == "Repetition"].sort_values("lag")
    autocorrelograms_2AFC_repetition.errorbar(
        _data_sub["lag"],
        _data_sub["autocorr"],
        yerr=_data_sub.get("autocorr_sem"),
        fmt="o",
        capsize=0,
        ms=1,
        color="tab:blue",
        ecolor="tab:blue",
        label="Data",
        zorder=4,
    )

    _model_sub = _model_autocorr[_model_autocorr["signal"] == "Repetition"].sort_values("lag")
    autocorrelograms_2AFC_repetition.plot(
        _model_sub["lag"],
        _model_sub["autocorr"],
        color="tab:red",
        label="GLM-HMM-T",
        zorder=3,
    )

    autocorrelograms_2AFC_repetition.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    autocorrelograms_2AFC_repetition.set_title("Repetition")
    autocorrelograms_2AFC_repetition.set_xlabel("Lag")
    autocorrelograms_2AFC_repetition.set_ylabel("Autocorrelation")
    autocorrelograms_2AFC_repetition.legend(frameon=False)

    if not mount_figure:
        autocorrelograms_2AFC_repetition.figure.savefig((path_panels / "2AFC_autocorrelogram_repetition").with_suffix(f".{format}"))

    autocorrelograms_2AFC_repetition
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    __ax,
    autocorrelograms_by_task,
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    task_labels,
):
    _data_autocorr = autocorrelograms_by_task["MCDR"]["data"]["autocorr"]
    _model_autocorr = autocorrelograms_by_task["MCDR"]["glmhmmt"]["autocorr"]
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    if not mount_figure:
        _axes = plt.gcf().subplots(1, 2)
    else:
        _axes = [
            axd.get("autocorrelograms_3CDR_outcome", plt.gca()),
            axd.get("autocorrelograms_3CDR_repetition", plt.gca()),
        ]
    autocorrelograms_3CDR_outcome, autocorrelograms_3CDR_repetition = _axes
    for _ax in _axes:
        __ax.clear()

    for _signal, _ax in zip(("Outcome", "Repetition"), _axes, strict=False):
        _data_sub = _data_autocorr[_data_autocorr["signal"] == _signal].sort_values("lag")
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
        if _model_autocorr is not None and not _model_autocorr.empty:
            _model_sub = _model_autocorr[_model_autocorr["signal"] == _signal].sort_values("lag")
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
    autocorrelograms_3CDR_outcome.figure.suptitle(task_labels["MCDR"])
    if not mount_figure:
        autocorrelograms_3CDR_outcome.figure.savefig((path_panels / "MCDR_glmhmmt_autocorrelograms").with_suffix(f".{format}"))
    autocorrelograms_3CDR_outcome
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Emission Weights
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
    add_paired_state_annotation,
    axd,
    boxplot_STYLE,
    emission_hue_orders,
    emission_orders,
    emission_plot_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    emission_weights_2ADC = plt.gca() if not mount_figure else axd.get("emission_weights_2ADC", plt.gca())
    emission_weights_2ADC.clear()
    sns.boxplot(
        data=emission_plot_dfs["2AFC_delay"],
        x="feature_label",
        y="weight",
        hue="state_label",
        order=emission_orders["2AFC_delay"],
        hue_order=emission_hue_orders["2AFC_delay"],
        gap=0.2,
        palette=state_palette,
        ax=emission_weights_2ADC,
        **boxplot_STYLE,
    )
    # add_subject_pair_lines(emission_weights_2ADC, emission_plot_dfs["2AFC_delay"], x="feature_label", y="weight", order=emission_orders["2AFC_delay"])
    add_paired_state_annotation(emission_weights_2ADC, emission_plot_dfs["2AFC_delay"], x="feature_label", y="weight", order=emission_orders["2AFC_delay"])
    emission_weights_2ADC.axhline(0, color="0.5", linestyle="--")
    # emission_weights_2ADC.set_title(task_labels["2AFC_delay"])
    emission_weights_2ADC.set_xlabel("")
    emission_weights_2ADC.set_ylabel("Emission weight")
    emission_weights_2ADC.tick_params(axis="x")
    # emission_weights_2ADC.legend(frameon=False, title="")
    emission_weights_2ADC.legend_.remove()
    if not mount_figure:
        emission_weights_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_emission_weights").with_suffix(f".{format}"))
    emission_weights_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_paired_state_annotation,
    axd,
    boxplot_STYLE,
    emission_hue_orders,
    emission_orders,
    emission_plot_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    emission_weights_2AFC = plt.gca() if not mount_figure else axd.get("emission_weights_2AFC", plt.gca())
    emission_weights_2AFC.clear()
    sns.boxplot(
        data=emission_plot_dfs["2AFC"],
        x="feature_label",
        y="weight",
        hue="state_label",
        order=emission_orders["2AFC"],
        hue_order=emission_hue_orders["2AFC"],
        gap=0.2,
        palette=state_palette,
        ax=emission_weights_2AFC,
        **boxplot_STYLE,
    )
    # add_subject_pair_lines(emission_weights_2AFC, emission_plot_dfs["2AFC"], x="feature_label", y="weight", order=emission_orders["2AFC"])
    add_paired_state_annotation(emission_weights_2AFC, emission_plot_dfs["2AFC"], x="feature_label", y="weight", order=emission_orders["2AFC"])
    emission_weights_2AFC.axhline(0, color="0.5", linestyle="--")
    # emission_weights_2AFC.set_title(task_labels["2AFC"])
    emission_weights_2AFC.set_xlabel("")
    emission_weights_2AFC.set_ylabel("Emission weight")
    emission_weights_2AFC.tick_params(axis="x")
    # emission_weights_2AFC.legend(frameon=False, title="")
    emission_weights_2AFC.legend_.remove()
    if not mount_figure:
        emission_weights_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_emission_weights").with_suffix(f".{format}"))
    emission_weights_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    add_paired_state_annotation,
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    emission_hue_orders,
    emission_orders,
    emission_plot_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    emission_weights_3CDR = plt.gca() if not mount_figure else axd.get("emission_weights_3CDR", plt.gca())
    emission_weights_3CDR.clear()
    sns.boxplot(
        data=emission_plot_dfs["MCDR"],
        x="feature_label",
        y="weight",
        hue="state_label",
        order=emission_orders["MCDR"],
        hue_order=emission_hue_orders["MCDR"],
        gap=0.2,
        palette=state_palette,
        ax=emission_weights_3CDR,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(emission_weights_3CDR, emission_plot_dfs["MCDR"], x="feature_label", y="weight", order=emission_orders["MCDR"])
    add_paired_state_annotation(emission_weights_3CDR, emission_plot_dfs["MCDR"], x="feature_label", y="weight", order=emission_orders["MCDR"])
    emission_weights_3CDR.axhline(0, color="0.5", linestyle="--")
    emission_weights_3CDR.set_title(task_labels["MCDR"])
    emission_weights_3CDR.set_xlabel("")
    emission_weights_3CDR.set_ylabel("Emission weight")
    emission_weights_3CDR.tick_params(axis="x")
    emission_weights_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        emission_weights_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_emission_weights").with_suffix(f".{format}"))
    emission_weights_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transition Weights
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
    add_one_sample_zero_annotations,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    model_type,
    mount_figure,
    path_panels,
    plt,
    sns,
    transition_orders,
    transition_plot_dfs,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    transition_weights_2ADC = plt.gca() if not mount_figure else axd.get("transition_weights_2ADC", plt.gca())
    transition_weights_2ADC.clear()
    if model_type != "glmhmmt":
        transition_weights_2ADC.set_axis_off()
        transition_weights_2ADC.text(0.5, 0.5, "Transition", ha="center", va="center",)
        transition_weights_2ADC.text(0.5, 0.3, "Weights", ha="center", va="center",)
    else: 
        sns.boxplot(
            data=transition_plot_dfs["2AFC_delay"],
            x="transition_feature_label",
            y="weight",
            order=transition_orders["2AFC_delay"],
            color="tab:gray",
            ax=transition_weights_2ADC,
            **boxplot_STYLE,
        )
        transition_weights_2ADC.axhline(0, color="0.5", linestyle="--")
        add_one_sample_zero_annotations(transition_weights_2ADC, transition_plot_dfs["2AFC_delay"], x="transition_feature_label", y="weight", order=transition_orders["2AFC_delay"])
    # transition_weights_2ADC.set_title(task_labels["2AFC_delay"])
    transition_weights_2ADC.set_xlabel("")
    transition_weights_2ADC.set_xticklabels(["E->E", "E->D", "D->E", "D->D"])
    transition_weights_2ADC.set_ylabel("Transition weight")
    transition_weights_2ADC.tick_params(axis="x", rotation=45)
    if not mount_figure:
        transition_weights_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_transition_weights").with_suffix(f".{format}"))
    transition_weights_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    model_type,
    mount_figure,
    path_panels,
    plt,
    sns,
    transition_orders,
    transition_plot_dfs,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    transition_weights_2AFC = plt.gca() if not mount_figure else axd.get("transition_weights_2AFC", plt.gca())
    transition_weights_2AFC.clear()
    if model_type != "glmhmmt":
        transition_weights_2AFC.set_axis_off()
        transition_weights_2AFC.text(0.5, 0.5, "Transition", ha="center", va="center", )
        transition_weights_2AFC.text(0.5, 0.3, "Weights", ha="center", va="center",)
    else:
        sns.boxplot(
            data=transition_plot_dfs["2AFC"],
            x="transition_feature_label",
            y="weight",
            order=transition_orders["2AFC"],
            color="tab:gray",
            ax=transition_weights_2AFC,
            **boxplot_STYLE,
        )
        transition_weights_2AFC.axhline(0, color="0.5", linestyle="--")
        add_one_sample_zero_annotations(transition_weights_2AFC, transition_plot_dfs["2AFC"], x="transition_feature_label", y="weight",
                                        order=transition_orders["2AFC"])
    # transition_weights_2AFC.set_title(task_labels["2AFC"])
    transition_weights_2AFC.set_xlabel("")
    transition_weights_2AFC.set_xticklabels(["E->E", "E->D", "D->E", "D->D"])
    transition_weights_2AFC.set_ylabel("Transition weight")
    transition_weights_2AFC.tick_params(axis="x", rotation=45)
    if not mount_figure:
        transition_weights_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_transition_weights").with_suffix(f".{format}"))
    transition_weights_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    model_type,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    transition_orders,
    transition_plot_dfs,
):
    plt.figure(figsize=fig_size(3, 1), constrained_layout=True)
    transition_weights_3CDR = plt.gca() if not mount_figure else axd.get("transition_weights_3CDR", plt.gca())
    transition_weights_3CDR.clear()
    if model_type != "glmhmmt":
        transition_weights_3CDR.set_axis_off()
        if not mount_figure:
            transition_weights_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_transition_weights").with_suffix(f".{format}"))
        transition_weights_3CDR
    else:
        sns.boxplot(
            data=transition_plot_dfs["MCDR"],
            x="transition_feature_label",
            y="weight",
            order=transition_orders["MCDR"],
            color="tab:gray",
            ax=transition_weights_3CDR,
            **boxplot_STYLE,
        )
        transition_weights_3CDR.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        add_one_sample_zero_annotations(transition_weights_3CDR, transition_plot_dfs["MCDR"], x="transition_feature_label", y="weight", order=transition_orders["MCDR"])
    transition_weights_3CDR.set_title(task_labels["MCDR"])
    transition_weights_3CDR.set_xlabel("")
    transition_weights_3CDR.set_ylabel("Transition weight")
    transition_weights_3CDR.tick_params(axis="x", rotation=30)
    if not mount_figure:
        transition_weights_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_transition_weights").with_suffix(f".{format}"))
    transition_weights_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Psychometrics By State
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
    delay_order,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    psychometric_dfs,
    psychometric_x_labels,
    sns,
    state_palette,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    psychometric_by_state_2ADC = plt.gca() if not mount_figure else axd.get("psychometric_by_state_2ADC", plt.gca())
    psychometric_by_state_2ADC.clear()
    _plot_df = psychometric_dfs["2AFC_delay"]
    _model = _plot_df[_plot_df["source"] == "Model"]
    _data = _plot_df[_plot_df["source"] == "Data"]
    sns.lineplot(
        data=_model,
        x="x_position",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
        palette=state_palette,
        ax=psychometric_by_state_2ADC,
    )
    sns.pointplot(
        data=_data,
        x="x_position",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        markers="o",
        linestyles="none",
        native_scale=True,
        palette=state_palette,
        legend=False,
        ax=psychometric_by_state_2ADC,
    )
    psychometric_by_state_2ADC.set_xticks(range(len(delay_order)))
    psychometric_by_state_2ADC.set_xticklabels([f"{value:g}" for value in delay_order])
    psychometric_by_state_2ADC.axhline(0.5, color="0.5", linestyle="--")
    # psychometric_by_state_2ADC.set_title(task_labels["2AFC_delay"])
    psychometric_by_state_2ADC.set_xlabel(psychometric_x_labels["2AFC_delay"])
    psychometric_by_state_2ADC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_state_2ADC.legend(frameon=False, title="")
    psychometric_by_state_2ADC.set_ylim(0,1)
    psychometric_by_state_2ADC.set_yticks([0, 0.5,1], [0, 0.5,1], )
    psychometric_by_state_2ADC.legend_.remove()
    if not mount_figure:
        psychometric_by_state_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_psychometric_by_state").with_suffix(f".{format}"))
    psychometric_by_state_2ADC
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    psychometric_dfs,
    psychometric_x_labels,
    sns,
    state_palette,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    psychometric_by_state_2AFC = plt.gca() if not mount_figure else axd.get("psychometric_by_state_2AFC", plt.gca())
    psychometric_by_state_2AFC.clear()
    _plot_df = psychometric_dfs["2AFC"]
    _model = _plot_df[_plot_df["source"] == "Model"]
    _data = _plot_df[_plot_df["source"] == "Data"]
    sns.lineplot(
        data=_model,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
        palette=state_palette,
        ax=psychometric_by_state_2AFC,
    )
    sns.pointplot(
        data=_data,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        markers="o",
        ms = 5,
        linestyles="none",
        native_scale=True,
        palette=state_palette,
        legend=False,
        ax=psychometric_by_state_2AFC,
    )
    xticks = sorted(psychometric_dfs["2AFC"]["x_numeric"].dropna().unique())
    xticks = [float(tick) for tick in xticks]
    psychometric_by_state_2AFC.set_xticks(xticks)
    psychometric_by_state_2AFC.set_xticklabels([f"{tick:g}" if abs(float(tick)) not in {2.0, 4.0} else "" for tick in xticks])
    psychometric_by_state_2AFC.axhline(0.5, color="0.5", linestyle="--")
    # psychometric_by_state_2AFC.set_title(task_labels["2AFC"])
    psychometric_by_state_2AFC.set_xlabel(psychometric_x_labels["2AFC"])
    psychometric_by_state_2AFC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_state_2AFC.legend(frameon=False, title="")
    psychometric_by_state_2AFC.set_yticks([0, 0.5,1], [0, 0.5,1], )
    psychometric_by_state_2AFC.legend_.remove()
    if not mount_figure:
        psychometric_by_state_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_psychometric_by_state").with_suffix(f".{format}"))
    psychometric_by_state_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
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
    psychometric_dfs,
    psychometric_orders,
    psychometric_x_labels,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    psychometric_by_state_3CDR = plt.gca() if not mount_figure else axd.get("psychometric_by_state_3CDR", plt.gca())
    psychometric_by_state_3CDR.clear()
    _plot_df = psychometric_dfs["MCDR"]
    _model = _plot_df[_plot_df["source"] == "Model"]
    _data = _plot_df[_plot_df["source"] == "Data"]
    sns.lineplot(
        data=_model,
        x="x_position",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
        palette=state_palette,
        ax=psychometric_by_state_3CDR,
    )
    sns.pointplot(
        data=_data,
        x="x_position",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        markers="o",
        linestyles="none",
        native_scale=True,
        palette=state_palette,
        legend=False,
        ax=psychometric_by_state_3CDR,
    )
    psychometric_by_state_3CDR.set_xticks(range(len(psychometric_orders["MCDR"])))
    psychometric_by_state_3CDR.set_xticklabels(psychometric_orders["MCDR"])
    psychometric_by_state_3CDR.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    psychometric_by_state_3CDR.set_title(task_labels["MCDR"])
    psychometric_by_state_3CDR.set_xlabel(psychometric_x_labels["MCDR"])
    psychometric_by_state_3CDR.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_state_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        psychometric_by_state_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_psychometric_by_state").with_suffix(f".{format}"))
    psychometric_by_state_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Psychometrics By Action Trace
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
    choice_lag_psychometric_dfs,
    feature_labels,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    psychometric_by_action_trace_2ADC = plt.gca() if not mount_figure else axd.get("psychometric_by_action_trace_2ADC", plt.gca())
    psychometric_by_action_trace_2ADC.clear()
    _plot_df = choice_lag_psychometric_dfs["2AFC_delay"]
    _model = _plot_df[_plot_df["source"] == "Model"]
    _data = _plot_df[_plot_df["source"] == "Data"]
    sns.lineplot(
        data=_model,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
        palette=state_palette,
        ax=psychometric_by_action_trace_2ADC,
    )
    sns.pointplot(
        data=_data,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        markers="o",
        linestyles="none",
        native_scale=True,
        palette=state_palette,
        legend=False,
        ax=psychometric_by_action_trace_2ADC,
    )
    # _xticks = sorted(choice_lag_psychometric_dfs["2AFC_delay"]["x_numeric"].dropna().unique())
    # _xticks = [float(tick) for tick in _xticks]
    # psychometric_by_action_trace_2ADC.set_xticks(_xticks)
    # psychometric_by_action_trace_2ADC.set_xticklabels([f"{tick:g}" for tick in _xticks])
    psychometric_by_action_trace_2ADC.axhline(0.5, color="0.5", linestyle="--")
    # psychometric_by_action_trace_2ADC.set_title(task_labels["2AFC_delay"])
    psychometric_by_action_trace_2ADC.set_xlabel(feature_labels["choice_lag_param"])
    psychometric_by_action_trace_2ADC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_action_trace_2ADC.legend(frameon=False, title="")
    psychometric_by_action_trace_2ADC.set_ylim(0,1)
    psychometric_by_action_trace_2ADC.set_yticks([0, 0.5,1], [0, 0.5,1], )
    psychometric_by_action_trace_2ADC.legend_.remove()
    if not mount_figure:
        psychometric_by_action_trace_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_psychometric_by_action_trace").with_suffix(f".{format}"))
    psychometric_by_action_trace_2ADC
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
    choice_lag_psychometric_dfs,
    feature_labels,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    psychometric_by_action_trace_2AFC = plt.gca() if not mount_figure else axd.get("psychometric_by_action_trace_2AFC", plt.gca())
    psychometric_by_action_trace_2AFC.clear()
    _plot_df = choice_lag_psychometric_dfs["2AFC"]
    _model = _plot_df[_plot_df["source"] == "Model"]
    _data = _plot_df[_plot_df["source"] == "Data"]
    sns.lineplot(
        data=_model,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
        palette=state_palette,
        ax=psychometric_by_action_trace_2AFC,
    )
    sns.pointplot(
        data=_data,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        markers="o",
        linestyles="none",
        native_scale=True,
        palette=state_palette,
        legend=False,
        ax=psychometric_by_action_trace_2AFC,
    )
    # _xticks = sorted(choice_lag_psychometric_dfs["2AFC"]["x_numeric"].dropna().unique())
    # _xticks = [float(tick) for tick in _xticks]
    # psychometric_by_action_trace_2AFC.set_xticks(_xticks)
    # psychometric_by_action_trace_2AFC.set_xticklabels([f"{tick:g}" for tick in _xticks])
    psychometric_by_action_trace_2AFC.axhline(0.5, color="0.5", linestyle="--")
    # psychometric_by_action_trace_2AFC.set_title(task_labels["2AFC"])
    psychometric_by_action_trace_2AFC.set_xlabel(feature_labels["choice_lag_param"])
    psychometric_by_action_trace_2AFC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_action_trace_2AFC.legend(frameon=False, title="")
    psychometric_by_action_trace_2AFC.set_ylim(0,1)
    psychometric_by_action_trace_2AFC.set_yticks([0, 0.5,1], [0, 0.5,1], )
    psychometric_by_action_trace_2AFC.legend_.remove()
    if not mount_figure:
        psychometric_by_action_trace_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_psychometric_by_action_trace").with_suffix(f".{format}"))
    psychometric_by_action_trace_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    axd,
    choice_lag_psychometric_dfs,
    feature_labels,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    psychometric_by_action_trace_3CDR = plt.gca() if not mount_figure else axd.get("psychometric_by_action_trace_3CDR", plt.gca())
    psychometric_by_action_trace_3CDR.clear()
    _plot_df = choice_lag_psychometric_dfs["MCDR"]
    _model = _plot_df[_plot_df["source"] == "Model"]
    _data = _plot_df[_plot_df["source"] == "Data"]
    sns.lineplot(
        data=_model,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
        palette=state_palette,
        ax=psychometric_by_action_trace_3CDR,
    )
    sns.pointplot(
        data=_data,
        x="x_numeric",
        y="p_right",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        markers="o",
        linestyles="none",
        native_scale=True,
        palette=state_palette,
        legend=False,
        ax=psychometric_by_action_trace_3CDR,
    )
    # _xticks = sorted(choice_lag_psychometric_dfs["MCDR"]["x_numeric"].dropna().unique())
    # _xticks = [float(tick) for tick in _xticks]
    # psychometric_by_action_trace_3CDR.set_xticks(_xticks)
    # psychometric_by_action_trace_3CDR.set_xticklabels([f"{tick:g}" for tick in _xticks])
    psychometric_by_action_trace_3CDR.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    psychometric_by_action_trace_3CDR.set_title(task_labels["MCDR"])
    psychometric_by_action_trace_3CDR.set_xlabel(feature_labels["choice_lag_param"])
    psychometric_by_action_trace_3CDR.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_action_trace_3CDR.legend(frameon=False, title="")
    psychometric_by_action_trace_3CDR.set_ylim(0,1)
    if not mount_figure:
        psychometric_by_action_trace_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_psychometric_by_action_trace").with_suffix(f".{format}"))
    psychometric_by_action_trace_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy By State
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
    accuracy_hue_orders,
    accuracy_plot_dfs,
    add_paired_state_annotation,
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_label_orders,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    accuracy_by_state_2ADC = plt.gca() if not mount_figure else axd.get("accuracy_by_state_2ADC", plt.gca())
    accuracy_by_state_2ADC.clear()
    sns.boxplot(
        data=accuracy_plot_dfs["2AFC_delay"],
        x="task_label",
        y="accuracy",
        hue="state_label",
        order=task_label_orders["2AFC_delay"],
        hue_order=accuracy_hue_orders["2AFC_delay"],
        palette=state_palette,
        ax=accuracy_by_state_2ADC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(accuracy_by_state_2ADC, accuracy_plot_dfs["2AFC_delay"], x="task_label", y="accuracy", order=task_label_orders["2AFC_delay"])
    add_paired_state_annotation(accuracy_by_state_2ADC, accuracy_plot_dfs["2AFC_delay"], x="task_label", y="accuracy", order=task_label_orders["2AFC_delay"])
    accuracy_by_state_2ADC.set_xlabel("")
    accuracy_by_state_2ADC.set_ylabel("Accuracy")
    accuracy_by_state_2ADC.legend(frameon=False, title="")
    if not mount_figure:
        accuracy_by_state_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_accuracy_by_state").with_suffix(f".{format}"))
    accuracy_by_state_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    accuracy_hue_orders,
    accuracy_plot_dfs,
    add_paired_state_annotation,
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_label_orders,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    accuracy_by_state_2AFC = plt.gca() if not mount_figure else axd.get("accuracy_by_state_2AFC", plt.gca())
    accuracy_by_state_2AFC.clear()
    sns.boxplot(
        data=accuracy_plot_dfs["2AFC"],
        x="task_label",
        y="accuracy",
        hue="state_label",
        order=task_label_orders["2AFC"],
        hue_order=accuracy_hue_orders["2AFC"],
        palette=state_palette,
        ax=accuracy_by_state_2AFC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(accuracy_by_state_2AFC, accuracy_plot_dfs["2AFC"], x="task_label", y="accuracy", order=task_label_orders["2AFC"])
    add_paired_state_annotation(accuracy_by_state_2AFC, accuracy_plot_dfs["2AFC"], x="task_label", y="accuracy", order=task_label_orders["2AFC"])
    accuracy_by_state_2AFC.set_xlabel("")
    accuracy_by_state_2AFC.set_ylabel("Accuracy")
    accuracy_by_state_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        accuracy_by_state_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_accuracy_by_state").with_suffix(f".{format}"))
    accuracy_by_state_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    accuracy_hue_orders,
    accuracy_plot_dfs,
    add_paired_state_annotation,
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_label_orders,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    accuracy_by_state_3CDR = plt.gca() if not mount_figure else axd.get("accuracy_by_state_3CDR", plt.gca())
    accuracy_by_state_3CDR.clear()
    sns.boxplot(
        data=accuracy_plot_dfs["MCDR"],
        x="task_label",
        y="accuracy",
        hue="state_label",
        order=task_label_orders["MCDR"],
        hue_order=accuracy_hue_orders["MCDR"],
        palette=state_palette,
        ax=accuracy_by_state_3CDR,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(accuracy_by_state_3CDR, accuracy_plot_dfs["MCDR"], x="task_label", y="accuracy", order=task_label_orders["MCDR"])
    add_paired_state_annotation(accuracy_by_state_3CDR, accuracy_plot_dfs["MCDR"], x="task_label", y="accuracy", order=task_label_orders["MCDR"])
    accuracy_by_state_3CDR.set_xlabel("")
    accuracy_by_state_3CDR.set_ylabel("Accuracy")
    accuracy_by_state_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        accuracy_by_state_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_accuracy_by_state").with_suffix(f".{format}"))
    accuracy_by_state_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Occupancy By State
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
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    occupancy_annotation_dfs,
    occupancy_hue_orders,
    occupancy_plot_dfs,
    path_panels,
    plt,
    sns,
    state_palette,
    task_label_orders,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    occupancy_by_state_2ADC = plt.gca() if not mount_figure else axd.get("occupancy_by_state_2ADC", plt.gca())
    occupancy_by_state_2ADC.clear()
    sns.boxplot(
        data=occupancy_plot_dfs["2AFC_delay"],
        x="task_label",
        y="occupancy",
        hue="state_label",
        order=task_label_orders["2AFC_delay"],
        hue_order=occupancy_hue_orders["2AFC_delay"],
        palette=state_palette,
        ax=occupancy_by_state_2ADC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(occupancy_by_state_2ADC, occupancy_annotation_dfs["2AFC_delay"], x="task_label", y="occupancy", order=task_label_orders["2AFC_delay"])
    occupancy_by_state_2ADC.set_xlabel("")
    occupancy_by_state_2ADC.set_ylabel("Occupancy")
    occupancy_by_state_2ADC.legend(frameon=False, title="")
    if not mount_figure:
        occupancy_by_state_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_occupancy_by_state").with_suffix(f".{format}"))
    occupancy_by_state_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    occupancy_annotation_dfs,
    occupancy_hue_orders,
    occupancy_plot_dfs,
    path_panels,
    plt,
    sns,
    state_palette,
    task_label_orders,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    occupancy_by_state_2AFC = plt.gca() if not mount_figure else axd.get("occupancy_by_state_2AFC", plt.gca())
    occupancy_by_state_2AFC.clear()
    sns.boxplot(
        data=occupancy_plot_dfs["2AFC"],
        x="task_label",
        y="occupancy",
        hue="state_label",
        order=task_label_orders["2AFC"],
        hue_order=occupancy_hue_orders["2AFC"],
        palette=state_palette,
        ax=occupancy_by_state_2AFC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(occupancy_by_state_2AFC, occupancy_annotation_dfs["2AFC"], x="task_label", y="occupancy", order=task_label_orders["2AFC"])
    occupancy_by_state_2AFC.set_xlabel("")
    occupancy_by_state_2AFC.set_ylabel("Occupancy")
    occupancy_by_state_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        occupancy_by_state_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_occupancy_by_state").with_suffix(f".{format}"))
    occupancy_by_state_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    mount_figure,
    occupancy_annotation_dfs,
    occupancy_hue_orders,
    occupancy_plot_dfs,
    path_panels,
    plt,
    sns,
    state_palette,
    task_label_orders,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    occupancy_by_state_3CDR = plt.gca() if not mount_figure else axd.get("occupancy_by_state_3CDR", plt.gca())
    occupancy_by_state_3CDR.clear()
    sns.boxplot(
        data=occupancy_plot_dfs["MCDR"],
        x="task_label",
        y="occupancy",
        hue="state_label",
        order=task_label_orders["MCDR"],
        hue_order=occupancy_hue_orders["MCDR"],
        palette=state_palette,
        ax=occupancy_by_state_3CDR,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(occupancy_by_state_3CDR, occupancy_annotation_dfs["MCDR"], x="task_label", y="occupancy", order=task_label_orders["MCDR"])
    occupancy_by_state_3CDR.set_xlabel("")
    occupancy_by_state_3CDR.set_ylabel("Occupancy")
    occupancy_by_state_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        occupancy_by_state_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_occupancy_by_state").with_suffix(f".{format}"))
    occupancy_by_state_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean Traces
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
    trace_dfs,
):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    mean_state_traces_2ADC = plt.gca() if not mount_figure else axd.get("mean_state_traces_2ADC", plt.gca())
    mean_state_traces_2ADC.clear()
    sns.lineplot(
        data=trace_dfs["2AFC_delay"],
        x="trial_bin",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=mean_state_traces_2ADC,
    )
    mean_state_traces_2ADC.set_title(task_labels["2AFC_delay"])
    mean_state_traces_2ADC.set_xlabel("Normalized session time")
    mean_state_traces_2ADC.set_ylabel("State posterior")
    mean_state_traces_2ADC.set_ylim(0,1)
    mean_state_traces_2ADC.legend(frameon=False, title="")
    if not mount_figure:
        mean_state_traces_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_mean_state_traces").with_suffix(f".{format}"))
    mean_state_traces_2ADC
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
    trace_dfs,
):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    mean_state_traces_2AFC = plt.gca() if not mount_figure else axd.get("mean_state_traces_2AFC", plt.gca())
    mean_state_traces_2AFC.clear()
    sns.lineplot(
        data=trace_dfs["2AFC"],
        x="trial_bin",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=mean_state_traces_2AFC,
    )
    mean_state_traces_2AFC.set_title(task_labels["2AFC"])
    mean_state_traces_2AFC.set_xlabel("Normalized session time")
    mean_state_traces_2AFC.set_ylabel("State posterior")
    mean_state_traces_2AFC.set_ylim(0,1)
    mean_state_traces_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        mean_state_traces_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_mean_state_traces").with_suffix(f".{format}"))
    mean_state_traces_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
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
    sns,
    state_palette,
    task_labels,
    trace_dfs,
):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    mean_state_traces_3CDR = plt.gca() if not mount_figure else axd.get("mean_state_traces_3CDR", plt.gca())
    mean_state_traces_3CDR.clear()
    sns.lineplot(
        data=trace_dfs["MCDR"],
        x="trial_bin",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=mean_state_traces_3CDR,
    )
    mean_state_traces_3CDR.set_title(task_labels["MCDR"])
    mean_state_traces_3CDR.set_xlabel("Normalized session time")
    mean_state_traces_3CDR.set_ylabel("State posterior")
    mean_state_traces_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        mean_state_traces_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_mean_state_traces").with_suffix(f".{format}"))
    mean_state_traces_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RTs And Licks By State
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    Annotator,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    metric_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    rt_by_state_2AFC = plt.gca() if not mount_figure else axd.get("rt_by_state_2AFC", plt.gca())
    rt_by_state_2AFC.clear()

    sns.boxplot(
        data=metric_dfs["2AFC"][metric_dfs["2AFC"]["metric"].eq("RT")],
        x="state_label",
        y="value",
        order=state_order,
        palette=state_palette,
        ax=rt_by_state_2AFC,
        gap=0.25,
        **boxplot_STYLE,
    )

    # paired annotation directly between the two x categories
    _annotator = Annotator(
        rt_by_state_2AFC,
        [("Engaged", "Disengaged")],
        data=metric_dfs["2AFC"][metric_dfs["2AFC"]["metric"].eq("RT")],
        x="state_label",
        y="value",
        order=state_order,
    )
    _annotator.configure(
        test="t-test_paired",
        text_format="star",
        line_height=0,
        verbose=False,
    ).apply_and_annotate()

    rt_by_state_2AFC.set_xlabel("")
    rt_by_state_2AFC.set_ylabel("RT")
    rt_by_state_2AFC.set_xticklabels(["Eng.", "Dis."])
    rt_by_state_2AFC.set_ylim(bottom=0)

    if not mount_figure:
        rt_by_state_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_rt_by_state").with_suffix(f".{format}"))

    rt_by_state_2AFC
    return


@app.cell
def _(axd, fig_size, format, mount_figure, path_panels, plt):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    nlicks_by_state_2ADC = plt.gca() if not mount_figure else axd.get("nlicks_by_state_2ADC", plt.gca())
    nlicks_by_state_2ADC.clear()
    nlicks_by_state_2ADC.set_axis_off()
    nlicks_by_state_2ADC.text(0.5, 0.5, "nLicks", ha="center", va="center", transform=nlicks_by_state_2ADC.transAxes)
    if not mount_figure:
        nlicks_by_state_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_nlicks_by_state").with_suffix(f".{format}"))
    nlicks_by_state_2ADC
    return


@app.cell
def _(
    Annotator,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    metric_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    nlicks_by_state_2AFC = plt.gca() if not mount_figure else axd.get("nlicks_by_state_2AFC", plt.gca())
    nlicks_by_state_2AFC.clear()

    sns.boxplot(
        data=metric_dfs["2AFC"][metric_dfs["2AFC"]["metric"].eq("nLicks")],
        x="state_label",
        y="value",
        order=state_order,
        palette=state_palette,
        ax=nlicks_by_state_2AFC,
        gap=0.25,
        **boxplot_STYLE,
    )

    _annotator = Annotator(
        nlicks_by_state_2AFC,
        [("Engaged", "Disengaged")],
        data=metric_dfs["2AFC"][metric_dfs["2AFC"]["metric"].eq("nLicks")],
        x="state_label",
        y="value",
        order=state_order,
    )
    _annotator.configure(
        test="t-test_paired",
        text_format="star",
        line_height=0,
        verbose=False,
    ).apply_and_annotate()

    nlicks_by_state_2AFC.set_xlabel("")
    nlicks_by_state_2AFC.set_ylabel("nLicks")
    nlicks_by_state_2AFC.set_ylim(bottom=0)
    nlicks_by_state_2AFC.set_xticklabels(["Eng.", "Dis."])

    if not mount_figure:
        nlicks_by_state_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_nlicks_by_state").with_suffix(f".{format}"))

    nlicks_by_state_2AFC
    return


@app.cell
def _(
    Annotator,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    metric_dfs,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_order,
    state_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    ili_by_state_2AFC = plt.gca() if not mount_figure else axd.get("ili_by_state_2AFC", plt.gca())
    ili_by_state_2AFC.clear()

    sns.boxplot(
        data=metric_dfs["2AFC"][metric_dfs["2AFC"]["metric"].eq("ILI")],
        x="state_label",
        y="value",
        order=state_order,
        palette=state_palette,
        ax=ili_by_state_2AFC,
        gap=0.25,
        **boxplot_STYLE,
    )

    # paired annotation directly between the two x categories
    _annotator = Annotator(
        ili_by_state_2AFC,
        [("Engaged", "Disengaged")],
        data=metric_dfs["2AFC"][metric_dfs["2AFC"]["metric"].eq("ILI")],
        x="state_label",
        y="value",
        order=state_order,
    )
    _annotator.configure(
        test="t-test_paired",
        text_format="star",
        line_height=0,
        verbose=False,
    ).apply_and_annotate()

    ili_by_state_2AFC.set_xlabel("")
    ili_by_state_2AFC.set_ylabel("ILI")
    ili_by_state_2AFC.set_ylim(bottom=0)

    if not mount_figure:
        ili_by_state_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_rt_by_state").with_suffix(f".{format}"))

    ili_by_state_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cumulative Dwell-Time Probability
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
    dwell_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_time_cdf_2ADC = plt.gca() if not mount_figure else axd.get("dwell_time_cdf_2ADC", plt.gca())
    dwell_time_cdf_2ADC.clear()
    sns.ecdfplot(
        data=dwell_dfs["2AFC_delay"],
        x="dwell_trials",
        hue="state_label",
        palette=state_palette,
        ax=dwell_time_cdf_2ADC,
    )
    dwell_time_cdf_2ADC.set_title(task_labels["2AFC_delay"])
    dwell_time_cdf_2ADC.set_xlabel("Dwell time (trials)")
    dwell_time_cdf_2ADC.set_ylabel("Cumulative probability")
    dwell_time_cdf_2ADC.legend(frameon=False, title="")
    if not mount_figure:
        dwell_time_cdf_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_dwell_time_cdf").with_suffix(f".{format}"))
    dwell_time_cdf_2ADC
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
    dwell_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_time_cdf_2AFC = plt.gca() if not mount_figure else axd.get("dwell_time_cdf_2AFC", plt.gca())
    dwell_time_cdf_2AFC.clear()
    sns.ecdfplot(
        data=dwell_dfs["2AFC"],
        x="dwell_trials",
        hue="state_label",
        palette=state_palette,
        ax=dwell_time_cdf_2AFC,
    )
    dwell_time_cdf_2AFC.set_title(task_labels["2AFC"])
    dwell_time_cdf_2AFC.set_xlabel("Dwell time (trials)")
    dwell_time_cdf_2AFC.set_ylabel("Cumulative probability")
    dwell_time_cdf_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        dwell_time_cdf_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_dwell_time_cdf").with_suffix(f".{format}"))
    dwell_time_cdf_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    axd,
    dwell_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    dwell_time_cdf_3CDR = plt.gca() if not mount_figure else axd.get("dwell_time_cdf_3CDR", plt.gca())
    dwell_time_cdf_3CDR.clear()
    sns.ecdfplot(
        data=dwell_dfs["MCDR"],
        x="dwell_trials",
        hue="state_label",
        palette=state_palette,
        ax=dwell_time_cdf_3CDR,
    )
    dwell_time_cdf_3CDR.set_title(task_labels["MCDR"])
    dwell_time_cdf_3CDR.set_xlabel("Dwell time (trials)")
    dwell_time_cdf_3CDR.set_ylabel("Cumulative probability")
    dwell_time_cdf_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        dwell_time_cdf_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_dwell_time_cdf").with_suffix(f".{format}"))
    dwell_time_cdf_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State Switch Histogram By Animal
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    switch_hist_dfs,
    switch_hist_summary_dfs,
    task_labels,
    task_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    state_switch_histogram_by_animal_2ADC = plt.gca() if not mount_figure else axd.get("state_switch_histogram_by_animal_2ADC", plt.gca())
    state_switch_histogram_by_animal_2ADC.clear()
    sns.histplot(
        data=switch_hist_dfs["2AFC_delay"],
        x="n_switches",
        weights="switch_probability",
        discrete=True,
        stat="probability",
        shrink=0.85,
        color=task_palette["2AFC_delay"],
        edgecolor="white",
        ax=state_switch_histogram_by_animal_2ADC,
    )
    state_switch_histogram_by_animal_2ADC.errorbar(
        switch_hist_summary_dfs["2AFC_delay"]["n_switches"],
        switch_hist_summary_dfs["2AFC_delay"]["switch_probability"],
        yerr=switch_hist_summary_dfs["2AFC_delay"]["sem"],
        fmt="none",
        color="0.2",
        capsize=2,
        linewidth=0.8,
    )
    state_switch_histogram_by_animal_2ADC.set_xticks(sorted(switch_hist_dfs["2AFC_delay"]["n_switches"].dropna().astype(int).unique()))
    state_switch_histogram_by_animal_2ADC.set_title(task_labels["2AFC_delay"])
    state_switch_histogram_by_animal_2ADC.set_xlabel("Mean switches per session")
    state_switch_histogram_by_animal_2ADC.set_ylabel("Animal probability")
    if not mount_figure:
        state_switch_histogram_by_animal_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_state_switch_histogram_by_animal").with_suffix(f".{format}"))
    state_switch_histogram_by_animal_2ADC
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
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    switch_hist_dfs,
    switch_hist_summary_dfs,
    task_labels,
    task_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    state_switch_histogram_by_animal_2AFC = plt.gca() if not mount_figure else axd.get("state_switch_histogram_by_animal_2AFC", plt.gca())
    state_switch_histogram_by_animal_2AFC.clear()
    sns.histplot(
        data=switch_hist_dfs["2AFC"],
        x="n_switches",
        weights="switch_probability",
        discrete=True,
        stat="probability",
        shrink=0.85,
        color=task_palette["2AFC"],
        edgecolor="white",
        ax=state_switch_histogram_by_animal_2AFC,
    )
    state_switch_histogram_by_animal_2AFC.errorbar(
        switch_hist_summary_dfs["2AFC"]["n_switches"],
        switch_hist_summary_dfs["2AFC"]["switch_probability"],
        yerr=switch_hist_summary_dfs["2AFC"]["sem"],
        fmt="none",
        color="0.2",
        capsize=2,
        linewidth=0.8,
    )
    state_switch_histogram_by_animal_2AFC.set_xticks(sorted(switch_hist_dfs["2AFC"]["n_switches"].dropna().astype(int).unique()))
    state_switch_histogram_by_animal_2AFC.set_title(task_labels["2AFC"])
    state_switch_histogram_by_animal_2AFC.set_xlabel("Mean switches per session")
    state_switch_histogram_by_animal_2AFC.set_ylabel("Animal probability")
    if not mount_figure:
        state_switch_histogram_by_animal_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_state_switch_histogram_by_animal").with_suffix(f".{format}"))
    state_switch_histogram_by_animal_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
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
    sns,
    switch_hist_dfs,
    switch_hist_summary_dfs,
    task_labels,
    task_palette,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    state_switch_histogram_by_animal_3CDR = plt.gca() if not mount_figure else axd.get("state_switch_histogram_by_animal_3CDR", plt.gca())
    state_switch_histogram_by_animal_3CDR.clear()
    sns.histplot(
        data=switch_hist_dfs["MCDR"],
        x="n_switches",
        weights="switch_probability",
        discrete=True,
        stat="probability",
        shrink=0.85,
        color=task_palette["MCDR"],
        edgecolor="white",
        ax=state_switch_histogram_by_animal_3CDR,
    )
    state_switch_histogram_by_animal_3CDR.errorbar(
        switch_hist_summary_dfs["MCDR"]["n_switches"],
        switch_hist_summary_dfs["MCDR"]["switch_probability"],
        yerr=switch_hist_summary_dfs["MCDR"]["sem"],
        fmt="none",
        color="0.2",
        capsize=2,
        linewidth=0.8,
    )
    state_switch_histogram_by_animal_3CDR.set_xticks(sorted(switch_hist_dfs["MCDR"]["n_switches"].dropna().astype(int).unique()))
    state_switch_histogram_by_animal_3CDR.set_title(task_labels["MCDR"])
    state_switch_histogram_by_animal_3CDR.set_xlabel("Mean switches per session")
    state_switch_histogram_by_animal_3CDR.set_ylabel("Animal probability")
    if not mount_figure:
        state_switch_histogram_by_animal_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_state_switch_histogram_by_animal").with_suffix(f".{format}"))
    state_switch_histogram_by_animal_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posteriors Around A Change
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
    change_posterior_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    posteriors_around_change_2ADC = plt.gca() if not mount_figure else axd.get("posteriors_around_change_2ADC", plt.gca())
    posteriors_around_change_2ADC.clear()
    sns.lineplot(
        data=change_posterior_dfs["2AFC_delay"],
        x="lag",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=posteriors_around_change_2ADC,
    )
    posteriors_around_change_2ADC.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
    posteriors_around_change_2ADC.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    posteriors_around_change_2ADC.set_title(task_labels["2AFC_delay"])
    posteriors_around_change_2ADC.set_xlabel("Trials from state change")
    posteriors_around_change_2ADC.set_ylabel("State posterior")
    posteriors_around_change_2ADC.legend(frameon=False, title="")
    if not mount_figure:
        posteriors_around_change_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_posteriors_around_change").with_suffix(f".{format}"))
    posteriors_around_change_2ADC
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
    change_posterior_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    posteriors_around_change_2AFC = plt.gca() if not mount_figure else axd.get("posteriors_around_change_2AFC", plt.gca())
    posteriors_around_change_2AFC.clear()
    sns.lineplot(
        data=change_posterior_dfs["2AFC"],
        x="lag",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=posteriors_around_change_2AFC,
    )
    posteriors_around_change_2AFC.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
    posteriors_around_change_2AFC.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    posteriors_around_change_2AFC.set_title(task_labels["2AFC"])
    posteriors_around_change_2AFC.set_xlabel("Trials from state change")
    posteriors_around_change_2AFC.set_ylabel("State posterior")
    posteriors_around_change_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        posteriors_around_change_2AFC.figure.savefig((path_panels / "2AFC_glmhmmt_posteriors_around_change").with_suffix(f".{format}"))
    posteriors_around_change_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _(
    axd,
    change_posterior_dfs,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    plt.figure(figsize=fig_size(1, 2), constrained_layout=True)
    posteriors_around_change_3CDR = plt.gca() if not mount_figure else axd.get("posteriors_around_change_3CDR", plt.gca())
    posteriors_around_change_3CDR.clear()
    sns.lineplot(
        data=change_posterior_dfs["MCDR"],
        x="lag",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=posteriors_around_change_3CDR,
    )
    posteriors_around_change_3CDR.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
    posteriors_around_change_3CDR.set_title(task_labels["MCDR"])
    posteriors_around_change_3CDR.set_xlabel("Trials from state change")
    posteriors_around_change_3CDR.set_ylabel("State posterior")
    posteriors_around_change_3CDR.legend(frameon=False, title="")
    if not mount_figure:
        posteriors_around_change_3CDR.figure.savefig((path_panels / "MCDR_glmhmmt_posteriors_around_change").with_suffix(f".{format}"))
    posteriors_around_change_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single Sessions
    """)
    return


@app.cell
def _(
    adapters,
    axd,
    build_session_repetition_data,
    fig_size,
    format,
    mount_figure,
    path_panels,
    pd,
    pl,
    plot_dfs,
    plt,
    state_palette,
    task_labels,
):
    _subject = "C37"
    _session = "35"
    _task_df = plot_dfs["2AFC_delay"]
    if _task_df.filter((pl.col("subject").cast(pl.Utf8) == _subject) & (pl.col("session").cast(pl.Utf8) == _session)).height == 0:
        _available = (
            _task_df
            .select(pl.col("subject").cast(pl.Utf8).alias("subject"), pl.col("session").cast(pl.Utf8).alias("session"))
            .unique()
            .sort(["subject", "session"])
        )
        _subject = _available["subject"][0]
        _session = _available["session"][0]
    session_repetition_data_2ADC = build_session_repetition_data(
        _task_df,
        subject=_subject,
        session=_session,
        adapter=adapters["2AFC_delay"],
        window=20,
    )
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_2ADC = plt.gca() if not mount_figure else axd.get("single_session_2ADC", plt.gca())
    single_session_2ADC.clear()
    _trial_col = "trial_idx" if "trial_idx" in _task_df.columns else "trial"
    _session_pdf = (
        _task_df
        .filter((pl.col("subject").cast(pl.Utf8) == _subject) & (pl.col("session").cast(pl.Utf8) == _session))
        .sort(_trial_col)
        .to_pandas()
        .reset_index(drop=True)
    )
    _posterior_cols = [
        _col for _col in _session_pdf.columns
        if str(_col).startswith("p_state_") and str(_col).rsplit("_", 1)[-1].isdigit()
    ]
    _engaged_col = None
    if "state_idx" in _session_pdf.columns:
        _engaged_rows = _session_pdf[_session_pdf["state_label"].astype(str).eq("Engaged")].dropna(subset=["state_idx"])
        if not _engaged_rows.empty:
            _candidate = f"p_state_{int(_engaged_rows['state_idx'].iloc[0])}"
            if _candidate in _posterior_cols:
                _engaged_col = _candidate
    if _engaged_col is None and _posterior_cols:
        _engaged_rows = _session_pdf[_session_pdf["state_label"].astype(str).eq("Engaged")]
        _engaged_col = _engaged_rows[_posterior_cols].mean().idxmax() if not _engaged_rows.empty else _posterior_cols[0]
    if _engaged_col is not None:
        _p_engaged = pd.to_numeric(_session_pdf[_engaged_col], errors="coerce").iloc[: len(session_repetition_data_2ADC)]


    _x = session_repetition_data_2ADC["trial_x"].iloc[: len(_session_pdf)]
    _p_engaged = pd.to_numeric(
        _session_pdf[_engaged_col],
        errors="coerce",
    ).iloc[: len(_x)].to_numpy()
    _p_disengaged = 1 - _p_engaged
    single_session_2ADC.fill_between(
        _x,
        0,
        _p_engaged,
        color=state_palette.get("Engaged", "tab:green"),
        alpha=0.25,
        linewidth=0,
        label="Engaged",
        zorder=0,
    )
    single_session_2ADC.fill_between(
        _x,
        _p_engaged,
        _p_disengaged + _p_engaged,
        color=state_palette.get("Disengaged", "tab:gray"),
        alpha=0.20,
        linewidth=0,
        label="Disengaged",
        zorder=0,
    )

    single_session_2ADC.plot("trial_x", "response_repeat_window_fraction", color="tab:brown", linewidth=1.5, label="Rep. Choices", data=session_repetition_data_2ADC, zorder=2, alpha = 0.5)
    single_session_2ADC.plot("trial_x", "stimulus_repeat_window_fraction", color="tab:blue", linewidth=1.5, label="Rep. Stimulus", data=session_repetition_data_2ADC, zorder=2, alpha = 0.5)
    single_session_2ADC.plot("trial_x", "accuracy_window_fraction", color="black", linewidth=1.5, label="Accuracy", data=session_repetition_data_2ADC, zorder=2, alpha = 0.5)
    single_session_2ADC.set_title(task_labels["2AFC_delay"])
    single_session_2ADC.set_xlabel("Trial")
    single_session_2ADC.set_ylabel("Running fraction")
    single_session_2ADC.set_ylim(0, 1)
    single_session_2ADC.set_xlim(-0.5, len(session_repetition_data_2ADC) - 0.5)
    single_session_2ADC.legend(frameon=False, loc="upper right")
    single_session_2ADC.set_yticks([0,0.5,1],[0,0.5,1])
    if not mount_figure:
        single_session_2ADC.figure.savefig((path_panels / "2AFC_delay_single_session").with_suffix(f".{format}"))
    single_session_2ADC
    return


@app.cell
def _(
    adapters,
    axd,
    build_session_repetition_data,
    fig_size,
    format,
    mount_figure,
    path_panels,
    pd,
    pl,
    plot_dfs,
    plt,
    state_palette,
    task_labels,
):
    _subject = "335"
    _session = "335_stage_training_v2_20220329-111341"
    _task_df = plot_dfs["2AFC"]
    if _task_df.filter((pl.col("subject").cast(pl.Utf8) == _subject) & (pl.col("session").cast(pl.Utf8) == _session)).height == 0:
        _available = (
            _task_df
            .select(pl.col("subject").cast(pl.Utf8).alias("subject"), pl.col("session").cast(pl.Utf8).alias("session"))
            .unique()
            .sort(["subject", "session"])
        )
        _subject = _available["subject"][0]
        _session = _available["session"][0]
    session_repetition_data_2AFC = build_session_repetition_data(
        _task_df,
        subject=_subject,
        session=_session,
        adapter=adapters["2AFC"],
        window=20,
    )
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_2AFC = plt.gca() if not mount_figure else axd.get("single_session_2AFC", plt.gca())
    single_session_2AFC.clear()
    _trial_col = "trial_idx" if "trial_idx" in _task_df.columns else "trial"
    _session_pdf = (
        _task_df
        .filter((pl.col("subject").cast(pl.Utf8) == _subject) & (pl.col("session").cast(pl.Utf8) == _session))
        .sort(_trial_col)
        .to_pandas()
        .reset_index(drop=True)
    )
    _posterior_cols = [
        _col for _col in _session_pdf.columns
        if str(_col).startswith("p_state_") and str(_col).rsplit("_", 1)[-1].isdigit()
    ]
    _engaged_col = None
    if "state_idx" in _session_pdf.columns:
        _engaged_rows = _session_pdf[_session_pdf["state_label"].astype(str).eq("Engaged")].dropna(subset=["state_idx"])
        if not _engaged_rows.empty:
            _candidate = f"p_state_{int(_engaged_rows['state_idx'].iloc[0])}"
            if _candidate in _posterior_cols:
                _engaged_col = _candidate
    if _engaged_col is None and _posterior_cols:
        _engaged_rows = _session_pdf[_session_pdf["state_label"].astype(str).eq("Engaged")]
        _engaged_col = _engaged_rows[_posterior_cols].mean().idxmax() if not _engaged_rows.empty else _posterior_cols[0]
    if _engaged_col is not None:
        _p_engaged = pd.to_numeric(_session_pdf[_engaged_col], errors="coerce").iloc[: len(session_repetition_data_2AFC)]

    _x = session_repetition_data_2AFC["trial_x"].iloc[: len(_session_pdf)]
    _p_engaged = pd.to_numeric(
        _session_pdf[_engaged_col],
        errors="coerce",
    ).iloc[: len(_x)].to_numpy()
    _p_disengaged = 1 - _p_engaged
    single_session_2AFC.fill_between(
        _x,
        0,
        _p_engaged,
        color=state_palette.get("Engaged", "tab:green"),
        alpha=0.25,
        linewidth=0,
        label="Engaged",
        zorder=0,
    )
    single_session_2AFC.fill_between(
        _x,
        _p_engaged,
        _p_disengaged + _p_engaged,
        color=state_palette.get("Disengaged", "tab:gray"),
        alpha=0.20,
        linewidth=0,
        label="Disengaged",
        zorder=0,
    )

    single_session_2AFC.plot("trial_x", "response_repeat_window_fraction", color="tab:brown", linewidth=1.5, label="Rep. Choices", data=session_repetition_data_2AFC, zorder=2, alpha = 0.5)
    single_session_2AFC.plot("trial_x", "stimulus_repeat_window_fraction", color="tab:blue", linewidth=1.5, label="Rep. Stimulus", data=session_repetition_data_2AFC, zorder=2, alpha = 0.5)
    if "accuracy_window_fraction" in session_repetition_data_2AFC:
        single_session_2AFC.plot("trial_x", "accuracy_window_fraction", color="black", linewidth=1.5, label="Accuracy", data=session_repetition_data_2AFC, zorder=2, alpha = 0.5)
    single_session_2AFC.set_title(task_labels["2AFC"])
    single_session_2AFC.set_xlabel("Trial")
    single_session_2AFC.set_ylabel("Running fraction")
    single_session_2AFC.set_ylim(0, 1)
    single_session_2AFC.set_xlim(-0.5, len(session_repetition_data_2AFC) - 0.5)
    single_session_2AFC.legend(frameon=False, loc="upper right")
    single_session_2AFC.set_yticks([0,0.5,1],[0,0.5,1])
    if not mount_figure:
        single_session_2AFC.figure.savefig((path_panels / "2AFC_single_session").with_suffix(f".{format}"))
    single_session_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Model Comparison
    """)
    return


@app.cell
def _(MODEL_BY_TASK, load_metrics_dir_raw, paths, pl):
    def load_model_metrics(task_name, model_kind, alias, model_name):
        metrics = load_metrics_dir_raw(
            task_name=task_name,
            model_kind=model_kind,
            alias=alias,
            local_root=paths.RESULTS / "fits" / task_name / model_kind,
            label_map={model_kind: model_name},
        )
        ll_col = "test_ll_per_trial_mean" if "test_ll_per_trial_mean" in metrics.columns else "ll_per_trial"
        return (
            metrics
            .with_columns(
                pl.lit(model_name).alias("model_name"),
                pl.col(ll_col).alias("test_ll_per_trial"),
            )
            .group_by("subject", "model_name")
            .agg(
                pl.col("test_ll_per_trial").mean().alias("test_ll_per_trial"),
                pl.col("bic").mean().alias("bic"),
            )
        )

    def model_delta_df(glm_metrics, glmhmmt_metrics):
        return (
            glmhmmt_metrics
            .join(glm_metrics, on="subject", suffix="_glm", how="inner")
            .select(
                "subject",
                pl.lit("GLM-HMM-T - GLM").alias("comparison"),
                (pl.col("test_ll_per_trial") - pl.col("test_ll_per_trial_glm")).alias("Delta LL/trial"),
                (pl.col("bic") - pl.col("bic_glm")).alias("Delta BIC"),
            )
            .unpivot(
                index=["subject", "comparison"],
                variable_name="metric",
                value_name="delta",
            )
        )

    ll_bic_delta_2ADC = model_delta_df(
        load_model_metrics("2AFC_delay", "glm", "one hot2", "GLM"),
        load_model_metrics("2AFC_delay", "glmhmm", MODEL_BY_TASK["2AFC_delay"], "GLM-HMM-T"),
    )
    ll_bic_delta_2AFC = model_delta_df(
        load_model_metrics("2AFC", "glm", "one hot2", "GLM"),
        load_model_metrics("2AFC", "glmhmm", MODEL_BY_TASK["2AFC"], "GLM-HMM-T"),
    )
    return ll_bic_delta_2ADC, ll_bic_delta_2AFC


@app.cell
def _(np, pd, plot_dfs):
    def roc_auc(target, score):
        target = np.asarray(target, dtype=bool)
        score = np.asarray(score, dtype=float)
        valid = np.isfinite(score)
        target = target[valid]
        score = score[valid]
        n_pos = int(target.sum())
        n_neg = int((~target).sum())
        if target.size == 0 or n_pos == 0 or n_neg == 0:
            return np.nan
        order = np.argsort(-score, kind="mergesort")
        target_sorted = target[order]
        score_sorted = score[order]
        threshold_idxs = np.r_[np.where(np.diff(score_sorted))[0], target_sorted.size - 1]
        tps = np.cumsum(target_sorted)[threshold_idxs]
        fps = (1 + threshold_idxs) - tps
        tpr = np.r_[0.0, tps / n_pos]
        fpr = np.r_[0.0, fps / n_neg]
        auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))
        return fpr, tpr, auc

    def engaged_target(labels):
        label_text = pd.Series(labels, copy=False).astype(str).str.strip().str.lower()
        positive = label_text.eq("engaged") | label_text.str.startswith("engaged ")
        negative = label_text.eq("disengaged") | label_text.str.startswith("disengaged ")
        return positive.to_numpy(dtype=bool), (positive | negative).to_numpy(dtype=bool)


    plot_df_2AFC = plot_dfs["2AFC"].to_pandas()
    fpr_grid_2AFC = np.linspace(0, 1, 101)

    roc_metrics_2AFC = {
        "ILI": ("ILI", -1),
        "RT": ("RT", -1),
        "nLicks": ("nLicks", 1),
    }

    roc_curve_rows_2AFC = []
    roc_auc_rows_2AFC = []

    for metric_label, (metric_col, sign) in roc_metrics_2AFC.items():
        if metric_col not in plot_df_2AFC.columns:
            continue

        for subject, subject_df in plot_df_2AFC.groupby("subject", sort=True):
            target, valid_labels = engaged_target(subject_df["state_label"])
            score = sign * pd.to_numeric(subject_df[metric_col], errors="coerce").to_numpy(dtype=float)
            result = roc_auc(target[valid_labels], score[valid_labels])

            if not isinstance(result, tuple):
                continue

            fpr, tpr, auc = result
            interp_tpr = np.interp(fpr_grid_2AFC, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0

            roc_auc_rows_2AFC.append(
                {"subject": str(subject), "metric": metric_label, "auc": auc}
            )
            for fpr_value, tpr_value in zip(fpr_grid_2AFC, interp_tpr, strict=False):
                roc_curve_rows_2AFC.append(
                    {
                        "subject": str(subject),
                        "metric": metric_label,
                        "fpr": fpr_value,
                        "tpr": tpr_value,
                    }
                )

    roc_curve_df_2AFC = pd.DataFrame(roc_curve_rows_2AFC)
    roc_auc_df_2AFC = pd.DataFrame(roc_auc_rows_2AFC)
    return roc_auc_df_2AFC, roc_curve_df_2AFC


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    align_zero_to_axis_fraction,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    ll_bic_delta_2ADC,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    model_comparison_ll_2ADC = plt.gca() if not mount_figure else axd.get("model_comparison_ll_2ADC", plt.gca())
    model_comparison_ll_2ADC.clear()
    ll_bic_delta_2ADC_pdf = ll_bic_delta_2ADC.to_pandas()
    ll_delta_2ADC_pdf = ll_bic_delta_2ADC_pdf[ll_bic_delta_2ADC_pdf["metric"].eq("Delta LL/trial")]
    sns.boxplot(
        data=ll_delta_2ADC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
        ax=model_comparison_ll_2ADC,
        **boxplot_STYLE,
    )
    model_comparison_ll_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_one_sample_zero_annotations(
        model_comparison_ll_2ADC,
        ll_delta_2ADC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
    )
    model_comparison_ll_2ADC.set_title(task_labels["2AFC_delay"])
    model_comparison_ll_2ADC.set_xlabel("")
    model_comparison_ll_2ADC.set_ylabel("Delta CV test LL / trial")
    align_zero_to_axis_fraction(model_comparison_ll_2ADC, zero_fraction=0.5)
    model_comparison_ll_2ADC.set_yticks([0, 0.15], [0, 0.15])
    model_comparison_ll_2ADC.set_xticklabels(["LL"])
    if not mount_figure:
        model_comparison_ll_2ADC.figure.savefig((path_panels / "2AFC_delay_model_comparison_ll").with_suffix(f".{format}"))
    model_comparison_ll_2ADC
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    align_zero_to_axis_fraction,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    format_delta_bic_axis,
    ll_bic_delta_2ADC,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    model_comparison_bic_2ADC = plt.gca() if not mount_figure else axd.get("model_comparison_bic_2ADC", plt.gca())
    model_comparison_bic_2ADC.clear()
    _ll_bic_delta_2ADC_pdf = ll_bic_delta_2ADC.to_pandas()
    _bic_delta_2ADC_pdf = _ll_bic_delta_2ADC_pdf[_ll_bic_delta_2ADC_pdf["metric"].eq("Delta BIC")]
    sns.boxplot(
        data=_bic_delta_2ADC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
        ax=model_comparison_bic_2ADC,
        **boxplot_STYLE,
    )
    model_comparison_bic_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_one_sample_zero_annotations(
        model_comparison_bic_2ADC,
        _bic_delta_2ADC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
    )
    align_zero_to_axis_fraction(model_comparison_bic_2ADC, zero_fraction=0.5)
    format_delta_bic_axis(model_comparison_bic_2ADC)
    model_comparison_bic_2ADC.set_xlabel("")
    model_comparison_bic_2ADC.set_ylabel(r"Delta BIC ($\times 10^3$)")
    model_comparison_bic_2ADC.set_xticklabels(["BIC"])
    if not mount_figure:
        model_comparison_bic_2ADC.figure.savefig((path_panels / "2AFC_delay_model_comparison_bic").with_suffix(f".{format}"))
    model_comparison_bic_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    align_zero_to_axis_fraction,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    ll_bic_delta_2AFC,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    model_comparison_ll_2AFC = plt.gca() if not mount_figure else axd.get("model_comparison_ll_2AFC", plt.gca())
    model_comparison_ll_2AFC.clear()
    ll_bic_delta_2AFC_pdf = ll_bic_delta_2AFC.to_pandas()
    ll_delta_2AFC_pdf = ll_bic_delta_2AFC_pdf[ll_bic_delta_2AFC_pdf["metric"].eq("Delta LL/trial")]
    sns.boxplot(
        data=ll_delta_2AFC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
        ax=model_comparison_ll_2AFC,
        **boxplot_STYLE,
    )
    model_comparison_ll_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_one_sample_zero_annotations(
        model_comparison_ll_2AFC,
        ll_delta_2AFC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
    )
    model_comparison_ll_2AFC.set_title(task_labels["2AFC"])
    model_comparison_ll_2AFC.set_xlabel("")
    model_comparison_ll_2AFC.set_ylabel("Delta CV test LL / trial")
    model_comparison_ll_2AFC.set_ylim(-0.1, 0.1)
    align_zero_to_axis_fraction(model_comparison_ll_2AFC, zero_fraction=0.5)
    model_comparison_ll_2AFC.set_yticks([0, 0.1], [0, 0])
    model_comparison_ll_2AFC.set_xticklabels(["LL"])
    if not mount_figure:
        model_comparison_ll_2AFC.figure.savefig((path_panels / "2AFC_model_comparison_ll").with_suffix(f".{format}"))
    model_comparison_ll_2AFC
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    align_zero_to_axis_fraction,
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    format_delta_bic_axis,
    ll_bic_delta_2AFC,
    mount_figure,
    path_panels,
    plt,
    sns,
):
    plt.figure(figsize=fig_size(4, 1), constrained_layout=True)
    model_comparison_bic_2AFC = plt.gca() if not mount_figure else axd.get("model_comparison_bic_2AFC", plt.gca())
    model_comparison_bic_2AFC.clear()
    _ll_bic_delta_2AFC_pdf = ll_bic_delta_2AFC.to_pandas()
    _bic_delta_2AFC_pdf = _ll_bic_delta_2AFC_pdf[_ll_bic_delta_2AFC_pdf["metric"].eq("Delta BIC")]
    sns.boxplot(
        data=_bic_delta_2AFC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
        ax=model_comparison_bic_2AFC,
        **boxplot_STYLE,
    )
    model_comparison_bic_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    add_one_sample_zero_annotations(
        model_comparison_bic_2AFC,
        _bic_delta_2AFC_pdf,
        x="comparison",
        y="delta",
        order=["GLM-HMM-T - GLM"],
    )
    align_zero_to_axis_fraction(model_comparison_bic_2AFC, zero_fraction=0.5)
    format_delta_bic_axis(model_comparison_bic_2AFC)
    # model_comparison_bic_2AFC.set_title(task_labels["2AFC"])
    model_comparison_bic_2AFC.set_xlabel("")
    # model_comparison_bic_2AFC.set_ylabel(r"Delta BIC ($\times 10^3$)")
    model_comparison_bic_2AFC.set_xticklabels(["BIC"])
    if not mount_figure:
        model_comparison_bic_2AFC.figure.savefig((path_panels / "2AFC_model_comparison_bic").with_suffix(f".{format}"))
    model_comparison_bic_2AFC
    return


@app.cell
def _(plt):
    # Alternative palette
    from matplotlib import colormaps
    cmap = plt.get_cmap("Set2")
    return (cmap,)


@app.cell
def _(
    cmap,
    fig_size,
    format,
    path_panels,
    plt,
    roc_auc_df_2AFC,
    roc_curve_df_2AFC,
    sns,
):
    lick_roc_2AFC = plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    lick_roc_2AFC.clear()
    lick_roc_2AFC_ax = lick_roc_2AFC.gca()
    lick_roc_summary_2AFC = (
        roc_curve_df_2AFC[roc_curve_df_2AFC["metric"].eq("nLicks")]
        .groupby("fpr", as_index=False)
        .agg(
            mean_tpr=("tpr", "mean"),
            sem_tpr=("tpr", "sem"),
        )
    )

    lick_auc_mean_2AFC = roc_auc_df_2AFC[roc_auc_df_2AFC["metric"].eq("nLicks")]["auc"].mean()
    lick_auc_sem_2AFC = roc_auc_df_2AFC[roc_auc_df_2AFC["metric"].eq("nLicks")]["auc"].sem()
    lick_roc_2AFC_ax.plot(
        lick_roc_summary_2AFC["fpr"],
        lick_roc_summary_2AFC["mean_tpr"],
        color=cmap.colors[0],
        lw=2,
        label=f"AUC={lick_auc_mean_2AFC:.3f} +/- {lick_auc_sem_2AFC:.3f}",
    )
    lick_roc_2AFC_ax.fill_between(
        lick_roc_summary_2AFC["fpr"],
        lick_roc_summary_2AFC["mean_tpr"] - lick_roc_summary_2AFC["sem_tpr"].fillna(0),
        lick_roc_summary_2AFC["mean_tpr"] + lick_roc_summary_2AFC["sem_tpr"].fillna(0),
        color=cmap.colors[0],
        alpha=0.2,
        linewidth=0,
    )
    lick_roc_2AFC_ax.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--")
    lick_roc_2AFC_ax.set_title("2AFC nLicks")
    lick_roc_2AFC_ax.set_xlabel("False positive rate")
    lick_roc_2AFC_ax.set_ylabel("True positive rate")
    lick_roc_2AFC_ax.set_xlim(0, 1)
    lick_roc_2AFC_ax.set_ylim(0, 1)
    lick_roc_2AFC_ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=lick_roc_2AFC_ax)
    lick_roc_2AFC.savefig((path_panels / "2AFC_nlicks_state_roc").with_suffix(f".{format}"))
    lick_roc_2AFC
    return


@app.cell
def _(
    cmap,
    fig_size,
    format,
    path_panels,
    plt,
    roc_auc_df_2AFC,
    roc_curve_df_2AFC,
    sns,
):
    ili_roc_2AFC = plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    ili_roc_2AFC.clear()
    ili_roc_2AFC_ax = ili_roc_2AFC.gca()
    ili_roc_summary_2AFC = (
        roc_curve_df_2AFC[roc_curve_df_2AFC["metric"].eq("ILI")]
        .groupby("fpr", as_index=False)
        .agg(
            mean_tpr=("tpr", "mean"),
            sem_tpr=("tpr", "sem"),
        )
    )

    ili_auc_mean_2AFC = roc_auc_df_2AFC[roc_auc_df_2AFC["metric"].eq("ILI")]["auc"].mean()
    ili_auc_sem_2AFC = roc_auc_df_2AFC[roc_auc_df_2AFC["metric"].eq("ILI")]["auc"].sem()
    ili_roc_2AFC_ax.plot(
        ili_roc_summary_2AFC["fpr"],
        ili_roc_summary_2AFC["mean_tpr"],
        color=cmap.colors[1],
        lw=2,
        label=f"AUC={ili_auc_mean_2AFC:.3f} +/- {ili_auc_sem_2AFC:.3f}",
    )
    ili_roc_2AFC_ax.fill_between(
        ili_roc_summary_2AFC["fpr"],
        ili_roc_summary_2AFC["mean_tpr"] - ili_roc_summary_2AFC["sem_tpr"].fillna(0),
        ili_roc_summary_2AFC["mean_tpr"] + ili_roc_summary_2AFC["sem_tpr"].fillna(0),
        color=cmap.colors[1],
        alpha=0.2,
        linewidth=0,
    )
    ili_roc_2AFC_ax.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--")
    ili_roc_2AFC_ax.set_title(f"2AFC $-ILI$")
    ili_roc_2AFC_ax.set_xlabel("False positive rate")
    ili_roc_2AFC_ax.set_ylabel("True positive rate")
    ili_roc_2AFC_ax.set_xlim(0, 1)
    ili_roc_2AFC_ax.set_ylim(0, 1)
    ili_roc_2AFC_ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ili_roc_2AFC_ax)
    ili_roc_2AFC.savefig((path_panels / "2AFC_ili_state_roc").with_suffix(f".{format}"))
    ili_roc_2AFC
    return


@app.cell
def _(
    cmap,
    fig_size,
    format,
    path_panels,
    plt,
    roc_auc_df_2AFC,
    roc_curve_df_2AFC,
    sns,
):
    rt_roc_2AFC = plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    rt_roc_2AFC.clear()
    rt_roc_2AFC_ax = rt_roc_2AFC.gca()
    rt_roc_summary_2AFC = (
        roc_curve_df_2AFC[roc_curve_df_2AFC["metric"].eq("RT")]
        .groupby("fpr", as_index=False)
        .agg(
            mean_tpr=("tpr", "mean"),
            sem_tpr=("tpr", "sem"),
        )
    )

    rt_auc_mean_2AFC = roc_auc_df_2AFC[roc_auc_df_2AFC["metric"].eq("RT")]["auc"].mean()
    rt_auc_sem_2AFC = roc_auc_df_2AFC[roc_auc_df_2AFC["metric"].eq("RT")]["auc"].sem()
    rt_roc_2AFC_ax.plot(
        rt_roc_summary_2AFC["fpr"],
        rt_roc_summary_2AFC["mean_tpr"],
        color=cmap.colors[2],
        lw=2,
        label=f"AUC={rt_auc_mean_2AFC:.3f} +/- {rt_auc_sem_2AFC:.3f}",
    )
    rt_roc_2AFC_ax.fill_between(
        rt_roc_summary_2AFC["fpr"],
        rt_roc_summary_2AFC["mean_tpr"] - rt_roc_summary_2AFC["sem_tpr"].fillna(0),
        rt_roc_summary_2AFC["mean_tpr"] + rt_roc_summary_2AFC["sem_tpr"].fillna(0),
        color=cmap.colors[2],
        alpha=0.2,
        linewidth=0,
    )
    rt_roc_2AFC_ax.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--")
    rt_roc_2AFC_ax.set_title(f"2AFC $-RT$")
    rt_roc_2AFC_ax.set_xlabel("False positive rate")
    rt_roc_2AFC_ax.set_ylabel("True positive rate")
    rt_roc_2AFC_ax.set_xlim(0, 1)
    rt_roc_2AFC_ax.set_ylim(0, 1)
    rt_roc_2AFC_ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=rt_roc_2AFC_ax)
    rt_roc_2AFC.savefig((path_panels / "2AFC_rt_state_roc").with_suffix(f".{format}"))
    rt_roc_2AFC
    return


@app.cell
def _(axd, fig, format, mount_figure, path_panels):
    if mount_figure:
        _legend_ax = axd.get("single_session_2ADC")
        _handles, _labels = _legend_ax.get_legend_handles_labels() if _legend_ax is not None else ([], [])
        _seen = set()
        _legend_items = [
            (_handle, _label)
            for _handle, _label in zip(_handles, _labels, strict=False)
            if _label and not _label.startswith("_") and not (_label in _seen or _seen.add(_label))
        ]
        for _legend in list(fig.legends):
            _legend.remove()

        for _name, _ax in axd.items():
            _ax.set_title("")
            _ax.set_xlabel("")
            if not _name.startswith("_model_comparison_parent"):
                _ax.set_ylabel("")
            _legend = _ax.get_legend()
            if _legend is not None:
                _legend.remove()

        axd["emission_weights_2ADC"].set_title("2ADC")
        axd["emission_weights_2AFC"].set_title("2AFC")

        axd["emission_weights_2ADC"].set_ylabel("Emission weight")
        axd["emission_weights_2AFC"].set_ylabel("Emission weight")
        axd["transition_weights_2ADC"].set_ylabel("Transition weight")
        axd["transition_weights_2AFC"].set_ylabel("Transition weight")
        axd["psychometric_by_state_2ADC"].set_ylabel(r"$p(\mathrm{right})$")
        axd["nlicks_by_state_2ADC"].set_ylabel("nLicks")
        axd["nlicks_by_state_2AFC"].set_ylabel("nLicks")
        # axd["model_comparison_ll_2ADC"].set_ylabel(r"$\Delta$ LL")
        # axd["model_comparison_ll_2AFC"].set_ylabel(r"$\Delta$ LL")
        axd["autocorrelograms_2ADC_repetition"].set_ylabel("Autocorrelation")
        axd["single_session_2ADC"].set_ylabel("Running fraction")
        # axd["single_session_2çADC"].set_xlabel("Trial")
        # axd["single_session_2AFC"].set_xlabel("Trial")

        if _legend_items:
            _handles, _labels = zip(*_legend_items, strict=False)
            fig.legend(
                _handles,
                _labels,
                loc="lower center",
                ncol=min(6, len(_labels)),
                frameon=False,
                bbox_to_anchor=(0.5, -0.1),
                handlelength=1.2,
                columnspacing=0.8,
            )
        fig.savefig((path_panels / "figure3").with_suffix(f".{format}"))
        fig.align_ylabels()
    fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
