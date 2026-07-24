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
        glmhmmt_state_psychometric_df,
        glmhmmt_state_switch_histogram_df,
        glmhmmt_state_switches_df,
        glmhmmt_state_trace_df,
        glmhmmt_transition_weights_df,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )
    from src.plots.common import (
        BOXPLOT_STYLE,
        build_session_repetition_data,
        fig_size,
    )

    return (
        Annotator,
        BOXPLOT_STYLE,
        FuncFormatter,
        Path,
        add_choice_lag_summary_regressor,
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
        glmhmmt_state_psychometric_df,
        glmhmmt_state_switch_histogram_df,
        glmhmmt_state_switches_df,
        glmhmmt_state_trace_df,
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
        process_mcdr,
        process_two_adc,
        process_two_afc,
        sns,
        ttest_1samp,
    )


@app.cell
def _():
    boxplot_STYLE = dict(
        fill=False,
        boxprops={"color": "0.5"},
        whiskerprops={"color": "0.5"},
        medianprops={"linewidth": 4},
        showfliers=False,
        showcaps=False,
    )
    return (boxplot_STYLE,)


@app.cell
def _(load_app_config, process_mcdr, process_two_adc, process_two_afc):
    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
        # if task_name in {"2ADC", "2ADC_DRUG", "2AFC", "2AFC_DRUG"}:
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
    mount_figure = True
    format = "pdf"
    return format, mount_figure


@app.cell
def _():
    model_type = "glmhmmt"
    return (model_type,)


@app.cell
def _():
    MODEL_BY_TASK = {
        "2AFC_DRUG": "drug_transitions2",
        "2ADC_DRUG": "drug_transitions2",
        # "MCDR": "param",
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
    path_panels = project_path / "figures" / "panels4" / format
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
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    task_labels = {
        "2AFC_DRUG": "2AFC",
        "2ADC_DRUG": "2ADC",
    }
    task_palette = {
        "2AFC_DRUG": "tab:green",
        "2ADC_DRUG": "tab:blue",
    }
    treatment_order = ["Saline", "Drug"]
    treatment_palette = {
        "Saline": "tab:gray",
        "Drug": "tab:pink",
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
        "stim_x_delay_param": "Stim.",
        "choice_lag_param": "A",
        "choice_lag_param_correct": "A",
        "prev_choice": "Prev. choice",
        "filtered_reward": "Reward trace",
        "filtered_choice": "Choice trace",
        "filtered_stim_side": "Stimulus trace",
        "prev_difficulty": "Previous difficulty",
        "cumulative_reward": "Cumulative reward",
        "trial_index": "Trial index",
        "Drug": "NMDAr block",
        "drug_code": "NMDAr block",
        "drug_x_filtered_reward": "NMDAr block : Rew trace",
    }
    delay_order = [-0.1, -1, -3, -10, 10, 3, 1, 0.1]
    delay_mapping = {value: index for index, value in enumerate(delay_order)}
    return (
        delay_order,
        feature_labels,
        state_palette,
        task_labels,
        treatment_order,
        treatment_palette,
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
            return "ns"
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
        dfs["2AFC_DRUG"] = dfs["2AFC_DRUG"].filter(pl.col("subject") != "326")
        dfs["2AFC_DRUG"] = dfs["2AFC_DRUG"].filter(pl.col("subject") != "19")
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
                f"{_task_name}: no arrays found for {_model_dir}"
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
    return active_task_names, model_load_report, plot_dfs, weight_dfs


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
    pd,
    pl,
    plot_dfs,
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
        if action_col is None or not {"subject", x_col, "response"}.issubset(df.columns):
            return empty

        out = df[["subject", x_col, "response", action_col]].copy()
        out["_action_trace"] = pd.to_numeric(out[action_col], errors="coerce")
        out["_response_right"] = (pd.to_numeric(out["response"], errors="coerce") > 0).astype(float)
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
            .agg(p_right=("_response_right", "mean"), n_trials=("_response_right", "size"))
            .sort_values(["action_bin", "x_position", "subject"])
        )
        summary["x_numeric"] = pd.to_numeric(summary["x_label"].astype(str), errors="coerce")
        return summary

    def choice_lag_psychometric_df(trial_df, *, n_bins=9):
        df = trial_df.to_pandas().copy() if hasattr(trial_df, "to_pandas") else pd.DataFrame(trial_df).copy()
        empty = pd.DataFrame(
            columns=[
                "subject",
                "state_label",
                "x_value",
                "x_label",
                "x_numeric",
                "x_position",
                "p_right",
                "n_trials",
            ]
        )
        required = {"subject", "state_label", "choice_lag_param", "response"}
        if df.empty or not required.issubset(df.columns):
            return empty

        out = df[["subject", "state_label", "choice_lag_param", "response"]].copy()
        out["_choice_lag"] = pd.to_numeric(out["choice_lag_param"], errors="coerce")
        out["_response_right"] = (pd.to_numeric(out["response"], errors="coerce") > 0).astype(float)
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

        return (
            out.groupby(["subject", "state_label", "x_value", "x_position", "x_numeric", "x_label"], as_index=False, observed=True)
            .agg(p_right=("_response_right", "mean"), n_trials=("_response_right", "size"))
            .sort_values(["state_label", "x_position", "subject"])
        )

    for _task_name in active_task_names:
        emission_plot_dfs[_task_name] = with_feature_labels(weight_dfs[_task_name].to_pandas())
        # transition_plot_dfs[_task_name] = with_feature_labels(
        #     glmhmmt_transition_weights_df(arrays_by_task[_task_name], views[_task_name])
        # )
        # transition_plot_dfs[_task_name] = transition_plot_dfs[_task_name][transition_plot_dfs[_task_name]["source_state_label"] != transition_plot_dfs[_task_name]["destination_state_label"]]
        _psychometric_source_df = plot_dfs[_task_name]
        if _task_name == "2ADC_DRUG":
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
                "2AFC_DRUG": "ILD",
                "MCDR": "ttype_c",
            }[_task_name]
        psychometric_x_cols[_task_name] = _x_col
        psychometric_x_labels[_task_name] = {
            "2AFC_DRUG": "ILD",
            "2ADC_DRUG": "Signed delay",
            "MCDR": "Trial type",
        }[_task_name]
        psychometric_dfs[_task_name] = glmhmmt_state_psychometric_df(
            _psychometric_source_df,
            x_col=_x_col,
            x_order=delay_order if _task_name == "2ADC_DRUG" else None,
        )
        choice_lag_psychometric_dfs[_task_name] = choice_lag_psychometric_df(_psychometric_source_df)
        action_trace_psychometric_dfs[_task_name] = action_trace_psychometric_df(
            _psychometric_source_df,
            x_col=_x_col,
            x_order=delay_order if _task_name == "2ADC_DRUG" else None,
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
        switch_hist_dfs,
        switch_session_dfs,
        trace_dfs,
        transition_plot_dfs,
    )


@app.cell
def _(dwell_dfs, np, pd, plot_dfs, switch_session_dfs, task_names):
    treatment_trial_dfs = {}
    treatment_psychometric_dfs = {}
    psychometric_limits = {}
    treatment_dwell_dfs = {}
    treatment_switch_dfs = {}
    session_options = {}

    for _task_name in task_names:
        _trials = plot_dfs[_task_name].to_pandas().copy()
        _condition = _trials["condition"].astype("string").str.lower()
        _trials["treatment"] = _condition.map(
            {"saline": "Saline", "drug": "Drug"}
        )
        _trials = _trials.dropna(subset=["subject", "session", "treatment"])
        _trials["subject"] = _trials["subject"].astype(str)
        _trials["session"] = _trials["session"].astype(str)

        if _task_name == "2ADC_DRUG":
            _stimulus_sign = np.where(
                pd.to_numeric(_trials["stimulus"], errors="coerce").eq(0),
                -1.0,
                1.0,
            )
            _trials["signed_delay"] = (
                pd.to_numeric(_trials["delays"], errors="coerce")
                * _stimulus_sign
            ).round(1)

        treatment_trial_dfs[_task_name] = _trials
        _session_treatments = (
            _trials[["subject", "session", "treatment"]]
            .drop_duplicates()
            .sort_values(["treatment", "subject", "session"])
        )
        session_options[_task_name] = _session_treatments

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
            _trials["p_right_model"] = pd.to_numeric(
                _trials[_model_col],
                errors="coerce",
            )
        _response = pd.to_numeric(_trials["response"], errors="coerce")
        _trials["p_right_data"] = np.where(
            _response.notna(),
            (_response > 0).astype(float),
            np.nan,
        )
        _trials["stimulus_evidence"] = pd.to_numeric(
            _trials[_evidence_col],
            errors="coerce",
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
        _full_evidence_limits = (
            float(_psychometric_trials["stimulus_evidence"].min()),
            float(_psychometric_trials["stimulus_evidence"].max()),
        )
        _psychometric_trials["_evidence_bin"] = pd.qcut(
            _psychometric_trials["stimulus_evidence"],
            q=9,
            duplicates="drop",
        )
        _psychometric_trials["stimulus_evidence"] = (
            _psychometric_trials.groupby(
                "_evidence_bin",
                observed=True,
            )["stimulus_evidence"].transform("mean")
        )
        treatment_psychometric_dfs[_task_name] = (
            _psychometric_trials.groupby(
                [
                    "subject",
                    "treatment",
                    "stimulus_evidence",
                ],
                as_index=False,
                observed=True,
            )
            .agg(
                p_right_data=("p_right_data", "mean"),
                p_right_model=("p_right_model", "mean"),
            )
            .sort_values(["treatment", "stimulus_evidence", "subject"])
        )
        psychometric_limits[_task_name] = _full_evidence_limits

        _dwell = dwell_dfs[_task_name].copy()
        _dwell["subject"] = _dwell["subject"].astype(str)
        _dwell["session"] = _dwell["session"].astype(str)
        treatment_dwell_dfs[_task_name] = (
            _dwell.merge(
                _session_treatments,
                on=["subject", "session"],
                how="inner",
            )
            .groupby(
                ["subject", "treatment", "state_label"],
                as_index=False,
                observed=True,
            )
            .agg(mean_dwell_trials=("dwell_trials", "mean"))
        )
        _switches = switch_session_dfs[_task_name].copy()
        _switches["subject"] = _switches["subject"].astype(str)
        _switches["session"] = _switches["session"].astype(str)
        treatment_switch_dfs[_task_name] = _switches.merge(
            _session_treatments,
            on=["subject", "session"],
            how="inner",
        )
    return (
        psychometric_limits,
        session_options,
        treatment_dwell_dfs,
        treatment_psychometric_dfs,
        treatment_switch_dfs,
        treatment_trial_dfs,
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

    def ordered_transition_feature_values(df, column):
        values = ordered_feature_values(df, column)

        return sorted(
            values,
            key=lambda v: (
                "Drug" in str(v), # False primero, True último
                values.index(v),            # mantiene el orden previo dentro de cada grupo
            ),
        )


    transition_orders = {}
    if model_type == "glmhmmt":
        transition_orders = {
            task: ordered_transition_feature_values(transition_plot_dfs[task], "feature_label")
            for task in task_names
        }
    if model_type == "glmhmmt":  # Check if it is a model with transitions
        transition_orders = {task: ordered_feature_values(transition_plot_dfs[task], "feature_label") for task in task_names}

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
    metric_order = ordered_values(metric_dfs["2AFC_DRUG"], "metric")
    metric_hue_order = ordered_states(metric_dfs["2AFC_DRUG"])
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
        task_label_orders,
        transition_orders,
    )


@app.cell
def _(metric_dfs):
    metric_dfs
    return


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
                [
                    "Diagram",
                    "Diagram",
                    "Diagram",
                    "Diagram",
                ],
                [
                    "single_session_saline_2ADC",
                    "single_session_saline_2ADC",
                    "single_session_saline_2AFC",
                    "single_session_saline_2AFC",
                ],
                [
                    "single_session_drug_2ADC",
                    "single_session_drug_2ADC",
                    "single_session_drug_2AFC",
                    "single_session_drug_2AFC",
                ],
                [
                    "histogram_transitions_2ADC",
                    "dwell_time_2ADC",
                    "histogram_transitions_2AFC",
                    "dwell_time_2AFC",
                ],
                [
                    "transition_weights_2ADC",
                    "psychometric_2ADC",
                    "transition_weights_2AFC",
                    "psychometric_2AFC",
                ],
            ],
            figsize=fig_size(1, 0.7),
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            w_pad=0.005,
            h_pad=0.01,
            wspace=0.005,
            hspace=0.04,
        )

        for name, axis in axd.items():
            axis.text(
                0.5,
                0.5,
                name,
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                wrap=True,
            )
            if name == "Diagram":
                axis.set_axis_off()

    else:
        fig, axd = None, {}
    return axd, fig


@app.cell
def _(fig):

    fig
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
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    emission_weights_2ADC = plt.gca() if not mount_figure else axd.get("emission_weights_2ADC", plt.gca())
    emission_weights_2ADC.clear()
    sns.boxplot(
        data=emission_plot_dfs["2ADC_DRUG"],
        x="feature_label",
        y="weight",
        hue="state_label",
        order=emission_orders["2ADC_DRUG"],
        hue_order=emission_hue_orders["2ADC_DRUG"],
        gap=0.2,
        palette=state_palette,
        ax=emission_weights_2ADC,
        **boxplot_STYLE,
    )
    # add_subject_pair_lines(emission_weights_2ADC, emission_plot_dfs["2ADC_DRUG"], x="feature_label", y="weight", order=emission_orders["2ADC_DRUG"])
    add_paired_state_annotation(emission_weights_2ADC, emission_plot_dfs["2ADC_DRUG"], x="feature_label", y="weight", order=emission_orders["2ADC_DRUG"])
    emission_weights_2ADC.axhline(0, color="0.5", linestyle="--")
    emission_weights_2ADC.set_title(task_labels["2ADC_DRUG"])
    emission_weights_2ADC.set_xlabel("")
    emission_weights_2ADC.set_ylabel("Emission weight")
    emission_weights_2ADC.tick_params(axis="x")
    emission_weights_2ADC.legend().remove()
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
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    emission_weights_2AFC = plt.gca() if not mount_figure else axd.get("emission_weights_2AFC", plt.gca())
    emission_weights_2AFC.clear()
    sns.boxplot(
        data=emission_plot_dfs["2AFC_DRUG"],
        x="feature_label",
        y="weight",
        hue="state_label",
        order=emission_orders["2AFC_DRUG"],
        hue_order=emission_hue_orders["2AFC_DRUG"],
        gap=0.2,
        palette=state_palette,
        ax=emission_weights_2AFC,
        **boxplot_STYLE,
    )
    # add_subject_pair_lines(emission_weights_2AFC, emission_plot_dfs["2AFC_DRUG"], x="feature_label", y="weight", order=emission_orders["2AFC_DRUG"])
    add_paired_state_annotation(emission_weights_2AFC, emission_plot_dfs["2AFC_DRUG"], x="feature_label", y="weight", order=emission_orders["2AFC_DRUG"])
    emission_weights_2AFC.axhline(0, color="0.5", linestyle="--")
    emission_weights_2AFC.set_title(task_labels["2AFC_DRUG"])
    emission_weights_2AFC.set_xlabel("")
    emission_weights_2AFC.set_ylabel("Emission weight")
    emission_weights_2AFC.tick_params(axis="x")
    emission_weights_2AFC.legend(frameon=False, title="")
    emission_weights_2AFC.legend().remove()
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
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
def _(transition_plot_dfs):
    transition_plot_dfs
    return


@app.cell
def _(
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
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    transition_weights_2ADC = (
        plt.gca() if not mount_figure else axd["transition_weights_2ADC"]
    )
    transition_weights_2ADC.clear()
    if model_type != "glmhmmt":
        transition_weights_2ADC.set_axis_off()
        transition_weights_2ADC.text(0.5, 0.5, "Transition", ha="center", va="center",)
        transition_weights_2ADC.text(0.5, 0.3, "Weights", ha="center", va="center",)
    else: 
        palette = {
            "Engaged -> Engaged": "tab:green",
            "Disengaged -> Engaged": "tab:green",
            "Engaged -> Disengaged": "0.5",
            "Disengaged -> Disengaged": "0.5",
        }

        sns.boxplot(
            data=transition_plot_dfs["2ADC_DRUG"],
            x="feature_label",
            y="weight",
            hue="transition_label",
            order=transition_orders["2ADC_DRUG"],
            palette=palette,
            ax=transition_weights_2ADC,
            **boxplot_STYLE,
        )
        transition_weights_2ADC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        # add_one_sample_zero_annotations(transition_weights_2ADC, transition_plot_dfs["2ADC_DRUG"], x="feature_label", y="weight", order=transition_orders["2ADC_DRUG"])
    # transition_weights_2ADC.set_title(task_labels["2ADC_DRUG"])
    transition_weights_2ADC.set_xlabel("")
    transition_weights_2ADC.set_ylabel("Transition weight")
    transition_weights_2ADC.tick_params(axis="x", labelrotation=30)
    transition_weights_2ADC.set_ylim(top = 10)
    handles, _ = transition_weights_2ADC.get_legend_handles_labels()

    transition_weights_2ADC.legend(
        handles,
        [r"$E\rightarrow D$", r"$D\rightarrow E$"],
        frameon=False, ncol = 2,  handlelength=0.8,handletextpad=0.4,columnspacing=0.8,
    )
    if not mount_figure:
        transition_weights_2ADC.figure.savefig((path_panels / "2AFC_delay_glmhmmt_transition_weights").with_suffix(f".{format}"))
    transition_weights_2ADC
    return (palette,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    axd,
    boxplot_STYLE,
    fig_size,
    format,
    model_type,
    mount_figure,
    palette,
    path_panels,
    plt,
    sns,
    transition_orders,
    transition_plot_dfs,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    transition_weights_2AFC = (
        plt.gca() if not mount_figure else axd["transition_weights_2AFC"]
    )
    transition_weights_2AFC.clear()
    if model_type != "glmhmmt":
        transition_weights_2AFC.set_axis_off()
        transition_weights_2AFC.text(0.5, 0.5, "Transition", ha="center", va="center")
        transition_weights_2AFC.text(0.5, 0.3, "Weights", ha="center", va="center")
    else:

        sns.boxplot(
            data=transition_plot_dfs["2AFC_DRUG"],
            x="feature_label",
            y="weight",
            hue="transition_label",
            order=transition_orders["2AFC_DRUG"],
            palette=palette,
            ax=transition_weights_2AFC,
            **boxplot_STYLE,
        )

        transition_weights_2AFC.axhline(0, color="0.5", linestyle="--", linewidth=0.8)

        # add_one_sample_zero_annotations(
        #     transition_weights_2AFC,
        #     transition_plot_dfs["2AFC_DRUG"],
        #     x="feature_label",
        #     y="weight",
        #     order=transition_orders["2AFC_DRUG"],
        # )

        _handles, _ = transition_weights_2AFC.get_legend_handles_labels()
        transition_weights_2AFC.legend(
            _handles,
            [r"$E\rightarrow D$", r"$D\rightarrow E$"],
            ncols=2,
            frameon=False,
             handlelength=0.8,handletextpad=0.4,columnspacing=0.8,
        )

    # transition_weights_2AFC.set_title(task_labels["2AFC_DRUG"])
    transition_weights_2AFC.set_xlabel("")
    transition_weights_2AFC.set_ylim(top = 10)
    transition_weights_2AFC.set_ylabel("Transition weight")
    transition_weights_2AFC.tick_params(axis="x", rotation=30)
    if not mount_figure:
        transition_weights_2AFC.figure.savefig(
            (path_panels / "2AFC_glmhmmt_transition_weights").with_suffix(f".{format}")
        )

    transition_weights_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
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
    psychometric_2ADC.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    psychometric_2ADC.set_title(task_labels["2ADC_DRUG"])
    psychometric_2ADC.set_xlabel("Stimulus evidence")
    psychometric_2ADC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_2ADC.legend(
        frameon=False,
        title=None,
        ncol=2,
        fontsize=5,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.6,
        loc="lower center",
    )
    psychometric_2ADC.set_xlim(*psychometric_limits["2ADC_DRUG"])
    psychometric_2ADC.set_ylim(0, 1)
    psychometric_2ADC.set_yticks([0, 0.5, 1])
    if not mount_figure:
        psychometric_2ADC.figure.savefig(
            (path_panels / "2ADC_drug_psychometric").with_suffix(
                f".{format}"
            )
        )
    psychometric_2ADC
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
    psychometric_limits,
    sns,
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_psychometric_dfs,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
        err_kws={
            "edgecolor": "none",
            "linewidth": 0,
        },
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
    psychometric_2AFC.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    psychometric_2AFC.set_title(task_labels["2AFC_DRUG"])
    psychometric_2AFC.set_xlabel("Stimulus evidence")
    psychometric_2AFC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_2AFC.legend(
        frameon=False,
        title=None,
        ncol=2,
        fontsize=5,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.6,
        loc="lower center",
    )
    psychometric_2AFC.set_xlim(*psychometric_limits["2AFC_DRUG"])
    psychometric_2AFC.set_ylim(0, 1)
    psychometric_2AFC.set_yticks([0, 0.5, 1])
    if not mount_figure:
        psychometric_2AFC.figure.savefig(
            (path_panels / "2AFC_drug_psychometric").with_suffix(
                f".{format}"
            )
        )
    psychometric_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    psychometric_by_action_trace_2ADC = plt.gca() if not mount_figure else axd.get("psychometric_by_action_trace_2ADC", plt.gca())
    psychometric_by_action_trace_2ADC.clear()
    sns.lineplot(
        data=choice_lag_psychometric_dfs["2ADC_DRUG"],
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
        ax=psychometric_by_action_trace_2ADC,
    )
    # _xticks = sorted(choice_lag_psychometric_dfs["2ADC_DRUG"]["x_numeric"].dropna().unique())
    # _xticks = [float(tick) for tick in _xticks]
    # psychometric_by_action_trace_2ADC.set_xticks(_xticks)
    # psychometric_by_action_trace_2ADC.set_xticklabels([f"{tick:g}" for tick in _xticks])
    psychometric_by_action_trace_2ADC.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    psychometric_by_action_trace_2ADC.set_title(task_labels["2ADC_DRUG"])
    psychometric_by_action_trace_2ADC.set_xlabel(feature_labels["choice_lag_param"])
    psychometric_by_action_trace_2ADC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_action_trace_2ADC.legend(frameon=False, title="")
    psychometric_by_action_trace_2ADC.set_ylim(0,1)
    psychometric_by_action_trace_2ADC.set_yticks([0, 0.5,1], [0, 0.5,1], )
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
    task_labels,
):
    plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    psychometric_by_action_trace_2AFC = plt.gca() if not mount_figure else axd.get("psychometric_by_action_trace_2AFC", plt.gca())
    psychometric_by_action_trace_2AFC.clear()
    sns.lineplot(
        data=choice_lag_psychometric_dfs["2AFC_DRUG"],
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
        ax=psychometric_by_action_trace_2AFC,
    )
    # _xticks = sorted(choice_lag_psychometric_dfs["2AFC_DRUG"]["x_numeric"].dropna().unique())
    # _xticks = [float(tick) for tick in _xticks]
    # psychometric_by_action_trace_2AFC.set_xticks(_xticks)
    # psychometric_by_action_trace_2AFC.set_xticklabels([f"{tick:g}" for tick in _xticks])
    psychometric_by_action_trace_2AFC.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    psychometric_by_action_trace_2AFC.set_title(task_labels["2AFC_DRUG"])
    psychometric_by_action_trace_2AFC.set_xlabel(feature_labels["choice_lag_param"])
    psychometric_by_action_trace_2AFC.set_ylabel(r"$p(\mathrm{right})$")
    psychometric_by_action_trace_2AFC.legend(frameon=False, title="")
    psychometric_by_action_trace_2AFC.set_ylim(0,1)
    psychometric_by_action_trace_2AFC.set_yticks([0, 0.5,1], [0, 0.5,1], )
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
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
        data=accuracy_plot_dfs["2ADC_DRUG"],
        x="task_label",
        y="accuracy",
        hue="state_label",
        order=task_label_orders["2ADC_DRUG"],
        hue_order=accuracy_hue_orders["2ADC_DRUG"],
        palette=state_palette,
        ax=accuracy_by_state_2ADC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(accuracy_by_state_2ADC, accuracy_plot_dfs["2ADC_DRUG"], x="task_label", y="accuracy", order=task_label_orders["2ADC_DRUG"])
    add_paired_state_annotation(accuracy_by_state_2ADC, accuracy_plot_dfs["2ADC_DRUG"], x="task_label", y="accuracy", order=task_label_orders["2ADC_DRUG"])
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
        data=accuracy_plot_dfs["2AFC_DRUG"],
        x="task_label",
        y="accuracy",
        hue="state_label",
        order=task_label_orders["2AFC_DRUG"],
        hue_order=accuracy_hue_orders["2AFC_DRUG"],
        palette=state_palette,
        ax=accuracy_by_state_2AFC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(accuracy_by_state_2AFC, accuracy_plot_dfs["2AFC_DRUG"], x="task_label", y="accuracy", order=task_label_orders["2AFC_DRUG"])
    add_paired_state_annotation(accuracy_by_state_2AFC, accuracy_plot_dfs["2AFC_DRUG"], x="task_label", y="accuracy", order=task_label_orders["2AFC_DRUG"])
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
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
        data=occupancy_plot_dfs["2ADC_DRUG"],
        x="task_label",
        y="occupancy",
        hue="state_label",
        order=task_label_orders["2ADC_DRUG"],
        hue_order=occupancy_hue_orders["2ADC_DRUG"],
        palette=state_palette,
        ax=occupancy_by_state_2ADC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(occupancy_by_state_2ADC, occupancy_annotation_dfs["2ADC_DRUG"], x="task_label", y="occupancy", order=task_label_orders["2ADC_DRUG"])
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
        data=occupancy_plot_dfs["2AFC_DRUG"],
        x="task_label",
        y="occupancy",
        hue="state_label",
        order=task_label_orders["2AFC_DRUG"],
        hue_order=occupancy_hue_orders["2AFC_DRUG"],
        palette=state_palette,
        ax=occupancy_by_state_2AFC,
        **boxplot_STYLE,
    )
    add_subject_pair_lines(occupancy_by_state_2AFC, occupancy_annotation_dfs["2AFC_DRUG"], x="task_label", y="occupancy", order=task_label_orders["2AFC_DRUG"])
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
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
        data=trace_dfs["2ADC_DRUG"],
        x="trial_bin",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=mean_state_traces_2ADC,
    )
    mean_state_traces_2ADC.set_title(task_labels["2ADC_DRUG"])
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
        data=trace_dfs["2AFC_DRUG"],
        x="trial_bin",
        y="p_state",
        hue="state_label",
        estimator="mean",
        errorbar="se",
        palette=state_palette,
        ax=mean_state_traces_2AFC,
    )
    mean_state_traces_2AFC.set_title(task_labels["2AFC_DRUG"])
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
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
        data=metric_dfs["2AFC_DRUG"][metric_dfs["2AFC_DRUG"]["metric"].eq("RT")],
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
        data=metric_dfs["2AFC_DRUG"][metric_dfs["2AFC_DRUG"]["metric"].eq("RT")],
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
        data=metric_dfs["2AFC_DRUG"][metric_dfs["2AFC_DRUG"]["metric"].eq("nLicks")],
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
        data=metric_dfs["2AFC_DRUG"][metric_dfs["2AFC_DRUG"]["metric"].eq("nLicks")],
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
        data=metric_dfs["2AFC_DRUG"][metric_dfs["2AFC_DRUG"]["metric"].eq("ILI")],
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
        data=metric_dfs["2AFC_DRUG"][metric_dfs["2AFC_DRUG"]["metric"].eq("ILI")],
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
    ## Mean Dwell Time
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
    BOXPLOT_STYLE,
    add_paired_state_annotation,
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_dwell_dfs,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    dwell_time_2ADC = plt.gca() if not mount_figure else axd["dwell_time_2ADC"]
    dwell_time_2ADC.clear()
    sns.boxplot(
        data=treatment_dwell_dfs["2ADC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        hue="treatment",
        order=["Engaged", "Disengaged"],
        hue_order=treatment_order,
        palette=treatment_palette,
        ax=dwell_time_2ADC,
        legend = False,
        **BOXPLOT_STYLE,
    )
    add_paired_state_annotation(
        dwell_time_2ADC,
        treatment_dwell_dfs["2ADC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        order=["Engaged", "Disengaged"],
        hue="treatment",
        hue_order=treatment_order,
    )
    dwell_time_2ADC.set_title(task_labels["2ADC_DRUG"])
    dwell_time_2ADC.set_xlabel("")
    dwell_time_2ADC.set_ylabel("Mean dwell time (trials)")
    dwell_time_2ADC.set_xticklabels(["Eng.", "Dis."])
    if not mount_figure:
        dwell_time_2ADC.figure.savefig(
            (path_panels / "2ADC_drug_dwell_time").with_suffix(f".{format}")
        )
    dwell_time_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    add_paired_state_annotation,
    axd,
    fig_size,
    format,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_dwell_dfs,
    treatment_order,
    treatment_palette,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    dwell_time_2AFC = plt.gca() if not mount_figure else axd["dwell_time_2AFC"]
    dwell_time_2AFC.clear()
    sns.boxplot(
        data=treatment_dwell_dfs["2AFC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        hue="treatment",
        order=["Engaged", "Disengaged"],
        hue_order=treatment_order,
        palette=treatment_palette,
        ax=dwell_time_2AFC,
        legend = False,
        **BOXPLOT_STYLE,
    )
    add_paired_state_annotation(
        dwell_time_2AFC,
        treatment_dwell_dfs["2AFC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        order=["Engaged", "Disengaged"],
        hue="treatment",
        hue_order=treatment_order,
    )
    dwell_time_2AFC.set_title(task_labels["2AFC_DRUG"])
    dwell_time_2AFC.set_xlabel("")
    dwell_time_2AFC.set_ylabel("Mean dwell time (trials)")
    dwell_time_2AFC.set_xticklabels(["Eng.", "Dis."])
    dwell_time_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        dwell_time_2AFC.figure.savefig(
            (path_panels / "2AFC_drug_dwell_time").with_suffix(f".{format}")
        )
    dwell_time_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State Switch Histograms
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
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_switch_dfs,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    histogram_transitions_2ADC = (
        plt.gca() if not mount_figure else axd["histogram_transitions_2ADC"]
    )
    histogram_transitions_2ADC.clear()
    sns.histplot(
        data=treatment_switch_dfs["2ADC_DRUG"],
        x="n_switches",
        hue="treatment",
        hue_order=treatment_order,
        palette=treatment_palette,
        bins=12,
        stat="probability",
        common_norm=False,
        element="step",
        multiple="layer",
        alpha=0.45,
        ax=histogram_transitions_2ADC,
    )
    _legend = histogram_transitions_2ADC.get_legend()
    _legend.set_frame_on(False)
    _legend.set_title(None)
    histogram_transitions_2ADC.set_title(task_labels["2ADC_DRUG"])
    histogram_transitions_2ADC.set_xlabel("State switches")
    histogram_transitions_2ADC.set_ylabel("Probability")
    if not mount_figure:
        histogram_transitions_2ADC.figure.savefig(
            (path_panels / "2ADC_drug_state_switch_histogram").with_suffix(
                f".{format}"
            )
        )
    histogram_transitions_2ADC
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
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_switch_dfs,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    histogram_transitions_2AFC = (
        plt.gca() if not mount_figure else axd["histogram_transitions_2AFC"]
    )
    histogram_transitions_2AFC.clear()
    sns.histplot(
        data=treatment_switch_dfs["2AFC_DRUG"],
        x="n_switches",
        hue="treatment",
        hue_order=treatment_order,
        palette=treatment_palette,
        bins=12,
        stat="probability",
        common_norm=False,
        element="step",
        multiple="layer",
        alpha=0.45,
        ax=histogram_transitions_2AFC,
    )
    _legend = histogram_transitions_2AFC.get_legend()
    _legend.set_frame_on(False)
    _legend.set_title(None)
    histogram_transitions_2AFC.set_title(task_labels["2AFC_DRUG"])
    histogram_transitions_2AFC.set_xlabel("State switches")
    histogram_transitions_2AFC.set_ylabel("Probability")
    if not mount_figure:
        histogram_transitions_2AFC.figure.savefig(
            (path_panels / "2AFC_drug_state_switch_histogram").with_suffix(
                f".{format}"
            )
        )
    histogram_transitions_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3CDR
    """)
    return


@app.cell
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
        data=change_posterior_dfs["2ADC_DRUG"],
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
    posteriors_around_change_2ADC.set_title(task_labels["2ADC_DRUG"])
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
        data=change_posterior_dfs["2AFC_DRUG"],
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
    posteriors_around_change_2AFC.set_title(task_labels["2AFC_DRUG"])
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
def _():
    # MCDR is not part of MODEL_BY_TASK for this figure; panel disabled.
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
    _subject = "E11"
    _session = "E11_StageTraining_Ephys_V1_20210409-115620"
    _task_df = plot_dfs["2ADC_DRUG"]
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
        adapter=adapters["2ADC_DRUG"],
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

    single_session_2ADC.plot("trial_x", "response_repeat_window_fraction", color="tab:brown", linewidth=1.5, label="Rep. Choices", data=session_repetition_data_2ADC, zorder=2)
    single_session_2ADC.plot("trial_x", "stimulus_repeat_window_fraction", color="tab:blue", linewidth=1.5, label="Rep. Stimulus", data=session_repetition_data_2ADC, zorder=2)
    if "accuracy_window_fraction" in session_repetition_data_2ADC:
        single_session_2ADC.plot("trial_x", "accuracy_window_fraction", color="black", linewidth=1.5, label="Accuracy", data=session_repetition_data_2ADC, zorder=2)
    single_session_2ADC.set_title(task_labels["2ADC_DRUG"])
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
    _subject = "875"
    _session = "875_stage_training_v4_20230717-112113" 
    _task_df = plot_dfs["2AFC_DRUG"]
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
        adapter=adapters["2AFC_DRUG"],
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

    single_session_2AFC.plot("trial_x", "response_repeat_window_fraction", color="tab:brown", linewidth=1.5, label="Rep. Choices", data=session_repetition_data_2AFC, zorder=2)
    single_session_2AFC.plot("trial_x", "stimulus_repeat_window_fraction", color="tab:blue", linewidth=1.5, label="Rep. Stimulus", data=session_repetition_data_2AFC, zorder=2)
    if "accuracy_window_fraction" in session_repetition_data_2AFC:
        single_session_2AFC.plot("trial_x", "accuracy_window_fraction", color="black", linewidth=1.5, label="Accuracy", data=session_repetition_data_2AFC, zorder=2)
    single_session_2AFC.set_title(task_labels["2AFC_DRUG"])
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


@app.cell
def _():
    # Replace these defaults with the four paper sessions once selected.
    SELECTED_SESSIONS = {
        "saline_2ADC": {"subject": None, "session": None},
        "drug_2ADC": {"subject": None, "session": None},
        "saline_2AFC": {"subject": None, "session": None},
        "drug_2AFC": {"subject": None, "session": None},
    }
    return (SELECTED_SESSIONS,)


@app.cell(hide_code=True)
def _(SELECTED_SESSIONS, mo, session_options):
    def _subject_dropdown(task_name, treatment, panel_key):
        _options = (
            session_options[task_name]
            .loc[lambda df: df["treatment"].eq(treatment), "subject"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        _selected = SELECTED_SESSIONS[panel_key]["subject"]
        _value = str(_selected) if str(_selected) in _options else _options[0]
        return mo.ui.dropdown(_options, value=_value, label="Subject")

    ui_saline_2ADC_subject = _subject_dropdown(
        "2ADC_DRUG", "Saline", "saline_2ADC"
    )
    ui_drug_2ADC_subject = _subject_dropdown(
        "2ADC_DRUG", "Drug", "drug_2ADC"
    )
    ui_saline_2AFC_subject = _subject_dropdown(
        "2AFC_DRUG", "Saline", "saline_2AFC"
    )
    ui_drug_2AFC_subject = _subject_dropdown(
        "2AFC_DRUG", "Drug", "drug_2AFC"
    )
    return (
        ui_drug_2ADC_subject,
        ui_drug_2AFC_subject,
        ui_saline_2ADC_subject,
        ui_saline_2AFC_subject,
    )


@app.cell(hide_code=True)
def _(
    SELECTED_SESSIONS,
    mo,
    session_options,
    ui_drug_2ADC_subject,
    ui_drug_2AFC_subject,
    ui_saline_2ADC_subject,
    ui_saline_2AFC_subject,
):
    def _session_dropdown(task_name, treatment, subject, panel_key):
        _options = (
            session_options[task_name]
            .loc[
                lambda df: df["treatment"].eq(treatment)
                & df["subject"].eq(str(subject)),
                "session",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        _selected = SELECTED_SESSIONS[panel_key]["session"]
        _value = str(_selected) if str(_selected) in _options else _options[0]
        return mo.ui.dropdown(_options, value=_value, label="Session")

    ui_saline_2ADC_session = _session_dropdown(
        "2ADC_DRUG",
        "Saline",
        ui_saline_2ADC_subject.value,
        "saline_2ADC",
    )
    ui_drug_2ADC_session = _session_dropdown(
        "2ADC_DRUG",
        "Drug",
        ui_drug_2ADC_subject.value,
        "drug_2ADC",
    )
    ui_saline_2AFC_session = _session_dropdown(
        "2AFC_DRUG",
        "Saline",
        ui_saline_2AFC_subject.value,
        "saline_2AFC",
    )
    ui_drug_2AFC_session = _session_dropdown(
        "2AFC_DRUG",
        "Drug",
        ui_drug_2AFC_subject.value,
        "drug_2AFC",
    )
    return (
        ui_drug_2ADC_session,
        ui_drug_2AFC_session,
        ui_saline_2ADC_session,
        ui_saline_2AFC_session,
    )


@app.cell(hide_code=True)
def _(
    mo,
    ui_drug_2ADC_session,
    ui_drug_2ADC_subject,
    ui_drug_2AFC_session,
    ui_drug_2AFC_subject,
    ui_saline_2ADC_session,
    ui_saline_2ADC_subject,
    ui_saline_2AFC_session,
    ui_saline_2AFC_subject,
):
    mo.vstack(
        [
            mo.md("### Selected example sessions"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**2ADC saline**"),
                            ui_saline_2ADC_subject,
                            ui_saline_2ADC_session,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**2ADC drug**"),
                            ui_drug_2ADC_subject,
                            ui_drug_2ADC_session,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**2AFC saline**"),
                            ui_saline_2AFC_subject,
                            ui_saline_2AFC_session,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**2AFC drug**"),
                            ui_drug_2AFC_subject,
                            ui_drug_2AFC_session,
                        ]
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(
    adapters,
    build_session_repetition_data,
    pd,
    plot_dfs,
    treatment_trial_dfs,
    ui_drug_2AFC_session,
    ui_drug_2AFC_subject,
    ui_saline_2AFC_session,
    ui_saline_2AFC_subject,
):
    _selections = {
        "saline_2ADC": (
            "2ADC_DRUG",
            "N25",
            "5",
        ),
        "drug_2ADC": (
            "2ADC_DRUG",
            "N25",
            "7",
        ),
        "saline_2AFC": (
            "2AFC_DRUG",
            ui_saline_2AFC_subject.value,
            ui_saline_2AFC_session.value,
        ),
        "drug_2AFC": (
            "2AFC_DRUG",
            ui_drug_2AFC_subject.value,
            ui_drug_2AFC_session.value,
        ),
    }
    selected_session_data = {}

    for _panel_key, (_task_name, _subject, _session) in _selections.items():
        _running = build_session_repetition_data(
            plot_dfs[_task_name],
            subject=_subject,
            session=_session,
            adapter=adapters[_task_name],
            window=20,
        )
        _trial_col = (
            "trial_idx"
            if "trial_idx" in treatment_trial_dfs[_task_name].columns
            else "trial"
        )
        _session_df = (
            treatment_trial_dfs[_task_name]
            .loc[
                lambda df: df["subject"].eq(str(_subject))
                & df["session"].eq(str(_session))
            ]
            .sort_values(_trial_col)
            .reset_index(drop=True)
        )
        _posterior_cols = [
            _col
            for _col in _session_df.columns
            if str(_col).startswith("p_state_")
            and str(_col).rsplit("_", 1)[-1].isdigit()
        ]
        _engaged_idx = int(
            _session_df.loc[
                _session_df["state_label"].astype(str).eq("Engaged"),
                "state_idx",
            ].dropna().iloc[0]
        )
        _engaged_col = f"p_state_{_engaged_idx}"
        if _engaged_col not in _posterior_cols:
            raise KeyError(
                f"{_panel_key}: missing engaged posterior column {_engaged_col}"
            )

        _n_trials = min(len(_running), len(_session_df))
        _running = _running.iloc[:_n_trials].copy()
        selected_session_data[_panel_key] = {
            "running": _running,
            "trial_x": _running["trial_x"].to_numpy(),
            "p_engaged": pd.to_numeric(
                _session_df[_engaged_col],
                errors="coerce",
            ).iloc[:_n_trials].to_numpy(),
            "subject": str(_subject),
            "session": str(_session),
        }
    return (selected_session_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC Saline
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
    selected_session_data,
    state_palette,
):
    _data = selected_session_data["saline_2ADC"]
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_saline_2ADC = (
        plt.gca() if not mount_figure else axd["single_session_saline_2ADC"]
    )
    single_session_saline_2ADC.clear()
    single_session_saline_2ADC.plot(
        _data["trial_x"],
        _data["p_engaged"],
        color=state_palette["Engaged"],
        label=r"$p$(Engaged)",
    )
    for _trial_x, _probability in zip(
        _data["trial_x"], _data["p_engaged"], strict=False
    ):
        single_session_saline_2ADC.axvspan(
            _trial_x - 0.5,
            _trial_x + 0.5,
            color=state_palette[
                "Engaged" if _probability > 0.5 else "Disengaged"
            ],
            alpha=0.25,
            linewidth=0,
        )
    single_session_saline_2ADC.plot(
        "trial_x",
        "accuracy_window_fraction",
        data=_data["running"],
        color="black",
        linewidth=1,
        alpha=0.5,
        label="Accuracy",
    )
    single_session_saline_2ADC.set(
        title="2ADC · Saline",
        xlabel="Trial",
        ylabel="Probability",
        xlim=(-0.5, len(_data["trial_x"]) - 0.5),
        ylim=(0, 1),
    )
    single_session_saline_2ADC.set_yticks([0, 0.5, 1])
    single_session_saline_2ADC.legend(frameon=False, loc="lower right")
    if not mount_figure:
        single_session_saline_2ADC.figure.savefig(
            (path_panels / "2ADC_saline_single_session").with_suffix(
                f".{format}"
            )
        )
    single_session_saline_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC Saline
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
    selected_session_data,
    state_palette,
):
    _data = selected_session_data["saline_2AFC"]
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_saline_2AFC = (
        plt.gca() if not mount_figure else axd["single_session_saline_2AFC"]
    )
    single_session_saline_2AFC.clear()
    single_session_saline_2AFC.plot(
        _data["trial_x"],
        _data["p_engaged"],
        color=state_palette["Engaged"],
        label=r"$p$(Engaged)",
    )
    for _trial_x, _probability in zip(
        _data["trial_x"], _data["p_engaged"], strict=False
    ):
        single_session_saline_2AFC.axvspan(
            _trial_x - 0.5,
            _trial_x + 0.5,
            color=state_palette[
                "Engaged" if _probability > 0.5 else "Disengaged"
            ],
            alpha=0.25,
            linewidth=0,
        )
    single_session_saline_2AFC.plot(
        "trial_x",
        "accuracy_window_fraction",
        data=_data["running"],
        color="black",
        linewidth=1,
        alpha=0.5,
        label="Accuracy",
    )
    single_session_saline_2AFC.set(
        title="2AFC · Saline",
        xlabel="Trial",
        ylabel="Probability",
        xlim=(-0.5, len(_data["trial_x"]) - 0.5),
        ylim=(0, 1),
    )
    single_session_saline_2AFC.set_yticks([0, 0.5, 1])
    single_session_saline_2AFC.legend(frameon=False, loc="lower right")
    if not mount_figure:
        single_session_saline_2AFC.figure.savefig(
            (path_panels / "2AFC_saline_single_session").with_suffix(
                f".{format}"
            )
        )
    single_session_saline_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC Drug
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
    selected_session_data,
    state_palette,
):
    _data = selected_session_data["drug_2ADC"]
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_drug_2ADC = (
        plt.gca() if not mount_figure else axd["single_session_drug_2ADC"]
    )
    single_session_drug_2ADC.clear()
    single_session_drug_2ADC.plot(
        _data["trial_x"],
        _data["p_engaged"],
        color=state_palette["Engaged"],
        label=r"$p$(Engaged)",
    )
    for _trial_x, _probability in zip(
        _data["trial_x"], _data["p_engaged"], strict=False
    ):
        single_session_drug_2ADC.axvspan(
            _trial_x - 0.5,
            _trial_x + 0.5,
            color=state_palette[
                "Engaged" if _probability > 0.5 else "Disengaged"
            ],
            alpha=0.25,
            linewidth=0,
        )
    single_session_drug_2ADC.plot(
        "trial_x",
        "accuracy_window_fraction",
        data=_data["running"],
        color="black",
        linewidth=1,
        alpha=0.5,
        label="Accuracy",
    )
    single_session_drug_2ADC.set(
        title="2ADC · Drug",
        xlabel="Trial",
        ylabel="Probability",
        xlim=(-0.5, len(_data["trial_x"]) - 0.5),
        ylim=(0, 1),
    )
    single_session_drug_2ADC.set_yticks([0, 0.5, 1])
    single_session_drug_2ADC.legend(frameon=False, loc="lower right")
    if not mount_figure:
        single_session_drug_2ADC.figure.savefig(
            (path_panels / "2ADC_drug_single_session").with_suffix(
                f".{format}"
            )
        )
    single_session_drug_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC Drug
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
    selected_session_data,
    state_palette,
):
    _data = selected_session_data["drug_2AFC"]
    plt.figure(figsize=fig_size(1, 3), constrained_layout=True)
    single_session_drug_2AFC = (
        plt.gca() if not mount_figure else axd["single_session_drug_2AFC"]
    )
    single_session_drug_2AFC.clear()
    single_session_drug_2AFC.plot(
        _data["trial_x"],
        _data["p_engaged"],
        color=state_palette["Engaged"],
        label=r"$p$(Engaged)",
    )
    for _trial_x, _probability in zip(
        _data["trial_x"], _data["p_engaged"], strict=False
    ):
        single_session_drug_2AFC.axvspan(
            _trial_x - 0.5,
            _trial_x + 0.5,
            color=state_palette[
                "Engaged" if _probability > 0.5 else "Disengaged"
            ],
            alpha=0.25,
            linewidth=0,
        )
    single_session_drug_2AFC.plot(
        "trial_x",
        "accuracy_window_fraction",
        data=_data["running"],
        color="black",
        linewidth=1,
        alpha=0.5,
        label="Accuracy",
    )
    single_session_drug_2AFC.set(
        title="2AFC · Drug",
        xlabel="Trial",
        ylabel="Probability",
        xlim=(-0.5, len(_data["trial_x"]) - 0.5),
        ylim=(0, 1),
    )
    single_session_drug_2AFC.set_yticks([0, 0.5, 1])
    single_session_drug_2AFC.legend(frameon=False, loc="lower right")
    if not mount_figure:
        single_session_drug_2AFC.figure.savefig(
            (path_panels / "2AFC_drug_single_session").with_suffix(
                f".{format}"
            )
        )
    single_session_drug_2AFC
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
        load_model_metrics("2ADC_DRUG", "glm", "one hot2", "GLM"),
        load_model_metrics("2ADC_DRUG", "glmhmm", MODEL_BY_TASK["2ADC_DRUG"], "GLM-HMM-T"),
    )
    ll_bic_delta_2AFC = model_delta_df(
        load_model_metrics("2AFC_DRUG", "glm", "one hot2", "GLM"),
        load_model_metrics("2AFC_DRUG", "glmhmm", MODEL_BY_TASK["2AFC_DRUG"], "GLM-HMM-T"),
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


    plot_df_2AFC = plot_dfs["2AFC_DRUG"].to_pandas()
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
    model_comparison_ll_2ADC.set_title(task_labels["2ADC_DRUG"])
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
    model_comparison_ll_2AFC.set_title(task_labels["2AFC_DRUG"])
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
    # model_comparison_bic_2AFC.set_title(task_labels["2AFC_DRUG"])
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
def _(dfs):
    dfs["2ADC_DRUG"]
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
        for _name in (
            "single_session_drug_2ADC",
            "single_session_saline_2AFC",
            "single_session_drug_2AFC",
            "histogram_transitions_2AFC",
            "dwell_time_2AFC",
            "transition_weights_2AFC",
            "psychometric_2AFC",
        ):
            _legend = axd[_name].get_legend()
            if _legend is not None:
                _legend.remove()

        axd["single_session_saline_2AFC"].set_yticklabels([])
        axd["single_session_drug_2AFC"].set_yticklabels([])
        axd["transition_weights_2AFC"].set_yticklabels([])
        axd["psychometric_2AFC"].set_yticklabels([])

        axd["histogram_transitions_2ADC"].set_title("2ADC")
        axd["histogram_transitions_2AFC"].set_title("2AFC")
        axd["transition_weights_2ADC"].set_title("2ADC")
        axd["transition_weights_2AFC"].set_title("2AFC")
        axd["psychometric_2ADC"].set_title("2ADC")
        axd["psychometric_2AFC"].set_title("2AFC")

        fig.savefig((path_panels / "figure4").with_suffix(f".{format}"))
        fig.align_ylabels()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
