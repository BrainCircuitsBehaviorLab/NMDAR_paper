# /// script
# [tool.marimo.opengraph]
# title = "Figure 4 Models"
# description = "Figure 4: model weights, state dynamics, and behavior under saline and drug."
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
        glmhmmt_state_dwell_df,
        glmhmmt_state_switches_df,
        glmhmmt_transition_weights_df,
    )
    from src.plots.common import (
        BOXPLOT_STYLE,
        fig_size,
    )

    return (
        Annotator,
        BOXPLOT_STYLE,
        Path,
        add_choice_lag_summary_regressor,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        glmhmmt_state_dwell_df,
        glmhmmt_state_switches_df,
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
def _(Annotator, np, pd, ttest_1samp):
    def add_one_sample_zero_annotations(ax, df, *, x, y, order, hue=None, hue_order=None, show_pvalue_if_ns=False):
        """Annotate each feature and transition with its one-sample test against zero."""
        if df is None or df.empty or not {x, y}.issubset(df.columns):
            return

        if hue is None:
            groups = [(x_value, None) for x_value in order]
        else:
            groups = [(x_value, hue_value) for x_value in order for hue_value in hue_order]

        pairs = []
        pvalues = []
        for x_value, hue_value in groups:
            mask = df[x] == x_value
            if hue_value is not None:
                mask &= df[hue] == hue_value
            values = pd.to_numeric(df.loc[mask, y], errors="coerce").dropna()
            if len(values) < 2:
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=0.0).pvalue)
            if not np.isfinite(pvalue):
                continue
            key = x_value if hue_value is None else (x_value, hue_value)
            pairs.append((key, key))
            pvalues.append(pvalue)

        if not pairs:
            return

        annotator_kwargs = dict(x=x, y=y, order=order)
        if hue is not None:
            annotator_kwargs.update(hue=hue, hue_order=hue_order)

        annotator = Annotator(ax, pairs, data=df, **annotator_kwargs)
        annotator.configure(line_width=0, text_format="star", verbose=0)

        texts_before = {id(t) for t in ax.texts}

        if show_pvalue_if_ns:
            def _label(pvalue):
                if pvalue >= 0.05:
                    return f"p={pvalue:.3f}"
                    # return f"p={pvalue:.3f}".replace("p=0.", "p=.")
                if pvalue < 0.001:
                    return "***"
                if pvalue < 0.01:
                    return "**"
                return "*"

            annotator.set_custom_annotations([_label(p) for p in pvalues])
            annotator.annotate()
        else:
            annotator.set_pvalues_and_annotate(pvalues)

        # Pin all new annotation texts to a uniform y (the highest one wins).
        new_texts = [t for t in ax.texts if id(t) not in texts_before]
        if new_texts:
            target_y = max(t.get_position()[1] for t in new_texts)
            for t in new_texts:
                t.set_position((t.get_position()[0], target_y))

        return annotator

    return (add_one_sample_zero_annotations,)


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
    return (mount_figure,)


@app.cell
def _():
    model_type = "glmhmmt"
    return (model_type,)


@app.cell
def _():
    MODEL_BY_TASK = {
        # "2AFC_DRUG": "drug_transitions2",
        # "2ADC_DRUG": "drug_transitions2",
        "2AFC_DRUG": "drug_transitions2_nocv",
        "2ADC_DRUG": "drug_transitions2_nocv",
        # "2AFC_DRUG": "drug_transitions_emissions",
        # "2ADC_DRUG": "drug_transitions_emissions",
        # "2AFC_DRUG": "base_model",
        # "2ADC_DRUG": "base_param",
        # "2AFC_DRUG": "drug_emissions",
        # "2ADC_DRUG": "drug_emissions",
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
def _(Path, configure_paths, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    project_path = ROOT
    path_panels = project_path / "figures" / "panels4_models"
    for _format in ("svg", "png"):
        os.makedirs(path_panels / _format, exist_ok=True)
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
    transition_palette = {
        "Engaged -> Disengaged": "tab:gray",
        "Disengaged -> Engaged": "tab:green",
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
        "cumulative_reward": "Cum.\nreward",
        "trial_index": "Trial index",
        "Drug": "Drug",
        "drug_code": "Drug",
        "drug_x_stim_param": "Drug ×\nStim.",
        "drug_x_stim_x_delay_param": "Drug ×\nStim.",
        "drug_x_choice_lag_param": "Drug ×\nA",
        "drug_x_filtered_reward": "NMDAr block : Rew trace",
    }
    return (
        feature_labels,
        state_palette,
        task_labels,
        transition_palette,
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
def _(Annotator, pd):
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
        show_pvalue_if_ns=False,
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
        )

        if show_pvalue_if_ns:
            annotator.apply_test()
            labels = [
                f"p={_ann.data.pvalue:.3f}" if _ann.data.pvalue >= 0.05 else _ann.text
                for _ann in annotator.annotations
            ]
            annotator.set_custom_annotations(labels)
            annotator.annotate()
        else:
            annotator.apply_and_annotate()


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


    return add_paired_state_annotation, add_subject_pair_lines, state_order


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
    pl,
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

    # Sessions excluded from 2AFC plotting (post model-alignment, so unaffected
    # subjects/sessions keep their full trial counts).
    EXCLUDED_2AFC_SESSIONS = []

    if "2AFC_DRUG" in trial_dfs:
        trial_dfs["2AFC_DRUG"] = trial_dfs["2AFC_DRUG"].filter(
            ~pl.col("session").is_in(EXCLUDED_2AFC_SESSIONS)
        )
        plot_dfs["2AFC_DRUG"] = plot_dfs["2AFC_DRUG"].filter(
            ~pl.col("session").is_in(EXCLUDED_2AFC_SESSIONS)
        )
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
    arrays_by_task,
    glmhmmt_state_dwell_df,
    glmhmmt_state_switches_df,
    glmhmmt_transition_weights_df,
    plot_dfs,
    views,
    weight_dfs,
    with_feature_labels,
):
    emission_plot_dfs = {}
    transition_plot_dfs = {}
    dwell_dfs = {}
    switch_session_dfs = {}
    for _task in active_task_names:
        emission_plot_dfs[_task] = with_feature_labels(weight_dfs[_task].to_pandas())
        _weights = glmhmmt_transition_weights_df(arrays_by_task[_task], views[_task])
        transition_plot_dfs[_task] = with_feature_labels(
            _weights[_weights["source_state_label"] != _weights["destination_state_label"]]
        )
        dwell_dfs[_task] = glmhmmt_state_dwell_df(plot_dfs[_task])
        switch_session_dfs[_task], _ = glmhmmt_state_switches_df(plot_dfs[_task])
    return (
        dwell_dfs,
        emission_plot_dfs,
        switch_session_dfs,
        transition_plot_dfs,
    )


@app.cell
def _(active_task_names, dwell_dfs, pl, plot_dfs, switch_session_dfs):
    treatment_dwell_dfs = {}
    treatment_switch_dfs = {}
    treatment_accuracy_dfs = {}
    engaged_accuracy_dfs = {}
    treatment_occupancy_dfs = {}
    for _task in active_task_names:
        _trials = plot_dfs[_task].with_columns(
            pl.col("condition").cast(pl.String).str.to_lowercase().replace_strict(
                {"saline": "Saline", "drug": "Drug"}, default=None,
            ).alias("treatment"),
            pl.col("subject").cast(pl.String),
            pl.col("session").cast(pl.String),
        ).drop_nulls(["subject", "session", "treatment"])
        _session_treatments = _trials.select("subject", "session", "treatment").unique().to_pandas()
        treatment_dwell_dfs[_task] = (
            dwell_dfs[_task].astype({"subject": str, "session": str})
            .merge(_session_treatments, on=["subject", "session"], validate="many_to_one")
            .groupby(["subject", "treatment", "state_label"], as_index=False, observed=True)
            .agg(mean_dwell_trials=("dwell_trials", "mean"))
        )
        treatment_switch_dfs[_task] = (
            switch_session_dfs[_task].astype({"subject": str, "session": str})
            .merge(_session_treatments, on=["subject", "session"], validate="one_to_one")
        )
        # Accuracy retains one observation per animal and treatment.
        treatment_accuracy_dfs[_task] = (
            _trials.group_by("subject", "treatment")
            .agg(pl.col("correct_bool").mean().alias("accuracy"))
            .sort("subject", "treatment").to_pandas()
        )
        engaged_accuracy_dfs[_task] = (
            _trials.filter(pl.col("state_label") == "Engaged")
            .group_by("subject", "treatment")
            .agg(pl.col("correct_bool").mean().alias("accuracy"))
            .sort("subject", "treatment").to_pandas()
        )
        # Count all classified trials, including sessions with zero engaged trials.
        treatment_occupancy_dfs[_task] = (
            _trials.drop_nulls("state_label")
            .group_by("subject", "session", "treatment")
            .agg(
                (pl.col("state_label") == "Engaged").sum().alias("n_engaged"),
                pl.len().alias("n_trials"),
            )
            .with_columns((pl.col("n_engaged") / pl.col("n_trials")).alias("occupancy"))
            .sort("subject", "session", "treatment").to_pandas()
        )
    return (
        engaged_accuracy_dfs,
        treatment_accuracy_dfs,
        treatment_dwell_dfs,
        treatment_occupancy_dfs,
        treatment_switch_dfs,
    )


@app.cell
def _(active_task_names, emission_plot_dfs, state_order, transition_plot_dfs):
    def ordered_features(df, *, bias_last=False):
        """Order regressors, optionally placing bias last for emission weights."""
        def priority(value):
            label = str(value).lower()
            return (4 if bias_last else 0) if label == "bias" else 1 if "stim" in label else 2 if label == "a" else 3
        return sorted(dict.fromkeys(df["feature_label"]), key=priority)

    emission_orders = {task: ordered_features(emission_plot_dfs[task], bias_last=True) for task in active_task_names}
    emission_hue_orders = {
        task: [state for state in state_order if state in set(emission_plot_dfs[task]["state_label"])]
        for task in active_task_names
    }
    transition_orders = {task: ordered_features(transition_plot_dfs[task]) for task in active_task_names}
    return emission_hue_orders, emission_orders, transition_orders


@app.cell
def _(treatment_switch_dfs):
    switch_xlim = (-0.5, max(df["n_switches"].max() for df in treatment_switch_dfs.values()) + 0.5)
    return (switch_xlim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Plots
    """)
    return


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [
                ["histogram_transitions_2ADC"] * 2 + ["dwell_time_2ADC"] + ["histogram_transitions_2AFC"] * 2 + ["dwell_time_2AFC"],
                ["emission_weights_2ADC"] * 2 + ["transition_weights_2ADC"] + ["emission_weights_2AFC"] * 2 + ["transition_weights_2AFC"],
                ["accuracy_2ADC", "engaged_accuracy_2ADC", "occupancy_2ADC", "accuracy_2AFC", "engaged_accuracy_2AFC", "occupancy_2AFC"],
            ],
            figsize=fig_size(1),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State Switch Histograms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
    """)
    return


@app.cell
def _(
    axd,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    switch_xlim,
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_switch_dfs,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
        binwidth=1,
        binrange=switch_xlim,
        stat="probability",
        common_norm=False,
        element="step",
        multiple="layer",
        alpha=0.5,
        ax=histogram_transitions_2ADC,
    )
    _legend = histogram_transitions_2ADC.get_legend()
    _legend.set_frame_on(False)
    _legend.set_title(None)
    _legend.set_loc("upper right")
    histogram_transitions_2ADC.set_title(task_labels["2ADC_DRUG"])
    histogram_transitions_2ADC.set_xlabel("State switches")
    histogram_transitions_2ADC.set_ylabel("Probability")
    histogram_transitions_2ADC.set_xlim(switch_xlim)
    if not mount_figure:
        for _format in ("svg", "png"):
            histogram_transitions_2ADC.figure.savefig((path_panels / _format / "2ADC_drug_state_switch_histogram").with_suffix(f".{_format}"))
    histogram_transitions_2ADC
    return (histogram_transitions_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
    """)
    return


@app.cell
def _(
    axd,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    switch_xlim,
    task_labels,
    treatment_order,
    treatment_palette,
    treatment_switch_dfs,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
        binwidth=1,
        binrange=switch_xlim,
        stat="probability",
        common_norm=False,
        element="step",
        multiple="layer",
        alpha=0.5,
        ax=histogram_transitions_2AFC,
    )
    _legend = histogram_transitions_2AFC.get_legend()
    _legend.set_frame_on(False)
    _legend.set_title(None)
    histogram_transitions_2AFC.set_title(task_labels["2AFC_DRUG"])
    histogram_transitions_2AFC.set_xlabel("State switches")
    histogram_transitions_2AFC.set_ylabel("Probability")
    histogram_transitions_2AFC.set_xlim(switch_xlim)
    if not mount_figure:
        for _format in ("svg", "png"):
            histogram_transitions_2AFC.figure.savefig((path_panels / _format / "2AFC_drug_state_switch_histogram").with_suffix(f".{_format}"))
    histogram_transitions_2AFC
    return (histogram_transitions_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Dwell Time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    add_paired_state_annotation,
    add_subject_pair_lines,
    axd,
    fig_size,
    mount_figure,
    plt,
    sns,
    treatment_dwell_dfs,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
    dwell_time_2ADC.set_yscale('log')
    add_subject_pair_lines(
        dwell_time_2ADC,
        treatment_dwell_dfs["2ADC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        order=["Engaged", "Disengaged"],
        hue="treatment",
        hue_order=treatment_order,
    )
    add_paired_state_annotation(
        dwell_time_2ADC,
        treatment_dwell_dfs["2ADC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        order=["Engaged", "Disengaged"],
        hue="treatment",
        hue_order=treatment_order,
        # show_pvalue_if_ns=True
    )
    # dwell_time_2ADC.set_title(task_labels["2ADC_DRUG"])
    dwell_time_2ADC.set_xlabel("State")
    dwell_time_2ADC.set_ylabel("Dwell time (trials)")
    dwell_time_2ADC.set_xticks([0, 1], ["Eng.", "Dis."])
    dwell_time_2ADC
    return (dwell_time_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    add_paired_state_annotation,
    add_subject_pair_lines,
    axd,
    fig_size,
    mount_figure,
    plt,
    sns,
    treatment_dwell_dfs,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
    dwell_time_2AFC.set_yscale('log')
    add_subject_pair_lines(
        dwell_time_2AFC,
        treatment_dwell_dfs["2AFC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        order=["Engaged", "Disengaged"],
        hue="treatment",
        hue_order=treatment_order,
    )
    add_paired_state_annotation(
        dwell_time_2AFC,
        treatment_dwell_dfs["2AFC_DRUG"],
        x="state_label",
        y="mean_dwell_trials",
        order=["Engaged", "Disengaged"],
        hue="treatment",
        hue_order=treatment_order,
        # show_pvalue_if_ns=True
    )
    # dwell_time_2AFC.set_title(task_labels["2AFC_DRUG"])
    dwell_time_2AFC.set_xlabel("State")
    dwell_time_2AFC.set_ylabel("Dwell time (trials)")
    dwell_time_2AFC.set_xticks([0, 1], ["Eng.", "Dis."])
    dwell_time_2AFC
    return (dwell_time_2AFC,)


@app.cell
def _(dwell_time_2ADC, dwell_time_2AFC, mount_figure, path_panels):
    dwell_ylim = (
        min(axis.get_ylim()[0] for axis in (dwell_time_2ADC, dwell_time_2AFC)),
        max(axis.get_ylim()[1] for axis in (dwell_time_2ADC, dwell_time_2AFC)),
    )
    for _task, _axis in (("2ADC", dwell_time_2ADC), ("2AFC", dwell_time_2AFC)):
        _axis.set_ylim(dwell_ylim)
        if not mount_figure:
            for _format in ("svg", "png"):
                _axis.figure.savefig((path_panels / _format / f"{_task}_drug_dwell_time").with_suffix(f".{_format}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Emission Weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
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
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    emission_weights_2ADC = plt.gca() if not mount_figure else axd["emission_weights_2ADC"]
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
    add_subject_pair_lines(emission_weights_2ADC, emission_plot_dfs["2ADC_DRUG"], x="feature_label", y="weight", order=emission_orders["2ADC_DRUG"])
    add_paired_state_annotation(emission_weights_2ADC, emission_plot_dfs["2ADC_DRUG"], x="feature_label", y="weight", order=emission_orders["2ADC_DRUG"])
    emission_weights_2ADC.axhline(0, color="0.5", linestyle="--")
    emission_weights_2ADC.set_title(task_labels["2ADC_DRUG"])
    emission_weights_2ADC.set_xlabel("")
    emission_weights_2ADC.set_ylabel("Emission weight")
    emission_weights_2ADC.tick_params(axis="x")
    emission_weights_2ADC.legend(frameon=False, title="", ncols=2, fontsize=6, handletextpad=0.3)
    if not mount_figure:
        for _format in ("svg", "png"):
            emission_weights_2ADC.figure.savefig((path_panels / _format / "2AFC_delay_glmhmmt_emission_weights").with_suffix(f".{_format}"))
    emission_weights_2ADC
    return (emission_weights_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
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
    mount_figure,
    path_panels,
    plt,
    sns,
    state_palette,
    task_labels,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    emission_weights_2AFC = plt.gca() if not mount_figure else axd["emission_weights_2AFC"]
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
    add_subject_pair_lines(emission_weights_2AFC, emission_plot_dfs["2AFC_DRUG"], x="feature_label", y="weight", order=emission_orders["2AFC_DRUG"])
    add_paired_state_annotation(emission_weights_2AFC, emission_plot_dfs["2AFC_DRUG"], x="feature_label", y="weight", order=emission_orders["2AFC_DRUG"])
    emission_weights_2AFC.axhline(0, color="0.5", linestyle="--")
    emission_weights_2AFC.set_title(task_labels["2AFC_DRUG"])
    emission_weights_2AFC.set_xlabel("")
    emission_weights_2AFC.set_ylabel("Emission weight")
    emission_weights_2AFC.tick_params(axis="x")
    emission_weights_2AFC.legend(frameon=False, title="")
    emission_weights_2AFC.legend().remove()
    if not mount_figure:
        for _format in ("svg", "png"):
            emission_weights_2AFC.figure.savefig((path_panels / _format / "2AFC_glmhmmt_emission_weights").with_suffix(f".{_format}"))
    emission_weights_2AFC
    return (emission_weights_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Transition Weights
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
    """)
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    model_type,
    mount_figure,
    path_panels,
    plt,
    sns,
    transition_orders,
    transition_palette,
    transition_plot_dfs,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    transition_weights_2ADC = (
        plt.gca() if not mount_figure else axd["transition_weights_2ADC"]
    )
    transition_weights_2ADC.clear()
    if model_type != "glmhmmt":
        transition_weights_2ADC.set_axis_off()
        transition_weights_2ADC.text(0.5, 0.5, "Transition", ha="center", va="center",)
        transition_weights_2ADC.text(0.5, 0.3, "Weights", ha="center", va="center",)
    else: 


        sns.boxplot(
            data=transition_plot_dfs["2ADC_DRUG"],
            x="feature_label",
            y="weight",
            hue="transition_label",
            order=transition_orders["2ADC_DRUG"],
            palette=transition_palette,
            ax=transition_weights_2ADC,
            **boxplot_STYLE,
        )
        transition_weights_2ADC.axhline(0, color="0.5", linestyle="--")
        add_subject_pair_lines(
            transition_weights_2ADC,
            transition_plot_dfs["2ADC_DRUG"],
            x="feature_label",
            y="weight",
            order=transition_orders["2ADC_DRUG"],
            hue="transition_label",
            hue_order=["Engaged -> Disengaged", "Disengaged -> Engaged"],
        )
        # add_one_sample_zero_annotations(transition_weights_2ADC, transition_plot_dfs["2ADC_DRUG"], x="feature_label", y="weight", order=transition_orders["2ADC_DRUG"])
        add_one_sample_zero_annotations(
            transition_weights_2ADC,
            transition_plot_dfs["2ADC_DRUG"],
            x="feature_label",
            y="weight",
            order=transition_orders["2ADC_DRUG"],
            hue="transition_label",
            hue_order=["Engaged -> Disengaged", "Disengaged -> Engaged"],
            show_pvalue_if_ns=False
        )
    # transition_weights_2ADC.set_title(task_labels["2ADC_DRUG"])
    transition_weights_2ADC.set_xlabel("")
    transition_weights_2ADC.set_ylabel("Transition weight")
    transition_weights_2ADC.tick_params(axis="x", labelrotation=0)
    handles, _ = transition_weights_2ADC.get_legend_handles_labels()

    transition_weights_2ADC.legend(
        handles,
        [r"E$\rightarrow$D", r"D$\rightarrow$E"],
        frameon=False, ncol=1, handlelength=1, handletextpad=0.5 ,columnspacing=1, loc='upper left', bbox_to_anchor=(-0.05, 1.05)
    )
    transition_weights_2ADC.legend_.remove()
    if not mount_figure:
        for _format in ("svg", "png"):
            transition_weights_2ADC.figure.savefig((path_panels / _format / "2AFC_delay_glmhmmt_transition_weights").with_suffix(f".{_format}"))
    transition_weights_2ADC
    return (transition_weights_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    add_subject_pair_lines,
    axd,
    boxplot_STYLE,
    fig_size,
    model_type,
    mount_figure,
    path_panels,
    plt,
    sns,
    transition_orders,
    transition_palette,
    transition_plot_dfs,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
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
            palette=transition_palette,
            ax=transition_weights_2AFC,
            **boxplot_STYLE,
        )

        transition_weights_2AFC.axhline(0, color="0.5", linestyle="--")

        add_subject_pair_lines(
            transition_weights_2AFC,
            transition_plot_dfs["2AFC_DRUG"],
            x="feature_label",
            y="weight",
            order=transition_orders["2AFC_DRUG"],
            hue="transition_label",
            hue_order=["Engaged -> Disengaged", "Disengaged -> Engaged"],
        )

        # add_one_sample_zero_annotations(
        #     transition_weights_2AFC,
        #     transition_plot_dfs["2AFC_DRUG"],
        #     x="feature_label",
        #     y="weight",
        #     order=transition_orders["2AFC_DRUG"],
        # )
        add_one_sample_zero_annotations(
            transition_weights_2AFC,
            transition_plot_dfs["2AFC_DRUG"],
            x="feature_label",
            y="weight",
            order=transition_orders["2AFC_DRUG"],
            hue="transition_label",
            hue_order=["Engaged -> Disengaged", "Disengaged -> Engaged"],
            show_pvalue_if_ns=False
        )

        _handles, _ = transition_weights_2AFC.get_legend_handles_labels()
        transition_weights_2AFC.legend(
            _handles,
            [r"$E\rightarrow D$", r"$D\rightarrow E$"],
            ncols=2,
            frameon=False,
             handlelength=0.8,handletextpad=0.4,columnspacing=0.8,
        )
    transition_weights_2AFC.legend_.remove()
    # transition_weights_2AFC.set_title(task_labels["2AFC_DRUG"])
    transition_weights_2AFC.set_xlabel("")
    transition_weights_2AFC.set_ylabel("Transition weight")
    transition_weights_2AFC.tick_params(axis="x", rotation=0)
    if not mount_figure:
        for _format in ("svg", "png"):
            transition_weights_2AFC.figure.savefig((path_panels / _format / "2AFC_glmhmmt_transition_weights").with_suffix(f".{_format}"))

    transition_weights_2AFC
    return (transition_weights_2AFC,)


@app.cell
def _(Annotator, treatment_order):
    from scipy.stats import ttest_rel

    def add_paired_treatment_comparison(ax, df, *, y):
        """Connect animal means and test paired Saline/Drug values across animals."""
        paired = df.pivot_table(
            index="subject", columns="treatment", values=y, aggfunc="mean",
        ).reindex(columns=treatment_order).dropna()
        ax.plot([0, 1], paired.to_numpy().T, color="0.75", linewidth=0.5, zorder=0)
        if len(paired) < 2:
            return
        pvalue = ttest_rel(paired[treatment_order[0]], paired[treatment_order[1]]).pvalue
        paired_df = paired.reset_index().melt(
            id_vars="subject", var_name="treatment", value_name=y,
        )
        annotator = Annotator(
            ax, [tuple(treatment_order)], data=paired_df,
            x="treatment", y=y, order=treatment_order,
        )
        annotator.configure(text_format="star", line_height=0, verbose=False)
        annotator.set_pvalues_and_annotate([pvalue])

    return (add_paired_treatment_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Accuracy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
    """)
    return


@app.cell
def _(
    add_paired_treatment_comparison,
    axd,
    boxplot_STYLE,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_accuracy_dfs,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    accuracy_2ADC = axd["accuracy_2ADC"] if mount_figure else plt.gca()
    accuracy_2ADC.clear()
    sns.boxplot(
        data=treatment_accuracy_dfs["2ADC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        ax=accuracy_2ADC, **boxplot_STYLE,
    )
    sns.stripplot(
        data=treatment_accuracy_dfs["2ADC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        jitter=0.15, size=2.5, alpha=0.6, ax=accuracy_2ADC,
    )
    accuracy_2ADC.set_title(task_labels["2ADC_DRUG"])
    accuracy_2ADC.set_xlabel("")
    accuracy_2ADC.set_ylabel("Accuracy")
    accuracy_2ADC.set_ylim(-0.03, 1.03)
    add_paired_treatment_comparison(accuracy_2ADC, treatment_accuracy_dfs["2ADC_DRUG"], y="accuracy")
    accuracy_2ADC.set_xticks([0, 1], treatment_order, rotation=45, ha="right")
    if not mount_figure:
        for _format in ("svg", "png"):
            accuracy_2ADC.figure.savefig((path_panels / _format / "2ADC_accuracy").with_suffix(f".{_format}"))
    accuracy_2ADC
    return (accuracy_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_paired_treatment_comparison,
    axd,
    boxplot_STYLE,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_accuracy_dfs,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    accuracy_2AFC = axd["accuracy_2AFC"] if mount_figure else plt.gca()
    accuracy_2AFC.clear()
    sns.boxplot(
        data=treatment_accuracy_dfs["2AFC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        ax=accuracy_2AFC, **boxplot_STYLE,
    )
    sns.stripplot(
        data=treatment_accuracy_dfs["2AFC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        jitter=0.15, size=2.5, alpha=0.6, ax=accuracy_2AFC,
    )
    accuracy_2AFC.set_title(task_labels["2AFC_DRUG"])
    accuracy_2AFC.set_xlabel("")
    accuracy_2AFC.set_ylabel("Accuracy")
    accuracy_2AFC.set_ylim(-0.03, 1.03)
    add_paired_treatment_comparison(accuracy_2AFC, treatment_accuracy_dfs["2AFC_DRUG"], y="accuracy")
    accuracy_2AFC.set_xticks([0, 1], treatment_order, rotation=45, ha="right")
    if not mount_figure:
        for _format in ("svg", "png"):
            accuracy_2AFC.figure.savefig((path_panels / _format / "2AFC_accuracy").with_suffix(f".{_format}"))
    accuracy_2AFC
    return (accuracy_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Accuracy Only In Engaged
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
    """)
    return


@app.cell
def _(
    add_paired_treatment_comparison,
    axd,
    boxplot_STYLE,
    engaged_accuracy_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    engaged_accuracy_2ADC = axd["engaged_accuracy_2ADC"] if mount_figure else plt.gca()
    engaged_accuracy_2ADC.clear()
    sns.boxplot(
        data=engaged_accuracy_dfs["2ADC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        ax=engaged_accuracy_2ADC, **boxplot_STYLE,
    )
    sns.stripplot(
        data=engaged_accuracy_dfs["2ADC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        jitter=0.15, size=2.5, alpha=0.6, ax=engaged_accuracy_2ADC,
    )
    engaged_accuracy_2ADC.set_title(task_labels["2ADC_DRUG"])
    engaged_accuracy_2ADC.set_xlabel("")
    engaged_accuracy_2ADC.set_ylabel("Accuracy (engaged)")
    engaged_accuracy_2ADC.set_ylim(-0.03, 1.03)
    add_paired_treatment_comparison(engaged_accuracy_2ADC, engaged_accuracy_dfs["2ADC_DRUG"], y="accuracy")
    engaged_accuracy_2ADC.set_xticks([0, 1], treatment_order, rotation=45, ha="right")
    if not mount_figure:
        for _format in ("svg", "png"):
            engaged_accuracy_2ADC.figure.savefig((path_panels / _format / "2ADC_engaged_accuracy").with_suffix(f".{_format}"))
    engaged_accuracy_2ADC
    return (engaged_accuracy_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_paired_treatment_comparison,
    axd,
    boxplot_STYLE,
    engaged_accuracy_dfs,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    engaged_accuracy_2AFC = axd["engaged_accuracy_2AFC"] if mount_figure else plt.gca()
    engaged_accuracy_2AFC.clear()
    sns.boxplot(
        data=engaged_accuracy_dfs["2AFC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        ax=engaged_accuracy_2AFC, **boxplot_STYLE,
    )
    sns.stripplot(
        data=engaged_accuracy_dfs["2AFC_DRUG"],
        x="treatment", y="accuracy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        jitter=0.15, size=2.5, alpha=0.6, ax=engaged_accuracy_2AFC,
    )
    engaged_accuracy_2AFC.set_title(task_labels["2AFC_DRUG"])
    engaged_accuracy_2AFC.set_xlabel("")
    engaged_accuracy_2AFC.set_ylabel("Accuracy (engaged)")
    engaged_accuracy_2AFC.set_ylim(-0.03, 1.03)
    add_paired_treatment_comparison(engaged_accuracy_2AFC, engaged_accuracy_dfs["2AFC_DRUG"], y="accuracy")
    engaged_accuracy_2AFC.set_xticks([0, 1], treatment_order, rotation=45, ha="right")
    if not mount_figure:
        for _format in ("svg", "png"):
            engaged_accuracy_2AFC.figure.savefig((path_panels / _format / "2AFC_engaged_accuracy").with_suffix(f".{_format}"))
    engaged_accuracy_2AFC
    return (engaged_accuracy_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Engaged Occupancy Per Session
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Each point is one session: engaged trials divided by all classified trials in that session.
    Paired lines and paired t-tests compare each animal’s mean session occupancy between Saline and Drug.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2ADC
    """)
    return


@app.cell
def _(
    add_paired_treatment_comparison,
    axd,
    boxplot_STYLE,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_occupancy_dfs,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    occupancy_2ADC = axd["occupancy_2ADC"] if mount_figure else plt.gca()
    occupancy_2ADC.clear()
    sns.boxplot(
        data=treatment_occupancy_dfs["2ADC_DRUG"],
        x="treatment", y="occupancy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        ax=occupancy_2ADC, **boxplot_STYLE,
    )
    sns.stripplot(
        data=treatment_occupancy_dfs["2ADC_DRUG"],
        x="treatment", y="occupancy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        jitter=0.15, size=2.5, alpha=0.6, ax=occupancy_2ADC,
    )
    occupancy_2ADC.set_title(task_labels["2ADC_DRUG"])
    occupancy_2ADC.set_xlabel("")
    occupancy_2ADC.set_ylabel("Occupancy")
    occupancy_2ADC.set_ylim(-0.03, 1.03)
    add_paired_treatment_comparison(occupancy_2ADC, treatment_occupancy_dfs["2ADC_DRUG"], y="occupancy")
    occupancy_2ADC.set_xticks([0, 1], treatment_order, rotation=45, ha="right")
    if not mount_figure:
        for _format in ("svg", "png"):
            occupancy_2ADC.figure.savefig((path_panels / _format / "2ADC_occupancy").with_suffix(f".{_format}"))
    occupancy_2ADC
    return (occupancy_2ADC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2AFC
    """)
    return


@app.cell
def _(
    add_paired_treatment_comparison,
    axd,
    boxplot_STYLE,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    sns,
    task_labels,
    treatment_occupancy_dfs,
    treatment_order,
    treatment_palette,
):
    if not mount_figure:
        plt.figure(figsize=fig_size(2, 1), constrained_layout=True)
    occupancy_2AFC = axd["occupancy_2AFC"] if mount_figure else plt.gca()
    occupancy_2AFC.clear()
    sns.boxplot(
        data=treatment_occupancy_dfs["2AFC_DRUG"],
        x="treatment", y="occupancy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        ax=occupancy_2AFC, **boxplot_STYLE,
    )
    sns.stripplot(
        data=treatment_occupancy_dfs["2AFC_DRUG"],
        x="treatment", y="occupancy", hue="treatment",
        order=treatment_order, hue_order=treatment_order,
        palette=treatment_palette, legend=False,
        jitter=0.15, size=2.5, alpha=0.6, ax=occupancy_2AFC,
    )
    occupancy_2AFC.set_title(task_labels["2AFC_DRUG"])
    occupancy_2AFC.set_xlabel("")
    occupancy_2AFC.set_ylabel("Occupancy")
    occupancy_2AFC.set_ylim(-0.03, 1.03)
    add_paired_treatment_comparison(occupancy_2AFC, treatment_occupancy_dfs["2AFC_DRUG"], y="occupancy")
    occupancy_2AFC.set_xticks([0, 1], treatment_order, rotation=45, ha="right")
    if not mount_figure:
        for _format in ("svg", "png"):
            occupancy_2AFC.figure.savefig((path_panels / _format / "2AFC_occupancy").with_suffix(f".{_format}"))
    occupancy_2AFC
    return (occupancy_2AFC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Mount Figure
    """)
    return


@app.cell
def _(
    MODEL_BY_TASK,
    accuracy_2ADC,
    accuracy_2AFC,
    dwell_time_2ADC,
    dwell_time_2AFC,
    emission_weights_2ADC,
    emission_weights_2AFC,
    engaged_accuracy_2ADC,
    engaged_accuracy_2AFC,
    fig,
    histogram_transitions_2ADC,
    histogram_transitions_2AFC,
    mount_figure,
    occupancy_2ADC,
    occupancy_2AFC,
    path_panels,
    transition_weights_2ADC,
    transition_weights_2AFC,
):
    if mount_figure:
        # Referencing each panel makes this export react to every plot change.
        for _axis in (
            dwell_time_2ADC,
            dwell_time_2AFC,
            emission_weights_2ADC,
            emission_weights_2AFC,
            transition_weights_2ADC,
            transition_weights_2AFC,
            accuracy_2ADC,
            accuracy_2AFC,
            engaged_accuracy_2ADC,
            engaged_accuracy_2AFC,
            occupancy_2ADC,
            occupancy_2AFC,
        ):
            _axis.set_title("")
        fig.align_ylabels()
        histogram_transitions_2ADC.set_title("2ADC")
        histogram_transitions_2AFC.set_title("2AFC")
        fig.savefig((path_panels / "figure4_models").with_suffix(".pdf"))
        for _format in ("svg", "png"):
            fig.savefig((path_panels / _format / f"figure4_{MODEL_BY_TASK["2ADC_DRUG"]}").with_suffix(f".{_format}"))
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
