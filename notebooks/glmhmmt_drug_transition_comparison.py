import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GLMHMM-T drug-transition comparison

    Load one GLMHMM-T fit whose transition design contains the task drug
    regressor, then re-evaluate the same fitted model with that transition
    covariate forced to 0 and 1.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import copy
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from matplotlib.lines import Line2D

    from glmhmmt.model import (
        ParamsInputDrivenTransitions,
        ParamsSoftmaxGLMHMM,
        ParamsSoftmaxGLMHMMEmissions,
        ParamsStandardHMMInitialState,
        SoftmaxGLMHMM,
    )
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        load_model_config,
        model_aliases_for_kind,
        select_subject_behavior_df,
    )
    from glmhmmt.postprocess import (
        build_emission_weights_df,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_session_trajectories_payload,
        build_transition_matrix_payload,
        build_transition_weights_df,
    )
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.plots.common import custom_boxplot, significance_label
    import glmhmmt.plots as model_plots

    from plot_saver import make_plot_saver
    from scipy.stats import ttest_rel
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import add_choice_lag_summary_regressor, attach_signed_delay_columns
    from src.plots.common import plot_mean_over_data
    from src.utils import fig_size

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    paths = get_runtime_paths()
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    sns.set_style("ticks")
    sns.set_context("notebook")

    TASK_OPTIONS = ["2AFC_DRUG", "2ADC_DRUG"]
    DRUG_COL_BY_TASK = {"2AFC_DRUG": "Drug", "2ADC_DRUG": "drug_code"}
    CONDITION_LABELS = {0: "saline", 1: "drug"}
    CONDITION_PALETTE = {
        "saline": "tab:gray",
        "rest": "tab:gray",
        "drug": "tab:pink",
        "nan": "#666666",
        "all": "#333333",
    }
    return (
        CONDITION_LABELS,
        CONDITION_PALETTE,
        DRUG_COL_BY_TASK,
        Line2D,
        ParamsInputDrivenTransitions,
        ParamsSoftmaxGLMHMM,
        ParamsSoftmaxGLMHMMEmissions,
        ParamsStandardHMMInitialState,
        SoftmaxGLMHMM,
        TASK_OPTIONS,
        add_choice_lag_summary_regressor,
        attach_signed_delay_columns,
        build_transition_matrix_payload,
        build_transition_weights_df,
        build_trial_and_weights_df,
        build_views,
        copy,
        custom_boxplot,
        fig_size,
        get_adapter,
        json,
        load_fit_arrays,
        load_model_config,
        make_plot_saver,
        mo,
        model_aliases_for_kind,
        model_plots,
        np,
        paths,
        pd,
        pl,
        plot_mean_over_data,
        plt,
        process_two_adc,
        process_two_afc,
        select_subject_behavior_df,
        significance_label,
        sns,
        ttest_rel,
    )


@app.cell
def _(TASK_OPTIONS, mo):
    ui_task = mo.ui.dropdown(options=TASK_OPTIONS, value="2AFC_DRUG", label="Task")
    return (ui_task,)


@app.cell
def _(DRUG_COL_BY_TASK, get_adapter, pl, ui_task):
    task_name = ui_task.value
    drug_col = DRUG_COL_BY_TASK[task_name]
    adapter = get_adapter(task_name)
    df_all = adapter.subject_filter(adapter.read_dataset())
    plots = adapter.get_plots()
    condition_counts = (
        df_all.group_by("condition")
        .agg(pl.len().alias("n_trials"))
        .sort("condition", nulls_last=True)
    )
    return adapter, condition_counts, df_all, drug_col, plots, task_name


@app.cell
def _(df_all):
    df_all
    return


@app.cell
def _(drug_col, json, model_aliases_for_kind, paths, task_name):
    fit_root = paths.RESULTS / "fits" / task_name / "glmhmmt"
    all_aliases = model_aliases_for_kind(
        task_name=task_name,
        model_kind="glmhmmt",
        local_root=fit_root,
    )

    def _config_for(alias: str) -> dict:
        config_path = fit_root / alias / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path) as handle:
            return json.load(handle)

    alias_configs = {alias: _config_for(alias) for alias in all_aliases}
    drug_aliases = [
        alias
        for alias, cfg in alias_configs.items()
        if drug_col in [str(col) for col in cfg.get("transition_cols", [])]
    ]
    fit_aliases = drug_aliases or all_aliases
    return alias_configs, fit_aliases, fit_root


@app.cell
def _(alias_configs, fit_aliases, mo, task_name):
    def _preferred_alias() -> str:
        preferred = {
            "2AFC_DRUG": ["param drug", "mohammadi drug 2states", "drug trial_index reward prev"],
            "2ADC_DRUG": ["base_param", "base_param pure"],
        }.get(task_name, [])
        for alias in preferred:
            if alias in fit_aliases:
                return alias
        return fit_aliases[0] if fit_aliases else ""

    default_alias = _preferred_alias()
    default_cfg = alias_configs.get(default_alias, {})
    k_options = [int(k) for k in default_cfg.get("K_list", [2])]
    ui_alias = mo.ui.dropdown(
        options=fit_aliases or [""],
        value=default_alias,
        label="GLMHMM-T fit",
    )
    ui_k = mo.ui.dropdown(
        options=k_options or [2],
        value=(k_options or [2])[0],
        label="K",
    )
    return ui_alias, ui_k


@app.cell
def _(condition_counts, fit_aliases, mo, ui_alias, ui_k, ui_task):
    mo.vstack(
        [
            mo.hstack([ui_task, ui_alias, ui_k]),
            mo.md(f"Saved GLMHMM-T aliases with drug transition column: `{len(fit_aliases)}`"),
            condition_counts,
        ]
    )
    return


@app.cell
def _(
    adapter,
    df_all,
    fit_root,
    load_fit_arrays,
    load_model_config,
    mo,
    task_name,
    ui_alias,
    ui_k,
):
    mo.stop(not ui_alias.value, mo.md("No saved GLMHMM-T fits found."))
    selected_alias = ui_alias.value
    K = int(ui_k.value)
    cfg = load_model_config(
        task_name=task_name,
        model_kind="glmhmmt",
        alias=selected_alias,
        local_root=fit_root,
    )
    subjects = [str(subject) for subject in cfg.get("subjects", [])]
    emission_cols = list(cfg.get("emission_cols", [])) or None
    transition_cols = list(cfg.get("transition_cols", []))
    arrays_store, _names = load_fit_arrays(
        out_dir=fit_root / selected_alias,
        arrays_suffix="glmhmmt_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=subjects,
        emission_cols=emission_cols,
        transition_cols=transition_cols,
        k=K,
    )
    selected = [subject for subject in subjects if subject in arrays_store]
    mo.stop(not selected, mo.md("No fitted array files found for the selected alias/K."))
    return K, arrays_store, selected, selected_alias


@app.cell
def _(
    ParamsInputDrivenTransitions,
    ParamsSoftmaxGLMHMM,
    ParamsSoftmaxGLMHMMEmissions,
    ParamsStandardHMMInitialState,
    SoftmaxGLMHMM,
    adapter,
    copy,
    df_all,
    drug_col,
    np,
    select_subject_behavior_df,
):
    def _conditioned_u(arr: dict, drug_value: int):
        u = np.asarray(arr["U"], dtype=float).copy()
        u_cols = [str(col) for col in arr.get("U_cols", [])]
        if drug_col in u_cols:
            u[:, u_cols.index(drug_col)] = float(drug_value)
        for idx, col in enumerate(u_cols):
            if not col.startswith("drug_x_"):
                continue
            source_col = col.removeprefix("drug_x_")
            if source_col in u_cols:
                u[:, idx] = float(drug_value) * u[:, u_cols.index(source_col)]
        return u

    def _params_from_arrays(arr: dict, model):
        weights = np.asarray(arr["transition_weights"], dtype=np.float32)
        weights = model.transition_component._coerce_weights(weights)
        return ParamsSoftmaxGLMHMM(
            initial=ParamsStandardHMMInitialState(
                probs=np.asarray(arr["initial_probs"], dtype=np.float32)
            ),
            transitions=ParamsInputDrivenTransitions(
                bias=np.asarray(arr["transition_bias"], dtype=np.float32),
                weights=weights,
            ),
            emissions=ParamsSoftmaxGLMHMMEmissions(
                weights=np.asarray(arr["emission_weights"], dtype=np.float32)
            ),
        )

    def condition_arrays_store(arrays_store: dict, subjects: list[str], drug_value: int):
        conditioned = {}
        skipped = []
        for subject in subjects:
            arr = arrays_store.get(subject)
            if arr is None or "transition_weights" not in arr or "U" not in arr:
                skipped.append(subject)
                continue

            y = np.asarray(arr["y"])
            x = np.asarray(arr["X"], dtype=np.float32)
            df_sub = select_subject_behavior_df(
                df_all,
                subject=subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
                min_session_length=2,
            )
            if df_sub.height != y.shape[0]:
                skipped.append(subject)
                continue
            session_ids = df_sub[adapter.session_col].to_numpy()
            u = _conditioned_u(arr, drug_value).astype(np.float32)

            model = SoftmaxGLMHMM(
                num_states=int(arr.get("K", np.asarray(arr["smoothed_probs"]).shape[1])),
                num_classes=int(arr.get("num_classes", adapter.num_classes)),
                emission_input_dim=x.shape[1],
                transition_input_dim=u.shape[1],
                emission_feature_names=list(arr.get("X_cols", [])),
                baseline_class_idx=int(np.asarray(arr.get("baseline_class_idx", 0)).reshape(())),
            )
            params = _params_from_arrays(arr, model)
            inputs = np.concatenate([x, u], axis=1).astype(np.float32)
            new_arr = copy.copy(arr)
            new_arr["X"] = x
            new_arr["U"] = u
            new_arr["y"] = y
            new_arr["smoothed_probs"] = np.asarray(
                model.smoother_multisession(params, y, inputs, session_ids=session_ids)
            )
            new_arr["p_pred"] = np.asarray(
                model.predict_choice_probs_multisession(params, y, inputs, session_ids=session_ids)
            )
            new_arr["predictive_state_probs"] = np.asarray(
                model.predict_state_probs_multisession(params, y, inputs, session_ids=session_ids)
            )
            transition_mats = np.asarray(
                model.transition_component._compute_transition_matrices(params.transitions, inputs)
            )
            new_arr["transition_matrix"] = (
                transition_mats.mean(axis=0)
                if transition_mats.ndim == 3 and transition_mats.shape[0] > 0
                else transition_mats
            )
            conditioned[subject] = new_arr
        return conditioned, skipped

    return (condition_arrays_store,)


@app.cell
def _(
    CONDITION_LABELS,
    K,
    adapter,
    arrays_store,
    build_trial_and_weights_df,
    build_views,
    condition_arrays_store,
    df_all,
    drug_col,
    mo,
    pl,
    selected,
):
    def _actual_condition_trials(trial_df, drug_value: int):
        if trial_df.is_empty():
            return trial_df
        if drug_col in trial_df.columns:
            return trial_df.filter(
                pl.col(drug_col).cast(pl.Float64, strict=False) == float(drug_value)
            )
        if "condition" in trial_df.columns:
            target = "drug" if int(drug_value) == 1 else "saline"
            return trial_df.filter(
                pl.col("condition").cast(pl.Utf8).str.to_lowercase() == target
            )
        return trial_df

    conditioned_arrays = {}
    conditioned_views = {}
    conditioned_trial_dfs = {}
    conditioned_weight_dfs = {}
    skipped_by_condition = {}
    trial_count_rows = []
    for _drug_value in [0, 1]:
        _label = CONDITION_LABELS[_drug_value]
        conditioned_arrays[_drug_value], skipped_by_condition[_drug_value] = condition_arrays_store(
            arrays_store,
            selected,
            _drug_value,
        )
        conditioned_views[_drug_value] = build_views(
            conditioned_arrays[_drug_value],
            adapter,
            K,
            selected,
        )
        trial_df, weights_df = build_trial_and_weights_df(
            df_all,
            views=conditioned_views[_drug_value],
            adapter=adapter,
            min_session_length=2,
        )
        trial_df = _actual_condition_trials(trial_df, _drug_value)
        conditioned_trial_dfs[_drug_value] = trial_df.with_columns(
            [
                pl.lit(_label).alias("fit_condition"),
                pl.lit(_drug_value).alias("transition_drug_value"),
            ]
        )
        trial_count_rows.append(
            {
                "transition_drug_value": int(_drug_value),
                "fit_condition": _label,
                "n_trials": int(trial_df.height),
                "n_subjects": (
                    int(trial_df.select(pl.col("subject").n_unique()).item())
                    if "subject" in trial_df.columns and trial_df.height
                    else 0
                ),
            }
        )
        conditioned_weight_dfs[_drug_value] = weights_df.with_columns(
            [
                pl.lit(_label).alias("fit_condition"),
                pl.lit(_drug_value).alias("transition_drug_value"),
            ]
        )
    condition_trial_counts = pl.DataFrame(trial_count_rows)
    mo.stop(
        any(conditioned_trial_dfs[value].is_empty() for value in [0, 1]),
        mo.vstack(
            [
                mo.md("One of the actual drug-condition trial sets is empty."),
                condition_trial_counts,
            ]
        ),
    )
    common_conditioned_subjects = sorted(
        set(conditioned_views[0].keys()) & set(conditioned_views[1].keys())
    )
    mo.stop(
        not common_conditioned_subjects,
        mo.md("No subjects could be re-evaluated for both drug transition values."),
    )
    skip_text = "; ".join(
        f"{CONDITION_LABELS[value]} skipped {len(skipped)}"
        for value, skipped in skipped_by_condition.items()
        if skipped
    )
    mo.md(
        f"Conditioned subjects: `{len(common_conditioned_subjects)}`"
        + (f" ({skip_text})" if skip_text else "")
    )
    condition_trial_counts
    return (
        conditioned_arrays,
        conditioned_trial_dfs,
        conditioned_views,
        conditioned_weight_dfs,
    )


@app.cell
def _(
    adapter,
    add_choice_lag_summary_regressor,
    conditioned_trial_dfs,
    conditioned_views,
    pl,
    process_two_adc,
    process_two_afc,
    task_name,
):
    def _choice_lag_cols_for(views: dict, trial_df):
        cols = []
        for view in views.values():
            for feat in list(getattr(view, "feat_names", []) or []):
                feat = str(feat)
                if feat.startswith("choice_lag_") and feat not in cols:
                    cols.append(feat)
        if not cols:
            cols = adapter.choice_lag_cols(trial_df)
        return cols

    def _prepare_plot_df(trial_df, views):
        processor = process_two_adc if task_name == "2ADC_DRUG" else process_two_afc
        plot_df = processor.prepare_predictions_df(trial_df)
        plot_df = add_choice_lag_summary_regressor(
            plot_df,
            choice_lag_cols=_choice_lag_cols_for(views, trial_df),
        )
        if "Choice" not in plot_df.columns and "response" in plot_df.columns:
            plot_df = plot_df.with_columns(pl.col("response").alias("Choice"))
        return plot_df

    plot_dfs = {
        drug_value: _prepare_plot_df(conditioned_trial_dfs[drug_value], conditioned_views[drug_value])
        for drug_value in [0, 1]
    }
    plot_dfs
    return (plot_dfs,)


@app.cell
def _():
    return


@app.cell
def _(mo, plot_dfs, plots):
    columns = set(plot_dfs[0].columns) & set(plot_dfs[1].columns)
    preferred = [
        "choice_lag_one_hot_sum",
        "at_choice",
        "at_choice_param",
        "stim_vals",
        "stim_param",
        "stim_x_delay_param",
        "prev_choice",
        "wsls",
    ]
    regressor_options = [col for col in preferred if col in columns]
    if not regressor_options:
        regressor_options = sorted(
            col
            for col in columns
            if col not in {"subject", "fit_alias", "fit_condition", "Session", "Filename"}
        )[:10]
    choice_history_regressor = (
        "choice_lag_one_hot_sum"
        if "choice_lag_one_hot_sum" in regressor_options
        else plots.pick_choice_history_regressor(regressor_options)
    )
    mo.stop(choice_history_regressor is None, mo.md("No choice-history regressor available."))
    return (choice_history_regressor,)


@app.cell
def _(
    CONDITION_LABELS,
    CONDITION_PALETTE,
    Line2D,
    adapter,
    choice_history_regressor,
    conditioned_views,
    plot_dfs,
    plots,
    plt,
    task_name,
):
    def _artist_snapshot(ax):
        return {
            "lines": set(ax.lines),
            "collections": set(ax.collections),
            "patches": set(ax.patches),
        }

    def _style_added_artists(ax, before: dict, *, color: str):
        for line in [artist for artist in ax.lines if artist not in before["lines"]]:
            line.set_color(color)
            line.set_markerfacecolor(color)
            line.set_markeredgecolor(color)
        for collection in [artist for artist in ax.collections if artist not in before["collections"]]:
            try:
                collection.set_edgecolor(color)
                collection.set_facecolor(color)
            except Exception:
                pass
        for patch in [artist for artist in ax.patches if artist not in before["patches"]]:
            patch.set_edgecolor(color)

    def _overlay(call, ax, *, color: str):
        before = _artist_snapshot(ax)
        result = call(ax)
        if result is not None:
            _style_added_artists(ax, before, color=color)
        return result

    fig_overlay, axd = plt.subplot_mosaic(
        [
            ["accuracy", "repeat_evidence"],
            ["binned_accuracy", "right_regressor"],
            ["repeat_bias", "repeat_bias"],
        ],
        figsize=(7.0, 8.3),
        layout="constrained",
    )
    _is_delay_task = task_name == "2ADC_DRUG"
    accuracy_x_axis = "raw_delay" if _is_delay_task else "ILD"
    for drug_value in [0, 1]:
        label = CONDITION_LABELS[drug_value]
        _color = CONDITION_PALETTE[label]
        plot_df = plot_dfs[drug_value]
        views = conditioned_views[drug_value]
        _overlay(
            lambda ax, df=plot_df, vs=views: plots.plot_accuracy_by_total_evidence(
                df,
                adapter=adapter,
                views=vs,
                ax=ax,
                legend=False,
            ),
            axd["accuracy"],
            color=_color,
        )
        _overlay(
            lambda ax, df=plot_df, vs=views: plots.plot_repeat_by_repeat_evidence(
                df,
                views=vs,
                ax=ax,
                legend=False,
            ),
            axd["repeat_evidence"],
            color=_color,
        )
        _overlay(
            lambda ax, df=plot_df, vs=views: plots.plot_binned_accuracy_figure(
                df,
                regressor_col=choice_history_regressor,
                x_axis=accuracy_x_axis,
                adapter=adapter,
                views=vs,
                axes=[ax],
                max_panels=1,
                legend=False,
            ),
            axd["binned_accuracy"],
            color=_color,
        )
        _overlay(
            lambda ax, df=plot_df: plots.plot_right_by_regressor(
                df,
                regressor_col=choice_history_regressor,
                ax=ax,
                legend=False,
            ),
            axd["right_regressor"],
            color=_color,
        )
        _overlay(
            lambda ax, df=plot_df: plots.plot_rb(df, ax=ax, title=None),
            axd["repeat_bias"],
            color=_color,
        )

    axd["accuracy"].set_title("Accuracy by fitted evidence")
    axd["repeat_evidence"].set_title("Repeat by fitted evidence")
    axd["binned_accuracy"].set_title("Psychometric by choice lag")
    axd["right_regressor"].set_title("Right choice by choice lag")
    axd["repeat_bias"].set_title("Repeat bias by delay" if _is_delay_task else "Repeat bias by stimulus strength")
    for axis in axd.values():
        axis.set_box_aspect(1)
    fig_overlay.legend(
        handles=[
            Line2D([0], [0], color=CONDITION_PALETTE[CONDITION_LABELS[value]], lw=2, marker="o", label=CONDITION_LABELS[value])
            for value in [0, 1]
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    return (fig_overlay,)


@app.cell
def _(
    CONDITION_LABELS,
    CONDITION_PALETTE,
    Line2D,
    attach_signed_delay_columns,
    fig_overlay,
    fig_size,
    make_plot_saver,
    mo,
    paths,
    pd,
    plot_dfs,
    plot_mean_over_data,
    plt,
    selected_alias,
    sns,
    task_name,
):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=task_name,
        model_id=f"glmhmmt_drug_transition/{selected_alias}",
    )
    signed_delay_order = ["0L", "-1", "-3", "-10", "10", "3", "1", "0R"]
    signed_delay_tick_labels = ["-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"]

    def to_pandas(frame):
        return frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)

    def add_p_right(pdf):
        pdf = pdf.copy()
        choice_col = next(
            (col for col in ["Choice", "choice", "choices", "response"] if col in pdf.columns),
            None,
        )
        if choice_col is None:
            return pdf
        choice = pd.to_numeric(pdf[choice_col], errors="coerce")
        finite_choice = choice.dropna()
        if not finite_choice.empty and set(finite_choice.unique()).issubset({0, 1}):
            pdf["p_right"] = choice.astype(float)
        else:
            pdf["p_right"] = (choice > 0).astype(float)
            pdf.loc[choice.isna(), "p_right"] = pd.NA
        return pdf

    def prepare_curve_df(frame, drug_value):
        curve_df = add_p_right(to_pandas(frame))
        curve_df["fit_condition"] = CONDITION_LABELS[drug_value]
        if "raw_delay" in curve_df.columns and not any(col in curve_df.columns for col in ["delay_raw", "delays", "delay"]):
            curve_df["delay_raw"] = curve_df["raw_delay"]
        if "stim" not in curve_df.columns:
            for candidate in ["stim_vals", "stim_param", "stimulus"]:
                if candidate in curve_df.columns:
                    curve_df["stim"] = pd.to_numeric(curve_df[candidate], errors="coerce")
                    break
        return curve_df

    curve_df = pd.concat(
        [prepare_curve_df(plot_dfs[value], value) for value in [0, 1]],
        ignore_index=True,
    )
    curve_df = curve_df.dropna(subset=["p_right"]) if "p_right" in curve_df.columns else pd.DataFrame()
    _is_delay_task = task_name == "2ADC_DRUG"
    fig_psychometric_comparison, ax_psychometric = plt.subplots(figsize=fig_size(2, 1))
    psychometric_available = False
    if not curve_df.empty and _is_delay_task:
        signed_df = attach_signed_delay_columns(curve_df)
        signed_df["signed_delay_plot"] = signed_df["_signed_delay_cat"].astype(str)
        signed_df = signed_df[signed_df["signed_delay_plot"].isin(signed_delay_order)].copy()
        for value in [0, 1]:
            _condition = CONDITION_LABELS[value]
            _condition_df = signed_df[signed_df["fit_condition"] == _condition]
            if _condition_df.empty:
                continue
            plot_mean_over_data(
                _condition_df,
                x_col="signed_delay_plot",
                x_order=signed_delay_order,
                x_tick_labels=signed_delay_tick_labels,
                y_col="p_right",
                xlabel="Signed delay (s)",
                ylabel=r"$p(\mathrm{right})$",
                title="",
                baseline=0.5,
                baseline_area=False,
                color=CONDITION_PALETTE[_condition],
                figsize=fig_size(2, 1),
                ax=ax_psychometric,
            )
            psychometric_available = True
    if not psychometric_available and not curve_df.empty and "ILD" in curve_df.columns:
        for value in [0, 1]:
            _condition = CONDITION_LABELS[value]
            _condition_df = curve_df[curve_df["fit_condition"] == _condition]
            if _condition_df.empty:
                continue
            plot_mean_over_data(
                _condition_df,
                x_col="ILD",
                y_col="p_right",
                xlabel="ILD (dB)",
                ylabel=r"$p(\mathrm{right})$",
                title="",
                baseline=0.5,
                baseline_area=False,
                color=CONDITION_PALETTE[_condition],
                figsize=fig_size(2, 1),
                ax=ax_psychometric,
            )
            psychometric_available = True
        ax_psychometric.set_xticks(
            [-20, -8, -4, -2, 0, 2, 4, 8, 20],
            labels=["-20", "-8", "", "", "0", "", "", "8", "20"],
        )
    if psychometric_available:
        ax_psychometric.legend(
            handles=[
                Line2D([0], [0], color=CONDITION_PALETTE[CONDITION_LABELS[value]], lw=2, marker="o", label=CONDITION_LABELS[value])
                for value in [0, 1]
            ],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.6),
            ncol=2,
        )
        ax_psychometric.set_title("")
        ax_psychometric.figure.subplots_adjust(bottom=0.34)
        sns.despine(ax=ax_psychometric)
    else:
        ax_psychometric.text(0.5, 0.5, "No psychometric curve data", ha="center", va="center")
        ax_psychometric.axis("off")
    mo.vstack(
        [
            fig_overlay,
            save_plot(fig_overlay, "drug transition overlay", stem="drug_transition_overlay"),
            fig_psychometric_comparison,
            save_plot(fig_psychometric_comparison, "overall psychometric by transition drug", stem="overall_psychometric_transition_drug"),
        ],
        align="center",
    )
    return (
        add_p_right,
        save_plot,
        signed_delay_order,
        signed_delay_tick_labels,
        to_pandas,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transition weights and conditioned matrices
    """)
    return


@app.cell
def _(
    K,
    build_transition_matrix_payload,
    build_transition_weights_df,
    conditioned_arrays,
    conditioned_views,
    fig_size,
    mo,
    model_plots,
    plt,
    save_plot,
):
    transition_weights_df = build_transition_weights_df(conditioned_views[0])
    mo.stop(transition_weights_df.is_empty(), mo.md("No transition weights found in the selected fit."))
    fig_transition_weights, ax_transition_weights = plt.subplots(figsize=fig_size(2, 1))
    model_plots.transition_weights_summary_boxplot(
        transition_weights_df,
        connect_subjects=True,
        show_ttests=True,
        ax=ax_transition_weights,
        tick_rotation=0,
    )
    ax_transition_weights.set_title("Transition weights")

    matrix_columns = []
    for _drug_value in [0, 1]:
        payload = build_transition_matrix_payload(
            arrays_store=conditioned_arrays[_drug_value],
            state_labels={
                subject: view.state_name_by_idx
                for subject, view in conditioned_views[_drug_value].items()
            },
            K=K,
            subjects=list(conditioned_views[_drug_value].keys()),
        )
        fig_matrix, ax_matrix = plt.subplots(figsize=fig_size(2, 1))
        model_plots.transition_matrix(**payload, ax=ax_matrix)
        ax_matrix.set_title(f"Transition matrix, drug={_drug_value}")
        matrix_columns.append(
            mo.vstack(
                [
                    fig_matrix,
                    save_plot(
                        fig_matrix,
                        f"transition matrix drug {_drug_value}",
                        stem=f"transition_matrix_drug_{_drug_value}",
                    ),
                ],
                align="center",
            )
        )
    mo.vstack(
        [
            fig_transition_weights,
            save_plot(fig_transition_weights, "transition weights", stem="transition_weights"),
            mo.hstack(matrix_columns, align="center"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State comparison
    """)
    return


@app.cell
def _(
    CONDITION_LABELS,
    CONDITION_PALETTE,
    Line2D,
    add_p_right,
    attach_signed_delay_columns,
    conditioned_trial_dfs,
    conditioned_weight_dfs,
    custom_boxplot,
    fig_size,
    np,
    pd,
    plots,
    plt,
    signed_delay_order,
    signed_delay_tick_labels,
    significance_label,
    sns,
    task_name,
    to_pandas,
    ttest_rel,
):
    state_labels = {0: "Engaged", 1: "Disengaged"}
    state_ranks = [0, 1]

    def state_name(rank):
        return state_labels.get(int(rank), f"State {int(rank)}")

    def build_state_rank_map(weights_df, rank_feature):
        weights = to_pandas(weights_df).copy()
        weights["subject"] = weights["subject"].astype(str)
        weights["fit_condition"] = weights["fit_condition"].astype(str)
        weights["state_idx"] = pd.to_numeric(weights["state_idx"], errors="coerce")
        weights["state_rank"] = pd.to_numeric(weights["state_rank"], errors="coerce")
        rank_columns = ["subject", "fit_condition", "state_idx", "state_rank", "state_label"]
        ranked = (
            weights[weights["feature"].astype(str) == str(rank_feature)]
            .groupby(rank_columns, as_index=False, observed=True)["weight"]
            .mean()
            .sort_values(["subject", "fit_condition", "weight"], ascending=[True, True, False])
        )
        ranked["plot_state_rank"] = ranked.groupby(["subject", "fit_condition"], observed=True).cumcount()
        ranked = ranked.dropna(subset=["state_idx", "plot_state_rank"]).copy()
        ranked["state_idx"] = ranked["state_idx"].astype(int)
        ranked["plot_state_rank"] = ranked["plot_state_rank"].astype(int)
        ranked["plot_state_label"] = ranked["plot_state_rank"].map(state_name)
        return ranked[["subject", "fit_condition", "state_idx", "plot_state_rank", "plot_state_label"]]

    def add_ranked_state_columns(trial_df, rank_map):
        trials = to_pandas(trial_df).copy()
        trials["subject"] = trials["subject"].astype(str)
        trials["fit_condition"] = trials["fit_condition"].astype(str)
        trials["state_idx"] = pd.to_numeric(trials["state_idx"], errors="coerce").astype(int)
        ranked = trials.merge(rank_map, on=["subject", "fit_condition", "state_idx"], how="inner")
        ranked["state_rank"] = ranked["plot_state_rank"].astype(int)
        ranked["state_group"] = ranked["state_rank"].map(state_name)
        engaged_states = (
            rank_map[rank_map["plot_state_rank"] == 0][["subject", "fit_condition", "state_idx"]]
            .rename(columns={"state_idx": "engaged_state_idx"})
        )
        ranked = ranked.merge(engaged_states, on=["subject", "fit_condition"], how="left")
        ranked["p_engaged"] = np.nan
        for engaged_state_idx, row_index in ranked.groupby("engaged_state_idx", observed=True).groups.items():
            state_column = f"p_state_{int(engaged_state_idx)}"
            ranked.loc[row_index, "p_engaged"] = pd.to_numeric(ranked.loc[row_index, state_column], errors="coerce")
        return ranked

    comparison_weights_df = pd.concat(
        [to_pandas(conditioned_weight_dfs[value]) for value in [0, 1]],
        ignore_index=True,
    )
    available_features = list(dict.fromkeys(comparison_weights_df["feature"].astype(str)))
    preferred_rank_features = [
        "stim_param",
        "stim_x_delay_param",
        "stim_vals",
        "at_choice_param",
        "choice_lag_param",
    ]
    rank_feature = next((feature for feature in preferred_rank_features if feature in available_features), available_features[0])
    rank_map = build_state_rank_map(comparison_weights_df, rank_feature)
    state_comparison_df = pd.concat(
        [
            add_ranked_state_columns(conditioned_trial_dfs[value], rank_map)
            for value in [0, 1]
        ],
        ignore_index=True,
    )
    state_comparison_df = state_comparison_df[state_comparison_df["state_rank"].isin(state_ranks)].copy()
    conditions = [CONDITION_LABELS[0], CONDITION_LABELS[1]]

    def condition_handles():
        return [
            Line2D([0], [0], color=CONDITION_PALETTE.get(condition, "#333333"), lw=3, label=condition)
            for condition in conditions
        ]

    def state_panel_figure():
        fig = plt.figure(figsize=fig_size(2, 1), constrained_layout=False)
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[1, 0.34],
            left=0.22,
            right=0.96,
            top=0.95,
            bottom=0.02,
            hspace=0.18,
        )
        ax = fig.add_subplot(gs[0])
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis("off")
        return fig, ax, legend_ax

    def format_state_summary_panel(fig, ax, legend_ax):
        if ax.legend_ is not None:
            ax.legend_.remove()
        fig.legends.clear()
        legend_ax.legend(handles=condition_handles(), frameon=False, loc="lower center", ncol=2)
        sns.despine(ax=ax)
        return fig

    def paired_state_boxplot(summary_df, *, value_col, ylabel, chance=None, ylim=None, plot_state_ranks=None):
        ranks = list(state_ranks if plot_state_ranks is None else plot_state_ranks)
        fig, ax, legend_ax = state_panel_figure()
        width = 0.30
        offsets = np.linspace(-width / 2, width / 2, len(conditions))
        values = []
        positions = []
        median_colors = []
        position_by_pair = {}
        for rank_idx, rank in enumerate(ranks):
            for condition_idx, condition in enumerate(conditions):
                position = rank_idx + offsets[condition_idx]
                vals = summary_df[
                    (summary_df["state_rank"] == rank)
                    & (summary_df["fit_condition"] == condition)
                ][value_col].dropna().to_numpy(dtype=float)
                values.append(vals)
                positions.append(position)
                median_colors.append(CONDITION_PALETTE.get(condition, "#333333"))
                position_by_pair[(rank, condition)] = position
        custom_boxplot(
            ax,
            values,
            positions=positions,
            widths=width * 0.85,
            median_colors=median_colors,
            showfliers=False,
            showcaps=False,
        )
        for rank in ranks:
            pivot = (
                summary_df[summary_df["state_rank"] == rank]
                .pivot(index="subject", columns="fit_condition", values=value_col)
            )
            if not set(conditions).issubset(pivot.columns):
                continue
            paired = pivot[conditions].dropna()
            for row in paired.itertuples(index=False):
                ax.plot(
                    [position_by_pair[(rank, conditions[0])], position_by_pair[(rank, conditions[1])]],
                    [row[0], row[1]],
                    color="#A7A7A7",
                    linewidth=0.8,
                    alpha=0.28,
                    zorder=1,
                )
            if len(paired) >= 2:
                rank_values = summary_df[summary_df["state_rank"] == rank][value_col].dropna().to_numpy(dtype=float)
                value_range = float(np.nanmax(rank_values) - np.nanmin(rank_values))
                value_range = value_range if np.isfinite(value_range) and value_range > 0 else 1.0
                x1 = position_by_pair[(rank, conditions[0])]
                x2 = position_by_pair[(rank, conditions[1])]
                y = float(np.nanmax(rank_values)) + 0.08 * value_range
                h = 0.025 * value_range
                pvalue = float(ttest_rel(paired[conditions[0]], paired[conditions[1]], nan_policy="omit").pvalue)
                label = significance_label(pvalue) if np.isfinite(pvalue) else "n.s."
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.9)
                ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", color="black")
        if chance is not None:
            ax.axhline(chance, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([state_name(rank) for rank in ranks])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("State")
        if ylim is not None:
            ax.set_ylim(*ylim)
        return format_state_summary_panel(fig, ax, legend_ax)

    state_accuracy_summary = (
        state_comparison_df.dropna(subset=["correct_bool"])
        .groupby(["subject", "fit_condition", "state_rank"], as_index=False, observed=True)
        .agg(accuracy=("correct_bool", "mean"), n_trials=("correct_bool", "size"))
    )
    fig_state_accuracy = paired_state_boxplot(
        state_accuracy_summary,
        value_col="accuracy",
        ylabel="Accuracy",
        chance=0.5,
        ylim=(0, 1),
    )

    state_occupancy_summary = (
        state_comparison_df.groupby(["subject", "fit_condition"], as_index=False, observed=True)
        .agg(
            n_subject_trials=("state_rank", "size"),
            n_state_trials=("state_rank", lambda values: (values == 0).sum()),
        )
    )
    state_occupancy_summary["state_rank"] = 0
    state_occupancy_summary["occupancy"] = (
        state_occupancy_summary["n_state_trials"] / state_occupancy_summary["n_subject_trials"]
    )
    fig_state_occupancy = paired_state_boxplot(
        state_occupancy_summary,
        value_col="occupancy",
        ylabel="Engaged occupancy",
        ylim=(0, 1),
        plot_state_ranks=[0],
    )

    switch_records = []
    for (subject, condition, session), group in state_comparison_df.sort_values(
        ["subject", "fit_condition", "session", "trial_idx"]
    ).groupby(["subject", "fit_condition", "session"], observed=True):
        states = pd.to_numeric(group["state_rank"], errors="coerce").to_numpy(dtype=float)
        states = states[np.isfinite(states)]
        switch_records.append(
            {
                "subject": str(subject),
                "fit_condition": str(condition),
                "session": str(session),
                "n_switches": int(np.sum(states[1:] != states[:-1])),
            }
        )
    switch_df = pd.DataFrame(switch_records)
    fig_switches, ax_switches, legend_ax_switches = state_panel_figure()
    max_switches = float(switch_df["n_switches"].max())
    for condition in conditions:
        values = switch_df[switch_df["fit_condition"] == condition]["n_switches"].dropna().to_numpy(dtype=float)
        if values.size >= 2 and np.nanstd(values) > 0:
            sns.kdeplot(
                x=values,
                ax=ax_switches,
                color=CONDITION_PALETTE.get(condition, "#333333"),
                linewidth=2.0,
                bw_adjust=0.8,
                clip=(0, None),
                label=condition,
            )
        elif values.size == 1:
            ax_switches.axvline(values[0], color=CONDITION_PALETTE.get(condition, "#333333"), linewidth=2.0, label=condition)
    ax_switches.set_xlim(left=-0.25, right=max(1.0, max_switches + 0.25))
    ax_switches.set_xlabel("State changes per session")
    ax_switches.set_ylabel("Density")
    format_state_summary_panel(fig_switches, ax_switches, legend_ax_switches)

    def add_model_p_right(data):
        prepared = data.copy()
        model_col = next(
            (col for col in ["p_model_right", "p_pred", "pR"] if col in prepared.columns),
            None,
        )
        if model_col is not None:
            prepared["p_model_right"] = pd.to_numeric(prepared[model_col], errors="coerce")
        return prepared

    def plot_state_psychometric_curve(
        ax,
        data,
        *,
        x_col,
        y_col,
        x_order=None,
        x_tick_labels=None,
        color,
        linestyle="-",
        marker="o",
        linewidth=1.8,
        alpha=1.0,
    ):
        source = data.copy()
        if y_col not in source.columns:
            return False
        source["_y"] = pd.to_numeric(source[y_col], errors="coerce")
        source = source[source[x_col].notna() & source["_y"].notna()].copy()
        if source.empty:
            return False

        if "subject" in source.columns:
            subject_summary = (
                source.groupby(["subject", x_col], observed=True)["_y"]
                .mean()
                .reset_index(name="subject_mean")
            )
            summary = (
                subject_summary.groupby(x_col, observed=True)["subject_mean"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
            )
        else:
            summary = (
                source.groupby(x_col, observed=True)["_y"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
            )

        if x_order is not None:
            summary = summary[summary[x_col].isin(x_order)].copy()
            if summary.empty:
                return False
            summary[x_col] = pd.Categorical(summary[x_col], categories=x_order, ordered=True)
            summary = summary.sort_values(x_col)
            x = np.arange(len(summary), dtype=float)
            label_map = dict(zip(x_order, x_tick_labels or x_order, strict=False))
            tick_labels = [label_map.get(value, str(value)) for value in summary[x_col]]
        else:
            summary["_x_numeric"] = pd.to_numeric(summary[x_col], errors="coerce")
            summary = summary.dropna(subset=["_x_numeric"]).sort_values("_x_numeric")
            if summary.empty:
                return False
            x = summary["_x_numeric"].to_numpy(dtype=float)
            tick_labels = [f"{value:g}" for value in x]

        sem = summary["std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(summary["n"].clip(lower=1).to_numpy(dtype=float))
        fmt = marker if marker else linestyle
        ax.errorbar(
            x,
            summary["mean"].to_numpy(dtype=float),
            yerr=sem,
            fmt=fmt,
            color=color,
            ecolor=color,
            capsize=0,
            linewidth=linewidth,
            markersize=4 if marker else 0,
            alpha=alpha,
        )
        ax.set_xticks(x, labels=tick_labels)
        return True

    def format_psychometric_state_axis(ax, *, xlabel, title):
        ax.axhline(0.5, color="gray", ls="--")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])

    categorical_source = add_p_right(state_comparison_df)
    categorical_source = add_model_p_right(categorical_source)
    has_model = "p_model_right" in categorical_source.columns and categorical_source["p_model_right"].notna().any()
    is_delay_task = task_name == "2ADC_DRUG"
    fig_state_categorical, axes = plt.subplots(
        1,
        len(state_ranks),
        figsize=(fig_size(2, 1)[0] * len(state_ranks), fig_size(2, 1)[1]),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    if is_delay_task:
        categorical_source = categorical_source.copy()
        if "delay_raw" not in categorical_source.columns and "raw_delay" in categorical_source.columns:
            categorical_source["delay_raw"] = categorical_source["raw_delay"]
        plot_source = attach_signed_delay_columns(categorical_source)
        plot_source["signed_delay_plot"] = plot_source["_signed_delay_cat"].astype(str)
        plot_source = plot_source[plot_source["signed_delay_plot"].isin(signed_delay_order)].copy()
        for ax, rank in zip(axes, state_ranks, strict=False):
            for condition in conditions:
                condition_df = plot_source[
                    (plot_source["state_rank"] == rank)
                    & (plot_source["fit_condition"] == condition)
                ]
                plot_state_psychometric_curve(
                    ax,
                    condition_df,
                    x_col="signed_delay_plot",
                    x_order=signed_delay_order,
                    x_tick_labels=signed_delay_tick_labels,
                    y_col="p_right",
                    color=CONDITION_PALETTE.get(condition, "#333333"),
                    linestyle="",
                )
                plot_state_psychometric_curve(
                    ax,
                    condition_df,
                    x_col="signed_delay_plot",
                    x_order=signed_delay_order,
                    x_tick_labels=signed_delay_tick_labels,
                    y_col="p_model_right",
                    color=CONDITION_PALETTE.get(condition, "#333333"),
                    linestyle="-",
                    marker="",
                    linewidth=2.0,
                    alpha=1.0,
                )
            format_psychometric_state_axis(ax, xlabel="Signed delay (s)", title=state_name(rank))
    else:
        for ax, rank in zip(axes, state_ranks, strict=False):
            for condition in conditions:
                condition_df = categorical_source[
                    (categorical_source["state_rank"] == rank)
                    & (categorical_source["fit_condition"] == condition)
                ]
                plot_state_psychometric_curve(
                    ax,
                    condition_df,
                    x_col="ILD",
                    y_col="p_right",
                    color=CONDITION_PALETTE.get(condition, "#333333"),
                    linestyle="",
                )
                plot_state_psychometric_curve(
                    ax,
                    condition_df,
                    x_col="ILD",
                    y_col="p_model_right",
                    color=CONDITION_PALETTE.get(condition, "#333333"),
                    linestyle="-",
                    marker="",
                    linewidth=2.0,
                    alpha=1.0,
                )
            ax.set_xticks(
                [-20, -8, -4, -2, 0, 2, 4, 8, 20],
                labels=["-20", "-8", "", "", "0", "", "", "8", "20"],
            )
            format_psychometric_state_axis(ax, xlabel="ILD (dB)", title=state_name(rank))
    for axis_index, ax in enumerate(axes):
        ax.set_ylabel(r"$p(\mathrm{right})$" if axis_index == 0 else "")
        sns.despine(ax=ax)
    legend_handles = []
    for condition in conditions:
        color = CONDITION_PALETTE.get(condition, "#333333")
        legend_handles.append(Line2D([0], [0], color=color, lw=2, marker="o", label=f"{condition} data"))
        if has_model:
            legend_handles.append(Line2D([0], [0], color=color, lw=2.0, linestyle="-", label=f"{condition} model"))
    fig_state_categorical.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
    )
    fig_state_categorical.subplots_adjust(bottom=0.26, wspace=0.25)

    def adapter_by_state_psychometric_figure(condition_payloads, conditions):
        plot_fn = getattr(plots, "plot_categorical_performance_by_state", None)
        if plot_fn is None:
            raise AttributeError("Task plots do not expose plot_categorical_performance_by_state.")
        payloads = [
            (condition, condition_payloads[condition][0], condition_payloads[condition][1])
            for condition in conditions
            if condition in condition_payloads and condition_payloads[condition][1]
        ]
        if not payloads:
            raise ValueError("No condition payloads available for by-state psychometric plotting.")

        def _num_states(views):
            return int(next(iter(views.values())).K) if views else 1

        n_rows = len(payloads)
        n_cols = max(_num_states(views) for _, _, views in payloads)
        panel_w, panel_h = fig_size(2, 1)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(panel_w * n_cols, panel_h * n_rows),
            sharey=True,
            squeeze=False,
        )
        state_plot_kwargs = dict(
            background_style="none",
            show_weighted_points=True,
            show_data_smooth=False,
            show_model_smooth=True,
            model_line_mode="smooth",
            state_assignment_mode="weighted",
            figure_dpi=300,
        )
        for row_idx, (condition, condition_df, condition_views) in enumerate(payloads):
            k = _num_states(condition_views)
            for extra_ax in axes[row_idx, k:]:
                extra_ax.axis("off")
            plot_fn(
                df=condition_df,
                views=condition_views,
                model_name=str(condition),
                axes=list(axes[row_idx, :k]),
                **state_plot_kwargs,
            )
            ylabel = axes[row_idx, 0].get_ylabel() or r"$p(\mathrm{right})$"
            axes[row_idx, 0].set_ylabel(f"{condition}\n{ylabel}")
        fig.tight_layout()
        return fig
    fig_state_categorical
    return (
        adapter_by_state_psychometric_figure,
        fig_state_accuracy,
        fig_state_occupancy,
        fig_switches,
        rank_feature,
    )


@app.cell
def _(
    CONDITION_LABELS,
    adapter_by_state_psychometric_figure,
    conditioned_views,
    fig_state_accuracy,
    fig_state_occupancy,
    fig_switches,
    mo,
    plot_dfs,
    rank_feature,
    save_plot,
):
    fig_state_categorical2 = adapter_by_state_psychometric_figure(
        {
            CONDITION_LABELS[value]: (plot_dfs[value], conditioned_views[value])
            for value in [0, 1]
        },
        [CONDITION_LABELS[0], CONDITION_LABELS[1]],
    )
    mo.vstack(
        [
            mo.md(f"State ranking coefficient: `{rank_feature}`"),
            mo.hstack(
                [
                    mo.vstack([fig_state_accuracy, save_plot(fig_state_accuracy, "state accuracy by transition drug", stem="state_accuracy_transition_drug")], align="center"),
                    mo.vstack([fig_state_occupancy, save_plot(fig_state_occupancy, "state occupancy by transition drug", stem="state_occupancy_transition_drug")], align="center"),
                    mo.vstack([fig_switches, save_plot(fig_switches, "state switches by transition drug", stem="state_switches_transition_drug")], align="center"),
                ],
                align="center",
            ),
            fig_state_categorical2,
            save_plot(fig_state_categorical2, "psychometric by state transition drug", stem="psychometric_transition_drug"),
        ],
        align="center",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
