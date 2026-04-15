import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from plot_saver import make_plot_saver
    from glmhmmt.notebook_support import (
        CoefficientEditorWidget,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        model_cfg as ModelCfg,
        wrap_anywidget,
    )
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        resolve_selected_model_id,
        select_subject_behavior_df,
    )
    from glmhmmt.cli.fit_glm import main as fit_main, generate_model_id
    from glmhmmt.postprocess import build_trial_df
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import get_state_color

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    sns.set_style("ticks")
    paths = get_runtime_paths()
    return (
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        build_trial_and_weights_df,
        build_trial_df,
        fit_main,
        generate_model_id,
        get_adapter,
        load_fit_arrays,
        make_plot_saver,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        resolve_selected_model_id,
        select_subject_behavior_df,
        sns,
        wrap_anywidget,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We import the ui widgets and the model adapters
    """)
    return


@app.cell
def _(get_adapter, model_cfg):
    task_name = model_cfg.task
    adapter = get_adapter(task_name)
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    plots = adapter.get_plots()
    return adapter, df_all, plots, task_name


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glm",
        task="2AFC",
        tau=50,
        lapse_mode="none",
        lapse_max=0.2,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return mm_widget, ui_model_manager


@app.cell
def _(ModelCfg, ui_model_manager):
    model_cfg = ModelCfg.from_value(ui_model_manager.value)
    is_2afc = (model_cfg.task != "MCDR")
    return is_2afc, model_cfg


@app.cell
def _(mo):
    get_last_fit_click, set_last_fit_click = mo.state(0)
    return get_last_fit_click, set_last_fit_click


@app.cell
def _(generate_model_id, model_cfg, task_name):
    current_hash = generate_model_id(
        task_name,
        model_cfg.tau,
        model_cfg.emission_cols,
        lapse_mode=model_cfg.lapse_mode,
        lapse_max=model_cfg.lapse_max,
    )
    return (current_hash,)


@app.cell
def _(current_hash, model_cfg, resolve_selected_model_id):
    selected_model_id = resolve_selected_model_id(
        current_hash,
        model_cfg.existing,
        model_cfg.alias,
    )
    return (selected_model_id,)


@app.cell
def _(make_plot_saver, mo, paths, selected_model_id, task_name):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=task_name,
        model_id=selected_model_id,
    )
    return (save_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Configuration
    """)
    return


@app.cell
def _(current_hash, mo, save_plot, ui_model_manager):
    mo.vstack([
        ui_model_manager,
        save_plot.save_all_widget(label="Save all model plots"),
        mo.md(f"**Current params hash:** `{current_hash}`"),
    ])
    return


@app.cell
def _(
    current_hash,
    fit_main,
    get_last_fit_click,
    mm_widget,
    mo,
    model_cfg,
    paths,
    set_last_fit_click,
    task_name,
):
    _last_fit_click = get_last_fit_click()
    mo.stop(
        model_cfg.run_fit_clicks <= _last_fit_click,
        mo.md("Configure parameters and press **Run fit**."),
    )
    set_last_fit_click(model_cfg.run_fit_clicks)

    _n_restarts = 1
    _selected_id = model_cfg.existing or (model_cfg.alias if model_cfg.alias else current_hash)
    _OUT = paths.RESULTS / "fits" / task_name / "glm" / _selected_id

    def _progress_title(info: dict) -> str:
        return (
            f"Fitting GLM subject {info['subject_index']}/{info['subject_total']}: "
            f"{info['subject']}"
        )

    def _progress_subtitle(info: dict) -> str:
        _base = f"Restart {info['restart_index']}/{info['restart_total']}"
        if info.get("event") == "restart_complete":
            return f"{_base} complete"
        return _base

    _total_progress = max(1, len(model_cfg.subjects) * _n_restarts)
    mm_widget.is_running = True
    try:
        with mo.status.progress_bar(
            total=_total_progress,
            title="Fitting GLM",
            subtitle=f"{len(model_cfg.subjects)} subjects × {_n_restarts} restart(s)",
            completion_title="Fit complete",
            completion_subtitle=f"Saved under {_selected_id}",
        ) as _bar:
            def _on_progress(info: dict) -> None:
                if info.get("event") == "restart_start":
                    _bar.update(
                        increment=0,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )
                    return
                if info.get("event") == "restart_complete":
                    _bar.update(
                        increment=1,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )

            fit_main(
                subjects=model_cfg.subjects,
                out_dir=_OUT,
                tau=model_cfg.tau,
                emission_cols=model_cfg.emission_cols,
                task=task_name,
                model_alias=model_cfg.alias if model_cfg.alias else None,
                lapse_mode=model_cfg.lapse_mode,
                lapse_max=model_cfg.lapse_max,
                n_restarts=_n_restarts,
                verbose=False,
                progress_callback=_on_progress,
            )
        mm_widget.saved_model_name = _selected_id
        mm_widget.alias_error = ""
        mm_widget.alias_status = ""
        if not model_cfg.alias:
            mm_widget.alias = _selected_id
        mm_widget._update_options()
        if _selected_id in mm_widget.existing_models:
            mm_widget.existing_model = _selected_id
    finally:
        mm_widget.is_running = False

    mo.md("✅ Fit complete — plots below update automatically.")
    return


@app.cell
def _(
    adapter,
    df_all,
    load_fit_arrays,
    mo,
    model_cfg,
    paths,
    selected_model_id,
    task_name,
):
    def _normalize_glm_arrays(arrays: dict) -> dict:
        # ── Backward-compatibility: old fit_glm.py saved W_R at index 0.
        # New convention stores W_L (negative stim weight) at index 0.
        _weights = arrays.get("emission_weights")
        if _weights is None:
            return arrays

        stim_idx = next(
            (idx for idx, col in enumerate(arrays.get("X_cols", [])) if col in {"stim_vals", "stim_d", "ild_norm"}),
            None,
        )
        if stim_idx is not None and float(_weights[0, 0, stim_idx]) > 0:
            arrays["emission_weights"] = -_weights
        return arrays


    OUT = paths.RESULTS / "fits" / task_name / "glm" / selected_model_id
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=list(model_cfg.subjects),
        emission_cols=list(model_cfg.emission_cols),
        postprocess_array=_normalize_glm_arrays,
    )

    mo.md(f"Loaded {len(arrays_store)} subjects from `{selected_model_id}`")
    return (arrays_store,)


@app.cell
def _(adapter, arrays_store, mo, model_cfg):
    selected = [s for s in model_cfg.subjects if s in arrays_store]
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    from glmhmmt.views import build_views
    K = 1
    views = build_views(arrays_store, adapter, K, selected)
    return K, build_views, selected, views


@app.cell
def _(adapter, arrays_store, build_views):
    editor_views = build_views(arrays_store, adapter, 1, list(arrays_store.keys()))
    return (editor_views,)


@app.cell
def _(is_2afc, np, pd, plt, sns):
    import re

    _stim_pattern = re.compile(r"^stim_(\d+)$")
    _choice_lag_pattern = re.compile(r"^choice_lag_(\d+)$")
    _bias_pattern = re.compile(r"^bias_(\d+)$")
    _stim_hot_order = [2, 4, 8, 20]

    def _weights_df_to_plot_df(weights_df) -> pd.DataFrame:
        df_plot = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        df_plot = df_plot[df_plot["class_idx"] == 0].copy()
        df_plot["feature"] = df_plot["feature"].astype(str)
        df_plot["weight"] = pd.to_numeric(df_plot["weight"], errors="coerce")
        df_plot["weight"] = -df_plot["weight"]
        df_plot["state_rank"] = 0
        df_plot["state_label"] = "State 0"
        return df_plot.dropna(subset=["weight"]).copy()

    def _raw_one_hot_mask(feature_series: pd.Series) -> pd.Series:
        feature_names = feature_series.astype(str)
        return (
            feature_names.str.match(_stim_pattern)
            | feature_names.str.match(_choice_lag_pattern)
            | feature_names.str.match(_bias_pattern)
        )

    def plot_emission_weights_summary_clean(weights_df) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(weights_df)
        if df_plot.empty:
            return None
        df_plot = df_plot.loc[~_raw_one_hot_mask(df_plot["feature"])].copy()
        if df_plot.empty:
            return None

        def _feature_group(name: str) -> tuple[int, str]:
            lname = str(name).lower()
            if lname.startswith("stim"):
                return (0, lname)
            if "bias" in lname:
                return (1, lname)
            if lname.startswith("at_"):
                return (2, lname)
            return (3, lname)

        feature_order = sorted(pd.unique(df_plot["feature"]), key=_feature_group)
        df_plot["feature_label"] = pd.Categorical(
            df_plot["feature"],
            categories=feature_order,
            ordered=True,
        )
        summary = (
            df_plot.groupby("feature_label", as_index=False, observed=False)["weight"]
            .mean()
            .sort_values("feature_label")
        )
        if summary.empty:
            return None

        positions = np.arange(len(feature_order))
        fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(feature_order)), 4.2))
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title("Emission weights")
        ax.set_xlabel("")
        ax.set_ylabel("Weight")
        ax.set_xticks(positions)
        ax.set_xticklabels(feature_order, rotation=35, ha="right")
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_stim_hot_weights(weights_df) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(weights_df)
        if df_plot.empty:
            return None
        df_plot["stim_level"] = df_plot["feature"].str.extract(_stim_pattern, expand=False)
        df_plot = df_plot[df_plot["stim_level"].notna()].copy()
        if df_plot.empty:
            return None
        df_plot["stim_level"] = df_plot["stim_level"].astype(int)
        df_plot = df_plot[df_plot["stim_level"].isin(_stim_hot_order)].copy()
        if df_plot.empty:
            return None
        observed_levels = [level for level in _stim_hot_order if level in set(df_plot["stim_level"])]
        summary = (
            df_plot.groupby("stim_level", as_index=False)["weight"]
            .mean()
            .sort_values("stim_level")
        )
        if summary.empty:
            return None

        positions = np.arange(len(observed_levels))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title("stim_hot")
        ax.set_xlabel("Stimulus level")
        ax.set_ylabel("Weight")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(level) for level in observed_levels])
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_choice_lag_weights(weights_df) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(weights_df)
        if df_plot.empty:
            return None
        df_plot["lag"] = df_plot["feature"].str.extract(_choice_lag_pattern, expand=False)
        df_plot = df_plot[df_plot["lag"].notna()].copy()
        if df_plot.empty:
            return None
        df_plot["lag"] = df_plot["lag"].astype(int)
        lag_ticks = sorted(pd.unique(df_plot["lag"]).tolist())
        summary = (
            df_plot.groupby("lag", as_index=False)["weight"]
            .mean()
            .sort_values("lag")
        )
        if summary.empty:
            return None

        positions = np.arange(len(lag_ticks))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title("choice_lag_*")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Weight")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(lag) for lag in lag_ticks])
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_bias_hot_weights(weights_df) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(weights_df)
        if df_plot.empty:
            return None
        df_plot["session_idx"] = df_plot["feature"].str.extract(_bias_pattern, expand=False)
        df_plot = df_plot[df_plot["session_idx"].notna()].copy()
        if df_plot.empty:
            return None
        df_plot["session_idx"] = df_plot["session_idx"].astype(int)
        session_order = sorted(pd.unique(df_plot["session_idx"]).tolist())
        summary = (
            df_plot.groupby("session_idx", as_index=False)["weight"]
            .mean()
            .sort_values("session_idx")
        )
        if summary.empty:
            return None

        positions = np.arange(len(session_order))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title("bias_hot")
        ax.set_xlabel("Session index")
        ax.set_ylabel("Weight")
        if len(session_order) == 1:
            ax.set_xticks([0])
            ax.set_xticklabels([str(session_order[0])])
        else:
            ax.set_xticks([positions[0], positions[-1]])
            ax.set_xticklabels([str(session_order[0]), str(session_order[-1])])
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_sequence_feature_weights(weights_df) -> plt.Figure | None:
        """Plot only sequential stimulus features (s_i / sf_i) from the canonical weights df."""
        feature_pattern = re.compile(r"^(?:s|sf)_(\d+)$")
        if weights_df is None or getattr(weights_df, "is_empty", lambda: False)():
            return None

        df_plot = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        if df_plot.empty:
            return None

        df_plot["feature_name"] = df_plot["feature"].astype(str)
        df_plot["seq_idx"] = df_plot["feature_name"].str.extract(feature_pattern, expand=False)
        df_plot = df_plot[df_plot["seq_idx"].notna()].copy()
        if df_plot.empty:
            return None

        df_plot["seq_idx"] = df_plot["seq_idx"].astype(int)
        if is_2afc:
            # Binary fits store logit(Left); flip sign so the plot keeps the intuitive rightward convention.
            df_plot["weight"] = -df_plot["weight"]

        # Collapse across class_idx so each subject/state/feature contributes one value.
        df_plot = (
            df_plot.groupby(
                ["subject", "state_rank", "state_label", "seq_idx", "feature_name"],
                as_index=False,
            )["weight"]
            .mean()
        )

        state_order = (
            df_plot[["state_rank", "state_label"]]
            .drop_duplicates()
            .sort_values("state_rank")
        )
        n_states = max(1, len(state_order))
        fig, axes = plt.subplots(1, n_states, figsize=(4.8 * n_states, 3.8), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, (_, state_row) in zip(axes, state_order.iterrows()):
            state_rank = int(state_row["state_rank"])
            state_label = str(state_row["state_label"])
            state_df = df_plot[df_plot["state_rank"] == state_rank].copy()
            state_df = state_df.sort_values(["subject", "seq_idx"])

            for _, subj_df in state_df.groupby("subject", sort=False):
                ax.plot(
                    subj_df["seq_idx"],
                    subj_df["weight"],
                    color="#bdbdbd",
                    alpha=0.35,
                    linewidth=1.0,
                )

            summary = (
                state_df.groupby(["seq_idx", "feature_name"], as_index=False)
                .agg(
                    mean=("weight", "mean"),
                    std=("weight", "std"),
                    count=("weight", "count"),
                )
            )
            summary["sem"] = np.where(
                summary["count"] > 1,
                summary["std"] / np.sqrt(summary["count"]),
                0.0,
            )
            summary = summary.sort_values("seq_idx")

            ax.plot(
                summary["seq_idx"],
                summary["mean"],
                color="#1f77b4",
                marker="o",
                linewidth=2.2,
            )
            if len(summary) > 1:
                ax.fill_between(
                    summary["seq_idx"],
                    summary["mean"] - summary["sem"],
                    summary["mean"] + summary["sem"],
                    color="#1f77b4",
                    alpha=0.15,
                )

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title(state_label)
            ax.set_xlabel("Sequential stimulus features")
            ax.set_xticks(summary["seq_idx"])
            ax.set_xticklabels(summary["feature_name"], rotation=35, ha="right")
            sns.despine(ax=ax)

        axes[0].set_ylabel("Weight")
        fig.suptitle("s_i / sf_i coefficients", y=1.02)
        fig.tight_layout()
        return fig

    return (
        plot_bias_hot_weights,
        plot_choice_lag_weights,
        plot_emission_weights_summary_clean,
        plot_sequence_feature_weights,
        plot_stim_hot_weights,
    )


@app.cell
def _(adapter, build_trial_and_weights_df, df_all, mo, views):
    trial_df, weights_df = build_trial_and_weights_df(
        df_all,
        views=views,
        adapter=adapter,
        min_session_length=1,
    )
    mo.stop(trial_df.height == 0, mo.md("No subjects with matching data lengths."))
    return trial_df, weights_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Emission Weights
    """)
    return


@app.cell
def _(weights_df):
    weights_df
    return


@app.cell
def _(
    K,
    arrays_store,
    mo,
    pl,
    plot_bias_hot_weights,
    plot_choice_lag_weights,
    plot_emission_weights_summary_clean,
    plot_sequence_feature_weights,
    plot_stim_hot_weights,
    plots,
    save_plot,
    selected,
    views,
    weights_df,
):
    mo.stop(not arrays_store, mo.md("No results loaded."))
    views_sel = {s: views[s] for s in selected}
    _weights_df_sel = weights_df.filter(pl.col("subject").is_in(selected))

    _fig_by_subject = plots.plot_emission_weights_by_subject(
        views=views_sel,
        K=K,
    )
    _fig_summary = plot_emission_weights_summary_clean(_weights_df_sel)
    _fig_stim_hot = plot_stim_hot_weights(_weights_df_sel)
    _fig_choice_lag = plot_choice_lag_weights(_weights_df_sel)
    _fig_bias_hot = plot_bias_hot_weights(_weights_df_sel)
    _fig_lapses = plots.plot_lapse_rates_boxplot(views=views_sel, K=K, collapse_lapses=False)
    _fig_seq = plot_sequence_feature_weights(_weights_df_sel)
    _items = [mo.md("#### By subject"), _fig_by_subject]
    if _fig_seq is not None:
        _items.extend([mo.md("#### Sequential coefficients"), _fig_seq])
    else:
        _items.extend(
            [
                mo.md("#### Sequential coefficients"),
                mo.md("No `s_i` / `sf_i` regressors found in the current GLM fit."),
            ]
        )
    _summary_cards = []
    if _fig_summary is not None:
        _summary_cards.append(
            mo.vstack(
                [_fig_summary, save_plot(_fig_summary, "emission weights", stem="emission_weights")],
                align="center",
            )
        )
    _summary_cards.append(
        mo.vstack([_fig_lapses, save_plot(_fig_lapses, "lapse rates", stem="lapse_rates")], align="center")
    )
    _summary_panel = mo.hstack(_summary_cards, align="start", justify="start", gap=1.0)
    _items.extend([mo.md("#### Summary"), _summary_panel])

    if _fig_stim_hot:
        _items.extend(
            [
                mo.md("#### One-hot families"),
                mo.hstack([_fig_stim_hot, _fig_choice_lag], align="start", justify="start", gap=1.0),
                _fig_bias_hot
            ]
        )
    mo.vstack(_items, align = "center")
    return (views_sel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary
    """)
    return


@app.cell
def _(is_2afc, mo, pl, plots, save_plot, selected, trial_df, views_sel):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))

    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))

    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    _plot_df_all = plots.prepare_predictions_df(_trial_df_sel)
    _perf_kwargs = {"views": views_sel} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df_all,
        "glm",
        background_style="model",
        **_perf_kwargs,
    )

    mo.vstack([_fig_all, save_plot(_fig_all, "overall psychometric", stem="categorical_overall"),], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## By subject (editable)
    """)
    return


@app.cell
def _(editor_views, mo):
    subjects = list(editor_views.keys())
    mo.stop(not subjects, mo.md("No fitted subjects available for coefficient editing."))
    ui_editor_subject = mo.ui.dropdown(
        options=subjects,
        value=subjects[0],
        label="Subject",
    )
    return (ui_editor_subject,)


@app.cell
def _(adapter, editor_views, mo, ui_editor_subject):
    _view = editor_views[ui_editor_subject.value]
    _state_options = [
        f"{_k} — {_view.state_name_by_idx.get(_k, f'State {_k}')}"
        for _k in _view.state_idx_order
    ]
    ui_editor_state = mo.ui.dropdown(
        options=_state_options,
        value=_state_options[0],
        label="State",
    )
    _choices = [str(label) for label in adapter.choice_labels]
    ui_editor_side = mo.ui.dropdown(
        options=_choices,
        value=_choices[0],
        label="Side",
    )
    mo.hstack([ui_editor_subject, ui_editor_state, ui_editor_side], justify="center")
    return ui_editor_side, ui_editor_state


@app.cell
def _(
    CoefficientEditorWidget,
    adapter,
    build_editor_payload,
    editor_views,
    np,
    ui_editor_side,
    ui_editor_state,
    ui_editor_subject,
    wrap_anywidget,
):
    subject = ui_editor_subject.value
    view = editor_views[subject]
    coef_state_idx = int(ui_editor_state.value.split(" — ", 1)[0])
    coef_state_label = view.state_name_by_idx.get(
        coef_state_idx, f"State {coef_state_idx}"
    )
    _stored_weights = np.asarray(view.emission_weights[coef_state_idx], dtype=float)
    _choice_labels = [str(label) for label in adapter.choice_labels]
    _stored_class_indices = [0] if view.num_classes == 2 else [0, 2]
    _reference_class_idx = 1 if view.num_classes > 2 else (view.num_classes - 1)
    if view.num_classes == 2 and ui_editor_side is not None:
        _display_class_idx = _choice_labels.index(ui_editor_side.value)
        _display_reference_class_idx = next(
            idx for idx in range(view.num_classes) if idx != _display_class_idx
        )
    else:
        _display_reference_class_idx = None

    coef_editor_payload = build_editor_payload(
        _stored_weights,
        choice_labels=_choice_labels,
        stored_class_indices=_stored_class_indices,
        reference_class_idx=_reference_class_idx,
        display_reference_class_idx=_display_reference_class_idx,
    )

    coef_editor = wrap_anywidget(
        CoefficientEditorWidget(
            title="Coefficient editor",
            subtitle=coef_editor_payload["subtitle"],
            features=list(view.feat_names),
            channel_labels=coef_editor_payload["channel_labels"],
            weights=coef_editor_payload["weights"].tolist(),
            original_weights=coef_editor_payload["weights"].tolist(),
            slider_min=-6.0,
            slider_max=6.0,
            slider_step=0.05,
        )
    )
    _controls = [ui_editor_subject, ui_editor_state]
    if ui_editor_side is not None:
        _controls.append(ui_editor_side)


    coef_editor
    return (
        coef_editor,
        coef_editor_payload,
        coef_state_idx,
        coef_state_label,
        subject,
        view,
    )


@app.cell
def _(
    adapter,
    build_trial_df,
    df_all,
    mo,
    select_subject_behavior_df,
    subject,
    view,
):
    _df_sub = select_subject_behavior_df(
        df_all,
        subject=subject,
        sort_col=adapter.sort_col,
        session_col=adapter.session_col,
        min_session_length=1,
    )
    mo.stop(_df_sub.height != view.T, mo.md(f"Subject {subject} does not match the loaded fit arrays."))
    editor_trial_df = build_trial_df(view, adapter, _df_sub, adapter.behavioral_cols)
    return (editor_trial_df,)


@app.cell
def _(
    adapter,
    apply_state_tweak_to_trial_df,
    apply_state_tweak_to_view,
    coef_editor,
    coef_editor_payload,
    coef_state_idx,
    coef_state_label,
    editor_trial_df,
    mo,
    np,
    plots,
    save_plot,
    subject,
    view,
):
    _trial_df_sub = editor_trial_df
    _edited_weights = np.asarray(coef_editor.value["weights"], dtype=float)

    _trial_df_tweaked = apply_state_tweak_to_trial_df(
        _trial_df_sub,
        adapter=adapter,
        view=view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        original_weights=np.asarray(coef_editor.value["original_weights"], dtype=float),
        explicit_class_indices=list(coef_editor_payload["explicit_class_indices"]),
        reference_class_idx=int(coef_editor_payload["reference_class_idx"]),
    )
    _view_tweaked = apply_state_tweak_to_view(
        view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        explicit_class_indices=list(coef_editor_payload["explicit_class_indices"]),
        reference_class_idx=int(coef_editor_payload["reference_class_idx"]),
        stored_class_indices=list(coef_editor_payload["stored_class_indices"]),
        stored_reference_class_idx=int(coef_editor_payload["stored_reference_class_idx"]),
    )
    _plot_df_tweaked = plots.prepare_predictions_df(_trial_df_tweaked)

    _title = f"{subject} — tweaked {coef_state_label}"
    _fig_all_tweaked, _ = plots.plot_categorical_performance_all(
        _plot_df_tweaked,
        _title,
        # background_style=ui_psychometric_background.value,
    )
    _side_plot_fn = getattr(plots, "plot_categorical_strat_by_side", None)
    if _side_plot_fn is None:
        _fig_side_tweaked = mo.md("This task does not expose a side-stratified categorical plot.")
    else:
        _fig_side_tweaked, _ = plots.plot_categorical_strat_by_side(
            _plot_df_tweaked,
            subject=subject,
            model_name=f"{subject}_tweaked_{coef_state_idx}",
        )


    mo.hstack(
        [
            mo.vstack(
                [
                    _fig_all_tweaked,
                    save_plot(
                        _fig_all_tweaked,
                        "tweaked overall psychometric",
                        stem="tweaked_categorical_overall",
                    ),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    _fig_side_tweaked,
                    save_plot(
                        _fig_side_tweaked,
                        "tweaked overall psychometric",
                        stem="tweaked_categorical_side",
                    ),
                ],
                align="center",
            ),
        ],
        widths=[2.5, 1],
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
