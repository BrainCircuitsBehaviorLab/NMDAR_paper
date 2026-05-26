import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    import importlib
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # _local_glmhmmt_src = Path(__file__).resolve().parents[2] / "glmhmmt" / "src"
    # if _local_glmhmmt_src.exists() and str(_local_glmhmmt_src) not in sys.path:
    #     sys.path.insert(0, str(_local_glmhmmt_src))

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
    import glmhmmt.cli.fit_glm as _fit_glm_cli
    _fit_glm_cli = importlib.reload(_fit_glm_cli)
    fit_main = _fit_glm_cli.main
    generate_model_id = _fit_glm_cli.generate_model_id
    from glmhmmt.glm import fit_glm
    import glmhmmt.postprocess as _postprocess
    build_trial_df = _postprocess.build_trial_df
    build_emission_weights_df = _postprocess.build_emission_weights_df
    build_weights_boxplot_payload = _postprocess.build_weights_boxplot_payload
    import glmhmmt.plots.emissions as _emission_plots
    import glmhmmt.plots as model_plots
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import get_state_color
    from src.plots.common import plot_prepared_weight_family, plot_regressor_net_impact
    from src.process import MCDR as process_mcdr
    process_mcdr = importlib.reload(process_mcdr)
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import add_choice_lag_summary_regressor, display_regressor_name

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    sns.set_style("ticks")
    sns.set_context("paper")

    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")

    paths = get_runtime_paths()
    from src.utils import fig_size

    return (
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        add_choice_lag_summary_regressor,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        build_trial_and_weights_df,
        build_trial_df,
        display_regressor_name,
        fig_size,
        fit_glm,
        fit_main,
        generate_model_id,
        get_adapter,
        load_fit_arrays,
        make_plot_saver,
        mo,
        model_plots,
        np,
        paths,
        pd,
        pl,
        plot_prepared_weight_family,
        plot_regressor_net_impact,
        plt,
        prepare_predictions_df,
        process_mcdr,
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
    if adapter.condition_filter_options():
        df_all = adapter.filter_condition_df(df_all, model_cfg.condition_filter)
    plots = adapter.get_plots()
    return adapter, df_all, plots, task_name


@app.cell
def _(adapter, plot_df_all, plots, views_sel):
    _fig_total_evidence = plots.plot_accuracy_by_total_evidence(
        plot_df_all,
        adapter=adapter,
        views=views_sel,
    )
    return


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
    ui_emission_model = mo.ui.dropdown(
        options=["standard", "private_alternative"],
        value="standard",
        label="Emission model",
    )
    return (ui_emission_model,)


@app.cell
def _(mo):
    ui_private_alternative_variant = mo.ui.dropdown(
        options={"private batch 11": "batch11", "private batch 3B": "batch3b"},
        value="private batch 11",
        label="Private design",
    )
    return (ui_private_alternative_variant,)


@app.cell
def _(task_name, ui_emission_model):
    emission_model = (
        ui_emission_model.value
        if task_name == "MCDR"
        else "standard"
    )
    return (emission_model,)


@app.cell
def _(emission_model, task_name, ui_private_alternative_variant):
    private_alternative_variant = (
        ui_private_alternative_variant.value
        if task_name == "MCDR" and emission_model == "private_alternative"
        else "batch11"
    )
    return (private_alternative_variant,)


@app.cell
def _(mo):
    get_last_fit_click, set_last_fit_click = mo.state(0)
    return get_last_fit_click, set_last_fit_click


@app.cell
def _(
    adapter,
    emission_model,
    generate_model_id,
    model_cfg,
    private_alternative_variant,
    process_mcdr,
    task_name,
):
    current_hash = process_mcdr.generate_private_alternative_model_id(
        generate_model_id,
        task_name,
        model_cfg.tau,
        model_cfg.emission_cols,
        lapse_mode=model_cfg.lapse_mode,
        lapse_max=model_cfg.lapse_max,
        baseline_class_idx=adapter.baseline_class_idx,
        condition_filter=model_cfg.condition_filter,
        emission_model=emission_model,
        private_alternative_variant=private_alternative_variant,
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
        model_id=f"glm/{selected_model_id}",
    )
    return (save_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Configuration
    """)
    return


@app.cell
def _(
    current_hash,
    mo,
    save_plot,
    ui_emission_model,
    ui_model_manager,
    ui_private_alternative_variant,
):
    mo.vstack([
        ui_model_manager,
        ui_emission_model,
        ui_private_alternative_variant,
        save_plot.save_all_widget(label="Save all model plots"),
        mo.md(f"**Current params hash:** `{current_hash}`"),
    ])
    return


@app.cell
def _():
    #trial_df.group_by(["batch", "subject", "drug"]).len()
    return


@app.cell
def _(weights_df):
    weights_df
    return


@app.cell
def _(
    adapter,
    current_hash,
    emission_model,
    fit_main,
    get_last_fit_click,
    mm_widget,
    mo,
    model_cfg,
    paths,
    private_alternative_variant,
    process_mcdr,
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

            if task_name == "MCDR" and emission_model == "private_alternative":
                process_mcdr.set_private_alternative_variant(private_alternative_variant)

            fit_main(
                subjects=model_cfg.subjects,
                out_dir=_OUT,
                tau=model_cfg.tau,
                emission_cols=model_cfg.emission_cols,
                task=task_name,
                model_alias=model_cfg.alias if model_cfg.alias else None,
                lapse_mode=model_cfg.lapse_mode,
                n_restarts=_n_restarts,
                verbose=True,
                progress_callback=_on_progress,
                baseline_class_idx=adapter.baseline_class_idx,
                condition_filter=model_cfg.condition_filter,
                emission_model=emission_model,
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
def _(private_alternative_variant):
    private_alternative_variant
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
def _(mo):
    ui_mcdr_one_hot_mode = mo.ui.dropdown(
        options=["folded", "split"],
        value="folded",
        label="MCDR one-hot view (folded or split)",
    )
    return (ui_mcdr_one_hot_mode,)


@app.cell
def _(
    adapter,
    fig_size,
    fit_glm,
    np,
    pd,
    plot_prepared_weight_family,
    plt,
    sns,
    task_name,
):
    from scipy.stats import ttest_1samp

    def _significance_stars(pvalue: float) -> str:
        if not np.isfinite(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def _annotate_weight_family_against_zero(prepared, ax: plt.Axes | None) -> None:
        if prepared is None or ax is None or prepared.plot_kind != "box":
            return

        df = pd.DataFrame(prepared.data).copy()
        required = {"subject", "x_label", "weight"}
        if df.empty or not required.issubset(df.columns):
            return

        df["subject"] = df["subject"].astype(str)
        df["x_label"] = df["x_label"].astype(str)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["weight"])
        if df.empty:
            return

        x_order = (
            list(prepared.x_order)
            if prepared.x_order is not None
            else pd.unique(df["x_label"]).tolist()
        )
        present_labels = set(df["x_label"])
        x_order = [str(label) for label in x_order if str(label) in present_labels]
        if not x_order:
            return

        _, y_top = ax.get_ylim()
        data_low = float(np.nanmin(df["weight"].to_numpy(dtype=float)))
        data_high = float(np.nanmax(df["weight"].to_numpy(dtype=float)))
        y_bottom = min(float(ax.get_ylim()[0]), data_low, 0.0)
        y_top = max(float(y_top), data_high, 0.0)
        y_span = y_top - y_bottom
        if not np.isfinite(y_span) or y_span <= 0:
            y_span = 1.0
        y_text = y_top + 0.04 * y_span

        annotated = False
        for xpos, x_label in enumerate(x_order, start=1):
            values = (
                df.loc[df["x_label"] == x_label]
                .groupby("subject", observed=False)["weight"]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            if values.size < 2:
                continue
            pvalue = float(ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue)
            label = _significance_stars(pvalue)
            if not label:
                continue
            ax.text(
                xpos,
                y_text,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                clip_on=False,
            )
            annotated = True

        if annotated:
            ax.set_ylim(y_bottom, y_text + 0.05 * y_span)

    def plot_stim_hot_weights(
        weights_df,
        *,
        mcdr_mode: str = "folded",
        ax: plt.Axes | None = None,
        connect_subjects: bool = False,
    ) -> plt.Figure | None:
        prepared = adapter.prepare_weight_family_plot(weights_df, "stim_hot", variant=mcdr_mode)
        fig = plot_prepared_weight_family(
            prepared,
            figsize=fig_size(2, 1),
            ax=ax,
            connect_subjects=connect_subjects,
        )
        if fig is not None:
            target_ax = ax if ax is not None else (fig.axes[0] if fig.axes else None)
            if target_ax is not None:
                _annotate_weight_family_against_zero(prepared, target_ax)
                if task_name == "2AFC":
                    target_ax.set_xlabel("Stimulus level")
                elif task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
                    target_ax.set_xlabel("Delay level")
        return fig

    def plot_choice_lag_weights(
        weights_df,
        *,
        mcdr_mode: str = "folded",
        ax: plt.Axes | None = None,
        connect_subjects: bool = False,
    ) -> plt.Figure | None:
        prepared = adapter.prepare_weight_family_plot(
            weights_df,
            "choice_lag",
            variant=mcdr_mode,
        )
        fig = plot_prepared_weight_family(
            prepared,
            figsize=fig_size(2, 1),
            ax=ax,
            connect_subjects=connect_subjects,
        )
        target_ax = (
            ax
            if ax is not None
            else (fig.axes[0] if fig is not None and fig.axes else None)
        )
        _annotate_weight_family_against_zero(prepared, target_ax)
        return fig

    def plot_bias_hot_weights(weights_df) -> plt.Figure | None:
        return plot_prepared_weight_family(
            adapter.prepare_weight_family_plot(weights_df, "bias_hot")
        )

    def plot_sequence_feature_weights(weights_df) -> plt.Figure | None:
        """Plot only sequential stimulus features (s_i / sf_i) from the canonical weights df."""
        if weights_df is None or getattr(weights_df, "is_empty", lambda: False)():
            return None

        df_plot = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        if df_plot.empty:
            return None

        df_plot["feature_name"] = df_plot["feature"].astype(str)
        seq_pattern = r"^(?:s|sf)_(\d+)$"
        df_plot["seq_idx"] = df_plot["feature_name"].str.extract(seq_pattern, expand=False)
        df_plot = df_plot[df_plot["seq_idx"].notna()].copy()
        if df_plot.empty:
            return None

        df_plot["seq_idx"] = df_plot["seq_idx"].astype(int)
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

    def build_regressor_lr_df(
        arrays_store,
        *,
        lapse_max: float = 0.2,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Build per-subject likelihood-ratio chi2 values for each regressor.

        The statistic is computed as ``2 * (LL_full - LL_reduced)`` where the
        full-model log-likelihood comes from the stored fitted predictive
        probabilities and the reduced model is re-fit after dropping one
        regressor column for the current subject.
        """
        records: list[dict[str, object]] = []

        for subject, arrays in arrays_store.items():
            x_raw = arrays.get("X")
            y_raw = arrays.get("y")
            weights_raw = arrays.get("emission_weights")
            x_cols_raw = arrays.get("X_cols")
            p_pred_raw = arrays.get("p_pred")
            if (
                x_raw is None
                or y_raw is None
                or weights_raw is None
                or x_cols_raw is None
                or p_pred_raw is None
            ):
                continue

            X = np.asarray(x_raw, dtype=float)
            y = np.asarray(y_raw, dtype=int).reshape(-1)
            emission_weights = np.asarray(weights_raw, dtype=float)
            p_pred = np.asarray(p_pred_raw, dtype=float)
            if X.ndim != 2 or emission_weights.ndim != 3 or emission_weights.shape[0] == 0:
                continue
            if p_pred.ndim != 2 or p_pred.shape[0] != X.shape[0] or y.shape[0] != X.shape[0]:
                continue

            feature_names = [str(col) for col in np.asarray(x_cols_raw).tolist()]
            if X.shape[1] != len(feature_names):
                continue

            if len(feature_names) <= 1:
                continue

            num_classes = int(p_pred.shape[1])
            baseline_class_idx = int(np.asarray(arrays.get("baseline_class_idx", 0)).reshape(()))
            lapse_mode = str(arrays.get("lapse_mode", "none"))

            full_probs = np.clip(p_pred, 1e-10, 1.0)
            full_probs /= full_probs.sum(axis=1, keepdims=True)
            full_ll = float(np.sum(np.log(full_probs[np.arange(y.shape[0]), y])))

            for feat_idx, feature_name in enumerate(feature_names):
                X_reduced = np.delete(X, feat_idx, axis=1)
                if X_reduced.shape[1] == 0:
                    continue

                reduced_fit = fit_glm(
                    X_reduced,
                    y,
                    num_classes=num_classes,
                    baseline_class_idx=baseline_class_idx,
                    lapse_mode=lapse_mode,
                    lapse_max=float(lapse_max),
                    n_restarts=1,
                    restart_noise_scale=0.0,
                    seed=seed,
                )
                reduced_probs = np.asarray(reduced_fit.predictive_probs, dtype=float)
                reduced_probs = np.clip(reduced_probs, 1e-10, 1.0)
                reduced_probs /= reduced_probs.sum(axis=1, keepdims=True)
                reduced_ll = float(np.sum(np.log(reduced_probs[np.arange(y.shape[0]), y])))
                lr_chi2 = max(0.0, 2.0 * (full_ll - reduced_ll))

                records.append(
                    {
                        "subject": str(subject),
                        "feature": feature_name,
                        "lr_chi2": lr_chi2 if np.isfinite(lr_chi2) else np.nan,
                    }
                )

        return pd.DataFrame.from_records(records, columns=["subject", "feature", "lr_chi2"])

    return (
        plot_bias_hot_weights,
        plot_choice_lag_weights,
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
def _(arrays_store, mo, selected, views):
    mo.stop(not arrays_store, mo.md("No results loaded."))
    views_sel = {s: views[s] for s in selected}
    return (views_sel,)


@app.cell
def _(ui_mcdr_one_hot_mode):
    ui_mcdr_one_hot_mode
    return


@app.cell
def _(pl, weights_df):
    weights_df.filter(pl.col("feature").str.contains("session").not_(), pl.col("feature").str.contains("bias").not_(),)
    return


@app.cell
def _(fig_size):
    fig_size(2, 1)
    return


@app.cell
def _():
    return


@app.cell
def _(
    K,
    fig_size,
    mo,
    model_plots,
    np,
    pl,
    plot_bias_hot_weights,
    plot_choice_lag_weights,
    plot_sequence_feature_weights,
    plot_stim_hot_weights,
    plt,
    save_plot,
    selected,
    sns,
    task_name,
    ui_mcdr_one_hot_mode,
    views_sel,
    weights_df,
):
    sns.set_context("paper")
    _weights_df_sel = weights_df.filter(pl.col("subject").is_in(selected))
    _mcdr_mode = ui_mcdr_one_hot_mode.value if task_name == "MCDR" else "folded"

    # _fig_by_subject = plots.plot_emission_weights_by_subject(
    #     _weights_df_sel,
    #     K=K,
    # )

    _fig_summary = model_plots.emission_weights_summary_boxplot(weights_df)
    # _lr_df = build_regressor_lr_df(
    #     {s: arrays_store[s] for s in selected},
    #     lapse_max=model_cfg.lapse_max,
    # )
    # _ax_lr = (
    #     None
    #     if _lr_df.empty
    #     else plot_emission_regressor_lr_boxplot(
    #         _lr_df,
    #         feature_order=list(model_cfg.emission_cols) if model_cfg.emission_cols else None,
    #         title="Regressor LR chi2",
    #     )
    # )
    # _fig_lr = None if _ax_lr is None else _ax_lr.figure
    def _plot_one_hot_family(plotter, *, mcdr_mode):
        _fig, _ax = plt.subplots(
            figsize=fig_size(2, 1),
            constrained_layout=True,
        )
        _plotted_fig = plotter(_weights_df_sel, mcdr_mode=mcdr_mode, ax=_ax)
        if _plotted_fig is None:
            plt.close(_fig)
            return None, None
        return _plotted_fig, _ax

    _fig_stim_hot, _ax_stim_hot = _plot_one_hot_family(
        plot_stim_hot_weights,
        mcdr_mode=_mcdr_mode,
    )
    _fig_choice_lag, _ax_choice_lag = _plot_one_hot_family(
        plot_choice_lag_weights,
        mcdr_mode=_mcdr_mode,
    )
    if _ax_stim_hot is not None and _ax_choice_lag is not None:
        _ylims = np.asarray(
            [_ax_stim_hot.get_ylim(), _ax_choice_lag.get_ylim()],
            dtype=float,
        )
        _lower = float(np.nanmin(_ylims[:, 0]))
        _upper = float(np.nanmax(_ylims[:, 1]))
        if np.isfinite(_lower) and np.isfinite(_upper):
            if _lower == _upper:
                _pad = 1.0 if _lower == 0 else abs(_lower) * 0.05
                _lower -= _pad
                _upper += _pad
            _ax_stim_hot.set_ylim(_lower, _upper)
            _ax_choice_lag.set_ylim(_lower, _upper)
    # _fig_choice_lag = model_plots.emission_weights_summary_boxplot(weights_df.filter(pl.col("feature").str.contains("choice")))
    _fig_bias_hot = plot_bias_hot_weights(_weights_df_sel)
    # _fig_bias_hot = model_plots.emission_weights_summary_boxplot(weights_df.filter(pl.col("feature").str.contains("choice").not_(), pl.col("feature").str.contains("stim").not_()))
    _fig_lapses = model_plots.lapse_rates_boxplot(views=views_sel, K=K, collapse_history_choices=True)
    _fig_seq = plot_sequence_feature_weights(_weights_df_sel)
    # _items = [mo.md("#### By subject"), _fig_by_subject]
    _items = []
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
                [_fig_summary, save_plot(_fig_summary.figure, "emission weights", stem="emission_weights")],
                align="center",
            )
        )
    # if _fig_lr is not None:
    #     _summary_cards.append(
    #         mo.vstack(
    #             [_fig_lr, save_plot(_fig_lr, "regressor LR chi2", stem="regressor_lr_chi2")],
    #             align="center",
    #         )
    #     )
    _summary_cards.append(mo.vstack([_fig_lapses, save_plot(_fig_lapses, "lapse rates", stem="lapse_rates")], align="center"))
    _summary_panel = mo.hstack(_summary_cards, align="start", justify="start", gap=1.0)
    _items.extend([mo.md("#### Summary"), _summary_panel])

    _one_hot_figs = []
    if _fig_stim_hot is not None:
        _one_hot_figs.append(
            mo.vstack(
                [
                    _fig_stim_hot,
                    save_plot(_fig_stim_hot.figure, "stim one-hot", stem="stim_one_hot"),
                ],
                align="center",
            )
        )
    if _fig_choice_lag is not None:
        _one_hot_figs.append(
            mo.vstack(
                [
                    _fig_choice_lag,
                    save_plot(_fig_choice_lag.figure, "choice lag one-hot", stem="prev_choice_one_hot"),
                ],
                align="center",
            )
        )
    if _fig_bias_hot is not None:
        _one_hot_figs.append(
            mo.vstack(
                [
                    _fig_bias_hot,
                    save_plot(_fig_bias_hot.figure, "bias hot", stem="bias_one_hot"),
                ],
                align="center",
            )
        )
    if _one_hot_figs:
        _one_hot_header_items = [mo.md("#### One-hot families")]
        if task_name == "MCDR":
            _one_hot_header_items.append(ui_mcdr_one_hot_mode)
        _items.extend(
            [
                mo.vstack(_one_hot_header_items, align="start"),
                mo.hstack(
                    _one_hot_figs,
                    align="start",
                    justify="start",
                    gap=1.0,
                ),
            ]
        )

    mo.vstack(_items, align="center")
    # _fig_summary
    # _fig_choice_lag
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy plots
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the total-evidence plots, the x-axis is the fitted emission evidence for the correct class on each trial.

    **2-choice task**

    As we take a class to be the reference, the fitted logits are:

    $$
    (0, \eta_R),
    $$

    so the baseline-class logit is fixed to zero. Therefore the total evidence for the correct choice is just the signed fitted logit:

    $$
    E_{\mathrm{tot}} = \eta_{\mathrm{correct}}
    = \log \frac{p(\mathrm{correct})}{p(\mathrm{other})}.
    $$

    **3-choice task**

    In general, if one class is taken as the reference, the fitted evidence for the correct choice is

    $$
    E_{\mathrm{tot}} = \eta_{\mathrm{correct}} - \log \sum_{j \neq \mathrm{correct}} e^{\eta_j}
    = \log \frac{p(\mathrm{correct})}{1 - p(\mathrm{correct})}.
    $$

    We take the center choice to be the reference class. The saved explicit rows are the left and right logits, so the reconstructed logits are

    $$
    (\eta_L, 0, \eta_R).
    $$

    Therefore, for MCDR in this notebook:

    $$
    E_{\mathrm{tot}} = \eta_{\mathrm{correct}} - \log \left(\sum_{j \neq \mathrm{correct}} e^{\eta_j}\right),
    \qquad
    (\eta_L, \eta_C, \eta_R) = (\eta_L, 0, \eta_R).
    $$
    """)
    return


@app.cell
def _(adapter, mo, task_name, trial_df, views):
    _view_feature_names = []
    for _view in views.values():
        for _feat in list(getattr(_view, "feat_names", []) or []):
            _feat = str(_feat)
            if _feat not in _view_feature_names:
                _view_feature_names.append(_feat)

    _available_cols = set(trial_df.columns) | set(_view_feature_names)
    _choice_lag_cols = [col for col in _view_feature_names if col.startswith("choice_lag_")]
    if not _choice_lag_cols:
        _choice_lag_cols = [col for col in adapter.choice_lag_cols(trial_df) if col in _available_cols]

    is_2adc = task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}
    if task_name == "MCDR" or is_2adc:
        regressor_options = [col for col in ["choice_lag_param"] if col in _available_cols]
    else:
        regressor_options = [col for col in ["at_choice_param"] if col in _available_cols]
    if _choice_lag_cols:
        regressor_options.append("choice_lag_one_hot_sum")

    _net_impact_options = []
    for _col in list(getattr(adapter, "emission_cols", []) or []) + _view_feature_names:
        _col = str(_col)
        if _col in _available_cols and _col not in _net_impact_options:
            _net_impact_options.append(_col)
    if _choice_lag_cols and "choice_lag_one_hot_sum" not in _net_impact_options:
        _net_impact_options.append("choice_lag_one_hot_sum")

    if not regressor_options:
        ui_accuracy_binning = None
        ui_accuracy_n_quantiles = None
        ui_accuracy_x_axis = None
        ui_accuracy_regressor = None
        ui_fit_lapse_by_subject = None
        ui_fit_lapse_logistic = None
        ui_share_lapse_logistic_core = None
        ui_show_lapses_in_legend = None
    else:
        ui_accuracy_binning = mo.ui.checkbox(value=False, label="Enable")
        if task_name == "MCDR":
            _accuracy_x_options = ["ttype", "stimd", "delay", "total_evidence"]
        elif is_2adc:
            _accuracy_x_options = ["delay", "weighted_stimulus", "total_evidence"]
        else:
            _accuracy_x_options = ["ILD", "weighted_stimulus", "total_evidence"]
        ui_accuracy_x_axis = mo.ui.dropdown(
            options=_accuracy_x_options,
            value=_accuracy_x_options[0],
            label="Accuracy x-axis",
        )
        ui_accuracy_regressor = mo.ui.dropdown(
            options=regressor_options,
            value=regressor_options[0],
            label="Regressor",
        )
        ui_accuracy_n_quantiles = mo.ui.slider(
            start=2,
            stop=8,
            step=2,
            value=4,
            label="Quantile bins",
        )
        ui_fit_lapse_logistic = mo.ui.checkbox(value=False, label="Fit lapse logistic")
        ui_fit_lapse_by_subject = mo.ui.checkbox(value=True, label="Fit by animal")
        ui_share_lapse_logistic_core = mo.ui.checkbox(value=False, label="Fix bias/slope across quartiles")
        ui_show_lapses_in_legend = mo.ui.checkbox(value=True, label="Show lapses in legend")

    if not _net_impact_options:
        ui_net_impact_x_axis = None
        ui_net_impact_y_axis = None
    else:
        _x_default = _net_impact_options[0]
        _y_default = next(
            (_col for _col in _net_impact_options if _col != _x_default),
            _x_default,
        )
        ui_net_impact_x_axis = mo.ui.dropdown(
            options=_net_impact_options,
            value=_x_default,
            label="Magnitude regressor",
        )
        ui_net_impact_y_axis = mo.ui.dropdown(
            options=_net_impact_options,
            value=_y_default,
            label="Impact regressor",
        )
    return (
        regressor_options,
        ui_accuracy_n_quantiles,
        ui_accuracy_regressor,
        ui_accuracy_x_axis,
        ui_fit_lapse_by_subject,
        ui_fit_lapse_logistic,
        ui_net_impact_x_axis,
        ui_net_impact_y_axis,
        ui_share_lapse_logistic_core,
        ui_show_lapses_in_legend,
    )


@app.cell
def _(mo, pl, selected, trial_df):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))

    trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    mo.stop(trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))
    return (trial_df_sel,)


@app.cell
def _(
    adapter,
    add_choice_lag_summary_regressor,
    np,
    pl,
    prepare_predictions_df,
    task_name,
    trial_df_sel,
    views_sel,
):
    _choice_lag_cols = []
    for _view in views_sel.values():
        for _feat in list(getattr(_view, "feat_names", []) or []):
            _feat = str(_feat)
            if _feat.startswith("choice_lag_") and _feat not in _choice_lag_cols:
                _choice_lag_cols.append(_feat)

    if not _choice_lag_cols:
        _choice_lag_cols = adapter.choice_lag_cols(trial_df_sel)

    plot_df_all = prepare_predictions_df(task_name, trial_df_sel)
    plot_df_all = add_choice_lag_summary_regressor(
        plot_df_all,
        choice_lag_cols=_choice_lag_cols,
    )
    _weighted_col = "choice_lag_glm_weighted_sum"
    if _choice_lag_cols:
        _weighted_chunks = []
        for _subject in plot_df_all["subject"].unique().to_list():
            _subject_df = plot_df_all.filter(pl.col("subject") == _subject)
            _view = views_sel.get(str(_subject), views_sel.get(_subject))
            if _view is None:
                _weighted_chunks.append(_subject_df.with_columns(pl.lit(None).cast(pl.Float64).alias(_weighted_col)))
                continue

            _feat_names = [str(_feat) for _feat in list(getattr(_view, "feat_names", []) or [])]
            _weights = np.asarray(getattr(_view, "emission_weights", []), dtype=float)
            if _weights.ndim == 3:
                _weights = _weights[0, 0]
            elif _weights.ndim == 2:
                _weights = _weights[0]
            if _weights.ndim != 1 or len(_feat_names) != _weights.shape[0]:
                _weighted_chunks.append(_subject_df.with_columns(pl.lit(None).cast(pl.Float64).alias(_weighted_col)))
                continue

            _terms = []
            for _col in _choice_lag_cols:
                if _col not in _subject_df.columns or _col not in _feat_names:
                    continue
                _terms.append(
                    pl.col(_col).cast(pl.Float64, strict=False).fill_null(0.0)
                    * float(_weights[_feat_names.index(_col)])
                )
            if not _terms:
                _weighted_chunks.append(_subject_df.with_columns(pl.lit(None).cast(pl.Float64).alias(_weighted_col)))
                continue
            _weighted_expr = _terms[0]
            for _term in _terms[1:]:
                _weighted_expr = _weighted_expr + _term
            _weighted_chunks.append(_subject_df.with_columns(_weighted_expr.alias(_weighted_col)))
        if _weighted_chunks:
            plot_df_all = pl.concat(_weighted_chunks, how="vertical")
    return (plot_df_all,)


@app.cell
def _(
    mo,
    ui_accuracy_n_quantiles,
    ui_accuracy_regressor,
    ui_accuracy_x_axis,
    ui_fit_lapse_by_subject,
    ui_fit_lapse_logistic,
    ui_share_lapse_logistic_core,
    ui_show_lapses_in_legend,
):
    mo.hstack(
        [
            ui_accuracy_regressor,
            ui_accuracy_x_axis,
            ui_accuracy_n_quantiles,
            ui_fit_lapse_logistic,
            ui_fit_lapse_by_subject,
            ui_share_lapse_logistic_core,
            ui_show_lapses_in_legend,
        ]
    )
    return


@app.cell
def _(
    fig_size,
    is_2afc,
    mo,
    plot_df_all,
    plots,
    regressor_options,
    save_plot,
    ui_accuracy_regressor,
    views_sel,
):
    _perf_kwargs = {"views": views_sel} if is_2afc else {}

    _fig_all, _ = plots.plot_categorical_performance_all(
        plot_df_all,
        "glm",
        background_style="model",
        **_perf_kwargs,
        figsize=fig_size(3, 1) if not is_2afc else fig_size(2, 1),
    )
    _fig_all_list = (
        list(_fig_all)
        if isinstance(_fig_all, (list, tuple))
        else [_fig_all]
    )

    _choice_history_regressor = plots.pick_choice_history_regressor(regressor_options)
    _regressor_for_right = _choice_history_regressor or ui_accuracy_regressor.value
    _regressor_label = plots.display_regressor_name(_regressor_for_right)

    _fig_regressor = plots.plot_repeat_by_regressor_simple(
        plot_df_all,
        regressor_col=_regressor_for_right,
        views=views_sel,
        title=None,
        figsize = fig_size(2,1)
    )

    # mo.stop(_fig_regressor is None, mo.md(f"No p(repeat) plot available for {_regressor_label}."))

    mo.hstack(
        [
            mo.vstack(
                [
                    _item
                    for _fig_idx, _fig in enumerate(_fig_all_list, start=1)
                    for _item in (
                        _fig,
                        save_plot(
                            _fig,
                            (
                                f"overall psychometric {_fig_idx}"
                                if len(_fig_all_list) > 1
                                else "overall psychometric"
                            ),
                            stem=(
                                f"accuracy_overall_{_fig_idx}"
                                if len(_fig_all_list) > 1
                                else "accuracy_overall"
                            ),
                        ),
                    )
                ],
                align="center",
            ),
            mo.vstack(
                [
                    _fig_regressor,
                    save_plot(
                        _fig_regressor,
                        f"p(repeat) by {_regressor_label}",
                        stem=f"psychometric_regressor_{_regressor_for_right}",
                    ),
                ],
                align="center",
            ),
        ]
    )
    return


@app.cell
def _(adapter, fig_size, mo, plot_df_all, plots, plt, save_plot, views_sel):
    _evidence_figsize = fig_size(3, 1)
    _fig_total_evidence, _ax_total_evidence = plt.subplots(
        1,
        1,
        figsize=_evidence_figsize,
        layout="constrained",
    )
    plots.plot_accuracy_by_total_evidence(
        plot_df_all,
        adapter=adapter,
        views=views_sel,
        ax=_ax_total_evidence,
        figsize=_evidence_figsize,
    )
    _ax_total_evidence.set_xlabel("Fitted Evidence")

    _fig_repeat_evidence, _ax_repeat_evidence = plt.subplots(
        1,
        1,
        figsize=_evidence_figsize,
        layout="constrained",
    )
    plots.plot_repeat_by_repeat_evidence(
        plot_df_all,
        views=views_sel,
        ax=_ax_repeat_evidence,
        figsize=_evidence_figsize,
    )
    _ax_repeat_evidence.set_xlabel("Rep. Evidence")

    # mo.stop(_fig_total_evidence is None, mo.md("Accuracy by fitted total evidence not available."))

    mo.hstack(
        [
            mo.vstack(
                [
                    _fig_total_evidence,
                    save_plot(
                        _fig_total_evidence,
                        "accuracy by fitted total evidence",
                        stem="accuracy_total_evidence",
                    ),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    _fig_repeat_evidence,
                    save_plot(
                        _fig_repeat_evidence,
                        "repeat probability by fitted repeat evidence",
                        stem="repeat_probability_repeat_evidence",
                    ),
                ],
                align="center",
            ),
        ], 
    )
    return


@app.cell
def _(plot_df_all, ui_accuracy_regressor):
    ui_accuracy_regressor.value
    plot_df_all
    return


@app.cell
def _(
    adapter,
    fig_size,
    mo,
    plot_df_all,
    plots,
    plt,
    save_plot,
    ui_accuracy_n_quantiles,
    ui_accuracy_regressor,
    ui_accuracy_x_axis,
    ui_fit_lapse_by_subject,
    ui_fit_lapse_logistic,
    ui_share_lapse_logistic_core,
    ui_show_lapses_in_legend,
    views_sel,
):
    _selected_regressor_label = plots.display_regressor_name(ui_accuracy_regressor.value)
    _panel_size = fig_size(2, 1)

    _fig_binned = plots.plot_binned_accuracy_figure(
        plot_df_all,
        regressor_col=ui_accuracy_regressor.value,
        x_axis=ui_accuracy_x_axis.value,
        adapter=adapter,
        views=views_sel,
        n_bins=int(ui_accuracy_n_quantiles.value),
        fit_lapse_logistic=bool(ui_fit_lapse_logistic.value),
        fit_lapse_by_subject=bool(ui_fit_lapse_by_subject.value),
        share_lapse_logistic_core=bool(ui_share_lapse_logistic_core.value),
        show_lapses_in_legend=bool(ui_show_lapses_in_legend.value),
        print_lapse_fits=not bool(ui_show_lapses_in_legend.value),
        figsize=_panel_size,
    )
    mo.stop(_fig_binned is None, mo.md(f"No binned accuracy plot available for {_selected_regressor_label}."))
    _fig_binned_base = _fig_binned[0] if isinstance(_fig_binned, tuple) else _fig_binned
    _right_figsize = tuple(float(_size) for _size in _fig_binned_base.get_size_inches())

    _plot_df_cols = set(getattr(plot_df_all, "columns", []))
    _secondary_regressor = "choice_lag_glm_weighted_sum" if "choice_lag_glm_weighted_sum" in _plot_df_cols else None
    mo.stop(_secondary_regressor is None, mo.md("No GLM-weighted choice-history regressor available."))

    _secondary_regressor_label = plots.display_regressor_name(_secondary_regressor)
    _fig_secondary_right_base, (_ax_secondary_right, _ax_secondary_right_legend) = plt.subplots(
        1,
        2,
        figsize=_right_figsize,
        gridspec_kw={"width_ratios": [1.0, 0.1], "wspace": 0.02},
    )
    _fig_secondary_right = plots.plot_right_by_regressor(
        plot_df_all,
        regressor_col=_secondary_regressor,
        title=None,
        n_bins=9,
        ax=_ax_secondary_right,
        legend_ax=_ax_secondary_right_legend,
    )

    mo.stop(_fig_secondary_right is None, mo.md(f"No p(right) plot available for {_secondary_regressor_label}."))

    mo.vstack(
        [
            mo.hstack(
                [
                    ui_accuracy_regressor,
                    ui_accuracy_x_axis,
                    ui_accuracy_n_quantiles,
                    ui_fit_lapse_logistic,
                    ui_fit_lapse_by_subject,
                    ui_share_lapse_logistic_core,
                    ui_show_lapses_in_legend,
                ]
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            _fig_binned,
                            save_plot(
                                _fig_binned[0],
                                f"binned accuracy {_selected_regressor_label}",
                                stem=f"accuracy_binned_{ui_accuracy_regressor.value}",
                            ),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            _fig_secondary_right_base,
                            save_plot(
                                _fig_secondary_right_base,
                                f"p(right) by {_secondary_regressor_label}",
                                stem=f"psychometric_binned_{_secondary_regressor}",
                            ),
                        ],
                        align="center",
                    ),
                ],
            ),
        ],
        align="center"
    )
    return


@app.cell
def _(df_all, mo):
    ui_filter_subject = mo.ui.multiselect(options = df_all["subject"].unique(), value = [df_all["subject"].unique()[1]])
    ui_filter_subject
    return (ui_filter_subject,)


@app.cell
def _(mo, pl, plot_df_all, ui_filter_subject):
    df_filter_subject = plot_df_all.filter(pl.col("subject") == ui_filter_subject.value[0]) 
    ui_filter_session = mo.ui.dropdown(options = df_filter_subject["session"].unique())
    ui_filter_session
    return df_filter_subject, ui_filter_session


@app.cell
def _(session_df):
    session_df
    return


@app.cell
def _(mo):
    ui_autocorr_apply_correction = mo.ui.checkbox(
        value=True,
        label="Apply cross-session correction",
    )
    ui_show_glm_autocorr = mo.ui.run_button(
        label="Run fitted GLM autocorrelogram simulations",
    )
    ui_glm_autocorr_n_simulations = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=5,
        label="GLM simulations",
    )
    ui_glm_autocorr_recursive = mo.ui.checkbox(
        value=False,
        label="Recursive GLM simulation",
    )
    mo.hstack([
        ui_autocorr_apply_correction,
        ui_show_glm_autocorr,
        ui_glm_autocorr_n_simulations,
        ui_glm_autocorr_recursive,
    ])
    return (
        ui_autocorr_apply_correction,
        ui_glm_autocorr_n_simulations,
        ui_glm_autocorr_recursive,
        ui_show_glm_autocorr,
    )


@app.cell
def _(
    adapter,
    arrays_store,
    df_all,
    df_filter_subject,
    fig_size,
    mo,
    model_cfg,
    pl,
    plot_df_all,
    plt,
    save_plot,
    ui_autocorr_apply_correction,
    ui_filter_session,
    ui_glm_autocorr_n_simulations,
    ui_glm_autocorr_recursive,
    ui_show_glm_autocorr,
):
    from src.process.common import (
        prepare_corrected_behavior_autocorrelograms,
        prepare_glm_simulated_corrected_behavior_autocorrelograms,
        prepare_session_accuracy_repetition_timescale,
    )
    from src.plots.common import (
        plot_corrected_behavior_autocorrelograms,
        plot_session_accuracy_repetition_timescale,
    )
    session_df = df_filter_subject.filter(pl.col("session") == ui_filter_session.value)
    _trial_col = "trial_idx" if "trial_idx" in plot_df_all.columns else ("trial" if "trial" in plot_df_all.columns else None)
    _autocorr_sort_cols = ["subject", "session"] + ([_trial_col] if _trial_col is not None else [])
    _autocorr_df = plot_df_all.sort(_autocorr_sort_cols)
    _session_trial_col = (
        _trial_col
        if _trial_col is not None and _trial_col in session_df.columns
        else ("trial_idx" if "trial_idx" in session_df.columns else ("trial" if "trial" in session_df.columns else None))
    )
    prepared_session_timescale = prepare_session_accuracy_repetition_timescale(
        session_df,
        choice_col="response",
        outcome_col="performance",
        trial_index_col=_session_trial_col,
        running_window=20,
        max_lag=50,
    )
    prepared_corrected_autocorr = prepare_corrected_behavior_autocorrelograms(
        _autocorr_df,
        subject_col="subject",
        session_col="session",
        choice_col="response",
        outcome_col="performance",
        trial_index_col=_trial_col,
        max_lag=50,
        min_cross_pairs=20,
        max_cross_pairs=80,
        seed=0,
    )
    if ui_show_glm_autocorr.value:
        prepared_glm_autocorr = prepare_glm_simulated_corrected_behavior_autocorrelograms(
            df_all,
            arrays_store,
            adapter=adapter,
            subject_col="subject",
            session_col=adapter.session_col,
            trial_index_col=adapter.behavioral_cols["trial"],
            correct_label_col=adapter.behavioral_cols["stimulus"],
            tau=model_cfg.tau,
            emission_cols=list(model_cfg.emission_cols),
            recursive=bool(ui_glm_autocorr_recursive.value),
            n_simulations=int(ui_glm_autocorr_n_simulations.value),
            max_lag=50,
            min_cross_pairs=20,
            max_cross_pairs=80,
            seed=1,
            summary_only=True,
        )
    else:
        prepared_glm_autocorr = None

    fig_session_timescale, _ = plot_session_accuracy_repetition_timescale(prepared_session_timescale)
    _glm_autocorr = (
        prepared_glm_autocorr["autocorr"]
        if prepared_glm_autocorr is not None
        else None
    )
    def _style_autocorr_axis(_ax):
        _ax.set_ylim(-0.05, 0.2)
        _ax.set_title("")

    if ui_autocorr_apply_correction.value:
        _fig_choice_autocorr, _ax_choice_autocorr = plt.subplots(figsize=fig_size(2, 1))
        _fig_choice_autocorr, _ = plot_corrected_behavior_autocorrelograms(
            prepared_corrected_autocorr,
            axes=[_ax_choice_autocorr],
            glm_autocorr=_glm_autocorr,
            signals=("Outcome",),
            figsize=fig_size(2, 1),
        )
        _style_autocorr_axis(_ax_choice_autocorr)
        _fig_repeat_autocorr, _ax_repeat_autocorr = plt.subplots(figsize=fig_size(2, 1))
        _fig_repeat_autocorr, _ = plot_corrected_behavior_autocorrelograms(
            prepared_corrected_autocorr,
            axes=[_ax_repeat_autocorr],
            glm_autocorr=_glm_autocorr,
            signals=("Repetition",),
            figsize=fig_size(2, 1),
        )
        _style_autocorr_axis(_ax_repeat_autocorr)
        _autocorr_display = mo.hstack(
            [
                mo.vstack(
                    [
                        _fig_choice_autocorr,
                        save_plot(_fig_choice_autocorr, "corrected choice autocorrelogram", stem="corrected_choice_autocorrelogram"),
                    ],
                    align="center",
                ),
                mo.vstack(
                    [
                        _fig_repeat_autocorr,
                        save_plot(_fig_repeat_autocorr, "corrected repeat autocorrelogram", stem="corrected_repeat_autocorrelogram"),
                    ],
                    align="center",
                ),
            ],
            align="center",
        )
    else:
        _fig_raw_choice_autocorr, _ax_raw_choice_autocorr = plt.subplots(figsize=fig_size(2, 1))
        _fig_raw_choice_autocorr, _ = plot_corrected_behavior_autocorrelograms(
            prepared_corrected_autocorr,
            axes=[_ax_raw_choice_autocorr],
            glm_autocorr=_glm_autocorr,
            autocorr_col="raw_autocorr",
            sem_col="raw_autocorr_sem",
            data_label="Data (raw)",
            model_label="Fitted GLM (raw)",
            ylabel="Raw autocorrelation",
            signals=("Outcome",),
            figsize=fig_size(2, 1),
        )
        _style_autocorr_axis(_ax_raw_choice_autocorr)
        _fig_raw_repeat_autocorr, _ax_raw_repeat_autocorr = plt.subplots(figsize=fig_size(2, 1))
        _fig_raw_repeat_autocorr, _ = plot_corrected_behavior_autocorrelograms(
            prepared_corrected_autocorr,
            axes=[_ax_raw_repeat_autocorr],
            glm_autocorr=_glm_autocorr,
            autocorr_col="raw_autocorr",
            sem_col="raw_autocorr_sem",
            data_label="Data (raw)",
            model_label="Fitted GLM (raw)",
            ylabel="Raw autocorrelation",
            signals=("Repetition",),
            figsize=fig_size(2, 1),
        )
        _style_autocorr_axis(_ax_raw_repeat_autocorr)
        _fig_correction_choice_autocorr, _ax_correction_choice_autocorr = plt.subplots(figsize=fig_size(2, 1))
        _fig_correction_choice_autocorr, _ = plot_corrected_behavior_autocorrelograms(
            prepared_corrected_autocorr,
            axes=[_ax_correction_choice_autocorr],
            glm_autocorr=_glm_autocorr,
            autocorr_col="crosscorr",
            sem_col="crosscorr_sem",
            data_label="Correction",
            model_label="Fitted GLM correction",
            ylabel="Cross-session correction",
            signals=("Outcome",),
            figsize=fig_size(2, 1),
        )
        _style_autocorr_axis(_ax_correction_choice_autocorr)
        _fig_correction_repeat_autocorr, _ax_correction_repeat_autocorr = plt.subplots(figsize=fig_size(2, 1))
        _fig_correction_repeat_autocorr, _ = plot_corrected_behavior_autocorrelograms(
            prepared_corrected_autocorr,
            axes=[_ax_correction_repeat_autocorr],
            glm_autocorr=_glm_autocorr,
            autocorr_col="crosscorr",
            sem_col="crosscorr_sem",
            data_label="Correction",
            model_label="Fitted GLM correction",
            ylabel="Cross-session correction",
            signals=("Repetition",),
            figsize=fig_size(2, 1),
        )
        _style_autocorr_axis(_ax_correction_repeat_autocorr)
        _autocorr_display = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack([_fig_raw_choice_autocorr, save_plot(_fig_raw_choice_autocorr, "raw choice autocorrelogram", stem="raw_choice_autocorrelogram")], align="center"),
                        mo.vstack([_fig_raw_repeat_autocorr, save_plot(_fig_raw_repeat_autocorr, "raw repeat autocorrelogram", stem="raw_repeat_autocorrelogram")], align="center"),
                    ],
                    align="center",
                ),
                mo.hstack(
                    [
                        mo.vstack([_fig_correction_choice_autocorr, save_plot(_fig_correction_choice_autocorr, "choice correction autocorrelogram", stem="choice_correction_autocorrelogram")], align="center"),
                        mo.vstack([_fig_correction_repeat_autocorr, save_plot(_fig_correction_repeat_autocorr, "repeat correction autocorrelogram", stem="repeat_correction_autocorrelogram")], align="center"),
                    ],
                    align="center",
                ),
            ],
            align="center",
        )
    mo.vstack([
        fig_session_timescale,
        _autocorr_display,
    ])
    return (session_df,)


@app.cell
def _(mo, np, pd, pl, plot_df_all, plt, save_plot, sns):
    from src.process.common import binary_autocorrelation

    _df = plot_df_all.to_pandas() if isinstance(plot_df_all, pl.DataFrame) else pd.DataFrame(plot_df_all).copy()
    _stimulus_col = next(
        (_col for _col in ("stimulus", "stim", "Side", "ILD", "weighted_stimulus") if _col in _df.columns),
        None,
    )
    print(plot_df_all["subject"].unique())
    mo.stop(_stimulus_col is None, mo.md("No stimulus column available for autocorrelogram."))

    _sort_cols = [_col for _col in ("subject", "session", "trial") if _col in _df.columns]
    if _sort_cols:
        _df = _df.sort_values(_sort_cols, kind="mergesort")

    _group_cols = [_col for _col in ("subject", "session") if _col in _df.columns]
    _groups = _df.groupby(_group_cols, observed=True) if _group_cols else [(None, _df)]

    _session_autocorr_rows = []
    for _group_key, _session_df in _groups:
        _ac = binary_autocorrelation(_session_df[_stimulus_col], max_lag=50)
        if _ac.empty:
            continue
        if _group_cols:
            _group_key = _group_key if isinstance(_group_key, tuple) else (_group_key,)
            for _col, _value in zip(_group_cols, _group_key, strict=False):
                _ac[_col] = _value
        _session_autocorr_rows.append(_ac)

    _session_autocorr = (
        pd.concat(_session_autocorr_rows, ignore_index=True)
        if _session_autocorr_rows
        else pd.DataFrame(columns=["lag", "autocorr", "n"])
    )
    if "subject" in _session_autocorr.columns:
        _subject_autocorr = (
            _session_autocorr.groupby(["subject", "lag"], observed=True)["autocorr"]
            .mean()
            .reset_index()
        )
        _stimulus_autocorr = (
            _subject_autocorr.groupby("lag", observed=True)
            .agg(
                autocorr=("autocorr", "mean"),
                autocorr_std=("autocorr", "std"),
                n_subjects=("subject", "count"),
            )
            .reset_index()
        )
        _stimulus_autocorr["autocorr_sem"] = (
            _stimulus_autocorr["autocorr_std"].fillna(0.0)
            / np.sqrt(_stimulus_autocorr["n_subjects"].clip(lower=1))
        )
    else:
        _stimulus_autocorr = _session_autocorr[["lag", "autocorr"]].copy()
        _stimulus_autocorr["autocorr_sem"] = 0.0

    _fig_stimulus_autocorr, _ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    _fig_stimulus_autocorr_sns, _ax_sns = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    if _stimulus_autocorr.empty or not {"lag", "autocorr"}.issubset(_stimulus_autocorr.columns):
        for _empty_ax in (_ax, _ax_sns):
            _empty_ax.text(0.5, 0.5, "No valid stimulus autocorrelation data", ha="center", va="center", transform=_empty_ax.transAxes)
            _empty_ax.axis("off")
    else:
        _stimulus_autocorr = _stimulus_autocorr.sort_values("lag")
        _ax.errorbar(
            _stimulus_autocorr["lag"].to_numpy(dtype=float),
            _stimulus_autocorr["autocorr"].to_numpy(dtype=float),
            yerr=_stimulus_autocorr["autocorr_sem"].to_numpy(dtype=float),
            fmt="o",
            ms=4,
            color="#4c78a8",
            ecolor="#4c78a8",
            elinewidth=1.0,
            capsize=2,
        )
        _ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
        _ax.set_xlabel("Lag")
        _ax.set_ylabel("Stimulus autocorrelation")
        _ax.set_title("Stimulus sequence")

        _lineplot_df = _subject_autocorr if "subject" in _session_autocorr.columns else _stimulus_autocorr
        sns.lineplot(
            data=_lineplot_df,
            x="lag",
            y="autocorr",
            errorbar="se" if "subject" in _lineplot_df.columns else None,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color="#4c78a8",
            ax=_ax_sns,
        )
        _ax_sns.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
        _ax_sns.set_xlabel("Lag")
        _ax_sns.set_ylabel("Stimulus autocorrelation")
        _ax_sns.set_title("Stimulus sequence")
    mo.hstack([
        mo.vstack([
            _fig_stimulus_autocorr,
            save_plot(_fig_stimulus_autocorr, "stimulus autocorrelogram", stem="stimulus_autocorrelogram"),
        ], align="center"),
        mo.vstack([
            _fig_stimulus_autocorr_sns,
            save_plot(
                _fig_stimulus_autocorr_sns,
                "stimulus autocorrelogram seaborn",
                stem="stimulus_autocorrelogram_seaborn",
            ),
        ], align="center"),
    ])
    return


@app.cell
def _():
    from src.process.common import (
        build_action_trace_model_prediction_rb,
        build_action_trace_parameter_fixed_simulations,
    )
    from src.plots.common import (
        plot_action_trace_parameter_fixed_lag_match,
        plot_action_trace_parameter_fixed_rb,
        plot_action_trace_parameter_fixed_subject_scatter,
    )

    return (
        build_action_trace_model_prediction_rb,
        build_action_trace_parameter_fixed_simulations,
        plot_action_trace_parameter_fixed_lag_match,
        plot_action_trace_parameter_fixed_rb,
        plot_action_trace_parameter_fixed_subject_scatter,
    )


@app.cell
def _(mo, task_name):
    ui_glm_model_rb_max_lag = mo.ui.slider(
        start=1,
        stop=15,
        step=1,
        value=10,
        label="Full-model RB max history lag",
    )
    _supported = task_name in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}
    (
        mo.hstack([ui_glm_model_rb_max_lag], justify="center")
        if _supported
        else mo.md("Trial-level full-model repetition bias is implemented only for 2AFC and 2ADC.")
    )
    return (ui_glm_model_rb_max_lag,)


@app.cell
def _(mo, task_name):
    ui_counterfactual_n_simulations = mo.ui.slider(
        start=50,
        stop=1000,
        step=50,
        value=200,
        label="Parameter-fixed simulations",
    )
    ui_counterfactual_seed = mo.ui.number(
        start=0,
        stop=1_000_000,
        step=1,
        value=7,
        label="Seed",
    )
    ui_counterfactual_max_lag = mo.ui.slider(
        start=1,
        stop=15,
        step=1,
        value=10,
        label="Max history lag",
    )

    _controls = (
        mo.hstack(
            [
                ui_counterfactual_n_simulations,
                ui_counterfactual_max_lag,
                ui_counterfactual_seed,
            ],
            justify="center",
        )
        if task_name in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}
        else mo.md("Action-trace parameter-fixed simulations are implemented only for 2AFC and 2ADC.")
    )
    _controls
    return (
        ui_counterfactual_max_lag,
        ui_counterfactual_n_simulations,
        ui_counterfactual_seed,
    )


@app.cell
def _(
    build_action_trace_model_prediction_rb,
    mo,
    plot_action_trace_parameter_fixed_lag_match,
    plot_action_trace_parameter_fixed_rb,
    plot_df_all,
    save_plot,
    task_name,
    ui_glm_model_rb_max_lag,
):
    mo.stop(
        task_name not in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"},
    )

    _summary, _lag_summary, _subject_scatter, _meta = build_action_trace_model_prediction_rb(
        plot_df_all,
        task_name=task_name,
        max_history_lag=int(ui_glm_model_rb_max_lag.value),
    )
    mo.stop(
        _summary.empty,
        mo.md("No trial-level model repetition-bias result; check pR/p_pred and previous choices."),
    )
    _model_scenarios = ["Data", "Full fitted"]
    _summary = _summary[_summary["scenario"].isin(_model_scenarios)].copy()
    _lag_summary = _lag_summary[_lag_summary["scenario"].isin(_model_scenarios)].copy()

    _fig_model_rb, _ax_model_rb = plot_action_trace_parameter_fixed_rb(
        _summary,
        _meta,
    )
    _fig_model_lag, _ax_model_lag = plot_action_trace_parameter_fixed_lag_match(
        _lag_summary,
        _meta,
    )
    _ax_model_lag.set_title("")

    mo.vstack(
        [
            mo.md("#### Trial-level full-model repetition bias: Data vs Full fitted"),
            _fig_model_rb,
            save_plot(
                _fig_model_rb,
                "glm trial-level full-model repetition bias",
                stem="glm_trial_model_rb",
            ),
            _fig_model_lag,
            save_plot(
                _fig_model_lag,
                "glm trial-level full-model lag match",
                stem="glm_trial_model_lag_match",
            ),
            mo.md(
                "Full fitted uses each trial's model P(right), compares it with the same animal's empirical previous choice, "
                "and shows Data vs Full fitted by stimulus/evidence and by history lag. No refit or choice simulation is run."
            ),
        ],
        align="center",
    )
    # _fig_model_rb
    # glm_model_summary = _summary
    return


@app.cell
def _(mo, ui_accuracy_n_quantiles):
    run_simulations = mo.ui.run_button(
        kind="neutral",
        label="Run parameter-fixed simulations from A_t-binned psychometrics",
    )
    mo.hstack([ui_accuracy_n_quantiles, run_simulations], justify="center")
    return (run_simulations,)


@app.cell
def _(
    build_action_trace_parameter_fixed_simulations,
    mo,
    plot_action_trace_parameter_fixed_lag_match,
    plot_action_trace_parameter_fixed_rb,
    plot_action_trace_parameter_fixed_subject_scatter,
    plot_df_all,
    run_simulations,
    save_plot,
    task_name,
    ui_accuracy_n_quantiles,
    ui_accuracy_regressor,
    ui_counterfactual_max_lag,
    ui_counterfactual_n_simulations,
    ui_counterfactual_seed,
):
    mo.stop(
        ((task_name not in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}) or (not run_simulations.value)),
    )

    parameter_fixed_summary, parameter_fixed_lag_summary, parameter_fixed_subject_scatter, parameter_fixed_fit_table, parameter_fixed_meta = build_action_trace_parameter_fixed_simulations(
        plot_df_all,
        task_name=task_name,
        regressor_col=ui_accuracy_regressor.value,
        n_bins=int(ui_accuracy_n_quantiles.value),
        n_simulations=int(ui_counterfactual_n_simulations.value),
        max_history_lag=int(ui_counterfactual_max_lag.value),
        seed=int(ui_counterfactual_seed.value),
    )
    mo.stop(
        parameter_fixed_summary.empty,
        mo.md("No parameter-fixed repetition-bias result; check that each Action-trace bin has enough psychometric x levels."),
    )

    _fig_simulations, _ax_simulations = plot_action_trace_parameter_fixed_rb(
        parameter_fixed_summary,
        parameter_fixed_meta,
    )
    _fig_lag_match, _ax_lag_match = plot_action_trace_parameter_fixed_lag_match(
        parameter_fixed_lag_summary,
        parameter_fixed_meta,
    )
    _fig_subject_scatter, _ax_subject_scatter = plot_action_trace_parameter_fixed_subject_scatter(
        parameter_fixed_subject_scatter,
        parameter_fixed_meta,
    )
    _fit_display = parameter_fixed_fit_table.copy()
    for _col in ["slope", "bias", "lapse_left", "lapse_right"]:
        if _col in _fit_display.columns:
            _fit_display[_col] = _fit_display[_col].round(3)

    _parameter_fixed_panel = mo.vstack(
        [
            _fig_simulations,
            save_plot(
                _fig_simulations,
                "action trace psychometric parameter-fixed repetition bias",
                stem=f"action_trace_parameter_fixed_rb_{ui_accuracy_regressor.value}",
            ),
            _fig_lag_match,
            save_plot(
                _fig_lag_match,
                "action trace psychometric parameter-fixed lag match",
                stem=f"action_trace_parameter_fixed_lag_match_{ui_accuracy_regressor.value}",
            ),
            _fig_subject_scatter,
            save_plot(
                _fig_subject_scatter,
                "action trace psychometric parameter-fixed full model by animal",
                stem=f"action_trace_parameter_fixed_full_by_animal_{ui_accuracy_regressor.value}",
            ),
            mo.md(
                "Each simulated trial compares the simulated current response with the same animal's empirical previous choice. "
                "RB is computed conditional on previous choice side within animal, then sides and animals are averaged. "
                "Full fitted uses the real continuous Action-trace value to interpolate fitted parameters; "
                "Fixed bias and Fixed lapses replace only that parameter family by its across-bin average. "
                "The lag-match panel plots p(simulated response at t equals experimental response at t-L). "
                "The scatter shows one animal per point for the full fitted simulation."
            ),
            mo.ui.table(_fit_display),
        ],
        align="center",
    )
    _parameter_fixed_panel
    return parameter_fixed_meta, parameter_fixed_summary


@app.cell
def _(
    glm_model_summary,
    mo,
    np,
    parameter_fixed_meta,
    parameter_fixed_summary,
    pd,
    plt,
    save_plot,
    ui_accuracy_regressor,
):
    _parameter_summary = parameter_fixed_summary.copy()
    _glm_summary = glm_model_summary.copy()

    def _append_whole_rb(_rows, _source, *, source_scenario, label):
        if "scenario" not in _source.columns:
            return
        _sub = _source[_source["scenario"].astype(str) == source_scenario]
        if _sub.empty:
            return
        _mean = float(np.nanmean(_sub["rb_mean"].to_numpy(dtype=float)))
        _lo = float(np.nanmean(_sub["rb_lo"].to_numpy(dtype=float)))
        _hi = float(np.nanmean(_sub["rb_hi"].to_numpy(dtype=float)))
        _rows.append(
            {
                "scenario": label,
                "rb_mean": _mean,
                "rb_lo": _lo,
                "rb_hi": _hi,
            }
        )

    _rows = []
    _data_source_label = "Data" if "Data" in set(_parameter_summary["scenario"].astype(str)) else "Empirical"
    _append_whole_rb(_rows, _parameter_summary, source_scenario=_data_source_label, label="Data")
    _append_whole_rb(_rows, _glm_summary, source_scenario="Full fitted", label="GLM")
    _append_whole_rb(_rows, _parameter_summary, source_scenario="Full fitted", label="Free Psychometrics")
    _append_whole_rb(_rows, _parameter_summary, source_scenario="Fixed bias", label="Fixed bias")
    _append_whole_rb(_rows, _parameter_summary, source_scenario="Fixed lapses", label="Fixed lapses")
    _whole_rb = pd.DataFrame(_rows)
    mo.stop(_whole_rb.empty, mo.md("No whole-RB summary available."))

    _palette = {
        "Data": "#1f77b4",
        "GLM": "#7b3294",
        "Free Psychometrics": "#111111",
        "Fixed bias": "#d55e00",
        "Fixed lapses": "#009e73",
    }
    _fig_whole_rb, _ax_whole_rb = plt.subplots(1, 1, figsize=(4.2, 3.0), layout="constrained")
    _x = np.arange(len(_whole_rb))
    _y = _whole_rb["rb_mean"].to_numpy(dtype=float)
    _yerr = np.vstack(
        [
            np.clip(_y - _whole_rb["rb_lo"].to_numpy(dtype=float), 0.0, None),
            np.clip(_whole_rb["rb_hi"].to_numpy(dtype=float) - _y, 0.0, None),
        ]
    )
    _colors = [_palette.get(_scenario, "0.2") for _scenario in _whole_rb["scenario"]]
    _ax_whole_rb.bar(_x, _y, color=_colors, alpha=0.85, width=0.72)
    _ax_whole_rb.errorbar(
        _x,
        _y,
        yerr=_yerr,
        fmt="none",
        ecolor="0.2",
        elinewidth=1.0,
        capsize=3,
        zorder=3,
    )
    _ax_whole_rb.axhline(float(parameter_fixed_meta.get("baseline", 0.5)), color="0.5", lw=0.9, ls="--", zorder=0)
    _ax_whole_rb.set_xticks(_x, _whole_rb["scenario"].tolist(), rotation=25, ha="right")
    _ax_whole_rb.set_ylim(0.4, 0.7)
    _ax_whole_rb.set_ylabel("Rep. bias")

    _whole_rb_display = _whole_rb.copy()
    for _col in ["rb_mean", "rb_lo", "rb_hi"]:
        _whole_rb_display[_col] = _whole_rb_display[_col].round(3)

    mo.vstack(
        [
            mo.md("#### Whole parameter-fixed repetition bias"),
            _fig_whole_rb,
            save_plot(
                _fig_whole_rb,
                "action trace psychometric parameter-fixed whole repetition bias",
                stem=f"action_trace_parameter_fixed_whole_rb_{ui_accuracy_regressor.value}",
            ),
        ],
        align="center",
    )
    _fig_whole_rb
    return


@app.cell
def _(
    mo,
    plot_df_all,
    plot_regressor_net_impact,
    plt,
    save_plot,
    ui_net_impact_x_axis,
    ui_net_impact_y_axis,
):
    mo.stop(
        ui_net_impact_x_axis is None or ui_net_impact_y_axis is None,
        mo.md("No regressor pair is available for the net-impact plot."),
    )
    _x_axis = ui_net_impact_x_axis.value
    _y_axis = ui_net_impact_y_axis.value
    _fig_net_impact, _ax_net_impact = plt.subplots(1, 1, figsize=(4, 3), layout="constrained")
    plot_regressor_net_impact(
        plot_df_all,
        x_axis=_x_axis,
        y_axis=_y_axis,
        axes=[_ax_net_impact],
    )
    mo.vstack(
        [
            mo.hstack([ui_net_impact_x_axis, ui_net_impact_y_axis], justify="center"),
            _fig_net_impact,
            save_plot(
                _fig_net_impact,
                f"net impact {_y_axis} by {_x_axis}",
                stem=f"net_impact_{_y_axis}_by_{_x_axis}",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(mo, plot_df_all, plots, save_plot):
    mo.stop(
        not hasattr(plots, "plot_right_integration_map"),
        mo.md("No p(right) integration map helper is available for this task."),
    )
    _fig_integration_map = plots.plot_right_integration_map(
        plot_df_all,
    )
    mo.stop(
        _fig_integration_map is None,
        mo.md("No p(right) integration map available for the selected task/features."),
    )

    mo.vstack(
        [
            _fig_integration_map,
            save_plot(
                _fig_integration_map,
                "p(right) integration map",
                stem="right_integration_map",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(adapter, plot_df_all, plots, plt, views_sel):
    fig, axd = plt.subplot_mosaic(
        [["a", "b"], ["c", "d"]],
        figsize=(7, 5),
        layout="constrained",
    )   
    plots.plot_right_by_regressor(plot_df_all, regressor_col="choice_lag_one_hot_sum", ax=axd["a"])
    # plots.plot_right_integration_map(
    #     plot_df_all,
    #     ax=axd["b"],
    #     x_col="evidence_param",
    #     xlabel="Evidence strength",
    # )
    plots.plot_right_by_regressor(plot_df_all, regressor_col="choice_lag_one_hot_sum", ax=axd["d"])
    # plots.plot_right_integration_map(
    #     plot_df_all,
    #     ax=axd["c"],
    #     x_col="evidence_param",
    #     xlabel="Evidence strength",
    # )
    plots.plot_accuracy_by_total_evidence(
        plot_df_all,
        adapter=adapter,
        views=views_sel,
        ax = axd["a"],
    )
    plots.plot_repeat_by_repeat_evidence(
        plot_df_all,
        views=views_sel,
        ax = axd["c"],
    )
    for label, ax in zip("abcd", axd.values()):
        ax.text(
            -0.25, 1.1, label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )
        ax.set_box_aspect(1)
    fig
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
    add_choice_lag_summary_regressor,
    apply_state_tweak_to_trial_df,
    apply_state_tweak_to_view,
    coef_editor,
    coef_editor_payload,
    coef_state_idx,
    coef_state_label,
    display_regressor_name,
    editor_trial_df,
    mo,
    np,
    plots,
    prepare_predictions_df,
    save_plot,
    subject,
    task_name,
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
    _choice_lag_cols = [
        str(_feat)
        for _feat in list(getattr(_view_tweaked, "feat_names", []) or [])
        if str(_feat).startswith("choice_lag_")
    ]
    if not _choice_lag_cols:
        _choice_lag_cols = adapter.choice_lag_cols(_trial_df_tweaked)
    _plot_df_tweaked = prepare_predictions_df(task_name, _trial_df_tweaked)
    _plot_df_tweaked = add_choice_lag_summary_regressor(
        _plot_df_tweaked,
        choice_lag_cols=_choice_lag_cols,
    )

    _title = f"{subject} — tweaked {coef_state_label}"
    _fig_all_tweaked, _ = plots.plot_categorical_performance_all(
        _plot_df_tweaked,
        _title,
        # background_style=ui_psychometric_background.value,
    )
    _fig_all_tweaked_list = (
        list(_fig_all_tweaked)
        if isinstance(_fig_all_tweaked, (list, tuple))
        else [_fig_all_tweaked]
    )
    _regressor_col = "choice_lag_one_hot_sum" if "choice_lag_one_hot_sum" in _plot_df_tweaked.columns else (
        "choice_lag_param" if "choice_lag_param" in _plot_df_tweaked.columns else (
            "at_choice_param" if "at_choice_param" in _plot_df_tweaked.columns else None
        )
    )
    if _regressor_col is None:
        _reg_section = mo.md("No choice-history regressor available for the tweaked psychometric plot.")
    else:
        _regressor_label = display_regressor_name(_regressor_col)
        _fig_reg_tweaked = plots.plot_repeat_by_regressor_simple(
            _plot_df_tweaked,
            regressor_col=_regressor_col,
            views={subject: _view_tweaked},
            title=None,
        )
        if _fig_reg_tweaked is None:
            _reg_section = mo.md("No valid trials available for the tweaked regressor psychometric plot.")
        else:
            _reg_section = mo.vstack(
                [
                    _fig_reg_tweaked,
                    save_plot(
                        _fig_reg_tweaked,
                        f"tweaked {_regressor_label} repeat probability",
                        stem=f"tweaked_regressor_{_regressor_col}",
                    ),
                ],
                align="center",
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
                    _item
                    for _fig_idx, _fig in enumerate(_fig_all_tweaked_list, start=1)
                    for _item in (
                        _fig,
                        save_plot(
                            _fig,
                            (
                                f"tweaked overall psychometric {_fig_idx}"
                                if len(_fig_all_tweaked_list) > 1
                                else "tweaked overall psychometric"
                            ),
                            stem=(
                                f"tweaked_categorical_overall_{_fig_idx}"
                                if len(_fig_all_tweaked_list) > 1
                                else "tweaked_categorical_overall"
                            ),
                        ),
                    )
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
            _reg_section,
        ],
        widths=[2.0, 1.0, 1.4],
    )
    return


if __name__ == "__main__":
    app.run()
