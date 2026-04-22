import marimo

__generated_with = "0.23.2"
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
    from glmhmmt.postprocess import (
        build_trial_df,
        build_emission_weights_df,
        build_weights_boxplot_payload,
    )
    from glmhmmt.plots import plot_feature_boxplot, plot_weights_boxplot
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import get_state_color
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_afc_delay as process_two_afc_delay
    from src.process.common import add_choice_lag_summary_regressor

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name == "2AFC_delay":
            return process_two_afc_delay.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    sns.set_style("ticks")
    paths = get_runtime_paths()
    return (
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        add_choice_lag_summary_regressor,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        build_emission_weights_df,
        build_trial_and_weights_df,
        build_trial_df,
        build_weights_boxplot_payload,
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
        plot_feature_boxplot,
        plot_weights_boxplot,
        plt,
        prepare_predictions_df,
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
def _(df_all):
    df_all
    return


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
                n_restarts=_n_restarts,
                verbose=True,
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
def _(mo):
    ui_mcdr_one_hot_mode = mo.ui.dropdown(
        options=["folded", "split"],
        value="folded",
        label="MCDR one-hot view (folded or split)",
    )
    return (ui_mcdr_one_hot_mode,)


@app.cell
def _(is_2afc, np, pd, plot_feature_boxplot, plt, sns, task_name):
    import re

    is_mcdr = task_name == "MCDR"
    flip_binary_one_hot = task_name in {"2AFC"}
    is_2afc_delay = task_name == "2AFC_delay"
    _stim_pattern = (
        re.compile(r"^stim_x_delay_hot_(m?\d+(?:p\d+)?)$")
        if is_2afc_delay
        else re.compile(r"^stim_(\d+)$")
    )
    _choice_lag_pattern = re.compile(r"^choice_lag_(\d+)$")
    _bias_pattern = re.compile(r"^bias_(\d+)$")
    _mcdr_stim_pattern = re.compile(r"^stim(?P<x_value>\d+)(?P<side>[LCR])$")
    _mcdr_choice_lag_pattern = re.compile(r"^choice_lag_(?P<x_value>\d+)(?P<side>[LCR])$")

    def _parse_feature_level_token(token: str) -> float | int | None:
        if is_2afc_delay:
            try:
                return float(token.replace("m", "-").replace("p", "."))
            except ValueError:
                return None
        try:
            return int(token)
        except ValueError:
            return None

    def _format_feature_level_label(value) -> str:
        if isinstance(value, str):
            return value
        if is_2afc_delay:
            value = float(value)
            if np.isclose(value, 0.1):
                return "0"
            if value.is_integer():
                return str(int(value))
            return format(value, "g")
        return str(int(value))

    def _weights_df_to_plot_df(
        weights_df,
        *,
        class_idx: int | None = 0,
        flip_sign: bool = False,
    ) -> pd.DataFrame:
        if weights_df is None or getattr(weights_df, "is_empty", lambda: False)():
            return pd.DataFrame()

        df_plot = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        if df_plot.empty:
            return df_plot

        if class_idx is not None and "class_idx" in df_plot.columns:
            df_plot = df_plot[df_plot["class_idx"] == class_idx].copy()
        else:
            df_plot = df_plot.copy()

        if df_plot.empty:
            return df_plot

        df_plot["feature"] = df_plot["feature"].astype(str)
        df_plot["weight"] = pd.to_numeric(df_plot["weight"], errors="coerce")
        if "class_idx" in df_plot.columns:
            df_plot["class_idx"] = pd.to_numeric(df_plot["class_idx"], errors="coerce")
        if flip_sign:
            df_plot["weight"] = -df_plot["weight"]
        if "state_rank" not in df_plot.columns:
            df_plot["state_rank"] = 0
        if "state_label" not in df_plot.columns:
            df_plot["state_label"] = "State 0"
        return df_plot.dropna(subset=["weight"]).copy()

    def _plot_grouped_boxplot(
        grouped_df: pd.DataFrame,
        *,
        value_col: str,
        title: str,
        xlabel: str,
        ylabel: str = "Weight",
        x_col: str = "x_value",
        order: list | None = None,
    ) -> plt.Figure | None:
        if grouped_df.empty:
            return None

        ordered_values = order or sorted(pd.unique(grouped_df[x_col]).tolist())
        grouped_df = grouped_df[grouped_df[x_col].isin(ordered_values)].copy()
        if grouped_df.empty:
            return None

        subject_order = pd.unique(grouped_df["subject"]).tolist()
        per_feature_values: list[np.ndarray] = []
        subject_lines = np.full((len(subject_order), len(ordered_values)), np.nan, dtype=float)

        for feature_idx, x_value in enumerate(ordered_values):
            feature_df = grouped_df[grouped_df[x_col] == x_value].copy()
            if feature_df.empty:
                per_feature_values.append(np.asarray([], dtype=float))
                continue
            by_subject = (
                feature_df.groupby("subject", observed=False)[value_col]
                .mean()
                .reindex(subject_order)
            )
            subject_lines[:, feature_idx] = by_subject.to_numpy(dtype=float)
            per_feature_values.append(by_subject.dropna().to_numpy(dtype=float))

        if not any(values.size for values in per_feature_values):
            return None

        return plot_feature_boxplot(
            per_feature_values,
            [_format_feature_level_label(value) for value in ordered_values],
            subject_lines=subject_lines,
            figsize=(max(5.0, 0.8 * len(ordered_values)), 4.0),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    def plot_feature_rule_boxplot(
        weights_df,
        *,
        pattern,
        title: str,
        xlabel: str,
        order: list[int] | None = None,
    ) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(
            weights_df,
            class_idx=0,
            flip_sign=flip_binary_one_hot,
        )
        if df_plot.empty:
            return None
        df_plot["x_value"] = df_plot["feature"].str.extract(pattern, expand=False)
        df_plot = df_plot[df_plot["x_value"].notna()].copy()
        if df_plot.empty:
            return None
        df_plot["x_value"] = df_plot["x_value"].map(_parse_feature_level_token)
        df_plot = df_plot[df_plot["x_value"].notna()].copy()
        if df_plot.empty:
            return None
        return _plot_grouped_boxplot(
            df_plot,
            value_col="weight",
            title=title,
            xlabel=xlabel,
            order=order,
        )

    def plot_mcdr_folded_boxplot(
        weights_df,
        *,
        pattern,
        title: str,
        xlabel: str,
        mode: str = "folded",
        order: list[int] | None = None,
        positive_label: str = "coh",
        neutral_label: str = "C",
        negative_label: str = "incoh",
    ) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(weights_df, class_idx=None, flip_sign=False)
        if df_plot.empty or "class_idx" not in df_plot.columns:
            return None

        parsed = df_plot["feature"].str.extract(pattern)
        df_plot = pd.concat([df_plot, parsed], axis=1)
        df_plot = df_plot[df_plot["x_value"].notna() & df_plot["side"].notna()].copy()
        if df_plot.empty:
            return None

        df_plot["x_value"] = df_plot["x_value"].astype(int)
        df_plot["class_idx"] = pd.to_numeric(df_plot["class_idx"], errors="coerce")
        df_plot = df_plot[df_plot["class_idx"].isin([0, 1])].copy()
        if df_plot.empty:
            return None

        pivoted = (
            df_plot.groupby(["subject", "x_value", "side", "class_idx"], as_index=False, observed=False)["weight"]
            .mean()
            .pivot(index=["subject", "x_value", "side"], columns="class_idx", values="weight")
            .reset_index()
        )
        for class_idx in (0, 1):
            if class_idx not in pivoted.columns:
                pivoted[class_idx] = np.nan
        pivoted = pivoted.dropna(subset=[0, 1]).copy()
        if pivoted.empty:
            return None
        pivoted = pivoted.rename(columns={0: "weight_0", 1: "weight_1"})

        records: list[dict[str, object]] = []
        for row in pivoted.itertuples(index=False):
            subject = str(row.subject)
            x_value = int(row.x_value)
            side = str(row.side)
            weight_0 = float(row.weight_0)
            weight_1 = float(row.weight_1)
            if side == "L":
                records.append({"subject": subject, "x_value": x_value, "group": positive_label, "value": weight_0})
                records.append({"subject": subject, "x_value": x_value, "group": negative_label, "value": weight_1})
            elif side == "R":
                records.append({"subject": subject, "x_value": x_value, "group": positive_label, "value": weight_1})
                records.append({"subject": subject, "x_value": x_value, "group": negative_label, "value": weight_0})
            elif side == "C":
                records.append({"subject": subject, "x_value": x_value, "group": neutral_label, "value": (weight_0 + weight_1) / 2.0})

        if not records:
            return None

        grouped_df = pd.DataFrame.from_records(records)
        if grouped_df.empty:
            return None

        if mode == "split":
            split_df = grouped_df[grouped_df["group"].isin([positive_label, neutral_label, negative_label])].copy()
            if split_df.empty:
                return None
            split_df = (
                split_df.groupby(["subject", "x_value", "group"], as_index=False, observed=False)["value"]
                .mean()
            )
            ordered_values = order or sorted(pd.unique(split_df["x_value"]).tolist())
            label_order = [
                f"{value} {group}"
                for value in ordered_values
                for group in (positive_label, neutral_label, negative_label)
            ]
            label_order = [label for label in label_order if label.split(" ", 1)[1] in set(split_df["group"])]
            split_df["x_label"] = (
                split_df["x_value"].astype(int).astype(str)
                + " "
                + split_df["group"].astype(str)
            )
            return _plot_grouped_boxplot(
                split_df,
                value_col="value",
                title=f"{title} (split {positive_label}/{neutral_label}/{negative_label})",
                xlabel=xlabel,
                ylabel="Weight",
                x_col="x_label",
                order=label_order,
            )

        folded_df = grouped_df[grouped_df["group"].isin([positive_label, negative_label])].copy()
        if folded_df.empty:
            return None
        folded_df["value"] = np.where(
            folded_df["group"] == negative_label,
            -folded_df["value"].to_numpy(dtype=float),
            folded_df["value"].to_numpy(dtype=float),
        )
        folded_df = (
            folded_df.groupby(["subject", "x_value", "group"], as_index=False, observed=False)["value"]
            .mean()
        )
        folded_df = (
            folded_df.groupby(["subject", "x_value"], as_index=False, observed=False)["value"]
            .mean()
        )
        return _plot_grouped_boxplot(
            folded_df,
            value_col="value",
            title=f"{title} (folded {positive_label}/{negative_label})",
            xlabel=xlabel,
            ylabel="Weight",
            order=order,
        )

    def plot_feature_rule_lineplot(
        weights_df,
        *,
        pattern,
        title: str,
        xlabel: str,
        order: list[int] | None = None,
    ) -> plt.Figure | None:
        df_plot = _weights_df_to_plot_df(
            weights_df,
            class_idx=0,
            flip_sign=flip_binary_one_hot,
        )
        if df_plot.empty:
            return None
        df_plot["x_value"] = df_plot["feature"].str.extract(pattern, expand=False)
        df_plot = df_plot[df_plot["x_value"].notna()].copy()
        if df_plot.empty:
            return None
        df_plot["x_value"] = df_plot["x_value"].map(_parse_feature_level_token)
        df_plot = df_plot[df_plot["x_value"].notna()].copy()
        if df_plot.empty:
            return None
        ordered_values = order or sorted(pd.unique(df_plot["x_value"]).tolist())
        df_plot = df_plot[df_plot["x_value"].isin(ordered_values)].copy()
        if df_plot.empty:
            return None
        df_plot["x_value"] = pd.Categorical(
            df_plot["x_value"],
            categories=ordered_values,
            ordered=True,
        )
        summary = (
            df_plot.groupby("x_value", as_index=False, observed=False)["weight"]
            .mean()
            .sort_values("x_value")
        )
        if summary.empty:
            return None

        positions = np.arange(len(summary))
        fig, ax = plt.subplots(figsize=(max(5.0, 0.8 * len(summary)), 4.0))
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Weight")
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_format_feature_level_label(value) for value in summary["x_value"].tolist()]
        )
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_stim_hot_weights(weights_df, *, mcdr_mode: str = "folded") -> plt.Figure | None:
        if is_mcdr:
            return plot_mcdr_folded_boxplot(
                weights_df,
                pattern=_mcdr_stim_pattern,
                title="stim one-hot",
                xlabel="Stimulus window",
                mode=mcdr_mode,
                order=[1, 2, 3, 4],
            )
        return plot_feature_rule_boxplot(
            weights_df,
            pattern=_stim_pattern,
            title="stim_hot" if task_name == "2AFC" else "stim×delay one-hot",
            xlabel="stimulus level" if task_name == "2AFC" else "delay level",
        )

    def plot_choice_lag_weights(weights_df, *, mcdr_mode: str = "folded") -> plt.Figure | None:
        if is_mcdr:
            return plot_mcdr_folded_boxplot(
                weights_df,
                pattern=_mcdr_choice_lag_pattern,
                title="choice_lag_*",
                xlabel="Lag",
                mode=mcdr_mode,
                positive_label="repeat",
                neutral_label="C",
                negative_label="switch",
            )
        return plot_feature_rule_boxplot(
            weights_df,
            pattern=_choice_lag_pattern,
            title="choice_lag_*",
            xlabel="Lag",
        )

    def plot_bias_hot_weights(weights_df) -> plt.Figure | None:
        return plot_feature_rule_lineplot(
            weights_df,
            pattern=_bias_pattern,
            title="bias_hot",
            xlabel="Session index",
        )

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
    _stim_pattern
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
def _(weights_df):
    weights_df
    return


@app.cell
def _(
    K,
    arrays_store,
    build_emission_weights_df,
    build_weights_boxplot_payload,
    mo,
    pl,
    plot_bias_hot_weights,
    plot_choice_lag_weights,
    plot_sequence_feature_weights,
    plot_stim_hot_weights,
    plot_weights_boxplot,
    plots,
    save_plot,
    selected,
    task_name,
    ui_mcdr_one_hot_mode,
    views,
    weights_df,
):
    mo.stop(not arrays_store, mo.md("No results loaded."))
    views_sel = {s: views[s] for s in selected}
    _weights_df_sel = weights_df.filter(pl.col("subject").is_in(selected))
    _mcdr_mode = ui_mcdr_one_hot_mode.value if task_name == "MCDR" else "folded"

    _fig_by_subject = plots.plot_emission_weights_by_subject(
        _weights_df_sel,
        K=K,
    )

    _fig_summary = plot_weights_boxplot(**build_weights_boxplot_payload(build_emission_weights_df(views_sel)))
    _fig_stim_hot = plot_stim_hot_weights(_weights_df_sel, mcdr_mode=_mcdr_mode)
    _fig_choice_lag = plot_choice_lag_weights(_weights_df_sel, mcdr_mode=_mcdr_mode)
    _fig_bias_hot = plot_bias_hot_weights(_weights_df_sel)
    _fig_lapses = plots.plot_lapse_rates_boxplot(views=views_sel, K=K)
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
                [_fig_summary, save_plot(_fig_summary, "emission weights", stem="emission_weights")],
                align="center",
            )
        )
    _summary_cards.append(mo.vstack([_fig_lapses, save_plot(_fig_lapses, "lapse rates", stem="lapse_rates")], align="center"))
    _summary_panel = mo.hstack(_summary_cards, align="start", justify="start", gap=1.0)
    _items.extend([mo.md("#### Summary"), _summary_panel])

    _one_hot_figs = []
    if _fig_stim_hot is not None:
        _one_hot_figs.append(
            mo.vstack(
                [
                    _fig_stim_hot,
                    save_plot(_fig_stim_hot, "stim one-hot", stem="stim_one_hot"),
                ],
                align="center",
            )
        )
    if _fig_choice_lag is not None:
        _one_hot_figs.append(
            mo.vstack(
                [
                    _fig_choice_lag,
                    save_plot(_fig_choice_lag, "choice lag one-hot", stem="prev_choice_one_hot"),
                ],
                align="center",
            )
        )
    if _fig_bias_hot is not None:
        _one_hot_figs.append(
            mo.vstack(
                [
                    _fig_bias_hot,
                    save_plot(_fig_bias_hot, "bias hot", stem="bias_one_hot"),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the total-evidence plots, the x-axis is the fitted emission evidence for the correct class on each trial.

    **2-choice task**

    As we take a class to be the reference, the fitted logits are:

    $$
    (\eta_L, 0),
    $$

    so the other logit is always fixed to zero. Therefore the total evidence for the correct choice is just the signed fitted logit:

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

    if task_name in {"MCDR", "2AFC_delay"}:
        regressor_options = [col for col in ["choice_lag_param"] if col in _available_cols]
    else:
        regressor_options = [col for col in ["at_choice_param"] if col in _available_cols]
    if _choice_lag_cols:
        regressor_options.append("choice_lag_one_hot_sum")

    if not regressor_options:
        ui_accuracy_binning = None
        ui_accuracy_regressor = None
    else:
        ui_accuracy_binning = mo.ui.checkbox(value=False, label="Enable")
        ui_accuracy_regressor = mo.ui.dropdown(
            options=regressor_options,
            value=regressor_options[0],
            label="Regressor",
        )
    return regressor_options, ui_accuracy_regressor


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
    plot_df_all
    return (plot_df_all,)


@app.cell
def _(
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
    )

    _choice_history_regressor = plots.pick_choice_history_regressor(regressor_options)
    _regressor_for_right = _choice_history_regressor or ui_accuracy_regressor.value
    _regressor_label = plots.display_regressor_name(_regressor_for_right)

    _fig_regressor = plots.plot_right_by_regressor_simple(
        plot_df_all,
        regressor_col=_regressor_for_right,
        title=None,
    )

    mo.stop(_fig_regressor is None, mo.md(f"No p(right) plot available for {_regressor_label}."))

    mo.hstack(
        [
            mo.vstack(
                [
                    _fig_all,
                    save_plot(_fig_all, "overall psychometric", stem="accuracy_overall"),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    _fig_regressor,
                    save_plot(
                        _fig_regressor,
                        f"p(right) by {_regressor_label}",
                        stem=f"psychometric_regressor_{_regressor_for_right}",
                    ),
                ],
                align="center",
            ),
        ]
    )
    return


@app.cell
def _(adapter, mo, plot_df_all, plots, save_plot, views_sel):
    _fig_total_evidence = plots.plot_accuracy_by_total_evidence(
        plot_df_all,
        adapter=adapter,
        views=views_sel,
    )
    _fig_repeat_evidence = plots.plot_repeat_by_repeat_evidence(
        plot_df_all,
        views=views_sel,
    )

    mo.stop(_fig_total_evidence is None, mo.md("Accuracy by fitted total evidence not available."))

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
def _(
    mo,
    plot_df_all,
    plots,
    regressor_options,
    save_plot,
    ui_accuracy_regressor,
):
    _selected_regressor_label = plots.display_regressor_name(ui_accuracy_regressor.value)

    _fig_binned = plots.plot_binned_accuracy_figure(
        plot_df_all,
        regressor_col=ui_accuracy_regressor.value,
    )
    _secondary_regressor = plots.pick_choice_history_regressor(regressor_options)
    mo.stop(_secondary_regressor is None, mo.md("No choice-history regressor available."))

    _secondary_regressor_label = plots.display_regressor_name(_secondary_regressor)
    _fig_secondary_right = plots.plot_right_by_regressor(
        plot_df_all,
        regressor_col=_secondary_regressor,
        title=None,
    )

    mo.stop(_fig_binned is None, mo.md(f"No binned accuracy plot available for {_selected_regressor_label}."))

    mo.vstack(
        [
            ui_accuracy_regressor,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            _fig_binned,
                            save_plot(
                                _fig_binned,
                                f"binned accuracy {_selected_regressor_label}",
                                stem=f"accuracy_binned_{ui_accuracy_regressor.value}",
                            ),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            _fig_secondary_right,
                            save_plot(
                                _fig_secondary_right,
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
def _(mo):
    ui_integration_map_smooth = mo.ui.checkbox(value=True, label="Smooth map")
    return (ui_integration_map_smooth,)


@app.cell
def _(mo, plot_df_all, plots, save_plot, ui_integration_map_smooth):
    mo.stop(
        not hasattr(plots, "plot_right_integration_map"),
        mo.md("No p(right) integration map helper is available for this task."),
    )
    _fig_integration_map = plots.plot_right_integration_map(
        plot_df_all,
        smooth=ui_integration_map_smooth.value,
    )
    mo.stop(
        _fig_integration_map is None,
        mo.md("No p(right) integration map available for the selected task/features."),
    )

    mo.vstack(
        [
            ui_integration_map_smooth,
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
    plot_right_by_regressor_simple,
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
    _regressor_col = "choice_lag_one_hot_sum" if "choice_lag_one_hot_sum" in _plot_df_tweaked.columns else (
        "choice_lag_param" if "choice_lag_param" in _plot_df_tweaked.columns else (
            "at_choice_param" if "at_choice_param" in _plot_df_tweaked.columns else None
        )
    )
    if _regressor_col is None:
        _reg_section = mo.md("No choice-history regressor available for the tweaked psychometric plot.")
    else:
        _regressor_label = display_regressor_name(_regressor_col)
        _fig_reg_tweaked = plot_right_by_regressor_simple(
            _plot_df_tweaked,
            regressor_col=_regressor_col,
            task_name=task_name,
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
                        f"tweaked {_regressor_label} psychometric",
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
            _reg_section,
        ],
        widths=[2.0, 1.0, 1.4],
    )
    return


if __name__ == "__main__":
    app.run()
