import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys

    _PROJECT_ROOT = next(
        (
            p
            for base in (Path.cwd(), Path(__file__).resolve())
            for p in (base, *base.parents)
            if (p / "config.toml").exists() and (p / "src").exists()
        ),
        Path.cwd(),
    )
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from plot_saver import make_plot_saver
    from glmhmmt.notebook_support import (
        CoefficientEditorWidget,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        wrap_anywidget,

        model_cfg as ModelCfg,
    )
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        resolve_selected_model_id,
        select_subject_behavior_df,
    )
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from glmhmmt.cli.fit_glmhmm import main as fit_main
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import (
        build_change_triggered_posteriors_payload,
        build_session_deepdive_payload,
        build_session_trajectories_payload,
        build_state_accuracy_payload,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_state_posterior_count_payload,
        build_trial_df,
        build_emission_weights_df,
        build_weights_boxplot_payload,
        build_transition_matrix_by_subject_payload,
        build_transition_matrix_payload,
    )
    import glmhmmt.plots as model_plots
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from src.process import MCDR as process_mcdr
    from src.process import mcdr_accuracy as _process_mcdr_accuracy
    from src.process import nuo_auditory as _process_nuo_auditory
    from src.process import two_afc as process_two_afc
    from src.process import two_afc_drug as _process_two_afc_drug
    from src.process import two_adc as process_two_adc
    from src.process import two_adc_drug as _process_two_adc_drug

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)


    def put_legend_inside_panel(ax, *, loc="upper right", anchor=(0.98, 0.98)):
        legend = ax.get_legend()
        if legend is None:
            return
        handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
        labels = [text.get_text() for text in legend.get_texts()]
        title = legend.get_title().get_text()
        legend.remove()
        ax.legend(
            handles,
            labels,
            title=title or None,
            frameon=legend.get_frame_on(),
            loc=loc,
            bbox_to_anchor=anchor,
            borderaxespad=0.2,
        )

    def put_figure_legend_at_bottom(fig, *, bottom=0.24, max_ncol=8, x_anchor=0.46, y_anchor=0.01):
        legend_entries = {}
        legend_title = None

        def _add_entries(handles, labels):
            for handle, label in zip(handles, labels, strict=False):
                label_text = str(label)
                label_key = label_text.lower().replace(" ", "")
                is_engaged_probability_trace = (
                    ("p(" in label_key or label_key.startswith("$p") or "mathit{p}" in label_key)
                    and ("engag" in label_key or "enag" in label_key)
                    and ("rolling" in label_key or "raw" in label_key)
                )
                if is_engaged_probability_trace:
                    continue
                if label_text and not label_text.startswith("_"):
                    legend_entries.setdefault(label_text, handle)

        for ax in fig.axes:
            handles, labels = ax.get_legend_handles_labels()
            _add_entries(handles, labels)

        legends = list(fig.legends)
        legends.extend(fig.findobj(lambda artist: artist.__class__.__name__ == "Legend"))
        for legend in dict.fromkeys(legends):
            handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
            labels = [text.get_text() for text in legend.get_texts()]
            legend_title = legend_title or legend.get_title().get_text() or None
            _add_entries(handles, labels)
            try:
                legend.remove()
            except ValueError:
                pass

        for ax in list(fig.axes):
            if ax.has_data():
                continue
            if ax.get_legend_handles_labels()[0]:
                continue
            if ax.get_visible() and not ax.get_xlabel() and not ax.get_ylabel() and not ax.get_title():
                fig.delaxes(ax)

        if not legend_entries:
            return

        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine(None)
        fig.legend(
            legend_entries.values(),
            legend_entries.keys(),
            title=legend_title,
            loc="lower center",
            bbox_to_anchor=(x_anchor, y_anchor),
            bbox_transform=fig.transFigure,
            ncol=min(max(1, len(legend_entries)), max_ncol),
            fontsize=8,
            title_fontsize=9,
            frameon=False,
            columnspacing=1.6,
            handlelength=1.6,
        )
        fig.subplots_adjust(bottom=bottom)


    from statannotations.Annotator import Annotator

    project_root = _PROJECT_ROOT
    configure_paths(config_path=project_root / "config.toml")

    sns.set_style("ticks")
    sns.set_context("paper")
    paths = get_runtime_paths()
    from src.process.common import adapter_behavioral_column
    from src.plots.common import fig_size

    return (
        Annotator,
        ModelCfg,
        ModelManagerWidget,
        adapter_behavioral_column,
        build_change_triggered_posteriors_payload,
        build_session_deepdive_payload,
        build_session_trajectories_payload,
        build_state_accuracy_payload,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_state_posterior_count_payload,
        build_transition_matrix_payload,
        build_trial_and_weights_df,
        build_views,
        fig_size,
        fit_main,
        get_adapter,
        load_fit_arrays,
        make_plot_saver,
        mo,
        model_plots,
        np,
        paths,
        pd,
        pl,
        plt,
        prepare_predictions_df,
        project_root,
        put_figure_legend_at_bottom,
        resolve_selected_model_id,
        sns,
        wrap_anywidget,
    )


@app.cell
def _(plt, project_root):
    plt.style.use(project_root / "paper.mplstyle")
    return


@app.cell
def _(get_adapter, model_cfg):
    task_name = model_cfg.task
    adapter = get_adapter(task_name)
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    df_all = adapter.filter_condition_df(df_all, model_cfg.condition_filter)
    is_2afc = adapter.num_classes == 2
    plots = adapter.get_plots()
    return adapter, df_all, is_2afc, plots, task_name


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glmhmm",
        task="2AFC",
        K=2,
        tau=50,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return mm_widget, ui_model_manager


@app.cell
def _(ModelCfg, ui_model_manager):
    model_cfg = ModelCfg.from_value(ui_model_manager.value)
    return (model_cfg,)


@app.cell
def _(mo):
    get_last_fit_click, set_last_fit_click = mo.state(0)
    return get_last_fit_click, set_last_fit_click


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _(get_adapter, model_cfg, task_name):
    from glmhmmt.cli.fit_glmhmm import generate_model_id as _gen_id

    baseline_class_idx = int(get_adapter(task_name).baseline_class_idx)

    current_hash = _gen_id(
        task=task_name,
        K=model_cfg.K,
        tau=model_cfg.tau,
        emission_cols=model_cfg.emission_cols,
        frozen_emissions=model_cfg.frozen_emissions,
        baseline_class_idx=baseline_class_idx,
        cv_mode=model_cfg.cv_mode,
        cv_repeats=model_cfg.cv_repeats,
        condition_filter=model_cfg.condition_filter,
    )
    return (current_hash,)


@app.cell
def _(
    current_hash,
    make_plot_saver,
    mo,
    model_cfg,
    paths,
    resolve_selected_model_id,
    task_name,
):
    selected_model_id = resolve_selected_model_id(
        current_hash,
        model_cfg.existing,
        model_cfg.alias,
    )
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=task_name,
        model_id=f"glmhmm/{selected_model_id}",
    )
    return save_plot, selected_model_id


@app.cell
def _(df_all, pl):
    df_all.group_by("subject").agg(
        pl.col("session").n_unique().alias("n_sessions")
    )
    return


@app.cell
def _(current_hash, mo, save_plot, ui_model_manager):
    mo.vstack(
        [
            ui_model_manager,
            save_plot.save_all_widget(label="Save all model plots"),
            mo.md(f"**Current params hash:** `{current_hash}`"),
        ],
        align="center",
    )
    return


@app.cell
def _(
    current_hash,
    fit_main,
    get_adapter,
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

    _n_restarts = 1 if model_cfg.cv_mode != "none" else 5
    _cv_repeats = int(model_cfg.cv_repeats) if model_cfg.cv_mode != "none" else 0
    _baseline_class_idx = int(get_adapter(task_name).baseline_class_idx)

    _selected_id = model_cfg.existing or (model_cfg.alias if model_cfg.alias else current_hash)
    _OUT = paths.RESULTS / "fits" / task_name / "glmhmm" / _selected_id


    def _progress_title(info: dict) -> str:
        return f"Fitting GLM-HMM K={info['K']} subject {info['subject_index']}/{info['subject_total']}: {info['subject']}"


    def _progress_subtitle(info: dict) -> str:
        _base = f"Restart {info['restart_index']}/{info['restart_total']}"
        if info.get("event") == "restart_complete":
            return f"{_base} complete"
        return _base


    _total_progress = max(
        1,
        len(model_cfg.subjects) * (_cv_repeats if model_cfg.cv_mode != "none" else _n_restarts),
    )
    mm_widget.is_running = True
    try:
        with mo.status.progress_bar(
            total=_total_progress,
            title=f"Fitting GLM-HMM K={model_cfg.K}",
            subtitle=(
                f"{len(model_cfg.subjects)} subjects × {_cv_repeats} CV repeat(s)"
                if model_cfg.cv_mode != "none"
                else f"{len(model_cfg.subjects)} subjects × {_n_restarts} restart(s)"
            ),
            completion_title="Fit complete",
            completion_subtitle=f"Saved under {_selected_id}",
        ) as _bar:

            def _on_progress(info: dict) -> None:
                if info.get("event") == "cv_repeat_start":
                    _bar.update(
                        increment=0,
                        title=_progress_title(info),
                        subtitle=f"CV repeat {info['cv_repeat_index']}/{info['cv_repeat_total']}",
                    )
                    return
                if info.get("event") == "cv_repeat_complete":
                    _bar.update(
                        increment=1,
                        title=_progress_title(info),
                        subtitle=f"CV repeat {info['cv_repeat_index']}/{info['cv_repeat_total']} complete",
                    )
                    return
                if info.get("event") == "restart_start":
                    _bar.update(
                        increment=0,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )
                    return
                if info.get("event") == "restart_complete":
                    _bar.update(
                        increment=0 if model_cfg.cv_mode != "none" else 1,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )

            fit_main(
                subjects=model_cfg.subjects,
                K_list=[model_cfg.K],
                out_dir=_OUT,
                tau=model_cfg.tau,
                emission_cols=model_cfg.emission_cols,
                frozen_emissions=model_cfg.frozen_emissions or None,
                task=task_name,
                cv_mode=model_cfg.cv_mode,
                cv_repeats=_cv_repeats,
                n_restarts=_n_restarts,
                verbose=False,
                condition_filter=model_cfg.condition_filter,
                baseline_class_idx=_baseline_class_idx,
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
    model_cfg,
    paths,
    selected_model_id,
    task_name,
):
    K = model_cfg.K

    OUT = paths.RESULTS / "fits" / task_name / "glmhmm" / selected_model_id
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glmhmm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=list(model_cfg.subjects),
        emission_cols=list(model_cfg.emission_cols),
        k=K,
    )

    selected = [s for s in model_cfg.subjects if s in arrays_store]
    _ = names
    return K, arrays_store, selected


@app.cell
def _(adapter, mo):
    # ── State-scoring regressor selector ─────────────────────────────────────
    _opts = list(adapter._SCORING_OPTIONS.keys()) if hasattr(adapter, "_SCORING_OPTIONS") else ["default"]
    _default_key = getattr(adapter, "scoring_key", _opts[0])
    if _default_key not in _opts:
        _default_key = _opts[0]
    ui_scoring_key = mo.ui.dropdown(
        options=_opts,
        value=_default_key,
        label="State scoring regressor (Engaged = highest score)",
    )
    mo.vstack([mo.md("### State labelling regressor"), ui_scoring_key])
    return (ui_scoring_key,)


@app.cell
def _(K, adapter, arrays_store, build_views, mo, selected, ui_scoring_key):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))

    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    views = build_views(arrays_store, adapter, K, selected)
    editor_views = views.copy()
    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    return state_labels, views


@app.cell
def _(adapter, build_trial_and_weights_df, df_all, mo, views):
    trial_df, weights_df = build_trial_and_weights_df(
        df_all,
        views=views,
        adapter=adapter,
        min_session_length=2,
    )
    mo.stop(trial_df.height == 0, mo.md("No subjects with matching data lengths."))
    return trial_df, weights_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(selected, views):
    views_sel = {s: views[s] for s in selected}
    return (views_sel,)


@app.cell
def _(mo, save_plot):
    def panel(title, fig=None, stem=None, description=None):
        content = [mo.md(f"#### {title}")]

        if fig is not None:
            content.append(fig)
            if stem is not None:
                content.append(save_plot(fig, description or title.lower(), stem=stem))

        return mo.vstack(content, align="center")

    return (panel,)


@app.cell
def _():
    BOXPLOT_STYLE = dict(
        fill=False,
        boxprops={"color": "0.5"},
        whiskerprops={"color": "0.5"},
        medianprops={"linewidth": 2},
        showfliers=False,
        showcaps=False,
    )
    return (BOXPLOT_STYLE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature labels
    """)
    return


@app.cell
def _(weights_df):
    feature_labels = {
        "stim_param": r"$\mathrm{Stim}_{\mathrm{param}}$",
        "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "at_choice_param": r"$\mathrm{A}_t$",
        "choice_lag_param": r"$\mathrm{A}$",
        "choice_lag_param_2": r"$\mathrm{A}$",
        "prev_choice": r"$choice_{t-1}$",
        "stim_x_delay_param":  r"$\mathrm{Stim}:\mathrm{Delay}_{\mathrm{param}}$"
    }

    feature_labeler = lambda feature: feature_labels.get(str(feature), str(feature))

    features = weights_df["feature"].unique()
    preferred_feature_order = []
    for _feature_group in (
        ["bias_param", "biasparam", "bias"],
        ["stim_param", "stim", "stim_x_delay_param"],
        ["at_choice_param", "choice_lag_param", "at_choice", "prev_choice"],
    ):
        preferred_feature_order.extend(
            _feature for _feature in _feature_group if _feature in features and _feature not in preferred_feature_order
        )
    plot_feature_order = preferred_feature_order + [_feature for _feature in features if _feature not in preferred_feature_order]

    state_order = ["Engaged", "Disengaged"]
    state_palette = {"Engaged": "tab:green", "Disengaged": "tab:gray"}
    return (
        feature_labeler,
        features,
        plot_feature_order,
        state_order,
        state_palette,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Emission weights
    """)
    return


@app.cell
def _():
    # fig_by_subject = model_plots.emission_weights_by_subject(
    #     weights_df,
    #     K=K,
    # )
    return


@app.cell
def _(
    Annotator,
    BOXPLOT_STYLE,
    feature_labeler,
    features,
    fig_size,
    mo,
    panel,
    plot_feature_order,
    plt,
    selected,
    sns,
    state_order,
    state_palette,
    weights_df,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    emissions_fig, emissions_ax = plt.subplots(figsize=fig_size(2, 1))

    sns.boxplot(
        data=weights_df,
        ax=emissions_ax,
        x="feature",
        y="weight",
        hue="state_label",
        order=plot_feature_order,
        hue_order=state_order,
        palette=state_palette,
        **BOXPLOT_STYLE,
    )
    emissions_ax.axhline(0, linestyle="--", color="0.5", zorder=0)


    # We take pairs for the annotation of the significance
    paired = weights_df.pivot(
        values="weight",
        index=["subject", "feature"],
        columns="state_label",
        aggregate_function="first",
    )

    for row in paired.iter_rows(named=True):
        x = plot_feature_order.index(row["feature"])
        emissions_ax.plot(
            [x - 0.2, x + 0.2],
            [row[state_order[0]], row[state_order[1]]],
            color="0.75",
            linewidth = 0.5,
            zorder=0,
        )

    _pairs = [((f, state_order[0]), (f, state_order[1])) for f in features]
    Annotator(
        emissions_ax,
        _pairs,
        data=weights_df.to_pandas(),
        x="feature",
        y="weight",
        hue="state_label",
        order=plot_feature_order,
        hue_order=state_order,
    ).configure(test="t-test_paired", text_format="star", line_height=0, verbose=False).apply_and_annotate()


    emissions_ax.legend(frameon=False)
    emissions_ax.set_xticklabels([feature_labeler(f) for f in plot_feature_order])

    panel(
        "Emission weights",
        emissions_ax.figure,
        "emissions",
        "emissions boxplot",
    ),
    return


@app.cell
def _(np, pd):
    from scipy.stats import ttest_1samp

    def significance_stars(pvalue: float) -> str:
        if not np.isfinite(pvalue):
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        if pvalue < 0.05:
            return "*"
        return ""

    def annotate_choice_lag_ttests(ax, panel_df: pd.DataFrame, lag_order: list[int], y: float = 3.75) -> None:
        for lag in lag_order:
            values = panel_df.loc[panel_df["lag"] == lag, "weight"].dropna().to_numpy(dtype=float)
            ax.text(
                lag,
                y,
                significance_stars(float(ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue)),
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                clip_on=False,
            )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Transition matrices
    """)
    return


@app.cell
def _(mo, selected):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    # _by_subject_payload = build_transition_matrix_by_subject_payload(
    #     arrays_store=arrays_store,
    #     state_labels=state_labels,
    #     K=K,
    #     subjects=selected,
    # )
    # _fig_by_subject = model_plots.transition_matrix_by_subject(**_by_subject_payload)
    return


@app.cell
def _(
    K,
    arrays_store,
    build_transition_matrix_payload,
    fig_size,
    model_plots,
    panel,
    plt,
    selected,
    state_labels,
):
    transitions_summary = build_transition_matrix_payload(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    fig_transition_matrix, ax_transition_matrix = plt.subplots(figsize=fig_size(2,1))
    fig_transition_matrix = model_plots.transition_matrix(**transitions_summary, ax=ax_transition_matrix)
    fig_transition_matrix.set_title("")

    panel("Mean Transition matrix",ax_transition_matrix.figure, "transition_matrix")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State dwell times
    """)
    return


@app.cell
def _(
    build_state_dwell_times_payload,
    fig_size,
    mo,
    model_plots,
    panel,
    plt,
    trial_df,
    views_sel,
    weights_df,
):
    dwell_times = build_state_dwell_times_payload(
        trial_df,
        session_col="session",
        sort_col="trial_idx",
        views=views_sel,
        max_dwell=90,
    )

    fig_dwell_time, ax_dwell_time = plt.subplots(
        1,
        (weights_df["state_label"].n_unique()),
        figsize=(fig_size(1,2)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig_dwell_time, ax_dwell_time = model_plots.state_dwell_times_summary(
        dwell_times,
        axes=ax_dwell_time,
    )
    fig_dwell_cumulative, ax_dwell_cumulative = plt.subplots(figsize=fig_size(2,1))
    fig_dwell_cumulative, ax_dwell_cumulative = model_plots.state_dwell_times_cumulative(
        dwell_times,
        ax=ax_dwell_cumulative,
    )
    fig_dwell_median, ax_dwell_median = plt.subplots(figsize=fig_size(3,1))
    fig_dwell_median, ax_dwell_median = model_plots.state_dwell_median_boxplot(
        dwell_times,
        ax=ax_dwell_median,
    )

    mo.hstack([panel("Mean Transition matrix",fig_dwell_time, "dwell_time"), panel("Mean Transition matrix",ax_dwell_cumulative.figure, "dwell_cumulative"), panel("Mean Transition matrix",ax_dwell_median.figure, "dwell_median")])
    return


@app.cell
def _(is_2afc, mo, views):
    _feature_names = []
    if is_2afc and views:
        for _view in views.values():
            for _feat in list(getattr(_view, "feat_names", []) or []):
                if _feat not in _feature_names:
                    _feature_names.append(_feat)
    if not _feature_names:
        _feature_names = ["at_choice"]
    _default_feature = "at_choice" if "at_choice" in _feature_names else _feature_names[0]
    ui_psychometric_regressor = mo.ui.dropdown(
        options=_feature_names,
        value=_default_feature,
        label="Regressor",
    )
    ui_psychometric_regressor
    return (ui_psychometric_regressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Task accuracy plots
    """)
    return


@app.cell
def _(
    K,
    fig_size,
    mo,
    panel,
    plots,
    plt,
    prepare_predictions_df,
    selected,
    task_name,
    trial_df,
    ui_psychometric_regressor,
    views_sel,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))

    plot_df_all = prepare_predictions_df(task_name, trial_df)
    _state_plot_kwargs = dict(
        background_style="model",
        show_weighted_points=True,
        show_data_smooth=True,
        show_model_smooth=True,
        model_line_mode="smooth",
        state_assignment_mode="map",
        figure_dpi=300,
    )

    fig_psychometric, ax_psychometric = plots.plot_categorical_performance_all(
        plot_df_all,
        f"glmhmm K={K}",
        background_style="model",
        views=views_sel,
    )

    fig_psychometric_state, ax_psychometric_state = plt.subplots(figsize=fig_size(2, 1))
    fig_psychometric_state, _ = plots.plot_categorical_performance_state_overlay(
        df=plot_df_all,
        views=views_sel,
        model_name=f"glmhmm K={K} — all states",
        ax=ax_psychometric_state,
        **_state_plot_kwargs,
    )

    fig_psychometric_state_detailed, _ = plots.plot_categorical_performance_by_state(
        df=plot_df_all,
        views=views_sel,
        model_name=f"glmhmm K={K} — per state",
        **_state_plot_kwargs,
    )

    fig_psychometric_state_at, ax_psychometric_state_at = plt.subplots(figsize=fig_size(2, 1))
    fig_psychometric_state_at, _ = plots.plot_regressor_psychometric_by_state(
        df=plot_df_all,
        views=views_sel,
        model_name=f"glmhmm K={K}",
        feature_col=ui_psychometric_regressor.value,
        overlay_only=True,
        ax=ax_psychometric_state_at,
        **_state_plot_kwargs,
    )
    ax_psychometric_state_at.set_xlabel(r"$A$")

    fig_psychometric_state_at_detailed, ax_psychometric_state_at_detailed = plt.subplots(1, K, figsize=(4 * K, 4), sharey=True)
    fig_psychometric_state_at_detailed, _ = plots.plot_regressor_psychometric_by_state(
        df=plot_df_all,
        views=views_sel,
        model_name=f"glmhmm K={K}",
        feature_col=ui_psychometric_regressor.value,
        axes=ax_psychometric_state_at_detailed,
        **_state_plot_kwargs,
    )
    mo.vstack([
        panel("Overall psychometric", fig_psychometric, "psychometric"),

        mo.hstack([panel("State categorical performance", fig_psychometric_state, "categorical_state_overlay"), panel("Per-state categorical performance", fig_psychometric_state_detailed, "categorical_by_state")]),

        mo.hstack([panel("Psychometric by regressor", fig_psychometric_state_at, f"regressor_state_overlay_{ui_psychometric_regressor.value}"), panel(f"{ui_psychometric_regressor.value} by state", fig_psychometric_state_at_detailed, f"regressor_by_state_{ui_psychometric_regressor.value}")]),

    ], align="center")
    return (plot_df_all,)


@app.cell
def _(adapter, fig_size, mo, panel, plot_df_all, plots, plt, views_sel):
    fig_total_evidence, ax_total_evidence = plt.subplots(
        1,
        1,
        figsize=fig_size(2, 1),
        layout="constrained",
    )
    plots.plot_accuracy_by_total_evidence(
        plot_df_all,
        adapter=adapter,
        views=views_sel,
        ax=ax_total_evidence,
        figsize=fig_size(2, 1),
    )
    ax_total_evidence.set_xlabel("Fitted Evidence")

    fig_repeat_evidence, ax_repeat_evidence = plt.subplots(1, 1, figsize=fig_size(2, 1), layout="constrained")
    plots.plot_repeat_by_repeat_evidence(
        plot_df_all,
        views=views_sel,
        ax=ax_repeat_evidence,
        figsize=fig_size(2, 1),
    )
    ax_repeat_evidence.set_xlabel("Rep. Evidence")

    mo.hstack([
        panel(
            "Accuracy by fitted evidence",
            fig_total_evidence,
            "accuracy_total_evidence",
        ),
        panel(
            "Repeat probability by repeat evidence",
            fig_repeat_evidence,
            "repeat_probability_repeat_evidence",
        ),
    ], align="center")
    return


@app.cell
def _(fig_size, mo, panel, plot_df_all, plots, plt):
    regressor = (
        "choice_lag_param_2"
        if "choice_lag_param_2" in plot_df_all.columns
        else (
            "choice_lag_param"
            if "choice_lag_param" in plot_df_all.columns
            else None
        )
    )
    mo.stop(
        regressor is None,
        mo.md("No choice-history regressor available for p(right) by regressor."),
    )

    fig, ax = plt.subplots(figsize=fig_size(2, 1))
    plots.plot_right_by_regressor(plot_df_all, regressor_col=regressor, title=None, ax=ax)
    ax.set_xlabel(r"$A$")

    panel("p(right) by choice history", fig, f"right_by_{regressor}", "p(right) by choice history")
    return (regressor,)


@app.cell
def _(fig_size, mo, panel, plot_df_all, plots, plt, regressor):
    mo.stop(
        regressor is None,
        mo.md("No choice-history regressor available for p(right) by regressor."),
    )
    map_w, map_h = fig_size(2, 1)

    fig_data, (ax_data, cax_data) = plt.subplots(
        1, 2,
        figsize=(map_w * 1.12, map_h),
        gridspec_kw={"width_ratios": [1, 0.08], "wspace": 0.12},
    )
    plots.plot_right_integration_map(
        plot_df_all,
        panel="data",
        ax=ax_data,
        cbar_ax=cax_data,
    )
    ax_data.set_ylabel(r"$A$")

    fig_model, (ax_model, cax_model) = plt.subplots(
        1, 2,
        figsize=(map_w * 1.12, map_h),
        gridspec_kw={"width_ratios": [1, 0.08], "wspace": 0.12},
    )
    plots.plot_right_integration_map(
        plot_df_all,
        panel="model",
        ax=ax_model,
        cbar_ax=cax_model,
    )

    mo.hstack(
        [
            panel(
                "p(right) integration map (data)",
                fig_data,
                "right_integration_map_data",
                "p(right) integration map data",
            ),
            panel(
                "p(right) integration map (model)",
                fig_model,
                "right_integration_map_model",
                "p(right) integration map model",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(
    adapter,
    fig_size,
    mo,
    panel,
    plot_df_all,
    plots,
    plt,
    regressor,
    views_sel,
):
    mo.stop(
        regressor is None,
        mo.md("No choice-history regressor available for p(right) by regressor."),
    )
    bin_w, bin_h = fig_size(2, 1)
    fig_binned, (ax_binned, ax_binned_legend) = plt.subplots(
        1,
        2,
        figsize=(bin_w * 1.45, bin_h),
        gridspec_kw={"width_ratios": [1, 0.45], "wspace": 0.02},
    )

    plots.plot_binned_accuracy_figure(
        plot_df_all,
        regressor_col=regressor,
        adapter=adapter,
        views=views_sel,
        max_panels=1,
        ax=ax_binned,
        legend_ax=ax_binned_legend,
    )

    if ax_binned_legend.get_legend():
        ax_binned_legend.get_legend().set_title(r"$A$")

    panel(
        "Binned accuracy by choice history",
        fig_binned,
        f"accuracy_binned_{regressor}",
        "binned accuracy by choice history",
    )
    return


@app.cell
def _():
    from src.process.common import build_action_trace_model_prediction_rb
    from src.plots.common import (
        plot_action_trace_parameter_fixed_lag_match,
        plot_action_trace_parameter_fixed_rb,
        plot_action_trace_parameter_fixed_subject_scatter,
    )

    return (
        build_action_trace_model_prediction_rb,
        plot_action_trace_parameter_fixed_lag_match,
        plot_action_trace_parameter_fixed_rb,
        plot_action_trace_parameter_fixed_subject_scatter,
    )


@app.cell
def _(mo, task_name):
    ui_glmhmm_model_rb_max_lag = mo.ui.slider(
        start=1,
        stop=15,
        step=1,
        value=10,
        label="Max history lag",
    )
    _supported = task_name in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}
    (
        mo.hstack([ui_glmhmm_model_rb_max_lag], justify="start")
        if _supported
        else mo.md("Trial-level model RB is implemented for 2AFC and 2ADC.")
    )
    return (ui_glmhmm_model_rb_max_lag,)


@app.cell
def _(
    build_action_trace_model_prediction_rb,
    mo,
    panel,
    plot_action_trace_parameter_fixed_lag_match,
    plot_action_trace_parameter_fixed_rb,
    plot_action_trace_parameter_fixed_subject_scatter,
    plot_df_all,
    task_name,
    ui_glmhmm_model_rb_max_lag,
):
    mo.stop(task_name not in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"})

    summary, lag_summary, subject_scatter, meta = build_action_trace_model_prediction_rb(
        plot_df_all,
        task_name=task_name,
        max_history_lag=int(ui_glmhmm_model_rb_max_lag.value),
    )

    mo.stop(summary.empty, mo.md("No trial-level model repetition-bias result."))

    fig_model_rb, ax_model_rb = plot_action_trace_parameter_fixed_rb(summary, meta)
    fig_model_lag, ax_model_lag = plot_action_trace_parameter_fixed_lag_match(lag_summary, meta)
    fig_model_scatter, ax_model_scatter = plot_action_trace_parameter_fixed_subject_scatter(subject_scatter, meta)

    mo.hstack(
        [
            panel("Repetition bias", fig_model_rb, "glmhmm_trial_model_rb", "glmhmm trial-level full-model repetition bias"),
            panel("Lag match", fig_model_lag, "glmhmm_trial_model_lag_match", "glmhmm trial-level full-model lag match"),
            panel("By animal", fig_model_scatter, "glmhmm_trial_model_rb_by_animal", "glmhmm trial-level full-model repetition bias by animal"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Editable accuracy plots
    """)
    return


@app.cell
def _(wrap_anywidget):
    from wigglystuff import TangleSlider

    THRESH_ui = wrap_anywidget(
        TangleSlider(
            amount=0.5,
            min_value=0.0,
            max_value=1,
            step=0.01,
            digits=2,
        )
    )
    return (THRESH_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State analysis
    """)
    return


@app.cell
def _(adapter, adapter_behavioral_column, df_all):
    session_col = adapter_behavioral_column(adapter, df_all, "session", "session", "Session")
    trial_col = adapter_behavioral_column(adapter, df_all, "trial_idx", "trial_idx", "trial", "Trial")
    return session_col, trial_col


@app.cell
def _(df_all, pl, session_col, trial_col, trial_df):

    licks_df = (
        trial_df
        .join(
            df_all.select(
                "subject",
                pl.col(session_col).alias("session"),
                pl.col(trial_col).alias("trial_idx"),
                "nLicks",
            ),
            on=["subject", "session", "trial_idx"],
            how="left",
        )
        .with_columns(pl.col("nLicks").cast(pl.Float64, strict=False))
        .drop_nulls(["nLicks", "state_label"])
        .group_by(["subject", "state_label"])
        .agg(pl.median("nLicks").alias("nLicks"))
        .to_pandas()
    )
    return (licks_df,)


@app.cell
def _(
    Annotator,
    adapter,
    build_state_accuracy_payload,
    build_state_posterior_count_payload,
    fig_size,
    licks_df,
    mo,
    model_plots,
    panel,
    plt,
    selected,
    sns,
    trial_df,
):
    mo.stop(not selected, mo.md("No fitted subjects available."))

    fig_acc,ax_acc  = plt.subplots(figsize=fig_size(3, 1))
    model_plots.state_accuracy(
        build_state_accuracy_payload(
            trial_df,
            performance_col="correct_bool",
            chance_level=1.0 / adapter.num_classes,
        ),
        ax=ax_acc,
    )
    ax_acc.set_title("")

    fig_post, ax_post = plt.subplots(figsize=fig_size(3, 1))
    model_plots.state_posterior_count_kde(
        build_state_posterior_count_payload(trial_df),
        ax=ax_post,
        figsize=fig_size(3, 1),
    )
    ax_post.spines["right"].set_visible(True)
    ax_post.set_title("")

    fig_nlicks, ax_nlicks = plt.subplots(figsize=fig_size(3, 1))
    sns.boxplot(
        data=licks_df,
        x="state_label",
        y="nLicks",
        hue="state_label",
        order=["Engaged", "Disengaged"],
        palette={"Engaged": "tab:green", "Disengaged": "tab:gray"},
        fill=False,
        showcaps=False,
        showfliers=False,
        medianprops={"linewidth": 2},
        boxprops={"color": "tab:gray"},
        whiskerprops={"color": "tab:gray"},
        ax=ax_nlicks,
    )

    annotator = Annotator(
        ax_nlicks,
        [("Engaged", "Disengaged")],
        data=licks_df,
        x="state_label",
        y="nLicks",
        order=["Engaged", "Disengaged"],
    )
    annotator.configure(test="t-test_paired", text_format="star", loc="outside", verbose=0, line_height=0)
    annotator.apply_and_annotate()

    ax_nlicks.set(xlabel="State", ylabel="Number of licks in correct trials")
    sns.despine(ax=ax_nlicks)
    fig_nlicks.tight_layout()

    mo.hstack(
        [
            panel("Accuracy by state", fig_acc, "state_accuracy", "accuracy by state"),
            panel("Number of licks by state", fig_nlicks, "state_nlicks", "number of licks by state"),
            panel("Posterior / trial-count KDE", fig_post, "state_posterior_count_kde", "posterior trial-count kde"),
        ],
        align="center",
    )
    return


@app.cell
def _(np, pd):
    # Convert state labels to a binary target:
    # Engaged = True, everything else = False
    def engaged_mask(labels):
        return pd.Series(labels).astype(str).str.lower().str.startswith("engaged").to_numpy()


    # Compute ROC curve and AUC from binary targets and continuous scores
    def roc_curve(target, score):
        target, score = np.asarray(target, bool), np.asarray(score, float)

        # Remove invalid scores
        keep = np.isfinite(score)
        target, score = target[keep], score[keep]

        # Sort by decreasing score
        order = np.argsort(-score, kind="mergesort")
        target = target[order]

        # Indices where score thresholds change
        thresholds = np.r_[np.where(np.diff(score[order]))[0], len(target) - 1]

        # Compute cumulative TP and FP counts
        tp = np.cumsum(target)[thresholds]
        fp = (thresholds + 1) - tp

        # Convert to rates
        tpr = np.r_[0, tp / target.sum()]
        fpr = np.r_[0, fp / (~target).sum()]

        # Area under the ROC curve
        auc = np.trapezoid(tpr, fpr)

        return fpr, tpr, auc

    return engaged_mask, roc_curve


@app.cell
def _(
    engaged_mask,
    fig_size,
    panel,
    pd,
    plot_df_all,
    plt,
    roc_curve,
    trial_df,
):
    # Metrics to evaluate:
    # (column, panel title, legend title, score direction)
    metrics = [
        ("nLicks", "Licking", "Higher lick count", 1),
        ("RT", "RT", "Faster RT", -1),  # negate RT so faster responses predict engagement
    ]


    # Create one ROC panel per behavioral metric
    fig_roc, axes_roc = plt.subplots(
        1,
        len(metrics),
        figsize=fig_size(1, len(metrics)),
        squeeze=False,
        layout="constrained",
    )


    # Attach behavioral variables to trial-level state assignments
    behavior_df = (
        trial_df.join(
            plot_df_all.select("subject", "session", "trial_idx", "nLicks", "RT",),
            on=["subject", "session", "trial_idx"],
            how="left",
        )
        .select("subject", "session","state_label", "nLicks", "RT")
        .to_pandas()
    )


    # Compute and plot ROC curve for each behavioral metric
    for ax_roc, (metric, title, direction, sign) in zip(axes_roc.ravel(), metrics):
        # Keep only trials with valid values for this metric
        df_roc = behavior_df[["state_label", metric]].dropna().copy()

        target = engaged_mask(df_roc["state_label"])
        score = sign * pd.to_numeric(df_roc[metric], errors="coerce").to_numpy(float)

        # Compute ROC and AUC
        fpr, tpr, auc = roc_curve(target, score)

        ax_roc.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], ls="--", color="0.5")
        ax_roc.set(
            title=title,
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )

        ax_roc.legend(
            title=direction,
            frameon=False,
            loc="lower right",
        )


    panel(
        "Behavioral ROC by state",
        fig_roc,
        "state_behavioral_roc",
        "behavioral ROC by state",
    )
    return


@app.cell
def _(mo, plot_df_all):
    def random_session():
        return (
            plot_df_all
            .select(["subject", "session"])
            .unique()
            .sample(n=1)
            .row(0, named=True)
        )
    ui_behavior_random_session = mo.ui.run_button(label="Pick random session")
    return random_session, ui_behavior_random_session


@app.cell
def _():
    return


@app.cell
def _(df_all, mo, pl, plot_df_all, random_session, ui_behavior_random_session):
    if ui_behavior_random_session.value:
        pick = random_session()
        subject_value = str(pick["subject"])
        session_value = str(pick["session"])
    else:
        subject_value = sorted(df_all["subject"].unique())[0]
        session_value = (
            plot_df_all.filter(
                pl.col("subject") == subject_value,
            )
            ["session"][0]
        )
    ui_behavior_session_subj = mo.ui.dropdown(
        options=sorted(plot_df_all["subject"].unique()),
        value=subject_value,
        label="Subject",
    )
    return session_value, subject_value, ui_behavior_session_subj


@app.cell
def _(mo, pl, plot_df_all, session_value, subject_value):
    ui_behavior_session_id = mo.ui.dropdown(
        options=(
            plot_df_all.filter(
                pl.col("subject") == subject_value,
            )["session"]
            .unique()
        ),
        value=session_value,
        label="Session",
    )
    return (ui_behavior_session_id,)


@app.cell
def _():
    return


@app.cell
def _(
    fig_size,
    mo,
    np,
    panel,
    pd,
    plot_df_all,
    plt,
    state_order,
    state_palette,
    ui_behavior_random_session,
    ui_behavior_session_id,
    ui_behavior_session_subj,
):
    session_df = plot_df_all.filter(
        (plot_df_all["subject"] == str(ui_behavior_session_subj.value)),
        (plot_df_all["session"]== str(ui_behavior_session_id.value))
    ).to_pandas()
    def ecdf(values):
        x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
        return x, np.arange(1, len(x) + 1) / len(x)
    _metrics = [("nLicks", "Licks"), ("RT", "RT")]
    fig_ecdf, axes_ecdf = plt.subplots(
        1,
        len(_metrics),
        figsize=fig_size(1, len(_metrics)),
        squeeze=False,
        layout="constrained",
    )

    for _ax_ecdf, (_metric, _title) in zip(axes_ecdf.ravel(), _metrics):
        for _state in state_order:
            _x, _y = ecdf(session_df.loc[session_df["state_label"].astype(str) == _state, _metric])
            _ax_ecdf.step(_x,_y, where="post", label=f"{_state} (n={len(_x)})", color = state_palette[_state])

        _ax_ecdf.set(title=_title, xlabel=_metric, ylabel="Cumulative probability", ylim=(0, 1))
        _ax_ecdf.legend(frameon=False, loc="lower right")

    mo.vstack([
            mo.hstack([ui_behavior_random_session, ui_behavior_session_subj, ui_behavior_session_id]),
            panel(
                "Example session RT/lick cumulative distributions",
                fig_ecdf,
                "example_session_behavior_ecdf_by_state",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(
    build_session_trajectories_payload,
    fig_size,
    mo,
    model_plots,
    panel,
    plt,
    selected,
    trial_df,
):
    mo.stop(not selected, mo.md("Select subjects above to view session trajectories."))
    fig_traj, ax_traj = plt.subplots(figsize=fig_size(1, 2))
    ax_traj = model_plots.session_trajectories(
        build_session_trajectories_payload(
            trial_df,
            session_col="session",
            sort_col="trial_idx",
        ),
        ax=ax_traj,
    )
    panel("Mean engagement", fig_traj, "mean_engagement")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Fractional occupancy & state changes per session
    """)
    return


@app.cell
def _(
    build_state_occupancy_payload,
    fig_size,
    mo,
    model_plots,
    panel,
    plt,
    trial_df,
):
    occupancy_payload = build_state_occupancy_payload(trial_df, session_col="session", sort_col="trial_idx")

    fig_occ_overall, ax_occ_overall = plt.subplots(figsize=fig_size(3, 1))
    model_plots.state_occupancy_overall_summary(occupancy_payload, ax=ax_occ_overall).set_title("")

    fig_occ_sessions, ax_occ_sessions = plt.subplots(figsize=fig_size(3, 1))
    model_plots.state_session_occupancy_summary(occupancy_payload, ax=ax_occ_sessions).set_title("")

    fig_switches, ax_switches = plt.subplots(figsize=fig_size(3, 1))
    model_plots.state_switches_summary(occupancy_payload, ax=ax_switches).set_title("")


    mo.hstack(
        [
            panel("Overall occupancy", fig_occ_overall, "state_occupancy_overall_summary", "fractional occupancy overall summary"),
            panel("Session occupancy", fig_occ_sessions, "state_session_occupancy_summary", "fractional occupancy by session summary"),
            panel("State switches", ax_switches, "state_switches_summary", "state switches summary"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Posteriors around a change
    """)
    return


@app.cell
def _(
    THRESH_ui,
    build_change_triggered_posteriors_payload,
    fig_size,
    mo,
    model_plots,
    panel,
    plt,
    selected,
    trial_df,
):
    mo.stop(not selected, mo.md("Select subjects above."))

    change_payload = build_change_triggered_posteriors_payload(
        trial_df,
        session_col="session",
        sort_col="trial_idx",
        switch_posterior_threshold=THRESH_ui.amount,
    )

    fig_change_base, axes_change = plt.subplots(
        1,
        len(change_payload.get("directions") or ["into_engaged", "out_of_engaged"]),
        figsize=fig_size(1, 3),
    )

    fig_change = model_plots.change_triggered_posteriors_summary(change_payload, axes=axes_change)

    for ax_change in fig_change[0].figure.axes:
        ax_change.set_title("")
        if ax_change.legend_:
            ax_change.legend_.remove()

    fig_change[0].figure.legend(
        handles=[
            plt.Line2D([0], [0], color="tab:green", lw=2, label="Engaged"),
            plt.Line2D([0], [0], color="tab:gray", lw=2, label="Disengaged"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    mo.vstack(
        [
            mo.md(f"> Change events use the same confident MAP switch rule as the histogram above: posterior ≥ {THRESH_ui.amount}."),
            panel(
                "Change-triggered posteriors",
                fig_change[0].figure,
                "change_triggered_posteriors_summary",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(mo, selected):
    _subj_opts = selected if selected else ["(no fitted subjects)"]

    ui_session_subj = mo.ui.dropdown(
        options=_subj_opts,
        value=_subj_opts[0],
        label="Subject",
    )
    return (ui_session_subj,)


@app.cell
def _(mo, pl, trial_df, ui_session_subj, views):
    _sess_opts = (
        sorted(trial_df.filter(pl.col("subject") == ui_session_subj.value)["session"].unique().to_list())
        if ui_session_subj.value in views
        else [0]
    )
    _sess_opts = _sess_opts or [0]
    ui_session_id = mo.ui.dropdown(
        options=[str(s) for s in _sess_opts],
        value=str(_sess_opts[0]),
        label="Session",
    )
    _win_opts = [1, 5, 10, 20, 50]
    ui_engaged_window = mo.ui.dropdown(
        options=[str(w) for w in _win_opts],
        value="20",
        label="P(engaged) window",
    )
    ui_engaged_trace_mode = mo.ui.radio(
        options={
            "Rolling": "rolling",
            "Raw": "raw",
        },
        value="Rolling",
        inline=False,
        label="P(engaged) trace",
    )
    return ui_engaged_trace_mode, ui_engaged_window


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session statistics
    """)
    return


@app.cell
def _(
    adapter,
    build_session_deepdive_payload,
    mo,
    model_plots,
    panel,
    put_figure_legend_at_bottom,
    trial_df,
    ui_behavior_random_session,
    ui_behavior_session_id,
    ui_behavior_session_subj,
    ui_engaged_trace_mode,
    ui_engaged_window,
    views,
):
    subject = ui_behavior_session_subj.value
    session = int(ui_behavior_session_id.value) if str(ui_behavior_session_id.value).isdigit() else ui_behavior_session_id.value

    mo.stop(subject not in views, mo.md("No fitted arrays for this subject — run the fit first."))

    deepdive_payload = build_session_deepdive_payload(
        trial_df,
        subject=subject,
        session=session,
        session_col="session",
        sort_col="trial",
        engaged_window=ui_engaged_window.value,
        engaged_trace_mode=ui_engaged_trace_mode.value,
        chance_level=1.0 / adapter.num_classes,
        num_classes=adapter.num_classes,
        views=views,
    )

    fig_deepdive = model_plots.session_deepdive(deepdive_payload)
    fig_traces = model_plots.session_deepdive_state_traces(deepdive_payload)

    put_figure_legend_at_bottom(fig_deepdive, bottom=0.18)
    put_figure_legend_at_bottom(fig_traces, bottom=0.28)

    mo.vstack(
        [
            mo.hstack(
                [
                    ui_behavior_random_session,
                    ui_behavior_session_subj,
                    ui_behavior_session_id,
                    ui_engaged_window,
                    ui_engaged_trace_mode,
                ],
                align="center",
            ),
            panel("Session statistics", fig_deepdive, f"session_stats_{subject}_{session}", "session statistics"),
            panel("Session state traces", fig_traces, f"session_state_traces_{subject}_{session}", "session state traces"),
        ],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
