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
    from src.utils import fig_size

    return (
        Annotator,
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        adapter_behavioral_column,
        build_change_triggered_posteriors_payload,
        build_editor_payload,
        build_session_deepdive_payload,
        build_session_trajectories_payload,
        build_state_accuracy_payload,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_state_posterior_count_payload,
        build_transition_matrix_payload,
        build_trial_and_weights_df,
        build_trial_df,
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
        select_subject_behavior_df,
        sns,
        wrap_anywidget,
    )


@app.cell
def _(plt, project_root):
    plt.style.use(project_root / "styles" / "paper.mplstyle")
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
    return editor_views, state_labels, views


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

    (
        panel(
            "Emission weights",
            emissions_ax.figure,
            "emissions",
            "emissions boxplot",
        ),
    )
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
def _(fig_size, mo, np, pd, plot_df_all, plt, save_plot):

    _stim_col = (
        "stim_x_delay"
        if "stim_x_delay" in plot_df_all.columns
        else "ILD"
        if "ILD" in plot_df_all.columns
        else "ild"
    )
    mo.stop(
        _stim_col not in plot_df_all.columns or "choice_lag_param" not in plot_df_all.columns,
        mo.md("Need `choice_lag_param` and raw `ILD`/`stim_x_delay` columns for conditional p(right) plots."),
    )

    def _response_right(_df):
        _response = pd.to_numeric(_df["response"], errors="coerce")
        if next(iter(_views_sel.values())).num_classes == 3:
            return (_response == 2).astype(float)
        return (_response > 0).astype(float)

    def _summary(_df, *, x_col, line_col, x_quantile=False, line_quantile=False):
        _df = _df.to_pandas().copy() if hasattr(_df, "to_pandas") else pd.DataFrame(_df).copy()
        _df["_response_right"] = _response_right(_df)
        _df["_x"] = pd.to_numeric(_df[x_col], errors="coerce")
        _df["_line_value"] = pd.to_numeric(_df[line_col], errors="coerce")
        _df["_p_right"] = pd.to_numeric(_df["pR"], errors="coerce")
        _df = _df.dropna(subset=["subject", "_response_right", "_x", "_line_value", "_p_right"])
        if _df.empty or _df["_x"].nunique() < 2 or _df["_line_value"].nunique() < 2:
            return None
        _df["_line"] = (
            pd.qcut(_df["_line_value"], q=min(4, _df["_line_value"].nunique()), labels=False, duplicates="drop")
            if line_quantile
            else _df["_line_value"]
        )
        _df["_xbin"] = (
            pd.qcut(_df["_x"], q=min(10, _df["_x"].nunique()), labels=False, duplicates="drop")
            if x_quantile
            else _df["_x"]
        )
        _df = _df.dropna(subset=["_line", "_xbin"])
        _subject = (
            _df.groupby(["_line", "subject", "_xbin"], observed=True)
            .agg(data=("_response_right", "mean"), model=("_p_right", "mean"), x=("_x", "mean"))
            .reset_index()
        )
        _out = (
            _subject.groupby(["_line", "_xbin"], observed=True)
            .agg(data_mean=("data", "mean"), data_std=("data", "std"), n=("data", "count"), model_mean=("model", "mean"), x=("x", "mean"))
            .reset_index()
            .sort_values(["_line", "x"])
        )
        _out["data_sem"] = _out["data_std"].fillna(0) / np.sqrt(_out["n"].clip(lower=1))
        return _out

    def _plot(_ax, _summary_df, *, legend_title, line_quantile, palette_name):
        if _summary_df is None or _summary_df.empty:
            _ax.set_axis_off()
            return False
        _order = sorted(_summary_df["_line"].dropna().unique().tolist())
        _colors = plt.get_cmap(palette_name)(np.linspace(0.15, 0.85, len(_order)))
        for _line, _color in zip(_order, _colors, strict=False):
            _sub = _summary_df[_summary_df["_line"] == _line]
            _label = f"Q{int(_line) + 1}" if line_quantile else f"{float(_line):g}"
            _ax.plot(_sub["x"], _sub["model_mean"], "-", color=_color, lw=2, label=_label)
            _ax.errorbar(_sub["x"], _sub["data_mean"], yerr=_sub["data_sem"], fmt="o", color=_color, ecolor=_color, ms=4, capsize=3)
        _ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel(r"$p(\mathrm{right})$")
        _ax.legend(title=legend_title, frameon=False, fontsize=8)
        return True

    _fig_conditional, (_ax_stim_by_a, _ax_a_by_stim) = plt.subplots(1, 2, figsize=fig_size(1, 2), layout="constrained")
    _drawn_1 = _plot(
        _ax_stim_by_a,
        _summary(plot_df_all, x_col=_stim_col, line_col="choice_lag_param", line_quantile=True),
        legend_title=r"$A$",
        line_quantile=True,
        palette_name="RdBu",
    )
    _ax_stim_by_a.set_xlabel(_stim_col)
    _drawn_2 = _plot(
        _ax_a_by_stim,
        _summary(plot_df_all, x_col="choice_lag_param", line_col=_stim_col, x_quantile=True),
        legend_title="Stim.",
        line_quantile=False,
        palette_name="viridis",
    )
    _ax_a_by_stim.set_xlabel(r"$A$")
    mo.stop(not (_drawn_1 or _drawn_2), mo.md("No conditional p(right) plots could be drawn."))

    mo.vstack(
        [
            _fig_conditional,
            save_plot(
                _fig_conditional,
                "conditional psychometrics by action and stimulus",
                stem="pright_conditional_action_stimulus",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(
    fig_size,
    mo,
    np,
    pd,
    plot_df_all,
    plt,
    save_plot,
    selected,
    sns,
    task_name,
    views,
):
    from glmhmmt.views import get_state_color
    from src.process.common import attach_repeat_choice_evidence

    _views_sel = {s: views[s] for s in selected}
    mo.stop(not _views_sel, mo.md("No fitted arrays found -- run the fit first."))

    _repeat_state_df = attach_repeat_choice_evidence(
        plot_df_all,
        views=_views_sel,
        is_mcdr=task_name == "MCDR",
    )
    mo.stop(_repeat_state_df.empty, mo.md("No repeat evidence data available."))
    mo.stop(
        "subject" not in _repeat_state_df.columns or "_repeat_choice_evidence" not in _repeat_state_df.columns,
        mo.md("No repeat evidence data available for the selected task/features."),
    )
    mo.stop(
        "state_idx" not in _repeat_state_df.columns and "state_rank" not in _repeat_state_df.columns,
        mo.md("No state assignment column available for repeat evidence distributions."),
    )

    _repeat_state_df = _repeat_state_df.copy().reset_index(drop=True)
    _repeat_state_df["_repeat_choice_evidence"] = pd.to_numeric(
        _repeat_state_df["_repeat_choice_evidence"],
        errors="coerce",
    )

    if "state_rank" in _repeat_state_df.columns:
        _repeat_state_df["_state_rank"] = pd.to_numeric(_repeat_state_df["state_rank"], errors="coerce")
    else:
        _repeat_state_df["_state_rank"] = np.nan
        for _subject, _idx in _repeat_state_df.groupby("subject", observed=True).groups.items():
            _view = _views_sel.get(_subject) or _views_sel.get(str(_subject))
            if _view is None:
                continue
            _rank_by_raw = {int(_raw_idx): int(_rank) for _raw_idx, _rank in _view.state_rank_by_idx.items()}
            _raw_state = pd.to_numeric(_repeat_state_df.loc[_idx, "state_idx"], errors="coerce")
            _repeat_state_df.loc[_idx, "_state_rank"] = _raw_state.map(_rank_by_raw).to_numpy(dtype=float)

    _repeat_state_df = _repeat_state_df.dropna(subset=["_repeat_choice_evidence", "_state_rank"]).copy()
    mo.stop(_repeat_state_df.empty, mo.md("No finite repeat evidence values available by state."))

    _repeat_state_df["_state_rank"] = _repeat_state_df["_state_rank"].astype(int)
    _state_labels = {}
    for _view in _views_sel.values():
        for _raw_idx, _label in _view.state_name_by_idx.items():
            _rank = int(_view.state_rank_by_idx.get(int(_raw_idx), int(_raw_idx)))
            _state_labels.setdefault(_rank, _label)

    _state_order = sorted(_repeat_state_df["_state_rank"].unique().tolist())
    _label_by_rank = {_rank: _state_labels.get(_rank, f"State {_rank}") for _rank in _state_order}
    _repeat_state_df["_state_label"] = _repeat_state_df["_state_rank"].map(_label_by_rank)
    _label_order = [_label_by_rank[_rank] for _rank in _state_order]
    _K = int(next(iter(_views_sel.values())).K) if _views_sel else len(_state_order)
    _palette = {
        _label_by_rank[_rank]: get_state_color(_label_by_rank[_rank], _rank, K=_K)
        for _rank in _state_order
    }

    _fig_repeat_state_dist, _ax_repeat_state_dist = plt.subplots(
        figsize=fig_size(3, 1),
        layout="constrained",
    )
    _kde_drawn = False
    for _label in _label_order:
        _values = _repeat_state_df.loc[
            _repeat_state_df["_state_label"] == _label,
            "_repeat_choice_evidence",
        ].to_numpy(dtype=float)
        _values = _values[np.isfinite(_values)]
        if _values.size >= 2 and np.nanstd(_values) > 0:
            sns.kdeplot(
                x=_values,
                ax=_ax_repeat_state_dist,
                label=_label,
                color=_palette[_label],
                fill=False,
                linewidth=1.8,
            )
            _kde_drawn = True

    if not _kde_drawn:
        sns.stripplot(
            data=_repeat_state_df,
            x="_repeat_choice_evidence",
            y="_state_label",
            order=_label_order,
            palette=_palette,
            ax=_ax_repeat_state_dist,
            size=2.5,
            alpha=0.45,
        )
        _ax_repeat_state_dist.set_ylabel("")
    else:
        _ax_repeat_state_dist.legend(title="State", frameon=False)
        _ax_repeat_state_dist.set_ylabel("Density")

    _ax_repeat_state_dist.axvline(0, color="black", linestyle="--", linewidth=0.8)
    _ax_repeat_state_dist.set_xlabel("Rep. Evidence")
    _ax_repeat_state_dist.set_title("Repeat evidence distribution by state")
    sns.despine(ax=_ax_repeat_state_dist)

    mo.vstack(
        [
            _fig_repeat_state_dist,
            save_plot(
                _fig_repeat_state_dist,
                "repeat evidence distribution by state",
                stem="repeat_evidence_distribution_by_state",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(fig_size, mo, plot_df_all, plots, plt, save_plot):
    _choice_history_regressor = (
        "choice_lag_param_2"
        if "choice_lag_param_2" in plot_df_all.columns
        else "choice_lag_param"
    )
    mo.stop(
        _choice_history_regressor not in plot_df_all.columns,
        mo.md("No choice-history regressor available for p(right) by regressor."),
    )

    fig_right, ax_right = plt.subplots(figsize=fig_size(2,1))
    _ax_right_regressor = plots.plot_right_by_regressor(
        plot_df_all,
        regressor_col=_choice_history_regressor,
        title=None,
        ax=ax_right,
    )
    _ax_right_regressor.set_xlabel(r"$A$")
    mo.stop(
        _ax_right_regressor is None,
        mo.md("No p(right) by choice-history regressor plot available."),
    )

    mo.vstack(
        [
            fig_right,
            save_plot(
                fig_right,
                "p(right) by choice history",
                stem=f"right_by_{_choice_history_regressor}",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(fig_size, mo, plot_df_all, plots, plt, save_plot):
    mo.stop(
        not hasattr(plots, "plot_right_integration_map"),
        mo.md("No p(right) integration map helper is available for this task."),
    )
    _map_w, _map_h = fig_size(2, 1)
    _fig_integration_data_base, (_ax_integration_data, _cax_integration_data) = plt.subplots(
        1,
        2,
        figsize=(_map_w * 1.12, _map_h),
        gridspec_kw={"width_ratios": [1.0, 0.08], "wspace": 0.12},
    )
    _integration_data_result = plots.plot_right_integration_map(
        plot_df_all,
        panel="data",
        ax=_ax_integration_data,
        cbar_ax=_cax_integration_data,
    )
    _ax_integration_data.set_ylabel(r"$A$")
    _fig_integration_data = None if _integration_data_result is None else _integration_data_result[0]

    _fig_integration_model_base, (_ax_integration_model, _cax_integration_model) = plt.subplots(
        1,
        2,
        figsize=(_map_w * 1.12, _map_h),
        gridspec_kw={"width_ratios": [1.0, 0.08], "wspace": 0.12},
    )
    _integration_model_result = plots.plot_right_integration_map(
        plot_df_all,
        panel="model",
        ax=_ax_integration_model,
        cbar_ax=_cax_integration_model,
    )
    _fig_integration_model = None if _integration_model_result is None else _integration_model_result[0]

    mo.stop(
        _fig_integration_data is None and _fig_integration_model is None,
        mo.md("No p(right) integration map available for the selected task/features."),
    )

    _integration_items = []
    if _fig_integration_data is not None:
        _integration_items.extend(
            [
                _fig_integration_data,
                save_plot(
                    _fig_integration_data,
                    "p(right) integration map data",
                    stem="right_integration_map_data",
                ),
            ]
        )
    if _fig_integration_model is not None:
        _integration_items.extend(
            [
                _fig_integration_model,
                save_plot(
                    _fig_integration_model,
                    "p(right) integration map model",
                    stem="right_integration_map_model",
                ),
            ]
        )

    mo.vstack(
        _integration_items,
        align="center",
    )
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
    selected,
    views,
):
    _choice_history_regressor = (
        "choice_lag_param_2"
        if "choice_lag_param_2" in plot_df_all.columns
        else "choice_lag_param"
    )
    mo.stop(
        _choice_history_regressor not in plot_df_all.columns,
        mo.md("No choice-history regressor available for binned accuracy."),
    )
    _views_sel = {s: views[s] for s in selected}
    _bin_w, _bin_h = fig_size(2, 1)
    _fig_binned_base, (_ax_binned, _ax_binned_legend) = plt.subplots(
        1,
        2,
        figsize=(_bin_w * 1.45, _bin_h),
        gridspec_kw={"width_ratios": [1.0, 0.45], "wspace": 0.02},
    )
    _binned_result = plots.plot_binned_accuracy_figure(
        plot_df_all,
        regressor_col=_choice_history_regressor,
        adapter=adapter,
        views=_views_sel,
        max_panels=1,
        ax=_ax_binned,
        legend_ax=_ax_binned_legend,
    )
    _binned_legend = _ax_binned_legend.get_legend()
    if _binned_legend is not None:
        _binned_legend.set_title(r"$A$")

    _fig_binned = None if _binned_result is None else _binned_result[0]
    mo.stop(
        _fig_binned is None,
        mo.md("No binned accuracy plot available for the choice-history regressor."),
    )

    mo.vstack(
        [
            _fig_binned,
            save_plot(
                _fig_binned,
                "binned accuracy by choice history",
                stem=f"accuracy_binned_{_choice_history_regressor}",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    ui_run_glmhmm_autocorr_simulations = mo.ui.run_button(
        label="Run fitted GLM-HMM autocorrelogram simulations",
    )
    ui_glmhmm_autocorr_n_simulations = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=5,
        label="GLM-HMM simulations",
    )
    mo.hstack([ui_run_glmhmm_autocorr_simulations, ui_glmhmm_autocorr_n_simulations], align="center")
    return ui_glmhmm_autocorr_n_simulations, ui_run_glmhmm_autocorr_simulations


@app.cell
def _(
    fig_size,
    mo,
    plot_df_all,
    plt,
    save_plot,
    ui_glmhmm_autocorr_n_simulations,
    ui_run_glmhmm_autocorr_simulations,
):
    from src.process.common import (
        prepare_corrected_behavior_autocorrelograms,
        prepare_model_simulated_corrected_behavior_autocorrelograms,
    )
    from src.plots.common import plot_corrected_behavior_autocorrelograms

    _trial_col = "trial_idx" if "trial_idx" in plot_df_all.columns else ("trial" if "trial" in plot_df_all.columns else None)
    _autocorr_sort_cols = ["subject", "session"] + ([_trial_col] if _trial_col is not None else [])
    _autocorr_df = plot_df_all.sort(_autocorr_sort_cols)
    _data_autocorr = prepare_corrected_behavior_autocorrelograms(
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
    if ui_run_glmhmm_autocorr_simulations.value:
        _model_autocorr = prepare_model_simulated_corrected_behavior_autocorrelograms(
            _autocorr_df,
            subject_col="subject",
            session_col="session",
            trial_index_col=_trial_col,
            response_col="response",
            performance_col="performance",
            n_simulations=int(ui_glmhmm_autocorr_n_simulations.value),
            max_lag=50,
            min_cross_pairs=20,
            max_cross_pairs=80,
            seed=1,
        )
        _model_autocorr_df = _model_autocorr["autocorr"]
    else:
        _model_autocorr_df = None
    def _style_autocorr_axis(_ax):
        _ax.set_ylim(-0.075, 0.2)
        _ax.set_title("")

    _fig_choice_autocorr, _ax_choice_autocorr = plt.subplots(figsize=fig_size(2, 1.5))
    _fig_choice_autocorr, _ = plot_corrected_behavior_autocorrelograms(
        _data_autocorr,
        axes=[_ax_choice_autocorr],
        model_autocorr=_model_autocorr_df,
        model_label="Fitted GLM-HMM",
        signals=("Outcome",),
        figsize=fig_size(2, 1),
    )
    _style_autocorr_axis(_ax_choice_autocorr)
    _fig_repeat_autocorr, _ax_repeat_autocorr = plt.subplots(figsize=fig_size(2, 1.25))
    _fig_repeat_autocorr, _ = plot_corrected_behavior_autocorrelograms(
        _data_autocorr,
        axes=[_ax_repeat_autocorr],
        model_autocorr=_model_autocorr_df,
        model_label="Fitted GLM-HMM",
        signals=("Repetition",),
        figsize=fig_size(2, 1.25),
    )
    _style_autocorr_axis(_ax_repeat_autocorr)
    mo.hstack(
        [
            mo.vstack([_fig_choice_autocorr, save_plot(_fig_choice_autocorr, "GLM-HMM choice autocorrelogram", stem="glmhmm_choice_autocorrelogram")], align="center"),
            mo.vstack([_fig_repeat_autocorr, save_plot(_fig_repeat_autocorr, "GLM-HMM repeat autocorrelogram", stem="glmhmm_repeat_autocorrelogram")], align="center"),
        ],
        align="center",
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
    plot_action_trace_parameter_fixed_lag_match,
    plot_action_trace_parameter_fixed_rb,
    plot_action_trace_parameter_fixed_subject_scatter,
    plot_df_all,
    save_plot,
    task_name,
    ui_glmhmm_model_rb_max_lag,
):
    mo.stop(
        task_name not in {"2AFC", "2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"},
    )

    _summary, _lag_summary, _subject_scatter, _meta = build_action_trace_model_prediction_rb(
        plot_df_all,
        task_name=task_name,
        max_history_lag=int(ui_glmhmm_model_rb_max_lag.value),
    )
    mo.stop(
        _summary.empty,
        mo.md("No trial-level model repetition-bias result; check pR/p_pred and previous choices."),
    )

    _fig_model_rb, _ax_model_rb = plot_action_trace_parameter_fixed_rb(
        _summary,
        _meta,
    )
    _fig_model_lag, _ax_model_lag = plot_action_trace_parameter_fixed_lag_match(
        _lag_summary,
        _meta,
    )
    _fig_model_scatter, _ax_model_scatter = plot_action_trace_parameter_fixed_subject_scatter(
        _subject_scatter,
        _meta,
    )

    mo.vstack(
        [
            mo.md("#### Trial-level full-model repetition bias"),
            _fig_model_rb,
            save_plot(
                _fig_model_rb,
                "glmhmm trial-level full-model repetition bias",
                stem="glmhmm_trial_model_rb",
            ),
            _fig_model_lag,
            save_plot(
                _fig_model_lag,
                "glmhmm trial-level full-model lag match",
                stem="glmhmm_trial_model_lag_match",
            ),
            _fig_model_scatter,
            save_plot(
                _fig_model_scatter,
                "glmhmm trial-level full-model repetition bias by animal",
                stem="glmhmm_trial_model_rb_by_animal",
            ),
            mo.md(
                "Full fitted uses each trial's inferred model P(right), compares it with the same animal's empirical previous choice, "
                "and aggregates RB conditional on previous choice side within animal. No refit or choice simulation is run."
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(editor_views, mo):
    _subjects = sorted(editor_views.keys(), key=str)
    mo.stop(not _subjects, mo.md("No fitted subjects available for coefficient editing."))
    ui_editor_subject = mo.ui.dropdown(
        options=_subjects,
        value=_subjects[0],
        label="Subject",
    )
    ui_editor_subject
    return (ui_editor_subject,)


@app.cell
def _(editor_views, mo, ui_editor_subject):
    _view = editor_views[ui_editor_subject.value]
    _state_options = [f"{_k} — {_view.state_name_by_idx.get(_k, f'State {_k}')}" for _k in _view.state_idx_order]
    ui_editor_state = mo.ui.dropdown(
        options=_state_options,
        value=_state_options[0],
        label="State",
    )
    ui_editor_state
    return (ui_editor_state,)


@app.cell
def _(adapter, mo):
    if adapter.num_classes != 2:
        ui_editor_side = None
    else:
        _choices = [str(label) for label in adapter.choice_labels]
        ui_editor_side = mo.ui.dropdown(
            options=_choices,
            value=_choices[0],
            label="Side",
        )
    ui_editor_side
    return (ui_editor_side,)


@app.cell
def _(
    CoefficientEditorWidget,
    adapter,
    build_editor_payload,
    editor_views,
    mo,
    np,
    ui_editor_side,
    ui_editor_state,
    ui_editor_subject,
    wrap_anywidget,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]
    coef_state_idx = int(ui_editor_state.value.split(" — ", 1)[0])
    coef_state_label = _view.state_name_by_idx.get(coef_state_idx, f"State {coef_state_idx}")
    _stored_weights = np.asarray(_view.emission_weights[coef_state_idx], dtype=float)
    _choice_labels = [str(label) for label in adapter.choice_labels]
    _stored_class_indices = list(range(_view.num_classes - 1))
    _reference_class_idx = _view.num_classes - 1
    if _view.num_classes == 2 and ui_editor_side is not None:
        _display_class_idx = _choice_labels.index(ui_editor_side.value)
        _display_reference_class_idx = next(idx for idx in range(_view.num_classes) if idx != _display_class_idx)
    else:
        _display_reference_class_idx = 1 if _view.num_classes == 3 else _reference_class_idx
    _payload = build_editor_payload(
        _stored_weights,
        choice_labels=_choice_labels,
        stored_class_indices=_stored_class_indices,
        reference_class_idx=_reference_class_idx,
        display_reference_class_idx=_display_reference_class_idx,
    )

    coef_editor = wrap_anywidget(
        CoefficientEditorWidget(
            title="Coefficient editor",
            subtitle=_payload["subtitle"],
            features=list(_view.feat_names),
            channel_labels=_payload["channel_labels"],
            weights=_payload["weights"].tolist(),
            original_weights=_payload["weights"].tolist(),
            slider_min=-6.0,
            slider_max=6.0,
            slider_step=0.05,
        )
    )
    _controls = [ui_editor_subject, ui_editor_state]
    if ui_editor_side is not None:
        _controls.append(ui_editor_side)

    coef_editor_panel = mo.vstack(
        [
            mo.md("### Interactive coefficient editor"),
            mo.md(
                "Only the selected state's emission coefficients are edited. "
                "The overall and per-state categorical plots update using the edited state."
            ),
            mo.hstack(_controls),
            coef_editor,
        ],
        align="center",
    )
    coef_editor_panel
    coef_editor_explicit_class_indices = _payload["explicit_class_indices"]
    coef_editor_reference_class_idx = _payload["reference_class_idx"]
    coef_editor_stored_class_indices = _payload["stored_class_indices"]
    coef_editor_stored_reference_class_idx = _payload["stored_reference_class_idx"]
    return


@app.cell
def _(
    adapter,
    build_trial_df,
    df_all,
    editor_views,
    mo,
    select_subject_behavior_df,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]

    _df_sub = select_subject_behavior_df(
        df_all,
        subject=_subj,
        sort_col=adapter.sort_col,
        session_col=adapter.session_col,
        min_session_length=2,
    )
    mo.stop(_df_sub.height != _view.T, mo.md(f"Subject {_subj} does not match the loaded fit arrays."))
    editor_trial_df = build_trial_df(_view, adapter, _df_sub, adapter.behavioral_cols)
    editor_view = _view
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
def _(adapter, adapter_behavioral_column, df_all, pl, trial_df):
    session_col = adapter_behavioral_column(adapter, df_all, "session", "session", "Session")
    trial_col = adapter_behavioral_column(adapter, df_all, "trial_idx", "trial_idx", "trial", "Trial")

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
    return licks_df, session_col, trial_col


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
    df_all,
    engaged_mask,
    fig_size,
    panel,
    pd,
    pl,
    plt,
    roc_curve,
    session_col,
    trial_col,
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
            df_all.select(
                "subject",
                pl.col(session_col).alias("session"),
                pl.col(trial_col).alias("trial_idx"),
                "nLicks",
                "RT",
            ),
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
    return (behavior_df,)


@app.cell
def _(behavior_df, mo):
    def random_session():
        return (
            behavior_df[["subject", "session"]]
            .dropna()
            .drop_duplicates()
            .sample(1)
            .iloc[0]
        )
    ui_behavior_random_session = mo.ui.run_button(label="Pick random session")
    return random_session, ui_behavior_random_session


@app.cell
def _(behavior_df, mo, random_session, ui_behavior_random_session):
    if ui_behavior_random_session.value:
        pick = random_session()
        subject_value = str(pick["subject"])
        session_value = str(pick["session"])
    else:
        subject_value = sorted(behavior_df["subject"].astype(str).unique())[0]
        session_value = (
            behavior_df.loc[
                behavior_df["subject"].astype(str) == subject_value,
                "session"
            ]
            .astype(str)
            .sort_values()
            .iloc[0]
        )
    ui_behavior_session_subj = mo.ui.dropdown(
        options=sorted(behavior_df["subject"].astype(str).unique()),
        value=subject_value,
        label="Subject",
    )
    return session_value, ui_behavior_session_subj


@app.cell
def _(behavior_df, mo, session_value, ui_behavior_session_subj):
    ui_behavior_session_id = mo.ui.dropdown(
        options=(
            behavior_df.loc[
                behavior_df["subject"].astype(str) == ui_behavior_session_subj.value,
                "session",
            ]
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        ),
        value=session_value,
        label="Session",
    )
    return (ui_behavior_session_id,)


@app.cell
def _(
    behavior_df,
    fig_size,
    mo,
    np,
    pd,
    plt,
    save_plot,
    state_order,
    state_palette,
    ui_behavior_random_session,
    ui_behavior_session_id,
    ui_behavior_session_subj,
):
    session_df = behavior_df[
        (behavior_df["subject"].astype(str) == str(ui_behavior_session_subj.value))
        & (behavior_df["session"].astype(str) == str(ui_behavior_session_id.value))
    ]
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
        mo.md("#### Example session RT/lick cumulative distributions"),
        mo.hstack([ui_behavior_random_session, ui_behavior_session_subj, ui_behavior_session_id]),
        fig_ecdf,
        save_plot(fig_ecdf, "example session behavioral ECDF by state", stem="example_session_behavior_ecdf_by_state"),
    ], align="center")
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
