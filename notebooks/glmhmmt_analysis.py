import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

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
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        from glmhmmt.cli.fit_glmhmmt import main as fit_main
        _FITTING_AVAILABLE = True
    except ImportError:
        fit_main = None
        _FITTING_AVAILABLE = False
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import (
        build_change_triggered_posteriors_payload,
        build_emission_weights_df,
        build_session_deepdive_payload,
        build_session_trajectories_payload,
        build_state_accuracy_payload,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_state_posterior_count_payload,
        build_trial_df,
        build_transition_matrix_by_subject_payload,
        build_transition_matrix_payload,
        build_transition_weights_df,
    )
    import glmhmmt.plots as model_plots
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_afc_drug as process_two_afc_drug
    from src.process import two_adc as process_two_adc
    from src.process import two_adc_drug as process_two_adc_drug
    from src.process.common import add_choice_lag_summary_regressor

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    # Set boxplot_tick_rotation to 35 to restore rotated labels.
    # Set boxplot_fixed_panel to False to let matplotlib choose panel bounds.
    boxplot_tick_rotation = 0
    boxplot_fixed_panel = True
    from src.utils import fig_size
    boxplot_figsize = fig_size(2,1)
    boxplot_panel_bounds = (0.16, 0.22, 0.80, 0.70)

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

    def format_boxplot_panel(ax):
        ax.figure.set_size_inches(*boxplot_figsize, forward=True)
        plt.setp(
            ax.get_xticklabels(),
            rotation=boxplot_tick_rotation,
            ha="right" if boxplot_tick_rotation else "center",
        )
        if boxplot_fixed_panel:
            ax.set_position(boxplot_panel_bounds)

    def make_boxplot_axis():
        fig, ax = plt.subplots(figsize=boxplot_figsize)
        if boxplot_fixed_panel:
            ax.set_position(boxplot_panel_bounds)
        return fig, ax

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    sns.set_style("ticks")
    sns.set_context("paper")
    paths = get_runtime_paths()
    from src.utils import fig_size

    return (
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        Path,
        add_choice_lag_summary_regressor,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        boxplot_figsize,
        boxplot_tick_rotation,
        build_change_triggered_posteriors_payload,
        build_editor_payload,
        build_emission_weights_df,
        build_session_deepdive_payload,
        build_session_trajectories_payload,
        build_state_accuracy_payload,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_state_posterior_count_payload,
        build_transition_matrix_payload,
        build_transition_weights_df,
        build_trial_and_weights_df,
        build_trial_df,
        build_views,
        fig_size,
        fit_main,
        format_boxplot_panel,
        get_adapter,
        load_fit_arrays,
        make_boxplot_axis,
        make_plot_saver,
        mo,
        model_plots,
        np,
        paths,
        pd,
        pl,
        plt,
        prepare_predictions_df,
        put_figure_legend_at_bottom,
        put_legend_inside_panel,
        resolve_selected_model_id,
        select_subject_behavior_df,
        sns,
        wrap_anywidget,
    )


@app.cell
def _(Path, plt):
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    return


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glmhmmt",
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
    ## GLM-HMM-T configuration
    """)
    return


@app.cell
def _(get_adapter, mo, model_cfg):
    task_name = model_cfg.task
    adapter = get_adapter(task_name)
    mo.stop(
        model_cfg.condition_filter != "all",
        mo.md(
            "Condition filters are not yet wired through `fit_glmhmmt`; set the condition filter to **all** "
            "so the fitted arrays and plotted dataframe use the same trials."
        ),
    )
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    df_all = adapter.filter_condition_df(df_all, model_cfg.condition_filter)
    is_2afc = adapter.num_classes == 2
    plots = adapter.get_plots()
    return adapter, df_all, is_2afc, plots, task_name


@app.cell
def _(get_adapter, model_cfg, task_name):
    from glmhmmt.cli.fit_glmhmmt import generate_model_id as _gen_id

    baseline_class_idx = int(get_adapter(task_name).baseline_class_idx)
    current_hash = _gen_id(
        task=task_name,
        K=model_cfg.K,
        tau=model_cfg.tau,
        emission_cols=model_cfg.emission_cols or None,
        transition_cols=list(model_cfg.transition_cols),
        frozen_emissions=model_cfg.frozen_emissions or None,
        baseline_class_idx=baseline_class_idx,
        cv_mode=model_cfg.cv_mode,
        cv_repeats=model_cfg.cv_repeats,
    )
    return baseline_class_idx, current_hash


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
        model_id=f"glmhmmt/{selected_model_id}",
    )
    return save_plot, selected_model_id


@app.cell
def _(current_hash, mo, save_plot, ui_model_manager):
    mo.vstack(
        [
            ui_model_manager,
            save_plot.save_all_widget(label="Save all GLM-HMM-T plots"),
            mo.md(f"**Current params hash:** `{current_hash}`"),
        ],
        align="center",
    )
    return


@app.cell
def _(
    baseline_class_idx,
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
    mo.stop(fit_main is None, mo.md("`glmhmmt.cli.fit_glmhmmt` is not available in this environment."))
    set_last_fit_click(model_cfg.run_fit_clicks)

    _n_restarts = 1
    _cv_repeats = int(model_cfg.cv_repeats) if model_cfg.cv_mode != "none" else 0
    _selected_id = model_cfg.existing or (model_cfg.alias if model_cfg.alias else current_hash)
    _OUT = paths.RESULTS / "fits" / task_name / "glmhmmt" / _selected_id

    def _progress_title(info: dict) -> str:
        return f"Fitting GLM-HMM-T K={info['K']} subject {info['subject_index']}/{info['subject_total']}: {info['subject']}"

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
            title=f"Fitting GLM-HMM-T K={model_cfg.K}",
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
                emission_cols=model_cfg.emission_cols or None,
                transition_cols=list(model_cfg.transition_cols),
                frozen_emissions=model_cfg.frozen_emissions or None,
                tau=model_cfg.tau,
                task=task_name,
                cv_mode=model_cfg.cv_mode,
                cv_repeats=_cv_repeats,
                n_restarts=_n_restarts,
                verbose=False,
                baseline_class_idx=baseline_class_idx,
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
    OUT = paths.RESULTS / "fits" / task_name / "glmhmmt" / selected_model_id
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glmhmmt_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=list(model_cfg.subjects),
        emission_cols=list(model_cfg.emission_cols) or None,
        transition_cols=list(model_cfg.transition_cols),
        k=K,
    )
    selected = [s for s in model_cfg.subjects if s in arrays_store]
    return K, arrays_store, selected


@app.cell
def _(K, adapter, arrays_store, build_views, mo, model_cfg, selected):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    adapter.state_scoring_feature = model_cfg.state_scoring_feature or None
    adapter.state_scoring_rule = model_cfg.state_scoring_rule or "+"
    adapter.state_split_feature = model_cfg.state_split_feature or None
    adapter.state_split_rule = model_cfg.state_split_rule or "+"
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
    return (trial_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Emission weights
    """)
    return


@app.cell
def _(
    K,
    boxplot_tick_rotation,
    build_emission_weights_df,
    format_boxplot_panel,
    make_boxplot_axis,
    mo,
    model_plots,
    np,
    pd,
    pl,
    put_legend_inside_panel,
    save_plot,
    selected,
    views,
):
    from glmhmmt.plots.emissions import (
        _emission_boxplot_payload,
        _fold_three_choice_raw_weights,
        _prepare_weights_df,
    )

    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _views_sel = {s: views[s] for s in selected}
    _weights_df = build_emission_weights_df(_views_sel)

    _feature_labels = {
        "stim_param": r"$\mathrm{Stim}_{\mathrm{param}}$",
        "stim_x_delay_param": r"$\mathrm{Stim:delay}_{\mathrm{param}}$",
        "drug_x_stim_param": r"$\mathrm{NMDAr}\times\mathrm{Stim}_{\mathrm{param}}$",
        "drug_x_stim_x_delay_param": r"$\mathrm{NMDAr}\times\mathrm{Stim:delay}_{\mathrm{param}}$",
        "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "bias": r"$\mathrm{Bias}$",
        "prev_choice": r"$\mathrm{Choice}_{t-1}$",
        "at_choice_param": r"$\mathrm{A}_t$",
        "choice_lag_param": r"$\mathrm{A}$",
        "drug_x_choice_lag_param": r"$\mathrm{NMDAr}\times\mathrm{A}$",
        "trial_index": r"$\mathrm{TrialIndex}$",
        "current_choice": r"$\mathrm{Choice}_t$",
        "current_stim_side": r"$\mathrm{StimSide}_t$",
        "current_reward": r"$\mathrm{Rew}_t$",
        "current_abs_stim": r"$|\mathrm{Stim}_t|$",
        "current_abs_delay": r"$|\mathrm{Delay}_t|$",
        "prev_abs_stim" :  r"$\mathrm{Stim}_{t-1}$",
        "prev_reward" :  r"$\mathrm{Rew}_{t-1}$",
        "cumulative_reward" :  r"$\sum_0^{t}\mathrm{Rew}$",
        "filtered_choice": r"filtered choice",
        "filtered_reward": r"filtered reward",
        "filtered_stim_side": r"filtered stim side",
        "drug_code" : "NMDAr hypofunction" ,
        "Drug" : "NMDAr hypofunction" 

    }
    feature_labeler = lambda feature: _feature_labels.get(str(feature), str(feature))
    _available_features = _weights_df.get_column("feature").unique().to_list()
    _preferred_feature_order = []
    for _candidates in (
        ["bias", "bias_param", "biasparam"],
        ["stim_param", "stim_vals", "stim_x_delay_param"],
        ["drug_x_stim_param", "drug_x_stim_x_delay_param"],
        ["at_choice_param", "choice_lag_param", "at_choice", "prev_choice"],
        ["drug_x_choice_lag_param"],
    ):
        _match = next((_feature for _feature in _candidates if _feature in _available_features), None)
        if _match is not None:
            _preferred_feature_order.append(_match)

    _fig_by_subject, _ = model_plots.emission_weights_by_subject(
        _weights_df,
        K=K,
        feature_labeler=feature_labeler,
    )
    _summary_weights_df = _weights_df.filter(pl.col("feature").str.contains("bias_(/d)").not_())
    _summary_fig, _summary_ax = make_boxplot_axis()
    _summary_ax = model_plots.emission_weights_summary_boxplot(
        _summary_weights_df,
        connect_subjects=True,
        show_ttests=True,
        feature_order=_preferred_feature_order,
        feature_labeler=feature_labeler,
        ax=_summary_ax,
        tick_rotation=boxplot_tick_rotation,
    )
    put_legend_inside_panel(_summary_ax, anchor=(0.98,0.2))
    format_boxplot_panel(_summary_ax)
    _summary_fig = _summary_ax.figure
    _plot_weights = _fold_three_choice_raw_weights(_summary_weights_df)
    if _plot_weights is None:
        _plot_weights = _summary_weights_df
    _df, _features, _, _state_order, _ = _prepare_weights_df(
        _plot_weights,
        K=K,
        feature_order=_preferred_feature_order,
        feature_labeler=feature_labeler,
    )
    _, _subject_lines = _emission_boxplot_payload(
        _df,
        features=_features,
        states=_state_order,
    )
    _subjects = sorted(pd.unique(_df["subject"]).tolist())
    _n_states = len(_state_order)
    _hue_width = 0.8 / max(1, _n_states)
    emission_summary_selection_points = []
    for _feat_idx, _feature in enumerate(_features):
        _positions = [
            _feat_idx + (_state_idx - (_n_states - 1) / 2.0) * _hue_width
            for _state_idx in range(_n_states)
        ]
        for _subject, _ys in zip(_subjects, _subject_lines[_feat_idx], strict=False):
            _ys = np.asarray(_ys, dtype=float)
            _valid_idx = np.flatnonzero(np.isfinite(_ys))
            if _valid_idx.size < 2:
                continue
            _split_points = np.where(np.diff(_valid_idx) > 1)[0] + 1
            for _segment in np.split(_valid_idx, _split_points):
                if _segment.size < 2:
                    continue
                _x_segment = [float(_positions[_idx]) for _idx in _segment]
                _y_segment = [float(_ys[_idx]) for _idx in _segment]
                for _left in range(len(_x_segment) - 1):
                    for _x, _y in zip(
                        np.linspace(_x_segment[_left], _x_segment[_left + 1], 24),
                        np.linspace(_y_segment[_left], _y_segment[_left + 1], 24),
                        strict=False,
                    ):
                        emission_summary_selection_points.append(
                            {
                                "subject": str(_subject),
                                "feature": str(_feature),
                                "feature_label": str(feature_labeler(_feature)),
                                "x": float(_x),
                                "y": float(_y),
                            }
                        )
    ui_emission_summary = mo.ui.matplotlib(_summary_ax, debounce=True)

    # _summary_ax = model_plots.emission_weights_summary_lineplot(
    #     _weights_df.filter(pl.col("feature").str.contains("bias_(/d)").not_()),
    #     feature_order=_preferred_feature_order,
    #     feature_labeler=feature_labeler,
    # )
    # _summary_fig = _summary_ax.figure
    emmissions_summary = _summary_fig
    mo.vstack(
        [
            # ui_emission_summary,
            _summary_fig,
            mo.hstack(
                [
                    save_plot(_summary_fig, "Emission weights summary", stem="emissions_summary"),
                    save_plot(_fig_by_subject, "Emission weights by subject", stem="emissions_by_subject"),
                ],
                gap="15",
            ),
        ],
        align="center",
    )
    return (
        emission_summary_selection_points,
        emmissions_summary,
        feature_labeler,
        ui_emission_summary,
    )


@app.cell
def _(emission_summary_selection_points, mo, pd, ui_emission_summary):
    _points = pd.DataFrame(emission_summary_selection_points)
    if _points.empty or not ui_emission_summary.value:
        selected_emission_subjects = []
    else:
        _mask = ui_emission_summary.value.get_mask(
            _points["x"].to_numpy(),
            _points["y"].to_numpy(),
        )
        selected_emission_subjects = sorted(_points.loc[_mask, "subject"].unique().tolist())
    mo.md("Selected subjects: " + (", ".join(selected_emission_subjects) if selected_emission_subjects else "_none_"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Transition weights and matrices
    """)
    return


@app.cell
def _(
    K,
    boxplot_tick_rotation,
    build_transition_weights_df,
    feature_labeler,
    format_boxplot_panel,
    make_boxplot_axis,
    mo,
    model_plots,
    put_legend_inside_panel,
    save_plot,
    selected,
    views,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    mo.stop(
        not any(getattr(views[s], "transition_weights", None) is not None for s in selected),
        mo.md("No transition weights found — run a GLM-HMM-T fit with transition regressors."),
    )
    _views_sel = {s: views[s] for s in selected}
    _weights_df = build_transition_weights_df(_views_sel)
    _transition_feature_order = ["filtered_choice", "filtered_stim_side", "filtered_reward"]
    _fig_by_subject, _ = model_plots.transition_weights_by_subject(
        _weights_df,
        K=K,
    )
    _summary_fig, _summary_ax = make_boxplot_axis()
    _summary_ax = model_plots.transition_weights_summary_boxplot(
        _weights_df,
        feature_order=_transition_feature_order,
        connect_subjects=False,
        show_ttests=True,
        feature_labeler=feature_labeler,
        ax=_summary_ax,
        tick_rotation=boxplot_tick_rotation,
    )
    put_legend_inside_panel(_summary_ax)
    format_boxplot_panel(_summary_ax)
    _summary_fig = _summary_ax.figure

    # _summary_ax = model_plots.transition_weights_summary_lineplot(
    #     _weights_df,
    #     feature_labeler=feature_labeler,
    # )
    # _summary_fig = _summary_ax.figure


    mo.vstack(
        [
            _summary_fig,
            mo.hstack(
                [
                    save_plot(_summary_fig, "Transition weights summary", stem="transition_weights_summary"),
                    save_plot(_fig_by_subject, "Transition weights by subject", stem="transition_weights_by_subject"),
                ],
                gap="15",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(build_transition_weights_df, selected, views):
    _views_sel = {s: views[s] for s in selected}
    _weights_df = build_transition_weights_df(_views_sel)
    # _weights_df
    return


@app.cell
def _(
    K,
    adapter,
    boxplot_tick_rotation,
    build_transition_weights_df,
    emmissions_summary,
    feature_labeler,
    format_boxplot_panel,
    make_boxplot_axis,
    mo,
    model_plots,
    pl,
    put_legend_inside_panel,
    save_plot,
    selected,
    views,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    mo.stop(
        not any(getattr(views[s], "transition_weights", None) is not None for s in selected),
        mo.md("No transition weights found — run a GLM-HMM-T fit with transition regressors."),
    )
    _views_sel = {s: views[s] for s in selected}
    _weights_df = build_transition_weights_df(_views_sel)
    _transition_feature_order = ["filtered_choice", "filtered_stim_side", "filtered_reward"]
    _fig_by_subject, _ = model_plots.transition_weights_by_subject(
        _weights_df,
        K=K,
    )
    _summary_fig, _summary_ax = make_boxplot_axis()
    _summary_ax = model_plots.transition_weights_summary_boxplot(
        _weights_df,
        feature_order=_transition_feature_order,
        connect_subjects=True,
        show_ttests=True,
        feature_labeler=feature_labeler,
        ax=_summary_ax,
        tick_rotation=boxplot_tick_rotation,
    )
    put_legend_inside_panel(_summary_ax)
    format_boxplot_panel(_summary_ax)
    _summary_fig = _summary_ax.figure
    _features = _weights_df.get_column("feature").unique().to_list()
    _transition_groups = (
        adapter.build_transition_groups(_features)
        if hasattr(adapter, "build_transition_groups")
        else []
    )
    _family_fig_rows = []
    for _group in _transition_groups:
        print(_group)
        _members = list(_group.get("toggle_members") or _group.get("members", {}).values())
        _members = [str(_member) for _member in _members if str(_member) in _features]
        if len(_members) <= 1:
            continue
        _family_df = _weights_df.filter(pl.col("feature").is_in(_members))
        if _family_df.height == 0:
            continue
        _family_key = str(_group.get("key", "transition_family"))
        _family_label = str(_group.get("label", _family_key))
        _family_fig, _family_ax = make_boxplot_axis()
        _family_ax = model_plots.transition_weights_summary_boxplot(
            _family_df,
            feature_order=[
                _feature
                for _feature in _transition_feature_order
                if _feature in _members
            ],
            connect_subjects=True,
            show_ttests=True,
            feature_labeler=feature_labeler,
            ax=_family_ax,
            tick_rotation=boxplot_tick_rotation,
        )
        put_legend_inside_panel(_family_ax)
        format_boxplot_panel(_family_ax)
        _family_fig = _family_ax.figure
        _family_fig.suptitle(_family_label)
        _family_fig_rows.append(
            mo.vstack(
                [
                    mo.md(f"#### {_family_label}"),
                    _family_fig,
                    save_plot(
                        _family_fig,
                        f"Transition weights: {_family_label}",
                        stem=f"transition_weights_{_family_key}",
                    ),
                ],
                align="center",
            )
        )

    mo.vstack(
        [
            mo.hstack([emmissions_summary,_summary_fig], align = "center"),
            mo.hstack(
                [
                    save_plot(_summary_fig, "Transition weights summary", stem="transition_weights_summary"),
                    save_plot(_fig_by_subject, "Transition weights by subject", stem="transition_weights_by_subject"),
                ],
                gap="15",
            ),
            *_family_fig_rows,
        ],
        align="center",
    )
    return


@app.cell
def _(
    K,
    arrays_store,
    build_transition_matrix_payload,
    fig_size,
    mo,
    model_plots,
    plt,
    save_plot,
    selected,
    state_labels,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    # _by_subject_payload = build_transition_matrix_by_subject_payload(
    #     arrays_store=arrays_store,
    #     state_labels=state_labels,
    #     K=K,
    #     subjects=selected,
    # )
    # _fig_by_subject, _ = model_plots.transition_matrix_by_subject(**_by_subject_payload)
    _summary_payload = build_transition_matrix_payload(

        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    _fig_summary, _ax_summary = plt.subplots(figsize=fig_size(2,1))
    _summary_ax = model_plots.transition_matrix(**_summary_payload, ax=_ax_summary)
    _summary_ax.set_title("")
    _fig_summary = _summary_ax.figure
    mo.vstack(
        [
            _fig_summary,
            save_plot(_fig_summary, "Mean transition matrix", stem="mean_transition_matrix"),
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


@app.cell(disabled=True)
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
        _ax.set_ylim(-0.1, 0.3)
        _ax.set_title("")

    _fig_choice_autocorr, _ax_choice_autocorr = plt.subplots(figsize=fig_size(2, 1))
    _fig_choice_autocorr, _ = plot_corrected_behavior_autocorrelograms(
        _data_autocorr,
        axes=[_ax_choice_autocorr],
        model_autocorr=_model_autocorr_df,
        model_label="Fitted GLM-HMM",
        signals=("Outcome",),
        figsize=fig_size(2, 1),
    )
    _style_autocorr_axis(_ax_choice_autocorr)
    _fig_repeat_autocorr, _ax_repeat_autocorr = plt.subplots(figsize=fig_size(2, 1))
    _fig_repeat_autocorr, _ = plot_corrected_behavior_autocorrelograms(
        _data_autocorr,
        axes=[_ax_repeat_autocorr],
        model_autocorr=_model_autocorr_df,
        model_label="Fitted GLM-HMM",
        signals=("Repetition",),
        figsize=fig_size(2, 1),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State dwell times
    """)
    return


@app.cell
def _(
    boxplot_figsize,
    build_state_dwell_times_payload,
    fig_size,
    format_boxplot_panel,
    mo,
    model_plots,
    pl,
    plt,
    save_plot,
    selected,
    trial_df,
    views,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _views_sel = {s: views[s] for s in selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    _dwell_payload = build_state_dwell_times_payload(
        _trial_df_sel,
        session_col="session",
        sort_col="trial_idx",
        views=_views_sel,
        max_dwell=None,
    )
    _dwell_ax_size = fig_size(2, 1)
    _n_dwell_states = max(1, len(_dwell_payload.get("state_order") or []))
    _n_dwell_subjects = max(1, len(_dwell_payload.get("subject_order") or []))
    _fig_dwell_summary, _axes_dwell_summary = plt.subplots(
        1,
        _n_dwell_states,
        figsize=(_dwell_ax_size[0] * _n_dwell_states, _dwell_ax_size[1]),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    _fig_dwell_summary, _axes_dwell_summary = model_plots.state_dwell_times_summary(
        _dwell_payload,
        axes=_axes_dwell_summary,
    )
    _fig_dwell_cumulative, _ax_dwell_cumulative = plt.subplots(figsize=_dwell_ax_size)
    _fig_dwell_cumulative, _ax_dwell_cumulative = model_plots.state_dwell_times_cumulative(
        _dwell_payload,
        ax=_ax_dwell_cumulative,
    )
    _fig_dwell_median, _ax_dwell_median = plt.subplots(figsize=boxplot_figsize)
    _fig_dwell_median, _ax_dwell_median = model_plots.state_dwell_median_boxplot(
        _dwell_payload,
        ax=_ax_dwell_median,
    )
    format_boxplot_panel(_ax_dwell_median)
    _fig_dwell_by_subject, _axes_dwell_by_subject = plt.subplots(
        _n_dwell_subjects,
        _n_dwell_states,
        figsize=(_dwell_ax_size[0] * _n_dwell_states, _dwell_ax_size[1] * _n_dwell_subjects),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    # _fig_dwell_by_subject, _axes_dwell_by_subject = model_plots.state_dwell_times_by_subject(
    #     _dwell_payload,
    #     axes=_axes_dwell_by_subject,
    # )
    mo.vstack(
        [
            _fig_dwell_summary,
            save_plot(_fig_dwell_summary, "state dwell times summary", stem="state_dwell_times_summary"),
            mo.hstack([
            mo.vstack([_fig_dwell_cumulative, save_plot(_fig_dwell_cumulative, "state dwell times cumulative", stem="state_dwell_times_cumulative")], align="center"),
            mo.vstack([_fig_dwell_median, save_plot(_fig_dwell_median, "state dwell median boxplot", stem="state_dwell_median_boxplot")], align="center"),
            ])
            # _fig_dwell_by_subject,
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    ui_psychometric_background = mo.ui.radio(
        options={"Data traces": "data", "Model curves": "model", "None": "none"},
        value="Model curves",
        inline=False,
        label="Psychometric background",
    )
    ui_state_show_weighted_points = mo.ui.checkbox(value=True, label="Weighted dots")
    ui_state_show_data_smooth = mo.ui.checkbox(value=False, label="Data smooth")
    ui_state_assignment_mode = mo.ui.radio(
        options={"Predictive weights": "weighted", "MAP state": "map"},
        value="MAP state",
        inline=False,
        label="State assignment",
    )
    ui_state_model_line_mode = mo.ui.radio(
        options={"Smooth curve": "smooth", "Trial-matched": "trial_matched", "None": "none"},
        value="Smooth curve",
        inline=False,
        label="Model line",
    )
    return (
        ui_psychometric_background,
        ui_state_assignment_mode,
        ui_state_model_line_mode,
        ui_state_show_data_smooth,
        ui_state_show_weighted_points,
    )


@app.cell
def _(
    adapter,
    add_choice_lag_summary_regressor,
    is_2afc,
    mo,
    pl,
    prepare_predictions_df,
    selected,
    task_name,
    trial_df,
    views,
):
    _feature_names = []
    if is_2afc and views:
        for _view in views.values():
            for _feat in list(getattr(_view, "feat_names", []) or []):
                if _feat not in _feature_names:
                    _feature_names.append(_feat)
    if is_2afc and selected:
        _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
        if _trial_df_sel.height:
            _plot_df_preview = prepare_predictions_df(task_name, _trial_df_sel)
            _choice_lag_cols = []
            for _view in views.values():
                for _feat in list(getattr(_view, "feat_names", []) or []):
                    _feat = str(_feat)
                    if (
                        _feat.startswith("choice_lag_")
                        and _feat.removeprefix("choice_lag_").isdigit()
                        and _feat not in _choice_lag_cols
                    ):
                        _choice_lag_cols.append(_feat)
            if not _choice_lag_cols and hasattr(adapter, "choice_lag_cols"):
                _choice_lag_cols = adapter.choice_lag_cols(_trial_df_sel)
            _plot_df_preview = add_choice_lag_summary_regressor(
                _plot_df_preview,
                choice_lag_cols=_choice_lag_cols,
            )
            _plot_columns = set(_plot_df_preview.columns)
            for _feat in (
                "choice_lag_one_hot_sum",
                "choice_lag_param",
                "at_choice_param",
                "at_choice",
                "stim_x_delay_param",
                "stim_param",
                "bias_param",
            ):
                if _feat in _plot_columns and _feat not in _feature_names:
                    _feature_names.append(_feat)
    if not _feature_names:
        _feature_names = ["at_choice"]
    _default_feature = next(
        (_feat for _feat in ("choice_lag_one_hot_sum", "choice_lag_param", "at_choice_param", "at_choice") if _feat in _feature_names),
        _feature_names[0],
    )
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
    adapter,
    add_choice_lag_summary_regressor,
    fig_size,
    is_2afc,
    mo,
    pl,
    plots,
    plt,
    prepare_predictions_df,
    save_plot,
    selected,
    task_name,
    trial_df,
    ui_psychometric_background,
    ui_psychometric_regressor,
    ui_state_assignment_mode,
    ui_state_model_line_mode,
    ui_state_show_data_smooth,
    ui_state_show_weighted_points,
    views,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _views_sel = {s: views[s] for s in selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    plot_df_all = prepare_predictions_df(task_name, _trial_df_sel)
    _choice_lag_cols = []
    for _view in _views_sel.values():
        for _feat in list(getattr(_view, "feat_names", []) or []):
            _feat = str(_feat)
            if (
                _feat.startswith("choice_lag_")
                and _feat.removeprefix("choice_lag_").isdigit()
                and _feat not in _choice_lag_cols
            ):
                _choice_lag_cols.append(_feat)
    if not _choice_lag_cols and hasattr(adapter, "choice_lag_cols"):
        _choice_lag_cols = adapter.choice_lag_cols(_trial_df_sel)
    plot_df_all = add_choice_lag_summary_regressor(
        plot_df_all,
        choice_lag_cols=_choice_lag_cols,
    )
    _perf_kwargs = {"views": _views_sel} if is_2afc else {}
    _state_legend_loc = "upper left"
    _state_legend_bbox_to_anchor = (1.01, 1.05)

    def _set_state_legend_location(_fig):
        for _ax in _fig.axes:
            _legend = _ax.get_legend()
            if _legend is None:
                continue
            _handles = getattr(_legend, "legend_handles", getattr(_legend, "legendHandles", []))
            _labels = [_text.get_text() for _text in _legend.get_texts()]
            _ax.legend(
                _handles,
                _labels,
                fontsize=8,
                frameon=_legend.get_frame_on(),
                loc=_state_legend_loc,
                bbox_to_anchor=_state_legend_bbox_to_anchor,
            )

    _fig_all, _ = plots.plot_categorical_performance_all(
        plot_df_all,
        f"glmhmmt K={K}",
        background_style=ui_psychometric_background.value,
        **_perf_kwargs,
    )
    _fig_all_list = list(_fig_all) if isinstance(_fig_all, (list, tuple)) else [_fig_all]
    for _fig in _fig_all_list:
        for _ax_idx, _ax in enumerate(_fig.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if _fig._suptitle is not None:
            _fig._suptitle.set_text("")
        _fig.tight_layout()

    _state_overlay_fn = getattr(plots, "plot_categorical_performance_state_overlay", None)
    if _state_overlay_fn is None:
        _fig_state_overlay, _ = plots.plot_categorical_performance_by_state(
            df=plot_df_all,
            views=_views_sel,
            model_name=f"glmhmmt K={K} — all states",
            background_style="none",
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            overlay_only=True,
        )
        _set_state_legend_location(_fig_state_overlay)
    else:
        _fig_state_overlay_base, _ax_state_overlay = plt.subplots(figsize=(3, 3))
        _fig_state_overlay, _ = _state_overlay_fn(
            df=plot_df_all,
            views=_views_sel,
            model_name=f"glmhmmt K={K} — all states",
            background_style="none",
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            ax=_ax_state_overlay,
        )
        _set_state_legend_location(_fig_state_overlay)
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=plot_df_all,
        views=_views_sel,
        model_name=f"glmhmmt K={K} — per state",
        background_style="none",
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
    )
    _set_state_legend_location(_fig_state)
    _state_plot_kwargs = dict(
        background_style="none",
        show_weighted_points=True,
        show_data_smooth=False,
        show_model_smooth=True,
        model_line_mode="smooth",
        state_assignment_mode="map",
        figure_dpi=300,
    )
    _fig_state_overlay, _ax_state_overlay = plt.subplots(figsize=fig_size(2, 1))
    _fig_state_overlay, _ = plots.plot_categorical_performance_state_overlay(
        df=plot_df_all,
        views=_views_sel,
        model_name=f"glmhmm K={K} — all states",
        ax=_ax_state_overlay,
        **_state_plot_kwargs,
    )
    _fig_reg_overlay, _ax_reg_overlay = plt.subplots(figsize=fig_size(2, 1))
    _fig_reg_overlay, _ = plots.plot_regressor_psychometric_by_state(
        df=plot_df_all,
        views=_views_sel,
        model_name=f"glmhmm K={K}",
        feature_col=ui_psychometric_regressor.value,
        overlay_only=True,
        ax=_ax_reg_overlay,
        **_state_plot_kwargs,
    )
    _ax_reg_overlay.set_xlabel(plots.display_regressor_name(ui_psychometric_regressor.value))

    _reg_plot_fn = getattr(plots, "plot_regressor_psychometric_by_state", None)
    if is_2afc and _reg_plot_fn is not None:
        _fig_reg_state, _ = _reg_plot_fn(
            df=plot_df_all,
            views=_views_sel,
            model_name=f"glmhmmt K={K}",
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
        )
        _reg_section = mo.vstack(
            [
                mo.hstack([mo.md("#### Per-state psychometric by regressor"), ui_psychometric_regressor], justify="space-between"),
                mo.vstack(
                    [
                        _fig_reg_state,
                        save_plot(
                            _fig_reg_state,
                            f"{ui_psychometric_regressor.value} by state",
                            stem=f"regressor_by_state_{ui_psychometric_regressor.value}",
                        ),
                    ],
                    align="center",
                ),
            ],
            align="center",
        )
    else:
        _reg_section = mo.md("This task does not expose a regressor psychometric plot.")

    mo.vstack(
        [
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
                                    f"overall psychometric {_fig_idx}" if len(_fig_all_list) > 1 else "overall psychometric",
                                    stem=f"categorical_overall_{_fig_idx}" if len(_fig_all_list) > 1 else "categorical_overall",
                                ),
                            )
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            mo.hstack([ui_psychometric_background, ui_state_model_line_mode], align="end"),
                            ui_state_show_weighted_points,
                            ui_state_show_data_smooth,
                            ui_state_assignment_mode,
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
            ),
            mo.md("#### State categorical performance — all states"),
            mo.vstack(
                [
                    _fig_reg_overlay,
                    save_plot(_fig_reg_overlay, "state-overlay psychometric", stem="categorical_regressor_state_overlay"),
                    _fig_state_overlay,
                    save_plot(_fig_state_overlay, "state-overlay psychometric", stem="categorical_state_overlay"),
                ],
                align="center",
            ),
            mo.md("#### Per-state categorical performance"),
            mo.vstack([_fig_state, save_plot(_fig_state, "per-state psychometric", stem="categorical_by_state")], align="center"),
            _reg_section,
        ],
        align="center",
    )
    return (plot_df_all,)


@app.cell
def _(adapter, fig_size, mo, np, pd, plot_df_all, plots, plt, save_plot, selected, views):
    _views_sel = {s: views[s] for s in selected}
    _evidence_figsize = fig_size(3, 1)

    def _response_right(_df):
        _response = pd.to_numeric(_df["response"], errors="coerce")
        if next(iter(_views_sel.values())).num_classes == 3:
            return (_response == 2).astype(float)
        return (_response > 0).astype(float)

    def _plot_pright_by_evidence(_df, *, group_col, legend_title, ax, discrete=False):
        _df = _df.to_pandas().copy() if hasattr(_df, "to_pandas") else pd.DataFrame(_df).copy()
        if group_col not in _df.columns or "pR" not in _df.columns:
            ax.set_axis_off()
            return None
        _df["_response_right"] = _response_right(_df)
        _df["_group"] = pd.to_numeric(_df[group_col], errors="coerce")
        _df["_p_right"] = pd.to_numeric(_df["pR"], errors="coerce")
        _df["_right_evidence"] = np.log(np.clip(_df["_p_right"], 1e-6, 1 - 1e-6) / np.clip(1 - _df["_p_right"], 1e-6, 1))
        _df = _df.dropna(subset=["_response_right", "_group", "_p_right", "_right_evidence"])
        if _df.empty or _df["_group"].nunique() < 2:
            ax.set_axis_off()
            return None
        _df["_line"] = _df["_group"] if discrete else pd.qcut(_df["_group"], q=4, labels=False, duplicates="drop")
        _df["_xbin"] = pd.qcut(_df["_right_evidence"], q=10, labels=False, duplicates="drop")
        _df = _df.dropna(subset=["_line", "_xbin"]).copy()
        _subject = (
            _df.groupby(["_line", "subject", "_xbin"], observed=True)
            .agg(data=("_response_right", "mean"), model=("_p_right", "mean"), x=("_right_evidence", "mean"))
            .reset_index()
        )
        _summary = (
            _subject.groupby(["_line", "_xbin"], observed=True)
            .agg(data_mean=("data", "mean"), data_std=("data", "std"), n=("data", "count"), model_mean=("model", "mean"), x=("x", "mean"))
            .reset_index()
            .sort_values(["_line", "x"])
        )
        _summary["data_sem"] = _summary["data_std"].fillna(0) / np.sqrt(_summary["n"].clip(lower=1))
        _order = sorted(_summary["_line"].dropna().unique().tolist())
        _colors = plt.get_cmap("viridis" if discrete else "RdBu")(np.linspace(0.15, 0.85, len(_order)))
        for _line, _color in zip(_order, _colors, strict=False):
            _sub = _summary[_summary["_line"] == _line]
            _label = f"{float(_line):g}" if discrete else f"Q{int(_line) + 1}"
            ax.plot(_sub["x"], _sub["model_mean"], "-", color=_color, lw=2, label=_label)
            ax.errorbar(_sub["x"], _sub["data_mean"], yerr=_sub["data_sem"], fmt="o", color=_color, ecolor=_color, ms=4, capsize=3)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Right-vs-rest fitted evidence")
        ax.set_ylabel(r"$p(\mathrm{right})$")
        ax.set_ylim(0, 1)
        ax.legend(title=legend_title, frameon=False, fontsize=8)
        return ax

    _fig_total_evidence, _ax_total_evidence = plt.subplots(
        1,
        1,
        figsize=_evidence_figsize,
        layout="constrained",
    )
    plots.plot_accuracy_by_total_evidence(
        plot_df_all,
        adapter=adapter,
        views=_views_sel,
        ax=_ax_total_evidence,
        figsize=_evidence_figsize,
    )
    _ax_total_evidence.set_xlabel("Fitted Evidence")

    _stimulus_group_col = "stim_x_delay" if "stim_x_delay" in plot_df_all.columns else "ILD"
    _pright_evidence_panels = []
    for _group_col, _legend_title, _stem_label, _discrete in [
        ("choice_lag_param", r"$A$", "action_trace", False),
        (_stimulus_group_col, "Stim.", "stimulus", True),
    ]:
        _fig_pright_evidence, _ax_pright_evidence = plt.subplots(
            1,
            1,
            figsize=_evidence_figsize,
            layout="constrained",
        )
        _plotted_pright = _plot_pright_by_evidence(
            plot_df_all,
            group_col=_group_col,
            legend_title=_legend_title,
            ax=_ax_pright_evidence,
            discrete=_discrete,
        )
        if _plotted_pright is None:
            plt.close(_fig_pright_evidence)
            continue
        _pright_evidence_panels.append(
            mo.vstack(
                [
                    _fig_pright_evidence,
                    save_plot(
                        _fig_pright_evidence,
                        f"p(right) by total evidence binned by {_stem_label}",
                        stem=f"pright_total_evidence_binned_{_stem_label}",
                    ),
                ],
                align="center",
            )
        )

    _fig_repeat_evidence = plots.plot_repeat_by_repeat_evidence(
        plot_df_all,
        views=_views_sel,
    )
    mo.stop(
        _fig_repeat_evidence is None,
        mo.md("No repeat evidence plot available for the selected task/features."),
    )

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
            *_pright_evidence_panels,
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
def _(fig_size, mo, np, pd, plot_df_all, plt, save_plot, selected, views):
    _views_sel = {s: views[s] for s in selected}
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

    _fig_conditional, (_ax_stim_by_a, _ax_a_by_stim) = plt.subplots(1, 2, figsize=fig_size(4, 1), layout="constrained")
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
def _(plot_df_all, plots):
    _regressor_options = list(getattr(plot_df_all, "columns", []))
    _picker = getattr(plots, "pick_choice_history_regressor", None)
    if _picker is None:
        _preferred = ["choice_lag_one_hot_sum", "choice_lag_param", "at_choice_param"]
        choice_history_regressor = next((_col for _col in _preferred if _col in _regressor_options), None)
    else:
        choice_history_regressor = _picker(_regressor_options)
    _labeler = getattr(plots, "display_regressor_name", lambda _col: str(_col).replace("_", " "))
    choice_history_regressor_label = None if choice_history_regressor is None else _labeler(choice_history_regressor)
    return choice_history_regressor, choice_history_regressor_label


@app.cell
def _(
    choice_history_regressor,
    choice_history_regressor_label,
    fig_size,
    mo,
    plot_df_all,
    plots,
    plt,
    save_plot,
):
    mo.stop(
        choice_history_regressor is None,
        mo.md("No choice-history regressor available for p(right) by regressor."),
    )

    _fig_right, _ax_right = plt.subplots(figsize=fig_size(2, 1))
    _ax_right_regressor = plots.plot_right_by_regressor(
        plot_df_all,
        regressor_col=choice_history_regressor,
        title=None,
        ax=_ax_right,
    )
    if _ax_right_regressor is not None:
        _ax_right_regressor.set_xlabel(choice_history_regressor_label)
    mo.stop(
        _ax_right_regressor is None,
        mo.md("No p(right) by choice-history regressor plot available."),
    )

    mo.vstack(
        [
            _fig_right,
            save_plot(
                _fig_right,
                "p(right) by choice history",
                stem=f"right_by_{choice_history_regressor}",
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
    choice_history_regressor,
    fig_size,
    mo,
    plot_df_all,
    plots,
    plt,
    save_plot,
    selected,
    views,
):
    mo.stop(
        choice_history_regressor is None,
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
        regressor_col=choice_history_regressor,
        adapter=adapter,
        views=_views_sel,
        max_panels=1,
        ax=_ax_binned,
        legend_ax=_ax_binned_legend,
    )
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
                stem=f"accuracy_binned_{choice_history_regressor}",
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactive coefficient editor
    """)
    return


@app.cell
def _(editor_views, mo):
    _subjects = sorted(editor_views.keys(), key=str)
    mo.stop(not _subjects, mo.md("No fitted subjects available for coefficient editing."))
    ui_editor_subject = mo.ui.dropdown(options=_subjects, value=_subjects[0], label="Subject")
    ui_editor_subject
    return (ui_editor_subject,)


@app.cell
def _(editor_views, mo, ui_editor_subject):
    _view = editor_views[ui_editor_subject.value]
    _state_options = [f"{_k} — {_view.state_name_by_idx.get(_k, f'State {_k}')}" for _k in _view.state_idx_order]
    ui_editor_state = mo.ui.dropdown(options=_state_options, value=_state_options[0], label="State")
    ui_editor_state
    return (ui_editor_state,)


@app.cell
def _(adapter, mo):
    if adapter.num_classes != 2:
        ui_editor_side = None
    else:
        _choices = [str(label) for label in adapter.choice_labels]
        ui_editor_side = mo.ui.dropdown(options=_choices, value=_choices[0], label="Side")
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

    mo.vstack(
        [
            mo.md("Only the selected state's emission coefficients are edited; plots below update with the edited state."),
            mo.hstack(_controls),
            coef_editor,
        ],
        align="center",
    )
    coef_editor_explicit_class_indices = _payload["explicit_class_indices"]
    coef_editor_reference_class_idx = _payload["reference_class_idx"]
    coef_editor_stored_class_indices = _payload["stored_class_indices"]
    coef_editor_stored_reference_class_idx = _payload["stored_reference_class_idx"]
    return (
        coef_editor,
        coef_editor_explicit_class_indices,
        coef_editor_reference_class_idx,
        coef_editor_stored_class_indices,
        coef_editor_stored_reference_class_idx,
        coef_state_idx,
        coef_state_label,
    )


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
    return editor_trial_df, editor_view


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Editable accuracy plots
    """)
    return


@app.cell
def _(
    adapter,
    add_choice_lag_summary_regressor,
    apply_state_tweak_to_trial_df,
    apply_state_tweak_to_view,
    coef_editor,
    coef_editor_explicit_class_indices,
    coef_editor_reference_class_idx,
    coef_editor_stored_class_indices,
    coef_editor_stored_reference_class_idx,
    coef_state_idx,
    coef_state_label,
    editor_trial_df,
    editor_view,
    mo,
    np,
    plots,
    plt,
    prepare_predictions_df,
    save_plot,
    task_name,
    ui_editor_subject,
    ui_psychometric_background,
    ui_psychometric_regressor,
    ui_state_assignment_mode,
    ui_state_model_line_mode,
    ui_state_show_data_smooth,
    ui_state_show_weighted_points,
):
    _subj = ui_editor_subject.value
    _view = editor_view
    _edited_weights = np.asarray(coef_editor.value["weights"], dtype=float)
    _trial_df_tweaked = apply_state_tweak_to_trial_df(
        editor_trial_df,
        adapter=adapter,
        view=_view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        original_weights=np.asarray(coef_editor.value["original_weights"], dtype=float),
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
    )
    _view_tweaked = apply_state_tweak_to_view(
        _view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
        stored_class_indices=list(coef_editor_stored_class_indices),
        stored_reference_class_idx=int(coef_editor_stored_reference_class_idx),
    )
    _plot_df_tweaked = prepare_predictions_df(task_name, _trial_df_tweaked)
    _choice_lag_cols = [
        str(_feat)
        for _feat in list(getattr(_view_tweaked, "feat_names", []) or [])
        if str(_feat).startswith("choice_lag_") and str(_feat).removeprefix("choice_lag_").isdigit()
    ]
    if not _choice_lag_cols and hasattr(adapter, "choice_lag_cols"):
        _choice_lag_cols = adapter.choice_lag_cols(_trial_df_tweaked)
    _plot_df_tweaked = add_choice_lag_summary_regressor(
        _plot_df_tweaked,
        choice_lag_cols=_choice_lag_cols,
    )
    _title = f"{_subj} — tweaked {coef_state_label}"
    _state_legend_loc = "upper left"
    _state_legend_bbox_to_anchor = (1.01, 1)

    def _set_state_legend_location(_fig):
        for _ax in _fig.axes:
            _legend = _ax.get_legend()
            if _legend is None:
                continue
            _handles = getattr(_legend, "legend_handles", getattr(_legend, "legendHandles", []))
            _labels = [_text.get_text() for _text in _legend.get_texts()]
            _ax.legend(
                _handles,
                _labels,
                fontsize=8,
                frameon=_legend.get_frame_on(),
                loc=_state_legend_loc,
                bbox_to_anchor=_state_legend_bbox_to_anchor,
            )

    _fig_all_tweaked, _ = plots.plot_categorical_performance_all(
        _plot_df_tweaked,
        _title,
        background_style=ui_psychometric_background.value,
    )
    _fig_all_tweaked_list = list(_fig_all_tweaked) if isinstance(_fig_all_tweaked, (list, tuple)) else [_fig_all_tweaked]
    for _fig in _fig_all_tweaked_list:
        for _ax_idx, _ax in enumerate(_fig.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if _fig._suptitle is not None:
            _fig._suptitle.set_text("")
        _fig.tight_layout()

    _state_overlay_fn = getattr(plots, "plot_categorical_performance_state_overlay", None)
    if _state_overlay_fn is None:
        _fig_state_overlay_tweaked, _ = plots.plot_categorical_performance_by_state(
            df=_plot_df_tweaked,
            views={_subj: _view_tweaked},
            model_name=f"{_title} — all states",
            background_style="none",
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            overlay_only=True,
        )
        _set_state_legend_location(_fig_state_overlay_tweaked)
    else:
        _fig_state_overlay_tweaked_base, _ax_state_overlay_tweaked = plt.subplots(figsize=(3, 3))
        _fig_state_overlay_tweaked, _ = _state_overlay_fn(
            df=_plot_df_tweaked,
            views={_subj: _view_tweaked},
            model_name=f"{_title} — all states",
            background_style="none",
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            ax=_ax_state_overlay_tweaked,
        )
        _set_state_legend_location(_fig_state_overlay_tweaked)
    _fig_state_tweaked, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_tweaked,
        views={_subj: _view_tweaked},
        model_name=f"{_title} — per state",
        background_style="none",
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
    )
    _set_state_legend_location(_fig_state_tweaked)
    _reg_plot_fn = getattr(plots, "plot_regressor_psychometric_by_state", None)
    if _reg_plot_fn is None:
        _reg_section = mo.md("This task does not expose a regressor psychometric plot.")
    else:
        _fig_reg_state_tweaked, _ = _reg_plot_fn(
            df=_plot_df_tweaked,
            views={_subj: _view_tweaked},
            model_name=_title,
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
        )
        _reg_section = mo.vstack(
            [
                mo.hstack([ui_psychometric_regressor], justify="space-between"),
                mo.vstack(
                    [
                        _fig_reg_state_tweaked,
                        save_plot(
                            _fig_reg_state_tweaked,
                            f"tweaked {ui_psychometric_regressor.value} by state",
                            stem=f"tweaked_regressor_by_state_{ui_psychometric_regressor.value}",
                        ),
                    ],
                    align="center",
                ),
            ],
            align="center",
        )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            *[
                                _item
                                for _fig_idx, _fig in enumerate(_fig_all_tweaked_list, start=1)
                                for _item in (
                                    _fig,
                                    save_plot(
                                        _fig,
                                        f"tweaked overall psychometric {_fig_idx}" if len(_fig_all_tweaked_list) > 1 else "tweaked overall psychometric",
                                        stem=f"tweaked_categorical_overall_{_fig_idx}" if len(_fig_all_tweaked_list) > 1 else "tweaked_categorical_overall",
                                    ),
                                )
                            ],
                        ],
                        align="center",
                    ),
                    mo.vstack([ui_psychometric_background], align="start"),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
            ),
            _fig_state_overlay_tweaked,
            save_plot(
                _fig_state_overlay_tweaked,
                "tweaked state-overlay psychometric",
                stem="tweaked_categorical_state_overlay",
            ),
            _fig_state_tweaked,
            save_plot(_fig_state_tweaked, "tweaked per-state psychometric", stem="tweaked_categorical_by_state"),
            _reg_section,
            coef_editor,
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### State analysis
    """)
    return


@app.cell
def _(mo):
    ui_switch_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        label="Switch posterior threshold",
    )
    ui_switch_threshold
    return


@app.cell
def _(
    adapter,
    build_state_accuracy_payload,
    build_state_posterior_count_payload,
    fig_size,
    mo,
    model_plots,
    np,
    pd,
    pl,
    plt,
    save_plot,
    selected,
    sns,
    trial_df,
):
    mo.stop(not selected, mo.md("No fitted subjects available."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))

    _fig_acc_base, _ax_acc = plt.subplots(figsize=fig_size(3,1))
    _fig_acc = model_plots.state_accuracy(
        build_state_accuracy_payload(
            _trial_df_sel,
            performance_col="correct_bool",
            chance_level=1.0 / adapter.num_classes,
        ),
        ax=_ax_acc,
    )
    _fig_acc.set_title("")
    fig, ax = plt.subplots(figsize=fig_size(3,1))
    _fig_post = model_plots.state_posterior_count_kde(build_state_posterior_count_payload(_trial_df_sel), ax=ax, figsize=fig_size(3,1))
    ax.spines["right"].set_visible(True)
    ax.set_title("")

    def _session_switch_density_df(_trial_df):
        _pdf = _trial_df.to_pandas() if hasattr(_trial_df, "to_pandas") else pd.DataFrame(_trial_df)
        if _pdf.empty or "state_idx" not in _pdf.columns:
            return pd.DataFrame(columns=["subject", "condition", "session", "n_switches"])
        _session_col = "session" if "session" in _pdf.columns else "Session"
        _trial_col = "trial_idx" if "trial_idx" in _pdf.columns else "trial" if "trial" in _pdf.columns else "Trial"
        if _session_col not in _pdf.columns or _trial_col not in _pdf.columns:
            return pd.DataFrame(columns=["subject", "condition", "session", "n_switches"])
        if "condition" in _pdf.columns:
            _pdf["_switch_condition"] = _pdf["condition"].astype("string").fillna("unlabeled").astype(str)
        elif "drug_code" in _pdf.columns:
            _drug = pd.to_numeric(_pdf["drug_code"], errors="coerce")
            _pdf["_switch_condition"] = np.select(
                [_drug == 0, _drug == 1],
                ["saline", "drug"],
                default="unlabeled",
            )
        elif "Drug" in _pdf.columns:
            _drug = pd.to_numeric(_pdf["Drug"], errors="coerce")
            _pdf["_switch_condition"] = np.select(
                [_drug == 0, _drug == 1],
                ["saline", "drug"],
                default="unlabeled",
            )
        else:
            _pdf["_switch_condition"] = "all"
        _records = []
        _sorted = _pdf.sort_values(["subject", "_switch_condition", _session_col, _trial_col])
        for (_subject, _condition, _session), _group in _sorted.groupby(
            ["subject", "_switch_condition", _session_col],
            observed=True,
        ):
            _states = pd.to_numeric(_group["state_idx"], errors="coerce").to_numpy(dtype=float)
            _states = _states[np.isfinite(_states)]
            if _states.size == 0:
                continue
            _records.append(
                {
                    "subject": str(_subject),
                    "condition": str(_condition),
                    "session": str(_session),
                    "n_switches": int(np.sum(_states[1:] != _states[:-1])),
                }
            )
        return pd.DataFrame(_records, columns=["subject", "condition", "session", "n_switches"])

    _switch_df = _session_switch_density_df(_trial_df_sel)
    _fig_switch_kde, _ax_switch_kde = plt.subplots(figsize=fig_size(3,1))
    _condition_order = [
        _condition
        for _condition in ["saline", "rest", "drug", "all", "unlabeled"]
        if _condition in set(_switch_df["condition"].astype(str)) if not _switch_df.empty
    ]
    if not _switch_df.empty:
        _condition_order.extend(
            _condition
            for _condition in _switch_df["condition"].astype(str).unique().tolist()
            if _condition not in _condition_order
        )
    _palette = {
        "saline": "tab:gray",
        "rest": "tab:red",
        "drug": "tab:pink",
        "all": "#333333",
        "unlabeled": "#777777",
    }
    for _condition in _condition_order:
        _values = _switch_df.loc[
            _switch_df["condition"].astype(str) == _condition,
            "n_switches",
        ].dropna().to_numpy(dtype=float)
        if _values.size == 0:
            continue
        _color = _palette.get(str(_condition), "#333333")
        if _values.size >= 2 and np.nanstd(_values) > 0:
            sns.kdeplot(
                x=_values,
                ax=_ax_switch_kde,
                color=_color,
                linewidth=2.0,
                bw_adjust=0.8,
                clip=(0, None),
                label=str(_condition),
            )
        else:
            _ax_switch_kde.axvline(float(_values[0]), color=_color, linewidth=2.0, label=str(_condition))
    _max_switches = float(_switch_df["n_switches"].max()) if not _switch_df.empty else 1.0
    _ax_switch_kde.set_xlim(left=-0.25, right=max(1.0, _max_switches + 0.25))
    _ax_switch_kde.set_xlabel("State changes per session")
    _ax_switch_kde.set_ylabel("Density")
    _ax_switch_kde.set_title("")
    if len(_condition_order) > 1:
        _ax_switch_kde.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.6), ncol=len(_condition_order))
        _fig_switch_kde.subplots_adjust(bottom=0.34)
    sns.despine(ax=_ax_switch_kde)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("#### Accuracy by state"),
                            _fig_acc.figure,
                            save_plot(
                                _fig_acc.figure,
                                "accuracy by state",
                                stem="state_accuracy",
                            ),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            mo.md("#### Posterior / trial-count KDE"),
                            _fig_post.figure,
                            save_plot(
                                _fig_post.figure,
                                "posterior trial-count kde",
                                stem="state_posterior_count_kde",
                            ),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            mo.md("#### State-change KDE"),
                            _fig_switch_kde,
                            save_plot(
                                _fig_switch_kde,
                                "state changes per session kde",
                                stem="state_switches_kde",
                            ),
                        ],
                        align="center",
                    ),
                ],
                align="center",
            ),
            mo.md("**Trial counts & mean accuracy per label:**"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fractional occupancy & state changes per session
    """)
    return


@app.cell
def _(
    build_state_occupancy_payload,
    fig_size,
    mo,
    model_plots,
    np,
    pd,
    pl,
    plt,
    save_plot,
    selected,
    trial_df,
):
    mo.stop(not selected, mo.md("Select subjects above."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    _occupancy_payload = build_state_occupancy_payload(
        _trial_df_sel,
        session_col="session",
        sort_col="trial_idx",
    )
    _fig_occ_overall_summary, _ax_occ_overall_summary = plt.subplots(figsize=fig_size(3,1))
    ax_occ_overall_summary = model_plots.state_occupancy_overall_summary(
        _occupancy_payload,
        ax=_ax_occ_overall_summary,
    )
    ax_occ_overall_summary.set_title("")
    _fig_occ_overall_by_subject = model_plots.state_occupancy_overall_by_subject(_occupancy_payload)
    _fig_occ_sessions_summary, _ax_occ_sessions_summary = plt.subplots(figsize=fig_size(3,1))
    ax_occ_sessions_summary = model_plots.state_session_occupancy_summary(
        _occupancy_payload,
        ax=_ax_occ_sessions_summary,
    )
    ax_occ_sessions_summary.set_title("")
    _fig_occ_sessions_by_subject = model_plots.state_session_occupancy_by_subject(_occupancy_payload)
    _fig_switches_summary, _ax_switches_summary = plt.subplots(figsize=fig_size(3,1))
    ax_switches_summary = model_plots.state_switches_summary(
        _occupancy_payload,
        ax=_ax_switches_summary,
    )
    ax_switches_summary.set_title("")
    ax_switches_summary.set_xlim(right = 60)
    _fig_occ_switches_by_subject = model_plots.state_switches_by_subject(_occupancy_payload)
    state_switch_sessions_df = _occupancy_payload["switches_df"]
    state_switch_sessions_df = (
        state_switch_sessions_df.to_pandas() if hasattr(state_switch_sessions_df, "to_pandas") else pd.DataFrame(state_switch_sessions_df)
    )
    state_switch_sessions_df = state_switch_sessions_df.copy()
    state_switch_sessions_df["subject"] = state_switch_sessions_df["subject"].astype(str)
    state_switch_sessions_df["session"] = state_switch_sessions_df["session"].astype(str)
    state_switch_sessions_df["n_switches"] = pd.to_numeric(
        state_switch_sessions_df["n_switches"],
        errors="coerce",
    )
    state_switch_sessions_df = state_switch_sessions_df.dropna(subset=["n_switches"]).copy()
    state_switch_sessions_df["n_switches"] = state_switch_sessions_df["n_switches"].astype(int)

    _switch_counts = (
        state_switch_sessions_df.groupby("n_switches", as_index=False, observed=True)
        .size()
        .rename(columns={"size": "n_sessions"})
        .sort_values("n_switches")
    )
    state_switch_selection_points = []
    for _row in _switch_counts.itertuples(index=False):
        for _y in range(int(_row.n_sessions)):
            for _x in np.linspace(float(_row.n_switches) - 0.45, float(_row.n_switches) + 0.45, 9):
                state_switch_selection_points.append(
                    {
                        "n_switches": int(_row.n_switches),
                        "x": float(_x),
                        "y": float(_y) + 0.5,
                    }
                )
    ui_occ_switches_summary = mo.ui.matplotlib(ax_switches_summary, debounce=True)
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.vstack(
                        [
                            ax_occ_overall_summary.figure,
                            save_plot(
                                ax_occ_overall_summary.figure,
                                "fractional occupancy overall summary",
                                stem="state_occupancy_overall_summary",
                            ),
                        ],
                        align="center",
                    ),
                    #     mo.vstack([
                    #         _fig_occ_overall_by_subject,
                    #         save_plot(
                    #             _fig_occ_overall_by_subject,
                    #             "fractional occupancy overall by subject",
                    #             stem="state_occupancy_overall_by_subject",
                    #             location=(0, 0),
                    #         ),
                    #     ], align="center"),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    mo.vstack(
                        [
                            ax_occ_sessions_summary.figure,
                            save_plot(
                                ax_occ_sessions_summary.figure,
                                "fractional occupancy by session summary",
                                stem="state_session_occupancy_summary",
                            ),
                        ],
                        align="center",
                    ),
                    # mo.vstack([
                    #     _fig_occ_sessions_by_subject,
                    #     save_plot(
                    #         _fig_occ_sessions_by_subject,
                    #         "fractional occupancy by session and subject",
                    #         stem="state_session_occupancy_by_subject",
                    #         location=(0, 0),
                    #     ),
                    # ], align="center"),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    mo.vstack(
                        [
                            ax_switches_summary.figure,
                            save_plot(
                                ax_switches_summary.figure,
                                "state switches summary",
                                stem="state_switches_summary",
                            ),
                        ],
                        align="center",
                    ),
                    # mo.vstack([
                    #     _fig_occ_switches_by_subject,
                    #     save_plot(
                    #         _fig_occ_switches_by_subject,
                    #         "state switches by subject",
                    #         stem="state_switches_by_subject",
                    #         location=(0, 0),
                    #     ),
                    # ], align="center"),
                ],
                align="center",
            ),
        ],
        align="center",
    )
    return (
        state_switch_selection_points,
        state_switch_sessions_df,
        ui_occ_switches_summary,
    )


@app.cell
def _(
    mo,
    pd,
    state_switch_selection_points,
    state_switch_sessions_df,
    ui_occ_switches_summary,
):
    _points = pd.DataFrame(state_switch_selection_points)
    if _points.empty or not ui_occ_switches_summary.value:
        selected_state_switch_counts = []
    else:
        _mask = ui_occ_switches_summary.value.get_mask(_points["x"].to_numpy(), _points["y"].to_numpy())
        selected_state_switch_counts = sorted(_points.loc[_mask, "n_switches"].unique().tolist())
    selected_state_switch_sessions = (
        state_switch_sessions_df[state_switch_sessions_df["n_switches"].isin(selected_state_switch_counts)]
        .sort_values(["n_switches", "subject", "session"])
        .reset_index(drop=True)
    )
    mo.vstack(
        [
            mo.md("Selected switch counts: " + (", ".join(map(str, selected_state_switch_counts)) if selected_state_switch_counts else "_none_")),
            selected_state_switch_sessions if selected_state_switch_counts else mo.md("Select one or more histogram bars to list matching sessions."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Posteriors around a change
    """)
    return


@app.cell
def _(
    build_change_triggered_posteriors_payload,
    fig_size,
    mo,
    model_plots,
    pl,
    plt,
    save_plot,
    selected,
    trial_df,
):
    mo.stop(not selected, mo.md("Select subjects above."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    _change_payload = build_change_triggered_posteriors_payload(
        _trial_df_sel,
        session_col="session",
        sort_col="trial_idx",
    )
    _change_directions = list(_change_payload.get("directions") or ["into_engaged", "out_of_engaged"])
    _fig_change_summary_base, _axes_change_summary = plt.subplots(
        1,
        len(_change_directions),
        figsize=fig_size(1,3),
        squeeze=False,
    )
    _fig_change_summary = model_plots.change_triggered_posteriors_summary(
        _change_payload,
        axes=_axes_change_summary,
    )
    _fig_change_by_subject = model_plots.change_triggered_posteriors_by_subject(
        _change_payload,
    )
    def _clear_titles(_plot_obj):
        if isinstance(_plot_obj, (list, tuple)):
            for _item in _plot_obj:
                _clear_titles(_item)
            return
        _fig = getattr(_plot_obj, "figure", _plot_obj)
        for _ax in getattr(_fig, "axes", []):
            _ax.set_title("")

    def _set_change_legend(_fig):
        for _ax in _fig.axes:
            if _ax.legend_ is not None:
                _ax.legend_.remove()
        _fig.legend(
            handles=[
                plt.Line2D([0], [0], color="tab:green", lw=2.2, label="Engaged"),
                plt.Line2D([0], [0], color="tab:gray", lw=2.0, label="Disengaged"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        _fig.subplots_adjust(top=0.86)

    for _plot_obj in (_fig_change_summary, _fig_change_by_subject):
        _clear_titles(_plot_obj)
    _set_change_legend(_fig_change_summary[0].figure)
    mo.vstack(
        [
            _fig_change_summary[0],
            save_plot(
                _fig_change_summary[0].figure,
                "change-triggered posteriors summary",
                stem="change_triggered_posteriors_summary",
            ),
            # _fig_change_by_subject,
            # save_plot(
            #     _fig_change_by_subject,
            #     "change-triggered posteriors by subject",
            #     stem="change_triggered_posteriors_by_subject",
            # ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session statistics
    """)
    return


@app.cell
def _(mo):
    get_session_pick, set_session_pick = mo.state(None)
    ui_random_session = mo.ui.run_button(label="Pick random session")
    return get_session_pick, set_session_pick, ui_random_session


@app.cell
def _(
    mo,
    np,
    pl,
    selected,
    set_session_pick,
    trial_df,
    ui_random_session,
    views,
):
    mo.stop(not ui_random_session.value)
    _subjects = [s for s in selected if s in views]
    mo.stop(not _subjects, mo.md("No fitted subjects available."))
    _sessions_df = (
        trial_df.filter(pl.col("subject").is_in(_subjects))
        .group_by(["subject", "session"])
        .agg(pl.len().alias("n_trials"))
        .filter(pl.col("n_trials") > 0)
        .sort(["subject", "session"])
    )
    mo.stop(_sessions_df.height == 0, mo.md("No sessions available for fitted subjects."))
    _rows = _sessions_df.select(["subject", "session"]).to_dicts()
    _row = _rows[int(np.random.default_rng().integers(len(_rows)))]
    set_session_pick({"subject": _row["subject"], "session": str(_row["session"])})
    return


@app.cell
def _(get_session_pick, mo, selected, set_session_pick):
    _subj_opts = selected if selected else ["(no fitted subjects)"]
    _pick = get_session_pick() or {}
    _picked_subj = _pick.get("subject")
    _default_subj = _picked_subj if _picked_subj in _subj_opts else _subj_opts[0]
    ui_session_subj = mo.ui.dropdown(
        options=_subj_opts,
        value=_default_subj,
        label="Subject",
        on_change=lambda value: set_session_pick({"subject": value, "session": None}),
    )
    return (ui_session_subj,)


@app.cell
def _(
    get_session_pick,
    mo,
    pl,
    set_session_pick,
    trial_df,
    ui_session_subj,
    views,
):
    _sess_opts = (
        sorted(
            trial_df.filter(pl.col("subject") == ui_session_subj.value)["session"].unique().to_list(),
            key=str,
        )
        if ui_session_subj.value in views
        else [0]
    )
    _sess_opts = _sess_opts or [0]
    _sess_values = [str(s) for s in _sess_opts]
    _pick = get_session_pick() or {}
    _picked_session = (
        str(_pick.get("session"))
        if _pick.get("subject") == ui_session_subj.value and _pick.get("session") is not None
        else None
    )
    _default_session = _picked_session if _picked_session in _sess_values else _sess_values[0]
    ui_session_id = mo.ui.dropdown(
        options=_sess_values,
        value=_default_session,
        label="Session",
        on_change=lambda value: set_session_pick({"subject": ui_session_subj.value, "session": value}),
    )
    _win_opts = [1, 5, 10, 20, 50]
    ui_engaged_window = mo.ui.dropdown(options=[str(w) for w in _win_opts], value="20", label="P(engaged) window")
    ui_engaged_trace_mode = mo.ui.radio(
        options={"Rolling": "rolling", "Raw": "raw"},
        value="Rolling",
        inline=False,
        label="P(engaged) trace",
    )
    return ui_engaged_trace_mode, ui_engaged_window, ui_session_id


@app.cell
def _(
    adapter,
    build_session_deepdive_payload,
    mo,
    model_plots,
    put_figure_legend_at_bottom,
    save_plot,
    trial_df,
    ui_engaged_trace_mode,
    ui_engaged_window,
    ui_random_session,
    ui_session_id,
    ui_session_subj,
    views,
):
    _subj = ui_session_subj.value
    mo.stop(_subj not in views, mo.md("No fitted arrays for this subject — run the fit first."))
    _sess = int(ui_session_id.value) if str(ui_session_id.value).isdigit() else ui_session_id.value
    _deepdive_payload = build_session_deepdive_payload(
        trial_df,
        subject=_subj,
        session=_sess,
        session_col="session",
        sort_col="trial_idx",
        engaged_window=int(ui_engaged_window.value),
        engaged_trace_mode=ui_engaged_trace_mode.value,
        chance_level=1.0 / adapter.num_classes,
        num_classes=adapter.num_classes,
        views=views,
    )
    _fig = model_plots.session_deepdive(_deepdive_payload)
    _fig_traces = model_plots.session_deepdive_state_traces(_deepdive_payload)
    put_figure_legend_at_bottom(_fig, bottom=0.18)
    put_figure_legend_at_bottom(_fig_traces, bottom=0.28)
    mo.vstack(
        [
            mo.hstack([ui_random_session, ui_session_subj, ui_session_id, ui_engaged_window, ui_engaged_trace_mode], align = "center"),
            _fig,
            _fig_traces,
            save_plot(_fig, "session statistics", stem=f"session_stats_{_subj}_{_sess}"),
            save_plot(_fig_traces, "session state traces", stem=f"session_state_traces_{_subj}_{_sess}"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session trajectories
    """)
    return


@app.cell
def _(mo):
    session_button = mo.ui.run_button(label = "Session trajectories")
    session_button
    return (session_button,)


@app.cell
def _(
    build_session_trajectories_payload,
    fig_size,
    mo,
    model_plots,
    pl,
    plt,
    save_plot,
    selected,
    session_button,
    trial_df,
):
    mo.stop(not selected or not session_button.value, mo.md("Click run button"))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    _fig_traj, _ax_traj = plt.subplots(figsize=fig_size(1, 3))
    _ax_traj = model_plots.session_trajectories(
        build_session_trajectories_payload(
            _trial_df_sel,
            session_col="session",
            sort_col="trial_idx",
        ),
        ax=_ax_traj,
    )
    _fig_traj = _ax_traj.figure
    mo.vstack(
        [
            _fig_traj,
            save_plot(_fig_traj, "session trajectories", stem="session_trajectories"),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drug-split session summaries
    """)
    return


@app.cell
def _(mo, pl, selected, task_name, trial_df):
    mo.stop(not selected, mo.md("No fitted subjects available."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    drug_col = "drug_code" if (task_name == "2ADC_DRUG") else "Drug"
    mo.stop(drug_col not in _trial_df_sel.columns, mo.md("`trial_df` has no `drug` column."))
    _session_drug = (
        _trial_df_sel.group_by(["subject", "session"])
        .agg(
            [
                pl.col(drug_col).drop_nulls().first().alias("drug"),
                pl.col(drug_col).drop_nulls().n_unique().alias("n_drug_values"),
            ]
        )
        .with_columns(pl.col("drug").cast(pl.Int64, strict=False).alias("_drug"))
        .filter(pl.col("_drug").is_in([0, 1]))
    )
    mo.stop(_session_drug.height == 0, mo.md("No sessions with `drug = 0` or `drug = 1`."))
    _ambiguous_sessions = _session_drug.filter(pl.col("n_drug_values") > 1)
    mo.stop(
        _ambiguous_sessions.height > 0,
        mo.md("Some sessions contain multiple non-null `drug` values; split the dataframe before comparing drug conditions."),
    )
    drug_trial_dfs = {}
    for _drug in [0, 1]:
        _sessions = _session_drug.filter(pl.col("_drug") == _drug).select(["subject", "session"])
        drug_trial_dfs[_drug] = _trial_df_sel.join(_sessions, on=["subject", "session"], how="inner")
    drug_session_counts = (
        _session_drug.group_by("_drug")
        .agg(pl.len().alias("n_sessions"))
        .rename({"_drug": "drug"})
        .sort("drug")
    )
    mo.ui.dataframe(drug_session_counts)
    return (drug_trial_dfs,)


@app.cell
def _(
    build_state_occupancy_payload,
    drug_trial_dfs,
    fig_size,
    mo,
    model_plots,
    np,
    plt,
    save_plot,
):
    _columns = []
    for _drug in [0, 1]:
        _trial_df_drug = drug_trial_dfs[_drug]
        if _trial_df_drug.height == 0:
            _columns.append(mo.md(f"No sessions for `drug = {_drug}`."))
            continue
        _payload = build_state_occupancy_payload(
            _trial_df_drug,
            session_col="session",
            sort_col="trial_idx",
            title=f"Drug = {_drug} - state changes",
        )
        _switches = _payload["switches_df"]["n_switches"].to_numpy(dtype=float)
        _median_switches = float(np.median(_switches)) if _switches.size else float("nan")
        _fig, _ax = plt.subplots(figsize=fig_size(2, 1))
        _ax = model_plots.state_switches_summary(_payload, ax=_ax)
        _ax.set_title(f"Drug = {_drug} - state changes\nmedian = {_median_switches:g}")
        _fig = _ax.figure
        _columns.append(
            mo.vstack(
                [
                    _fig,
                    save_plot(
                        _fig,
                        f"state switches summary drug {_drug}",
                        stem=f"state_switches_summary_drug_{_drug}",
                    ),
                ],
                align="center",
            )
        )
    mo.hstack(_columns, align="center")
    return


@app.cell
def _(
    boxplot_figsize,
    build_state_dwell_times_payload,
    drug_trial_dfs,
    format_boxplot_panel,
    mo,
    model_plots,
    plt,
    save_plot,
    views,
):
    _columns = []
    for _drug in [0, 1]:
        _trial_df_drug = drug_trial_dfs[_drug]
        if _trial_df_drug.height == 0:
            _columns.append(mo.md(f"No sessions for `drug = {_drug}`."))
            continue
        _subjects = _trial_df_drug["subject"].unique().to_list()
        _views_sel = {s: views[s] for s in _subjects if s in views}
        _payload = build_state_dwell_times_payload(
            _trial_df_drug,
            session_col="session",
            sort_col="trial_idx",
            views=_views_sel,
            max_dwell=None,
            title=f"Drug = {_drug} - dwell times",
        )
        _fig, _ax = plt.subplots(figsize=boxplot_figsize)
        _fig, _ax = model_plots.state_dwell_median_boxplot(_payload, ax=_ax)
        format_boxplot_panel(_ax)
        _ax.set_title(f"Drug = {_drug} - median dwell time")
        _columns.append(
            mo.vstack(
                [
                    _fig,
                    save_plot(
                        _fig,
                        f"state dwell median boxplot drug {_drug}",
                        stem=f"state_dwell_median_boxplot_drug_{_drug}",
                    ),
                ],
                align="center",
            )
        )
    mo.hstack(_columns, align="center")
    return


@app.cell
def _(
    build_session_trajectories_payload,
    drug_trial_dfs,
    fig_size,
    mo,
    model_plots,
    plt,
    put_figure_legend_at_bottom,
    save_plot,
):
    _columns = []
    for _drug in [0, 1]:
        _trial_df_drug = drug_trial_dfs[_drug]
        if _trial_df_drug.height == 0:
            _columns.append(mo.md(f"No sessions for `drug = {_drug}`."))
            continue
        _payload = build_session_trajectories_payload(
            _trial_df_drug,
            session_col="session",
            sort_col="trial_idx",
            title=""
            # title=f"{"NMDAr hypofunction" if _drug == 1 else "Control"}",
        )
        _fig, _ax = plt.subplots(figsize=fig_size(1, 3))
        _ax = model_plots.session_trajectories(_payload, ax=_ax)
        _fig = _ax.figure
        put_figure_legend_at_bottom(_fig, bottom=0.34)
        _columns.append(
            mo.vstack(
                [
                    _fig,
                    save_plot(
                        _fig,
                        f"session trajectories drug {_drug}",
                        stem=f"session_trajectories_drug_{_drug}",
                    ),
                ],
                align="center",
            )
        )
    mo.hstack(_columns, align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### τ sweep
    """)
    return


@app.cell
def _(K, df_all, mo, model_cfg, np, paths, pl, plt, save_plot, sns):
    _sweep_path = paths.RESULTS / "fits" / "tau_sweep" / f"glmhmmt_K{K}" / "tau_sweep_summary.parquet"
    mo.stop(
        not _sweep_path.exists(),
        mo.md(
            f"**τ sweep results not found.** Run the sweep first:\n```\n"
            f"uv run glmhmmt-fit-tau-sweep --model glmhmmt --K {K}\n```"
        ),
    )
    _df_sweep = pl.read_parquet(_sweep_path)
    _subjects = [s for s in model_cfg.subjects if s in _df_sweep["subject"].unique().to_list()]
    mo.stop(not _subjects, mo.md("No sweep data for selected subjects."))

    _fig_sweep, _axes_sw = plt.subplots(1, 2, figsize=(12, 4))
    _ax_bic, _ax_ll = _axes_sw
    _palette = sns.color_palette("tab10", n_colors=len(_subjects))
    _n_trials = df_all.group_by("subject").agg(pl.len().alias("n_trials"))
    for _i, _subj in enumerate(_subjects):
        _d = _df_sweep.filter((pl.col("subject") == _subj) & (pl.col("K") == K)).sort("tau")
        _tau = _d["tau"].to_numpy()
        _bic = _d["bic"].to_numpy()
        _ll = _d["ll_per_trial"].to_numpy()
        _color = _palette[_i]
        _ax_bic.plot(_tau, _bic, "-o", ms=3, color=_color, label=_subj)
        _ax_ll.plot(_tau, _ll, "-o", ms=3, color=_color, label=_subj)
        _best_idx = int(np.argmin(_bic))
        _ax_bic.axvline(_tau[_best_idx], color=_color, lw=0.8, linestyle="--", alpha=0.6)
    for _ax, _ylabel, _title in [
        (_ax_bic, "BIC", "BIC vs τ  (lower is better)"),
        (_ax_ll, "LL / trial", "Log-likelihood per trial vs τ"),
    ]:
        _ax.set_xlabel("τ (action-trace half-life)")
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=_ax)
    _fig_sweep.tight_layout()

    _best = (
        _df_sweep.filter(pl.col("subject").is_in(_subjects) & (pl.col("K") == K))
        .sort("bic")
        .group_by(["subject", "K"])
        .first()
        .select(["subject", "K", "tau", "bic", "ll_per_trial", "acc"])
        .sort(["subject", "K"])
    )
    _best_all = (
        _df_sweep.filter(pl.col("subject").is_in(_subjects) & (pl.col("K") == K))
        .join(_n_trials, on="subject", how="left")
        .group_by("tau")
        .agg(
            [
                (pl.col("bic") * pl.col("n_trials")).sum().alias("bic_wsum"),
                (pl.col("ll_per_trial") * pl.col("n_trials")).sum().alias("llpt_wsum"),
                (pl.col("acc") * pl.col("n_trials")).sum().alias("acc_wsum"),
                pl.col("n_trials").sum().alias("n_total"),
                pl.n_unique("subject").alias("n_subjects"),
            ]
        )
        .with_columns(
            [
                (pl.col("bic_wsum") / pl.col("n_total")).alias("bic_mean_w"),
                (pl.col("llpt_wsum") / pl.col("n_total")).alias("ll_per_trial_mean_w"),
                (pl.col("acc_wsum") / pl.col("n_total")).alias("acc_mean_w"),
            ]
        )
        .select(["tau", "bic_mean_w", "ll_per_trial_mean_w", "acc_mean_w", "n_subjects", "n_total"])
        .sort("bic_mean_w")
    )
    mo.vstack(
        [
            mo.md(f"### τ sweep results — glmhmmt K={K}"),
            _fig_sweep,
            save_plot(_fig_sweep, "tau sweep results", stem=f"tau_sweep_glmhmmt_k{K}"),
            mo.md("**Best τ per subject (min BIC):**"),
            mo.plain_text(_best.to_pandas().to_string(index=False)),
            mo.ui.dataframe(_best_all),
        ],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
