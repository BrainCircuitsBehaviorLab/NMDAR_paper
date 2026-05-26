import marimo

__generated_with = "0.23.8"
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
    try:
        from glmhmmt.cli.fit_glmhmm import main as fit_main
        _FITTING_AVAILABLE = True
    except ImportError:
        fit_main = None
        _FITTING_AVAILABLE = False
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

    project_root = _PROJECT_ROOT
    configure_paths(config_path=project_root / "config.toml")

    sns.set_style("ticks")
    paths = get_runtime_paths()
    from src.utils import fig_size

    return (
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_change_triggered_posteriors_payload,
        build_editor_payload,
        build_emission_weights_df,
        build_session_deepdive_payload,
        build_session_trajectories_payload,
        build_state_accuracy_payload,
        build_state_dwell_times_payload,
        build_state_occupancy_payload,
        build_state_posterior_count_payload,
        build_transition_matrix_by_subject_payload,
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
    build_emission_weights_df,
    mo,
    model_plots,
    np,
    paths,
    pd,
    save_plot,
    selected,
    views,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _save_path = paths.RESULTS / "plots/GLMHMM/emissions_coefs.png"
    _views_sel = {s: views[s] for s in selected}
    _weights_df = build_emission_weights_df(_views_sel)
    _fig_by_subject = model_plots.emission_weights_by_subject(
        _weights_df,
        K=K,
    )

    _feature_labels = {
        "stim_param": r"$\mathrm{Stim}_{\mathrm{param}}$",
        "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "at_choice_param": r"$\mathrm{A}_t$",
        "choice_lag_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    }

    feature_labeler = lambda feature: _feature_labels.get(str(feature), str(feature))

    _summary_figs = model_plots.emission_weights_summary_boxplot(
        _weights_df,
        connect_subjects=True,
        show_ttests=True,
        feature_labeler=feature_labeler,
    )
    _emission_subject_lines = [
        _line
        for _line in _summary_figs.lines
        if _line.get_alpha() == 0.15 and _line.get_linestyle() == "-" and len(_line.get_xdata()) >= 2 and len(_line.get_ydata()) >= 2
    ]
    _weights_pdf = _weights_df.to_pandas() if hasattr(_weights_df, "to_pandas") else pd.DataFrame(_weights_df)
    _weights_pdf = _weights_pdf.copy()
    _weights_pdf["subject"] = _weights_pdf["subject"].astype(str)
    _weights_pdf["feature"] = _weights_pdf["feature"].astype(str)
    _weights_pdf["weight"] = pd.to_numeric(_weights_pdf["weight"], errors="coerce")
    _weights_pdf = _weights_pdf.dropna(subset=["weight"])
    _line_keys = []
    for _feature in pd.unique(_weights_pdf["feature"]):
        _feature_df = _weights_pdf[_weights_pdf["feature"] == _feature]
        for _subject in sorted(pd.unique(_feature_df["subject"]).tolist()):
            if _feature_df[_feature_df["subject"] == _subject]["weight"].notna().sum() >= 2:
                _line_keys.append((str(_subject), str(_feature), str(feature_labeler(_feature))))
    emission_summary_selection_points = []
    for (_subject, _feature, _feature_label), _line in zip(_line_keys, _emission_subject_lines, strict=False):
        _xs = _line.get_xdata()
        _ys = _line.get_ydata()
        for _left in range(len(_xs) - 1):
            if not (np.isfinite(_xs[_left]) and np.isfinite(_xs[_left + 1]) and np.isfinite(_ys[_left]) and np.isfinite(_ys[_left + 1])):
                continue
            for _x, _y in zip(np.linspace(_xs[_left], _xs[_left + 1], 24), np.linspace(_ys[_left], _ys[_left + 1], 24), strict=False):
                emission_summary_selection_points.append(
                    {
                        "subject": _subject,
                        "feature": _feature,
                        "feature_label": _feature_label,
                        "x": float(_x),
                        "y": float(_y),
                    }
                )
    ui_emission_summary = mo.ui.matplotlib(_summary_figs, debounce=True)
    mo.vstack(
        [
            # _fig_by_subject,
            #  save_plot(_fig_by_subject, f"Emission Weights",
            #                      stem=f"emissions_summary", location = (0,1)),
            ui_emission_summary,
            mo.hstack(
                [
                    save_plot(_summary_figs, f"Emission Weights lineplot", stem=f"emissions_lineplot", location=(0, 0)),
                    save_plot(_summary_figs, f"Emission Weights boxplot", stem=f"emissions_boxplot", location=(0, 1)),
                ],
                gap="15",
            ),
        ],
        align="center",
    )
    return emission_summary_selection_points, ui_emission_summary


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


@app.cell
def _(K, build_emission_weights_df, mo, np, pd, selected, views):
    from itertools import combinations

    import plotly.graph_objects as _go
    from scipy.stats import ttest_rel

    from glmhmmt.plots.common import significance_label
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
        "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
        "at_choice_param": r"$\mathrm{A}_t$",
        "choice_lag_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    }
    _feature_labeler = lambda feature: _feature_labels.get(str(feature), str(feature))

    _plot_weights = _fold_three_choice_raw_weights(_weights_df)
    if _plot_weights is None:
        _plot_weights = _weights_df

    _df, _features, _display_features, _state_order, _palette = _prepare_weights_df(
        _plot_weights,
        K=K,
        feature_labeler=_feature_labeler,
    )
    _grouped_values, _subject_lines = _emission_boxplot_payload(
        _df,
        features=_features,
        states=_state_order,
    )

    _n_features = len(_features)
    _n_states = len(_state_order)
    _group_width = 0.8
    _hue_width = _group_width / max(1, _n_states)
    _colors = [_palette[_state] for _state in _state_order]
    _subjects = sorted(pd.unique(_df["subject"]).tolist())

    _fig_plotly = _go.Figure()

    for _feat_idx, _feature in enumerate(_features):
        _positions = [_feat_idx + (_state_idx - (_n_states - 1) / 2.0) * _hue_width for _state_idx in range(_n_states)]

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
                _fig_plotly.add_trace(
                    _go.Scatter(
                        x=_x_segment,
                        y=_y_segment,
                        mode="lines+markers",
                        line=dict(color="rgba(122, 122, 122, 0.15)", width=1.0),
                        marker=dict(color="rgba(122, 122, 122, 0.01)", size=10),
                        customdata=[[_subject, _feature, _feature_labeler(_feature)] for _ in _segment],
                        hovertemplate=("Subject: %{customdata[0]}<br>Feature: %{customdata[2]}<br>Weight: %{y:.4f}<extra></extra>"),
                        showlegend=False,
                        selected=dict(marker=dict(color="rgba(0, 0, 0, 0.45)", size=10)),
                        unselected=dict(marker=dict(opacity=0.01)),
                    )
                )

        for _state_idx, _state in enumerate(_state_order):
            _values = np.asarray(_grouped_values[_state_idx][_feat_idx], dtype=float)
            _values = _values[np.isfinite(_values)]
            if _values.size == 0:
                continue
            _fig_plotly.add_trace(
                _go.Box(
                    x=[_positions[_state_idx]] * len(_values),
                    y=_values,
                    name=str(_state),
                    width=_hue_width * 0.78,
                    boxpoints=False,
                    fillcolor="rgba(255, 255, 255, 0.85)",
                    line=dict(color="#666666", width=1.1),
                    marker_color=_colors[_state_idx],
                    quartilemethod="linear",
                    whiskerwidth=0,
                    showlegend=_feat_idx == 0,
                    hovertemplate=(f"State: {_state}<br>Feature: {_feature_labeler(_feature)}<br>Weight: %{{y:.4f}}<extra></extra>"),
                )
            )
            _box_half_width = (_hue_width * 0.78) / 2.0
            _fig_plotly.add_shape(
                type="line",
                x0=_positions[_state_idx] - _box_half_width,
                x1=_positions[_state_idx] + _box_half_width,
                y0=float(np.median(_values)),
                y1=float(np.median(_values)),
                line=dict(color=_colors[_state_idx], width=3.0),
            )

    for _feat_idx, _per_subject_values in enumerate(_subject_lines):
        _finite_groups = [
            _values[np.isfinite(_values)]
            for _values in (_grouped_values[_state_idx][_feat_idx] for _state_idx in range(_n_states))
            if np.isfinite(_values).any()
        ]
        if not _finite_groups or _n_states < 2:
            continue

        _finite_values = np.concatenate(_finite_groups)
        _y_range = float(np.max(_finite_values) - np.min(_finite_values))
        if not np.isfinite(_y_range) or _y_range <= 0:
            _y_range = 1.0
        _y_base = float(np.max(_finite_values))
        _step = _y_range * 0.08
        _bracket_height = _y_range * 0.02

        for _pair_idx, (_state_a, _state_b) in enumerate(combinations(range(_n_states), 2)):
            _paired = _per_subject_values[np.isfinite(_per_subject_values[:, _state_a]) & np.isfinite(_per_subject_values[:, _state_b])]
            if _paired.shape[0] < 2:
                continue

            _pvalue = float(ttest_rel(_paired[:, _state_a], _paired[:, _state_b]).pvalue)
            _x1 = _feat_idx + (_state_a - (_n_states - 1) / 2.0) * _hue_width
            _x2 = _feat_idx + (_state_b - (_n_states - 1) / 2.0) * _hue_width
            _y = _y_base + _step * (_pair_idx + 1)
            _fig_plotly.add_shape(
                type="path",
                path=(f"M {_x1},{_y} L {_x1},{_y + _bracket_height} L {_x2},{_y + _bracket_height} L {_x2},{_y}"),
                line=dict(color="black", width=1.0),
            )
            _fig_plotly.add_annotation(
                x=(_x1 + _x2) / 2.0,
                y=_y + _bracket_height,
                text=significance_label(_pvalue),
                showarrow=False,
                yanchor="bottom",
                font=dict(color="black", size=12),
            )

    _fig_plotly.add_hline(y=0, line_color="black", line_dash="dash", line_width=0.8)
    _fig_plotly.update_xaxes(
        tickmode="array",
        tickvals=list(range(_n_features)),
        ticktext=_display_features,
        tickangle=35,
        title_text="",
    )
    _fig_plotly.update_yaxes(title_text="Weight", zeroline=False)
    _fig_plotly.update_layout(
        template="simple_white",
        boxmode="overlay",
        dragmode="select",
        height=430,
        width=max(650, int(140 * max(1, _n_features))),
        margin=dict(l=60, r=140, t=25, b=95),
        legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top"),
    )

    ui_emission_summary_plotly = mo.ui.plotly(_fig_plotly)
    selected_emission_subjects_plotly = sorted(
        {str(_point.get("customdata", [None])[0]) for _point in ui_emission_summary_plotly.points if _point.get("customdata")}
    )
    mo.vstack(
        [
            ui_emission_summary_plotly,
            mo.md("Selected subjects: " + (", ".join(selected_emission_subjects_plotly) if selected_emission_subjects_plotly else "_none_")),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Transition matrices
    """)
    return


@app.cell
def _(
    K,
    arrays_store,
    build_transition_matrix_by_subject_payload,
    build_transition_matrix_payload,
    mo,
    model_plots,
    save_plot,
    selected,
    state_labels,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _by_subject_payload = build_transition_matrix_by_subject_payload(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    _fig_by_subject = model_plots.transition_matrix_by_subject(**_by_subject_payload)

    _summary_payload = build_transition_matrix_payload(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    _fig_summary = model_plots.transition_matrix(**_summary_payload)
    mo.vstack(
        [
            _fig_summary,
            save_plot(
                _fig_summary,
                f"Mean Transition Matrix",
                stem=f"mean_transition_matrix",
            ),
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
    build_state_dwell_times_payload,
    fig_size,
    mo,
    model_plots,
    pl,
    save_plot,
    selected,
    trial_df,
    views,
):
    _views_sel = {s: views[s] for s in selected}
    mo.stop(not _views_sel, mo.md("No fitted arrays found — run the fit first."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    _dwell_payload = build_state_dwell_times_payload(
        _trial_df_sel,
        session_col="session",
        sort_col="trial_idx",
        views=_views_sel,
        max_dwell=90,
    )
    _dwell_ax_size = fig_size(2, 1)
    _n_dwell_states = max(1, len(_dwell_payload.get("state_order") or []))
    _n_dwell_subjects = max(1, len(_dwell_payload.get("subject_order") or []))
    _fig_dwell_summary, _axes_dwell_summary = model_plots.state_dwell_times_summary(
        _dwell_payload,
        figsize=(_dwell_ax_size[0] * _n_dwell_states, _dwell_ax_size[1]),
    )
    _fig_dwell_cumulative, _ax_dwell_cumulative = model_plots.state_dwell_times_cumulative(
        _dwell_payload,
        figsize=_dwell_ax_size,
    )
    _fig_dwell_median, _ax_dwell_median = model_plots.state_dwell_median_boxplot(
        _dwell_payload,
        figsize=_dwell_ax_size,
    )
    _fig_dwell_by_subject, _axes_dwell_by_subject = model_plots.state_dwell_times_by_subject(
        _dwell_payload,
        figsize=(_dwell_ax_size[0] * _n_dwell_states, _dwell_ax_size[1] * _n_dwell_subjects),
    )
    mo.vstack(
        [
            _fig_dwell_summary,
            save_plot(_fig_dwell_summary, "state dwell times summary", stem="state_dwell_times_summary"),
            _fig_dwell_cumulative,
            save_plot(_fig_dwell_cumulative, "state dwell times cumulative", stem="state_dwell_times_cumulative"),
            _fig_dwell_median,
            save_plot(_fig_dwell_median, "state dwell median boxplot", stem="state_dwell_median_boxplot"),
            _fig_dwell_by_subject,
            # save_plot(_fig_dwell_by_subject, "state dwell times by subject", stem="state_dwell_times_by_subject"),
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    ui_psychometric_background = mo.ui.radio(
        options={
            "Data traces": "data",
            "Model curves": "model",
            "None": "none",
        },
        value="Data traces",
        inline=False,
        label="Psychometric background",
    )
    ui_state_show_weighted_points = mo.ui.checkbox(value=True, label="Weighted dots")
    ui_state_show_data_smooth = mo.ui.checkbox(value=False, label="Data smooth")
    ui_state_assignment_mode = mo.ui.radio(
        options={
            "Predictive weights": "weighted",
            "MAP state": "map",
        },
        value="MAP state",
        inline=False,
        label="State assignment",
    )
    ui_state_model_line_mode = mo.ui.radio(
        options={
            "Smooth curve": "smooth",
            "Trial-matched": "trial_matched",
            "None": "none",
        },
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
    pl,
    plots,
    plt,
    prepare_predictions_df,
    save_plot,
    selected,
    task_name,
    trial_df,
    ui_psychometric_regressor,
    views,
):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _views_sel = {s: views[s] for s in selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    plot_df_all = prepare_predictions_df(task_name, _trial_df_sel)
    _state_plot_kwargs = dict(
        background_style="model",
        show_weighted_points=True,
        show_data_smooth=True,
        show_model_smooth=True,
        model_line_mode="smooth",
        state_assignment_mode="map",
        figure_dpi=300,
    )

    _fig_all, _ = plots.plot_categorical_performance_all(
        plot_df_all,
        f"glmhmm K={K}",
        background_style="model",
        views=_views_sel,
    )
    _fig_all_list = list(_fig_all) if isinstance(_fig_all, (list, tuple)) else [_fig_all]
    for _fig in _fig_all_list:
        for _ax_idx, _ax in enumerate(_fig.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if _fig._suptitle is not None:
            _fig._suptitle.set_text("")
        _fig.tight_layout()

    _fig_state_overlay, _ax_state_overlay = plt.subplots(figsize=fig_size(2, 1))
    _fig_state_overlay, _ = plots.plot_categorical_performance_state_overlay(
        df=plot_df_all,
        views=_views_sel,
        model_name=f"glmhmm K={K} — all states",
        ax=_ax_state_overlay,
        **_state_plot_kwargs,
    )

    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=plot_df_all,
        views=_views_sel,
        model_name=f"glmhmm K={K} — per state",
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

    _fig_reg_state, _axes_reg_state = plt.subplots(1, K, figsize=(4 * K, 4), sharey=True)
    _fig_reg_state, _ = plots.plot_regressor_psychometric_by_state(
        df=plot_df_all,
        views=_views_sel,
        model_name=f"glmhmm K={K}",
        feature_col=ui_psychometric_regressor.value,
        axes=_axes_reg_state,
        **_state_plot_kwargs,
    )

    mo.vstack(
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
            mo.md("#### State categorical performance — all states"),
            mo.vstack(
                [
                    _fig_state_overlay,
                    save_plot(_fig_state_overlay, "state-overlay psychometric", stem="categorical_state_overlay"),
                ],
                justify="space-between",
                align="center",
            ),
            mo.md("#### Per-state categorical performance"),
            _fig_state,
            save_plot(_fig_state, "per-state psychometric", stem="categorical_by_state"),
            mo.hstack([mo.md("#### Per-state psychometric by regressor"), ui_psychometric_regressor]),
            _fig_reg_overlay,
            save_plot(
                _fig_reg_overlay,
                f"{ui_psychometric_regressor.value} all states",
                stem=f"regressor_state_overlay_{ui_psychometric_regressor.value}",
            ),
            _fig_reg_state,
            save_plot(
                _fig_reg_state,
                f"{ui_psychometric_regressor.value} by state",
                stem=f"regressor_by_state_{ui_psychometric_regressor.value}",
            ),
        ],
        align="center",
    )
    return (plot_df_all,)


@app.cell
def _(mo, plot_df_all, plots, save_plot, selected, views):
    _views_sel = {s: views[s] for s in selected}
    _fig_repeat_evidence = plots.plot_repeat_by_repeat_evidence(
        plot_df_all,
        views=_views_sel,
    )
    mo.stop(
        _fig_repeat_evidence is None,
        mo.md("No repeat evidence plot available for the selected task/features."),
    )

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
    )
    return


@app.cell
def _(fig_size, mo, plot_df_all, plots, plt, save_plot):
    mo.stop(
        "choice_lag_param" not in plot_df_all.columns,
        mo.md("No choice-history regressor available for p(right) by regressor."),
    )

    fig_right, ax_right = plt.subplots(figsize=fig_size(2,1))
    _ax_right_regressor = plots.plot_right_by_regressor(
        plot_df_all,
        regressor_col="choice_lag_param",
        title=None,
        ax=ax_right,
    )

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
                stem="right_by_choice_lag_param",
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
    _ax_integration_data.set_ylabel(r"$A_t^{\mathrm{choice,param}}$")
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
    mo.stop(
        "choice_lag_param" not in plot_df_all.columns,
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
        regressor_col="choice_lag_param",
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
                stem="accuracy_binned_choice_lag_param",
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
    _trial_df_sub = editor_trial_df
    _edited_weights = np.asarray(coef_editor.value["weights"], dtype=float)

    _trial_df_tweaked = apply_state_tweak_to_trial_df(
        _trial_df_sub,
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

    _title = f"{_subj} — tweaked {coef_state_label}"
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
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            overlay_only=True,
        )
    else:
        _fig_state_overlay_tweaked_base, _ax_state_overlay_tweaked = plt.subplots(figsize=(3, 3))
        _fig_state_overlay_tweaked, _ = _state_overlay_fn(
            df=_plot_df_tweaked,
            views={_subj: _view_tweaked},
            model_name=f"{_title} — all states",
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            ax=_ax_state_overlay_tweaked,
        )
    _fig_state_tweaked, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_tweaked,
        views={_subj: _view_tweaked},
        model_name=f"{_title} — per state",
        background_style=ui_psychometric_background.value,
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
    )
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
    _side_plot_fn = getattr(plots, "plot_categorical_strat_by_side", None)
    if _side_plot_fn is None:
        _side_section = mo.md("This task does not expose a side-stratified categorical plot.")
    else:
        _fig_side_tweaked, _ = plots.plot_categorical_strat_by_side(
            _plot_df_tweaked,
            subject=_subj,
            model_name=f"{_subj}_tweaked_{coef_state_idx}",
        )
        _side_section = mo.vstack(
            [
                _fig_side_tweaked,
                save_plot(
                    _fig_side_tweaked,
                    "tweaked psychometric by stimulus side",
                    stem="tweaked_categorical_by_side",
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
                    mo.vstack(
                        [
                            ui_psychometric_background,
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
            ),
            mo.vstack(
                [
                    _fig_state_overlay_tweaked,
                    save_plot(
                        _fig_state_overlay_tweaked,
                        "tweaked state-overlay psychometric",
                        stem="tweaked_categorical_state_overlay",
                    ),
                    _fig_state_tweaked,
                    save_plot(
                        _fig_state_tweaked,
                        "tweaked per-state psychometric",
                        stem="tweaked_categorical_by_state",
                    ),
                ],
                align="center",
            ),
            _reg_section,
            _side_section,
            coef_editor,
        ],
        align="center",
    )
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
def _(
    adapter,
    build_state_accuracy_payload,
    build_state_posterior_count_payload,
    mo,
    model_plots,
    pl,
    plt,
    save_plot,
    selected,
    trial_df,
):
    mo.stop(not selected, mo.md("No fitted subjects available."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))

    _fig_acc = model_plots.state_accuracy(
        build_state_accuracy_payload(
            _trial_df_sel,
            performance_col="correct_bool",
            chance_level=1.0 / adapter.num_classes,
        )
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    _fig_post = model_plots.state_posterior_count_kde(build_state_posterior_count_payload(_trial_df_sel), ax=ax)
    ax.spines["right"].set_visible(True)

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
                ],
                align="center",
            ),
            mo.md("**Trial counts & mean accuracy per label:**"),
        ]
    )
    return


@app.cell
def _(df_all, mo):
    # ── controls for session-trajectory & occupancy plots ─────────────────────
    ui_subjects_traj = mo.ui.multiselect(
        options=sorted(df_all["subject"].unique().to_list(), key=str), label="Subjects (session trajectories & occupancy)", value=""
    )
    mo.vstack([mo.md("### Session trajectory & occupancy"), ui_subjects_traj])
    return


@app.cell
def _(
    build_session_trajectories_payload,
    mo,
    model_plots,
    pl,
    selected,
    trial_df,
):
    mo.stop(not selected, mo.md("Select subjects above to view session trajectories."))
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    _fig_traj = model_plots.session_trajectories(
        build_session_trajectories_payload(
            _trial_df_sel,
            session_col="session",
            sort_col="trial_idx",
        )
    )
    mo.vstack(
        [
            # mo.md(f"### c. Average state-probability trajectories within a session  (K={K})"),
            # _fig_traj,
            # mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
        ],
        align="center",
    )
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
    mo,
    model_plots,
    np,
    pd,
    pl,
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
    _fig_occ_overall_summary = model_plots.state_occupancy_overall_summary(_occupancy_payload)
    _fig_occ_overall_by_subject = model_plots.state_occupancy_overall_by_subject(_occupancy_payload)
    _fig_occ_sessions_summary = model_plots.state_session_occupancy_summary(_occupancy_payload)
    _fig_occ_sessions_by_subject = model_plots.state_session_occupancy_by_subject(_occupancy_payload)
    _fig_occ_switches_summary = model_plots.state_switches_summary(_occupancy_payload)
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
    ui_occ_switches_summary = mo.ui.matplotlib(_fig_occ_switches_summary, debounce=True)
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.vstack(
                        [
                            _fig_occ_overall_summary.figure,
                            save_plot(
                                _fig_occ_overall_summary.figure,
                                "fractional occupancy overall summary",
                                stem="state_occupancy_overall_summary",
                                location=(0, 0),
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
                            _fig_occ_sessions_summary.figure,
                            save_plot(
                                _fig_occ_sessions_summary.figure,
                                "fractional occupancy by session summary",
                                stem="state_session_occupancy_summary",
                                location=(0, 0),
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
                            ui_occ_switches_summary,
                            save_plot(
                                _fig_occ_switches_summary.figure,
                                "state switches summary",
                                stem="state_switches_summary",
                                location=(0, 0),
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
        _mask = ui_occ_switches_summary.value.get_mask(
            _points["x"].to_numpy(),
            _points["y"].to_numpy(),
        )
        selected_state_switch_counts = sorted(_points.loc[_mask, "n_switches"].unique().tolist())
    selected_state_switch_sessions = (
        state_switch_sessions_df[state_switch_sessions_df["n_switches"].isin(selected_state_switch_counts)]
        .sort_values(["n_switches", "subject", "session"])
        .reset_index(drop=True)
    )
    mo.vstack(
        [
            mo.md(
                "Selected switch counts: " + (", ".join(map(str, selected_state_switch_counts)) if selected_state_switch_counts else "_none_")
            ),
            selected_state_switch_sessions
            if selected_state_switch_counts
            else mo.md("Select one or more histogram bars to list matching sessions."),
        ]
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
    mo,
    model_plots,
    pl,
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
        switch_posterior_threshold=THRESH_ui.amount,
    )
    _fig_change_summary = model_plots.change_triggered_posteriors_summary(
        _change_payload,
    )
    _fig_change_by_subject = model_plots.change_triggered_posteriors_by_subject(
        _change_payload,
    )
    mo.vstack(
        [
            mo.md(f"> Change events use the same confident MAP switch rule as the histogram above: posterior ≥ {THRESH_ui}. "),
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
    return ui_engaged_trace_mode, ui_engaged_window, ui_session_id


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
    save_plot,
    trial_df,
    ui_engaged_trace_mode,
    ui_engaged_window,
    ui_session_id,
    ui_session_subj,
    views,
):
    _subj = ui_session_subj.value
    mo.stop(
        _subj not in views,
        mo.md("No fitted arrays for this subject — run the fit first."),
    )

    _sess = int(ui_session_id.value) if str(ui_session_id.value).isdigit() else ui_session_id.value
    _deepdive_payload = build_session_deepdive_payload(
        trial_df,
        subject=_subj,
        session=_sess,
        session_col="session",
        sort_col="trial",
        engaged_window=ui_engaged_window.value,
        engaged_trace_mode=ui_engaged_trace_mode.value,
        chance_level=1.0 / adapter.num_classes,
        num_classes=adapter.num_classes,
        views=views,
    )
    _fig = model_plots.session_deepdive(_deepdive_payload)
    _fig_traces = model_plots.session_deepdive_state_traces(_deepdive_payload)
    mo.vstack(
        [
            mo.hstack([ui_session_subj, ui_session_id, ui_engaged_window, ui_engaged_trace_mode]),
            _fig,
            _fig_traces,
            save_plot(
                _fig,
                "session statistics",
                stem=f"session_stats_{_subj}_{_sess}",
            ),
            save_plot(
                _fig_traces,
                "session state traces",
                stem=f"session_state_traces_{_subj}_{_sess}",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(pl, trial_df):
    trial_df.filter(pl.col("subject") == "E10", pl.col("session") == 32)["drug"].unique()
    return


@app.cell
def _(K, df_all, mo, model_cfg, np, paths, pl, plt, save_plot, sns):
    # ── τ sweep analysis ────────────────────────────────────────────────────────
    # Loads results produced by:
    #   uv run glmhmmt-fit-tau-sweep --model glmhmm --K <K>
    # Expects: RESULTS/fits/tau_sweep/glmhmm_K<K>/tau_sweep_summary.parquet

    _sweep_path = paths.RESULTS / "fits" / "tau_sweep" / f"glmhmm_K{K}" / "tau_sweep_summary.parquet"
    mo.stop(
        not _sweep_path.exists(),
        mo.md(
            f"**τ sweep results not found.**  \
     Run the sweep first:\n```\n"
            f"uv run glmhmmt-fit-tau-sweep --model glmhmm --K {K}\n```"
        ),
    )

    _df_sweep = pl.read_parquet(_sweep_path)
    _subjects = [s for s in model_cfg.subjects if s in _df_sweep["subject"].unique().to_list()]
    mo.stop(not _subjects, mo.md("No sweep data for selected subjects."))

    # ── BIC vs τ plot ────────────────────────────────────────────────────
    _fig_sweep, _axes_sw = plt.subplots(1, 2, figsize=(12, 4))
    _ax_bic, _ax_ll = _axes_sw
    _palette = sns.color_palette("tab10", n_colors=len(_subjects))
    n_trials = df_all.group_by("subject").agg(pl.len().alias("n_trials"))

    for _i, _subj in enumerate(_subjects):
        _d = _df_sweep.filter((pl.col("subject") == _subj) & (pl.col("K") == K)).sort("tau")
        _tau = _d["tau"].to_numpy()
        _bic = _d["bic"].to_numpy()
        _ll = _d["ll_per_trial"].to_numpy()
        _c = _palette[_i]
        _ax_bic.plot(_tau, _bic, "-o", ms=3, color=_c, label=_subj)
        _ax_ll.plot(_tau, _ll, "-o", ms=3, color=_c, label=_subj)
        # mark best τ
        _best_idx = int(np.argmin(_bic))
        _ax_bic.axvline(_tau[_best_idx], color=_c, lw=0.8, linestyle="--", alpha=0.6)
    4
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

    # ── best τ table ────────────────────────────────────────────────────────
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
        .join(n_trials, on="subject", how="left")
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
        .select(
            [
                "tau",
                "bic_mean_w",
                "ll_per_trial_mean_w",
                "acc_mean_w",
                "n_subjects",
                "n_total",
            ]
        )
        .sort("bic_mean_w")
    )

    mo.vstack(
        [
            mo.md(f"### τ sweep results — glmhmm K={K}"),
            _fig_sweep,
            save_plot(
                _fig_sweep,
                "tau sweep results",
                stem=f"tau_sweep_glmhmm_k{K}",
            ),
            mo.md("**Best τ per subject (min BIC):**"),
            mo.plain_text(_best.to_pandas().to_string(index=False)),
            mo.ui.dataframe(_best_all),
        ],
        align="center",
    )
    return


@app.cell
def _(mo, task_name):
    # ── SSM GLM-HMM safety check (2AFC only) ──────────────────────────────────
    mo.stop(
        task_name != "2AFC_DRUG",
        mo.md("ℹ️ **SSM safety check is only available for the 2AFC task.** Switch task to 2AFC above."),
    )
    ssm_run_btn = mo.ui.run_button(label="▶ Run SSM safety check")
    mo.vstack(
        [
            mo.md("### SSM GLM-HMM safety check (2AFC)"),
            mo.md(
                "Fits a K-state GLM-HMM using the **SSM library** (`input_driven_obs`, `standard` "
                "transitions) with the exact same covariates as the custom model.  \n"
            ),
            ssm_run_btn,
        ]
    )
    return (ssm_run_btn,)


@app.cell
def _(
    K,
    adapter,
    build_trial_df,
    build_views,
    df_all,
    mo,
    model_cfg,
    np,
    paths,
    pl,
    plots,
    prepare_predictions_df,
    selected_model_id,
    ssm_run_btn,
    task_name,
    trial_df,
    views,
):
    # ── SSM fit + comparison tables ────────────────────────────────────────────
    mo.stop(not ssm_run_btn.value, mo.md("Press **▶ Run SSM safety check** above to fit."))
    try:
        import ssm as ssm_lib
    except Exception as exc:
        mo.stop(
            True,
            mo.md(
                "SSM could not be imported in the current environment, so the SSM vs custom log-likelihood comparison cannot run. "
                f"Import error: `{type(exc).__name__}: {exc}`. "
                "In this project, that usually means `ssm` is installed but incompatible with the currently resolved "
                "`autograd`/`numpy` versions, not that `uv` is using the wrong environment."
            ),
        )
    from glmhmmt.cli.fit_common import valid_trial_mask

    ssm_subjects = [subject for subject in model_cfg.subjects if subject in views]
    mo.stop(not ssm_subjects, mo.md("No fitted arrays found — run the custom fit first."))

    ssm_arrays = {}
    cmp_rows = []
    missing_metric_subjects = []
    out_dir = paths.RESULTS / "fits" / task_name / "glmhmm" / selected_model_id


    def load_custom_metrics(subject: str, n_trials: int):
        candidates = [
            out_dir / f"{subject}_K{K}_glmhmm_metrics.parquet",
            out_dir / f"{subject}_glmhmm_metrics.parquet",
            *sorted(out_dir.glob(f"{subject}*_glmhmm_metrics.parquet")),
        ]
        for path in dict.fromkeys(candidates):
            if not path.exists():
                continue
            metrics_df = pl.read_parquet(path)
            if metrics_df.height == 0:
                continue
            row = metrics_df.row(0, named=True)
            raw_ll = row.get("raw_ll")
            ll_per_trial = row.get("ll_per_trial")
            if raw_ll is None and ll_per_trial is not None:
                raw_ll = float(ll_per_trial) * n_trials
            if ll_per_trial is None and raw_ll is not None:
                ll_per_trial = float(raw_ll) / max(n_trials, 1)
            if raw_ll is None or ll_per_trial is None:
                continue
            return float(raw_ll), float(ll_per_trial), path.name
        return np.nan, np.nan, None


    def ssm_data_loglik(model, choices_list, inputs_list):
        if hasattr(model, "log_likelihood"):
            return float(model.log_likelihood(choices_list, inputs=inputs_list)), "log_likelihood"
        if hasattr(model, "log_probability"):
            return float(model.log_probability(choices_list, inputs=inputs_list)), "log_probability"
        raise AttributeError("SSM HMM object exposes neither log_likelihood nor log_probability.")


    def stable_softmax_np(logits: np.ndarray) -> np.ndarray:
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


    with mo.status.spinner(title="Fitting SSM GLM-HMM…"):
        for _subject in ssm_subjects:
            view = views[_subject]
            X = np.asarray(view.X)  # (T, n_feat) — already session-filtered
            y = np.asarray(view.y)  # (T,)

            # Reconstruct session ids with same mask as fit_subject()
            subject_df = df_all.filter(pl.col("subject") == _subject).sort(adapter.sort_col)
            session_ids_raw = subject_df[adapter.session_col].to_numpy()
            valid_mask = valid_trial_mask(session_ids_raw)
            session_ids = session_ids_raw[valid_mask]

            # Split into per-session lists — SSM expects list of arrays
            unique_sessions = list(dict.fromkeys(session_ids.tolist()))
            choices_list, inputs_list = [], []
            for session_id in unique_sessions:
                idx = np.where(session_ids == session_id)[0]
                choices_list.append(y[idx].reshape(-1, 1).astype(int))
                inputs_list.append(X[idx].astype(float))

            # Initialise and fit
            obs_dim = 1
            n_cats = 2
            n_feat = X.shape[1]
            glmhmm_ssm = ssm_lib.HMM(
                K,
                obs_dim,
                n_feat,
                observations="input_driven_obs",
                observation_kwargs=dict(C=n_cats),
                transitions="standard",
            )
            glmhmm_ssm.fit(
                choices_list,
                inputs=inputs_list,
                method="em",
                num_iters=200,
                tolerance=1e-4,
            )

            W_ssm = glmhmm_ssm.observations.params  # (K, C-1, n_feat); flip sign
            transition_matrix_ssm = glmhmm_ssm.transitions.transition_matrix  # (K, K)
            smoothed_probs_ssm = np.vstack(
                [glmhmm_ssm.expected_states(data=data, input=inp)[0] for data, inp in zip(choices_list, inputs_list)]
            )
            initial_state_distn_ssm = np.asarray(glmhmm_ssm.init_state_distn.initial_state_distn, dtype=float)
            p_pred_ssm_parts = []
            for data, inp in zip(choices_list, inputs_list):
                filtered_probs = np.asarray(glmhmm_ssm.filter(data=data, input=inp), dtype=float)  # (T_s, K)
                n_trials_session = int(inp.shape[0])
                pred_z_session = (
                    np.vstack(
                        [
                            initial_state_distn_ssm[None, :],
                            filtered_probs[:-1] @ transition_matrix_ssm,
                        ]
                    )
                    if n_trials_session > 1
                    else initial_state_distn_ssm[None, :]
                )
                logits_ce_session = np.einsum("kcf,tf->tkc", W_ssm, np.asarray(inp, dtype=float))
                logits_session = np.concatenate(
                    [
                        logits_ce_session,
                        np.zeros((n_trials_session, K, 1), dtype=float),
                    ],
                    axis=-1,
                )
                p_y_given_z_session = stable_softmax_np(logits_session)  # (T_s, K, C)
                p_pred_ssm_parts.append(np.einsum("tk,tkc->tc", pred_z_session, p_y_given_z_session))
            p_pred_ssm = np.concatenate(p_pred_ssm_parts, axis=0)
            ssm_raw_ll, ssm_ll_source = ssm_data_loglik(glmhmm_ssm, choices_list, inputs_list)
            _n_trials = int(y.shape[0])
            ssm_ll_per_trial = ssm_raw_ll / max(_n_trials, 1)
            custom_raw_ll, custom_ll_per_trial, metric_file = load_custom_metrics(_subject, _n_trials)
            if metric_file is None:
                missing_metric_subjects.append(_subject)

            cmp_rows.append(
                {
                    "subject": _subject,
                    "n_trials": _n_trials,
                    "custom_raw_ll": custom_raw_ll,
                    "ssm_raw_ll": ssm_raw_ll,
                    "delta_raw_ll_ssm_minus_custom": ssm_raw_ll - custom_raw_ll,
                    "custom_ll_per_trial": custom_ll_per_trial,
                    "ssm_ll_per_trial": ssm_ll_per_trial,
                    "delta_ll_per_trial_ssm_minus_custom": ssm_ll_per_trial - custom_ll_per_trial,
                    "custom_metrics_file": metric_file,
                    "ssm_ll_source": ssm_ll_source,
                }
            )

            ssm_arrays[_subject] = {
                "smoothed_probs": smoothed_probs_ssm,
                "emission_weights": W_ssm,
                "transition_matrix": transition_matrix_ssm,
                "X": X,
                "y": y,
                "X_cols": np.array(list(view.feat_names), dtype=object),
                "p_pred": p_pred_ssm,
            }

    ssm_views = build_views(ssm_arrays, adapter, K, ssm_subjects)
    views_sel = {subject: views[subject] for subject in ssm_subjects}
    ssm_views_sel = {subject: ssm_views[subject] for subject in ssm_subjects}
    trial_df_custom_sel = trial_df.filter(pl.col("subject").is_in(ssm_subjects))
    sort_col = adapter.sort_col
    session_col = adapter.session_col
    behavioral_cols = adapter.behavioral_cols
    trial_frames_ssm = []
    for _subject, view in ssm_views_sel.items():
        subject_df = df_all.filter(pl.col("subject") == _subject).sort(sort_col).filter(pl.col(session_col).count().over(session_col) >= 2)
        if subject_df.height != view.T:
            continue
        trial_frames_ssm.append(build_trial_df(view, adapter, subject_df, behavioral_cols))
    trial_df_ssm = pl.concat(trial_frames_ssm) if trial_frames_ssm else pl.DataFrame()

    ssm_psych_fig_custom = None
    ssm_psych_fig_ssm = None
    if trial_df_custom_sel.height > 0 and trial_df_ssm.height > 0:
        plot_df_custom = prepare_predictions_df(task_name, trial_df_custom_sel)
        ssm_psych_fig_custom, _ = plots.plot_categorical_performance_all(
            plot_df_custom,
            f"Dynamax glmhmm K={K}",
            views=views_sel,
        )
        for _ax_idx, _ax in enumerate(ssm_psych_fig_custom.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if ssm_psych_fig_custom._suptitle is not None:
            ssm_psych_fig_custom._suptitle.set_text("")
        ssm_psych_fig_custom.tight_layout()
        plot_df_ssm = prepare_predictions_df(task_name, trial_df_ssm)
        ssm_psych_fig_ssm, _ = plots.plot_categorical_performance_all(
            plot_df_ssm,
            f"SSM glmhmm K={K}",
            views=ssm_views_sel,
        )
        for _ax_idx, _ax in enumerate(ssm_psych_fig_ssm.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if ssm_psych_fig_ssm._suptitle is not None:
            ssm_psych_fig_ssm._suptitle.set_text("")
        ssm_psych_fig_ssm.tight_layout()

    ssm_cmp_df = pl.DataFrame(cmp_rows)
    contrast_labels = list(adapter.choice_labels[:-1]) or ["contrast_0"]
    coef_rows = []
    for _subject in ssm_subjects:
        custom_view = views[_subject]
        _ssm_view = ssm_views[_subject]
        custom_feat_names = list(custom_view.feat_names)
        ssm_feat_names = list(_ssm_view.feat_names)
        feat_names = (
            custom_feat_names
            if custom_feat_names == ssm_feat_names
            else [
                custom_feat_names[i] if i < len(custom_feat_names) else ssm_feat_names[i]
                for i in range(min(len(custom_feat_names), len(ssm_feat_names)))
            ]
        )

        for state_rank, (custom_k, ssm_k) in enumerate(zip(custom_view.state_idx_order, _ssm_view.state_idx_order, strict=False)):
            custom_label = custom_view.state_name_by_idx.get(int(custom_k), f"State {custom_k}")
            ssm_label = _ssm_view.state_name_by_idx.get(int(ssm_k), f"State {ssm_k}")
            state_label = custom_label if custom_label == ssm_label else f"{custom_label} | {ssm_label}"
            custom_w = np.asarray(custom_view.emission_weights[int(custom_k)], dtype=float)
            ssm_w = np.asarray(_ssm_view.emission_weights[int(ssm_k)], dtype=float)
            n_contrasts = min(custom_w.shape[0], ssm_w.shape[0], len(contrast_labels))
            n_features = min(custom_w.shape[1], ssm_w.shape[1], len(feat_names))

            for contrast_idx in range(n_contrasts):
                for feature_idx in range(n_features):
                    custom_coef = float(custom_w[contrast_idx, feature_idx])
                    ssm_coef = -float(ssm_w[contrast_idx, feature_idx])
                    coef_rows.append(
                        {
                            "subject": _subject,
                            "state_rank": int(state_rank),
                            "state_label": state_label,
                            "custom_state_idx": int(custom_k),
                            "ssm_state_idx": int(ssm_k),
                            "contrast": contrast_labels[contrast_idx],
                            "feature": feat_names[feature_idx],
                            "dynamax_coef": custom_coef,
                            "ssm_coef": ssm_coef,
                            "delta_ssm_minus_dynamax": abs(ssm_coef + custom_coef),
                        }
                    )

    ssm_coef_df = (
        pl.DataFrame(coef_rows)
        if coef_rows
        else pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "state_rank": pl.Int64,
                "state_label": pl.Utf8,
                "custom_state_idx": pl.Int64,
                "ssm_state_idx": pl.Int64,
                "contrast": pl.Utf8,
                "feature": pl.Utf8,
                "dynamax_coef": pl.Float64,
                "ssm_coef": pl.Float64,
                "delta_ssm_minus_dynamax": pl.Float64,
            }
        )
    )
    ssm_coef_df = ssm_coef_df.sort(["subject", "state_rank", "contrast", "feature"])
    ssm_coef_display = ssm_coef_df.select(
        [
            "subject",
            "state_rank",
            "state_label",
            "custom_state_idx",
            "ssm_state_idx",
            "contrast",
            "feature",
            "dynamax_coef",
            "ssm_coef",
            "delta_ssm_minus_dynamax",
        ]
    )

    cmp_valid = ssm_cmp_df.filter(pl.col("custom_raw_ll").is_finite())
    if cmp_valid.height > 0:
        custom_total_raw = float(cmp_valid["custom_raw_ll"].sum())
        ssm_total_raw = float(cmp_valid["ssm_raw_ll"].sum())
        total_trials = int(cmp_valid["n_trials"].sum())
        ssm_summary_md = "\n".join(
            [
                "### Log-likelihood comparison",
                "",
                f"- Compared on **{cmp_valid.height} subject(s)** and **{total_trials} trials**.",
                f"- **Custom / Dynamax total raw LL:** `{custom_total_raw:.3f}`",
                f"- **SSM total raw LL:** `{ssm_total_raw:.3f}`",
                f"- **Δ raw LL (SSM - custom):** `{ssm_total_raw - custom_total_raw:.3f}`",
                f"- **Custom / Dynamax LL per trial:** `{custom_total_raw / max(total_trials, 1):.6f}`",
                f"- **SSM LL per trial:** `{ssm_total_raw / max(total_trials, 1):.6f}`",
                f"- **Δ LL per trial (SSM - custom):** `{(ssm_total_raw - custom_total_raw) / max(total_trials, 1):.6f}`",
            ]
        )
    else:
        ssm_summary_md = (
            "### Log-likelihood comparison\n\n"
            "No matching saved custom metrics were found for the selected fit, so only the "
            "SSM posterior overlay is shown below."
        )

    notes = []
    if missing_metric_subjects:
        notes.append("Missing custom metrics for: " + ", ".join(sorted(dict.fromkeys(missing_metric_subjects))))
    ssm_sources = sorted(dict.fromkeys(ssm_cmp_df["ssm_ll_source"].to_list())) if ssm_cmp_df.height > 0 else []
    if ssm_sources and ssm_sources != ["log_likelihood"]:
        notes.append("SSM LL used fallback method(s): " + ", ".join(ssm_sources))
    ssm_notes_md = (
        "  \n".join(f"- {note}" for note in notes)
        if notes
        else "- `raw_ll` is the data log-likelihood from the saved custom fit metrics.  \n"
        "- `delta` columns are defined as **SSM - custom / Dynamax**."
    )
    ssm_notes_md += (
        "  \n- Emission coefficients are compared after each model's states are reordered by the notebook's "
        "semantic state labelling (`state_idx_order`), not by raw fitted state index."
    )

    mo.vstack(
        [
            mo.md("### SSM GLM-HMM fit summary"),
            mo.md(ssm_summary_md),
            mo.md(ssm_notes_md),
            mo.ui.dataframe(ssm_cmp_df),
            mo.md("### Emission coefficients — SSM vs Dynamax"),
            mo.md(
                "Each row below is one fitted emission coefficient for one subject, aligned by the notebook's "
                "state order. `delta_ssm_minus_dynamax > 0` means the SSM coefficient is larger."
            ),
            mo.ui.dataframe(ssm_coef_display),
        ],
        align="center",
    )
    return (
        cmp_valid,
        ssm_coef_df,
        ssm_psych_fig_custom,
        ssm_psych_fig_ssm,
        ssm_subjects,
        ssm_views,
    )


@app.cell
def _(
    K,
    adapter,
    cmp_valid,
    np,
    plt,
    sns,
    ssm_coef_df,
    ssm_subjects,
    ssm_views,
    ui_trial_range,
    views,
):
    def choice_meta(num_classes: int):
        if num_classes == 2:
            return {0: "royalblue", 1: "tomato"}
        return {0: "royalblue", 1: "gold", 2: "tomato"}


    def choice_short_labels(labels):
        return {int(i): str(label)[0].upper() for i, label in enumerate(labels)}


    def posterior_color(rank: int):
        palette = ["tab:green", "tab:grey", *sns.color_palette("tab10", n_colors=max(0, K - 2))]
        if rank < len(palette):
            return palette[rank]
        return sns.color_palette("tab10", n_colors=K)[rank % K]


    def plot_view_posterior(
        ax,
        view,
        title: str,
        t0_plot: int,
        t1_plot: int,
        overlay_line=None,
        overlay_label: str | None = None,
    ):
        probs = np.asarray(view.smoothed_probs)[t0_plot : t1_plot + 1]
        y_window = np.asarray(view.y).astype(int)[t0_plot : t1_plot + 1]
        n_trials_window = probs.shape[0]
        x_window = np.arange(t0_plot, t0_plot + n_trials_window)
        bottom = np.zeros(n_trials_window)

        for state_idx in list(view.state_idx_order):
            rank = view.state_rank_by_idx.get(int(state_idx), int(state_idx))
            color = posterior_color(rank)
            ax.fill_between(
                x_window,
                bottom,
                bottom + probs[:, state_idx],
                alpha=0.7,
                color=color,
                label=view.state_name_by_idx.get(state_idx, f"State {state_idx}"),
            )
            bottom += probs[:, state_idx]

        engaged_state = view.engaged_k()
        engaged_label = view.state_name_by_idx.get(engaged_state, f"State {engaged_state}")
        ax.plot(
            x_window,
            probs[:, engaged_state],
            color="black",
            lw=1.4,
            alpha=0.95,
            label=f"P({engaged_label})",
        )
        if overlay_line is not None:
            ax.plot(
                x_window,
                np.asarray(overlay_line)[:n_trials_window],
                color="darkorange",
                lw=2,
                alpha=0.95,
                linestyle="--",
                label=overlay_label or "Overlay",
            )

        choice_colors = choice_meta(view.num_classes)
        choice_labels = choice_short_labels(adapter.choice_labels)
        for response, color in choice_colors.items():
            mask = y_window == response
            if not np.any(mask):
                continue
            ax.scatter(
                x_window[mask],
                np.ones(mask.sum()) * 1.03,
                c=color,
                s=4,
                marker="|",
                label=choice_labels.get(response, str(response)),
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

        ax.set_xlim(t0_plot, t0_plot + n_trials_window - 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("State probability")
        ax.set_title(title)
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            ncol=1,
            frameon=False,
        )


    ssm_ll_fig = None
    if cmp_valid.height > 0:
        import plotly.graph_objects as go

        cmp_pd = cmp_valid.select(["subject", "custom_ll_per_trial", "ssm_ll_per_trial"]).to_pandas()
        ssm_ll_fig = go.Figure()

        for row in cmp_pd.itertuples(index=False):
            ssm_ll_fig.add_trace(
                go.Scatter(
                    x=["Dynamax", "SSM"],
                    y=[row.custom_ll_per_trial, row.ssm_ll_per_trial],
                    mode="lines+markers",
                    line=dict(color="rgba(120, 120, 120, 0.22)", width=1.2),
                    marker=dict(color="rgba(0, 0, 0, 0.65)", size=7),
                    customdata=[row.subject, row.subject],
                    hovertemplate="Subject: %{customdata}<br>Model: %{x}<br>LL/trial: %{y:.6f}<extra></extra>",
                    showlegend=False,
                )
            )

        ssm_ll_fig.add_trace(
            go.Box(
                x=["Dynamax"] * len(cmp_pd),
                y=cmp_pd["custom_ll_per_trial"],
                name="Dynamax",
                marker_color="rgba(180, 180, 180, 0.9)",
                fillcolor="rgba(217, 217, 217, 0.6)",
                line=dict(color="rgba(90, 90, 90, 0.9)"),
                boxpoints=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        ssm_ll_fig.add_trace(
            go.Box(
                x=["SSM"] * len(cmp_pd),
                y=cmp_pd["ssm_ll_per_trial"],
                name="SSM",
                marker_color="rgba(180, 180, 180, 0.9)",
                fillcolor="rgba(217, 217, 217, 0.6)",
                line=dict(color="rgba(90, 90, 90, 0.9)"),
                boxpoints=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        ssm_ll_fig.update_layout(
            title="Per-subject LL comparison",
            xaxis_title=None,
            yaxis_title="Log-likelihood per trial",
            template="simple_white",
            width=560,
            height=420,
            margin=dict(l=60, r=20, t=60, b=50),
        )
        ssm_ll_fig.update_yaxes(zeroline=False)
        ssm_ll_fig.update_xaxes(categoryorder="array", categoryarray=["Dynamax", "SSM"])

    ssm_coef_fig = None
    if ssm_coef_df.height > 0:
        coef_pd = ssm_coef_df.to_pandas()
        panel_keys = (
            coef_pd[["state_rank", "state_label", "contrast"]].drop_duplicates().sort_values(["state_rank", "contrast"]).to_dict("records")
        )
        n_panels = len(panel_keys)
        n_cols = 1 if n_panels == 1 else min(2, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        ssm_coef_fig, coef_axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(8, 5.5 * n_cols), max(3.6, 3.6 * n_rows)),
            squeeze=False,
            sharey=True,
        )
        axes_flat = coef_axes.ravel()
        for _ax, key in zip(axes_flat, panel_keys, strict=False):
            mask = (
                (coef_pd["state_rank"] == key["state_rank"])
                & (coef_pd["state_label"] == key["state_label"])
                & (coef_pd["contrast"] == key["contrast"])
            )
            panel_df = coef_pd.loc[mask].copy()
            sns.boxplot(
                data=panel_df,
                x="feature",
                y="delta_ssm_minus_dynamax",
                ax=_ax,
                showfliers=False,
                color="#D9D9D9",
                boxprops={"alpha": 0.8},
            )
            sns.stripplot(
                data=panel_df,
                x="feature",
                y="delta_ssm_minus_dynamax",
                ax=_ax,
                color="black",
                alpha=0.7,
                size=4,
                jitter=0.22,
            )
            _ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.7)
            _ax.set_title(f"{key['state_label']}  ({key['contrast']})")
            _ax.set_xlabel("")
            _ax.set_ylabel("SSM - Dynamax coefficient")
            _ax.tick_params(axis="x", rotation=35)
            _ax.set_yscale("log")
            sns.despine(ax=_ax)
        for _ax in axes_flat[n_panels:]:
            _ax.set_visible(False)
        ssm_coef_fig.tight_layout()

    t0_ssm, t1_ssm = ui_trial_range.value
    n_subjects = len(ssm_subjects)
    ssm_posterior_fig, axes_ssm = plt.subplots(n_subjects, 1, figsize=(14, 3.4 * n_subjects), squeeze=False)

    for i, subject in enumerate(ssm_subjects):
        ssm_view = ssm_views[subject]
        ssm_engaged_probs = np.asarray(ssm_view.smoothed_probs)[t0_ssm : t1_ssm + 1, ssm_view.engaged_k()]
        plot_view_posterior(
            axes_ssm[i, 0],
            views[subject],
            f"Subject {subject} — Custom posterior + SSM line",
            t0_ssm,
            t1_ssm,
            overlay_line=ssm_engaged_probs,
            overlay_label="SSM P(Engaged)",
        )

    axes_ssm[-1, 0].set_xlabel("Trial")
    ssm_posterior_fig.tight_layout()
    ssm_posterior_fig.subplots_adjust(right=0.84)
    sns.despine(fig=ssm_posterior_fig)
    return ssm_coef_fig, ssm_ll_fig, ssm_posterior_fig


@app.cell
def _(mo, save_plot, ssm_posterior_fig):
    mo.vstack(
        [
            ssm_posterior_fig,
            save_plot(
                ssm_posterior_fig,
                "ssm posterior overlay",
                stem="ssm_posterior_overlay",
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SSM GLM-HMM plots
    """)
    return


@app.cell
def _(
    K,
    mo,
    save_plot,
    ssm_coef_fig,
    ssm_ll_fig,
    ssm_psych_fig_custom,
    ssm_psych_fig_ssm,
):
    mo.vstack(
        [
            mo.md("### Log-likelihood comparison"),
            ssm_ll_fig if ssm_ll_fig is not None else mo.md("LL comparison unavailable because subject-level metrics are missing."),
            mo.md("### Categorical psychometrics — Dynamax vs SSM"),
            (
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                mo.md("#### Dynamax"),
                                ssm_psych_fig_custom,
                                save_plot(
                                    ssm_psych_fig_custom,
                                    "ssm comparison dynamax psychometric",
                                    stem=f"ssm_comparison_dynamax_psychometric_k{K}",
                                ),
                            ],
                            align="center",
                        ),
                        mo.vstack(
                            [
                                mo.md("#### SSM"),
                                ssm_psych_fig_ssm,
                                save_plot(
                                    ssm_psych_fig_ssm,
                                    "ssm comparison ssm psychometric",
                                    stem=f"ssm_comparison_ssm_psychometric_k{K}",
                                ),
                            ],
                            align="center",
                        ),
                    ],
                    justify="start",
                )
                if ssm_psych_fig_custom is not None and ssm_psych_fig_ssm is not None
                else mo.md("Psychometric comparison unavailable because one of the trial-level prediction tables could not be built.")
            ),
            mo.md("### Coefficient differences"),
            (
                mo.vstack(
                    [
                        ssm_coef_fig,
                        save_plot(
                            ssm_coef_fig,
                            "ssm coefficient differences",
                            stem=f"ssm_coefficient_differences_k{K}",
                        ),
                    ],
                    align="center",
                )
                if ssm_coef_fig is not None
                else mo.md("No coefficient comparison available.")
            ),
        ],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
