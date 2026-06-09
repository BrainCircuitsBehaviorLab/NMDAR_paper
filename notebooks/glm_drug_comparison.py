import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Drug versus saline fits
    """)
    return


@app.cell
def _():
    import importlib
    from pathlib import Path
    import sys
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from matplotlib.lines import Line2D

    _local_glmhmmt_src = Path(__file__).resolve().parents[2] / "glmhmmt" / "src"
    if _local_glmhmmt_src.exists() and str(_local_glmhmmt_src) not in sys.path:
        sys.path.insert(0, str(_local_glmhmmt_src))

    from glmhmmt.notebook_support.analysis_common import (
        load_fit_bundle,
        load_model_config,
        model_aliases_for_kind,
    )
    from scipy.stats import ttest_rel

    from glmhmmt.plots.common import custom_boxplot, significance_label
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from plot_saver import make_plot_saver
    from src.process import common as process_common
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.plots.common import plot_mean_over_data
    from src.process.common import add_choice_lag_summary_regressor, attach_signed_delay_columns
    from src.utils import fig_size

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    paths = get_runtime_paths()
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    sns.set_style("ticks")
    sns.set_context("notebook")

    CONDITION_PALETTE = {
        "saline": "tab:gray",
        "rest": "tab:gray",
        "drug": "tab:pink",
        "nan": "#666666",
        "all": "#333333",
    }
    TASK_OPTIONS = ["2AFC_DRUG", "2ADC_DRUG", "MCDR"]
    DEFAULT_TASK_NAME = "2ADC_DRUG"
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    return (
        CONDITION_PALETTE,
        DEFAULT_TASK_NAME,
        Line2D,
        TASK_OPTIONS,
        add_choice_lag_summary_regressor,
        attach_signed_delay_columns,
        build_emission_weights_df,
        build_trial_df,
        build_views,
        custom_boxplot,
        fig_size,
        get_adapter,
        load_fit_bundle,
        load_model_config,
        make_plot_saver,
        mo,
        model_aliases_for_kind,
        np,
        paths,
        pd,
        pl,
        plot_mean_over_data,
        plt,
        process_mcdr,
        process_two_adc,
        process_two_afc,
        significance_label,
        sns,
        ttest_rel,
    )


@app.cell
def _(DEFAULT_TASK_NAME, TASK_OPTIONS, mo):
    ui_task = mo.ui.dropdown(
        options=TASK_OPTIONS,
        value=DEFAULT_TASK_NAME,
        label="Task",
    )
    return (ui_task,)


@app.cell
def _(ui_task):
    TASK_NAME = ui_task.value
    return (TASK_NAME,)


@app.cell
def _(TASK_NAME, get_adapter, pl):
    adapter = get_adapter(TASK_NAME)
    plots = adapter.get_plots()
    df_all = adapter.subject_filter(adapter.read_dataset())
    condition_counts = (
        df_all.group_by("condition")
        .agg(pl.len().alias("n_trials"))
        .sort("condition", nulls_last=True)
    )
    return adapter, condition_counts, df_all, plots


@app.cell
def _(TASK_NAME, mo, model_aliases_for_kind, paths):
    aliases_by_kind = {
        _kind: model_aliases_for_kind(
            task_name=TASK_NAME,
            model_kind=_kind,
            local_root=paths.RESULTS / "fits" / TASK_NAME / _kind,
        )
        for _kind in ["glm", "glmhmm", "glmhmmt"]
    }
    _default_kind = "glm" if aliases_by_kind["glm"] else next(
        (_kind for _kind, _aliases in aliases_by_kind.items() if _aliases),
        "glm",
    )
    ui_model_kind = mo.ui.dropdown(
        options=["glm", "glmhmm", "glmhmmt"],
        value=_default_kind,
        label="Model kind",
    )
    return aliases_by_kind, ui_model_kind


@app.cell
def _(aliases_by_kind, mo, ui_model_kind):
    _kind = ui_model_kind.value
    fit_aliases = aliases_by_kind[_kind]

    def _find_alias(needle: str):
        return next((alias for alias in fit_aliases if needle in alias.lower()), "")

    _fallback_alias = fit_aliases[0] if fit_aliases else ""
    _default_rest_alias = _find_alias("rest") or _find_alias("saline") or _fallback_alias
    _default_drug_alias = _find_alias("drug") or _fallback_alias

    ui_rest_alias = mo.ui.dropdown(
        options=fit_aliases or [""],
        value=_default_rest_alias,
        label="Saline fit",
    )
    ui_drug_alias = mo.ui.dropdown(
        options=fit_aliases or [""],
        value=_default_drug_alias,
        label="Drug fit",
    )
    ui_k = mo.ui.dropdown(
        options=[1] if _kind == "glm" else [2, 3, 4],
        value=1 if _kind == "glm" else 2,
        label="K",
    )
    return fit_aliases, ui_drug_alias, ui_k, ui_rest_alias


@app.cell
def _(
    condition_counts,
    fit_aliases,
    mo,
    ui_drug_alias,
    ui_k,
    ui_model_kind,
    ui_rest_alias,
    ui_task,
):
    mo.vstack(
        [
            mo.hstack([ui_task, ui_model_kind, ui_k, ui_rest_alias, ui_drug_alias]),
            mo.md(f"Available aliases: `{len(fit_aliases)}`"),
            condition_counts,
        ]
    )
    return


@app.cell
def _(
    TASK_NAME,
    adapter,
    load_model_config,
    mo,
    paths,
    ui_drug_alias,
    ui_model_kind,
    ui_rest_alias,
):
    _root = paths.RESULTS / "fits" / TASK_NAME / ui_model_kind.value

    def _saved_condition(alias: str, fallback: str) -> str:
        cfg = load_model_config(
            task_name=TASK_NAME,
            model_kind=ui_model_kind.value,
            alias=alias,
            local_root=_root,
        )
        return str(cfg.get("condition_filter", fallback) or fallback).lower()

    _condition_options = adapter.condition_filter_options() or ["all"]

    ui_rest_condition = mo.ui.dropdown(
        options=_condition_options,
        value=_saved_condition(ui_rest_alias.value, "rest" if "rest" in _condition_options else _condition_options[0]),
        label="Condition A",
    )
    ui_drug_condition = mo.ui.dropdown(
        options=_condition_options,
        value=_saved_condition(ui_drug_alias.value, "drug" if "drug" in _condition_options else _condition_options[0]),
        label="Condition B",
    )
    return ui_drug_condition, ui_rest_condition


@app.cell
def _(mo, ui_drug_condition, ui_rest_condition):
    mo.hstack([ui_rest_condition, ui_drug_condition])
    return


@app.cell
def _(
    TASK_NAME,
    build_views,
    get_adapter,
    load_fit_bundle,
    mo,
    paths,
    ui_drug_alias,
    ui_k,
    ui_model_kind,
    ui_rest_alias,
):
    mo.stop(not ui_rest_alias.value or not ui_drug_alias.value, mo.md("No saved fits found for this model kind."))
    model_kind = ui_model_kind.value
    K = int(ui_k.value)

    # load_fit_bundle expects an explicit subject list; infer it from filenames.
    _fit_root = paths.RESULTS / "fits" / TASK_NAME / model_kind
    _suffix = "glm_arrays.npz" if model_kind == "glm" else f"K{K}_{model_kind}_arrays.npz"
    _subjects = sorted(
        {
            path.name.split("_")[0]
            for alias in [ui_rest_alias.value, ui_drug_alias.value]
            for path in (_fit_root / alias).glob(f"*_{_suffix}")
        }
    )
    mo.stop(not _subjects, mo.md("No array files found for the selected aliases."))

    rest_adapter, rest_arrays, rest_names, rest_views = load_fit_bundle(
        task_name=TASK_NAME,
        model_kind=model_kind,
        alias=ui_rest_alias.value,
        k=K,
        subjects=_subjects,
        get_adapter=get_adapter,
        build_views=build_views,
        local_root=_fit_root,
    )
    drug_adapter, drug_arrays, drug_names, drug_views = load_fit_bundle(
        task_name=TASK_NAME,
        model_kind=model_kind,
        alias=ui_drug_alias.value,
        k=K,
        subjects=_subjects,
        get_adapter=get_adapter,
        build_views=build_views,
        local_root=_fit_root,
    )
    common_subjects = sorted(set(rest_views) & set(drug_views))
    mo.stop(not common_subjects, mo.md("The selected fits have no subjects in common."))
    return common_subjects, drug_views, model_kind, rest_views


@app.cell
def _(
    adapter,
    build_trial_df,
    common_subjects,
    df_all,
    drug_views,
    mo,
    pl,
    rest_views,
    ui_drug_alias,
    ui_drug_condition,
    ui_rest_alias,
    ui_rest_condition,
):
    def _fit_label(condition: str) -> str:
        value = str(condition or "all").lower()
        return "saline" if value in {"rest", "saline"} else value

    def _trial_df_for_fit(views: dict, *, condition_filter: str, label: str, alias: str):
        _df = adapter.filter_condition_df(df_all, condition_filter)
        _frames = []
        for _subject in common_subjects:
            _view = views.get(_subject)
            if _view is None:
                continue
            _df_sub = _df.filter(pl.col("subject") == _subject).sort(adapter.sort_col)
            if _df_sub.height != _view.T:
                continue
            _frames.append(build_trial_df(_view, adapter, _df_sub, adapter.behavioral_cols))
        if not _frames:
            return pl.DataFrame()
        return pl.concat(_frames, how="diagonal_relaxed").with_columns(
            [
                pl.lit(label).alias("fit_condition"),
                pl.lit(alias).alias("fit_alias"),
            ]
        )

    rest_trial_df = _trial_df_for_fit(
        rest_views,
        condition_filter=ui_rest_condition.value,
        label=_fit_label(ui_rest_condition.value),
        alias=ui_rest_alias.value,
    )
    drug_trial_df = _trial_df_for_fit(
        drug_views,
        condition_filter=ui_drug_condition.value,
        label=_fit_label(ui_drug_condition.value),
        alias=ui_drug_alias.value,
    )
    comparison_trial_df = (
        pl.concat([rest_trial_df, drug_trial_df], how="diagonal_relaxed")
        if rest_trial_df.height and drug_trial_df.height
        else pl.DataFrame()
    )
    mo.stop(comparison_trial_df.is_empty(), mo.md("No aligned trials for the selected condition filters."))
    return drug_trial_df, rest_trial_df


@app.cell
def _(
    TASK_NAME,
    adapter,
    add_choice_lag_summary_regressor,
    drug_trial_df,
    drug_views,
    pl,
    process_mcdr,
    process_two_adc,
    process_two_afc,
    rest_trial_df,
    rest_views,
):
    def _choice_lag_cols_for(views: dict, trial_df):
        _cols = []
        for _view in views.values():
            for _feat in list(getattr(_view, "feat_names", []) or []):
                _feat = str(_feat)
                if _feat.startswith("choice_lag_") and _feat not in _cols:
                    _cols.append(_feat)
        if not _cols:
            _cols = adapter.choice_lag_cols(trial_df)
        return _cols

    def _prepare_plot_df(trial_df, views):
        _processor = (
            process_mcdr
            if TASK_NAME == "MCDR"
            else
            process_two_adc
            if TASK_NAME in {"2ADC_DRUG", "2AFC_delay_DRUG", "2ADC", "2AFC_delay"}
            else process_two_afc
        )
        _plot_df = _processor.prepare_predictions_df(trial_df)
        _plot_df = add_choice_lag_summary_regressor(
            _plot_df,
            choice_lag_cols=_choice_lag_cols_for(views, trial_df),
        )
        if isinstance(_plot_df, pl.DataFrame) and "Choice" not in _plot_df.columns and "response" in _plot_df.columns:
            _plot_df = _plot_df.with_columns(pl.col("response").alias("Choice"))
        return _plot_df

    rest_plot_df = _prepare_plot_df(rest_trial_df, rest_views)
    drug_plot_df = _prepare_plot_df(drug_trial_df, drug_views)
    return drug_plot_df, rest_plot_df


@app.cell
def _(drug_plot_df, mo, plots, rest_plot_df):
    _columns = set(rest_plot_df.columns) & set(drug_plot_df.columns)
    _preferred = [
        "choice_lag_one_hot_sum",
        "at_choice",
        "at_choice_param",
        "stim_vals",
        "stim_param",
        "prev_choice",
        "wsls",
    ]
    regressor_options = [col for col in _preferred if col in _columns]
    if not regressor_options:
        regressor_options = sorted(
            col
            for col in _columns
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
    CONDITION_PALETTE,
    Line2D,
    TASK_NAME,
    adapter,
    choice_history_regressor,
    drug_plot_df,
    drug_views,
    plots,
    plt,
    rest_plot_df,
    rest_views,
    ui_drug_condition,
    ui_rest_condition,
):
    def _artist_snapshot(ax):
        return {
            "lines": set(ax.lines),
            "collections": set(ax.collections),
            "patches": set(ax.patches),
        }

    def _style_added_artists(ax, before: dict, *, color: str):
        for _line in [artist for artist in ax.lines if artist not in before["lines"]]:
            _line.set_color(color)
            _line.set_markerfacecolor(color)
            _line.set_markeredgecolor(color)
        for _collection in [artist for artist in ax.collections if artist not in before["collections"]]:
            try:
                _collection.set_edgecolor(color)
                _collection.set_facecolor(color)
            except Exception:
                pass
        for _patch in [artist for artist in ax.patches if artist not in before["patches"]]:
            _patch.set_edgecolor(color)

    def _overlay(call, ax, *, label: str, color: str):
        _before = _artist_snapshot(ax)
        _result = call(ax)
        if _result is not None:
            _style_added_artists(ax, _before, color=color)
        return _result

    fig_overlay, axd = plt.subplot_mosaic(
        [
            ["accuracy", "repeat_evidence"],
            ["binned_accuracy", "right_regressor"],
            ["repeat_bias", "repeat_bias"],
        ],
        figsize=(7.0, 8.3),
        layout="constrained",
    )
    _payloads = [
        (str(ui_rest_condition.value), rest_plot_df, rest_views),
        (str(ui_drug_condition.value), drug_plot_df, drug_views),
    ]
    _is_delay_task = TASK_NAME in {"2ADC_DRUG", "2AFC_delay_DRUG", "2ADC", "2AFC_delay"}
    _accuracy_x_axis = "raw_delay" if _is_delay_task else "ILD"
    for _label, _plot_df, _views in _payloads:
        _color = CONDITION_PALETTE.get(_label, "#333333")
        _overlay(
            lambda _ax, _df=_plot_df, _vs=_views: plots.plot_accuracy_by_total_evidence(
                _df,
                adapter=adapter,
                views=_vs,
                ax=_ax,
                legend=False,
            ),
            axd["accuracy"],
            label=_label,
            color=_color,
        )
        _overlay(
            lambda _ax, _df=_plot_df, _vs=_views: plots.plot_repeat_by_repeat_evidence(
                _df,
                views=_vs,
                ax=_ax,
                legend=False,
            ),
            axd["repeat_evidence"],
            label=_label,
            color=_color,
        )
        _overlay(
            lambda _ax, _df=_plot_df, _vs=_views: plots.plot_binned_accuracy_figure(
                _df,
                regressor_col=choice_history_regressor,
                x_axis=_accuracy_x_axis,
                adapter=adapter,
                views=_vs,
                axes=[_ax],
                max_panels=1,
                legend=False,
            ),
            axd["binned_accuracy"],
            label=_label,
            color=_color,
        )
        _overlay(
            lambda _ax, _df=_plot_df: plots.plot_right_by_regressor(
                _df,
                regressor_col=choice_history_regressor,
                ax=_ax,
                legend=False,
            ),
            axd["right_regressor"],
            label=_label,
            color=_color,
        )
        _overlay(
            lambda _ax, _df=_plot_df: plots.plot_rb(
                _df,
                ax=_ax,
                title=None,
            ),
            axd["repeat_bias"],
            label=_label,
            color=_color,
        )

    axd["accuracy"].set_title("Accuracy by fitted evidence")
    axd["repeat_evidence"].set_title("Repeat by fitted evidence")
    axd["binned_accuracy"].set_title("Psychometric by choice lag")
    axd["right_regressor"].set_title("Right choice by choice lag")
    axd["repeat_bias"].set_title("Repeat bias by delay" if _is_delay_task else "Repeat bias by stimulus strength")
    for _key, _axis in axd.items():
        _axis.set_box_aspect(1)
    fig_overlay.legend(
        handles=[
            Line2D(
                [0], [0],
                color=CONDITION_PALETTE.get(str(ui_rest_condition.value), "#333333"),
                lw=2,
                marker="o",
                label=str(ui_rest_condition.value),
            ),
            Line2D(
                [0], [0],
                color=CONDITION_PALETTE.get(str(ui_drug_condition.value), "#333333"),
                lw=2,
                marker="o",
                label=str(ui_drug_condition.value),
            ),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    return (fig_overlay,)


@app.cell
def _(
    CONDITION_PALETTE,
    Line2D,
    TASK_NAME,
    attach_signed_delay_columns,
    drug_plot_df,
    fig_overlay,
    fig_size,
    make_plot_saver,
    mo,
    model_kind,
    paths,
    pd,
    plot_mean_over_data,
    plt,
    rest_plot_df,
    sns,
    ui_drug_alias,
    ui_drug_condition,
    ui_rest_alias,
    ui_rest_condition,
):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=TASK_NAME,
        model_id=f"drug_comparison/{model_kind}/{ui_rest_alias.value}_vs_{ui_drug_alias.value}",
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

    def prepare_curve_df(frame, condition):
        curve_df = add_p_right(to_pandas(frame))
        curve_df["fit_condition"] = str(condition)
        if "p_right" not in curve_df.columns:
            return curve_df
        if "raw_delay" in curve_df.columns and not any(col in curve_df.columns for col in ["delay_raw", "delays", "delay"]):
            curve_df["delay_raw"] = curve_df["raw_delay"]
        if "stim" not in curve_df.columns:
            for candidate in ["stim_vals", "stim_param"]:
                if candidate in curve_df.columns:
                    curve_df["stim"] = pd.to_numeric(curve_df[candidate], errors="coerce")
                    break
            if "stim" not in curve_df.columns and "stimulus" in curve_df.columns:
                stimulus = pd.to_numeric(curve_df["stimulus"], errors="coerce")
                if stimulus.min(skipna=True) < 0:
                    curve_df["stim"] = stimulus
        return curve_df

    rest_curve_df = prepare_curve_df(rest_plot_df, ui_rest_condition.value)
    drug_curve_df = prepare_curve_df(drug_plot_df, ui_drug_condition.value)
    curve_df = pd.concat([rest_curve_df, drug_curve_df], ignore_index=True)
    curve_df = curve_df.dropna(subset=["p_right"]) if "p_right" in curve_df.columns else pd.DataFrame()
    is_delay_task = TASK_NAME in {"2ADC_DRUG", "2AFC_delay_DRUG", "2ADC", "2AFC_delay"}
    fig_psychometric_comparison, ax_psychometric = plt.subplots(figsize=fig_size(2, 1))
    psychometric_available = False
    if not curve_df.empty and is_delay_task:
        signed_df = attach_signed_delay_columns(curve_df)
        signed_df["signed_delay_plot"] = signed_df["_signed_delay_cat"].astype(str)
        signed_df = signed_df[signed_df["signed_delay_plot"].isin(signed_delay_order)].copy()
        if not signed_df.empty:
            for condition in [str(ui_rest_condition.value), str(ui_drug_condition.value)]:
                condition_df = signed_df[signed_df["fit_condition"] == condition]
                if condition_df.empty:
                    continue
                plot_mean_over_data(
                    condition_df,
                    x_col="signed_delay_plot",
                    x_order=signed_delay_order,
                    x_tick_labels=signed_delay_tick_labels,
                    y_col="p_right",
                    xlabel="Signed delay (s)",
                    ylabel=r"$p(\mathrm{right})$",
                    title="",
                    baseline=0.5,
                    baseline_area=False,
                    color=CONDITION_PALETTE.get(condition, "#333333"),
                    figsize=fig_size(2, 1),
                    ax=ax_psychometric,
                )
                psychometric_available = True
    if not psychometric_available and not curve_df.empty and "ILD" in curve_df.columns:
        for condition in [str(ui_rest_condition.value), str(ui_drug_condition.value)]:
            condition_df = curve_df[curve_df["fit_condition"] == condition]
            if condition_df.empty:
                continue
            plot_mean_over_data(
                condition_df,
                x_col="ILD",
                y_col="p_right",
                xlabel="ILD (dB)",
                ylabel=r"$p(\mathrm{right})$",
                title="",
                baseline=0.5,
                baseline_area=False,
                color=CONDITION_PALETTE.get(condition, "#333333"),
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
                Line2D([0], [0], color=CONDITION_PALETTE.get(str(ui_rest_condition.value), "#333333"), lw=2, marker="o", label=str(ui_rest_condition.value)),
                Line2D([0], [0], color=CONDITION_PALETTE.get(str(ui_drug_condition.value), "#333333"), lw=2, marker="o", label=str(ui_drug_condition.value)),
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
            save_plot(fig_overlay, "drug saline overlay", stem="drug_saline_overlay"),
            fig_psychometric_comparison,
            save_plot(fig_psychometric_comparison, "overall psychometric saline drug", stem="overall_psychometric_saline_drug"),
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
    ### State comparison
    """)
    return


@app.cell
def _(
    CONDITION_PALETTE,
    Line2D,
    add_p_right,
    attach_signed_delay_columns,
    custom_boxplot,
    fig_size,
    np,
    pd,
    pl,
    plots,
    plt,
    signed_delay_order,
    signed_delay_tick_labels,
    significance_label,
    sns,
    to_pandas,
    ttest_rel,
):
    state_labels = {0: "Engaged", 1: "Disengaged"}
    state_ranks = [0, 1]


    def state_name(rank):
        return state_labels.get(int(rank), f"State {int(rank)}")

    def ordered_conditions(data):
        available = [str(condition) for condition in pd.unique(data["fit_condition"]).tolist()]
        preferred = ["saline", "rest", "drug"]
        ordered = [condition for condition in preferred if condition in available]
        ordered.extend(condition for condition in available if condition not in ordered)
        return ordered[:2]

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

    def build_state_comparison_df(rest_trial_df, drug_trial_df, comparison_weights_df, rank_feature):
        rank_map = build_state_rank_map(comparison_weights_df, rank_feature)
        rest_trials = add_ranked_state_columns(rest_trial_df, rank_map)
        drug_trials = add_ranked_state_columns(drug_trial_df, rank_map)
        comparison = pd.concat([rest_trials, drug_trials], ignore_index=True)
        comparison = comparison[comparison["state_rank"].isin(state_ranks)].copy()
        comparison["fit_condition"] = comparison["fit_condition"].astype(str)
        return comparison

    def condition_handles(conditions):
        return [
            plt.Line2D([0], [0], color=CONDITION_PALETTE.get(cond, "#333333"), lw=3, label=cond)
            for cond in conditions
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

    def format_state_summary_panel(fig, ax, legend_ax, conditions):
        if ax.legend_ is not None:
            ax.legend_.remove()
        fig.legends.clear()
        legend_ax.legend(
            handles=condition_handles(conditions),
            frameon=False,
            loc="lower center",
            ncol=len(conditions),
        )
        sns.despine(ax=ax)
        return fig

    def plot_condition_mean_sem(ax, data, conditions, *, x_col, y_col, x_transform=None):
        transform = x_transform or (lambda values: values)
        for condition in conditions:
            condition_df = data[data["fit_condition"] == condition]
            summary = (
                condition_df.groupby(x_col, as_index=False, observed=True)
                .agg(mean=(y_col, "mean"), sem=(y_col, lambda values: values.sem()))
                .sort_values(x_col)
            )
            x = transform(summary[x_col].to_numpy(dtype=float))
            mean = summary["mean"].to_numpy(dtype=float)
            sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)
            color = CONDITION_PALETTE.get(condition, "#333333")
            ax.plot(x, mean, color=color, lw=2.0, label=condition)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)

    def paired_state_boxplot(summary_df, conditions, *, value_col, ylabel, chance=None, ylim=None, plot_state_ranks=None):
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
        return format_state_summary_panel(fig, ax, legend_ax, conditions)

    def state_accuracy_df(state_comparison_df):
        return (
            state_comparison_df.dropna(subset=["correct_bool"])
            .groupby(["subject", "fit_condition", "state_rank"], as_index=False, observed=True)
            .agg(accuracy=("correct_bool", "mean"), n_trials=("correct_bool", "size"))
        )

    def engaged_occupancy_df(state_comparison_df):
        return (
            pl.from_pandas(state_comparison_df[["subject", "fit_condition", "state_rank"]])
            .with_columns(pl.col("state_rank").cast(pl.Int64, strict=False))
            .group_by(["subject", "fit_condition"])
            .agg(
                [
                    pl.len().alias("n_subject_trials"),
                    (pl.col("state_rank") == 0).sum().alias("n_state_trials"),
                ]
            )
            .with_columns(
                [
                    pl.lit(0).alias("state_rank"),
                    (pl.col("n_state_trials") / pl.col("n_subject_trials")).alias("occupancy"),
                ]
            )
            .to_pandas()
        )

    def session_switches(state_comparison_df):
        records = []
        for (subject, condition, session), group in state_comparison_df.sort_values(
            ["subject", "fit_condition", "session", "trial_idx"]
        ).groupby(["subject", "fit_condition", "session"], observed=True):
            states = pd.to_numeric(group["state_rank"], errors="coerce").to_numpy(dtype=float)
            states = states[np.isfinite(states)]
            records.append(
                {
                    "subject": str(subject),
                    "fit_condition": str(condition),
                    "session": str(session),
                    "n_switches": int(np.sum(states[1:] != states[:-1])),
                }
            )
        columns = ["subject", "fit_condition", "session", "n_switches"]
        return pd.DataFrame(records, columns=columns)

    def switch_density_figure(switch_df, conditions):
        fig, ax, legend_ax = state_panel_figure()
        max_switches = float(switch_df["n_switches"].max())
        for condition in conditions:
            values = switch_df[switch_df["fit_condition"] == condition]["n_switches"].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            color = CONDITION_PALETTE.get(condition, "#333333")
            if values.size >= 2 and np.nanstd(values) > 0:
                sns.kdeplot(
                    x=values,
                    ax=ax,
                    color=color,
                    linewidth=2.0,
                    bw_adjust=0.8,
                    clip=(0, None),
                    label=condition,
                )
            else:
                ax.axvline(
                    float(values[0]),
                    color=color,
                    linewidth=2.0,
                    label=condition,
                )
        ax.set_xlim(left=-0.25, right=max(1.0, max_switches + 0.25))
        ax.set_xlabel("State changes per session")
        ax.set_ylabel("Density")
        return format_state_summary_panel(fig, ax, legend_ax, conditions)

    def normalize_delay_task_columns(data):
        prepared = data.copy()
        if "delay_raw" not in prepared.columns and "raw_delay" in prepared.columns:
            prepared["delay_raw"] = prepared["raw_delay"]
        if "stim" not in prepared.columns:
            stim_col = next(col for col in ["stim_vals", "stim_param", "stimulus"] if col in prepared.columns)
            prepared["stim"] = pd.to_numeric(prepared[stim_col], errors="coerce")
        return prepared

    def add_model_p_right(data):
        prepared = data.copy()
        model_col = next(
            (col for col in ["p_model_right", "p_pred", "pR"] if col in prepared.columns),
            None,
        )
        if model_col is not None:
            prepared["p_model_right"] = pd.to_numeric(prepared[model_col], errors="coerce")
        return prepared

    def categorical_source_df(state_comparison_df, is_delay_task):
        prepared = add_p_right(state_comparison_df)
        prepared = add_model_p_right(prepared)
        if is_delay_task:
            prepared = normalize_delay_task_columns(prepared)
        return prepared.dropna(subset=["p_right"]).copy()

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
        fmt = f"{marker}{linestyle}" if marker else linestyle
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

    def categorical_by_state_figure(state_comparison_df, conditions, is_delay_task):
        source = categorical_source_df(state_comparison_df, is_delay_task)
        has_model = "p_model_right" in source.columns and source["p_model_right"].notna().any()
        fig, axes = plt.subplots(
            1,
            len(state_ranks),
            figsize=(fig_size(2, 1)[0] * len(state_ranks), fig_size(2, 1)[1]),
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()
        if is_delay_task:
            plot_source = attach_signed_delay_columns(source)
            plot_source["signed_delay_plot"] = plot_source["_signed_delay_cat"].astype(str)
            plot_source = plot_source[plot_source["signed_delay_plot"].isin(signed_delay_order)].copy()
            for ax, rank in zip(axes, state_ranks, strict=False):
                state_df = plot_source[plot_source["state_rank"] == rank]
                for condition in conditions:
                    condition_df = state_df[state_df["fit_condition"] == condition]
                    plot_state_psychometric_curve(
                        ax,
                        condition_df,
                        x_col="signed_delay_plot",
                        x_order=signed_delay_order,
                        x_tick_labels=signed_delay_tick_labels,
                        y_col="p_right",
                        color=CONDITION_PALETTE.get(condition, "#333333"),
                    )
                    plot_state_psychometric_curve(
                        ax,
                        condition_df,
                        x_col="signed_delay_plot",
                        x_order=signed_delay_order,
                        x_tick_labels=signed_delay_tick_labels,
                        y_col="p_model_right",
                        color=CONDITION_PALETTE.get(condition, "#333333"),
                        linestyle="--",
                        marker="",
                        linewidth=1.6,
                        alpha=0.8,
                    )
                format_psychometric_state_axis(ax, xlabel="Signed delay (s)", title=state_name(rank))
        else:
            for ax, rank in zip(axes, state_ranks, strict=False):
                state_df = source[source["state_rank"] == rank]
                for condition in conditions:
                    condition_df = state_df[state_df["fit_condition"] == condition]
                    plot_state_psychometric_curve(
                        ax,
                        condition_df,
                        x_col="ILD",
                        y_col="p_right",
                        color=CONDITION_PALETTE.get(condition, "#333333"),
                    )
                    plot_state_psychometric_curve(
                        ax,
                        condition_df,
                        x_col="ILD",
                        y_col="p_model_right",
                        color=CONDITION_PALETTE.get(condition, "#333333"),
                        linestyle="--",
                        marker="",
                        linewidth=1.6,
                        alpha=0.8,
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
                legend_handles.append(Line2D([0], [0], color=color, lw=1.8, linestyle="--", label=f"{condition} model"))
        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
        )
        fig.subplots_adjust(bottom=0.26, wspace=0.25)
        return fig

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
            state_assignment_mode="map",
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

    def mean_session_trace_df(state_comparison_df, n_bins=20):
        records = []
        sorted_df = state_comparison_df.dropna(subset=["p_engaged"]).sort_values(
            ["subject", "fit_condition", "session", "trial_idx"]
        )
        for (subject, condition, session), group in sorted_df.groupby(["subject", "fit_condition", "session"], observed=True):
            position = np.linspace(0.0, 1.0, len(group))
            position_bin = np.minimum((position * n_bins).astype(int), n_bins - 1)
            session_df = pd.DataFrame(
                {
                    "subject": str(subject),
                    "fit_condition": str(condition),
                    "session": str(session),
                    "position_bin": position_bin,
                    "p_engaged": group["p_engaged"].to_numpy(dtype=float),
                }
            )
            records.extend(
                session_df.groupby(["subject", "fit_condition", "session", "position_bin"], as_index=False, observed=True)["p_engaged"]
                .mean()
                .to_dict("records")
            )
        columns = ["subject", "fit_condition", "session", "position_bin", "p_engaged"]
        return pd.DataFrame(records, columns=columns)

    def mean_session_trace_figure(trace_df, conditions, n_bins=20):
        fig, ax = plt.subplots(figsize=fig_size(2, 1))
        plot_condition_mean_sem(
            ax,
            trace_df,
            conditions,
            x_col="position_bin",
            y_col="p_engaged",
            x_transform=lambda values: (values + 0.5) / n_bins,
        )
        ax.set_xlabel("Normalized session time")
        ax.set_ylabel("P(engaged)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=len(conditions))
        fig.subplots_adjust(bottom=0.34)
        sns.despine(ax=ax)
        return fig

    def change_triggered_trace_df(state_comparison_df, window=5):
        records = []
        sorted_df = state_comparison_df.dropna(subset=["p_engaged"]).sort_values(
            ["subject", "fit_condition", "session", "trial_idx"]
        )
        for (subject, condition, session), group in sorted_df.groupby(["subject", "fit_condition", "session"], observed=True):
            states = pd.to_numeric(group["state_rank"], errors="coerce").to_numpy(dtype=float)
            p_engaged = group["p_engaged"].to_numpy(dtype=float)
            for idx in np.flatnonzero(states[1:] != states[:-1]) + 1:
                lo = idx - window
                hi = idx + window + 1
                direction = "Into engaged" if states[idx] == 0 else "Out of engaged" if states[idx - 1] == 0 else None
                if lo < 0 or hi > len(states) or direction is None:
                    continue
                for rel_trial, value in zip(range(-window, window + 1), p_engaged[lo:hi], strict=False):
                    records.append(
                        {
                            "subject": str(subject),
                            "fit_condition": str(condition),
                            "session": str(session),
                            "direction": direction,
                            "rel_trial": int(rel_trial),
                            "p_engaged": float(value),
                        }
                    )
        columns = ["subject", "fit_condition", "session", "direction", "rel_trial", "p_engaged"]
        return pd.DataFrame(records, columns=columns)

    def change_triggered_trace_figure(trace_df, conditions):
        fig, axes = plt.subplots(1, 2, figsize=fig_size(1, 3), sharey=True, squeeze=False)
        for ax, direction in zip(axes.ravel(), ["Into engaged", "Out of engaged"], strict=False):
            direction_df = trace_df[trace_df["direction"] == direction]
            plot_condition_mean_sem(ax, direction_df, conditions, x_col="rel_trial", y_col="p_engaged")
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(direction)
            ax.set_xlabel("Trials from change")
            ax.set_ylim(0, 1)
            sns.despine(ax=ax)
        axes.ravel()[0].set_ylabel("P(engaged)")
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color=CONDITION_PALETTE.get(condition, "#333333"), lw=2.2, label=condition)
                for condition in conditions
            ],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(conditions),
        )
        fig.subplots_adjust(bottom=0.23, wspace=0.25)
        return fig

    return (
        adapter_by_state_psychometric_figure,
        build_state_comparison_df,
        change_triggered_trace_df,
        change_triggered_trace_figure,
        engaged_occupancy_df,
        mean_session_trace_df,
        mean_session_trace_figure,
        ordered_conditions,
        paired_state_boxplot,
        session_switches,
        state_accuracy_df,
        switch_density_figure,
    )


@app.cell
def _(
    adapter,
    adapter_by_state_psychometric_figure,
    build_state_comparison_df,
    change_triggered_trace_df,
    change_triggered_trace_figure,
    comparison_weights_df,
    drug_plot_df,
    drug_trial_df,
    drug_views,
    engaged_occupancy_df,
    mean_session_trace_df,
    mean_session_trace_figure,
    mo,
    ordered_conditions,
    paired_state_boxplot,
    rest_plot_df,
    rest_trial_df,
    rest_views,
    save_plot,
    session_switches,
    state_accuracy_df,
    switch_density_figure,
    ui_state_rank_feature,
):
    state_comparison_df = build_state_comparison_df(
        rest_trial_df,
        drug_trial_df,
        comparison_weights_df,
        ui_state_rank_feature.value,
    )
    mo.stop(state_comparison_df.empty, mo.md("No state assignments available."))
    state_condition_order = ordered_conditions(state_comparison_df)
    mo.stop(len(state_condition_order) < 2, mo.md("Need both saline/rest and drug trials for state comparisons."))

    fig_state_accuracy_comparison = paired_state_boxplot(
        state_accuracy_df(state_comparison_df),
        state_condition_order,
        value_col="accuracy",
        ylabel="Accuracy",
        chance=1.0 / adapter.num_classes,
        ylim=(0, 1),
    )
    fig_state_occupancy_comparison = paired_state_boxplot(
        engaged_occupancy_df(state_comparison_df),
        state_condition_order,
        value_col="occupancy",
        ylabel="Engaged occupancy",
        ylim=(0, 1),
        plot_state_ranks=[0],
    )
    state_switch_sessions_df = session_switches(state_comparison_df)
    fig_switch_distribution = switch_density_figure(state_switch_sessions_df, state_condition_order)
    fig_state_categorical = adapter_by_state_psychometric_figure(
        {
            state_condition_order[0]: (rest_plot_df, rest_views),
            state_condition_order[1]: (drug_plot_df, drug_views),
        },
        state_condition_order,
    )
    mean_session_trace_data = mean_session_trace_df(state_comparison_df)
    fig_session_trace = mean_session_trace_figure(mean_session_trace_data, state_condition_order)
    change_triggered_trace_data = change_triggered_trace_df(state_comparison_df)
    fig_change_trace = change_triggered_trace_figure(change_triggered_trace_data, state_condition_order)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([fig_state_accuracy_comparison, save_plot(fig_state_accuracy_comparison, "state accuracy saline drug", stem="state_accuracy_saline_drug")], align="center"),
                    mo.vstack([fig_state_occupancy_comparison, save_plot(fig_state_occupancy_comparison, "state occupancy saline drug", stem="state_occupancy_saline_drug")], align="center"),
                    mo.vstack([fig_switch_distribution, save_plot(fig_switch_distribution, "state changes distribution saline drug", stem="state_changes_distribution_saline_drug")], align="center"),
                ],
                align="center",
            ),
            mo.vstack([fig_state_categorical, save_plot(fig_state_categorical, "psychometric by state saline drug", stem="psychometric_saline_drug")], align="center"),
            mo.hstack(
                [
                    mo.vstack([fig_session_trace, save_plot(fig_session_trace, "mean session engaged trace saline drug", stem="mean_session_engaged_trace_saline_drug")], align="center"),
                    mo.vstack([fig_change_trace, save_plot(fig_change_trace, "change-triggered engaged trace saline drug", stem="change_triggered_engaged_trace_saline_drug")], align="center"),
                ],
                align="center",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(
    build_emission_weights_df,
    drug_views,
    mo,
    pl,
    rest_views,
    ui_drug_condition,
    ui_rest_condition,
):
    rest_weights_df = build_emission_weights_df(rest_views).with_columns(pl.lit(str(ui_rest_condition.value)).alias("fit_condition"))
    drug_weights_df = build_emission_weights_df(drug_views).with_columns(pl.lit(str(ui_drug_condition.value)).alias("fit_condition"))
    comparison_weights_df = pl.concat([rest_weights_df, drug_weights_df], how="diagonal_relaxed")
    _features = (
        comparison_weights_df.select("feature").unique().sort("feature").to_series().to_list()
        if not comparison_weights_df.is_empty()
        else []
    )
    _preferred_rank_features = [
        "stim_param",
        "stim_vals",
        "stim",
        "evidence_param",
        "at_choice_param",
        "choice_lag_param",
    ]
    _default_rank_feature = next(
        (feature for feature in _preferred_rank_features if feature in _features),
        _features[0] if _features else "",
    )
    ui_state_rank_feature = mo.ui.dropdown(
        options=_features or [""],
        value=_default_rank_feature,
        label="State ranking coefficient",
    )
    return comparison_weights_df, ui_state_rank_feature


@app.cell
def _(mo, ui_state_rank_feature):
    mo.hstack([ui_state_rank_feature])
    return


@app.cell
def _(
    CONDITION_PALETTE,
    TASK_NAME,
    comparison_weights_df,
    custom_boxplot,
    fig_size,
    np,
    pd,
    pl,
    plt,
    significance_label,
    sns,
    ttest_rel,
    ui_drug_condition,
    ui_rest_condition,
    ui_state_rank_feature,
):
    def plot_emission_condition_boxplot(
        weights_df,
        *,
        feature_order: list[str] | None = None,
    ):
        _raw = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        if _raw.empty:
            return None
        _feature_labels = {
            "stim_param": r"$\mathrm{Stim}_{\mathrm{param}}$",
            "stim_x_delay_param": r"$\mathrm{Stim:delay}_{\mathrm{param}}$",
            "delay_param": r"$\mathrm{Delay}_{\mathrm{param}}$",
            "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
            "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
            "at_choice_param": r"$\mathrm{A}_t$",
            "at_choice": r"$\mathrm{A}_t$",
            "choice_lag_param": r"$\mathrm{A}$",
            "choice_param": r"$\mathrm{A}$",
        }
        _feature_labeler = lambda feature: _feature_labels.get(str(feature), str(feature))
        _raw = (
            _raw.groupby(
                ["subject", "fit_condition", "state_rank", "state_label", "feature"],
                as_index=False,
            )["weight"]
            .mean()
        )
        _rank_feature = str(ui_state_rank_feature.value or "")
        if _rank_feature in set(_raw["feature"]):
            _rank_source = (
                _raw[_raw["feature"] == _rank_feature]
                .groupby(["subject", "fit_condition", "state_rank", "state_label"], as_index=False)["weight"]
                .mean()
            )
            _rank_source = _rank_source.sort_values(
                ["subject", "fit_condition", "weight"],
                ascending=[True, True, False],
            )
            _rank_source["plot_state_rank"] = (
                _rank_source.groupby(["subject", "fit_condition"]).cumcount()
            )
            _rank_source["plot_state_label"] = _rank_source["plot_state_rank"].map(
                lambda rank: "Engaged" if int(rank) == 0 else "Disengaged" if int(rank) == 1 else f"Disengaged {int(rank)}"
            )
            _raw = _raw.merge(
                _rank_source[
                    [
                        "subject",
                        "fit_condition",
                        "state_rank",
                        "state_label",
                        "plot_state_rank",
                        "plot_state_label",
                    ]
                ],
                on=["subject", "fit_condition", "state_rank", "state_label"],
                how="left",
            )
        else:
            _raw["plot_state_rank"] = _raw["state_rank"]
            _raw["plot_state_label"] = _raw["state_label"]
        _raw = _raw.dropna(subset=["plot_state_rank"]).copy()
        _raw["plot_state_rank"] = _raw["plot_state_rank"].astype(int)
        _feature_order = [feature for feature in (feature_order or list(dict.fromkeys(_raw["feature"].astype(str)))) if feature in set(_raw["feature"])]
        if not _feature_order:
            return None
        _raw = _raw[_raw["feature"].isin(_feature_order)].copy()
        _state_rows = (
            _raw[["plot_state_rank", "plot_state_label"]]
            .drop_duplicates()
            .sort_values("plot_state_rank")
            .head(2)
        )
        _cond_order = [
            cond
            for cond in [str(ui_rest_condition.value), str(ui_drug_condition.value)]
            if cond in set(_raw["fit_condition"])
        ]
        _panel_width, _panel_height = fig_size(2, 1)
        _fig, _axes = plt.subplots(
            1,
            max(1, len(_state_rows)),
            figsize=(_panel_width * max(1, len(_state_rows)), _panel_height),
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        _axes = _axes.ravel()
        _width = 0.34
        _offsets = np.linspace(-_width / 2, _width / 2, max(1, len(_cond_order)))
        _selection_points = []
        _subject_line_axes = []

        def _add_line_selection_points(_subject, _feature, _state_label, _x1, _x2, _y1, _y2):
            for _x, _y in zip(np.linspace(_x1, _x2, 24), np.linspace(_y1, _y2, 24), strict=False):
                _selection_points.append(
                    {
                        "subject": str(_subject),
                        "feature": str(_feature),
                        "state": str(_state_label),
                        "x": float(_x),
                        "y": float(_y),
                    }
                )

        def _annotate_feature_significance(_ax, _state_df, _feature_order, _pos_by_pair):
            _y_top = None
            for _feature in _feature_order:
                _pivot = (
                    _state_df[_state_df["feature"] == _feature]
                    .pivot(index="subject", columns="fit_condition", values="weight")
                )
                if len(_cond_order) < 2 or not set(_cond_order[:2]).issubset(_pivot.columns):
                    continue
                _paired = _pivot[_cond_order[:2]].dropna()
                if len(_paired) < 2:
                    continue

                _pvalue = float(ttest_rel(_paired[_cond_order[0]], _paired[_cond_order[1]], nan_policy="omit").pvalue)
                _label = significance_label(_pvalue)
                _feature_values = _state_df[_state_df["feature"] == _feature]["weight"].dropna().to_numpy(dtype=float)
                if _feature_values.size == 0:
                    continue

                _x1 = _pos_by_pair.get((_feature, _cond_order[0]))
                _x2 = _pos_by_pair.get((_feature, _cond_order[1]))
                if _x1 is None or _x2 is None:
                    continue

                _yrange = float(np.nanmax(_state_df["weight"]) - np.nanmin(_state_df["weight"]))
                if not np.isfinite(_yrange) or _yrange <= 0:
                    _yrange = 1.0
                _y = float(np.nanmax(_feature_values)) + 0.08 * _yrange
                _h = 0.025 * _yrange
                _ax.plot([_x1, _x1, _x2, _x2], [_y, _y + _h, _y + _h, _y], color="black", lw=1.0)
                _ax.text((_x1 + _x2) / 2, _y + _h, _label, ha="center", va="bottom", color="black")
                _y_top = max(_y_top if _y_top is not None else _y + _h, _y + _h)
            if _y_top is not None:
                _lo, _hi = _ax.get_ylim()
                if _y_top >= _hi:
                    _ax.set_ylim(_lo, _y_top + 0.08 * max(_hi - _lo, 1.0))

        for _ax, (_, _state_row) in zip(_axes, _state_rows.iterrows(), strict=False):
            _subject_line_axes.append((str(_state_row["plot_state_label"]), _ax))
            _state_df = _raw[_raw["plot_state_rank"] == _state_row["plot_state_rank"]].copy()
            _values = []
            _positions = []
            _median_colors = []
            _pos_by_pair = {}
            for _fx, _feature in enumerate(_feature_order):
                for _cx, _cond in enumerate(_cond_order):
                    _pos = _fx + _offsets[_cx]
                    _vals = _state_df[
                        (_state_df["feature"] == _feature)
                        & (_state_df["fit_condition"] == _cond)
                    ]["weight"].dropna().to_numpy(dtype=float)
                    _values.append(_vals)
                    _positions.append(_pos)
                    _median_colors.append(CONDITION_PALETTE.get(_cond, "#333333"))
                    _pos_by_pair[(_feature, _cond)] = _pos
            custom_boxplot(
                _ax,
                _values,
                positions=_positions,
                widths=_width * 0.8,
                median_colors=_median_colors,
                showfliers=False,
                showcaps=False,
            )
            for _feature in _feature_order:
                _pivot = (
                    _state_df[_state_df["feature"] == _feature]
                    .pivot(index="subject", columns="fit_condition", values="weight")
                )
                if len(_cond_order) >= 2 and set(_cond_order[:2]).issubset(_pivot.columns):
                    for _subject, _row in _pivot.iterrows():
                        _ys = [_row[_cond_order[0]], _row[_cond_order[1]]]
                        if np.all(np.isfinite(_ys)):
                            _x1 = _pos_by_pair[(_feature, _cond_order[0])]
                            _x2 = _pos_by_pair[(_feature, _cond_order[1])]
                            _ax.plot(
                                [_x1, _x2],
                                _ys,
                                color="#B0B0B0",
                                alpha=0.25,
                                linewidth=1.0,
                                zorder=1,
                            )
                            _add_line_selection_points(
                                _subject,
                                _feature,
                                _state_row["plot_state_label"],
                                _x1,
                                _x2,
                                _ys[0],
                                _ys[1],
                            )
            _annotate_feature_significance(_ax, _state_df, _feature_order, _pos_by_pair)
            _ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
            _ax.set_title(str(_state_row["plot_state_label"]))
            _ax.set_xticks(range(len(_feature_order)))
            _ax.set_xticklabels([_feature_labeler(_feature) for _feature in _feature_order])
            _ax.set_xlabel("")
            sns.despine(ax=_ax)
        _axes[0].set_ylabel("Emission weight")
        _fig.legend(
            handles=[
                plt.Line2D([0], [0], color=CONDITION_PALETTE.get(str(ui_rest_condition.value), "#333333"), lw=3, label=str(ui_rest_condition.value)),
                plt.Line2D([0], [0], color=CONDITION_PALETTE.get(str(ui_drug_condition.value), "#333333"), lw=3, label=str(ui_drug_condition.value)),
            ],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=2,
        )
        _fig.tight_layout()
        _fig.subplots_adjust(bottom=0.30)
        _fig._subject_line_selection_points = _selection_points
        _fig._subject_line_axes = _subject_line_axes
        return _fig

    _features = (
        comparison_weights_df.select("feature").unique().sort("feature").to_series().to_list()
        if not comparison_weights_df.is_empty()
        else []
    )

    def _suffix_int(feature: str) -> int:
        suffix = feature.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 10**9

    if TASK_NAME == "MCDR":
        _stimulus_features = [
            feature
            for feature in [
                "stim_param",
                "SL", "SC", "SR",
                "SLxdelay", "SCxdelay", "SRxdelay",
                "SLxD", "SCxD", "SRxD",
                "stim1L", "stim1C", "stim1R",
                "stim2L", "stim2C", "stim2R",
                "stim3L", "stim3C", "stim3R",
                "stim4L", "stim4C", "stim4R",
            ]
            if feature in _features
        ]
        _choice_lag_features = [
            feature
            for feature in ["choice_param", "choice_lag_param"]
            if feature in _features
        ] + sorted(
            [
                feature
                for feature in _features
                if str(feature).startswith("choice_lag_")
            ],
            key=_suffix_int,
        )
        _combined_features = list(dict.fromkeys([*_stimulus_features, *_choice_lag_features]))
        fig_emission_coefficients = plot_emission_condition_boxplot(
            comparison_weights_df,
            feature_order=_combined_features,
        )
    else:
        _is_delay_task = TASK_NAME in {"2ADC_DRUG", "2AFC_delay_DRUG", "2ADC", "2AFC_delay"}
        _stimulus_base_features = (
            ["stim", "stim_vals", "stim_param", "delay", "delay_param", "stim_x_delay", "stim_x_delay_param"]
            if _is_delay_task
            else ["stim_vals", "stim_param", "stim"]
        )
        _stimulus_prefixes = (
            ("stim_", "delay_", "stim_x_delay_hot_")
            if _is_delay_task
            else ("stim_",)
        )

        _stimulus_features = [
            feature
            for feature in _stimulus_base_features
            if feature in _features
        ] + sorted(
            [
                feature
                for feature in _features
                if any(str(feature).startswith(prefix) for prefix in _stimulus_prefixes)
                and feature not in _stimulus_base_features
                and not str(feature).startswith("stim_x_delay_param")
            ],
            key=_suffix_int,
        )
        _choice_lag_features = [
            feature
            for feature in ["choice_param", "at_choice", "at_choice_param", "prev_choice"]
            if feature in _features
        ] + sorted(
            [
                feature
                for feature in _features
                if str(feature).startswith("choice_lag_")
            ],
            key=_suffix_int,
        )

        _combined_features = list(dict.fromkeys([*_stimulus_features, *_choice_lag_features]))
        fig_emission_coefficients = plot_emission_condition_boxplot(
            comparison_weights_df.filter(pl.col("subject").is_in(["E10"]).not_()),
            feature_order=_combined_features,
        )
    return (fig_emission_coefficients,)


@app.cell
def _(fig_emission_coefficients, mo, save_plot):
    mo.stop(
        fig_emission_coefficients is None,
        mo.md("No emission weights available for the selected fits."),
    )
    ui_emission_coefficients_by_state = {
        _state_label: mo.ui.matplotlib(_ax, debounce=True)
        for _state_label, _ax in getattr(fig_emission_coefficients, "_subject_line_axes", [])
    }
    ui_emission_coefficients = mo.ui.tabs(
        ui_emission_coefficients_by_state,
        value=next(iter(ui_emission_coefficients_by_state), None),
    )
    mo.vstack(
        [
            ui_emission_coefficients,
            save_plot(
                fig_emission_coefficients,
                "emission weights saline drug",
                stem="emission_weights_saline_drug",
            ),
        ],
        align="center",
    )
    return (ui_emission_coefficients_by_state,)


@app.cell
def _(fig_emission_coefficients, mo, pd, ui_emission_coefficients_by_state):
    _points = pd.DataFrame(
        getattr(fig_emission_coefficients, "_subject_line_selection_points", [])
    )
    _selected_subjects = set()
    if not _points.empty:
        for _state_label, _ui in ui_emission_coefficients_by_state.items():
            if not _ui.value:
                continue
            _state_points = _points[_points["state"] == _state_label]
            if _state_points.empty:
                continue
            _mask = _ui.value.get_mask(
                _state_points["x"].to_numpy(),
                _state_points["y"].to_numpy(),
            )
            _selected_subjects.update(_state_points.loc[_mask, "subject"].tolist())
    selected_emission_subjects = sorted(_selected_subjects)
    mo.md(
        "Selected subjects: "
        + (", ".join(selected_emission_subjects) if selected_emission_subjects else "_none_")
    )
    return


if __name__ == "__main__":
    app.run()
