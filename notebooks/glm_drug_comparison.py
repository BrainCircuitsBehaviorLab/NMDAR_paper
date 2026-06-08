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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Task key .* already registered", category=RuntimeWarning)
        process_common = importlib.reload(process_common)
        process_mcdr = importlib.reload(process_mcdr)
        process_two_afc = importlib.reload(process_two_afc)
        process_two_adc = importlib.reload(process_two_adc)
    add_choice_lag_summary_regressor = process_common.add_choice_lag_summary_regressor

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    paths = get_runtime_paths()
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    sns.set_style("ticks")
    sns.set_context("notebook")

    CONDITION_PALETTE = {
        "saline": "tab:cyan",
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
        build_emission_weights_df,
        build_trial_df,
        build_views,
        custom_boxplot,
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
    TASK_NAME,
    fig_overlay,
    make_plot_saver,
    mo,
    model_kind,
    paths,
    ui_drug_alias,
    ui_rest_alias,
):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=TASK_NAME,
        model_id=f"drug_comparison/{model_kind}/{ui_rest_alias.value}_vs_{ui_drug_alias.value}",
    )
    mo.vstack(
        [
            fig_overlay,
            save_plot(fig_overlay, "drug saline overlay", stem="drug_saline_overlay"),
        ],
        align="center",
    )
    return (save_plot,)


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
        title: str | None = None,
    ):
        _raw = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        if _raw.empty:
            return None
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
        _fig, _axes = plt.subplots(
            1,
            max(1, len(_state_rows)),
            figsize=(max(5.0, 0.9 * len(_feature_order)) * max(1, len(_state_rows)), 4.2),
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
            _ax.set_title(f"{_state_row['plot_state_label']} by {_rank_feature}")
            _ax.set_xticks(range(len(_feature_order)))
            _ax.set_xticklabels(_feature_order, rotation=35, ha="right")
            _ax.set_xlabel("")
            sns.despine(ax=_ax)
        _axes[0].set_ylabel("Emission weight")
        _fig.legend(
            handles=[
                plt.Line2D([0], [0], color=CONDITION_PALETTE.get(str(ui_rest_condition.value), "#333333"), lw=3, label=str(ui_rest_condition.value)),
                plt.Line2D([0], [0], color=CONDITION_PALETTE.get(str(ui_drug_condition.value), "#333333"), lw=3, label=str(ui_drug_condition.value)),
            ],
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        if title is not None:
            _fig.suptitle(title, y=1.02)
        _fig.tight_layout()
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
            title="Emission weights",
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
            title="Emission weights",
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
