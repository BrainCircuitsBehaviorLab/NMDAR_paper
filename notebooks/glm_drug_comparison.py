import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2AFC drug versus rest fits
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys

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
    from glmhmmt.plots.common import custom_boxplot
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from plot_saver import make_plot_saver
    from src.process import two_afc as process_two_afc

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    paths = get_runtime_paths()
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    sns.set_style("ticks")
    sns.set_context("notebook")

    CONDITION_PALETTE = {
        "rest": "#3B6EA8",
        "drug": "#C44E52",
        "nan": "#666666",
        "all": "#333333",
    }
    TASK_NAME = "2AFC_DRUG"

    return (
        CONDITION_PALETTE,
        Line2D,
        TASK_NAME,
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
        process_two_afc,
        sns,
    )


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
        label="Rest fit",
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
def _(condition_counts, fit_aliases, mo, ui_drug_alias, ui_k, ui_model_kind, ui_rest_alias):
    mo.vstack(
        [
            mo.hstack([ui_model_kind, ui_k, ui_rest_alias, ui_drug_alias]),
            mo.md(f"Available aliases: `{len(fit_aliases)}`"),
            condition_counts,
        ]
    )
    return


@app.cell
def _(TASK_NAME, load_model_config, mo, paths, ui_drug_alias, ui_model_kind, ui_rest_alias):
    _root = paths.RESULTS / "fits" / TASK_NAME / ui_model_kind.value

    def _saved_condition(alias: str, fallback: str) -> str:
        cfg = load_model_config(
            task_name=TASK_NAME,
            model_kind=ui_model_kind.value,
            alias=alias,
            local_root=_root,
        )
        value = str(cfg.get("condition_filter", fallback) or fallback).lower()
        return "rest" if value == "saline" else value

    ui_rest_condition = mo.ui.dropdown(
        options=["rest", "drug", "nan", "all", "saline"],
        value=_saved_condition(ui_rest_alias.value, "rest"),
        label="Rest condition",
    )
    ui_drug_condition = mo.ui.dropdown(
        options=["drug", "rest", "nan", "all", "saline"],
        value=_saved_condition(ui_drug_alias.value, "drug"),
        label="Drug condition",
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
    return (
        K,
        common_subjects,
        drug_adapter,
        drug_arrays,
        drug_names,
        drug_views,
        model_kind,
        rest_adapter,
        rest_arrays,
        rest_names,
        rest_views,
    )


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
        return "rest" if value == "saline" else value

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
    return comparison_trial_df, drug_trial_df, rest_trial_df


@app.cell
def _(drug_trial_df, process_two_afc, rest_trial_df):
    rest_plot_df = process_two_afc.prepare_predictions_df(rest_trial_df)
    drug_plot_df = process_two_afc.prepare_predictions_df(drug_trial_df)
    return drug_plot_df, rest_plot_df


@app.cell
def _(drug_plot_df, mo, plots, rest_plot_df):
    _columns = set(rest_plot_df.columns) & set(drug_plot_df.columns)
    _preferred = [
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
    ui_regressor = mo.ui.dropdown(
        options=regressor_options or [""],
        value=plots.pick_choice_history_regressor(regressor_options) or (regressor_options[0] if regressor_options else ""),
        label="Regressor",
    )
    return regressor_options, ui_regressor


@app.cell
def _(mo, ui_regressor):
    ui_regressor
    return


@app.cell
def _(CONDITION_PALETTE, Line2D, adapter, drug_plot_df, drug_views, np, plots, plt, rest_plot_df, rest_views, ui_regressor):
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
        [["accuracy", "repeat_evidence"], ["repeat_regressor", "right_regressor"]],
        figsize=(7.0, 5.8),
        layout="constrained",
    )
    _payloads = [
        ("rest", rest_plot_df, rest_views),
        ("drug", drug_plot_df, drug_views),
    ]
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
            lambda _ax, _df=_plot_df, _vs=_views: plots.plot_repeat_by_regressor_simple(
                _df,
                regressor_col=ui_regressor.value,
                views=_vs,
                ax=_ax,
                legend=False,
            ),
            axd["repeat_regressor"],
            label=_label,
            color=_color,
        )
        _overlay(
            lambda _ax, _df=_plot_df: plots.plot_right_by_regressor(
                _df,
                regressor_col=ui_regressor.value,
                ax=_ax,
                legend=False,
            ),
            axd["right_regressor"],
            label=_label,
            color=_color,
        )

    axd["accuracy"].set_title("Accuracy by fitted evidence")
    axd["repeat_evidence"].set_title("Repeat by fitted evidence")
    axd["repeat_regressor"].set_title("Repeat by regressor")
    axd["right_regressor"].set_title("Right choice by regressor")
    for _axis in axd.values():
        _axis.set_box_aspect(1)
    fig_overlay.legend(
        handles=[
            Line2D([0], [0], color=CONDITION_PALETTE["rest"], lw=2, marker="o", label="rest"),
            Line2D([0], [0], color=CONDITION_PALETTE["drug"], lw=2, marker="o", label="drug"),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    return (fig_overlay,)


@app.cell
def _(TASK_NAME, fig_overlay, make_plot_saver, mo, model_kind, paths, ui_drug_alias, ui_rest_alias):
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
            save_plot(fig_overlay, "drug rest overlay", stem="drug_rest_overlay"),
        ],
        align="center",
    )
    return (save_plot,)


@app.cell
def _(build_emission_weights_df, drug_views, pl, rest_views):
    rest_weights_df = build_emission_weights_df(rest_views).with_columns(pl.lit("rest").alias("fit_condition"))
    drug_weights_df = build_emission_weights_df(drug_views).with_columns(pl.lit("drug").alias("fit_condition"))
    comparison_weights_df = pl.concat([rest_weights_df, drug_weights_df], how="diagonal_relaxed")
    return comparison_weights_df, drug_weights_df, rest_weights_df


@app.cell
def _(CONDITION_PALETTE, comparison_weights_df, custom_boxplot, np, pd, plt, sns):
    def plot_emission_condition_boxplot(weights_df, *, feature_order: list[str] | None = None):
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
        _feature_order = feature_order or list(dict.fromkeys(_raw["feature"].astype(str)))
        _state_rows = (
            _raw[["state_rank", "state_label"]]
            .drop_duplicates()
            .sort_values("state_rank")
        )
        _cond_order = [cond for cond in ["rest", "drug"] if cond in set(_raw["fit_condition"])]
        _fig, _axes = plt.subplots(
            1,
            max(1, len(_state_rows)),
            figsize=(max(5.0, 1.0 * len(_feature_order)) * max(1, len(_state_rows)), 3.8),
            sharey=True,
            squeeze=False,
        )
        _axes = _axes.ravel()
        _width = 0.34
        _offsets = np.linspace(-_width / 2, _width / 2, max(1, len(_cond_order)))
        for _ax, (_, _state_row) in zip(_axes, _state_rows.iterrows(), strict=False):
            _state_df = _raw[_raw["state_rank"] == _state_row["state_rank"]].copy()
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
                    _median_colors.append(CONDITION_PALETTE[_cond])
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
                if {"rest", "drug"}.issubset(_pivot.columns):
                    for _, _row in _pivot.iterrows():
                        _ys = [_row["rest"], _row["drug"]]
                        if np.all(np.isfinite(_ys)):
                            _ax.plot(
                                [_pos_by_pair[(_feature, "rest")], _pos_by_pair[(_feature, "drug")]],
                                _ys,
                                color="#B0B0B0",
                                alpha=0.25,
                                linewidth=1.0,
                                zorder=1,
                            )
            _ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
            _ax.set_title(str(_state_row["state_label"]))
            _ax.set_xticks(range(len(_feature_order)))
            _ax.set_xticklabels(_feature_order, rotation=35, ha="right")
            _ax.set_xlabel("")
            sns.despine(ax=_ax)
        _axes[0].set_ylabel("Emission weight")
        _fig.legend(
            handles=[
                plt.Line2D([0], [0], color=CONDITION_PALETTE["rest"], lw=3, label="rest"),
                plt.Line2D([0], [0], color=CONDITION_PALETTE["drug"], lw=3, label="drug"),
            ],
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        _fig.tight_layout()
        return _fig

    fig_emission = plot_emission_condition_boxplot(comparison_weights_df)
    return fig_emission, plot_emission_condition_boxplot


@app.cell
def _(fig_emission, mo, save_plot):
    mo.stop(fig_emission is None, mo.md("No emission weights available for the selected fits."))
    mo.vstack(
        [
            fig_emission,
            save_plot(fig_emission, "emission weights rest drug", stem="emission_weights_rest_drug"),
        ],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
