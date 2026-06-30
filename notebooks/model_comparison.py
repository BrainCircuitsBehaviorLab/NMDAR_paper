import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from glmhmmt.runtime import get_runtime_paths

    paths = get_runtime_paths()
    from glmhmmt.notebook_support.analysis_common import (
        load_fit_bundle as load_fit_bundle_raw,
        load_metrics_dir as load_metrics_dir_raw,
        model_aliases_for_kind,
    )
    from glmhmmt.tasks import get_adapter, get_task_options
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df
    from glmhmmt.views import build_views
    from matplotlib.lines import Line2D
    from glmhmmt.plots_common import custom_boxplot
    from plot_saver import make_plot_saver
    from src.process.common import adapter_behavioral_column
    from src.plots.common import fig_size
    sns.set_style("white")
    sns.set_context("paper")
    return (
        Line2D,
        adapter_behavioral_column,
        build_emission_weights_df,
        build_trial_df,
        build_views,
        custom_boxplot,
        fig_size,
        get_adapter,
        get_task_options,
        load_fit_bundle_raw,
        load_metrics_dir_raw,
        make_plot_saver,
        mo,
        model_aliases_for_kind,
        np,
        paths,
        pl,
        plt,
        sns,
    )


@app.cell
def _(mo, save_plot):
    def panel(title, fig=None, stem=None, description=None):
        content = [mo.md(f"#### {title}")]

        if fig is not None:
            content.append(fig)
            if stem is not None:
                content.append(save_plot(fig, description or title.lower(), stem=stem))

        return mo.vstack(content, align="center")

    BOXPLOT_STYLE = dict(
        fill=False,
        boxprops={"color": "0.5"},
        whiskerprops={"color": "0.5"},
        medianprops={"linewidth": 2},
        showfliers=False,
        showcaps=False,
    )
    return BOXPLOT_STYLE, panel


@app.cell
def _(get_task_options, mo):
    _task_options = get_task_options()
    ui_task = mo.ui.dropdown(
        options={opt["label"]: opt["value"] for opt in _task_options},
        value="MCDR",
        label="Task",
    )
    return (ui_task,)


@app.cell
def _(load_metrics_dir_raw, model_aliases_for_kind, paths, pl):
    _MODEL_LABELS = {
        "glm": "GLM",
        "glmhmm": "GLMHMM",
        "glmhmmt": "GLMHMMT",
    }

    def model_aliases(task: str, kind: str) -> list[str]:
        return model_aliases_for_kind(
            task_name=task,
            model_kind=kind,
            local_root=paths.RESULTS / "fits" / task / kind,
        )

    def load_metrics_dir_for_notebook(task_name: str, folder_name: str | None, expected_model_kind: str):
        df = load_metrics_dir_raw(
            task_name=task_name,
            model_kind=expected_model_kind,
            alias=folder_name,
            local_root=paths.RESULTS / "fits" / task_name / expected_model_kind,
            label_map=_MODEL_LABELS,
        )
        if df is None:
            return None
        if "test_ll_per_trial_mean" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("test_ll_per_trial_mean"))
        df = df.with_columns(pl.col("test_ll_per_trial_mean").alias("test_ll_per_trial"))
        df = df.with_columns(pl.lit(expected_model_kind).alias("model_kind"))
        keep = [
            "subject",
            "K",
            "model_kind",
            "model_alias",
            "model_label",
            "ll_per_trial",
            "test_ll_per_trial",
            "test_ll_per_trial_mean",
            "bic",
            "acc",
            "n_trials",
        ]
        return df.select([c for c in keep if c in df.columns])

    def model_k_options(task: str, kind: str, alias: str | None) -> list[int]:
        df = load_metrics_dir_for_notebook(task, alias, kind)
        if df is None or df.is_empty():
            return []
        return sorted(
            {
                int(k)
                for k in df["K"].drop_nulls().to_list()
            }
        )

    load_metrics_dir = load_metrics_dir_for_notebook
    return load_metrics_dir, model_aliases


@app.cell
def _(make_plot_saver, mo, paths, ui_task):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=ui_task.value,
        model_id=f"model_comparison",
    )
    return (save_plot,)


@app.cell
def _(build_views, get_adapter, load_fit_bundle_raw, paths):
    def load_fit_bundle_for_notebook(task_name, model_kind, alias, K, subjects, scoring_key=None):
        return load_fit_bundle_raw(
            task_name=task_name,
            model_kind=model_kind,
            alias=alias,
            k=K,
            subjects=list(subjects),
            get_adapter=get_adapter,
            build_views=build_views,
            scoring_key=scoring_key,
            local_root=paths.RESULTS / "fits" / task_name / model_kind,
        )

    load_fit_bundle = load_fit_bundle_for_notebook
    return (load_fit_bundle,)


@app.cell
def _(get_adapter, mo, model_aliases, ui_task):
    adapter = get_adapter(ui_task.value)

    ui_glm_dir = mo.ui.multiselect(
        options=model_aliases(ui_task.value, "glm"),
        value=[],
        label="GLM aliases",
    )
    ui_glmhmm_dir = mo.ui.multiselect(
        options=model_aliases(ui_task.value, "glmhmm"),
        value=[],
        label="GLMHMM aliases",
    )
    ui_glmhmmt_dir = mo.ui.multiselect(
        options=model_aliases(ui_task.value, "glmhmmt"),
        value=[],
        label="GLMHMM-T aliases",
    )

    mo.vstack([
        mo.md("### Model Comparison — Configuration"),
        mo.md(
            "Select one or more aliases for each model kind. "
            "Leave empty to skip that model."
        ),
        mo.hstack([ui_task]),
        mo.hstack([ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir]),
    ])
    return adapter, ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir


@app.cell
def _(adapter, mo):
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    _all_subjects = df_all["subject"].unique().sort().to_list()

    ui_subjects = mo.ui.multiselect(
        options=_all_subjects,
        value=_all_subjects,
        label="Subjects",
    )
    ui_K_range = mo.ui.range_slider(
        start=1, stop=10, step=1, value=[1, 5],
        full_width=True, label="K range",
    )

    mo.vstack([
        mo.hstack([ui_subjects]),
        mo.hstack([mo.md("K range:"), ui_K_range]),
    ])
    return df_all, ui_K_range, ui_subjects


@app.cell
def _(
    load_metrics_dir,
    mo,
    pl,
    ui_glm_dir,
    ui_glmhmm_dir,
    ui_glmhmmt_dir,
    ui_task,
):
    _parts = []
    for _names, _kind in [
        (ui_glm_dir.value, "glm"),
        (ui_glmhmm_dir.value, "glmhmm"),
        (ui_glmhmmt_dir.value, "glmhmmt"),
    ]:
        for _name in _names:
            _p = load_metrics_dir(ui_task.value, _name, _kind)
            if _p is not None:
                _parts.append(_p)

    if _parts:
        results_long = pl.concat(_parts, how="diagonal")
    else:
        results_long = pl.DataFrame(
            schema={
                "subject": pl.Utf8, "K": pl.Int64, "model_kind": pl.Utf8,
                "model_alias": pl.Utf8, "model_label": pl.Utf8,
                "ll_per_trial": pl.Float64, "test_ll_per_trial": pl.Float64,
                "bic": pl.Float64, "acc": pl.Float64,
            }
        )

    mo.stop(
        results_long.is_empty(),
        mo.md("⚠️  No metrics loaded — select at least one fit folder above."),
    )
    mo.md(
        f"Loaded **{results_long.height}** fit rows from "
        f"**{len(_parts)}** model folder(s)."
    )
    return (results_long,)


@app.cell
def _(mo, pl, results_long):
    _MODEL_ORDER = [
        ("glm", "GLM"),
        ("glmhmmt", "GLMHMMT"),
        ("glmhmm", "GLMHMM"),
    ]
    _elements = {}
    _metadata = {}

    for _kind, _kind_label in _MODEL_ORDER:
        _kind_df = results_long.filter(pl.col("model_kind") == _kind)
        for _K in _kind_df["K"].drop_nulls().unique().sort().to_list():
            _aliases = (
                _kind_df
                .filter(pl.col("K") == _K)
                .select("model_alias")
                .unique()
                .sort("model_alias")
                .get_column("model_alias")
                .to_list()
            )
            if not _aliases:
                continue
            _key = f"{_kind}:{int(_K)}"
            _metadata[_key] = {
                "model_kind": _kind,
                "model_kind_label": _kind_label,
                "K": int(_K),
            }
            _elements[_key] = mo.ui.dropdown(
                options={"Skip": "__skip__", **{_alias: _alias for _alias in _aliases}},
                value=_aliases[0],
                label=f"{_kind_label} K={int(_K)}",
            )

    ui_model_picks = mo.ui.dictionary(_elements, label="Model picks")
    _rows = [
        mo.hstack([
            mo.md(f"**{_metadata[_key]['model_kind_label']} K={_metadata[_key]['K']}**"),
            ui_model_picks[_key],
        ])
        for _key in _elements
    ]

    mo.vstack([
        mo.md("### One model per family and state count"),
        mo.md("Pick at most one alias for each model family and `K`. Use `Skip` to omit a point."),
        *(_rows if _rows else [mo.md("No model/K combinations available for the selected aliases.")]),
    ])
    return (ui_model_picks,)


@app.cell
def _(pl, ui_model_picks):
    _selected = []
    for _key, _alias in ui_model_picks.value.items():
        if _alias == "__skip__":
            continue
        _kind, _K = _key.split(":", 1)
        _selected.append(
            {
                "model_kind": _kind,
                "K": int(_K),
                "model_alias": _alias,
            }
        )

    selected_model_specs = pl.DataFrame(
        _selected,
        schema={
            "model_kind": pl.Utf8,
            "K": pl.Int64,
            "model_alias": pl.Utf8,
        },
    )
    return (selected_model_specs,)


@app.cell
def _(pl, results_long, selected_model_specs, ui_K_range, ui_subjects):
    K_min, K_max = ui_K_range.value
    if selected_model_specs.is_empty():
        results_filtered = results_long.head(0)
    else:
        results_filtered = (
            results_long
            .join(selected_model_specs, on=["model_kind", "K", "model_alias"], how="inner")
            .filter(
                pl.col("subject").is_in(ui_subjects.value)
                & pl.col("K").is_between(K_min, K_max)
            )
        )
    return (results_filtered,)


@app.cell
def _(adapter, df_all, mo, pl):
    _enum_dtype = getattr(pl, "Enum", None)
    if getattr(adapter, "num_classes", None) == 3:
        _preferred = [
            "stimd_n",
            "stimd_c",
            "ttype_n",
            "ttype_c",
            "condition",
            "Condition",
            "Experiment",
            adapter.session_col,
        ]
        _default_candidates = ["stimd_n", "stimd_c", "ttype_n", "ttype_c"]
    else:
        _preferred = [
            "ILD",
            "ild",
            "stim_vals",
            "stim_d",
            "stim_strength",
            "condition",
            "Condition",
            "Experiment",
            adapter.session_col,
        ]
        _default_candidates = ["ILD", "ild", "stim_vals", "stim_d", "stim_strength"]
    _seen = set()
    _options = []
    for _col in _preferred:
        if _col in df_all.columns and _col not in _seen:
            _options.append(_col)
            _seen.add(_col)
    for _col, _dtype in df_all.schema.items():
        if _col in _seen or _col == "subject":
            continue
        if _dtype in tuple(
            _dt for _dt in (pl.Utf8, pl.Categorical, _enum_dtype, pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
            if _dt is not None
        ):
            _options.append(_col)
            _seen.add(_col)

    _default = next((_col for _col in _default_candidates if _col in _options), None)
    if _default is None:
        _default = "condition" if "condition" in _options else (_options[0] if _options else None)
    ui_ce_condition = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="Cross-entropy grouping",
    )
    mo.hstack([ui_ce_condition])
    return


@app.cell
def _(mo, results_filtered):
    _baseline_options = results_filtered["model_label"].unique().sort().to_list()
    _baseline_value = _baseline_options[0] if _baseline_options else None
    ui_bic_baseline = mo.ui.dropdown(
        options=_baseline_options,
        value=_baseline_value,
        label="BIC baseline model",
    )
    mo.hstack([ui_bic_baseline])
    return (ui_bic_baseline,)


@app.cell
def _(pl, results_filtered, ui_bic_baseline):
    if results_filtered.is_empty() or ui_bic_baseline.value is None:
        results_plot = results_filtered.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("bic_delta")
        )
    else:
        _baseline_bic = (
            results_filtered
            .filter(pl.col("model_label") == ui_bic_baseline.value)
            .group_by("subject")
            .agg(pl.first("bic").alias("bic_baseline"))
        )
        results_plot = (
            results_filtered
            .join(_baseline_bic, on="subject", how="left")
            .with_columns(((pl.col("bic") - pl.col("bic_baseline"))/pl.col("bic_baseline")).alias("bic_delta"))
        )
    return (results_plot,)


@app.cell
def _(mo, results_filtered):
    _highlight_options = results_filtered["subject"].unique().sort().to_list()
    ui_highlight_subject = mo.ui.dropdown(
        options={"None": "__none__", **{_subject: _subject for _subject in _highlight_options}},
        value="None",
        label="Dashed subject",
    )
    mo.hstack([ui_highlight_subject])
    return (ui_highlight_subject,)


@app.cell
def _(np):
    def observed_choice_index(adapter, trial_df):
        _resp = np.asarray(trial_df["response"]).astype(object)
        _out = np.full(len(_resp), -1, dtype=int)

        if adapter.num_classes == 2:
            for _i, _val in enumerate(_resp):
                if _val is None:
                    continue
                try:
                    _f = float(_val)
                    if _f in (0.0, 1.0):
                        _out[_i] = int(_f)
                    elif _f in (-1.0, 1.0):
                        _out[_i] = 1 if _f > 0 else 0
                except (TypeError, ValueError):
                    _s = str(_val).strip().upper()
                    if _s in {"L", "LEFT"}:
                        _out[_i] = 0
                    elif _s in {"R", "RIGHT"}:
                        _out[_i] = 1
        else:
            for _i, _val in enumerate(_resp):
                if _val is None:
                    continue
                try:
                    _f = float(_val)
                    if _f in (0.0, 1.0, 2.0):
                        _out[_i] = int(_f)
                    elif _f in (1.0, 2.0, 3.0):
                        _out[_i] = int(_f) - 1
                except (TypeError, ValueError):
                    _s = str(_val).strip().upper()
                    if _s in {"L", "LEFT"}:
                        _out[_i] = 0
                    elif _s in {"C", "CENTER", "CENTRE"}:
                        _out[_i] = 1
                    elif _s in {"R", "RIGHT"}:
                        _out[_i] = 2
        return _out

    return


@app.cell
def _(pl, results_filtered):
    agg = (
        results_filtered.group_by(["model_kind", "model_alias", "model_label", "K"])
        .agg([
            pl.len().alias("n_subjects"),
            pl.mean("test_ll_per_trial").alias("test_ll_mean"),
            pl.std("test_ll_per_trial").alias("test_ll_std"),
            pl.mean("bic").alias("bic_mean"),
            pl.std("bic").alias("bic_std"),
        ])
        .with_columns([
            (pl.col("test_ll_std")  / pl.col("n_subjects").sqrt()).alias("test_ll_sem"),
            (pl.col("bic_std") / pl.col("n_subjects").sqrt()).alias("bic_sem"),
        ])
        .sort(["model_kind", "model_alias", "K"])
    )
    return


@app.cell
def _(Line2D, mo, np, pl, plt, results_filtered, sns, ui_highlight_subject):
    mo.stop(results_filtered.is_empty(), mo.md("No selected model metrics to plot."))
    mo.stop(
        results_filtered.filter(pl.col("model_kind") == "glm").is_empty(),
        mo.md("Select a GLM model to use as the LL increment baseline."),
    )

    _MODEL_STYLES = {
        "glm": {"label": "GLM", "marker": "s", "color": "#4C78A8"},
        "glmhmmt": {"label": "GLMHMMT", "marker": "^", "color": "#54A24B"},
        "glmhmm": {"label": "GLMHMM", "marker": "o", "color": "#F58518"},
    }
    _model_order = ["glm", "glmhmmt", "glmhmm"]
    _highlight = ui_highlight_subject.value
    _glm_baseline = (
        results_filtered
        .filter(pl.col("model_kind") == "glm")
        .sort(["subject", "K", "model_alias"])
        .group_by("subject")
        .agg(pl.first("test_ll_per_trial").alias("glm_test_ll_per_trial"))
    )

    _plot_df = (
        results_filtered
        .join(_glm_baseline, on="subject", how="inner")
        .with_columns(
            ((pl.col("test_ll_per_trial") - pl.col("glm_test_ll_per_trial")) / np.log(2)).alias("test_ll_increment_bits")
        )
        .sort(["model_kind", "subject", "K"])
        .to_pandas()
    )

    _fig_ll_bits, _ax_ll_bits = plt.subplots(figsize=(7.2, 4.6))

    for _kind in _model_order:
        _sub = _plot_df[_plot_df["model_kind"] == _kind]
        if _sub.empty:
            continue
        _style = _MODEL_STYLES[_kind]
        for _subject, _subject_df in _sub.groupby("subject"):
            _subject_df = _subject_df.sort_values("K")
            _is_highlight = _highlight != "__none__" and _subject == _highlight
            _ax_ll_bits.plot(
                _subject_df["K"],
                _subject_df["test_ll_increment_bits"],
                color=_style["color"],
                linestyle="--" if _is_highlight else "-",
                linewidth=2.0 if _is_highlight else 0.8,
                alpha=0.9 if _is_highlight else 0.16,
                marker=_style["marker"] if _is_highlight else None,
                markersize=4,
                zorder=3 if _is_highlight else 1,
            )

        _mean_df = (
            _sub
            .groupby("K", as_index=False)["test_ll_increment_bits"]
            .mean()
            .sort_values("K")
        )
        _ax_ll_bits.plot(
            _mean_df["K"],
            _mean_df["test_ll_increment_bits"],
            color=_style["color"],
            marker=_style["marker"],
            linewidth=2.6,
            markersize=5,
            label=f"{_style['label']} mean",
            zorder=4,
        )

    _ax_ll_bits.axhline(0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.6)
    _ax_ll_bits.set_xlabel("Number of states K")
    _ax_ll_bits.set_ylabel("Test LL increment vs GLM (bits / trial)")
    _ax_ll_bits.set_title("Per-animal and mean test LL increment vs GLM")
    _K_values = sorted(_plot_df["K"].dropna().unique())
    if _K_values:
        _ax_ll_bits.set_xticks(_K_values)

    _handles, _labels = _ax_ll_bits.get_legend_handles_labels()
    if _highlight != "__none__":
        _handles.append(Line2D([0], [0], color="0.2", linestyle="--", linewidth=2, label=f"{_highlight}"))
        _labels.append(f"{_highlight} dashed")
    _ax_ll_bits.legend(_handles, _labels, frameon=False, loc="best")
    sns.despine(ax=_ax_ll_bits)
    _fig_ll_bits.tight_layout()
    _fig_ll_bits
    return


@app.cell
def _():
    import itertools
    import pandas as pd
    from scipy.stats import ttest_1samp, ttest_rel, ttest_ind

    def _sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    def add_sig_bars(ax, df, *, x_col, y_col, hue_col, order, hue_order, pair_col=None):
        n_hue = max(1, len(hue_order))
        hue_width = 0.8 / n_hue
        y_range = df[y_col].max() - df[y_col].min()
        if pd.isna(y_range) or y_range == 0:
            y_range = 1.0

        for m, xval in enumerate(order):
            sub = df[df[x_col] == xval]
            if sub.empty:
                continue

            current_y = sub[y_col].max() + y_range * 0.05
            h = y_range * 0.02

            for p1, p2 in itertools.combinations(range(n_hue), 2):
                g1 = hue_order[p1]
                g2 = hue_order[p2]

                s1 = sub[sub[hue_col] == g1]
                s2 = sub[sub[hue_col] == g2]

                if pair_col is not None:
                    v1 = s1.set_index(pair_col)[y_col]
                    v2 = s2.set_index(pair_col)[y_col]
                    common = v1.index.intersection(v2.index)
                    if len(common) < 2:
                        continue
                    _, pval = ttest_rel(v1.loc[common].values, v2.loc[common].values)
                else:
                    v1 = s1[y_col].dropna().values
                    v2 = s2[y_col].dropna().values
                    if min(len(v1), len(v2)) < 2:
                        continue
                    _, pval = ttest_ind(v1, v2, equal_var=False)

                star = _sig_label(pval)
                if star == "ns":
                    continue

                x1 = m + (p1 - (n_hue - 1) / 2) * hue_width
                x2 = m + (p2 - (n_hue - 1) / 2) * hue_width

                ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y], lw=1, c="k")
                ax.text((x1 + x2) / 2, current_y + h, star, ha="center", va="bottom", color="k")
                current_y += y_range * 0.075


    return add_sig_bars, pd, ttest_1samp, ttest_rel


@app.cell
def _(
    Line2D,
    add_sig_bars,
    custom_boxplot,
    np,
    plt,
    results_plot,
    sns,
    ui_bic_baseline,
):
    from matplotlib.colors import to_rgb, to_hex

    _MODEL_STYLES = {
        "glm": {"marker": "s", "label": "GLM"},
        "glmhmm": {"marker": "o", "label": "GLMHMM"},
        "glmhmmt": {"marker": "^", "label": "GLMHMM-T"},
    }

    def darken(color, factor=0.75):
        rgb = np.array(to_rgb(color))
        return to_hex(np.clip(rgb * factor, 0, 1))

    raw = results_plot.to_pandas()

    _label_df = raw[["model_kind", "model_label"]].drop_duplicates()
    hue_order = _label_df["model_label"].tolist()
    _base_colors = sns.color_palette("tab20", n_colors=max(1, len(hue_order)))
    palette = {
        _label: to_hex(_base_colors[_i])
        for _i, _label in enumerate(hue_order)
    }
    strip_palette = {
        _label: darken(palette[_label], 0.70)
        for _label in hue_order
    }
    K_order = sorted(raw["K"].unique())

    fig_cmp, (ax_ll, ax_bic) = plt.subplots(1, 2, figsize=(8, 4.8), constrained_layout=False)

    def _grouped_custom_boxplot(ax, ycol: str) -> None:
        if not hue_order or not K_order:
            return

        hue_width = 0.8 / len(hue_order)
        grouped_values = []
        positions = []
        median_colors = []

        for x_idx, k_val in enumerate(K_order):
            for hue_idx, hue_label in enumerate(hue_order):
                vals = raw[
                    (raw["K"] == k_val)
                    & (raw["model_label"] == hue_label)
                ][ycol].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                positions.append(x_idx + (hue_idx - (len(hue_order) - 1) / 2) * hue_width)
                grouped_values.append(vals)
                median_colors.append(palette[hue_label])

        if grouped_values:
            custom_boxplot(
                ax,
                grouped_values,
                positions=positions,
                widths=hue_width * 0.9,
                median_colors=median_colors,
                showfliers=False,
                showcaps=False,
                zorder=1,
            )

    for ax, ycol in [(ax_ll, "test_ll_per_trial"), (ax_bic, "bic_delta")]:
        _grouped_custom_boxplot(ax, ycol)

        sns.stripplot(
            data=raw,
            x="K",
            y=ycol,
            hue="model_label",
            order=K_order,
            hue_order=hue_order,
            palette=strip_palette,
            dodge=True,
            jitter=0.18,
            alpha=0.85,
            size=4,
            ax=ax,
            legend=False,
        )

    add_sig_bars(
        ax_ll, raw,
        x_col="K", y_col="test_ll_per_trial", hue_col="model_label",
        order=K_order, hue_order=hue_order, pair_col="subject",
    )

    add_sig_bars(
        ax_bic, raw,
        x_col="K", y_col="bic_delta", hue_col="model_label",
        order=K_order, hue_order=hue_order, pair_col="subject",
    )

    ax_ll.set_ylabel("CV test log-likelihood / trial")
    ax_ll.set_title("CV test LL / trial (higher = better)")

    ax_bic.axhline(0, color="grey", lw=0.9, linestyle="--", alpha=0.7)
    ax_bic.set_ylabel("ΔBIC vs baseline")
    ax_bic.set_title(f"ΔBIC vs {ui_bic_baseline.value} (lower = better)")

    _legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=strip_palette[_label], label=_label, markersize=6)
        for _label in hue_order
    ]
    _legend_labels = list(hue_order)
    if ax_ll.get_legend() is not None:
        ax_ll.get_legend().remove()
    if ax_bic.get_legend() is not None:
        ax_bic.get_legend().remove()
    fig_cmp.legend(
        _legend_handles,
        _legend_labels,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, max(1, len(_legend_labels))),
        frameon=False,
    )

    sns.despine(fig=fig_cmp)
    fig_cmp.tight_layout(rect=(0, 0.12, 1, 1))
    fig_cmp
    return


@app.cell
def _(plt, results_plot, sns, ui_bic_baseline):
    def _cov_group(label):
        _label = str(label).lower()
        if "3 cov" in _label or  "3cov" in _label:
            return "3 covs"
        if "2 cov" in _label:
            return "2 covs"
        if "base_lapses" in _label:
            return "GLM"
        return None

    _raw = results_plot.to_pandas()
    _cov_mean = (
        _raw.assign(cov_group=_raw["model_label"].map(_cov_group))
        .dropna(subset=["cov_group"])
        .groupby(["cov_group", "K"], as_index=False)[["test_ll_per_trial", "bic_delta"]]
        .mean()
        .sort_values(["cov_group", "K"])
    )

    _cov_order = [grp for grp in ["GLM", "2 covs", "3 covs"] if grp in _cov_mean["cov_group"].unique()]
    _palette = {
        _group: _color
        for _group, _color in zip(_cov_order, sns.color_palette("tab10", n_colors=max(1, len(_cov_order))))
    }

    fig_cov_mean, (ax_cov_ll, ax_cov_bic) = plt.subplots(
        1, 2, figsize=(8, 4.2), constrained_layout=False
    )

    for _group in _cov_order:
        _group_df = _cov_mean[_cov_mean["cov_group"] == _group]
        ax_cov_ll.plot(
            _group_df["K"],
            _group_df["test_ll_per_trial"],
            marker="o",
            linewidth=2,
            color=_palette[_group],
            label=_group,
        )
        ax_cov_bic.plot(
            _group_df["K"],
            _group_df["bic_delta"],
            marker="o",
            linewidth=2,
            color=_palette[_group],
            label=_group,
        )

    ax_cov_ll.set_xlabel("Number of states K")
    ax_cov_ll.set_ylabel("Mean CV test LL / trial")
    ax_cov_ll.set_title("Mean CV test LL / trial by covariate count")

    ax_cov_bic.axhline(0, color="grey", lw=0.9, linestyle="--", alpha=0.7)
    ax_cov_bic.set_xlabel("Number of states K")
    ax_cov_bic.set_ylabel("Mean ΔBIC vs baseline")
    ax_cov_bic.set_title(f"Mean ΔBIC vs {ui_bic_baseline.value}")

    if _cov_order:
        fig_cov_mean.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(_cov_order),
            frameon=False,
            title="Grouped models",
        )

    sns.despine(fig=fig_cov_mean)
    fig_cov_mean.tight_layout(rect=(0, 0.12, 1, 1))
    fig_cov_mean
    return


@app.cell
def _(mo, results_long):
    _MODEL_KIND_LABELS = {
        "glm": "GLM",
        "glmhmm": "GLMHMM",
        "glmhmmt": "GLMHMM-T",
    }

    def _pairwise_model_key(kind: str, K: int, alias: str) -> str:
        return f"{kind}\t{int(K)}\t{alias}"

    def parse_pairwise_model_key(key: str) -> tuple[str, int, str]:
        _kind, _K, _alias = key.split("\t", 2)
        return _kind, int(_K), _alias

    _spec_rows = (
        results_long
        .select(["model_kind", "model_alias", "model_label", "K"])
        .unique()
        .sort(["model_kind", "model_alias", "K"])
        .iter_rows(named=True)
    )
    _pair_options = {}
    for _row in _spec_rows:
        _kind = _row["model_kind"]
        if _kind != "glm":
            _K = int(_row["K"])
        else:
            _K = 1
        _alias = _row["model_alias"]
        _kind_label = _MODEL_KIND_LABELS.get(_kind, str(_kind).upper())
        _label = f"{_kind_label} K={_K}: {_alias}"
        _pair_options[_label] = _pairwise_model_key(_kind, _K, _alias)

    _labels = list(_pair_options.keys())
    _default_a = _labels[0] if _labels else None
    _default_b = _labels[1] if len(_labels) > 1 else _default_a
    ui_pairwise_model_a = mo.ui.dropdown(
        options=_pair_options,
        value=_default_a,
        label="Model A",
    )
    ui_pairwise_model_b = mo.ui.dropdown(
        options=_pair_options,
        value=_default_b,
        label="Model B",
    )
    mo.vstack(
        [
            mo.md("### Pairwise model comparison"),
            mo.md(
                "Compare any two loaded model fits. `K` is selected independently for each side, "
                "so paired test LL and BIC do not require a shared state count."
            ),
            mo.hstack([ui_pairwise_model_a, ui_pairwise_model_b]),
            mo.md(
                "Uses the subjects selected above. Pick a smaller subset there if you want a tighter visual comparison."
            ),
        ]
    )
    return parse_pairwise_model_key, ui_pairwise_model_a, ui_pairwise_model_b


@app.cell
def _(adapter, mo):
    _opts = list(adapter._SCORING_OPTIONS.keys()) if hasattr(adapter, "_SCORING_OPTIONS") else ["default"]
    _default_key = getattr(adapter, "scoring_key", _opts[0]) if _opts else None
    if _opts and _default_key not in _opts:
        _default_key = _opts[0]

    ui_pairwise_scoring_key_a = mo.ui.dropdown(
        options=_opts,
        value=_default_key,
        label="Model A state scoring regressor",
    )
    ui_pairwise_scoring_key_b = mo.ui.dropdown(
        options=_opts,
        value=_default_key,
        label="Model B state scoring regressor",
    )
    mo.hstack([ui_pairwise_scoring_key_a, ui_pairwise_scoring_key_b])
    return ui_pairwise_scoring_key_a, ui_pairwise_scoring_key_b


@app.cell
def _(
    load_fit_bundle,
    load_metrics_dir,
    mo,
    parse_pairwise_model_key,
    pl,
    ui_pairwise_model_a,
    ui_pairwise_model_b,
    ui_pairwise_scoring_key_a,
    ui_pairwise_scoring_key_b,
    ui_subjects,
    ui_task,
):
    mo.stop(
        not ui_pairwise_model_a.value or not ui_pairwise_model_b.value,
        mo.md("Select two model fits above."),
    )
    mo.stop(
        ui_pairwise_model_a.value == ui_pairwise_model_b.value,
        mo.md("Choose two different model fits for a pairwise comparison."),
    )

    pairwise_kind_a, pairwise_K_a, pairwise_alias_a = parse_pairwise_model_key(ui_pairwise_model_a.value)
    pairwise_kind_b, pairwise_K_b, pairwise_alias_b = parse_pairwise_model_key(ui_pairwise_model_b.value)
    requested_subjects = list(ui_subjects.value)

    pairwise_fit_error_a = None
    pairwise_fit_error_b = None
    try:
        pairwise_adapter_a, pairwise_arrays_a, pairwise_names_a, pairwise_views_a = load_fit_bundle(
            ui_task.value,
            pairwise_kind_a,
            pairwise_alias_a,
            pairwise_K_a,
            requested_subjects,
            scoring_key=ui_pairwise_scoring_key_a.value,
        )
    except Exception as _e:
        pairwise_fit_error_a = str(_e)
        pairwise_adapter_a, pairwise_arrays_a, pairwise_names_a, pairwise_views_a = None, {}, {}, {}

    try:
        pairwise_adapter_b, pairwise_arrays_b, pairwise_names_b, pairwise_views_b = load_fit_bundle(
            ui_task.value,
            pairwise_kind_b,
            pairwise_alias_b,
            pairwise_K_b,
            requested_subjects,
            scoring_key=ui_pairwise_scoring_key_b.value,
        )
    except Exception as _e:
        pairwise_fit_error_b = str(_e)
        pairwise_adapter_b, pairwise_arrays_b, pairwise_names_b, pairwise_views_b = None, {}, {}, {}

    _metric_schema = {
        "subject": pl.Utf8,
        "K": pl.Int64,
        "model_kind": pl.Utf8,
        "model_alias": pl.Utf8,
        "model_label": pl.Utf8,
        "ll_per_trial": pl.Float64,
        "test_ll_per_trial": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
    }

    def _pair_metrics(kind: str, alias: str, K: int):
        _df = load_metrics_dir(ui_task.value, alias, kind)
        if _df is None:
            return pl.DataFrame(schema=_metric_schema)
        return _df.filter(
            pl.col("subject").is_in(requested_subjects)
            & (pl.col("K") == K)
        )

    pairwise_metrics_a = _pair_metrics(pairwise_kind_a, pairwise_alias_a, pairwise_K_a)
    pairwise_metrics_b = _pair_metrics(pairwise_kind_b, pairwise_alias_b, pairwise_K_b)
    _metric_subjects_a = set(pairwise_metrics_a["subject"].to_list()) if not pairwise_metrics_a.is_empty() else set()
    _metric_subjects_b = set(pairwise_metrics_b["subject"].to_list()) if not pairwise_metrics_b.is_empty() else set()
    _cached_subjects_a = set(pairwise_views_a)
    _cached_subjects_b = set(pairwise_views_b)
    pairwise_common_subjects = [
        _subject
        for _subject in requested_subjects
        if _subject in _metric_subjects_a and _subject in _metric_subjects_b
    ]
    pairwise_cached_common_subjects = [
        _subject
        for _subject in requested_subjects
        if _subject in _cached_subjects_a and _subject in _cached_subjects_b
    ]
    if pairwise_common_subjects:
        pairwise_metrics_a = pairwise_metrics_a.filter(pl.col("subject").is_in(pairwise_common_subjects))
        pairwise_metrics_b = pairwise_metrics_b.filter(pl.col("subject").is_in(pairwise_common_subjects))
    mo.stop(
        not pairwise_common_subjects and not pairwise_cached_common_subjects,
        mo.md(
            "No common subjects were found for the selected model fits. "
            "Check the subject subset or the cached metrics."
        ),
    )
    pairwise_missing_a = [s for s in requested_subjects if s not in pairwise_views_a]
    pairwise_missing_b = [s for s in requested_subjects if s not in pairwise_views_b]
    return (
        pairwise_K_a,
        pairwise_K_b,
        pairwise_adapter_a,
        pairwise_adapter_b,
        pairwise_alias_a,
        pairwise_alias_b,
        pairwise_arrays_a,
        pairwise_arrays_b,
        pairwise_cached_common_subjects,
        pairwise_common_subjects,
        pairwise_fit_error_a,
        pairwise_fit_error_b,
        pairwise_kind_a,
        pairwise_kind_b,
        pairwise_metrics_a,
        pairwise_metrics_b,
        pairwise_missing_a,
        pairwise_missing_b,
        pairwise_names_a,
        pairwise_names_b,
        pairwise_views_a,
        pairwise_views_b,
        requested_subjects,
    )


@app.cell
def _(pairwise_metrics_b):
    pairwise_metrics_b
    return


@app.cell
def _(
    mo,
    pairwise_K_a,
    pairwise_K_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_cached_common_subjects,
    pairwise_common_subjects,
    pairwise_fit_error_a,
    pairwise_fit_error_b,
    pairwise_kind_a,
    pairwise_kind_b,
    pairwise_missing_a,
    pairwise_missing_b,
    requested_subjects,
    ui_pairwise_scoring_key_a,
    ui_pairwise_scoring_key_b,
):
    _notes = [
        f"- Comparing A `{pairwise_kind_a}` / `{pairwise_alias_a}` / `K={pairwise_K_a}` vs B `{pairwise_kind_b}` / `{pairwise_alias_b}` / `K={pairwise_K_b}`.",
        f"- Common metric subjects: **{len(pairwise_common_subjects)} / {len(requested_subjects)}**.",
        f"- Common cached fit subjects for state-level plots: **{len(pairwise_cached_common_subjects)} / {len(requested_subjects)}**.",
        f"- `{pairwise_alias_a}` scoring key: `{ui_pairwise_scoring_key_a.value}`.",
        f"- `{pairwise_alias_b}` scoring key: `{ui_pairwise_scoring_key_b.value}`.",
        "- Transition deltas are aligned by semantic state label.",
    ]
    if pairwise_missing_a:
        _notes.append(
            f"- Missing in `{pairwise_alias_a}`: {', '.join(map(str, pairwise_missing_a[:8]))}"
            + (" ..." if len(pairwise_missing_a) > 8 else "")
        )
    if pairwise_missing_b:
        _notes.append(
            f"- Missing in `{pairwise_alias_b}`: {', '.join(map(str, pairwise_missing_b[:8]))}"
            + (" ..." if len(pairwise_missing_b) > 8 else "")
        )
    if pairwise_fit_error_a:
        _notes.append(f"- State-level cache for A could not be loaded: `{pairwise_fit_error_a}`")
    if pairwise_fit_error_b:
        _notes.append(f"- State-level cache for B could not be loaded: `{pairwise_fit_error_b}`")
    mo.md("\n".join(_notes))
    return


@app.cell
def _(pairwise_metrics_b):
    pairwise_metrics_b
    return


@app.cell
def _(
    pairwise_K_a,
    pairwise_K_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_kind_a,
    pairwise_kind_b,
    pairwise_metrics_a,
    pairwise_metrics_b,
    pl,
):
    _frames = []
    if not pairwise_metrics_a.is_empty():
        _frames.append(
            pairwise_metrics_a.with_columns(pl.lit("A").alias("model_slot"))
        )
    if not pairwise_metrics_b.is_empty():
        _frames.append(
            pairwise_metrics_b.with_columns(pl.lit("B").alias("model_slot"))
        )

    if _frames:
        pairwise_metrics = pl.concat(_frames, how="diagonal")
    else:
        pairwise_metrics = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "K": pl.Int64,
                "model_kind": pl.Utf8,
                "model_alias": pl.Utf8,
                "model_label": pl.Utf8,
                "ll_per_trial": pl.Float64,
                "test_ll_per_trial": pl.Float64,
                "bic": pl.Float64,
                "acc": pl.Float64,
                "model_slot": pl.Utf8,
            }
        )

    pairwise_metric_summary = (
        pairwise_metrics
        .group_by(["model_slot", "model_kind", "model_alias", "K"])
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("test_ll_per_trial").alias("test_ll_mean"),
                pl.mean("bic").alias("bic_mean"),
                pl.mean("acc").alias("acc_mean"),
            ]
        )
        .sort("model_slot")
    )

    pairwise_metric_deltas = (
        pairwise_metrics_a.select(
            [
                "subject",
                pl.col("test_ll_per_trial").alias("ll_a"),
                pl.col("bic").alias("bic_a"),
                pl.col("acc").alias("acc_a"),
            ]
        )
        .join(
            pairwise_metrics_b.select(
                [
                    "subject",
                    pl.col("test_ll_per_trial").alias("ll_b"),
                    pl.col("bic").alias("bic_b"),
                    pl.col("acc").alias("acc_b"),
                ]
            ),
            on="subject",
            how="inner",
        )
        .with_columns(
            [
                pl.lit(pairwise_kind_a).alias("model_kind_a"),
                pl.lit(pairwise_kind_b).alias("model_kind_b"),
                pl.lit(pairwise_alias_a).alias("model_a"),
                pl.lit(pairwise_alias_b).alias("model_b"),
                pl.lit(pairwise_K_a).alias("K_a"),
                pl.lit(pairwise_K_b).alias("K_b"),
                (pl.col("ll_b") - pl.col("ll_a")).alias("delta_ll_per_trial"),
                (pl.col("bic_b") - pl.col("bic_a")).alias("delta_bic"),
            ]
        )
        .sort("subject")
    )

    pairwise_metric_delta_summary = (
        pairwise_metric_deltas
        .select(["delta_ll_per_trial", "delta_bic"])
        .mean()
        .with_columns(
            [
                pl.lit(pairwise_kind_b).alias("model_kind_b"),
                pl.lit(pairwise_alias_b).alias("model_b"),
                pl.lit(pairwise_K_b).alias("K_b"),
                pl.lit(pairwise_kind_a).alias("model_kind_a"),
                pl.lit(pairwise_alias_a).alias("model_a"),
                pl.lit(pairwise_K_a).alias("K_a"),
            ]
        )
        .select([
            "model_kind_b",
            "model_b",
            "K_b",
            "model_kind_a",
            "model_a",
            "K_a",
            "delta_ll_per_trial",
            "delta_bic",
        ])
    )
    pairwise_metrics
    return (pairwise_metric_deltas,)


@app.cell
def _(
    fig_size,
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metric_deltas,
    panel,
    plt,
    pretty_names,
    sns,
):
    mo.stop(
        pairwise_metric_deltas.is_empty(),
        mo.md("No common subject metrics were found for the selected model fits."),
    )

    pairwise_ll_scatter = pairwise_metric_deltas.to_pandas()

    fig_pairwise_ll, ax_pairwise_ll = plt.subplots(figsize = fig_size(2,1))

    sns.scatterplot(
        data=pairwise_ll_scatter,
        x="ll_a",
        y="ll_b",
        ax=ax_pairwise_ll,
    )

    pairwise_ll_xlim = ax_pairwise_ll.get_xlim()
    pairwise_ll_ylim = ax_pairwise_ll.get_ylim()
    pairwise_ll_min = min(*pairwise_ll_xlim, *pairwise_ll_ylim)
    pairwise_ll_max = max(*pairwise_ll_xlim, *pairwise_ll_ylim)

    ax_pairwise_ll.plot(
        [pairwise_ll_min, pairwise_ll_max],
        [pairwise_ll_min, pairwise_ll_max],
        linestyle="--",
        color=ax_pairwise_ll.spines["left"].get_edgecolor(),
        linewidth=ax_pairwise_ll.spines["left"].get_linewidth(),
    )
    ax_pairwise_ll.set_xlim(pairwise_ll_min, pairwise_ll_max)
    ax_pairwise_ll.set_ylim(pairwise_ll_min, pairwise_ll_max)
    ax_pairwise_ll.set_aspect("equal", adjustable="box")
    ax_pairwise_ll.set_xlabel(f"Model A: {pretty_names[pairwise_alias_a]}")
    ax_pairwise_ll.set_ylabel(f"Model B: {pretty_names[pairwise_alias_b]}")
    sns.despine(ax=ax_pairwise_ll)

    panel(
        "Pairwise test LL",
        fig_pairwise_ll,
        stem=f"pairwise_test_ll_scatter_{pairwise_alias_a}_{pairwise_alias_b}",
        description="Pairwise model A vs model B test LL scatter",
    )
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    fig_size,
    mo,
    np,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metric_deltas,
    plt,
    save_plot,
    sns,
    ttest_1samp,
):
    mo.stop(
        pairwise_metric_deltas.is_empty(),
        mo.md("No common subject metrics were found for the selected model fits."),
    )

    pretty_names = {
        "one hot2 lapses":  "lapse model",
        "param frozen": "pure GLM-HMM",
        "one hot": "GLM",
        "param2": "GLM-HMM",
        "mohammadi at 2states dif" : "GLM-HMMt",
        "mohammadi at 2states dif frozen" : "GLM-HMMt pure",
        "2afc_bias-stimD0-choice-lag-paramE0": "Both fixed",
        "2afc_bias-stimD0-choice-lag-param_free": "Stim fixed",
        "param": "Carles"
    }

    _pd = pairwise_metric_deltas.to_pandas()
    _panels = [
        ("delta_ll_per_trial", "Δ test LL / trial", (-0.05, 0.05)),
        ("delta_bic", "Δ BIC", (-800, 800)),
    ]

    def p_label(_values):
        _values = _values[np.isfinite(_values)]
        if len(_values) < 2:
            return ""
        if np.allclose(_values, 0):
            return "ns"
        _, _pval = ttest_1samp(_values, popmean=0.0, nan_policy="omit")
        if not np.isfinite(_pval):
            return ""
        if _pval < 0.0001:
            return "****"
        if _pval < 0.001:
            return "***"
        if _pval < 0.01:
            return "**"
        if _pval < 0.05:
            return "*"
        return "ns"

    _fig_pair_metrics, _axd = plt.subplot_mosaic(
        [["ll", "bic"]],
        figsize=fig_size(1,2),
        constrained_layout=True,
    )
    _axes = [_axd["ll"], _axd["bic"]]
    for _ax, (_delta_col, _ylabel, _ylim) in zip(_axes, _panels, strict=False):
        _delta = _pd[_delta_col].to_numpy(dtype=float)
        _finite = _delta[np.isfinite(_delta)]
        _ax.axhline(0, color="0.35", linewidth=1.0, linestyle="--", alpha=0.75, zorder=0)
        if len(_finite) > 0:
            sns.boxplot(
                y=_finite,
                ax=_ax,
                color = "tab:red",
                # width=.5,
                **BOXPLOT_STYLE
            )
        _p_txt = p_label(_finite)
        # _ax.set_ylim(*_ylim)
        if _p_txt:
            _ax.text(0.5, 0.88, _p_txt, ha="center", va="bottom", transform=_ax.transAxes)
        _ax.set_xticks([0])
        _ax.set_xticklabels([
            f"{pretty_names.get(pairwise_alias_b, pairwise_alias_b)} $-$ {pretty_names.get(pairwise_alias_a, pairwise_alias_a)}"
        ])
        _ax.set_ylabel(_ylabel)
        # _ax.set_xlim(-0.45, 0.45)
        sns.despine(ax=_ax)

    mo.vstack([_fig_pair_metrics, save_plot(_fig_pair_metrics, "Metric comparisom", stem=f"metric_comparison_{pairwise_alias_a}_{pairwise_alias_b}"),])
    return (pretty_names,)


@app.cell
def _(np, plt, sns):
    def _resolve_transition_matrix(arrays: dict) -> np.ndarray | None:
        if "transition_matrix" in arrays:
            return np.asarray(arrays["transition_matrix"], dtype=float)
        if "transition_bias" in arrays:
            _bias = np.asarray(arrays["transition_bias"], dtype=float)
            _exp = np.exp(_bias - _bias.max(axis=-1, keepdims=True))
            return _exp / _exp.sum(axis=-1, keepdims=True)
        return None

    def _reindex_transition_matrix(
        matrix: np.ndarray,
        source_labels: list[str],
        target_labels: list[str],
    ) -> np.ndarray:
        _source_index = {label: idx for idx, label in enumerate(source_labels)}
        _aligned = np.full((len(target_labels), len(target_labels)), np.nan, dtype=float)
        for _row_idx, _row_label in enumerate(target_labels):
            _src_row = _source_index.get(_row_label)
            if _src_row is None:
                continue
            for _col_idx, _col_label in enumerate(target_labels):
                _src_col = _source_index.get(_col_label)
                if _src_col is None:
                    continue
                _aligned[_row_idx, _col_idx] = matrix[_src_row, _src_col]
        return _aligned

    def _finite_max_abs(matrix: np.ndarray) -> float:
        _finite = matrix[np.isfinite(matrix)]
        if _finite.size == 0:
            return 1e-12
        return max(float(np.max(np.abs(_finite))), 1e-12)

    def plot_pairwise_transition_matrices(
        *,
        arrays_a: dict,
        arrays_b: dict,
        views_a: dict,
        views_b: dict,
        subjects: list,
        alias_a: str,
        alias_b: str,
    ) -> plt.Figure:
        def _mean_transition(arrays_store: dict, views: dict):
            _subject_entries = []
            _labels = []
            for _subject in subjects:
                if _subject not in views:
                    continue
                _mat = _resolve_transition_matrix(arrays_store.get(_subject, {}))
                if _mat is None:
                    continue
                _order = [int(k) for k in views[_subject].state_idx_order]
                _subject_labels = [
                    views[_subject].state_name_by_idx.get(int(k), f"State {k}")
                    for k in _order
                ]
                for _label in _subject_labels:
                    if _label not in _labels:
                        _labels.append(_label)
                _subject_entries.append(
                    (
                        _mat[np.ix_(_order, _order)],
                        _subject_labels,
                    )
                )
            if not _subject_entries:
                return None, []
            _aligned_mats = [
                _reindex_transition_matrix(_mat, _subject_labels, _labels)
                for _mat, _subject_labels in _subject_entries
            ]
            return np.nanmean(np.stack(_aligned_mats, axis=0), axis=0), _labels

        _mat_a, _labels_a = _mean_transition(arrays_a, views_a)
        _mat_b, _labels_b = _mean_transition(arrays_b, views_b)
        if _mat_a is None or _mat_b is None:
            raise ValueError("No transition matrices were available for the common subject set.")

        _labels = []
        for _label in _labels_a + _labels_b:
            if _label not in _labels:
                _labels.append(_label)
        _mat_a = _reindex_transition_matrix(_mat_a, _labels_a, _labels)
        _mat_b = _reindex_transition_matrix(_mat_b, _labels_b, _labels)

        _vmax = max(_finite_max_abs(_mat_a), _finite_max_abs(_mat_b))
        _delta = _mat_b - _mat_a
        _dmax = _finite_max_abs(_delta)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=False)
        for _ax, _mat, _title in [
            (axes[0], _mat_a, alias_a),
            (axes[1], _mat_b, alias_b),
        ]:
            sns.heatmap(
                _mat,
                ax=_ax,
                cmap="Blues",
                vmin=0,
                vmax=_vmax,
                annot=True,
                fmt=".2f",
                square=True,
                cbar=False,
            )
            _ax.set_title(_title)
            _ax.set_xticklabels(_labels, rotation=25, ha="right")
            _ax.set_yticklabels(_labels, rotation=0)
            _ax.set_xlabel("To state")
            _ax.set_ylabel("From state")

        sns.heatmap(
            _delta,
            ax=axes[2],
            cmap="RdBu_r",
            center=0,
            vmin=-_dmax,
            vmax=_dmax,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=False,
        )
        axes[2].set_title(f"{alias_b} - {alias_a}")
        axes[2].set_xticklabels(_labels, rotation=25, ha="right")
        axes[2].set_yticklabels(_labels, rotation=0)
        axes[2].set_xlabel("To state")
        axes[2].set_ylabel("From state")
        fig.suptitle(
            f"Mean transition matrices aligned by semantic state label  (n={len(subjects)} subjects)"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    return (plot_pairwise_transition_matrices,)


@app.cell
def _(
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_arrays_a,
    pairwise_arrays_b,
    pairwise_cached_common_subjects,
    pairwise_views_a,
    pairwise_views_b,
    plot_pairwise_transition_matrices,
):
    try:
        _fig_transition_pair = plot_pairwise_transition_matrices(
            arrays_a=pairwise_arrays_a,
            arrays_b=pairwise_arrays_b,
            views_a=pairwise_views_a,
            views_b=pairwise_views_b,
            subjects=pairwise_cached_common_subjects,
            alias_a=pairwise_alias_a,
            alias_b=pairwise_alias_b,
        )
        _transition_output = mo.vstack([mo.md("#### Transition matrices"), _fig_transition_pair])
    except Exception as _e:
        _transition_output = mo.md(f"#### Transition matrices\n\nCould not render the pairwise transition comparison: `{_e}`")
    _transition_output
    return


@app.cell
def _(
    build_emission_weights_df,
    mo,
    pairwise_K_a,
    pairwise_K_b,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_arrays_a,
    pairwise_arrays_b,
    pairwise_names_a,
    pairwise_names_b,
    pairwise_views_a,
    pairwise_views_b,
):
    def _emission_summary(_adapter, _views, _arrays_store, _names, _K):
        if _adapter is None:
            raise ValueError("fit bundle was not loaded")
        _plots = _adapter.get_plots()
        return _plots.plot_emission_weights_summary(
            build_emission_weights_df(_views),
            K=_K,
        )

    try:
        _fig_emission_a = _emission_summary(
            pairwise_adapter_a,
            pairwise_views_a,
            pairwise_arrays_a,
            pairwise_names_a,
            pairwise_K_a,
        )
        _fig_emission_b = _emission_summary(
            pairwise_adapter_b,
            pairwise_views_b,
            pairwise_arrays_b,
            pairwise_names_b,
            pairwise_K_b,
        )
        _emission_output = mo.vstack(
            [
                mo.md("#### Emission weights"),
                mo.hstack(
                    [
                        mo.vstack([mo.md(f"**A** — `{pairwise_alias_a}`"), _fig_emission_a]),
                        mo.vstack([mo.md(f"**B** — `{pairwise_alias_b}`"), _fig_emission_b]),
                    ],
                    widths="equal",
                ),
            ]
        )
    except Exception as _e:
        _emission_output = mo.md(f"#### Emission weights\n\nCould not render the pairwise emission summaries: `{_e}`")
    _emission_output
    return


@app.cell
def _(pl):
    def subject_behavior_df(df_all, *, subject, sort_col, session_col):
        df_sub = df_all.filter(pl.col("subject") == subject).sort(sort_col)
        if session_col in df_sub.columns:
            df_sub = df_sub.filter(
                pl.col(session_col).count().over(session_col) >= 2
            )
        return df_sub

    return (subject_behavior_df,)


@app.cell
def _(adapter_behavioral_column, df_all, pd):
    RT_METRIC_CANDIDATES = (
        "RT",
        "RT2",
        "response_time_first",
        "reaction_time",
        "ReactionTime",
        "timepoint_4",
    )
    BEHAVIOR_METRIC_CANDIDATES = ("nLicks", *RT_METRIC_CANDIDATES)

    def augment_behavior_metrics(trial_df, adapter):
        if trial_df.is_empty():
            return pd.DataFrame()

        out = trial_df.to_pandas()
        missing_cols = [
            col
            for col in BEHAVIOR_METRIC_CANDIDATES
            if col not in out.columns and col in df_all.columns
        ]
        if adapter is None or not missing_cols:
            return out

        session_col = adapter_behavioral_column(adapter, df_all, "session", "session", "Session")
        trial_col = adapter_behavioral_column(adapter, df_all, "trial_idx", "trial_idx", "trial", "Trial")
        if session_col is None or trial_col is None or not {"subject", "session", "trial_idx"}.issubset(out.columns):
            return out

        behavior_df = (
            df_all
            .select(["subject", session_col, trial_col, *missing_cols])
            .rename({session_col: "session", trial_col: "trial_idx"})
            .to_pandas()
        )
        return out.merge(behavior_df, on=["subject", "session", "trial_idx"], how="left")

    return RT_METRIC_CANDIDATES, augment_behavior_metrics


@app.cell
def _(
    build_trial_df,
    df_all,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_cached_common_subjects,
    pairwise_views_a,
    pairwise_views_b,
    pl,
    subject_behavior_df,
):
    def _pairwise_trial_df(adapter, views):
        if adapter is None:
            return pl.DataFrame()
        _frames = []
        for _subject in pairwise_cached_common_subjects:
            if _subject not in views:
                continue
            _df_sub = subject_behavior_df(
                df_all,
                subject=_subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
            )
            if _df_sub.height != views[_subject].T:
                continue
            try:
                _frames.append(
                    build_trial_df(
                        views[_subject],
                        adapter,
                        _df_sub,
                        adapter.behavioral_cols,
                    )
                )
            except Exception:
                pass
        if not _frames:
            return pl.DataFrame()
        return pl.concat(_frames, how="diagonal")

    pairwise_trial_df_a = _pairwise_trial_df(pairwise_adapter_a, pairwise_views_a)
    pairwise_trial_df_b = _pairwise_trial_df(pairwise_adapter_b, pairwise_views_b)
    return pairwise_trial_df_a, pairwise_trial_df_b


@app.cell
def _(
    df_all,
    np,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_cached_common_subjects,
    pairwise_views_a,
    pairwise_views_b,
    pl,
    subject_behavior_df,
):
    def _session_occupancy_records(*, alias: str, adapter, views: dict):
        if adapter is None:
            return []
        _records = []
        for _subject in pairwise_cached_common_subjects:
            if _subject not in views:
                continue
            _view = views[_subject]
            _df_sub = subject_behavior_df(
                df_all,
                subject=_subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
            )
            if _df_sub.height != _view.T:
                continue
            if adapter.session_col not in _df_sub.columns:
                continue
            _session_col = adapter.session_col
            _sessions = np.asarray(_df_sub[_session_col])
            _probs = np.asarray(_view.smoothed_probs, dtype=float)
            for _session in np.unique(_sessions):
                _mask = _sessions == _session
                if not np.any(_mask):
                    continue
                for _state_idx in _view.state_idx_order:
                    _records.append(
                        {
                            "subject": str(_subject),
                            "session": str(_session),
                            "model_alias": alias,
                            "state_label": _view.state_name_by_idx.get(
                                int(_state_idx), f"State {_state_idx}"
                            ),
                            "occupancy": float(np.mean(_probs[_mask, int(_state_idx)])),
                        }
                    )
        return _records

    _records = _session_occupancy_records(
        alias=pairwise_alias_a,
        adapter=pairwise_adapter_a,
        views=pairwise_views_a,
    )
    _records += _session_occupancy_records(
        alias=pairwise_alias_b,
        adapter=pairwise_adapter_b,
        views=pairwise_views_b,
    )

    if _records:
        pairwise_session_occupancy = pl.DataFrame(_records)
    else:
        pairwise_session_occupancy = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "session": pl.Utf8,
                "model_alias": pl.Utf8,
                "state_label": pl.Utf8,
                "occupancy": pl.Float64,
            }
        )

    pairwise_subject_occupancy = (
        pairwise_session_occupancy
        .group_by(["subject", "model_alias", "state_label"])
        .agg(pl.mean("occupancy").alias("occupancy"))
        .sort(["state_label", "model_alias", "subject"])
    )

    pairwise_occupancy_summary = (
        pairwise_subject_occupancy
        .group_by(["model_alias", "state_label"])
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("occupancy").alias("occupancy_mean"),
                pl.std("occupancy").alias("occupancy_std"),
            ]
        )
        .with_columns(
            (pl.col("occupancy_std") / pl.col("n_subjects").sqrt()).alias("occupancy_sem")
        )
        .sort(["state_label", "model_alias"])
    )
    return pairwise_session_occupancy, pairwise_subject_occupancy


@app.cell
def _(
    BOXPLOT_STLYE,
    fig_size,
    mo,
    np,
    pairwise_K_a,
    pairwise_K_b,
    pairwise_adapter_a,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_session_occupancy,
    pairwise_subject_occupancy,
    pairwise_trial_df_a,
    pairwise_trial_df_b,
    pl,
    plt,
    sns,
    ttest_rel,
):
    mo.stop(pairwise_session_occupancy.is_empty(), mo.md("#### No session occupancy for current subset."))

    def subject_accuracy(alias, df):
        if df.is_empty() or "state_label" not in df.columns:
            return pl.DataFrame()
        if "correct_bool" not in df.columns:
            df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
        return (
            df.drop_nulls(["state_label", "correct_bool"])
            .group_by("subject", "state_label")
            .agg((pl.mean("correct_bool") * 100).alias("value"))
            .with_columns(pl.lit(alias).alias("model_alias"), pl.lit("Accuracy (%)").alias("metric"))
        )

    occ = pairwise_subject_occupancy.rename({"occupancy": "value"}).with_columns(
        pl.lit("Occupancy").alias("metric")
    )
    acc = pl.concat([
        subject_accuracy(pairwise_alias_a, pairwise_trial_df_a),
        subject_accuracy(pairwise_alias_b, pairwise_trial_df_b),
    ], how="diagonal")

    plot_df = pl.concat([occ, acc], how="diagonal").to_pandas()

    models = [pairwise_alias_a, pairwise_alias_b]
    palette2 = {pairwise_alias_a: "#1B6CA8", pairwise_alias_b: "#C76D3A"}
    fig, axs = plt.subplot_mosaic([["occ", "acc"]], figsize=fig_size(1, 2))

    for key, metric, ylim, chance in [
        ("occ", "Occupancy", (0, 1), 1 / pairwise_K_a if pairwise_K_a == pairwise_K_b else None),
        ("acc", "Accuracy (%)", (0, 100), 100 / pairwise_adapter_a.num_classes if pairwise_adapter_a else None),
    ]:
        _ax = axs[key]
        df = plot_df[plot_df["metric"].eq(metric)]
        order = sorted(df["state_label"].dropna().unique())

        if chance is not None:
            _ax.axhline(chance, ls="--", lw=1, c="gray")

        sns.boxplot(
            data=df, x="state_label", y="value", hue="model_alias",
            order=order, hue_order=models, palette=palette2,
            width=.55, ax=_ax, **BOXPLOT_STLYE
        )

        for i, state in enumerate(order):
            wide = (
                df[df["state_label"].eq(state)]
                .pivot(index="subject", columns="model_alias", values="value")
                .dropna()
            )
            for _, row in wide.iterrows():
                _ax.plot([i - .2, i + .2], [row[models[0]], row[models[1]]], alpha = 0.15, color = "tab:gray")

            if len(wide) >= 2 and not np.allclose(wide[models[0]], wide[models[1]]):
                p = ttest_rel(wide[models[0]], wide[models[1]], nan_policy="omit").pvalue
                stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                if stars:
                    y = wide.max().max() + .05 * (ylim[1] - ylim[0])
                    _ax.plot([i - .2, i + .2], [y, y], c="k", lw=1)
                    _ax.text(i, y, stars, ha="center", va="bottom")

        _ax.set(xlabel="", ylabel=metric, ylim=ylim)
        _ax.tick_params(axis="x")
        sns.despine(ax=_ax)

    handles, labels = axs["occ"].get_legend_handles_labels()
    for _ax in axs.values():
        _ax.legend_.remove()
    fig.legend(handles[:2], labels[:2], title="Model", loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))

    fig
    return


@app.cell
def _(
    RT_METRIC_CANDIDATES,
    augment_behavior_metrics,
    fig_size,
    mo,
    n_shuffles,
    np,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_trial_df_a,
    pairwise_trial_df_b,
    pd,
    plt,
    pretty_names,
    save_plot,
    sns,
):
    _df_a = augment_behavior_metrics(pairwise_trial_df_a, pairwise_adapter_a)
    _df_b = augment_behavior_metrics(pairwise_trial_df_b, pairwise_adapter_b)
    if not _df_a.empty:
        _df_a = _df_a.assign(
            model_slot="A", 
            model_name=f"{pretty_names.get(pairwise_alias_a,pairwise_alias_a)}",
        )
    if not _df_b.empty:
        _df_b = _df_b.assign(
            model_slot="B",
            model_name=f"{pretty_names.get(pairwise_alias_b,pairwise_alias_b)}",
        )
    _frames = [_df for _df in [_df_a, _df_b] if not _df.empty]
    mo.stop(not _frames, mo.md("No pairwise trial data were available for behavioral ROC curves."))
    roc_df = pd.concat(_frames, ignore_index=True)

    state_col = next(
        (_col for _col in ["state_label", "state_label_pred"] if _col in roc_df.columns),
        None,
    )
    mo.stop(state_col is None, mo.md("State labels are not available for pairwise behavioral ROC curves."))

    _rt_col = next(
        (_col for _col in RT_METRIC_CANDIDATES if _col in roc_df.columns),
        None,
    )
    metric_specs = [("nLicks", "Licking", "Higher lick count")]
    if _rt_col is not None:
        metric_specs.append((_rt_col, "RT", "Faster RT"))
    metric_specs.append(("ILI", "ILI", "Faster ILI"))
    metric_specs = [_spec for _spec in metric_specs if _spec[0] in roc_df.columns]
    mo.stop(not metric_specs, mo.md("No `nLicks` or RT column was found for pairwise behavioral ROC curves."))

    def binary_engaged_target(_labels):
        _label_text = pd.Series(_labels, copy=False).astype(str).str.strip().str.lower()
        _positive = _label_text.eq("engaged") | _label_text.str.startswith("engaged ")
        _negative = _label_text.eq("disengaged") | _label_text.str.startswith("disengaged ")
        return _positive.to_numpy(dtype=bool), (_positive | _negative).to_numpy(dtype=bool)

    def roc_curve(_target, _score):
        _target = np.asarray(_target, dtype=bool)
        _score = np.asarray(_score, dtype=float)
        _valid = np.isfinite(_score)
        _target = _target[_valid]
        _score = _score[_valid]
        _n_pos = int(_target.sum())
        _n_neg = int((~_target).sum())
        if _target.size == 0 or _n_pos == 0 or _n_neg == 0:
            return None
        _order = np.argsort(-_score, kind="mergesort")
        _target_sorted = _target[_order]
        _score_sorted = _score[_order]
        _threshold_idxs = np.r_[np.where(np.diff(_score_sorted))[0], _target_sorted.size - 1]
        _tps = np.cumsum(_target_sorted)[_threshold_idxs]
        _fps = (1 + _threshold_idxs) - _tps
        _tpr = np.r_[0.0, _tps / _n_pos]
        _fpr = np.r_[0.0, _fps / _n_neg]
        _auc = float(np.sum(np.diff(_fpr) * (_tpr[:-1] + _tpr[1:]) / 2.0))
        return _fpr, _tpr, _auc

    # def _shuffled_auc(_target, _score, _rng, n_shuffles=200):
        _aucs = []
        for _ in range(n_shuffles):
            _result = roc_curve(_rng.permutation(_target), _score)
            if _result is not None:
                _aucs.append(_result[2])
        return float(np.mean(_aucs)) if _aucs else np.nan

    _palette = {
        "A": "tab:blue",
        "B": "tab:orange",
    }
    _rng = np.random.default_rng(0)
    _fig_roc, _axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=fig_size(1,2),

    )
    _plotted = False
    for _ax, (_metric_col, _title, _direction_label) in zip(_axes.ravel(), metric_specs, strict=False):
        for _slot, _slot_df in roc_df.groupby("model_slot", sort=True):
            _target, _valid_labels = binary_engaged_target(_slot_df[state_col])
            _score = pd.to_numeric(_slot_df[_metric_col], errors="coerce").to_numpy(dtype=float)
            if _metric_col != "nLicks":
                _score = -_score
            _target = _target[_valid_labels]
            _score = _score[_valid_labels]
            _result = roc_curve(_target, _score)
            if _result is None:
                continue
            _fpr, _tpr, _auc = _result
            _name = str(_slot_df["model_name"].iloc[0])
            _ax.plot(
                _fpr,
                _tpr,
                color=_palette.get(_slot, "tab:gray"),
                lw=2,
                label=f"{_name} (AUC={_auc:.3f})",
            )
            _plotted = True
        _ax.plot([0, 1], [0, 1], color="tab:gray", lw=1, ls="--")
        _ax.set_title(_title)
        _ax.set_xlabel("False positive rate")
        _ax.set_ylabel("True positive rate")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.legend(frameon=False, loc="lower right")
        sns.despine(ax=_ax)

    mo.stop(not _plotted, mo.md("Pairwise behavioral ROC curves require Engaged and Disengaged trials."))
    mo.vstack(
        [
            mo.md("#### Pairwise behavioral ROC by state"),
            _fig_roc,
            save_plot(
                _fig_roc,
                "pairwise behavioral ROC by state",
                stem="pairwise_state_behavioral_roc",
            ),
        ],
        align="center",
    )
    return binary_engaged_target, metric_specs, roc_curve, roc_df, state_col


@app.cell
def _(
    binary_engaged_target,
    metric_specs,
    mo,
    np,
    pd,
    roc_curve,
    roc_df,
    state_col,
):
    _subject_col = "subject"
    mo.stop(_subject_col not in roc_df.columns, mo.md("No subject column was found."))

    _fpr_grid = np.linspace(0, 1, 101)

    _curve_rows = []
    _auc_rows = []

    for (_slot, _subject), _subj_df in roc_df.groupby(["model_slot", _subject_col], sort=True):
        for _metric_col, _title, _direction_label in metric_specs:
            _target, _valid_labels = binary_engaged_target(_subj_df[state_col])
            _score = pd.to_numeric(_subj_df[_metric_col], errors="coerce").to_numpy(dtype=float)

            if _metric_col != "nLicks":
                _score = -_score

            _target = _target[_valid_labels]
            _score = _score[_valid_labels]

            _result = roc_curve(_target, _score)
            if _result is None:
                continue

            _fpr, _tpr, _auc = _result
            _interp_tpr = np.interp(_fpr_grid, _fpr, _tpr)
            _interp_tpr[0] = 0.0
            _interp_tpr[-1] = 1.0

            _name = str(_subj_df["model_name"].iloc[0])

            _auc_rows.append(
                {
                    "model_slot": _slot,
                    "model_name": _name,
                    "subject": _subject,
                    "metric": _metric_col,
                    "metric_label": _title,
                    "auc": _auc,
                }
            )

            for _fpr_value, _tpr_value in zip(_fpr_grid, _interp_tpr):
                _curve_rows.append(
                    {
                        "model_slot": _slot,
                        "model_name": _name,
                        "subject": _subject,
                        "metric": _metric_col,
                        "metric_label": _title,
                        "fpr": _fpr_value,
                        "tpr": _tpr_value,
                    }
                )

    auc_df = pd.DataFrame(_auc_rows)
    curve_df = pd.DataFrame(_curve_rows)

    mo.stop(
        auc_df.empty or curve_df.empty,
        mo.md("Pairwise behavioral ROC curves require Engaged and Disengaged trials per subject."),
    )
    return auc_df, curve_df


@app.cell
def _(
    BOXPLOT_STYLE,
    auc_df,
    curve_df,
    fig_size,
    metric_specs,
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    plt,
    save_plot,
    sns,
):
    from statannotations.Annotator import Annotator
    _fig_roc, _axes_roc = plt.subplots(
        1,
        len(metric_specs),
        figsize=fig_size(1, len(metric_specs)),
        squeeze=False,
        layout="constrained",
    )
    _palette = {
        "A": "tab:blue",
        "B": "tab:orange",
    }
    for _ax, (_metric_col, _title, _direction_label) in zip(
        _axes_roc.ravel(), metric_specs, strict=False
    ):
        _metric_curve_df = curve_df[curve_df["metric"].eq(_metric_col)]
        _metric_auc_df = auc_df[auc_df["metric"].eq(_metric_col)]

        for _slot, _slot_curve_df in _metric_curve_df.groupby("model_slot", sort=True):
            _summary_df = (
                _slot_curve_df
                .groupby("fpr", as_index=False)
                .agg(
                    mean_tpr=("tpr", "mean"),
                    sem_tpr=("tpr", "sem"),
                )
            )

            _slot_auc_df = _metric_auc_df[_metric_auc_df["model_slot"].eq(_slot)]
            _name = str(_slot_auc_df["model_name"].iloc[0])
            _mean_auc = _slot_auc_df["auc"].mean()
            _sem_auc = _slot_auc_df["auc"].sem()

            _color = _palette.get(_slot, "tab:gray")

            _ax.plot(
                _summary_df["fpr"],
                _summary_df["mean_tpr"],
                color=_color,
                lw=2,
                label=f"{_name} AUC={_mean_auc:.3f} ± {_sem_auc:.3f}",
            )

            _ax.fill_between(
                _summary_df["fpr"],
                _summary_df["mean_tpr"] - _summary_df["sem_tpr"].fillna(0),
                _summary_df["mean_tpr"] + _summary_df["sem_tpr"].fillna(0),
                color=_color,
                alpha=0.2,
                linewidth=0,
            )

        _ax.plot([0, 1], [0, 1], color="tab:gray", lw=1, ls="--")
        _ax.set_title(_title)
        _ax.set_xlabel("False positive rate")
        _ax.set_ylabel("True positive rate")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.legend(frameon=False, loc="lower right")
        sns.despine(ax=_ax)

    _fig_auc, _axes_auc = plt.subplots(
        1,
        len(metric_specs),
        figsize=fig_size(1, len(metric_specs)),
        squeeze=False,
        layout="constrained",
    )

    for _ax, (_metric_col, _title, _direction_label) in zip(
        _axes_auc.ravel(), metric_specs, strict=False
    ):
        _metric_auc_df = auc_df[auc_df["metric"].eq(_metric_col)].copy()

        sns.boxplot(
            data=_metric_auc_df,
            x="model_name",
            y="auc",
            hue="model_slot",
            order=[
                auc_df[auc_df["model_slot"].eq("A")]["model_name"].iloc[0],
                auc_df[auc_df["model_slot"].eq("B")]["model_name"].iloc[0],
            ],
            hue_order=["A", "B"],
            palette=_palette,
            ax=_ax,
            zorder = 1,
            **BOXPLOT_STYLE
        )

        sns.stripplot(
            data=_metric_auc_df,
            x="model_name",
            y="auc",
            hue="model_slot",
            order=[
                auc_df[auc_df["model_slot"].eq("A")]["model_name"].iloc[0],
                auc_df[auc_df["model_slot"].eq("B")]["model_name"].iloc[0],
            ],
            hue_order=["A", "B"],
            palette=_palette,
            ax=_ax,
            dodge=False,
            alpha=0.2,
            size=0,
            legend=False,
            zorder = 0
        )

        _paired_auc_df = (
            _metric_auc_df
            .pivot_table(
                index="subject",
                columns="model_slot",
                values="auc",
                aggfunc="mean",
            )
            .dropna(subset=["A", "B"])
        )

        _model_name_a = _metric_auc_df[_metric_auc_df["model_slot"].eq("A")]["model_name"].iloc[0]
        _model_name_b = _metric_auc_df[_metric_auc_df["model_slot"].eq("B")]["model_name"].iloc[0]

        for _subject, _row in _paired_auc_df.iterrows():
            _ax.plot(
                [_model_name_a, _model_name_b],
                [_row["A"], _row["B"]],
                color="0.75",
                linewidth=0.5,
                zorder=0,
            )

        _pairs = [(_model_name_a, _model_name_b)]

        Annotator(
            _ax,
            _pairs,
            data=_metric_auc_df,
            x="model_name",
            y="auc",
            order=[_model_name_a, _model_name_b],
        ).configure(
            test="t-test_paired",
            text_format="star",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()

        _ax.axhline(0.5, color="tab:gray", lw=1, ls="--")
        _ax.set_title(f"{_title} AUC by subject")
        _ax.set_xlabel("")
        _ax.set_ylabel("AUC")
        _ax.set_ylim(0, 1)
        sns.despine(ax=_ax)
        _handles, _labels = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend_.remove()
    mo.vstack(
        [
            mo.md("#### Pairwise behavioral ROC by state, per subject"),
            _fig_roc,
            mo.md("#### Subject-level AUC distributions"),
            _fig_auc,
            mo.ui.table(auc_df, pagination=True),
            save_plot(
                _fig_roc,
                "pairwise behavioral ROC by state per subject",
                stem="pairwise_state_behavioral_roc_by_subject",
            ),
            save_plot(
                _fig_auc,
                "pairwise behavioral ROC AUC by subject",
                stem="pairwise_state_behavioral_roc_auc_by_subject",
            ),
        ],
        align="center",
    )

    mo.vstack([_fig_auc, save_plot(_fig_auc, "AUC comparisom", stem=f"auc_comparison_{pairwise_alias_a}_{pairwise_alias_b}"),])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
