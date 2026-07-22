import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from matplotlib.colors import to_hex, to_rgb
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
    from statannotations.Annotator import Annotator

    from glmhmmt.notebook_support.analysis_common import (
        load_fit_bundle as load_fit_bundle_raw,
        load_metrics_dir as load_metrics_dir_raw,
        model_aliases_for_kind,
    )
    from glmhmmt.plots_common import custom_boxplot
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df
    from glmhmmt.runtime import get_runtime_paths
    from glmhmmt.tasks import get_adapter, get_task_options
    from glmhmmt.views import build_views
    from plot_saver import make_plot_saver
    from src.plots.common import fig_size
    from src.process.common import adapter_behavioral_column

    paths = get_runtime_paths()
    return (
        Annotator,
        Line2D,
        adapter_behavioral_column,
        build_emission_weights_df,
        build_trial_df,
        build_views,
        custom_boxplot,
        fig_size,
        get_adapter,
        get_task_options,
        itertools,
        load_fit_bundle_raw,
        load_metrics_dir_raw,
        make_plot_saver,
        mo,
        model_aliases_for_kind,
        np,
        paths,
        pd,
        pl,
        plt,
        sns,
        to_hex,
        to_rgb,
        ttest_1samp,
        ttest_ind,
        ttest_rel,
    )


@app.cell
def _(np, plt, sns, to_hex, to_rgb):
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    MODEL_LABELS = {
        "glm": "GLM",
        "glmhmm": "GLMHMM",
        "glmhmmt": "GLMHMMT",
    }
    MODEL_ORDER = ["glm", "glmhmmt", "glmhmm"]
    MODEL_STYLES = {
        "glm": {"label": "GLM", "marker": "s", "color": "#4C78A8"},
        "glmhmmt": {"label": "GLMHMM-T", "marker": "^", "color": "#54A24B"},
        "glmhmm": {"label": "GLMHMM", "marker": "o", "color": "#F58518"},
    }
    PAIRWISE_PRETTY_NAMES = {
        "one hot2 lapses": "lapse model",
        "param frozen": "pure GLM-HMM",
        "one hot": "GLM",
        "param2": "GLM-HMM",
        "mohammadi at 2states dif": "GLM-HMM-T",
        "mohammadi at 2states dif frozen": "pure GLM-HMM-T",
        "2afc_bias-stimD0-choice-lag-paramE0": "Both fixed",
        "2afc_bias-stimD0-choice-lag-param_free": "Stim fixed",
        "param": "Carles",
    }
    ROC_PALETTE = {"A": "tab:blue", "B": "tab:orange"}
    BOXPLOT_STYLE = {
        "fill": False,
        "boxprops": {"color": "0.5"},
        "whiskerprops": {"color": "0.5"},
        "medianprops": {"linewidth": 2},
        "showfliers": False,
        "showcaps": False,
    }

    def darken(color, factor=0.75):
        """Scale an RGB color by ``factor`` and return a hex color."""
        return to_hex(np.clip(np.array(to_rgb(color)) * factor, 0, 1))

    return (
        BOXPLOT_STYLE,
        MODEL_LABELS,
        MODEL_ORDER,
        MODEL_STYLES,
        PAIRWISE_PRETTY_NAMES,
        ROC_PALETTE,
        darken,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Model comparison

    Compare cross-validated model fit, latent-state structure, and the behavioral
    meaning of states for two selected fits.
    """)
    return


@app.cell
def _(mo, save_plot):
    def panel(title, fig=None, stem=None, description=None):
        """Display a titled figure and its plot-saver controls."""
        content = [mo.md(f"#### {title}")]

        if fig is not None:
            content.append(fig)
            if stem is not None:
                content.append(save_plot(fig, description or title.lower(), stem=stem))

        return mo.vstack(content, align="center")

    return (panel,)


@app.cell
def _(get_task_options, mo):
    ui_task = mo.ui.dropdown(
        options={option["label"]: option["value"] for option in get_task_options()},
        value="MCDR",
        label="Task",
    )
    return (ui_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select fits
    """)
    return


@app.cell
def _(MODEL_LABELS, load_metrics_dir_raw, model_aliases_for_kind, paths, pl):
    def model_aliases(task: str, kind: str) -> list[str]:
        """Return available fitted-model aliases for one task and model family."""
        return model_aliases_for_kind(
            task_name=task,
            model_kind=kind,
            local_root=paths.RESULTS / "fits" / task / kind,
        )

    def load_metrics_dir(task_name: str, folder_name: str | None, expected_model_kind: str):
        """Load one fit's subject metrics as a consistent Polars dataframe."""
        metrics_df = load_metrics_dir_raw(
            task_name=task_name,
            model_kind=expected_model_kind,
            alias=folder_name,
            local_root=paths.RESULTS / "fits" / task_name / expected_model_kind,
            label_map=MODEL_LABELS,
        )
        if metrics_df is None:
            return None
        if "test_ll_per_trial_mean" not in metrics_df.columns:
            metrics_df = metrics_df.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("test_ll_per_trial_mean")
            )
        metrics_df = metrics_df.with_columns(
            pl.col("test_ll_per_trial_mean").alias("test_ll_per_trial"),
            pl.lit(expected_model_kind).alias("model_kind"),
        )
        metric_columns = [
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
        return metrics_df.select(
            [column for column in metric_columns if column in metrics_df.columns]
        )

    return load_metrics_dir, model_aliases


@app.cell
def _(make_plot_saver, mo, paths, ui_task):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=ui_task.value,
        model_id="model_comparison",
    )
    return (save_plot,)


@app.cell
def _(build_views, get_adapter, load_fit_bundle_raw, paths):
    def load_fit_bundle(task_name, model_kind, alias, K, subjects, scoring_key=None):
        """Load cached arrays and semantic state views for one fitted model."""
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
    all_subjects = df_all["subject"].unique().sort().to_list()

    ui_subjects = mo.ui.multiselect(
        options=all_subjects,
        value=all_subjects,
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
    selected_aliases_by_kind = [
        (ui_glm_dir.value, "glm"),
        (ui_glmhmm_dir.value, "glmhmm"),
        (ui_glmhmmt_dir.value, "glmhmmt"),
    ]
    metric_frames = [
        metrics_df
        for aliases, model_kind in selected_aliases_by_kind
        for alias in aliases
        if (metrics_df := load_metrics_dir(ui_task.value, alias, model_kind)) is not None
    ]

    if metric_frames:
        results_long = pl.concat(metric_frames, how="diagonal")
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
        f"**{len(metric_frames)}** model folder(s)."
    )
    return (results_long,)


@app.cell
def _(MODEL_LABELS, MODEL_ORDER, mo, pl, results_long):
    model_pick_elements = {}
    model_pick_metadata = {}

    for pick_model_kind in MODEL_ORDER:
        model_kind_df = results_long.filter(pl.col("model_kind") == pick_model_kind)
        for pick_state_count in model_kind_df["K"].drop_nulls().unique().sort().to_list():
            model_aliases_at_k = (
                model_kind_df
                .filter(pl.col("K") == pick_state_count)
                .select("model_alias")
                .unique()
                .sort("model_alias")
                .get_column("model_alias")
                .to_list()
            )
            if not model_aliases_at_k:
                continue
            pick_model_key = f"{pick_model_kind}:{int(pick_state_count)}"
            model_pick_metadata[pick_model_key] = {
                "model_kind": pick_model_kind,
                "model_kind_label": MODEL_LABELS[pick_model_kind],
                "K": int(pick_state_count),
            }
            model_pick_elements[pick_model_key] = mo.ui.dropdown(
                options={
                    "Skip": "__skip__",
                    **{alias: alias for alias in model_aliases_at_k},
                },
                value=model_aliases_at_k[0],
                label=f"{MODEL_LABELS[pick_model_kind]} K={int(pick_state_count)}",
            )

    ui_model_picks = mo.ui.dictionary(model_pick_elements, label="Model picks")
    model_pick_rows = [
        mo.hstack([
            mo.md(
                f"**{model_pick_metadata[pick_row_key]['model_kind_label']} "
                f"K={model_pick_metadata[pick_row_key]['K']}**"
            ),
            ui_model_picks[pick_row_key],
        ])
        for pick_row_key in model_pick_elements
    ]

    mo.vstack([
        mo.md("### One model per family and state count"),
        mo.md("Pick at most one alias for each model family and `K`. Use `Skip` to omit a point."),
        *(model_pick_rows if model_pick_rows else [mo.md("No model/K combinations available for the selected aliases.")]),
    ])
    return (ui_model_picks,)


@app.cell
def _(pl, ui_model_picks):
    selected_model_rows = []
    for selected_model_key, selected_model_alias in ui_model_picks.value.items():
        if selected_model_alias == "__skip__":
            continue
        selected_model_kind, selected_state_count = selected_model_key.split(":", 1)
        selected_model_rows.append(
            {
                "model_kind": selected_model_kind,
                "K": int(selected_state_count),
                "model_alias": selected_model_alias,
            }
        )

    selected_model_specs = pl.DataFrame(
        selected_model_rows,
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
    enum_dtype = getattr(pl, "Enum", None)
    if getattr(adapter, "num_classes", None) == 3:
        preferred_condition_columns = [
            "stimd_n",
            "stimd_c",
            "ttype_n",
            "ttype_c",
            "condition",
            "Condition",
            "Experiment",
            adapter.session_col,
        ]
        default_condition_candidates = ["stimd_n", "stimd_c", "ttype_n", "ttype_c"]
    else:
        preferred_condition_columns = [
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
        default_condition_candidates = ["ILD", "ild", "stim_vals", "stim_d", "stim_strength"]
    seen_condition_columns = set()
    condition_columns = []
    for column in preferred_condition_columns:
        if column in df_all.columns and column not in seen_condition_columns:
            condition_columns.append(column)
            seen_condition_columns.add(column)
    for column, dtype in df_all.schema.items():
        if column in seen_condition_columns or column == "subject":
            continue
        if dtype in tuple(
            candidate_dtype
            for candidate_dtype in (
                pl.Utf8, pl.Categorical, enum_dtype, pl.Boolean,
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            )
            if candidate_dtype is not None
        ):
            condition_columns.append(column)
            seen_condition_columns.add(column)

    default_condition = next(
        (column for column in default_condition_candidates if column in condition_columns),
        None,
    )
    if default_condition is None:
        default_condition = (
            "condition"
            if "condition" in condition_columns
            else (condition_columns[0] if condition_columns else None)
        )
    ui_ce_condition = mo.ui.dropdown(
        options=condition_columns,
        value=default_condition,
        label="Cross-entropy grouping",
    )
    mo.hstack([ui_ce_condition])
    return


@app.cell
def _(mo, results_filtered):
    bic_baseline_options = results_filtered["model_label"].unique().sort().to_list()
    bic_baseline_default = bic_baseline_options[0] if bic_baseline_options else None
    ui_bic_baseline = mo.ui.dropdown(
        options=bic_baseline_options,
        value=bic_baseline_default,
        label="BIC baseline model",
    )
    mo.hstack([ui_bic_baseline])
    return (ui_bic_baseline,)


@app.cell
def _(pl, results_filtered, ui_bic_baseline):
    if results_filtered.is_empty() or ui_bic_baseline.value is None:
        bic_baseline_by_subject = pl.DataFrame(
            schema={"subject": pl.Utf8, "bic_baseline": pl.Float64}
        )
        results_plot = results_filtered.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("bic_delta")
        )
    else:
        bic_baseline_by_subject = (
            results_filtered
            .filter(pl.col("model_label") == ui_bic_baseline.value)
            .group_by("subject")
            .agg(pl.first("bic").alias("bic_baseline"))
        )
        results_plot = (
            results_filtered
            .join(bic_baseline_by_subject, on="subject", how="left")
            .with_columns(
                ((pl.col("bic") - pl.col("bic_baseline")) / pl.col("bic_baseline"))
                .alias("bic_delta")
            )
        )
    return (results_plot,)


@app.cell
def _(mo, results_filtered):
    highlight_subject_options = results_filtered["subject"].unique().sort().to_list()
    ui_highlight_subject = mo.ui.dropdown(
        options={
            "None": "__none__",
            **{subject: subject for subject in highlight_subject_options},
        },
        value="None",
        label="Dashed subject",
    )
    mo.hstack([ui_highlight_subject])
    return (ui_highlight_subject,)


@app.cell
def _(pl, results_filtered):
    model_metric_summary = (
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare all selected models
    """)
    return


@app.cell
def _(mo, np, pl, results_filtered):
    mo.stop(results_filtered.is_empty(), mo.md("No selected model metrics to plot."))
    mo.stop(
        results_filtered.filter(pl.col("model_kind") == "glm").is_empty(),
        mo.md("Select a GLM model to use as the LL increment baseline."),
    )

    glm_ll_by_subject = (
        results_filtered
        .filter(pl.col("model_kind") == "glm")
        .sort(["subject", "K", "model_alias"])
        .group_by("subject")
        .agg(pl.first("test_ll_per_trial").alias("glm_test_ll_per_trial"))
    )

    ll_increment_df = (
        results_filtered
        .join(glm_ll_by_subject, on="subject", how="inner")
        .with_columns(
            ((pl.col("test_ll_per_trial") - pl.col("glm_test_ll_per_trial")) / np.log(2)).alias("test_ll_increment_bits")
        )
        .sort(["model_kind", "subject", "K"])
        .to_pandas()
    )
    ll_state_counts = sorted(ll_increment_df["K"].dropna().unique())
    return ll_increment_df, ll_state_counts


@app.cell
def _(
    Line2D,
    MODEL_ORDER,
    MODEL_STYLES,
    ll_increment_df,
    ll_state_counts,
    plt,
    sns,
    ui_highlight_subject,
):
    highlighted_subject = ui_highlight_subject.value
    fig_ll_bits, ax_ll_bits = plt.subplots(figsize=(7.2, 4.6))

    for ll_model_kind in MODEL_ORDER:
        model_df = ll_increment_df[ll_increment_df["model_kind"] == ll_model_kind]
        if model_df.empty:
            continue
        model_style = MODEL_STYLES[ll_model_kind]
        for subject, subject_df in model_df.groupby("subject"):
            subject_df = subject_df.sort_values("K")
            is_highlighted = (
                highlighted_subject != "__none__" and subject == highlighted_subject
            )
            ax_ll_bits.plot(
                subject_df["K"],
                subject_df["test_ll_increment_bits"],
                color=model_style["color"],
                linestyle="--" if is_highlighted else "-",
                linewidth=2.0 if is_highlighted else 0.8,
                alpha=0.9 if is_highlighted else 0.16,
                marker=model_style["marker"] if is_highlighted else None,
                markersize=4,
                zorder=3 if is_highlighted else 1,
            )

        model_mean_df = (
            model_df
            .groupby("K", as_index=False)["test_ll_increment_bits"]
            .mean()
            .sort_values("K")
        )
        ax_ll_bits.plot(
            model_mean_df["K"],
            model_mean_df["test_ll_increment_bits"],
            color=model_style["color"],
            marker=model_style["marker"],
            linewidth=2.6,
            markersize=5,
            label=f"{model_style['label']} mean",
            zorder=4,
        )

    ax_ll_bits.axhline(0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.6)
    ax_ll_bits.set(
        xlabel="Number of states K",
        ylabel="Test LL increment vs GLM (bits / trial)",
        title="Per-animal and mean test LL increment vs GLM",
        xticks=ll_state_counts,
    )

    ll_legend_handles, ll_legend_labels = ax_ll_bits.get_legend_handles_labels()
    if highlighted_subject != "__none__":
        ll_legend_handles.append(
            Line2D([0], [0], color="0.2", linestyle="--", linewidth=2)
        )
        ll_legend_labels.append(f"{highlighted_subject} dashed")
    ax_ll_bits.legend(ll_legend_handles, ll_legend_labels, frameon=False)
    sns.despine(ax=ax_ll_bits)
    fig_ll_bits.tight_layout()
    fig_ll_bits
    return


@app.cell
def _(itertools, pd, ttest_ind, ttest_rel):
    def significance_label(p_value):
        """Convert a p-value to the paper's significance-star notation."""
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    def add_sig_bars(ax, df, *, x_col, y_col, hue_col, order, hue_order, pair_col=None):
        """Annotate significant hue-level comparisons within each x level."""
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

                star = significance_label(pval)
                if star == "ns":
                    continue

                x1 = m + (p1 - (n_hue - 1) / 2) * hue_width
                x2 = m + (p2 - (n_hue - 1) / 2) * hue_width

                ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y], lw=1, c="k")
                ax.text((x1 + x2) / 2, current_y + h, star, ha="center", va="bottom", color="k")
                current_y += y_range * 0.075

    return (add_sig_bars,)


@app.cell
def _(darken, results_plot, sns, to_hex):
    comparison_df = results_plot.to_pandas()
    comparison_hue_order = (
        comparison_df[["model_kind", "model_label"]]
        .drop_duplicates()["model_label"]
        .tolist()
    )
    comparison_base_colors = sns.color_palette(
        "tab20", n_colors=max(1, len(comparison_hue_order))
    )
    comparison_palette = {
        label: to_hex(comparison_base_colors[index])
        for index, label in enumerate(comparison_hue_order)
    }
    comparison_strip_palette = {
        label: darken(comparison_palette[label], 0.70)
        for label in comparison_hue_order
    }
    comparison_K_order = sorted(comparison_df["K"].unique())
    return (
        comparison_K_order,
        comparison_df,
        comparison_hue_order,
        comparison_palette,
        comparison_strip_palette,
    )


@app.cell
def _(
    Line2D,
    add_sig_bars,
    comparison_K_order,
    comparison_df,
    comparison_hue_order,
    comparison_palette,
    comparison_strip_palette,
    custom_boxplot,
    plt,
    sns,
    ui_bic_baseline,
):
    fig_cmp, (ax_ll, ax_bic) = plt.subplots(1, 2, figsize=(8, 4.8), constrained_layout=False)

    def grouped_custom_boxplot(ax, y_column):
        """Draw grouped boxes using the project's custom boxplot helper."""
        hue_width = 0.8 / len(comparison_hue_order)
        grouped_values = []
        positions = []
        median_colors = []

        for K_index, K_value in enumerate(comparison_K_order):
            for hue_index, hue_label in enumerate(comparison_hue_order):
                values = comparison_df[
                    (comparison_df["K"] == K_value)
                    & (comparison_df["model_label"] == hue_label)
                ][y_column].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                positions.append(
                    K_index
                    + (hue_index - (len(comparison_hue_order) - 1) / 2) * hue_width
                )
                grouped_values.append(values)
                median_colors.append(comparison_palette[hue_label])

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

    for comparison_axis, y_column in [(ax_ll, "test_ll_per_trial"), (ax_bic, "bic_delta")]:
        grouped_custom_boxplot(comparison_axis, y_column)

        sns.stripplot(
            data=comparison_df,
            x="K",
            y=y_column,
            hue="model_label",
            order=comparison_K_order,
            hue_order=comparison_hue_order,
            palette=comparison_strip_palette,
            dodge=True,
            jitter=0.18,
            alpha=0.85,
            size=4,
            ax=comparison_axis,
            legend=False,
        )

    add_sig_bars(
        ax_ll, comparison_df,
        x_col="K", y_col="test_ll_per_trial", hue_col="model_label",
        order=comparison_K_order, hue_order=comparison_hue_order, pair_col="subject",
    )

    add_sig_bars(
        ax_bic, comparison_df,
        x_col="K", y_col="bic_delta", hue_col="model_label",
        order=comparison_K_order, hue_order=comparison_hue_order, pair_col="subject",
    )

    ax_ll.set_ylabel("CV test log-likelihood / trial")
    ax_ll.set_title("CV test LL / trial (higher = better)")

    ax_bic.axhline(0, color="grey", lw=0.9, linestyle="--", alpha=0.7)
    ax_bic.set_ylabel("ΔBIC vs baseline")
    ax_bic.set_title(f"ΔBIC vs {ui_bic_baseline.value} (lower = better)")

    comparison_legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            color=comparison_strip_palette[label], label=label, markersize=6,
        )
        for label in comparison_hue_order
    ]
    if ax_ll.get_legend() is not None:
        ax_ll.get_legend().remove()
    if ax_bic.get_legend() is not None:
        ax_bic.get_legend().remove()
    fig_cmp.legend(
        comparison_legend_handles,
        comparison_hue_order,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, max(1, len(comparison_hue_order))),
        frameon=False,
    )

    sns.despine(fig=fig_cmp)
    fig_cmp.tight_layout(rect=(0, 0.12, 1, 1))
    fig_cmp
    return


@app.cell
def _(results_plot, sns):
    def covariate_group(label):
        """Map model labels to the GLM, two-, or three-covariate family."""
        normalized_label = str(label).lower()
        if "3 cov" in normalized_label or "3cov" in normalized_label:
            return "3 covs"
        if "2 cov" in normalized_label:
            return "2 covs"
        if "base_lapses" in normalized_label:
            return "GLM"
        return None

    covariate_comparison_df = results_plot.to_pandas()
    covariate_mean_df = (
        covariate_comparison_df.assign(
            cov_group=covariate_comparison_df["model_label"].map(covariate_group)
        )
        .dropna(subset=["cov_group"])
        .groupby(["cov_group", "K"], as_index=False)[["test_ll_per_trial", "bic_delta"]]
        .mean()
        .sort_values(["cov_group", "K"])
    )

    covariate_order = [
        group
        for group in ["GLM", "2 covs", "3 covs"]
        if group in covariate_mean_df["cov_group"].unique()
    ]
    covariate_palette = {
        group: color
        for group, color in zip(
            covariate_order,
            sns.color_palette("tab10", n_colors=max(1, len(covariate_order))),
        )
    }
    return covariate_mean_df, covariate_order, covariate_palette


@app.cell
def _(
    covariate_mean_df,
    covariate_order,
    covariate_palette,
    plt,
    sns,
    ui_bic_baseline,
):

    fig_cov_mean, (ax_cov_ll, ax_cov_bic) = plt.subplots(
        1, 2, figsize=(8, 4.2), constrained_layout=False
    )

    for covariate_group_name in covariate_order:
        covariate_group_df = covariate_mean_df[
            covariate_mean_df["cov_group"] == covariate_group_name
        ]
        ax_cov_ll.plot(
            covariate_group_df["K"],
            covariate_group_df["test_ll_per_trial"],
            marker="o",
            linewidth=2,
            color=covariate_palette[covariate_group_name],
            label=covariate_group_name,
        )
        ax_cov_bic.plot(
            covariate_group_df["K"],
            covariate_group_df["bic_delta"],
            marker="o",
            linewidth=2,
            color=covariate_palette[covariate_group_name],
            label=covariate_group_name,
        )

    ax_cov_ll.set_xlabel("Number of states K")
    ax_cov_ll.set_ylabel("Mean CV test LL / trial")
    ax_cov_ll.set_title("Mean CV test LL / trial by covariate count")

    ax_cov_bic.axhline(0, color="grey", lw=0.9, linestyle="--", alpha=0.7)
    ax_cov_bic.set_xlabel("Number of states K")
    ax_cov_bic.set_ylabel("Mean ΔBIC vs baseline")
    ax_cov_bic.set_title(f"Mean ΔBIC vs {ui_bic_baseline.value}")

    if covariate_order:
        fig_cov_mean.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(covariate_order),
            frameon=False,
            title="Grouped models",
        )

    sns.despine(fig=fig_cov_mean)
    fig_cov_mean.tight_layout(rect=(0, 0.12, 1, 1))
    fig_cov_mean
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare two models
    """)
    return


@app.cell
def _(MODEL_LABELS, mo, results_long):
    def pairwise_model_key(kind: str, K: int, alias: str) -> str:
        """Encode a model family, state count, and alias for a dropdown value."""
        return f"{kind}\t{int(K)}\t{alias}"

    def parse_pairwise_model_key(key: str) -> tuple[str, int, str]:
        """Decode a pairwise model dropdown value."""
        kind, state_count, alias = key.split("\t", 2)
        return kind, int(state_count), alias

    pairwise_model_specs = list(
        results_long
        .select(["model_kind", "model_alias", "model_label", "K"])
        .unique()
        .sort(["model_kind", "model_alias", "K"])
        .iter_rows(named=True)
    )
    pairwise_model_options = {}
    for pairwise_model_spec in pairwise_model_specs:
        pairwise_option_kind = pairwise_model_spec["model_kind"]
        pairwise_option_K = (
            int(pairwise_model_spec["K"])
            if pairwise_option_kind != "glm"
            else 1
        )
        pairwise_option_alias = pairwise_model_spec["model_alias"]
        option_label = (
            f"{MODEL_LABELS[pairwise_option_kind]} K={pairwise_option_K}: "
            f"{pairwise_option_alias}"
        )
        pairwise_model_options[option_label] = pairwise_model_key(
            pairwise_option_kind, pairwise_option_K, pairwise_option_alias
        )

    pairwise_model_labels = list(pairwise_model_options)
    pairwise_default_a = pairwise_model_labels[0] if pairwise_model_labels else None
    pairwise_default_b = (
        pairwise_model_labels[1]
        if len(pairwise_model_labels) > 1
        else pairwise_default_a
    )
    ui_pairwise_model_a = mo.ui.dropdown(
        options=pairwise_model_options,
        value=pairwise_default_a,
        label="Model A",
    )
    ui_pairwise_model_b = mo.ui.dropdown(
        options=pairwise_model_options,
        value=pairwise_default_b,
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
    pairwise_scoring_options = (
        list(adapter._SCORING_OPTIONS)
        if hasattr(adapter, "_SCORING_OPTIONS")
        else ["default"]
    )
    pairwise_default_scoring = getattr(
        adapter,
        "scoring_key",
        pairwise_scoring_options[0],
    )
    if pairwise_default_scoring not in pairwise_scoring_options:
        pairwise_default_scoring = pairwise_scoring_options[0]

    ui_pairwise_scoring_key_a = mo.ui.dropdown(
        options=pairwise_scoring_options,
        value=pairwise_default_scoring,
        label="Model A state scoring regressor",
    )
    ui_pairwise_scoring_key_b = mo.ui.dropdown(
        options=pairwise_scoring_options,
        value=pairwise_default_scoring,
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

    pairwise_adapter_a, pairwise_arrays_a, pairwise_names_a, pairwise_views_a = load_fit_bundle(
        ui_task.value,
        pairwise_kind_a,
        pairwise_alias_a,
        pairwise_K_a,
        requested_subjects,
        scoring_key=ui_pairwise_scoring_key_a.value,
    )
    pairwise_adapter_b, pairwise_arrays_b, pairwise_names_b, pairwise_views_b = load_fit_bundle(
        ui_task.value,
        pairwise_kind_b,
        pairwise_alias_b,
        pairwise_K_b,
        requested_subjects,
        scoring_key=ui_pairwise_scoring_key_b.value,
    )

    pairwise_metric_schema = {
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

    pairwise_metrics_a = load_metrics_dir(
        ui_task.value, pairwise_alias_a, pairwise_kind_a
    )
    pairwise_metrics_b = load_metrics_dir(
        ui_task.value, pairwise_alias_b, pairwise_kind_b
    )
    pairwise_metrics_a = (
        pl.DataFrame(schema=pairwise_metric_schema)
        if pairwise_metrics_a is None
        else pairwise_metrics_a.filter(
            pl.col("subject").is_in(requested_subjects)
            & (pl.col("K") == pairwise_K_a)
        )
    )
    pairwise_metrics_b = (
        pl.DataFrame(schema=pairwise_metric_schema)
        if pairwise_metrics_b is None
        else pairwise_metrics_b.filter(
            pl.col("subject").is_in(requested_subjects)
            & (pl.col("K") == pairwise_K_b)
        )
    )
    metric_subjects_a = set(pairwise_metrics_a["subject"].to_list())
    metric_subjects_b = set(pairwise_metrics_b["subject"].to_list())
    cached_subjects_a = set(pairwise_views_a)
    cached_subjects_b = set(pairwise_views_b)
    pairwise_common_subjects = [
        subject
        for subject in requested_subjects
        if subject in metric_subjects_a and subject in metric_subjects_b
    ]
    pairwise_cached_common_subjects = [
        subject
        for subject in requested_subjects
        if subject in cached_subjects_a and subject in cached_subjects_b
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
        pairwise_kind_a,
        pairwise_kind_b,
        pairwise_metrics_a,
        pairwise_metrics_b,
        pairwise_missing_a,
        pairwise_missing_b,
        pairwise_views_a,
        pairwise_views_b,
        requested_subjects,
    )


@app.cell
def _(
    mo,
    pairwise_K_a,
    pairwise_K_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_cached_common_subjects,
    pairwise_common_subjects,
    pairwise_kind_a,
    pairwise_kind_b,
    pairwise_missing_a,
    pairwise_missing_b,
    requested_subjects,
    ui_pairwise_scoring_key_a,
    ui_pairwise_scoring_key_b,
):
    pairwise_notes = [
        f"- Comparing A `{pairwise_kind_a}` / `{pairwise_alias_a}` / `K={pairwise_K_a}` vs B `{pairwise_kind_b}` / `{pairwise_alias_b}` / `K={pairwise_K_b}`.",
        f"- Common metric subjects: **{len(pairwise_common_subjects)} / {len(requested_subjects)}**.",
        f"- Common cached fit subjects for state-level plots: **{len(pairwise_cached_common_subjects)} / {len(requested_subjects)}**.",
        f"- `{pairwise_alias_a}` scoring key: `{ui_pairwise_scoring_key_a.value}`.",
        f"- `{pairwise_alias_b}` scoring key: `{ui_pairwise_scoring_key_b.value}`.",
        "- Transition deltas are aligned by semantic state label.",
    ]
    if pairwise_missing_a:
        pairwise_notes.append(
            f"- Missing in `{pairwise_alias_a}`: {', '.join(map(str, pairwise_missing_a[:8]))}"
            + (" ..." if len(pairwise_missing_a) > 8 else "")
        )
    if pairwise_missing_b:
        pairwise_notes.append(
            f"- Missing in `{pairwise_alias_b}`: {', '.join(map(str, pairwise_missing_b[:8]))}"
            + (" ..." if len(pairwise_missing_b) > 8 else "")
        )
    mo.md("\n".join(pairwise_notes))
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
    pairwise_metric_frames = []
    if not pairwise_metrics_a.is_empty():
        pairwise_metric_frames.append(
            pairwise_metrics_a.with_columns(pl.lit("A").alias("model_slot"))
        )
    if not pairwise_metrics_b.is_empty():
        pairwise_metric_frames.append(
            pairwise_metrics_b.with_columns(pl.lit("B").alias("model_slot"))
        )

    if pairwise_metric_frames:
        pairwise_metrics = pl.concat(pairwise_metric_frames, how="diagonal")
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
    PAIRWISE_PRETTY_NAMES,
    fig_size,
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metric_deltas,
    panel,
    plt,
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
    ax_pairwise_ll.set_xlabel(
        f"Model A: {PAIRWISE_PRETTY_NAMES.get(pairwise_alias_a, pairwise_alias_a)}"
    )
    ax_pairwise_ll.set_ylabel(
        f"Model B: {PAIRWISE_PRETTY_NAMES.get(pairwise_alias_b, pairwise_alias_b)}"
    )
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
    PAIRWISE_PRETTY_NAMES,
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

    pairwise_delta_df = pairwise_metric_deltas.to_pandas()
    pairwise_delta_panels = [
        ("delta_ll_per_trial", "Δ test LL / trial", (-0.05, 0.05)),
        ("delta_bic", "Δ BIC", (-800, 800)),
    ]

    def one_sample_p_label(values):
        """Test a finite subject-level delta against zero and return stars."""
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return ""
        if np.allclose(values, 0):
            return "ns"
        p_value = ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue
        if not np.isfinite(p_value):
            return ""
        if p_value < 0.0001:
            return "****"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    fig_pair_metrics, pairwise_metric_axes = plt.subplot_mosaic(
        [["ll", "bic"]],
        figsize=fig_size(1,2),
        constrained_layout=True,
    )
    pairwise_metric_plot_axes = [
        pairwise_metric_axes["ll"],
        pairwise_metric_axes["bic"],
    ]
    for pairwise_metric_axis, (delta_column, ylabel, pairwise_metric_ylim) in zip(
        pairwise_metric_plot_axes, pairwise_delta_panels, strict=False
    ):
        finite_deltas = pairwise_delta_df[delta_column].to_numpy(dtype=float)
        finite_deltas = finite_deltas[np.isfinite(finite_deltas)]
        pairwise_metric_axis.axhline(
            0, color="0.35", linewidth=1.0, linestyle="--", alpha=0.75
        )
        if len(finite_deltas) > 0:
            sns.boxplot(
                y=finite_deltas,
                ax=pairwise_metric_axis,
                color="tab:red",
                **BOXPLOT_STYLE,
            )
        p_text = one_sample_p_label(finite_deltas)
        if p_text:
            pairwise_metric_axis.text(
                0.5, 0.88, p_text, ha="center", transform=pairwise_metric_axis.transAxes
            )
        pairwise_metric_axis.set_xticks([0])
        pairwise_metric_axis.set_xticklabels([
            f"{PAIRWISE_PRETTY_NAMES.get(pairwise_alias_b, pairwise_alias_b)} $-$ "
            f"{PAIRWISE_PRETTY_NAMES.get(pairwise_alias_a, pairwise_alias_a)}"
        ])
        pairwise_metric_axis.set_ylabel(ylabel)
        sns.despine(ax=pairwise_metric_axis)

    mo.vstack([
        fig_pair_metrics,
        save_plot(
            fig_pair_metrics,
            "Metric comparison",
            stem=f"metric_comparison_{pairwise_alias_a}_{pairwise_alias_b}",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Latent-state parameters
    """)
    return


@app.cell
def _(np, plt, sns):
    def resolve_transition_matrix(arrays: dict) -> np.ndarray | None:
        """Return a fitted K x K transition matrix, converting logits if needed."""
        if "transition_matrix" in arrays:
            return np.asarray(arrays["transition_matrix"], dtype=float)
        if "transition_bias" in arrays:
            transition_bias = np.asarray(arrays["transition_bias"], dtype=float)
            transition_exp = np.exp(
                transition_bias - transition_bias.max(axis=-1, keepdims=True)
            )
            return transition_exp / transition_exp.sum(axis=-1, keepdims=True)
        return None

    def reindex_transition_matrix(
        matrix: np.ndarray,
        source_labels: list[str],
        target_labels: list[str],
    ) -> np.ndarray:
        """Map a K x K matrix from ``source_labels`` into an L x L target order."""
        source_index = {label: index for index, label in enumerate(source_labels)}
        aligned = np.full((len(target_labels), len(target_labels)), np.nan)
        for row_index, row_label in enumerate(target_labels):
            source_row = source_index.get(row_label)
            if source_row is None:
                continue
            for column_index, column_label in enumerate(target_labels):
                source_column = source_index.get(column_label)
                if source_column is None:
                    continue
                aligned[row_index, column_index] = matrix[source_row, source_column]
        return aligned

    def finite_max_abs(matrix: np.ndarray) -> float:
        """Return a non-zero color limit from the finite entries of a 2D matrix."""
        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size == 0:
            return 1e-12
        return max(float(np.max(np.abs(finite_values))), 1e-12)

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
        """Plot two mean K x K transition matrices and their semantic-state delta."""
        def mean_transition(arrays_store: dict, views: dict):
            subject_entries = []
            state_labels = []
            for subject in subjects:
                if subject not in views:
                    continue
                transition_matrix = resolve_transition_matrix(
                    arrays_store.get(subject, {})
                )
                if transition_matrix is None:
                    continue
                state_order = [int(k) for k in views[subject].state_idx_order]
                subject_labels = [
                    views[subject].state_name_by_idx.get(int(k), f"State {k}")
                    for k in state_order
                ]
                for state_label in subject_labels:
                    if state_label not in state_labels:
                        state_labels.append(state_label)
                subject_entries.append(
                    (
                        transition_matrix[np.ix_(state_order, state_order)],
                        subject_labels,
                    )
                )
            if not subject_entries:
                return None, []
            aligned_matrices = [
                reindex_transition_matrix(matrix, subject_labels, state_labels)
                for matrix, subject_labels in subject_entries
            ]
            return np.nanmean(np.stack(aligned_matrices), axis=0), state_labels

        matrix_a, labels_a = mean_transition(arrays_a, views_a)
        matrix_b, labels_b = mean_transition(arrays_b, views_b)
        if matrix_a is None or matrix_b is None:
            raise ValueError("No transition matrices were available for the common subject set.")

        state_labels = list(dict.fromkeys(labels_a + labels_b))
        matrix_a = reindex_transition_matrix(matrix_a, labels_a, state_labels)
        matrix_b = reindex_transition_matrix(matrix_b, labels_b, state_labels)

        matrix_max = max(finite_max_abs(matrix_a), finite_max_abs(matrix_b))
        matrix_delta = matrix_b - matrix_a
        delta_max = finite_max_abs(matrix_delta)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=False)
        for axis, matrix, title in [
            (axes[0], matrix_a, alias_a),
            (axes[1], matrix_b, alias_b),
        ]:
            sns.heatmap(
                matrix,
                ax=axis,
                cmap="Blues",
                vmin=0,
                vmax=matrix_max,
                annot=True,
                fmt=".2f",
                square=True,
                cbar=False,
            )
            axis.set_title(title)
            axis.set_xticklabels(state_labels, rotation=25, ha="right")
            axis.set_yticklabels(state_labels, rotation=0)
            axis.set_xlabel("To state")
            axis.set_ylabel("From state")

        sns.heatmap(
            matrix_delta,
            ax=axes[2],
            cmap="RdBu_r",
            center=0,
            vmin=-delta_max,
            vmax=delta_max,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=False,
        )
        axes[2].set_title(f"{alias_b} - {alias_a}")
        axes[2].set_xticklabels(state_labels, rotation=25, ha="right")
        axes[2].set_yticklabels(state_labels, rotation=0)
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
    fig_transition_pair = plot_pairwise_transition_matrices(
        arrays_a=pairwise_arrays_a,
        arrays_b=pairwise_arrays_b,
        views_a=pairwise_views_a,
        views_b=pairwise_views_b,
        subjects=pairwise_cached_common_subjects,
        alias_a=pairwise_alias_a,
        alias_b=pairwise_alias_b,
    )
    mo.vstack([mo.md("#### Transition matrices"), fig_transition_pair])
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
    pairwise_views_a,
    pairwise_views_b,
):
    emission_weights_a = build_emission_weights_df(pairwise_views_a)
    emission_weights_b = build_emission_weights_df(pairwise_views_b)
    fig_emission_a = pairwise_adapter_a.get_plots().plot_emission_weights_summary(
        emission_weights_a,
        K=pairwise_K_a,
    )
    fig_emission_b = pairwise_adapter_b.get_plots().plot_emission_weights_summary(
        emission_weights_b,
        K=pairwise_K_b,
    )
    mo.vstack([
        mo.md("#### Emission weights"),
        mo.hstack([
            mo.vstack([mo.md(f"**A** — `{pairwise_alias_a}`"), fig_emission_a]),
            mo.vstack([mo.md(f"**B** — `{pairwise_alias_b}`"), fig_emission_b]),
        ], widths="equal"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Behavioral interpretation of states
    """)
    return


@app.cell
def _(pl):
    def subject_behavior_df(df_all, *, subject, sort_col, session_col):
        """Return one subject's chronologically sorted trials from valid sessions."""
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
        """Join behavioral columns onto an N-trial model dataframe by trial identity."""
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
    def build_pairwise_trial_df(adapter, views):
        """Concatenate N-trial state dataframes for subjects cached by one fit."""
        trial_frames = []
        for subject in pairwise_cached_common_subjects:
            if subject not in views:
                continue
            subject_df = subject_behavior_df(
                df_all,
                subject=subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
            )
            if subject_df.height != views[subject].T:
                continue
            trial_frames.append(
                build_trial_df(
                    views[subject],
                    adapter,
                    subject_df,
                    adapter.behavioral_cols,
                )
            )
        return pl.concat(trial_frames, how="diagonal") if trial_frames else pl.DataFrame()

    pairwise_trial_df_a = build_pairwise_trial_df(pairwise_adapter_a, pairwise_views_a)
    pairwise_trial_df_b = build_pairwise_trial_df(pairwise_adapter_b, pairwise_views_b)
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
    def session_occupancy_records(*, alias: str, adapter, views: dict):
        """Return one occupancy record per subject, session, and semantic state."""
        records = []
        for subject in pairwise_cached_common_subjects:
            if subject not in views:
                continue
            view = views[subject]
            subject_df = subject_behavior_df(
                df_all,
                subject=subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
            )
            if subject_df.height != view.T:
                continue
            if adapter.session_col not in subject_df.columns:
                continue
            sessions = np.asarray(subject_df[adapter.session_col])
            state_probabilities = np.asarray(view.smoothed_probs, dtype=float)
            for session in np.unique(sessions):
                session_mask = sessions == session
                for state_index in view.state_idx_order:
                    records.append(
                        {
                            "subject": str(subject),
                            "session": str(session),
                            "model_alias": alias,
                            "state_label": view.state_name_by_idx.get(
                                int(state_index), f"State {state_index}"
                            ),
                            "occupancy": float(
                                np.mean(state_probabilities[session_mask, int(state_index)])
                            ),
                        }
                    )
        return records

    occupancy_records = session_occupancy_records(
        alias=pairwise_alias_a,
        adapter=pairwise_adapter_a,
        views=pairwise_views_a,
    )
    occupancy_records += session_occupancy_records(
        alias=pairwise_alias_b,
        adapter=pairwise_adapter_b,
        views=pairwise_views_b,
    )

    if occupancy_records:
        pairwise_session_occupancy = pl.DataFrame(occupancy_records)
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
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_subject_occupancy,
    pairwise_trial_df_a,
    pairwise_trial_df_b,
    pl,
):
    def subject_accuracy(alias, trial_df):
        """Return subject x semantic-state accuracy percentages from N trials."""
        accuracy_df = trial_df
        if "correct_bool" not in accuracy_df.columns:
            accuracy_df = accuracy_df.with_columns(
                pl.col("performance").cast(pl.Boolean).alias("correct_bool")
            )
        return (
            accuracy_df.drop_nulls(["state_label", "correct_bool"])
            .group_by("subject", "state_label")
            .agg((pl.mean("correct_bool") * 100).alias("value"))
            .with_columns(
                pl.lit(alias).alias("model_alias"),
                pl.lit("Accuracy (%)").alias("metric"),
            )
        )

    pairwise_accuracy = pl.concat([
        subject_accuracy(pairwise_alias_a, pairwise_trial_df_a),
        subject_accuracy(pairwise_alias_b, pairwise_trial_df_b),
    ], how="diagonal")
    pairwise_occupancy = pairwise_subject_occupancy.rename(
        {"occupancy": "value"}
    ).with_columns(pl.lit("Occupancy").alias("metric"))
    occupancy_accuracy_df = pl.concat(
        [pairwise_occupancy, pairwise_accuracy], how="diagonal"
    ).to_pandas()
    pairwise_model_order = [pairwise_alias_a, pairwise_alias_b]
    pairwise_model_palette = {
        pairwise_alias_a: "#1B6CA8",
        pairwise_alias_b: "#C76D3A",
    }
    return occupancy_accuracy_df, pairwise_model_order, pairwise_model_palette


@app.cell
def _(
    BOXPLOT_STYLE,
    fig_size,
    mo,
    np,
    occupancy_accuracy_df,
    pairwise_K_a,
    pairwise_K_b,
    pairwise_adapter_a,
    pairwise_model_order,
    pairwise_model_palette,
    pairwise_session_occupancy,
    plt,
    sns,
    ttest_rel,
):
    mo.stop(pairwise_session_occupancy.is_empty(), mo.md("#### No session occupancy for current subset."))
    fig_occupancy_accuracy, occupancy_accuracy_axes = plt.subplot_mosaic(
        [["occ", "acc"]], figsize=fig_size(1, 2)
    )

    for panel_key, metric_name, metric_ylim, chance_level in [
        ("occ", "Occupancy", (0, 1), 1 / pairwise_K_a if pairwise_K_a == pairwise_K_b else None),
        ("acc", "Accuracy (%)", (0, 100), 100 / pairwise_adapter_a.num_classes if pairwise_adapter_a else None),
    ]:
        occupancy_accuracy_axis = occupancy_accuracy_axes[panel_key]
        panel_df = occupancy_accuracy_df[
            occupancy_accuracy_df["metric"].eq(metric_name)
        ]
        state_order = sorted(panel_df["state_label"].dropna().unique())

        if chance_level is not None:
            occupancy_accuracy_axis.axhline(chance_level, ls="--", lw=1, c="gray")

        sns.boxplot(
            data=panel_df, x="state_label", y="value", hue="model_alias",
            order=state_order, hue_order=pairwise_model_order,
            palette=pairwise_model_palette, width=.55,
            ax=occupancy_accuracy_axis, **BOXPLOT_STYLE,
        )

        for state_index, state_label in enumerate(state_order):
            paired_state_df = (
                panel_df[panel_df["state_label"].eq(state_label)]
                .pivot(index="subject", columns="model_alias", values="value")
                .dropna()
            )
            for _, subject_values in paired_state_df.iterrows():
                occupancy_accuracy_axis.plot(
                    [state_index - .2, state_index + .2],
                    [subject_values[pairwise_model_order[0]], subject_values[pairwise_model_order[1]]],
                    alpha=0.15,
                    color="tab:gray",
                )

            if len(paired_state_df) >= 2 and not np.allclose(
                paired_state_df[pairwise_model_order[0]],
                paired_state_df[pairwise_model_order[1]],
            ):
                p_value = ttest_rel(
                    paired_state_df[pairwise_model_order[0]],
                    paired_state_df[pairwise_model_order[1]],
                    nan_policy="omit",
                ).pvalue
                stars = "***" if p_value < .001 else "**" if p_value < .01 else "*" if p_value < .05 else "ns"
                annotation_y = paired_state_df.max().max() + .05 * (metric_ylim[1] - metric_ylim[0])
                occupancy_accuracy_axis.plot(
                    [state_index - .2, state_index + .2],
                    [annotation_y, annotation_y], c="k", lw=1,
                )
                occupancy_accuracy_axis.text(
                    state_index, annotation_y, stars, ha="center", va="bottom"
                )

        occupancy_accuracy_axis.set(xlabel="", ylabel=metric_name, ylim=metric_ylim)
        sns.despine(ax=occupancy_accuracy_axis)

    occupancy_handles, occupancy_labels = occupancy_accuracy_axes["occ"].get_legend_handles_labels()
    for panel_axis in occupancy_accuracy_axes.values():
        panel_axis.legend_.remove()
    fig_occupancy_accuracy.legend(
        occupancy_handles[:2], occupancy_labels[:2], title="Model",
        loc="lower center", ncol=2, frameon=False,
    )
    fig_occupancy_accuracy.tight_layout(rect=(0, .12, 1, 1))
    fig_occupancy_accuracy
    return


@app.cell
def _(
    PAIRWISE_PRETTY_NAMES,
    RT_METRIC_CANDIDATES,
    augment_behavior_metrics,
    mo,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_trial_df_a,
    pairwise_trial_df_b,
    pd,
):
    roc_trials_a = augment_behavior_metrics(pairwise_trial_df_a, pairwise_adapter_a)
    roc_trials_b = augment_behavior_metrics(pairwise_trial_df_b, pairwise_adapter_b)
    if not roc_trials_a.empty:
        roc_trials_a = roc_trials_a.assign(
            model_slot="A",
            model_name=PAIRWISE_PRETTY_NAMES.get(pairwise_alias_a, pairwise_alias_a),
        )
    if not roc_trials_b.empty:
        roc_trials_b = roc_trials_b.assign(
            model_slot="B",
            model_name=PAIRWISE_PRETTY_NAMES.get(pairwise_alias_b, pairwise_alias_b),
        )
    roc_trial_frames = [
        trial_df for trial_df in [roc_trials_a, roc_trials_b] if not trial_df.empty
    ]
    mo.stop(
        not roc_trial_frames,
        mo.md("No pairwise trial data were available for behavioral ROC curves."),
    )
    roc_df = pd.concat(roc_trial_frames, ignore_index=True)

    roc_state_column = next(
        (
            column
            for column in ["state_label", "state_label_pred"]
            if column in roc_df.columns
        ),
        None,
    )
    mo.stop(
        roc_state_column is None,
        mo.md("State labels are not available for pairwise behavioral ROC curves."),
    )

    roc_rt_column = next(
        (column for column in RT_METRIC_CANDIDATES if column in roc_df.columns),
        None,
    )
    roc_metric_specs = [
        ("nLicks", "Licking (correct trials)", "Higher lick count")
    ]
    if roc_rt_column is not None:
        roc_metric_specs.append((roc_rt_column, "RT", "Faster RT"))
    roc_metric_specs.append(("ILI", "ILI", "Faster ILI"))
    roc_metric_specs = [
        metric_spec
        for metric_spec in roc_metric_specs
        if metric_spec[0] in roc_df.columns
    ]
    mo.stop(
        not roc_metric_specs,
        mo.md("No lick, RT, or ILI column was found for behavioral ROC curves."),
    )
    return roc_df, roc_metric_specs, roc_state_column


@app.cell
def _(np, pd):
    def correct_trial_mask(trial_df):
        """Return a mask selecting trials with a correct behavioral response."""
        correct_column = next(
            (
                column
                for column in ["correct_bool", "performance"]
                if column in trial_df.columns
            ),
            None,
        )
        if correct_column is None:
            raise ValueError(
                "Lick ROC curves require a `correct_bool` or `performance` column."
            )
        return (
            pd.to_numeric(trial_df[correct_column], errors="coerce")
            .eq(1)
            .fillna(False)
            .to_numpy(bool)
        )

    def binary_engaged_target(labels):
        """Return two length-N masks: Engaged targets and valid binary-state rows."""
        label_text = pd.Series(labels, copy=False).astype(str).str.strip().str.lower()
        positive = label_text.eq("engaged") | label_text.str.startswith("engaged ")
        negative = label_text.eq("disengaged") | label_text.str.startswith("disengaged ")
        return positive.to_numpy(bool), (positive | negative).to_numpy(bool)

    def roc_curve(target, score):
        """Return FPR, TPR, and AUC from two length-N target and score arrays."""
        target = np.asarray(target, dtype=bool)
        score = np.asarray(score, dtype=float)
        finite_score = np.isfinite(score)
        target = target[finite_score]
        score = score[finite_score]
        n_positive = int(target.sum())
        n_negative = int((~target).sum())
        if target.size == 0 or n_positive == 0 or n_negative == 0:
            return None

        score_order = np.argsort(-score, kind="mergesort")
        sorted_target = target[score_order]
        sorted_score = score[score_order]
        threshold_indices = np.r_[
            np.where(np.diff(sorted_score))[0],
            sorted_target.size - 1,
        ]
        true_positives = np.cumsum(sorted_target)[threshold_indices]
        false_positives = 1 + threshold_indices - true_positives
        true_positive_rate = np.r_[0.0, true_positives / n_positive]
        false_positive_rate = np.r_[0.0, false_positives / n_negative]
        auc = float(
            np.sum(
                np.diff(false_positive_rate)
                * (true_positive_rate[:-1] + true_positive_rate[1:])
                / 2
            )
        )
        return false_positive_rate, true_positive_rate, auc

    return binary_engaged_target, correct_trial_mask, roc_curve


@app.cell
def _(
    ROC_PALETTE,
    binary_engaged_target,
    correct_trial_mask,
    mo,
    pd,
    plt,
    roc_curve,
    roc_df,
    roc_metric_specs,
    roc_state_column,
    save_plot,
    sns,
):
    fig_pooled_roc, pooled_roc_axes = plt.subplots(
        1,
        len(roc_metric_specs),
        # figsize=fig_size(1, len(roc_metric_specs)),
        figsize = (9,3),
        squeeze=False,
    )
    pooled_roc_plotted = False
    for pooled_roc_axis, (roc_metric, roc_title, roc_direction) in zip(
        pooled_roc_axes.ravel(), roc_metric_specs, strict=False
    ):
        for roc_model_slot, roc_slot_df in roc_df.groupby("model_slot", sort=True):
            roc_target, valid_state_labels = binary_engaged_target(
                roc_slot_df[roc_state_column]
            )
            roc_score = pd.to_numeric(
                roc_slot_df[roc_metric], errors="coerce"
            ).to_numpy(float)
            valid_roc_trials = valid_state_labels.copy()
            if roc_metric == "nLicks":
                valid_roc_trials &= correct_trial_mask(roc_slot_df)
            if roc_metric != "nLicks":
                roc_score = -roc_score
            roc_result = roc_curve(
                roc_target[valid_roc_trials], roc_score[valid_roc_trials]
            )
            if roc_result is None:
                continue
            false_positive_rate, true_positive_rate, pooled_auc = roc_result
            roc_model_name = str(roc_slot_df["model_name"].iloc[0])
            pooled_roc_axis.plot(
                false_positive_rate,
                true_positive_rate,
                color=ROC_PALETTE[roc_model_slot],
                lw=2,
                label=f"{roc_model_name} (AUC={pooled_auc:.3f})",
            )
            pooled_roc_plotted = True

        pooled_roc_axis.plot([0, 1], [0, 1], color="tab:gray", lw=1, ls="--")
        pooled_roc_axis.set(
            title=roc_title,
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        pooled_roc_axis.legend(frameon=False, loc="lower right")
        sns.despine(ax=pooled_roc_axis)

    mo.stop(
        not pooled_roc_plotted,
        mo.md("Behavioral ROC curves require Engaged and Disengaged trials."),
    )
    mo.vstack([
        mo.md("#### Pairwise behavioral ROC by state"),
        fig_pooled_roc,
        save_plot(
            fig_pooled_roc,
            "pairwise behavioral ROC by state",
            stem="pairwise_state_behavioral_roc",
        ),
    ], align="center")
    return


@app.cell
def _(
    binary_engaged_target,
    correct_trial_mask,
    mo,
    np,
    pd,
    roc_curve,
    roc_df,
    roc_metric_specs,
    roc_state_column,
):
    roc_fpr_grid = np.linspace(0, 1, 101)
    subject_roc_curve_rows = []
    subject_auc_rows = []

    for (subject_roc_slot, roc_subject), subject_roc_df in roc_df.groupby(
        ["model_slot", "subject"], sort=True
    ):
        for subject_roc_metric, subject_roc_title, subject_roc_direction in roc_metric_specs:
            subject_target, valid_subject_labels = binary_engaged_target(
                subject_roc_df[roc_state_column]
            )
            subject_score = pd.to_numeric(
                subject_roc_df[subject_roc_metric], errors="coerce"
            ).to_numpy(float)
            valid_subject_trials = valid_subject_labels.copy()
            if subject_roc_metric == "nLicks":
                valid_subject_trials &= correct_trial_mask(subject_roc_df)
            if subject_roc_metric != "nLicks":
                subject_score = -subject_score

            subject_roc_result = roc_curve(
                subject_target[valid_subject_trials],
                subject_score[valid_subject_trials],
            )
            if subject_roc_result is None:
                continue

            subject_fpr, subject_tpr, subject_auc = subject_roc_result
            interpolated_tpr = np.interp(roc_fpr_grid, subject_fpr, subject_tpr)
            interpolated_tpr[[0, -1]] = [0.0, 1.0]
            subject_model_name = str(subject_roc_df["model_name"].iloc[0])

            subject_auc_rows.append(
                {
                    "model_slot": subject_roc_slot,
                    "model_name": subject_model_name,
                    "subject": roc_subject,
                    "metric": subject_roc_metric,
                    "metric_label": subject_roc_title,
                    "auc": subject_auc,
                }
            )

            for fpr_value, tpr_value in zip(roc_fpr_grid, interpolated_tpr):
                subject_roc_curve_rows.append(
                    {
                        "model_slot": subject_roc_slot,
                        "model_name": subject_model_name,
                        "subject": roc_subject,
                        "metric": subject_roc_metric,
                        "metric_label": subject_roc_title,
                        "fpr": fpr_value,
                        "tpr": tpr_value,
                    }
                )

    auc_df = pd.DataFrame(subject_auc_rows)
    curve_df = pd.DataFrame(subject_roc_curve_rows)

    mo.stop(
        auc_df.empty or curve_df.empty,
        mo.md("Pairwise behavioral ROC curves require Engaged and Disengaged trials per subject."),
    )
    return auc_df, curve_df


@app.cell
def _(
    Annotator,
    BOXPLOT_STYLE,
    ROC_PALETTE,
    auc_df,
    curve_df,
    fig_size,
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    plt,
    roc_metric_specs,
    save_plot,
    sns,
):
    fig_subject_roc, subject_roc_axes = plt.subplots(
        1,
        len(roc_metric_specs),
        figsize=fig_size(1, len(roc_metric_specs)),
        squeeze=False,
        layout="constrained",
    )
    for subject_roc_axis, (subject_plot_metric, subject_plot_title, subject_plot_direction) in zip(
        subject_roc_axes.ravel(), roc_metric_specs, strict=False
    ):
        metric_curve_df = curve_df[curve_df["metric"].eq(subject_plot_metric)]
        metric_auc_df = auc_df[auc_df["metric"].eq(subject_plot_metric)]

        for subject_plot_slot, slot_curve_df in metric_curve_df.groupby("model_slot", sort=True):
            roc_summary_df = (
                slot_curve_df
                .groupby("fpr", as_index=False)
                .agg(
                    mean_tpr=("tpr", "mean"),
                    sem_tpr=("tpr", "sem"),
                )
            )

            slot_auc_df = metric_auc_df[metric_auc_df["model_slot"].eq(subject_plot_slot)]
            subject_plot_model_name = str(slot_auc_df["model_name"].iloc[0])
            mean_auc = slot_auc_df["auc"].mean()
            sem_auc = slot_auc_df["auc"].sem()

            subject_plot_color = ROC_PALETTE.get(subject_plot_slot, "tab:gray")

            subject_roc_axis.plot(
                roc_summary_df["fpr"],
                roc_summary_df["mean_tpr"],
                color=subject_plot_color,
                lw=2,
                label=f"{subject_plot_model_name} AUC={mean_auc:.3f} ± {sem_auc:.3f}",
            )

            subject_roc_axis.fill_between(
                roc_summary_df["fpr"],
                roc_summary_df["mean_tpr"] - roc_summary_df["sem_tpr"].fillna(0),
                roc_summary_df["mean_tpr"] + roc_summary_df["sem_tpr"].fillna(0),
                color=subject_plot_color,
                alpha=0.2,
                linewidth=0,
            )

        subject_roc_axis.plot([0, 1], [0, 1], color="tab:gray", lw=1, ls="--")
        subject_roc_axis.set(
            title=subject_plot_title,
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        subject_roc_axis.legend(frameon=False, loc="lower right")
        sns.despine(ax=subject_roc_axis)

    fig_subject_auc, subject_auc_axes = plt.subplots(
        1,
        len(roc_metric_specs),
        figsize=fig_size(1, len(roc_metric_specs)),
        squeeze=False,
        layout="constrained",
    )

    for subject_auc_axis, (auc_metric, auc_title, auc_direction) in zip(
        subject_auc_axes.ravel(), roc_metric_specs, strict=False
    ):
        auc_metric_df = auc_df[auc_df["metric"].eq(auc_metric)].copy()
        auc_model_name_a = auc_metric_df[
            auc_metric_df["model_slot"].eq("A")
        ]["model_name"].iloc[0]
        auc_model_name_b = auc_metric_df[
            auc_metric_df["model_slot"].eq("B")
        ]["model_name"].iloc[0]
        auc_model_order = [auc_model_name_a, auc_model_name_b]

        sns.boxplot(
            data=auc_metric_df,
            x="model_name",
            y="auc",
            hue="model_slot",
            order=auc_model_order,
            hue_order=["A", "B"],
            palette=ROC_PALETTE,
            ax=subject_auc_axis,
            zorder=1,
            **BOXPLOT_STYLE,
        )

        sns.stripplot(
            data=auc_metric_df,
            x="model_name",
            y="auc",
            hue="model_slot",
            order=auc_model_order,
            hue_order=["A", "B"],
            palette=ROC_PALETTE,
            ax=subject_auc_axis,
            dodge=False,
            alpha=0.2,
            size=0,
            legend=False,
            zorder=0,
        )

        paired_auc_df = (
            auc_metric_df
            .pivot_table(
                index="subject",
                columns="model_slot",
                values="auc",
                aggfunc="mean",
            )
            .dropna(subset=["A", "B"])
        )

        for auc_subject, auc_subject_values in paired_auc_df.iterrows():
            subject_auc_axis.plot(
                auc_model_order,
                [auc_subject_values["A"], auc_subject_values["B"]],
                color="0.75",
                linewidth=0.5,
                zorder=0,
            )

        Annotator(
            subject_auc_axis,
            [(auc_model_name_a, auc_model_name_b)],
            data=auc_metric_df,
            x="model_name",
            y="auc",
            order=auc_model_order,
        ).configure(
            test="t-test_paired",
            text_format="star",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()

        subject_auc_axis.axhline(0.5, color="tab:gray", lw=1, ls="--")
        subject_auc_axis.set(
            title=f"{auc_title} AUC by subject",
            xlabel="",
            ylabel="AUC",
            ylim=(0, 1),
        )
        sns.despine(ax=subject_auc_axis)
        if subject_auc_axis.get_legend() is not None:
            subject_auc_axis.legend_.remove()
    mo.vstack(
        [
            mo.md("#### Pairwise behavioral ROC by state, per subject"),
            fig_subject_roc,
            mo.md("#### Subject-level AUC distributions"),
            fig_subject_auc,
            mo.ui.table(auc_df, pagination=True),
            save_plot(
                fig_subject_roc,
                "pairwise behavioral ROC by state per subject",
                stem="pairwise_state_behavioral_roc_by_subject",
            ),
            save_plot(
                fig_subject_auc,
                "pairwise behavioral ROC AUC by subject",
                stem="pairwise_state_behavioral_roc_auc_by_subject",
            ),
        ],
        align="center",
    )

    mo.vstack([
        fig_subject_auc,
        save_plot(
            fig_subject_auc,
            "AUC comparison",
            stem=f"auc_comparison_{pairwise_alias_a}_{pairwise_alias_b}",
        ),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
