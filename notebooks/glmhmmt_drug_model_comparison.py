import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import itertools
    import json
    import math
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

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

    from glmhmmt.cli.fit_glmhmmt import main as fit_main
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter

    from src.plots.common import fig_size

    project_root = _PROJECT_ROOT
    configure_paths(config_path=project_root / "config.toml")
    paths = get_runtime_paths()

    sns.set_theme(style="ticks", context="paper")
    return (
        fig_size,
        fit_main,
        get_adapter,
        itertools,
        json,
        math,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        project_root,
        sns,
    )


@app.cell
def _(plt, project_root):
    plt.style.use(project_root / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return


@app.cell
def _(fig_size, project_root):
    path_panels = project_root / "figures" / "panels_glmhmmt_drug_model_comparison"
    path_panels.mkdir(parents=True, exist_ok=True)
    figsize = fig_size(2, 1)
    heatmap_figsize = fig_size(2, 1.25)
    return figsize, heatmap_figsize, path_panels


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # GLM-HMM-T drug model comparison

    "
        "This notebook fits and compares 2-state GLM-HMM-T drug batteries for "
        "`2AFC_DRUG` and `2ADC_DRUG`.
    """)
    return


@app.cell
def _(itertools, pd):
    K = 2
    TAU = 50

    TASK_CONFIGS = {
        "2AFC_DRUG": {
            "label": "2AFC drug",
            "drug_col": "Drug",
            "stim_feature": "stim_param",
            "history_feature": "choice_lag_param",
            "base_emission_cols": ["bias", "stim_param", "choice_lag_param"],
            "base_transition_cols": ["trial_index", "filtered_reward"],
            "stim_fixed_state": "0",
        },
        "2ADC_DRUG": {
            "label": "2ADC drug",
            "drug_col": "drug_code",
            "stim_feature": "stim_x_delay_param",
            "history_feature": "choice_lag_param",
            "base_emission_cols": ["bias", "stim_x_delay_param", "choice_lag_param"],
            "base_transition_cols": ["trial_index", "filtered_reward"],
            "stim_fixed_state": "1",
        },
    }
    TASK_OPTIONS = list(TASK_CONFIGS)

    def _powerset(items: list[str]) -> list[tuple[str, ...]]:
        return [
            tuple(combo)
            for size in range(len(items) + 1)
            for combo in itertools.combinations(items, size)
        ]

    def _effect_key(prefix: str, interactions: tuple[str, ...]) -> str:
        del prefix
        if not interactions:
            return "main"
        return "_".join(
            name.replace("filtered_", "filt_").replace("choice_lag_param", "choice")
            for name in interactions
        )

    def _effect_label(drug_col: str, interactions: tuple[str, ...]) -> str:
        if not interactions:
            return drug_col
        return drug_col + " x " + "+".join(interactions)

    def _effect_levels(
        *,
        drug_col: str,
        interaction_sources: list[str],
        prefix: str,
    ) -> list[dict]:
        return [
            {
                "key": "none",
                "label": "none",
                "drug_main": False,
                "interactions": (),
                "cols": [],
            },
            *[
                {
                    "key": _effect_key(prefix, combo),
                    "label": _effect_label(drug_col, combo),
                    "drug_main": True,
                    "interactions": combo,
                    "cols": [drug_col, *[f"drug_x_{col}" for col in combo]],
                }
                for combo in _powerset(interaction_sources)
            ],
        ]

    def drug_model_specs(task_name: str) -> list[dict]:
        cfg = TASK_CONFIGS[task_name]
        drug_col = cfg["drug_col"]
        stim_feature = cfg["stim_feature"]
        history_feature = cfg["history_feature"]
        emission_levels = _effect_levels(
            drug_col=drug_col,
            interaction_sources=[stim_feature, history_feature],
            prefix="E",
        )
        transition_levels = _effect_levels(
            drug_col=drug_col,
            interaction_sources=list(cfg["base_transition_cols"]),
            prefix="T",
        )
        freeze_levels = [
            {
                "key": "stim_free",
                "label": "stim free",
                "frozen_emissions": {},
            },
            {
                "key": "stim_fixed",
                "label": "stim fixed",
                "frozen_emissions": {
                    cfg["stim_fixed_state"]: {stim_feature: 0.0}
                },
            },
        ]

        specs = []
        order = 0
        for freeze, emission, transition in itertools.product(
            freeze_levels,
            emission_levels,
            transition_levels,
        ):
            emission_cols = [
                *cfg["base_emission_cols"],
                *emission["cols"],
            ]
            transition_cols = [
                *cfg["base_transition_cols"],
                *transition["cols"],
            ]
            if emission["key"] == "none" and transition["key"] == "none":
                placement = "no drug"
            elif emission["key"] != "none" and transition["key"] != "none":
                placement = "both"
            elif emission["key"] != "none":
                placement = "emissions"
            else:
                placement = "transitions"
            model_id = (
                f"druggrid_{freeze['key']}"
                f"_E_{emission['key']}"
                f"_T_{transition['key']}"
            )
            specs.append(
                {
                    "task": task_name,
                    "model_id": model_id,
                    "model_order": order,
                    "freeze_key": freeze["key"],
                    "freeze_label": freeze["label"],
                    "drug_placement": placement,
                    "emission_key": emission["key"],
                    "emission_label": emission["label"],
                    "transition_key": transition["key"],
                    "transition_label": transition["label"],
                    "short_label": (
                        f"{freeze['label']} | E:{emission['label']} | "
                        f"T:{transition['label']}"
                    ),
                    "emission_cols": list(dict.fromkeys(emission_cols)),
                    "transition_cols": list(dict.fromkeys(transition_cols)),
                    "frozen_emissions": freeze["frozen_emissions"],
                }
            )
            order += 1
        return specs

    model_specs_by_task = {
        task_name: drug_model_specs(task_name)
        for task_name in TASK_OPTIONS
    }

    model_counts = pd.DataFrame(
        [
            {
                "task": task_name,
                "n_models": len(specs),
                "n_stim_free": sum(spec["freeze_key"] == "stim_free" for spec in specs),
                "n_stim_fixed": sum(spec["freeze_key"] == "stim_fixed" for spec in specs),
            }
            for task_name, specs in model_specs_by_task.items()
        ]
    )
    return K, TASK_OPTIONS, TAU, model_counts, model_specs_by_task


@app.cell
def _(TASK_OPTIONS, mo):
    ui_task = mo.ui.dropdown(options=TASK_OPTIONS, value="2AFC_DRUG", label="Task")
    ui_cv_mode = mo.ui.dropdown(
        options=["balanced_session_holdout", "none"],
        value="balanced_session_holdout",
        label="Fit score",
    )
    ui_num_iters = mo.ui.number(start=1, stop=500, step=1, value=50, label="EM iterations")
    ui_n_restarts = mo.ui.number(start=1, stop=20, step=1, value=1, label="Restarts")
    ui_run_grid = mo.ui.run_button(label="Fit selected task grid")
    return ui_cv_mode, ui_n_restarts, ui_num_iters, ui_run_grid, ui_task


@app.cell
def _(get_adapter, pl, ui_task):
    task_name = ui_task.value
    adapter = get_adapter(task_name)
    df_all = adapter.subject_filter(adapter.read_dataset())
    all_subjects = [str(subject) for subject in df_all["subject"].unique().sort().to_list()]
    subject_trial_counts = (
        df_all
        .group_by("subject")
        .agg(pl.len().cast(pl.Int64).alias("n_trials"))
        .with_columns(pl.col("subject").cast(pl.Utf8))
    )
    condition_counts = (
        df_all
        .group_by("condition")
        .agg(pl.len().alias("n_trials"))
        .sort("condition", nulls_last=True)
        if "condition" in df_all.columns
        else pl.DataFrame()
    )
    baseline_class_idx = int(adapter.baseline_class_idx)
    return (
        all_subjects,
        baseline_class_idx,
        condition_counts,
        subject_trial_counts,
        task_name,
    )


@app.cell
def _(all_subjects, mo):
    ui_subjects = mo.ui.multiselect(
        options=all_subjects,
        value=all_subjects,
        label="Subjects",
    )
    return (ui_subjects,)


@app.cell
def _(
    condition_counts,
    mo,
    model_counts,
    model_specs_by_task,
    pd,
    task_name,
    ui_cv_mode,
    ui_n_restarts,
    ui_num_iters,
    ui_run_grid,
    ui_subjects,
    ui_task,
):
    model_table = pd.DataFrame(
        [
            {
                "order": spec["model_order"],
                "model_id": spec["model_id"],
                "freeze": spec["freeze_label"],
                "placement": spec["drug_placement"],
                "emission": spec["emission_label"],
                "transition": spec["transition_label"],
                "emission_cols": ", ".join(spec["emission_cols"]),
                "transition_cols": ", ".join(spec["transition_cols"]),
                "frozen_emissions": spec["frozen_emissions"],
            }
            for spec in model_specs_by_task[task_name]
        ]
    )
    mo.vstack(
        [
            mo.hstack([ui_task, ui_cv_mode, ui_num_iters, ui_n_restarts]),
            ui_subjects,
            ui_run_grid,
            condition_counts,
            model_counts,
        ]
    )
    return


@app.cell
def _():
    # model_table
    return


@app.cell
def _(
    K,
    TAU,
    baseline_class_idx,
    fit_main,
    json,
    mo,
    np,
    paths,
    pl,
    task_name,
):
    def free_parameter_count(spec: dict, *, k: int = K, num_classes: int = 2) -> int:
        n_emission = k * (num_classes - 1) * len(spec["emission_cols"])
        n_frozen = sum(len(features) for features in spec.get("frozen_emissions", {}).values())
        n_transition = k * (k - 1) * (1 + len(spec["transition_cols"]))
        return int(n_transition + n_emission - n_frozen)

    def _normalise_frozen(value) -> dict:
        if not value:
            return {}
        return {
            str(state): {str(name): float(weight) for name, weight in features.items()}
            for state, features in dict(value).items()
        }

    def spec_matches_config(spec: dict, model_dir) -> bool:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            return False
        return (
            list(config.get("emission_cols", [])) == list(spec["emission_cols"])
            and list(config.get("transition_cols", [])) == list(spec["transition_cols"])
            and _normalise_frozen(config.get("frozen_emissions")) == _normalise_frozen(spec.get("frozen_emissions"))
        )

    def subject_arrays_match_spec(spec: dict, model_dir, metrics_path) -> bool:
        subject = metrics_path.name.split("_K", maxsplit=1)[0]
        arrays_path = model_dir / f"{subject}_K{K}_glmhmmt_arrays.npz"
        if not arrays_path.exists():
            return False
        try:
            with np.load(arrays_path, allow_pickle=True) as arrays:
                x_cols = [str(col) for col in arrays["X_cols"]]
                u_cols = [str(col) for col in arrays["U_cols"]]
                frozen_json = str(arrays["frozen_emissions_json"].item())
        except Exception:
            return False
        try:
            array_frozen = json.loads(frozen_json) if frozen_json else {}
        except Exception:
            return False
        return (
            x_cols == list(spec["emission_cols"])
            and u_cols == list(spec["transition_cols"])
            and _normalise_frozen(array_frozen) == _normalise_frozen(spec.get("frozen_emissions"))
        )

    def read_metrics(model_specs: list[dict]) -> pl.DataFrame:
        frames = []
        for spec in model_specs:
            model_dir = paths.RESULTS / "fits" / task_name / "glmhmmt" / spec["model_id"]
            if not spec_matches_config(spec, model_dir):
                continue
            for path in sorted(model_dir.glob("*_glmhmmt_metrics.parquet")):
                if not subject_arrays_match_spec(spec, model_dir, path):
                    continue
                frame = pl.read_parquet(path).with_columns(
                    pl.col("subject").cast(pl.Utf8),
                    pl.lit(spec["model_id"]).alias("model_id"),
                    pl.lit(spec["short_label"]).alias("model_label"),
                    pl.lit(spec["short_label"]).alias("short_label"),
                    pl.lit(spec["model_order"], dtype=pl.Int64).alias("model_order"),
                    pl.lit(spec["freeze_key"]).alias("freeze_key"),
                    pl.lit(spec["freeze_label"]).alias("freeze_label"),
                    pl.lit(spec["drug_placement"]).alias("drug_placement"),
                    pl.lit(spec["emission_key"]).alias("emission_key"),
                    pl.lit(spec["emission_label"]).alias("emission_label"),
                    pl.lit(spec["transition_key"]).alias("transition_key"),
                    pl.lit(spec["transition_label"]).alias("transition_label"),
                    pl.lit(free_parameter_count(spec), dtype=pl.Int64).alias("n_free_params"),
                )
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal")

    def score_column(metrics: pl.DataFrame) -> str:
        if "test_ll_per_trial_mean" in metrics.columns:
            non_null = metrics.select(pl.col("test_ll_per_trial_mean").is_not_null().sum()).item()
            if non_null:
                return "test_ll_per_trial_mean"
        return "ll_per_trial"

    def fit_grid(
        *,
        model_specs: list[dict],
        subjects: list[str],
        cv_mode: str,
        num_iters: int,
        n_restarts: int,
    ) -> None:
        cv_repeats = 5 if cv_mode != "none" else 0
        steps_per_subject = cv_repeats if cv_mode != "none" else n_restarts
        total = max(1, len(model_specs) * len(subjects) * steps_per_subject)
        with mo.status.progress_bar(
            total=total,
            title="Fitting GLM-HMM-T drug grid",
            subtitle=f"{len(model_specs)} models x {len(subjects)} subjects",
            completion_title="Fit grid complete",
        ) as bar:
            for spec in model_specs:
                out_dir = paths.RESULTS / "fits" / task_name / "glmhmmt" / spec["model_id"]

                def on_progress(info: dict) -> None:
                    event = info.get("event")
                    if event == "cv_repeat_complete":
                        bar.update(
                            increment=1,
                            title=f"Fitting {spec['model_id']}",
                            subtitle=(
                                f"{info.get('subject')} CV fold "
                                f"{info.get('cv_repeat_index')}/{info.get('cv_repeat_total')}"
                            ),
                        )
                    elif event == "restart_complete" and cv_mode == "none":
                        bar.update(
                            increment=1,
                            title=f"Fitting {spec['model_id']}",
                            subtitle=(
                                f"{info.get('subject')} restart "
                                f"{info.get('restart_index')}/{info.get('restart_total')}"
                            ),
                        )

                fit_main(
                    subjects=subjects,
                    K_list=[K],
                    num_iters=num_iters,
                    n_restarts=1 if cv_mode != "none" else n_restarts,
                    base_seed=0,
                    out_dir=out_dir,
                    tau=TAU,
                    emission_cols=list(spec["emission_cols"]),
                    transition_cols=list(spec["transition_cols"]),
                    frozen_emissions=spec["frozen_emissions"] or None,
                    task=task_name,
                    cv_mode=cv_mode,
                    cv_repeats=cv_repeats,
                    verbose=False,
                    baseline_class_idx=baseline_class_idx,
                    progress_callback=on_progress,
                )

    return fit_grid, read_metrics, score_column


@app.cell
def _(
    fit_grid,
    mo,
    model_specs_by_task,
    task_name,
    ui_cv_mode,
    ui_n_restarts,
    ui_num_iters,
    ui_run_grid,
    ui_subjects,
):
    if ui_run_grid.value:
        fit_grid(
            model_specs=model_specs_by_task[task_name],
            subjects=list(ui_subjects.value),
            cv_mode=ui_cv_mode.value,
            num_iters=int(ui_num_iters.value),
            n_restarts=int(ui_n_restarts.value),
        )
        fit_output = mo.md("Selected task grid fitted.")
    else:
        fit_output = mo.md("Saved fits are loaded below. Press the run button to fit or refresh the selected task grid.")
    fit_output
    return


@app.cell
def _(model_specs_by_task, read_metrics, task_name):
    model_metrics = read_metrics(model_specs_by_task[task_name])
    return (model_metrics,)


@app.cell
def _(math, mo, model_metrics, pl, score_column, subject_trial_counts):
    score_schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "model_label": pl.Utf8,
        "short_label": pl.Utf8,
        "model_order": pl.Int64,
        "freeze_key": pl.Utf8,
        "freeze_label": pl.Utf8,
        "drug_placement": pl.Utf8,
        "emission_key": pl.Utf8,
        "emission_label": pl.Utf8,
        "transition_key": pl.Utf8,
        "transition_label": pl.Utf8,
        "score": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
        "n_free_params": pl.Int64,
        "n_trials": pl.Int64,
    }
    if model_metrics.is_empty():
        score_col = "test_ll_per_trial_mean"
        score_df = pl.DataFrame(schema=score_schema)
        score_summary = pl.DataFrame()
        best_score_models = pl.DataFrame()
        best_bic_models = pl.DataFrame()
        metrics_output = mo.md("No matching GLM-HMM-T drug-grid metrics found yet.")
    else:
        score_col = score_column(model_metrics)
        score_df = (
            model_metrics
            .with_columns((pl.col(score_col) / math.log(2.0)).alias("score"))
            .select([col for col in score_schema if col != "n_trials"])
            .join(subject_trial_counts, on="subject", how="left")
            .with_columns(pl.col("n_trials").cast(pl.Int64))
            .sort(["model_order", "subject"])
        )
        score_summary = (
            score_df
            .group_by(
                [
                    "model_order",
                    "model_id",
                    "freeze_label",
                    "drug_placement",
                    "emission_label",
                    "transition_label",
                    "n_free_params",
                ]
            )
            .agg(
                pl.mean("score").alias("mean_score_bits"),
                pl.std("score").alias("sd_score_bits"),
                pl.mean("bic").alias("mean_bic"),
                pl.mean("acc").alias("mean_acc"),
                pl.len().alias("n_subjects"),
            )
            .sort("mean_score_bits", descending=True)
        )
        best_score_models = score_summary.head(15)
        best_bic_models = score_summary.sort("mean_bic").head(15)
        metrics_output = best_score_models
    metrics_output
    return best_bic_models, score_col, score_df, score_summary


@app.cell(hide_code=True)
def _(mo, score_col):
    mo.md(f"""
    ## Model ranking

    Score column: `{score_col}`. Log-likelihood values are shown as bits per trial.
    """)
    return


@app.cell
def _(best_bic_models):
    best_bic_models
    return


@app.cell
def _(pl, score_df):
    if score_df.is_empty():
        delta_vs_no_drug_df = pl.DataFrame()
        delta_vs_free_stim_df = pl.DataFrame()
        delta_summary = pl.DataFrame()
        freeze_summary = pl.DataFrame()
    else:
        _no_drug = (
            score_df
            .filter((pl.col("emission_key") == "none") & (pl.col("transition_key") == "none"))
            .select(
                [
                    "subject",
                    "freeze_key",
                    pl.col("score").alias("no_drug_score"),
                    pl.col("bic").alias("no_drug_bic"),
                ]
            )
        )
        delta_vs_no_drug_df = (
            score_df
            .join(_no_drug, on=["subject", "freeze_key"], how="inner")
            .with_columns(
                (pl.col("score") - pl.col("no_drug_score")).alias("delta_score_vs_no_drug"),
                (pl.col("bic") - pl.col("no_drug_bic")).alias("delta_bic_vs_no_drug"),
            )
            .sort(["model_order", "subject"])
        )
        delta_summary = (
            delta_vs_no_drug_df
            .group_by(
                [
                    "model_order",
                    "freeze_label",
                    "drug_placement",
                    "emission_label",
                    "transition_label",
                    "n_free_params",
                ]
            )
            .agg(
                pl.mean("delta_score_vs_no_drug").alias("mean_delta_score_bits"),
                pl.std("delta_score_vs_no_drug").alias("sd_delta_score_bits"),
                pl.mean("delta_bic_vs_no_drug").alias("mean_delta_bic"),
                pl.len().alias("n_subjects"),
            )
            .sort("mean_delta_score_bits", descending=True)
        )
        _stim_free = (
            score_df
            .filter(pl.col("freeze_key") == "stim_free")
            .select(
                [
                    "subject",
                    "emission_key",
                    "transition_key",
                    pl.col("score").alias("stim_free_score"),
                    pl.col("bic").alias("stim_free_bic"),
                ]
            )
        )
        delta_vs_free_stim_df = (
            score_df
            .filter(pl.col("freeze_key") == "stim_fixed")
            .join(_stim_free, on=["subject", "emission_key", "transition_key"], how="inner")
            .with_columns(
                (pl.col("score") - pl.col("stim_free_score")).alias("delta_score_vs_stim_free"),
                (pl.col("bic") - pl.col("stim_free_bic")).alias("delta_bic_vs_stim_free"),
            )
            .sort(["model_order", "subject"])
        )
        freeze_summary = (
            delta_vs_free_stim_df
            .group_by(["drug_placement", "emission_label", "transition_label", "n_free_params"])
            .agg(
                pl.mean("delta_score_vs_stim_free").alias("mean_delta_score_bits"),
                pl.mean("delta_bic_vs_stim_free").alias("mean_delta_bic"),
                pl.len().alias("n_subjects"),
            )
            .sort("mean_delta_score_bits", descending=True)
        )
    return delta_summary, freeze_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Drug terms vs no-drug baseline
    """)
    return


@app.cell
def _(delta_summary):
    delta_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fixed stimulus vs free stimulus
    """)
    return


@app.cell
def _(freeze_summary):
    freeze_summary
    return


@app.function
def clean_lineplot_edges(ax):
    for line in ax.lines:
        line.set_markeredgewidth(0)
        line.set_markeredgecolor("none")
    for collection in ax.collections:
        collection.set_edgecolor("none")
        collection.set_linewidth(0)


@app.cell
def _(figsize, mo, path_panels, plt, score_summary, sns):
    if score_summary.is_empty():
        top_score_ax = mo.md("No score summary rows to plot.")
    else:
        _top = score_summary.head(15).sort("mean_score_bits").to_pandas()
        _labels = (
            _top["freeze_label"]
            + " | "
            + _top["drug_placement"]
            + " | E:"
            + _top["emission_label"]
            + " | T:"
            + _top["transition_label"]
        )
        plt.figure(figsize=figsize, constrained_layout=True)
        top_score_ax = plt.gca()
        sns.barplot(
            data=_top.assign(plot_label=_labels),
            x="mean_score_bits",
            y="plot_label",
            color="0.25",
            ax=top_score_ax,
        )
        top_score_ax.set_xlabel("Mean LL (bits/trial)")
        top_score_ax.set_ylabel("")
        sns.despine(ax=top_score_ax)
        top_score_ax.figure.savefig((path_panels / "top_mean_score").with_suffix(".svg"))
        top_score_ax.figure.savefig((path_panels / "top_mean_score").with_suffix(".png"))
    top_score_ax
    return


@app.cell
def _(delta_summary, heatmap_figsize, mo, path_panels, plt, sns):
    if delta_summary.is_empty():
        delta_heatmap_ax = mo.md("No delta summary rows to plot.")
    else:
        _df = delta_summary.to_pandas()
        _df = _df.loc[_df["freeze_label"] == "stim free"]
        if _df.empty:
            delta_heatmap_ax = mo.md("No stim-free delta rows to plot.")
        else:
            _pivot = _df.pivot_table(
                index="emission_label",
                columns="transition_label",
                values="mean_delta_score_bits",
                aggfunc="mean",
            )
            plt.figure(figsize=heatmap_figsize, constrained_layout=True)
            delta_heatmap_ax = plt.gca()
            sns.heatmap(
                _pivot,
                center=0,
                cmap="vlag",
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Delta LL vs no drug"},
                ax=delta_heatmap_ax,
            )
            delta_heatmap_ax.set_xlabel("Transition drug terms")
            delta_heatmap_ax.set_ylabel("Emission drug terms")
            delta_heatmap_ax.set_title("Stim free")
            delta_heatmap_ax.figure.savefig((path_panels / "stim_free_delta_heatmap").with_suffix(".svg"))
            delta_heatmap_ax.figure.savefig((path_panels / "stim_free_delta_heatmap").with_suffix(".png"))
    delta_heatmap_ax
    return


@app.cell
def _(delta_summary, heatmap_figsize, mo, path_panels, plt, sns):
    if delta_summary.is_empty():
        fixed_delta_heatmap_ax = mo.md("No delta summary rows to plot.")
    else:
        _df = delta_summary.to_pandas()
        _df = _df.loc[_df["freeze_label"] == "stim fixed"]
        if _df.empty:
            fixed_delta_heatmap_ax = mo.md("No stim-fixed delta rows to plot.")
        else:
            _pivot = _df.pivot_table(
                index="emission_label",
                columns="transition_label",
                values="mean_delta_score_bits",
                aggfunc="mean",
            )
            plt.figure(figsize=heatmap_figsize, constrained_layout=True)
            fixed_delta_heatmap_ax = plt.gca()
            sns.heatmap(
                _pivot,
                center=0,
                cmap="vlag",
                annot=True,
                fmt=".3f",
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": "Delta LL vs no drug"},
                ax=fixed_delta_heatmap_ax,
            )
            fixed_delta_heatmap_ax.set_xlabel("Transition drug terms")
            fixed_delta_heatmap_ax.set_ylabel("Emission drug terms")
            fixed_delta_heatmap_ax.set_title("Stim fixed")
            fixed_delta_heatmap_ax.figure.savefig((path_panels / "stim_fixed_delta_heatmap").with_suffix(".svg"))
            fixed_delta_heatmap_ax.figure.savefig((path_panels / "stim_fixed_delta_heatmap").with_suffix(".png"))
    fixed_delta_heatmap_ax
    return


if __name__ == "__main__":
    app.run()
