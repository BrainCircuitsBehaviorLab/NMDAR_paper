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
    from scipy.stats import ttest_1samp

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
        ttest_1samp,
    )


@app.cell
def _(plt, project_root):
    plt.style.use(project_root / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return


@app.cell
def _(fig_size, project_root):
    path_panels = project_root / "figures" / "panels_glmhmmt_transition_regressor_comparison"
    path_panels.mkdir(parents=True, exist_ok=True)
    curve_figsize = fig_size(2, 2)
    term_figsize = fig_size(2, 1)
    return curve_figsize, path_panels, term_figsize


@app.cell
def _(np, pd, ttest_1samp):
    def _stars(pvalue):
        if not np.isfinite(pvalue) or pvalue >= 0.05:
            return "ns"
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_one_sample_zero_annotations(ax, df, *, x, y, order, show_ns=True):
        if df is None or df.empty or not {x, y}.issubset(df.columns):
            return
        y_values = pd.to_numeric(df[y], errors="coerce")
        finite = y_values[np.isfinite(y_values)]
        if finite.empty:
            return
        y_min = float(finite.min())
        y_max = float(finite.max())
        pad = max((y_max - y_min) * 0.08, 0.002)
        text_y = y_max + pad
        ax.set_ylim(top=text_y + pad)

        for x_value in order:
            values = pd.to_numeric(df.loc[df[x] == x_value, y], errors="coerce").dropna()
            if len(values) < 2 or float(values.std()) == 0:
                label = "ns"
            else:
                pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=0.0).pvalue)
                label = _stars(pvalue)
            if label != "ns" or show_ns:
                ax.text(x_value, text_y, label, ha="center", va="bottom")

    return (add_one_sample_zero_annotations,)


@app.function
def clean_lineplot_edges(ax):
    for line in ax.lines:
        line.set_markeredgewidth(0)
        line.set_markeredgecolor("none")
    for collection in ax.collections:
        collection.set_edgecolor("none")
        collection.set_linewidth(0)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # GLM-HMM-T transition regressor comparison

    Compare no-drug GLM-HMM-T transition designs for `2AFC` and `2ADC`
    (`2AFC_delay` in the fit folders). Emission constraints are selected once,
    then transition regressors are tested alone and in combinations.
    """)
    return


@app.cell
def _(itertools, pd):
    K = 2
    TAU = 50
    TASK_CONFIGS = {
        "2AFC": {
            "label": "2AFC",
            "emission_cols": ["bias", "stim_param", "choice_lag_param"],
            "stim_feature": "stim_param",
            "history_feature": "choice_lag_param",
            "candidate_terms": [
                "trial_index",
                "filtered_reward",
                "cumulative_reward",
                "prev_difficulty",
                "filtered_difficulty",
            ],
            "default_terms": [
                "trial_index",
                "filtered_reward",
                "cumulative_reward",
                "prev_difficulty",
                "filtered_difficulty",
            ],
            "term_labels": {
                "trial_index": "Trial index",
                "filtered_reward": "Filtered reward",
                "cumulative_reward": "Cumulative reward",
                "prev_difficulty": "Previous difficulty",
                "filtered_difficulty": "Filtered difficulty",
            },
        },
        "2AFC_delay": {
            "label": "2ADC",
            "emission_cols": ["bias", "stim_x_delay_param", "choice_lag_param"],
            "stim_feature": "stim_x_delay_param",
            "history_feature": "choice_lag_param",
            "candidate_terms": [
                "trial_index",
                "filtered_reward",
                "cumulative_reward",
                "prev_difficulty",
                "filtered_difficulty",
            ],
            "default_terms": [
                "trial_index",
                "filtered_reward",
                "cumulative_reward",
                "prev_difficulty",
                "filtered_difficulty",
            ],
            "term_labels": {
                "trial_index": "Trial index",
                "filtered_reward": "Filtered reward",
                "cumulative_reward": "Cumulative reward",
                "prev_difficulty": "Previous difficulty",
                "filtered_difficulty": "Filtered difficulty",
            },
        },
    }
    TASK_OPTIONS = list(TASK_CONFIGS)

    def _terms_key(terms: tuple[str, ...]) -> str:
        return "|".join(terms) if terms else "intercept_only"

    def _model_key(terms: tuple[str, ...]) -> str:
        return "_".join(terms) if terms else "intercept_only"

    def _term_label(task_name: str, term: str) -> str:
        return TASK_CONFIGS[task_name]["term_labels"].get(term, term)

    def _combo_label(task_name: str, terms: tuple[str, ...]) -> str:
        if not terms:
            return "Transition intercept"
        return " + ".join(_term_label(task_name, term) for term in terms)

    def _parent_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
        return terms[:-1]

    EMISSION_CONSTRAINTS = {
        "free": {
            "label": "Free",
            "model_prefix": "transreg",
            "description": "Stimulus and choice-history emission weights free in both raw states.",
        },
        "stim0": {
            "label": "Stim0",
            "model_prefix": "transreg_stim0",
            "description": "Stimulus emission weight frozen to 0 in raw state 0.",
        },
        "stim0_choice1": {
            "label": "Both0",
            "model_prefix": "transreg_stim0_choice1",
            "description": "Stimulus frozen to 0 in raw state 0; choice-history frozen to 0 in raw state 1.",
        },
    }
    EMISSION_CONSTRAINT_OPTIONS = list(EMISSION_CONSTRAINTS)

    def _frozen_emissions(task_name: str, emission_constraint: str) -> dict[str, dict[str, float]]:
        cfg = TASK_CONFIGS[task_name]
        stim_feature = cfg["stim_feature"]
        history_feature = cfg["history_feature"]
        if emission_constraint == "free":
            return {}
        if emission_constraint == "stim0":
            return {"0": {stim_feature: 0.0}}
        if emission_constraint == "stim0_choice1":
            return {"0": {stim_feature: 0.0}, "1": {history_feature: 0.0}}
        raise ValueError(f"Unknown emission constraint: {emission_constraint}")

    def _constraint_model_id(emission_constraint: str, terms: tuple[str, ...]) -> str:
        prefix = EMISSION_CONSTRAINTS[emission_constraint]["model_prefix"]
        return f"{prefix}_{_model_key(terms)}"

    def build_model_specs(task_name: str, selected_terms: list[str], emission_constraint: str) -> list[dict]:
        cfg = TASK_CONFIGS[task_name]
        constraint = EMISSION_CONSTRAINTS[emission_constraint]
        frozen_emissions = _frozen_emissions(task_name, emission_constraint)
        ordered_terms = [term for term in cfg["candidate_terms"] if term in set(selected_terms)]
        raw_specs = []
        for size in range(len(ordered_terms) + 1):
            for terms in itertools.combinations(ordered_terms, size):
                parent = _parent_terms(terms)
                raw_specs.append(
                    {
                        "task": task_name,
                        "model_id": _constraint_model_id(emission_constraint, terms),
                        "parent_model_id": (
                            _constraint_model_id(emission_constraint, parent)
                            if terms
                            else None
                        ),
                        "emission_constraint": emission_constraint,
                        "emission_constraint_label": constraint["label"],
                        "frozen_emissions": frozen_emissions,
                        "terms_key": _terms_key(terms),
                        "parent_terms_key": _terms_key(parent) if terms else None,
                        "n_terms": len(terms),
                        "combo_label": _combo_label(task_name, terms),
                        "added_term": _term_label(task_name, terms[-1]) if terms else "baseline",
                        "emission_cols": list(cfg["emission_cols"]),
                        "transition_cols": list(terms),
                    }
                )
        specs = []
        for order, spec in enumerate(raw_specs):
            spec = dict(spec)
            spec["model_order"] = order
            specs.append(spec)
        return specs

    model_defaults = pd.DataFrame(
        [
            {
                "task": task_name,
                "label": cfg["label"],
                "fixed_emissions": ", ".join(cfg["emission_cols"]),
                "default_transition_regressors": ", ".join(cfg["default_terms"]),
                "all_transition_regressors": ", ".join(cfg["candidate_terms"]),
            }
            for task_name, cfg in TASK_CONFIGS.items()
        ]
    )
    return (
        EMISSION_CONSTRAINTS,
        EMISSION_CONSTRAINT_OPTIONS,
        K,
        TASK_CONFIGS,
        TASK_OPTIONS,
        TAU,
        build_model_specs,
        model_defaults,
    )


@app.cell
def _(EMISSION_CONSTRAINT_OPTIONS, TASK_OPTIONS, mo):
    ui_task = mo.ui.dropdown(options=TASK_OPTIONS, value="2AFC", label="Task")
    ui_emission_constraint = mo.ui.dropdown(
        options=EMISSION_CONSTRAINT_OPTIONS,
        value="free",
        label="Emission constraint",
    )
    ui_cv_mode = mo.ui.dropdown(
        options=["balanced_session_holdout", "none"],
        value="balanced_session_holdout",
        label="Fit score",
    )
    ui_num_iters = mo.ui.number(start=1, stop=500, step=1, value=50, label="EM iterations")
    ui_n_restarts = mo.ui.number(start=1, stop=20, step=1, value=1, label="Restarts")
    ui_run_grid = mo.ui.run_button(label="Fit selected transition grid")
    return (
        ui_cv_mode,
        ui_emission_constraint,
        ui_n_restarts,
        ui_num_iters,
        ui_run_grid,
        ui_task,
    )


@app.cell
def _(TASK_CONFIGS, get_adapter, mo, pl, ui_task):
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
    candidate_options = [
        term
        for term in TASK_CONFIGS[task_name]["candidate_terms"]
    ]
    default_candidate_terms = [
        term
        for term in TASK_CONFIGS[task_name]["default_terms"]
        if term in candidate_options
    ]
    baseline_class_idx = int(adapter.baseline_class_idx)
    ui_candidate_terms = mo.ui.multiselect(
        options=candidate_options,
        value=default_candidate_terms,
        label="Transition regressors",
    )
    return (
        all_subjects,
        baseline_class_idx,
        condition_counts,
        subject_trial_counts,
        task_name,
        ui_candidate_terms,
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
    EMISSION_CONSTRAINTS,
    TASK_CONFIGS,
    build_model_specs,
    json,
    pd,
    task_name,
    ui_candidate_terms,
    ui_emission_constraint,
):
    selected_candidate_terms = list(ui_candidate_terms.value)
    emission_constraint = str(ui_emission_constraint.value)
    model_specs = build_model_specs(task_name, selected_candidate_terms, emission_constraint)
    selected_emission_constraint = EMISSION_CONSTRAINTS[emission_constraint]
    selected_frozen_emissions = model_specs[0]["frozen_emissions"] if model_specs else {}
    emission_constraint_table = pd.DataFrame(
        [
            {
                "selected_emission_constraint": selected_emission_constraint["label"],
                "description": selected_emission_constraint["description"],
                "stim_feature": TASK_CONFIGS[task_name]["stim_feature"],
                "history_feature": TASK_CONFIGS[task_name]["history_feature"],
                "frozen_emissions": json.dumps(selected_frozen_emissions, sort_keys=True),
            }
        ]
    )
    model_table = pd.DataFrame(
        [
            {
                "order": spec["model_order"],
                "emission_constraint": spec["emission_constraint_label"],
                "n_terms": spec["n_terms"],
                "combo": spec["combo_label"],
                "added_term": spec["added_term"],
                "model_id": spec["model_id"],
                "parent_model_id": spec["parent_model_id"],
                "emission_cols": ", ".join(spec["emission_cols"]),
                "frozen_emissions": json.dumps(spec["frozen_emissions"], sort_keys=True),
                "transition_cols": ", ".join(spec["transition_cols"]),
            }
            for spec in model_specs
        ]
    )
    model_counts = pd.DataFrame(
        [
            {
                "task": task_name,
                "emission_constraint": selected_emission_constraint["label"],
                "selected_transition_regressors": ", ".join(selected_candidate_terms),
                "n_selected_transition_regressors": len(selected_candidate_terms),
                "n_models": len(model_specs),
            }
        ]
    )
    return (
        emission_constraint_table,
        model_counts,
        model_specs,
        model_table,
        selected_candidate_terms,
    )


@app.cell
def _(
    condition_counts,
    emission_constraint_table,
    mo,
    model_counts,
    model_defaults,
    model_table,
    ui_candidate_terms,
    ui_cv_mode,
    ui_emission_constraint,
    ui_n_restarts,
    ui_num_iters,
    ui_run_grid,
    ui_subjects,
    ui_task,
):
    mo.vstack(
        [
            mo.hstack([ui_task, ui_emission_constraint, ui_cv_mode, ui_num_iters, ui_n_restarts]),
            ui_candidate_terms,
            ui_subjects,
            ui_run_grid,
            condition_counts,
            model_defaults,
            emission_constraint_table,
            model_counts,
            model_table,
        ]
    )
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
        n_transition = k * (k - 1) + k * k * len(spec["transition_cols"])
        return int(n_transition + n_emission - n_frozen * (num_classes - 1))

    def _normalise_frozen(value) -> dict:
        if not value:
            return {}
        return {
            str(state): {str(name): float(weight) for name, weight in features.items()}
            for state, features in dict(value).items()
        }

    def subject_arrays_match_spec(spec: dict, model_dir, metrics_path) -> bool:
        subject = metrics_path.name.split("_K", maxsplit=1)[0]
        arrays_path = model_dir / f"{subject}_K{K}_glmhmmt_arrays.npz"
        if not arrays_path.exists():
            return False
        try:
            with np.load(arrays_path, allow_pickle=True) as arrays:
                x_cols = [str(col) for col in arrays["X_cols"]]
                u_cols = [str(col) for col in arrays["U_cols"]]
                if "frozen_emissions_json" in arrays.files:
                    frozen_json = str(arrays["frozen_emissions_json"].item())
                else:
                    frozen_json = ""
        except Exception:
            return False
        try:
            array_frozen = json.loads(frozen_json) if frozen_json else {}
        except Exception:
            return False
        return (
            x_cols == list(spec["emission_cols"])
            and u_cols == list(spec["transition_cols"])
            and _normalise_frozen(array_frozen) == _normalise_frozen(spec.get("frozen_emissions", {}))
        )

    def read_metrics(model_specs: list[dict]) -> pl.DataFrame:
        frames = []
        for spec in model_specs:
            model_dir = paths.RESULTS / "fits" / task_name / "glmhmmt" / spec["model_id"]
            if not model_dir.exists():
                continue
            for path in sorted(model_dir.glob("*_glmhmmt_metrics.parquet")):
                if not subject_arrays_match_spec(spec, model_dir, path):
                    continue
                frame = pl.read_parquet(path).with_columns(
                    pl.col("subject").cast(pl.Utf8),
                    pl.lit(spec["model_id"]).alias("model_id"),
                    pl.lit(spec["parent_model_id"], dtype=pl.Utf8).alias("parent_model_id"),
                    pl.lit(spec["emission_constraint"]).alias("emission_constraint"),
                    pl.lit(spec["emission_constraint_label"]).alias("emission_constraint_label"),
                    pl.lit(json.dumps(spec["frozen_emissions"], sort_keys=True)).alias("frozen_emissions_json"),
                    pl.lit(spec["terms_key"]).alias("terms_key"),
                    pl.lit(spec["parent_terms_key"], dtype=pl.Utf8).alias("parent_terms_key"),
                    pl.lit(spec["combo_label"]).alias("model_label"),
                    pl.lit(spec["combo_label"]).alias("combo_label"),
                    pl.lit(spec["added_term"]).alias("added_term"),
                    pl.lit(spec["model_order"], dtype=pl.Int64).alias("model_order"),
                    pl.lit(spec["n_terms"], dtype=pl.Int64).alias("n_terms"),
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
            title="Fitting GLM-HMM-T transition grid",
            subtitle=f"{len(model_specs)} models x {len(subjects)} subjects",
            completion_title="Transition grid complete",
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
    model_specs,
    ui_cv_mode,
    ui_n_restarts,
    ui_num_iters,
    ui_run_grid,
    ui_subjects,
):
    if ui_run_grid.value:
        fit_grid(
            model_specs=model_specs,
            subjects=list(ui_subjects.value),
            cv_mode=ui_cv_mode.value,
            num_iters=int(ui_num_iters.value),
            n_restarts=int(ui_n_restarts.value),
        )
        fit_output = mo.md("Selected transition grid fitted.")
    else:
        fit_output = mo.md("Saved fits are loaded below. Press the run button to fit or refresh the selected transition grid.")
    fit_output
    return


@app.cell
def _(model_specs, read_metrics):
    model_metrics = read_metrics(model_specs)
    return (model_metrics,)


@app.cell
def _(math, model_metrics, pl, score_column, subject_trial_counts):
    score_schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "parent_model_id": pl.Utf8,
        "emission_constraint": pl.Utf8,
        "emission_constraint_label": pl.Utf8,
        "frozen_emissions_json": pl.Utf8,
        "terms_key": pl.Utf8,
        "parent_terms_key": pl.Utf8,
        "model_label": pl.Utf8,
        "combo_label": pl.Utf8,
        "added_term": pl.Utf8,
        "model_order": pl.Int64,
        "n_terms": pl.Int64,
        "score": pl.Float64,
        "raw_ll": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
        "n_free_params": pl.Int64,
        "n_trials": pl.Int64,
    }
    if model_metrics.is_empty():
        score_col = "test_ll_per_trial_mean"
        score_df = pl.DataFrame(schema=score_schema)
    else:
        score_col = score_column(model_metrics)
        _metrics = model_metrics.with_columns((pl.col(score_col) / math.log(2.0)).alias("score"))
        for col in ["raw_ll", "bic", "acc"]:
            if col not in _metrics.columns:
                _metrics = _metrics.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        score_df = (
            _metrics
            .select([col for col in score_schema if col != "n_trials"])
            .join(subject_trial_counts, on="subject", how="left")
            .with_columns(pl.col("n_trials").cast(pl.Int64))
            .sort(["model_order", "subject"])
        )
    return score_col, score_df


@app.cell
def _(np, pl, score_df, selected_candidate_terms, ttest_1samp):
    delta_schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "model_order": pl.Int64,
        "n_terms": pl.Int64,
        "terms_key": pl.Utf8,
        "combo_label": pl.Utf8,
        "score": pl.Float64,
        "raw_ll": pl.Float64,
        "bic": pl.Float64,
        "n_free_params": pl.Int64,
        "n_trials": pl.Int64,
        "delta_score_vs_base": pl.Float64,
        "delta_raw_ll_vs_base": pl.Float64,
        "delta_bic_vs_base": pl.Float64,
        "param_delta_vs_base": pl.Int64,
    }
    term_gain_schema = {
        "subject": pl.Utf8,
        "term": pl.Utf8,
        "with_terms_key": pl.Utf8,
        "without_terms_key": pl.Utf8,
        "score_with": pl.Float64,
        "score_without": pl.Float64,
        "delta_score": pl.Float64,
    }

    def _split_terms(key: str) -> tuple[str, ...]:
        if key == "intercept_only" or key is None:
            return ()
        return tuple(str(key).split("|"))

    def _terms_key(terms: tuple[str, ...]) -> str:
        return "|".join(terms) if terms else "intercept_only"

    def _stars(pvalue):
        if not np.isfinite(pvalue) or pvalue >= 0.05:
            return "ns"
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    if score_df.is_empty():
        delta_vs_base_df = pl.DataFrame(schema=delta_schema)
        term_gain_summary = pl.DataFrame()
    else:
        _base = (
            score_df
            .filter(pl.col("terms_key") == "intercept_only")
            .select(
                [
                    "subject",
                    pl.col("score").alias("base_score"),
                    pl.col("raw_ll").alias("base_raw_ll"),
                    pl.col("bic").alias("base_bic"),
                    pl.col("n_free_params").alias("base_n_free_params"),
                ]
            )
        )
        delta_vs_base_df = (
            score_df
            .join(_base, on="subject", how="inner")
            .with_columns(
                (pl.col("score") - pl.col("base_score")).alias("delta_score_vs_base"),
                (pl.col("raw_ll") - pl.col("base_raw_ll")).alias("delta_raw_ll_vs_base"),
                (pl.col("bic") - pl.col("base_bic")).alias("delta_bic_vs_base"),
                (pl.col("n_free_params") - pl.col("base_n_free_params")).cast(pl.Int64).alias("param_delta_vs_base"),
            )
            .select(list(delta_schema))
            .sort(["model_order", "subject"])
        )

        score_pd = score_df.to_pandas()
        rows = []
        lookup = {
            (str(_row.subject), str(_row.terms_key)): _row
            for _row in score_pd.itertuples(index=False)
        }
        for _row in score_pd.itertuples(index=False):
            terms = _split_terms(str(_row.terms_key))
            for term in terms:
                without_terms = tuple(item for item in terms if item != term)
                without_key = _terms_key(without_terms)
                base = lookup.get((str(_row.subject), without_key))
                if base is None:
                    continue
                rows.append(
                    {
                        "subject": str(_row.subject),
                        "term": term,
                        "with_terms_key": str(_row.terms_key),
                        "without_terms_key": without_key,
                        "score_with": float(_row.score),
                        "score_without": float(base.score),
                        "delta_score": float(_row.score) - float(base.score),
                    }
                )
        term_gain_df = pl.DataFrame(rows, schema=term_gain_schema) if rows else pl.DataFrame(schema=term_gain_schema)
        if term_gain_df.is_empty():
            term_gain_summary = pl.DataFrame()
        else:
            summary_rows = []
            for term in selected_candidate_terms:
                values = (
                    term_gain_df
                    .filter(pl.col("term") == term)
                    .get_column("delta_score")
                    .to_numpy()
                )
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    continue
                if len(values) < 2 or float(np.std(values)) == 0:
                    pvalue = float("nan")
                else:
                    pvalue = float(ttest_1samp(values, popmean=0.0).pvalue)
                mean_delta = float(np.mean(values))
                summary_rows.append(
                    {
                        "term": term,
                        "mean_marginal_delta_bits": mean_delta,
                        "sd_marginal_delta_bits": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n_paired_context_subjects": int(len(values)),
                        "delta_bits_10k_trials": mean_delta * 10000.0,
                        "log10_likelihood_ratio_10k": mean_delta * 10000.0 * np.log10(2.0),
                        "pvalue_vs_zero": pvalue,
                        "significance": _stars(pvalue),
                    }
                )
            term_gain_summary = (
                pl.DataFrame(summary_rows).sort("mean_marginal_delta_bits", descending=True)
                if summary_rows
                else pl.DataFrame()
            )
    return delta_vs_base_df, term_gain_summary


@app.cell
def _(
    add_one_sample_zero_annotations,
    curve_figsize,
    delta_vs_base_df,
    mo,
    path_panels,
    pl,
    plt,
    score_col,
    sns,
):
    if delta_vs_base_df.is_empty():
        delta_base_ax = mo.md("No delta-vs-base scores to plot.")
    else:
        _df = delta_vs_base_df.filter(pl.col("model_order") > 0).to_pandas()
        if _df.empty:
            delta_base_ax = mo.md("No non-base delta-vs-base scores to plot.")
        else:
            _ticks = (
                delta_vs_base_df
                .filter(pl.col("model_order") > 0)
                .select(["model_order", "combo_label"])
                .unique()
                .sort("model_order")
            )
            _positions = _ticks.get_column("model_order").to_list()
            _labels = _ticks.get_column("combo_label").to_list()
            plt.figure(figsize=curve_figsize, constrained_layout=True)
            delta_base_ax = plt.gca()
            sns.lineplot(
                data=_df,
                x="model_order",
                y="delta_score_vs_base",
                units="subject",
                estimator=None,
                color="0.84",
                linewidth=0.6,
                sort=False,
                ax=delta_base_ax,
            )
            sns.lineplot(
                data=_df,
                x="model_order",
                y="delta_score_vs_base",
                errorbar=("se", 1),
                marker="o",
                markersize=3.5,
                markeredgewidth=0,
                markeredgecolor="none",
                err_kws={"edgecolor": "none", "linewidth": 0},
                color="black",
                linewidth=1,
                sort=False,
                ax=delta_base_ax,
            )
            delta_base_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
            add_one_sample_zero_annotations(
                delta_base_ax,
                _df,
                x="model_order",
                y="delta_score_vs_base",
                order=_positions,
                show_ns=True,
            )
            _score_label = "CV test LL" if score_col == "test_ll_per_trial_mean" else "LL"
            delta_base_ax.set_xlabel("Transition model")
            delta_base_ax.set_ylabel(f"Delta {_score_label} vs transition intercept (bits/trial)")
            delta_base_ax.set_xticks(_positions)
            delta_base_ax.set_xticklabels(_labels, rotation=45, ha="right")
            clean_lineplot_edges(delta_base_ax)
            sns.despine(ax=delta_base_ax)
            delta_base_ax.figure.savefig((path_panels / "delta_vs_transition_intercept").with_suffix(".svg"))
            delta_base_ax.figure.savefig((path_panels / "delta_vs_transition_intercept").with_suffix(".png"))
    delta_base_ax
    return


@app.cell
def _(mo, path_panels, plt, sns, term_figsize, term_gain_summary):
    if term_gain_summary.is_empty():
        term_gain_ax = mo.md("No marginal regressor-gain rows to plot.")
    else:
        _df = term_gain_summary.sort("mean_marginal_delta_bits").to_pandas()

        plt.figure(figsize=term_figsize, constrained_layout=True)
        term_gain_ax = plt.gca()

        sns.barplot(
            data=_df,
            x="term",
            y="mean_marginal_delta_bits",
            color="0.35",
            ax=term_gain_ax,
        )

        term_gain_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        term_gain_ax.set_xlabel("")
        term_gain_ax.set_ylabel("Mean marginal delta LL (bits/trial)")
        term_gain_ax.tick_params(axis="x", rotation=90)

        sns.despine(ax=term_gain_ax)

        term_gain_ax.figure.savefig((path_panels / "term_marginal_gain").with_suffix(".svg"))
        term_gain_ax.figure.savefig((path_panels / "term_marginal_gain").with_suffix(".png"))

    term_gain_ax
    return


if __name__ == "__main__":
    app.run()
