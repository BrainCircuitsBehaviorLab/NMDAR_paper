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
    from scipy.stats import chi2, ttest_1samp, ttest_rel

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
        chi2,
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
        ttest_rel,
    )


@app.cell
def _(plt, project_root):
    plt.style.use(project_root / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"
    return


@app.cell
def _(np, pd, ttest_1samp, ttest_rel):
    def _stars(pvalue):
        if not np.isfinite(pvalue) or pvalue >= 0.05:
            return "ns"
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_numeric_pair_annotations(
        ax,
        df,
        *,
        x,
        y,
        pairs,
        subject_col="subject",
        show_ns=True,
    ):
        if df is None or df.empty or not {x, y, subject_col}.issubset(df.columns):
            return
        y_values = pd.to_numeric(df[y], errors="coerce")
        finite = y_values[np.isfinite(y_values)]
        if finite.empty:
            return
        y_min = float(finite.min())
        y_max = float(finite.max())
        pad = max((y_max - y_min) * 0.08, 0.002)
        base_y = y_max + pad
        ax.set_ylim(top=base_y + pad * (len(pairs) + 1))

        annotation_idx = 0
        for left, right in pairs:
            sub = df[df[x].isin([left, right])]
            paired = sub.pivot_table(
                values=y,
                index=subject_col,
                columns=x,
                aggfunc="first",
            )
            if left not in paired.columns or right not in paired.columns:
                continue
            paired = paired.dropna(subset=[left, right])
            if len(paired) < 2:
                continue
            pvalue = float(ttest_rel(paired[left], paired[right]).pvalue)
            label = _stars(pvalue)
            if label == "ns" and not show_ns:
                continue
            y_pos = base_y + pad * annotation_idx
            ax.plot(
                [left, left, right, right],
                [y_pos, y_pos + pad * 0.25, y_pos + pad * 0.25, y_pos],
                color="0.35",
                linewidth=0.7,
            )
            ax.text((left + right) / 2, y_pos + pad * 0.3, label, ha="center", va="bottom")
            annotation_idx += 1

    def add_one_sample_zero_annotations(ax, df, *, x, y, order, show_ns=False):
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
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=0.0).pvalue)
            stars = _stars(pvalue)
            if stars != "ns" or show_ns:
                ax.text(x_value, text_y, stars, ha="center", va="bottom")

    return (add_one_sample_zero_annotations,)


@app.cell
def _(fig_size, project_root):
    path_panels = project_root / "figures" / "panels_glmhmmt_drug_final_model_comparison"
    path_panels.mkdir(parents=True, exist_ok=True)
    figsize = fig_size(2, 2.2)
    ranking_figsize = fig_size(2, 1.2)
    return figsize, path_panels, ranking_figsize


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # GLM-HMM-T drug final-model comparison

    This notebook fits a combination battery of 2-state GLM-HMM-T drug models
    for `2AFC_DRUG` and `2ADC_DRUG`. All models keep stimulus sensitivity fixed
    in one raw state and leave choice history free.
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

    def _short_feature_name(name: str) -> str:
        replacements = {
            "stim_param": "stim",
            "stim_x_delay_param": "stim_delay",
            "choice_lag_param": "choice",
            "filtered_reward": "filt_reward",
            "trial_index": "trial",
        }
        return replacements.get(name, name)

    def _term_label(term: str, cfg: dict, *, target: str) -> str:
        drug_col = cfg["drug_col"]
        if term == drug_col:
            return "T:drug"
        prefix = "drug_x_"
        if term.startswith(prefix):
            feature = _short_feature_name(term.removeprefix(prefix))
            return f"{target}:drug x {feature}"
        return term

    def drug_model_specs(task_name: str) -> list[dict]:
        cfg = TASK_CONFIGS[task_name]
        drug_col = cfg["drug_col"]
        stim_feature = cfg["stim_feature"]
        history_feature = cfg["history_feature"]
        stim_drug_term = f"drug_x_{stim_feature}"
        choice_drug_term = f"drug_x_{history_feature}"
        trial_drug_term = "drug_x_trial_index"
        reward_drug_term = "drug_x_filtered_reward"

        term_defs = {
            "E_stim": {
                "label": "E stim",
                "detail": _term_label(stim_drug_term, cfg, target="E"),
                "emission_cols": [stim_drug_term],
                "transition_cols": [],
            },
            "E_choice": {
                "label": "E choice",
                "detail": _term_label(choice_drug_term, cfg, target="E"),
                "emission_cols": [choice_drug_term],
                "transition_cols": [],
            },
            "T_drug": {
                "label": "T drug",
                "detail": _term_label(drug_col, cfg, target="T"),
                "emission_cols": [],
                "transition_cols": [drug_col],
            },
            "T_trial": {
                "label": "T trial",
                "detail": _term_label(trial_drug_term, cfg, target="T"),
                "emission_cols": [],
                "transition_cols": [trial_drug_term],
            },
            "T_reward": {
                "label": "T reward",
                "detail": _term_label(reward_drug_term, cfg, target="T"),
                "emission_cols": [],
                "transition_cols": [reward_drug_term],
            },
        }
        term_order = ["E_stim", "E_choice", "T_drug", "T_trial", "T_reward"]
        emission_levels = [
            tuple(combo)
            for size in range(3)
            for combo in itertools.combinations(["E_stim", "E_choice"], size)
        ]
        transition_levels = [
            (),
            ("T_drug",),
            ("T_drug", "T_trial"),
            ("T_drug", "T_reward"),
            ("T_drug", "T_trial", "T_reward"),
        ]

        def combo_key(terms: tuple[str, ...]) -> str:
            if not terms:
                return "base"
            parts = []
            if "E_stim" in terms and "E_choice" in terms:
                parts.append("E_stim_choice")
            elif "E_stim" in terms:
                parts.append("E_stim")
            elif "E_choice" in terms:
                parts.append("E_choice")
            if "T_drug" in terms:
                transition_key = "T_drug"
                if "T_trial" in terms:
                    transition_key += "_trial"
                if "T_reward" in terms:
                    transition_key += "_reward"
                parts.append(transition_key)
            return "_".join(parts)

        def sort_key(terms: tuple[str, ...]) -> tuple:
            return len(terms), tuple(term_order.index(term) for term in terms)

        def parent_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
            return terms[:-1]

        raw_specs = []
        for emission_terms, transition_terms in itertools.product(emission_levels, transition_levels):
            terms = tuple(term for term in term_order if term in {*emission_terms, *transition_terms})
            emission_drug_cols = [
                col
                for term in terms
                for col in term_defs[term]["emission_cols"]
            ]
            transition_drug_cols = [
                col
                for term in terms
                for col in term_defs[term]["transition_cols"]
            ]
            emission_cols = list(dict.fromkeys([*cfg["base_emission_cols"], *emission_drug_cols]))
            transition_cols = list(dict.fromkeys([*cfg["base_transition_cols"], *transition_drug_cols]))
            frozen_features = {stim_feature: 0.0}
            if stim_drug_term in emission_cols:
                frozen_features[stim_drug_term] = 0.0
            frozen_emissions = {cfg["stim_fixed_state"]: frozen_features}
            has_emission_drug = any(term.startswith("E_") for term in terms)
            has_transition_drug = any(term.startswith("T_") for term in terms)
            if has_emission_drug and has_transition_drug:
                placement = "emissions + transitions"
            elif has_emission_drug:
                placement = "emissions"
            elif has_transition_drug:
                placement = "transitions"
            else:
                placement = "no drug"
            key = combo_key(terms)
            parent = parent_terms(terms)
            raw_specs.append(
                {
                    "task": task_name,
                    "terms": terms,
                    "model_id": f"drugfinal_stim_fixed_{key}",
                    "parent_model_id": (
                        f"drugfinal_stim_fixed_{combo_key(parent)}"
                        if terms
                        else None
                    ),
                    "step_key": key,
                    "step_label": " + ".join(term_defs[term]["label"] for term in terms) or "Base",
                    "added_terms": term_defs[terms[-1]]["detail"] if terms else "baseline",
                    "n_drug_terms": len(terms),
                    "freeze_key": "stim_fixed",
                    "freeze_label": f"stim fixed state {cfg['stim_fixed_state']}",
                    "drug_placement": placement,
                    "emission_key": "+".join(term for term in terms if term.startswith("E_")) or "none",
                    "emission_label": ", ".join(term_defs[term]["detail"] for term in terms if term.startswith("E_")) or "none",
                    "transition_key": "+".join(term for term in terms if term.startswith("T_")) or "none",
                    "transition_label": ", ".join(term_defs[term]["detail"] for term in terms if term.startswith("T_")) or "none",
                    "short_label": " + ".join(term_defs[term]["label"] for term in terms) or "Base",
                    "emission_cols": emission_cols,
                    "transition_cols": transition_cols,
                    "frozen_emissions": frozen_emissions,
                }
            )
        specs = []
        for order, spec in enumerate(sorted(raw_specs, key=lambda item: sort_key(item["terms"]))):
            spec = dict(spec)
            spec["model_order"] = order
            spec.pop("terms")
            specs.append(spec)
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
                "max_drug_terms": max(spec["n_drug_terms"] for spec in specs),
                "n_stim_fixed": sum(spec["freeze_key"] == "stim_fixed" for spec in specs),
                "first_model": specs[0]["model_id"],
                "final_model": specs[-1]["model_id"],
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
                "n_drug_terms": spec["n_drug_terms"],
                "step": spec["step_label"],
                "added_terms": spec["added_terms"],
                "model_id": spec["model_id"],
                "parent_model_id": spec["parent_model_id"],
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
                    pl.lit(spec["parent_model_id"], dtype=pl.Utf8).alias("parent_model_id"),
                    pl.lit(spec["short_label"]).alias("model_label"),
                    pl.lit(spec["short_label"]).alias("short_label"),
                    pl.lit(spec["model_order"], dtype=pl.Int64).alias("model_order"),
                    pl.lit(spec["n_drug_terms"], dtype=pl.Int64).alias("n_drug_terms"),
                    pl.lit(spec["step_key"]).alias("step_key"),
                    pl.lit(spec["step_label"]).alias("step_label"),
                    pl.lit(spec["added_terms"]).alias("added_terms"),
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
            title="Fitting GLM-HMM-T final-model battery",
            subtitle=f"{len(model_specs)} models x {len(subjects)} subjects",
            completion_title="Final-model battery complete",
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
        fit_output = mo.md("Saved fits are loaded below. Press the run button to fit or refresh the selected task battery.")
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
        "parent_model_id": pl.Utf8,
        "model_label": pl.Utf8,
        "short_label": pl.Utf8,
        "model_order": pl.Int64,
        "n_drug_terms": pl.Int64,
        "step_key": pl.Utf8,
        "step_label": pl.Utf8,
        "added_terms": pl.Utf8,
        "freeze_key": pl.Utf8,
        "freeze_label": pl.Utf8,
        "drug_placement": pl.Utf8,
        "emission_key": pl.Utf8,
        "emission_label": pl.Utf8,
        "transition_key": pl.Utf8,
        "transition_label": pl.Utf8,
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
        score_summary = pl.DataFrame()
        best_score_models = pl.DataFrame()
        best_bic_models = pl.DataFrame()
        metrics_output = mo.md("No matching GLM-HMM-T final-model metrics found yet.")
    else:
        score_col = score_column(model_metrics)
        _metrics = model_metrics.with_columns((pl.col(score_col) / math.log(2.0)).alias("score"))
        if "raw_ll" not in _metrics.columns:
            _metrics = _metrics.with_columns(pl.lit(None, dtype=pl.Float64).alias("raw_ll"))
        score_df = (
            _metrics
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
                    "n_drug_terms",
                    "step_label",
                    "added_terms",
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
            .sort("model_order")
        )
        best_score_models = score_summary.sort("mean_score_bits", descending=True).head(15)
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
def _(chi2, np, pl, score_df):
    delta_vs_base_schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "parent_model_id": pl.Utf8,
        "model_order": pl.Int64,
        "n_drug_terms": pl.Int64,
        "step_label": pl.Utf8,
        "added_terms": pl.Utf8,
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
    nested_delta_schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "parent_model_id": pl.Utf8,
        "model_order": pl.Int64,
        "n_drug_terms": pl.Int64,
        "step_label": pl.Utf8,
        "added_terms": pl.Utf8,
        "score": pl.Float64,
        "parent_score": pl.Float64,
        "raw_ll": pl.Float64,
        "parent_raw_ll": pl.Float64,
        "bic": pl.Float64,
        "parent_bic": pl.Float64,
        "n_free_params": pl.Int64,
        "parent_n_free_params": pl.Int64,
        "n_trials": pl.Int64,
        "delta_score_vs_parent": pl.Float64,
        "delta_raw_ll_vs_parent": pl.Float64,
        "delta_bic_vs_parent": pl.Float64,
        "df": pl.Int64,
        "lr_stat": pl.Float64,
        "subject_lrt_pvalue": pl.Float64,
    }

    def _lrt_pvalue(row):
        lr_stat = row["lr_stat"]
        df = row["df"]
        if lr_stat is None or df is None or int(df) <= 0:
            return None
        if not np.isfinite(float(lr_stat)):
            return None
        return float(chi2.sf(max(float(lr_stat), 0.0), int(df)))

    if score_df.is_empty():
        delta_vs_base_df = pl.DataFrame(schema=delta_vs_base_schema)
        nested_delta_df = pl.DataFrame(schema=nested_delta_schema)
        delta_summary = pl.DataFrame()
        nested_delta_summary = pl.DataFrame()
        nested_lrt_summary = pl.DataFrame()
    else:
        _base = (
            score_df
            .filter(pl.col("model_order") == 0)
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
            .select(list(delta_vs_base_schema))
            .sort(["model_order", "subject"])
        )
        delta_summary = (
            delta_vs_base_df
            .group_by(["model_order", "model_id", "n_drug_terms", "step_label", "added_terms", "n_free_params"])
            .agg(
                pl.mean("delta_score_vs_base").alias("mean_delta_score_bits"),
                pl.std("delta_score_vs_base").alias("sd_delta_score_bits"),
                pl.mean("delta_bic_vs_base").alias("mean_delta_bic"),
                pl.first("param_delta_vs_base").alias("param_delta_vs_base"),
                pl.len().alias("n_subjects"),
            )
            .sort("model_order")
        )
        _parents = (
            score_df
            .select(
                [
                    "subject",
                    pl.col("model_id").alias("parent_model_id"),
                    pl.col("score").alias("parent_score"),
                    pl.col("raw_ll").alias("parent_raw_ll"),
                    pl.col("bic").alias("parent_bic"),
                    pl.col("n_free_params").alias("parent_n_free_params"),
                ]
            )
        )
        nested_delta_df = (
            score_df
            .filter(pl.col("parent_model_id").is_not_null())
            .join(_parents, on=["subject", "parent_model_id"], how="inner")
            .with_columns(
                (pl.col("score") - pl.col("parent_score")).alias("delta_score_vs_parent"),
                (pl.col("raw_ll") - pl.col("parent_raw_ll")).alias("delta_raw_ll_vs_parent"),
                (pl.col("bic") - pl.col("parent_bic")).alias("delta_bic_vs_parent"),
                (pl.col("n_free_params") - pl.col("parent_n_free_params")).cast(pl.Int64).alias("df"),
            )
            .with_columns((2.0 * pl.col("delta_raw_ll_vs_parent")).alias("lr_stat"))
            .with_columns(
                pl.struct(["lr_stat", "df"])
                .map_elements(_lrt_pvalue, return_dtype=pl.Float64)
                .alias("subject_lrt_pvalue")
            )
            .select(list(nested_delta_schema))
            .sort(["model_order", "subject"])
        )
        nested_delta_summary = (
            nested_delta_df
            .group_by(["model_order", "model_id", "n_drug_terms", "step_label", "added_terms", "n_free_params"])
            .agg(
                pl.mean("delta_score_vs_parent").alias("mean_delta_score_bits"),
                pl.std("delta_score_vs_parent").alias("sd_delta_score_bits"),
                pl.mean("delta_bic_vs_parent").alias("mean_delta_bic"),
                pl.first("df").alias("df_per_subject"),
                pl.len().alias("n_subjects"),
            )
            .sort("model_order")
        )
        nested_lrt_summary = (
            nested_delta_df
            .group_by(["model_order", "model_id", "n_drug_terms", "step_label", "added_terms"])
            .agg(
                pl.sum("lr_stat").alias("total_lr_stat"),
                pl.first("df").alias("df_per_subject"),
                pl.len().alias("n_subjects"),
                pl.mean("subject_lrt_pvalue").alias("mean_subject_lrt_pvalue"),
                pl.mean("delta_score_vs_parent").alias("mean_delta_score_bits"),
            )
            .with_columns((pl.col("df_per_subject") * pl.col("n_subjects")).cast(pl.Int64).alias("total_df"))
            .with_columns(
                pl.struct(["total_lr_stat", "total_df"])
                .map_elements(lambda row: _lrt_pvalue({"lr_stat": row["total_lr_stat"], "df": row["total_df"]}), return_dtype=pl.Float64)
                .alias("aggregate_lrt_pvalue")
            )
            .sort("model_order")
        )
    return (
        delta_summary,
        delta_vs_base_df,
        nested_delta_summary,
        nested_lrt_summary,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Drug terms vs base model
    """)
    return


@app.cell
def _(delta_summary):
    delta_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Nested parent comparisons
    """)
    return


@app.cell
def _(nested_delta_summary):
    nested_delta_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nested likelihood-ratio tests

    The LRT table is for in-sample comparisons against each model's canonical
    nested parent, defined by dropping the last term in the fixed term order.
    Use it when the battery was fit with `Fit score = none`; with
    cross-validation, the held-out LL plots below are the primary comparison.
    """)
    return


@app.cell
def _(nested_lrt_summary):
    nested_lrt_summary
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
def _(mo, path_panels, plt, ranking_figsize, score_summary, sns):
    if score_summary.is_empty():
        top_score_ax = mo.md("No score summary rows to plot.")
    else:
        _top = score_summary.sort("mean_score_bits").to_pandas()
        _labels = _top["step_label"] + " (" + _top["added_terms"] + ")"
        plt.figure(figsize=ranking_figsize, constrained_layout=True)
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
def _(figsize, mo, path_panels, plt, score_col, score_df, sns):
    if score_df.is_empty():
        nested_ll_ax = mo.md("No model scores to plot.")
    else:
        _df = score_df.to_pandas()
        _ticks = (
            score_df
            .select(["model_order", "step_label"])
            .unique()
            .sort("model_order")
        )
        _positions = _ticks.get_column("model_order").to_list()
        _labels = _ticks.get_column("step_label").to_list()
        plt.figure(figsize=figsize, constrained_layout=True)
        nested_ll_ax = plt.gca()
        sns.lineplot(
            data=_df,
            x="model_order",
            y="score",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.7,
            sort=False,
            ax=nested_ll_ax,
        )
        sns.lineplot(
            data=_df,
            x="model_order",
            y="score",
            errorbar=("se", 1),
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=nested_ll_ax,
        )
        nested_ll_ax.set_xlabel("Model combination")
        _score_label = "CV test LL" if score_col == "test_ll_per_trial_mean" else "LL"
        nested_ll_ax.set_ylabel(f"{_score_label} (bits/trial)")
        nested_ll_ax.set_xticks(_positions)
        nested_ll_ax.set_xticklabels(_labels, rotation=30, ha="right")
        clean_lineplot_edges(nested_ll_ax)
        sns.despine(ax=nested_ll_ax)
        nested_ll_ax.figure.savefig((path_panels / "nested_ll_curve").with_suffix(".svg"))
        nested_ll_ax.figure.savefig((path_panels / "nested_ll_curve").with_suffix(".png"))
    nested_ll_ax
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    delta_vs_base_df,
    fig_size,
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
                .select(["model_order", "step_label"])
                .unique()
                .sort("model_order")
            )
            _positions = _ticks.get_column("model_order").to_list()
            _labels = _ticks.get_column("step_label").to_list()
            plt.figure(figsize=fig_size(1,2), constrained_layout=True)
            delta_base_ax = plt.gca()
            sns.lineplot(
                data=_df,
                x="model_order",
                y="delta_score_vs_base",
                units="subject",
                estimator=None,
                color="0.84",
                linewidth=0.7,
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
            delta_base_ax.set_xlabel("Model combination")
            _score_label = "CV test LL" if score_col == "test_ll_per_trial_mean" else "LL"
            delta_base_ax.set_ylabel(f"Delta {_score_label} vs base (bits/trial)")
            delta_base_ax.set_xticks(_positions)
            delta_base_ax.set_xticklabels(_labels, rotation=30, ha="right")
            clean_lineplot_edges(delta_base_ax)
            sns.despine(ax=delta_base_ax)
            delta_base_ax.figure.savefig((path_panels / "nested_delta_vs_base").with_suffix(".svg"))
            delta_base_ax.figure.savefig((path_panels / "nested_delta_vs_base").with_suffix(".png"))
    delta_base_ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
