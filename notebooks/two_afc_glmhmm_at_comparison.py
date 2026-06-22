import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import json
    import math
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_1samp, ttest_rel
    from statannotations.Annotator import Annotator

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

    from glmhmmt.cli.fit_glmhmm import main as fit_main
    from glmhmmt.notebook_support.analysis_common import load_fit_arrays
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views

    from src.plots.common import boxplot_STYLE
    from src.utils import fig_size

    project_root = _PROJECT_ROOT
    configure_paths(config_path=project_root / "config.toml")
    paths = get_runtime_paths()

    sns.set_theme(style="ticks", context="paper")
    return (
        Annotator,
        boxplot_STYLE,
        build_emission_weights_df,
        build_trial_df,
        build_views,
        fig_size,
        fit_main,
        get_adapter,
        json,
        load_fit_arrays,
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
def _(fig_size, project_root):
    path_panels = project_root / "figures" / "panels_glmhmm_at_comparison"
    path_panels.mkdir(parents=True, exist_ok=True)
    figsize = fig_size(2, 1)
    state_palette = {
        "Engaged": "tab:green",
        "Disengaged": "tab:gray",
    }
    state_hue_order = ["Engaged", "Disengaged"]
    freeze_model_palette = {
        "Free": "black",
        "Both0": "tab:red",
        "Stim0": "tab:blue",
        "Hist0": "tab:orange",
    }
    freeze_model_order = ["Free", "Both0", "Stim0", "Hist0"]
    return (
        figsize,
        freeze_model_order,
        freeze_model_palette,
        path_panels,
        state_hue_order,
        state_palette,
    )


@app.cell
def _(Annotator, np, pd, ttest_1samp, ttest_rel):
    def _stars(pvalue):
        if not np.isfinite(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def add_model_pair_annotations(ax, df, *, x, y, order, pairs, subject_col="subject"):
        if df is None or df.empty or not {x, y, subject_col}.issubset(df.columns):
            return
        available_pairs = []
        paired_frames = []
        for left, right in pairs:
            if left not in order or right not in order:
                continue
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
            paired_subjects = set(paired.index.astype(str))
            paired_frames.append(sub[sub[subject_col].astype(str).isin(paired_subjects)].copy())
            available_pairs.append((left, right))
        if not available_pairs or not paired_frames:
            return

        annotator = Annotator(
            ax,
            available_pairs,
            data=pd.concat(paired_frames, ignore_index=True),
            x=x,
            y=y,
            order=order,
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            loc="outside",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()

    def add_one_sample_zero_annotations(ax, df, *, x, y, order):
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

        for x_idx, x_value in enumerate(order):
            values = pd.to_numeric(df.loc[df[x] == x_value, y], errors="coerce").dropna()
            if len(values) < 2 or float(values.std()) == 0:
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=0.0).pvalue)
            stars = _stars(pvalue)
            if stars:
                text_x = x_value if isinstance(x_value, (int, float, np.integer, np.floating)) else x_idx
                ax.text(text_x, text_y, stars, ha="center", va="bottom")

    def add_numeric_pair_annotations(
        ax,
        df,
        *,
        x,
        y,
        pairs,
        subject_col="subject",
        show_ns=False,
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
            stars = _stars(pvalue)
            label = stars or ("ns" if show_ns else "")
            if not label:
                continue
            y_pos = base_y + pad * annotation_idx
            ax.plot([left, left, right, right], [y_pos, y_pos + pad * 0.25, y_pos + pad * 0.25, y_pos], color="0.35", linewidth=0.7)
            ax.text((left + right) / 2, y_pos + pad * 0.3, label, ha="center", va="bottom")
            annotation_idx += 1

    def add_paired_state_annotation(
        ax,
        df,
        *,
        x,
        y,
        order,
        hue="state_label",
        subject_col="subject",
        hue_order=("Engaged", "Disengaged"),
    ):
        if df is None or df.empty or len(hue_order) != 2:
            return
        if not {x, y, hue, subject_col}.issubset(df.columns):
            return

        available_pairs = []
        paired_frames = []
        for x_value in order:
            sub = df[df[x] == x_value]
            paired = sub.pivot_table(
                values=y,
                index=subject_col,
                columns=hue,
                aggfunc="first",
            )
            if not all(state in paired.columns for state in hue_order):
                continue
            paired = paired.dropna(subset=list(hue_order))
            if len(paired) < 2:
                continue
            paired_subjects = set(paired.index.astype(str))
            paired_frames.append(sub[sub[subject_col].astype(str).isin(paired_subjects)].copy())
            available_pairs.append(((x_value, hue_order[0]), (x_value, hue_order[1])))

        if not available_pairs or not paired_frames:
            return

        annotator = Annotator(
            ax,
            available_pairs,
            data=pd.concat(paired_frames, ignore_index=True),
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=list(hue_order),
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()

    def add_subject_pair_lines(
        ax,
        df,
        *,
        x,
        y,
        order,
        hue="state_label",
        subject_col="subject",
        hue_order=("Engaged", "Disengaged"),
        offset=0.2,
    ):
        if df is None or df.empty or len(hue_order) != 2:
            return
        if not {x, y, hue, subject_col}.issubset(df.columns):
            return
        for x_idx, x_value in enumerate(order):
            sub = df[df[x] == x_value]
            paired = sub.pivot_table(
                values=y,
                index=subject_col,
                columns=hue,
                aggfunc="first",
            )
            if not all(state in paired.columns for state in hue_order):
                continue
            paired = paired.dropna(subset=list(hue_order))
            for _, row in paired.iterrows():
                ax.plot(
                    [x_idx - offset, x_idx + offset],
                    [row[hue_order[0]], row[hue_order[1]]],
                    color="0.75",
                    linewidth=0.5,
                    zorder=0,
                )

    return (
        add_model_pair_annotations,
        add_numeric_pair_annotations,
        add_one_sample_zero_annotations,
        add_paired_state_annotation,
        add_subject_pair_lines,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2AFC GLM-HMM: stimulus vs choice history

    This notebook fits a small set of 2-state GLM-HMMs for 2AFC and compares:

    1. whether the disengaged state is better described by frozen stimulus sensitivity or by frozen choice-history sensitivity;
    2. how much choice history adds beyond the previous choice.

    The frozen-emission tests use raw state indices during fitting. The convention below is raw state `0` = disengaged candidate and raw state `1` = engaged candidate; the weight plot at the end checks the post-hoc state labels after fitting.
    """)
    return


@app.cell
def _():
    TASK = "2AFC"
    K = 2
    TAU = 50

    HISTORY_FEATURE = "choice_lag_param"
    HISTORY_LABEL = "choice lag param"
    FREEZE_MODEL_SPECS = []
    core_cols = ["bias", "stim_param", HISTORY_FEATURE]
    feature_id = HISTORY_FEATURE.replace("_", "-")
    FREEZE_MODEL_SPECS.extend(
        [
            {
                "model_id": f"2afc_bias-stim-{feature_id}_free",
                "label": f"free stim + {HISTORY_LABEL}",
                "short_label": "Free",
                "history_feature": HISTORY_FEATURE,
                "description": "Control: bias, stimulus and choice-history parameter free in both states.",
                "emission_cols": core_cols,
                "frozen_emissions": {},
            },
            {
                "model_id": f"2afc_bias-stimD0-{feature_id}E0",
                "label": f"stim_D=0, {HISTORY_LABEL}_E=0",
                "short_label": "Both0",
                "history_feature": HISTORY_FEATURE,
                "description": "Stimulus frozen in raw state 0, choice-history parameter frozen in raw state 1.",
                "emission_cols": core_cols,
                "frozen_emissions": {"0": {"stim_param": 0.0}, "1": {HISTORY_FEATURE: 0.0}},
            },
            {
                "model_id": f"2afc_bias-stimD0-{feature_id}_free",
                "label": f"stim_D=0 ({HISTORY_LABEL} free)",
                "short_label": "Stim0",
                "history_feature": HISTORY_FEATURE,
                "description": "Stimulus frozen in raw state 0, choice-history parameter free in both states.",
                "emission_cols": core_cols,
                "frozen_emissions": {"0": {"stim_param": 0.0}},
            },
            {
                "model_id": f"2afc_bias-stim_free-{feature_id}E0",
                "label": f"{HISTORY_LABEL}_E=0",
                "short_label": "Hist0",
                "history_feature": HISTORY_FEATURE,
                "description": "Choice-history parameter frozen in raw state 1, stimulus free in both states.",
                "emission_cols": core_cols,
                "frozen_emissions": {"1": {HISTORY_FEATURE: 0.0}},
            },
        ]
    )

    def choice_lag_cols(n_lags: int) -> list[str]:
        return [f"choice_lag_{lag:02d}" for lag in range(1, int(n_lags) + 1)]

    LAG_MODEL_SPECS = [
        {
            "model_id": f"2afc_bias-stim-choice_lags_{n_lags:02d}",
            "label": "prev choice" if n_lags == 1 else f"choice lags 1-{n_lags}",
            "description": (
                "Previous choice only"
                if n_lags == 1
                else f"Nested choice-history model with lags 1-{n_lags}."
            ),
            "n_lags": n_lags,
            "emission_cols": ["bias", "stim_param", *choice_lag_cols(n_lags)],
            "frozen_emissions": {},
        }
        for n_lags in range(1, 11)
    ]
    return FREEZE_MODEL_SPECS, K, LAG_MODEL_SPECS, TASK, TAU


@app.cell
def _(FREEZE_MODEL_SPECS, LAG_MODEL_SPECS, pd):
    model_table = pd.DataFrame(
        [
            {
                "suite": "freeze",
                "model_id": spec["model_id"],
                "label": spec["label"],
                "short_label": spec.get("short_label"),
                "history_feature": spec.get("history_feature"),
                "emission_cols": ", ".join(spec["emission_cols"]),
                "frozen_emissions": spec["frozen_emissions"],
            }
            for spec in FREEZE_MODEL_SPECS
        ]
        + [
            {
                "suite": "lag curve",
                "model_id": spec["model_id"],
                "label": spec["label"],
                "short_label": spec.get("short_label"),
                "history_feature": spec.get("history_feature"),
                "emission_cols": ", ".join(spec["emission_cols"]),
                "frozen_emissions": spec["frozen_emissions"],
            }
            for spec in LAG_MODEL_SPECS
        ]
    )
    model_table
    return


@app.cell
def _(TASK, get_adapter, pl):
    adapter = get_adapter(TASK)
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    all_subjects = df_all["subject"].unique().sort().to_list()
    subject_trial_counts = (
        df_all
        .group_by("subject")
        .agg(pl.len().cast(pl.Int64).alias("n_trials"))
    )
    baseline_class_idx = int(adapter.baseline_class_idx)
    return (
        adapter,
        all_subjects,
        baseline_class_idx,
        df_all,
        subject_trial_counts,
    )


@app.cell
def _(all_subjects, mo):
    ui_subjects = mo.ui.multiselect(
        options=all_subjects,
        value=all_subjects,
        label="Subjects",
    )
    ui_num_iters = mo.ui.number(start=1, stop=500, step=1, value=50, label="EM iterations")
    ui_n_restarts = mo.ui.number(start=1, stop=20, step=1, value=5, label="Restarts")
    ui_cv_mode = mo.ui.dropdown(
        options=["balanced_session_holdout", "none"],
        value="balanced_session_holdout",
        label="Fit score",
    )
    ui_run_freeze = mo.ui.run_button(label="Fit freeze battery")
    ui_run_lags = mo.ui.run_button(label="Fit lag curve")
    mo.vstack(
        [
            mo.hstack([ui_subjects]),
            mo.hstack([ui_num_iters, ui_n_restarts, ui_cv_mode]),
            mo.hstack([ui_run_freeze, ui_run_lags]),
        ]
    )
    return (
        ui_cv_mode,
        ui_n_restarts,
        ui_num_iters,
        ui_run_freeze,
        ui_run_lags,
        ui_subjects,
    )


@app.cell
def _(
    K,
    TASK,
    TAU,
    baseline_class_idx,
    fit_main,
    json,
    math,
    mo,
    np,
    paths,
    pl,
):
    def free_parameter_count(spec: dict, *, k: int = K, num_classes: int = 2) -> int:
        n_emission = k * (num_classes - 1) * len(spec["emission_cols"])
        n_frozen = sum(len(features) for features in spec.get("frozen_emissions", {}).values())
        n_transition = k * (k - 1)
        return int(n_transition + n_emission - n_frozen)

    def chi2_sf_even_df(x: float, df: int) -> float:
        if not math.isfinite(x) or x < 0 or df <= 0:
            return float("nan")
        if df % 2 != 0:
            return float("nan")
        half_x = x / 2.0
        terms = sum((half_x**i) / math.factorial(i) for i in range(df // 2))
        return float(math.exp(-half_x) * terms)

    def spec_matches_config(spec: dict, model_dir) -> bool:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            return False
        config_frozen = config.get("frozen_emissions") or {}
        spec_frozen = spec.get("frozen_emissions") or {}
        return (
            list(config.get("emission_cols", [])) == list(spec["emission_cols"])
            and config_frozen == spec_frozen
        )

    def subject_arrays_match_spec(spec: dict, model_dir, metrics_path) -> bool:
        subject = metrics_path.name.split("_K", maxsplit=1)[0]
        arrays_path = model_dir / f"{subject}_K{K}_glmhmm_arrays.npz"
        if not arrays_path.exists():
            return False
        try:
            with np.load(arrays_path, allow_pickle=True) as arrays:
                x_cols = [str(col) for col in arrays["X_cols"]]
                frozen_json = str(arrays["frozen_emissions_json"].item())
        except Exception:
            return False
        try:
            array_frozen = json.loads(frozen_json) if frozen_json else {}
        except Exception:
            return False
        return (
            x_cols == list(spec["emission_cols"])
            and array_frozen == (spec.get("frozen_emissions") or {})
        )

    def read_metrics(model_specs: list[dict]) -> pl.DataFrame:
        frames = []
        for order, spec in enumerate(model_specs):
            model_dir = paths.RESULTS / "fits" / TASK / "glmhmm" / spec["model_id"]
            if not spec_matches_config(spec, model_dir):
                continue
            for path in sorted(model_dir.glob("*_metrics.parquet")):
                if not subject_arrays_match_spec(spec, model_dir, path):
                    continue
                frame = pl.read_parquet(path).with_columns(
                    pl.lit(spec["model_id"]).alias("model_id"),
                    pl.lit(spec["label"]).alias("model_label"),
                    pl.lit(spec.get("short_label", spec["label"])).alias("short_label"),
                    pl.lit(order, dtype=pl.Int64).alias("model_order"),
                    pl.lit(int(spec.get("n_lags", 0)), dtype=pl.Int64).alias("n_lags"),
                    pl.lit(free_parameter_count(spec), dtype=pl.Int64).alias("n_free_params"),
                )
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal")

    def read_glm_metrics(alias: str = "one hot") -> pl.DataFrame:
        model_dir = paths.RESULTS / "fits" / TASK / "glm" / alias
        frames = [
            pl.read_parquet(path).with_columns(
                pl.lit(alias).alias("model_id"),
                pl.lit("GLM one hot").alias("model_label"),
                pl.lit("GLM").alias("short_label"),
                pl.lit(-1, dtype=pl.Int64).alias("model_order"),
                pl.lit(0, dtype=pl.Int64).alias("n_lags"),
                pl.lit(None, dtype=pl.Int64).alias("n_free_params"),
            )
            for path in sorted(model_dir.glob("*_metrics.parquet"))
        ]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal")

    def score_column(metrics: pl.DataFrame) -> str:
        if "test_ll_per_trial_mean" in metrics.columns:
            non_null = metrics.select(pl.col("test_ll_per_trial_mean").is_not_null().sum()).item()
            if non_null:
                return "test_ll_per_trial_mean"
        return "ll_per_trial"

    def fit_suite(
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
            title="Fitting GLM-HMM suite",
            subtitle=f"{len(model_specs)} models x {len(subjects)} subjects",
            completion_title="Fit suite complete",
        ) as bar:
            for spec in model_specs:
                out_dir = paths.RESULTS / "fits" / TASK / "glmhmm" / spec["model_id"]

                def on_progress(info: dict) -> None:
                    event = info.get("event")
                    if event == "cv_repeat_complete":
                        bar.update(
                            increment=1,
                            title=f"Fitting {spec['label']}",
                            subtitle=(
                                f"{info.get('subject')} CV fold "
                                f"{info.get('cv_repeat_index')}/{info.get('cv_repeat_total')}"
                            ),
                        )
                    elif event == "restart_complete" and cv_mode == "none":
                        bar.update(
                            increment=1,
                            title=f"Fitting {spec['label']}",
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
                    out_dir=out_dir,
                    tau=TAU,
                    emission_cols=list(spec["emission_cols"]),
                    frozen_emissions=spec["frozen_emissions"] or None,
                    task=TASK,
                    cv_mode=cv_mode,
                    cv_repeats=cv_repeats,
                    verbose=False,
                    baseline_class_idx=baseline_class_idx,
                    progress_callback=on_progress,
                )

    return (
        chi2_sf_even_df,
        fit_suite,
        read_glm_metrics,
        read_metrics,
        score_column,
        spec_matches_config,
        subject_arrays_match_spec,
    )


@app.cell
def _(
    FREEZE_MODEL_SPECS,
    fit_suite,
    mo,
    ui_cv_mode,
    ui_n_restarts,
    ui_num_iters,
    ui_run_freeze,
    ui_subjects,
):
    if ui_run_freeze.value:
        fit_suite(
            model_specs=FREEZE_MODEL_SPECS,
            subjects=list(ui_subjects.value),
            cv_mode=ui_cv_mode.value,
            num_iters=int(ui_num_iters.value),
            n_restarts=int(ui_n_restarts.value),
        )
        _output = mo.md("Freeze battery fitted.")
    else:
        _output = mo.md("Run the freeze battery when you want to create or refresh those fits.")
    _output
    return


@app.cell
def _(freeze_model_order, freeze_trial_df, mo, np, pd):
    _schema_auc = [
        "model_id",
        "model_label",
        "short_label",
        "model_order",
        "subject",
        "metric",
        "metric_label",
        "auc",
    ]
    _schema_curve = [
        "model_id",
        "model_label",
        "short_label",
        "model_order",
        "subject",
        "metric",
        "metric_label",
        "fpr",
        "tpr",
    ]

    def binary_engaged_target(labels):
        label_text = pd.Series(labels, copy=False).astype(str).str.strip().str.lower()
        positive = label_text.eq("engaged") | label_text.str.startswith("engaged ")
        negative = label_text.eq("disengaged") | label_text.str.startswith("disengaged ")
        return positive.to_numpy(dtype=bool), (positive | negative).to_numpy(dtype=bool)

    def roc_curve(target, score):
        target = np.asarray(target, dtype=bool)
        score = np.asarray(score, dtype=float)
        valid = np.isfinite(score)
        target = target[valid]
        score = score[valid]
        n_pos = int(target.sum())
        n_neg = int((~target).sum())
        if target.size == 0 or n_pos == 0 or n_neg == 0:
            return None
        order = np.argsort(-score, kind="mergesort")
        target_sorted = target[order]
        score_sorted = score[order]
        threshold_idxs = np.r_[np.where(np.diff(score_sorted))[0], target_sorted.size - 1]
        tps = np.cumsum(target_sorted)[threshold_idxs]
        fps = (1 + threshold_idxs) - tps
        tpr = np.r_[0.0, tps / n_pos]
        fpr = np.r_[0.0, fps / n_neg]
        auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))
        return fpr, tpr, auc

    if freeze_trial_df.is_empty():
        freeze_aux_metric_specs = []
        freeze_aux_auc_df = pd.DataFrame(columns=_schema_auc)
        freeze_aux_curve_df = pd.DataFrame(columns=_schema_curve)
        _output = mo.md("No freeze trial data available for auxiliary behavior ROC.")
    else:
        _trial_pdf = freeze_trial_df.to_pandas()
        _state_col = next(
            (col for col in ["state_label", "state_label_pred"] if col in _trial_pdf.columns),
            None,
        )
        _rt_col = next(
            (
                col
                for col in ["RT", "RT2", "response_time_first", "reaction_time", "ReactionTime", "timepoint_4"]
                if col in _trial_pdf.columns
            ),
            None,
        )
        freeze_aux_metric_specs = [("nLicks", "nLicks", "Higher lick count")]
        if _rt_col is not None:
            freeze_aux_metric_specs.append((_rt_col, "RT", "Faster RT"))
        freeze_aux_metric_specs = [
            spec for spec in freeze_aux_metric_specs if spec[0] in _trial_pdf.columns
        ]

        _fpr_grid = np.linspace(0, 1, 101)
        _auc_rows = []
        _curve_rows = []
        if _state_col is not None and freeze_aux_metric_specs:
            for (_short_label, _subject), _subj_df in _trial_pdf.groupby(["short_label", "subject"], sort=False):
                for _metric_col, _metric_label, _direction_label in freeze_aux_metric_specs:
                    _target, _valid_labels = binary_engaged_target(_subj_df[_state_col])
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
                    _first = _subj_df.iloc[0]
                    _auc_rows.append(
                        {
                            "model_id": _first["model_id"],
                            "model_label": _first["model_label"],
                            "short_label": _short_label,
                            "model_order": int(_first["model_order"]),
                            "subject": str(_subject),
                            "metric": _metric_col,
                            "metric_label": _metric_label,
                            "auc": _auc,
                        }
                    )
                    for _fpr_value, _tpr_value in zip(_fpr_grid, _interp_tpr, strict=False):
                        _curve_rows.append(
                            {
                                "model_id": _first["model_id"],
                                "model_label": _first["model_label"],
                                "short_label": _short_label,
                                "model_order": int(_first["model_order"]),
                                "subject": str(_subject),
                                "metric": _metric_col,
                                "metric_label": _metric_label,
                                "fpr": float(_fpr_value),
                                "tpr": float(_tpr_value),
                            }
                        )
        freeze_aux_auc_df = pd.DataFrame(_auc_rows, columns=_schema_auc)
        freeze_aux_curve_df = pd.DataFrame(_curve_rows, columns=_schema_curve)
        if freeze_aux_auc_df.empty:
            _output = mo.md("Auxiliary behavior ROC needs Engaged and Disengaged trials per subject.")
        else:
            freeze_aux_auc_df["short_label"] = pd.Categorical(
                freeze_aux_auc_df["short_label"],
                categories=freeze_model_order,
                ordered=True,
            )
            freeze_aux_curve_df["short_label"] = pd.Categorical(
                freeze_aux_curve_df["short_label"],
                categories=freeze_model_order,
                ordered=True,
            )
            _output = freeze_aux_auc_df
    _output
    return freeze_aux_auc_df, freeze_aux_curve_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Freeze models: auxiliary behavior by state

    These panels ask whether `nLicks` and RT discriminate the post-hoc `Engaged` state similarly across the four frozen-emission models. For RT the score is negated, so larger AUC means faster responses in `Engaged`.
    """)
    return


@app.cell
def _(figsize, freeze_aux_auc_df, mo, path_panels, plt, sns):
    _metric_df = freeze_aux_auc_df[freeze_aux_auc_df["metric"].eq("nLicks")]
    if _metric_df.empty:
        _output = mo.md("No nLicks AUC rows to plot.")
    else:
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_nlicks_auc_ax = plt.gca()
        sns.lineplot(
            data=_metric_df,
            x="short_label",
            y="auc",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=freeze_nlicks_auc_ax,
        )
        sns.lineplot(
            data=_metric_df,
            x="short_label",
            y="auc",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=freeze_nlicks_auc_ax,
        )
        freeze_nlicks_auc_ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
        # add_model_pair_annotations(
        #     freeze_nlicks_auc_ax,
        #     _metric_df,
        #     x="short_label",
        #     y="auc",
        #     order=freeze_model_order,
        #     pairs=[("Free", "Both0"), ("Free", "Stim0"), ("Free", "Hist0")],
        # )
        freeze_nlicks_auc_ax.set_title("nLicks")
        freeze_nlicks_auc_ax.set_xlabel("")
        freeze_nlicks_auc_ax.set_ylabel("State AUC")
        freeze_nlicks_auc_ax.set_ylim(0, 1)
        clean_lineplot_edges(freeze_nlicks_auc_ax)
        sns.despine(ax=freeze_nlicks_auc_ax)
        plt.savefig(f"{path_panels}/freeze_nlicks_auc.svg")
        plt.savefig(f"{path_panels}/freeze_nlicks_auc.png")
        _output = freeze_nlicks_auc_ax
    _output
    return


@app.cell
def _(figsize, freeze_aux_auc_df, mo, path_panels, plt, sns):
    _metric_df = freeze_aux_auc_df[freeze_aux_auc_df["metric_label"].eq("RT")]
    if _metric_df.empty:
        _output = mo.md("No RT AUC rows to plot.")
    else:
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_rt_auc_ax = plt.gca()
        sns.lineplot(
            data=_metric_df,
            x="short_label",
            y="auc",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=freeze_rt_auc_ax,
        )
        sns.lineplot(
            data=_metric_df,
            x="short_label",
            y="auc",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=freeze_rt_auc_ax,
        )
        freeze_rt_auc_ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
        # add_model_pair_annotations(
        #     freeze_rt_auc_ax,
        #     _metric_df,
        #     x="short_label",
        #     y="auc",
        #     order=freeze_model_order,
        #     pairs=[("Free", "Both0"), ("Free", "Stim0"), ("Free", "Hist0")],
        # )
        freeze_rt_auc_ax.set_title("RT")
        freeze_rt_auc_ax.set_xlabel("")
        freeze_rt_auc_ax.set_ylabel("State AUC")
        freeze_rt_auc_ax.set_ylim(0, 1)
        clean_lineplot_edges(freeze_rt_auc_ax)
        sns.despine(ax=freeze_rt_auc_ax)
        plt.savefig(f"{path_panels}/freeze_rt_auc.svg")
        plt.savefig(f"{path_panels}/freeze_rt_auc.png")
        _output = freeze_rt_auc_ax
    _output
    return


@app.cell
def _(
    LAG_MODEL_SPECS,
    fit_suite,
    mo,
    ui_cv_mode,
    ui_n_restarts,
    ui_num_iters,
    ui_run_lags,
    ui_subjects,
):
    if ui_run_lags.value:
        fit_suite(
            model_specs=LAG_MODEL_SPECS,
            subjects=list(ui_subjects.value),
            cv_mode=ui_cv_mode.value,
            num_iters=int(ui_num_iters.value),
            n_restarts=int(ui_n_restarts.value),
        )
        _output = mo.md("Lag curve fitted.")
    else:
        _output = mo.md("Run the lag curve when you want to create or refresh those fits.")
    _output
    return


@app.cell
def _(FREEZE_MODEL_SPECS, LAG_MODEL_SPECS, read_glm_metrics, read_metrics):
    freeze_metrics = read_metrics(FREEZE_MODEL_SPECS)
    glm_one_hot_metrics = read_glm_metrics("one hot")
    lag_metrics = read_metrics(LAG_MODEL_SPECS)
    return freeze_metrics, glm_one_hot_metrics, lag_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frozen stimulus vs frozen choice history

    The primary score is held-out log-likelihood per trial when the fits were run with session CV. If the models were run in-sample, the notebook falls back to in-sample log-likelihood per trial.
    """)
    return


@app.cell
def _(freeze_metrics, math, mo, pl, score_column):
    if freeze_metrics.is_empty():
        freeze_score_col = "test_ll_per_trial_mean"
        freeze_plot_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "model_id": pl.Utf8,
                "model_label": pl.Utf8,
                "short_label": pl.Utf8,
                "model_order": pl.Int64,
                "score": pl.Float64,
                "bic": pl.Float64,
                "acc": pl.Float64,
                "n_free_params": pl.Int64,
            }
        )
        freeze_summary = pl.DataFrame()
        _output = mo.md("No freeze metrics found yet.")
    else:
        freeze_score_col = score_column(freeze_metrics)
        freeze_plot_df = (
            freeze_metrics
            .with_columns((pl.col(freeze_score_col) / math.log(2)).alias("score"))
            .select(["subject", "model_id", "model_label", "short_label", "model_order", "score", "bic", "acc", "n_free_params"])
            .sort(["model_order", "subject"])
        )
        freeze_summary = (
            freeze_plot_df
            .group_by(["model_order", "model_label", "short_label", "n_free_params"])
            .agg(
                pl.mean("score").alias("mean_score"),
                pl.std("score").alias("sd_score"),
                pl.mean("bic").alias("mean_bic"),
                pl.mean("acc").alias("mean_acc"),
                pl.len().alias("n_subjects"),
            )
            .sort("model_order")
        )
        _output = freeze_summary
    _output
    return (freeze_plot_df,)


@app.cell
def _(
    freeze_plot_df,
    glm_one_hot_metrics,
    math,
    mo,
    pl,
    score_column,
    subject_trial_counts,
):
    _schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "model_label": pl.Utf8,
        "short_label": pl.Utf8,
        "model_order": pl.Int64,
        "score": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
        "n_free_params": pl.Int64,
        "n_trials": pl.Int64,
        "delta_vs_glm": pl.Float64,
        "delta_vs_free": pl.Float64,
    }
    if glm_one_hot_metrics.is_empty() or freeze_plot_df.is_empty():
        glm_one_hot_plot_df = pl.DataFrame(schema={k: v for k, v in _schema.items() if k not in {"delta_vs_glm", "delta_vs_free"}})
        freeze_vs_glm_df = pl.DataFrame(schema=_schema)
        freeze_score_with_glm_df = pl.DataFrame(schema=_schema)
        freeze_vs_glm_summary = pl.DataFrame()
        _output = mo.md("No `glm/one hot` and freeze metrics overlap yet.")
    else:
        glm_score_col = score_column(glm_one_hot_metrics)
        glm_one_hot_plot_df = (
            glm_one_hot_metrics
            .with_columns((pl.col(glm_score_col) / math.log(2)).alias("score"))
            .select(["subject", "model_id", "model_label", "short_label", "model_order", "score", "bic", "acc", "n_free_params"])
            .join(subject_trial_counts, on="subject", how="left")
            .with_columns(
                pl.col("model_order").cast(pl.Int64),
                pl.col("n_free_params").cast(pl.Int64),
                pl.col("n_trials").cast(pl.Int64),
            )
        )
        _glm_baseline = glm_one_hot_plot_df.select(["subject", pl.col("score").alias("glm_score")])
        _free_baseline = (
            freeze_plot_df
            .filter(pl.col("model_order") == 0)
            .select(["subject", pl.col("score").alias("free_score")])
        )
        freeze_vs_glm_df = (
            freeze_plot_df
            .join(_glm_baseline, on="subject", how="inner")
            .join(_free_baseline, on="subject", how="inner")
            .join(subject_trial_counts, on="subject", how="left")
            .with_columns(
                (pl.col("score") - pl.col("glm_score")).alias("delta_vs_glm"),
                (pl.col("score") - pl.col("free_score")).alias("delta_vs_free"),
                pl.col("model_order").cast(pl.Int64),
                pl.col("n_free_params").cast(pl.Int64),
                pl.col("n_trials").cast(pl.Int64),
            )
            .sort(["model_order", "subject"])
        )
        freeze_score_with_glm_df = pl.concat(
            [
                glm_one_hot_plot_df.with_columns(
                    pl.lit(0.0).alias("delta_vs_glm"),
                    pl.lit(None, dtype=pl.Float64).alias("delta_vs_free"),
                ).select(list(_schema)),
                freeze_vs_glm_df.select(list(_schema)),
            ],
            how="diagonal",
        ).sort(["model_order", "subject"])
        freeze_vs_glm_summary = (
            freeze_vs_glm_df
            .group_by(["model_order", "short_label"])
            .agg(
                pl.mean("delta_vs_glm").alias("mean_delta_vs_glm"),
                pl.std("delta_vs_glm").alias("sd_delta_vs_glm"),
                pl.len().alias("n_subjects"),
            )
            .sort("model_order")
        )
        _output = freeze_vs_glm_summary
    _output
    return freeze_score_with_glm_df, freeze_vs_glm_df


@app.function
def clean_lineplot_edges(ax):
    for line in ax.lines:
        line.set_markeredgewidth(0)
        line.set_markeredgecolor("none")
    for collection in ax.collections:
        collection.set_edgecolor("none")
        collection.set_linewidth(0)


@app.cell
def _(
    add_model_pair_annotations,
    figsize,
    freeze_plot_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if freeze_plot_df.is_empty():
        _output = mo.md("No freeze metrics to plot.")
    else:
        _df = freeze_plot_df.to_pandas()
        _order = (
            freeze_plot_df
            .select(["model_order", "short_label"])
            .unique()
            .sort("model_order")
            .get_column("short_label")
            .to_list()
        )
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_score_ax = plt.gca()
        sns.lineplot(
            data=_df,
            x="short_label",
            y="score",
            units="subject",
            estimator=None,
            color="0.82",
            linewidth=0.8,
            sort=False,
            ax=freeze_score_ax,
        )
        sns.lineplot(
            data=_df,
            x="short_label",
            y="score",
            marker="o",
            markersize=4,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            sort=False,
            ax=freeze_score_ax,
        )
        add_model_pair_annotations(
            freeze_score_ax,
            _df,
            x="short_label",
            y="score",
            order=_order,
            pairs=[("Free", "Both0"), ("Stim0", "Hist0")],
        )
        freeze_score_ax.set_xlabel("")
        freeze_score_ax.set_ylabel("CV test LL (bits/trial)")
        clean_lineplot_edges(freeze_score_ax)
        sns.despine(ax=freeze_score_ax)
        plt.savefig(f"{path_panels}/freeze_cv_ll.svg")
        plt.savefig(f"{path_panels}/freeze_cv_ll.png")
        _output = freeze_score_ax
    _output
    return


@app.cell
def _(
    figsize,
    freeze_score_with_glm_df,
    freeze_vs_glm_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if freeze_vs_glm_df.is_empty():
        _output = mo.md("No freeze-vs-GLM score rows available.")
    else:
        _score_df = freeze_score_with_glm_df.to_pandas()
        _score_order = (
            freeze_score_with_glm_df
            .select(["model_order", "short_label"])
            .unique()
            .sort("model_order")
            .get_column("short_label")
            .to_list()
        )
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_glm_score_ax = plt.gca()
        sns.lineplot(
            data=_score_df,
            x="short_label",
            y="score",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=freeze_glm_score_ax,
        )
        sns.lineplot(
            data=_score_df,
            x="short_label",
            y="score",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=freeze_glm_score_ax,
        )
        # add_model_pair_annotations(
        #     freeze_glm_score_ax,
        #     _score_df,
        #     x="short_label",
        #     y="score",
        #     order=_score_order,
        #     pairs=[("GLM", "Free"), ("GLM", "Both0"), ("GLM", "Stim0"), ("GLM", "Hist0")],
        # )
        freeze_glm_score_ax.set_title("CV LL")
        freeze_glm_score_ax.set_xlabel("")
        freeze_glm_score_ax.set_ylabel("CV test LL (bits/trial)")
        clean_lineplot_edges(freeze_glm_score_ax)
        sns.despine(ax=freeze_glm_score_ax)
        plt.savefig(f"{path_panels}/freeze_vs_glm_cv_ll.svg")
        plt.savefig(f"{path_panels}/freeze_vs_glm_cv_ll.png")
        _output = freeze_glm_score_ax
    _output
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    figsize,
    freeze_vs_glm_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if freeze_vs_glm_df.is_empty():
        _output = mo.md("No freeze-vs-GLM delta rows available.")
    else:
        _delta_df = freeze_vs_glm_df.to_pandas()
        _delta_order = (
            freeze_vs_glm_df
            .select(["model_order", "short_label"])
            .unique()
            .sort("model_order")
            .get_column("short_label")
            .to_list()
        )
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_delta_glm_ax = plt.gca()
        sns.lineplot(
            data=_delta_df,
            x="short_label",
            y="delta_vs_glm",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=freeze_delta_glm_ax,
        )
        sns.lineplot(
            data=_delta_df,
            x="short_label",
            y="delta_vs_glm",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=freeze_delta_glm_ax,
        )
        freeze_delta_glm_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        add_one_sample_zero_annotations(
            freeze_delta_glm_ax,
            _delta_df,
            x="short_label",
            y="delta_vs_glm",
            order=_delta_order,
        )
        freeze_delta_glm_ax.set_title("vs GLM")
        freeze_delta_glm_ax.set_xlabel("")
        freeze_delta_glm_ax.set_ylabel("Delta CV LL (bits/trial)")
        clean_lineplot_edges(freeze_delta_glm_ax)
        sns.despine(ax=freeze_delta_glm_ax)
        plt.savefig(f"{path_panels}/freeze_delta_vs_glm.svg")
        plt.savefig(f"{path_panels}/freeze_delta_vs_glm.png")
        _output = freeze_delta_glm_ax
    _output
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    figsize,
    freeze_vs_glm_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if freeze_vs_glm_df.is_empty():
        _output = mo.md("No freeze-vs-Free delta rows available.")
    else:
        _delta_df = freeze_vs_glm_df.to_pandas()
        _delta_order = (
            freeze_vs_glm_df
            .select(["model_order", "short_label"])
            .unique()
            .sort("model_order")
            .get_column("short_label")
            .to_list()
        )
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_delta_free_ax = plt.gca()
        sns.lineplot(
            data=_delta_df,
            x="short_label",
            y="delta_vs_free",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=freeze_delta_free_ax,
        )
        sns.lineplot(
            data=_delta_df,
            x="short_label",
            y="delta_vs_free",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=freeze_delta_free_ax,
        )
        freeze_delta_free_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        add_one_sample_zero_annotations(
            freeze_delta_free_ax,
            _delta_df,
            x="short_label",
            y="delta_vs_free",
            order=_delta_order,
        )
        freeze_delta_free_ax.set_title("vs Free")
        freeze_delta_free_ax.set_xlabel("")
        freeze_delta_free_ax.set_ylabel("Delta CV LL (bits/trial)")
        clean_lineplot_edges(freeze_delta_free_ax)
        sns.despine(ax=freeze_delta_free_ax)
        plt.savefig(f"{path_panels}/freeze_delta_vs_free.svg")
        plt.savefig(f"{path_panels}/freeze_delta_vs_free.png")
        _output = freeze_delta_free_ax
    _output
    return


@app.cell
def _(freeze_plot_df, mo, pl):
    if freeze_plot_df.is_empty():
        freeze_delta = pl.DataFrame()
        freeze_delta_summary = pl.DataFrame()
        _output = mo.md("No freeze metrics to compare.")
    else:
        _control = (
            freeze_plot_df
            .filter(pl.col("model_order") == 0)
            .select(["subject", pl.col("score").alias("control_score")])
        )
        freeze_delta = (
            freeze_plot_df
            .join(_control, on="subject", how="inner")
            .with_columns((pl.col("score") - pl.col("control_score")).alias("delta_vs_control"))
            .sort(["model_order", "subject"])
        )
        freeze_delta_summary = (
            freeze_delta
            .group_by(["model_order", "model_label"])
            .agg(
                pl.mean("delta_vs_control").alias("mean_delta_vs_control"),
                pl.std("delta_vs_control").alias("sd_delta_vs_control"),
                pl.len().alias("n_subjects"),
            )
            .sort("model_order")
        )
        _output = freeze_delta_summary
    _output
    return


@app.cell
def _(freeze_vs_glm_df, math, mo, pl):
    if freeze_vs_glm_df.is_empty():
        ll_bits_equivalence = pl.DataFrame()
        _output = mo.md("No LL improvement equivalence to show yet.")
    else:
        ll_bits_equivalence = (
            freeze_vs_glm_df
            .group_by(["model_order", "short_label"])
            .agg(
                pl.mean("delta_vs_glm").alias("mean_delta_bits_vs_glm"),
                pl.mean("delta_vs_free").alias("mean_delta_bits_vs_free"),
                pl.mean("n_trials").alias("mean_n_trials"),
                pl.median("n_trials").alias("median_n_trials"),
                (pl.col("delta_vs_glm") * pl.col("n_trials")).mean().alias("mean_subject_bits_vs_glm"),
                (pl.col("delta_vs_glm") * pl.col("n_trials") * math.log10(2.0)).mean().alias("mean_log10_likelihood_ratio_vs_glm"),
                (pl.col("delta_vs_free") * pl.col("n_trials")).mean().alias("mean_subject_bits_vs_free"),
                (pl.col("delta_vs_free") * pl.col("n_trials") * math.log10(2.0)).mean().alias("mean_log10_likelihood_ratio_vs_free"),
                pl.len().alias("n_subjects"),
            )
            .with_columns(
                ((pl.lit(2.0) ** pl.col("mean_delta_bits_vs_glm") - 1.0) * 100.0).alias("pct_likelihood_per_trial_vs_glm"),
                ((pl.lit(2.0) ** pl.col("mean_delta_bits_vs_free") - 1.0) * 100.0).alias("pct_likelihood_per_trial_vs_free"),
            )
            .sort("model_order")
        )
        _positive_increases = (
            ll_bits_equivalence
            .filter(pl.col("mean_delta_bits_vs_glm") > 0)
            .sort("mean_delta_bits_vs_glm")
        )
        if _positive_increases.is_empty():
            _insight = "No positive mean increase vs GLM in the current fits."
        else:
            _smallest = _positive_increases.row(0, named=True)
            _delta_bits = float(_smallest["mean_delta_bits_vs_glm"])
            _per_trial_pct = (2.0 ** _delta_bits - 1.0) * 100.0
            _mean_n_trials = float(_smallest["mean_n_trials"])
            _median_n_trials = float(_smallest["median_n_trials"])
            _mean_subject_bits = float(_smallest["mean_subject_bits_vs_glm"])
            _mean_log10_ratio = float(_smallest["mean_log10_likelihood_ratio_vs_glm"])
            _factor_text = (
                f"{10.0 ** _mean_log10_ratio:.2e}"
                if _mean_log10_ratio < 300
                else f"10^{_mean_log10_ratio:.1f}"
            )
            _insight = (
                f"The smallest positive mean increase vs GLM is "
                f"{_delta_bits:.3f} bits/trial ({_smallest['short_label']}). "
                f"That is only { _per_trial_pct:.1f}% more likelihood on an average trial, "
                f"but using each subject's actual trial count in this model "
                f"(mean {_mean_n_trials:.0f}, median {_median_n_trials:.0f} trials), "
                f"it averages {_mean_subject_bits:.1f} extra bits per subject, "
                f"or about {_factor_text} times more likely for each subject's observed choice sequence."
            )
        _output = mo.vstack(
            [
                mo.md(
                    fr"""
                    **Equivalence.** A gain of \(\Delta\) bits/trial means the model assigns \(2^\Delta\) times more likelihood to the observed choice on an average trial.

                    {_insight}
                    """
                ),
                ll_bits_equivalence,
            ]
        )
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Previous choice vs choice-history lags

    Here \(N=1\) is the previous choice model. Models with \(N=2,\dots,10\) add signed choice lags one at a time, so they are nested in the \(N=1\) model and in the previous point of the curve. With CV the held-out curve is the main comparison; for formal likelihood-ratio p-values, run the suite with `Fit score = none`.
    """)
    return


@app.cell
def _(lag_metrics, math, mo, pl, score_column):
    if lag_metrics.is_empty():
        lag_score_col = "test_ll_per_trial_mean"
        lag_plot_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "model_id": pl.Utf8,
                "model_label": pl.Utf8,
                "model_order": pl.Int64,
                "n_lags": pl.Int64,
                "lag_label": pl.Utf8,
                "score": pl.Float64,
                "raw_ll": pl.Float64,
                "n_free_params": pl.Int64,
            }
        )
        _output = mo.md("No lag-curve metrics found yet.")
    else:
        lag_score_col = score_column(lag_metrics)
        lag_plot_df = (
            lag_metrics
            .with_columns(
                (pl.col(lag_score_col) / math.log(2)).alias("score"),
                pl.col("n_lags").cast(pl.Utf8).alias("lag_label"),
            )
            .select(["subject", "model_id", "model_label", "model_order", "n_lags", "lag_label", "score", "raw_ll", "n_free_params"])
            .with_columns(
                pl.col("model_order").cast(pl.Int64),
                pl.col("n_lags").cast(pl.Int64),
                pl.col("n_free_params").cast(pl.Int64),
            )
            .sort(["n_lags", "subject"])
        )
        _output = lag_plot_df
    _output
    return (lag_plot_df,)


@app.cell
def _(
    freeze_plot_df,
    glm_one_hot_metrics,
    lag_plot_df,
    math,
    mo,
    pl,
    score_column,
):
    _schema = {
        "subject": pl.Utf8,
        "model_id": pl.Utf8,
        "model_label": pl.Utf8,
        "model_order": pl.Int64,
        "n_lags": pl.Int64,
        "lag_label": pl.Utf8,
        "score": pl.Float64,
        "raw_ll": pl.Float64,
        "n_free_params": pl.Int64,
        "plot_order": pl.Int64,
        "is_glm": pl.Boolean,
        "is_full": pl.Boolean,
    }
    if lag_plot_df.is_empty():
        lag_plot_with_full_df = pl.DataFrame(schema=_schema)
        _output = mo.md("No lag metrics for the GLM/full-reference lag plot.")
    else:
        _lag_rows = lag_plot_df.with_columns(
            pl.col("n_lags").alias("plot_order"),
            pl.lit(False).alias("is_glm"),
            pl.lit(False).alias("is_full"),
        )
        _parts = [_lag_rows.select(list(_schema))]
        if not glm_one_hot_metrics.is_empty():
            _glm_score_col = score_column(glm_one_hot_metrics)
            _glm_rows = (
                glm_one_hot_metrics
                .with_columns((pl.col(_glm_score_col) / math.log(2)).alias("score"))
                .select(["subject", "model_id", "model_label", "score", "n_free_params"])
                .with_columns(
                    pl.lit(-1, dtype=pl.Int64).alias("model_order"),
                    pl.lit(0, dtype=pl.Int64).alias("n_lags"),
                    pl.lit("GLM").alias("lag_label"),
                    pl.lit(None, dtype=pl.Float64).alias("raw_ll"),
                    pl.lit(0, dtype=pl.Int64).alias("plot_order"),
                    pl.lit(True).alias("is_glm"),
                    pl.lit(False).alias("is_full"),
                )
            )
            _parts.append(_glm_rows.select(list(_schema)))
        if not freeze_plot_df.is_empty():
            _full_rows = (
                freeze_plot_df
                .filter(pl.col("model_order") == 0)
                .select(["subject", "model_id", "model_label", "score", "n_free_params"])
                .with_columns(
                    pl.lit(15, dtype=pl.Int64).alias("model_order"),
                    pl.lit(15, dtype=pl.Int64).alias("n_lags"),
                    pl.lit("15").alias("lag_label"),
                    pl.lit(None, dtype=pl.Float64).alias("raw_ll"),
                    pl.lit(12, dtype=pl.Int64).alias("plot_order"),
                    pl.lit(False).alias("is_glm"),
                    pl.lit(True).alias("is_full"),
                )
            )
            _parts.append(_full_rows.select(list(_schema)))
        lag_plot_with_full_df = pl.concat(_parts, how="diagonal").sort(["plot_order", "subject"])
        _output = (
            lag_plot_with_full_df
            .group_by(["plot_order", "lag_label", "is_glm", "is_full"])
            .agg(
                pl.mean("score").alias("mean_score_bits"),
                pl.std("score").alias("sd_score_bits"),
                pl.len().alias("n_subjects"),
            )
            .sort("plot_order")
        )
    _output
    return (lag_plot_with_full_df,)


@app.cell
def _(lag_plot_with_full_df, mo, pl):
    if lag_plot_with_full_df.is_empty():
        lag_delta_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "n_lags": pl.Int64,
                "lag_label": pl.Utf8,
                "plot_order": pl.Int64,
                "is_glm": pl.Boolean,
                "is_full": pl.Boolean,
                "score": pl.Float64,
                "delta_vs_1_lag": pl.Float64,
            }
        )
        _output = mo.md("No lag metrics for delta LL plot.")
    else:
        _one_lag = (
            lag_plot_with_full_df
            .filter(pl.col("n_lags") == 1)
            .select(["subject", pl.col("score").alias("one_lag_score")])
        )
        lag_delta_df = (
            lag_plot_with_full_df
            .join(_one_lag, on="subject", how="inner")
            .with_columns((pl.col("score") - pl.col("one_lag_score")).alias("delta_vs_1_lag"))
            .sort(["plot_order", "subject"])
        )
        _output = (
            lag_delta_df
            .group_by(["plot_order", "n_lags", "lag_label", "is_full"])
            .agg(
                pl.mean("delta_vs_1_lag").alias("mean_delta_bits"),
                pl.std("delta_vs_1_lag").alias("sd_delta_bits"),
                pl.len().alias("n_subjects"),
            )
            .sort("plot_order")
        )
    _output
    return (lag_delta_df,)


@app.cell
def _(lag_plot_with_full_df, mo, pl):
    if lag_plot_with_full_df.is_empty() or not lag_plot_with_full_df.get_column("is_glm").any():
        lag_delta_vs_glm_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "n_lags": pl.Int64,
                "lag_label": pl.Utf8,
                "plot_order": pl.Int64,
                "is_glm": pl.Boolean,
                "is_full": pl.Boolean,
                "score": pl.Float64,
                "delta_vs_glm": pl.Float64,
            }
        )
        _output = mo.md("No GLM baseline available for lag delta plot.")
    else:
        _glm_score = (
            lag_plot_with_full_df
            .filter(pl.col("is_glm"))
            .select(["subject", pl.col("score").alias("glm_score")])
        )
        lag_delta_vs_glm_df = (
            lag_plot_with_full_df
            .join(_glm_score, on="subject", how="inner")
            .with_columns((pl.col("score") - pl.col("glm_score")).alias("delta_vs_glm"))
            .sort(["plot_order", "subject"])
        )
        _output = (
            lag_delta_vs_glm_df
            .group_by(["plot_order", "n_lags", "lag_label", "is_glm", "is_full"])
            .agg(
                pl.mean("delta_vs_glm").alias("mean_delta_bits"),
                pl.std("delta_vs_glm").alias("sd_delta_bits"),
                pl.len().alias("n_subjects"),
            )
            .sort("plot_order")
        )
    _output
    return (lag_delta_vs_glm_df,)


@app.cell
def _(
    add_numeric_pair_annotations,
    figsize,
    lag_plot_with_full_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if lag_plot_with_full_df.is_empty():
        _output = mo.md("No lag metrics to plot.")
    else:
        _df = lag_plot_with_full_df.to_pandas()
        _ticks_df = (
            lag_plot_with_full_df
            .select(["plot_order", "lag_label"])
            .unique()
            .sort("plot_order")
        )
        _tick_positions = _ticks_df.get_column("plot_order").to_list()
        _order = _ticks_df.get_column("lag_label").to_list()
        plt.figure(figsize=figsize, constrained_layout=True)
        lag_curve_ax = plt.gca()
        sns.lineplot(
            data=_df,
            x="plot_order",
            y="score",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.7,
            sort=False,
            ax=lag_curve_ax,
        )
        sns.lineplot(
            data=_df,
            x="plot_order",
            y="score",
            errorbar=("se", 1),
            marker="o",
            markeredgewidth=0,
            markeredgecolor="none",
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            sort=False,
            ax=lag_curve_ax,
        )
        add_numeric_pair_annotations(
            lag_curve_ax,
            _df,
            x="plot_order",
            y="score",
            pairs=[(0, 1), (9, 10), (10, 12)],
            show_ns=True,
        )
        lag_curve_ax.set_xlabel("Choice-history lags")
        lag_curve_ax.set_ylabel("CV test LL (bits/trial)")
        lag_curve_ax.set_xticks(_tick_positions)
        lag_curve_ax.set_xticklabels(_order)
        clean_lineplot_edges(lag_curve_ax)
        sns.despine(ax=lag_curve_ax)
        plt.savefig(f"{path_panels}/choice_lag_curve.svg")
        plt.savefig(f"{path_panels}/choice_lag_curve.png")
        _output = lag_curve_ax
    _output
    return


@app.cell
def _(
    add_numeric_pair_annotations,
    figsize,
    lag_delta_vs_glm_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if lag_delta_vs_glm_df.is_empty():
        _output = mo.md("No lag delta-vs-GLM rows to plot.")
    else:
        _df = lag_delta_vs_glm_df.to_pandas()
        _ticks_df = (
            lag_delta_vs_glm_df
            .select(["plot_order", "lag_label"])
            .unique()
            .sort("plot_order")
        )
        _tick_positions = _ticks_df.get_column("plot_order").to_list()
        _order = _ticks_df.get_column("lag_label").to_list()
        plt.figure(figsize=figsize, constrained_layout=True)
        lag_delta_glm_ax = plt.gca()
        sns.lineplot(
            data=_df,
            x="plot_order",
            y="delta_vs_glm",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=lag_delta_glm_ax,
        )
        sns.lineplot(
            data=_df,
            x="plot_order",
            y="delta_vs_glm",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=lag_delta_glm_ax,
        )
        lag_delta_glm_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        add_numeric_pair_annotations(
            lag_delta_glm_ax,
            _df,
            x="plot_order",
            y="delta_vs_glm",
            pairs=[(0, 1), (9, 10), (10, 12)],
            show_ns=True,
        )
        lag_delta_glm_ax.set_xlabel("Choice-history lags")
        lag_delta_glm_ax.set_ylabel("Delta CV LL vs GLM (bits/trial)")
        lag_delta_glm_ax.set_xticks(_tick_positions)
        lag_delta_glm_ax.set_xticklabels(_order)
        clean_lineplot_edges(lag_delta_glm_ax)
        sns.despine(ax=lag_delta_glm_ax)
        plt.savefig(f"{path_panels}/choice_lag_delta_vs_glm.svg")
        plt.savefig(f"{path_panels}/choice_lag_delta_vs_glm.png")
        _output = lag_delta_glm_ax
    _output
    return


@app.cell
def _(
    add_one_sample_zero_annotations,
    figsize,
    lag_delta_df,
    mo,
    path_panels,
    plt,
    sns,
):
    if lag_delta_df.is_empty():
        _output = mo.md("No lag delta LL rows to plot.")
    else:
        _df = lag_delta_df.to_pandas()
        _ticks_df = (
            lag_delta_df
            .select(["plot_order", "lag_label"])
            .unique()
            .sort("plot_order")
        )
        _tick_positions = _ticks_df.get_column("plot_order").to_list()
        _order = _ticks_df.get_column("lag_label").to_list()
        plt.figure(figsize=figsize, constrained_layout=True)
        lag_delta_ll_ax = plt.gca()
        sns.lineplot(
            data=_df,
            x="plot_order",
            y="delta_vs_1_lag",
            units="subject",
            estimator=None,
            color="0.84",
            linewidth=0.6,
            sort=False,
            ax=lag_delta_ll_ax,
        )
        sns.lineplot(
            data=_df,
            x="plot_order",
            y="delta_vs_1_lag",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            markeredgecolor="none",
            errorbar=("se", 1),
            err_kws={"edgecolor": "none", "linewidth": 0},
            color="black",
            linewidth=1,
            sort=False,
            ax=lag_delta_ll_ax,
        )
        lag_delta_ll_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
        add_one_sample_zero_annotations(
            lag_delta_ll_ax,
            _df,
            x="plot_order",
            y="delta_vs_1_lag",
            order=_tick_positions,
        )
        lag_delta_ll_ax.set_xlabel("Choice-history lags")
        lag_delta_ll_ax.set_ylabel("Delta CV LL vs 1 lag (bits/trial)")
        lag_delta_ll_ax.set_xticks(_tick_positions)
        lag_delta_ll_ax.set_xticklabels(_order)
        clean_lineplot_edges(lag_delta_ll_ax)
        sns.despine(ax=lag_delta_ll_ax)
        plt.savefig(f"{path_panels}/choice_lag_delta_vs_1_lag.svg")
        plt.savefig(f"{path_panels}/choice_lag_delta_vs_1_lag.png")
        _output = lag_delta_ll_ax
    _output
    return


@app.cell
def _(chi2_sf_even_df, lag_plot_df, mo, pd, pl):
    if lag_plot_df.is_empty():
        lag_lrt_table = pl.DataFrame()
        _output = mo.md("No lag metrics for likelihood-ratio tests.")
    else:
        _wide = (
            lag_plot_df
            .select(["subject", "n_lags", "raw_ll", "n_free_params"])
            .to_pandas()
            .pivot(index="subject", columns="n_lags", values=["raw_ll", "n_free_params"])
        )
        _rows = []
        for n_lags in range(2, 11):
            if ("raw_ll", n_lags - 1) not in _wide.columns or ("raw_ll", n_lags) not in _wide.columns:
                continue
            _ll0 = _wide[("raw_ll", n_lags - 1)]
            _ll1 = _wide[("raw_ll", n_lags)]
            _df0 = _wide[("n_free_params", n_lags - 1)]
            _df1 = _wide[("n_free_params", n_lags)]
            _valid = _ll0.notna() & _ll1.notna() & _df0.notna() & _df1.notna()
            if not _valid.any():
                continue
            _lrt = 2.0 * float((_ll1[_valid] - _ll0[_valid]).sum())
            _dof = int((_df1[_valid].iloc[0] - _df0[_valid].iloc[0]) * int(_valid.sum()))
            _rows.append(
                {
                    "comparison": f"{n_lags - 1} -> {n_lags} lags",
                    "n_subjects": int(_valid.sum()),
                    "delta_raw_ll": float((_ll1[_valid] - _ll0[_valid]).sum()),
                    "lrt": _lrt,
                    "df": _dof,
                    "p_value": chi2_sf_even_df(_lrt, _dof),
                }
            )
        lag_lrt_table = pl.from_pandas(pd.DataFrame(_rows))
        _output = lag_lrt_table
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weight check for the frozen models

    This checks whether the raw frozen state index is aligned with the semantic state labels after fitting. If the frozen state does not land where expected, compare the metrics but do not overinterpret the `Engaged`/`Disengaged` names for that fit.
    """)
    return


@app.cell
def _(adapter, mo):
    _options = list(adapter._SCORING_OPTIONS.keys()) if hasattr(adapter, "_SCORING_OPTIONS") else ["default"]
    _default = getattr(adapter, "scoring_key", _options[0])
    if _default not in _options:
        _default = _options[0]
    ui_scoring_key = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="State scoring regressor",
    )
    mo.hstack([ui_scoring_key])
    return (ui_scoring_key,)


@app.cell
def _(
    FREEZE_MODEL_SPECS,
    K,
    adapter,
    build_emission_weights_df,
    build_trial_df,
    build_views,
    df_all,
    load_fit_arrays,
    mo,
    paths,
    pl,
    spec_matches_config,
    subject_arrays_match_spec,
    ui_scoring_key,
    ui_subjects,
):
    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    _frames = []
    _trial_frames = []
    _raw_rows = []
    _subjects = list(ui_subjects.value)
    for _model_order, spec in enumerate(FREEZE_MODEL_SPECS):
        _out = paths.RESULTS / "fits" / "2AFC" / "glmhmm" / spec["model_id"]
        if not _out.exists() or not spec_matches_config(spec, _out):
            continue
        _matching_subjects = [
            _subject
            for _subject in _subjects
            if subject_arrays_match_spec(
                spec,
                _out,
                _out / f"{_subject}_K{K}_glmhmm_metrics.parquet",
            )
        ]
        if not _matching_subjects:
            continue
        _arrays_store, _names = load_fit_arrays(
            out_dir=_out,
            arrays_suffix="glmhmm_arrays.npz",
            adapter=adapter,
            df_all=df_all,
            subjects=_matching_subjects,
            emission_cols=list(spec["emission_cols"]),
            k=K,
        )
        _selected = [subject for subject in _matching_subjects if subject in _arrays_store]
        if not _selected:
            continue
        _feature = spec.get("history_feature")
        for _subject in _selected:
            _payload = _arrays_store[_subject]
            _x_cols = [str(col) for col in _payload["X_cols"]]
            if _feature not in _x_cols:
                continue
            _feature_idx = _x_cols.index(_feature)
            for _raw_state in range(K):
                _raw_rows.append(
                    {
                        "subject": str(_subject),
                        "model_id": spec["model_id"],
                        "model_label": spec["label"],
                        "short_label": spec["short_label"],
                        "raw_state": int(_raw_state),
                        "feature": _feature,
                        "weight": float(_payload["emission_weights"][_raw_state, 0, _feature_idx]),
                        "frozen_emissions_json": str(_payload.get("frozen_emissions_json", "")),
                    }
                )
        _views = build_views(_arrays_store, adapter, K, _selected)
        for _subject in _selected:
            _df_sub = df_all.filter(pl.col("subject") == _subject).sort(adapter.sort_col)
            if _df_sub.height != _views[_subject].T:
                continue
            try:
                _trial_frames.append(
                    build_trial_df(
                        _views[_subject],
                        adapter,
                        _df_sub,
                        adapter.behavioral_cols,
                    ).with_columns(
                        pl.lit(spec["model_id"]).alias("model_id"),
                        pl.lit(spec["label"]).alias("model_label"),
                        pl.lit(spec["short_label"]).alias("short_label"),
                        pl.lit(_model_order, dtype=pl.Int64).alias("model_order"),
                    )
                )
            except Exception:
                pass
        _weights = build_emission_weights_df(_views).with_columns(
            pl.lit(spec["model_id"]).alias("model_id"),
            pl.lit(spec["label"]).alias("model_label"),
            pl.lit(spec["short_label"]).alias("short_label"),
        )
        _frames.append(_weights)
    if _frames:
        freeze_weights_df = pl.concat(_frames, how="diagonal")
        _output = freeze_weights_df
    else:
        freeze_weights_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "feature": pl.Utf8,
                "state_label": pl.Utf8,
                "weight": pl.Float64,
                "model_id": pl.Utf8,
                "model_label": pl.Utf8,
                "short_label": pl.Utf8,
            }
        )
        _output = mo.md("No fitted freeze arrays found yet.")
    if _trial_frames:
        freeze_trial_df = pl.concat(_trial_frames, how="diagonal")
    else:
        freeze_trial_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "model_id": pl.Utf8,
                "model_label": pl.Utf8,
                "short_label": pl.Utf8,
                "model_order": pl.Int64,
                "state_label": pl.Utf8,
                "state_label_pred": pl.Utf8,
                "nLicks": pl.Float64,
                "RT": pl.Float64,
                "RT2": pl.Float64,
            }
        )
    if _raw_rows:
        freeze_raw_history_weights_df = pl.from_dicts(_raw_rows)
    else:
        freeze_raw_history_weights_df = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "model_id": pl.Utf8,
                "model_label": pl.Utf8,
                "short_label": pl.Utf8,
                "raw_state": pl.Int64,
                "feature": pl.Utf8,
                "weight": pl.Float64,
                "frozen_emissions_json": pl.Utf8,
            }
        )
    _output
    return freeze_raw_history_weights_df, freeze_trial_df, freeze_weights_df


@app.cell
def _(freeze_raw_history_weights_df, freeze_weights_df, mo, pl):
    if freeze_raw_history_weights_df.is_empty() or freeze_weights_df.is_empty():
        hist0_choice_lag_audit = pl.DataFrame()
        _output = mo.md("No Hist0 choice-lag weights to audit.")
    else:
        _raw_audit = (
            freeze_raw_history_weights_df
            .filter(
                (pl.col("short_label") == "Hist0")
                & (pl.col("feature") == "choice_lag_param")
            )
            .with_columns(
                pl.format("raw {}", pl.col("raw_state")).alias("state"),
                (pl.col("weight").abs() < 1e-8).alias("is_zero"),
                pl.lit("raw fit").alias("view"),
            )
            .group_by(["view", "state"])
            .agg(
                pl.len().alias("n_subjects"),
                pl.sum("is_zero").alias("n_zero"),
                pl.mean("weight").alias("mean_weight"),
                pl.col("weight").abs().mean().alias("mean_abs_weight"),
            )
        )
        _semantic_audit = (
            freeze_weights_df
            .filter(
                (pl.col("short_label") == "Hist0")
                & (pl.col("feature") == "choice_lag_param")
            )
            .with_columns(
                pl.col("state_label").alias("state"),
                (pl.col("weight").abs() < 1e-8).alias("is_zero"),
                pl.lit("semantic label").alias("view"),
            )
            .group_by(["view", "state"])
            .agg(
                pl.len().alias("n_subjects"),
                pl.sum("is_zero").alias("n_zero"),
                pl.mean("weight").alias("mean_weight"),
                pl.col("weight").abs().mean().alias("mean_abs_weight"),
            )
        )
        hist0_choice_lag_audit = pl.concat([_raw_audit, _semantic_audit], how="diagonal").sort(["view", "state"])
        _output = hist0_choice_lag_audit
    _output
    return


@app.cell
def _(
    figsize,
    freeze_aux_auc_df,
    freeze_aux_curve_df,
    freeze_model_order,
    freeze_model_palette,
    mo,
    path_panels,
    plt,
    sns,
):
    _curve_df = freeze_aux_curve_df[freeze_aux_curve_df["metric"].eq("nLicks")]
    _auc_df = freeze_aux_auc_df[freeze_aux_auc_df["metric"].eq("nLicks")]
    if _curve_df.empty or _auc_df.empty:
        _output = mo.md("No nLicks ROC curve rows to plot.")
    else:
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_nlicks_roc_ax = plt.gca()
        for _label in freeze_model_order:
            _label_curve_df = _curve_df[_curve_df["short_label"].astype(str).eq(_label)]
            _label_auc_df = _auc_df[_auc_df["short_label"].astype(str).eq(_label)]
            if _label_curve_df.empty or _label_auc_df.empty:
                continue
            _summary_df = (
                _label_curve_df
                .groupby("fpr", as_index=False)
                .agg(mean_tpr=("tpr", "mean"), sem_tpr=("tpr", "sem"))
            )
            _mean_auc = _label_auc_df["auc"].mean()
            _sem_auc = _label_auc_df["auc"].sem()
            _color = freeze_model_palette.get(_label, "0.5")
            freeze_nlicks_roc_ax.plot(
                _summary_df["fpr"],
                _summary_df["mean_tpr"],
                color=_color,
                linewidth=1.4,
                label=f"{_label} AUC={_mean_auc:.3f} ± {_sem_auc:.3f}",
            )
            freeze_nlicks_roc_ax.fill_between(
                _summary_df["fpr"],
                _summary_df["mean_tpr"] - _summary_df["sem_tpr"].fillna(0),
                _summary_df["mean_tpr"] + _summary_df["sem_tpr"].fillna(0),
                color=_color,
                alpha=0.18,
                edgecolor="none",
                linewidth=0,
            )
        freeze_nlicks_roc_ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, linestyle="--")
        freeze_nlicks_roc_ax.set_title("nLicks ROC")
        freeze_nlicks_roc_ax.set_xlabel("False positive rate")
        freeze_nlicks_roc_ax.set_ylabel("True positive rate")
        freeze_nlicks_roc_ax.set_xlim(0, 1)
        freeze_nlicks_roc_ax.set_ylim(0, 1)
        freeze_nlicks_roc_ax.legend(frameon=False, fontsize=6, loc="lower right")
        sns.despine(ax=freeze_nlicks_roc_ax)
        plt.savefig(f"{path_panels}/freeze_nlicks_roc.svg")
        plt.savefig(f"{path_panels}/freeze_nlicks_roc.png")
        _output = freeze_nlicks_roc_ax
    _output
    return


@app.cell
def _(
    figsize,
    freeze_aux_auc_df,
    freeze_aux_curve_df,
    freeze_model_order,
    freeze_model_palette,
    mo,
    path_panels,
    plt,
    sns,
):
    _curve_df = freeze_aux_curve_df[freeze_aux_curve_df["metric_label"].eq("RT")]
    _auc_df = freeze_aux_auc_df[freeze_aux_auc_df["metric_label"].eq("RT")]
    if _curve_df.empty or _auc_df.empty:
        _output = mo.md("No RT ROC curve rows to plot.")
    else:
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_rt_roc_ax = plt.gca()
        for _label in freeze_model_order:
            _label_curve_df = _curve_df[_curve_df["short_label"].astype(str).eq(_label)]
            _label_auc_df = _auc_df[_auc_df["short_label"].astype(str).eq(_label)]
            if _label_curve_df.empty or _label_auc_df.empty:
                continue
            _summary_df = (
                _label_curve_df
                .groupby("fpr", as_index=False)
                .agg(mean_tpr=("tpr", "mean"), sem_tpr=("tpr", "sem"))
            )
            _mean_auc = _label_auc_df["auc"].mean()
            _sem_auc = _label_auc_df["auc"].sem()
            _color = freeze_model_palette.get(_label, "0.5")
            freeze_rt_roc_ax.plot(
                _summary_df["fpr"],
                _summary_df["mean_tpr"],
                color=_color,
                linewidth=1.4,
                label=f"{_label} AUC={_mean_auc:.3f} ± {_sem_auc:.3f}",
            )
            freeze_rt_roc_ax.fill_between(
                _summary_df["fpr"],
                _summary_df["mean_tpr"] - _summary_df["sem_tpr"].fillna(0),
                _summary_df["mean_tpr"] + _summary_df["sem_tpr"].fillna(0),
                color=_color,
                alpha=0.18,
                edgecolor="none",
                linewidth=0,
            )
        freeze_rt_roc_ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8, linestyle="--")
        freeze_rt_roc_ax.set_title("RT ROC")
        freeze_rt_roc_ax.set_xlabel("False positive rate")
        freeze_rt_roc_ax.set_ylabel("True positive rate")
        freeze_rt_roc_ax.set_xlim(0, 1)
        freeze_rt_roc_ax.set_ylim(0, 1)
        freeze_rt_roc_ax.legend(frameon=False, fontsize=6, loc="lower right")
        sns.despine(ax=freeze_rt_roc_ax)
        plt.savefig(f"{path_panels}/freeze_rt_roc.svg")
        plt.savefig(f"{path_panels}/freeze_rt_roc.png")
        _output = freeze_rt_roc_ax
    _output
    return


@app.cell
def _(
    add_paired_state_annotation,
    add_subject_pair_lines,
    boxplot_STYLE,
    figsize,
    freeze_weights_df,
    mo,
    path_panels,
    pl,
    plt,
    sns,
    state_hue_order,
    state_palette,
):
    if freeze_weights_df.is_empty():
        _output = mo.md("No bias weights to plot.")
    else:
        _plot_df = (
            freeze_weights_df
            .filter(pl.col("feature") == "bias")
            .to_pandas()
        )
        _order = ["Free", "Both0", "Stim0", "Hist0"]
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_bias_weight_ax = plt.gca()
        sns.boxplot(
            data=_plot_df,
            x="short_label",
            y="weight",
            hue="state_label",
            order=_order,
            hue_order=state_hue_order,
            palette=state_palette,
            gap=0.2,
            ax=freeze_bias_weight_ax,
            **boxplot_STYLE,
        )
        add_subject_pair_lines(
            freeze_bias_weight_ax,
            _plot_df,
            x="short_label",
            y="weight",
            order=_order,
            hue_order=state_hue_order,
        )
        add_paired_state_annotation(
            freeze_bias_weight_ax,
            _plot_df,
            x="short_label",
            y="weight",
            order=_order,
            hue_order=state_hue_order,
        )
        freeze_bias_weight_ax.axhline(0, linestyle="--", color="0.6", linewidth=0.8, zorder=0)
        freeze_bias_weight_ax.set_title("bias")
        freeze_bias_weight_ax.set_xlabel("")
        freeze_bias_weight_ax.set_ylabel("Emission weight")
        freeze_bias_weight_ax.legend(frameon=False, fontsize=7, title="")
        sns.despine(ax=freeze_bias_weight_ax)
        plt.savefig(f"{path_panels}/freeze_bias_weight.svg")
        plt.savefig(f"{path_panels}/freeze_bias_weight.png")
        _output = freeze_bias_weight_ax
    _output
    return


@app.cell
def _(
    add_paired_state_annotation,
    add_subject_pair_lines,
    boxplot_STYLE,
    figsize,
    freeze_weights_df,
    mo,
    path_panels,
    pl,
    plt,
    sns,
    state_hue_order,
    state_palette,
):
    if freeze_weights_df.is_empty():
        _output = mo.md("No stimulus weights to plot.")
    else:
        _plot_df = (
            freeze_weights_df
            .filter(pl.col("feature") == "stim_param")
            .to_pandas()
        )
        _order = ["Free", "Both0", "Stim0", "Hist0"]
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_stim_weight_ax = plt.gca()
        sns.boxplot(
            data=_plot_df,
            x="short_label",
            y="weight",
            hue="state_label",
            order=_order,
            hue_order=state_hue_order,
            palette=state_palette,
            gap=0.2,
            ax=freeze_stim_weight_ax,
            **boxplot_STYLE,
        )
        add_subject_pair_lines(
            freeze_stim_weight_ax,
            _plot_df,
            x="short_label",
            y="weight",
            order=_order,
            hue_order=state_hue_order,
        )
        add_paired_state_annotation(
            freeze_stim_weight_ax,
            _plot_df,
            x="short_label",
            y="weight",
            order=_order,
            hue_order=state_hue_order,
        )
        freeze_stim_weight_ax.axhline(0, linestyle="--", color="0.6", linewidth=0.8, zorder=0)
        freeze_stim_weight_ax.set_title("stim_param")
        freeze_stim_weight_ax.set_xlabel("")
        freeze_stim_weight_ax.set_ylabel("Emission weight")
        freeze_stim_weight_ax.legend(frameon=False, fontsize=7, title="")
        sns.despine(ax=freeze_stim_weight_ax)
        plt.savefig(f"{path_panels}/freeze_stim_weight.svg")
        plt.savefig(f"{path_panels}/freeze_stim_weight.png")
        _output = freeze_stim_weight_ax
    _output
    return


@app.cell
def _(
    add_paired_state_annotation,
    add_subject_pair_lines,
    boxplot_STYLE,
    figsize,
    freeze_weights_df,
    mo,
    path_panels,
    pl,
    plt,
    sns,
    state_hue_order,
    state_palette,
):
    if freeze_weights_df.is_empty():
        _output = mo.md("No choice-lag weights to plot.")
    else:
        _plot_df = (
            freeze_weights_df
            .filter(pl.col("feature") == "choice_lag_param")
            .to_pandas()
        )
        _order = ["Free", "Both0", "Stim0", "Hist0"]
        plt.figure(figsize=figsize, constrained_layout=True)
        freeze_choice_lag_weight_ax = plt.gca()
        sns.boxplot(
            data=_plot_df,
            x="short_label",
            y="weight",
            hue="state_label",
            order=_order,
            hue_order=state_hue_order,
            palette=state_palette,
            gap=0.2,
            ax=freeze_choice_lag_weight_ax,
            **boxplot_STYLE,
        )
        add_subject_pair_lines(
            freeze_choice_lag_weight_ax,
            _plot_df,
            x="short_label",
            y="weight",
            order=_order,
            hue_order=state_hue_order,
        )
        add_paired_state_annotation(
            freeze_choice_lag_weight_ax,
            _plot_df,
            x="short_label",
            y="weight",
            order=_order,
            hue_order=state_hue_order,
        )
        freeze_choice_lag_weight_ax.axhline(0, linestyle="--", color="0.6", linewidth=0.8, zorder=0)
        freeze_choice_lag_weight_ax.set_title("choice_lag_param")
        freeze_choice_lag_weight_ax.set_xlabel("")
        freeze_choice_lag_weight_ax.set_ylabel("Emission weight")
        freeze_choice_lag_weight_ax.legend(frameon=False, fontsize=7, title="")
        sns.despine(ax=freeze_choice_lag_weight_ax)
        plt.savefig(f"{path_panels}/freeze_choice_lag_weight.svg")
        plt.savefig(f"{path_panels}/freeze_choice_lag_weight.png")
        _output = freeze_choice_lag_weight_ax
    _output
    return


if __name__ == "__main__":
    app.run()
