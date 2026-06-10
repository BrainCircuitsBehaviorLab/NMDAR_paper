from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Literal
import warnings

import matplotlib

matplotlib.use("Agg")
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


PROJECT_ROOT = next(
    (
        p
        for base in (Path.cwd(), Path(__file__).resolve())
        for p in (base, *base.parents)
        if (p / "config.toml").exists() and (p / "src").exists()
    ),
    Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"Task key .* already registered",
    category=RuntimeWarning,
)

from glmhmmt.cli.fit_glmhmm import main as fit_glmhmm  # noqa: E402
from glmhmmt.cli.fit_glmhmmt import main as fit_glmhmmt  # noqa: E402
from glmhmmt.notebook_support.analysis_common import (  # noqa: E402
    build_trial_and_weights_df,
    load_fit_arrays,
)
from glmhmmt.postprocess import (  # noqa: E402
    build_emission_weights_df,
    build_state_accuracy_payload,
    build_state_posterior_count_payload,
    build_transition_matrix_by_subject_payload,
    build_transition_matrix_payload,
    build_transition_weights_df,
)
import glmhmmt.plots as model_plots  # noqa: E402
from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config  # noqa: E402
from glmhmmt.tasks import get_adapter  # noqa: E402
from glmhmmt.views import build_views  # noqa: E402

from src.process.common import adapter_behavioral_column  # noqa: E402
from src.utils import fig_size  # noqa: E402


ModelKind = Literal["glmhmm", "glmhmmt"]
MODEL_KINDS = ("glmhmm", "glmhmmt")
FIT_FUNCTIONS = {
    "glmhmm": fit_glmhmm,
    "glmhmmt": fit_glmhmmt,
}


def prepare_predictions_df(task_name: str, df: pl.DataFrame, adapter) -> pd.DataFrame:
    adapter_module = sys.modules.get(adapter.__module__)
    prepare_fn = getattr(adapter_module, "prepare_predictions_df", None)
    if prepare_fn is None and task_name == "2AFC_DRUG":
        prepare_fn = importlib.import_module("src.process.two_afc").prepare_predictions_df
    if prepare_fn is None and task_name in {"2ADC_DRUG", "2AFC_delay_DRUG"}:
        prepare_fn = importlib.import_module("src.process.two_adc").prepare_predictions_df
    if prepare_fn is None:
        raise AttributeError(f"{adapter.__module__} does not define prepare_predictions_df")
    if task_name == "MCDR":
        return prepare_fn(df, cfg=load_app_config())
    return prepare_fn(df)


def read_fit_config(fit_dir: Path) -> dict:
    config_path = fit_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    return json.loads(config_path.read_text())


def infer_fit_location(
    fit_dir: Path,
    cfg: dict,
    model_kind: str | None = None,
) -> tuple[str, ModelKind, str]:
    parts = fit_dir.resolve().parts
    task = cfg.get("task")
    path_model_kind = None
    model_id = cfg.get("model_id") or fit_dir.name
    if "fits" in parts:
        idx = parts.index("fits")
        if len(parts) > idx + 3:
            task = task or parts[idx + 1]
            path_model_kind = parts[idx + 2]
            model_id = model_id or parts[idx + 3]
    model_kind = model_kind or cfg.get("model_kind") or cfg.get("model_type") or path_model_kind
    if model_kind not in MODEL_KINDS:
        raise ValueError(
            f"Could not infer model kind for {fit_dir}; pass --model-kind "
            f"with one of {', '.join(MODEL_KINDS)}."
        )
    if not task:
        raise ValueError(f"Could not infer task for {fit_dir}")
    return str(task), model_kind, str(model_id)


def plot_format_from_config(paths, override: str | None) -> str:
    if override:
        return override
    cfg = load_app_config(paths.CONFIG)
    return str(cfg.get("plot-saver", {}).get("format", "png")).lstrip(".") or "png"


def figure_from(plot_obj):
    if isinstance(plot_obj, (list, tuple)):
        if not plot_obj:
            raise ValueError("Cannot save an empty plot object.")
        return figure_from(plot_obj[0])
    if hasattr(plot_obj, "savefig"):
        return plot_obj
    if hasattr(plot_obj, "figure"):
        return plot_obj.figure
    raise TypeError(f"Cannot resolve a matplotlib figure from {type(plot_obj)!r}")


def save_figure(fig, out_dir: Path, stem: str, fmt: str, dpi: int) -> Path:
    fig = figure_from(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def save_axis_figure(ax, out_dir: Path, stem: str, fmt: str, dpi: int) -> Path:
    fig = ax.figure
    return save_figure(fig, out_dir, stem, fmt, dpi)


def clean_title(fig_or_ax) -> None:
    fig = getattr(fig_or_ax, "figure", fig_or_ax)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")


def pick_regressor(views: dict, fallback: str = "at_choice") -> str:
    feature_names: list[str] = []
    for view in views.values():
        for feature in list(getattr(view, "feat_names", []) or []):
            if feature not in feature_names:
                feature_names.append(str(feature))
    if not feature_names:
        return fallback
    return fallback if fallback in feature_names else feature_names[0]


def pick_column(df, candidates: list[str | None]) -> str | None:
    for candidate in candidates:
        if candidate and candidate in df.columns:
            return candidate
    return None


def plot_state_nlicks(
    trial_df_sel: pl.DataFrame,
    df_all: pl.DataFrame,
    adapter,
) -> plt.Figure | None:
    trial_pdf = trial_df_sel.to_pandas()
    _state_col = pick_column(trial_pdf, ["state_label", "state_label_pred"])
    if _state_col is None:
        return None

    behavior_metric_candidates = [
        "nLicks",
        "RT",
        "RT2",
        "response_time_first",
        "reaction_time",
        "ReactionTime",
        "timepoint_4",
    ]
    state_behavior_df = trial_pdf.copy()
    missing_behavior_cols = [
        col
        for col in behavior_metric_candidates
        if col not in state_behavior_df.columns and col in df_all.columns
    ]
    if missing_behavior_cols:
        session_col = adapter_behavioral_column(adapter, df_all, "session", "session", "Session")
        trial_col = adapter_behavioral_column(adapter, df_all, "trial_idx", "trial_idx", "trial", "Trial")
        if (
            session_col is not None
            and trial_col is not None
            and {"subject", "session", "trial_idx"}.issubset(state_behavior_df.columns)
        ):
            behavior_pdf = (
                df_all.select(["subject", session_col, trial_col, *missing_behavior_cols])
                .rename({session_col: "session", trial_col: "trial_idx"})
                .to_pandas()
            )
            state_behavior_df = state_behavior_df.merge(
                behavior_pdf,
                on=["subject", "session", "trial_idx"],
                how="left",
            )

    if "nLicks" not in state_behavior_df.columns:
        return None

    state_nlicks_df = state_behavior_df.copy()
    state_nlicks_df["nLicks"] = pd.to_numeric(state_nlicks_df["nLicks"], errors="coerce")
    state_nlicks_df = state_nlicks_df.dropna(subset=["nLicks", _state_col])
    if state_nlicks_df.empty:
        return None

    plot_df = (
        state_nlicks_df
        .groupby(["subject", _state_col], as_index=False)["nLicks"]
        .median()
    )
    if plot_df.empty:
        return None

    _fig_nlicks, _ax_nlicks = plt.subplots(figsize=fig_size(3, 1))
    sns.boxplot(
        data=plot_df,
        x=_state_col,
        hue=_state_col,
        y="nLicks",
        ax=_ax_nlicks,
        showfliers=False,
        palette={"Engaged": "tab:green", "Disengaged": "tab:gray"},
        order=["Engaged", "Disengaged"],
        fill=False,
        whiskerprops={"color": "gray"},
        boxprops={"color": "gray"},
        medianprops={"linewidth": 2},
        showcaps=False,
    )
    _ax_nlicks.set_xlabel("State")
    _ax_nlicks.set_ylabel("Number of licks in correct trials")
    _ax_nlicks.tick_params(axis="x")
    sns.despine(ax=_ax_nlicks)
    _fig_nlicks.tight_layout()
    return _fig_nlicks


def fit(fit_dir: Path, cfg: dict, args, *, model_kind: ModelKind) -> None:
    if not args.fit:
        return

    task = str(cfg["task"])
    K_list = [int(args.K)] if args.K is not None else [int(k) for k in cfg.get("K_list", [2])]
    cv_mode = str(cfg.get("cv_mode", "none"))
    cv_repeats = int(cfg.get("cv_repeats", 0)) if cv_mode != "none" else 0
    n_restarts = args.n_restarts
    if n_restarts is None:
        n_restarts = 1 if model_kind == "glmhmmt" or cv_mode != "none" else 5

    fit_kwargs = dict(
        subjects=[str(subject) for subject in cfg.get("subjects") or []] or None,
        K_list=K_list,
        num_iters=int(args.num_iters),
        n_restarts=int(n_restarts),
        base_seed=int(args.seed),
        out_dir=fit_dir,
        tau=float(cfg.get("tau", 50.0)),
        emission_cols=list(cfg.get("emission_cols") or []),
        frozen_emissions=cfg.get("frozen_emissions") or None,
        task=task,
        cv_mode=cv_mode,
        cv_repeats=cv_repeats,
        verbose=True,
        baseline_class_idx=int(cfg.get("baseline_class_idx", get_adapter(task).baseline_class_idx)),
    )

    if model_kind == "glmhmm":
        fit_kwargs["condition_filter"] = str(cfg.get("condition_filter", "all"))
    else:
        if str(cfg.get("condition_filter", "all")) != "all":
            raise ValueError("GLM-HMM-T fitting does not support condition_filter; use saved arrays or refit with all.")
        fit_kwargs["transition_cols"] = list(cfg.get("transition_cols") or [])

    FIT_FUNCTIONS[model_kind](**fit_kwargs)


def save_model_plots(fit_dir: Path, args, *, model_kind: str | None = None) -> list[Path]:
    cfg = read_fit_config(fit_dir)
    task_name, model_kind, model_id = infer_fit_location(fit_dir, cfg, model_kind)
    fit(fit_dir, cfg, args, model_kind=model_kind)

    paths = get_runtime_paths()
    fmt = plot_format_from_config(paths, args.format)
    out_dir = paths.RESULTS / "plots" / task_name / model_kind / model_id

    adapter = get_adapter(task_name)
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    df_all = adapter.filter_condition_df(df_all, str(cfg.get("condition_filter", "all")))
    plots = adapter.get_plots()

    K = int(args.K) if args.K is not None else int((cfg.get("K_list") or [2])[0])
    cfg_subjects = [str(subject) for subject in cfg.get("subjects") or []]
    subjects = [str(subject) for subject in args.subjects] if args.subjects else cfg_subjects
    emission_cols = list(cfg.get("emission_cols") or [])
    transition_cols = list(cfg.get("transition_cols") or [])

    arrays_store, _names = load_fit_arrays(
        out_dir=fit_dir,
        arrays_suffix=f"{model_kind}_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=subjects,
        emission_cols=emission_cols,
        transition_cols=transition_cols if model_kind == "glmhmmt" else None,
        k=K,
    )
    if not arrays_store:
        raise RuntimeError(f"No fitted arrays found in {fit_dir}")

    selected = [subject for subject in subjects if subject in arrays_store] or sorted(arrays_store)
    if args.scoring_key and hasattr(adapter, "scoring_key"):
        adapter.scoring_key = args.scoring_key
    views = build_views(arrays_store, adapter, K, selected)
    state_labels = {subject: view.state_name_by_idx for subject, view in views.items()}
    views_sel = {subject: views[subject] for subject in selected if subject in views}

    trial_df, _weights_df = build_trial_and_weights_df(
        df_all,
        views=views_sel,
        adapter=adapter,
        min_session_length=2,
    )
    if trial_df.height == 0:
        raise RuntimeError(
            "No subjects with matching data lengths. "
            "This usually means the saved arrays were fit against an older processed dataset "
            "or a different condition filter."
        )
    trial_df_sel = trial_df.filter(pl.col("subject").is_in(selected))
    plot_df_all = prepare_predictions_df(task_name, trial_df_sel, adapter)
    regressor = args.regressor or pick_regressor(views_sel)

    saved: list[Path] = []
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.style.use(PROJECT_ROOT / "styles" / "paper.mplstyle")

    weights_df = build_emission_weights_df(views_sel)
    if not weights_df.is_empty():
        fig_em_subj, _axes = model_plots.emission_weights_by_subject(weights_df, K=K)
        saved.append(save_figure(fig_em_subj, out_dir, "emissions_by_subject", fmt, args.dpi))

        fig_em_summary, ax_em_summary = plt.subplots(figsize=(4, 3))
        ax_em_summary = model_plots.emission_weights_summary_boxplot(
            weights_df,
            K=K,
            connect_subjects=True,
            show_ttests=True,
            ax=ax_em_summary,
            tick_rotation=0,
        )
        saved.append(save_axis_figure(ax_em_summary, out_dir, "emissions_boxplot", fmt, args.dpi))

    # GLM-HMM-T adds transition regressors; plot them beside the shared state diagnostics.
    if model_kind == "glmhmmt":
        transition_weights_df = build_transition_weights_df(views_sel)
        if not transition_weights_df.is_empty():
            fig_trans_weights, _axes = model_plots.transition_weights_by_subject(transition_weights_df, K=K)
            saved.append(
                save_figure(fig_trans_weights, out_dir, "transition_weights_by_subject", fmt, args.dpi)
            )

            fig_trans_summary, ax_trans_summary = plt.subplots(figsize=(4, 3))
            ax_trans_summary = model_plots.transition_weights_summary_boxplot(
                transition_weights_df,
                K=K,
                connect_subjects=True,
                show_ttests=True,
                ax=ax_trans_summary,
                tick_rotation=0,
            )
            saved.append(
                save_axis_figure(ax_trans_summary, out_dir, "transition_weights_boxplot", fmt, args.dpi)
            )

    by_subject_payload = build_transition_matrix_by_subject_payload(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    if by_subject_payload.get("matrices"):
        fig_transition_by_subject = model_plots.transition_matrix_by_subject(**by_subject_payload)
        saved.append(save_figure(fig_transition_by_subject, out_dir, "transition_matrix_by_subject", fmt, args.dpi))

    summary_payload = build_transition_matrix_payload(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    if np.asarray(summary_payload.get("matrix", [])).size:
        fig_transition, ax_transition = plt.subplots(figsize=fig_size(2, 1))
        ax_transition = model_plots.transition_matrix(**summary_payload, ax=ax_transition)
        clean_title(ax_transition)
        saved.append(save_axis_figure(ax_transition, out_dir, "mean_transition_matrix", fmt, args.dpi))

    views_for_plot = views_sel
    state_plot_kwargs = dict(
        background_style="model",
        show_weighted_points=True,
        show_data_smooth=True,
        show_model_smooth=True,
        model_line_mode="smooth",
        state_assignment_mode="map",
        figure_dpi=args.dpi,
    )

    try:
        fig_overall, _ = plots.plot_categorical_performance_all(
            plot_df_all,
            f"{model_kind} K={K}",
            background_style="model",
            views=views_for_plot,
        )
        fig_list = list(fig_overall) if isinstance(fig_overall, (list, tuple)) else [fig_overall]
        for idx, fig in enumerate(fig_list, start=1):
            clean_title(fig)
            stem = f"categorical_overall_{idx}" if len(fig_list) > 1 else "categorical_overall"
            saved.append(save_figure(fig, out_dir, stem, fmt, args.dpi))
    except Exception as exc:
        print(f"Skipping categorical_overall: {exc}")

    if hasattr(plots, "plot_categorical_performance_state_overlay"):
        try:
            fig_overlay, ax_overlay = plt.subplots(figsize=fig_size(2, 1))
            fig_overlay, _ = plots.plot_categorical_performance_state_overlay(
                df=plot_df_all,
                views=views_for_plot,
                model_name=f"{model_kind} K={K} - all states",
                ax=ax_overlay,
                **state_plot_kwargs,
            )
            clean_title(fig_overlay)
            saved.append(save_figure(fig_overlay, out_dir, "categorical_state_overlay", fmt, args.dpi))
        except Exception as exc:
            print(f"Skipping categorical_state_overlay: {exc}")

    if hasattr(plots, "plot_categorical_performance_by_state"):
        try:
            fig_by_state, _ = plots.plot_categorical_performance_by_state(
                df=plot_df_all,
                views=views_for_plot,
                model_name=f"{model_kind} K={K} - per state",
                **state_plot_kwargs,
            )
            clean_title(fig_by_state)
            saved.append(save_figure(fig_by_state, out_dir, "categorical_by_state", fmt, args.dpi))
        except Exception as exc:
            print(f"Skipping categorical_by_state: {exc}")

    if hasattr(plots, "plot_regressor_psychometric_by_state") and regressor in plot_df_all.columns:
        try:
            fig_reg_overlay, ax_reg_overlay = plt.subplots(figsize=fig_size(2, 1))
            fig_reg_overlay, _ = plots.plot_regressor_psychometric_by_state(
                df=plot_df_all,
                views=views_for_plot,
                model_name=f"{model_kind} K={K}",
                feature_col=regressor,
                overlay_only=True,
                ax=ax_reg_overlay,
                **state_plot_kwargs,
            )
            clean_title(fig_reg_overlay)
            saved.append(save_figure(fig_reg_overlay, out_dir, f"regressor_state_overlay_{regressor}", fmt, args.dpi))
        except Exception as exc:
            print(f"Skipping regressor_state_overlay_{regressor}: {exc}")

        try:
            axes = None
            if K > 1:
                _fig_reg_state, axes = plt.subplots(1, K, figsize=(4 * K, 4), sharey=True)
            fig_reg_state, _ = plots.plot_regressor_psychometric_by_state(
                df=plot_df_all,
                views=views_for_plot,
                model_name=f"{model_kind} K={K}",
                feature_col=regressor,
                axes=axes,
                **state_plot_kwargs,
            )
            clean_title(fig_reg_state)
            saved.append(save_figure(fig_reg_state, out_dir, f"regressor_by_state_{regressor}", fmt, args.dpi))
        except Exception as exc:
            print(f"Skipping regressor_by_state_{regressor}: {exc}")

    try:
        fig_acc, ax_acc = plt.subplots(figsize=fig_size(3, 1))
        ax_acc = model_plots.state_accuracy(
            build_state_accuracy_payload(
                trial_df_sel,
                performance_col="correct_bool",
                chance_level=1.0 / adapter.num_classes,
            ),
            ax=ax_acc,
        )
        clean_title(ax_acc)
        saved.append(save_axis_figure(ax_acc, out_dir, "state_accuracy", fmt, args.dpi))
    except Exception as exc:
        print(f"Skipping state_accuracy: {exc}")

    try:
        fig_post, ax_post = plt.subplots(figsize=fig_size(3, 1))
        ax_post = model_plots.state_posterior_count_kde(
            build_state_posterior_count_payload(trial_df_sel),
            ax=ax_post,
            figsize=fig_size(3, 1),
        )
        clean_title(ax_post)
        saved.append(save_axis_figure(ax_post, out_dir, "state_posterior_count_kde", fmt, args.dpi))
    except Exception as exc:
        print(f"Skipping state_posterior_count_kde: {exc}")

    fig_nlicks = plot_state_nlicks(trial_df_sel, df_all, adapter)
    if fig_nlicks is not None:
        saved.append(save_figure(fig_nlicks, out_dir, "state_nlicks", fmt, args.dpi))

    return saved


def resolve_fit_dirs(args) -> list[Path]:
    paths = get_runtime_paths()
    fit_dirs = [Path(path).expanduser().resolve() for path in args.fit_dir]
    if args.model_id:
        if not args.task:
            raise ValueError("--task is required when using --model-id")
        fit_dirs.extend(
            paths.RESULTS / "fits" / args.task / args.model_kind / model_id
            for model_id in args.model_id
        )
    if not fit_dirs:
        raise ValueError("Pass --fit-dir or --task with one or more --model-id values.")
    return fit_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or load saved GLM-HMM/GLM-HMM-T fits and save notebook-style plots.",
    )
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--task", default=None, help="Task name, required with --model-id.")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, default="glmhmm")
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Saved model id under results/fits/<task>/<model-kind>.",
    )
    parser.add_argument("--fit-dir", action="append", default=[], help="Direct path to a saved model fit directory.")
    parser.add_argument("--fit", action="store_true", help="Rerun the fit from each saved config before plotting.")
    parser.add_argument("--K", type=int, default=None, help="K to load. Defaults to the first K in config.json.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional subset of fitted subjects to plot.")
    parser.add_argument("--scoring-key", default=None, help="Optional adapter state-scoring key.")
    parser.add_argument("--regressor", default=None, help="Optional regressor for per-state psychometric plots.")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        config_path=args.config_path,
    )
    all_saved: list[Path] = []
    for fit_dir in resolve_fit_dirs(args):
        saved = save_model_plots(fit_dir, args, model_kind=None if args.fit_dir else args.model_kind)
        all_saved.extend(saved)
        print(f"Saved {len(saved)} plot(s) under {saved[0].parent if saved else fit_dir}")
    if all_saved:
        print("Saved files:")
        for path in all_saved:
            print(path)


if __name__ == "__main__":
    main()
