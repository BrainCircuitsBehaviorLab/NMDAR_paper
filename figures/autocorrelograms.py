from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from src.utils import fig_size


# Hardcoded analysis selection.
TASK = "2AFC"
GLM_MODEL_ID = "one hot"
GLMHMM_MODEL_ID = "param2"

N_SIMULATIONS = 1
MAX_LAG = 50
MIN_CROSS_PAIRS = 20
MAX_CROSS_PAIRS = 80
SEED = 1


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

warnings.filterwarnings("ignore", message=r"Task key .* already registered.*")

from glmhmmt.glm import glm_probs_from_weights  # noqa: E402
from glmhmmt.notebook_support.analysis_common import (  # noqa: E402
    load_fit_arrays,
    select_subject_behavior_df,
)
from glmhmmt.runtime import configure_paths, get_runtime_paths  # noqa: E402
from glmhmmt.tasks import get_adapter  # noqa: E402
from glmhmmt.tasks.fitted_regressors import (  # noqa: E402
    mean_feature_weights_from_fit,
    resolved_source_features,
    subject_feature_weights_from_fit,
)

import src.process.two_afc  # noqa: E402,F401
from src.process.common import prepare_corrected_behavior_autocorrelograms  # noqa: E402


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model config: {path}")
    return json.loads(path.read_text())


def array_subjects(out_dir: Path, suffix: str, *, k: int | None = None) -> list[str]:
    subjects = []
    pattern = f"*_{suffix}" if k is None else f"*_K{k}_{suffix}"
    for path in sorted(out_dir.glob(pattern)):
        subject = path.name.removesuffix(f"_{suffix}")
        if k is not None:
            subject = subject.removesuffix(f"_K{k}")
        subjects.append(subject)
    return subjects


def fitted_lag_weights(adapter, subject: str, target_col: str) -> dict[int, float]:
    spec_attr = {
        "choice_lag_param": "choice_lag_param_spec",
        "at_choice_param": "at_choice_param_spec",
    }.get(target_col)
    if spec_attr is None or not hasattr(adapter, spec_attr):
        return {}

    spec = getattr(adapter, spec_attr)
    try:
        weights = subject_feature_weights_from_fit(spec, subject)
    except (FileNotFoundError, ValueError):
        weights = mean_feature_weights_from_fit(spec)

    out = {}
    for feature in resolved_source_features(spec):
        match = re.fullmatch(r"choice_lag_(\d+)", str(feature))
        if match and feature in weights:
            out[int(match.group(1))] = float(weights[feature])
    return out


def class_count(arrays: dict) -> int:
    p_pred = np.asarray(arrays.get("p_pred", []), dtype=float)
    if p_pred.ndim == 2 and p_pred.shape[1] > 0:
        return int(p_pred.shape[1])
    y = np.asarray(arrays.get("y", []), dtype=float)
    finite = y[np.isfinite(y)]
    return int(np.nanmax(finite) + 1) if finite.size else 2


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    out = np.asarray(probs, dtype=float).copy()
    out = np.clip(out, 1e-12, np.inf)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0:
        return np.full_like(out, 1.0 / out.size, dtype=float)
    return out / total


def apply_lapse(
    probs: np.ndarray,
    *,
    previous_choice: int | None,
    lapse_mode: str,
    lapse_rates: np.ndarray,
) -> np.ndarray:
    out = normalize_probs(probs)
    if previous_choice is None:
        return out

    num_classes = out.size
    lapse_rates = np.asarray(lapse_rates, dtype=float).reshape(-1)
    if lapse_mode == "class" and lapse_rates.size:
        total_mass = float(np.sum(lapse_rates))
        out = lapse_rates[:num_classes] + (1.0 - total_mass) * out
    elif lapse_mode == "history":
        repeat_rate = float(lapse_rates[0]) if lapse_rates.size > 0 else 0.0
        alternate_rate = float(lapse_rates[1]) if lapse_rates.size > 1 else 0.0
        repeat_target = np.zeros(num_classes, dtype=float)
        repeat_target[int(previous_choice)] = 1.0
        alternate_target = np.full(num_classes, 1.0 / max(1, num_classes - 1), dtype=float)
        alternate_target[int(previous_choice)] = 0.0
        out = (1.0 - repeat_rate - alternate_rate) * out
        out += repeat_rate * repeat_target + alternate_rate * alternate_target
    elif lapse_mode == "history_conditioned":
        repeat_rates = lapse_rates[:num_classes]
        alternate_rates = lapse_rates[num_classes : 2 * num_classes]
        repeat_rate = float(repeat_rates[int(previous_choice)]) if repeat_rates.size > previous_choice else 0.0
        alternate_rate = (
            float(alternate_rates[int(previous_choice)])
            if alternate_rates.size > previous_choice
            else 0.0
        )
        repeat_target = np.zeros(num_classes, dtype=float)
        repeat_target[int(previous_choice)] = 1.0
        alternate_target = np.full(num_classes, 1.0 / max(1, num_classes - 1), dtype=float)
        alternate_target[int(previous_choice)] = 0.0
        out = (1.0 - repeat_rate - alternate_rate) * out
        out += repeat_rate * repeat_target + alternate_rate * alternate_target
    return normalize_probs(out)


def infer_correct_class(subject_df: pd.DataFrame, adapter) -> np.ndarray:
    stimulus_col = adapter.behavioral_cols["stimulus"]
    stimulus = pd.to_numeric(subject_df[stimulus_col], errors="coerce").to_numpy(dtype=float)
    return np.where(
        np.isin(stimulus, [0.0, 1.0]),
        stimulus,
        np.where(np.isfinite(stimulus), (stimulus > 0.0).astype(float), np.nan),
    )


def session_starts(sessions: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(sessions), dtype=int)
    start = 0
    for idx in range(len(sessions)):
        if idx > 0 and sessions[idx] != sessions[idx - 1]:
            start = idx
        starts[idx] = start
    return starts


def history_value(
    choices: np.ndarray,
    trial_idx: int,
    lag: int,
    starts: np.ndarray,
) -> float:
    source_idx = trial_idx - int(lag)
    if source_idx < starts[trial_idx]:
        return 0.0
    choice = choices[source_idx]
    if not np.isfinite(choice):
        return 0.0
    return float(2.0 * int(choice) - 1.0)


def closed_loop_x(
    base_x: np.ndarray,
    *,
    trial_idx: int,
    choices: np.ndarray,
    starts: np.ndarray,
    x_cols: list[str],
    lag_param_weights: dict[str, dict[int, float]],
) -> np.ndarray:
    x = np.asarray(base_x, dtype=float).copy()
    for col_idx, col in enumerate(x_cols):
        match = re.fullmatch(r"choice_lag_(\d+)", str(col))
        if match:
            x[col_idx] = history_value(choices, trial_idx, int(match.group(1)), starts)
        elif col in lag_param_weights:
            x[col_idx] = sum(
                weight * history_value(choices, trial_idx, lag, starts)
                for lag, weight in lag_param_weights[col].items()
            )
        elif col == "prev_choice":
            x[col_idx] = history_value(choices, trial_idx, 1, starts)
    return x


def simulate_subject_closed_loop(
    subject_df: pd.DataFrame,
    arrays: dict,
    *,
    adapter,
    subject: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base_X = np.asarray(arrays["X"], dtype=float)
    x_cols = [str(v) for v in np.asarray(arrays.get("X_cols", []), dtype=object).tolist()]
    if base_X.ndim != 2 or base_X.shape[0] != len(subject_df):
        raise ValueError(f"{subject}: X rows ({base_X.shape}) do not match data rows ({len(subject_df)}).")

    weights = np.asarray(arrays["emission_weights"], dtype=float)
    if weights.ndim == 2:
        weights = weights[None, :, :]
    if weights.ndim != 3:
        raise ValueError(f"{subject}: expected emission_weights with 3 dimensions, got {weights.shape}.")

    K = int(weights.shape[0])
    num_classes = class_count(arrays)
    baseline_class_idx = int(np.asarray(arrays.get("baseline_class_idx", 0)).reshape(()))
    lapse_mode = str(np.asarray(arrays.get("lapse_mode", "none")).reshape(()))
    lapse_rates = np.asarray(arrays.get("lapse_rates", []), dtype=float)
    transition_matrix = np.asarray(arrays.get("transition_matrix", np.eye(K)), dtype=float)
    initial_probs = normalize_probs(np.asarray(arrays.get("initial_probs", np.ones(K) / K), dtype=float))

    sessions = subject_df[adapter.behavioral_cols["session"]].to_numpy()
    starts = session_starts(sessions)
    choices = np.full(len(subject_df), np.nan, dtype=float)
    states = np.zeros(len(subject_df), dtype=int)
    lag_param_weights = {
        col: fitted_lag_weights(adapter, subject, col)
        for col in ("choice_lag_param", "at_choice_param")
        if col in x_cols
    }

    for trial_idx in range(len(subject_df)):
        if trial_idx == starts[trial_idx]:
            state = int(rng.choice(K, p=initial_probs))
        else:
            state_probs = normalize_probs(transition_matrix[states[trial_idx - 1]])
            state = int(rng.choice(K, p=state_probs))
        states[trial_idx] = state

        x_trial = closed_loop_x(
            base_X[trial_idx],
            trial_idx=trial_idx,
            choices=choices,
            starts=starts,
            x_cols=x_cols,
            lag_param_weights=lag_param_weights,
        )
        probs = glm_probs_from_weights(
            x_trial[None, :],
            weights[state],
            baseline_class_idx=baseline_class_idx,
            num_classes=num_classes,
        )[0]
        previous_choice = int(choices[trial_idx - 1]) if trial_idx > starts[trial_idx] else None
        probs = apply_lapse(
            probs,
            previous_choice=previous_choice,
            lapse_mode=lapse_mode,
            lapse_rates=lapse_rates,
        )
        choices[trial_idx] = int(rng.choice(num_classes, p=probs))

    correct_class = infer_correct_class(subject_df, adapter)
    performance = (choices == correct_class).astype(float)
    performance[~np.isfinite(correct_class)] = np.nan
    return choices, performance


def prepare_closed_loop_model_autocorrelograms(
    df_all: pl.DataFrame,
    arrays_store: dict,
    *,
    adapter,
    n_simulations: int,
    max_lag: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    frames = []
    for subject, arrays in arrays_store.items():
        subject_df_pl = select_subject_behavior_df(
            df_all,
            subject=subject,
            sort_col=adapter.sort_col,
            session_col=adapter.session_col,
            min_session_length=2,
        )
        if subject_df_pl.height == 0:
            continue
        subject_df = subject_df_pl.to_pandas()
        sessions = subject_df[adapter.behavioral_cols["session"]].to_numpy()
        trial_index = subject_df[adapter.behavioral_cols["trial"]].to_numpy()
        for sim_idx in range(int(n_simulations)):
            choices, performance = simulate_subject_closed_loop(
                subject_df,
                arrays,
                adapter=adapter,
                subject=str(subject),
                rng=rng,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "subject": f"{subject}__closed_loop_{sim_idx:03d}",
                        "session": sessions,
                        "trial_index": trial_index,
                        "response": choices,
                        "performance": performance,
                    }
                )
            )

    simulated_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    prepared = prepare_corrected_behavior_autocorrelograms(
        simulated_df,
        subject_col="subject",
        session_col="session",
        choice_col="response",
        outcome_col="performance",
        trial_index_col="trial_index",
        max_lag=max_lag,
        min_cross_pairs=MIN_CROSS_PAIRS,
        max_cross_pairs=MAX_CROSS_PAIRS,
        seed=seed,
    )
    prepared["simulated_df"] = simulated_df
    return prepared


def plot_overlay(data_ac: pd.DataFrame, glm_ac: pd.DataFrame, glmhmm_ac: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=fig_size(1, 2), layout="constrained")
    colors = {
        "data": "#1f77b4",
        "glm": "#333333",
        "glmhmm": "#c23b22",
    }
    for ax, signal in zip(axes, ("Outcome", "Repetition"), strict=True):
        data_sub = data_ac[data_ac["signal"] == signal].sort_values("lag")
        ax.errorbar(
            data_sub["lag"],
            data_sub["autocorr"],
            yerr=data_sub.get("autocorr_sem"),
            fmt="o",
            ms=3.5,
            capsize=2,
            color=colors["data"],
            ecolor=colors["data"],
            elinewidth=0.9,
            label="Data",
            zorder=4,
        )
        for label, model_ac, color in (
            (f"GLM {GLM_MODEL_ID}", glm_ac, colors["glm"]),
            (f"GLM-HMM {GLMHMM_MODEL_ID}", glmhmm_ac, colors["glmhmm"]),
        ):
            sub = model_ac[model_ac["signal"] == signal].sort_values("lag")
            if sub.empty:
                continue
            ax.plot(sub["lag"], sub["autocorr"], lw=1.8, color=color, label=label, zorder=3)
        ax.axhline(0.0, color="0.55", lw=0.8, ls="--", alpha=0.7)
        ax.set_title("Choice outcomes" if signal == "Outcome" else "Repeated responses")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Corrected autocorrelation")
        ax.set_ylim(-0.075, 0.20)
        ax.legend(frameon=False, fontsize=7)
    return fig


def main():
    configure_paths(config_path=PROJECT_ROOT / "config.toml")
    paths = get_runtime_paths()
    plt.style.use(PROJECT_ROOT / "styles" / "paper.mplstyle")

    adapter = get_adapter(TASK)
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    df_all = adapter.filter_condition_df(df_all, "all")

    glm_out = paths.RESULTS / "fits" / TASK / "glm" / GLM_MODEL_ID
    glmhmm_out = paths.RESULTS / "fits" / TASK / "glmhmm" / GLMHMM_MODEL_ID
    glm_cfg = load_config(glm_out / "config.json")
    glmhmm_cfg = load_config(glmhmm_out / "config.json")
    glm_K = 1
    glmhmm_K = int(glmhmm_cfg.get("K_list", [2])[0])

    glm_subjects = glm_cfg.get("subjects") or array_subjects(glm_out, "glm_arrays.npz")
    glmhmm_subjects = glmhmm_cfg.get("subjects") or array_subjects(
        glmhmm_out,
        "glmhmm_arrays.npz",
        k=glmhmm_K,
    )
    subjects = sorted(set(map(str, glm_subjects)).intersection(map(str, glmhmm_subjects)))
    if not subjects:
        raise ValueError("No overlapping subjects between the selected GLM and GLM-HMM fits.")

    df_model = df_all.filter(pl.col("subject").cast(pl.Utf8).is_in(subjects))
    trial_col = adapter.behavioral_cols["trial"]
    session_col = adapter.behavioral_cols["session"]
    data_autocorr = prepare_corrected_behavior_autocorrelograms(
        df_model,
        subject_col="subject",
        session_col=session_col,
        choice_col=adapter.behavioral_cols["response"],
        outcome_col=adapter.behavioral_cols["performance"],
        trial_index_col=trial_col,
        max_lag=MAX_LAG,
        min_cross_pairs=MIN_CROSS_PAIRS,
        max_cross_pairs=MAX_CROSS_PAIRS,
        seed=0,
    )

    glm_arrays, _ = load_fit_arrays(
        out_dir=glm_out,
        arrays_suffix="glm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=subjects,
        emission_cols=list(glm_cfg.get("emission_cols", [])),
    )
    glmhmm_arrays, _ = load_fit_arrays(
        out_dir=glmhmm_out,
        arrays_suffix="glmhmm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=subjects,
        emission_cols=list(glmhmm_cfg.get("emission_cols", [])),
        k=glmhmm_K,
    )

    glm_autocorr = prepare_closed_loop_model_autocorrelograms(
        df_model,
        glm_arrays,
        adapter=adapter,
        n_simulations=N_SIMULATIONS,
        max_lag=MAX_LAG,
        seed=SEED,
    )
    glmhmm_autocorr = prepare_closed_loop_model_autocorrelograms(
        df_model,
        glmhmm_arrays,
        adapter=adapter,
        n_simulations=N_SIMULATIONS,
        max_lag=MAX_LAG,
        seed=SEED + 100,
    )

    out_dir = paths.RESULTS / "plots" / "autocorrelograms_closed_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_autocorr["autocorr"].to_csv(out_dir / f"{TASK}_data_autocorrelogram.csv", index=False)
    glm_autocorr["autocorr"].to_csv(out_dir / f"{TASK}_glm_{GLM_MODEL_ID}_closed_loop_autocorrelogram.csv", index=False)
    glmhmm_autocorr["autocorr"].to_csv(out_dir / f"{TASK}_glmhmm_{GLMHMM_MODEL_ID}_closed_loop_autocorrelogram.csv", index=False)

    fig = plot_overlay(
        data_autocorr["autocorr"],
        glm_autocorr["autocorr"],
        glmhmm_autocorr["autocorr"],
    )
    png_path = out_dir / f"{TASK}_closed_loop_autocorrelograms.png"
    pdf_path = out_dir / f"{TASK}_closed_loop_autocorrelograms.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Subjects: {len(subjects)}; simulations per model: {N_SIMULATIONS}")


if __name__ == "__main__":
    main()
