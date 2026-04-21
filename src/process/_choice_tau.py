from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import polars as pl

from glmhmmt.runtime import get_results_dir


_TASK_TAU_TABLE_ENVS: dict[str, str] = {
    "2AFC": "GLMHMMT_2AFC_CHOICE_TAU_TABLE",
    "2AFC_delay": "GLMHMMT_2AFC_DELAY_CHOICE_TAU_TABLE",
    "MCDR": "GLMHMMT_MCDR_CHOICE_TAU_TABLE",
    "mcdr": "GLMHMMT_MCDR_CHOICE_TAU_TABLE",
}
_LOG_2 = float(np.log(2.0))
_LOG_HALF = float(np.log(0.5))


def tau_decay_to_half_life(tau_decay: float) -> float:
    tau_value = float(tau_decay)
    if not np.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError(f"tau_decay must be positive and finite; got {tau_decay!r}.")
    return float(tau_value * _LOG_2)


def compute_choice_ewma(
    values: np.ndarray,
    *,
    half_life: float,
) -> np.ndarray:
    """Return the adjust=False EWMA used for per-subject action-history traces."""
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_np.size == 0:
        return values_np.copy()

    half_life_value = float(half_life)
    if not np.isfinite(half_life_value) or half_life_value <= 0.0:
        raise ValueError(f"half_life must be positive and finite; got {half_life!r}.")

    alpha = np.float32(1.0 - np.exp(_LOG_HALF / half_life_value))
    decay = np.float32(1.0 - alpha)
    trace = np.empty_like(values_np, dtype=np.float32)
    trace[0] = values_np[0]
    for trial_idx in range(1, values_np.size):
        trace[trial_idx] = (decay * trace[trial_idx - 1]) + (alpha * values_np[trial_idx])
    return trace


def _choice_tau_table_path(task_key: str, fit_model_id: str) -> Path:
    env_name = _TASK_TAU_TABLE_ENVS.get(str(task_key))
    env_value = os.environ.get(env_name) if env_name else None
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (
        get_results_dir()
        / "fits"
        / str(task_key)
        / "glm"
        / str(fit_model_id)
        / "choice_lag_tau.parquet"
    ).resolve()


def _table_signature(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=None)
def _load_tau_table_cached(
    path_str: str,
    signature: tuple[int, int] | None,
) -> dict[str, dict[str, float]]:
    if signature is None:
        return {}

    path = Path(path_str)
    if path.suffix.lower() == ".csv":
        table = pl.read_csv(path)
    else:
        table = pl.read_parquet(path)

    required = {"subject"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in choice-tau table {path}: {sorted(missing)}")
    if "tau_ewma_half_life" not in table.columns and "tau_decay" not in table.columns:
        raise ValueError(
            f"Choice-tau table {path} must include at least one of "
            f"['tau_ewma_half_life', 'tau_decay']."
        )

    tau_map: dict[str, dict[str, float]] = {}
    select_cols = ["subject"]
    if "tau_decay" in table.columns:
        select_cols.append("tau_decay")
    if "tau_ewma_half_life" in table.columns:
        select_cols.append("tau_ewma_half_life")
    for row in table.select(select_cols).iter_rows(named=True):
        subject = str(row["subject"])
        tau_decay_raw = row.get("tau_decay")
        tau_half_life_raw = row.get("tau_ewma_half_life")
        tau_decay = (
            float(tau_decay_raw)
            if tau_decay_raw is not None and np.isfinite(tau_decay_raw) and float(tau_decay_raw) > 0.0
            else np.nan
        )
        tau_half_life = (
            float(tau_half_life_raw)
            if tau_half_life_raw is not None and np.isfinite(tau_half_life_raw) and float(tau_half_life_raw) > 0.0
            else np.nan
        )
        if not np.isfinite(tau_half_life) and np.isfinite(tau_decay):
            tau_half_life = tau_decay_to_half_life(tau_decay)
        if not np.isfinite(tau_decay) and np.isfinite(tau_half_life):
            tau_decay = float(tau_half_life / _LOG_2)
        if np.isfinite(tau_decay) and tau_decay > 0.0 and np.isfinite(tau_half_life) and tau_half_life > 0.0:
            tau_map[subject] = {
                "tau_decay": tau_decay,
                "tau_ewma_half_life": tau_half_life,
            }
    return tau_map


def _subject_tau_record(
    *,
    task_key: str,
    fit_model_id: str,
    subject: str | None,
) -> dict[str, float] | None:
    if subject is None:
        return None
    path = _choice_tau_table_path(task_key, fit_model_id)
    tau_map = _load_tau_table_cached(str(path), _table_signature(path))
    return tau_map.get(str(subject))


def load_subject_choice_tau(
    *,
    task_key: str,
    fit_model_id: str,
    subject: str | None,
) -> float | None:
    record = _subject_tau_record(task_key=task_key, fit_model_id=fit_model_id, subject=subject)
    return None if record is None else float(record["tau_decay"])


def load_subject_choice_half_life(
    *,
    task_key: str,
    fit_model_id: str,
    subject: str | None,
) -> float | None:
    record = _subject_tau_record(task_key=task_key, fit_model_id=fit_model_id, subject=subject)
    return None if record is None else float(record["tau_ewma_half_life"])


def clear_choice_tau_cache() -> None:
    _load_tau_table_cached.cache_clear()
