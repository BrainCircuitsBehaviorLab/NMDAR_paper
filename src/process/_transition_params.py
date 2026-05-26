"""Helpers for transition regressors derived from fitted GLM-HMM-T weights."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from glmhmmt.runtime import get_results_dir


def _fit_signature(fit_task: str, fit_model_id: str) -> tuple[tuple[str, int, int], ...]:
    fit_dir = get_results_dir() / "fits" / fit_task / "glmhmmt" / fit_model_id
    if not fit_dir.exists():
        raise FileNotFoundError(f"Missing glmhmmt fit directory: {fit_dir}")
    paths = [fit_dir / "config.json", *sorted(fit_dir.glob("*_glmhmmt_arrays.npz"))]
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            signature.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


def mean_transition_feature_weights(
    *,
    fit_task: str,
    fit_model_id: str,
    source_features: Sequence[str],
    from_state_idx: int = 0,
    to_state_idx: int = 0,
) -> dict[str, float]:
    return _mean_transition_feature_weights_cached(
        str(fit_task),
        str(fit_model_id),
        tuple(str(col) for col in source_features),
        int(from_state_idx),
        int(to_state_idx),
        _fit_signature(str(fit_task), str(fit_model_id)),
    )


@lru_cache(maxsize=None)
def _mean_transition_feature_weights_cached(
    fit_task: str,
    fit_model_id: str,
    source_features: tuple[str, ...],
    from_state_idx: int,
    to_state_idx: int,
    _signature: tuple[tuple[str, int, int], ...],
) -> dict[str, float]:
    del _signature
    fit_dir = get_results_dir() / "fits" / fit_task / "glmhmmt" / fit_model_id
    collected: dict[str, list[float]] = {feature: [] for feature in source_features}
    for arr_path in sorted(fit_dir.glob("*_glmhmmt_arrays.npz")):
        with np.load(arr_path, allow_pickle=True) as arr:
            if "U_cols" not in arr or "transition_weights" not in arr:
                continue
            u_cols = [str(col) for col in arr["U_cols"].tolist()]
            weights = np.asarray(arr["transition_weights"], dtype=float)
        if weights.ndim != 3:
            continue
        if weights.shape[0] <= from_state_idx or weights.shape[1] <= to_state_idx:
            continue
        col_to_idx = {col: idx for idx, col in enumerate(u_cols)}
        if any(feature not in col_to_idx for feature in source_features):
            continue
        for feature in source_features:
            collected[feature].append(
                float(weights[from_state_idx, to_state_idx, col_to_idx[feature]])
            )

    if not any(collected.values()):
        raise ValueError(
            f"No transition weights found for {fit_task}/glmhmmt/{fit_model_id} "
            f"with features {list(source_features)}."
        )

    return {
        feature: float(np.nanmean(values)) if values else 0.0
        for feature, values in collected.items()
    }


def transition_weighted_sum(
    frame,
    *,
    fit_task: str,
    fit_model_id: str,
    source_features: Sequence[str],
    fallback=None,
    dtype=np.float32,
) -> np.ndarray:
    try:
        weights = mean_transition_feature_weights(
            fit_task=fit_task,
            fit_model_id=fit_model_id,
            source_features=tuple(source_features),
        )
    except (FileNotFoundError, ValueError):
        if fallback is not None:
            return np.asarray(fallback, dtype=dtype)
        return np.zeros(len(frame), dtype=dtype)

    total = np.zeros(len(frame), dtype=np.float64)
    columns = set(frame.columns)
    for feature in source_features:
        if feature not in columns:
            continue
        total += np.asarray(frame[feature], dtype=np.float64) * float(weights.get(feature, 0.0))
    return total.astype(dtype)
