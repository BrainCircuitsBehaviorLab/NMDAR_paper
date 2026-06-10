"""Small helpers for task-owned design matrices."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def dedupe(items: Iterable[str]) -> list[str]:
    """Return items in first-seen order without duplicates."""
    return list(dict.fromkeys(str(item) for item in items))


def numeric_suffix_sort_key(name: str, prefix: str) -> tuple[int, str]:
    """Sort columns named like ``prefix01`` or ``prefix_1`` by numeric suffix."""
    suffix = str(name).removeprefix(prefix)
    return (int(suffix), str(name)) if suffix.isdigit() else (10**9, str(name))


def numeric_prefixed(
    columns: Sequence[str],
    prefix: str,
    *,
    max_count: int | None = None,
) -> list[str]:
    """Return columns with a numeric suffix after ``prefix``."""
    return sorted(
        [
            str(col)
            for col in columns
            if str(col).startswith(prefix)
            and str(col).removeprefix(prefix).isdigit()
            and (max_count is None or int(str(col).removeprefix(prefix)) <= int(max_count))
        ],
        key=lambda col: numeric_suffix_sort_key(col, prefix),
    )


def lag_names(prefix: str, n_lags: int) -> list[str]:
    """Return zero-padded lag column names."""
    return [f"{prefix}{idx:02d}" for idx in range(1, int(n_lags) + 1)]


def lag_level_prefixed(columns: Sequence[str], prefix: str) -> list[str]:
    """Return columns named like ``prefix01_8`` sorted by lag then level."""
    def sort_key(name: str) -> tuple[int, int, str]:
        suffix = str(name).removeprefix(prefix)
        lag, sep, level = suffix.partition("_")
        if sep and lag.isdigit() and level.isdigit():
            return (int(lag), int(level), str(name))
        return (10**9, 10**9, str(name))

    return sorted(
        [
            str(col)
            for col in columns
            if str(col).startswith(prefix) and sort_key(str(col))[0] < 10**9
        ],
        key=sort_key,
    )


def constant_frame(values: dict[str, float], index: pd.Index) -> pd.DataFrame:
    """Return float32 columns filled with one value per column."""
    return pd.DataFrame(
        {
            col: np.full(len(index), value, dtype=np.float32)
            for col, value in values.items()
        },
        index=index,
    )


def session_one_hot_frame(session_idx: int, max_sessions: int, index: pd.Index) -> pd.DataFrame:
    """Return bias_0..bias_n one-hot columns for one session."""
    return pd.DataFrame(
        {
            f"bias_{idx}": np.full(len(index), idx == session_idx, dtype=np.float32)
            for idx in range(int(max_sessions))
        },
        index=index,
    )


def shifted_lag_frame(series: pd.Series, columns: Sequence[str], index: pd.Index) -> pd.DataFrame:
    """Return columns from successive positive lags of one series."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(
        {
            col: values.shift(lag_idx).fillna(0.0).astype(np.float32)
            for lag_idx, col in enumerate(columns, start=1)
        },
        index=index,
    )


def level_indicator_frame(
    series: pd.Series,
    levels: Sequence,
    *,
    prefix: str,
    index: pd.Index,
    label=str,
) -> pd.DataFrame:
    """Return one float32 indicator column per level."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(
        {
            f"{prefix}{label(level)}": (values == level).astype(np.float32)
            for level in levels
        },
        index=index,
    )


def lagged_level_indicator_frame(
    series: pd.Series,
    levels: Sequence,
    *,
    prefix: str,
    n_lags: int,
    index: pd.Index,
    label=str,
) -> pd.DataFrame:
    """Return one indicator per positive lag and level."""
    values = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(
        {
            f"{prefix}{lag_idx:02d}_{label(level)}": (
                values.shift(lag_idx).fillna(0.0) == level
            ).astype(np.float32)
            for lag_idx in range(1, int(n_lags) + 1)
            for level in levels
        },
        index=index,
    )


def choice_outcome_lag_frames(
    choice_signed: pd.Series,
    hit: pd.Series,
    *,
    n_lags: int,
    index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return previous-choice lag columns split by previous correct/incorrect."""
    choice = pd.to_numeric(choice_signed, errors="coerce")
    reward = pd.to_numeric(hit, errors="coerce")
    corr = pd.DataFrame(
        {
            f"choice_lag_corr_{lag_idx:02d}": (
                reward.shift(lag_idx).fillna(0.0).astype(np.float32)
                * choice.shift(lag_idx).fillna(0.0).astype(np.float32)
            )
            for lag_idx in range(1, int(n_lags) + 1)
        },
        index=index,
    )
    inc = pd.DataFrame(
        {
            f"choice_lag_inc_{lag_idx:02d}": (
                (1.0 - reward.shift(lag_idx).fillna(0.0).astype(np.float32))
                * choice.shift(lag_idx).fillna(0.0).astype(np.float32)
            )
            for lag_idx in range(1, int(n_lags) + 1)
        },
        index=index,
    )
    return corr, inc
