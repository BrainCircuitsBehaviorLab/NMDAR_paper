"""Task adapter for the Tiffany 2AFC delay task."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd

from ._choice_tau import compute_choice_ewma, load_subject_choice_half_life
from .design import (
    choice_outcome_lag_frames,
    constant_frame,
    lag_names,
    lagged_level_indicator_frame,
    level_indicator_frame,
    numeric_prefixed,
    session_one_hot_frame,
    shifted_lag_frame,
)
from ._transition_params import transition_weighted_sum
from glmhmmt.cli.alexis_functions import get_action_trace
from glmhmmt.runtime import get_data_dir
from glmhmmt.tasks import TaskAdapter, _register
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    mean_feature_weights_from_fit,
    resolved_source_features,
    weighted_sum_regressor,
)

try:
    from glmhmmt.tasks import build_selector_groups as _build_selector_groups
except ImportError:
    def _build_selector_groups(available_cols: list[str], registry: list[dict]) -> list[dict]:
        available = set(available_cols)
        registered: set[str] = set()
        result: list[dict] = []
        for group in registry:
            filtered = {k: v for k, v in group["members"].items() if v in available}
            if filtered:
                result.append({**group, "members": filtered})
                registered.update(filtered.values())
        for col in available_cols:
            if col not in registered:
                result.append({"key": col, "label": col, "members": {"N": col}})
        return result

from src.process.common import label_states_by_regressor

_DELAY_HOT_COL_PREFIX = "delay_"
_NUM_LEGACY_CHOICE_LAGS = 15
_NUM_MEDIUM_CHOICE_LAGS = 50
_NUM_REWARD_LAGS = 15
_NUM_DIFFICULTY_LAGS = 20
_NUM_DAY_REWARD_LAGS = 5
_FILTERED_REGRESSOR_TAU = 4.0
_RAW_PARAM_MODEL_ID = "one hot2"
_TRANSITION_PARAM_MODEL_ID = "one hot"
EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "stim",
    "stim_param",
    "delay",
    "delay_param",
    "stim_x_delay",
    "stim_x_delay_hot",
    "stim_x_delay_param",
    "at_choice",
    "choice_lag_param",
    "choice_lag_param_2",
    "choice_lag_param_correct",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "wsls",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
    "prev_abs_delay",
    "after_correct",
    "repeat",
    "repeat_choice_side",
]
TRANSITION_COLS: list[str] = [
    "trial_index",
    "filtered_choice",
    "filtered_stim_side",
    "filtered_reward",
    "filtered_difficulty",
    "filtered_bad_stim",
    "filtered_bad_choice",
    "filtered_bad_reward",
    "cumulative_reward",
    "delay",
    "prev_day_total_reward",
    "prev_day_total_reward_x_cumulative_reward",
    "reward_lag_param",
    "difficulty_hot_param",
]
_LEGACY_TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_delay",
    "prev_reward",
    "prev_difficulty",
    "prev_difficulty_param",
]

_BIAS_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="bias_param",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=("bias_",),
)
_STIM_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_param",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_features=("stim",),
)
_DELAY_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="delay_param",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_DELAY_HOT_COL_PREFIX,),
)
_STIM_X_DELAY_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_x_delay_param",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=("stim_x_delay_hot_",),
)
_CHOICE_LAG_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="choice_lag_param",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)),
)
_CHOICE_LAG_PARAM_CORRECT_SPEC = FittedWeightRegressorSpec(
    target_name="choice_lag_param_correct",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(lag_names("choice_lag_corr_", _NUM_LEGACY_CHOICE_LAGS)),
)
_CHOICE_LAG_PARAM_2_SPEC = FittedWeightRegressorSpec(
    target_name="choice_lag_param_2",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)[1:]),
)

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim": r"$\mathrm{Stimulus}$",
    "stim_param": r"$\mathrm{Stimulus}_{\mathrm{param}}$",
    "delay": r"$\mathrm{Delay}$",
    "delay_param": r"$\mathrm{Delay}_{\mathrm{param}}$",
    "stim_x_delay": r"$\mathrm{Stimulus}\times\mathrm{Delay}$",
    "stim_x_delay_param": r"$(\mathrm{Stimulus}\times\mathrm{Delay})_{\mathrm{param}}$",
    "bias": r"$\mid\mathrm{bias}\mid$",
    "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
    "at_choice": r"$\mathrm{A}_t^{\mathrm{choice}}$",
    "choice_lag_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    "choice_lag_param_2": r"$\mathrm{A}_{t,\geq 2}^{\mathrm{choice,param}}$",
    "choice_lag_param_correct": r"$\mathrm{A}_t^{\mathrm{choice,correct,param}}$",
    "at_error": r"$\mathrm{A}_t^{\mathrm{error}}$",
    "at_correct": r"$\mathrm{A}_t^{\mathrm{correct}}$",
    "reward_trace": r"$\mathrm{Reward}_{\mathrm{trace}}$",
    "prev_choice": r"$\mathrm{PrevChoice}$",
    "prev_reward": r"$\mathrm{PrevReward}$",
    "prev_abs_stim": r"$|\mathrm{PrevStim}|$",
    "prev_abs_delay": r"$|\mathrm{PrevDelay}|$",
    "filtered_bad_stim": r"$\mathrm{FilteredBadStimSide}$",
    "filtered_bad_choice": r"$\mathrm{FilteredBadChoice}$",
    "filtered_bad_reward": r"$\mathrm{FilteredBadReward}$",
    "cumulative_reward": r"$\mathrm{CumReward}$",
    "wsls": r"$\mathrm{WSLS}$",
    "after_correct": r"$\mathrm{AfterCorrect}$",
    "repeat": r"$\mathrm{Repeat}$",
    "repeat_choice_side": r"$\mathrm{RepeatSide}$",
    "WM": r"$\mathrm{WM}$",
    "RL": r"$\mathrm{RL}$",
}

_EMISSION_GROUPS: list[dict] = [
    {"key": "bias", "label": "bias", "members": {"N": "bias"}},
    {"key": "bias_param", "label": "bias param", "members": {"N": "bias_param"}},
    {"key": "stim", "label": "stim", "members": {"N": "stim"}},
    {"key": "stim_param", "label": "stim param", "members": {"N": "stim_param"}},
    {"key": "delay", "label": "delay", "members": {"N": "delay"}},
    {"key": "delay_param", "label": "delay param", "members": {"N": "delay_param"}},
    {"key": "stim_x_delay", "label": "stim×delay", "members": {"N": "stim_x_delay"}},
    {"key": "stim_x_delay_hot", "label": "stim×delay one-hot", "members": {"N": "stim_x_delay_hot"}},
    {"key": "stim_x_delay_param", "label": "stim×delay param", "members": {"N": "stim_x_delay_param"}},
    {"key": "at_choice", "label": "action (choice)", "members": {"N": "at_choice"}},
    {"key": "choice_lag_param", "label": "choice lag param", "members": {"N": "choice_lag_param"}},
    {"key": "choice_lag_param_2", "label": "choice lag param 2+", "members": {"N": "choice_lag_param_2"}},
    {"key": "choice_lag_param_correct", "label": "choice lag correct param", "members": {"N": "choice_lag_param_correct"}},
    {"key": "at_error", "label": "action (error)", "members": {"N": "at_error"}},
    {"key": "at_correct", "label": "action (correct)", "members": {"N": "at_correct"}},
    {"key": "reward_trace", "label": "reward trace", "members": {"N": "reward_trace"}},
    {"key": "prev_choice", "label": "prev choice", "members": {"N": "prev_choice"}},
    {"key": "wsls", "label": "WSLS", "members": {"N": "wsls"}},
    {"key": "prev_reward", "label": "prev reward", "members": {"N": "prev_reward"}},
    {"key": "cumulative_reward", "label": "cumulative reward", "members": {"N": "cumulative_reward"}},
    {"key": "prev_abs_stim", "label": "prev abs stim", "members": {"N": "prev_abs_stim"}},
    {"key": "prev_abs_delay", "label": "prev abs delay", "members": {"N": "prev_abs_delay"}},
    {"key": "after_correct", "label": "after correct", "members": {"N": "after_correct"}},
    {"key": "repeat", "label": "repeat", "members": {"N": "repeat"}},
    {"key": "repeat_choice_side", "label": "repeat side", "members": {"N": "repeat_choice_side"}},
]


def _safe_weighted_sum_regressor(
    part,
    spec: FittedWeightRegressorSpec,
) -> np.ndarray | None:
    try:
        source_features = resolved_source_features(spec)
        missing_cols = [col for col in source_features if col not in part.columns]
        if missing_cols:
            part = part.copy()
            for col in missing_cols:
                part[col] = np.float32(0.0)
        return weighted_sum_regressor(part, spec, dtype=np.float32)
    except (FileNotFoundError, ValueError):
        return None


def _delay_level_token(delay_value: float) -> str:
    delay_value = float(delay_value)
    if delay_value.is_integer():
        return str(int(delay_value))
    return format(delay_value, "g").replace("-", "m").replace(".", "p")


def _parse_delay_level_token(token: str) -> float | None:
    try:
        return float(token.replace("m", "-").replace("p", "."))
    except ValueError:
        return None


def _format_delay_level_label(delay_value: float) -> str:
    delay_value = float(delay_value)
    if delay_value.is_integer():
        return str(int(delay_value))
    return format(delay_value, "g")


def _delay_param_weight_map() -> dict[float, float]:
    try:
        feature_weights = mean_feature_weights_from_fit(_DELAY_PARAM_SPEC)
    except (FileNotFoundError, ValueError):
        return {}
    resolved: dict[float, float] = {}
    for feat, weight in feature_weights.items():
        if not feat.startswith(_DELAY_HOT_COL_PREFIX):
            continue
        parsed = _parse_delay_level_token(feat.removeprefix(_DELAY_HOT_COL_PREFIX))
        if parsed is None:
            continue
        resolved[parsed] = weight
    return resolved


def _stim_param_weight_map() -> dict[int, float]:
    """Legacy compatibility shim for task-owned plotting helpers."""
    return {}


def _delay_hot_sort_key(name: str) -> tuple[float, str]:
    suffix = name.removeprefix(_DELAY_HOT_COL_PREFIX)
    parsed = _parse_delay_level_token(suffix)
    return (parsed, name) if parsed is not None else (float("inf"), name)


def _ewma_time_series(values: Sequence[float], period: float = _FILTERED_REGRESSOR_TAU) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_np.size == 0:
        return values_np.copy()
    ewma = pd.DataFrame(data=values_np).ewm(span=float(period)).mean()
    return ewma.iloc[:, 0].to_numpy(dtype=np.float32)


def _difficulty_hot_sort_key(name: str) -> tuple[float, str]:
    suffix = name.removeprefix("difficulty_hot_")
    parsed = _parse_delay_level_token(suffix)
    return (parsed, name) if parsed is not None else (float("inf"), name)


def _prev_difficulty_hot_sort_key(name: str) -> tuple[float, str]:
    suffix = name.removeprefix("prev_difficulty_hot_")
    parsed = _parse_delay_level_token(suffix)
    return (parsed, name) if parsed is not None else (float("inf"), name)


def _prev_difficulty_lag_hot_sort_key(name: str) -> tuple[int, float, str]:
    suffix = name.removeprefix("prev_difficulty_lag_hot_")
    lag, sep, level_token = suffix.partition("_")
    parsed = _parse_delay_level_token(level_token) if sep and lag.isdigit() else None
    return (int(lag), parsed, name) if parsed is not None else (10**9, float("inf"), name)


def _stim_x_delay_hot_sort_key(name: str) -> tuple[float, str]:
    suffix = name.removeprefix("stim_x_delay_hot_")
    parsed = _parse_delay_level_token(suffix)
    return (parsed, name) if parsed is not None else (float("inf"), name)


def _delay_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_DELAY_HOT_COL_PREFIX)
            and _parse_delay_level_token(col.removeprefix(_DELAY_HOT_COL_PREFIX)) is not None
        ],
        key=_delay_hot_sort_key,
    )


def _is_bias_hot_col(col: str) -> bool:
    return col.startswith("bias_") and col.removeprefix("bias_").isdigit()


def _drop_unavailable_bias_hot_cols(cols: list[str], available_cols: set[str]) -> list[str]:
    return [col for col in cols if col in available_cols or not _is_bias_hot_col(col)]


def _zscore_sequence(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std <= 0:
        return [0.0 for _ in arr]
    return ((arr - mean) / std).astype(np.float32).tolist()


def _scale_by_max_sequence(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    max_value = float(np.nanmax(arr))
    if not np.isfinite(max_value) or max_value <= 0:
        return [0.0 for _ in arr]
    return (arr / max_value).astype(np.float32).tolist()


def _session_trial_index(n_trials: int) -> np.ndarray:
    """Return within-session trial progress scaled from 0 to 1."""
    n = int(n_trials)
    if n <= 1:
        return np.zeros(max(n, 0), dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def _difficulty_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith("difficulty_hot_")
            and _parse_delay_level_token(col.removeprefix("difficulty_hot_")) is not None
        ],
        key=_difficulty_hot_sort_key,
    )


def _prev_difficulty_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith("prev_difficulty_hot_")
            and _parse_delay_level_token(col.removeprefix("prev_difficulty_hot_")) is not None
        ],
        key=_prev_difficulty_hot_sort_key,
    )


def _prev_difficulty_lag_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith("prev_difficulty_lag_hot_")
            and _prev_difficulty_lag_hot_sort_key(col)[0] < 10**9
        ],
        key=_prev_difficulty_lag_hot_sort_key,
    )


def _stim_x_delay_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith("stim_x_delay_hot_")
            and _parse_delay_level_token(col.removeprefix("stim_x_delay_hot_")) is not None
        ],
        key=_stim_x_delay_hot_sort_key,
    )   


def _infer_delay_hot_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = _delay_hot_cols(columns)
    if existing:
        return existing
    delay_col = "delay_raw" if "delay_raw" in columns else "delays" if "delays" in columns else None
    if delay_col is None:
        return []
    delay_series = df[delay_col].drop_nulls() if isinstance(df, pl.DataFrame) else df[delay_col].dropna()
    delay_levels = sorted({float(v) for v in delay_series.to_list()})
    return [f"{_DELAY_HOT_COL_PREFIX}{_delay_level_token(delay_value)}" for delay_value in delay_levels]


@lru_cache(maxsize=1)
def _all_delay_levels() -> tuple[float, ...]:
    dataset_path = get_data_dir() / "tiffany.parquet"
    df = pl.read_parquet(dataset_path)
    if "drug" in df.columns:
        df = df.filter(pl.col("drug") == "Rest")
    if "delays" not in df.columns:
        return tuple()
    delay_vals = sorted({float(v) for v in df["delays"].drop_nulls().to_list()})
    return tuple(delay_vals)


def _build_emission_groups(available_cols: list[str]) -> list[dict]:
    available = set(available_cols)
    result: list[dict] = []
    registered: set[str] = set()

    def add_scalar(group: dict) -> None:
        filtered = {k: v for k, v in group["members"].items() if v in available}
        if filtered:
            result.append({**group, "members": filtered})
            registered.update(filtered.values())

    def add_hidden_family(*, key: str, label: str, family_cols: list[str]) -> None:
        if not family_cols:
            return
        result.append(
            {
                "key": key,
                "label": label,
                "members": {},
                "toggle_members": list(family_cols),
                "hide_members": True,
            }
        )
        registered.update(family_cols)

    delay_hot_cols = _delay_hot_cols(available_cols)
    bias_hot_cols = numeric_prefixed(available_cols, "bias_")
    choice_lag_cols = numeric_prefixed(available_cols, "choice_lag_")
    choice_lag_corr_cols = numeric_prefixed(available_cols, "choice_lag_corr_")
    choice_lag_inc_cols = numeric_prefixed(available_cols, "choice_lag_inc_")
    choice_lag_15_cols = numeric_prefixed(available_cols, "choice_lag_", max_count=_NUM_LEGACY_CHOICE_LAGS)
    choice_lag_50_cols = numeric_prefixed(available_cols, "choice_lag_", max_count=_NUM_MEDIUM_CHOICE_LAGS)
    stim_x_delay_hot_cols = _stim_x_delay_hot_cols(available_cols)

    for group in _EMISSION_GROUPS:
        key = group["key"]
        if key == "bias":
            add_scalar(group)
            add_hidden_family(key="bias_hot", label="bias_hot", family_cols=bias_hot_cols)
            continue
        if key == "delay_param":
            add_scalar(group)
            add_hidden_family(key="delay_hot", label="delay_hot", family_cols=delay_hot_cols)
            continue
        if key == "at_choice":
            add_scalar(group)
            add_hidden_family(
                key="choice_lag_15_lags",
                label="choice lag (15)",
                family_cols=choice_lag_15_cols,
            )
            add_hidden_family(
                key="choice_lag_50_lags",
                label="choice lag (50)",
                family_cols=choice_lag_50_cols,
            )
            add_hidden_family(
                key="choice_lag_100_lags",
                label="choice lag (100)",
                family_cols=choice_lag_cols,
            )
            add_hidden_family(
                key="choice_lag_correct",
                label="choice lag correct",
                family_cols=choice_lag_corr_cols,
            )
            add_hidden_family(
                key="choice_lag_inc",
                label="choice lag incorrect",
                family_cols=choice_lag_inc_cols,
            )
            continue
        if key == "stim_x_delay":
            add_scalar(group)
            add_hidden_family(
                key="stim_x_delay_one_hot",
                label="stim×delay one-hot",
                family_cols=stim_x_delay_hot_cols,
            )
            continue
        add_scalar(group)

    remaining = [col for col in available_cols if col not in registered]
    if remaining:
        result.extend(_build_selector_groups(remaining, []))
    return result


def _build_transition_groups(available_cols: list[str]) -> list[dict]:
    available = set(available_cols)
    result: list[dict] = []
    registered: set[str] = set()

    def add_scalar(col: str, label: str | None = None) -> None:
        if col in available:
            result.append({"key": col, "label": label or col, "members": {"N": col}})
            registered.add(col)

    for col in TRANSITION_COLS:
        add_scalar(col)

    reward_lag_cols = numeric_prefixed(available_cols, "reward_lag_")
    if reward_lag_cols:
        result.append(
            {
                "key": "reward_lag",
                "label": "reward lag",
                "members": {},
                "toggle_members": list(reward_lag_cols),
                "hide_members": True,
            }
        )
        registered.update(reward_lag_cols)

    prev_day_reward_lag_cols = numeric_prefixed(available_cols, "prev_day_total_reward_lag_")
    if prev_day_reward_lag_cols:
        result.append(
            {
                "key": "prev_day_total_reward_lag",
                "label": "prev day reward lag",
                "members": {},
                "toggle_members": list(prev_day_reward_lag_cols),
                "hide_members": True,
            }
        )
        registered.update(prev_day_reward_lag_cols)

    difficulty_hot_cols = _difficulty_hot_cols(available_cols)
    if difficulty_hot_cols:
        result.append(
            {
                "key": "difficulty_hot",
                "label": "difficulty one-hot",
                "members": {},
                "toggle_members": list(difficulty_hot_cols),
                "hide_members": True,
            }
        )
        registered.update(difficulty_hot_cols)

    prev_difficulty_hot_cols = _prev_difficulty_hot_cols(available_cols)
    if prev_difficulty_hot_cols:
        result.append(
            {
                "key": "prev_difficulty_hot",
                "label": "prev difficulty one-hot",
                "members": {},
                "toggle_members": list(prev_difficulty_hot_cols),
                "hide_members": True,
            }
        )
        registered.update(prev_difficulty_hot_cols)

    prev_difficulty_lag_cols = numeric_prefixed(available_cols, "prev_difficulty_lag_")
    if prev_difficulty_lag_cols:
        result.append(
            {
                "key": "prev_difficulty_lag",
                "label": "prev difficulty lag",
                "members": {},
                "toggle_members": list(prev_difficulty_lag_cols),
                "hide_members": True,
            }
        )
        registered.update(prev_difficulty_lag_cols)

    prev_difficulty_lag_hot_cols = _prev_difficulty_lag_hot_cols(available_cols)
    if prev_difficulty_lag_hot_cols:
        result.append(
            {
                "key": "prev_difficulty_lag_hot",
                "label": "prev difficulty lag one-hot",
                "members": {},
                "toggle_members": list(prev_difficulty_lag_hot_cols),
                "hide_members": True,
            }
        )
        registered.update(prev_difficulty_lag_hot_cols)

    remaining = [col for col in available_cols if col not in registered]
    if remaining:
        result.extend(_build_selector_groups(remaining, []))
    return result


def _max_sessions_from_df(df: pl.DataFrame | pd.DataFrame) -> int:
    if "session" not in df.columns:
        return _max_subject_sessions()
    if "subject" not in df.columns:
        session_series = df["session"]
        if isinstance(df, pl.DataFrame):
            return int(session_series.n_unique())
        return int(session_series.nunique())
    if isinstance(df, pl.DataFrame):
        return int(
            df.group_by("subject")
            .agg(pl.col("session").n_unique().alias("n_sessions"))
            .select(pl.col("n_sessions").max())
            .item()
            or 0
        )
    grouped = df.groupby("subject", sort=False)["session"].nunique()
    return int(grouped.max()) if len(grouped) else 0


def _infer_bias_hot_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = numeric_prefixed(columns, "bias_")
    if existing:
        return existing
    max_sessions = _max_sessions_from_df(df)
    return [f"bias_{idx}" for idx in range(max_sessions)]


@lru_cache(maxsize=1)
def _max_subject_sessions() -> int:
    dataset_path = get_data_dir() / "tiffany.parquet"
    df = pl.read_parquet(dataset_path)
    if "drug" in df.columns:
        df = df.filter(pl.col("drug") == "Rest")
    return int(
        df.group_by("subject")
        .agg(pl.col("session").n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


def _difficulty_hot_names(levels: Sequence[float]) -> list[str]:
    return [f"difficulty_hot_{_delay_level_token(level)}" for level in levels]


def _prev_difficulty_hot_names(levels: Sequence[float]) -> list[str]:
    return [f"prev_difficulty_hot_{_delay_level_token(level)}" for level in levels]


def _prev_difficulty_lag_hot_names(levels: Sequence[float]) -> list[str]:
    return [
        f"prev_difficulty_lag_hot_{lag_idx:02d}_{_delay_level_token(level)}"
        for lag_idx in range(1, _NUM_DIFFICULTY_LAGS + 1)
        for level in levels
    ]


def _all_difficulty_hot_names() -> list[str]:
    return _difficulty_hot_names(_all_delay_levels())


def _all_prev_difficulty_hot_names() -> list[str]:
    return _prev_difficulty_hot_names(_all_delay_levels())


def _all_prev_difficulty_lag_hot_names() -> list[str]:
    return _prev_difficulty_lag_hot_names(_all_delay_levels())



def _choice_to_binary(series: pd.Series) -> np.ndarray:
    return series.astype(np.int32).to_numpy()


def _signed_to_binary(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    unique = set(vals.dropna().unique().tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return vals.astype(np.float32)
    if unique.issubset({-1, 1, -1.0, 1.0}):
        return (vals > 0).astype(np.float32)
    return vals.astype(np.float32)


def _signed_to_model_binary(series: pd.Series) -> pd.Series:
    """Map choices to model classes with Right=0 and Left=1.

    TwoAFCDelayAdapter uses baseline_class_idx=1, so class 0 is the
    explicit binary logit and the saved weights are Right against Left.
    """
    vals = pd.to_numeric(series, errors="coerce")
    unique = set(vals.dropna().unique().tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return (vals == 0).astype(np.float32)
    if unique.issubset({-1, 1, -1.0, 1.0}):
        return (vals < 0).astype(np.float32)
    return (vals < 0).astype(np.float32)




def _signed_choice(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    unique = set(vals.dropna().unique().tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return ((2.0 * vals.astype(np.float32)) - 1.0).astype(np.float32)
    if unique.issubset({-1, 1, -1.0, 1.0}):
        return vals.astype(np.float32)
    return pd.Series(
        np.where(vals > 0, 1.0, -1.0),
        index=series.index,
        dtype=np.float32,
    )

from src.process.common import (
    attach_quantile_bin_column,
    attach_response_right_column,
    attach_signed_delay_columns,
    display_regressor_name,
    mean_glm_feature_curve as _mean_glm_feature_curve,
    mean_glm_ild_curve as _mean_glm_ild_curve,
    p_right_label,
    prepare_simple_regressor_curve,
    resolve_grouping,
    summarize_grouped_panel,
    subject_glm_feature_curves as _subject_glm_feature_curves,
    subject_glm_ild_curves as _subject_glm_ild_curves,
    to_pandas_df,
)
from src.process.plot_payloads import (
    TWO_ADC_PROFILE,
    prepare_binned_accuracy_figure as _prepare_binned_accuracy_figure,
    prepare_predictions_df as _prepare_predictions_df,
    prepare_right_by_regressor as _prepare_right_by_regressor,
    prepare_right_by_regressor_simple as _prepare_right_by_regressor_simple,
)

PRED_COL = "p_pred"
RESPONSE_MODE = "pm1_or_prob"
BASELINE = 0.5

SIGNED_DELAY_ORDER = ["-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"]
SIGNED_DELAY_LABELS = ["-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"]


def format_delay_tick(value: float) -> str:
    value = float(value)
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def delay_ticks_from_df(df: pd.DataFrame, *, delay_col: str = "delay") -> tuple[list[float], list[str]]:
    if delay_col not in df.columns:
        return [], []
    ticks = sorted(pd.to_numeric(df[delay_col], errors="coerce").dropna().astype(float).unique())
    return ticks, [format_delay_tick(value) for value in ticks]


def signed_delay_label(value: str) -> str:
    return format_delay_tick(float(value))


def signed_delay_order_and_labels(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    values = [str(value) for value in df["_signed_delay_cat"].dropna().unique()]
    preferred = [value for value in SIGNED_DELAY_ORDER if value in set(values)]
    extras = sorted(
        (value for value in values if value not in set(preferred)),
        key=lambda value: float(value),
    )
    order = preferred + extras
    return order, [signed_delay_label(value) for value in order]


def _correct_prob_expr(pl_module):
    stim = pl_module.col("stimulus").cast(pl_module.Float64)
    return (
        pl_module.when(stim.is_in([0.0, 1.0]))
        .then(pl_module.when(stim == 0.0).then(pl_module.col("pL")).otherwise(pl_module.col("pR")))
        .when(stim.is_in([-1.0, 1.0]))
        .then(pl_module.when(stim < 0.0).then(pl_module.col("pL")).otherwise(pl_module.col("pR")))
        .otherwise(None)
    )


def _correct_prob_pandas(df: pd.DataFrame) -> pd.Series:
    stim = pd.to_numeric(df["stimulus"], errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)

    binary_mask = stim.isin([0.0, 1.0])
    signed_mask = stim.isin([-1.0, 1.0])

    out.loc[binary_mask & (stim == 0.0)] = pd.to_numeric(df.loc[binary_mask & (stim == 0.0), "pL"], errors="coerce")
    out.loc[binary_mask & (stim == 1.0)] = pd.to_numeric(df.loc[binary_mask & (stim == 1.0), "pR"], errors="coerce")
    out.loc[signed_mask & (stim < 0.0)] = pd.to_numeric(df.loc[signed_mask & (stim < 0.0), "pL"], errors="coerce")
    out.loc[signed_mask & (stim > 0.0)] = pd.to_numeric(df.loc[signed_mask & (stim > 0.0), "pR"], errors="coerce")
    return out


def prepare_predictions_df(df_pred):
    """Compatibility wrapper for the shared 2ADC trial payload builder."""
    return _prepare_predictions_df(df_pred, profile=TWO_ADC_PROFILE)


def prepare_delay_accuracy_summary(
    trial_df,
    *,
    delay_col: str = "delay",
    weight_col: str | None = None,
    model_col: str = "p_model_correct",
) -> tuple[pd.DataFrame, dict]:
    df_pd = to_pandas_df(trial_df)
    if delay_col not in df_pd.columns:
        return pd.DataFrame(), {}

    delay_values = pd.to_numeric(df_pd[delay_col], errors="coerce")
    rows: list[dict[str, float]] = []
    for delay_value in sorted(delay_values.dropna().unique()):
        group = df_pd[delay_values == delay_value].copy()
        if group.empty:
            continue
        weights = (
            np.ones(len(group), dtype=float)
            if weight_col is None
            else pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        correct = pd.to_numeric(group["correct_bool"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        model_vals = pd.to_numeric(group[model_col], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
        rows.append(
            {
                "delay": float(delay_value),
                "data_acc": float(np.average(correct, weights=weights)),
                "model_acc": float(np.average(model_vals, weights=weights)),
            }
        )

    summary = pd.DataFrame(rows).sort_values("delay").reset_index(drop=True)
    xticks, tick_labels = delay_ticks_from_df(df_pd, delay_col=delay_col)
    return summary, {
        "xlabel": "Delay",
        "ylabel": "Accuracy",
        "baseline": BASELINE,
        "xticks": xticks,
        "x_tick_labels": tick_labels,
    }


def mean_glm_ild_curve(arrays_store, subjects, X_cols, *, ild_max, state_k=None):
    return _mean_glm_ild_curve(
        arrays_store,
        subjects,
        X_cols,
        ild_max=ild_max,
        state_k=state_k,
        stim_param_weight_map=_stim_param_weight_map,
        right_logit_sign=1.0,
    )


def subject_glm_ild_curves(arrays_store, subjects, X_cols, *, ild_max, state_k=None):
    return _subject_glm_ild_curves(
        arrays_store,
        subjects,
        X_cols,
        ild_max=ild_max,
        state_k=state_k,
        stim_param_weight_map=_stim_param_weight_map,
        right_logit_sign=1.0,
    )


def mean_glm_feature_curve(
    arrays_store,
    subjects,
    X_cols,
    *,
    feature_name,
    grid_min,
    grid_max,
    state_k=None,
    n_grid: int = 300,
):
    return _mean_glm_feature_curve(
        arrays_store,
        subjects,
        X_cols,
        feature_name=feature_name,
        grid_min=grid_min,
        grid_max=grid_max,
        state_k=state_k,
        n_grid=n_grid,
        right_logit_sign=1.0,
    )


def subject_glm_feature_curves(
    arrays_store,
    subjects,
    X_cols,
    *,
    feature_name,
    grid_min,
    grid_max,
    state_k=None,
    n_grid: int = 300,
):
    return _subject_glm_feature_curves(
        arrays_store,
        subjects,
        X_cols,
        feature_name=feature_name,
        grid_min=grid_min,
        grid_max=grid_max,
        state_k=state_k,
        n_grid=n_grid,
        right_logit_sign=1.0,
    )


def prepare_right_by_regressor_simple(
    trial_df,
    *,
    regressor_col: str,
    xlabel: str | None = None,
    n_bins: int = 10,
):
    return _prepare_right_by_regressor_simple(
        trial_df,
        profile=TWO_ADC_PROFILE,
        regressor_col=regressor_col,
        xlabel=xlabel,
        n_bins=n_bins,
    )


def prepare_binned_accuracy_figure(
    trial_df,
    *,
    regressor_col: str,
    x_col: str | None = None,
    xlabel: str | None = None,
    n_bins: int = 4,
) -> tuple[list[dict] | None, str | None]:
    return _prepare_binned_accuracy_figure(
        trial_df,
        profile=TWO_ADC_PROFILE,
        regressor_col=regressor_col,
        x_col=x_col,
        xlabel=xlabel,
        n_bins=n_bins,
    )


def prepare_right_by_regressor(
    trial_df,
    *,
    regressor_col: str,
    xlabel: str | None = None,
    n_bins: int = 10,
    group_col: str | None = None,
    group_order: Sequence | None = None,
):
    return _prepare_right_by_regressor(
        trial_df,
        profile=TWO_ADC_PROFILE,
        regressor_col=regressor_col,
        xlabel=xlabel,
        n_bins=n_bins,
        group_col=group_col,
        group_order=group_order,
    )



@_register(["two_afc_delay", "2afc_delay", "2AFC_delay", "2adc", "2ADC"])
class TwoAFCDelayAdapter(TaskAdapter):
    """Adapter for the Tiffany binary 2AFC task with trial delay."""

    task_key: str = "2AFC_delay"
    task_label: str = "2ADC"
    num_classes: int = 2
    baseline_class_idx: int = 1
    data_file: str = "tiffany.parquet"
    sort_col = ["session", "trial"]
    session_col: str = "session"
    prediction_col: str = PRED_COL
    response_mode: str = RESPONSE_MODE
    psychometric_x_col: str = "delay"
    psychometric_x_label: str = "Delay"
    accuracy_x_col: str = "delays"
    accuracy_x_label: str = "Delay"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS
    bias_param_spec: FittedWeightRegressorSpec = _BIAS_PARAM_SPEC
    stim_param_spec: FittedWeightRegressorSpec = _STIM_PARAM_SPEC
    delay_param_spec: FittedWeightRegressorSpec = _DELAY_PARAM_SPEC
    stim_x_delay_param_spec: FittedWeightRegressorSpec = _STIM_X_DELAY_PARAM_SPEC
    choice_lag_param_spec: FittedWeightRegressorSpec = _CHOICE_LAG_PARAM_SPEC
    choice_lag_param_2_spec: FittedWeightRegressorSpec = _CHOICE_LAG_PARAM_2_SPEC
    choice_lag_param_correct_spec: FittedWeightRegressorSpec = _CHOICE_LAG_PARAM_CORRECT_SPEC

    _SCORING_OPTIONS: dict = {
        "stim (w)": [("stim", "pos")],
        "stim (|w|)": [("stim", "abs")],
        "stim_param (w)": [("stim_param", "pos")],
        "stim_x_delay (w)": [("stim_x_delay", "pos")],
        "stim_x_delay_param (w)": [("stim_x_delay_param", "pos")],
        "4-state high stim_param + bias": [],
        "4-state high stim_x_delay_param + bias": [],
        "delay (|w|)": [("delay", "abs")],
        "stim_x_delay (|w|)": [("stim_x_delay", "abs")],
        "stim_x_delay_param (|w|)": [("stim_x_delay_param", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_x_delay_param (w)"
    state_scoring_feature: str | None = None
    state_scoring_rule: str = "+"
    state_split_feature: str | None = None
    state_split_rule: str = "+"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("drug") == "Rest")

    def choice_half_life(self, subject: str | None) -> float | None:
        return load_subject_choice_half_life(
            task_key=self.task_key,
            fit_model_id=_RAW_PARAM_MODEL_ID,
            subject=subject,
        )

    def _build_feature_df(
        self,
        df_sub: pl.DataFrame,
        tau: float = 50.0,
    ) -> pl.DataFrame:
        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["session", "trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)
        subject_half_life = self.choice_half_life(
            str(df_pd["subject"].iloc[0]) if "subject" in df_pd.columns and len(df_pd) else None,
        )
        delay_levels = list(_all_delay_levels())
        if not delay_levels:
            delay_levels = sorted(
                {
                    float(v)
                    for v in pd.to_numeric(df_pd["delays"], errors="coerce").dropna().tolist()
                }
            )
        max_sessions = _max_sessions_from_df(df_pd)
        session_order = list(dict.fromkeys(df_pd["session"].tolist()))
        session_to_idx = {session_name: idx for idx, session_name in enumerate(session_order)}
        session_reward_totals = {
            session_name: float(pd.to_numeric(df_session["hit"], errors="coerce").fillna(0.0).sum())
            for session_name, df_session in df_pd.groupby("session", sort=False)
        }
        choice_lag_cols = lag_names("choice_lag_", 15)
        reward_lag_cols = lag_names("reward_lag_", _NUM_REWARD_LAGS)
        prev_day_reward_lag_cols = lag_names("prev_day_total_reward_lag_", _NUM_DAY_REWARD_LAGS)
        prev_day_reward_lag_maps: dict[str, dict[Any, float]] = {}
        for lag_idx, lag_col in enumerate(prev_day_reward_lag_cols, start=1):
            raw_values = [
                float(session_reward_totals.get(session_order[idx - lag_idx], 0.0))
                if idx >= lag_idx
                else 0.0
                for idx in range(len(session_order))
            ]
            scaled_values = _scale_by_max_sequence(raw_values)
            prev_day_reward_lag_maps[lag_col] = dict(zip(session_order, scaled_values))
        prev_difficulty_lag_cols = lag_names("prev_difficulty_lag_", _NUM_DIFFICULTY_LAGS)

        parts: list[pd.DataFrame] = []
        for _, df_session in df_pd.groupby("session", sort=False):
            part = df_session.copy().reset_index(drop=True)
            part["bias"] = 1.0
            part["stim_signed"] = pd.to_numeric(part["stim"], errors="coerce").astype(np.float32)
            part["stim"] = part["stim_signed"].astype(np.float32)
            part["choice_signed"] = _signed_choice(part["choices"]).astype(np.float32)
            part["choice_bin"] = _signed_to_binary(part["choices"]).astype(np.float32)
            part["model_choice_bin"] = _signed_to_model_binary(part["choices"]).astype(np.float32)
            part["delay_raw"] = part["delays"].astype(np.float32)
            session_name = df_session["session"].iloc[0]
            session_idx = session_to_idx[df_session["session"].iloc[0]]
            prev_day_reward_lags = {
                lag_col: float(prev_day_reward_lag_maps[lag_col].get(session_name, 0.0))
                for lag_col in prev_day_reward_lag_cols
            }
            prev_day_total_reward = prev_day_reward_lags.get(
                "prev_day_total_reward_lag_01",
                0.0,
            )
            bias_hot = session_one_hot_frame(session_idx, max_sessions, part.index)
            delay_hot_cols = {
                f"{_DELAY_HOT_COL_PREFIX}{_delay_level_token(delay_value)}": np.where(
                    part["delay_raw"] == np.float32(delay_value),
                    1.0,
                    0.0,
                ).astype(np.float32)
                for delay_value in delay_levels
            }
            stimx_delay_hot_cols = {
                f"stim_x_delay_hot_{_delay_level_token(delay_value)}": (
                    part["stim"].to_numpy(dtype=np.float32) * np.where(
                        part["delay_raw"] == np.float32(delay_value),
                        1.0,
                        0.0,
                    ).astype(np.float32)
                ).astype(np.float32)
                for delay_value in delay_levels
            }
            choice_lag_df = shifted_lag_frame(part["choice_signed"], choice_lag_cols, part.index)
            choice_lag_corr_df, choice_lag_inc_df = choice_outcome_lag_frames(
                part["choice_signed"],
                part["hit"],
                n_lags=15,
                index=part.index,
            )
            reward_lag_df = shifted_lag_frame(part["hit"], reward_lag_cols, part.index)
            prev_day_reward_lag_df = constant_frame(prev_day_reward_lags, part.index)
            difficulty_hot_df = level_indicator_frame(
                part["delay_raw"],
                delay_levels,
                prefix="difficulty_hot_",
                index=part.index,
                label=_delay_level_token,
            )
            prev_difficulty_hot_df = level_indicator_frame(
                part["delay_raw"].shift(1).fillna(0.0),
                delay_levels,
                prefix="prev_difficulty_hot_",
                index=part.index,
                label=_delay_level_token,
            )
            prev_difficulty_lag_df = shifted_lag_frame(
                part["delay_raw"].abs(),
                prev_difficulty_lag_cols,
                part.index,
            )
            prev_difficulty_lag_hot_df = lagged_level_indicator_frame(
                part["delay_raw"],
                delay_levels,
                prefix="prev_difficulty_lag_hot_",
                n_lags=_NUM_DIFFICULTY_LAGS,
                index=part.index,
                label=_delay_level_token,
            )

            trace_input = pd.DataFrame(
                {
                    "Choice": _choice_to_binary(part["choice_bin"]),
                    "Hit": part["hit"].astype(float).to_numpy(),
                    "Punish": (1.0 - part["hit"].astype(float)).to_numpy(),
                }
            )
            at_choice, at_error, at_correct, reward_trace = get_action_trace(trace_input)
            if subject_half_life is not None:
                prev_signed_choice = part["choice_signed"].shift(1).fillna(0.0).astype(np.float32)
                at_choice = compute_choice_ewma(
                    prev_signed_choice.to_numpy(dtype=np.float32),
                    half_life=subject_half_life,
                )
            part["at_choice"] = np.asarray(at_choice, dtype=np.float32)
            part["at_error"] = np.asarray(at_error, dtype=np.float32)
            part["at_correct"] = np.asarray(at_correct, dtype=np.float32)
            part["reward_trace"] = np.asarray(reward_trace, dtype=np.float32)

            prev_choice = part["choice_signed"].shift(1).fillna(0).astype(np.float32)
            prev_reward = part["hit"].shift(1).fillna(0).astype(np.float32)
            current_reward = part["hit"].fillna(0).astype(np.float32)
            part["prev_choice"] = prev_choice
            part["prev_reward"] = prev_reward
            current_stim_side = np.sign(part["stim"].fillna(0.0).astype(float)).astype(np.float32)
            lagged_stim_side = current_stim_side.shift(1).fillna(0.0).astype(np.float32)
            lagged_abs_delay = part["delay_raw"].abs().shift(1).fillna(0.0).astype(np.float32)

            cumulative_reward = part["hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            cumulative_reward = pd.Series(
                _zscore_sequence(cumulative_reward.to_numpy()),
                index=part.index,
                dtype=np.float32,
            )
            part["cumulative_reward"] = cumulative_reward.astype(np.float32)
            part["prev_abs_stim"] = part["stim"].abs().shift(1).fillna(0).astype(np.float32)
            part["prev_abs_delay"] = lagged_abs_delay
            part["prev_day_total_reward"] = np.full(len(part), prev_day_total_reward, dtype=np.float32)
            part["prev_day_total_reward_x_cumulative_reward"] = (
                prev_day_total_reward * cumulative_reward
            ).astype(np.float32)
            part["trial_index"] = _session_trial_index(len(part))
            part["prev_difficulty"] = part["prev_abs_delay"].astype(np.float32)
            prev_choice_signed = prev_choice.to_numpy().astype(np.float32)
            signed_prev_reward = np.where(prev_reward.to_numpy() > 0, 1.0, -1.0).astype(np.float32)
            part["wsls"] = (prev_choice_signed * signed_prev_reward).astype(np.float32)
            part["filtered_choice"] = _ewma_time_series(prev_choice)
            part["filtered_stim_side"] = _ewma_time_series(lagged_stim_side)
            part["filtered_reward"] = _ewma_time_series(prev_reward)
            part["filtered_difficulty"] = _ewma_time_series(lagged_abs_delay)
            part["filtered_bad_stim"] = _ewma_time_series(current_stim_side)
            part["filtered_bad_choice"] = _ewma_time_series(
                part["choice_signed"].fillna(0.0).astype(np.float32)
            )
            part["filtered_bad_reward"] = _ewma_time_series(current_reward)

            part["after_correct"] = part["after_correct"].fillna(0).astype(np.float32)
            part["repeat"] = part["repeat"].fillna(0).astype(np.float32)
            part["repeat_choice_side"] = part["repeat_choice_side"].fillna(0).astype(np.float32)
            part["WM"] = part["WM"].fillna(0).astype(np.float32)
            part["RL"] = part["RL"].fillna(0).astype(np.float32)
            part["ILD"] = part["stim"].astype(np.float32)
            feature_frames = [
                ("bias_hot", bias_hot),
                ("delay_hot", pd.DataFrame(delay_hot_cols, index=part.index)),
                ("stim_x_delay_hot", pd.DataFrame(stimx_delay_hot_cols, index=part.index)),
                ("choice_lag", choice_lag_df),
                ("choice_lag_correct", choice_lag_corr_df),
                ("choice_lag_inc", choice_lag_inc_df),
                ("reward_lag", reward_lag_df),
                ("prev_day_total_reward_lag", prev_day_reward_lag_df),
                ("difficulty_hot", difficulty_hot_df),
                ("prev_difficulty_hot", prev_difficulty_hot_df),
                ("prev_difficulty_lag", prev_difficulty_lag_df),
                ("prev_difficulty_lag_hot", prev_difficulty_lag_hot_df),
            ]
            part = pd.concat(
                [part, *(frame for _, frame in feature_frames)],
                axis=1,
            )
            parts.append(part)

        feature_df = pd.concat(parts, ignore_index=True)
        delay_raw = pd.to_numeric(feature_df["delay_raw"], errors="coerce").astype(np.float32)
        delay_mean = float(np.nanmean(delay_raw.to_numpy())) if len(delay_raw) else 0.0
        delay_std = float(np.nanstd(delay_raw.to_numpy())) if len(delay_raw) else 0.0
        if delay_std > 0:
            delay_z = ((delay_raw - delay_mean) / delay_std).astype(np.float32)
        else:
            delay_z = pd.Series(np.zeros(len(feature_df), dtype=np.float32), index=feature_df.index)
        bias_param = _safe_weighted_sum_regressor(feature_df, self.bias_param_spec)
        stim_param = _safe_weighted_sum_regressor(feature_df, self.stim_param_spec)
        delay_param = _safe_weighted_sum_regressor(feature_df, self.delay_param_spec)
        stim_x_delay_param = _safe_weighted_sum_regressor(feature_df, self.stim_x_delay_param_spec)
        choice_lag_param = _safe_weighted_sum_regressor(feature_df, self.choice_lag_param_spec)
        choice_lag_param_2 = _safe_weighted_sum_regressor(feature_df, self.choice_lag_param_2_spec)
        choice_lag_param_correct = _safe_weighted_sum_regressor(feature_df, self.choice_lag_param_correct_spec)
        reward_lag_cols = numeric_prefixed(list(feature_df.columns), "reward_lag_")
        difficulty_hot_cols = _difficulty_hot_cols(list(feature_df.columns))
        prev_difficulty_hot_cols = _prev_difficulty_lag_hot_cols(list(feature_df.columns))
        delay_np = np.asarray(delay_z, dtype=np.float32)
        derived_cols = pd.DataFrame(
            {
                "delay": delay_np,
                "stim_x_delay": (
                    feature_df["stim"].to_numpy(dtype=np.float32) * delay_np
                ).astype(np.float32),
                "bias_param": (
                    np.asarray(bias_param, dtype=np.float32)
                    if bias_param is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "stim_param": (
                    np.asarray(stim_param, dtype=np.float32)
                    if stim_param is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "delay_param": (
                    np.asarray(delay_param, dtype=np.float32)
                    if delay_param is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "stim_x_delay_param": (
                    np.asarray(stim_x_delay_param, dtype=np.float32)
                    if stim_x_delay_param is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "choice_lag_param": (
                    np.asarray(choice_lag_param, dtype=np.float32)
                    if choice_lag_param is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "choice_lag_param_2": (
                    np.asarray(choice_lag_param_2, dtype=np.float32)
                    if choice_lag_param_2 is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "choice_lag_param_correct": (
                    np.asarray(choice_lag_param_correct, dtype=np.float32)
                    if choice_lag_param_correct is not None
                    else np.zeros(len(feature_df), dtype=np.float32)
                ),
                "reward_lag_param": transition_weighted_sum(
                    feature_df,
                    fit_task=self.task_key,
                    fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                    source_features=reward_lag_cols,
                    fallback=np.asarray(feature_df[reward_lag_cols], dtype=np.float32).mean(axis=1),
                ),
                "difficulty_hot_param": transition_weighted_sum(
                    feature_df,
                    fit_task=self.task_key,
                    fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                    source_features=difficulty_hot_cols,
                    fallback=delay_np,
                ),
                "prev_difficulty_param": transition_weighted_sum(
                    feature_df,
                    fit_task=self.task_key,
                    fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                    source_features=prev_difficulty_hot_cols,
                    fallback=np.asarray(feature_df["prev_difficulty"], dtype=np.float32),
                ),
            },
            index=feature_df.index,
        )
        feature_df = pd.concat([feature_df, derived_cols], axis=1)
        return pl.from_pandas(feature_df)

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        return self._build_feature_df(df_sub, tau=tau)

    def _resolved_emission_cols(
        self,
        feature_df: pl.DataFrame,
        emission_cols: List[str] | None,
    ) -> list[str]:
        requested = emission_cols if emission_cols is not None else self.default_emission_cols(feature_df)
        expanded: list[str] = []
        delay_hot_cols = _infer_delay_hot_cols_from_df(feature_df)
        stim_x_delay_hot_cols = _stim_x_delay_hot_cols(list(feature_df.columns))
        if not stim_x_delay_hot_cols:
            stim_x_delay_hot_cols = [
                f"stim_x_delay_hot_{col.removeprefix(_DELAY_HOT_COL_PREFIX)}"
                for col in delay_hot_cols
            ]
        choice_lag_cols = numeric_prefixed(list(feature_df.columns), "choice_lag_")
        family_aliases = {
            "bias_hot": _infer_bias_hot_cols_from_df(feature_df),
            "delay_hot": delay_hot_cols,
            "choice_lag": choice_lag_cols,
            "at_choice_lag": choice_lag_cols,
            "choice_lag_correct": numeric_prefixed(list(feature_df.columns), "choice_lag_corr_"),
            "choice_lag_inc": numeric_prefixed(list(feature_df.columns), "choice_lag_inc_"),
            "choice_lag_15_lags": numeric_prefixed(list(feature_df.columns), "choice_lag_", max_count=_NUM_LEGACY_CHOICE_LAGS),
            "choice_lag_50_lags": numeric_prefixed(list(feature_df.columns), "choice_lag_", max_count=_NUM_MEDIUM_CHOICE_LAGS),
            "choice_lag_100_lags": choice_lag_cols,
            "at_choice_lag_15_lags": numeric_prefixed(list(feature_df.columns), "choice_lag_", max_count=_NUM_LEGACY_CHOICE_LAGS),
            "at_choice_lag_50_lags": numeric_prefixed(list(feature_df.columns), "choice_lag_", max_count=_NUM_MEDIUM_CHOICE_LAGS),
            "at_choice_lag_100_lags": choice_lag_cols,
            "stim_x_delay_hot": stim_x_delay_hot_cols,
            "stim_x_delay_one_hot": stim_x_delay_hot_cols,
        }
        for col in requested:
            expanded.extend(family_aliases.get(col, [col]))
        return list(dict.fromkeys(expanded))

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        feature_df = self._build_feature_df(df_sub, tau=tau)
        return self.build_design_matrices(
            feature_df,
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )

    def build_design_matrices(
        self,
        feature_df,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        ecols = self._resolved_emission_cols(feature_df, emission_cols)
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(feature_df))
        ecols = _drop_unavailable_bias_hot_cols(list(ecols), allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        dynamic_ucols = [
            *numeric_prefixed(list(feature_df.columns), "reward_lag_"),
            *_difficulty_hot_cols(list(feature_df.columns)),
            *_prev_difficulty_hot_cols(list(feature_df.columns)),
        ]
        allowed_ucols = list(dict.fromkeys([*self.available_transition_cols(), *dynamic_ucols]))
        bad_u = [c for c in ucols if c not in allowed_ucols]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {allowed_ucols}"
            )
        missing_e = [c for c in ecols if c not in feature_df.columns]
        missing_u = [c for c in ucols if c not in feature_df.columns]
        if missing_e or missing_u:
            raise ValueError(
                "Requested design columns are not present in feature_df. "
                f"Missing emission columns: {missing_e}; missing transition columns: {missing_u}. "
                "Build optional features through load_subject or build_feature_df before calling build_design_matrices."
            )

        y_np = feature_df["model_choice_bin"].to_numpy().astype(np.int32)
        y = jnp.asarray(y_np)
        X = (
            jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32))
            if ecols
            else jnp.empty((len(y), 0), dtype=jnp.float32)
        )
        U = (
            jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32))
            if ucols
            else jnp.empty((len(y), 0), dtype=jnp.float32)
        )
        names = {
            "X_cols": list(ecols),
            "U_cols": list(ucols),
        }
        return y, X, U, names

    def cv_balance_labels(self, feature_df: pl.DataFrame):
        if "stim" not in feature_df.columns:
            return None
        return feature_df["stim"].cast(pl.Float64)

    def default_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        if df is None:
            delay_hot_cols = [
                f"{_DELAY_HOT_COL_PREFIX}{_delay_level_token(delay_value)}"
                for delay_value in _all_delay_levels()
            ]
            stim_x_delay_hot_cols = [
                f"stim_x_delay_hot_{_delay_level_token(delay_value)}"
                for delay_value in _all_delay_levels()
            ]
            bias_hot_cols = [f"bias_{idx}" for idx in range(_max_subject_sessions())]
            choice_lag_cols = lag_names("choice_lag_", 15)
        else:
            delay_hot_cols = _infer_delay_hot_cols_from_df(df)
            stim_x_delay_hot_cols = _stim_x_delay_hot_cols(list(df.columns))
            if not stim_x_delay_hot_cols:
                stim_x_delay_hot_cols = [
                    f"stim_x_delay_hot_{col.removeprefix(_DELAY_HOT_COL_PREFIX)}"
                    for col in delay_hot_cols
                ]
            bias_hot_cols = _infer_bias_hot_cols_from_df(df)
            choice_lag_cols = (
                numeric_prefixed(list(df.columns), "choice_lag_")
                or lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
            )

        default_cols = [
            *self.emission_cols,
            *bias_hot_cols,
            *choice_lag_cols,
            *delay_hot_cols,
            *stim_x_delay_hot_cols,
        ]
        return list(dict.fromkeys(default_cols))

    def default_transition_cols(self) -> List[str]:
        return list(dict.fromkeys(self.transition_cols))

    def available_transition_cols(self) -> List[str]:
        return list(
            dict.fromkeys(
                [
                    *self.default_transition_cols(),
                    *_LEGACY_TRANSITION_COLS,
                    *lag_names("reward_lag_", _NUM_REWARD_LAGS),
                    *lag_names("prev_day_total_reward_lag_", _NUM_DAY_REWARD_LAGS),
                    *_all_difficulty_hot_names(),
                    *_all_prev_difficulty_hot_names(),
                    *lag_names("prev_difficulty_lag_", _NUM_DIFFICULTY_LAGS),
                    *_all_prev_difficulty_lag_hot_names(),
                ]
            )
        )

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        available_cols = list(self.emission_cols)
        available_cols.extend(
            [
                "choice_lag_15_lags",
                "choice_lag_50_lags",
                "choice_lag_100_lags",
                "choice_lag_correct",
                "choice_lag_inc",
                "at_choice_lag_15_lags",
                "at_choice_lag_50_lags",
                "at_choice_lag_100_lags",
            ]
        )
        available_cols.extend(
            (
                numeric_prefixed(list(df.columns), "choice_lag_")
                or lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
            )
            if df is not None
            else lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
        )
        available_cols.extend(
            (
                numeric_prefixed(list(df.columns), "choice_lag_corr_")
                or lag_names("choice_lag_corr_", _NUM_LEGACY_CHOICE_LAGS)
            )
            if df is not None
            else lag_names("choice_lag_corr_", _NUM_LEGACY_CHOICE_LAGS)
        )
        available_cols.extend(
            (
                numeric_prefixed(list(df.columns), "choice_lag_inc_")
                or lag_names("choice_lag_inc_", _NUM_LEGACY_CHOICE_LAGS)
            )
            if df is not None
            else lag_names("choice_lag_inc_", _NUM_LEGACY_CHOICE_LAGS)
        )
        if df is not None:
            delay_hot_cols = _infer_delay_hot_cols_from_df(df)
            stim_x_delay_hot_cols = _stim_x_delay_hot_cols(list(df.columns))
            if not stim_x_delay_hot_cols:
                stim_x_delay_hot_cols = [
                    f"stim_x_delay_hot_{col.removeprefix(_DELAY_HOT_COL_PREFIX)}"
                    for col in delay_hot_cols
                ]
            available_cols.extend(delay_hot_cols)
            available_cols.extend(stim_x_delay_hot_cols)
            available_cols.extend(_infer_bias_hot_cols_from_df(df))
        return list(dict.fromkeys(available_cols))

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df: pl.DataFrame | None = None,
    ) -> Dict[str, List[str]]:
        requested_ecols = list(emission_cols) if emission_cols is not None else self.default_emission_cols(df)
        requested_ucols = list(transition_cols) if transition_cols is not None else self.default_transition_cols()
        expanded_ecols: list[str] = []
        if df is not None:
            columns = list(df.columns)
            delay_hot_cols = _infer_delay_hot_cols_from_df(df)
            stim_x_delay_hot_cols = _stim_x_delay_hot_cols(columns)
            if not stim_x_delay_hot_cols:
                stim_x_delay_hot_cols = [
                    f"stim_x_delay_hot_{col.removeprefix(_DELAY_HOT_COL_PREFIX)}"
                    for col in delay_hot_cols
                ]
            choice_lag_cols = (
                numeric_prefixed(columns, "choice_lag_")
                or lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
            )
            choice_lag_corr_cols = (
                numeric_prefixed(columns, "choice_lag_corr_")
                or lag_names("choice_lag_corr_", _NUM_LEGACY_CHOICE_LAGS)
            )
            choice_lag_inc_cols = (
                numeric_prefixed(columns, "choice_lag_inc_")
                or lag_names("choice_lag_inc_", _NUM_LEGACY_CHOICE_LAGS)
            )
            family_aliases = {
                "bias_hot": _infer_bias_hot_cols_from_df(df),
                "delay_hot": delay_hot_cols,
                "choice_lag": choice_lag_cols,
                "at_choice_lag": choice_lag_cols,
                "choice_lag_correct": choice_lag_corr_cols,
                "choice_lag_inc": choice_lag_inc_cols,
                "choice_lag_15_lags": (
                    numeric_prefixed(columns, "choice_lag_", max_count=_NUM_LEGACY_CHOICE_LAGS)
                    or lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
                ),
                "choice_lag_50_lags": (
                    numeric_prefixed(columns, "choice_lag_", max_count=_NUM_MEDIUM_CHOICE_LAGS)
                    or lag_names("choice_lag_", _NUM_MEDIUM_CHOICE_LAGS)
                ),
                "choice_lag_100_lags": choice_lag_cols,
                "at_choice_lag_15_lags": (
                    numeric_prefixed(columns, "choice_lag_", max_count=_NUM_LEGACY_CHOICE_LAGS)
                    or lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
                ),
                "at_choice_lag_50_lags": (
                    numeric_prefixed(columns, "choice_lag_", max_count=_NUM_MEDIUM_CHOICE_LAGS)
                    or lag_names("choice_lag_", _NUM_MEDIUM_CHOICE_LAGS)
                ),
                "at_choice_lag_100_lags": choice_lag_cols,
                "stim_x_delay_hot": stim_x_delay_hot_cols,
                "stim_x_delay_one_hot": stim_x_delay_hot_cols,
            }
            for col in requested_ecols:
                expanded_ecols.extend(family_aliases.get(col, [col]))
        else:
            choice_lag_cols = lag_names("choice_lag_", 15)
            family_aliases = {
                "choice_lag": choice_lag_cols,
                "at_choice_lag": choice_lag_cols,
                "choice_lag_correct": lag_names("choice_lag_corr_", 15),
                "choice_lag_inc": lag_names("choice_lag_inc_", 15),
                "choice_lag_15_lags": lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS),
                "choice_lag_50_lags": lag_names("choice_lag_", _NUM_MEDIUM_CHOICE_LAGS),
                "choice_lag_100_lags": choice_lag_cols,
                "at_choice_lag_15_lags": lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS),
                "at_choice_lag_50_lags": lag_names("choice_lag_", _NUM_MEDIUM_CHOICE_LAGS),
                "at_choice_lag_100_lags": choice_lag_cols,
            }
            for col in requested_ecols:
                expanded_ecols.extend(family_aliases.get(col, [col]))
        allowed_ecols = set(self.available_emission_cols(df))
        expanded_ecols = _drop_unavailable_bias_hot_cols(expanded_ecols, allowed_ecols)
        bad_e = [c for c in expanded_ecols if c not in allowed_ecols]
        dynamic_ucols: list[str] = []
        if df is not None:
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "reward_lag_"))
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "prev_day_total_reward_lag_"))
            dynamic_ucols.extend(_difficulty_hot_cols(list(df.columns)))
            dynamic_ucols.extend(_prev_difficulty_hot_cols(list(df.columns)))
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "prev_difficulty_lag_"))
            dynamic_ucols.extend(_prev_difficulty_lag_hot_cols(list(df.columns)))
        allowed_ucols = list(dict.fromkeys([*self.available_transition_cols(), *dynamic_ucols]))
        bad_u = [c for c in requested_ucols if c not in allowed_ucols]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {allowed_ucols}"
            )
        return {"X_cols": list(dict.fromkeys(expanded_ecols)), "U_cols": list(requested_ucols)}

    def build_emission_groups(self, available_cols: List[str]) -> list[dict]:
        return _build_emission_groups(list(available_cols))

    def build_transition_groups(self, available_cols: List[str]) -> list[dict]:
        return _build_transition_groups(list(available_cols))

    @property
    def choice_labels(self) -> list[str]:
        return ["Right", "Left"]

    @property
    def probability_columns(self) -> list[str]:
        return ["pR", "pL"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        stim = df["stimulus"].to_numpy().astype(float)
        unique = set(np.unique(stim[~np.isnan(stim)]).tolist())
        if unique.issubset({0.0, 1.0}):
            return np.where(stim > 0, 0, 1).astype(int)
        if unique.issubset({-1.0, 1.0}):
            return np.where(stim > 0, 0, 1).astype(int)
        return np.where(stim > 0, 0, np.where(stim < 0, 1, -1)).astype(int)

    @property
    def behavioral_cols(self) -> dict:
        return {
            "trial_idx": "trial",
            "trial": "trial",
            "session": "session",
            "stimulus": "stim",
            "response": "choices",
            "performance": "hit",
        }

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        return label_states_by_regressor(
            arrays_store,
            names,
            K,
            subjects,
            scoring_key=getattr(self, "scoring_key", None),
            scoring_options=getattr(self, "_SCORING_OPTIONS", None),
            primary_feature=getattr(self, "state_scoring_feature", None),
            primary_rule=getattr(self, "state_scoring_rule", "+"),
            split_feature=getattr(self, "state_split_feature", None),
            split_rule=getattr(self, "state_split_rule", "+"),
            preferred_features=("stim_x_delay_param", "stim_param", "stim_x_delay", "stim"),
        )
