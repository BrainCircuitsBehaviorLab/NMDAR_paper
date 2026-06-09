"""Task adapter for the 2AFC (Alexis human) task."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import jax.numpy as jnp
import polars as pl

from ._choice_tau import compute_choice_ewma, load_subject_choice_half_life
from ._transition_params import transition_weighted_sum
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    mean_feature_weights_from_fit,
    resolved_source_features,
    weighted_sum_regressor,
)
from glmhmmt.tasks import TaskAdapter, _register
from glmhmmt.runtime import get_data_dir

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
    
from src.process.common import (
    PreparedWeightFamilyPlot,
    attach_quantile_bin_column,
    attach_response_right_column,
    display_regressor_name,
    label_states_by_regressor,
    mean_glm_feature_curve as _mean_glm_feature_curve,
    mean_glm_ild_curve as _mean_glm_ild_curve,
    p_right_label,
    prepare_grouped_weight_family_plot,
    prepare_simple_regressor_curve,
    resolve_grouping,
    summarize_grouped_panel,
    subject_glm_feature_curves as _subject_glm_feature_curves,
    subject_glm_ild_curves as _subject_glm_ild_curves,
    to_pandas_df,
)


# Default experiments to keep (avoids habituation / drug sessions)
_KEEP_EXPERIMENTS = ["2AFC_2", "2AFC_3", "2AFC_4", "2AFC_6"]
_SF_COL_PREFIX = "sf_"
_STIM_ABS_COL_PREFIX = "stim_"
_ABS_ILD_HOT_COL_PREFIX = "abs_ILD_hot_"
_BIAS_HOT_COL_PREFIX = "bias_"
_CHOICE_LAG_COL_PREFIX = "choice_lag_"
_CHOICE_LAG_CORR_COL_PREFIX = "choice_lag_corr_"
_CHOICE_LAG_INC_COL_PREFIX = "choice_lag_inc_"
_CHOICE_LAG_CORR_ALIAS = "choice_lag_corr"
_CHOICE_LAG_INC_ALIAS = "choice_lag_inc"
_CHOICE_LAG_15_ALIAS = "choice_lag_15_lags"
_CHOICE_LAG_50_ALIAS = "choice_lag_50_lags"
_CHOICE_LAG_100_ALIAS = "choice_lag_100_lags"
_AT_CHOICE_LAG_15_ALIAS = "at_choice_lag_15_lags"
_AT_CHOICE_LAG_50_ALIAS = "at_choice_lag_50_lags"
_AT_CHOICE_LAG_100_ALIAS = "at_choice_lag_100_lags"
_REWARD_LAG_COL_PREFIX = "reward_lag_"
_DIFFICULTY_HOT_COL_PREFIX = "difficulty_hot_"
_PREV_DIFFICULTY_HOT_COL_PREFIX = "prev_difficulty_hot_"
_PREV_DIFFICULTY_LAG_COL_PREFIX = "prev_difficulty_lag_"
_PREV_DIFFICULTY_LAG_HOT_COL_PREFIX = "prev_difficulty_lag_hot_"
_PREV_DAY_REWARD_LAG_COL_PREFIX = "prev_day_total_reward_lag_"
_NUM_CHOICE_LAGS = 40
_NUM_LEGACY_CHOICE_LAGS = 15
_NUM_MEDIUM_CHOICE_LAGS = 40
_NUM_REWARD_LAGS = 40
_NUM_DIFFICULTY_LAGS = 20
_NUM_DAY_REWARD_LAGS = 5
_FILTERED_REGRESSOR_TAU = 4.0
_RAW_PARAM_MODEL_ID = "one hot"
_TRANSITION_PARAM_MODEL_ID = "one hot"
EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "stim_vals",
    "stim_side",
    "abs_ILD",
    "stim_param",
    "stim_strength",
    "at_choice",
    "at_choice_param",
    "choice_lag_param",
    "choice_lag_param_2",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "wsls",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
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
    "prev_abs_stim",
    "prev_reward",
    "prev_difficulty",
    "prev_difficulty_param",
]
_STIM_PARAM_COL = "stim_param"
_CHOICE_LAG_PARAM_COL = "choice_lag_param"
_CHOICE_LAG_PARAM_2_COL = "choice_lag_param_2"
_STIM_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_param",
    fit_task="2AFC",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_STIM_ABS_COL_PREFIX,),
    exclude_features=("stim_0",),
    excluded_subjects=("325", "325.0"),
    sign=1.0,
)
_BIAS_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="bias_param",
    fit_task="2AFC",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_BIAS_HOT_COL_PREFIX,),
)
_AT_CHOICE_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="at_choice_param",
    fit_task="2AFC",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_CHOICE_LAG_COL_PREFIX,),
)
_CHOICE_LAG_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name=_CHOICE_LAG_PARAM_COL,
    fit_task="2AFC",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_CHOICE_LAG_COL_PREFIX,),
)
_CHOICE_LAG_PARAM_2_SPEC = FittedWeightRegressorSpec(
    target_name=_CHOICE_LAG_PARAM_2_COL,
    fit_task="2AFC",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_CHOICE_LAG_COL_PREFIX,),
    exclude_features=(f"{_CHOICE_LAG_COL_PREFIX}01",),
)

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim_vals": r"$\mathrm{Stimulus}$",
    "stim_side": r"$\mathrm{StimSide}$",
    "abs_ILD": r"$|\mathrm{ILD}|$",
    "stim_param": r"$\mathrm{Stimulus}_{\mathrm{param}}$",
    "stim_strength": r"$\mathrm{Stimulus}_{\mathrm{strength}}$",
    "bias": r"$\mid\mathrm{bias}\mid$",
    "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
    "at_choice": r"$\mathrm{A}_t^{\mathrm{choice}}$",
    "at_choice_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    "choice_lag_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    "choice_lag_param_2": r"$\mathrm{A}_{t,\geq 2}^{\mathrm{choice,param}}$",
    "at_error": r"$\mathrm{A}_t^{\mathrm{error}}$",
    "at_correct": r"$\mathrm{A}_t^{\mathrm{correct}}$",
    "reward_trace": r"$\mathrm{Reward}_{\mathrm{trace}}$",
    "prev_choice": r"$\mathrm{PrevChoice}$",
    "prev_reward": r"$\mathrm{PrevReward}$",
    "prev_abs_stim": r"$|\mathrm{PrevStim}|$",
    "filtered_bad_stim": r"$\mathrm{FilteredBadStimSide}$",
    "filtered_bad_choice": r"$\mathrm{FilteredBadChoice}$",
    "filtered_bad_reward": r"$\mathrm{FilteredBadReward}$",
    "cumulative_reward": r"$\mathrm{CumReward}$",
    "wsls": r"$\mathrm{WSLS}$",
}

_EMISSION_GROUPS: list[dict] = [
    {"key": "bias", "label": "bias", "members": {"N": "bias"}},
    {"key": "bias_param", "label": "bias param", "members": {"N": "bias_param"}},
    {"key": "stim_vals", "label": "stim vals", "members": {"N": "stim_vals"}},
    {"key": "stim_side", "label": "stim side", "members": {"N": "stim_side"}},
    {"key": "abs_ILD", "label": "abs ILD", "members": {"N": "abs_ILD"}},
    {"key": "stim_param", "label": "stim param", "members": {"N": "stim_param"}},
    {"key": "stim_strength", "label": "stim strength", "members": {"N": "stim_strength"}},
    {"key": "at_choice", "label": "action (choice)", "members": {"N": "at_choice"}},
    {"key": "at_choice_param", "label": "choice param", "members": {"N": "at_choice_param"}},
    {"key": "choice_lag_param", "label": "choice lag param", "members": {"N": "choice_lag_param"}},
    {"key": "choice_lag_param_2", "label": "choice lag param 2+", "members": {"N": "choice_lag_param_2"}},
    {"key": "at_error", "label": "action (error)", "members": {"N": "at_error"}},
    {"key": "at_correct", "label": "action (correct)", "members": {"N": "at_correct"}},
    {"key": "reward_trace", "label": "reward trace", "members": {"N": "reward_trace"}},
    {"key": "prev_choice", "label": "prev choice", "members": {"N": "prev_choice"}},
    {"key": "wsls", "label": "WSLS", "members": {"N": "wsls"}},
    {"key": "prev_reward", "label": "prev reward", "members": {"N": "prev_reward"}},
    {"key": "cumulative_reward", "label": "cumulative reward", "members": {"N": "cumulative_reward"}},
    {"key": "prev_abs_stim", "label": "prev abs stim", "members": {"N": "prev_abs_stim"}},
]


def _sf_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_SF_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _stim_abs_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_STIM_ABS_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _abs_ild_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_ABS_ILD_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _bias_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_BIAS_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_corr_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_CORR_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_inc_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_INC_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _reward_lag_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_REWARD_LAG_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _difficulty_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_DIFFICULTY_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _prev_difficulty_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_PREV_DIFFICULTY_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _prev_difficulty_lag_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_PREV_DIFFICULTY_LAG_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _prev_difficulty_lag_hot_sort_key(name: str) -> tuple[int, int, str]:
    suffix = name.removeprefix(_PREV_DIFFICULTY_LAG_HOT_COL_PREFIX)
    lag, sep, level = suffix.partition("_")
    if sep and lag.isdigit() and level.isdigit():
        return (int(lag), int(level), name)
    return (10**9, 10**9, name)


def _prev_day_reward_lag_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_PREV_DAY_REWARD_LAG_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _ewma_time_series(values: Sequence[float], period: float = _FILTERED_REGRESSOR_TAU) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_np.size == 0:
        return values_np.copy()
    ewma = pd.DataFrame(data=values_np).ewm(span=float(period)).mean()
    return ewma.iloc[:, 0].to_numpy(dtype=np.float32)


def _stim_abs_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_STIM_ABS_COL_PREFIX)
            and col.removeprefix(_STIM_ABS_COL_PREFIX).isdigit()
        ],
        key=_stim_abs_sort_key,
    )


def _abs_ild_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_ABS_ILD_HOT_COL_PREFIX)
            and col.removeprefix(_ABS_ILD_HOT_COL_PREFIX).isdigit()
        ],
        key=_abs_ild_hot_sort_key,
    )


def _bias_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_BIAS_HOT_COL_PREFIX)
            and col.removeprefix(_BIAS_HOT_COL_PREFIX).isdigit()
        ],
        key=_bias_hot_sort_key,
    )


def _is_bias_hot_col(col: str) -> bool:
    return col.startswith(_BIAS_HOT_COL_PREFIX) and col.removeprefix(_BIAS_HOT_COL_PREFIX).isdigit()


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


def _choice_lag_cols(columns: list[str], max_lags: int | None = None) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_CHOICE_LAG_COL_PREFIX)
            and col.removeprefix(_CHOICE_LAG_COL_PREFIX).isdigit()
            and (
                max_lags is None
                or int(col.removeprefix(_CHOICE_LAG_COL_PREFIX)) <= int(max_lags)
            )
        ],
        key=_choice_lag_sort_key,
    )


def _choice_lag_corr_cols(columns: list[str], max_lags: int | None = None) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_CHOICE_LAG_CORR_COL_PREFIX)
            and col.removeprefix(_CHOICE_LAG_CORR_COL_PREFIX).isdigit()
            and (
                max_lags is None
                or int(col.removeprefix(_CHOICE_LAG_CORR_COL_PREFIX)) <= int(max_lags)
            )
        ],
        key=_choice_lag_corr_sort_key,
    )


def _choice_lag_inc_cols(columns: list[str], max_lags: int | None = None) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_CHOICE_LAG_INC_COL_PREFIX)
            and col.removeprefix(_CHOICE_LAG_INC_COL_PREFIX).isdigit()
            and (
                max_lags is None
                or int(col.removeprefix(_CHOICE_LAG_INC_COL_PREFIX)) <= int(max_lags)
            )
        ],
        key=_choice_lag_inc_sort_key,
    )


def _reward_lag_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_REWARD_LAG_COL_PREFIX)
            and col.removeprefix(_REWARD_LAG_COL_PREFIX).isdigit()
        ],
        key=_reward_lag_sort_key,
    )


def _difficulty_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_DIFFICULTY_HOT_COL_PREFIX)
            and col.removeprefix(_DIFFICULTY_HOT_COL_PREFIX).isdigit()
        ],
        key=_difficulty_hot_sort_key,
    )


def _prev_difficulty_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_PREV_DIFFICULTY_HOT_COL_PREFIX)
            and col.removeprefix(_PREV_DIFFICULTY_HOT_COL_PREFIX).isdigit()
        ],
        key=_prev_difficulty_hot_sort_key,
    )


def _prev_difficulty_lag_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_PREV_DIFFICULTY_LAG_COL_PREFIX)
            and col.removeprefix(_PREV_DIFFICULTY_LAG_COL_PREFIX).isdigit()
        ],
        key=_prev_difficulty_lag_sort_key,
    )


def _prev_difficulty_lag_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_PREV_DIFFICULTY_LAG_HOT_COL_PREFIX)
            and _prev_difficulty_lag_hot_sort_key(col)[0] < 10**9
        ],
        key=_prev_difficulty_lag_hot_sort_key,
    )


def _prev_day_reward_lag_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_PREV_DAY_REWARD_LAG_COL_PREFIX)
            and col.removeprefix(_PREV_DAY_REWARD_LAG_COL_PREFIX).isdigit()
        ],
        key=_prev_day_reward_lag_sort_key,
    )


def _infer_stim_abs_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = _stim_abs_cols(columns)
    if existing:
        return existing
    if "ILD" not in columns:
        return []
    ild_series = df["ILD"].drop_nulls() if isinstance(df, pl.DataFrame) else df["ILD"].dropna()
    stim_abs_levels = sorted({int(abs(v)) for v in ild_series.to_list()})
    return [f"{_STIM_ABS_COL_PREFIX}{stim_abs}" for stim_abs in stim_abs_levels]


def _infer_abs_ild_hot_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = _abs_ild_hot_cols(columns)
    if existing:
        return existing
    if "ILD" not in columns:
        return []
    ild_series = df["ILD"].drop_nulls() if isinstance(df, pl.DataFrame) else df["ILD"].dropna()
    stim_abs_levels = sorted({int(abs(v)) for v in ild_series.to_list()})
    return [f"{_ABS_ILD_HOT_COL_PREFIX}{stim_abs}" for stim_abs in stim_abs_levels]


def _choice_lag_names(max_lags: int | None = None) -> list[str]:
    n_lags = _NUM_CHOICE_LAGS if max_lags is None else min(int(max_lags), _NUM_CHOICE_LAGS)
    return [f"{_CHOICE_LAG_COL_PREFIX}{idx:02d}" for idx in range(1, n_lags + 1)]


def _choice_lag_corr_names(max_lags: int | None = None) -> list[str]:
    n_lags = _NUM_CHOICE_LAGS if max_lags is None else min(int(max_lags), _NUM_CHOICE_LAGS)
    return [f"{_CHOICE_LAG_CORR_COL_PREFIX}{idx:02d}" for idx in range(1, n_lags + 1)]


def _choice_lag_inc_names(max_lags: int | None = None) -> list[str]:
    n_lags = _NUM_CHOICE_LAGS if max_lags is None else min(int(max_lags), _NUM_CHOICE_LAGS)
    return [f"{_CHOICE_LAG_INC_COL_PREFIX}{idx:02d}" for idx in range(1, n_lags + 1)]


def _reward_lag_names() -> list[str]:
    return [f"{_REWARD_LAG_COL_PREFIX}{idx:02d}" for idx in range(1, _NUM_REWARD_LAGS + 1)]


def _prev_day_reward_lag_names() -> list[str]:
    return [f"{_PREV_DAY_REWARD_LAG_COL_PREFIX}{idx:02d}" for idx in range(1, _NUM_DAY_REWARD_LAGS + 1)]


def _prev_difficulty_lag_names() -> list[str]:
    return [f"{_PREV_DIFFICULTY_LAG_COL_PREFIX}{idx:02d}" for idx in range(1, _NUM_DIFFICULTY_LAGS + 1)]


def _difficulty_hot_names(levels: Sequence[int]) -> list[str]:
    return [f"{_DIFFICULTY_HOT_COL_PREFIX}{int(level)}" for level in levels]


def _prev_difficulty_hot_names(levels: Sequence[int]) -> list[str]:
    return [f"{_PREV_DIFFICULTY_HOT_COL_PREFIX}{int(level)}" for level in levels]


def _prev_difficulty_lag_hot_names(levels: Sequence[int]) -> list[str]:
    return [
        f"{_PREV_DIFFICULTY_LAG_HOT_COL_PREFIX}{lag_idx:02d}_{int(level)}"
        for lag_idx in range(1, _NUM_DIFFICULTY_LAGS + 1)
        for level in levels
    ]


@lru_cache(maxsize=1)
def _all_stim_abs_levels() -> tuple[int, ...]:
    try:
        dataset_path = get_data_dir() / "alexis_combined.parquet"
        df = pl.read_parquet(dataset_path)
        df = df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))
        levels = sorted({int(abs(v)) for v in df["ILD"].drop_nulls().to_list()})
        return tuple(levels)
    except Exception:
        return tuple()


def _all_difficulty_hot_names() -> list[str]:
    return _difficulty_hot_names(_all_stim_abs_levels())


def _all_prev_difficulty_hot_names() -> list[str]:
    return _prev_difficulty_hot_names(_all_stim_abs_levels())


def _all_prev_difficulty_lag_hot_names() -> list[str]:
    return _prev_difficulty_lag_hot_names(_all_stim_abs_levels())


def _build_emission_groups(available_cols: list[str]) -> list[dict]:
    available = set(available_cols)
    result: list[dict] = []
    registered: set[str] = set()

    def add_scalar(group: dict) -> None:
        filtered = {k: v for k, v in group["members"].items() if v in available}
        if filtered:
            result.append({**group, "members": filtered})
            registered.update(filtered.values())

    def add_hidden_family(*, key: str, label: str, family_cols: list[str], toggle_cols: list[str] | None = None) -> None:
        if not family_cols:
            return
        members = list(toggle_cols if toggle_cols is not None else family_cols)
        result.append(
            {
                "key": key,
                "label": label,
                "members": {},
                "toggle_members": members,
                "hide_members": True,
            }
        )
        registered.update(family_cols)

    stim_cols = _stim_abs_cols(available_cols)
    abs_ild_hot_cols = _abs_ild_hot_cols(available_cols)
    bias_hot_cols = _bias_hot_cols(available_cols)
    choice_lag_cols = _choice_lag_cols(available_cols)
    choice_lag_corr_cols = _choice_lag_corr_cols(available_cols)
    choice_lag_inc_cols = _choice_lag_inc_cols(available_cols)
    choice_lag_15_cols = _choice_lag_cols(available_cols, max_lags=_NUM_LEGACY_CHOICE_LAGS)
    choice_lag_50_cols = _choice_lag_cols(available_cols, max_lags=_NUM_MEDIUM_CHOICE_LAGS)

    for group in _EMISSION_GROUPS:
        key = group["key"]
        if key == "bias":
            add_scalar(group)
            add_hidden_family(key="bias_hot", label="bias_hot", family_cols=bias_hot_cols)
            continue
        if key == "stim_param":
            add_scalar(group)
            add_hidden_family(
                key="stim_hot",
                label="stim_hot",
                family_cols=stim_cols,
                toggle_cols=[col for col in stim_cols if col != "stim_0"],
            )
            continue
        if key == "abs_ILD":
            add_scalar(group)
            add_hidden_family(
                key="abs_ILD_hot",
                label="abs_ILD_hot",
                family_cols=abs_ild_hot_cols,
            )
            continue
        if key == "at_choice":
            add_scalar(group)
            add_hidden_family(
                key=_CHOICE_LAG_15_ALIAS,
                label="choice lag (15)",
                family_cols=choice_lag_15_cols,
            )
            add_hidden_family(
                key=_CHOICE_LAG_50_ALIAS,
                label="choice lag (50)",
                family_cols=choice_lag_50_cols,
            )
            add_hidden_family(
                key=_CHOICE_LAG_100_ALIAS,
                label="choice lag (100)",
                family_cols=choice_lag_cols,
            )
            add_hidden_family(
                key=_CHOICE_LAG_CORR_ALIAS,
                label="choice lag correct",
                family_cols=choice_lag_corr_cols,
            )
            add_hidden_family(
                key=_CHOICE_LAG_INC_ALIAS,
                label="choice lag incorrect",
                family_cols=choice_lag_inc_cols,
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

    reward_lag_cols = _reward_lag_cols(available_cols)
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

    prev_day_reward_lag_cols = _prev_day_reward_lag_cols(available_cols)
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

    prev_difficulty_lag_cols = _prev_difficulty_lag_cols(available_cols)
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
    if "subject" not in df.columns or "Session" not in df.columns:
        return _max_subject_sessions()
    if isinstance(df, pl.DataFrame):
        return int(
            df.group_by("subject")
            .agg(pl.col("Session").n_unique().alias("n_sessions"))
            .select(pl.col("n_sessions").max())
            .item()
            or 0
        )
    grouped = df.groupby("subject", sort=False)["Session"].nunique()
    return int(grouped.max()) if len(grouped) else 0


def _infer_bias_hot_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = _bias_hot_cols(columns)
    if existing:
        return existing
    max_sessions = _max_sessions_from_df(df)
    return [f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(max_sessions)]


@lru_cache(maxsize=1)
def _max_subject_sessions() -> int:
    dataset_path = get_data_dir() / "alexis_combined.parquet"
    df = pl.read_parquet(dataset_path)
    df = df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))
    return int(
        df.group_by("subject")
        .agg(pl.col("Session").n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


def _stim_param_weight_map() -> dict[int, float]:
    """Return pooled one-hot stimulus weights used to build ``stim_param``."""
    feature_weights = mean_feature_weights_from_fit(_STIM_PARAM_SPEC)
    return {
        int(feat.removeprefix(_STIM_ABS_COL_PREFIX)): weight
        for feat, weight in feature_weights.items()
        if feat.startswith(_STIM_ABS_COL_PREFIX)
        and feat.removeprefix(_STIM_ABS_COL_PREFIX).isdigit()
    }


def _build_stim_param_from_spec(
    part: pd.DataFrame,
    stim_abs_levels: list[int],
    spec: FittedWeightRegressorSpec,
) -> np.ndarray:
    """Return the pooled one-hot stimulus contribution for each trial."""
    required_features = {
        f"{_STIM_ABS_COL_PREFIX}{stim_abs}"
        for stim_abs in stim_abs_levels
        if stim_abs != 0
    }
    source_features = set(resolved_source_features(spec))
    missing = sorted(required_features - source_features)
    if missing:
        raise ValueError(
            "stim_param is missing pooled weights for absolute ILD levels "
            f"{missing}. Available fitted features: {sorted(source_features)}"
        )
    return _weighted_sum_regressor_zero_fill(part, spec)


def _build_stim_param(part: pd.DataFrame, stim_abs_levels: list[int]) -> np.ndarray:
    return _build_stim_param_from_spec(part, stim_abs_levels, _STIM_PARAM_SPEC)


def _weighted_sum_regressor_zero_fill(
    part: pd.DataFrame,
    spec: FittedWeightRegressorSpec,
) -> np.ndarray:
    """Project onto fitted one-hot weights, treating absent fitted columns as zero."""
    source_features = resolved_source_features(spec)
    missing_cols = [col for col in source_features if col not in part.columns]
    if missing_cols:
        part = part.copy()
        for col in missing_cols:
            part[col] = np.float32(0.0)
    return weighted_sum_regressor(part, spec, dtype=np.float32)

PRED_COL = "p_pred"
RESPONSE_MODE = "pm1_or_prob"
BASELINE = 0.5


def prepare_predictions_df(df_pred):
    """Prepare a canonical 2AFC trial-level predictions dataframe."""
    if isinstance(df_pred, pl.DataFrame):
        df = df_pred.clone()
        required = {"stimulus", "response", "performance"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required 2AFC columns: {missing}")

        if "correct_bool" not in df.columns:
            df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        return df.with_columns(
            pl.col("pR").alias("p_pred"),
            pl.when(pl.col("stimulus") == 0)
            .then(pl.col("pL"))
            .otherwise(pl.col("pR"))
            .alias("p_model_correct"),
        )

    df = df_pred.copy()
    required = {"stimulus", "response", "performance"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required 2AFC columns: {missing}")

    if "correct_bool" not in df.columns:
        df["correct_bool"] = df["performance"].astype(bool)
    if "pL" not in df.columns or "pR" not in df.columns:
        raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

    df["p_pred"] = df["pR"]
    df["p_model_correct"] = df.apply(
        lambda row: row["pL"] if row["stimulus"] == 0 else row["pR"],
        axis=1,
    )
    return df


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
    return prepare_simple_regressor_curve(
        trial_df,
        regressor_col=regressor_col,
        pred_col=PRED_COL,
        response_mode=RESPONSE_MODE,
        baseline=BASELINE,
        ylabel=p_right_label(),
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
    df_pd = to_pandas_df(trial_df)
    if regressor_col not in df_pd.columns:
        return None, None

    df_pd, bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=int(n_bins),
        quantiles=None,
    )
    if df_pd is None:
        return None, None
    reg_bin_labels = bin_centers["_reg_bin"].tolist()

    df_pd = attach_response_right_column(df_pd, response_mode=RESPONSE_MODE)
    if df_pd.empty:
        return None, None

    if x_col is None:
        plot_x_col = "_plot_ild"
        df_pd[plot_x_col] = pd.to_numeric(df_pd["ILD"], errors="coerce")
        df_pd[plot_x_col] = np.where(
            np.isclose(df_pd[plot_x_col], -70.0),
            -16.0,
            np.where(np.isclose(df_pd[plot_x_col], 70.0), 16.0, df_pd[plot_x_col]),
        )
        plot_xlabel = "ILD (dB)"
    else:
        if x_col not in df_pd.columns:
            return None, None
        plot_x_col = x_col
        df_pd[plot_x_col] = pd.to_numeric(df_pd[plot_x_col], errors="coerce")
        plot_xlabel = xlabel or display_regressor_name(x_col)
    conds = sorted(df_pd["condition"].dropna().unique()) if "condition" in df_pd.columns else []
    exps = sorted(df_pd["experiment"].dropna().unique()) if "experiment" in df_pd.columns else []
    ild_ticks = sorted(pd.to_numeric(df_pd[plot_x_col], errors="coerce").dropna().unique()) if x_col is None else []

    # Build sparse tick labels: show central labels and remapped extremes.
    if ild_ticks:
        allowed = {-16.0, -8.0, 0.0, 8.0, 16.0}
        ild_tick_labels: list[str] = []
        for t in ild_ticks:
            val = float(t)
            if any(np.isclose(val, a) for a in allowed):
                if np.isclose(val, 0.0):
                    ild_tick_labels.append("0")
                elif val > 0:
                    ild_tick_labels.append(f"+{int(round(val))}")
                else:
                    ild_tick_labels.append(str(int(round(val))))
            else:
                ild_tick_labels.append("")
    else:
        ild_tick_labels = []

    panels: list[dict] = []

    def _subject_summary(*, subgroup_col: str | None = None, subgroup_value=None) -> pd.DataFrame:
        plot_df = df_pd.copy()
        if subgroup_col is not None:
            plot_df = plot_df[plot_df[subgroup_col] == subgroup_value].copy()
        plot_df = plot_df[
            plot_df["_reg_bin"].notna()
            & plot_df[plot_x_col].notna()
            & plot_df["_reg_bin"].isin(reg_bin_labels)
        ].copy()
        if plot_df.empty:
            return pd.DataFrame()
        return (
            plot_df.groupby(["_reg_bin", "subject", plot_x_col], observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=(PRED_COL, "mean"),
                n_trials=("_response_right", "count"),
            )
            .reset_index()
        )

    panels.append(
        {
            "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col=plot_x_col,
                subject_col="subject",
                data_col="_response_right",
                model_col=PRED_COL,
                line_order=reg_bin_labels,
            ),
            "subject_summary": _subject_summary(),
            "meta": {
                "xlabel": plot_xlabel,
                "ylabel": p_right_label(),
                "legend_title": display_regressor_name(regressor_col),
                "baseline": BASELINE,
                "xticks": ild_ticks if x_col is None else None,
                "x_tick_labels": ild_tick_labels if x_col is None else None,
                "x_col": plot_x_col,
                "fit_x_col": plot_x_col,
            }
        }
    )

    for cond in conds:
        panels.append(
            {
                "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col=plot_x_col,
                subject_col="subject",
                data_col="_response_right",
                model_col=PRED_COL,
                    line_order=reg_bin_labels,
                    subgroup_col="condition",
                    subgroup_value=cond,
                ),
                "subject_summary": _subject_summary(
                    subgroup_col="condition",
                    subgroup_value=cond,
                ),
                "meta": {
                    "xlabel": plot_xlabel,
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "xticks": ild_ticks if x_col is None else None,
                    "x_tick_labels": ild_tick_labels if x_col is None else None,
                    "x_col": plot_x_col,
                    "fit_x_col": plot_x_col,
                },
            }
        )

    for exp in exps:
        panels.append(
            {
                "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col=plot_x_col,
                subject_col="subject",
                data_col="_response_right",
                model_col=PRED_COL,
                    line_order=reg_bin_labels,
                    subgroup_col="experiment",
                    subgroup_value=exp,
                ),
                "subject_summary": _subject_summary(
                    subgroup_col="experiment",
                    subgroup_value=exp,
                ),
                "meta": {
                    "xlabel": plot_xlabel,
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "xticks": ild_ticks if x_col is None else None,
                    "x_tick_labels": ild_tick_labels if x_col is None else None,
                    "x_col": plot_x_col,
                    "fit_x_col": plot_x_col,
                },
            }
        )

    return panels, display_regressor_name(regressor_col)


def prepare_right_by_regressor(
    trial_df,
    *,
    regressor_col: str,
    xlabel: str | None = None,
    n_bins: int = 10,
    group_col: str | None = None,
    group_order: Sequence | None = None,
):
    df_pd = to_pandas_df(trial_df)
    required = {regressor_col, "response", PRED_COL, "subject", "ILD"}
    if not required.issubset(df_pd.columns):
        return None, None
    resolved_group_col, resolved_group_order = resolve_grouping(
        df_pd,
        group_col=group_col,
        group_order=group_order,
    )

    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd[PRED_COL] = pd.to_numeric(df_pd[PRED_COL], errors="coerce")
    df_pd["ILD"] = pd.to_numeric(df_pd["ILD"], errors="coerce")
    df_pd = attach_response_right_column(df_pd, response_mode=RESPONSE_MODE)

    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd[PRED_COL])
        & np.isfinite(df_pd["_response_right"])
        & np.isfinite(df_pd["ILD"])
    ].copy()
    if df_pd.empty:
        return None, None

    df_pd, bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=n_bins,
        quantiles=None,
    )
    if df_pd is None:
        return None, None
    bin_order = bin_centers["_reg_bin"].tolist()

    ild_order = sorted(df_pd["ILD"].dropna().unique().tolist())

    if resolved_group_col is None:
        summary = summarize_grouped_panel(
            df_pd,
            line_group_col="ILD",
            x_col="_reg_bin",
            subject_col="subject",
            data_col="_response_right",
            model_col=PRED_COL,
            line_order=ild_order,
            x_order=bin_order,
        )
        line_group_col = "ILD"
        line_order = ild_order
        legend_title = "Signed ILD"
    else:
        df_pd = df_pd[df_pd[resolved_group_col].notna()].copy()
        df_pd = df_pd[df_pd[resolved_group_col].isin(resolved_group_order)].copy()
        subj = (
            df_pd.groupby(["subject", resolved_group_col, "_reg_bin"], observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=(PRED_COL, "mean"),
            )
            .reset_index()
        )
        summary = (
            subj.groupby([resolved_group_col, "_reg_bin"], observed=True)
            .agg(
                md=("data_mean", "mean"),
                sd=("data_mean", "std"),
                nd=("data_mean", "count"),
                mm=("model_mean", "mean"),
            )
            .reset_index()
        )
        summary["sem"] = summary["sd"].fillna(0.0) / np.sqrt(summary["nd"].clip(lower=1))
        summary[resolved_group_col] = pd.Categorical(
            summary[resolved_group_col],
            categories=resolved_group_order,
            ordered=True,
        )
        summary["_reg_bin"] = pd.Categorical(summary["_reg_bin"], categories=bin_order, ordered=True)
        summary = summary.sort_values([resolved_group_col, "_reg_bin"])
        line_group_col = resolved_group_col
        line_order = resolved_group_order
        legend_title = resolved_group_col
    if summary.empty:
        return None, None

    summary = summary.merge(bin_centers, on="_reg_bin", how="left")

    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": p_right_label(),
        "legend_title": legend_title,
        "baseline": BASELINE,
        "line_group_col": line_group_col,
        "line_order": line_order,
        "legend_outside": True,
    }
    return summary, meta





@_register(["two_afc", "2afc"])
class TwoAFCAdapter(TaskAdapter):
    """Adapter for the binary 2-AFC human data (Alexis)."""

    task_key: str    = "2AFC"
    task_label: str  = "2AFC"
    num_classes: int = 2
    data_file: str   = "alexis_combined.parquet"
    # Session-local trial numbers must be sorted within session to match the
    # per-session concatenation order used during fitting.
    sort_col         = ["Session", "Trial"]
    session_col: str = "Session"
    prediction_col: str = PRED_COL
    response_mode: str = RESPONSE_MODE
    psychometric_x_col: str = "ILD"
    psychometric_x_label: str = "ILD (dB)"
    accuracy_x_col: str = "abs_ILD"
    accuracy_x_label: str = "Absolute ILD (dB)"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS
    stim_param_spec: FittedWeightRegressorSpec = _STIM_PARAM_SPEC
    bias_param_spec: FittedWeightRegressorSpec = _BIAS_PARAM_SPEC
    at_choice_param_spec: FittedWeightRegressorSpec = _AT_CHOICE_PARAM_SPEC
    choice_lag_param_spec: FittedWeightRegressorSpec = _CHOICE_LAG_PARAM_SPEC
    choice_lag_param_2_spec: FittedWeightRegressorSpec = _CHOICE_LAG_PARAM_2_SPEC

    # ── state-scoring options ────────────────────────────────────────────────
    # For 2AFC the weight matrix is (K, 1, M) where W[k,0,:] is the
    # right-class logit against the left-class baseline.
    # Modes:
    #   "neg"  – legacy alias kept for saved configs
    #   "abs"  – |W[k, 0, fi]|  (unsigned magnitude)
    #   "pos"  – +W[k, 0, fi]  (raw positive = anti-stimulus tendency)
    # Score per state = mean over listed pairs.
    _SCORING_OPTIONS: dict = {
        "stim_vals (w)": [("stim_vals", "pos")],
        "stim_vals (-w)": [("stim_vals", "neg")],
        "stim_vals (|w|)": [("stim_vals", "abs")],
        "stim_param (w)": [("stim_param", "pos")],
        "stim_param (-w)": [("stim_param", "neg")],
        "stim_param (|w|)": [("stim_param", "abs")],
        "4-state high stim_param + bias": [],
        "4-state signed stim_param + bias": [],
        "4-state high abs_ILD_hot_8 + bias": [],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_param (w)"
    state_scoring_feature: str | None = None
    state_scoring_rule: str = "+"
    state_split_feature: str | None = None
    state_split_rule: str = "+"

    # ── data preparation ────────────────────────────────────────────────────

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))

    def condition_filter_options(self) -> list[str]:
        return ["all"]

    def filter_condition_df(
        self,
        df: pl.DataFrame | pd.DataFrame,
        condition_filter: str = "all",
    ) -> pl.DataFrame | pd.DataFrame:
        selected = str(condition_filter or "all").strip().lower()
        if selected in {"all", ""}:
            return df
        raise ValueError(
            f"Unknown 2AFC condition filter {condition_filter!r}. "
            "Expected one of: all."
        )

    def _build_stim_param(self, part: pd.DataFrame, stim_abs_levels: list[int]) -> np.ndarray:
        return _build_stim_param_from_spec(part, stim_abs_levels, self.stim_param_spec)

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
        include_stim_strength: bool = False,
        include_stim_param: bool = False,
        include_bias_param: bool = False,
        include_at_choice_param: bool = False,
        include_choice_lag_param: bool = False,
        include_choice_lag_param_2: bool = False,
    ) -> pl.DataFrame:
        """Return the Alexis 2AFC feature dataframe owned by this adapter."""
        from glmhmmt.cli.alexis_functions import get_action_trace, make_frames_dm

        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["Session", "Trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)
        subject_half_life = self.choice_half_life(
            str(df_pd["subject"].iloc[0]) if "subject" in df_pd.columns and len(df_pd) else None
        )

        stim_scale = float(df_pd["ILD"].abs().max() or 0.0)
        if stim_scale <= 0:
            stim_scale = 1.0

        stim_set = 6 if df_pd["Experiment"].iloc[0] == "2AFC_6" else 2
        stim_abs_levels = sorted(
            {
                int(abs(v))
                for v in df_pd["ILD"].dropna().astype(int).tolist()
            }
        )
        difficulty_levels = list(_all_stim_abs_levels()) or stim_abs_levels
        max_sessions = _max_sessions_from_df(df_pd)
        session_order = list(dict.fromkeys(df_pd["Session"].tolist()))
        session_to_idx = {session_name: idx for idx, session_name in enumerate(session_order)}
        session_reward_totals = {
            session_name: float(pd.to_numeric(df_session["Hit"], errors="coerce").fillna(0.0).sum())
            for session_name, df_session in df_pd.groupby("Session", sort=False)
        }
        choice_lag_cols = _choice_lag_names()
        reward_lag_cols = _reward_lag_names()
        prev_day_reward_lag_cols = _prev_day_reward_lag_names()
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
        prev_difficulty_lag_cols = _prev_difficulty_lag_names()
        parts = []
        for _, df_session in df_pd.groupby("Session", sort=False):
            part = df_session.copy().reset_index(drop=True)
            session_name = df_session["Session"].iloc[0]
            session_idx = session_to_idx[df_session["Session"].iloc[0]]
            prev_day_reward_lags = {
                lag_col: float(prev_day_reward_lag_maps[lag_col].get(session_name, 0.0))
                for lag_col in prev_day_reward_lag_cols
            }
            prev_day_total_reward = prev_day_reward_lags.get(
                f"{_PREV_DAY_REWARD_LAG_COL_PREFIX}01",
                0.0,
            )
            bias_hot = pd.get_dummies(
                pd.Series(
                    np.full(len(part), session_idx, dtype=np.int32),
                    index=part.index,
                ),
                prefix=_BIAS_HOT_COL_PREFIX.removesuffix("_"),
                prefix_sep="_",
                dtype=np.float32,
            ).reindex(
                columns=[f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(max_sessions)],
                fill_value=0.0,
            )

            stim_hot_cols: dict[str, np.ndarray] = {}
            for stim_abs in stim_abs_levels:
                if stim_abs == 0:
                    stim_col = np.where(part["ILD"] == 0, 1.0, 0.0).astype(np.float32)
                else:
                    stim_col = np.select(
                        [part["ILD"] == stim_abs, part["ILD"] == -stim_abs],
                        [1.0, -1.0],
                        default=0.0,
                    ).astype(np.float32)
                stim_hot_cols[f"{_STIM_ABS_COL_PREFIX}{stim_abs}"] = stim_col
            abs_ild_hot_df = pd.DataFrame(
                {
                    f"{_ABS_ILD_HOT_COL_PREFIX}{stim_abs}": (
                        part["ILD"].abs() == stim_abs
                    ).astype(np.float32)
                    for stim_abs in stim_abs_levels
                },
                index=part.index,
            )
            signed_choice = (2.0 * part["Choice"].fillna(0).astype(np.float32)) - 1.0

            choice_lag_df = pd.DataFrame(
                {
                    lag_col: signed_choice.shift(lag_idx).fillna(0.0).astype(np.float32)
                    for lag_idx, lag_col in enumerate(choice_lag_cols, start=1)
                },
                index=part.index,
            )
            choice_lag_corr_df = pd.DataFrame(
                {
                    f"{_CHOICE_LAG_CORR_COL_PREFIX}{lag_idx:02d}": (
                        part["Hit"].shift(lag_idx).fillna(0.0).astype(np.float32)
                        * signed_choice.shift(lag_idx).fillna(0.0).astype(np.float32)
                    )
                    for lag_idx in range(1, _NUM_CHOICE_LAGS + 1)
                },
                index=part.index,
            )
            choice_lag_inc_df = pd.DataFrame(
                {
                    f"{_CHOICE_LAG_INC_COL_PREFIX}{lag_idx:02d}": (
                        (1.0 - part["Hit"].shift(lag_idx).fillna(0.0).astype(np.float32))
                        * signed_choice.shift(lag_idx).fillna(0.0).astype(np.float32)
                    )
                    for lag_idx in range(1, _NUM_CHOICE_LAGS + 1)
                },
                index=part.index,
            )
            reward_lag_df = pd.DataFrame(
                {
                    lag_col: part["Hit"].shift(lag_idx).fillna(0.0).astype(np.float32)
                    for lag_idx, lag_col in enumerate(reward_lag_cols, start=1)
                },
                index=part.index,
            )
            prev_day_reward_lag_df = pd.DataFrame(
                {
                    lag_col: np.full(len(part), value, dtype=np.float32)
                    for lag_col, value in prev_day_reward_lags.items()
                },
                index=part.index,
            )
            difficulty_hot_df = pd.DataFrame(
                {
                    f"{_DIFFICULTY_HOT_COL_PREFIX}{difficulty_level}": (
                        part["ILD"].abs() == difficulty_level
                    ).astype(np.float32)
                    for difficulty_level in difficulty_levels
                },
                index=part.index,
            )
            prev_difficulty_hot_df = pd.DataFrame(
                {
                    f"{_PREV_DIFFICULTY_HOT_COL_PREFIX}{difficulty_level}": (
                        part["ILD"].abs().shift(1).fillna(0) == difficulty_level
                    ).astype(np.float32)
                    for difficulty_level in difficulty_levels
                },
                index=part.index,
            )
            prev_difficulty_lag_df = pd.DataFrame(
                {
                    lag_col: (
                        part["ILD"].abs().shift(lag_idx).fillna(0) / stim_scale
                    ).astype(np.float32)
                    for lag_idx, lag_col in enumerate(prev_difficulty_lag_cols, start=1)
                },
                index=part.index,
            )
            prev_difficulty_lag_hot_df = pd.DataFrame(
                {
                    f"{_PREV_DIFFICULTY_LAG_HOT_COL_PREFIX}{lag_idx:02d}_{difficulty_level}": (
                        part["ILD"].abs().shift(lag_idx).fillna(0) == difficulty_level
                    ).astype(np.float32)
                    for lag_idx in range(1, _NUM_DIFFICULTY_LAGS + 1)
                    for difficulty_level in difficulty_levels
                },
                index=part.index,
            )
            part = pd.concat(
                [
                    part,
                    pd.DataFrame(
                        {
                            "bias": np.ones(len(part), dtype=np.float32),
                            "stim_vals": (part["ILD"].astype(float) / stim_scale).astype(np.float32),
                            "stim_side": part["Side"].fillna(0).replace({0: -1, 1: 1}).astype(np.float32),
                            "abs_ILD": (part["ILD"].abs().astype(float) / stim_scale).astype(np.float32),
                        },
                        index=part.index,
                    ),
                    bias_hot,
                    pd.DataFrame(stim_hot_cols, index=part.index),
                    abs_ild_hot_df,
                    choice_lag_df,
                    choice_lag_corr_df,
                    choice_lag_inc_df,
                    reward_lag_df,
                    prev_day_reward_lag_df,
                    difficulty_hot_df,
                    prev_difficulty_hot_df,
                    prev_difficulty_lag_df,
                    prev_difficulty_lag_hot_df,
                ],
                axis=1,
            )
            if include_stim_param:
                part[_STIM_PARAM_COL] = self._build_stim_param(part, stim_abs_levels)

            existing_sf_cols = [
                c for c in part.columns if str(c).startswith(_SF_COL_PREFIX)
            ]
            if include_stim_strength and not existing_sf_cols and "Filename" in part.columns:
                stim_strength, _ = make_frames_dm(part, stim_set=stim_set, residuals=True, zscore=False)
                stim_strength = stim_strength.reset_index(drop=True)
                max_val = float(np.nanmax(np.abs(stim_strength.to_numpy()))) if not stim_strength.empty else 0.0
                if max_val > 0:
                    stim_strength = stim_strength / max_val
                stim_strength.columns = [f"{_SF_COL_PREFIX}{col}" for col in stim_strength.columns]
                part = pd.concat([part.reset_index(drop=True), stim_strength], axis=1)

            at_choice, at_error, at_correct, reward_trace = get_action_trace(part)
            if subject_half_life is not None:
                prev_signed_choice = signed_choice.shift(1).fillna(0.0).astype(np.float32)
                at_choice = compute_choice_ewma(
                    prev_signed_choice.to_numpy(dtype=np.float32),
                    half_life=subject_half_life,
                )
            signed_stim = np.sign(part["ILD"].fillna(0).astype(float)).astype(np.float32)
            lagged_choice = signed_choice.shift(1).fillna(0.0).astype(np.float32)
            lagged_stim_side = signed_stim.shift(1).fillna(0.0).astype(np.float32)
            lagged_reward = part["Hit"].shift(1).fillna(0.0).astype(np.float32)
            current_reward = part["Hit"].fillna(0.0).astype(np.float32)
            abs_stim = (part["ILD"].abs().astype(float) / stim_scale).astype(np.float32)
            lagged_abs_stim = abs_stim.shift(1).fillna(0.0).astype(np.float32)

            cumulative_reward = part["Hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            cumulative_reward = pd.Series(
                _zscore_sequence(cumulative_reward.to_numpy()),
                index=part.index,
                dtype=np.float32,
            )
            derived_cols = pd.DataFrame(
                {
                    "trial_index": _session_trial_index(len(part)),
                    "at_choice": np.asarray(at_choice, dtype=np.float32),
                    "at_error": np.asarray(at_error, dtype=np.float32),
                    "at_correct": np.asarray(at_correct, dtype=np.float32),
                    "reward_trace": np.asarray(reward_trace, dtype=np.float32),
                    "prev_choice": lagged_choice,
                    "prev_reward": lagged_reward,
                    "cumulative_reward": cumulative_reward.astype(np.float32),
                    "prev_abs_stim": lagged_abs_stim,
                    "prev_day_total_reward": np.full(len(part), prev_day_total_reward, dtype=np.float32),
                    "prev_day_total_reward_x_cumulative_reward": (
                        prev_day_total_reward * cumulative_reward
                    ).astype(np.float32),
                    "prev_difficulty": -lagged_abs_stim,
                    "wsls": part["Side"].shift(1).fillna(0).replace({0: -1, 1: 1}).astype(np.float32),
                },
                index=part.index,
            )
            part = pd.concat([part, derived_cols], axis=1)
            filtered_cols = pd.DataFrame(
                {
                    "filtered_choice": _ewma_time_series(lagged_choice),
                    "filtered_stim_side": _ewma_time_series(lagged_stim_side),
                    "filtered_reward": _ewma_time_series(lagged_reward),
                    "filtered_difficulty": _ewma_time_series(-lagged_abs_stim),
                    "filtered_bad_stim": _ewma_time_series(signed_stim),
                    "filtered_bad_choice": _ewma_time_series(signed_choice),
                    "filtered_bad_reward": _ewma_time_series(current_reward),
                    "filtered_abs_stim": _ewma_time_series(abs_stim),
                },
                index=part.index,
            )
            part = pd.concat([part, filtered_cols], axis=1)
            part["reward_lag_param"] = transition_weighted_sum(
                part,
                fit_task=self.task_key,
                fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                source_features=reward_lag_cols,
                fallback=np.asarray(part[reward_lag_cols], dtype=np.float32).mean(axis=1),
            )
            difficulty_hot_cols = _difficulty_hot_cols(list(part.columns))
            part["difficulty_hot_param"] = transition_weighted_sum(
                part,
                fit_task=self.task_key,
                fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                source_features=difficulty_hot_cols,
                fallback=part["stim_vals"].abs().to_numpy(dtype=np.float32),
            )
            prev_difficulty_hot_cols = _prev_difficulty_lag_hot_cols(list(part.columns))
            part["prev_difficulty_param"] = transition_weighted_sum(
                part,
                fit_task=self.task_key,
                fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                source_features=prev_difficulty_hot_cols,
                fallback=part["prev_difficulty"].to_numpy(dtype=np.float32),
            )
            if include_bias_param:
                try:
                    bias_param = _weighted_sum_regressor_zero_fill(part, self.bias_param_spec)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"Cannot build {self.bias_param_spec.target_name!r}; pooled fitted weights are unavailable "
                        f"for {self.bias_param_spec.fit_task}/{self.bias_param_spec.fit_model_kind}/{self.bias_param_spec.fit_model_id}."
                    ) from exc
                part = pd.concat(
                    [part, pd.DataFrame({"bias_param": bias_param}, index=part.index)],
                    axis=1,
                )
            if include_at_choice_param:
                try:
                    at_choice_param = _weighted_sum_regressor_zero_fill(part, self.at_choice_param_spec)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"Cannot build {self.at_choice_param_spec.target_name!r}; pooled fitted weights are unavailable "
                        f"for {self.at_choice_param_spec.fit_task}/{self.at_choice_param_spec.fit_model_kind}/{self.at_choice_param_spec.fit_model_id}."
                    ) from exc
                part = pd.concat(
                    [part, pd.DataFrame({"at_choice_param": at_choice_param}, index=part.index)],
                    axis=1,
                )
            if include_choice_lag_param:
                try:
                    choice_lag_param = _weighted_sum_regressor_zero_fill(part, self.choice_lag_param_spec)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"Cannot build {self.choice_lag_param_spec.target_name!r}; pooled fitted weights are unavailable "
                        f"for {self.choice_lag_param_spec.fit_task}/{self.choice_lag_param_spec.fit_model_kind}/{self.choice_lag_param_spec.fit_model_id}."
                    ) from exc
                part = pd.concat(
                    [part, pd.DataFrame({_CHOICE_LAG_PARAM_COL: choice_lag_param}, index=part.index)],
                    axis=1,
                )
            if include_choice_lag_param_2:
                try:
                    choice_lag_param_2 = _weighted_sum_regressor_zero_fill(part, self.choice_lag_param_2_spec)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"Cannot build {self.choice_lag_param_2_spec.target_name!r}; pooled fitted weights are unavailable "
                        f"for {self.choice_lag_param_2_spec.fit_task}/{self.choice_lag_param_2_spec.fit_model_kind}/{self.choice_lag_param_2_spec.fit_model_id}."
                    ) from exc
                part = pd.concat(
                    [part, pd.DataFrame({_CHOICE_LAG_PARAM_2_COL: choice_lag_param_2}, index=part.index)],
                    axis=1,
                )
            parts.append(part)

        return pl.from_pandas(pd.concat(parts, ignore_index=True))

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        """Return the default 2AFC feature dataframe without frame regressors."""
        return self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=False,
            include_stim_param=False,
            include_bias_param=False,
            include_at_choice_param=False,
            include_choice_lag_param=False,
            include_choice_lag_param_2=False,
        )

    def _resolved_emission_cols(
        self,
        feature_df: pl.DataFrame,
        emission_cols: List[str] | None,
    ) -> list[str]:
        requested = emission_cols if emission_cols is not None else self.default_emission_cols(feature_df)
        resolved: list[str] = []
        dynamic_sf_cols = sorted(
            [c for c in feature_df.columns if c.startswith(_SF_COL_PREFIX)],
            key=_sf_sort_key,
        )
        family_aliases = {
            "bias_hot": self.bias_hot_cols(feature_df),
            "choice_lag": self.choice_lag_cols(feature_df),
            "at_choice_lag": self.choice_lag_cols(feature_df),
            _CHOICE_LAG_CORR_ALIAS: _choice_lag_corr_cols(list(feature_df.columns)),
            _CHOICE_LAG_INC_ALIAS: _choice_lag_inc_cols(list(feature_df.columns)),
            _CHOICE_LAG_15_ALIAS: self.choice_lag_cols(
                feature_df,
                max_lags=_NUM_LEGACY_CHOICE_LAGS,
            ),
            _CHOICE_LAG_50_ALIAS: self.choice_lag_cols(
                feature_df,
                max_lags=_NUM_MEDIUM_CHOICE_LAGS,
            ),
            _CHOICE_LAG_100_ALIAS: self.choice_lag_cols(feature_df),
            _AT_CHOICE_LAG_15_ALIAS: self.choice_lag_cols(
                feature_df,
                max_lags=_NUM_LEGACY_CHOICE_LAGS,
            ),
            _AT_CHOICE_LAG_50_ALIAS: self.choice_lag_cols(
                feature_df,
                max_lags=_NUM_MEDIUM_CHOICE_LAGS,
            ),
            _AT_CHOICE_LAG_100_ALIAS: self.choice_lag_cols(feature_df),
            "stim_hot": [col for col in self.stim_abs_cols(feature_df) if col != "stim_0"],
            "stim_one_hot": [col for col in self.stim_abs_cols(feature_df) if col != "stim_0"],
            "abs_ILD_hot": self.abs_ild_hot_cols(feature_df),
            "abs_ild_hot": self.abs_ild_hot_cols(feature_df),
        }
        for col in requested:
            if col == "stim_strength":
                if not dynamic_sf_cols:
                    raise ValueError(
                        "Requested emission col 'stim_strength', but no frame-level "
                        f"'{_SF_COL_PREFIX}*' columns are available for {self.task_key}."
                    )
                resolved.extend(dynamic_sf_cols)
            else:
                resolved.extend(family_aliases.get(col, [col]))
        return list(dict.fromkeys(resolved))

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for the 2AFC task."""
        requested_emission_cols = emission_cols if emission_cols is not None else self.default_emission_cols(df_sub)
        include_stim_strength = "stim_strength" in requested_emission_cols or any(
            col.startswith(_SF_COL_PREFIX) for col in requested_emission_cols
        )
        include_stim_param = _STIM_PARAM_COL in requested_emission_cols
        include_bias_param = "bias_param" in requested_emission_cols
        include_at_choice_param = "at_choice_param" in requested_emission_cols
        include_choice_lag_param = _CHOICE_LAG_PARAM_COL in requested_emission_cols
        include_choice_lag_param_2 = _CHOICE_LAG_PARAM_2_COL in requested_emission_cols
        feature_df = self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=include_stim_strength,
            include_stim_param=include_stim_param,
            include_bias_param=include_bias_param,
            include_at_choice_param=include_at_choice_param,
            include_choice_lag_param=include_choice_lag_param,
            include_choice_lag_param_2=include_choice_lag_param_2,
        )
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
        """Return ``(y, X, U, names)`` for the 2AFC task."""
        requested_emission_cols = list(emission_cols) if emission_cols is not None else self.default_emission_cols(feature_df)
        include_stim_strength = "stim_strength" in requested_emission_cols or any(
            str(col).startswith(_SF_COL_PREFIX) for col in requested_emission_cols
        )
        include_stim_param = _STIM_PARAM_COL in requested_emission_cols
        include_bias_param = "bias_param" in requested_emission_cols
        include_at_choice_param = "at_choice_param" in requested_emission_cols
        include_choice_lag_param = _CHOICE_LAG_PARAM_COL in requested_emission_cols
        include_choice_lag_param_2 = _CHOICE_LAG_PARAM_2_COL in requested_emission_cols
        missing_optional = (
            (include_stim_strength and not any(str(col).startswith(_SF_COL_PREFIX) for col in feature_df.columns))
            or (include_stim_param and _STIM_PARAM_COL not in feature_df.columns)
            or (include_bias_param and "bias_param" not in feature_df.columns)
            or (include_at_choice_param and "at_choice_param" not in feature_df.columns)
            or (include_choice_lag_param and _CHOICE_LAG_PARAM_COL not in feature_df.columns)
            or (include_choice_lag_param_2 and _CHOICE_LAG_PARAM_2_COL not in feature_df.columns)
        )
        if missing_optional:
            raw_cols = [
                col
                for col in [
                    "subject",
                    "Trial",
                    "Side",
                    "Drug",
                    "drug",
                    "Choice",
                    "Hit",
                    "Punish",
                    "Session",
                    "ILD",
                    "Filename",
                    "Experiment",
                    "Task",
                    "P",
                    "AW",
                    "WarmUp",
                    "Date",
                    "condition",
                ]
                if col in feature_df.columns
            ]
            if raw_cols:
                feature_df = feature_df.select(raw_cols)
            feature_df = self._build_feature_df(
                feature_df,
                include_stim_strength=include_stim_strength,
                include_stim_param=include_stim_param,
                include_bias_param=include_bias_param,
                include_at_choice_param=include_at_choice_param,
                include_choice_lag_param=include_choice_lag_param,
                include_choice_lag_param_2=include_choice_lag_param_2,
            )
        ecols = self._resolved_emission_cols(feature_df, emission_cols)
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(feature_df))
        ecols = _drop_unavailable_bias_hot_cols(list(ecols), allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        dynamic_ucols = [
            *_reward_lag_cols(list(feature_df.columns)),
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

        y = jnp.asarray(feature_df["Choice"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {
            "X_cols": list(ecols),
            "U_cols": list(ucols),
        }
        return y, X, U, names

    def cv_balance_labels(self, feature_df: pl.DataFrame):
        """Return signed-ILD balance labels for CV splits."""
        if "ILD" not in feature_df.columns:
            return None
        return feature_df["ILD"].cast(pl.Float64)

    # ── column defaults ─────────────────────────────────────────────────────

    def default_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        default_cols = [
            c
            for c in self.emission_cols
            if c not in {
                "stim_strength",
                _STIM_PARAM_COL,
                "bias_param",
                "at_choice_param",
                _CHOICE_LAG_PARAM_COL,
                _CHOICE_LAG_PARAM_2_COL,
            }
        ]
        if df is not None:
            default_cols.extend(self.sf_cols(df))
        return list(dict.fromkeys(default_cols))

    def default_transition_cols(self) -> List[str]:
        return list(dict.fromkeys(self.transition_cols))

    def available_transition_cols(self) -> List[str]:
        return list(
            dict.fromkeys(
                [
                    *self.default_transition_cols(),
                    *_LEGACY_TRANSITION_COLS,
                    *_reward_lag_names(),
                    *_prev_day_reward_lag_names(),
                    *_all_difficulty_hot_names(),
                    *_all_prev_difficulty_hot_names(),
                    *_prev_difficulty_lag_names(),
                    *_all_prev_difficulty_lag_hot_names(),
                ]
            )
        )

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        available_cols = list(self.emission_cols)
        available_cols.extend(
            [
                _CHOICE_LAG_15_ALIAS,
                _CHOICE_LAG_50_ALIAS,
                _CHOICE_LAG_100_ALIAS,
                _CHOICE_LAG_CORR_ALIAS,
                _CHOICE_LAG_INC_ALIAS,
                _AT_CHOICE_LAG_15_ALIAS,
                _AT_CHOICE_LAG_50_ALIAS,
                _AT_CHOICE_LAG_100_ALIAS,
            ]
        )
        available_cols.extend(self.choice_lag_cols(df))
        available_cols.extend(
            _choice_lag_corr_cols(list(df.columns))
            if df is not None
            else _choice_lag_corr_names()
        )
        available_cols.extend(
            _choice_lag_inc_cols(list(df.columns))
            if df is not None
            else _choice_lag_inc_names()
        )
        if df is not None:
            available_cols.extend(self.sf_cols(df))
            available_cols.extend(self.stim_abs_cols(df))
            available_cols.extend(self.abs_ild_hot_cols(df))
            available_cols.extend(self.bias_hot_cols(df))
        return list(dict.fromkeys(available_cols))

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df: pl.DataFrame | None = None,
    ) -> Dict[str, List[str]]:
        requested_ecols = list(emission_cols) if emission_cols is not None else self.default_emission_cols(df)
        requested_ucols = list(transition_cols) if transition_cols is not None else self.default_transition_cols()

        resolved_ecols: list[str] = []
        family_aliases = {}
        if df is not None:
            family_aliases = {
                "bias_hot": self.bias_hot_cols(df),
                "choice_lag": self.choice_lag_cols(df),
                "at_choice_lag": self.choice_lag_cols(df),
                _CHOICE_LAG_CORR_ALIAS: _choice_lag_corr_cols(list(df.columns)),
                _CHOICE_LAG_INC_ALIAS: _choice_lag_inc_cols(list(df.columns)),
                _CHOICE_LAG_15_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_LEGACY_CHOICE_LAGS,
                ),
                _CHOICE_LAG_50_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_MEDIUM_CHOICE_LAGS,
                ),
                _CHOICE_LAG_100_ALIAS: self.choice_lag_cols(df),
                _AT_CHOICE_LAG_15_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_LEGACY_CHOICE_LAGS,
                ),
                _AT_CHOICE_LAG_50_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_MEDIUM_CHOICE_LAGS,
                ),
                _AT_CHOICE_LAG_100_ALIAS: self.choice_lag_cols(df),
                "stim_hot": [col for col in self.stim_abs_cols(df) if col != "stim_0"],
                "stim_one_hot": [col for col in self.stim_abs_cols(df) if col != "stim_0"],
                "abs_ILD_hot": self.abs_ild_hot_cols(df),
                "abs_ild_hot": self.abs_ild_hot_cols(df),
            }
        else:
            family_aliases = {
                "choice_lag": self.choice_lag_cols(df),
                "at_choice_lag": self.choice_lag_cols(df),
                _CHOICE_LAG_CORR_ALIAS: _choice_lag_corr_names(),
                _CHOICE_LAG_INC_ALIAS: _choice_lag_inc_names(),
                _CHOICE_LAG_15_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_LEGACY_CHOICE_LAGS,
                ),
                _CHOICE_LAG_50_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_MEDIUM_CHOICE_LAGS,
                ),
                _CHOICE_LAG_100_ALIAS: self.choice_lag_cols(df),
                _AT_CHOICE_LAG_15_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_LEGACY_CHOICE_LAGS,
                ),
                _AT_CHOICE_LAG_50_ALIAS: self.choice_lag_cols(
                    df,
                    max_lags=_NUM_MEDIUM_CHOICE_LAGS,
                ),
                _AT_CHOICE_LAG_100_ALIAS: self.choice_lag_cols(df),
            }
        for col in requested_ecols:
            if col == "stim_strength":
                sf_cols = self.sf_cols(df) if df is not None else []
                if not sf_cols:
                    raise ValueError(
                        "Requested emission col 'stim_strength', but no frame-level "
                        f"'{_SF_COL_PREFIX}*' columns are available without rebuilding features."
                    )
                resolved_ecols.extend(sf_cols)
            else:
                resolved_ecols.extend(family_aliases.get(col, [col]))

        allowed_ecols = set(self.available_emission_cols(df))
        resolved_ecols = _drop_unavailable_bias_hot_cols(resolved_ecols, allowed_ecols)
        bad_e = [c for c in resolved_ecols if c not in allowed_ecols]
        dynamic_ucols: list[str] = []
        if df is not None:
            dynamic_ucols.extend(_reward_lag_cols(list(df.columns)))
            dynamic_ucols.extend(_prev_day_reward_lag_cols(list(df.columns)))
            dynamic_ucols.extend(_difficulty_hot_cols(list(df.columns)))
            dynamic_ucols.extend(_prev_difficulty_hot_cols(list(df.columns)))
            dynamic_ucols.extend(_prev_difficulty_lag_cols(list(df.columns)))
            dynamic_ucols.extend(_prev_difficulty_lag_hot_cols(list(df.columns)))
        allowed_ucols = list(dict.fromkeys([*self.available_transition_cols(), *dynamic_ucols]))
        bad_u = [c for c in requested_ucols if c not in allowed_ucols]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {allowed_ucols}"
            )
        return {"X_cols": list(resolved_ecols), "U_cols": list(requested_ucols)}

    def sf_cols(self, df: pl.DataFrame) -> List[str]:
        """Return any stimulus-frame (sf_*) columns present in *df*."""
        return [c for c in df.columns if c.startswith(_SF_COL_PREFIX)]

    def stim_abs_cols(self, df: pl.DataFrame) -> List[str]:
        """Return signed one-hot columns for absolute ILD magnitudes."""
        return _infer_stim_abs_cols_from_df(df)

    def abs_ild_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return unsigned one-hot columns for absolute ILD magnitudes."""
        return _infer_abs_ild_hot_cols_from_df(df)

    def bias_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return subject-local session one-hot columns."""
        return _infer_bias_hot_cols_from_df(df)

    def choice_lag_cols(
        self,
        df: pl.DataFrame | None = None,
        max_lags: int | None = None,
    ) -> List[str]:
        """Return the previous-choice one-hot lag columns."""
        if df is not None:
            existing = _choice_lag_cols(list(df.columns), max_lags=max_lags)
            if existing:
                return existing
        return _choice_lag_names(max_lags=max_lags)

    def weight_family_specs(self, weights_df=None) -> Dict[str, dict]:
        df = to_pandas_df(weights_df) if weights_df is not None else None
        feature_names = [] if df is None or df.empty or "feature" not in df.columns else pd.unique(df["feature"].astype(str)).tolist()
        stim_cols = _stim_abs_cols(feature_names)
        abs_ild_hot_cols = _abs_ild_hot_cols(feature_names)
        choice_cols = _choice_lag_cols(feature_names)
        choice_corr_cols = _choice_lag_corr_cols(feature_names)
        choice_inc_cols = _choice_lag_inc_cols(feature_names)
        choice_15_cols = _choice_lag_cols(feature_names, max_lags=_NUM_LEGACY_CHOICE_LAGS)
        choice_50_cols = _choice_lag_cols(feature_names, max_lags=_NUM_MEDIUM_CHOICE_LAGS)
        bias_cols = _bias_hot_cols(feature_names)
        return {
            "stim_hot": {
                "title": "stim_hot",
                "xlabel": "stimulus level",
                "plot_kind": "box",
                "feature_groups": [(col.removeprefix(_STIM_ABS_COL_PREFIX), [col]) for col in stim_cols],
            },
            "abs_ILD_hot": {
                "title": "abs_ILD_hot",
                "xlabel": "absolute ILD",
                "plot_kind": "box",
                "feature_groups": [(col.removeprefix(_ABS_ILD_HOT_COL_PREFIX), [col]) for col in abs_ild_hot_cols],
            },
            "choice_lag": {
                "title": "choice_lag_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_cols],
            },
            _CHOICE_LAG_15_ALIAS: {
                "title": "choice_lag_01-15",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_15_cols],
            },
            _CHOICE_LAG_50_ALIAS: {
                "title": "choice_lag_01-50",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_50_cols],
            },
            _CHOICE_LAG_100_ALIAS: {
                "title": "choice_lag_01-100",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_cols],
            },
            _CHOICE_LAG_CORR_ALIAS: {
                "title": "choice_lag_corr_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_CORR_COL_PREFIX))), [col]) for col in choice_corr_cols],
            },
            _CHOICE_LAG_INC_ALIAS: {
                "title": "choice_lag_inc_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_INC_COL_PREFIX))), [col]) for col in choice_inc_cols],
            },
            "at_choice_lag": {
                "title": "choice_lag_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_cols],
            },
            _AT_CHOICE_LAG_15_ALIAS: {
                "title": "choice_lag_01-15",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_15_cols],
            },
            _AT_CHOICE_LAG_50_ALIAS: {
                "title": "choice_lag_01-50",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_50_cols],
            },
            _AT_CHOICE_LAG_100_ALIAS: {
                "title": "choice_lag_01-100",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_cols],
            },
            "bias_hot": {
                "title": "bias_hot",
                "xlabel": "Session index",
                "plot_kind": "line",
                "feature_groups": [(col.removeprefix(_BIAS_HOT_COL_PREFIX), [col]) for col in bias_cols],
            },
        }

    def prepare_weight_family_plot(
        self,
        weights_df,
        family_key: str,
        *,
        variant: str | None = None,
    ) -> PreparedWeightFamilyPlot | None:
        del variant
        spec = self.weight_family_specs(weights_df).get(family_key)
        if spec is None:
            return None
        return prepare_grouped_weight_family_plot(
            weights_df,
            feature_groups=spec["feature_groups"],
            title=spec["title"],
            xlabel=spec["xlabel"],
            plot_kind=spec["plot_kind"],
            weight_row_indices=(0,),
        )

    def build_emission_groups(self, available_cols: List[str]) -> list[dict]:
        return _build_emission_groups(list(available_cols))

    def build_transition_groups(self, available_cols: List[str]) -> list[dict]:
        return _build_transition_groups(list(available_cols))

    @property
    def choice_labels(self) -> list[str]:
        return ["Left", "Right"]

    @property
    def probability_columns(self) -> list[str]:
        return ["pL", "pR"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        stim = df["stimulus"].to_numpy().astype(float)
        unique = set(np.unique(stim[~np.isnan(stim)]).tolist())
        if unique.issubset({0.0, 1.0}):
            return stim.astype(int)
        if unique.issubset({-1.0, 1.0}):
            return np.where(stim > 0, 1, 0).astype(int)
        return np.where(stim > 0, 1, np.where(stim < 0, 0, -1)).astype(int)

    # ── column mapping ───────────────────────────────────────────────────────

    @property
    def behavioral_cols(self) -> dict:
        """2AFC column mapping (canonical → actual)."""
        return {
            "trial_idx":   "Trial",
            "trial":       "Trial",
            "session":     "Session",
            "stimulus":    "Side",
            "response":    "Choice",
            "performance": "Hit",
        }

    # ── state labelling ─────────────────────────────────────────────────────

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
            preferred_features=("stim_param", "stim_vals", "abs_ILD", "stim_strength"),
        )
