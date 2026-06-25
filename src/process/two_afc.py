"""Task adapter for the 2AFC (Alexis human) task."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import jax.numpy as jnp
import polars as pl

from dataclasses import replace

from ._choice_tau import compute_choice_ewma, load_subject_choice_half_life
from .design import (
    choice_outcome_lag_frames,
    constant_frame,
    lag_level_prefixed,
    lag_names,
    lagged_level_indicator_frame,
    level_indicator_frame,
    numeric_prefixed,
    session_one_hot_frame,
    shifted_lag_frame,
)
from ._transition_params import transition_weighted_sum
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    mean_feature_weights_from_fit,
    resolved_source_features,
    weighted_sum_regressor,
)
from glmhmmt.tasks import TaskAdapter, _register, build_selector_groups as _build_selector_groups
from glmhmmt.runtime import get_data_dir
from src.process.common import (
    attach_quantile_bin_column,
    attach_response_right_column,
    display_regressor_name,
    label_states_by_regressor,
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
    TWO_AFC_PROFILE,
    prepare_binned_accuracy_figure as _prepare_binned_accuracy_figure,
    prepare_predictions_df as _prepare_predictions_df,
    prepare_right_by_regressor as _prepare_right_by_regressor,
    prepare_right_by_regressor_simple as _prepare_right_by_regressor_simple,
)


# Default experiments to keep (avoids habituation / drug sessions)
_KEEP_EXPERIMENTS = ["2AFC_2", "2AFC_3", "2AFC_4", "2AFC_6"]
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
    "choice_lag_param_correct",
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
_CHOICE_LAG_PARAM_CORRECT_COL = "choice_lag_param_correct"

ParamWeightSource = tuple[str, str]

PARAM_WEIGHT_SOURCE: ParamWeightSource | str = ("2AFC", "one hot")

PARAM_WEIGHT_SOURCE_BY_SELECTED_PARAM: dict[str, ParamWeightSource | str] = {
    _CHOICE_LAG_PARAM_CORRECT_COL: ("2AFC", "one hot2"),
}

PARAM_WEIGHT_SOURCE_BY_PARAM: dict[str, ParamWeightSource | str] = {}



def _fitted_weight_spec(
    *,
    target_name: str,
    source_feature_prefixes: tuple[str, ...] = (),
    source_features: tuple[str, ...] = (),
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
    exclude_features: tuple[str, ...] = (),
    excluded_subjects: tuple[str, ...] = (),
    sign: float = 1.0,
) -> FittedWeightRegressorSpec:
    return FittedWeightRegressorSpec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_kind="glm",
        fit_model_id=fit_model_id,
        arrays_suffix="glm_arrays.npz",
        source_features=source_features,
        source_feature_prefixes=source_feature_prefixes,
        exclude_features=exclude_features,
        excluded_subjects=excluded_subjects,
        sign=sign,
    )


def _stim_param_spec(
    *,
    target_name: str = _STIM_PARAM_COL,
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
) -> FittedWeightRegressorSpec:
    return _fitted_weight_spec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_id=fit_model_id,
        source_feature_prefixes=("stim_",),
        exclude_features=("stim_0",),
        excluded_subjects=("325", "325.0"),
    )


def _bias_param_spec(
    *,
    target_name: str = "bias_param",
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
) -> FittedWeightRegressorSpec:
    return _fitted_weight_spec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_id=fit_model_id,
        source_feature_prefixes=("bias_",),
    )


def _at_choice_param_spec(
    *,
    target_name: str = "at_choice_param",
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
) -> FittedWeightRegressorSpec:
    return _fitted_weight_spec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_id=fit_model_id,
        source_features=tuple(lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)),
    )


def _choice_lag_param_spec(
    *,
    target_name: str = _CHOICE_LAG_PARAM_COL,
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
) -> FittedWeightRegressorSpec:
    return _fitted_weight_spec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_id=fit_model_id,
        source_features=tuple(lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)),
    )


def _choice_lag_param_2_spec(
    *,
    target_name: str = _CHOICE_LAG_PARAM_2_COL,
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
) -> FittedWeightRegressorSpec:
    return _fitted_weight_spec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_id=fit_model_id,
        source_features=tuple(lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)[1:]),
    )


def _choice_lag_param_correct_spec(
    *,
    target_name: str = _CHOICE_LAG_PARAM_CORRECT_COL,
    fit_task: str = "2AFC",
    fit_model_id: str = _RAW_PARAM_MODEL_ID,
) -> FittedWeightRegressorSpec:
    return _fitted_weight_spec(
        target_name=target_name,
        fit_task=fit_task,
        fit_model_id=fit_model_id,
        source_features=tuple(lag_names("choice_lag_corr_", _NUM_LEGACY_CHOICE_LAGS)),
    )


_STIM_PARAM_SPEC = _stim_param_spec()
_BIAS_PARAM_SPEC = _bias_param_spec()
_AT_CHOICE_PARAM_SPEC = _at_choice_param_spec()
_CHOICE_LAG_PARAM_SPEC = _choice_lag_param_spec()
_CHOICE_LAG_PARAM_2_SPEC = _choice_lag_param_2_spec()
_CHOICE_LAG_PARAM_CORRECT_SPEC = _choice_lag_param_correct_spec()


def _coerce_param_weight_source(source: ParamWeightSource | str) -> ParamWeightSource:
    if isinstance(source, str):
        if source == "2AFC":
            return (source, "one hot")
        raise ValueError(
            "String PARAM_WEIGHT_SOURCE values must be '2AFC'. "
            "Use a (fit_task, fit_model_id) tuple for a specific GLM model."
        )

    if len(source) != 2:
        raise ValueError("Param weight sources must be (fit_task, fit_model_id) tuples.")

    fit_task, fit_model_id = source
    return (str(fit_task), str(fit_model_id))


def _param_weight_source_for_request(requested: set[str]) -> ParamWeightSource:
    for param_col, source in PARAM_WEIGHT_SOURCE_BY_SELECTED_PARAM.items():
        if param_col in requested:
            return _coerce_param_weight_source(source)
    return _coerce_param_weight_source(PARAM_WEIGHT_SOURCE)


def _with_param_weight_source(
    spec: FittedWeightRegressorSpec,
    source: ParamWeightSource | str,
) -> FittedWeightRegressorSpec:
    fit_task, fit_model_id = _coerce_param_weight_source(source)
    return replace(spec, fit_task=fit_task, fit_model_id=fit_model_id)

def _source_for_param(
    target_name: str,
    selected_source: ParamWeightSource,
) -> ParamWeightSource:
    return _coerce_param_weight_source(
        PARAM_WEIGHT_SOURCE_BY_PARAM.get(target_name, selected_source)
    )


def _standard_param_specs_for_request(
    adapter: "TwoAFCAdapter",
    requested: set[str],
) -> dict[str, FittedWeightRegressorSpec]:
    selected_source = _param_weight_source_for_request(requested)

    base_specs = {
        _STIM_PARAM_COL: adapter.stim_param_spec,
        "bias_param": adapter.bias_param_spec,
        "at_choice_param": adapter.at_choice_param_spec,
        _CHOICE_LAG_PARAM_COL: adapter.choice_lag_param_spec,
        _CHOICE_LAG_PARAM_2_COL: adapter.choice_lag_param_2_spec,
        _CHOICE_LAG_PARAM_CORRECT_COL: adapter.choice_lag_param_correct_spec,
    }

    return {
        target_name: _with_param_weight_source(
            spec,
            _source_for_param(target_name, selected_source),
        )
        for target_name, spec in base_specs.items()
    }

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
    "choice_lag_param_correct": r"$\mathrm{A}_t^{\mathrm{choice,correct,param}}$",
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
    {"key": "choice_lag_param_correct", "label": "choice lag correct param", "members": {"N": "choice_lag_param_correct"}},
    {"key": "at_error", "label": "action (error)", "members": {"N": "at_error"}},
    {"key": "at_correct", "label": "action (correct)", "members": {"N": "at_correct"}},
    {"key": "reward_trace", "label": "reward trace", "members": {"N": "reward_trace"}},
    {"key": "prev_choice", "label": "prev choice", "members": {"N": "prev_choice"}},
    {"key": "wsls", "label": "WSLS", "members": {"N": "wsls"}},
    {"key": "prev_reward", "label": "prev reward", "members": {"N": "prev_reward"}},
    {"key": "cumulative_reward", "label": "cumulative reward", "members": {"N": "cumulative_reward"}},
    {"key": "prev_abs_stim", "label": "prev abs stim", "members": {"N": "prev_abs_stim"}},
]


def _ewma_time_series(values: Sequence[float], period: float = _FILTERED_REGRESSOR_TAU) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_np.size == 0:
        return values_np.copy()
    ewma = pd.DataFrame(data=values_np).ewm(span=float(period)).mean()
    return ewma.iloc[:, 0].to_numpy(dtype=np.float32)


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


def _infer_stim_abs_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = numeric_prefixed(columns, "stim_")
    if existing:
        return existing
    fallback = [f"stim_{level}" for level in _all_stim_abs_levels()]
    if "ILD" not in columns:
        return fallback
    ild_series = df["ILD"].drop_nulls() if isinstance(df, pl.DataFrame) else df["ILD"].dropna()
    stim_abs_levels = sorted({int(abs(v)) for v in ild_series.to_list()})
    inferred = [f"stim_{stim_abs}" for stim_abs in stim_abs_levels]
    return list(dict.fromkeys([*inferred, *fallback]))


def _infer_abs_ild_hot_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = numeric_prefixed(columns, "abs_ILD_hot_")
    if existing:
        return existing
    fallback = [f"abs_ILD_hot_{level}" for level in _all_stim_abs_levels()]
    if "ILD" not in columns:
        return fallback
    ild_series = df["ILD"].drop_nulls() if isinstance(df, pl.DataFrame) else df["ILD"].dropna()
    stim_abs_levels = sorted({int(abs(v)) for v in ild_series.to_list()})
    inferred = [f"abs_ILD_hot_{stim_abs}" for stim_abs in stim_abs_levels]
    return list(dict.fromkeys([*inferred, *fallback]))


def _difficulty_hot_names(levels: Sequence[int]) -> list[str]:
    return [f"difficulty_hot_{int(level)}" for level in levels]


def _prev_difficulty_hot_names(levels: Sequence[int]) -> list[str]:
    return [f"prev_difficulty_hot_{int(level)}" for level in levels]


def _prev_difficulty_lag_hot_names(levels: Sequence[int]) -> list[str]:
    return [
        f"prev_difficulty_lag_hot_{lag_idx:02d}_{int(level)}"
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

    stim_cols = numeric_prefixed(available_cols, "stim_")
    abs_ild_hot_cols = numeric_prefixed(available_cols, "abs_ILD_hot_")
    bias_hot_cols = numeric_prefixed(available_cols, "bias_")
    choice_lag_cols = numeric_prefixed(available_cols, "choice_lag_")
    choice_lag_corr_cols = numeric_prefixed(available_cols, "choice_lag_corr_")
    choice_lag_inc_cols = numeric_prefixed(available_cols, "choice_lag_inc_")
    choice_lag_15_cols = numeric_prefixed(available_cols, "choice_lag_", max_count=_NUM_LEGACY_CHOICE_LAGS)
    choice_lag_50_cols = numeric_prefixed(available_cols, "choice_lag_", max_count=_NUM_MEDIUM_CHOICE_LAGS)

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

    difficulty_hot_cols = numeric_prefixed(available_cols, "difficulty_hot_")
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

    prev_difficulty_hot_cols = numeric_prefixed(available_cols, "prev_difficulty_hot_")
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

    prev_difficulty_lag_hot_cols = lag_level_prefixed(available_cols, "prev_difficulty_lag_hot_")
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
    existing = numeric_prefixed(columns, "bias_")
    if existing:
        return existing
    max_sessions = _max_sessions_from_df(df)
    return [f"bias_{idx}" for idx in range(max_sessions)]


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
        int(feat.removeprefix("stim_")): weight
        for feat, weight in feature_weights.items()
        if feat.startswith("stim_")
        and feat.removeprefix("stim_").isdigit()
    }


def _build_stim_param_from_spec(
    part: pd.DataFrame,
    stim_abs_levels: list[int],
    spec: FittedWeightRegressorSpec,
) -> np.ndarray:
    """Return the pooled one-hot stimulus contribution for each trial."""
    required_features = {
        f"stim_{stim_abs}"
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


def _safe_build_stim_param_from_spec(
    part: pd.DataFrame,
    stim_abs_levels: list[int],
    spec: FittedWeightRegressorSpec,
) -> np.ndarray:
    try:
        return _build_stim_param_from_spec(part, stim_abs_levels, spec)
    except (FileNotFoundError, ValueError):
        return np.zeros(len(part), dtype=np.float32)


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


def _safe_weighted_sum_regressor(
    part: pd.DataFrame,
    spec: FittedWeightRegressorSpec,
) -> np.ndarray:
    try:
        return _weighted_sum_regressor_zero_fill(part, spec)
    except (FileNotFoundError, ValueError):
        return np.zeros(len(part), dtype=np.float32)

PRED_COL = "p_pred"
RESPONSE_MODE = "pm1_or_prob"
BASELINE = 0.5


def prepare_predictions_df(df_pred):
    """Compatibility wrapper for the shared 2AFC trial payload builder."""
    return _prepare_predictions_df(df_pred, profile=TWO_AFC_PROFILE)


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
        profile=TWO_AFC_PROFILE,
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
        profile=TWO_AFC_PROFILE,
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
        profile=TWO_AFC_PROFILE,
        regressor_col=regressor_col,
        xlabel=xlabel,
        n_bins=n_bins,
        group_col=group_col,
        group_order=group_order,
    )





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
    choice_lag_param_correct_spec: FittedWeightRegressorSpec = _CHOICE_LAG_PARAM_CORRECT_SPEC

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
        param_specs: dict[str, FittedWeightRegressorSpec] | None = None,
        strict_param_cols: set[str] | None = None,
    ) -> pl.DataFrame:
        """Return the Alexis 2AFC feature dataframe owned by this adapter."""
        from glmhmmt.cli.alexis_functions import get_action_trace, make_frames_dm

        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["Session", "Trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)

        param_specs = param_specs or _standard_param_specs_for_request(self, set())
        strict_param_cols = strict_param_cols or set()
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
                "prev_day_total_reward_lag_01",
                0.0,
            )
            bias_hot = session_one_hot_frame(session_idx, max_sessions, part.index)

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
                stim_hot_cols[f"stim_{stim_abs}"] = stim_col
            abs_ild_hot_df = pd.DataFrame(
                {
                    f"abs_ILD_hot_{stim_abs}": (
                        part["ILD"].abs() == stim_abs
                    ).astype(np.float32)
                    for stim_abs in stim_abs_levels
                },
                index=part.index,
            )
            signed_choice = (2.0 * part["Choice"].fillna(0).astype(np.float32)) - 1.0

            choice_lag_df = shifted_lag_frame(signed_choice, choice_lag_cols, part.index)
            choice_lag_corr_df, choice_lag_inc_df = choice_outcome_lag_frames(
                signed_choice,
                part["Hit"],
                n_lags=15,
                index=part.index,
            )
            reward_lag_df = shifted_lag_frame(part["Hit"], reward_lag_cols, part.index)
            prev_day_reward_lag_df = constant_frame(prev_day_reward_lags, part.index)
            difficulty_hot_df = level_indicator_frame(
                part["ILD"].abs(),
                difficulty_levels,
                prefix="difficulty_hot_",
                index=part.index,
            )
            prev_difficulty_hot_df = level_indicator_frame(
                part["ILD"].abs().shift(1).fillna(0),
                difficulty_levels,
                prefix="prev_difficulty_hot_",
                index=part.index,
            )
            prev_difficulty_lag_df = shifted_lag_frame(
                part["ILD"].abs() / stim_scale,
                prev_difficulty_lag_cols,
                part.index,
            )
            prev_difficulty_lag_hot_df = lagged_level_indicator_frame(
                part["ILD"].abs(),
                difficulty_levels,
                prefix="prev_difficulty_lag_hot_",
                n_lags=_NUM_DIFFICULTY_LAGS,
                index=part.index,
            )
            feature_frames = [
                (
                    "core",
                    pd.DataFrame(
                        {
                            "bias": np.ones(len(part), dtype=np.float32),
                            "stim_vals": (part["ILD"].astype(float) / stim_scale).astype(np.float32),
                            "stim_side": part["Side"].fillna(0).replace({0: -1, 1: 1}).astype(np.float32),
                            "abs_ILD": (part["ILD"].abs().astype(float) / stim_scale).astype(np.float32),
                        },
                        index=part.index,
                    ),
                ),
                ("bias_hot", bias_hot),
                ("stim_hot", pd.DataFrame(stim_hot_cols, index=part.index)),
                ("abs_ILD_hot", abs_ild_hot_df),
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
            # Raw one-hot fits create the weights used by stim_param, so only
            # require source weights when stim_param itself is requested.
            stim_param_builder = (
                _build_stim_param_from_spec
                if _STIM_PARAM_COL in strict_param_cols
                else _safe_build_stim_param_from_spec
            )
            part[_STIM_PARAM_COL] = stim_param_builder(
                part,
                stim_abs_levels,
                param_specs[_STIM_PARAM_COL],
            )

            existing_sf_cols = [c for c in part.columns if str(c).startswith("sf_")]
            if include_stim_strength and not existing_sf_cols and "Filename" in part.columns:
                stim_strength, _ = make_frames_dm(part, stim_set=stim_set, residuals=True, zscore=False)
                stim_strength = stim_strength.reset_index(drop=True)
                max_val = float(np.nanmax(np.abs(stim_strength.to_numpy()))) if not stim_strength.empty else 0.0
                if max_val > 0:
                    stim_strength = stim_strength / max_val
                stim_strength.columns = [f"sf_{col}" for col in stim_strength.columns]
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
            difficulty_hot_cols = numeric_prefixed(list(part.columns), "difficulty_hot_")
            part["difficulty_hot_param"] = transition_weighted_sum(
                part,
                fit_task=self.task_key,
                fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                source_features=difficulty_hot_cols,
                fallback=part["stim_vals"].abs().to_numpy(dtype=np.float32),
            )
            prev_difficulty_hot_cols = lag_level_prefixed(list(part.columns), "prev_difficulty_lag_hot_")
            part["prev_difficulty_param"] = transition_weighted_sum(
                part,
                fit_task=self.task_key,
                fit_model_id=_TRANSITION_PARAM_MODEL_ID,
                source_features=prev_difficulty_hot_cols,
                fallback=part["prev_difficulty"].to_numpy(dtype=np.float32),
            )
            fitted_summary_cols = pd.DataFrame(
                {
                    "bias_param": _safe_weighted_sum_regressor(
                        part,
                        param_specs["bias_param"],
                    ),
                    "at_choice_param": _safe_weighted_sum_regressor(
                        part,
                        param_specs["at_choice_param"],
                    ),
                    _CHOICE_LAG_PARAM_COL: _safe_weighted_sum_regressor(
                        part,
                        param_specs[_CHOICE_LAG_PARAM_COL],
                    ),
                    _CHOICE_LAG_PARAM_2_COL: _safe_weighted_sum_regressor(
                        part,
                        param_specs[_CHOICE_LAG_PARAM_2_COL],
                    ),
                    _CHOICE_LAG_PARAM_CORRECT_COL: _safe_weighted_sum_regressor(
                        part,
                        param_specs[_CHOICE_LAG_PARAM_CORRECT_COL],
                    ),
                },
                index=part.index,
            )
            part = pd.concat([part, fitted_summary_cols], axis=1)
            parts.append(part)

        return pl.from_pandas(pd.concat(parts, ignore_index=True))

    def build_feature_df(self,df_sub: pl.DataFrame,tau: float = 50.0,emission_cols: List[str] | None = None, transition_cols: List[str] | None = None) -> pl.DataFrame:
        requested = list(emission_cols) if emission_cols is not None else []
        return self._build_feature_df(
            df_sub,
            tau=tau,
            param_specs=_standard_param_specs_for_request(self, set(requested)),
            strict_param_cols=set(requested),
        )
    
    def _resolved_emission_cols(
        self,
        feature_df: pl.DataFrame,
        emission_cols: List[str] | None,
    ) -> list[str]:
        requested = emission_cols if emission_cols is not None else self.default_emission_cols(feature_df)
        resolved: list[str] = []
        dynamic_sf_cols = numeric_prefixed(list(feature_df.columns), "sf_")
        choice_lag_cols = numeric_prefixed(list(feature_df.columns), "choice_lag_")
        stim_abs_cols = _infer_stim_abs_cols_from_df(feature_df)
        abs_ild_hot_cols = _infer_abs_ild_hot_cols_from_df(feature_df)
        family_aliases = {
            "bias_hot": _infer_bias_hot_cols_from_df(feature_df),
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
            "stim_hot": [col for col in stim_abs_cols if col != "stim_0"],
            "stim_one_hot": [col for col in stim_abs_cols if col != "stim_0"],
            "abs_ILD_hot": abs_ild_hot_cols,
            "abs_ild_hot": abs_ild_hot_cols,
        }
        for col in requested:
            if col == "stim_strength":
                if not dynamic_sf_cols:
                    raise ValueError(
                        "Requested emission col 'stim_strength', but no frame-level "
                        f"'sf_*' columns are available for {self.task_key}."
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
        requested = list(emission_cols) if emission_cols is not None else self.default_emission_cols()
        requested_set = set(requested)
        param_specs = _standard_param_specs_for_request(self, requested_set)
        include_stim_strength = "stim_strength" in requested or any(
            str(col).startswith("sf_") for col in requested
        )
        feature_df = self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=include_stim_strength,
            param_specs=param_specs,
            strict_param_cols=requested_set,
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
        ecols = self._resolved_emission_cols(feature_df, emission_cols)
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(feature_df))
        ecols = _drop_unavailable_bias_hot_cols(list(ecols), allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        dynamic_ucols = [
            *numeric_prefixed(list(feature_df.columns), "reward_lag_"),
            *numeric_prefixed(list(feature_df.columns), "difficulty_hot_"),
            *numeric_prefixed(list(feature_df.columns), "prev_difficulty_hot_"),
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
                "Build features through load_subject or build_feature_df before calling build_design_matrices."
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
        default_cols = [col for col in self.emission_cols if col != "stim_strength"]
        if df is not None:
            columns = list(df.columns)
            default_cols.extend(_infer_bias_hot_cols_from_df(df))
            default_cols.extend(_infer_stim_abs_cols_from_df(df))
            default_cols.extend(_infer_abs_ild_hot_cols_from_df(df))
            default_cols.extend(
                numeric_prefixed(columns, "choice_lag_")
                or lag_names("choice_lag_", _NUM_LEGACY_CHOICE_LAGS)
            )
            default_cols.extend(
                numeric_prefixed(columns, "choice_lag_corr_")
                or lag_names("choice_lag_corr_", _NUM_LEGACY_CHOICE_LAGS)
            )
            default_cols.extend(
                numeric_prefixed(columns, "choice_lag_inc_")
                or lag_names("choice_lag_inc_", _NUM_LEGACY_CHOICE_LAGS)
            )
            default_cols.extend([c for c in columns if c.startswith("sf_")])
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
            available_cols.extend([c for c in df.columns if c.startswith("sf_")])
            available_cols.extend(_infer_stim_abs_cols_from_df(df))
            available_cols.extend(_infer_abs_ild_hot_cols_from_df(df))
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

        resolved_ecols: list[str] = []
        family_aliases = {}
        if df is not None:
            columns = list(df.columns)
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
            stim_abs_cols = _infer_stim_abs_cols_from_df(df)
            abs_ild_hot_cols = _infer_abs_ild_hot_cols_from_df(df)
            family_aliases = {
                "bias_hot": _infer_bias_hot_cols_from_df(df),
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
                "stim_hot": [col for col in stim_abs_cols if col != "stim_0"],
                "stim_one_hot": [col for col in stim_abs_cols if col != "stim_0"],
                "abs_ILD_hot": abs_ild_hot_cols,
                "abs_ild_hot": abs_ild_hot_cols,
            }
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
            if col == "stim_strength":
                sf_cols = [c for c in df.columns if c.startswith("sf_")] if df is not None else []
                if not sf_cols:
                    raise ValueError(
                        "Requested emission col 'stim_strength', but no frame-level "
                        f"'sf_*' columns are available without rebuilding features."
                    )
                resolved_ecols.extend(sf_cols)
            else:
                resolved_ecols.extend(family_aliases.get(col, [col]))

        allowed_ecols = set(self.available_emission_cols(df))
        resolved_ecols = _drop_unavailable_bias_hot_cols(resolved_ecols, allowed_ecols)
        bad_e = [c for c in resolved_ecols if c not in allowed_ecols]
        dynamic_ucols: list[str] = []
        if df is not None:
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "reward_lag_"))
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "prev_day_total_reward_lag_"))
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "difficulty_hot_"))
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "prev_difficulty_hot_"))
            dynamic_ucols.extend(numeric_prefixed(list(df.columns), "prev_difficulty_lag_"))
            dynamic_ucols.extend(lag_level_prefixed(list(df.columns), "prev_difficulty_lag_hot_"))
        allowed_ucols = list(dict.fromkeys([*self.available_transition_cols(), *dynamic_ucols]))
        bad_u = [c for c in requested_ucols if c not in allowed_ucols]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {allowed_ucols}"
            )
        return {"X_cols": list(dict.fromkeys(resolved_ecols)), "U_cols": list(requested_ucols)}

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
