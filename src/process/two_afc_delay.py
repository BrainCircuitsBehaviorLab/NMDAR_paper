"""Task adapter for the Tiffany 2AFC delay task."""
from __future__ import annotations

from functools import lru_cache
import types
from typing import Any, Dict, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd

from ._choice_tau import compute_choice_ewma, load_subject_choice_half_life
from glmhmmt.cli.alexis_functions import get_action_trace
from glmhmmt.runtime import get_data_dir
from glmhmmt.tasks import TaskAdapter, _register, resolve_plots_module
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    mean_feature_weights_from_fit,
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

from src.process.common import (
    PreparedWeightFamilyPlot,
    prepare_grouped_weight_family_plot,
    to_pandas_df,
)

_DELAY_HOT_COL_PREFIX = "delay_"
_BIAS_HOT_COL_PREFIX = "bias_"
_CHOICE_LAG_COL_PREFIX = "choice_lag_"
_NUM_CHOICE_LAGS = 15
_RAW_PARAM_MODEL_ID = "one hot"
EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "stim",
    "delay",
    "delay_param",
    "stim_x_delay",
    "stim_x_delay_hot",
    "stim_x_delay_param",
    "at_choice",
    "choice_lag_param",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "wsls",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
    "after_correct",
    "repeat",
    "repeat_choice_side",
]
TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
    "delay",
]

_BIAS_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="bias_param",
    fit_task="2AFC_delay",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_BIAS_HOT_COL_PREFIX,),
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
    source_feature_prefixes=(_CHOICE_LAG_COL_PREFIX,),
)

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim": r"$\mathrm{Stimulus}$",
    "delay": r"$\mathrm{Delay}$",
    "delay_param": r"$\mathrm{Delay}_{\mathrm{param}}$",
    "stim_x_delay": r"$\mathrm{Stimulus}\times\mathrm{Delay}$",
    "bias": r"$\mid\mathrm{bias}\mid$",
    "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
    "at_choice": r"$\mathrm{A}_t^{\mathrm{choice}}$",
    "choice_lag_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    "at_error": r"$\mathrm{A}_t^{\mathrm{error}}$",
    "at_correct": r"$\mathrm{A}_t^{\mathrm{correct}}$",
    "reward_trace": r"$\mathrm{Reward}_{\mathrm{trace}}$",
    "prev_choice": r"$\mathrm{PrevChoice}$",
    "prev_reward": r"$\mathrm{PrevReward}$",
    "prev_abs_stim": r"$|\mathrm{PrevStim}|$",
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
    {"key": "delay", "label": "delay", "members": {"N": "delay"}},
    {"key": "delay_param", "label": "delay param", "members": {"N": "delay_param"}},
    {"key": "stim_x_delay", "label": "stim×delay", "members": {"N": "stim_x_delay"}},
    {"key": "stim_x_delay_hot", "label": "stim×delay one-hot", "members": {"N": "stim_x_delay_hot"}},
    {"key": "at_choice", "label": "action (choice)", "members": {"N": "at_choice"}},
    {"key": "choice_lag_param", "label": "choice lag param", "members": {"N": "choice_lag_param"}},
    {"key": "at_error", "label": "action (error)", "members": {"N": "at_error"}},
    {"key": "at_correct", "label": "action (correct)", "members": {"N": "at_correct"}},
    {"key": "reward_trace", "label": "reward trace", "members": {"N": "reward_trace"}},
    {"key": "prev_choice", "label": "prev choice", "members": {"N": "prev_choice"}},
    {"key": "wsls", "label": "WSLS", "members": {"N": "wsls"}},
    {"key": "prev_reward", "label": "prev reward", "members": {"N": "prev_reward"}},
    {"key": "cumulative_reward", "label": "cumulative reward", "members": {"N": "cumulative_reward"}},
    {"key": "prev_abs_stim", "label": "prev abs stim", "members": {"N": "prev_abs_stim"}},
    {"key": "after_correct", "label": "after correct", "members": {"N": "after_correct"}},
    {"key": "repeat", "label": "repeat", "members": {"N": "repeat"}},
    {"key": "repeat_choice_side", "label": "repeat side", "members": {"N": "repeat_choice_side"}},
]


def _safe_weighted_sum_regressor(
    part,
    spec: FittedWeightRegressorSpec,
) -> np.ndarray | None:
    try:
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
    if np.isclose(delay_value, 0.1):
        return "0"
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


def _bias_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_BIAS_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)

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


def _choice_lag_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_CHOICE_LAG_COL_PREFIX)
            and col.removeprefix(_CHOICE_LAG_COL_PREFIX).isdigit()
        ],
        key=_choice_lag_sort_key,
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
    bias_hot_cols = _bias_hot_cols(available_cols)
    choice_lag_cols = _choice_lag_cols(available_cols)
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
            add_hidden_family(key="at_choice_lag", label="choice_lag", family_cols=choice_lag_cols)
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
    existing = _bias_hot_cols(columns)
    if existing:
        return existing
    max_sessions = _max_sessions_from_df(df)
    return [f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(max_sessions)]


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


def _choice_lag_names() -> list[str]:
    return [f"{_CHOICE_LAG_COL_PREFIX}{idx:02d}" for idx in range(1, _NUM_CHOICE_LAGS + 1)]



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

    This makes class 0 the explicit binary logit and therefore Left the
    softmax reference class for 2AFC-delay fits.
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

PRED_COL = "p_pred"
RESPONSE_MODE = "pm1_or_prob"
BASELINE = 0.5

SIGNED_DELAY_ORDER = ["0L", "-1", "-3", "-10", "10", "3", "1", "0R"]
SIGNED_DELAY_LABELS = ["0", "-1", "-3", "-10", "10", "3", "1", "0"]


def format_delay_tick(value: float) -> str:
    value = 0.0 if np.isclose(float(value), 0.1) else float(value)
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def delay_ticks_from_df(df: pd.DataFrame, *, delay_col: str = "delay") -> tuple[list[float], list[str]]:
    if delay_col not in df.columns:
        return [], []
    ticks = sorted(pd.to_numeric(df[delay_col], errors="coerce").dropna().astype(float).unique())
    return ticks, [format_delay_tick(value) for value in ticks]


def signed_delay_label(value: str) -> str:
    if value in {"0L", "0R"}:
        return "0"
    return format_delay_tick(float(value))


def signed_delay_order_and_labels(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    values = [str(value) for value in df["_signed_delay_cat"].dropna().unique()]
    preferred = [value for value in SIGNED_DELAY_ORDER if value in set(values)]
    extras = sorted(
        (value for value in values if value not in set(preferred)),
        key=lambda value: float(value) if value not in {"0L", "0R"} else 0.0,
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
    """Prepare a canonical 2AFC-delay trial predictions dataframe."""
    if isinstance(df_pred, pl.DataFrame):
        df = df_pred.clone()
        required = {"stimulus", "response", "performance"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required 2AFC-delay columns: {missing}")

        if "correct_bool" not in df.columns:
            df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        df = df.with_columns(
            pl.col("pR").alias("p_pred"),
            _correct_prob_expr(pl).alias("p_model_correct"),
        )
        if "delays" in df.columns:
            return df.with_columns(pl.col("delays").cast(pl.Float64).alias("delay"))
        if "delay_raw" in df.columns:
            return df.with_columns(pl.col("delay_raw").cast(pl.Float64).alias("delay"))
        if "delay" in df.columns:
            return df.with_columns(pl.col("delay").cast(pl.Float64).alias("delay"))
        return df

    df = df_pred.copy()
    required = {"stimulus", "response", "performance"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required 2AFC-delay columns: {missing}")

    if "correct_bool" not in df.columns:
        df["correct_bool"] = df["performance"].astype(bool)
    if "pL" not in df.columns or "pR" not in df.columns:
        raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

    df["p_pred"] = df["pR"]
    df["p_model_correct"] = _correct_prob_pandas(df)
    if "delays" in df.columns:
        df["delay"] = pd.to_numeric(df["delays"], errors="coerce")
    elif "delay_raw" in df.columns:
        df["delay"] = pd.to_numeric(df["delay_raw"], errors="coerce")
    elif "delay" in df.columns:
        df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
    return df


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
) -> tuple[list[dict] | None, str | None]:
    df_pd = to_pandas_df(trial_df)
    if regressor_col not in df_pd.columns:
        return None, None

    df_pd, bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=4,
        quantiles=None,
    )
    if df_pd is None:
        return None, None
    reg_bin_labels = bin_centers["_reg_bin"].tolist()

    df_pd = attach_response_right_column(df_pd, response_mode=RESPONSE_MODE)
    df_pd = attach_signed_delay_columns(df_pd)
    if df_pd.empty:
        return None, None

    conds = sorted(df_pd["condition"].dropna().unique()) if "condition" in df_pd.columns else []
    exps = sorted(df_pd["experiment"].dropna().unique()) if "experiment" in df_pd.columns else []
    delay_ticks, delay_tick_labels = delay_ticks_from_df(df_pd)

    def _subject_summary(
        *,
        x_col: str,
        subgroup_col: str | None = None,
        subgroup_value=None,
    ) -> pd.DataFrame:
        plot_df = df_pd.copy()
        if subgroup_col is not None:
            plot_df = plot_df[plot_df[subgroup_col] == subgroup_value].copy()
        plot_df = plot_df[
            plot_df["_reg_bin"].notna()
            & plot_df[x_col].notna()
            & plot_df["_reg_bin"].isin(reg_bin_labels)
        ].copy()
        if plot_df.empty:
            return pd.DataFrame()
        return (
            plot_df.groupby(["_reg_bin", "subject", x_col], observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=(PRED_COL, "mean"),
                n_trials=("_response_right", "count"),
            )
            .reset_index()
        )

    panels: list[dict] = []

    panels.append(
        {
            "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col="delay",
                subject_col="subject",
                data_col="_response_right",
                model_col=PRED_COL,
                line_order=reg_bin_labels,
            ),
            "subject_summary": _subject_summary(x_col="delay"),
            "meta": {
                "xlabel": "Delay",
                "ylabel": p_right_label(),
                "legend_title": display_regressor_name(regressor_col),
                "baseline": BASELINE,
                "xticks": delay_ticks,
                "x_tick_labels": delay_tick_labels,
                "x_col": "delay",
            },
        }
    )

    if "_signed_delay_cat" in df_pd.columns and df_pd["_signed_delay_cat"].notna().any():
        x_order, x_tick_labels = signed_delay_order_and_labels(df_pd)
        signed_delay_code_col = "_signed_delay_code"
        code_map = {value: idx for idx, value in enumerate(x_order)}
        df_pd[signed_delay_code_col] = df_pd["_signed_delay_cat"].astype(str).map(code_map)
        signed_summary = summarize_grouped_panel(
            df_pd,
            line_group_col="_reg_bin",
            x_col="_signed_delay_cat",
            subject_col="subject",
            data_col="_response_right",
            model_col=PRED_COL,
            line_order=reg_bin_labels,
            x_order=x_order,
        )
        if not signed_summary.empty:
            signed_summary[signed_delay_code_col] = signed_summary["_signed_delay_cat"].astype(str).map(code_map)
        signed_subject_summary = _subject_summary(x_col="_signed_delay_cat")
        if not signed_subject_summary.empty:
            signed_subject_summary[signed_delay_code_col] = (
                signed_subject_summary["_signed_delay_cat"].astype(str).map(code_map)
            )
        panels.append(
            {
                "summary": signed_summary,
                "subject_summary": signed_subject_summary,
                "meta": {
                    "xlabel": "Signed delay",
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "x_order": x_order,
                    "x_tick_labels": x_tick_labels,
                    "categorical_x": True,
                    "x_col": "_signed_delay_cat",
                    "fit_x_col": signed_delay_code_col,
                },
            }
        )

    for cond in conds:
        panels.append(
            {
                "summary": summarize_grouped_panel(
                    df_pd,
                    line_group_col="_reg_bin",
                    x_col="delay",
                    subject_col="subject",
                    data_col="_response_right",
                    model_col=PRED_COL,
                    line_order=reg_bin_labels,
                    subgroup_col="condition",
                    subgroup_value=cond,
                ),
                "subject_summary": _subject_summary(
                    x_col="delay",
                    subgroup_col="condition",
                    subgroup_value=cond,
                ),
                "meta": {
                    "xlabel": "Delay",
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "xticks": delay_ticks,
                    "x_tick_labels": delay_tick_labels,
                    "x_col": "delay",
                },
            }
        )

    for exp in exps:
        panels.append(
            {
                "summary": summarize_grouped_panel(
                    df_pd,
                    line_group_col="_reg_bin",
                    x_col="delay",
                    subject_col="subject",
                    data_col="_response_right",
                    model_col=PRED_COL,
                    line_order=reg_bin_labels,
                    subgroup_col="experiment",
                    subgroup_value=exp,
                ),
                "subject_summary": _subject_summary(
                    x_col="delay",
                    subgroup_col="experiment",
                    subgroup_value=exp,
                ),
                "meta": {
                    "xlabel": "Delay",
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "xticks": delay_ticks,
                    "x_tick_labels": delay_tick_labels,
                    "x_col": "delay",
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
    required = {regressor_col, "response", PRED_COL, "subject"}
    if not required.issubset(df_pd.columns):
        return None, None
    resolved_group_col, resolved_group_order = resolve_grouping(
        df_pd,
        group_col=group_col,
        group_order=group_order,
    )

    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd[PRED_COL] = pd.to_numeric(df_pd[PRED_COL], errors="coerce")
    df_pd = attach_response_right_column(df_pd, response_mode=RESPONSE_MODE)
    df_pd = attach_signed_delay_columns(df_pd)

    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd[PRED_COL])
        & np.isfinite(df_pd["_response_right"])
    ].copy()
    if df_pd.empty:
        return None, None

    df_pd = df_pd[df_pd["_signed_delay_cat"].notna()].copy()
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

    delay_order, delay_labels = signed_delay_order_and_labels(df_pd)

    if resolved_group_col is None:
        summary = summarize_grouped_panel(
            df_pd,
            line_group_col="_signed_delay_cat",
            x_col="_reg_bin",
            subject_col="subject",
            data_col="_response_right",
            model_col=PRED_COL,
            line_order=delay_order,
            x_order=bin_order,
        )
        line_group_col = "_signed_delay_cat"
        line_order = delay_order
        line_labels = delay_labels
        legend_title = "Signed delay"
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
        line_labels = []
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
        "line_labels": line_labels,
        "legend_outside": True,
    }
    return summary, meta



@_register(["two_afc_delay", "2afc_delay", "2AFC_delay"])
class TwoAFCDelayAdapter(TaskAdapter):
    """Adapter for the Tiffany binary 2AFC task with trial delay."""

    task_key: str = "2AFC_delay"
    task_label: str = "2AFC delay"
    num_classes: int = 2
    data_file: str = "tiffany.parquet"
    sort_col = ["session", "trial"]
    session_col: str = "session"

    _SCORING_OPTIONS: dict = {
        "stim (w)": [("stim", "pos")],
        "stim (|w|)": [("stim", "abs")],
        "delay (|w|)": [("delay", "abs")],
        "stim_x_delay (|w|)": [("stim_x_delay", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim (w)"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("drug") == "Rest")

    def _build_feature_df(
        self,
        df_sub: pl.DataFrame,
        tau: float = 50.0,
    ) -> pl.DataFrame:
        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["session", "trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)
        subject_half_life = load_subject_choice_half_life(
            task_key=self.task_key,
            fit_model_id=_RAW_PARAM_MODEL_ID,
            subject=str(df_pd["subject"].iloc[0]) if "subject" in df_pd.columns and len(df_pd) else None,
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
        choice_lag_cols = _choice_lag_names()

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
            session_idx = session_to_idx[df_session["session"].iloc[0]]
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
            choice_lag_df = pd.DataFrame(
                {
                    lag_col: part["choice_signed"].shift(lag_idx).fillna(0.0).astype(np.float32)
                    for lag_idx, lag_col in enumerate(choice_lag_cols, start=1)
                },
                index=part.index,
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
            part["prev_choice"] = prev_choice
            part["prev_reward"] = prev_reward

            cumulative_reward = part["hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            part["cumulative_reward"] = cumulative_reward.astype(np.float32)
            part["prev_abs_stim"] = part["stim"].abs().shift(1).fillna(0).astype(np.float32)
            prev_choice_signed = prev_choice.to_numpy().astype(np.float32)
            signed_prev_reward = np.where(prev_reward.to_numpy() > 0, 1.0, -1.0).astype(np.float32)
            part["wsls"] = (prev_choice_signed * signed_prev_reward).astype(np.float32)

            part["after_correct"] = part["after_correct"].fillna(0).astype(np.float32)
            part["repeat"] = part["repeat"].fillna(0).astype(np.float32)
            part["repeat_choice_side"] = part["repeat_choice_side"].fillna(0).astype(np.float32)
            part["WM"] = part["WM"].fillna(0).astype(np.float32)
            part["RL"] = part["RL"].fillna(0).astype(np.float32)
            part["ILD"] = part["stim"].astype(np.float32)
            part = pd.concat(
                [
                    part,
                    bias_hot,
                    pd.DataFrame(delay_hot_cols, index=part.index),
                    pd.DataFrame(stimx_delay_hot_cols, index=part.index),
                    choice_lag_df,
                ],
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
        bias_param = _safe_weighted_sum_regressor(feature_df, _BIAS_PARAM_SPEC)
        delay_param = _safe_weighted_sum_regressor(feature_df, _DELAY_PARAM_SPEC)
        stim_x_delay_param = _safe_weighted_sum_regressor(feature_df, _STIM_X_DELAY_PARAM_SPEC)
        choice_lag_param = _safe_weighted_sum_regressor(feature_df, _CHOICE_LAG_PARAM_SPEC)
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
        family_aliases = {
            "bias_hot": self.bias_hot_cols(feature_df),
            "delay_hot": self.delay_hot_cols(feature_df),
            "choice_lag": self.choice_lag_cols(feature_df),
            "at_choice_lag": self.choice_lag_cols(feature_df),
            "stim_x_delay_hot": self.stim_x_delay_hot_cols(feature_df),
            "stim_x_delay_one_hot": self.stim_x_delay_hot_cols(feature_df),
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
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}"
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
            bias_hot_cols = [f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(_max_subject_sessions())]
            choice_lag_cols = _choice_lag_names()
        else:
            delay_hot_cols = self.delay_hot_cols(df)
            stim_x_delay_hot_cols = self.stim_x_delay_hot_cols(df)
            bias_hot_cols = self.bias_hot_cols(df)
            choice_lag_cols = self.choice_lag_cols(df)

        default_cols = [
            *bias_hot_cols,
            *choice_lag_cols,
            "stim",
            *delay_hot_cols,
            *stim_x_delay_hot_cols,
        ]
        return list(dict.fromkeys(default_cols))

    def default_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        available_cols = list(EMISSION_COLS)
        if df is not None:
            available_cols.extend(self.sf_cols(df))
            available_cols.extend(self.delay_hot_cols(df))
            available_cols.extend(self.stim_x_delay_hot_cols(df))
            available_cols.extend(self.bias_hot_cols(df))
            available_cols.extend(self.choice_lag_cols(df))
        return list(dict.fromkeys(available_cols))

    def available_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

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
            family_aliases = {
                "bias_hot": self.bias_hot_cols(df),
                "delay_hot": self.delay_hot_cols(df),
                "choice_lag": self.choice_lag_cols(df),
                "at_choice_lag": self.choice_lag_cols(df),
                "stim_x_delay_hot": self.stim_x_delay_hot_cols(df),
                "stim_x_delay_one_hot": self.stim_x_delay_hot_cols(df),
            }
            for col in requested_ecols:
                expanded_ecols.extend(family_aliases.get(col, [col]))
        else:
            expanded_ecols = list(requested_ecols)
        allowed_ecols = set(self.available_emission_cols(df))
        expanded_ecols = _drop_unavailable_bias_hot_cols(expanded_ecols, allowed_ecols)
        bad_e = [c for c in expanded_ecols if c not in allowed_ecols]
        bad_u = [c for c in requested_ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}"
            )
        return {"X_cols": list(dict.fromkeys(expanded_ecols)), "U_cols": list(requested_ucols)}

    def delay_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return delay one-hot columns."""
        return _infer_delay_hot_cols_from_df(df)

    def stim_x_delay_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return stimulus×delay one-hot columns."""
        existing = _stim_x_delay_hot_cols(list(df.columns))
        if existing:
            return existing
        return [
            f"stim_x_delay_hot_{col.removeprefix(_DELAY_HOT_COL_PREFIX)}"
            for col in self.delay_hot_cols(df)
        ]

    def bias_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return session one-hot bias columns."""
        return _infer_bias_hot_cols_from_df(df)

    def choice_lag_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        """Return explicit previous-choice lag columns."""
        if df is not None:
            existing = _choice_lag_cols(list(df.columns))
            if existing:
                return existing
        return _choice_lag_names()

    def weight_family_specs(self, weights_df=None) -> Dict[str, dict]:
        df = to_pandas_df(weights_df) if weights_df is not None else None
        feature_names = [] if df is None or df.empty or "feature" not in df.columns else pd.unique(df["feature"].astype(str)).tolist()
        delay_cols = _delay_hot_cols(feature_names)
        stim_x_delay_cols = _stim_x_delay_hot_cols(feature_names)
        choice_cols = _choice_lag_cols(feature_names)
        bias_cols = _bias_hot_cols(feature_names)

        def _delay_groups(columns: list[str], prefix: str) -> list[tuple[str, list[str]]]:
            groups: list[tuple[str, list[str]]] = []
            for col in columns:
                token = col.removeprefix(prefix)
                parsed = _parse_delay_level_token(token)
                if parsed is None:
                    continue
                groups.append((_format_delay_level_label(parsed), [col]))
            return groups

        return {
            "stim_hot": {
                "title": "stim×delay one-hot",
                "xlabel": "delay level",
                "plot_kind": "box",
                "feature_groups": _delay_groups(stim_x_delay_cols, "stim_x_delay_hot_"),
            },
            "delay_hot": {
                "title": "delay_hot",
                "xlabel": "delay level",
                "plot_kind": "box",
                "feature_groups": _delay_groups(delay_cols, _DELAY_HOT_COL_PREFIX),
            },
            "stim_x_delay_hot": {
                "title": "stim×delay one-hot",
                "xlabel": "delay level",
                "plot_kind": "box",
                "feature_groups": _delay_groups(stim_x_delay_cols, "stim_x_delay_hot_"),
            },
            "stim_x_delay_one_hot": {
                "title": "stim×delay one-hot",
                "xlabel": "delay level",
                "plot_kind": "box",
                "feature_groups": _delay_groups(stim_x_delay_cols, "stim_x_delay_hot_"),
            },
            "choice_lag": {
                "title": "choice_lag_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_COL_PREFIX))), [col]) for col in choice_cols],
            },
            "at_choice_lag": {
                "title": "choice_lag_*",
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
        del available_cols
        return []

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

    def get_plots(self) -> types.ModuleType:
        return resolve_plots_module(
            adapter_module_name=__name__,
            task_key=self.task_key,
        )

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "stim (w)"),
            self._SCORING_OPTIONS["stim (w)"],
        )

        def _score_states(W_np: np.ndarray, feat_names: list[str], *, stim: str = "stim_vals") -> np.ndarray:
            name2fi_local = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(W_np.shape[0], dtype=float)
            n_terms = 0
            for feat_name, mode in pairs:
                fi = name2fi_local.get(feat_name)
                if fi is None:
                    continue
                vals = W_np[:, 0, fi].astype(float)
                if mode == "neg":
                    vals = -vals
                elif mode == "abs":
                    vals = np.abs(vals)
                elif mode == "pos":
                    vals = vals
                else:
                    raise ValueError(f"Unknown 2AFC-delay scoring mode {mode!r}.")
                scores += vals
                n_terms += 1

            if n_terms > 0:
                return scores / n_terms

            stim_candidates = [stim]
            if stim != "stim_vals":
                stim_candidates.append("stim_vals")
            for stim_name in stim_candidates:
                stim_fi_local = name2fi_local.get(stim_name)
                if stim_fi_local is not None:
                    return -W_np[:, 0, stim_fi_local]
            return -W_np[:, 0, :].mean(axis=1)

        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict = {}

        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj] = list(range(K))
                continue

            feat = list(arrays_store[subj].get("X_cols", base_feat))
            W = np.asarray(W)
            name2fi = {n: i for i, n in enumerate(feat)}

            selected_stim = "stim"
            state_scores = _score_states(W, feat, stim=selected_stim)

            engaged_k = int(np.argmax(state_scores))
            others = [k for k in range(K) if k != engaged_k]

            if K == 2:
                labels = {engaged_k: "Engaged", others[0]: "Disengaged"}
                order = [engaged_k, others[0]]
            elif K == 3:
                bias_fi = name2fi.get("bias", None)
                if bias_fi is None:
                    bias_vals = np.zeros(len(others))
                else:
                    bias_vals = W[others, 0, bias_fi]
                left_k = others[int(np.argmin(bias_vals))]
                right_k = others[int(np.argmax(bias_vals))]
                labels = {
                    engaged_k: "Engaged",
                    left_k: "Biased L",
                    right_k: "Biased R",
                }
                order = [engaged_k, left_k, right_k]
            else:
                ranked_rest = sorted(others, key=lambda k: state_scores[k], reverse=True)
                labels = {engaged_k: "Engaged"}
                for idx, k in enumerate(ranked_rest, start=1):
                    labels[k] = f"Disengaged {idx}"
                order = [engaged_k, *ranked_rest]

            state_labels[subj] = labels
            state_order[subj] = order

        return state_labels, state_order
