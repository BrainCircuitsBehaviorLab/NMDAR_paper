"""Task adapter for the Nuo auditory 2AFC task."""
from __future__ import annotations

from typing import Any, List, Tuple, Dict

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl

from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    mean_feature_weights_from_fit,
    weighted_sum_regressor,
)
from glmhmmt.tasks import TaskAdapter, _register
from src.process.common import (
    PreparedWeightFamilyPlot,
    binned_feature_summary,
    prepare_grouped_weight_family_plot,
    to_pandas_df,
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

_NUM_STIM_BINS = 9
_NUM_CHOICE_LAGS = 15
_STIM_BIN_PREFIX = "stim_bin_"
_CHOICE_LAG_PREFIX = "choice_lag_"
_DIFFICULTY_COL_PREFIX = "difficulty_"
_BIAS_HOT_COL_PREFIX = "bias_"
_STIM_PARAM_COL = "stim_param"
_EVIDENCE_PARAM_COL = "evidence_param"
_DIFFICULTY_PARAM_COL = "difficulty_param"
_AT_CHOICE_PARAM_COL = "at_choice_param"
_DIFFICULTY_LEVELS = ("easy", "medium", "hard")
PRED_COL = "p_pred"
RESPONSE_MODE = "pm1_or_prob"
BASELINE = 0.5
RESPONSE_COL = "response"
PERFORMANCE_COL = "performance"
EVIDENCE_COL = "total_evidence_strength"
PLOT_CHOICE_RIGHT_COL = "_plot_choice_right"


def _stim_bin_names() -> list[str]:
    return [f"{_STIM_BIN_PREFIX}{idx:02d}" for idx in range(_NUM_STIM_BINS)]


def _choice_lag_names() -> list[str]:
    return [f"{_CHOICE_LAG_PREFIX}{idx:02d}" for idx in range(1, _NUM_CHOICE_LAGS + 1)]


def _difficulty_col_names() -> list[str]:
    return [f"{_DIFFICULTY_COL_PREFIX}{level}" for level in _DIFFICULTY_LEVELS]


_STIM_BIN_COLS = _stim_bin_names()
_CHOICE_LAG_COLS = _choice_lag_names()
_DIFFICULTY_COLS = _difficulty_col_names()

EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "stim_vals",
    "stim_param",
    "difficulty_param",
    "at_choice",
    "at_choice_param",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
]

TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
]

_STIM_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_param",
    fit_task="nuo_auditory",
    fit_model_kind="glm",
    fit_model_id="one hot lapses",
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(_STIM_BIN_COLS),
)

_EVIDENCE_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="evidence_param",
    fit_task="nuo_auditory",
    fit_model_kind="glm",
    fit_model_id="one hot lapses",
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(_STIM_BIN_COLS),
)

_BIAS_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="bias_param",
    fit_task="nuo_auditory",
    fit_model_kind="glm",
    fit_model_id="one hot lapses",
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_BIAS_HOT_COL_PREFIX,),
)

_DIFFICULTY_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="difficulty_param",
    fit_task="nuo_auditory",
    fit_model_kind="glm",
    fit_model_id="one hot lapses",
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(_DIFFICULTY_COLS),
)

_AT_CHOICE_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="at_choice_param",
    fit_task="nuo_auditory",
    fit_model_kind="glm",
    fit_model_id="lagged choices",
    arrays_suffix="glm_arrays.npz",
    source_features=tuple(_CHOICE_LAG_COLS),
)

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim_vals": r"$\mathrm{Stimulus}$",
    "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
    "stim_param": r"$\mathrm{Stimulus}_{\mathrm{param}}$",
    "evidence_param": r"$\mathrm{Evidence}_{\mathrm{param}}$",
    "difficulty_param": r"$\mathrm{Difficulty}_{\mathrm{param}}$",
    "difficulty_easy": r"$\mathrm{Easy}$",
    "difficulty_medium": r"$\mathrm{Medium}$",
    "difficulty_hard": r"$\mathrm{Hard}$",
    "bias": r"$\mid\mathrm{bias}\mid$",
    "at_choice": r"$\mathrm{A}_t^{\mathrm{choice}}$",
    "at_choice_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    "at_error": r"$\mathrm{A}_t^{\mathrm{error}}$",
    "at_correct": r"$\mathrm{A}_t^{\mathrm{correct}}$",
    "reward_trace": r"$\mathrm{Reward}_{\mathrm{trace}}$",
    "prev_choice": r"$\mathrm{PrevChoice}$",
    "prev_reward": r"$\mathrm{PrevReward}$",
    "prev_abs_stim": r"$|\mathrm{PrevStim}|$",
    "cumulative_reward": r"$\mathrm{CumReward}$",
}

_EMISSION_GROUPS: list[dict] = [
    {"key": "bias", "label": "bias", "members": {"N": "bias"}},
    {"key": "bias_param", "label": "bias param", "members": {"N": "bias_param"}},
    {"key": "stim_vals", "label": "stim vals", "members": {"N": "stim_vals"}},
    {"key": "stim_param", "label": "stim param", "members": {"N": "stim_param"}},
    {"key": "evidence_param", "label": "evidence param", "members": {"N": "evidence_param"}},
    {"key": "difficulty_param", "label": "difficulty param", "members": {"N": "difficulty_param"}},
    {"key": "at_choice", "label": "action (choice)", "members": {"N": "at_choice"}},
    {"key": "at_choice_param", "label": "choice param", "members": {"N": "at_choice_param"}},
    {"key": "at_error", "label": "action (error)", "members": {"N": "at_error"}},
    {"key": "at_correct", "label": "action (correct)", "members": {"N": "at_correct"}},
    {"key": "reward_trace", "label": "reward trace", "members": {"N": "reward_trace"}},
    {"key": "prev_choice", "label": "prev choice", "members": {"N": "prev_choice"}},
    {"key": "prev_reward", "label": "prev reward", "members": {"N": "prev_reward"}},
    {"key": "cumulative_reward", "label": "cumulative reward", "members": {"N": "cumulative_reward"}},
    {"key": "prev_abs_stim", "label": "prev abs stim", "members": {"N": "prev_abs_stim"}},
]


def _validated_half_life(tau: float) -> float:
    """Return a positive half-life for Polars EWMA features."""
    half_life = float(tau)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError(f"Nuo auditory requires tau > 0, got {tau!r}")
    return half_life


def _stim_bin_centers() -> np.ndarray:
    return np.linspace(-1.0, 1.0, _NUM_STIM_BINS, dtype=np.float32)


def _stim_bin_edges() -> np.ndarray:
    centers = _stim_bin_centers().astype(np.float64)
    mids = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate(([-np.inf], mids, [np.inf]))


def _stim_bin_indices(values: np.ndarray) -> np.ndarray:
    edges = _stim_bin_edges()
    idx = np.digitize(values.astype(np.float64), edges[1:-1], right=False)
    return np.clip(idx, 0, _NUM_STIM_BINS - 1).astype(np.int32)


def with_plot_choice_right(
    df: pd.DataFrame,
    choice_col: str = RESPONSE_COL,
) -> tuple[pd.DataFrame, str]:
    """Return a copy with a numeric right-choice column for plotting."""

    out = df.copy()
    out[PLOT_CHOICE_RIGHT_COL] = pd.to_numeric(out[choice_col], errors="coerce")
    return out, PLOT_CHOICE_RIGHT_COL


def attach_natural_evidence_bins(
    df_pd: pd.DataFrame,
    *,
    evidence_col: str = EVIDENCE_COL,
    bin_col: str = "_evidence_bin",
) -> tuple[pd.DataFrame | None, list[float]]:
    """Assign Nuo's native evidence-bin centers to each trial."""

    df = df_pd.copy()
    evidence = pd.to_numeric(df[evidence_col], errors="coerce")
    valid = evidence.notna() & np.isfinite(evidence)
    centers = _stim_bin_centers().astype(float)
    if int(valid.sum()) == 0:
        return None, centers.tolist()

    mids = (centers[:-1] + centers[1:]) / 2.0
    df[bin_col] = np.nan
    bin_idx = np.digitize(evidence.loc[valid].to_numpy(dtype=float), mids, right=False)
    bin_idx = np.clip(bin_idx, 0, len(centers) - 1)
    df.loc[valid, bin_col] = centers[bin_idx]
    return df, centers.tolist()


def prepare_predictions_df(df_pred):
    """Prepare canonical Nuo auditory trial-level prediction columns."""

    if isinstance(df_pred, pl.DataFrame):
        df = df_pred.clone()
        required = {"stimulus", RESPONSE_COL, PERFORMANCE_COL, EVIDENCE_COL}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required Nuo auditory columns: {missing}")

        if "correct_bool" not in df.columns:
            df = df.with_columns(pl.col(PERFORMANCE_COL).cast(pl.Boolean).alias("correct_bool"))
        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        return df.with_columns(
            pl.col("pR").alias(PRED_COL),
            pl.when(pl.col("stimulus") == 0)
            .then(pl.col("pL"))
            .otherwise(pl.col("pR"))
            .alias("p_model_correct"),
        )

    df = df_pred.copy()
    required = {"stimulus", RESPONSE_COL, PERFORMANCE_COL, EVIDENCE_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required Nuo auditory columns: {missing}")

    if "correct_bool" not in df.columns:
        df["correct_bool"] = df[PERFORMANCE_COL].astype(bool)
    if "pL" not in df.columns or "pR" not in df.columns:
        raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

    df[PRED_COL] = df["pR"]
    df["p_model_correct"] = df.apply(
        lambda row: row["pL"] if row["stimulus"] == 0 else row["pR"],
        axis=1,
    )
    return df


def prepare_compat_plot_df(plot_df) -> pd.DataFrame:
    """Return pandas trial data with Nuo prediction aliases present."""

    df_pd = to_pandas_df(plot_df)
    if {"correct_bool", "p_model_correct"} - set(df_pd.columns):
        df_pd = to_pandas_df(prepare_predictions_df(df_pd))

    df_pd = df_pd.copy()
    if "pR" not in df_pd.columns:
        if PRED_COL in df_pd.columns:
            df_pd["pR"] = pd.to_numeric(df_pd[PRED_COL], errors="coerce")
        elif "pL" in df_pd.columns:
            df_pd["pR"] = 1.0 - pd.to_numeric(df_pd["pL"], errors="coerce")
    if "pL" not in df_pd.columns and "pR" in df_pd.columns:
        df_pd["pL"] = 1.0 - pd.to_numeric(df_pd["pR"], errors="coerce")
    return df_pd


def ensure_previous_choice_column(df_pd: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    df = df_pd.copy()
    group_cols = [col for col in ("subject", "session") if col in df.columns]
    sort_cols = [col for col in ("subject", "session", "trial_idx", "trial") if col in df.columns]
    if group_cols and sort_cols and RESPONSE_COL in df.columns:
        df = df.sort_values(sort_cols).copy()
        df["_prev_choice_plot"] = df.groupby(group_cols, observed=True)[RESPONSE_COL].shift(1)
        return df, "_prev_choice_plot"
    if "prev_choice" in df.columns:
        return df, "prev_choice"
    return df, None


def build_simple_summary_from_feature(
    df_pd: pd.DataFrame,
    *,
    feature_col: str,
    data_col: str,
    model_col: str,
    subj_col: str = "subject",
    n_bins: int = 10,
) -> tuple[pd.DataFrame | None, dict]:
    summary = binned_feature_summary(
        df_pd,
        feature_col=feature_col,
        choice_col=data_col,
        pred_col=model_col,
        subj_col=subj_col,
        n_bins=n_bins,
    )
    if summary is None:
        return None, {}

    subj_agg, _ = summary
    overall = (
        subj_agg.groupby("_x_bin", observed=True)
        .agg(
            x_center=("center", "median"),
            data_mean=("data_mean", "mean"),
            data_std=("data_mean", "std"),
            data_count=("data_mean", "count"),
            model_mean=("model_mean", "mean"),
            model_std=("model_mean", "std"),
        )
        .reset_index(drop=True)
        .sort_values("x_center")
    )
    overall["data_sem"] = overall["data_std"].fillna(0.0) / np.sqrt(overall["data_count"].clip(lower=1))
    overall["model_sem"] = overall["model_std"].fillna(0.0) / np.sqrt(overall["data_count"].clip(lower=1))
    return overall, {}


def _safe_weighted_sum_regressor(part, spec: FittedWeightRegressorSpec) -> np.ndarray | None:
    try:
        return weighted_sum_regressor(part, spec, dtype=np.float32)
    except (FileNotFoundError, ValueError):
        return None


def _bias_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_BIAS_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


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


def _max_sessions_from_df(df: pl.DataFrame) -> int:
    if "session" not in df.columns:
        return 0
    if "subject" not in df.columns:
        return int(df.select(pl.col("session").drop_nulls().n_unique()).item() or 0)
    return int(
        df.group_by("subject")
        .agg(pl.col("session").drop_nulls().n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


def _infer_bias_hot_cols_from_df(df: pl.DataFrame) -> list[str]:
    existing = _bias_hot_cols(list(df.columns))
    if existing:
        return existing
    return [f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(_max_sessions_from_df(df))]


def _stim_param_weight_map() -> dict[str, float]:
    try:
        return mean_feature_weights_from_fit(_STIM_PARAM_SPEC)
    except (FileNotFoundError, ValueError):
        return {}


def _build_emission_groups(available_cols: list[str]) -> list[dict]:
    available = set(available_cols)
    result: list[dict] = []
    registered: set[str] = set()

    def add_scalar(group: dict) -> None:
        filtered = {k: v for k, v in group["members"].items() if v in available}
        if filtered:
            result.append({**group, "members": filtered})
            registered.update(filtered.values())

    bias_hot_cols = _bias_hot_cols(available_cols)
    stim_bin_cols = [col for col in available_cols if col in _STIM_BIN_COLS]
    difficulty_cols = [col for col in available_cols if col in _DIFFICULTY_COLS]
    choice_lag_cols = [col for col in available_cols if col in _CHOICE_LAG_COLS]
    for group in _EMISSION_GROUPS:
        add_scalar(group)
        if group["key"] == "stim_param" and stim_bin_cols:
            stim_hot_group = {
                "key": "stim_hot",
                "label": "stim_hot",
                "members": {},
                "toggle_members": list(stim_bin_cols),
                "hide_members": True,
            }
            result.append(stim_hot_group)
            result.append({**stim_hot_group, "key": "evidence_one_hot", "label": "evidence one-hot"})
            registered.update(stim_bin_cols)
        if group["key"] == "bias_param" and bias_hot_cols:
            result.append(
                {
                    "key": "bias_hot",
                    "label": "bias_hot",
                    "members": {},
                    "toggle_members": list(bias_hot_cols),
                    "hide_members": True,
                }
            )
            registered.update(bias_hot_cols)
        if group["key"] == "difficulty_param" and difficulty_cols:
            result.append(
                {
                    "key": "difficulty_hot",
                    "label": "difficulty_hot",
                    "members": {},
                    "toggle_members": list(difficulty_cols),
                    "hide_members": True,
                }
            )
            registered.update(difficulty_cols)
        if group["key"] == "at_choice_param" and choice_lag_cols:
            result.append(
                {
                    "key": "at_choice_lag",
                    "label": "choice_lag",
                    "members": {},
                    "toggle_members": list(choice_lag_cols),
                    "hide_members": True,
                }
            )
            registered.update(choice_lag_cols)

    remaining = [col for col in available_cols if col not in registered]
    if remaining:
        result.extend(_build_selector_groups(remaining, []))
    return result


@_register(["nuo_auditory", "auditory_2afc", "nuo_auditive"])
class NuoAuditoryAdapter(TaskAdapter):
    """Adapter for the binary Nuo auditory task."""

    task_key: str = "nuo_auditory"
    task_label: str = "Nuo auditory"
    num_classes: int = 2
    data_file: str = "hernando.parquet"
    sort_col = ["session", "trial"]
    session_col: str = "session"
    prediction_col: str = PRED_COL
    response_mode: str = RESPONSE_MODE
    psychometric_x_col: str = "total_evidence_strength"
    psychometric_x_label: str = "Evidence strength"
    accuracy_x_col: str = "total_evidence_strength"
    accuracy_x_label: str = "Evidence strength"
    emission_cols: list[str] = EMISSION_COLS
    transition_cols: list[str] = TRANSITION_COLS

    _SCORING_OPTIONS: dict = {
        "stim_vals (w)": [("stim_vals", "pos")],
        "stim_vals (-w)": [("stim_vals", "pos")],
        "stim_vals (|w|)": [("stim_vals", "abs")],
        "stim_param (w)": [("stim_param", "pos")],
        "stim_param (-w)": [("stim_param", "pos")],
        "stim_param (|w|)": [("stim_param", "abs")],
        "evidence_param (w)": [("evidence_param", "pos")],
        "evidence_param (-w)": [("evidence_param", "pos")],
        "evidence_param (|w|)": [("evidence_param", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_vals (w)"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop miss trials and add the canonical binary-task columns.

        The adapter keeps a task-owned feature contract by adding the canonical
        columns it will later use to build its own design matrices.
        """

        side_expr = (
            pl.when(pl.col("correct_side").str.to_lowercase() == "left")
            .then(pl.lit(0))
            .when(pl.col("correct_side").str.to_lowercase() == "right")
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .cast(pl.Int64)
        )
        response_expr = (
            pl.when(pl.col("last_choice").str.to_lowercase() == "left")
            .then(pl.lit(0))
            .when(pl.col("last_choice").str.to_lowercase() == "right")
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .cast(pl.Int64)
        )
        trial_idx_expr = (
            pl.col("__index_level_0__").cast(pl.Int64)
            if "__index_level_0__" in df.columns
            else pl.int_range(0, pl.len(), eager=False).cast(pl.Int64)
        )

        return (
            df.filter(~pl.col("miss_trial"))
            .with_columns(
                [
                    trial_idx_expr.alias("trial_idx"),
                    side_expr.alias("stimulus"),
                    response_expr.alias("response"),
                    pl.col("correct").cast(pl.Int64).alias("performance"),
                ]
            )
        )

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        """Return the Nuo auditory trial dataframe with derived regressors."""
        df_sub = df_sub.sort(["session", "trial"])
        half_life = _validated_half_life(tau)
        stim_scale = float(df_sub.select(pl.col("total_evidence_strength").abs().max()).item() or 0.0)
        if stim_scale <= 0:
            stim_scale = 1.0

        session_order = list(dict.fromkeys(df_sub["session"].to_list()))
        session_map = pl.DataFrame(
            {
                "session": session_order,
                "_session_idx": np.arange(len(session_order), dtype=np.int32),
            }
        )
        df_sub = df_sub.join(session_map, on="session", how="left")
        session_idx_np = df_sub["_session_idx"].to_numpy().astype(np.int32)
        bias_hot_exprs = [
            pl.Series(
                f"{_BIAS_HOT_COL_PREFIX}{idx}",
                (session_idx_np == idx).astype(np.float32),
            )
            for idx in range(len(session_order))
        ]

        choice_signed_expr = (
            pl.when(pl.col("response") == 1)
            .then(pl.lit(1.0))
            .when(pl.col("response") == 0)
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float32)
        )

        df_sub = df_sub.with_columns(
            [
                (-pl.col("total_evidence_strength").cast(pl.Float32) / pl.lit(stim_scale)).alias("stim_vals"),
                pl.lit(1.0).cast(pl.Float32).alias("bias"),
                choice_signed_expr.alias("_choice_signed"),
                pl.when(pl.col("correct_side").str.to_lowercase() == "left")
                .then(pl.lit(-1.0))
                .when(pl.col("correct_side").str.to_lowercase() == "right")
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .cast(pl.Float32)
                .alias("_correct_side_signed"),
            ]
        )
        if bias_hot_exprs:
            df_sub = df_sub.with_columns(bias_hot_exprs)

        stim_vals_np = df_sub["stim_vals"].to_numpy().astype(np.float32)
        stim_bin_idx = _stim_bin_indices(stim_vals_np)
        stim_bin_exprs = [
            pl.Series(
                name,
                (stim_bin_idx == idx).astype(np.float32),
            )
            for idx, name in enumerate(_STIM_BIN_COLS)
        ]
        df_sub = df_sub.with_columns(stim_bin_exprs)
        df_sub = df_sub.with_columns(
            [
                pl.when(pl.col("difficulty").str.to_lowercase() == level)
                .then(pl.col("_correct_side_signed"))
                .otherwise(pl.lit(0.0))
                .cast(pl.Float32)
                .alias(col)
                for level, col in zip(_DIFFICULTY_LEVELS, _DIFFICULTY_COLS)
            ]
        )

        df_sub = df_sub.with_columns(
            [
                pl.col("_choice_signed").shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("_prev_choice_signed"),
                (pl.col("_choice_signed") * pl.col("performance")).shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("_prev_correct_signed"),
                (pl.col("_choice_signed") * (1.0 - pl.col("performance"))).shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("_prev_error_signed"),
                pl.col("response").shift(1).fill_null(0).over("session").cast(pl.Float32).alias("prev_choice"),
                pl.col("performance").shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("prev_reward"),
                pl.col("stim_vals").abs().shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("prev_abs_stim"),
                pl.col("performance").shift(1).fill_null(0.0).cum_sum().over("session").cast(pl.Float32).alias("_cumulative_reward_raw"),
            ]
        )
        lag_exprs = [
            pl.col("_choice_signed")
            .shift(lag)
            .fill_null(0.0)
            .over("session")
            .cast(pl.Float32)
            .alias(name)
            for lag, name in enumerate(_CHOICE_LAG_COLS, start=1)
        ]
        df_sub = df_sub.with_columns(lag_exprs)

        bias_param = _safe_weighted_sum_regressor(df_sub, _BIAS_PARAM_SPEC)
        stim_param = _safe_weighted_sum_regressor(df_sub, _STIM_PARAM_SPEC)
        evidence_param = _safe_weighted_sum_regressor(df_sub, _EVIDENCE_PARAM_SPEC)
        difficulty_param = _safe_weighted_sum_regressor(df_sub, _DIFFICULTY_PARAM_SPEC)
        at_choice_param = _safe_weighted_sum_regressor(df_sub, _AT_CHOICE_PARAM_SPEC)
        df_sub = df_sub.with_columns(
            [
                pl.when(pl.col("_cumulative_reward_raw").max().over("session") > 0)
                .then(pl.col("_cumulative_reward_raw") / pl.col("_cumulative_reward_raw").max().over("session"))
                .otherwise(pl.lit(0.0))
                .cast(pl.Float32)
                .alias("cumulative_reward"),
                pl.col("_prev_choice_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("at_choice"),
                pl.col("_prev_correct_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("at_correct"),
                pl.col("_prev_error_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("at_error"),
                pl.col("prev_reward").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("reward_trace"),
                (
                    pl.Series("bias_param", bias_param)
                    if bias_param is not None
                    else pl.col("bias").cast(pl.Float32).alias("bias_param")
                ),
                (
                    pl.Series(_STIM_PARAM_COL, stim_param)
                    if stim_param is not None
                    else pl.col("stim_vals").cast(pl.Float32).alias(_STIM_PARAM_COL)
                ),
                (
                    pl.Series(_EVIDENCE_PARAM_COL, evidence_param)
                    if evidence_param is not None
                    else pl.col("stim_vals").cast(pl.Float32).alias(_EVIDENCE_PARAM_COL)
                ),
                (
                    pl.Series(_DIFFICULTY_PARAM_COL, difficulty_param)
                    if difficulty_param is not None
                    else pl.sum_horizontal([pl.col(col) for col in _DIFFICULTY_COLS]).cast(pl.Float32).alias(_DIFFICULTY_PARAM_COL)
                ),
                (
                    pl.Series(_AT_CHOICE_PARAM_COL, at_choice_param)
                    if at_choice_param is not None
                    else pl.col("_prev_choice_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias(_AT_CHOICE_PARAM_COL)
                ),
            ]
        )
        return df_sub

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for one subject."""
        feature_df = self.build_feature_df(df_sub, tau=tau)
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
        """Return ``(y, X, U, names)`` for one subject."""
        ecols = emission_cols if emission_cols is not None else self.default_emission_cols(feature_df)
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(feature_df))
        ecols = _drop_unavailable_bias_hot_cols(list(ecols), allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        allowed_ucols = self.available_transition_cols()
        bad_u = [c for c in ucols if c not in allowed_ucols]
        if bad_e:
            raise ValueError(
                f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}"
            )
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {allowed_ucols}"
            )

        y = jnp.asarray(feature_df["response"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {"X_cols": list(ecols), "U_cols": list(ucols)}
        return y, X, U, names

    def _dynamic_emission_cols(self, df: pl.DataFrame | None) -> list[str]:
        if df is None:
            return []

        dynamic_cols = [
            c
            for c in list(_STIM_BIN_COLS) + list(_DIFFICULTY_COLS) + list(_CHOICE_LAG_COLS)
            if c in df.columns
        ]
        dynamic_cols.extend(_bias_hot_cols(list(df.columns)))
        dynamic_cols = list(dict.fromkeys(dynamic_cols))
        if dynamic_cols:
            return dynamic_cols

        raw_has_stim = "total_evidence_strength" in df.columns
        raw_has_difficulty = "difficulty" in df.columns and "correct_side" in df.columns
        raw_has_choice = "last_choice" in df.columns or "response" in df.columns
        raw_has_session = "session" in df.columns
        inferred: list[str] = []
        if raw_has_stim:
            inferred.extend(_STIM_BIN_COLS)
        if raw_has_difficulty:
            inferred.extend(_DIFFICULTY_COLS)
        if raw_has_session:
            inferred.extend(_infer_bias_hot_cols_from_df(df))
        if raw_has_choice:
            inferred.extend(_CHOICE_LAG_COLS)
        return inferred

    def default_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        return list(self.emission_cols) + self._dynamic_emission_cols(df)

    def default_transition_cols(self) -> List[str]:
        return list(self.transition_cols)

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        available_cols = list(self.emission_cols) + [_EVIDENCE_PARAM_COL]
        return list(dict.fromkeys(available_cols + self._dynamic_emission_cols(df)))

    def available_transition_cols(self) -> List[str]:
        return list(self.transition_cols)

    def build_emission_groups(self, available_cols: List[str]) -> list[dict]:
        return _build_emission_groups(list(available_cols))

    def build_transition_groups(self, available_cols: List[str]) -> list[dict]:
        del available_cols
        return []

    def weight_family_specs(self, weights_df=None) -> Dict[str, dict]:
        df = to_pandas_df(weights_df) if weights_df is not None else None
        feature_names = [] if df is None or df.empty or "feature" not in df.columns else pd.unique(df["feature"].astype(str)).tolist()
        stim_cols = [col for col in feature_names if col in _STIM_BIN_COLS]
        difficulty_cols = [col for col in feature_names if col in _DIFFICULTY_COLS]
        choice_cols = [col for col in feature_names if col in _CHOICE_LAG_COLS]
        bias_cols = _bias_hot_cols(feature_names)
        stim_groups = [
            (str(idx), [col])
            for idx, col in enumerate(stim_cols)
        ]
        return {
            "stim_hot": {
                "title": "stim_hot",
                "xlabel": "stimulus bin",
                "plot_kind": "box",
                "feature_groups": stim_groups,
            },
            "evidence_one_hot": {
                "title": "evidence one-hot",
                "xlabel": "stimulus bin",
                "plot_kind": "box",
                "feature_groups": stim_groups,
            },
            "difficulty_hot": {
                "title": "difficulty_hot",
                "xlabel": "Difficulty",
                "plot_kind": "box",
                "feature_groups": [
                    (col.removeprefix(_DIFFICULTY_COL_PREFIX), [col])
                    for col in difficulty_cols
                ],
            },
            "choice_lag": {
                "title": "choice_lag_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_PREFIX))), [col]) for col in choice_cols],
            },
            "at_choice_lag": {
                "title": "choice_lag_*",
                "xlabel": "Lag",
                "plot_kind": "box",
                "feature_groups": [(str(int(col.removeprefix(_CHOICE_LAG_PREFIX))), [col]) for col in choice_cols],
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

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df=None,
    ) -> Dict[str, List[str]]:
        ecols = list(emission_cols) if emission_cols is not None else self.default_emission_cols(df)
        ucols = list(transition_cols) if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(df))
        ecols = _drop_unavailable_bias_hot_cols(ecols, allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        allowed_ucols = self.available_transition_cols()
        bad_u = [c for c in ucols if c not in allowed_ucols]
        if bad_e:
            raise ValueError(
                f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}"
            )
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {allowed_ucols}"
            )
        return {"X_cols": list(ecols), "U_cols": list(ucols)}

    def cv_balance_labels(self, feature_df: pl.DataFrame):
        """Return signed evidence labels for balanced session-holdout CV."""
        if self.psychometric_x_col not in feature_df.columns:
            return None
        return feature_df[self.psychometric_x_col].cast(pl.Float64)

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

    @property
    def behavioral_cols(self) -> dict:
        return {
            "trial_idx": "trial_idx",
            "trial": "trial",
            "session": "session",
            "stimulus": "stimulus",
            "response": "response",
            "performance": "performance",
        }

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """Binary-task state labels using the task's native stimulus sign."""
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

            stim_fi = name2fi.get("stim_vals")
            if stim_fi is not None:
                stim_scores = W[:, 0, stim_fi]
            else:
                stim_scores = W[:, 0, :].mean(axis=1)

            engaged_k = int(np.argmax(stim_scores))
            others = [k for k in range(K) if k != engaged_k]
            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k, others[0]]
            elif K == 3:
                bias_fi = name2fi.get("bias")
                if bias_fi is not None:
                    bias_disp = W[others, 0, bias_fi]
                    biased_l = others[int(np.argmin(bias_disp))]
                    biased_r = others[int(np.argmax(bias_disp))]
                else:
                    biased_l, biased_r = others[0], others[1]
                labels[biased_l] = "Biased L"
                labels[biased_r] = "Biased R"
                order = [engaged_k, biased_l, biased_r]
            else:
                others_sorted = sorted(others, key=lambda k: stim_scores[k], reverse=True)
                for dis, k in enumerate(others_sorted, start=1):
                    labels[k] = f"Disengaged {dis}"
                order = [engaged_k] + others_sorted

            state_labels[subj] = labels
            state_order[subj] = order

        return state_labels, state_order
