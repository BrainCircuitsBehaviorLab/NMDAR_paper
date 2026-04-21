"""Task adapter for the 2AFC (Alexis human) task."""
from __future__ import annotations

from functools import lru_cache
import types
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import jax.numpy as jnp
import polars as pl

from ._choice_tau import compute_choice_ewma, load_subject_choice_half_life
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
    mean_feature_weights_from_fit,
    resolved_source_features,
    weighted_sum_regressor,
)
from glmhmmt.tasks import TaskAdapter, _register, resolve_plots_module
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
    attach_quantile_bin_column,
    attach_response_right_column,
    display_regressor_name,
    mean_glm_feature_curve as _mean_glm_feature_curve,
    mean_glm_ild_curve as _mean_glm_ild_curve,
    p_right_label,
    prepare_simple_regressor_curve,
    summarize_grouped_panel,
    subject_glm_feature_curves as _subject_glm_feature_curves,
    subject_glm_ild_curves as _subject_glm_ild_curves,
    to_pandas_df,
)


# Default experiments to keep (avoids habituation / drug sessions)
_KEEP_EXPERIMENTS = ["2AFC_2", "2AFC_3", "2AFC_4", "2AFC_6"]
_SF_COL_PREFIX = "sf_"
_STIM_ABS_COL_PREFIX = "stim_"
_BIAS_HOT_COL_PREFIX = "bias_"
_CHOICE_LAG_COL_PREFIX = "choice_lag_"
_NUM_CHOICE_LAGS = 15
_RAW_PARAM_MODEL_ID = "one hot"
EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "stim_vals",
    "stim_param",
    "stim_strength",
    "at_choice",
    "at_choice_param",
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
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
]
_STIM_PARAM_COL = "stim_param"
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

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim_vals": r"$\mathrm{Stimulus}$",
    "stim_param": r"$\mathrm{Stimulus}_{\mathrm{param}}$",
    "stim_strength": r"$\mathrm{Stimulus}_{\mathrm{strength}}$",
    "bias": r"$\mid\mathrm{bias}\mid$",
    "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
    "at_choice": r"$\mathrm{A}_t^{\mathrm{choice}}$",
    "at_choice_param": r"$\mathrm{A}_t^{\mathrm{choice,param}}$",
    "at_error": r"$\mathrm{A}_t^{\mathrm{error}}$",
    "at_correct": r"$\mathrm{A}_t^{\mathrm{correct}}$",
    "reward_trace": r"$\mathrm{Reward}_{\mathrm{trace}}$",
    "prev_choice": r"$\mathrm{PrevChoice}$",
    "prev_reward": r"$\mathrm{PrevReward}$",
    "prev_abs_stim": r"$|\mathrm{PrevStim}|$",
    "cumulative_reward": r"$\mathrm{CumReward}$",
    "wsls": r"$\mathrm{WSLS}$",
}

_EMISSION_GROUPS: list[dict] = [
    {"key": "bias", "label": "bias", "members": {"N": "bias"}},
    {"key": "bias_param", "label": "bias param", "members": {"N": "bias_param"}},
    {"key": "stim_vals", "label": "stim vals", "members": {"N": "stim_vals"}},
    {"key": "stim_param", "label": "stim param", "members": {"N": "stim_param"}},
    {"key": "stim_strength", "label": "stim strength", "members": {"N": "stim_strength"}},
    {"key": "at_choice", "label": "action (choice)", "members": {"N": "at_choice"}},
    {"key": "at_choice_param", "label": "choice param", "members": {"N": "at_choice_param"}},
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


def _bias_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_BIAS_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


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


def _choice_lag_names() -> list[str]:
    return [f"{_CHOICE_LAG_COL_PREFIX}{idx:02d}" for idx in range(1, _NUM_CHOICE_LAGS + 1)]


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
    bias_hot_cols = _bias_hot_cols(available_cols)
    choice_lag_cols = _choice_lag_cols(available_cols)

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
        if key == "at_choice":
            add_scalar(group)
            add_hidden_family(key="at_choice_lag", label="choice_lag", family_cols=choice_lag_cols)
            continue
        add_scalar(group)

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


def _build_stim_param(part: pd.DataFrame, stim_abs_levels: list[int]) -> np.ndarray:
    """Return the pooled one-hot stimulus contribution for each trial."""
    required_features = {
        f"{_STIM_ABS_COL_PREFIX}{stim_abs}"
        for stim_abs in stim_abs_levels
        if stim_abs != 0
    }
    source_features = set(resolved_source_features(_STIM_PARAM_SPEC))
    missing = sorted(required_features - source_features)
    if missing:
        raise ValueError(
            "stim_param is missing pooled weights for absolute ILD levels "
            f"{missing}. Available fitted features: {sorted(source_features)}"
        )
    return weighted_sum_regressor(part, _STIM_PARAM_SPEC, dtype=np.float32)

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
        right_logit_sign=-1.0,
    )


def subject_glm_ild_curves(arrays_store, subjects, X_cols, *, ild_max, state_k=None):
    return _subject_glm_ild_curves(
        arrays_store,
        subjects,
        X_cols,
        ild_max=ild_max,
        state_k=state_k,
        stim_param_weight_map=_stim_param_weight_map,
        right_logit_sign=-1.0,
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
        right_logit_sign=-1.0,
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
        right_logit_sign=-1.0,
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

    df_pd, bin_centers = attach_quantile_bin_column(df_pd, value_col=regressor_col, max_bins=4)
    if df_pd is None:
        return None, None
    reg_bin_labels = bin_centers["_reg_bin"].tolist()

    df_pd = attach_response_right_column(df_pd, response_mode=RESPONSE_MODE)
    if df_pd.empty:
        return None, None

    conds = sorted(df_pd["condition"].dropna().unique()) if "condition" in df_pd.columns else []
    exps = sorted(df_pd["experiment"].dropna().unique()) if "experiment" in df_pd.columns else []
    ild_ticks = sorted(pd.to_numeric(df_pd["ILD"], errors="coerce").dropna().unique()) if "ILD" in df_pd.columns else []

    panels: list[dict] = []

    panels.append(
        {
            "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col="ILD",
                subject_col="subject",
                data_col="_response_right",
                model_col=PRED_COL,
                line_order=reg_bin_labels,
            ),
            "meta": {
                "xlabel": "ILD (dB)",
                "ylabel": p_right_label(),
                "legend_title": display_regressor_name(regressor_col),
                "baseline": BASELINE,
                "xticks": ild_ticks,
            }
        }
    )

    for cond in conds:
        panels.append(
            {
                "summary": summarize_grouped_panel(
                    df_pd,
                    line_group_col="_reg_bin",
                    x_col="ILD",
                    subject_col="subject",
                    data_col="_response_right",
                    model_col=PRED_COL,
                    line_order=reg_bin_labels,
                    subgroup_col="condition",
                    subgroup_value=cond,
                ),
                "meta": {
                    "xlabel": "ILD (dB)",
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "xticks": ild_ticks,
                },
            }
        )

    for exp in exps:
        panels.append(
            {
                "summary": summarize_grouped_panel(
                    df_pd,
                    line_group_col="_reg_bin",
                    x_col="ILD",
                    subject_col="subject",
                    data_col="_response_right",
                    model_col=PRED_COL,
                    line_order=reg_bin_labels,
                    subgroup_col="experiment",
                    subgroup_value=exp,
                ),
                "meta": {
                    "xlabel": "ILD (dB)",
                    "ylabel": p_right_label(),
                    "legend_title": display_regressor_name(regressor_col),
                    "baseline": BASELINE,
                    "xticks": ild_ticks,
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
):
    df_pd = to_pandas_df(trial_df)
    required = {regressor_col, "response", PRED_COL, "subject", "ILD"}
    if not required.issubset(df_pd.columns):
        return None, None

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

    df_pd, bin_centers = attach_quantile_bin_column(df_pd, value_col=regressor_col, max_bins=n_bins)
    if df_pd is None:
        return None, None
    bin_order = bin_centers["_reg_bin"].tolist()

    ild_order = sorted(df_pd["ILD"].dropna().unique().tolist())

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
    if summary.empty:
        return None, None

    summary = summary.merge(bin_centers, on="_reg_bin", how="left")

    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": p_right_label(),
        "legend_title": "Signed ILD",
        "baseline": BASELINE,
        "line_order": ild_order,
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

    # ── state-scoring options ────────────────────────────────────────────────
    # For 2AFC the weight matrix is (K, 1, M) where W[k,0,:] = logit(Left)
    # weights (reference = Right).  The plot shows -W for intuition.
    # Modes:
    #   "neg"  – -W[k, 0, fi]  (more negative raw = more stimulus-following)
    #   "abs"  – |W[k, 0, fi]|  (unsigned magnitude)
    #   "pos"  – +W[k, 0, fi]  (raw positive = anti-stimulus tendency)
    # Score per state = mean over listed pairs.
    _SCORING_OPTIONS: dict = {
        "stim_vals (-w)": [("stim_vals", "neg")],
        "stim_vals (|w|)": [("stim_vals", "abs")],
        "stim_param (-w)": [("stim_param", "neg")],
        "stim_param (|w|)": [("stim_param", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_vals (-w)"

    # ── data preparation ────────────────────────────────────────────────────

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))

    def _build_feature_df(
        self,
        df_sub: pl.DataFrame,
        tau: float = 50.0,
        include_stim_strength: bool = False,
        include_stim_param: bool = False,
        include_bias_param: bool = False,
        include_at_choice_param: bool = False,
    ) -> pl.DataFrame:
        """Return the Alexis 2AFC feature dataframe owned by this adapter."""
        from glmhmmt.cli.alexis_functions import get_action_trace, make_frames_dm

        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["Session", "Trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)
        subject_half_life = load_subject_choice_half_life(
            task_key=self.task_key,
            fit_model_id=_RAW_PARAM_MODEL_ID,
            subject=str(df_pd["subject"].iloc[0]) if "subject" in df_pd.columns and len(df_pd) else None,
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
        max_sessions = _max_subject_sessions()
        session_order = list(dict.fromkeys(df_pd["Session"].tolist()))
        session_to_idx = {session_name: idx for idx, session_name in enumerate(session_order)}
        choice_lag_cols = _choice_lag_names()
        parts = []
        for _, df_session in df_pd.groupby("Session", sort=False):
            part = df_session.copy().reset_index(drop=True)
            session_idx = session_to_idx[df_session["Session"].iloc[0]]
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
            signed_choice = (2.0 * part["Choice"].fillna(0).astype(np.float32)) - 1.0

            choice_lag_df = pd.DataFrame(
                {
                    lag_col: signed_choice.shift(lag_idx).fillna(0.0).astype(np.float32)
                    for lag_idx, lag_col in enumerate(choice_lag_cols, start=1)
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
                        },
                        index=part.index,
                    ),
                    bias_hot,
                    pd.DataFrame(stim_hot_cols, index=part.index),
                    choice_lag_df,
                ],
                axis=1,
            )
            if include_stim_param:
                part[_STIM_PARAM_COL] = _build_stim_param(part, stim_abs_levels)

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
            cumulative_reward = part["Hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            derived_cols = pd.DataFrame(
                {
                    "at_choice": np.asarray(at_choice, dtype=np.float32),
                    "at_error": np.asarray(at_error, dtype=np.float32),
                    "at_correct": np.asarray(at_correct, dtype=np.float32),
                    "reward_trace": np.asarray(reward_trace, dtype=np.float32),
                    "prev_choice": part["Choice"].shift(1).fillna(0).astype(np.float32),
                    "prev_reward": part["Hit"].shift(1).fillna(0).astype(np.float32),
                    "cumulative_reward": cumulative_reward.astype(np.float32),
                    "prev_abs_stim": (part["ILD"].abs().shift(1).fillna(0) / stim_scale).astype(np.float32),
                    "wsls": part["Side"].shift(1).fillna(0).replace({0: -1, 1: 1}).astype(np.float32),
                },
                index=part.index,
            )
            part = pd.concat([part, derived_cols], axis=1)
            if include_bias_param:
                try:
                    bias_param = weighted_sum_regressor(part, _BIAS_PARAM_SPEC, dtype=np.float32)
                except (FileNotFoundError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot build {_BIAS_PARAM_SPEC.target_name!r}; pooled fitted weights are unavailable "
                        f"for {_BIAS_PARAM_SPEC.fit_task}/{_BIAS_PARAM_SPEC.fit_model_kind}/{_BIAS_PARAM_SPEC.fit_model_id}."
                    ) from exc
                part = pd.concat(
                    [part, pd.DataFrame({"bias_param": bias_param}, index=part.index)],
                    axis=1,
                )
            if include_at_choice_param:
                try:
                    at_choice_param = weighted_sum_regressor(part, _AT_CHOICE_PARAM_SPEC, dtype=np.float32)
                except (FileNotFoundError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot build {_AT_CHOICE_PARAM_SPEC.target_name!r}; pooled fitted weights are unavailable "
                        f"for {_AT_CHOICE_PARAM_SPEC.fit_task}/{_AT_CHOICE_PARAM_SPEC.fit_model_kind}/{_AT_CHOICE_PARAM_SPEC.fit_model_id}."
                    ) from exc
                part = pd.concat(
                    [part, pd.DataFrame({"at_choice_param": at_choice_param}, index=part.index)],
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
        for col in requested:
            if col == "stim_strength":
                if not dynamic_sf_cols:
                    raise ValueError(
                        "Requested emission col 'stim_strength', but no frame-level "
                        f"'{_SF_COL_PREFIX}*' columns are available for 2AFC."
                    )
                resolved.extend(dynamic_sf_cols)
            else:
                resolved.append(col)
        return resolved

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
        feature_df = self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=include_stim_strength,
            include_stim_param=include_stim_param,
            include_bias_param=include_bias_param,
            include_at_choice_param=include_at_choice_param,
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
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}"
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
            for c in EMISSION_COLS
            if c not in {"stim_strength", _STIM_PARAM_COL, "bias_param", "at_choice_param"}
        ]
        if df is not None:
            default_cols.extend(self.sf_cols(df))
        return list(dict.fromkeys(default_cols))

    def default_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        available_cols = list(EMISSION_COLS)
        if df is not None:
            available_cols.extend(self.sf_cols(df))
            available_cols.extend(self.stim_abs_cols(df))
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

        resolved_ecols: list[str] = []
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
                resolved_ecols.append(col)

        allowed_ecols = set(self.available_emission_cols(df))
        bad_e = [c for c in resolved_ecols if c not in allowed_ecols]
        bad_u = [c for c in requested_ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}"
            )
        return {"X_cols": list(resolved_ecols), "U_cols": list(requested_ucols)}

    def sf_cols(self, df: pl.DataFrame) -> List[str]:
        """Return any stimulus-frame (sf_*) columns present in *df*."""
        return [c for c in df.columns if c.startswith(_SF_COL_PREFIX)]

    def stim_abs_cols(self, df: pl.DataFrame) -> List[str]:
        """Return signed one-hot columns for absolute ILD magnitudes."""
        return _infer_stim_abs_cols_from_df(df)

    def bias_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return subject-local session one-hot columns."""
        return _infer_bias_hot_cols_from_df(df)

    def choice_lag_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        """Return the previous-choice one-hot lag columns."""
        if df is not None:
            existing = _choice_lag_cols(list(df.columns))
            if existing:
                return existing
        return _choice_lag_names()

    def build_emission_groups(self, available_cols: List[str]) -> list[dict]:
        return _build_emission_groups(list(available_cols))

    def build_transition_groups(self, available_cols: List[str]) -> list[dict]:
        del available_cols
        return []

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

    # ── plots ────────────────────────────────────────────────────────────────

    def get_plots(self) -> types.ModuleType:
        return resolve_plots_module(
            adapter_module_name=__name__,
            task_key=self.task_key,
        )
    # ── state labelling ─────────────────────────────────────────────────────

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """2AFC engagement scoring.

        K=2: Engaged = argmax(selected score), Disengaged = the other.
        K=3: Engaged = argmax(selected score); the remaining two are split
             by bias weight: min(displayed bias) = "Biased L",
             max(displayed bias) = "Biased R".
        K>3: remaining states labelled "Disengaged 1", "Disengaged 2", ...
             ordered by descending selected score.
        """
        import numpy as np

        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "stim_vals (-w)"),
            self._SCORING_OPTIONS["stim_vals (-w)"],
        )

        def _score_states(
            W_np: np.ndarray,
            feat_names: list[str],
            *,
            stim: str = "stim_vals",
        ) -> np.ndarray:
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
                    raise ValueError(f"Unknown 2AFC scoring mode {mode!r}.")
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
        state_order: dict  = {}

        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj]  = list(range(K))
                continue

            feat    = list(arrays_store[subj].get("X_cols", base_feat))
            W       = np.asarray(W)   # (K, 1, M)
            name2fi = {n: i for i, n in enumerate(feat)}

            selected_stim = "stim_param" if getattr(self, "scoring_key", "").startswith("stim_param") else "stim_vals"
            state_scores = _score_states(W, feat, stim=selected_stim)

            engaged_k = int(np.argmax(state_scores))
            others    = [k for k in range(K) if k != engaged_k]

            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k, others[0]]

            elif K == 3:
                bias_fi = name2fi.get("bias")
                if bias_fi is not None:
                    # displayed bias = -raw; lower displayed = more left-biased
                    bias_disp = -W[others, 0, bias_fi]
                    biased_l = others[int(np.argmin(bias_disp))]
                    biased_r = others[int(np.argmax(bias_disp))]
                else:
                    biased_l, biased_r = others[0], others[1]
                labels[biased_l] = "Biased L"
                labels[biased_r] = "Biased R"
                order = [engaged_k, biased_l, biased_r]

            else:
                # K>3: rank remaining by selected score descending
                others_sorted = sorted(others, key=lambda k: state_scores[k], reverse=True)
                for dis, k in enumerate(others_sorted, start=1):
                    labels[k] = f"Disengaged {dis}"
                order = [engaged_k] + others_sorted

            state_labels[subj] = labels
            state_order[subj]  = order

        return state_labels, state_order
