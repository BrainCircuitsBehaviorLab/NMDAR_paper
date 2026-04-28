"""Task adapter for the MCDR (3-AFC rats) task."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import types
from typing import List, Tuple, Dict, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl

from ._choice_tau import load_subject_choice_half_life
from glmhmmt.runtime import get_data_dir, get_results_dir
from glmhmmt.tasks import TaskAdapter, _register, resolve_plots_module
from glmhmmt.tasks.fitted_regressors import (
    FittedWeightRegressorSpec,
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

_BIAS_HOT_COL_PREFIX = "bias_"
_CHOICE_LAG_COL_PREFIX = "choice_lag_"
_CHOICE_LAG_SIDES = ("L", "C", "R")
_CHOICE_LAG_REFERENCE_SIDE = "C"
_CHOICE_SIDE_TO_CLASS = {"L": 0, "C": 1, "R": 2}
_NUM_CHOICE_LAGS = 15
_STIM_PARAM_MODEL_ID = "One-hot"
_RAW_PARAM_MODEL_ID = "one hot"
_STIM_HOT_COLS = tuple(
    f"stim{stim_idx}{side}"
    for stim_idx in range(1, 5)
    for side in _CHOICE_LAG_SIDES
)

EMISSION_COLS: list[str] = [
    "bias",
    "bias_param",
    "biasL", "biasC", "biasR", "onsetL", "onsetC", "onsetR", "delay",
    "SL", "SC", "SR",
    "SLxdelay", "SCxdelay", "SRxdelay",
    "SLxD", "SCxD", "SRxD",
    "D", "DL", "DC", "DR",
    "A_L", "A_C", "A_R",
    "choice_lag_param",
    "stim_param",
    "speed1", "speed2", "speed3",
    "stim1L", "stim1C", "stim1R",
    "stim2L", "stim2C", "stim2R",
    "stim3L", "stim3C", "stim3R",
    "stim4L", "stim4C", "stim4R",
]

TRANSITION_COLS: list[str] = ["A_plus", "A_minus", "A_L", "A_C", "A_R"]

_BIAS_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="bias_param",
    fit_task="MCDR",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_BIAS_HOT_COL_PREFIX,),
    class_idx=0,
)
_CHOICE_LAG_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="choice_lag_param",
    fit_task="MCDR",
    fit_model_kind="glm",
    fit_model_id=_RAW_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_feature_prefixes=(_CHOICE_LAG_COL_PREFIX,),
    class_idx=0,
)
_STIM_PARAM_SPEC = FittedWeightRegressorSpec(
    target_name="stim_param",
    fit_task="MCDR",
    fit_model_kind="glm",
    fit_model_id=_STIM_PARAM_MODEL_ID,
    arrays_suffix="glm_arrays.npz",
    source_features=_STIM_HOT_COLS,
    class_idx=0,
)

_EMISSION_GROUPS: list[dict] = [
    {"key": "bias", "label": "bias", "members": {"N": "bias"}},
    {"key": "bias_param", "label": "bias param", "members": {"N": "bias_param"}},
    {"key": "bias_side", "label": "bias side", "members": {"L": "biasL", "C": "biasC", "R": "biasR"}},
    {"key": "onset", "label": "onset", "members": {"L": "onsetL", "C": "onsetC", "R": "onsetR"}},
    {"key": "delay", "label": "delay", "members": {"N": "delay"}},
    {"key": "S", "label": "S", "members": {"L": "SL", "C": "SC", "R": "SR"}},
    {"key": "SxDelay", "label": "S×delay", "members": {"L": "SLxdelay", "C": "SCxdelay", "R": "SRxdelay"}},
    {"key": "SxD", "label": "S×D", "members": {"L": "SLxD", "C": "SCxD", "R": "SRxD"}},
    {"key": "D", "label": "D (type)", "members": {"N": "D"}},
    {"key": "D_side", "label": "D side", "members": {"L": "DL", "C": "DC", "R": "DR"}},
    {"key": "A", "label": "A (action)", "members": {"L": "A_L", "C": "A_C", "R": "A_R"}},
    {"key": "choice_lag_param", "label": "choice lag param", "members": {"N": "choice_lag_param"}},
    {"key": "stim_param", "label": "stim param", "members": {"N": "stim_param"}},
    {"key": "speed1", "label": "speed 1", "members": {"N": "speed1"}},
    {"key": "speed2", "label": "speed 2", "members": {"N": "speed2"}},
    {"key": "speed3", "label": "speed 3", "members": {"N": "speed3"}},
    {"key": "stim1", "label": "stim 1", "members": {"L": "stim1L", "C": "stim1C", "R": "stim1R"}},
    {"key": "stim2", "label": "stim 2", "members": {"L": "stim2L", "C": "stim2C", "R": "stim2R"}},
    {"key": "stim3", "label": "stim 3", "members": {"L": "stim3L", "C": "stim3C", "R": "stim3R"}},
    {"key": "stim4", "label": "stim 4", "members": {"L": "stim4L", "C": "stim4C", "R": "stim4R"}},
]

_TRANSITION_GROUPS: list[dict] = [
    {"key": "A_plus", "label": "A+", "members": {"N": "A_plus"}},
    {"key": "A_minus", "label": "A−", "members": {"N": "A_minus"}},
    {"key": "A_trans", "label": "A (action)", "members": {"L": "A_L", "C": "A_C", "R": "A_R"}},
]


def _safe_weighted_sum_regressor(
    part,
    spec: FittedWeightRegressorSpec,
) -> np.ndarray | None:
    try:
        return weighted_sum_regressor(part, spec, dtype=np.float32)
    except (FileNotFoundError, ValueError):
        return None


def _bias_hot_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_BIAS_HOT_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _choice_lag_sort_key(name: str) -> tuple[int, int, str]:
    suffix = name.removeprefix(_CHOICE_LAG_COL_PREFIX)
    if len(suffix) >= 2 and suffix[:-1].isdigit() and suffix[-1] in _CHOICE_LAG_SIDES:
        return (int(suffix[:-1]), _CHOICE_LAG_SIDES.index(suffix[-1]), name)
    return (10**9, 10**9, name)


def _stim_hot_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("stim") and len(name) >= 6 and name[4].isdigit() and name[-1] in _CHOICE_LAG_SIDES:
        return (int(name[4]), _CHOICE_LAG_SIDES.index(name[-1]), name)
    return (10**9, 10**9, name)


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
            and len(col.removeprefix(_CHOICE_LAG_COL_PREFIX)) >= 2
            and col.removeprefix(_CHOICE_LAG_COL_PREFIX)[:-1].isdigit()
            and col.removeprefix(_CHOICE_LAG_COL_PREFIX)[-1] in _CHOICE_LAG_SIDES
        ],
        key=_choice_lag_sort_key,
    )


def _reference_coded_choice_lag_cols(columns: list[str]) -> list[str]:
    return [
        col
        for col in _choice_lag_cols(columns)
        if not col.endswith(_CHOICE_LAG_REFERENCE_SIDE)
    ]


def _reference_choice_lag_cols(columns: list[str]) -> list[str]:
    return [
        col
        for col in _choice_lag_cols(columns)
        if col.endswith(_CHOICE_LAG_REFERENCE_SIDE)
    ]


def _stim_hot_cols(columns: list[str]) -> list[str]:
    return sorted(
        [col for col in columns if col in _STIM_HOT_COLS],
        key=_stim_hot_sort_key,
    )


def _build_emission_groups(available_cols: list[str]) -> list[dict]:
    available = set(available_cols)
    result: list[dict] = []
    registered: set[str] = set()

    def add_group(group: dict) -> None:
        filtered = {k: v for k, v in group["members"].items() if v in available}
        if filtered:
            result.append({**group, "members": filtered})
            registered.update(filtered.values())

    bias_hot_cols = _bias_hot_cols(available_cols)
    stim_hot_cols = _stim_hot_cols(available_cols)
    choice_lag_cols = _choice_lag_cols(available_cols)
    choice_lag_toggle_cols = _reference_coded_choice_lag_cols(available_cols)
    choice_lag_exclude_cols = _reference_choice_lag_cols(available_cols)

    for group in _EMISSION_GROUPS:
        add_group(group)
        if group["key"] == "bias":
            if bias_hot_cols:
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
    if stim_hot_cols:
        stim_hot_group = {
            "key": "stim_hot",
            "label": "stim one-hot",
            "members": {},
            "toggle_members": list(stim_hot_cols),
            "hide_members": True,
        }
        result.append(stim_hot_group)
        result.append({**stim_hot_group, "key": "stim_one_hot"})
        registered.update(stim_hot_cols)

    grouped_choice_lags: dict[str, dict[str, str]] = {}
    for col in choice_lag_cols:
        suffix = col.removeprefix(_CHOICE_LAG_COL_PREFIX)
        lag_token, side = suffix[:-1], suffix[-1]
        grouped_choice_lags.setdefault(lag_token, {})[side] = col
        registered.add(col)

    # for lag_token in sorted(grouped_choice_lags, key=int):
    #     result.append(
    #         {
    #             "key": f"choice_lag_{lag_token}",
    #             "label": f"choice lag {lag_token}",
    #             "members": grouped_choice_lags[lag_token],
    #         }
    #     )

    if choice_lag_cols:
        result.append(
            {
                "key": "choice_lag",
                "label": "choice lag",
                "members": {},
                "toggle_members": list(choice_lag_toggle_cols),
                "exclude_members": list(choice_lag_exclude_cols),
                "hide_members": True,
            }
        )

    remaining = [col for col in available_cols if col not in registered]
    if remaining:
        result.extend(_build_selector_groups(remaining, []))
    return result


def _choice_lag_names(*, include_reference: bool = True) -> list[str]:
    sides = (
        _CHOICE_LAG_SIDES
        if include_reference
        else tuple(side for side in _CHOICE_LAG_SIDES if side != _CHOICE_LAG_REFERENCE_SIDE)
    )
    return [
        f"{_CHOICE_LAG_COL_PREFIX}{lag_idx:02d}{side}"
        for lag_idx in range(1, _NUM_CHOICE_LAGS + 1)
        for side in sides
    ]


def _max_sessions_from_df(df: pl.DataFrame) -> int:
    if "session" not in df.columns:
        return _max_subject_sessions()
    if "subject" not in df.columns:
        return int(df["session"].n_unique())
    return int(
        df.group_by("subject")
        .agg(pl.col("session").n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


def _infer_bias_hot_cols_from_df(df: pl.DataFrame) -> list[str]:
    existing = _bias_hot_cols(list(df.columns))
    if existing:
        return existing
    max_sessions = _max_sessions_from_df(df)
    return [f"{_BIAS_HOT_COL_PREFIX}{idx}" for idx in range(max_sessions)]


@lru_cache(maxsize=1)
def _max_subject_sessions() -> int:
    dataset_path = get_data_dir() / "df_filtered.parquet"
    df = pl.read_parquet(dataset_path)
    df = df.filter(pl.col("subject") != "A84")
    return int(
        df.group_by("subject")
        .agg(pl.col("session").n_unique().alias("n_sessions"))
        .select(pl.col("n_sessions").max())
        .item()
        or 0
    )


def _config_has_choice_lag_family(cfg: dict[str, Any]) -> bool:
    emission_cols = [str(col) for col in (cfg.get("emission_cols") or [])]
    return any(
        col.startswith(_CHOICE_LAG_COL_PREFIX) or col == "choice_lag_param"
        for col in emission_cols
    )


def _choice_lag_config_sort_key(path: Path, cfg: dict[str, Any]) -> tuple[int, int, str]:
    emission_cols = [str(col) for col in cfg.get("emission_cols", [])]
    model_id = str(cfg.get("model_id", path.parent.name))
    choice_lag_count = sum(col.startswith(_CHOICE_LAG_COL_PREFIX) for col in emission_cols)
    exact_model_match = 0 if model_id == _RAW_PARAM_MODEL_ID or path.parent.name == _RAW_PARAM_MODEL_ID else 1
    return (exact_model_match, -choice_lag_count, model_id)


def _resolve_choice_action_half_life(
    *,
    subject: str | None,
    default_half_life: float,
    results_dir: Path | None = None,
) -> float:
    subject_half_life = load_subject_choice_half_life(
        task_key="MCDR",
        fit_model_id=_RAW_PARAM_MODEL_ID,
        subject=subject,
    )
    if subject_half_life is not None:
        return float(subject_half_life)

    fits_root = (results_dir or get_results_dir()) / "fits" / "MCDR" / "glm"
    if not fits_root.exists():
        return float(default_half_life)

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for cfg_path in fits_root.glob("*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _config_has_choice_lag_family(cfg):
            candidates.append((cfg_path, cfg))

    if not candidates:
        return float(default_half_life)

    cfg_path, cfg = min(
        candidates,
        key=lambda item: _choice_lag_config_sort_key(item[0], item[1]),
    )
    del cfg_path
    tau_value = cfg.get("tau")
    if tau_value is None:
        return float(default_half_life)
    try:
        tau_float = float(tau_value)
    except (TypeError, ValueError):
        return float(default_half_life)
    return tau_float if np.isfinite(tau_float) and tau_float > 0.0 else float(default_half_life)


from src.process.common import (
    PreparedWeightFamilyPlot,
    attach_group_quantile_bin_column,
    attach_quantile_bin_column,
    attach_response_right_column,
    display_regressor_name,
    p_right_label,
    prepare_grouped_weight_family_plot,
    prepare_weight_family_base_df,
    prepare_simple_regressor_curve,
    summarize_grouped_panel,
    to_pandas_df,
)

PRED_COL = "pR"
RESPONSE_MODE = "mcdr_3class"
BASELINE = 1.0 / 3.0


def prepare_predictions_df(df_pred: pl.DataFrame, *, cfg) -> pl.DataFrame:
    """Prepare a canonical MCDR trial-level predictions dataframe."""
    df = df_pred.clone() if isinstance(df_pred, pl.DataFrame) else pl.from_pandas(df_pred)

    if "correct_bool" not in df.columns:
        if "performance" in df.columns:
            df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
        else:
            raise ValueError("No encuentro 'performance' ni 'correct_bool' en df.")

    for col in ["pL", "pC", "pR"]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en df (predicciones por trial).")
    if "response" not in df.columns:
        raise ValueError("Falta la columna 'response' (0/1/2) en df.")

    if "p_model_correct" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("stimulus") == 0)
            .then(pl.col("pL"))
            .when(pl.col("stimulus") == 1)
            .then(pl.col("pC"))
            .when(pl.col("stimulus") == 2)
            .then(pl.col("pR"))
            .otherwise(None)
            .alias("p_model_correct")
        )

    if "stimd_c" not in df.columns:
        if "stimd_n" in df.columns:
            df = df.with_columns(pl.col("stimd_n").replace(cfg["encoding"]["stimd"], default=None).alias("stimd_c"))
        else:
            raise ValueError("Falta 'stimd_c' y no existe 'stimd_n' para mapear.")

    if "ttype_c" not in df.columns:
        if "ttype_n" in df.columns:
            df = df.with_columns(pl.col("ttype_n").replace(cfg["encoding"]["ttype"], default=None).alias("ttype_c"))
        else:
            raise ValueError("Falta 'ttype_c' y no existe 'ttype_n' para mapear.")

    return df


def _prepare_mcdr_side_family_plot(
    weights_df,
    *,
    level_groups: list[tuple[str, dict[str, str]]],
    title: str,
    xlabel: str,
    variant: str = "folded",
    positive_label: str = "coh",
    neutral_label: str = "C",
    negative_label: str = "incoh",
) -> PreparedWeightFamilyPlot | None:
    df = prepare_weight_family_base_df(weights_df, weight_row_indices=(0, 1))
    if df.empty or "weight_row_idx" not in df.columns:
        return None

    feature_meta: dict[str, tuple[str, str]] = {}
    level_order: list[str] = []
    for level_label, members in level_groups:
        level_key = str(level_label)
        level_order.append(level_key)
        for side, feature in members.items():
            feature_meta[str(feature)] = (level_key, str(side))

    if not feature_meta:
        return None

    df = df[df["feature"].isin(feature_meta)].copy()
    if df.empty:
        return None

    df[["x_label", "side"]] = df["feature"].map(feature_meta).apply(pd.Series)
    df["weight_row_idx"] = pd.to_numeric(df["weight_row_idx"], errors="coerce")
    df = df[df["weight_row_idx"].isin([0, 1])].copy()
    if df.empty:
        return None

    pivoted = (
        df.groupby(["subject", "x_label", "side", "weight_row_idx"], as_index=False, observed=False)["weight"]
        .mean()
        .pivot(index=["subject", "x_label", "side"], columns="weight_row_idx", values="weight")
        .reset_index()
    )
    for row_idx in (0, 1):
        if row_idx not in pivoted.columns:
            pivoted[row_idx] = np.nan
    pivoted = pivoted.dropna(subset=[0, 1]).copy()
    if pivoted.empty:
        return None

    records: list[dict[str, object]] = []
    for _, row in pivoted.iterrows():
        subject = str(row["subject"])
        x_label = str(row["x_label"])
        side = str(row["side"])
        left_weight = float(row[0])
        right_weight = float(row[1])
        if side == "L":
            records.append({"subject": subject, "x_label": x_label, "group": positive_label, "weight": left_weight})
            records.append({"subject": subject, "x_label": x_label, "group": negative_label, "weight": right_weight})
        elif side == "R":
            records.append({"subject": subject, "x_label": x_label, "group": positive_label, "weight": right_weight})
            records.append({"subject": subject, "x_label": x_label, "group": negative_label, "weight": left_weight})
        elif side == "C":
            records.append({"subject": subject, "x_label": x_label, "group": neutral_label, "weight": (left_weight + right_weight) / 2.0})

    if not records:
        return None

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return None

    if variant == "split":
        split_order = [
            f"{x_label} {group}"
            for x_label in level_order
            for group in (positive_label, neutral_label, negative_label)
        ]
        out["x_label"] = out["x_label"].astype(str) + " " + out["group"].astype(str)
        out = (
            out.groupby(["subject", "x_label"], as_index=False, observed=False)["weight"]
            .mean()
        )
        present = set(out["x_label"].astype(str))
        return PreparedWeightFamilyPlot(
            data=out,
            plot_kind="box",
            title=f"{title} (split {positive_label}/{neutral_label}/{negative_label})",
            xlabel=xlabel,
            x_order=tuple(label for label in split_order if label in present),
        )

    if variant != "folded":
        raise ValueError(f"Unknown MCDR one-hot variant {variant!r}.")

    out = out[out["group"].isin([positive_label, negative_label])].copy()
    if out.empty:
        return None
    out["weight"] = np.where(
        out["group"] == negative_label,
        -out["weight"].to_numpy(dtype=float),
        out["weight"].to_numpy(dtype=float),
    )
    out = (
        out.groupby(["subject", "x_label"], as_index=False, observed=False)["weight"]
        .mean()
    )
    present = set(out["x_label"].astype(str))
    return PreparedWeightFamilyPlot(
        data=out,
        plot_kind="box",
        title=f"{title} (folded {positive_label}/{negative_label})",
        xlabel=xlabel,
        x_order=tuple(label for label in level_order if label in present),
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
    cfg,
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
    if df_pd.empty:
        return None, None

    panels: list[dict] = []

    panels.append(
        {
            "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col="ttype_c",
                subject_col="subject",
                data_col="correct_bool",
                model_col="p_model_correct",
                line_order=reg_bin_labels,
                x_order=list(cfg["plots"]["ttype"]["order"]),
            ),
            "meta": {
                "xlabel":"Trial difficulty",
                "ylabel":"Accuracy",
                "legend_title":display_regressor_name(regressor_col),
                "baseline":BASELINE,
                "x_order":list(cfg["plots"]["ttype"]["order"]),
                "x_tick_labels":list(cfg["plots"]["ttype"]["labels"]),
                "categorical_x":True,
                },
        }
    )

    panels.append(
        {
            "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col="stimd_c",
                subject_col="subject",
                data_col="correct_bool",
                model_col="p_model_correct",
                line_order=reg_bin_labels,
                x_order=list(cfg["plots"]["stimd"]["order"]),
                base_filter=df_pd["ttype_c"] == "DS",
            ),
            "meta": {
                "xlabel":"Stimulus type",
                "ylabel":"Accuracy",
                "legend_title":display_regressor_name(regressor_col),
                "baseline":BASELINE,
                "x_order":list(cfg["plots"]["stimd"]["order"]),
                "x_tick_labels":list(cfg["plots"]["stimd"]["labels"]),
                "categorical_x":True,
            },
        }
    )

    panels.append(
        {
            "summary": summarize_grouped_panel(
                df_pd,
                line_group_col="_reg_bin",
                x_col="ttype_c",
                subject_col="subject",
                data_col="correct_bool",
                model_col="p_model_correct",
                line_order=reg_bin_labels,
                x_order=list(cfg["plots"]["delay"]["order"]),
                base_filter=df_pd["stimd_c"] == "SS",
            ),
            "meta": {
                "xlabel":"Delay type",
                "ylabel":"Accuracy",
                "legend_title":display_regressor_name(regressor_col),
                "baseline":BASELINE,
                "x_order":list(cfg["plots"]["delay"]["order"]),
                "x_tick_labels":list(cfg["plots"]["delay"]["labels"]),
                "categorical_x":True,
            },
        }
    )

    return panels, display_regressor_name(regressor_col)


def prepare_categorical_performance_df(trial_df) -> pl.DataFrame:
    df = trial_df if isinstance(trial_df, pl.DataFrame) else pl.from_pandas(trial_df)
    if "p_model_correct_marginal" in df.columns:
        if "p_model_correct" in df.columns:
            df = df.drop("p_model_correct")
        df = df.rename({"p_model_correct_marginal": "p_model_correct"})
    return df


def prepare_cat_panel_payload(
    trial_df,
    *,
    group_col: str,
    order: list,
) -> dict | None:
    df = trial_df if isinstance(trial_df, pl.DataFrame) else pl.from_pandas(trial_df)
    df_pd = df.filter(pl.col(group_col).is_in(order)).to_pandas()
    if df_pd.empty:
        return None

    subj = (
        df_pd.groupby([group_col, "subject"], observed=True)
        .agg(correct_mean=("correct_bool", "mean"), model_mean=("p_model_correct", "mean"))
        .reset_index()
    )
    if subj.empty:
        return None

    grouped = (
        subj.groupby(group_col, observed=True)
        .agg(
            md=("correct_mean", "mean"),
            sd=("correct_mean", "std"),
            nd=("correct_mean", "count"),
            mm=("model_mean", "mean"),
            sm=("model_mean", "std"),
            nm=("model_mean", "count"),
        )
        .reset_index()
    )
    rows = grouped.set_index(group_col).to_dict("index")
    cats = [cat for cat in order if cat in rows]
    if not cats:
        return None
    return {
        "cats": cats,
        "md": np.array([rows[cat]["md"] for cat in cats], dtype=float),
        "sd": np.nan_to_num(np.array([rows[cat]["sd"] for cat in cats], dtype=float)),
        "nd": np.array([max(rows[cat]["nd"], 1) for cat in cats], dtype=float),
        "mm": np.array([rows[cat]["mm"] for cat in cats], dtype=float),
        "sm": np.nan_to_num(np.array([rows[cat]["sm"] for cat in cats], dtype=float)),
        "nm": np.array([max(rows[cat]["nm"], 1) for cat in cats], dtype=float),
        "n_subjects": int(subj["subject"].nunique()),
    }


def prepare_state_panel_payload(
    trial_df,
    *,
    group_col: str,
    order: list,
) -> dict | None:
    df = trial_df if isinstance(trial_df, pl.DataFrame) else pl.from_pandas(trial_df)
    df_pd = df.filter(pl.col(group_col).is_in(order)).to_pandas()
    if df_pd.empty:
        return None

    subj = (
        df_pd.groupby([group_col, "subject"], observed=True)
        .agg(acc=("correct_bool", "mean"), model=("p_model_correct", "mean"))
        .reset_index()
    )
    if subj.empty:
        return None

    agg = (
        subj.groupby(group_col, observed=True)
        .agg(md=("acc", "mean"), sd=("acc", "std"), mm=("model", "mean"), sm=("model", "std"))
        .reset_index()
    )
    rows = agg.set_index(group_col).to_dict("index")
    cats = [cat for cat in order if cat in rows]
    if not cats:
        return None
    return {
        "cats": cats,
        "xpos": np.array([order.index(cat) for cat in cats], dtype=float),
        "md": np.array([rows[cat]["md"] for cat in cats], dtype=float),
        "sd": np.nan_to_num(np.array([rows[cat]["sd"] for cat in cats], dtype=float)),
        "mm": np.array([rows[cat]["mm"] for cat in cats], dtype=float),
        "sm": np.nan_to_num(np.array([rows[cat]["sm"] for cat in cats], dtype=float)),
        "n_subjects": int(subj["subject"].nunique()),
    }


def prepare_delay_or_stim_1d_payload(trial_df, *, subject, n_bins: int, which: str) -> dict | None:
    df = to_pandas_df(trial_df)
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim = df.copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", "stim_d"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)

    if which == "delay":
        data = df_delay
        x_col = "delay_d"
        meta = {"xlabel": "Delay duration", "title_suffix": "Delay", "band_floor": BASELINE, "palette": "Purples_r"}
    elif which == "stim":
        data = df_stim
        x_col = "stim_d"
        meta = {"xlabel": "Stimulus duration", "title_suffix": "Stimulus", "band_floor": BASELINE, "palette": "Oranges"}
    else:
        raise ValueError("which must be 'delay' or 'stim'")

    if data.empty:
        return None

    data, centers = attach_quantile_bin_column(
        data,
        value_col=x_col,
        bin_col="x_bin",
        max_bins=n_bins,
        quantiles=None,
        center_col="center",
        center_agg="median",
    )
    if data is None:
        return None
    subj = (
        data.groupby(["x_bin", "subject"], observed=True)
        .agg(data_acc=("correct_bool", "mean"), model_acc=("p_model_correct", "mean"))
        .reset_index()
        .merge(centers, on="x_bin", how="left")
    )
    plot_df = subj.melt(
        id_vars=["x_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_df["kind"] = plot_df["kind"].map({"data_acc": "Data", "model_acc": "Model"})
    return {"plot_df": plot_df, "meta": meta}


def prepare_categorical_strat_by_side_payload(
    trial_df,
    *,
    df_silent=None,
    cond_col: str = "stimd_c",
    cond_order=None,
    cond_labels=None,
) -> dict:
    df = to_pandas_df(trial_df)
    df["x_c"] = df["x_c"].astype("string").str.strip().str.upper()

    if cond_order is None:
        cond_order = sorted(df[cond_col].dropna().unique())
    if cond_labels is None:
        cond_labels = cond_order

    summary = (
        df.groupby([cond_col, "x_c"], observed=True)
        .agg(
            data_mean=("correct_bool", "mean"),
            model_mean=("p_model_correct", "mean"),
            n=("correct_bool", "size"),
        )
        .reset_index()
    )
    summary["data_sem"] = np.sqrt(summary["data_mean"] * (1.0 - summary["data_mean"]) / summary["n"].clip(lower=1))
    summary["x_pos"] = summary[cond_col].map({cond: idx for idx, cond in enumerate(cond_order)})

    p_silent = None
    if df_silent is not None:
        p_silent = {"L": df_silent["pL_mean"], "C": df_silent["pC_mean"], "R": df_silent["pR_mean"]}

    return {
        "summary": summary,
        "p_silent": p_silent,
        "meta": {"cond_order": cond_order, "cond_labels": cond_labels, "baseline": BASELINE},
    }


def prepare_delay_binned_1d_payload(trial_df, *, subject=None, n_bins: int = 7) -> dict | None:
    df = to_pandas_df(trial_df)
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim = df[df["ttype_c"] == "DS"].copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", "stim_d"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)
    if df_delay.empty or df_stim.empty:
        return None

    df_delay = attach_group_quantile_bin_column(
        df_delay,
        value_col="delay_d",
        group_cols=["ttype_c"],
        bin_col="delay_bin",
        max_bins=n_bins,
    )
    if df_delay is None:
        return None
    centers_delay = (
        df_delay.groupby(["ttype_c", "delay_bin"], observed=True)["delay_d"].median().rename("center").reset_index()
    )
    subj_delay = (
        df_delay.groupby(["ttype_c", "delay_bin", "subject"], observed=True)
        .agg(data_acc=("correct_bool", "mean"), model_acc=("p_model_correct", "mean"))
        .reset_index()
        .merge(centers_delay, on=["ttype_c", "delay_bin"], how="left")
    )

    df_stim = attach_group_quantile_bin_column(
        df_stim,
        value_col="stim_d",
        group_cols=["stimd_c"],
        bin_col="stim_bin",
        max_bins=n_bins,
    )
    if df_stim is None:
        return None
    centers_stim = (
        df_stim.groupby(["stimd_c", "stim_bin"], observed=True)["stim_d"].median().rename("center").reset_index()
    )
    subj_stim = (
        df_stim.groupby(["stimd_c", "stim_bin", "subject"], observed=True)
        .agg(data_acc=("correct_bool", "mean"), model_acc=("p_model_correct", "mean"))
        .reset_index()
        .merge(centers_stim, on=["stimd_c", "stim_bin"], how="left")
    )

    plot_delay = subj_delay.melt(
        id_vars=["delay_bin", "subject", "ttype_c", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_delay["kind"] = plot_delay["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    plot_stim = subj_stim.melt(
        id_vars=["stim_bin", "subject", "center", "stimd_c"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    return {"plot_delay": plot_delay, "plot_stim": plot_stim}


def prepare_right_by_regressor(
    trial_df,
    *,
    regressor_col: str,
    cfg,
    xlabel: str | None = None,
    n_bins: int = 10,
):
    df_pd = to_pandas_df(trial_df)
    required = {regressor_col, "response", PRED_COL, "subject", "ttype_c", "stimd_c"}
    if not required.issubset(df_pd.columns):
        return None, None

    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd[PRED_COL] = pd.to_numeric(df_pd[PRED_COL], errors="coerce")
    df_pd = attach_response_right_column(df_pd, response_mode=RESPONSE_MODE)

    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd[PRED_COL])
        & np.isfinite(df_pd["_response_right"])
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

    delay_order = list(cfg["plots"]["delay"]["order"])

    summary = summarize_grouped_panel(
        df_pd,
        line_group_col="ttype_c",
        x_col="_reg_bin",
        subject_col="subject",
        data_col="_response_right",
        model_col=PRED_COL,
        line_order=delay_order,
        x_order=bin_order,
        base_filter=df_pd["stimd_c"] == "SS",
    )
    if summary.empty:
        return None, None

    summary = summary.merge(bin_centers, on="_reg_bin", how="left")

    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": p_right_label(),
        "legend_title": "Delay type",
        "baseline": BASELINE,
        "line_order": delay_order,
        "x_order": bin_order,
        "x_tick_labels": [f"{bin_center:.2f}" for bin_center in bin_centers["x_center"]],
        "categorical_x": True,
    }
    return summary, meta

@_register(["mcdr"])
class MCDRAdapter(TaskAdapter):
    """Adapter for the 3-AFC MCDR rat data."""

    task_key: str    = "MCDR"
    task_label: str  = "MCDR"
    num_classes: int = 3
    baseline_class_idx: int = 1
    data_file: str   = "df_filtered.parquet"
    sort_col: str    = "trial_idx"
    session_col: str = "session"

    # ── state-scoring options ────────────────────────────────────────────────
    # Each entry maps a label to a list of (feature_name, class_idx) pairs;
    # class_idx 0 = Left weight, 1 = Right weight.
    # The score for each state k is the mean of W[k, cls, feat_idx] over the
    # listed pairs.  States are then ranked highest-score → "Engaged".
    _SCORING_OPTIONS: dict = {
        "S_coh":     [("SL", 0), ("SR", 1)],
        "S1_coh":   [("stim1L", 0), ("stim1R", 1)],
        "S2_coh":   [("stim2L", 0), ("stim2R", 1)],
        "S3_coh":   [("stim3L", 0), ("stim3R", 1)],
        "S4_coh":   [("stim4L", 0), ("stim4R", 1)],
        "onset_coh": [("onsetL", 0), ("onsetR", 1)],
        "bias_coh":  [("biasL", 0), ("biasR", 1)],
    }
    scoring_key: str = "S_coh"

    # ── data preparation ────────────────────────────────────────────────────

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("subject") != "A84")

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        """Return the MCDR trial dataframe with all derived regressors."""
        df_sub = df_sub.sort(self.sort_col)
        subject = str(df_sub["subject"][0]) if "subject" in df_sub.columns and df_sub.height else None
        action_half_life = _resolve_choice_action_half_life(
            subject=subject,
            default_half_life=float(tau),
        )
        session_order = df_sub["session"].unique(maintain_order=True).to_list()
        session_to_idx = {session_name: idx for idx, session_name in enumerate(session_order)}
        max_sessions = len(session_order)
        session_idx_expr = pl.col("session").replace_strict(session_to_idx).cast(pl.Int32)
        df_sub = df_sub.with_columns(
            [
                session_idx_expr.alias("_session_idx"),
                ((pl.col("stimd_n") - pl.col("stimd_n").mean()) / pl.col("stimd_n").std()).alias("stimd_n_z"),
            ]
        )
        lag_exprs: list[pl.Expr] = []
        for lag_idx in range(1, _NUM_CHOICE_LAGS + 1):
            lagged_response = pl.col("response").shift(lag_idx).over(self.session_col)
            for side, class_idx in _CHOICE_SIDE_TO_CLASS.items():
                lag_exprs.append(
                    lagged_response.eq(class_idx).fill_null(False).cast(pl.Float32).alias(
                        f"{_CHOICE_LAG_COL_PREFIX}{lag_idx:02d}{side}"
                    )
                )
        session_bias_exprs = [
            pl.col("_session_idx").eq(idx).cast(pl.Float32).alias(f"{_BIAS_HOT_COL_PREFIX}{idx}")
            for idx in range(max_sessions)
        ]
        df_sub = df_sub.with_columns(
            [
                pl.col("response").cast(pl.Int32),
                (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
                (pl.col("x_c") == "C").cast(pl.Float32).alias("biasC"),
                (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
                pl.lit(1.0).cast(pl.Float32).alias("bias"),
                pl.col("delay_d").cast(pl.Float32).alias("delay"),
                ((pl.col("x_c") == "L") * pl.col("onset")).cast(pl.Float32).alias("onsetL"),
                ((pl.col("x_c") == "C") * pl.col("onset")).cast(pl.Float32).alias("onsetC"),
                ((pl.col("x_c") == "R") * pl.col("onset")).cast(pl.Float32).alias("onsetR"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SL"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SC"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SR"),
                ((pl.col("x_c") == "L") * pl.col("delay_d")).cast(pl.Float32).alias("DL"),
                ((pl.col("x_c") == "C") * pl.col("delay_d")).cast(pl.Float32).alias("DC"),
                ((pl.col("x_c") == "R") * pl.col("delay_d")).cast(pl.Float32).alias("DR"),
                pl.col("ttype_n").cast(pl.Float32).alias("D"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SLxD"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SCxD"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SRxD"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SLxdelay"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SCxdelay"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SRxdelay"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim1L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim1C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim1R"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim2L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim2C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim2R"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim3L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim3C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim3R"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim4L"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim4C"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim4R"),
                pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).over(self.session_col).alias("previous_outcome"),
                pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=action_half_life, adjust=False).over(self.session_col).alias("A_L"),
                pl.col("response").shift(1).fill_null(0.0).eq(1).cast(pl.Float32).ewm_mean(half_life=action_half_life, adjust=False).over(self.session_col).alias("A_C"),
                pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=action_half_life, adjust=False).over(self.session_col).alias("A_R"),
                (1 / (pl.col("timepoint_3") - pl.col("timepoint_4"))).cast(pl.Float32).alias("speed3"),
                (1 / (pl.col("timepoint_3") - pl.col("timepoint_2"))).cast(pl.Float32).alias("speed2"),
                (1 / (pl.col("timepoint_2") - pl.col("timepoint_1"))).cast(pl.Float32).alias("speed1"),
                *session_bias_exprs,
                *lag_exprs,
            ]
        )
        df_sub = df_sub.with_columns(
            [
                pl.col("previous_outcome").ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_plus"),
                (1.0 - pl.col("previous_outcome")).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_minus"),
            ]
        )
        df_sub = df_sub.with_columns(
            [
                ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).cast(pl.Float32).alias(c)
                for c in ["speed1", "speed2", "speed3"]
            ]
        ).drop("_session_idx")
        bias_param = _safe_weighted_sum_regressor(df_sub, _BIAS_PARAM_SPEC)
        choice_lag_param = _safe_weighted_sum_regressor(df_sub, _CHOICE_LAG_PARAM_SPEC)
        stim_param = _safe_weighted_sum_regressor(df_sub, _STIM_PARAM_SPEC)
        return df_sub.with_columns(
            [
                (
                    pl.Series("bias_param", bias_param)
                    if bias_param is not None
                    else pl.lit(0.0).cast(pl.Float32).alias("bias_param")
                ),
                (
                    pl.Series("choice_lag_param", choice_lag_param)
                    if choice_lag_param is not None
                    else pl.lit(0.0).cast(pl.Float32).alias("choice_lag_param")
                ),
                (
                    pl.Series("stim_param", stim_param)
                    if stim_param is not None
                    else pl.lit(0.0).cast(pl.Float32).alias("stim_param")
                ),
            ]
        )

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` from the MCDR feature dataframe."""
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
        """Return ``(y, X, U, names)`` from the MCDR feature dataframe."""
        ecols = emission_cols if emission_cols is not None else list(EMISSION_COLS)
        ucols = transition_cols if transition_cols is not None else list(TRANSITION_COLS)
        allowed_ecols = set(self.available_emission_cols(feature_df))
        ecols = _drop_unavailable_bias_hot_cols(list(ecols), allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}")

        y = jnp.asarray(feature_df["response"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {"X_cols": list(ecols), "U_cols": list(ucols)}
        return y, X, U, names

    # ── column defaults ─────────────────────────────────────────────────────

    def default_emission_cols(self, df=None) -> List[str]:
        cols = [
            "bias",
        ]
        if df is not None:
            cols.extend(self.stim_hot_cols(df))
            cols.extend(self.choice_lag_cols(df))
        return list(dict.fromkeys(cols))

    def default_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def available_emission_cols(self, df=None) -> List[str]:
        available_cols = list(EMISSION_COLS)
        if df is not None:
            available_cols.extend(self.bias_hot_cols(df))
            available_cols.extend(_choice_lag_cols(list(df.columns)) or _choice_lag_names())
            available_cols.extend(self.stim_hot_cols(df))
        return list(dict.fromkeys(available_cols))

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df=None,
    ) -> Dict[str, List[str]]:
        ecols = list(emission_cols) if emission_cols is not None else list(EMISSION_COLS)
        ucols = list(transition_cols) if transition_cols is not None else list(TRANSITION_COLS)
        allowed_ecols = set(self.available_emission_cols(df))
        ecols = _drop_unavailable_bias_hot_cols(ecols, allowed_ecols)
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}")
        return {"X_cols": ecols, "U_cols": ucols}

    def bias_hot_cols(self, df: pl.DataFrame) -> List[str]:
        """Return session one-hot bias columns."""
        return _infer_bias_hot_cols_from_df(df)

    def choice_lag_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        """Return reference-coded previous-choice lag columns."""
        if df is not None:
            existing = _reference_coded_choice_lag_cols(list(df.columns))
            if existing:
                return existing
        return _choice_lag_names(include_reference=False)

    def stim_hot_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        """Return stimulus-window one-hot columns."""
        if df is None:
            return list(_STIM_HOT_COLS)
        existing = _stim_hot_cols(list(df.columns))
        return existing if existing else list(_STIM_HOT_COLS)

    def weight_family_specs(self, weights_df=None) -> Dict[str, dict]:
        df = to_pandas_df(weights_df) if weights_df is not None else None
        feature_names = [] if df is None or df.empty or "feature" not in df.columns else pd.unique(df["feature"].astype(str)).tolist()
        stim_cols = _stim_hot_cols(feature_names)
        choice_cols = _reference_coded_choice_lag_cols(feature_names)
        bias_cols = _bias_hot_cols(feature_names)

        def _group_by_level(columns: list[str], prefix: str) -> list[tuple[str, dict[str, str]]]:
            grouped: dict[str, dict[str, str]] = {}
            for col in columns:
                suffix = col.removeprefix(prefix)
                if len(suffix) < 2:
                    continue
                lag_token, side = suffix[:-1], suffix[-1]
                grouped.setdefault(str(int(lag_token)), {})[side] = col
            return [(level, grouped[level]) for level in sorted(grouped, key=int)]

        return {
            "stim_hot": {
                "title": "stim one-hot",
                "xlabel": "Stimulus window",
                "levels": _group_by_level(stim_cols, "stim"),
                "variants": ("folded", "split"),
            },
            "stim_one_hot": {
                "title": "stim one-hot",
                "xlabel": "Stimulus window",
                "levels": _group_by_level(stim_cols, "stim"),
                "variants": ("folded", "split"),
            },
            "choice_lag": {
                "title": "choice_lag_*",
                "xlabel": "Lag",
                "levels": _group_by_level(choice_cols, _CHOICE_LAG_COL_PREFIX),
                "variants": ("folded", "split"),
                "positive_label": "repeat",
                "neutral_label": "C",
                "negative_label": "switch",
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
        spec = self.weight_family_specs(weights_df).get(family_key)
        if spec is None:
            return None
        if "levels" in spec:
            return _prepare_mcdr_side_family_plot(
                weights_df,
                level_groups=spec["levels"],
                title=spec["title"],
                xlabel=spec["xlabel"],
                variant=variant or "folded",
                positive_label=spec.get("positive_label", "coh"),
                neutral_label=spec.get("neutral_label", "C"),
                negative_label=spec.get("negative_label", "incoh"),
            )
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
        return _build_selector_groups(list(available_cols), _TRANSITION_GROUPS)

    @property
    def choice_labels(self) -> list[str]:
        return ["Left", "Center", "Right"]

    @property
    def probability_columns(self) -> list[str]:
        return ["pL", "pC", "pR"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        if "stimulus" in df.columns:
            vals = df["stimulus"].to_numpy().astype(int)
            unique = set(np.unique(vals).tolist())
            if unique.issubset({0, 1, 2}):
                return vals.astype(int)
            if unique.issubset({1, 2, 3}):
                return (vals - 1).astype(int)

        if "x_c" in df.columns:
            vals = df["x_c"].to_numpy()
            mapping = {"L": 0, "C": 1, "R": 2}
            out = np.array([mapping.get(str(v), -1) for v in vals], dtype=int)
            if np.any(out < 0):
                bad = sorted({str(v) for v, idx in zip(vals, out) if idx < 0})
                raise ValueError(f"Unexpected MCDR x_c values: {bad}")
            return out

        raise ValueError(
            "Could not derive MCDR correct class. Expected stimulus coded as "
            "0/1/2 or 1/2/3, or x_c coded as L/C/R."
        )
    # ── column mapping ───────────────────────────────────────────────────────

    @property
    def behavioral_cols(self) -> dict:
        """MCDR column mapping (canonical → actual)."""
        return {
            "trial_idx":   "trial_idx",
            "trial":       "trial",
            "session":     "session",
            "stimulus":    "stimulus",
            "response":    "response",
            "performance": "performance",
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
        """MCDR engagement scoring driven by ``self.scoring_key``.

        The state with the highest mean coherent weight (as defined by the
        selected scoring regressor) is labelled "Engaged"; the rest are
        "Disengaged" (or "Disengaged 1", "Disengaged 2", … for K>2).
        """
        import numpy as np

        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "S_coh"),
            self._SCORING_OPTIONS["S_coh"],
        )

        def _scoh(W, feat_names):
            name2fi = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(W.shape[0])
            n = 0
            for feat, cls in pairs:
                if feat in name2fi:
                    scores += W[:, cls, name2fi[feat]]
                    n += 1
            return scores / max(1, n)

        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict  = {}
        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj]  = list(range(K))
                continue
            W_np    = np.asarray(W)
            feat    = list(arrays_store[subj].get("X_cols", base_feat))
            scores  = _scoh(W_np, feat)
            ranking = list(np.argsort(scores)[::-1])
            engaged_k = int(ranking[0])
            others    = [int(k) for k in ranking[1:]]
            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k] + others

            elif K == 4:
                name2fi = {n: i for i, n in enumerate(feat)}
                sl_fi   = name2fi.get("SL")
                sr_fi   = name2fi.get("SR")

                # Disengaged L: state most driven by SL (left-choice weight)
                if sl_fi is not None:
                    dis_l = others[int(np.argmax(W_np[others, 0, sl_fi]))]
                else:
                    dis_l = others[0]
                remaining = [k for k in others if k != dis_l]

                # Disengaged R: state most driven by SR (right-choice weight)
                if sr_fi is not None:
                    dis_r = remaining[int(np.argmax(W_np[remaining, 1, sr_fi]))]
                else:
                    dis_r = remaining[0]
                dis_c = [k for k in remaining if k != dis_r][0]

                labels[dis_l] = "Disengaged L"
                labels[dis_r] = "Disengaged R"
                labels[dis_c] = "Disengaged C"
                order = [engaged_k, dis_l, dis_r, dis_c]

            else:
                dis = 1
                for k in others:
                    labels[k] = f"Disengaged {dis}"
                    dis += 1
                order = [engaged_k] + others

            state_labels[subj] = labels
            state_order[subj]  = order
        return state_labels, state_order
