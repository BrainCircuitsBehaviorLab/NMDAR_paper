from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
import polars as pl
from typing import Callable, Literal, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PreparedWeightFamilyPlot:
    data: pd.DataFrame
    plot_kind: Literal["box", "line"]
    title: str
    xlabel: str
    ylabel: str = "Weight"
    x_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LapseLogisticFit:
    group: object
    slope: float
    bias: float
    lapse_left: float
    lapse_right: float
    x_fit: np.ndarray
    y_fit: np.ndarray
    n_points: int
    success: bool = True


@dataclass(frozen=True)
class TaskPlotColumns:
    """Canonical columns used by task-owned payload builders and plots."""

    response_col: str = "response"
    prediction_col: str = "p_pred"
    correct_col: str = "correct_bool"
    model_correct_col: str = "p_model_correct"
    psychometric_x_col: str = "stimulus"
    psychometric_x_label: str = "Stimulus"
    subject_col: str = "subject"
    response_mode: str = "pm1_or_prob"
    baseline: float = 0.5


def resolve_task_plot_columns(
    adapter=None,
    *,
    response_col: str | None = None,
    prediction_col: str | None = None,
    correct_col: str | None = None,
    model_correct_col: str | None = None,
    psychometric_x_col: str | None = None,
    psychometric_x_label: str | None = None,
    subject_col: str | None = None,
    response_mode: str | None = None,
    baseline: float | None = None,
) -> TaskPlotColumns:
    """Resolve plotting columns from an adapter plus optional overrides."""

    return TaskPlotColumns(
        response_col=response_col or getattr(adapter, "response_col", "response"),
        prediction_col=prediction_col or getattr(adapter, "prediction_col", "p_pred"),
        correct_col=correct_col or getattr(adapter, "correct_bool_col", "correct_bool"),
        model_correct_col=model_correct_col or getattr(adapter, "model_correct_col", "p_model_correct"),
        psychometric_x_col=psychometric_x_col or getattr(adapter, "psychometric_x_col", "stimulus"),
        psychometric_x_label=psychometric_x_label or getattr(adapter, "psychometric_x_label", "Stimulus"),
        subject_col=subject_col or "subject",
        response_mode=response_mode or getattr(adapter, "response_mode", "pm1_or_prob"),
        baseline=float(baseline if baseline is not None else getattr(adapter, "baseline", 0.5)),
    )


def _iter_column_candidates(candidates: Sequence[object]):
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str):
            yield candidate
        elif isinstance(candidate, Sequence):
            yield from _iter_column_candidates(candidate)


def pick_existing_column(df_like, candidates: Sequence[object]) -> str | None:
    """Return the first candidate column present in a pandas/polars dataframe."""

    columns = set(getattr(df_like, "columns", []))
    for candidate in _iter_column_candidates(candidates):
        if candidate in columns:
            return candidate
    return None


def adapter_behavioral_column(adapter, df_like, key: str, *fallbacks: object) -> str | None:
    """Resolve a raw behavioral column, preferring the adapter mapping."""

    behavioral_cols = dict(getattr(adapter, "behavioral_cols", {}) or {})
    candidates: list[object] = []
    if key == "subject":
        candidates.extend([behavioral_cols.get("subject"), getattr(adapter, "subject_col", None), "subject"])
    elif key == "session":
        candidates.extend([behavioral_cols.get("session"), getattr(adapter, "session_col", None)])
    elif key in {"trial", "trial_idx"}:
        session_col = behavioral_cols.get("session") or getattr(adapter, "session_col", None)
        sort_col = getattr(adapter, "sort_col", None)
        sort_candidates = [
            col
            for col in _iter_column_candidates([sort_col])
            if col != session_col
        ]
        candidates.extend([
            behavioral_cols.get(key),
            behavioral_cols.get("trial_idx"),
            behavioral_cols.get("trial"),
            sort_candidates,
        ])
    else:
        candidates.append(behavioral_cols.get(key))
    candidates.extend(fallbacks)
    return pick_existing_column(df_like, candidates)


def to_pandas_df(df_like) -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    if hasattr(df_like, "to_pandas"):
        return df_like.to_pandas().copy()
    return pd.DataFrame(df_like).copy()


def prepare_weight_family_base_df(
    weights_df,
    *,
    weight_row_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    if weights_df is None or getattr(weights_df, "is_empty", lambda: False)():
        return pd.DataFrame(columns=["subject", "feature", "weight"])

    df = to_pandas_df(weights_df)
    if df.empty:
        return pd.DataFrame(columns=["subject", "feature", "weight"])

    required = {"feature", "weight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "weights dataframe must contain at least 'feature' and 'weight'. "
            f"Missing: {sorted(missing)}."
        )

    df = df.copy()
    df["feature"] = df["feature"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if "subject" not in df.columns:
        df["subject"] = "subject-0"
    else:
        df["subject"] = df["subject"].astype(str)

    if weight_row_indices is not None and "weight_row_idx" in df.columns:
        df["weight_row_idx"] = pd.to_numeric(df["weight_row_idx"], errors="coerce")
        df = df[df["weight_row_idx"].isin(list(weight_row_indices))].copy()

    return df.dropna(subset=["feature", "weight"]).copy()


def prepare_grouped_weight_family_plot(
    weights_df,
    *,
    feature_groups: Sequence[tuple[str, Sequence[str]]],
    title: str,
    xlabel: str,
    plot_kind: Literal["box", "line"] = "box",
    ylabel: str = "Weight",
    weight_row_indices: Sequence[int] | None = (0,),
) -> PreparedWeightFamilyPlot | None:
    feature_to_label: dict[str, str] = {}
    x_order: list[str] = []
    for label, features in feature_groups:
        x_label = str(label)
        x_order.append(x_label)
        for feature in features:
            feature_to_label[str(feature)] = x_label

    if not feature_to_label:
        return None

    df = prepare_weight_family_base_df(
        weights_df,
        weight_row_indices=weight_row_indices,
    )
    if df.empty:
        return None

    df = df[df["feature"].isin(feature_to_label)].copy()
    if df.empty:
        return None

    df["x_label"] = df["feature"].map(feature_to_label)
    df = (
        df.groupby(["subject", "x_label"], as_index=False, observed=False)["weight"]
        .mean()
    )
    if df.empty:
        return None

    present = set(df["x_label"].astype(str))
    resolved_order = tuple(label for label in x_order if label in present)
    return PreparedWeightFamilyPlot(
        data=df,
        plot_kind=plot_kind,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        x_order=resolved_order or None,
    )


def display_regressor_name(regressor_col: str) -> str:
    if regressor_col == "choice_lag_param_2":
        return r"$A_{t,\geq 2}^{\mathrm{choice,param}}$"
    if regressor_col == "choice_lag_one_hot_sum":
        return r"$A$"
    if regressor_col in {"at_choice_param", "choice_lag_param"}:
        return r"$A$"
    if regressor_col == "choice_lag_glm_weighted_sum":
        return r"$A$"
    return regressor_col.replace("_", " ")


def p_right_label() -> str:
    return r"$p(\mathrm{right})$"


def add_choice_lag_summary_regressor(
    plot_df,
    *,
    choice_lag_cols: list[str],
    regressor_col: str = "choice_lag_one_hot_sum",
):
    available_cols = [
        col for col in choice_lag_cols if col in getattr(plot_df, "columns", [])
    ]
    if not available_cols:
        return plot_df

    if isinstance(plot_df, pl.DataFrame):
        return plot_df.with_columns(
            pl.sum_horizontal(
                [
                    pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0)
                    for col in available_cols
                ]
            ).alias(regressor_col)
        )

    df_pd = to_pandas_df(plot_df)
    df_pd[regressor_col] = (
        df_pd[available_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )
    return df_pd


REPEAT_EVIDENCE_TAIL_QUANTILES = (0.0,0.01,0.025,0.05,0.10,0.20,0.35,0.50,0.65,0.80,0.90,0.95,0.975,0.99,1.0,)
# REPEAT_EVIDENCE_TAIL_QUANTILES = (
#     0.0,
#     0.025,
#     0.05,
#     0.10,
#     0.25,
#     0.50,
#     0.75,
#     0.90,
#     0.95,
#     0.975,
#     1.0,
# )


def assign_quantile_bins(
    values,
    *,
    max_bins: int = 4,
    quantiles: Optional[Sequence[float]] = None,
):
    numeric = pd.to_numeric(values, errors="coerce")
    valid_mask = numeric.notna() & np.isfinite(numeric)
    labels = pd.Series(pd.NA, index=values.index, dtype="object")

    if int(valid_mask.sum()) < 2:
        return labels, []

    n_unique = int(pd.Series(numeric[valid_mask]).nunique())
    if n_unique < 2:
        return labels, []

    if quantiles is None:
        q_spec = min(max_bins, n_unique)
    else:
        quantile_grid = np.asarray(quantiles, dtype=float)
        quantile_grid = quantile_grid[np.isfinite(quantile_grid)]
        quantile_grid = np.clip(quantile_grid, 0.0, 1.0)
        q_spec = np.unique(np.concatenate(([0.0], quantile_grid, [1.0])))
        if q_spec.size < 2:
            return labels, []

    qcut = pd.qcut(numeric[valid_mask], q=q_spec, duplicates="drop")
    resolved_labels = [f"Q{idx + 1}" for idx in range(len(qcut.cat.categories))]
    labels.loc[valid_mask] = (
        qcut.cat.rename_categories(resolved_labels).astype(str).to_numpy()
    )
    return labels, resolved_labels


def _stable_sigmoid(z) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    z_arr = np.clip(z_arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_arr))


def lapse_logistic_probability(
    x,
    *,
    slope: float,
    bias: float,
    lapse_left: float,
    lapse_right: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    slope_arr = np.asarray(slope, dtype=float)
    bias_arr = np.asarray(bias, dtype=float)
    lapse_left_arr = np.asarray(lapse_left, dtype=float)
    lapse_right_arr = np.asarray(lapse_right, dtype=float)
    return lapse_left_arr + (1.0 - lapse_left_arr - lapse_right_arr) * _stable_sigmoid(
        slope_arr * (x_arr - bias_arr)
    )


def fit_lapse_logistic_curve(
    x,
    y,
    *,
    weights=None,
    group=None,
    lapse_max: float = 0.4,
    min_points: int = 4,
    n_fit_points: int = 300,
) -> LapseLogisticFit | None:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float).reshape(-1)
        if w_arr.shape != x_arr.shape:
            w_arr = None
        else:
            mask &= np.isfinite(w_arr) & (w_arr > 0)
    else:
        w_arr = None

    x_arr = x_arr[mask]
    y_arr = np.clip(y_arr[mask], 1e-6, 1.0 - 1e-6)
    if w_arr is not None:
        w_arr = w_arr[mask]

    if x_arr.size < min_points or np.unique(x_arr).size < min_points:
        return None

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if w_arr is not None:
        w_arr = w_arr[order]

    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    x_span = max(x_max - x_min, 1e-6)
    pad = 0.25 * x_span
    lapse_bound = float(np.clip(lapse_max, 0.0, 1.0 - 1e-6))
    lapse_sum_bound = 1.0 - 1e-6

    slope0 = 4.0 / x_span
    bias0 = float(x_arr[np.argmin(np.abs(y_arr - 0.5))])
    lapse_left0 = float(np.clip(np.min(y_arr), 0.0, lapse_bound))
    lapse_right0 = float(np.clip(1.0 - np.max(y_arr), 0.0, lapse_bound))
    if lapse_left0 + lapse_right0 > lapse_sum_bound:
        scale = lapse_sum_bound / (lapse_left0 + lapse_right0)
        lapse_left0 *= scale
        lapse_right0 *= scale

    sqrt_w = None
    if w_arr is not None:
        sqrt_w = np.sqrt(w_arr / np.nanmean(w_arr))

    def _residual(params):
        slope, bias, lapse_left, lapse_right = params
        pred = lapse_logistic_probability(
            x_arr,
            slope=slope,
            bias=bias,
            lapse_left=lapse_left,
            lapse_right=lapse_right,
        )
        residual = pred - y_arr
        return residual * sqrt_w if sqrt_w is not None else residual

    x0 = np.asarray([slope0, bias0, lapse_left0, lapse_right0], dtype=float)
    try:
        if lapse_bound <= 0.49:
            from scipy.optimize import least_squares

            result = least_squares(
                _residual,
                x0=x0,
                bounds=(
                    np.asarray([1e-9, x_min - pad, 0.0, 0.0], dtype=float),
                    np.asarray([np.inf, x_max + pad, lapse_bound, lapse_bound], dtype=float),
                ),
                max_nfev=20000,
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                return None
            slope, bias, lapse_left, lapse_right = (float(v) for v in result.x)
        else:
            from scipy.optimize import minimize

            def _objective(params):
                residual = _residual(params)
                return float(np.nansum(residual * residual))

            result = minimize(
                _objective,
                x0=x0,
                method="SLSQP",
                bounds=[
                    (1e-9, None),
                    (x_min - pad, x_max + pad),
                    (0.0, lapse_bound),
                    (0.0, lapse_bound),
                ],
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda params: lapse_sum_bound - params[2] - params[3],
                    }
                ],
                options={"maxiter": 20000, "ftol": 1e-12},
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                return None
            slope, bias, lapse_left, lapse_right = (float(v) for v in result.x)
    except Exception:
        return None
    x_fit = np.linspace(x_min, x_max, int(n_fit_points))
    y_fit = lapse_logistic_probability(
        x_fit,
        slope=slope,
        bias=bias,
        lapse_left=lapse_left,
        lapse_right=lapse_right,
    )
    return LapseLogisticFit(
        group=group,
        slope=slope,
        bias=bias,
        lapse_left=lapse_left,
        lapse_right=lapse_right,
        x_fit=x_fit,
        y_fit=y_fit,
        n_points=int(x_arr.size),
    )


def fit_lapse_logistic_by_group(
    summary_df: pd.DataFrame,
    *,
    line_group_col: str,
    x_col: str,
    y_col: str = "md",
    weight_col: str | None = "nd",
    line_order: Sequence | None = None,
    lapse_max: float = 0.4,
    min_points: int = 4,
    n_fit_points: int = 300,
    shared_core: bool = False,
) -> dict[object, LapseLogisticFit]:
    if summary_df is None or summary_df.empty:
        return {}
    required = {line_group_col, x_col, y_col}
    if not required.issubset(summary_df.columns):
        return {}

    order = list(line_order) if line_order is not None else list(summary_df[line_group_col].dropna().unique())
    if shared_core:
        return _fit_lapse_logistic_by_group_shared_core(
            summary_df,
            line_group_col=line_group_col,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            line_order=order,
            lapse_max=lapse_max,
            min_points=min_points,
            n_fit_points=n_fit_points,
        )

    fits: dict[object, LapseLogisticFit] = {}
    for group_value in order:
        sub = summary_df[summary_df[line_group_col] == group_value].copy()
        if sub.empty:
            continue
        weights = sub[weight_col].to_numpy(dtype=float) if weight_col is not None and weight_col in sub.columns else None
        fit = fit_lapse_logistic_curve(
            sub[x_col].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            weights=weights,
            group=group_value,
            lapse_max=lapse_max,
            min_points=min_points,
            n_fit_points=n_fit_points,
        )
        if fit is not None:
            fits[group_value] = fit
    return fits


def fit_lapse_logistic_by_subject_group(
    subject_summary_df: pd.DataFrame,
    *,
    subject_col: str,
    line_group_col: str,
    x_col: str,
    y_col: str = "data_mean",
    weight_col: str | None = "n_trials",
    line_order: Sequence | None = None,
    lapse_max: float = 0.4,
    min_points: int = 4,
    n_fit_points: int = 300,
    shared_core: bool = False,
) -> dict[object, LapseLogisticFit]:
    if subject_summary_df is None or subject_summary_df.empty:
        return {}
    required = {subject_col, line_group_col, x_col, y_col}
    if not required.issubset(subject_summary_df.columns):
        return {}

    order = (
        list(line_order)
        if line_order is not None
        else list(subject_summary_df[line_group_col].dropna().unique())
    )
    per_group: dict[object, list[LapseLogisticFit]] = {group_value: [] for group_value in order}

    for _subject, subj_df in subject_summary_df.groupby(subject_col, observed=True):
        subject_fits = fit_lapse_logistic_by_group(
            subj_df,
            line_group_col=line_group_col,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            line_order=order,
            lapse_max=lapse_max,
            min_points=min_points,
            n_fit_points=n_fit_points,
            shared_core=shared_core,
        )
        for group_value, fit in subject_fits.items():
            if group_value in per_group:
                per_group[group_value].append(fit)

    averaged: dict[object, LapseLogisticFit] = {}
    for group_value in order:
        group_fits = per_group.get(group_value, [])
        if not group_fits:
            continue

        group_rows = subject_summary_df[subject_summary_df[line_group_col] == group_value]
        x_values = pd.to_numeric(group_rows[x_col], errors="coerce").to_numpy(dtype=float)
        x_values = x_values[np.isfinite(x_values)]
        if x_values.size == 0:
            continue

        slope = float(np.mean([fit.slope for fit in group_fits]))
        bias = float(np.mean([fit.bias for fit in group_fits]))
        lapse_left = float(np.mean([fit.lapse_left for fit in group_fits]))
        lapse_right = float(np.mean([fit.lapse_right for fit in group_fits]))
        x_fit = np.linspace(float(np.min(x_values)), float(np.max(x_values)), int(n_fit_points))
        y_fit = lapse_logistic_probability(
            x_fit,
            slope=slope,
            bias=bias,
            lapse_left=lapse_left,
            lapse_right=lapse_right,
        )
        averaged[group_value] = LapseLogisticFit(
            group=group_value,
            slope=slope,
            bias=bias,
            lapse_left=lapse_left,
            lapse_right=lapse_right,
            x_fit=x_fit,
            y_fit=y_fit,
            n_points=len(group_fits),
        )
    return averaged


def _fit_lapse_logistic_by_group_shared_core(
    summary_df: pd.DataFrame,
    *,
    line_group_col: str,
    x_col: str,
    y_col: str,
    weight_col: str | None,
    line_order: Sequence,
    lapse_max: float,
    min_points: int,
    n_fit_points: int,
) -> dict[object, LapseLogisticFit]:
    groups: list[tuple[object, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for group_value in line_order:
        sub = summary_df[summary_df[line_group_col] == group_value].copy()
        if sub.empty:
            continue

        x_arr = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y_arr = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        w_arr = None
        if weight_col is not None and weight_col in sub.columns:
            candidate = pd.to_numeric(sub[weight_col], errors="coerce").to_numpy(dtype=float)
            if candidate.shape == x_arr.shape:
                mask &= np.isfinite(candidate) & (candidate > 0)
                w_arr = candidate

        x_arr = x_arr[mask]
        y_arr = np.clip(y_arr[mask], 1e-6, 1.0 - 1e-6)
        if w_arr is not None:
            w_arr = w_arr[mask]

        if x_arr.size < min_points or np.unique(x_arr).size < min_points:
            continue
        order = np.argsort(x_arr)
        groups.append((group_value, x_arr[order], y_arr[order], w_arr[order] if w_arr is not None else None))

    if not groups:
        return {}

    x_all = np.concatenate([x for _, x, _, _ in groups])
    y_all = np.concatenate([y for _, _, y, _ in groups])
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    x_span = max(x_max - x_min, 1e-6)
    pad = 0.25 * x_span
    lapse_bound = float(np.clip(lapse_max, 0.0, 0.49))

    pooled = fit_lapse_logistic_curve(
        x_all,
        y_all,
        lapse_max=lapse_bound,
        min_points=min_points,
        n_fit_points=n_fit_points,
    )
    if pooled is not None:
        slope0 = pooled.slope
        bias0 = pooled.bias
    else:
        slope0 = 4.0 / x_span
        bias0 = float(x_all[np.argmin(np.abs(y_all - 0.5))])

    initial = [slope0, bias0]
    lower = [1e-9, x_min - pad]
    upper = [np.inf, x_max + pad]
    for _group_value, _x, y_arr, _w in groups:
        initial.extend(
            [
                float(np.clip(np.min(y_arr), 0.0, lapse_bound)),
                float(np.clip(1.0 - np.max(y_arr), 0.0, lapse_bound)),
            ]
        )
        lower.extend([0.0, 0.0])
        upper.extend([lapse_bound, lapse_bound])

    def _residual(params):
        slope = float(params[0])
        bias = float(params[1])
        residuals = []
        for group_idx, (_group_value, x_arr, y_arr, w_arr) in enumerate(groups):
            lapse_left = float(params[2 + 2 * group_idx])
            lapse_right = float(params[3 + 2 * group_idx])
            pred = lapse_logistic_probability(
                x_arr,
                slope=slope,
                bias=bias,
                lapse_left=lapse_left,
                lapse_right=lapse_right,
            )
            resid = pred - y_arr
            if w_arr is not None:
                mean_w = float(np.nanmean(w_arr))
                if np.isfinite(mean_w) and mean_w > 0:
                    resid = resid * np.sqrt(w_arr / mean_w)
            residuals.append(resid)
        return np.concatenate(residuals)

    try:
        from scipy.optimize import least_squares

        result = least_squares(
            _residual,
            x0=np.asarray(initial, dtype=float),
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            max_nfev=30000,
        )
    except Exception:
        return {}

    if not result.success or not np.all(np.isfinite(result.x)):
        return {}

    slope = float(result.x[0])
    bias = float(result.x[1])
    fits: dict[object, LapseLogisticFit] = {}
    for group_idx, (group_value, x_arr, _y_arr, _w_arr) in enumerate(groups):
        lapse_left = float(result.x[2 + 2 * group_idx])
        lapse_right = float(result.x[3 + 2 * group_idx])
        x_fit = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), int(n_fit_points))
        y_fit = lapse_logistic_probability(
            x_fit,
            slope=slope,
            bias=bias,
            lapse_left=lapse_left,
            lapse_right=lapse_right,
        )
        fits[group_value] = LapseLogisticFit(
            group=group_value,
            slope=slope,
            bias=bias,
            lapse_left=lapse_left,
            lapse_right=lapse_right,
            x_fit=x_fit,
            y_fit=y_fit,
            n_points=int(x_arr.size),
        )
    return fits


def lapse_logistic_label(
    label,
    fit: LapseLogisticFit | None,
    *,
    decimals: int = 2,
) -> str:
    if fit is None:
        return str(label)
    return (
        f"{label} (L={fit.lapse_left:.{decimals}f}, "
        f"R={fit.lapse_right:.{decimals}f}, "
        f"bias={fit.bias:.{decimals}f})"
    )


def glmhmmt_transition_weights_df(arrays_by_subject: dict, views_by_subject: dict | None = None) -> pd.DataFrame:
    """Build a long transition-weight dataframe from loaded GLM-HMM-T arrays."""

    def _state_label(view, idx: int) -> str:
        if view is not None and hasattr(view, "state_name_by_idx"):
            label = getattr(view, "state_name_by_idx", {}).get(idx)
            if label is not None:
                return str(label)
        return f"state {idx + 1}"

    records: list[dict] = []
    for subject, arrays in (arrays_by_subject or {}).items():
        weights = np.asarray(arrays.get("transition_weights", []), dtype=float)
        if weights.size == 0:
            continue
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)

        raw_feature_names = arrays.get("U_cols")
        if raw_feature_names is None:
            feature_names = []
        elif hasattr(raw_feature_names, "tolist"):
            feature_names = list(raw_feature_names.tolist())
        else:
            feature_names = list(raw_feature_names)
        if len(feature_names) != weights.shape[-1]:
            feature_names = [f"transition_{idx}" for idx in range(weights.shape[-1])]

        view = (views_by_subject or {}).get(subject)
        if weights.ndim == 3:
            K = int(getattr(view, "K", weights.shape[0]))
            weights = _expand_self_baseline_transition_array(
                weights,
                K=K,
                trailing_shape=(weights.shape[-1],),
                name="transition_weights",
            )
            for source_idx in range(weights.shape[0]):
                source_label = _state_label(view, source_idx)
                for destination_idx in range(weights.shape[1]):
                    destination_label = _state_label(view, destination_idx)
                    transition_idx = source_idx * weights.shape[1] + destination_idx
                    transition_label = f"{source_label} -> {destination_label}"
                    for feature_idx, feature in enumerate(feature_names):
                        records.append(
                            {
                                "subject": str(subject),
                                "transition_idx": transition_idx,
                                "transition_label": transition_label,
                                "source_state_idx": source_idx,
                                "source_state_label": source_label,
                                "destination_state_idx": destination_idx,
                                "destination_state_label": destination_label,
                                "feature": str(feature),
                                "weight": float(weights[source_idx, destination_idx, feature_idx]),
                            }
                        )
            continue

        if weights.ndim != 2:
            raise ValueError(f"transition_weights must be 1D, 2D, or full 3D, got shape {weights.shape}.")

        for row_idx in range(weights.shape[0]):
            transition_label = f"transition {row_idx + 1}"
            if view is not None and hasattr(view, "state_name_by_idx") and weights.shape[0] == 1 and getattr(view, "K", None) == 2:
                transition_label = "state transition"
            for feature_idx, feature in enumerate(feature_names):
                records.append(
                    {
                        "subject": str(subject),
                        "transition_idx": row_idx,
                        "transition_label": transition_label,
                        "source_state_idx": None,
                        "source_state_label": None,
                        "destination_state_idx": None,
                        "destination_state_label": None,
                        "feature": str(feature),
                        "weight": float(weights[row_idx, feature_idx]),
                    }
                )
    return pd.DataFrame.from_records(records)


def glmhmmt_state_psychometric_df(
    trial_df,
    *,
    x_col: str = "stimulus",
    x_order: Sequence | None = None,
    response_col: str = "response",
    state_col: str = "state_label",
    subject_col: str = "subject",
) -> pd.DataFrame:
    """Summarize P(right) by subject, state, and a categorical x variable."""

    df = to_pandas_df(trial_df)
    required = {subject_col, state_col, x_col, response_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=[subject_col, state_col, "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"])

    out = df[[subject_col, state_col, x_col, response_col]].copy()
    out[response_col] = pd.to_numeric(out[response_col], errors="coerce")
    out["right"] = (out[response_col] > 0).astype(float)
    out = out.dropna(subset=[subject_col, state_col, x_col, "right"])
    if out.empty:
        return pd.DataFrame(columns=[subject_col, state_col, "x_value", "x_label", "x_numeric", "x_position", "p_right", "n_trials"])

    if x_order is None:
        unique_values = list(pd.unique(out[x_col]))
        numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")
        if numeric_values.notna().all():
            order = [
                value
                for _, value in sorted(
                    zip(numeric_values.astype(float), unique_values, strict=False),
                    key=lambda item: item[0],
                )
            ]
        else:
            order = sorted(unique_values, key=lambda value: str(value))
    else:
        order = list(x_order)

    def _format_x_label(value) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return f"{float(numeric):g}"
        return str(value)

    label_order = [_format_x_label(value) for value in order]
    label_by_position = dict(enumerate(label_order))
    out["x_value"] = pd.Categorical(out[x_col], categories=order, ordered=True)
    out = out.dropna(subset=["x_value"])
    out["x_position"] = out["x_value"].cat.codes
    out["x_label"] = pd.Categorical(
        out["x_position"].map(label_by_position),
        categories=label_order,
        ordered=True,
    )

    summary = (
        out.groupby([subject_col, state_col, "x_value", "x_position", "x_label"], as_index=False, observed=True)
        .agg(p_right=("right", "mean"), n_trials=("right", "size"))
        .sort_values([state_col, "x_position", subject_col])
    )
    summary["x_numeric"] = pd.to_numeric(summary["x_label"].astype(str), errors="coerce")
    return summary


def glmhmmt_state_accuracy_df(
    trial_df,
    *,
    state_col: str = "state_label",
    subject_col: str = "subject",
    correct_col: str = "correct_bool",
) -> pd.DataFrame:
    """Summarize trial accuracy by subject and state."""

    df = to_pandas_df(trial_df)
    required = {subject_col, state_col, correct_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=[subject_col, state_col, "accuracy", "n_trials"])

    out = df[[subject_col, state_col, correct_col]].copy()
    out[correct_col] = pd.to_numeric(out[correct_col], errors="coerce")
    out = out.dropna(subset=[subject_col, state_col, correct_col])
    return (
        out.groupby([subject_col, state_col], as_index=False, observed=True)
        .agg(accuracy=(correct_col, "mean"), n_trials=(correct_col, "size"))
        .sort_values([state_col, subject_col])
    )


def glmhmmt_state_occupancy_df(
    trial_df,
    *,
    state_col: str = "state_label",
    subject_col: str = "subject",
    session_col: str = "session",
) -> pd.DataFrame:
    """Summarize MAP-state occupancy per session."""

    df = to_pandas_df(trial_df)
    required = {subject_col, session_col, state_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=[subject_col, session_col, state_col, "occupancy", "n_trials"])

    counts = (
        df.groupby([subject_col, session_col, state_col], observed=True)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    totals = counts.groupby([subject_col, session_col], observed=True)["n_trials"].transform("sum")
    counts["occupancy"] = counts["n_trials"] / totals.replace(0, np.nan)
    return counts.sort_values([subject_col, session_col, state_col])


def glmhmmt_state_trace_df(
    trial_df,
    *,
    state_col: str = "state_label",
    subject_col: str = "subject",
    session_col: str = "session",
    trial_col: str = "trial_idx",
    n_bins: int = 50,
) -> pd.DataFrame:
    """Build mean posterior traces over normalized session time."""

    df = to_pandas_df(trial_df)
    required = {subject_col, session_col, trial_col, state_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=[subject_col, state_col, "trial_bin", "p_state"])

    posterior_cols = [col for col in df.columns if re.fullmatch(r"p_state_\d+", str(col))]
    if not posterior_cols:
        return pd.DataFrame(columns=[subject_col, state_col, "trial_bin", "p_state"])

    state_map = (
        df[[state_col, "state_idx"]].dropna().drop_duplicates()
        if "state_idx" in df.columns
        else pd.DataFrame()
    )
    label_by_idx = {
        int(row.state_idx): str(getattr(row, state_col))
        for row in state_map.itertuples(index=False)
        if pd.notna(row.state_idx)
    }

    work = df[[subject_col, session_col, trial_col, *posterior_cols]].copy()
    work[trial_col] = pd.to_numeric(work[trial_col], errors="coerce")
    work = work.dropna(subset=[subject_col, session_col, trial_col])
    if work.empty:
        return pd.DataFrame(columns=[subject_col, state_col, "trial_bin", "p_state"])

    session_max = work.groupby([subject_col, session_col], observed=True)[trial_col].transform("max")
    session_min = work.groupby([subject_col, session_col], observed=True)[trial_col].transform("min")
    denom = (session_max - session_min).replace(0, np.nan)
    norm_trial = ((work[trial_col] - session_min) / denom).fillna(0.0)
    work["trial_bin"] = np.floor(norm_trial.clip(0, 0.999999) * int(n_bins)).astype(int)

    long = work.melt(
        id_vars=[subject_col, session_col, "trial_bin"],
        value_vars=posterior_cols,
        var_name="state_idx",
        value_name="p_state",
    )
    long["state_idx"] = long["state_idx"].str.extract(r"(\d+)").astype(int)
    long[state_col] = long["state_idx"].map(label_by_idx).fillna(long["state_idx"].map(lambda idx: f"State {idx}"))
    return (
        long.groupby([subject_col, state_col, "trial_bin"], as_index=False, observed=True)["p_state"]
        .mean()
        .sort_values([state_col, "trial_bin", subject_col])
    )


def glmhmmt_state_metric_df(
    trial_df,
    *,
    metrics: Sequence[str],
    state_col: str = "state_label",
    subject_col: str = "subject",
    correct_only_metrics: Sequence[str] = (),
    correct_col: str = "correct_bool",
) -> pd.DataFrame:
    """Summarize numeric trial metrics by subject and state.

    Metrics listed in ``correct_only_metrics`` are computed only from trials
    where ``correct_col`` equals one. Other metrics continue to use all trials.
    """

    df = to_pandas_df(trial_df)
    available_metrics = [metric for metric in metrics if metric in df.columns]
    required = {subject_col, state_col}
    if df.empty or not required.issubset(df.columns) or not available_metrics:
        return pd.DataFrame(columns=[subject_col, state_col, "metric", "value", "n_trials"])

    filtered_metrics = set(correct_only_metrics).intersection(available_metrics)
    if filtered_metrics and correct_col not in df.columns:
        raise ValueError(
            f"Cannot restrict {sorted(filtered_metrics)} to correct trials: "
            f"missing column {correct_col!r}."
        )

    work_cols = [subject_col, state_col, *available_metrics]
    if filtered_metrics:
        work_cols.append(correct_col)
    work = df[work_cols].copy()
    for metric in available_metrics:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    id_vars = [subject_col, state_col]
    if filtered_metrics:
        work[correct_col] = pd.to_numeric(work[correct_col], errors="coerce")
        id_vars.append(correct_col)
    long = work.melt(
        id_vars=id_vars,
        value_vars=available_metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if filtered_metrics:
        keep = ~long["metric"].isin(filtered_metrics) | long[correct_col].eq(1)
        long = long.loc[keep].drop(columns=correct_col)
    if long.empty:
        return pd.DataFrame(columns=[subject_col, state_col, "metric", "value", "n_trials"])
    return (
        long.groupby([subject_col, state_col, "metric"], as_index=False, observed=True)
        .agg(value=("value", "mean"), n_trials=("value", "size"))
        .sort_values(["metric", state_col, subject_col])
    )


def glmhmmt_state_dwell_df(
    trial_df,
    *,
    state_col: str = "state_label",
    subject_col: str = "subject",
    session_col: str = "session",
    trial_col: str = "trial_idx",
) -> pd.DataFrame:
    """Return contiguous MAP-state run lengths per subject and session."""

    df = to_pandas_df(trial_df)
    required = {subject_col, session_col, trial_col, state_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=[subject_col, session_col, state_col, "dwell_trials"])

    records: list[dict] = []
    ordered = df[[subject_col, session_col, trial_col, state_col]].dropna().sort_values(
        [subject_col, session_col, trial_col]
    )
    for (subject, session), group in ordered.groupby([subject_col, session_col], observed=True):
        current_state = None
        run_length = 0
        for state in group[state_col].astype(str):
            if current_state is None:
                current_state = state
                run_length = 1
            elif state == current_state:
                run_length += 1
            else:
                records.append(
                    {
                        subject_col: subject,
                        session_col: session,
                        state_col: current_state,
                        "dwell_trials": run_length,
                    }
                )
                current_state = state
                run_length = 1
        if current_state is not None:
            records.append(
                {
                    subject_col: subject,
                    session_col: session,
                    state_col: current_state,
                    "dwell_trials": run_length,
                }
            )
    return pd.DataFrame.from_records(records)


def glmhmmt_state_switches_df(
    trial_df,
    *,
    state_col: str = "state_label",
    subject_col: str = "subject",
    session_col: str = "session",
    trial_col: str = "trial_idx",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return session-level switches and subject-averaged switches."""

    df = to_pandas_df(trial_df)
    required = {subject_col, session_col, trial_col, state_col}
    columns = [subject_col, session_col, "n_switches", "n_trials", "switch_rate"]
    if df.empty or not required.issubset(df.columns):
        empty = pd.DataFrame(columns=columns)
        return empty, pd.DataFrame(columns=[subject_col, "n_switches", "switch_rate"])

    records: list[dict] = []
    ordered = df[[subject_col, session_col, trial_col, state_col]].dropna().sort_values(
        [subject_col, session_col, trial_col]
    )
    for (subject, session), group in ordered.groupby([subject_col, session_col], observed=True):
        states = group[state_col].astype(str).to_numpy()
        n_trials = int(states.size)
        n_switches = int(np.sum(states[1:] != states[:-1])) if n_trials > 1 else 0
        records.append(
            {
                subject_col: subject,
                session_col: session,
                "n_switches": n_switches,
                "n_trials": n_trials,
                "switch_rate": n_switches / max(n_trials - 1, 1),
            }
        )
    session_df = pd.DataFrame.from_records(records, columns=columns)
    subject_df = (
        session_df.groupby(subject_col, as_index=False, observed=True)
        .agg(n_switches=("n_switches", "mean"), switch_rate=("switch_rate", "mean"))
        .sort_values(subject_col)
    )
    return session_df, subject_df


def glmhmmt_state_switch_histogram_df(
    session_df,
    *,
    subject_col: str = "subject",
    switch_col: str = "n_switches",
    value_col: str = "switch_probability",
) -> pd.DataFrame:
    """Return integer-binned histograms of session-averaged switches per animal."""

    df = to_pandas_df(session_df)
    columns = [subject_col, "mean_n_switches", switch_col, value_col]
    required = {subject_col, switch_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    df = df[[subject_col, switch_col]].copy()
    df[switch_col] = pd.to_numeric(df[switch_col], errors="coerce")
    df = df.dropna(subset=[subject_col, switch_col])
    if df.empty:
        return pd.DataFrame(columns=columns)

    subject_means = (
        df.groupby(subject_col, as_index=False, observed=True)
        .agg(mean_n_switches=(switch_col, "mean"))
        .sort_values(subject_col)
    )
    subject_means[switch_col] = np.floor(subject_means["mean_n_switches"] + 0.5).astype(int)
    switch_values = range(0, int(subject_means[switch_col].max()) + 1)
    records: list[dict] = []
    for row in subject_means.itertuples(index=False):
        subject = getattr(row, subject_col)
        mean_n_switches = float(row.mean_n_switches)
        switch_bin = int(getattr(row, switch_col))
        for n_switches in switch_values:
            records.append(
                {
                    subject_col: subject,
                    "mean_n_switches": mean_n_switches,
                    switch_col: int(n_switches),
                    value_col: float(n_switches == switch_bin),
                }
            )
    return pd.DataFrame.from_records(records, columns=columns)


def glmhmmt_change_triggered_posterior_df(
    trial_df,
    *,
    state_col: str = "state_label",
    subject_col: str = "subject",
    session_col: str = "session",
    trial_col: str = "trial_idx",
    window: int = 20,
) -> pd.DataFrame:
    """Align posterior probabilities to MAP-state changes."""

    df = to_pandas_df(trial_df)
    required = {subject_col, session_col, trial_col, state_col}
    posterior_cols = [col for col in df.columns if re.fullmatch(r"p_state_\d+", str(col))]
    if df.empty or not required.issubset(df.columns) or not posterior_cols:
        return pd.DataFrame(columns=[subject_col, state_col, "event_id", "lag", "p_state"])

    state_map = (
        df[[state_col, "state_idx"]].dropna().drop_duplicates()
        if "state_idx" in df.columns
        else pd.DataFrame()
    )
    label_by_idx = {
        int(row.state_idx): str(getattr(row, state_col))
        for row in state_map.itertuples(index=False)
        if pd.notna(row.state_idx)
    }

    records: list[dict] = []
    event_id = 0
    keep_cols = [subject_col, session_col, trial_col, state_col, *posterior_cols]
    ordered = df[keep_cols].dropna(subset=[subject_col, session_col, trial_col, state_col]).sort_values(
        [subject_col, session_col, trial_col]
    )
    for (subject, session), group in ordered.groupby([subject_col, session_col], observed=True):
        group = group.reset_index(drop=True)
        states = group[state_col].astype(str).to_numpy()
        change_positions = np.flatnonzero(states[1:] != states[:-1]) + 1
        for pos in change_positions:
            event_id += 1
            lo = max(0, pos - int(window))
            hi = min(len(group), pos + int(window) + 1)
            segment = group.iloc[lo:hi].copy()
            for local_idx, row in segment.iterrows():
                lag = int(local_idx - pos)
                for posterior_col in posterior_cols:
                    state_idx = int(str(posterior_col).rsplit("_", 1)[-1])
                    records.append(
                        {
                            subject_col: subject,
                            session_col: session,
                            "event_id": event_id,
                            "lag": lag,
                            state_col: label_by_idx.get(state_idx, f"State {state_idx}"),
                            "p_state": float(row[posterior_col]),
                        }
                    )
    return pd.DataFrame.from_records(records)


def format_lapse_logistic_fits(
    fits: dict[object, LapseLogisticFit],
    *,
    title: str | None = None,
    decimals: int = 3,
) -> str:
    if not fits:
        return ""
    lines = []
    if title:
        lines.append(str(title))
    for group_value, fit in fits.items():
        lines.append(
            f"{group_value}: lapse_left={fit.lapse_left:.{decimals}f}, "
            f"lapse_right={fit.lapse_right:.{decimals}f}, "
            f"bias={fit.bias:.{decimals}f}, slope={fit.slope:.{decimals}f}"
        )
    return "\n".join(lines)


def padded_numeric_limits(
    values,
    *,
    absolute_pad: float = 0.0,
) -> tuple[float, float] | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) == 0:
        return None
    xmin = float(np.min(numeric))
    xmax = float(np.max(numeric))
    if xmax <= xmin:
        return None
    return xmin - absolute_pad, xmax + absolute_pad


def pick_choice_history_regressor(regressor_options: list[str]) -> str | None:
    preferred_order = [
        "choice_lag_param_2",
        "choice_lag_one_hot_sum",
        "choice_lag_param",
        "at_choice_param",
    ]
    for regressor in preferred_order:
        if regressor in regressor_options:
            return regressor
    return None


def resolve_grouping(
    df: pd.DataFrame,
    *,
    group_col: str | None,
    group_order: Sequence | None = None,
) -> tuple[str | None, list]:
    if group_col is None:
        return None, []
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}.")
    if group_order is None:
        order = list(pd.unique(df.loc[df[group_col].notna(), group_col]))
    else:
        order = list(group_order)
    return group_col, order


def attach_response_right_column(
    df_pd: pd.DataFrame,
    *,
    response_mode: str,
) -> pd.DataFrame:
    df = df_pd.copy()
    df["response"] = pd.to_numeric(df["response"], errors="coerce")

    if response_mode == "pm1_or_prob":
        unique_response = set(df["response"].dropna().unique().tolist())
        if unique_response.issubset({-1.0, 1.0}):
            df["_response_right"] = (df["response"] > 0).astype(float)
        else:
            df["_response_right"] = df["response"].astype(float)
    elif response_mode == "mcdr_3class":
        df["_response_right"] = (df["response"] == 2).astype(float)
    else:
        raise ValueError(f"Unknown response_mode={response_mode}")

    return df


STIM_EVIDENCE_CANDIDATES = (
    "stim_x_delay_param",
    "stim_x_delay_one_hot_sum",
    "stim_x_delay",
    "stim_vals",
    "stim_param",
    "stimd_n_z",
    "stim_d",
    "ild_norm",
    "total_evidence_strength",
    "ILD",
)

ACTION_TRACE_CANDIDATES = (
    "at_choice",
    "at_choice_param",
    "choice_lag_one_hot_sum",
    "choice_lag_param_2",
    "choice_lag_param",
    "A_R",
    "A_L",
)


def _first_available(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((col for col in candidates if col in available), None)


def _first_usable_numeric_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) > 0 and float(finite.max() - finite.min()) > 0:
            return col
    return None


def _stim_x_delay_hot_cols(columns: Sequence[str]) -> list[str]:
    return sorted([col for col in columns if str(col).startswith("stim_x_delay_hot_")])


def _attach_stim_x_delay_one_hot_sum(df: pd.DataFrame) -> pd.DataFrame:
    hot_cols = _stim_x_delay_hot_cols(df.columns)
    if not hot_cols or "stim_x_delay_one_hot_sum" in df.columns:
        return df
    out = df.copy()
    out["stim_x_delay_one_hot_sum"] = (
        out[hot_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )
    return out


def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.ones(1, dtype=float)
    n_conv_bins = 4.0 * float(sigma_bins)
    x = np.arange(-n_conv_bins, n_conv_bins + 1.0, 1.0, dtype=float)
    return np.exp(-(x**2) / (2.0 * sigma_bins**2))


def _smooth_2d(
    values: np.ndarray,
    *,
    sigma_x_bins: float,
    sigma_y_bins: float,
) -> np.ndarray:
    if sigma_x_bins <= 0 and sigma_y_bins <= 0:
        return values

    def _same_length_convolve(row: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        convolved = np.convolve(row, kernel, mode="same")
        if convolved.shape[0] == row.shape[0]:
            return convolved
        start = (convolved.shape[0] - row.shape[0]) // 2
        return convolved[start : start + row.shape[0]]

    out = values
    if sigma_x_bins > 0:
        kernel_x = _gaussian_kernel_1d(sigma_x_bins)
        out = np.apply_along_axis(_same_length_convolve, 0, out, kernel_x)
    if sigma_y_bins > 0:
        kernel_y = _gaussian_kernel_1d(sigma_y_bins)
        out = np.apply_along_axis(_same_length_convolve, 1, out, kernel_y)
    return out


def _fill_nan_grid(values: np.ndarray) -> np.ndarray:
    if np.isfinite(values).all():
        return values
    filled = pd.DataFrame(values).interpolate(
        axis=0,
        limit_direction="both",
    ).interpolate(
        axis=1,
        limit_direction="both",
    )
    return filled.to_numpy(dtype=float)


def integration_map_2d(
    x,
    y,
    values,
    *,
    bnd: float | None = None,
    dx: float | None = None,
    n_bins: int = 64,
    sigma: float | None = None,
    fill_empty: bool = True,
    default_sigma_dx: float = 2.0,
    x_edges=None,
    y_edges=None,
) -> dict | None:
    """Return a MATLAB-style smoothed 2D integration map over x/y bins."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if int(mask.sum()) < 10:
        return None

    x = x[mask]
    y = y[mask]
    values = values[mask]

    if bnd is None and (x_edges is None or y_edges is None):
        bnd = float(np.nanmax(np.abs(np.concatenate([x, y]))))
    if bnd is not None and (not np.isfinite(bnd) or bnd <= 0):
        return None

    if dx is None and (x_edges is None or y_edges is None):
        dx = (2.0 * bnd) / float(n_bins)
    if dx is not None and (not np.isfinite(dx) or dx <= 0):
        return None

    if x_edges is None:
        x_edges = np.arange(-bnd, bnd + dx, dx, dtype=float)
    else:
        x_edges = np.asarray(x_edges, dtype=float)
    if y_edges is None:
        y_edges = np.arange(-bnd, bnd + dx, dx, dtype=float)
    else:
        y_edges = np.asarray(y_edges, dtype=float)
    if x_edges.size < 3 or y_edges.size < 3:
        return None

    x_step = float(np.nanmedian(np.diff(x_edges)))
    y_step = float(np.nanmedian(np.diff(y_edges)))
    if x_step <= 0 or y_step <= 0:
        return None

    if sigma is None:
        sigma_x = float(default_sigma_dx) * x_step
        sigma_y = float(default_sigma_dx) * y_step
    else:
        sigma_x = float(sigma)
        sigma_y = float(sigma)
    if sigma_x < 0 or sigma_y < 0:
        return None

    def _extended_axis(base_edges: np.ndarray, step: float, axis_sigma: float):
        n_conv = 4.0 * axis_sigma
        boundaries = np.arange(
            float(base_edges[0] - n_conv),
            float(base_edges[-1] + n_conv) + step * 0.5,
            step,
            dtype=float,
        )
        hist_edges = np.concatenate(([-np.inf], boundaries, [np.inf]))
        centers = np.concatenate(([boundaries[0] - step / 2.0], boundaries + step / 2.0))
        keep = (centers > base_edges[0]) & (centers < base_edges[-1])
        return hist_edges, centers, keep

    x_hist_edges, x_centers_full, x_keep = _extended_axis(x_edges, x_step, sigma_x)
    y_hist_edges, y_centers_full, y_keep = _extended_axis(y_edges, y_step, sigma_y)

    weighted_sum, _, _ = np.histogram2d(
        x,
        y,
        bins=(x_hist_edges, y_hist_edges),
        weights=values,
    )
    counts, _, _ = np.histogram2d(x, y, bins=(x_hist_edges, y_hist_edges))

    sigma_x_bins = sigma_x / x_step if x_step > 0 else 0.0
    sigma_y_bins = sigma_y / y_step if y_step > 0 else 0.0

    weighted_sum = _smooth_2d(
        weighted_sum,
        sigma_x_bins=sigma_x_bins,
        sigma_y_bins=sigma_y_bins,
    )
    counts = _smooth_2d(
        counts,
        sigma_x_bins=sigma_x_bins,
        sigma_y_bins=sigma_y_bins,
    )

    weighted_sum = weighted_sum[np.ix_(x_keep, y_keep)]
    counts = counts[np.ix_(x_keep, y_keep)]
    x_centers = x_centers_full[x_keep]
    y_centers = y_centers_full[y_keep]

    mean_map = np.divide(
        weighted_sum,
        counts,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=counts > 1e-9,
    )

    return {
        "map": mean_map,
        "n_datapoints": counts,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "dx": dx,
        "bnd": bnd,
        "sigma": sigma,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_x_bins": sigma_x_bins,
        "sigma_y_bins": sigma_y_bins,
    }

def compute_rb_by_x(
    df,
    x_col: str,
    choice_col: str,
    subject_col: str = "subject",
    trial_col: str | None = None,
):
    df = df.copy()
    if trial_col is not None and trial_col in df.columns:
        df = df.sort_values([subject_col, trial_col]).copy()

    df["_prev_choice"] = df.groupby(subject_col, observed=True)[choice_col].shift(1)
    df = df.dropna(subset=[x_col, choice_col, "_prev_choice"]).copy()

    choice_set = sorted(df[choice_col].unique())

    results = []

    for (subject, x_val), df_sx in df.groupby([subject_col, x_col], observed=True):
        probs = []
        for c in choice_set:
            df_c = df_sx[df_sx["_prev_choice"] == c]
            if df_c.empty:
                continue
            probs.append((df_c[choice_col] == c).mean())

        if probs:
            results.append(
                {
                    subject_col: subject,
                    x_col: x_val,
                    "rb": float(np.mean(probs)),
                }
            )

    return pd.DataFrame(results)


def prepare_treatment_accuracy_repetition_curves(
    plot_df,
    *,
    task_name: str,
    treatment_order: Sequence[str] = ("Saline", "Drug"),
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Summarize observed and model-predicted accuracy and repetition bias.

    Values are first computed within animal. Repetition bias is additionally
    balanced over the previous left/right response before animals are averaged,
    matching :func:`compute_rb_by_x`. Model repetition is the expected
    probability of repeating the observed previous response.
    """
    df = to_pandas_df(plot_df)
    required = {
        "subject",
        "session",
        "condition",
        "response",
        "correct_bool",
        "p_model_correct",
    }
    if df.empty or not required.issubset(df.columns):
        return {"accuracy": pd.DataFrame(), "repetition_bias": pd.DataFrame()}, {}

    task_key = str(task_name).upper()
    if task_key in {"2ADC", "2ADC_DRUG", "2AFC_DELAY", "2AFC_DELAY_DRUG"}:
        x_source = pick_existing_column(df, ("delays", "delay"))
        x_label = "Delay (s)"
        invert_x = False
    elif task_key in {"2AFC", "2AFC_DRUG"}:
        x_source = pick_existing_column(df, ("ILD", "stimulus"))
        x_label = "|ILD| (dB)"
        invert_x = True
    else:
        raise ValueError(f"Unsupported binary task for treatment curves: {task_name!r}.")
    if x_source is None:
        return {"accuracy": pd.DataFrame(), "repetition_bias": pd.DataFrame()}, {}

    p_right_col = pick_existing_column(df, ("p_model_right", "p_pred", "pR"))
    if p_right_col is None:
        return {"accuracy": pd.DataFrame(), "repetition_bias": pd.DataFrame()}, {}

    work = df.copy()
    condition = work["condition"].astype("string").str.strip().str.lower()
    treatment_map = {str(value).lower(): str(value) for value in treatment_order}
    work["treatment"] = condition.map(treatment_map)
    work["x_value"] = pd.to_numeric(work[x_source], errors="coerce")
    if task_key in {"2AFC", "2AFC_DRUG"}:
        work["x_value"] = work["x_value"].abs()

    response = pd.to_numeric(work["response"], errors="coerce")
    finite_response = response.dropna()
    if not finite_response.empty and finite_response.min() < 0:
        work["_response_right"] = (response > 0).astype(float)
    else:
        work["_response_right"] = response.where(response.isin([0.0, 1.0]))
    work["_data_accuracy"] = _binary_indicator_series(work["correct_bool"])
    work["_model_accuracy"] = pd.to_numeric(work["p_model_correct"], errors="coerce")
    work["_model_p_right"] = pd.to_numeric(work[p_right_col], errors="coerce")

    sort_cols = ["subject", "session"]
    trial_col = pick_existing_column(work, ("trial_idx", "trial"))
    if trial_col is not None:
        sort_cols.append(trial_col)
    work = work.sort_values(sort_cols).copy()
    work["_previous_response_right"] = work.groupby(
        ["subject", "session", "treatment"], observed=True
    )["_response_right"].shift(1)
    work["_data_repeat"] = (
        work["_response_right"] == work["_previous_response_right"]
    ).astype(float)
    work.loc[work["_previous_response_right"].isna(), "_data_repeat"] = np.nan
    work["_model_repeat"] = np.where(
        np.isclose(work["_previous_response_right"], 1.0),
        work["_model_p_right"],
        1.0 - work["_model_p_right"],
    )

    accuracy_subject = (
        work.dropna(
            subset=[
                "subject",
                "treatment",
                "x_value",
                "_data_accuracy",
                "_model_accuracy",
            ]
        )
        .groupby(["subject", "treatment", "x_value"], observed=True)
        .agg(
            data_value=("_data_accuracy", "mean"),
            model_value=("_model_accuracy", "mean"),
        )
        .reset_index()
    )

    repetition_by_side = (
        work.dropna(
            subset=[
                "subject",
                "treatment",
                "x_value",
                "_previous_response_right",
                "_data_repeat",
                "_model_repeat",
            ]
        )
        .groupby(
            [
                "subject",
                "treatment",
                "x_value",
                "_previous_response_right",
            ],
            observed=True,
        )
        .agg(
            data_value=("_data_repeat", "mean"),
            model_value=("_model_repeat", "mean"),
        )
        .reset_index()
    )
    repetition_subject = (
        repetition_by_side.groupby(
            ["subject", "treatment", "x_value"], observed=True
        )[["data_value", "model_value"]]
        .mean()
        .reset_index()
    )

    def _summarize(subject_df: pd.DataFrame) -> pd.DataFrame:
        if subject_df.empty:
            return pd.DataFrame()
        summary = (
            subject_df.groupby(["treatment", "x_value"], observed=True)
            .agg(
                data_mean=("data_value", "mean"),
                data_std=("data_value", "std"),
                data_count=("data_value", "count"),
                model_mean=("model_value", "mean"),
                model_std=("model_value", "std"),
                model_count=("model_value", "count"),
            )
            .reset_index()
        )
        summary["data_sem"] = summary["data_std"].fillna(0.0) / np.sqrt(
            summary["data_count"].clip(lower=1)
        )
        summary["model_sem"] = summary["model_std"].fillna(0.0) / np.sqrt(
            summary["model_count"].clip(lower=1)
        )
        summary["treatment"] = pd.Categorical(
            summary["treatment"], categories=list(treatment_order), ordered=True
        )
        return summary.sort_values(["treatment", "x_value"]).reset_index(drop=True)

    summaries = {
        "accuracy": _summarize(accuracy_subject),
        "repetition_bias": _summarize(repetition_subject),
    }
    x_values = sorted(
        {
            float(value)
            for summary in summaries.values()
            for value in summary.get("x_value", pd.Series(dtype=float)).dropna()
        }
    )
    meta = {
        "xlabel": x_label,
        "baseline": 0.5,
        "invert_x": invert_x,
        "x_values": x_values,
        "p_right_col": p_right_col,
    }
    return summaries, meta


def prepare_right_integration_maps(
    plot_df,
    *,
    response_mode: str,
    pred_col: str | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    value_col: str | None = None,
    include_model: bool = True,
    bnd: float | None = None,
    dx: float | None = None,
    n_bins: int = 64,
    sigma: float | None = None,
    fill_empty: bool = True,
    default_sigma_dx: float = 2.0,
    x_edges=None,
    y_edges=None,
    xticks: list[float] | None = None,
    x_tick_labels: list[str] | None = None,
) -> tuple[list[dict], dict]:
    df_pd = to_pandas_df(plot_df)
    if df_pd.empty:
        return [], {}
    df_pd = _attach_stim_x_delay_one_hot_sum(df_pd)

    x_col = x_col or _first_usable_numeric_col(df_pd, STIM_EVIDENCE_CANDIDATES)
    y_col = y_col or _first_available(df_pd.columns, ACTION_TRACE_CANDIDATES)
    if x_col is None or y_col is None or "response" not in df_pd.columns:
        return [], {}

    df_pd = attach_response_right_column(df_pd, response_mode=response_mode)

    value_specs: list[tuple[str, str]] = []
    if value_col is not None:
        if value_col in df_pd.columns:
            value_specs.append((value_col, value_col))
    else:
        value_specs.append(("_response_right", "Data"))
        if include_model:
            model_col = pred_col if pred_col in df_pd.columns else _first_available(df_pd.columns, ("p_pred", "pR"))
            if model_col is not None:
                value_specs.append((model_col, "Model"))

    panels = []
    for selected_col, label in value_specs:
        result = integration_map_2d(
            df_pd[x_col],
            df_pd[y_col],
            df_pd[selected_col],
            bnd=bnd,
            dx=dx,
            n_bins=n_bins,
            sigma=sigma,
            fill_empty=fill_empty,
            default_sigma_dx=default_sigma_dx,
            x_edges=x_edges,
            y_edges=y_edges,
        )
        if result is not None:
            panels.append({"label": label, **result})

    meta = {
        "xlabel": display_regressor_name(x_col),
        "ylabel": display_regressor_name(y_col),
        "zlabel": p_right_label(),
        "x_col": x_col,
        "y_col": y_col,
        "xticks": xticks,
        "x_tick_labels": x_tick_labels,
    }
    return panels, meta


def attach_signed_delay_columns(df_pd: pd.DataFrame) -> pd.DataFrame:
    df = df_pd.copy()

    stim_col = None
    for col in ["stim", "stimulus"]:
        if col in df.columns:
            stim_col = col
            break

    delay_col = None
    for col in ["delay_raw", "delays", "delay"]:
        if col in df.columns:
            delay_col = col
            break

    if stim_col is None or delay_col is None:
        df["_signed_delay"] = np.nan
        df["_signed_delay_cat"] = pd.Series(pd.NA, index=df.index, dtype="object")
        return df

    stim_values = pd.to_numeric(df[stim_col], errors="coerce")
    unique_stim = set(stim_values.dropna().unique().tolist())
    if unique_stim and unique_stim.issubset({0, 1, 0.0, 1.0}):
        stim_sign = pd.Series(
            np.where(stim_values == 0, -1.0, np.where(stim_values == 1, 1.0, np.nan)),
            index=df.index,
            dtype=float,
        )
    else:
        stim_sign = np.sign(stim_values)
    delay_values = pd.to_numeric(df[delay_col], errors="coerce")

    signed_delay = delay_values * stim_sign
    df["_signed_delay"] = signed_delay

    cat = pd.Series(pd.NA, index=df.index, dtype="object")
    valid = np.isfinite(delay_values) & np.isfinite(stim_sign)
    cat.loc[valid] = signed_delay.loc[valid].map(lambda value: f"{float(value):g}")

    preferred_order = ["-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"]
    present = set(cat.dropna())
    existing = [x for x in preferred_order if x in present]
    extras = sorted(
        (x for x in present if x not in set(existing)),
        key=lambda x: float(x),
    )

    df["_signed_delay_cat"] = pd.Categorical(cat, categories=existing + extras, ordered=True)
    return df


def label_condition_session_windows(
    df: pl.DataFrame,
    *,
    subject_col: str,
    session_col: str,
    condition_col: str,
    order_col: str | None = None,
    first_session: int = 1,
    late_sessions: Sequence[int] | None = (3, 4, 5),
    late_from_session: int | None = None,
    exposure_col: str = "condition_session_number",
    window_col: str = "session_window",
    first_label: str = "First",
    late_label: str = "Late (3–5)",
) -> pl.DataFrame:
    """Label first and late sessions within each subject and condition.

    Session numbers are assigned after sorting each subject-condition pair by
    ``order_col`` (or ``session_col`` when no separate ordering column is
    supplied). Rows outside the requested windows receive a null label.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("label_condition_session_windows expects a Polars DataFrame.")

    order_col = order_col or session_col
    required = {subject_col, session_col, condition_col, order_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")
    if first_session < 1:
        raise ValueError("first_session must be at least 1.")

    if late_from_session is not None and late_sessions is not None:
        raise ValueError("Specify either late_sessions or late_from_session, not both.")
    if late_from_session is not None:
        late_from_session = int(late_from_session)
        if late_from_session < 1:
            raise ValueError("late_from_session must be at least 1.")
        is_late = pl.col(exposure_col) >= late_from_session
    else:
        late_session_numbers = sorted({int(value) for value in late_sessions or ()})
        if not late_session_numbers or late_session_numbers[0] < 1:
            raise ValueError("late_sessions must contain positive session numbers.")
        is_late = pl.col(exposure_col).is_in(late_session_numbers)

    session_keys = [subject_col, condition_col, session_col]
    selected_cols = list(dict.fromkeys([*session_keys, order_col]))
    session_table = (
        df.select(selected_cols)
        .unique()
        .sort([subject_col, condition_col, order_col, session_col])
        .with_columns(
            pl.int_range(1, pl.len() + 1)
            .over([subject_col, condition_col])
            .alias(exposure_col)
        )
        .with_columns(
            pl.when(pl.col(exposure_col) == first_session)
            .then(pl.lit(first_label))
            .when(is_late)
            .then(pl.lit(late_label))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias(window_col)
        )
    )

    return df.join(
        session_table.select([*session_keys, exposure_col, window_col]),
        on=session_keys,
        how="left",
        validate="m:1",
    )


def prepare_session_rolling_accuracy(
    df_like,
    *,
    subject_col: str,
    session_col: str,
    trial_col: str,
    accuracy_col: str,
    condition_col: str,
    window_col: str = "session_window",
    rolling_window: int = 20,
    progress_points: int = 101,
) -> dict:
    """Compute session-wise rolling accuracy and its across-subject summary.

    Accuracy is smoothed within each session with a centered rolling window,
    interpolated onto normalized session progress, averaged across sessions
    within each subject, and then summarized across subjects. The returned
    summary includes the mean, variance, standard deviation, SEM, and number of
    contributing subjects. Whole-session accuracy and the temporal variance of
    rolling accuracy follow the same hierarchy.
    """
    df = to_pandas_df(df_like)
    required = {
        subject_col,
        session_col,
        trial_col,
        accuracy_col,
        condition_col,
        window_col,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")

    rolling_window = int(rolling_window)
    progress_points = int(progress_points)
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2.")
    if progress_points < 2:
        raise ValueError("progress_points must be at least 2.")

    min_periods = max(2, int(np.ceil(rolling_window / 2)))
    progress_grid = np.linspace(0.0, 100.0, progress_points)
    session_rows = []
    session_accuracy_rows = []
    group_cols = [subject_col, condition_col, window_col, session_col]
    selected = df[df[window_col].notna()].copy()

    for group_key, session_df in selected.groupby(group_cols, observed=True, sort=False):
        session_df = session_df.copy()
        session_df["_trial_numeric"] = pd.to_numeric(session_df[trial_col], errors="coerce")
        session_df["_accuracy"] = _binary_indicator_series(session_df[accuracy_col])
        session_df = session_df.sort_values("_trial_numeric", kind="mergesort")
        if session_df["_accuracy"].notna().sum() < min_periods:
            continue

        rolling_accuracy = session_df["_accuracy"].rolling(
            rolling_window,
            min_periods=min_periods,
            center=True,
        ).mean()
        session_progress = np.linspace(0.0, 100.0, len(session_df))
        valid = rolling_accuracy.notna().to_numpy()
        if valid.sum() < 2:
            continue

        interpolated = np.interp(
            progress_grid,
            session_progress[valid],
            rolling_accuracy.to_numpy(dtype=float)[valid],
            left=np.nan,
            right=np.nan,
        )
        subject, condition, window, session = group_key
        session_accuracy_rows.append(
            {
                "subject": subject,
                "condition": condition,
                "session_window": window,
                "session": session,
                "session_mean_accuracy": float(session_df["_accuracy"].mean()),
                "session_rolling_variance": float(np.nanvar(interpolated, ddof=1)),
            }
        )
        session_rows.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "condition": condition,
                    "session_window": window,
                    "session": session,
                    "session_progress": progress_grid,
                    "rolling_accuracy": interpolated,
                }
            )
        )

    session_traces = (
        pd.concat(session_rows, ignore_index=True)
        if session_rows
        else pd.DataFrame(
            columns=[
                "subject",
                "condition",
                "session_window",
                "session",
                "session_progress",
                "rolling_accuracy",
            ]
        )
    )
    session_accuracy = pd.DataFrame(
        session_accuracy_rows,
        columns=[
            "subject",
            "condition",
            "session_window",
            "session",
            "session_mean_accuracy",
            "session_rolling_variance",
        ],
    )
    subject_traces = (
        session_traces.groupby(
            ["subject", "condition", "session_window", "session_progress"],
            observed=True,
        )["rolling_accuracy"]
        .mean()
        .reset_index()
        if not session_traces.empty
        else pd.DataFrame(
            columns=[
                "subject",
                "condition",
                "session_window",
                "session_progress",
                "rolling_accuracy",
            ]
        )
    )
    subject_session_accuracy = (
        session_accuracy.groupby(
            ["subject", "condition", "session_window"],
            observed=True,
        ).agg(
            whole_session_accuracy=("session_mean_accuracy", "mean"),
            whole_session_variance=("session_rolling_variance", "mean"),
        )
        .reset_index()
        if not session_accuracy.empty
        else pd.DataFrame(
            columns=[
                "subject",
                "condition",
                "session_window",
                "whole_session_accuracy",
                "whole_session_variance",
            ]
        )
    )
    if subject_traces.empty:
        summary = pd.DataFrame(
            columns=[
                "condition",
                "session_window",
                "session_progress",
                "mean_accuracy",
                "variance_accuracy",
                "std_accuracy",
                "sem_accuracy",
                "n_subjects",
            ]
        )
    else:
        summary = (
            subject_traces.groupby(
                ["condition", "session_window", "session_progress"],
                observed=True,
            )["rolling_accuracy"]
            .agg(
                mean_accuracy="mean",
                variance_accuracy="var",
                std_accuracy="std",
                n_subjects="count",
            )
            .reset_index()
        )
        summary["sem_accuracy"] = summary["std_accuracy"] / np.sqrt(
            summary["n_subjects"].clip(lower=1)
        )

    included_sessions = (
        session_accuracy.groupby(["condition", "session_window"], observed=True)
        .agg(
            included_sessions=("session", "size"),
            included_subjects=("subject", "nunique"),
        )
        .reset_index()
        if not session_accuracy.empty
        else pd.DataFrame(
            columns=[
                "condition",
                "session_window",
                "included_sessions",
                "included_subjects",
            ]
        )
    )
    return {
        "session_traces": session_traces,
        "subject_traces": subject_traces,
        "subject_session_accuracy": subject_session_accuracy,
        "summary": summary,
        "session_counts": included_sessions,
        "meta": {
            "rolling_window": rolling_window,
            "rolling_min_periods": min_periods,
            "progress_points": progress_points,
            "aggregation_unit": "subject_after_session_mean",
        },
    }


def build_bin_centers(
    df: pd.DataFrame,
    *,
    regressor_col: str,
    reg_bin_col: str = "_reg_bin",
    center_col: str = "x_center",
    center_agg: str = "mean",
) -> pd.DataFrame:
    return (
        df.groupby(reg_bin_col, observed=True)
        .agg(**{center_col: (regressor_col, center_agg)})
        .reset_index()
        .sort_values(center_col)
    )


def attach_quantile_bin_column(
    df: pd.DataFrame,
    *,
    value_col: str,
    bin_col: str = "_reg_bin",
    max_bins: int = 10,
    quantiles: Optional[Sequence[float]] = None,
    center_col: str = "x_center",
    center_agg: str = "mean",
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    bin_values, _ = assign_quantile_bins(
        df[value_col],
        max_bins=max_bins,
        quantiles=quantiles,
    )
    if bin_values.dropna().nunique() < 2:
        return None, pd.DataFrame()

    out = df.copy()
    out[bin_col] = bin_values
    out = out[out[bin_col].notna()].copy()
    if out.empty:
        return None, pd.DataFrame()

    centers = build_bin_centers(
        out,
        regressor_col=value_col,
        reg_bin_col=bin_col,
        center_col=center_col,
        center_agg=center_agg,
    )
    bin_order = centers[bin_col].tolist()
    out[bin_col] = pd.Categorical(out[bin_col], categories=bin_order, ordered=True)
    return out, centers


def attach_group_quantile_bin_column(
    df: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Sequence[str],
    bin_col: str,
    max_bins: int,
) -> pd.DataFrame | None:
    out = df.copy()
    out[bin_col] = pd.NA
    for _, idx in out.groupby(list(group_cols), observed=True).groups.items():
        bin_values, _ = assign_quantile_bins(out.loc[idx, value_col], max_bins=max_bins)
        out.loc[idx, bin_col] = bin_values

    out = out[out[bin_col].notna()].copy()
    return None if out.empty else out


def summarize_simple_curve(
    df: pd.DataFrame,
    *,
    subject_col: str,
    reg_bin_col: str,
    regressor_col: str,
    data_col: str,
    model_col: str,
) -> pd.DataFrame:
    summary = (
        df.groupby([subject_col, reg_bin_col], observed=True)
        .agg(
            data_mean=(data_col, "mean"),
            model_mean=(model_col, "mean"),
            x_center=(regressor_col, "mean"),
        )
        .reset_index()
    )
    if summary.empty:
        return summary

    overall = (
        summary.groupby(reg_bin_col, observed=True)
        .agg(
            data_mean=("data_mean", "mean"),
            data_std=("data_mean", "std"),
            data_count=("data_mean", "count"),
            model_mean=("model_mean", "mean"),
            model_std=("model_mean", "std"),
            x_center=("x_center", "mean"),
        )
        .reset_index()
        .sort_values("x_center")
    )
    overall["data_sem"] = overall["data_std"].fillna(0.0) / np.sqrt(
        overall["data_count"].clip(lower=1)
    )
    overall["model_sem"] = overall["model_std"].fillna(0.0) / np.sqrt(
        overall["data_count"].clip(lower=1)
    )
    return overall


def summarize_grouped_panel(
    df: pd.DataFrame,
    *,
    line_group_col: str,
    x_col: str,
    subject_col: str,
    data_col: str,
    model_col: str,
    line_order: list,
    x_order: list | None = None,
    subgroup_col: str | None = None,
    subgroup_value=None,
    base_filter: pd.Series | None = None,
) -> pd.DataFrame:
    plot_df = df.copy()

    if base_filter is not None:
        plot_df = plot_df.loc[base_filter].copy()
    if subgroup_col is not None:
        plot_df = plot_df[plot_df[subgroup_col] == subgroup_value].copy()

    plot_df = plot_df[
        plot_df[line_group_col].notna()
        & plot_df[x_col].notna()
        & plot_df[line_group_col].isin(line_order)
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()

    subj = (
        plot_df.groupby([line_group_col, subject_col, x_col], observed=True)
        .agg(
            data_mean=(data_col, "mean"),
            model_mean=(model_col, "mean"),
        )
        .reset_index()
    )
    if subj.empty:
        return pd.DataFrame()

    agg = (
        subj.groupby([line_group_col, x_col], observed=True)
        .agg(
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
            mm=("model_mean", "mean"),
        )
        .reset_index()
    )
    agg["sem"] = agg["sd"].fillna(0.0) / np.sqrt(agg["nd"].clip(lower=1))

    agg[line_group_col] = pd.Categorical(
        agg[line_group_col], categories=line_order, ordered=True
    )
    if x_order is not None:
        agg[x_col] = pd.Categorical(agg[x_col], categories=x_order, ordered=True)
        agg = agg.sort_values([line_group_col, x_col])
    else:
        agg = agg.sort_values([line_group_col, x_col])

    return agg


def prepare_simple_regressor_curve(
    plot_df,
    *,
    regressor_col: str,
    pred_col: str,
    response_mode: str,
    baseline: float,
    ylabel: str,
    xlabel: str | None = None,
    n_bins: int = 10,
) -> tuple[pd.DataFrame | None, dict]:
    df_pd = to_pandas_df(plot_df)
    required_cols = {regressor_col, "response", pred_col, "subject"}
    if not required_cols.issubset(df_pd.columns):
        return None, {}

    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd[pred_col] = pd.to_numeric(df_pd[pred_col], errors="coerce")
    df_pd = attach_response_right_column(df_pd, response_mode=response_mode)

    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd[pred_col])
        & np.isfinite(df_pd["_response_right"])
    ].copy()
    if df_pd.empty:
        return None, {}

    df_pd, bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=n_bins,
        quantiles=None,
    )
    if df_pd is None:
        return None, {}
    bin_order = bin_centers["_reg_bin"].tolist()

    summary = summarize_simple_curve(
        df_pd,
        subject_col="subject",
        reg_bin_col="_reg_bin",
        regressor_col=regressor_col,
        data_col="_response_right",
        model_col=pred_col,
    )
    if summary.empty:
        return None, {}

    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": ylabel,
        "baseline": baseline,
        "xlim": padded_numeric_limits(
            bin_centers["x_center"]
            if "_signed_delay_cat" in df_pd.columns
            else df_pd[regressor_col],
            absolute_pad=0.25,
        ),
    }
    return summary, meta


def build_trial_logits(view, *, is_mcdr: bool) -> np.ndarray:
    logits_ce = np.einsum("kcf,tf->tkc", view.emission_weights, view.X)
    map_k = view.map_states()
    explicit_logits = logits_ce[np.arange(view.T), map_k, :]
    num_classes = view.num_classes
    baseline_class_idx = int(getattr(view, "baseline_class_idx", 0))
    if not 0 <= baseline_class_idx < int(num_classes):
        raise ValueError(
            f"baseline_class_idx={baseline_class_idx} is invalid for num_classes={num_classes}."
        )
    if explicit_logits.shape[1] != num_classes - 1:
        raise ValueError(
            "Explicit emission logits have incompatible shape: "
            f"expected {num_classes - 1} columns, got {explicit_logits.shape[1]}."
        )

    zero = np.zeros((view.T, 1), dtype=float)
    return np.concatenate(
        [
            explicit_logits[:, :baseline_class_idx],
            zero,
            explicit_logits[:, baseline_class_idx:],
        ],
        axis=1,
    )


def attach_total_fitted_evidence(
    plot_df,
    *,
    adapter,
    views: dict,
    is_mcdr: bool,
) -> pd.DataFrame:
    df_pd = to_pandas_df(plot_df)
    if df_pd.empty or "subject" not in df_pd.columns:
        return df_pd

    df_pd = df_pd.copy().reset_index(drop=True)
    df_pd["_fitted_total_evidence"] = np.nan
    df_pd["_fitted_correct_prob"] = np.nan

    for subject, view in views.items():
        subj_mask = df_pd["subject"].astype(str) == str(subject)
        if not subj_mask.any():
            continue

        subj_df = df_pd.loc[subj_mask].copy().reset_index()
        if len(subj_df) != int(view.T):
            continue

        logits = build_trial_logits(view, is_mcdr=is_mcdr)
        correct_class = adapter.get_correct_class(
            pl.from_pandas(subj_df.drop(columns="index"))
        )
        correct_class = np.asarray(correct_class, dtype=int)

        valid_mask = (correct_class >= 0) & (correct_class < logits.shape[1])
        if not np.any(valid_mask):
            continue

        valid_logits = logits[valid_mask]
        valid_classes = correct_class[valid_mask]
        row_idx = np.arange(valid_logits.shape[0], dtype=int)

        correct_logits = valid_logits[row_idx, valid_classes]
        other_mask = np.ones_like(valid_logits, dtype=bool)
        other_mask[row_idx, valid_classes] = False
        other_logits = other_mask.reshape(valid_logits.shape) & other_mask
        other_logits = valid_logits[other_mask].reshape(
            valid_logits.shape[0], valid_logits.shape[1] - 1
        )

        other_max = np.max(other_logits, axis=1, keepdims=True)
        other_logsumexp = other_max[:, 0] + np.log(
            np.exp(other_logits - other_max).sum(axis=1)
        )

        fitted_evidence = correct_logits - other_logsumexp
        fitted_correct_prob = 1.0 / (1.0 + np.exp(-fitted_evidence))

        target_idx = subj_df.loc[valid_mask, "index"].to_numpy(dtype=int)
        df_pd.loc[target_idx, "_fitted_total_evidence"] = fitted_evidence
        df_pd.loc[target_idx, "_fitted_correct_prob"] = fitted_correct_prob

    return df_pd


def attach_repeat_choice_evidence(
    plot_df,
    *,
    views: dict,
    is_mcdr: bool,
) -> pd.DataFrame:
    df_pd = to_pandas_df(plot_df)
    if df_pd.empty or "subject" not in df_pd.columns:
        return df_pd

    df_pd = df_pd.copy().reset_index(drop=True)
    df_pd["_repeat_choice_evidence"] = np.nan
    df_pd["_p_repeat_model"] = np.nan
    df_pd["_repeat_choice"] = np.nan

    for subject, view in views.items():
        subj_mask = df_pd["subject"].astype(str) == str(subject)
        if not subj_mask.any():
            continue

        subj_df = df_pd.loc[subj_mask].copy().reset_index()
        if len(subj_df) != int(view.T):
            continue

        logits = build_trial_logits(view, is_mcdr=is_mcdr)
        choices = np.asarray(view.y, dtype=int)
        if choices.shape[0] != logits.shape[0]:
            continue

        if "session" in subj_df.columns:
            session_vals = subj_df["session"].astype(str).to_numpy()
        else:
            session_vals = np.zeros(len(subj_df), dtype=str)

        prev_choice = np.full_like(choices, -1, dtype=int)
        prev_choice[1:] = choices[:-1]

        same_session = np.zeros(len(subj_df), dtype=bool)
        same_session[1:] = session_vals[1:] == session_vals[:-1]

        valid_mask = same_session & (prev_choice >= 0) & (prev_choice < logits.shape[1])
        if not np.any(valid_mask):
            continue

        valid_logits = logits[valid_mask]
        valid_prev_choice = prev_choice[valid_mask]
        row_idx = np.arange(valid_logits.shape[0], dtype=int)

        repeat_logits = valid_logits[row_idx, valid_prev_choice]
        other_mask = np.ones_like(valid_logits, dtype=bool)
        other_mask[row_idx, valid_prev_choice] = False
        switched_logits = valid_logits[other_mask].reshape(
            valid_logits.shape[0], valid_logits.shape[1] - 1
        )

        other_max = np.max(switched_logits, axis=1, keepdims=True)
        other_logsumexp = other_max[:, 0] + np.log(
            np.exp(switched_logits - other_max).sum(axis=1)
        )

        repeat_evidence = repeat_logits - other_logsumexp
        p_repeat_model = 1.0 / (1.0 + np.exp(-repeat_evidence))
        repeat_choice = (choices[valid_mask] == valid_prev_choice).astype(float)

        target_idx = subj_df.loc[valid_mask, "index"].to_numpy(dtype=int)
        df_pd.loc[target_idx, "_repeat_choice_evidence"] = repeat_evidence
        df_pd.loc[target_idx, "_p_repeat_model"] = p_repeat_model
        df_pd.loc[target_idx, "_repeat_choice"] = repeat_choice

    return df_pd


def prepare_evidence_curve(
    df_pd: pd.DataFrame,
    *,
    evidence_col: str,
    data_col: str,
    model_col: str,
    baseline: float,
    xlabel: str,
    ylabel: str,
    n_bins: int = 10,
    quantiles: Optional[Sequence[float]] = None,
    group_col: str | None = None,
    group_order: Sequence | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    df = df_pd.copy()
    resolved_group_col, resolved_group_order = resolve_grouping(
        df,
        group_col=group_col,
        group_order=group_order,
    )
    df[evidence_col] = pd.to_numeric(df[evidence_col], errors="coerce")
    df[data_col] = pd.to_numeric(df[data_col], errors="coerce")
    df[model_col] = pd.to_numeric(df[model_col], errors="coerce")

    df = df[
        np.isfinite(df[evidence_col])
        & np.isfinite(df[data_col])
        & np.isfinite(df[model_col])
    ].copy()
    if df.empty:
        return None, {}

    df, bin_centers = attach_quantile_bin_column(
        df,
        value_col=evidence_col,
        bin_col="_bin",
        max_bins=n_bins,
        quantiles=quantiles,
    )
    if df is None:
        return None, {}
    bin_order = bin_centers["_bin"].tolist()

    subject_group_cols = ["subject", "_bin"]
    summary_group_cols = ["_bin"]
    if resolved_group_col is not None:
        df = df[df[resolved_group_col].notna()].copy()
        df = df[df[resolved_group_col].isin(resolved_group_order)].copy()
        subject_group_cols.append(resolved_group_col)
        summary_group_cols.append(resolved_group_col)

    subj = (
        df.groupby(subject_group_cols, observed=True)
        .agg(
            data_mean=(data_col, "mean"),
            model_mean=(model_col, "mean"),
            x_center=(evidence_col, "mean"),
        )
        .reset_index()
    )
    if subj.empty:
        return None, {}

    overall = (
        subj.groupby(summary_group_cols, observed=True)
        .agg(
            data_mean=("data_mean", "mean"),
            data_std=("data_mean", "std"),
            data_count=("data_mean", "count"),
            model_mean=("model_mean", "mean"),
            model_std=("model_mean", "std"),
            x_center=("x_center", "mean"),
        )
        .reset_index()
        .sort_values("x_center")
    )
    overall["data_sem"] = overall["data_std"].fillna(0.0) / np.sqrt(
        overall["data_count"].clip(lower=1)
    )
    overall["model_sem"] = overall["model_std"].fillna(0.0) / np.sqrt(
        overall["data_count"].clip(lower=1)
    )

    meta = {
        "xlabel": xlabel,
        "ylabel": ylabel,
        "baseline": baseline,
    }
    if resolved_group_col is not None:
        meta["line_order"] = resolved_group_order
        meta["legend_title"] = resolved_group_col
    return overall, meta


def resolve_ild_max(
    df: pd.DataFrame,
    ild_col: str,
    ild_max: Optional[float] = None,
) -> float:
    """Return an explicit |ILD| max or infer it from the plotted dataframe."""
    if ild_max is not None:
        value = float(ild_max)
        if np.isfinite(value) and value > 0:
            return value

    if ild_col not in df.columns:
        return 1.0

    values = pd.to_numeric(df[ild_col], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0

    inferred = float(np.max(np.abs(finite)))
    return inferred if inferred > 0 else 1.0


def _normalized_lapse_rates(lapse_rates: Optional[np.ndarray]) -> tuple[float, float]:
    if lapse_rates is None:
        return 0.0, 0.0
    values = np.asarray(lapse_rates, dtype=float).ravel()
    if len(values) >= 2:
        return float(values[0]), float(values[1])
    if len(values) == 1:
        value = float(values[0])
        return value, value
    return 0.0, 0.0


def _valid_trial_weights(X_data: Optional[np.ndarray], trial_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if X_data is None or trial_weights is None:
        return None
    weights = np.asarray(trial_weights, dtype=float).reshape(-1)
    if weights.shape[0] != np.asarray(X_data).shape[0]:
        return None
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    return weights if float(weights.sum()) > 0 else None


def _right_probability(logit, *, right_logit_sign: float) -> np.ndarray:
    right_logit = float(right_logit_sign) * logit
    return 1.0 / (1.0 + np.exp(-right_logit))


def _drug_product_feature_specs(
    names: Sequence[str],
    swept_source_indices: set[int],
) -> list[tuple[int, int, int]]:
    """Return ``(product, drug, source)`` specs for ``drug_x_<source>`` columns."""
    name_to_idx = {str(name): idx for idx, name in enumerate(names)}
    drug_idx = next((idx for idx, name in enumerate(names) if name in {"Drug", "drug", "drug_code"}), None)
    if drug_idx is None:
        return []

    specs: list[tuple[int, int, int]] = []
    for product_idx, name in enumerate(names):
        name = str(name)
        if not name.startswith("drug_x_"):
            continue
        source_idx = name_to_idx.get(name.removeprefix("drug_x_"))
        if source_idx in swept_source_indices:
            specs.append((product_idx, drug_idx, source_idx))
    return specs


def _apply_sweep_values(
    X: np.ndarray,
    values: dict[int, float],
    product_specs: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Overwrite swept columns and recompute simple product features in-place."""
    for idx, value in values.items():
        X[:, idx] = value
    for product_idx, multiplier_idx, source_idx in product_specs:
        X[:, product_idx] = X[:, multiplier_idx] * X[:, source_idx]
    return X


def _stimulus_grid_components(
    X_cols: Sequence[str],
    *,
    ild_max: float,
    n_grid: int,
    stim_param_weight_map: Optional[Callable[[], dict[int, float]]],
) -> dict:
    names = list(X_cols)
    stim_abs_indices = {
        int(name.removeprefix("stim_")): idx
        for idx, name in enumerate(names)
        if isinstance(name, str)
        and name.startswith("stim_")
        and name.removeprefix("stim_").isdigit()
    }
    abs_level_indices: dict[str, dict[int, int]] = {}
    for prefix in ("abs_ILD_hot_", "abs_ild_hot_"):
        indices = {
            int(name.removeprefix(prefix)): idx
            for idx, name in enumerate(names)
            if isinstance(name, str)
            and name.startswith(prefix)
            and name.removeprefix(prefix).isdigit()
        }
        if indices:
            abs_level_indices[prefix] = indices
    stim_param_idx = next((idx for idx, name in enumerate(names) if name == "stim_param"), None)
    stim_param_weights = stim_param_weight_map() if stim_param_idx is not None and stim_param_weight_map else {}
    ild_idx = next(
        (idx for idx, name in enumerate(names) if name in {"stim_vals", "stim_d", "ild_norm", "ILD", "ild", "stimulus"}),
        None,
    )
    stim_side_idx = next((idx for idx, name in enumerate(names) if name == "stim_side"), None)
    abs_ild_idx = next((idx for idx, name in enumerate(names) if name == "abs_ILD"), None)

    abs_levels = set().union(*(set(indices) for indices in abs_level_indices.values())) if abs_level_indices else set()
    if stim_abs_indices or abs_levels or stim_param_idx is not None:
        levels = sorted(set(stim_abs_indices) | abs_levels | set(stim_param_weights) | {0})
        grid = np.asarray(
            sorted({0.0} | {signed for level in levels if level != 0 for signed in (-float(level), float(level))}),
            dtype=float,
        )
    else:
        grid = np.linspace(-ild_max, ild_max, n_grid)

    feature_indices = sorted(
        set(
            ([ild_idx] if ild_idx is not None else [])
            + ([stim_side_idx] if stim_side_idx is not None else [])
            + ([abs_ild_idx] if abs_ild_idx is not None else [])
            + list(stim_abs_indices.values())
            + [idx for indices in abs_level_indices.values() for idx in indices.values()]
            + ([stim_param_idx] if stim_param_idx is not None else [])
        )
    )
    product_specs = _drug_product_feature_specs(names, set(feature_indices))
    feature_indices = sorted(set(feature_indices + [product_idx for product_idx, _, _ in product_specs]))
    return {
        "grid": grid,
        "norm": grid / ild_max,
        "ild_idx": ild_idx,
        "stim_abs_indices": stim_abs_indices,
        "abs_level_indices": abs_level_indices,
        "stim_param_idx": stim_param_idx,
        "stim_param_weights": stim_param_weights,
        "stim_side_idx": stim_side_idx,
        "abs_ild_idx": abs_ild_idx,
        "product_specs": product_specs,
        "feature_indices": feature_indices,
    }


def _stimulus_values_for_grid(component: dict, ild_value: float, ild_norm: float) -> dict[int, float]:
    values = {}
    if component["ild_idx"] is not None:
        values[component["ild_idx"]] = float(ild_norm)
    if component.get("stim_side_idx") is not None:
        values[component["stim_side_idx"]] = 0.0 if np.isclose(ild_value, 0.0) else float(np.sign(ild_value))
    if component.get("abs_ild_idx") is not None:
        values[component["abs_ild_idx"]] = float(abs(ild_norm))
    for stim_abs, stim_abs_idx in component["stim_abs_indices"].items():
        if stim_abs == 0:
            values[stim_abs_idx] = 1.0 if ild_value == 0 else 0.0
        else:
            values[stim_abs_idx] = float(np.sign(ild_value)) if abs(ild_value) == float(stim_abs) else 0.0
    for level_indices in component.get("abs_level_indices", {}).values():
        for abs_level, abs_level_idx in level_indices.items():
            values[abs_level_idx] = 1.0 if abs(ild_value) == float(abs_level) else 0.0
    if component["stim_param_idx"] is not None:
        weights = component["stim_param_weights"]
        if ild_value == 0:
            values[component["stim_param_idx"]] = float(weights.get(0, 0.0))
        else:
            values[component["stim_param_idx"]] = float(np.sign(ild_value)) * float(weights.get(int(abs(ild_value)), 0.0))
    return values


def eval_glm_on_ild_grid(
    weights: np.ndarray,
    X_cols: Sequence[str],
    ild_max: float,
    *,
    n_grid: int = 300,
    lapse_rates: Optional[np.ndarray] = None,
    X_data: Optional[np.ndarray] = None,
    trial_weights: Optional[np.ndarray] = None,
    stim_param_weight_map: Optional[Callable[[], dict[int, float]]] = None,
    right_logit_sign: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a fitted binary GLM-HMM emission model as P(right) over ILD."""
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[None, ...]
    K, _C_m1, M = W.shape

    X_cols_list = list(X_cols)
    component = _stimulus_grid_components(
        X_cols_list,
        ild_max=ild_max,
        n_grid=n_grid,
        stim_param_weight_map=stim_param_weight_map,
    )
    ild_grid = component["grid"]
    ild_norm = component["norm"]
    bias_idx = next((idx for idx, name in enumerate(X_cols_list) if name == "bias"), None)

    gL, gR = _normalized_lapse_rates(lapse_rates)
    p_right = np.zeros((K, len(ild_grid)))
    weights_t = _valid_trial_weights(X_data, trial_weights)
    stim_feature_indices = component["feature_indices"]
    product_specs = component.get("product_specs", [])

    if X_data is not None and stim_feature_indices:
        X_base = np.asarray(X_data, dtype=float).copy()
        for k in range(K):
            w = W[k, 0, :]
            for grid_idx, (ild_value, stim_value_norm) in enumerate(zip(ild_grid, ild_norm, strict=False)):
                stim_values = _stimulus_values_for_grid(component, ild_value, stim_value_norm)
                X_sweep = _apply_sweep_values(X_base.copy(), stim_values, product_specs)
                p_trial = gL + (1.0 - gL - gR) * _right_probability(
                    X_sweep @ w,
                    right_logit_sign=right_logit_sign,
                )
                p_right[k, grid_idx] = (
                    float(np.average(p_trial, weights=weights_t))
                    if weights_t is not None
                    else float(np.mean(p_trial))
                )
    else:
        if X_data is not None:
            col_means = np.asarray(X_data, dtype=float).mean(axis=0)
        else:
            col_means = np.zeros(M)
        if bias_idx is not None:
            col_means[bias_idx] = 1.0

        X_grid = np.tile(col_means, (len(ild_grid), 1))
        if stim_feature_indices:
            X_grid[:, stim_feature_indices] = 0.0
        for row_idx, (ild_value, stim_value_norm) in enumerate(zip(ild_grid, ild_norm, strict=False)):
            stim_values = _stimulus_values_for_grid(component, ild_value, stim_value_norm)
            for idx, value in stim_values.items():
                X_grid[row_idx, idx] = value
        if bias_idx is not None:
            X_grid[:, bias_idx] = 1.0
        _apply_sweep_values(X_grid, {}, product_specs)

        for k in range(K):
            p_right[k] = gL + (1.0 - gL - gR) * _right_probability(
                X_grid @ W[k, 0, :],
                right_logit_sign=right_logit_sign,
            )

    return (ild_grid, p_right[0]) if K == 1 else (ild_grid, p_right)


def eval_glm_on_feature_grid(
    weights: np.ndarray,
    X_cols: Sequence[str],
    feature_name: str,
    grid_min: float,
    grid_max: float,
    *,
    n_grid: int = 300,
    lapse_rates: Optional[np.ndarray] = None,
    X_data: Optional[np.ndarray] = None,
    trial_weights: Optional[np.ndarray] = None,
    right_logit_sign: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a fitted binary GLM-HMM emission model as P(right) over a regressor."""
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[None, ...]
    K, _C_m1, M = W.shape

    X_cols_list = list(X_cols)
    feat_idx = next((idx for idx, name in enumerate(X_cols_list) if name == feature_name), None)
    bias_idx = next((idx for idx, name in enumerate(X_cols_list) if name == "bias"), None)
    if feat_idx is None:
        raise KeyError(f"Feature {feature_name!r} not found in X_cols.")

    grid = np.linspace(float(grid_min), float(grid_max), int(n_grid))
    gL, gR = _normalized_lapse_rates(lapse_rates)
    p_right = np.zeros((K, len(grid)))
    product_specs = _drug_product_feature_specs(X_cols_list, {feat_idx})

    X_base = None
    if X_data is not None:
        candidate = np.asarray(X_data, dtype=float).copy()
        if candidate.ndim == 2 and candidate.shape[1] == M:
            X_base = candidate
    weights_t = _valid_trial_weights(X_base, trial_weights)

    if X_base is not None:
        for k in range(K):
            w = W[k, 0, :]
            for grid_idx, grid_value in enumerate(grid):
                X_sweep = _apply_sweep_values(X_base.copy(), {feat_idx: float(grid_value)}, product_specs)
                p_trial = gL + (1.0 - gL - gR) * _right_probability(
                    X_sweep @ w,
                    right_logit_sign=right_logit_sign,
                )
                p_right[k, grid_idx] = (
                    float(np.average(p_trial, weights=weights_t))
                    if weights_t is not None
                    else float(np.mean(p_trial))
                )
    else:
        col_means = np.zeros(M)
        if bias_idx is not None:
            col_means[bias_idx] = 1.0
        X_grid = np.tile(col_means, (len(grid), 1))
        X_grid[:, feat_idx] = grid
        _apply_sweep_values(X_grid, {}, product_specs)
        if bias_idx is not None:
            X_grid[:, bias_idx] = 1.0
        for k in range(K):
            p_right[k] = gL + (1.0 - gL - gR) * _right_probability(
                X_grid @ W[k, 0, :],
                right_logit_sign=right_logit_sign,
            )

    return (grid, p_right[0]) if K == 1 else (grid, p_right)


def rank_ordered_arrays_store(views: dict, *, include_lapse_rates: bool = True) -> dict:
    """Return the minimal arrays-store payload with state axis ordered by rank."""
    out = {}
    for subject, view in views.items():
        order = sorted(view.state_rank_by_idx, key=lambda raw_idx: view.state_rank_by_idx[raw_idx])
        payload = {
            "emission_weights": view.emission_weights[order],
            "X_cols": view.feat_names,
            "X": view.X,
            "smoothed_probs": view.smoothed_probs[:, order],
        }
        if include_lapse_rates:
            payload["lapse_rates"] = getattr(view, "lapse_rates", None)
        out[subject] = payload
    return out


def _view_columns(arrays_store: dict, subject, X_cols: Optional[Sequence[str]]) -> Optional[list[str]]:
    cols = X_cols
    if cols is None:
        raw = arrays_store[subject].get("X_cols")
        if raw is None:
            return None
        cols = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]
    return list(cols)


def _valid_view_X(subject_store: dict, weights: np.ndarray) -> Optional[np.ndarray]:
    X_data = subject_store.get("X")
    if X_data is None:
        return None
    X_data = np.asarray(X_data, dtype=float)
    if X_data.ndim != 2 or X_data.shape[1] != np.asarray(weights).shape[-1]:
        return None
    return X_data


def _state_restricted_X(subject_store: dict, X_data: Optional[np.ndarray], state_k: Optional[int]) -> Optional[np.ndarray]:
    if X_data is None or state_k is None:
        return X_data
    gamma = subject_store.get("smoothed_probs")
    if gamma is None:
        return X_data
    mask = np.argmax(np.asarray(gamma), axis=1) == state_k
    return X_data[mask] if mask.sum() > 0 else X_data


def mean_glm_ild_curve(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    *,
    ild_max: float,
    state_k: Optional[int] = None,
    stim_param_weight_map: Optional[Callable[[], dict[int, float]]] = None,
    right_logit_sign: float = -1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    all_p: list[np.ndarray] = []
    ild_grid: Optional[np.ndarray] = None

    for subject in subjects:
        if subject not in arrays_store:
            continue
        subject_store = arrays_store[subject]
        weights = subject_store.get("emission_weights")
        if weights is None:
            continue
        cols = _view_columns(arrays_store, subject, X_cols)
        if cols is None:
            continue
        X_data = _state_restricted_X(subject_store, _valid_view_X(subject_store, weights), state_k)
        lapse_rates = subject_store.get("lapse_rates")
        if lapse_rates is not None:
            lapse_rates = np.asarray(lapse_rates, dtype=float).ravel()
            if not np.any(lapse_rates > 0):
                lapse_rates = None

        try:
            grid, probs = eval_glm_on_ild_grid(
                weights,
                cols,
                ild_max=ild_max,
                lapse_rates=lapse_rates,
                X_data=X_data,
                stim_param_weight_map=stim_param_weight_map,
                right_logit_sign=right_logit_sign,
            )
        except Exception:
            continue

        if probs.ndim == 2 and state_k is not None:
            probs = probs[state_k]
        elif probs.ndim == 2:
            gamma = subject_store.get("smoothed_probs")
            if gamma is not None:
                weights_k = np.asarray(gamma, dtype=float).mean(axis=0)
                weight_sum = float(weights_k.sum())
                probs = np.average(probs, axis=0, weights=weights_k / weight_sum) if weight_sum > 0 else probs.mean(axis=0)
            else:
                probs = probs.mean(axis=0)
        all_p.append(probs)
        ild_grid = grid

    if not all_p or ild_grid is None:
        return None
    return ild_grid, np.mean(all_p, axis=0)


def subject_glm_ild_curves(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    *,
    ild_max: float,
    state_k: Optional[int] = None,
    stim_param_weight_map: Optional[Callable[[], dict[int, float]]] = None,
    right_logit_sign: float = -1.0,
) -> dict:
    return {
        subject: curve
        for subject in subjects
        if (
            curve := mean_glm_ild_curve(
                arrays_store,
                [subject],
                X_cols,
                ild_max=ild_max,
                state_k=state_k,
                stim_param_weight_map=stim_param_weight_map,
                right_logit_sign=right_logit_sign,
            )
        )
        is not None
    }


def mean_glm_feature_curve(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    *,
    feature_name: str,
    grid_min: float,
    grid_max: float,
    state_k: Optional[int] = None,
    n_grid: int = 300,
    right_logit_sign: float = -1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    all_p: list[np.ndarray] = []
    feature_grid: Optional[np.ndarray] = None

    for subject in subjects:
        if subject not in arrays_store:
            continue
        subject_store = arrays_store[subject]
        weights = subject_store.get("emission_weights")
        if weights is None:
            continue
        cols = _view_columns(arrays_store, subject, X_cols)
        if cols is None or feature_name not in cols:
            continue
        X_data = _state_restricted_X(subject_store, _valid_view_X(subject_store, weights), state_k)
        lapse_rates = subject_store.get("lapse_rates")
        if lapse_rates is not None:
            lapse_rates = np.asarray(lapse_rates, dtype=float).ravel()
            if not np.any(lapse_rates > 0):
                lapse_rates = None

        try:
            grid, probs = eval_glm_on_feature_grid(
                weights,
                cols,
                feature_name=feature_name,
                grid_min=grid_min,
                grid_max=grid_max,
                n_grid=n_grid,
                lapse_rates=lapse_rates,
                X_data=X_data,
                right_logit_sign=right_logit_sign,
            )
        except Exception:
            continue

        if probs.ndim == 2 and state_k is not None:
            probs = probs[state_k]
        elif probs.ndim == 2:
            gamma = subject_store.get("smoothed_probs")
            if gamma is not None:
                weights_k = np.asarray(gamma, dtype=float).mean(axis=0)
                weight_sum = float(weights_k.sum())
                probs = np.average(probs, axis=0, weights=weights_k / weight_sum) if weight_sum > 0 else probs.mean(axis=0)
            else:
                probs = probs.mean(axis=0)
        all_p.append(probs)
        feature_grid = grid

    if not all_p or feature_grid is None:
        return None
    return feature_grid, np.mean(all_p, axis=0)


def subject_glm_feature_curves(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    *,
    feature_name: str,
    grid_min: float,
    grid_max: float,
    state_k: Optional[int] = None,
    n_grid: int = 300,
    right_logit_sign: float = -1.0,
) -> dict:
    return {
        subject: curve
        for subject in subjects
        if (
            curve := mean_glm_feature_curve(
                arrays_store,
                [subject],
                X_cols,
                feature_name=feature_name,
                grid_min=grid_min,
                grid_max=grid_max,
                state_k=state_k,
                n_grid=n_grid,
                right_logit_sign=right_logit_sign,
            )
        )
        is not None
    }


def mean_weighted_empirical_curve(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    subj_col: str,
    weight_col: Optional[str] = None,
    grid: Optional[np.ndarray] = None,
    grid_min: Optional[float] = None,
    grid_max: Optional[float] = None,
    n_grid: int = 300,
    bandwidth: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Average subject-level kernel smoothers for observed choices."""
    cols = [x_col, y_col, subj_col]
    if weight_col is not None:
        cols.append(weight_col)
    d = df.dropna(subset=[col for col in cols if col in df.columns]).copy()
    if d.empty:
        return None

    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if weight_col is not None and weight_col in d.columns:
        d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return None

    finite_x = d[x_col].to_numpy(dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.size == 0:
        return None

    if grid is None:
        lo = float(np.min(finite_x)) if grid_min is None else float(grid_min)
        hi = float(np.max(finite_x)) if grid_max is None else float(grid_max)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        grid = np.linspace(lo, hi, int(n_grid))
    else:
        grid = np.asarray(grid, dtype=float)

    curves: list[np.ndarray] = []
    for _, grp in d.groupby(subj_col, observed=True):
        x = pd.to_numeric(grp[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp[y_col], errors="coerce").to_numpy(dtype=float)
        if weight_col is not None and weight_col in grp.columns:
            weights = pd.to_numeric(grp[weight_col], errors="coerce").to_numpy(dtype=float)
        else:
            weights = np.ones_like(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
        if not np.any(mask):
            continue
        x = x[mask]
        y = y[mask]
        weights = weights[mask]
        if x.size == 0 or float(weights.sum()) <= 0:
            continue

        bw = bandwidth
        if bw is None:
            unique_x = np.unique(np.sort(x))
            if unique_x.size >= 2:
                bw = float(np.median(np.diff(unique_x))) * 1.5
            else:
                span = float(np.max(x) - np.min(x))
                bw = span / 6.0 if span > 0 else 1.0
        bw = max(float(bw), 1e-6)

        z = (grid[:, None] - x[None, :]) / bw
        kernel_weights = np.exp(-0.5 * z * z) * weights[None, :]
        denom = kernel_weights.sum(axis=1)
        numer = kernel_weights @ y
        curves.append(np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0))

    if not curves:
        return None
    return grid, np.nanmean(np.vstack(curves), axis=0)


def quantile_bin_spec(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot bin an empty array.")

    unique_vals = np.unique(x)
    if unique_vals.size == 1:
        value = float(unique_vals[0])
        return np.asarray([value - 0.5, value + 0.5], dtype=float), np.asarray([value], dtype=float)

    bin_edges = np.unique(np.asarray(np.quantile(x, np.linspace(0.0, 1.0, max(int(n_bins), 1) + 1)), dtype=float))
    if bin_edges.size < 2:
        value = float(unique_vals[0])
        return np.asarray([value - 0.5, value + 0.5], dtype=float), np.asarray([value], dtype=float)
    return bin_edges, 0.5 * (bin_edges[:-1] + bin_edges[1:])


def quantile_bin_assignments(
    values: np.ndarray,
    n_bins: int,
    *,
    bin_edges: Optional[np.ndarray] = None,
    bin_centers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if bin_edges is None or bin_centers is None:
        bin_edges, bin_centers = quantile_bin_spec(values, n_bins=n_bins)
    bin_idx = np.digitize(np.asarray(values, dtype=float), bin_edges, right=True) - 1
    return np.clip(bin_idx, 0, len(bin_centers) - 1).astype(int), bin_centers


def binned_feature_summary(
    df: pd.DataFrame,
    *,
    feature_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    n_bins: int = 9,
    weight_col: Optional[str] = None,
    bin_edges: Optional[np.ndarray] = None,
    bin_centers: Optional[np.ndarray] = None,
) -> Optional[Tuple[pd.DataFrame, list[float]]]:
    needed = [feature_col, choice_col, pred_col, subj_col]
    d = df.dropna(subset=[col for col in needed if col in df.columns]).copy()
    if d.empty:
        return None

    d[feature_col] = pd.to_numeric(d[feature_col], errors="coerce")
    d[choice_col] = pd.to_numeric(d[choice_col], errors="coerce")
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d = d.dropna(subset=[feature_col, choice_col, pred_col])
    if d.empty:
        return None

    bin_idx, bin_centers = quantile_bin_assignments(
        d[feature_col].to_numpy(dtype=float),
        n_bins=n_bins,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
    )
    d["_x_bin"] = bin_idx
    centers = pd.DataFrame({"_x_bin": np.arange(len(bin_centers), dtype=int), "center": bin_centers})

    if weight_col is not None and weight_col in d.columns:
        rows = []
        for (x_bin, subj), grp in d.groupby(["_x_bin", subj_col], observed=True):
            weights = pd.to_numeric(grp[weight_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp[choice_col], errors="coerce").to_numpy(dtype=float)
            model = pd.to_numeric(grp[pred_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(weights) & np.isfinite(y) & np.isfinite(model) & (weights > 0)
            if not np.any(mask):
                continue
            weights = weights[mask]
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                continue
            rows.append(
                {
                    "_x_bin": x_bin,
                    subj_col: subj,
                    "data_mean": float(np.dot(y[mask], weights) / weight_sum),
                    "model_mean": float(np.dot(model[mask], weights) / weight_sum),
                }
            )
        subj = pd.DataFrame(rows)
        if not subj.empty:
            subj = subj.merge(centers, on="_x_bin", how="left")
    else:
        subj = (
            d.groupby(["_x_bin", subj_col], observed=True)
            .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
            .reset_index()
            .merge(centers, on="_x_bin", how="left")
        )
    if subj.empty:
        return None

    agg = (
        subj.groupby("_x_bin", observed=True)
        .agg(
            x=("center", "median"),
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
            mm=("model_mean", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("x")
    )
    agg["sd"] = agg["sd"].fillna(0.0)
    agg["sem"] = agg["sd"] / np.sqrt(agg["nd"].clip(lower=1))
    return subj, agg["x"].tolist()


def _rank_source_maps(views: dict) -> tuple[int, dict]:
    K = next(iter(views.values())).K
    maps = {
        subj: {int(rank): int(raw_idx) for raw_idx, rank in view.state_rank_by_idx.items()}
        for subj, view in views.items()
    }
    return K, maps


def _attach_ranked_source_cols(
    df: pd.DataFrame,
    views: dict,
    *,
    subj_col: str,
    source_col: Callable[[int], str],
    target_col: Callable[[int], str],
    missing_message: str,
) -> pd.DataFrame:
    if df.empty or subj_col not in df.columns or not views:
        return df

    K, raw_by_rank_by_subj = _rank_source_maps(views)
    out = df.copy()
    target_cols = [target_col(rank) for rank in range(K)]
    if all(col in out.columns for col in target_cols):
        return out

    for rank, dst_col in enumerate(target_cols):
        if dst_col in out.columns:
            continue
        values = np.full(len(out), np.nan, dtype=float)
        for subj, idx in out.groupby(subj_col, observed=True).groups.items():
            raw_by_rank = raw_by_rank_by_subj.get(subj) or raw_by_rank_by_subj.get(str(subj))
            raw_idx = None if raw_by_rank is None else raw_by_rank.get(rank)
            if raw_idx is None:
                continue
            src_col = source_col(raw_idx)
            if src_col not in out.columns:
                raise KeyError(missing_message.format(src_col=src_col))
            row_idx = np.asarray(idx, dtype=int)
            values[row_idx] = pd.to_numeric(out.iloc[row_idx][src_col], errors="coerce").to_numpy(dtype=float)
        out[dst_col] = values
    return out


def attach_rank_posterior_cols(
    df: pd.DataFrame,
    views: dict,
    *,
    subj_col: str = "subject",
) -> pd.DataFrame:
    """Attach rank-aligned posterior columns from raw predictive-state columns."""
    return _attach_ranked_source_cols(
        df,
        views,
        subj_col=subj_col,
        source_col=lambda raw_idx: f"p_state_pred_{raw_idx}",
        target_col=lambda rank: f"_p_state_rank_{rank}",
        missing_message=(
            "Missing required predictive state column {src_col!r}. "
            "Rebuild trial_df with the updated predictive-state export."
        ),
    )


def attach_rank_state_model_cols(
    df: pd.DataFrame,
    views: dict,
    *,
    subj_col: str = "subject",
    base_col: str = "pR_state",
) -> pd.DataFrame:
    """Attach rank-aligned state model columns from raw-state trial_df columns."""
    return _attach_ranked_source_cols(
        df,
        views,
        subj_col=subj_col,
        source_col=lambda raw_idx: f"{base_col}_{raw_idx}",
        target_col=lambda rank: f"_{base_col}_rank_{rank}",
        missing_message=(
            "Missing required state-conditional model column {src_col!r}. "
            "Rebuild trial_df with the updated per-state prediction export."
        ),
    )


def ranked_state_labels(views: dict) -> dict[int, str]:
    labels: dict[int, str] = {}
    for view in views.values():
        for raw_idx, label in view.state_name_by_idx.items():
            labels.setdefault(view.state_rank_by_idx.get(int(raw_idx), int(raw_idx)), label)
    return labels


STATE_SCORING_RULES: tuple[str, ...] = ("+", "-", "abs")


def normalize_state_scoring_rule(rule: str | None) -> str:
    """Normalize UI/legacy state-scoring rule names."""
    key = str(rule or "+").strip().lower()
    if key in {"+", "pos", "positive", "w"}:
        return "+"
    if key in {"-", "neg", "negative", "-w"}:
        return "-"
    if key in {"abs", "|w|", "absolute", "absolute value"}:
        return "abs"
    raise ValueError(f"Unknown state scoring rule {rule!r}; expected one of {STATE_SCORING_RULES}.")


def choose_state_scoring_feature(
    feature_names: Sequence[str],
    preferred: Sequence[str] = (),
    requested: str | None = None,
) -> str | None:
    """Pick a fitted regressor for state scoring."""
    features = [str(feature) for feature in feature_names]
    feature_set = set(features)
    if requested and requested in feature_set:
        return str(requested)
    for feature in preferred:
        if feature in feature_set:
            return str(feature)
    for feature in features:
        if feature != "bias" and not feature.startswith("bias_"):
            return feature
    return features[0] if features else None


def score_states_by_regressor(
    weights: np.ndarray,
    feature_names: Sequence[str],
    feature_name: str | None,
    rule: str | None = "+",
    *,
    weight_row_idx: int = 0,
) -> np.ndarray:
    """Score each state from one fitted emission-regressor column."""
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[:, None, :]
    if W.ndim != 3:
        raise ValueError(f"Expected emission weights with shape (K, C-1, M), got {W.shape}.")
    if W.shape[0] == 0:
        return np.asarray([], dtype=float)

    feature = choose_state_scoring_feature(feature_names, requested=feature_name)
    name2fi = {str(name): idx for idx, name in enumerate(feature_names)}
    if feature is None or feature not in name2fi:
        vals = W[:, min(max(int(weight_row_idx), 0), W.shape[1] - 1), :].mean(axis=1)
    else:
        row_idx = min(max(int(weight_row_idx), 0), W.shape[1] - 1)
        vals = W[:, row_idx, name2fi[feature]]

    normalized_rule = normalize_state_scoring_rule(rule)
    if normalized_rule == "-":
        return -vals
    if normalized_rule == "abs":
        return np.abs(vals)
    return vals


def score_states_by_scoring_terms(
    weights: np.ndarray,
    feature_names: Sequence[str],
    terms: Sequence[tuple[str, str | int]],
    *,
    default_rule: str | None = "+",
    weight_row_idx: int = 0,
) -> np.ndarray:
    """Score each state from one or more explicit scoring terms.

    A term is ``(feature, rule_or_row)``. String selectors are scoring rules
    such as ``"pos"``, ``"neg"``, or ``"abs"`` applied to ``weight_row_idx``.
    Integer selectors are explicit emission-weight rows, used by multi-class
    tasks where a semantic score averages over multiple class-specific weights.
    """
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[:, None, :]
    if W.ndim != 3:
        raise ValueError(f"Expected emission weights with shape (K, C-1, M), got {W.shape}.")
    if W.shape[0] == 0:
        return np.asarray([], dtype=float)

    name2fi = {str(name): idx for idx, name in enumerate(feature_names)}
    scores = np.zeros(W.shape[0], dtype=float)
    n_terms = 0

    for feature_name, selector in terms:
        feature = str(feature_name)
        if feature not in name2fi:
            continue
        if isinstance(selector, str):
            row_idx = min(max(int(weight_row_idx), 0), W.shape[1] - 1)
            rule = selector
        else:
            row_idx = min(max(int(selector), 0), W.shape[1] - 1)
            rule = default_rule

        vals = W[:, row_idx, name2fi[feature]]
        normalized_rule = normalize_state_scoring_rule(rule)
        if normalized_rule == "-":
            vals = -vals
        elif normalized_rule == "abs":
            vals = np.abs(vals)

        scores += vals
        n_terms += 1

    if n_terms == 0:
        return np.asarray([], dtype=float)
    return scores / n_terms


def label_states_by_regressor(
    arrays_store: dict,
    names: dict,
    K: int,
    subjects: Sequence,
    *,
    scoring_key: str | None = None,
    scoring_options: dict | None = None,
    primary_feature: str | None = None,
    primary_rule: str | None = "+",
    split_feature: str | None = None,
    split_rule: str | None = "+",
    preferred_features: Sequence[str] = (),
    preferred_split_features: Sequence[str] = ("bias", "bias_param"),
    weight_row_idx: int = 0,
) -> tuple[dict, dict]:
    """Label HMM states from fitted emission weights using a generic rule.

    The primary regressor ranks engagement. For K=4 the top two primary-score
    states become Engaged L/R and the remaining two become Disengaged L/R; the
    optional split regressor orders each pair into L/R. For K=3 the split
    regressor separates the two non-engaged states into Biased L/R.
    """
    base_feat = [str(feature) for feature in names.get("X_cols", [])]
    state_labels: dict = {}
    state_order: dict = {}

    for subj in subjects:
        subject_store = arrays_store.get(subj) if subj in arrays_store else arrays_store.get(str(subj))
        W = subject_store.get("emission_weights") if subject_store is not None else None
        if W is None:
            state_labels[subj] = {k: f"State {k + 1}" for k in range(K)}
            state_order[subj] = list(range(K))
            continue

        feat = [str(feature) for feature in subject_store.get("X_cols", base_feat)]
        option_terms = []
        if primary_feature is None and scoring_key is not None and scoring_options:
            option_terms = list(scoring_options.get(scoring_key, []) or [])

        if option_terms:
            primary_scores = score_states_by_scoring_terms(
                W,
                feat,
                option_terms,
                default_rule=primary_rule,
                weight_row_idx=weight_row_idx,
            )
        else:
            score_feature = choose_state_scoring_feature(
                feat,
                preferred=preferred_features,
                requested=primary_feature,
            )
            primary_scores = score_states_by_regressor(
                W,
                feat,
                score_feature,
                primary_rule,
                weight_row_idx=weight_row_idx,
            )
        if primary_scores.size == 0:
            state_labels[subj] = {k: f"State {k + 1}" for k in range(K)}
            state_order[subj] = list(range(K))
            continue

        split_requested = None if split_feature in {None, "", "(none)", "None"} else split_feature
        resolved_split_feature = choose_state_scoring_feature(
            feat,
            preferred=preferred_split_features,
            requested=split_requested,
        )
        split_scores = score_states_by_regressor(
            W,
            feat,
            resolved_split_feature,
            split_rule,
            weight_row_idx=weight_row_idx,
        )

        ranking = sorted(range(K), key=lambda k: (primary_scores[k], -k), reverse=True)

        if K <= 0:
            labels = {}
            order = []
        elif K == 1:
            labels = {0: "Engaged"}
            order = [0]
        elif K == 2:
            engaged_k = int(ranking[0])
            other_k = next(k for k in range(K) if k != engaged_k)
            labels = {engaged_k: "Engaged", other_k: "Disengaged"}
            order = [engaged_k, other_k]
        elif K == 3:
            engaged_k = int(ranking[0])
            others = [k for k in range(K) if k != engaged_k]
            ordered_by_split = sorted(others, key=lambda k: (split_scores[k], k))
            biased_l, biased_r = int(ordered_by_split[0]), int(ordered_by_split[-1])
            labels = {
                engaged_k: "Engaged",
                biased_l: "Biased L",
                biased_r: "Biased R",
            }
            order = [engaged_k, biased_l, biased_r]
        elif K == 4:
            engaged_states = [int(k) for k in ranking[:2]]
            disengaged_states = [int(k) for k in range(K) if k not in set(engaged_states)]

            def _split_left_right(state_ids: Sequence[int]) -> tuple[int, int]:
                ordered = sorted(state_ids, key=lambda k: (split_scores[k], k))
                return int(ordered[0]), int(ordered[-1])

            engaged_l, engaged_r = _split_left_right(engaged_states)
            disengaged_l, disengaged_r = _split_left_right(disengaged_states)
            labels = {
                engaged_l: "Engaged L",
                engaged_r: "Engaged R",
                disengaged_l: "Disengaged L",
                disengaged_r: "Disengaged R",
            }
            order = [engaged_l, engaged_r, disengaged_l, disengaged_r]
        else:
            engaged_k = int(ranking[0])
            labels = {engaged_k: "Engaged"}
            rest = [int(k) for k in ranking[1:]]
            for idx, state_idx in enumerate(rest, start=1):
                labels[state_idx] = f"Disengaged {idx}"
            order = [engaged_k, *rest]

        state_labels[subj] = labels
        state_order[subj] = order

    return state_labels, state_order


def _ticks_from_values(values, tick_values: Optional[Sequence[float]]) -> np.ndarray:
    source = tick_values if tick_values is not None else values
    ticks = sorted({float(value) for value in source if pd.notna(value)})
    if not ticks and tick_values is not None:
        ticks = sorted({float(value) for value in values if pd.notna(value)})
    return np.asarray(ticks, dtype=float)


def _curve_payload_from_subject_summary(
    subj_agg: pd.DataFrame,
    *,
    x_col: str,
    tick_values: Optional[Sequence[float]] = None,
    empirical_smooth=None,
) -> dict | None:
    if subj_agg.empty:
        return None
    x_values = sorted(subj_agg[x_col].unique())
    ticks = _ticks_from_values(x_values, tick_values)
    agg = (
        subj_agg.groupby(x_col, observed=True)
        .agg(
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
            mm=("model_mean", "mean"),
        )
        .reindex(x_values)
    )
    nd = agg["nd"].clip(lower=1).to_numpy(dtype=float)
    payload = {
        "subject_summary": subj_agg,
        "x": np.array(x_values, dtype=float),
        "ticks": ticks,
        "data_mean": agg["md"].to_numpy(dtype=float),
        "data_sem": agg["sd"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(nd),
        "model_mean": agg["mm"].to_numpy(dtype=float),
    }
    payload["empirical_smooth"] = empirical_smooth
    return payload


def prepare_psych_panel_payload(
    df: pd.DataFrame,
    *,
    x_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    tick_values: Optional[Sequence[float]] = None,
) -> dict | None:
    if df.empty:
        return None

    subj_agg = (
        df.groupby([subj_col, x_col], observed=True)
        .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
        .reset_index()
    )
    if subj_agg.empty:
        return None

    return _curve_payload_from_subject_summary(subj_agg, x_col=x_col, tick_values=tick_values)


def prepare_psych_state_panel_payload(
    df: pd.DataFrame,
    *,
    x_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    weight_col: Optional[str] = None,
    smooth_grid: Optional[np.ndarray] = None,
    tick_values: Optional[Sequence[float]] = None,
) -> dict | None:
    if df.empty:
        return None

    empirical_smooth = None
    if weight_col is not None and weight_col in df.columns:
        empirical_smooth = mean_weighted_empirical_curve(
            df,
            x_col=x_col,
            y_col=choice_col,
            subj_col=subj_col,
            weight_col=weight_col,
            grid=smooth_grid,
        )
        rows = []
        for (subj, x_value), grp in df.groupby([subj_col, x_col], observed=True):
            weights = pd.to_numeric(grp[weight_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp[choice_col], errors="coerce").to_numpy(dtype=float)
            model = pd.to_numeric(grp[pred_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(weights) & np.isfinite(y) & np.isfinite(model) & (weights > 0)
            if not np.any(mask):
                continue
            weights = weights[mask]
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                continue
            rows.append(
                {
                    subj_col: subj,
                    x_col: x_value,
                    "data_mean": float(np.dot(y[mask], weights) / weight_sum),
                    "model_mean": float(np.dot(model[mask], weights) / weight_sum),
                }
            )
        subj_agg = pd.DataFrame(rows)
    else:
        subj_agg = (
            df.groupby([subj_col, x_col], observed=True)
            .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
            .reset_index()
        )
    if subj_agg.empty:
        return None

    return _curve_payload_from_subject_summary(
        subj_agg,
        x_col=x_col,
        tick_values=tick_values,
        empirical_smooth=empirical_smooth,
    )


def prepare_regressor_state_panel_payload(
    df: pd.DataFrame,
    *,
    feature_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    n_bins: int = 9,
    weight_col: Optional[str] = None,
    bin_edges: Optional[np.ndarray] = None,
    bin_centers: Optional[np.ndarray] = None,
    smooth_grid: Optional[np.ndarray] = None,
) -> dict | None:
    if df.empty:
        return None

    empirical_smooth = None
    if weight_col is not None and weight_col in df.columns:
        empirical_smooth = mean_weighted_empirical_curve(
            df,
            x_col=feature_col,
            y_col=choice_col,
            subj_col=subj_col,
            weight_col=weight_col,
            grid=smooth_grid,
        )

    summary = binned_feature_summary(
        df,
        feature_col=feature_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        n_bins=n_bins,
        weight_col=weight_col,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
    )
    if summary is None:
        return None
    subj_agg, _x_ticks = summary

    agg = (
        subj_agg.groupby("_x_bin", observed=True)
        .agg(
            x=("center", "median"),
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
            mm=("model_mean", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("x")
    )
    nd = agg["nd"].clip(lower=1).to_numpy(dtype=float)
    return {
        "subject_summary": subj_agg,
        "empirical_smooth": empirical_smooth,
        "x": agg["x"].to_numpy(dtype=float),
        "data_mean": agg["md"].to_numpy(dtype=float),
        "data_sem": agg["sd"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(nd),
        "model_mean": agg["mm"].to_numpy(dtype=float),
    }


COUNTERFACTUAL_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("Full fitted", "full"),
    ("Fixed bias", "fixed_bias"),
    ("Fixed lapses", "fixed_lapses"),
)


def _counterfactual_sem(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def _counterfactual_subject_rb(
    work_df: pd.DataFrame,
    repeat_values,
    *,
    x_col: str,
    subject_col: str,
    previous_choice_col: str | None = None,
) -> pd.DataFrame:
    keep_cols = [subject_col, x_col]
    if previous_choice_col is not None:
        keep_cols.append(previous_choice_col)
    tmp = work_df[keep_cols].copy()
    tmp["_repeat_value"] = np.asarray(repeat_values, dtype=float)
    if previous_choice_col is not None:
        by_side = (
            tmp.dropna(subset=[subject_col, x_col, previous_choice_col, "_repeat_value"])
            .groupby([subject_col, x_col, previous_choice_col], observed=True)["_repeat_value"]
            .mean()
            .reset_index(name="rb_by_previous_choice")
        )
        subject = (
            by_side.groupby([subject_col, x_col], observed=True)["rb_by_previous_choice"]
            .mean()
            .reset_index(name="rb")
        )
        subject_counts = (
            tmp.dropna(subset=[subject_col, x_col, previous_choice_col, "_repeat_value"])
            .groupby([subject_col, x_col], observed=True)["_repeat_value"]
            .size()
            .reset_index(name="n_trials")
        )
        subject = subject.merge(subject_counts, on=[subject_col, x_col], how="left")
    else:
        subject = (
            tmp.dropna(subset=[subject_col, x_col, "_repeat_value"])
            .groupby([subject_col, x_col], observed=True)["_repeat_value"]
            .mean()
            .reset_index(name="rb")
        )
    if subject.empty:
        return pd.DataFrame()

    # Average animals, not trials, so high-trial-count animals do not dominate.
    summary = (
        subject.groupby(x_col, observed=True)["rb"]
        .agg(rb_mean="mean", rb_sem=_counterfactual_sem, n_subjects="count")
        .reset_index()
    )
    return summary


def _counterfactual_subject_overall_rb(
    work_df: pd.DataFrame,
    repeat_values,
    *,
    subject_col: str,
    previous_choice_col: str | None = None,
) -> pd.DataFrame:
    """Compute one repetition-bias value per animal."""
    keep_cols = [subject_col]
    if previous_choice_col is not None:
        keep_cols.append(previous_choice_col)
    tmp = work_df[keep_cols].copy()
    tmp["_repeat_value"] = np.asarray(repeat_values, dtype=float)
    if previous_choice_col is not None:
        valid = tmp.dropna(subset=[subject_col, previous_choice_col, "_repeat_value"])
        by_side = (
            valid.groupby([subject_col, previous_choice_col], observed=True)["_repeat_value"]
            .mean()
            .reset_index(name="rb_by_previous_choice")
        )
        subject = (
            by_side.groupby(subject_col, observed=True)["rb_by_previous_choice"]
            .mean()
            .reset_index(name="rb")
        )
        subject_counts = (
            valid.groupby(subject_col, observed=True)["_repeat_value"]
            .size()
            .reset_index(name="n_trials")
        )
        return subject.merge(subject_counts, on=subject_col, how="left")

    return (
        tmp.dropna(subset=[subject_col, "_repeat_value"])
        .groupby(subject_col, observed=True)["_repeat_value"]
        .agg(rb="mean", n_trials="count")
        .reset_index()
    )


def _counterfactual_subject_lag_match(
    work_df: pd.DataFrame,
    choice_values,
    *,
    max_lag: int,
    subject_col: str,
) -> pd.DataFrame:
    """Compute p(choice_t = experimental choice_{t-L}) per animal, then average animals."""
    choice = np.asarray(choice_values, dtype=float)
    rows = []
    for lag in range(1, int(max_lag) + 1):
        lag_col = f"_response_right_lag_{lag:02d}"
        if lag_col not in work_df.columns:
            continue

        tmp = work_df[[subject_col, lag_col]].copy()
        tmp["_choice_value"] = choice
        tmp = tmp.dropna(subset=[subject_col, lag_col, "_choice_value"]).copy()
        if tmp.empty:
            continue

        tmp["_lag_match"] = (
            tmp["_choice_value"].to_numpy(dtype=float) == tmp[lag_col].to_numpy(dtype=float)
        ).astype(float)
        subject = (
            tmp.groupby(subject_col, observed=True)["_lag_match"]
            .mean()
            .reset_index(name="lag_match")
        )
        if subject.empty:
            continue

        values = subject["lag_match"].to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        sem = _counterfactual_sem(values)
        rows.append(
            {
                "lag": int(lag),
                "lag_match_mean": mean,
                "lag_match_sem": sem,
                "lag_match_lo": float(np.clip(mean - sem, 0.0, 1.0)),
                "lag_match_hi": float(np.clip(mean + sem, 0.0, 1.0)),
                "n_subjects": int(len(subject)),
            }
        )
    return pd.DataFrame(rows)


def _counterfactual_subject_expected_lag_match(
    work_df: pd.DataFrame,
    p_right_values,
    *,
    max_lag: int,
    subject_col: str,
) -> pd.DataFrame:
    """Compute expected p(choice_t = experimental choice_{t-L}) per animal."""
    p_right = np.asarray(p_right_values, dtype=float)
    rows = []
    for lag in range(1, int(max_lag) + 1):
        lag_col = f"_response_right_lag_{lag:02d}"
        if lag_col not in work_df.columns:
            continue

        tmp = work_df[[subject_col, lag_col]].copy()
        tmp["_p_right"] = p_right
        tmp = tmp.dropna(subset=[subject_col, lag_col, "_p_right"]).copy()
        if tmp.empty:
            continue

        lag_choice = tmp[lag_col].to_numpy(dtype=float)
        p_right_sub = tmp["_p_right"].to_numpy(dtype=float)
        tmp["_lag_match"] = np.where(
            np.isclose(lag_choice, 1.0),
            p_right_sub,
            1.0 - p_right_sub,
        )
        subject = (
            tmp.groupby(subject_col, observed=True)["_lag_match"]
            .mean()
            .reset_index(name="lag_match")
        )
        if subject.empty:
            continue

        values = subject["lag_match"].to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        sem = _counterfactual_sem(values)
        rows.append(
            {
                "lag": int(lag),
                "lag_match_mean": mean,
                "lag_match_sem": sem,
                "lag_match_lo": float(np.clip(mean - sem, 0.0, 1.0)),
                "lag_match_hi": float(np.clip(mean + sem, 0.0, 1.0)),
                "n_subjects": int(len(subject)),
            }
        )
    return pd.DataFrame(rows)


def _format_delay_tick(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _signed_delay_order_and_labels(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    series = df["_signed_delay_cat"]
    if hasattr(series, "cat"):
        order = [str(value) for value in series.cat.categories if str(value) in set(series.dropna().astype(str))]
    else:
        order = [str(value) for value in series.dropna().unique()]

    def label(value: str) -> str:
        return _format_delay_tick(float(value))

    return order, [label(value) for value in order]


def _counterfactual_fit_axes(df: pd.DataFrame, *, task_name: str) -> tuple[pd.DataFrame, dict]:
    """Choose the psychometric x-axis and the RB summary x-axis for each task."""
    out = df.copy()
    if task_name == "2AFC":
        if "ILD" in out.columns:
            x = pd.to_numeric(out["ILD"], errors="coerce")
            out["_counterfactual_fit_x"] = np.where(
                np.isclose(x, -70.0),
                -16.0,
                np.where(np.isclose(x, 70.0), 16.0, x),
            )
            out["_counterfactual_rb_x"] = x.abs()
            ticks = sorted(out["_counterfactual_rb_x"].dropna().unique())
            return out, {
                "xlabel": "|ILD| (dB)",
                "fit_xlabel": "ILD (dB)",
                "x_col": "_counterfactual_rb_x",
                "xticks": ticks,
                "x_tick_labels": [f"{float(value):g}" for value in ticks],
                "invert_x": True,
                "baseline": 0.5,
            }
        if "stimulus" in out.columns:
            x = pd.to_numeric(out["stimulus"], errors="coerce")
            out["_counterfactual_fit_x"] = x
            out["_counterfactual_rb_x"] = x.abs()
            return out, {
                "xlabel": "|Stimulus|",
                "fit_xlabel": "Stimulus",
                "x_col": "_counterfactual_rb_x",
                "invert_x": True,
                "baseline": 0.5,
            }

    delay_tasks = {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}
    if task_name in delay_tasks:
        out = attach_signed_delay_columns(out)
        if "_signed_delay_cat" in out.columns and out["_signed_delay_cat"].notna().any():
            order, labels = _signed_delay_order_and_labels(out)
            code_map = {value: idx for idx, value in enumerate(order)}
            out["_counterfactual_fit_x"] = out["_signed_delay_cat"].astype(str).map(code_map).astype(float)
            fit_ticks = list(range(len(order)))
            fit_tick_labels = labels
        elif "delay" in out.columns:
            out["_counterfactual_fit_x"] = pd.to_numeric(out["delay"], errors="coerce")
            fit_ticks = None
            fit_tick_labels = None
        else:
            raise ValueError("Delay counterfactuals require a delay column.")

        delay_source = "delay" if "delay" in out.columns else "delays"
        delay = pd.to_numeric(out[delay_source], errors="coerce")
        out["_counterfactual_rb_x"] = delay
        ticks = sorted(delay.dropna().unique())
        return out, {
            "xlabel": "Delay (s)",
            "fit_xlabel": "Signed delay" if fit_ticks is not None else "Delay (s)",
            "fit_xticks": fit_ticks,
            "fit_x_tick_labels": fit_tick_labels,
            "x_col": "_counterfactual_rb_x",
            "xticks": ticks,
            "x_tick_labels": [_format_delay_tick(value) for value in ticks],
            "invert_x": False,
            "baseline": 0.5,
        }

    raise ValueError("Action-trace counterfactuals are implemented only for 2AFC and delay tasks.")


def _fit_counterfactual_full_model(
    curve_table: pd.DataFrame,
    *,
    bin_order: Sequence[str],
    lapse_max: float,
) -> pd.DataFrame:
    rows = []
    for bin_label in bin_order:
        curve = curve_table[curve_table["action_bin"] == bin_label]
        if curve.empty:
            continue
        fit = fit_lapse_logistic_curve(
            curve["_counterfactual_fit_x"].to_numpy(dtype=float),
            curve["p_right"].to_numpy(dtype=float),
            weights=curve["n_trials"].to_numpy(dtype=float),
            group=bin_label,
            lapse_max=float(lapse_max),
            min_points=4,
        )
        if fit is None:
            continue
        rows.append(
            {
                "scenario": "Full fitted",
                "kind": "full",
                "action_bin": bin_label,
                "n_trials": int(curve["n_trials"].sum()),
                "slope": float(fit.slope),
                "bias": float(fit.bias),
                "lapse_left": float(fit.lapse_left),
                "lapse_right": float(fit.lapse_right),
            }
        )
    return pd.DataFrame(rows)


def _interpolate_counterfactual_params(
    work: pd.DataFrame,
    fit_table: pd.DataFrame,
    bin_centers: pd.DataFrame,
    *,
    regressor_col: str,
) -> pd.DataFrame:
    """Map each real A_t value to fitted psychometric parameters by interpolation."""
    center_table = bin_centers.rename(columns={"_reg_bin": "action_bin"}).copy()
    center_table["action_bin"] = center_table["action_bin"].astype(str)
    param_table = (
        fit_table.merge(center_table[["action_bin", "x_center"]], on="action_bin", how="left")
        .dropna(subset=["x_center"])
        .sort_values("x_center")
    )
    if param_table.empty:
        raise ValueError("Cannot interpolate parameter-fixed simulations without A_t bin centers.")

    at_values = pd.to_numeric(work[regressor_col], errors="coerce").to_numpy(dtype=float)
    if param_table["x_center"].nunique() == 1:
        return pd.DataFrame(
            {
                param: np.full(len(work), float(param_table[param].iloc[0]))
                for param in ["slope", "bias", "lapse_left", "lapse_right"]
            },
            index=work.index,
        )

    centers = param_table["x_center"].to_numpy(dtype=float)
    params = {}
    for param in ["slope", "bias", "lapse_left", "lapse_right"]:
        params[param] = np.interp(
            at_values,
            centers,
            param_table[param].to_numpy(dtype=float),
            left=float(param_table[param].iloc[0]),
            right=float(param_table[param].iloc[-1]),
        )
    return pd.DataFrame(params, index=work.index)


def build_action_trace_counterfactual(
    plot_df,
    *,
    task_name: str,
    regressor_col: str,
    n_bins: int = 4,
    n_simulations: int = 200,
    lapse_max: float = 1.0,
    max_history_lag: int = 10,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Fit psychometrics by A_t bin, then run parameter-fixed simulations.

    The observed choice history is kept fixed per animal. Each simulation only resamples
    the current response from the fitted psychometrics, and repetition bias compares that
    simulated current response with the same animal's observed previous response. RB is
    first computed conditional on previous response side and then averaged across sides
    within animal, matching compute_rb_by_x for binary tasks. The full fitted simulation
    interpolates the fitted parameters at each trial's real A_t value; the controls fix
    bias or lapses.
    """
    df = to_pandas_df(plot_df)
    if regressor_col not in df.columns:
        raise ValueError(f"Missing action-trace regressor column {regressor_col!r}.")
    if "subject" not in df.columns or "response" not in df.columns:
        raise ValueError("Counterfactual analysis requires 'subject' and 'response' columns.")

    df, meta = _counterfactual_fit_axes(df, task_name=task_name)
    df = attach_response_right_column(df, response_mode="pm1_or_prob")
    df, bin_centers = attach_quantile_bin_column(df, value_col=regressor_col, max_bins=int(n_bins))
    if df is None or bin_centers.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    bin_order = [str(value) for value in bin_centers["_reg_bin"].tolist()]
    df["_counterfactual_action_bin"] = df["_reg_bin"].astype(str)

    # Keep experimental history fixed; only the current simulated response changes.
    max_history_lag = max(1, int(max_history_lag))
    for lag in range(1, max_history_lag + 1):
        df[f"_response_right_lag_{lag:02d}"] = df.groupby("subject", observed=True)["_response_right"].shift(lag)
    df["_prev_response_right"] = df["_response_right_lag_01"]

    required = [
        "_counterfactual_fit_x",
        "_counterfactual_rb_x",
        "_response_right",
        "_prev_response_right",
        "_counterfactual_action_bin",
    ]
    work = df.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    # First build the empirical psychometric curves: P(Right) vs stimulus evidence
    # separately for each A_t quantile.
    curve_table = (
        work.groupby(["_counterfactual_action_bin", "_counterfactual_fit_x"], observed=True)
        .agg(p_right=("_response_right", "mean"), n_trials=("_response_right", "count"))
        .reset_index()
        .rename(columns={"_counterfactual_action_bin": "action_bin"})
    )

    full_fit_table = _fit_counterfactual_full_model(
        curve_table,
        bin_order=bin_order,
        lapse_max=float(lapse_max),
    )
    if full_fit_table.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), full_fit_table, meta

    valid_bins = [bin_label for bin_label in bin_order if bin_label in set(full_fit_table["action_bin"])]
    work = work[work["_counterfactual_action_bin"].isin(valid_bins)].copy()
    fit_table = full_fit_table.copy()

    fit_weights = full_fit_table["n_trials"].to_numpy(dtype=float)
    pooled = {
        param: float(np.average(full_fit_table[param].to_numpy(dtype=float), weights=fit_weights))
        for param in ["slope", "bias", "lapse_left", "lapse_right"]
    }
    full_trial_params = _interpolate_counterfactual_params(
        work,
        full_fit_table,
        bin_centers,
        regressor_col=regressor_col,
    )

    empirical_repeat = (
        work["_response_right"].to_numpy(dtype=float) == work["_prev_response_right"].to_numpy(dtype=float)
    ).astype(float)
    empirical_subject = _counterfactual_subject_overall_rb(
        work,
        empirical_repeat,
        subject_col="subject",
        previous_choice_col="_prev_response_right",
    ).rename(columns={"rb": "empirical_rb", "n_trials": "empirical_n_trials"})
    empirical = _counterfactual_subject_rb(
        work,
        empirical_repeat,
        x_col="_counterfactual_rb_x",
        subject_col="subject",
        previous_choice_col="_prev_response_right",
    )
    if not empirical.empty:
        empirical["scenario"] = "Empirical"
        empirical["rb_lo"] = np.clip(empirical["rb_mean"] - empirical["rb_sem"], 0.0, 1.0)
        empirical["rb_hi"] = np.clip(empirical["rb_mean"] + empirical["rb_sem"], 0.0, 1.0)

    empirical_lag = _counterfactual_subject_lag_match(
        work,
        work["_response_right"].to_numpy(dtype=float),
        max_lag=max_history_lag,
        subject_col="subject",
    )
    if not empirical_lag.empty:
        empirical_lag["scenario"] = "Empirical"

    x = work["_counterfactual_fit_x"].to_numpy(dtype=float)
    previous_animal_response = work["_prev_response_right"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    simulation_rows = []
    lag_simulation_rows = []
    full_subject_rows = []

    for scenario, kind in COUNTERFACTUAL_SCENARIOS:
        params = full_trial_params.copy()
        # Full fitted: interpolate the fitted psychometric parameters at each real A_t.
        # Fixed bias: keep the real-A_t slope/lapses, but average the bias across bins.
        # Fixed lapses: keep the real-A_t slope/bias, but average lapse rates across bins.
        if kind == "fixed_bias":
            params["bias"] = pooled["bias"]
        elif kind == "fixed_lapses":
            params["lapse_left"] = pooled["lapse_left"]
            params["lapse_right"] = pooled["lapse_right"]

        params_np = params.to_numpy(dtype=float)
        p_right = lapse_logistic_probability(
            x,
            slope=params_np[:, 0],
            bias=params_np[:, 1],
            lapse_left=params_np[:, 2],
            lapse_right=params_np[:, 3],
        )
        p_right = np.clip(p_right, 1e-6, 1.0 - 1e-6)

        for simulation_idx in range(int(n_simulations)):
            sim_right = (rng.random(len(work)) < p_right).astype(float)
            previous_choice_for_rb = previous_animal_response
            # previous_choice_for_rb = (
            #     pd.Series(sim_right, index=work.index)
            #     .groupby(work["subject"], observed=True)
            #     .shift(1)
            #     .to_numpy(dtype=float)
            # )
            sim_repeat = np.where(
                np.isfinite(previous_choice_for_rb),
                (sim_right == previous_choice_for_rb).astype(float),
                np.nan,
            )
            if kind == "full":
                subject_rb = _counterfactual_subject_overall_rb(
                    work,
                    sim_repeat,
                    subject_col="subject",
                    previous_choice_col="_prev_response_right",
                )
                if not subject_rb.empty:
                    subject_rb["simulation"] = simulation_idx
                    full_subject_rows.append(subject_rb)

            summary = _counterfactual_subject_rb(
                work,
                sim_repeat,
                x_col="_counterfactual_rb_x",
                subject_col="subject",
                previous_choice_col="_prev_response_right",
            )
            if not summary.empty:
                summary["scenario"] = scenario
                summary["simulation"] = simulation_idx
                simulation_rows.append(summary)

            lag_summary = _counterfactual_subject_lag_match(
                work,
                sim_right,
                max_lag=max_history_lag,
                subject_col="subject",
            )
            if not lag_summary.empty:
                lag_summary["scenario"] = scenario
                lag_summary["simulation"] = simulation_idx
                lag_simulation_rows.append(lag_summary)

    if simulation_rows:
        simulated = pd.concat(simulation_rows, ignore_index=True)
        model_summary = (
            simulated.groupby(["scenario", "_counterfactual_rb_x"], observed=True)["rb_mean"]
            .agg(
                rb_mean="mean",
                rb_lo=lambda values: float(np.nanquantile(values, 0.025)),
                rb_hi=lambda values: float(np.nanquantile(values, 0.975)),
            )
            .reset_index()
        )
        model_summary["rb_sem"] = (model_summary["rb_hi"] - model_summary["rb_lo"]) / 3.92
        model_summary["n_subjects"] = np.nan
    else:
        model_summary = pd.DataFrame()

    if lag_simulation_rows:
        lag_simulated = pd.concat(lag_simulation_rows, ignore_index=True)
        lag_model_summary = (
            lag_simulated.groupby(["scenario", "lag"], observed=True)["lag_match_mean"]
            .agg(
                lag_match_mean="mean",
                lag_match_lo=lambda values: float(np.nanquantile(values, 0.025)),
                lag_match_hi=lambda values: float(np.nanquantile(values, 0.975)),
            )
            .reset_index()
        )
        lag_model_summary["lag_match_sem"] = (
            lag_model_summary["lag_match_hi"] - lag_model_summary["lag_match_lo"]
        ) / 3.92
        lag_model_summary["n_subjects"] = np.nan
    else:
        lag_model_summary = pd.DataFrame()

    if full_subject_rows:
        full_subject = pd.concat(full_subject_rows, ignore_index=True)
        full_subject_summary = (
            full_subject.groupby("subject", observed=True)["rb"]
            .agg(
                full_fitted_rb="mean",
                full_fitted_rb_lo=lambda values: float(np.nanquantile(values, 0.025)),
                full_fitted_rb_hi=lambda values: float(np.nanquantile(values, 0.975)),
            )
            .reset_index()
        )
        subject_scatter = empirical_subject.merge(full_subject_summary, on="subject", how="inner")
        subject_scatter["delta_full_minus_empirical"] = (
            subject_scatter["full_fitted_rb"] - subject_scatter["empirical_rb"]
        )
    else:
        subject_scatter = pd.DataFrame()

    rb_summary = pd.concat(
        [frame for frame in [empirical, model_summary] if frame is not None and not frame.empty],
        ignore_index=True,
    )
    lag_summary = pd.concat(
        [frame for frame in [empirical_lag, lag_model_summary] if frame is not None and not frame.empty],
        ignore_index=True,
    )
    meta.update(
        {
            "regressor_col": regressor_col,
            "regressor_label": display_regressor_name(regressor_col),
            "n_simulations": int(n_simulations),
            "max_history_lag": int(max_history_lag),
            "history_reference": "same animal's observed previous response",
            "rb_estimator": "mean over previous response sides within animal, then animal average",
            "pooled": pooled,
            "models": {
                "Full fitted": "slope, bias, and lapses are interpolated at each trial's real A_t value",
                "Fixed bias": "bias is averaged across A_t bins; slope and lapses use each trial's real A_t value",
                "Fixed lapses": "lapses are averaged across A_t bins; slope and bias use each trial's real A_t value",
            },
        }
    )
    return rb_summary, lag_summary, subject_scatter, fit_table, meta


def build_action_trace_parameter_fixed_simulations(*args, **kwargs):
    """Alias for the A_t psychometric parameter-fixed simulation analysis."""
    return build_action_trace_counterfactual(*args, **kwargs)


def build_action_trace_model_prediction_rb(
    plot_df,
    *,
    task_name: str,
    max_history_lag: int = 10,
    p_right_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Compare empirical RB with full-model trial-level predicted probabilities.

    This uses existing per-trial model inference, so no psychometric refit and no
    choice simulation are performed. The model RB is the expected p(repeat) from
    each trial's predicted P(right), compared against the same animal's empirical
    previous choice and aggregated with the same side-balanced estimator as
    compute_rb_by_x for binary tasks.
    """
    df = to_pandas_df(plot_df)
    if "subject" not in df.columns or "response" not in df.columns:
        raise ValueError("Model-prediction RB requires 'subject' and 'response' columns.")

    p_candidates = [p_right_col] if p_right_col is not None else []
    p_candidates.extend(["p_model_right", "p_pred", "pR"])
    p_col = next((col for col in p_candidates if col in df.columns), None)
    if p_col is None:
        raise ValueError("Model-prediction RB requires one of: 'p_model_right', 'p_pred', or 'pR'.")

    df, meta = _counterfactual_fit_axes(df, task_name=task_name)
    df = attach_response_right_column(df, response_mode="pm1_or_prob")
    max_history_lag = max(1, int(max_history_lag))
    for lag in range(1, max_history_lag + 1):
        df[f"_response_right_lag_{lag:02d}"] = df.groupby("subject", observed=True)["_response_right"].shift(lag)
    df["_prev_response_right"] = df["_response_right_lag_01"]
    df["_model_p_right"] = pd.to_numeric(df[p_col], errors="coerce")

    required = [
        "_counterfactual_rb_x",
        "_response_right",
        "_prev_response_right",
        "_model_p_right",
    ]
    work = df.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    empirical_repeat = (
        work["_response_right"].to_numpy(dtype=float) == work["_prev_response_right"].to_numpy(dtype=float)
    ).astype(float)
    p_right = np.clip(work["_model_p_right"].to_numpy(dtype=float), 0.0, 1.0)
    previous_response = work["_prev_response_right"].to_numpy(dtype=float)
    model_repeat = np.where(np.isclose(previous_response, 1.0), p_right, 1.0 - p_right)

    empirical_subject = _counterfactual_subject_overall_rb(
        work,
        empirical_repeat,
        subject_col="subject",
        previous_choice_col="_prev_response_right",
    ).rename(columns={"rb": "empirical_rb", "n_trials": "empirical_n_trials"})
    model_subject = _counterfactual_subject_overall_rb(
        work,
        model_repeat,
        subject_col="subject",
        previous_choice_col="_prev_response_right",
    ).rename(columns={"rb": "full_fitted_rb", "n_trials": "full_fitted_n_trials"})
    subject_scatter = empirical_subject.merge(model_subject, on="subject", how="inner")
    if not subject_scatter.empty:
        subject_scatter["delta_full_minus_empirical"] = (
            subject_scatter["full_fitted_rb"] - subject_scatter["empirical_rb"]
        )

    empirical = _counterfactual_subject_rb(
        work,
        empirical_repeat,
        x_col="_counterfactual_rb_x",
        subject_col="subject",
        previous_choice_col="_prev_response_right",
    )
    if not empirical.empty:
        empirical["scenario"] = "Data"
        empirical["rb_lo"] = np.clip(empirical["rb_mean"] - empirical["rb_sem"], 0.0, 1.0)
        empirical["rb_hi"] = np.clip(empirical["rb_mean"] + empirical["rb_sem"], 0.0, 1.0)

    model = _counterfactual_subject_rb(
        work,
        model_repeat,
        x_col="_counterfactual_rb_x",
        subject_col="subject",
        previous_choice_col="_prev_response_right",
    )
    if not model.empty:
        model["scenario"] = "Full fitted"
        model["rb_lo"] = np.clip(model["rb_mean"] - model["rb_sem"], 0.0, 1.0)
        model["rb_hi"] = np.clip(model["rb_mean"] + model["rb_sem"], 0.0, 1.0)

    empirical_lag = _counterfactual_subject_lag_match(
        work,
        work["_response_right"].to_numpy(dtype=float),
        max_lag=max_history_lag,
        subject_col="subject",
    )
    if not empirical_lag.empty:
        empirical_lag["scenario"] = "Data"
    model_lag = _counterfactual_subject_expected_lag_match(
        work,
        p_right,
        max_lag=max_history_lag,
        subject_col="subject",
    )
    if not model_lag.empty:
        model_lag["scenario"] = "Full fitted"

    rb_summary = pd.concat(
        [frame for frame in [empirical, model] if frame is not None and not frame.empty],
        ignore_index=True,
    )
    lag_summary = pd.concat(
        [frame for frame in [empirical_lag, model_lag] if frame is not None and not frame.empty],
        ignore_index=True,
    )
    meta.update(
        {
            "max_history_lag": int(max_history_lag),
            "model_probability_col": p_col,
            "history_reference": "same animal's observed previous response",
            "rb_estimator": "mean over previous response sides within animal, then animal average",
            "models": {
                "Full fitted": f"trial-level model P(right) from {p_col!r}; no refit or simulation",
            },
        }
    )
    return rb_summary, lag_summary, subject_scatter, meta



def _binary_indicator_series(values) -> pd.Series:
    """Coerce common correct/error encodings to a numeric 1/0 series."""
    series = pd.Series(values).copy()
    numeric = pd.to_numeric(series, errors="coerce")
    missing_numeric = numeric.isna() & series.notna()
    if not missing_numeric.any():
        return numeric.astype(float)

    labels = series.astype("string").str.strip().str.lower()
    mapped = pd.Series(np.nan, index=series.index, dtype=float)
    mapped[labels.isin({"1", "true", "correct", "hit", "success"})] = 1.0
    mapped[labels.isin({"0", "false", "error", "incorrect", "miss", "failure"})] = 0.0
    return numeric.where(numeric.notna(), mapped).astype(float)


def _normalized_lag_correlation(
    x,
    y=None,
    *,
    max_lag: int = 50,
    demean: bool = True,
    bidirectional: bool = False,
) -> pd.DataFrame:
    """Normalized lag correlation for one or two binary-like vectors."""
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")

    x_values = _binary_indicator_series(x).to_numpy(dtype=float)
    y_values = x_values if y is None else _binary_indicator_series(y).to_numpy(dtype=float)
    n_values = min(x_values.size, y_values.size)
    if n_values < 2:
        return pd.DataFrame({"lag": [], "corr": [], "n": []})

    x_values = x_values[:n_values]
    y_values = y_values[:n_values]

    if demean:
        x_values = x_values - np.nanmean(x_values)
        y_values = y_values - np.nanmean(y_values)

    def corr_at_lag(a_values: np.ndarray, b_values: np.ndarray, lag: int) -> tuple[float, int]:
        if lag == 0:
            a = a_values
            b = b_values
        else:
            a = a_values[:-lag]
            b = b_values[lag:]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            return np.nan, int(valid.sum())
        av = a[valid]
        bv = b[valid]
        denom = float(np.sqrt(np.nansum(av * av) * np.nansum(bv * bv)))
        corr = np.nan if denom <= 0 else float(np.nansum(av * bv) / denom)
        return corr, int(valid.sum())

    rows = []
    for lag in range(1, min(max_lag, n_values - 1) + 1):
        corr, n = corr_at_lag(x_values, y_values, lag)
        if bidirectional:
            reverse_corr, reverse_n = corr_at_lag(y_values, x_values, lag)
            finite_corr = [value for value in (corr, reverse_corr) if np.isfinite(value)]
            corr = float(np.mean(finite_corr)) if finite_corr else np.nan
            n = int(np.nanmean([n, reverse_n]))
        rows.append({"lag": lag, "corr": corr, "n": n})

    return pd.DataFrame(rows)


def binary_autocorrelation(x, *, max_lag: int = 50, demean: bool = True) -> pd.DataFrame:
    """Normalized autocorrelation of a binary vector for lags 1..max_lag."""
    corr = _normalized_lag_correlation(x, max_lag=max_lag, demean=demean)
    return corr.rename(columns={"corr": "autocorr"})


def binary_crosscorrelation(x, y, *, max_lag: int = 50, demean: bool = True) -> pd.DataFrame:
    """Normalized cross-correlation between two binary vectors for lags 1..max_lag."""
    corr = _normalized_lag_correlation(
        x,
        y,
        max_lag=max_lag,
        demean=demean,
        bidirectional=True,
    )
    return corr.rename(columns={"corr": "crosscorr"})


def prepare_session_accuracy_repetition_timescale(
    session_df,
    *,
    choice_col: str,
    outcome_col: str,
    trial_index_col: str | None = None,
    running_window: int = 20,
    max_lag: int = 50,
) -> dict:
    """Prepare running accuracy/repetition and autocorrelations for one session."""
    df = to_pandas_df(session_df).copy()

    required = {choice_col, outcome_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    if trial_index_col is not None:
        if trial_index_col not in df.columns:
            raise ValueError(f"Missing trial index column {trial_index_col!r}.")
        df["_trial_index_numeric"] = pd.to_numeric(df[trial_index_col], errors="coerce")
        df = df.sort_values("_trial_index_numeric", kind="mergesort").copy()
        trial_index = df.pop("_trial_index_numeric")
    else:
        trial_index = pd.Series(np.arange(len(df)), index=df.index, dtype=float)

    running_window = int(running_window)
    if running_window < 1:
        raise ValueError("running_window must be at least 1.")
    min_periods = min(running_window, max(2, running_window // 4))
    max_lag = int(max_lag)

    outcome = _binary_indicator_series(df[outcome_col])
    choice = df[choice_col]
    valid_choice_pair = choice.notna() & choice.shift(1).notna()

    repetition = choice.eq(choice.shift(1)).astype(float)
    repetition = repetition.where(valid_choice_pair, np.nan)

    trace = pd.DataFrame(
        {
            "trial_index": trial_index.to_numpy(dtype=float),
            "outcome": outcome.to_numpy(dtype=float),
            "repetition": repetition.to_numpy(dtype=float),
        }
    )

    trace["running_accuracy"] = (
        trace["outcome"]
        .rolling(running_window, min_periods=min_periods)
        .mean()
    )
    trace["running_repetition"] = (
        trace["repetition"]
        .rolling(running_window, min_periods=min_periods)
        .mean()
    )
    trace["running_repetition_bias"] = trace["running_repetition"]

    outcome_ac = binary_autocorrelation(trace["outcome"], max_lag=max_lag)
    outcome_ac["signal"] = "Outcome"

    repetition_ac = binary_autocorrelation(trace["repetition"], max_lag=max_lag)
    repetition_ac["signal"] = "Repetition"

    autocorr = pd.concat([outcome_ac, repetition_ac], ignore_index=True)

    return {
        "trace": trace,
        "autocorr": autocorr,
        "meta": {
            "running_window": running_window,
            "running_min_periods": min_periods,
            "max_lag": max_lag,
        },
    }


def _prepare_session_binary_sequences(
    df: pd.DataFrame,
    *,
    subject_col: str,
    session_col: str,
    choice_col: str,
    outcome_col: str,
    trial_index_col: str | None,
) -> pd.DataFrame:
    required = {subject_col, session_col, choice_col, outcome_col}
    if trial_index_col is not None:
        required.add(trial_index_col)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}.")

    out = df.copy()
    if trial_index_col is not None:
        out["_trial_index_numeric"] = pd.to_numeric(out[trial_index_col], errors="coerce")
    else:
        out["_trial_index_numeric"] = out.groupby(
            [subject_col, session_col],
            observed=True,
        ).cumcount()
    out = out.sort_values(
        [subject_col, session_col, "_trial_index_numeric"],
        kind="mergesort",
    ).copy()
    out["_outcome_binary"] = _binary_indicator_series(out[outcome_col]).to_numpy(dtype=float)
    valid_choice_pair = (
        out[choice_col].notna()
        & out.groupby([subject_col, session_col], observed=True)[choice_col].shift(1).notna()
    )
    repetition = out[choice_col].eq(
        out.groupby([subject_col, session_col], observed=True)[choice_col].shift(1)
    ).astype(float)
    out["_repetition_binary"] = repetition.where(valid_choice_pair, np.nan).to_numpy(dtype=float)
    return out[
        [
            subject_col,
            session_col,
            "_trial_index_numeric",
            "_outcome_binary",
            "_repetition_binary",
        ]
    ].rename(
        columns={
            subject_col: "subject",
            session_col: "session",
            "_trial_index_numeric": "trial_index",
            "_outcome_binary": "outcome",
            "_repetition_binary": "repetition",
        }
    )


def prepare_corrected_behavior_autocorrelograms(
    df_like,
    *,
    subject_col: str = "subject",
    session_col: str = "session",
    choice_col: str,
    outcome_col: str,
    trial_index_col: str | None = None,
    max_lag: int = 50,
    min_cross_pairs: int = 20,
    max_cross_pairs: int = 80,
    seed: int = 0,
) -> dict:
    """Prepare Tiffany-style drift-corrected autocorrelograms for outcome/repetition.

    Raw autocorrelograms are computed per session, averaged within mouse, and corrected
    by subtracting cross-correlograms from different sessions of the same mouse.
    """
    df = to_pandas_df(df_like)
    max_lag = int(max_lag)
    sequences = _prepare_session_binary_sequences(
        df,
        subject_col=subject_col,
        session_col=session_col,
        choice_col=choice_col,
        outcome_col=outcome_col,
        trial_index_col=trial_index_col,
    )
    if sequences.empty:
        empty = pd.DataFrame(
            columns=[
                "signal",
                "lag",
                "autocorr",
                "autocorr_sem",
                "raw_autocorr",
                "crosscorr",
                "n_subjects",
            ]
        )
        return {
            "autocorr": empty,
            "subject_autocorr": empty,
            "session_autocorr": pd.DataFrame(),
            "crosscorr": pd.DataFrame(),
            "meta": {"max_lag": max_lag},
        }

    signal_cols = {"Outcome": "outcome", "Repetition": "repetition"}
    raw_rows = []
    for (subject, session), session_df in sequences.groupby(["subject", "session"], observed=True):
        for signal, value_col in signal_cols.items():
            ac = binary_autocorrelation(session_df[value_col], max_lag=max_lag)
            if ac.empty:
                continue
            ac["subject"] = subject
            ac["session"] = session
            ac["signal"] = signal
            raw_rows.append(ac)
    session_autocorr = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()

    if session_autocorr.empty:
        return {
            "autocorr": pd.DataFrame(),
            "subject_autocorr": pd.DataFrame(),
            "session_autocorr": session_autocorr,
            "crosscorr": pd.DataFrame(),
            "meta": {"max_lag": max_lag},
        }

    subject_raw = (
        session_autocorr.groupby(["subject", "signal", "lag"], observed=True)["autocorr"]
        .mean()
        .reset_index(name="raw_autocorr")
    )

    rng = np.random.default_rng(int(seed))
    cross_rows = []
    for subject, subject_df in sequences.groupby("subject", observed=True):
        sessions = list(pd.unique(subject_df["session"]))
        if len(sessions) < 2:
            continue
        all_pairs = [(left, right) for idx, left in enumerate(sessions) for right in sessions[idx + 1 :]]
        if len(all_pairs) > int(max_cross_pairs):
            pair_idx = rng.choice(len(all_pairs), size=int(max_cross_pairs), replace=False)
            pairs = [all_pairs[int(idx)] for idx in pair_idx]
        else:
            pairs = all_pairs
        if len(pairs) < int(min_cross_pairs) and len(all_pairs) >= int(min_cross_pairs):
            pair_idx = rng.choice(len(all_pairs), size=int(min_cross_pairs), replace=False)
            pairs = [all_pairs[int(idx)] for idx in pair_idx]

        session_map = {session: sdf for session, sdf in subject_df.groupby("session", observed=True)}
        for left, right in pairs:
            left_df = session_map[left]
            right_df = session_map[right]
            for signal, value_col in signal_cols.items():
                cc = binary_crosscorrelation(
                    left_df[value_col],
                    right_df[value_col],
                    max_lag=max_lag,
                )
                if cc.empty:
                    continue
                cc["subject"] = subject
                cc["session_left"] = left
                cc["session_right"] = right
                cc["signal"] = signal
                cross_rows.append(cc)
    crosscorr = pd.concat(cross_rows, ignore_index=True) if cross_rows else pd.DataFrame()
    if crosscorr.empty:
        subject_cross = pd.DataFrame(columns=["subject", "signal", "lag", "crosscorr"])
    else:
        subject_cross = (
            crosscorr.groupby(["subject", "signal", "lag"], observed=True)["crosscorr"]
            .mean()
            .reset_index()
        )

    subject_autocorr = subject_raw.merge(
        subject_cross,
        on=["subject", "signal", "lag"],
        how="left",
    )
    subject_autocorr["crosscorr"] = subject_autocorr["crosscorr"].fillna(0.0)
    subject_autocorr["autocorr"] = subject_autocorr["raw_autocorr"] - subject_autocorr["crosscorr"]

    summary = (
        subject_autocorr.groupby(["signal", "lag"], observed=True)
        .agg(
            autocorr=("autocorr", "mean"),
            autocorr_std=("autocorr", "std"),
            raw_autocorr=("raw_autocorr", "mean"),
            raw_autocorr_std=("raw_autocorr", "std"),
            crosscorr=("crosscorr", "mean"),
            crosscorr_std=("crosscorr", "std"),
            n_subjects=("subject", "count"),
        )
        .reset_index()
    )
    _sem_denominator = np.sqrt(summary["n_subjects"].clip(lower=1))
    summary["autocorr_sem"] = summary["autocorr_std"].fillna(0.0) / _sem_denominator
    summary["raw_autocorr_sem"] = summary["raw_autocorr_std"].fillna(0.0) / _sem_denominator
    summary["crosscorr_sem"] = summary["crosscorr_std"].fillna(0.0) / _sem_denominator

    return {
        "autocorr": summary,
        "subject_autocorr": subject_autocorr,
        "session_autocorr": session_autocorr,
        "crosscorr": crosscorr,
        "sequences": sequences,
        "meta": {
            "max_lag": max_lag,
            "min_cross_pairs": int(min_cross_pairs),
            "max_cross_pairs": int(max_cross_pairs),
            "seed": int(seed),
        },
    }


def autocorrelogram_array_subjects(out_dir, suffix: str, *, k: int | None = None) -> list[str]:
    subjects = []
    pattern = f"*_{suffix}" if k is None else f"*_K{k}_{suffix}"
    for path in sorted(out_dir.glob(pattern)):
        subject = path.name.removesuffix(f"_{suffix}")
        if k is not None:
            subject = subject.removesuffix(f"_K{k}")
        subjects.append(subject)
    return subjects


def fitted_lag_weights(adapter, subject: str, target_col: str) -> dict[int, float]:
    from glmhmmt.tasks.fitted_regressors import (
        mean_feature_weights_from_fit,
        resolved_source_features,
        subject_feature_weights_from_fit,
    )

    spec_attr = {
        "choice_lag_param": "choice_lag_param_spec",
        "at_choice_param": "at_choice_param_spec",
    }.get(target_col)
    if spec_attr is None or not hasattr(adapter, spec_attr):
        return {}

    spec = getattr(adapter, spec_attr)
    try:
        weights = subject_feature_weights_from_fit(spec, subject)
    except (FileNotFoundError, ValueError):
        try:
            weights = mean_feature_weights_from_fit(spec)
        except (FileNotFoundError, ValueError):
            return {}

    out = {}
    for feature in resolved_source_features(spec):
        match = re.fullmatch(r"choice_lag_(\d+)", str(feature))
        if match and feature in weights:
            out[int(match.group(1))] = float(weights[feature])
    return out


def infer_autocorrelogram_lag_param_weights(
    y: np.ndarray,
    base_X: np.ndarray,
    sessions: np.ndarray,
    x_cols: list[str],
    target_col: str,
    *,
    max_lag: int = 20,
) -> dict[int, float]:
    """Infer an aggregate choice-history regressor as a weighted lag sum."""
    if target_col not in x_cols:
        return {}

    y = np.asarray(y, dtype=float)
    base_X = np.asarray(base_X, dtype=float)
    if y.shape[0] != base_X.shape[0] or y.size < 2:
        return {}

    target_idx = x_cols.index(target_col)
    target = base_X[:, target_idx]
    starts = autocorrelogram_session_starts(sessions)
    choice_history_values = infer_autocorrelogram_choice_history_values(y, base_X, sessions, x_cols)
    lag_count = min(int(max_lag), max(1, y.size - 1))
    lags = list(range(1, lag_count + 1))

    design = np.column_stack(
        [
            [
                autocorrelogram_history_value(
                    y,
                    trial_idx,
                    lag,
                    starts,
                    choice_history_values=choice_history_values,
                )
                for trial_idx in range(y.size)
            ]
            for lag in lags
        ]
    )
    valid = np.isfinite(target) & np.isfinite(design).all(axis=1)
    if int(valid.sum()) <= len(lags):
        return {}

    coefs, *_ = np.linalg.lstsq(design[valid], target[valid], rcond=None)
    return {
        lag: float(weight)
        for lag, weight in zip(lags, coefs, strict=False)
        if np.isfinite(weight) and abs(float(weight)) > 1e-10
    }


def autocorrelogram_class_count(arrays: dict) -> int:
    p_pred = np.asarray(arrays.get("p_pred", []), dtype=float)
    if p_pred.ndim == 2 and p_pred.shape[1] > 0:
        return int(p_pred.shape[1])
    y = np.asarray(arrays.get("y", []), dtype=float)
    finite = y[np.isfinite(y)]
    return int(np.nanmax(finite) + 1) if finite.size else 2


def normalize_probability_vector(probs: np.ndarray) -> np.ndarray:
    out = np.asarray(probs, dtype=float).copy()
    out = np.clip(out, 1e-12, np.inf)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0:
        return np.full_like(out, 1.0 / out.size, dtype=float)
    return out / total


def _softmax_last_axis(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.nanmax(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


def _input_driven_transition_weights_with_baseline(weights: np.ndarray, *, K: int) -> np.ndarray:
    """Return target-wise transition weights with the implicit baseline target restored."""
    weights = np.asarray(weights, dtype=float)
    baseline_target_idx = K - 1
    if weights.ndim == 3 and weights.shape[:2] != (K, K) and weights.shape[1] == K:
        weights = weights.mean(axis=0)
    if weights.ndim != 2:
        raise ValueError(f"target-wise transition_weights must be 2D or legacy 3D, got shape {weights.shape}.")
    if weights.shape[0] == K:
        return weights
    if weights.shape[0] != K - 1:
        raise ValueError(f"transition_weights first dimension must be K or K-1; got {weights.shape} for K={K}.")

    zero = np.zeros((1, weights.shape[1]), dtype=float)
    return np.concatenate(
        [
            weights[:baseline_target_idx],
            zero,
            weights[baseline_target_idx:],
        ],
        axis=0,
    )


def _expand_self_baseline_transition_array(
    values: np.ndarray,
    *,
    K: int,
    trailing_shape: tuple[int, ...] = (),
    name: str,
) -> np.ndarray:
    """Restore zero-valued self-transition references in GLM-HMM-T arrays."""
    values = np.asarray(values, dtype=float)
    full_shape = (K, K, *trailing_shape)
    compact_shape = (K, K - 1, *trailing_shape)
    if values.shape == full_shape:
        return values
    if values.shape != compact_shape:
        raise ValueError(
            f"{name} must have full shape {full_shape} or self-baseline shape "
            f"{compact_shape}; got {values.shape}."
        )

    full = np.zeros(full_shape, dtype=values.dtype)
    for source_idx in range(K):
        nonself_targets = [target_idx for target_idx in range(K) if target_idx != source_idx]
        full[source_idx, nonself_targets, ...] = values[source_idx]
    return full


def autocorrelogram_transition_matrices(arrays: dict, *, K: int, T: int) -> np.ndarray:
    """Build A[t, i, j] = p(z[t+1]=j | z[t]=i) for closed-loop simulations."""
    if T <= 1:
        return np.empty((0, K, K), dtype=float)

    transition_bias = arrays.get("transition_bias")
    transition_weights = arrays.get("transition_weights")
    U = arrays.get("U")
    if transition_bias is not None and transition_weights is not None and U is not None:
        bias = np.asarray(transition_bias, dtype=float)
        weights = np.asarray(transition_weights, dtype=float)
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[0] != T:
            raise ValueError(f"U must have shape (T, D) with T={T}, got {U.shape}.")

        bias = _expand_self_baseline_transition_array(
            bias,
            K=K,
            name="transition_bias",
        )

        # GLM-HMM-T transition inputs are destination/current-trial aligned:
        # U[t + 1] drives the transition from trial t to t + 1.
        if weights.ndim == 3:
            weights = _expand_self_baseline_transition_array(
                weights,
                K=K,
                trailing_shape=(U.shape[1],),
                name="transition_weights",
            )
            if U.shape[1] != weights.shape[2]:
                raise ValueError(f"U width ({U.shape[1]}) does not match transition_weights width ({weights.shape[2]}).")
            input_logits = np.einsum("td,ijd->tij", U[1:], weights)
            logits = bias[None, :, :] + input_logits
        else:
            weights = _input_driven_transition_weights_with_baseline(weights, K=K)
            if U.shape[1] != weights.shape[1]:
                raise ValueError(f"U width ({U.shape[1]}) does not match transition_weights width ({weights.shape[1]}).")
            input_logits = U[1:] @ weights.T
            logits = bias[None, :, :] + input_logits[:, None, :]
        return _softmax_last_axis(logits)

    transition_matrix = arrays.get("transition_matrix")
    if transition_matrix is not None:
        matrix = np.asarray(transition_matrix, dtype=float)
        if matrix.ndim == 2:
            matrix = np.broadcast_to(matrix[None, :, :], (max(T - 1, 0), K, K)).copy()
        elif matrix.ndim != 3:
            raise ValueError(f"transition_matrix must be 2D or 3D, got {matrix.shape}.")
        if matrix.shape[-2:] != (K, K):
            raise ValueError(f"transition_matrix trailing dimensions must be {(K, K)}, got {matrix.shape}.")
        if matrix.shape[0] != max(T - 1, 0):
            raise ValueError(f"transition_matrix has {matrix.shape[0]} rows, expected {max(T - 1, 0)}.")
        row_sums = np.sum(matrix, axis=-1, keepdims=True)
        return np.where(row_sums > 0, matrix / row_sums, np.full_like(matrix, 1.0 / K))

    return np.broadcast_to(np.eye(K, dtype=float)[None, :, :], (max(T - 1, 0), K, K)).copy()


def apply_autocorrelogram_lapse(
    probs: np.ndarray,
    *,
    previous_choice: int | None,
    lapse_mode: str,
    lapse_rates: np.ndarray,
) -> np.ndarray:
    out = normalize_probability_vector(probs)
    if previous_choice is None:
        return out

    num_classes = out.size
    lapse_rates = np.asarray(lapse_rates, dtype=float).reshape(-1)
    if lapse_mode == "class" and lapse_rates.size:
        class_rates = lapse_rates[:num_classes]
        total_mass = float(np.sum(class_rates))
        out = class_rates + (1.0 - total_mass) * out
    elif lapse_mode == "history":
        repeat_rate = float(lapse_rates[0]) if lapse_rates.size > 0 else 0.0
        alternate_rate = float(lapse_rates[1]) if lapse_rates.size > 1 else 0.0
        repeat_target = np.zeros(num_classes, dtype=float)
        repeat_target[int(previous_choice)] = 1.0
        alternate_target = np.full(num_classes, 1.0 / max(1, num_classes - 1), dtype=float)
        alternate_target[int(previous_choice)] = 0.0
        out = (1.0 - repeat_rate - alternate_rate) * out
        out += repeat_rate * repeat_target + alternate_rate * alternate_target
    elif lapse_mode == "history_conditioned":
        repeat_rates = lapse_rates[:num_classes]
        alternate_rates = lapse_rates[num_classes : 2 * num_classes]
        repeat_rate = float(repeat_rates[int(previous_choice)]) if repeat_rates.size > previous_choice else 0.0
        alternate_rate = (
            float(alternate_rates[int(previous_choice)])
            if alternate_rates.size > previous_choice
            else 0.0
        )
        repeat_target = np.zeros(num_classes, dtype=float)
        repeat_target[int(previous_choice)] = 1.0
        alternate_target = np.full(num_classes, 1.0 / max(1, num_classes - 1), dtype=float)
        alternate_target[int(previous_choice)] = 0.0
        out = (1.0 - repeat_rate - alternate_rate) * out
        out += repeat_rate * repeat_target + alternate_rate * alternate_target
    return normalize_probability_vector(out)


def infer_autocorrelogram_correct_class(
    subject_df: pd.DataFrame,
    adapter,
    arrays: dict | None = None,
) -> np.ndarray:
    behavioral_cols = dict(getattr(adapter, "behavioral_cols", {}) or {})
    stimulus_col = behavioral_cols.get("stimulus")
    performance_col = behavioral_cols.get("performance")

    if (
        arrays is not None
        and stimulus_col in subject_df.columns
        and performance_col in subject_df.columns
    ):
        y = np.asarray(arrays.get("y", []), dtype=float)
        if y.shape[0] == len(subject_df):
            performance = pd.to_numeric(subject_df[performance_col], errors="coerce")
            num_classes = autocorrelogram_class_count(arrays)
            if num_classes == 2:
                correct_class = np.where(performance.to_numpy(dtype=float) > 0, y, 1.0 - y)
                correct_class[~np.isfinite(y) | ~np.isfinite(performance.to_numpy(dtype=float))] = np.nan
                return correct_class
            tmp = pd.DataFrame(
                {
                    "stimulus": subject_df[stimulus_col].to_numpy(),
                    "performance": performance.to_numpy(dtype=float),
                    "class": y,
                }
            )
            mapping = {}
            for stim_value, group in tmp[tmp["performance"] > 0].groupby("stimulus", observed=True):
                finite_classes = group["class"][np.isfinite(group["class"])]
                if finite_classes.empty:
                    continue
                mode = finite_classes.astype(int).mode(dropna=True)
                if not mode.empty:
                    mapping[stim_value] = int(mode.iloc[0])
            if mapping:
                return np.asarray(
                    [mapping.get(value, np.nan) for value in subject_df[stimulus_col].to_numpy()],
                    dtype=float,
                )

    if stimulus_col not in subject_df.columns:
        return np.full(len(subject_df), np.nan, dtype=float)

    stimulus = pd.to_numeric(subject_df[stimulus_col], errors="coerce").to_numpy(dtype=float)
    if arrays is not None:
        num_classes = autocorrelogram_class_count(arrays)
        class_values = np.arange(num_classes, dtype=float)
        if np.isin(stimulus[np.isfinite(stimulus)], class_values).all():
            return stimulus
    return np.where(
        np.isin(stimulus, [0.0, 1.0]),
        stimulus,
        np.where(np.isfinite(stimulus), (stimulus > 0.0).astype(float), np.nan),
    )


def autocorrelogram_session_starts(sessions: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(sessions), dtype=int)
    start = 0
    for idx in range(len(sessions)):
        if idx > 0 and sessions[idx] != sessions[idx - 1]:
            start = idx
        starts[idx] = start
    return starts


def autocorrelogram_history_value(
    choices: np.ndarray,
    trial_idx: int,
    lag: int,
    starts: np.ndarray,
    choice_history_values: dict[int, float] | None = None,
) -> float:
    source_idx = trial_idx - int(lag)
    if source_idx < starts[trial_idx]:
        return 0.0
    choice = choices[source_idx]
    if not np.isfinite(choice):
        return 0.0
    choice_idx = int(choice)
    if choice_history_values is not None and choice_idx in choice_history_values:
        return float(choice_history_values[choice_idx])
    return float(2.0 * choice_idx - 1.0)


def infer_autocorrelogram_choice_history_values(
    y: np.ndarray,
    base_X: np.ndarray,
    sessions: np.ndarray,
    x_cols: list[str],
) -> dict[int, float]:
    """Infer class-to-history coding from saved direct choice-lag columns."""
    y = np.asarray(y, dtype=float)
    base_X = np.asarray(base_X, dtype=float)
    if y.shape[0] != base_X.shape[0]:
        return {}

    values_by_class: dict[int, list[float]] = {}
    for col_idx, col in enumerate(x_cols):
        match = re.fullmatch(r"choice_lag_(\d+)", str(col))
        if not match:
            continue
        lag = int(match.group(1))
        if lag <= 0 or lag >= len(y):
            continue
        same_session = sessions[lag:] == sessions[:-lag]
        source_y = y[:-lag][same_session]
        lag_values = base_X[lag:, col_idx][same_session]
        finite = np.isfinite(source_y) & np.isfinite(lag_values)
        for class_value, lag_value in zip(source_y[finite], lag_values[finite], strict=False):
            values_by_class.setdefault(int(class_value), []).append(float(lag_value))

    out = {}
    for class_value, values in values_by_class.items():
        if values:
            out[int(class_value)] = float(np.nanmedian(values))
    if out:
        return out

    aggregate_history_cols = {
        "choice_lag_param",
        "choice_lag_param_2",
        "at_choice_param",
        "prev_choice",
    }
    for col_idx, col in enumerate(x_cols):
        if str(col) not in aggregate_history_cols:
            continue
        if len(y) < 2:
            continue
        same_session = sessions[1:] == sessions[:-1]
        source_y = y[:-1][same_session]
        lag_values = base_X[1:, col_idx][same_session]
        finite = np.isfinite(source_y) & np.isfinite(lag_values)
        if not finite.any():
            continue

        medians_by_class = {
            int(class_value): float(np.nanmedian(lag_values[finite & (source_y == class_value)]))
            for class_value in np.unique(source_y[finite]).astype(int)
            if np.any(finite & (source_y == class_value))
        }
        if len(medians_by_class) < 2:
            continue
        medians = np.asarray(list(medians_by_class.values()), dtype=float)
        if not np.isfinite(medians).all() or float(np.nanmax(medians) - np.nanmin(medians)) <= 1e-8:
            continue

        center = float(np.nanmean(medians))
        scale = float(np.nanmax(np.abs(medians - center)))
        if scale <= 0:
            continue
        return {
            class_value: float((median - center) / scale)
            for class_value, median in medians_by_class.items()
        }
    return out


def infer_autocorrelogram_choice_indicator_classes(
    y: np.ndarray,
    base_X: np.ndarray,
    sessions: np.ndarray,
    x_cols: list[str],
) -> dict[str, int]:
    """Infer side-coded one-hot choice-lag columns, e.g. choice_lag_01L."""
    y = np.asarray(y, dtype=float)
    base_X = np.asarray(base_X, dtype=float)
    if y.shape[0] != base_X.shape[0]:
        return {}

    out: dict[str, int] = {}
    for col_idx, col in enumerate(x_cols):
        match = re.fullmatch(r"choice_lag_(\d+)([A-Za-z]+)", str(col))
        if not match:
            continue
        lag = int(match.group(1))
        if lag <= 0 or lag >= len(y):
            continue
        same_session = sessions[lag:] == sessions[:-lag]
        source_y = y[:-lag][same_session]
        lag_values = base_X[lag:, col_idx][same_session]
        finite = np.isfinite(source_y) & np.isfinite(lag_values)
        if not finite.any():
            continue

        class_scores = {}
        for class_value in np.unique(source_y[finite]).astype(int):
            class_mask = finite & (source_y == class_value)
            if class_mask.any():
                class_scores[int(class_value)] = float(np.nanmean(lag_values[class_mask]))
        if not class_scores:
            continue
        best_class, best_score = max(class_scores.items(), key=lambda item: item[1])
        if best_score > 0.5:
            out[str(col)] = int(best_class)
    return out


def closed_loop_autocorrelogram_x(
    base_x: np.ndarray,
    *,
    trial_idx: int,
    choices: np.ndarray,
    starts: np.ndarray,
    x_cols: list[str],
    lag_param_weights: dict[str, dict[int, float]],
    choice_history_values: dict[int, float] | None = None,
    choice_indicator_classes: dict[str, int] | None = None,
) -> np.ndarray:
    x = np.asarray(base_x, dtype=float).copy()
    for col_idx, col in enumerate(x_cols):
        match = re.fullmatch(r"choice_lag_(\d+)", str(col))
        if match:
            x[col_idx] = autocorrelogram_history_value(
                choices,
                trial_idx,
                int(match.group(1)),
                starts,
                choice_history_values=choice_history_values,
            )
        else:
            side_match = re.fullmatch(r"choice_lag_(\d+)([A-Za-z]+)", str(col))
            if side_match:
                source_idx = trial_idx - int(side_match.group(1))
                if source_idx < starts[trial_idx] or not np.isfinite(choices[source_idx]):
                    x[col_idx] = 0.0
                else:
                    class_idx = (choice_indicator_classes or {}).get(str(col))
                    x[col_idx] = float(class_idx is not None and int(choices[source_idx]) == class_idx)
                continue
        if col in lag_param_weights:
            x[col_idx] = sum(
                weight * autocorrelogram_history_value(
                    choices,
                    trial_idx,
                    lag,
                    starts,
                    choice_history_values=choice_history_values,
                )
                for lag, weight in lag_param_weights[col].items()
            )
        elif col == "prev_choice":
            x[col_idx] = autocorrelogram_history_value(
                choices,
                trial_idx,
                1,
                starts,
                choice_history_values=choice_history_values,
            )
    return x


def simulate_subject_closed_loop_autocorrelogram(
    subject_df: pd.DataFrame,
    arrays: dict,
    *,
    adapter,
    subject: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    from glmhmmt.glm import glm_probs_from_weights

    base_X = np.asarray(arrays["X"], dtype=float)
    x_cols = [str(v) for v in np.asarray(arrays.get("X_cols", []), dtype=object).tolist()]
    if base_X.ndim != 2 or base_X.shape[0] != len(subject_df):
        raise ValueError(f"{subject}: X rows ({base_X.shape}) do not match data rows ({len(subject_df)}).")

    weights = np.asarray(arrays["emission_weights"], dtype=float)
    if weights.ndim == 2:
        weights = weights[None, :, :]
    if weights.ndim != 3:
        raise ValueError(f"{subject}: expected emission_weights with 3 dimensions, got {weights.shape}.")

    K = int(weights.shape[0])
    num_classes = autocorrelogram_class_count(arrays)
    baseline_class_idx = int(np.asarray(arrays.get("baseline_class_idx", 0)).reshape(()))
    lapse_mode = str(np.asarray(arrays.get("lapse_mode", "none")).reshape(()))
    lapse_rates = np.asarray(arrays.get("lapse_rates", []), dtype=float)
    initial_probs = normalize_probability_vector(np.asarray(arrays.get("initial_probs", np.ones(K) / K), dtype=float))
    transition_matrices = autocorrelogram_transition_matrices(arrays, K=K, T=len(subject_df))

    sessions = subject_df[adapter.behavioral_cols["session"]].to_numpy()
    starts = autocorrelogram_session_starts(sessions)
    choices = np.full(len(subject_df), np.nan, dtype=float)
    states = np.zeros(len(subject_df), dtype=int)
    choice_history_values = infer_autocorrelogram_choice_history_values(
        np.asarray(arrays.get("y", []), dtype=float),
        base_X,
        sessions,
        x_cols,
    )
    choice_indicator_classes = infer_autocorrelogram_choice_indicator_classes(
        np.asarray(arrays.get("y", []), dtype=float),
        base_X,
        sessions,
        x_cols,
    )
    lag_param_weights = {}
    for col in ("choice_lag_param", "at_choice_param"):
        if col not in x_cols:
            continue
        weights_for_col = fitted_lag_weights(adapter, subject, col)
        if not weights_for_col:
            weights_for_col = infer_autocorrelogram_lag_param_weights(
                np.asarray(arrays.get("y", []), dtype=float),
                base_X,
                sessions,
                x_cols,
                col,
            )
        if weights_for_col:
            lag_param_weights[col] = weights_for_col

    predicted_state = initial_probs
    for trial_idx in range(len(subject_df)):
        if trial_idx == starts[trial_idx]:
            predicted_state = initial_probs

        x_trial = closed_loop_autocorrelogram_x(
            base_X[trial_idx],
            trial_idx=trial_idx,
            choices=choices,
            starts=starts,
            x_cols=x_cols,
            lag_param_weights=lag_param_weights,
            choice_history_values=choice_history_values,
            choice_indicator_classes=choice_indicator_classes,
        )
        previous_choice = int(choices[trial_idx - 1]) if trial_idx > starts[trial_idx] else None

        state_conditional_probs = np.vstack(
            [
                apply_autocorrelogram_lapse(
                    glm_probs_from_weights(
                        x_trial[None, :],
                        weights[state_idx],
                        baseline_class_idx=baseline_class_idx,
                        num_classes=num_classes,
                    )[0],
                    previous_choice=previous_choice,
                    lapse_mode=lapse_mode,
                    lapse_rates=lapse_rates,
                )
                for state_idx in range(K)
            ]
        )
        probs = normalize_probability_vector(predicted_state @ state_conditional_probs)
        choices[trial_idx] = int(rng.choice(num_classes, p=probs))

        likelihood = state_conditional_probs[:, int(choices[trial_idx])]
        filtered_state = normalize_probability_vector(predicted_state * likelihood)
        states[trial_idx] = int(np.argmax(filtered_state))

        next_idx = trial_idx + 1
        if next_idx < len(subject_df) and starts[next_idx] == starts[trial_idx]:
            predicted_state = normalize_probability_vector(filtered_state @ transition_matrices[trial_idx])

    correct_class = infer_autocorrelogram_correct_class(subject_df, adapter, arrays)
    performance = (choices == correct_class).astype(float)
    performance[~np.isfinite(correct_class)] = np.nan
    return choices, performance


def closed_loop_arrays_store_from_views(views: dict) -> dict:
    """Convert SubjectFitView objects to the arrays dict used by closed-loop simulation."""
    arrays_store = {}
    for subject, view in views.items():
        arrays = {
            "X": np.asarray(view.X),
            "X_cols": np.asarray(view.feat_names, dtype=object),
            "y": np.asarray(view.y),
            "emission_weights": np.asarray(view.emission_weights),
            "baseline_class_idx": np.asarray(view.baseline_class_idx),
            "lapse_mode": np.asarray(view.lapse_mode),
        }
        if view.p_pred is not None:
            arrays["p_pred"] = np.asarray(view.p_pred)
        if view.lapse_rates is not None:
            arrays["lapse_rates"] = np.asarray(view.lapse_rates)
        if view.initial_probs is not None:
            arrays["initial_probs"] = np.asarray(view.initial_probs)
        if view.transition_matrix is not None:
            arrays["transition_matrix"] = np.asarray(view.transition_matrix)
        if getattr(view, "transition_bias", None) is not None:
            arrays["transition_bias"] = np.asarray(view.transition_bias)
        if getattr(view, "transition_weights", None) is not None:
            arrays["transition_weights"] = np.asarray(view.transition_weights)
        if getattr(view, "U", None) is not None:
            arrays["U"] = np.asarray(view.U)
        arrays_store[str(subject)] = arrays
    return arrays_store


def prepare_closed_loop_model_autocorrelograms(
    df_all: pl.DataFrame,
    arrays_store: dict | None = None,
    *,
    views: dict | None = None,
    adapter,
    n_simulations: int,
    max_lag: int,
    seed: int,
    min_cross_pairs: int = 20,
    max_cross_pairs: int = 80,
    progress_label: str = "closed-loop simulations",
) -> dict:
    from glmhmmt.notebook_support.analysis_common import select_subject_behavior_df
    from tqdm.auto import tqdm

    if arrays_store is None:
        if views is None:
            raise ValueError("Provide either arrays_store or views.")
        arrays_store = closed_loop_arrays_store_from_views(views)

    rng = np.random.default_rng(seed)
    frames = []
    n_simulations = int(n_simulations)
    total_jobs = len(arrays_store) * n_simulations
    with tqdm(total=total_jobs, desc=progress_label, unit="sim") as progress:
        for subject, arrays in arrays_store.items():
            subject_df_pl = select_subject_behavior_df(
                df_all,
                subject=subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
                min_session_length=1,
            )
            if subject_df_pl.height == 0:
                progress.update(n_simulations)
                continue
            subject_df = subject_df_pl.to_pandas()
            sessions = subject_df[adapter.behavioral_cols["session"]].to_numpy()
            trial_index = subject_df[adapter.behavioral_cols["trial"]].to_numpy()
            for sim_idx in range(n_simulations):
                choices, performance = simulate_subject_closed_loop_autocorrelogram(
                    subject_df,
                    arrays,
                    adapter=adapter,
                    subject=str(subject),
                    rng=rng,
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "subject": f"{subject}__closed_loop_{sim_idx:03d}",
                            "session": sessions,
                            "trial_index": trial_index,
                            "response": choices,
                            "performance": performance,
                        }
                    )
                )
                progress.update()

    simulated_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    prepared = prepare_corrected_behavior_autocorrelograms(
        simulated_df,
        subject_col="subject",
        session_col="session",
        choice_col="response",
        outcome_col="performance",
        trial_index_col="trial_index",
        max_lag=max_lag,
        min_cross_pairs=min_cross_pairs,
        max_cross_pairs=max_cross_pairs,
        seed=seed,
    )
    prepared["simulated_df"] = simulated_df
    prepared["meta"] = {
        **prepared.get("meta", {}),
        "n_simulations": n_simulations,
        "simulation": "closed_loop_model",
    }
    return prepared


def animal_chunk_histogram_weights(
    chunk_lengths: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    stat: str,
) -> pd.DataFrame:
    """Average per-subject chunk-length counts/frequencies for plotting."""
    group_cols = list(group_cols)
    counts = (
        chunk_lengths
        .groupby(["subject", *group_cols, "chunk_length"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["frequency"] = (
        counts["count"]
        / counts.groupby(["subject", *group_cols], observed=True)["count"].transform("sum")
    )
    counts["n_subjects"] = counts.groupby(group_cols, observed=True)["subject"].transform("nunique")
    counts["hist_weight"] = (
        counts["frequency"] if stat == "probability" else counts["count"]
    ) / counts["n_subjects"]
    return counts


def transition_chunk_lengths_for_sequence(
    plot_df: pl.DataFrame,
    *,
    task_name: str,
    task_label: str,
    sequence_col: str,
    sequence_label: str,
) -> tuple[pd.DataFrame, dict]:
    """Return transition run lengths and subject-balanced repeat probability."""
    trials = (
        plot_df
        .select(["subject", "session", "trial_idx", sequence_col])
        .to_pandas()
        .dropna(subset=[sequence_col])
        .sort_values(["subject", "session", "trial_idx"])
    )
    trials["previous_value"] = (
        trials.groupby(["subject", "session"], observed=True)[sequence_col].shift(1)
    )
    trials = trials.dropna(subset=["previous_value"])
    trials["transition"] = (
        trials[sequence_col]
        .eq(trials["previous_value"])
        .map({True: "repeating", False: "alternating"})
    )
    trials["transition_chunk"] = (
        trials
        .groupby(["subject", "session"], observed=True)["transition"]
        .transform(lambda transition: transition.ne(transition.shift()).cumsum())
    )

    chunks = (
        trials
        .groupby(["subject", "session", "transition", "transition_chunk"], observed=True)
        .size()
        .rename("chunk_length")
        .reset_index()
    )
    chunks["task"] = task_name
    chunks["task_label"] = task_label
    chunks["sequence"] = sequence_label

    subject_repeat_probability = (
        trials.assign(is_repeat=trials["transition"] == "repeating")
        .groupby("subject", observed=True)["is_repeat"]
        .mean()
    )
    repeat_probability = {
        "task": task_name,
        "task_label": task_label,
        "sequence": sequence_label,
        "p_repeat": float(subject_repeat_probability.mean()),
        "n_subjects": int(subject_repeat_probability.shape[0]),
        "aggregation": "mean_subject_after_pooling_sessions",
    }
    return chunks, repeat_probability


def geometric_transition_chunk_probability(
    chunk_lengths,
    repeat_probability: float,
    transition: str,
):
    """Geometric chunk-length probability for iid repeat/alternate transitions."""
    continue_probability = (
        repeat_probability
        if transition == "repeating"
        else 1.0 - repeat_probability
    )
    return (1.0 - continue_probability) * (continue_probability ** (chunk_lengths - 1))


def fixed_accuracy_repeat_probabilities(
    accuracy: float,
    *,
    n_classes: int = 2,
) -> tuple[float, float]:
    """Return P(choice repeats) for repeated and changed stimuli at fixed accuracy."""

    p = float(np.clip(accuracy, 0.0, 1.0))
    n_classes = max(2, int(n_classes))
    n_alternatives = n_classes - 1
    same_stimulus = (p**2) + (((1.0 - p) ** 2) / n_alternatives)
    changed_stimulus = (
        (2.0 * p * (1.0 - p) / n_alternatives)
        + (max(0, n_classes - 2) * (((1.0 - p) / n_alternatives) ** 2))
    )
    return float(same_stimulus), float(changed_stimulus)


def _response_class_count(df: pd.DataFrame, response_col: str = "response") -> int:
    if response_col not in df.columns:
        return 2
    values = pd.Series(df[response_col]).dropna().unique()
    if len(values) < 2 or len(values) > 10:
        return 2
    return int(len(values))


def add_fixed_accuracy_repetition_band(
    session_data: pd.DataFrame,
    *,
    accuracy: float | None = None,
    n_classes: int | None = None,
    z: float = 1.96,
    prefix: str = "fixed_accuracy",
) -> pd.DataFrame:
    """Add the fixed-accuracy stimulus-following repeat band."""

    out = session_data.copy()
    if out.empty:
        return out

    if accuracy is None:
        accuracy = float(pd.to_numeric(out.get("correct"), errors="coerce").mean())
    if not np.isfinite(accuracy):
        return out

    same_prob, changed_prob = fixed_accuracy_repeat_probabilities(
        accuracy,
        n_classes=n_classes or _response_class_count(out),
    )
    n = pd.to_numeric(out["repeat_window_n"], errors="coerce").to_numpy(dtype=float)
    stimulus_count = pd.to_numeric(
        out["stimulus_repeat_window_count"], errors="coerce"
    ).to_numpy(dtype=float)
    stimulus_count = np.clip(stimulus_count, 0.0, n)

    expected = np.divide(
        stimulus_count,
        n,
        out=np.full_like(n, np.nan),
        where=n > 0,
    )
    preserve_probability = same_prob
    sem = np.divide(
        np.sqrt(preserve_probability * (1.0 - preserve_probability) * n),
        n,
        out=np.full_like(n, np.nan),
        where=n > 0,
    )
    lower = np.clip(expected - (float(z) * sem), 0.0, 1.0)
    upper = np.clip(expected + (float(z) * sem), 0.0, 1.0)

    observed = pd.to_numeric(
        out["response_repeat_window_fraction"], errors="coerce"
    ).to_numpy(dtype=float)
    out[f"{prefix}_repeat_fraction"] = expected
    out[f"{prefix}_repeat_low"] = lower
    out[f"{prefix}_repeat_high"] = upper
    out[f"{prefix}_choice_above"] = observed > upper
    out.attrs.update(
        {
            f"{prefix}_accuracy": float(accuracy),
            f"{prefix}_same_stimulus_repeat_probability": same_prob,
            f"{prefix}_changed_stimulus_repeat_probability": changed_prob,
        }
    )
    return out


def add_stationary_accuracy_band(
    session_data: pd.DataFrame,
    *,
    z: float = 1.96,
    prefix: str = "stationary_accuracy",
) -> pd.DataFrame:
    """Add rolling stimulus-conditioned accuracy and its analytical CI.

    Each trial receives the empirical ``P(correct)`` for its stimulus
    condition.  When available, ``level`` is included with stimulus side so
    different stimulus strengths retain different stationary accuracies.  The
    trial probabilities are then averaged over the same window as observed
    running accuracy.
    """

    out = session_data.copy()
    if out.empty or not {"stimulus", "correct"}.issubset(out.columns):
        return out

    correct = pd.to_numeric(out["correct"], errors="coerce")
    if not correct.notna().any():
        return out

    condition_cols = ["stimulus"]
    if "level" in out.columns and out["level"].notna().any():
        condition_cols.append("level")

    condition_data = out[condition_cols].copy()
    condition_data["_correct"] = correct
    condition_summary = (
        condition_data
        .groupby(condition_cols, observed=True, dropna=False)["_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "_stationary_pc", "count": "_stationary_n"})
        .reset_index()
    )
    condition_summary["_stationary_sem"] = np.sqrt(
        condition_summary["_stationary_pc"]
        * (1.0 - condition_summary["_stationary_pc"])
        / condition_summary["_stationary_n"]
    )

    trial_summary = (
        out[condition_cols]
        .merge(condition_summary, on=condition_cols, how="left", sort=False)
    )
    trial_probability = trial_summary["_stationary_pc"].astype(float)
    if "accuracy_window_n" in out.columns:
        window = int(pd.to_numeric(out["accuracy_window_n"], errors="coerce").max())
    elif "repeat_window_n" in out.columns:
        window = int(pd.to_numeric(out["repeat_window_n"], errors="coerce").max())
    else:
        window = len(out)
    window = max(1, window)

    rolling_n = trial_probability.rolling(window, min_periods=1).count()
    expected = trial_probability.rolling(window, min_periods=1).mean()
    rolling_variance = (
        trial_probability.mul(1.0 - trial_probability)
        .rolling(window, min_periods=1)
        .sum()
    )
    sem = np.sqrt(rolling_variance) / rolling_n

    out[f"{prefix}_trial_fraction"] = trial_probability.to_numpy(dtype=float)
    out[f"{prefix}_fraction"] = expected.to_numpy(dtype=float)
    out[f"{prefix}_low"] = np.clip(
        (expected - (float(z) * sem)).to_numpy(dtype=float), 0.0, 1.0
    )
    out[f"{prefix}_high"] = np.clip(
        (expected + (float(z) * sem)).to_numpy(dtype=float), 0.0, 1.0
    )
    out[f"{prefix}_n"] = rolling_n.to_numpy(dtype=float)
    out[f"{prefix}_condition_n"] = trial_summary["_stationary_n"].to_numpy(dtype=float)
    return out


def repeat_probability_for_transition_chunks(
    repeat_probabilities: pd.DataFrame,
    *,
    task_label: str,
    sequence: str,
) -> float | None:
    matches = repeat_probabilities.loc[
        (repeat_probabilities["task_label"] == task_label)
        & (repeat_probabilities["sequence"] == sequence),
        "p_repeat",
    ]
    if matches.empty:
        return None
    return float(matches.iloc[0])


def _transition_chunks_from_simulation(
    simulated_df,
    *,
    task_name: str,
    task_label: str,
) -> pd.DataFrame:
    df = to_pandas_df(simulated_df)
    if df.empty:
        return pd.DataFrame()
    if "trial_idx" not in df.columns and "trial_index" in df.columns:
        df = df.rename(columns={"trial_index": "trial_idx"})
    if not {"subject", "session", "trial_idx", "response"}.issubset(df.columns):
        return pd.DataFrame()
    chunks, _ = transition_chunk_lengths_for_sequence(
        pl.from_pandas(df[["subject", "session", "trial_idx", "response"]]),
        task_name=task_name,
        task_label=task_label,
        sequence_col="response",
        sequence_label="Choices",
    )
    return chunks


def _weighted_chunk_histogram_y(
    chunks: pd.DataFrame,
    *,
    stat: str,
    plot_x: np.ndarray,
) -> tuple[np.ndarray, float]:
    if chunks.empty:
        return np.zeros_like(plot_x, dtype=float), 0.0
    hist_data = animal_chunk_histogram_weights(
        chunks,
        group_cols=["transition"],
        stat=stat,
    )
    hist_data = hist_data[hist_data["transition"] == "repeating"]
    y = (
        hist_data
        .groupby("chunk_length", observed=True)["hist_weight"]
        .sum()
        .reindex(plot_x, fill_value=0)
        .sort_index()
        .to_numpy(dtype=float)
    )
    return y, float(hist_data["hist_weight"].sum())


def outcome_streak_lengths_for_sequence(
    plot_df,
    *,
    task_name: str,
    task_label: str,
    outcome_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return correct/incorrect run lengths and subject-balanced accuracy."""

    df = to_pandas_df(plot_df)
    if df.empty:
        return pd.DataFrame(), {
            "task": task_name,
            "task_label": task_label,
            "p_correct": np.nan,
            "n_subjects": 0,
            "aggregation": "mean_subject_after_pooling_sessions",
        }

    subject_col = pick_existing_column(df, ["subject", "Subject"])
    session_col = pick_existing_column(df, ["session", "Session"])
    trial_col = pick_existing_column(df, ["trial_idx", "trial_index", "Trial", "trial"])
    outcome_col = pick_existing_column(
        df,
        [outcome_col, "performance", "correct_bool", "correct", "outcome"],
    )
    if any(col is None for col in [subject_col, session_col, trial_col, outcome_col]):
        return pd.DataFrame(), {
            "task": task_name,
            "task_label": task_label,
            "p_correct": np.nan,
            "n_subjects": 0,
            "aggregation": "mean_subject_after_pooling_sessions",
        }

    trials = df[[subject_col, session_col, trial_col, outcome_col]].copy()
    trials.columns = ["subject", "session", "trial_idx", "outcome_value"]
    trials["outcome_value"] = _binary_indicator_series(trials["outcome_value"])
    trials = (
        trials
        .dropna(subset=["subject", "session", "trial_idx", "outcome_value"])
        .sort_values(["subject", "session", "trial_idx"])
    )
    trials = trials[trials["outcome_value"].isin([0.0, 1.0])].copy()
    if trials.empty:
        return pd.DataFrame(), {
            "task": task_name,
            "task_label": task_label,
            "p_correct": np.nan,
            "n_subjects": 0,
            "aggregation": "mean_subject_after_pooling_sessions",
        }

    trials["outcome"] = trials["outcome_value"].map({1.0: "Correct", 0.0: "Incorrect"})
    trials["outcome_streak"] = (
        trials
        .groupby(["subject", "session"], observed=True)["outcome"]
        .transform(lambda outcome: outcome.ne(outcome.shift()).cumsum())
    )
    chunks = (
        trials
        .groupby(["subject", "session", "outcome", "outcome_streak"], observed=True)
        .size()
        .rename("chunk_length")
        .reset_index()
    )
    chunks["task"] = task_name
    chunks["task_label"] = task_label

    subject_accuracy = (
        trials
        .assign(is_correct=trials["outcome"] == "Correct")
        .groupby("subject", observed=True)["is_correct"]
        .mean()
    )
    correct_probability = {
        "task": task_name,
        "task_label": task_label,
        "p_correct": float(subject_accuracy.mean()),
        "n_subjects": int(subject_accuracy.shape[0]),
        "aggregation": "mean_subject_after_pooling_sessions",
    }
    return chunks, correct_probability


def _outcome_streak_chunks_from_simulation(
    simulated_df,
    *,
    task_name: str,
    task_label: str,
) -> pd.DataFrame:
    df = to_pandas_df(simulated_df)
    if df.empty:
        return pd.DataFrame()
    chunks, _ = outcome_streak_lengths_for_sequence(
        df,
        task_name=task_name,
        task_label=task_label,
        outcome_col="performance",
    )
    return chunks


def _weighted_outcome_streak_histogram_y(
    chunks: pd.DataFrame,
    *,
    outcome: str,
    stat: str,
    plot_x: np.ndarray,
) -> tuple[np.ndarray, float]:
    if chunks.empty:
        return np.zeros_like(plot_x, dtype=float), 0.0
    hist_data = animal_chunk_histogram_weights(
        chunks,
        group_cols=["outcome"],
        stat=stat,
    )
    hist_data = hist_data[hist_data["outcome"] == outcome]
    y = (
        hist_data
        .groupby("chunk_length", observed=True)["hist_weight"]
        .sum()
        .reindex(plot_x, fill_value=0)
        .sort_index()
        .to_numpy(dtype=float)
    )
    return y, float(hist_data["hist_weight"].sum())


def geometric_outcome_streak_probability(
    chunk_lengths,
    correct_probability: float,
    outcome: str,
):
    """Geometric streak-length probability for iid correct/incorrect outcomes."""

    continue_probability = (
        correct_probability
        if outcome == "Correct"
        else 1.0 - correct_probability
    )
    return (1.0 - continue_probability) * (continue_probability ** (chunk_lengths - 1))


def build_outcome_streak_plot_data(
    plot_dfs: dict,
    task_names: Sequence[str],
    *,
    glm_simulated_dfs: dict | None = None,
    stat: str = "count",
    task_labels: dict[str, str] | None = None,
    task_order: Sequence[str] = ("2ADC", "2AFC", "MCDR"),
    outcome_palette: dict[str, str] | None = None,
    max_chunk_length: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Build correct/incorrect streak tables with independent and GLM predictions."""

    task_labels = {
        "2AFC": "2AFC",
        "2AFC_delay": "2ADC",
        "2ADC": "2ADC",
        "MCDR": "MCDR",
        **(task_labels or {}),
    }
    glm_simulated_dfs = glm_simulated_dfs or {}
    outcome_palette = outcome_palette or {"Correct": "tab:green", "Incorrect": "tab:red"}
    plot_x = np.arange(1, int(max_chunk_length) + 1)

    chunk_frames = []
    correct_probability_rows = []
    plot_rows = []
    for task_name in task_names:
        task_label = task_labels.get(task_name, task_name)
        if task_label not in task_order:
            continue

        chunks, correct_probability = outcome_streak_lengths_for_sequence(
            plot_dfs[task_name],
            task_name=task_name,
            task_label=task_label,
        )
        if not chunks.empty:
            chunk_frames.append(chunks)
        correct_probability_rows.append(correct_probability)
        glm_chunks = _outcome_streak_chunks_from_simulation(
            glm_simulated_dfs.get(task_name, pd.DataFrame()),
            task_name=task_name,
            task_label=task_label,
        )

        for outcome in outcome_palette:
            data_y, data_total = _weighted_outcome_streak_histogram_y(
                chunks,
                outcome=outcome,
                stat=stat,
                plot_x=plot_x,
            )
            plot_rows.extend(
                {
                    "task_label": task_label,
                    "chunk_length": chunk_length,
                    "outcome": outcome,
                    "source": "Data",
                    "weight": weight,
                }
                for chunk_length, weight in zip(plot_x, data_y, strict=False)
            )

            p_correct = correct_probability["p_correct"]
            if np.isfinite(p_correct):
                independent_y = geometric_outcome_streak_probability(
                    plot_x,
                    float(p_correct),
                    outcome,
                )
                if stat == "count":
                    independent_y = independent_y * data_total
                plot_rows.extend(
                    {
                        "task_label": task_label,
                        "chunk_length": chunk_length,
                        "outcome": outcome,
                        "source": "Independent choices",
                        "weight": weight,
                    }
                    for chunk_length, weight in zip(plot_x, independent_y, strict=False)
                )

            glm_y, _ = _weighted_outcome_streak_histogram_y(
                glm_chunks,
                outcome=outcome,
                stat=stat,
                plot_x=plot_x,
            )
            plot_rows.extend(
                {
                    "task_label": task_label,
                    "chunk_length": chunk_length,
                    "outcome": outcome,
                    "source": "GLM",
                    "weight": weight,
                }
                for chunk_length, weight in zip(plot_x, glm_y, strict=False)
            )

    outcome_streak_lengths = (
        pd.concat(chunk_frames, ignore_index=True)
        if chunk_frames
        else pd.DataFrame()
    )
    outcome_streak_probabilities = pd.DataFrame(correct_probability_rows)
    return (
        outcome_streak_lengths,
        pd.DataFrame(plot_rows),
        outcome_palette,
        outcome_streak_probabilities,
    )


def build_repetition_chunk_plot_data(
    plot_dfs: dict,
    task_names: Sequence[str],
    *,
    glm_simulated_dfs: dict | None = None,
    stat: str = "count",
    task_labels: dict[str, str] | None = None,
    task_order: Sequence[str] = ("2ADC", "2AFC", "MCDR"),
    max_chunk_length: int = 100,
) -> pd.DataFrame:
    """Build repeated-choice streak plot data with independent and GLM predictions."""

    task_labels = {
        "2AFC": "2AFC",
        "2AFC_delay": "2ADC",
        "2ADC": "2ADC",
        "MCDR": "MCDR",
        **(task_labels or {}),
    }
    glm_simulated_dfs = glm_simulated_dfs or {}
    plot_x = np.arange(1, int(max_chunk_length) + 1)

    rows = []
    for task_name in task_names:
        task_label = task_labels.get(task_name, task_name)
        if task_label not in task_order:
            continue
        chunks, repeat_probability = transition_chunk_lengths_for_sequence(
            plot_dfs[task_name],
            task_name=task_name,
            task_label=task_label,
            sequence_col="response",
            sequence_label="Choices",
        )
        chunks = chunks[chunks["transition"] == "repeating"]
        data_y, data_total = _weighted_chunk_histogram_y(
            chunks,
            stat=stat,
            plot_x=plot_x,
        )
        rows.extend(
            {
                "task_label": task_label,
                "chunk_length": chunk_length,
                "transition": "repeating",
                "source": "Data",
                "weight": weight,
            }
            for chunk_length, weight in zip(plot_x, data_y, strict=False)
        )

        independent_y = geometric_transition_chunk_probability(
            plot_x,
            float(repeat_probability["p_repeat"]),
            "repeating",
        )
        if stat == "count":
            independent_y = independent_y * data_total
        rows.extend(
            {
                "task_label": task_label,
                "chunk_length": chunk_length,
                "transition": "repeating",
                "source": "Independent choices",
                "weight": weight,
            }
            for chunk_length, weight in zip(plot_x, independent_y, strict=False)
        )

        glm_chunks = _transition_chunks_from_simulation(
            glm_simulated_dfs.get(task_name, pd.DataFrame()),
            task_name=task_name,
            task_label=task_label,
        )
        if "transition" in glm_chunks.columns:
            glm_chunks = glm_chunks[glm_chunks["transition"] == "repeating"]
        glm_y, _ = _weighted_chunk_histogram_y(
            glm_chunks,
            stat=stat,
            plot_x=plot_x,
        )
        rows.extend(
            {
                "task_label": task_label,
                "chunk_length": chunk_length,
                "transition": "repeating",
                "source": "GLM",
                "weight": weight,
            }
            for chunk_length, weight in zip(plot_x, glm_y, strict=False)
        )

    return pd.DataFrame(rows)


def build_transition_chunk_plot_data(
    plot_dfs: dict,
    task_names: Sequence[str],
    *,
    stat: str = "count",
    task_labels: dict[str, str] | None = None,
    task_order: Sequence[str] = ("2ADC", "2AFC", "MCDR"),
    transition_palette: dict[str, str] | None = None,
    max_chunk_length: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Build transition chunk tables and tidy plot data for task choice sequences."""
    task_labels = {
        "2AFC": "2AFC",
        "2AFC_delay": "2ADC",
        "2ADC": "2ADC",
        "MCDR": "MCDR",
        **(task_labels or {}),
    }
    transition_palette = transition_palette or {
        "repeating": "tab:brown",
        "alternating": "tab:purple",
    }

    chunk_frames = []
    repeat_probability_rows = []
    for task_name in task_names:
        task_label = task_labels.get(task_name, task_name)
        for sequence_col, sequence_label in [("response", "Choices"), ("stimulus", "Stimulus")]:
            if {"subject", "session", "trial_idx", sequence_col}.issubset(plot_dfs[task_name].columns):
                chunks, repeat_probability = transition_chunk_lengths_for_sequence(
                    plot_dfs[task_name],
                    task_name=task_name,
                    task_label=task_label,
                    sequence_col=sequence_col,
                    sequence_label=sequence_label,
                )
                chunk_frames.append(chunks)
                repeat_probability_rows.append(repeat_probability)

    transition_chunk_lengths = (
        pd.concat(chunk_frames, ignore_index=True)
        if chunk_frames
        else pd.DataFrame()
    )
    transition_repeat_probabilities = pd.DataFrame(repeat_probability_rows)
    plot_x = np.arange(1, int(max_chunk_length) + 1)

    plot_rows = []
    for task_label in task_order:
        data = transition_chunk_lengths[
            (transition_chunk_lengths["task_label"] == task_label)
            & (transition_chunk_lengths["sequence"] == "Choices")
        ]
        if data.empty:
            continue

        hist_data = animal_chunk_histogram_weights(
            data,
            group_cols=["transition"],
            stat=stat,
        )
        choice_probability = repeat_probability_for_transition_chunks(
            transition_repeat_probabilities,
            task_label=task_label,
            sequence="Choices",
        )
        for transition in transition_palette:
            transition_data = hist_data[hist_data["transition"] == transition]
            transition_total = transition_data["hist_weight"].sum()
            animal_y = (
                transition_data
                .groupby("chunk_length", observed=True)["hist_weight"]
                .sum()
                .reindex(plot_x, fill_value=0)
                .sort_index()
                .to_numpy(dtype=float)
            )
            plot_rows.extend(
                {
                    "task_label": task_label,
                    "chunk_length": chunk_length,
                    "transition": transition,
                    "source": "Data",
                    "weight": weight,
                }
                for chunk_length, weight in zip(plot_x, animal_y, strict=False)
            )
            if choice_probability is not None:
                generated_y = geometric_transition_chunk_probability(
                    plot_x,
                    choice_probability,
                    transition,
                )
                if stat == "count":
                    generated_y = generated_y * transition_total
                plot_rows.extend(
                    {
                        "task_label": task_label,
                        "chunk_length": chunk_length,
                        "transition": transition,
                        "source": "Independent choices",
                        "weight": weight,
                    }
                    for chunk_length, weight in zip(plot_x, generated_y, strict=False)
                )

    transition_chunk_plot_data = pd.DataFrame(plot_rows)
    return (
        transition_chunk_lengths,
        transition_chunk_plot_data,
        transition_palette,
        transition_repeat_probabilities,
    )


def build_transition_chunk_drug_plot_data(
    get_adapter,
    *,
    task_specs: Sequence[tuple[str, str, str]] = (
        ("2AFC_DRUG", "2AFC", "Drug"),
        ("2ADC_DRUG", "2ADC", "drug_code"),
        ("MCDR", "MCDR", "Drug"),
    ),
    task_order: Sequence[str] = ("2AFC", "2ADC", "MCDR"),
    drug_order: Sequence[str] = ("No drug", "Drug"),
    transition_palette: dict[str, str] | None = None,
    max_chunk_length: int = 100,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build tidy drug-condition transition chunk plot data."""
    transition_palette = transition_palette or {
        "repeating": "tab:brown",
        "alternating": "tab:purple",
    }
    x_values = np.arange(1, int(max_chunk_length) + 1)

    def drug_label(series):
        numeric = pd.to_numeric(series, errors="coerce")
        label = pd.Series(pd.NA, index=series.index, dtype="object")
        label[numeric == 0] = "No drug"
        label[numeric == 1] = "Drug"
        text = series.astype(str).str.strip().str.lower()
        label[text.isin(["saline", "no drug", "nodrug"])] = "No drug"
        label[text.isin(["drug", "nr2b"])] = "Drug"
        return label

    def transition_chunks_for_drug_task(task_name, task_label, drug_col):
        adapter = get_adapter(task_name)
        df = adapter.subject_filter(adapter.read_dataset()).to_pandas()
        drug_col = pick_existing_column(df, [drug_col, "Drug", "condition", "drug_code"])
        subject_col = pick_existing_column(df, ["subject", "Subject"])
        session_col = pick_existing_column(df, ["session", "Session"])
        trial_col = pick_existing_column(df, ["trial_idx", "Trial", "trial"])
        response_col = pick_existing_column(df, ["response", "Choice", "choice", "choices"])
        if any(col is None for col in [drug_col, subject_col, session_col, trial_col, response_col]):
            raise ValueError(f"{task_name}: missing columns needed for drug transition chunks.")

        trials = df[
            [subject_col, session_col, trial_col, response_col, drug_col]
        ].copy()
        trials.columns = ["subject", "session", "trial_idx", "response", "drug"]
        trials["drug_label"] = drug_label(trials["drug"])
        trials = (
            trials
            .dropna(subset=["subject", "session", "trial_idx", "response", "drug_label"])
            .sort_values(["subject", "drug_label", "session", "trial_idx"])
        )
        trials["previous_response"] = (
            trials
            .groupby(["subject", "drug_label", "session"], observed=True)["response"]
            .shift(1)
        )
        trials = trials.dropna(subset=["previous_response"])
        trials["transition"] = (
            trials["response"]
            .eq(trials["previous_response"])
            .map({True: "repeating", False: "alternating"})
        )
        trials["transition_chunk"] = (
            trials
            .groupby(["subject", "drug_label", "session"], observed=True)["transition"]
            .transform(lambda transition: transition.ne(transition.shift()).cumsum())
        )
        chunks = (
            trials
            .groupby(
                ["subject", "drug_label", "session", "transition", "transition_chunk"],
                observed=True,
            )
            .size()
            .rename("chunk_length")
            .reset_index()
        )
        chunks["task"] = task_name
        chunks["task_label"] = task_label
        repeat_probabilities = (
            trials
            .assign(is_repeating=trials["transition"] == "repeating")
            .groupby("drug_label", observed=True)["is_repeating"]
            .mean()
            .rename("p_repeat")
            .reset_index()
        )
        repeat_probabilities["task"] = task_name
        repeat_probabilities["task_label"] = task_label
        return chunks, repeat_probabilities

    chunk_frames = []
    repeat_probability_frames = []
    for task_name, task_label, drug_col in task_specs:
        chunks, repeat_probabilities = transition_chunks_for_drug_task(
            task_name,
            task_label,
            drug_col,
        )
        chunk_frames.append(chunks)
        repeat_probability_frames.append(repeat_probabilities)

    chunk_lengths = pd.concat(chunk_frames, ignore_index=True)
    repeat_probabilities = pd.concat(repeat_probability_frames, ignore_index=True)

    def geometric_chunk_probability(chunk_lengths, repeat_probability, transition):
        continue_probability = (
            repeat_probability
            if transition == "repeating"
            else 1.0 - repeat_probability
        )
        return (1.0 - continue_probability) * (continue_probability ** (chunk_lengths - 1))

    def repeat_probability_for(task_label, drug_label):
        matches = repeat_probabilities.loc[
            (repeat_probabilities["task_label"] == task_label)
            & (repeat_probabilities["drug_label"] == drug_label),
            "p_repeat",
        ]
        return float(matches.iloc[0])

    plot_rows = []
    for task_label in task_order:
        for current_drug_label in drug_order:
            data = chunk_lengths[
                (chunk_lengths["task_label"] == task_label)
                & (chunk_lengths["drug_label"] == current_drug_label)
            ]
            hist_data = animal_chunk_histogram_weights(
                data,
                group_cols=["drug_label", "transition"],
                stat="probability",
            )
            choice_probability = repeat_probability_for(task_label, current_drug_label)

            for transition in transition_palette:
                transition_data = hist_data[hist_data["transition"] == transition]
                animal_y = (
                    transition_data
                    .groupby("chunk_length", observed=True)["hist_weight"]
                    .sum()
                    .reindex(x_values, fill_value=0)
                    .sort_index()
                    .to_numpy(dtype=float)
                )
                generated_y = geometric_chunk_probability(
                    x_values,
                    choice_probability,
                    transition,
                )
                plot_rows.extend(
                    {
                        "task_label": task_label,
                        "drug_label": current_drug_label,
                        "chunk_length": chunk_length,
                        "transition": transition,
                        "source": "Data",
                        "weight": weight,
                    }
                    for chunk_length, weight in zip(x_values, animal_y, strict=False)
                )
                plot_rows.extend(
                    {
                        "task_label": task_label,
                        "drug_label": current_drug_label,
                        "chunk_length": chunk_length,
                        "transition": transition,
                        "source": "Independent choices",
                        "weight": weight,
                    }
                    for chunk_length, weight in zip(x_values, generated_y, strict=False)
                )

    return pd.DataFrame(plot_rows), transition_palette


def prepare_glm_simulated_corrected_behavior_autocorrelograms(
    df_like,
    arrays_store: dict,
    *,
    adapter=None,
    subject_col: str = "subject",
    session_col: str = "session",
    trial_index_col: str | None = None,
    correct_label_col: str = "stimulus",
    tau: float = 50.0,
    emission_cols: Sequence[str] | None = None,
    recursive: bool = False,
    n_simulations: int = 20,
    max_lag: int = 50,
    min_cross_pairs: int = 20,
    max_cross_pairs: int = 80,
    seed: int = 0,
    summary_only: bool = False,
) -> dict:
    """Simulate fitted GLM choices and prepare corrected autocorrelograms."""
    from glmhmmt.glm import glm_probs_from_weights, simulate_glm_choices

    df = to_pandas_df(df_like)
    required = {subject_col, session_col}
    adapter_behavioral_cols = dict(getattr(adapter, "behavioral_cols", {})) if adapter is not None else {}
    can_infer_correct_label = (
        not recursive
        and bool(adapter_behavioral_cols)
        and adapter_behavioral_cols.get("stimulus") in df.columns
        and adapter_behavioral_cols.get("performance") in df.columns
    )
    if not recursive and not can_infer_correct_label:
        required.add(correct_label_col)
    if trial_index_col is not None:
        required.add(trial_index_col)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for GLM simulation: {sorted(missing)}.")
    if recursive and adapter is None:
        raise ValueError("recursive=True requires adapter=... so history regressors can be rebuilt.")

    rng = np.random.default_rng(int(seed))
    frames = []
    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    sort_cols = [session_col]
    if trial_index_col is not None:
        sort_cols.append(trial_index_col)

    correction_rng = np.random.default_rng(int(seed))
    subject_summary_rows = []

    def summarize_simulated_subject(
        *,
        pseudo_subject: str,
        sessions: np.ndarray,
        choices: np.ndarray,
        correct_label: np.ndarray,
    ) -> None:
        sim_df = pd.DataFrame(
            {
                "session": sessions,
                "response": np.asarray(choices, dtype=int),
                "performance": (np.asarray(choices, dtype=float) == correct_label).astype(float),
            }
        )
        sim_df["_order"] = np.arange(len(sim_df), dtype=float)
        sim_df = sim_df.sort_values(["session", "_order"], kind="mergesort")
        sim_df["repetition"] = (
            sim_df["response"]
            .eq(sim_df.groupby("session", observed=True)["response"].shift(1))
            .where(sim_df.groupby("session", observed=True)["response"].shift(1).notna(), np.nan)
            .astype(float)
        )

        signal_cols = {"Outcome": "performance", "Repetition": "repetition"}
        raw_rows = []
        session_map = {}
        for session, session_df in sim_df.groupby("session", observed=True):
            session_map[session] = session_df
            for signal, value_col in signal_cols.items():
                ac = binary_autocorrelation(session_df[value_col], max_lag=max_lag)
                if ac.empty:
                    continue
                ac["signal"] = signal
                raw_rows.append(ac[["signal", "lag", "autocorr"]])
        if not raw_rows:
            return
        subject_raw = (
            pd.concat(raw_rows, ignore_index=True)
            .groupby(["signal", "lag"], observed=True)["autocorr"]
            .mean()
            .reset_index(name="raw_autocorr")
        )

        session_values = list(session_map)
        cross_rows = []
        if len(session_values) >= 2:
            all_pairs = [
                (left, right)
                for idx, left in enumerate(session_values)
                for right in session_values[idx + 1 :]
            ]
            if len(all_pairs) > int(max_cross_pairs):
                pair_idx = correction_rng.choice(len(all_pairs), size=int(max_cross_pairs), replace=False)
                pairs = [all_pairs[int(idx)] for idx in pair_idx]
            else:
                pairs = all_pairs
            if len(pairs) < int(min_cross_pairs) and len(all_pairs) >= int(min_cross_pairs):
                pair_idx = correction_rng.choice(len(all_pairs), size=int(min_cross_pairs), replace=False)
                pairs = [all_pairs[int(idx)] for idx in pair_idx]
            for left, right in pairs:
                left_df = session_map[left]
                right_df = session_map[right]
                for signal, value_col in signal_cols.items():
                    cc = binary_crosscorrelation(
                        left_df[value_col],
                        right_df[value_col],
                        max_lag=max_lag,
                    )
                    if cc.empty:
                        continue
                    cc["signal"] = signal
                    cross_rows.append(cc[["signal", "lag", "crosscorr"]])

        if cross_rows:
            subject_cross = (
                pd.concat(cross_rows, ignore_index=True)
                .groupby(["signal", "lag"], observed=True)["crosscorr"]
                .mean()
                .reset_index()
            )
        else:
            subject_cross = pd.DataFrame(columns=["signal", "lag", "crosscorr"])
        subject_autocorr = subject_raw.merge(subject_cross, on=["signal", "lag"], how="left")
        subject_autocorr["crosscorr"] = subject_autocorr["crosscorr"].fillna(0.0)
        subject_autocorr["autocorr"] = (
            subject_autocorr["raw_autocorr"] - subject_autocorr["crosscorr"]
        )
        subject_autocorr["subject"] = pseudo_subject
        subject_summary_rows.append(subject_autocorr)

    def apply_lapse_to_step_probs(
        probs: np.ndarray,
        *,
        previous_choice: int | None,
        lapse_mode: str,
        lapse_rates: np.ndarray,
    ) -> np.ndarray:
        out = np.asarray(probs, dtype=float).copy()
        if previous_choice is None:
            out = np.clip(out, 1e-12, 1.0)
            return out / out.sum()
        num_classes = out.size
        if lapse_mode == "class" and lapse_rates.size:
            total_mass = float(np.sum(lapse_rates))
            out = lapse_rates + (1.0 - total_mass) * out
        elif lapse_mode == "history":
            repeat_rate = float(lapse_rates[0]) if lapse_rates.size > 0 else 0.0
            alternate_rate = float(lapse_rates[1]) if lapse_rates.size > 1 else 0.0
            repeat_target = np.zeros(num_classes, dtype=float)
            repeat_target[int(previous_choice)] = 1.0
            alternate_target = np.full(num_classes, 1.0 / max(1, num_classes - 1), dtype=float)
            alternate_target[int(previous_choice)] = 0.0
            out = (1.0 - repeat_rate - alternate_rate) * out
            out += repeat_rate * repeat_target + alternate_rate * alternate_target
        elif lapse_mode == "history_conditioned":
            repeat_rates = lapse_rates[:num_classes] if lapse_rates.size >= num_classes else np.zeros(num_classes)
            alternate_rates = (
                lapse_rates[num_classes : 2 * num_classes]
                if lapse_rates.size >= 2 * num_classes
                else np.zeros(num_classes)
            )
            repeat_rate = float(repeat_rates[int(previous_choice)])
            alternate_rate = float(alternate_rates[int(previous_choice)])
            repeat_target = np.zeros(num_classes, dtype=float)
            repeat_target[int(previous_choice)] = 1.0
            alternate_target = np.full(num_classes, 1.0 / max(1, num_classes - 1), dtype=float)
            alternate_target[int(previous_choice)] = 0.0
            out = (1.0 - repeat_rate - alternate_rate) * out
            out += repeat_rate * repeat_target + alternate_rate * alternate_target
        out = np.clip(out, 1e-12, 1.0)
        return out / out.sum()

    def infer_class_to_response(raw_response: pd.Series, y_values: np.ndarray) -> dict[int, object]:
        out = {}
        tmp = pd.DataFrame({"response": raw_response.to_numpy(), "class": np.asarray(y_values, dtype=int)})
        for class_idx, group in tmp.dropna().groupby("class", observed=True):
            mode = group["response"].mode(dropna=True)
            out[int(class_idx)] = mode.iloc[0] if not mode.empty else int(class_idx)
        return out

    def infer_correct_classes(
        raw_df: pd.DataFrame,
        *,
        stimulus_col: str,
        performance_col: str,
        y_values: np.ndarray,
    ) -> np.ndarray:
        performance = pd.to_numeric(raw_df[performance_col], errors="coerce")
        stimulus = raw_df[stimulus_col]
        mapping = {}
        correct_trials = pd.DataFrame(
            {
                "stimulus": stimulus.to_numpy(),
                "performance": performance.to_numpy(dtype=float),
                "class": np.asarray(y_values, dtype=int),
            }
        )
        for stim_value, group in correct_trials[correct_trials["performance"] > 0].groupby("stimulus", observed=True):
            mode = group["class"].mode(dropna=True)
            if not mode.empty:
                mapping[stim_value] = int(mode.iloc[0])
        return np.asarray([mapping.get(value, np.nan) for value in stimulus.to_numpy()], dtype=float)

    for subject, arrays in arrays_store.items():
        subject_df = df[df[subject_col].astype(str) == str(subject)].copy()
        if subject_df.empty:
            continue
        subject_df = subject_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        X = np.asarray(arrays.get("X"), dtype=float)
        if X.ndim != 2 or X.shape[0] != len(subject_df):
            continue

        weights = np.asarray(arrays.get("emission_weights"), dtype=float)
        if weights.ndim == 3:
            weights = weights[0]
        p_pred = np.asarray(arrays.get("p_pred"), dtype=float)
        y = np.asarray(arrays.get("y", []), dtype=int)
        num_classes = int(p_pred.shape[1]) if p_pred.ndim == 2 else int(np.nanmax(y) + 1)
        baseline_class_idx = int(np.asarray(arrays.get("baseline_class_idx", 0)).reshape(()))
        lapse_mode = np.asarray(arrays.get("lapse_mode", "none")).reshape(()).item()
        lapse_rates = np.asarray(arrays.get("lapse_rates", []), dtype=float)

        if recursive:
            behavioral_cols = adapter_behavioral_cols
            response_col = behavioral_cols["response"]
            performance_col = behavioral_cols["performance"]
            stimulus_col = behavioral_cols["stimulus"]
            if any(col not in subject_df.columns for col in [response_col, performance_col, stimulus_col]):
                continue
            y_original = np.asarray(arrays.get("y", []), dtype=int)
            if y_original.shape[0] != len(subject_df):
                continue
            class_to_response = infer_class_to_response(subject_df[response_col], y_original)
            correct_class = infer_correct_classes(
                subject_df,
                stimulus_col=stimulus_col,
                performance_col=performance_col,
                y_values=y_original,
            )
            x_cols = list(emission_cols or arrays.get("X_cols", []))
            if not x_cols:
                x_cols = list(arrays.get("X_cols", []))
            simulations = []
            for _simulation_idx in range(n_simulations):
                sim_df = subject_df.copy()
                sim_choices = np.zeros(len(sim_df), dtype=int)
                for trial_idx in range(len(sim_df)):
                    _, X_current, _, _names = adapter.load_subject(
                        pl.from_pandas(sim_df),
                        tau=float(tau),
                        emission_cols=x_cols if x_cols else None,
                    )
                    x_trial = np.asarray(X_current, dtype=float)[trial_idx : trial_idx + 1]
                    probs = glm_probs_from_weights(
                        x_trial,
                        weights,
                        baseline_class_idx=baseline_class_idx,
                        num_classes=num_classes,
                    )[0]
                    previous_choice = int(sim_choices[trial_idx - 1]) if trial_idx > 0 else None
                    probs = apply_lapse_to_step_probs(
                        probs,
                        previous_choice=previous_choice,
                        lapse_mode=str(lapse_mode),
                        lapse_rates=lapse_rates,
                    )
                    choice = int(rng.choice(num_classes, p=probs))
                    sim_choices[trial_idx] = choice
                    sim_df.loc[trial_idx, response_col] = class_to_response.get(choice, choice)
                    sim_df.loc[trial_idx, performance_col] = float(choice == correct_class[trial_idx])
                simulations.append(sim_choices)
            simulations = np.asarray(simulations, dtype=int)
            correct_label = correct_class
        else:
            simulations = simulate_glm_choices(
                X,
                weights,
                baseline_class_idx=baseline_class_idx,
                num_classes=num_classes,
                lapse_mode=str(lapse_mode),
                lapse_rates=lapse_rates,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                n_simulations=n_simulations,
            )
            correct_label = None
            if can_infer_correct_label and y.shape[0] == len(subject_df):
                stimulus_col = adapter_behavioral_cols["stimulus"]
                performance_col = adapter_behavioral_cols["performance"]
                inferred_correct_label = infer_correct_classes(
                    subject_df,
                    stimulus_col=stimulus_col,
                    performance_col=performance_col,
                    y_values=y,
                )
                if np.isfinite(inferred_correct_label).any():
                    correct_label = inferred_correct_label
            if correct_label is None:
                if correct_label_col not in subject_df.columns:
                    raise ValueError(
                        f"Missing required columns for GLM simulation: {[correct_label_col]}."
                    )
                correct_label = pd.to_numeric(subject_df[correct_label_col], errors="coerce").to_numpy(dtype=float)

        for simulation_idx, simulated_choice in enumerate(simulations):
            if summary_only:
                summarize_simulated_subject(
                    pseudo_subject=f"{subject}__glm_sim_{simulation_idx:03d}",
                    sessions=subject_df[session_col].to_numpy(),
                    choices=simulated_choice,
                    correct_label=correct_label,
                )
                continue
            sim_frame = pd.DataFrame(
                {
                    "subject": f"{subject}__glm_sim_{simulation_idx:03d}",
                    "session": subject_df[session_col].to_numpy(),
                    "trial_index": (
                        pd.to_numeric(subject_df[trial_index_col], errors="coerce").to_numpy(dtype=float)
                        if trial_index_col is not None
                        else np.arange(len(subject_df), dtype=float)
                    ),
                    "response": simulated_choice.astype(int),
                    "performance": (simulated_choice.astype(float) == correct_label).astype(float),
                }
            )
            frames.append(sim_frame)

    if summary_only and subject_summary_rows:
        subject_autocorr = pd.concat(subject_summary_rows, ignore_index=True)
        summary = (
            subject_autocorr.groupby(["signal", "lag"], observed=True)
            .agg(
                autocorr=("autocorr", "mean"),
                autocorr_std=("autocorr", "std"),
                raw_autocorr=("raw_autocorr", "mean"),
                raw_autocorr_std=("raw_autocorr", "std"),
                crosscorr=("crosscorr", "mean"),
                crosscorr_std=("crosscorr", "std"),
                n_subjects=("subject", "count"),
            )
            .reset_index()
        )
        sem_denominator = np.sqrt(summary["n_subjects"].clip(lower=1))
        summary["autocorr_sem"] = summary["autocorr_std"].fillna(0.0) / sem_denominator
        summary["raw_autocorr_sem"] = summary["raw_autocorr_std"].fillna(0.0) / sem_denominator
        summary["crosscorr_sem"] = summary["crosscorr_std"].fillna(0.0) / sem_denominator
        return {
            "autocorr": summary,
            "subject_autocorr": subject_autocorr,
            "session_autocorr": pd.DataFrame(),
            "crosscorr": pd.DataFrame(),
            "sequences": pd.DataFrame(),
            "meta": {
                "max_lag": int(max_lag),
                "min_cross_pairs": int(min_cross_pairs),
                "max_cross_pairs": int(max_cross_pairs),
                "seed": int(seed),
                "n_simulations": n_simulations,
                "simulation": "recursive_glm" if recursive else "fixed_design_glm",
                "summary_only": True,
            },
        }

    if not frames:
        return {
            "autocorr": pd.DataFrame(),
            "subject_autocorr": pd.DataFrame(),
            "session_autocorr": pd.DataFrame(),
            "crosscorr": pd.DataFrame(),
            "sequences": pd.DataFrame(),
            "meta": {
                "max_lag": int(max_lag),
                "n_simulations": n_simulations,
                "seed": int(seed),
                "simulation": "recursive_glm" if recursive else "fixed_design_glm",
                "summary_only": bool(summary_only),
            },
        }

    simulated_df = pd.concat(frames, ignore_index=True)
    prepared = prepare_corrected_behavior_autocorrelograms(
        simulated_df,
        subject_col="subject",
        session_col="session",
        choice_col="response",
        outcome_col="performance",
        trial_index_col="trial_index",
        max_lag=max_lag,
        min_cross_pairs=min_cross_pairs,
        max_cross_pairs=max_cross_pairs,
        seed=seed,
    )
    prepared["meta"] = {
        **prepared.get("meta", {}),
        "n_simulations": n_simulations,
        "simulation": "recursive_glm" if recursive else "fixed_design_glm",
    }
    return prepared


def prepare_model_simulated_corrected_behavior_autocorrelograms(
    df_like,
    *,
    subject_col: str = "subject",
    session_col: str = "session",
    trial_index_col: str | None = None,
    response_col: str = "response",
    performance_col: str = "performance",
    stimulus_col: str = "stimulus",
    prob_cols: Sequence[str] | None = None,
    n_simulations: int = 20,
    max_lag: int = 50,
    min_cross_pairs: int = 20,
    max_cross_pairs: int = 80,
    seed: int = 0,
) -> dict:
    """Simulate choices from fitted trial-level probabilities and prepare autocorrelograms."""
    df = to_pandas_df(df_like).copy()
    required = {subject_col, session_col}
    if trial_index_col is not None:
        required.add(trial_index_col)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for model autocorrelogram simulation: {sorted(missing)}.")

    if prob_cols is None:
        if {"pL", "pC", "pR"}.issubset(df.columns):
            prob_cols = ("pL", "pC", "pR")
            classes = np.asarray([0, 1, 2], dtype=int)
        elif {"pL", "pR"}.issubset(df.columns):
            prob_cols = ("pL", "pR")
            classes = np.asarray([0, 1], dtype=int)
        elif "p_pred" in df.columns:
            prob_cols = ("_p_model_left", "p_pred")
            df["_p_model_left"] = 1.0 - pd.to_numeric(df["p_pred"], errors="coerce")
            classes = np.asarray([0, 1], dtype=int)
        else:
            raise ValueError("Need pL/pR, pL/pC/pR, or p_pred to simulate model choices.")
    else:
        prob_cols = tuple(prob_cols)
        classes = np.arange(len(prob_cols), dtype=int)

    for col in prob_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [subject_col, session_col]
    if trial_index_col is not None:
        sort_cols.append(trial_index_col)
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    stimulus_values = pd.to_numeric(df[stimulus_col], errors="coerce") if stimulus_col in df.columns else None
    correct_class = None
    if stimulus_values is not None:
        stim = stimulus_values.to_numpy(dtype=float)
        if len(classes) == 2:
            correct_class = np.where(
                np.isin(stim, [0.0, 1.0]),
                stim,
                np.where(np.isfinite(stim), (stim > 0.0).astype(float), np.nan),
            )
        elif len(classes) == 3:
            correct_class = np.where(np.isin(stim, classes.astype(float)), stim, np.nan)

    p_model_correct = (
        pd.to_numeric(df["p_model_correct"], errors="coerce")
        if "p_model_correct" in df.columns
        else None
    )
    observed_performance = (
        pd.to_numeric(df[performance_col], errors="coerce")
        if performance_col in df.columns
        else None
    )

    rng = np.random.default_rng(int(seed))
    frames = []
    n_simulations = int(n_simulations)
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    probabilities = df[list(prob_cols)].to_numpy(dtype=float)
    row_sums = np.nansum(probabilities, axis=1)
    valid_probs = np.isfinite(probabilities).all(axis=1) & (row_sums > 0)
    probabilities[valid_probs] = probabilities[valid_probs] / row_sums[valid_probs, None]

    for simulation_idx in range(n_simulations):
        simulated_response = np.full(len(df), np.nan, dtype=float)
        for row_idx, probs in enumerate(probabilities):
            if not valid_probs[row_idx]:
                continue
            simulated_response[row_idx] = int(rng.choice(classes, p=probs))

        if correct_class is not None and np.isfinite(correct_class).any():
            simulated_performance = (simulated_response == correct_class).astype(float)
            simulated_performance[~np.isfinite(simulated_response) | ~np.isfinite(correct_class)] = np.nan
        elif p_model_correct is not None:
            p_correct = p_model_correct.to_numpy(dtype=float)
            simulated_performance = (
                rng.random(len(df)) < np.clip(np.nan_to_num(p_correct, nan=0.0), 0.0, 1.0)
            ).astype(float)
            simulated_performance[~np.isfinite(p_correct)] = np.nan
        elif observed_performance is not None:
            simulated_performance = observed_performance.to_numpy(dtype=float)
        else:
            simulated_performance = np.full(len(df), np.nan, dtype=float)

        frames.append(
            pd.DataFrame(
                {
                    "subject": df[subject_col].astype(str) + f"__model_sim_{simulation_idx:03d}",
                    "session": df[session_col].to_numpy(),
                    "trial_index": (
                        pd.to_numeric(df[trial_index_col], errors="coerce").to_numpy(dtype=float)
                        if trial_index_col is not None
                        else np.arange(len(df), dtype=float)
                    ),
                    "response": simulated_response,
                    "performance": simulated_performance,
                }
            )
        )

    simulated_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    prepared = prepare_corrected_behavior_autocorrelograms(
        simulated_df,
        subject_col="subject",
        session_col="session",
        choice_col="response",
        outcome_col="performance",
        trial_index_col="trial_index",
        max_lag=max_lag,
        min_cross_pairs=min_cross_pairs,
        max_cross_pairs=max_cross_pairs,
        seed=seed,
    )
    prepared["meta"] = {
        **prepared.get("meta", {}),
        "n_simulations": n_simulations,
        "simulation": "trial_probability_model",
    }
    return prepared
