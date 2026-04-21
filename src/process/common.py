from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from typing import Callable, Optional, Sequence, Tuple


def to_pandas_df(df_like) -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()
    if hasattr(df_like, "to_pandas"):
        return df_like.to_pandas().copy()
    return pd.DataFrame(df_like).copy()


def display_regressor_name(regressor_col: str) -> str:
    if regressor_col == "choice_lag_one_hot_sum":
        return r"$A_t$"
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


REPEAT_EVIDENCE_TAIL_QUANTILES = (
    0.0,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
    0.995,
    0.9975,
    1.0,
)


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
        "choice_lag_one_hot_sum",
        "choice_lag_param",
        "at_choice_param",
    ]
    for regressor in preferred_order:
        if regressor in regressor_options:
            return regressor
    return None


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
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma_bins**2))
    return kernel / kernel.sum()


def _smooth_2d(values: np.ndarray, *, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return values
    kernel = _gaussian_kernel_1d(sigma_bins)

    def _same_length_convolve(row: np.ndarray) -> np.ndarray:
        convolved = np.convolve(row, kernel, mode="same")
        if convolved.shape[0] == row.shape[0]:
            return convolved
        start = (convolved.shape[0] - row.shape[0]) // 2
        return convolved[start : start + row.shape[0]]

    out = np.apply_along_axis(_same_length_convolve, 0, values)
    out = np.apply_along_axis(_same_length_convolve, 1, out)
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
    """Return a smoothed 2D map of mean values over x/y bins."""
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
        bnd = float(np.nanpercentile(np.abs(np.concatenate([x, y])), 98.0))
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

    weighted_sum, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=values)
    counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))

    x_step = float(np.nanmedian(np.diff(x_edges)))
    y_step = float(np.nanmedian(np.diff(y_edges)))
    smooth_step = min(x_step, y_step)

    if sigma is None:
        sigma = float(default_sigma_dx) * smooth_step
    sigma_bins = float(sigma) / smooth_step if smooth_step > 0 else 0.0

    weighted_sum = _smooth_2d(weighted_sum, sigma_bins=sigma_bins)
    counts = _smooth_2d(counts, sigma_bins=sigma_bins)

    mean_map = np.divide(
        weighted_sum,
        counts,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=counts > 1e-9,
    )
    if fill_empty:
        mean_map = _fill_nan_grid(mean_map)

    return {
        "map": mean_map,
        "n_datapoints": counts,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": (x_edges[:-1] + x_edges[1:]) / 2.0,
        "y_centers": (y_edges[:-1] + y_edges[1:]) / 2.0,
        "dx": dx,
        "bnd": bnd,
        "sigma": sigma,
        "sigma_bins": sigma_bins,
    }


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

    stim_sign = np.sign(pd.to_numeric(df[stim_col], errors="coerce"))
    delay_values = pd.to_numeric(df[delay_col], errors="coerce")

    display_delay = delay_values.mask(np.isclose(delay_values, 0.1), 0.0)
    signed_delay = display_delay * stim_sign
    df["_signed_delay"] = signed_delay

    # eje categórico para permitir dos ceros
    cat = pd.Series(pd.NA, index=df.index, dtype="object")
    valid = np.isfinite(display_delay) & np.isfinite(stim_sign)

    zero_left = valid & np.isclose(display_delay, 0.0) & (stim_sign < 0)
    zero_right = valid & np.isclose(display_delay, 0.0) & (stim_sign > 0)
    nonzero = valid & ~np.isclose(display_delay, 0.0)

    cat.loc[zero_left] = "0L"
    cat.loc[zero_right] = "0R"
    cat.loc[nonzero] = signed_delay.loc[nonzero].map(lambda value: f"{float(value):g}")

    preferred_order = ["0L", "-1", "-3", "-10", "10", "3", "1", "0R"]
    present = set(cat.dropna())
    existing = [x for x in preferred_order if x in present]
    extras = sorted(
        (x for x in present if x not in set(existing)),
        key=lambda x: float(x) if x not in {"0L", "0R"} else 0.0,
    )

    df["_signed_delay_cat"] = pd.Categorical(cat, categories=existing + extras, ordered=True)
    return df


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
    logits = np.zeros((view.T, num_classes), dtype=float)

    if is_mcdr and num_classes == 3 and explicit_logits.shape[1] == 2:
        logits[:, 0] = explicit_logits[:, 0]
        logits[:, 1] = 0.0
        logits[:, 2] = explicit_logits[:, 1]
        return logits

    logits[:, : explicit_logits.shape[1]] = explicit_logits
    return logits


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
) -> tuple[pd.DataFrame | None, dict]:
    df = df_pd.copy()
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

    subj = (
        df.groupby(["subject", "_bin"], observed=True)
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
        subj.groupby("_bin", observed=True)
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
    stim_param_idx = next((idx for idx, name in enumerate(names) if name == "stim_param"), None)
    stim_param_weights = stim_param_weight_map() if stim_param_idx is not None and stim_param_weight_map else {}
    ild_idx = next(
        (idx for idx, name in enumerate(names) if name in {"stim_vals", "stim_d", "ild_norm", "ILD", "ild", "stimulus"}),
        None,
    )

    if stim_abs_indices or stim_param_idx is not None:
        levels = sorted(set(stim_abs_indices) | set(stim_param_weights) | {0})
        grid = np.asarray(
            sorted({0.0} | {signed for level in levels if level != 0 for signed in (-float(level), float(level))}),
            dtype=float,
        )
    else:
        grid = np.linspace(-ild_max, ild_max, n_grid)

    feature_indices = sorted(
        set(
            ([ild_idx] if ild_idx is not None else [])
            + list(stim_abs_indices.values())
            + ([stim_param_idx] if stim_param_idx is not None else [])
        )
    )
    return {
        "grid": grid,
        "norm": grid / ild_max,
        "ild_idx": ild_idx,
        "stim_abs_indices": stim_abs_indices,
        "stim_param_idx": stim_param_idx,
        "stim_param_weights": stim_param_weights,
        "feature_indices": feature_indices,
    }


def _stimulus_values_for_grid(component: dict, ild_value: float, ild_norm: float) -> dict[int, float]:
    values = {}
    if component["ild_idx"] is not None:
        values[component["ild_idx"]] = float(ild_norm)
    for stim_abs, stim_abs_idx in component["stim_abs_indices"].items():
        if stim_abs == 0:
            values[stim_abs_idx] = 1.0 if ild_value == 0 else 0.0
        else:
            values[stim_abs_idx] = float(np.sign(ild_value)) if abs(ild_value) == float(stim_abs) else 0.0
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

    if X_data is not None and stim_feature_indices:
        X_base = np.asarray(X_data, dtype=float).copy()
        for k in range(K):
            w = W[k, 0, :]
            other_logit = X_base @ w
            base_logit = other_logit - (X_base[:, stim_feature_indices] @ w[stim_feature_indices])
            for grid_idx, (ild_value, stim_value_norm) in enumerate(zip(ild_grid, ild_norm, strict=False)):
                stim_logit = sum(
                    value * w[idx]
                    for idx, value in _stimulus_values_for_grid(component, ild_value, stim_value_norm).items()
                )
                p_trial = gL + (1.0 - gL - gR) * _right_probability(
                    base_logit + stim_logit,
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
            for idx, value in _stimulus_values_for_grid(component, ild_value, stim_value_norm).items():
                X_grid[row_idx, idx] = value
        if bias_idx is not None:
            X_grid[:, bias_idx] = 1.0

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

    X_base = None
    if X_data is not None:
        candidate = np.asarray(X_data, dtype=float).copy()
        if candidate.ndim == 2 and candidate.shape[1] == M:
            X_base = candidate
    weights_t = _valid_trial_weights(X_base, trial_weights)

    if X_base is not None:
        for k in range(K):
            w = W[k, 0, :]
            base_logit = (X_base @ w) - (X_base[:, feat_idx] * w[feat_idx])
            for grid_idx, grid_value in enumerate(grid):
                p_trial = gL + (1.0 - gL - gR) * _right_probability(
                    base_logit + grid_value * w[feat_idx],
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
