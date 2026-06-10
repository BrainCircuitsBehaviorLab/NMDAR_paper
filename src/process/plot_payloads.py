"""Shared trial-level payload builders for binary task plots.

Task modules pass a small profile into this module. The profile owns only the
facts that vary by task: correct-probability rule, default x-axis, and legend
metadata. Everything else stays behind this interface so plotting and notebooks
do not need to learn each task's private dataframe shape.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from src.process.common import (
    attach_quantile_bin_column,
    attach_response_right_column,
    attach_signed_delay_columns,
    display_regressor_name,
    p_right_label,
    prepare_simple_regressor_curve,
    resolve_grouping,
    summarize_grouped_panel,
    to_pandas_df,
)


CorrectRule = Literal["binary_zero_left", "signed_or_binary"]
AxisKind = Literal["ild", "delay"]


@dataclass(frozen=True)
class TaskPlotProfile:
    """Task-owned facts needed by the shared plot payload implementation."""

    task_label: str
    pred_col: str
    response_mode: str
    baseline: float
    correct_rule: CorrectRule
    binned_axis: AxisKind
    right_group_axis: AxisKind
    delay_sources: tuple[str, ...] = ()
    signed_delay_order: tuple[str, ...] = ()
    signed_delay_labels: tuple[str, ...] = ()


TWO_AFC_PROFILE = TaskPlotProfile(
    task_label="2AFC",
    pred_col="p_pred",
    response_mode="pm1_or_prob",
    baseline=0.5,
    correct_rule="binary_zero_left",
    binned_axis="ild",
    right_group_axis="ild",
)

TWO_ADC_PROFILE = TaskPlotProfile(
    task_label="2AFC-delay",
    pred_col="p_pred",
    response_mode="pm1_or_prob",
    baseline=0.5,
    correct_rule="signed_or_binary",
    binned_axis="delay",
    right_group_axis="delay",
    delay_sources=("delays", "delay_raw", "delay"),
    signed_delay_order=("-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"),
    signed_delay_labels=("-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"),
)


def _require_prediction_columns(df, *, task_label: str) -> None:
    required = {"stimulus", "response", "performance"}
    missing = sorted(required.difference(getattr(df, "columns", [])))
    if missing:
        raise ValueError(f"Missing required {task_label} columns: {missing}")
    if "pL" not in df.columns or "pR" not in df.columns:
        raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")


def _correct_prob_expr(profile: TaskPlotProfile):
    stim = pl.col("stimulus").cast(pl.Float64)
    if profile.correct_rule == "binary_zero_left":
        return pl.when(stim == 0).then(pl.col("pL")).otherwise(pl.col("pR"))
    return (
        pl.when(stim.is_in([0.0, 1.0]))
        .then(pl.when(stim == 0.0).then(pl.col("pL")).otherwise(pl.col("pR")))
        .when(stim.is_in([-1.0, 1.0]))
        .then(pl.when(stim < 0.0).then(pl.col("pL")).otherwise(pl.col("pR")))
        .otherwise(None)
    )


def _correct_prob_pandas(df: pd.DataFrame, profile: TaskPlotProfile) -> pd.Series:
    stim = pd.to_numeric(df["stimulus"], errors="coerce")
    if profile.correct_rule == "binary_zero_left":
        return pd.Series(
            np.where(stim == 0, df["pL"], df["pR"]),
            index=df.index,
            dtype=float,
        )

    out = pd.Series(np.nan, index=df.index, dtype=float)
    binary_mask = stim.isin([0.0, 1.0])
    signed_mask = stim.isin([-1.0, 1.0])
    out.loc[binary_mask & (stim == 0.0)] = pd.to_numeric(
        df.loc[binary_mask & (stim == 0.0), "pL"],
        errors="coerce",
    )
    out.loc[binary_mask & (stim == 1.0)] = pd.to_numeric(
        df.loc[binary_mask & (stim == 1.0), "pR"],
        errors="coerce",
    )
    out.loc[signed_mask & (stim < 0.0)] = pd.to_numeric(
        df.loc[signed_mask & (stim < 0.0), "pL"],
        errors="coerce",
    )
    out.loc[signed_mask & (stim > 0.0)] = pd.to_numeric(
        df.loc[signed_mask & (stim > 0.0), "pR"],
        errors="coerce",
    )
    return out


def prepare_predictions_df(df_pred, *, profile: TaskPlotProfile):
    """Prepare canonical trial-level prediction columns for a binary task."""

    if isinstance(df_pred, pl.DataFrame):
        df = df_pred.clone()
        _require_prediction_columns(df, task_label=profile.task_label)
        if "correct_bool" not in df.columns:
            df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
        df = df.with_columns(
            pl.col("pR").alias(profile.pred_col),
            _correct_prob_expr(profile).alias("p_model_correct"),
        )
        for source_col in profile.delay_sources:
            if source_col in df.columns:
                return df.with_columns(pl.col(source_col).cast(pl.Float64).alias("delay"))
        return df

    df = df_pred.copy()
    _require_prediction_columns(df, task_label=profile.task_label)
    if "correct_bool" not in df.columns:
        df["correct_bool"] = df["performance"].astype(bool)
    df[profile.pred_col] = df["pR"]
    df["p_model_correct"] = _correct_prob_pandas(df, profile)
    for source_col in profile.delay_sources:
        if source_col in df.columns:
            df["delay"] = pd.to_numeric(df[source_col], errors="coerce")
            break
    return df


def prepare_right_by_regressor_simple(
    trial_df,
    *,
    profile: TaskPlotProfile,
    regressor_col: str,
    xlabel: str | None = None,
    n_bins: int = 10,
):
    return prepare_simple_regressor_curve(
        trial_df,
        regressor_col=regressor_col,
        pred_col=profile.pred_col,
        response_mode=profile.response_mode,
        baseline=profile.baseline,
        ylabel=p_right_label(),
        xlabel=xlabel,
        n_bins=n_bins,
    )


def _format_delay_tick(value: float) -> str:
    value = float(value)
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _delay_ticks_from_df(df: pd.DataFrame, *, delay_col: str = "delay") -> tuple[list[float], list[str]]:
    if delay_col not in df.columns:
        return [], []
    ticks = sorted(pd.to_numeric(df[delay_col], errors="coerce").dropna().astype(float).unique())
    return ticks, [_format_delay_tick(value) for value in ticks]


def _signed_delay_order_and_labels(
    df: pd.DataFrame,
    profile: TaskPlotProfile,
) -> tuple[list[str], list[str]]:
    values = [str(value) for value in df["_signed_delay_cat"].dropna().unique()]
    preferred = [value for value in profile.signed_delay_order if value in set(values)]
    extras = sorted(
        (value for value in values if value not in set(preferred)),
        key=lambda value: float(value),
    )
    order = preferred + extras
    label_map = dict(zip(profile.signed_delay_order, profile.signed_delay_labels, strict=False))
    return order, [label_map.get(value, _format_delay_tick(float(value))) for value in order]


def _ild_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, list[float], list[str]]:
    out = df.copy()
    plot_x_col = "_plot_ild"
    out[plot_x_col] = pd.to_numeric(out["ILD"], errors="coerce")
    out[plot_x_col] = np.where(
        np.isclose(out[plot_x_col], -70.0),
        -16.0,
        np.where(np.isclose(out[plot_x_col], 70.0), 16.0, out[plot_x_col]),
    )
    ticks = sorted(pd.to_numeric(out[plot_x_col], errors="coerce").dropna().unique())
    allowed = {-16.0, -8.0, 0.0, 8.0, 16.0}
    tick_labels: list[str] = []
    for tick in ticks:
        value = float(tick)
        if any(np.isclose(value, allowed_value) for allowed_value in allowed):
            if np.isclose(value, 0.0):
                tick_labels.append("0")
            elif value > 0:
                tick_labels.append(f"+{int(round(value))}")
            else:
                tick_labels.append(str(int(round(value))))
        else:
            tick_labels.append("")
    return out, plot_x_col, "ILD (dB)", ticks, tick_labels


def _panel(
    df_pd: pd.DataFrame,
    *,
    profile: TaskPlotProfile,
    reg_bin_labels: list,
    x_col: str,
    xlabel: str,
    regressor_col: str,
    subgroup_col: str | None = None,
    subgroup_value=None,
    meta_extra: dict | None = None,
) -> dict:
    panel_df = df_pd if subgroup_col is None else df_pd[df_pd[subgroup_col] == subgroup_value].copy()
    meta = {
        "xlabel": xlabel,
        "ylabel": p_right_label(),
        "legend_title": display_regressor_name(regressor_col),
        "baseline": profile.baseline,
        "x_col": x_col,
        "fit_x_col": x_col,
    }
    if meta_extra:
        meta.update(meta_extra)

    subject_df = panel_df[
        panel_df["_reg_bin"].notna()
        & panel_df[x_col].notna()
        & panel_df["_reg_bin"].isin(reg_bin_labels)
    ].copy()
    if subject_df.empty:
        subject_summary = pd.DataFrame()
    else:
        subject_summary = (
            subject_df.groupby(["_reg_bin", "subject", x_col], observed=True)
            .agg(
                data_mean=("_response_right", "mean"),
                model_mean=(profile.pred_col, "mean"),
                n_trials=("_response_right", "count"),
            )
            .reset_index()
        )

    return {
        "summary": summarize_grouped_panel(
            df_pd,
            line_group_col="_reg_bin",
            x_col=x_col,
            subject_col="subject",
            data_col="_response_right",
            model_col=profile.pred_col,
            line_order=reg_bin_labels,
            subgroup_col=subgroup_col,
            subgroup_value=subgroup_value,
        ),
        "subject_summary": subject_summary,
        "meta": meta,
    }


def _append_subgroup_panels(
    panels: list[dict],
    df_pd: pd.DataFrame,
    *,
    profile: TaskPlotProfile,
    reg_bin_labels: list,
    x_col: str,
    xlabel: str,
    regressor_col: str,
    meta_extra: dict | None = None,
) -> None:
    for subgroup_col in ("condition", "experiment"):
        if subgroup_col not in df_pd.columns:
            continue
        for value in sorted(df_pd[subgroup_col].dropna().unique()):
            panels.append(
                _panel(
                    df_pd,
                    profile=profile,
                    reg_bin_labels=reg_bin_labels,
                    x_col=x_col,
                    xlabel=xlabel,
                    regressor_col=regressor_col,
                    subgroup_col=subgroup_col,
                    subgroup_value=value,
                    meta_extra=meta_extra,
                )
            )


def prepare_binned_accuracy_figure(
    trial_df,
    *,
    profile: TaskPlotProfile,
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

    df_pd = attach_response_right_column(df_pd, response_mode=profile.response_mode)
    if profile.binned_axis == "delay":
        df_pd = attach_signed_delay_columns(df_pd)
    if df_pd.empty:
        return None, None

    if x_col is not None:
        if x_col not in df_pd.columns:
            return None, None
        df_pd[x_col] = pd.to_numeric(df_pd[x_col], errors="coerce")
        plot_xlabel = xlabel or display_regressor_name(x_col)
        panels = [
            _panel(
                df_pd,
                profile=profile,
                reg_bin_labels=reg_bin_labels,
                x_col=x_col,
                xlabel=plot_xlabel,
                regressor_col=regressor_col,
            )
        ]
        _append_subgroup_panels(
            panels,
            df_pd,
            profile=profile,
            reg_bin_labels=reg_bin_labels,
            x_col=x_col,
            xlabel=plot_xlabel,
            regressor_col=regressor_col,
        )
        return panels, display_regressor_name(regressor_col)

    if profile.binned_axis == "ild":
        df_pd, plot_x_col, plot_xlabel, ticks, tick_labels = _ild_axis(df_pd)
        meta_extra = {"xticks": ticks, "x_tick_labels": tick_labels}
        panels = [
            _panel(
                df_pd,
                profile=profile,
                reg_bin_labels=reg_bin_labels,
                x_col=plot_x_col,
                xlabel=plot_xlabel,
                regressor_col=regressor_col,
                meta_extra=meta_extra,
            )
        ]
        _append_subgroup_panels(
            panels,
            df_pd,
            profile=profile,
            reg_bin_labels=reg_bin_labels,
            x_col=plot_x_col,
            xlabel=plot_xlabel,
            regressor_col=regressor_col,
            meta_extra=meta_extra,
        )
        return panels, display_regressor_name(regressor_col)

    delay_ticks, delay_tick_labels = _delay_ticks_from_df(df_pd)
    panels = [
        _panel(
            df_pd,
            profile=profile,
            reg_bin_labels=reg_bin_labels,
            x_col="delay",
            xlabel="Delay",
            regressor_col=regressor_col,
            meta_extra={"xticks": delay_ticks, "x_tick_labels": delay_tick_labels},
        )
    ]

    if "_signed_delay_cat" in df_pd.columns and df_pd["_signed_delay_cat"].notna().any():
        x_order, tick_labels = _signed_delay_order_and_labels(df_pd, profile)
        code_col = "_signed_delay_code"
        code_map = {value: idx for idx, value in enumerate(x_order)}
        df_pd[code_col] = df_pd["_signed_delay_cat"].astype(str).map(code_map)
        signed_panel = _panel(
            df_pd,
            profile=profile,
            reg_bin_labels=reg_bin_labels,
            x_col="_signed_delay_cat",
            xlabel="Signed delay",
            regressor_col=regressor_col,
            meta_extra={
                "x_order": x_order,
                "x_tick_labels": tick_labels,
                "categorical_x": True,
                "fit_x_col": code_col,
            },
        )
        if not signed_panel["summary"].empty:
            signed_panel["summary"][code_col] = (
                signed_panel["summary"]["_signed_delay_cat"].astype(str).map(code_map)
            )
        if not signed_panel["subject_summary"].empty:
            signed_panel["subject_summary"][code_col] = (
                signed_panel["subject_summary"]["_signed_delay_cat"].astype(str).map(code_map)
            )
        panels.append(signed_panel)

    _append_subgroup_panels(
        panels,
        df_pd,
        profile=profile,
        reg_bin_labels=reg_bin_labels,
        x_col="delay",
        xlabel="Delay",
        regressor_col=regressor_col,
        meta_extra={"xticks": delay_ticks, "x_tick_labels": delay_tick_labels, "x_col": "delay"},
    )
    return panels, display_regressor_name(regressor_col)


def _grouped_regressor_summary(
    df_pd: pd.DataFrame,
    *,
    profile: TaskPlotProfile,
    regressor_col: str,
    bin_order: list,
    group_col: str | None,
    group_order: Sequence | None,
) -> tuple[pd.DataFrame, str, Sequence, list, str]:
    resolved_group_col, resolved_group_order = resolve_grouping(
        df_pd,
        group_col=group_col,
        group_order=group_order,
    )

    if resolved_group_col is None and profile.right_group_axis == "ild":
        line_order = sorted(df_pd["ILD"].dropna().unique().tolist())
        summary = summarize_grouped_panel(
            df_pd,
            line_group_col="ILD",
            x_col="_reg_bin",
            subject_col="subject",
            data_col="_response_right",
            model_col=profile.pred_col,
            line_order=line_order,
            x_order=bin_order,
        )
        return summary, "ILD", line_order, [], "Signed ILD"

    if resolved_group_col is None:
        line_order, line_labels = _signed_delay_order_and_labels(df_pd, profile)
        summary = summarize_grouped_panel(
            df_pd,
            line_group_col="_signed_delay_cat",
            x_col="_reg_bin",
            subject_col="subject",
            data_col="_response_right",
            model_col=profile.pred_col,
            line_order=line_order,
            x_order=bin_order,
        )
        return summary, "_signed_delay_cat", line_order, line_labels, "Signed delay"

    df_grouped = df_pd[df_pd[resolved_group_col].notna()].copy()
    df_grouped = df_grouped[df_grouped[resolved_group_col].isin(resolved_group_order)].copy()
    subj = (
        df_grouped.groupby(["subject", resolved_group_col, "_reg_bin"], observed=True)
        .agg(
            data_mean=("_response_right", "mean"),
            model_mean=(profile.pred_col, "mean"),
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
    return summary, resolved_group_col, resolved_group_order, [], resolved_group_col


def prepare_right_by_regressor(
    trial_df,
    *,
    profile: TaskPlotProfile,
    regressor_col: str,
    xlabel: str | None = None,
    n_bins: int = 10,
    group_col: str | None = None,
    group_order: Sequence | None = None,
):
    df_pd = to_pandas_df(trial_df)
    required = {regressor_col, "response", profile.pred_col, "subject"}
    if profile.right_group_axis == "ild":
        required.add("ILD")
    if not required.issubset(df_pd.columns):
        return None, None

    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd[profile.pred_col] = pd.to_numeric(df_pd[profile.pred_col], errors="coerce")
    df_pd = attach_response_right_column(df_pd, response_mode=profile.response_mode)
    if profile.right_group_axis == "ild":
        df_pd["ILD"] = pd.to_numeric(df_pd["ILD"], errors="coerce")
        finite_mask = np.isfinite(df_pd["ILD"])
    else:
        df_pd = attach_signed_delay_columns(df_pd)
        finite_mask = df_pd["_signed_delay_cat"].notna()

    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd[profile.pred_col])
        & np.isfinite(df_pd["_response_right"])
        & finite_mask
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

    summary, line_group_col, line_order, line_labels, legend_title = _grouped_regressor_summary(
        df_pd,
        profile=profile,
        regressor_col=regressor_col,
        bin_order=bin_order,
        group_col=group_col,
        group_order=group_order,
    )
    if summary.empty:
        return None, None

    summary = summary.merge(bin_centers, on="_reg_bin", how="left")
    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": p_right_label(),
        "legend_title": legend_title,
        "baseline": profile.baseline,
        "line_group_col": line_group_col,
        "line_order": line_order,
        "legend_outside": True,
    }
    if line_labels:
        meta["line_labels"] = line_labels
    return summary, meta
