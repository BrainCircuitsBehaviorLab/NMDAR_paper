from __future__ import annotations

from collections.abc import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import ttest_1samp
from src.process.common import (
    PreparedWeightFamilyPlot,
    attach_repeat_choice_evidence,
    attach_quantile_bin_column,
    attach_total_fitted_evidence,
    display_regressor_name,
    fit_lapse_logistic_by_group,
    fit_lapse_logistic_by_subject_group,
    padded_numeric_limits,
    summarize_grouped_panel,
    summarize_simple_curve,
    to_pandas_df,
)
from glmhmmt.plots.common import (
    custom_boxplot,
    resolve_single_axis as resolve_glmhmmt_single_axis,
)


BOXPLOT_STYLE = dict(
    fill=False,
    boxprops={"color": "0.5"},
    whiskerprops={"color": "0.5"},
    medianprops={"linewidth": 2},
    showfliers=False,
    showcaps=False,
)
boxplot_STYLE = BOXPLOT_STYLE


def _significance_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue) or pvalue >= 0.05:
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    return "*"


def apply_axis_style(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    title: str | None = None,
    grid: bool = False,
):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    if title is not None:
        ax.set_title(title)

    if grid:
        ax.grid(True)


def fig_size(n_cols=1, ratio=None):
    """
    Get figure size for A4 page with n_cols columns and specified ratio (width/height).
    :param n_cols: Number of columns (0 for full page)
    :param ratio: Width/height ratio (None for default)
    :return:
    """

    if ratio is None:
        default_figsize = np.array(plt.rcParams["figure.figsize"])
        default_ratio = default_figsize[0] / default_figsize[1]
        ratio = default_ratio  # 4:3

    # All measurements are in inches
    A4_size = np.array((8.27, 11.69))  # A4 measurements
    margins = 2  # On both dimension
    size = A4_size - margins  # Effective size after margins removal (2 per dimension)
    width = size[0]
    height = size[1]

    # Full page (minus margins)
    if n_cols == 0:
        # Full A4 minus margins
        figsize = (width, height)
        if ratio == 1:  # Square
            figsize = (size[0], size[0])
        return figsize

    else:
        fig_width = width / n_cols
        fig_height = fig_width / ratio
        figsize = (fig_width, fig_height)
        return figsize


def pick_existing_column(df_like, candidates: Sequence[str | None]) -> str | None:
    columns = set(getattr(df_like, "columns", []))
    for candidate in candidates:
        if candidate and candidate in columns:
            return candidate
    return None


def _coerce_correct_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(float).to_numpy() > 0
    return (
        series.astype(str)
        .str.lower()
        .isin(["1", "true", "correct", "hit", "yes"])
        .to_numpy()
    )


def _difficulty_values(pdf: pd.DataFrame, task_name: str) -> tuple[pd.Series, str]:
    def numeric(column: str, *, abs_value: bool = False) -> pd.Series:
        values = pd.to_numeric(pdf[column], errors="coerce")
        if abs_value:
            values = values.abs()
        return values

    if task_name == "2AFC" and "ILD" in pdf.columns:
        return numeric("ILD", abs_value=True), "Difficulty (|ILD| dB)"

    if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
        for column in ["delay", "delays"]:
            if column in pdf.columns:
                return numeric(column), "Difficulty (delay, s)"

    if task_name == "MCDR":
        if "stimd_c" in pdf.columns:
            return pdf["stimd_c"].astype(str), "Difficulty"
        if "stimd_n" in pdf.columns:
            return numeric("stimd_n"), "Difficulty"

    for column in ["difficulty", "stimd_n", "delay", "delays", "ILD", "stimulus", "stim"]:
        if column in pdf.columns:
            return numeric(column, abs_value=column in {"ILD", "stimulus", "stim"}), "Difficulty"

    raise ValueError("No difficulty-like column found for this task.")


def _difficulty_labels(difficulty: pd.Series) -> pd.Series:
    difficulty = pd.Series(difficulty).reset_index(drop=True)
    numeric = pd.to_numeric(difficulty, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float).map(lambda value: f"{value:g}")
    return difficulty.astype(str)


def build_session_trial_outcomes_data(
    plot_df,
    *,
    task_name: str,
    subject,
    session,
    adapter=None,
) -> tuple[pd.DataFrame, str, str]:
    behavioral_cols = getattr(adapter, "behavioral_cols", {}) or {}
    session_col = pick_existing_column(
        plot_df,
        ["session", getattr(adapter, "session_col", None), behavioral_cols.get("session"), "Session"],
    )
    trial_col = pick_existing_column(
        plot_df,
        ["trial", "trial_idx", behavioral_cols.get("trial"), behavioral_cols.get("trial_idx"), "Trial"],
    )
    correct_col = pick_existing_column(
        plot_df,
        ["correct_bool", "performance", behavioral_cols.get("performance"), "Hit", "hit"],
    )
    if session_col is None or trial_col is None or correct_col is None:
        raise ValueError("Session plot needs session, trial, and correctness columns.")

    session_df = (
        plot_df
        .filter(
            (pl.col("subject").cast(pl.Utf8) == str(subject))
            & (pl.col(session_col).cast(pl.Utf8) == str(session))
        )
        .sort(trial_col)
    )
    if session_df.height == 0:
        raise ValueError("No trials for the selected subject/session.")

    pdf = session_df.to_pandas()
    x = np.arange(len(pdf), dtype=float)
    difficulty, ylabel = _difficulty_values(pdf, task_name)
    difficulty = pd.Series(difficulty)
    valid = difficulty.notna().to_numpy()
    if not valid.any():
        raise ValueError("No valid difficulty values for this session.")

    correct = _coerce_correct_values(pdf[correct_col])
    x = x[valid]
    difficulty_label = _difficulty_labels(difficulty[valid]).to_numpy()
    correct = correct[valid]

    edges = np.empty(len(x) + 1, dtype=float)
    if len(x) == 1:
        edges[:] = [x[0] - 0.5, x[0] + 0.5]
    else:
        midpoints = (x[:-1] + x[1:]) / 2.0
        edges[1:-1] = midpoints
        edges[0] = x[0] - (midpoints[0] - x[0])
        edges[-1] = x[-1] + (x[-1] - midpoints[-1])

    return (
        pd.DataFrame(
            {
                "trial_x": x,
                "trial_left": edges[:-1],
                "trial_right": edges[1:],
                "difficulty_label": difficulty_label,
                "correct": correct.astype(bool),
            }
        ),
        "Trial number (within session)",
        ylabel,
    )


def plot_session_trial_outcomes(
    data: pd.DataFrame,
    *,
    xlabel: str = "Trial number (within session)",
    easy_difficulty: str | None = None,
    trial_tick_step: int = 20,
    ax: plt.Axes | None = None,
    figsize=(6, 3),
    dpi=150,
):
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=False)
    if data.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    df = data.copy().reset_index(drop=True)
    has_easy_selection = easy_difficulty not in {None, "None"}
    df["is_easy"] = (
        has_easy_selection
        & (df["difficulty_label"].astype(str) == str(easy_difficulty))
    )
    df["color"] = [
        "#006d2c" if correct and easy else
        "#2ca02c" if correct else
        "#7f0000" if easy else
        "#d62728"
        for correct, easy in zip(df["correct"], df["is_easy"], strict=False)
    ]

    fig.set_dpi(dpi)
    for _, row in df.iterrows():
        ax.add_patch(
            mpatches.Rectangle(
                (row["trial_left"], -0.38),
                row["trial_right"] - row["trial_left"],
                0.76,
                facecolor=row["color"],
                alpha=0.75,
                linewidth=0,
            )
        )

    ax.set_xlabel(xlabel)
    ax.set_xlim(df["trial_left"].iloc[0], df["trial_right"].iloc[-1])
    last_trial = int(df["trial_x"].max())
    ax.set_xticks(list(range(0, last_trial + 1, int(trial_tick_step))))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel("")

    legend_handles = [
        mpatches.Patch(color="#2ca02c", alpha=0.75, label="Correct"),
        mpatches.Patch(color="#d62728", alpha=0.75, label="Incorrect"),
    ]
    if has_easy_selection:
        legend_handles.extend(
            [
                mpatches.Patch(color="#006d2c", alpha=0.75, label=f"Easy correct ({easy_difficulty})"),
                mpatches.Patch(color="#7f0000", alpha=0.75, label=f"Easy incorrect ({easy_difficulty})"),
            ]
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(legend_handles),
        frameon=False,
    )
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return fig, ax


def build_session_repetition_data(
    plot_df,
    *,
    subject,
    session,
    window: int,
    adapter=None,
) -> pd.DataFrame:
    behavioral_cols = getattr(adapter, "behavioral_cols", {}) or {}
    session_col = pick_existing_column(
        plot_df,
        ["session", getattr(adapter, "session_col", None), behavioral_cols.get("session"), "Session"],
    )
    trial_col = pick_existing_column(
        plot_df,
        ["trial_idx", "trial", behavioral_cols.get("trial_idx"), behavioral_cols.get("trial"), "Trial"],
    )
    response_col = pick_existing_column(
        plot_df,
        ["response", "choices", "choice", behavioral_cols.get("response"), "Choice"],
    )
    stimulus_col = pick_existing_column(
        plot_df,
        ["stimulus", "stim", "side", behavioral_cols.get("stimulus"), "Side"],
    )
    if session_col is None or trial_col is None or response_col is None or stimulus_col is None:
        raise ValueError("Example repetition plot needs session, trial, response, and stimulus columns.")

    session_df = (
        plot_df
        .filter(
            (pl.col("subject").cast(pl.Utf8) == str(subject))
            & (pl.col(session_col).cast(pl.Utf8) == str(session))
        )
        .sort(trial_col)
    )
    if session_df.height == 0:
        raise ValueError("No trials for the selected subject/session.")

    data = (
        session_df
        .select([trial_col, response_col, stimulus_col])
        .to_pandas()
        .rename(columns={trial_col: "trial", response_col: "response", stimulus_col: "stimulus"})
        .reset_index(drop=True)
    )
    data["trial_x"] = np.arange(len(data))
    data["previous_response"] = data["response"].shift(1)
    data["previous_stimulus"] = data["stimulus"].shift(1)
    data["response_repeat"] = data["response"].eq(data["previous_response"]).fillna(False)
    data["stimulus_repeat"] = data["stimulus"].eq(data["previous_stimulus"]).fillna(False)
    window = int(window)
    response_repeat = data["response_repeat"].astype(float)
    stimulus_repeat = data["stimulus_repeat"].astype(float)
    data["repeat_window_n"] = response_repeat.rolling(window, min_periods=1).count()
    data["response_repeat_window_count"] = response_repeat.rolling(window, min_periods=1).sum()
    data["stimulus_repeat_window_count"] = stimulus_repeat.rolling(window, min_periods=1).sum()
    data["response_repeat_window_fraction"] = (
        data["response_repeat_window_count"] / data["repeat_window_n"]
    )
    data["stimulus_repeat_window_fraction"] = (
        data["stimulus_repeat_window_count"] / data["repeat_window_n"]
    )
    return data


def plot_session_response_raster(
    data: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    figsize=(6, 1.6),
    dpi=150,
):
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=False)
    fig.set_dpi(dpi)
    response_labels = sorted(data["response"].dropna().unique(), key=lambda value: str(value))
    response_y = {value: index for index, value in enumerate(response_labels)}
    colors = sns.color_palette("tab10", n_colors=max(len(response_labels), 1))

    for idx, response in enumerate(response_labels):
        mask = data["response"].eq(response)
        ax.scatter(
            data.loc[mask, "trial_x"],
            [response_y[response]] * int(mask.sum()),
            s=10,
            color=colors[idx],
            label=str(response),
        )

    ax.set_xlabel("Trial number (within session)")
    ax.set_ylabel("Response")
    ax.set_yticks(list(response_y.values()))
    ax.set_yticklabels([str(label) for label in response_labels])
    ax.set_xlim(-0.5, len(data) - 0.5)
    if len(response_labels) > 1:
        ax.legend(title="Response", frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_session_repetition_running_count(
    data: pd.DataFrame,
    *,
    window: int,
    ax: plt.Axes | None = None,
    figsize=None,
    dpi=150,
):
    fig, ax = resolve_single_axis(
        ax=ax,
        figsize=fig_size(1, 3) if figsize is None else figsize,
        constrained_layout=False,
    )
    fig.set_dpi(dpi)
    ax.plot(
        data["trial_x"],
        data["response_repeat_window_fraction"],
        color="tab:brown",
        linewidth=1.5,
        label="Response",
    )
    ax.plot(
        data["trial_x"],
        data["stimulus_repeat_window_fraction"],
        color="tab:blue",
        linewidth=1.5,
        label="Stimulus",
    )
    ax.set_xlabel("Trial number (within session)")
    ax.set_ylabel("Repetition fraction")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.legend(frameon=False, loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax


def _drug_label_expr():
    drug_number = pl.col("drug").cast(pl.Float64, strict=False)
    drug_text = pl.col("drug").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return (
        pl.when(drug_number == 0)
        .then(pl.lit("Saline"))
        .when(drug_number == 1)
        .then(pl.lit("Drug"))
        .when(drug_text == "saline")
        .then(pl.lit("Saline"))
        .when(drug_text.is_in(["drug", "nr2b"]))
        .then(pl.lit("Drug"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("drug_label")
    )


def _overlapping_rolling_binomial_variance(n_windows, window: int, p: float) -> float | None:
    if n_windows is None or n_windows < 2:
        return None
    n_windows = int(n_windows)
    sigma2 = p * (1 - p)
    gamma0 = sigma2 / window
    max_lag = min(window - 1, n_windows - 1)
    lag_sum = (
        max_lag * n_windows * window
        - (n_windows + window) * max_lag * (max_lag + 1) / 2
        + max_lag * (max_lag + 1) * (2 * max_lag + 1) / 6
    )
    mean_variance = (
        n_windows * gamma0
        + 2 * sigma2 * lag_sum / (window ** 2)
    ) / (n_windows ** 2)
    return n_windows * (gamma0 - mean_variance) / (n_windows - 1)


def _repetition_variance_for_task(get_adapter, task_name: str, task_label: str, drug_col: str, window: int):
    adapter = get_adapter(task_name)
    df = adapter.subject_filter(adapter.read_dataset())
    if drug_col not in df.columns:
        return None

    subject_col = pick_existing_column(df, ["subject", "Subject"])
    session_col = pick_existing_column(df, ["session", "Session"])
    trial_col = pick_existing_column(df, ["trial_idx", "Trial", "trial"])
    response_col = pick_existing_column(df, ["response", "Choice", "choice", "choices"])
    stimulus_col = pick_existing_column(df, ["stimulus", "Side", "side", "stim"])
    if any(col is None for col in [subject_col, session_col, trial_col, response_col, stimulus_col]):
        return None

    group_cols = ["subject", "drug_label", "session"]
    return (
        df
        .select(
            pl.col(subject_col).alias("subject"),
            pl.col(session_col).alias("session"),
            pl.col(trial_col).alias("trial_idx"),
            pl.col(response_col).alias("response"),
            pl.col(stimulus_col).alias("stimulus"),
            pl.col(drug_col).alias("drug"),
        )
        .with_columns(_drug_label_expr())
        .drop_nulls(["subject", "session", "trial_idx", "response", "stimulus", "drug_label"])
        .sort(["subject", "drug_label", "session", "trial_idx"])
        .with_columns(pl.col("trial_idx").cum_count().over(group_cols).alias("trial_position"))
        .filter(pl.col("trial_position") > 10)
        .with_columns(
            pl.col("response").shift().over(group_cols).alias("previous_response"),
            pl.col("stimulus").shift().over(group_cols).alias("previous_stimulus"),
        )
        .drop_nulls(["previous_response", "previous_stimulus"])
        .with_columns(
            (pl.col("response") == pl.col("previous_response")).alias("response_repeat"),
            (pl.col("stimulus") == pl.col("previous_stimulus")).alias("stimulus_repeat"),
        )
        .with_columns(
            pl.col("response_repeat")
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=window)
            .over(group_cols)
            .alias("response_repeat_window_fraction"),
            pl.col("stimulus_repeat")
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=window)
            .over(group_cols)
            .alias("stimulus_repeat_window_fraction"),
        )
        .group_by(group_cols)
        .agg(
            pl.col("response_repeat_window_fraction").var().alias("response_repeat_variance"),
            pl.col("stimulus_repeat_window_fraction").var().alias("stimulus_repeat_variance"),
            pl.col("stimulus_repeat_window_fraction").is_not_null().sum().alias("n_windows"),
        )
        .with_columns(
            pl.lit(task_name).alias("task"),
            pl.lit(task_label).alias("task_label"),
        )
        .with_columns(
            pl.struct(["task_label", "n_windows"])
            .map_elements(
                lambda row: _overlapping_rolling_binomial_variance(
                    row["n_windows"],
                    window,
                    1 / 3 if row["task_label"] == "MCDR" else 0.5,
                ),
                return_dtype=pl.Float64,
            )
            .alias("stimulus_repeat_binomial_variance")
        )
    )


def build_repetition_variance_by_drug_task(
    get_adapter,
    *,
    window: int,
    task_specs: Sequence[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    if task_specs is None:
        task_specs = [
            ("2AFC_DRUG", "2AFC", "Drug"),
            ("2ADC_DRUG", "2ADC", "drug_code"),
            ("MCDR", "MCDR", "condition"),
        ]

    variance_frames = [
        frame
        for frame in (
            _repetition_variance_for_task(get_adapter, task_name, task_label, drug_col, int(window))
            for task_name, task_label, drug_col in task_specs
        )
        if frame is not None and not frame.is_empty()
    ]
    if variance_frames:
        by_session = pl.concat(variance_frames, how="diagonal_relaxed")
    else:
        by_session = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "drug_label": pl.Utf8,
                "session": pl.Int64,
                "response_repeat_variance": pl.Float64,
                "stimulus_repeat_variance": pl.Float64,
                "n_windows": pl.UInt32,
                "stimulus_repeat_binomial_variance": pl.Float64,
                "task": pl.Utf8,
                "task_label": pl.Utf8,
            }
        )

    by_task = (
        by_session
        .group_by(["task", "task_label", "subject", "drug_label"])
        .agg(
            pl.col("response_repeat_variance").mean(),
            pl.col("stimulus_repeat_variance").mean(),
            pl.col("stimulus_repeat_binomial_variance").mean(),
        )
    )
    long = (
        by_task
        .unpivot(
            index=["task", "task_label", "subject", "drug_label"],
            on=["response_repeat_variance", "stimulus_repeat_variance"],
            variable_name="signal",
            value_name="variance",
        )
        .with_columns(
            pl.col("signal").replace(
                {
                    "response_repeat_variance": "Repetition",
                    "stimulus_repeat_variance": "Stimulus",
                }
            )
        )
        .drop_nulls("variance")
        .to_pandas()
    )
    baselines = dict(
        by_task
        .group_by("task_label")
        .agg(pl.col("stimulus_repeat_binomial_variance").mean().alias("variance"))
        .to_pandas()
        .set_index("task_label")["variance"]
    )
    return long, baselines


def plot_drug_repetition_variance_by_task(
    data: pd.DataFrame,
    baselines: dict,
    *,
    task_order: Sequence[str] = ("2AFC", "2ADC", "MCDR"),
    signal_order: Sequence[str] = ("Repetition", "Stimulus"),
    drug_order: Sequence[str] = ("Saline", "Drug"),
    ax: Sequence[plt.Axes] | None = None,
    figsize=None,
):
    from math import isfinite
    from scipy.stats import ttest_1samp
    from statannotations.Annotator import Annotator

    fig, axes = resolve_axes(
        ax,
        n_axes=len(task_order),
        figsize=fig_size(1, 3) if figsize is None else figsize,
        squeeze=False,
        sharey=True,
    )
    legend_handles = []
    legend_labels = []

    for axis, task_label in zip(axes, task_order, strict=False):
        task_data = data[data["task_label"] == task_label]
        if task_data.empty:
            axis.axis("off")
            axis.set_title(task_label)
            continue

        sns.boxplot(
            data=task_data,
            x="signal",
            y="variance",
            order=list(signal_order),
            hue="drug_label",
            hue_order=list(drug_order),
            palette={"Saline": "tab:gray", "Drug": "tab:pink"},
            ax=axis,
            **BOXPLOT_STYLE,
        )
        baseline = baselines.get(task_label)
        if baseline is not None and np.isfinite(baseline):
            axis.axhline(
                baseline,
                color="tab:blue",
                linestyle="--",
                label="Stimulus binomial",
            )
        axis.set_title(task_label)
        axis.set_xlabel("")
        axis.set_ylabel("Variance of running fraction" if axis is axes[0] else "")
        if not legend_handles:
            handles, labels = axis.get_legend_handles_labels()
            legend = dict(zip(labels, handles, strict=False))
            legend_handles = list(legend.values())
            legend_labels = list(legend.keys())
        if axis.get_legend() is not None:
            axis.get_legend().remove()

        if baseline is None or not np.isfinite(baseline):
            continue
        pairs = []
        pvalues = []
        for signal in signal_order:
            for drug_label in drug_order:
                values = task_data.loc[
                    (task_data["signal"] == signal)
                    & (task_data["drug_label"] == drug_label),
                    "variance",
                ].dropna()
                if len(values) < 2:
                    continue
                pvalue = ttest_1samp(values, popmean=baseline).pvalue
                if isfinite(pvalue):
                    pairs.append(((signal, drug_label), (signal, drug_label)))
                    pvalues.append(pvalue)
        if pairs:
            annotator = Annotator(
                axis,
                pairs,
                data=task_data,
                x="signal",
                y="variance",
                hue="drug_label",
                order=list(signal_order),
                hue_order=list(drug_order),
            )
            annotator.configure(line_width=0, text_format="star", verbose=0)
            annotator.set_pvalues_and_annotate(pvalues)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, -0.15),
        )
    sns.despine(fig=fig)
    return fig, axes


def animal_chunk_histogram(chunk_lengths: pd.DataFrame, *, group_cols, stat: str) -> pd.DataFrame:
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


def _transition_chunks_for_sequence(plot_df, task_name: str, sequence_col: str, sequence_label: str):
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
    task_labels = {"2AFC": "2AFC", "2AFC_delay": "2ADC", "2ADC": "2ADC", "MCDR": "MCDR"}
    chunks["task"] = task_name
    chunks["task_label"] = task_labels.get(task_name, task_name)
    chunks["sequence"] = sequence_label
    repeat_probability = {
        "task": task_name,
        "task_label": task_labels.get(task_name, task_name),
        "sequence": sequence_label,
        "p_repeat": (trials["transition"] == "repeating").mean(),
    }
    return chunks, repeat_probability


def build_transition_chunks_by_task(plot_payloads: dict, task_names: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_frames = []
    repeat_probability_rows = []
    for task_name in task_names:
        plot_df = plot_payloads[task_name]["plot_df"]
        available_columns = set(plot_df.columns)
        for sequence_col, sequence_label in [("response", "Choices"), ("stimulus", "Stimulus")]:
            if {"subject", "session", "trial_idx", sequence_col}.issubset(available_columns):
                chunks, repeat_probability = _transition_chunks_for_sequence(
                    plot_df,
                    task_name,
                    sequence_col,
                    sequence_label,
                )
                chunk_frames.append(chunks)
                repeat_probability_rows.append(repeat_probability)

    return (
        pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame(),
        pd.DataFrame(repeat_probability_rows),
    )


def two_afc_repeat_alternate_trials(plot_df) -> pd.DataFrame:
    df = plot_df.to_pandas() if hasattr(plot_df, "to_pandas") else pd.DataFrame(plot_df)
    subject_col = pick_existing_column(df, ["subject"])
    session_col = pick_existing_column(df, ["session", "Session"])
    trial_col = pick_existing_column(df, ["trial_idx", "trial", "Trial"])
    choice_col = pick_existing_column(df, ["response", "choice", "choices", "Choice"])
    correct_col = pick_existing_column(df, ["correct_bool", "performance", "Hit", "hit"])
    if any(col is None for col in [subject_col, session_col, trial_col, choice_col, correct_col]):
        return pd.DataFrame()

    out = df[[subject_col, session_col, trial_col, choice_col, correct_col]].copy()
    out.columns = ["subject", "session", "trial", "choice", "correct"]
    out["correct"] = _coerce_correct_values(out["correct"]).astype(float)
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out = out.dropna(subset=["subject", "session", "trial", "choice", "correct"])
    out = out.sort_values(["subject", "session", "trial"])
    out["previous_choice"] = out.groupby(["subject", "session"], observed=True)["choice"].shift(1)
    out = out.dropna(subset=["previous_choice"]).copy()
    out["transition"] = [
        "Repeating" if choice == previous else "Alternating"
        for choice, previous in zip(out["choice"], out["previous_choice"], strict=False)
    ]
    return out


def two_afc_transition_chunk_lengths(plot_df) -> pd.DataFrame:
    trials = two_afc_repeat_alternate_trials(plot_df)
    if trials.empty:
        return trials
    chunks = []
    for (_subject, _session), session_df in trials.groupby(["subject", "session"], observed=True):
        session_df = session_df.copy()
        session_df["chunk"] = (session_df["transition"] != session_df["transition"].shift()).cumsum()
        chunks.append(
            session_df.groupby("chunk", as_index=False, observed=True)
            .agg(
                subject=("subject", "first"),
                session=("session", "first"),
                transition=("transition", "first"),
                chunk_length=("transition", "size"),
            )
        )
    return pd.concat(chunks, ignore_index=True)


def two_afc_session_repeat_alternate_accuracy(plot_df) -> pd.DataFrame:
    trials = two_afc_repeat_alternate_trials(plot_df)
    if trials.empty:
        return trials
    accuracy = (
        trials.groupby(["subject", "session", "transition"], observed=True)["correct"]
        .mean()
        .unstack("transition")
        .reset_index()
    )
    if {"Repeating", "Alternating"}.difference(accuracy.columns):
        return pd.DataFrame()
    return accuracy.rename(
        columns={
            "Repeating": "repeat_accuracy",
            "Alternating": "alternate_accuracy",
        }
    ).dropna(subset=["repeat_accuracy", "alternate_accuracy"])


def plot_prepared_weight_family(
    prepared: PreparedWeightFamilyPlot | None,
    figsize=(4.0, 4.0),
    title: str | None = None,
    ax: plt.Axes | None = None,
    connect_subjects: bool = True,
):
    if prepared is None:
        return None

    df = to_pandas_df(prepared.data)
    if df.empty:
        return None

    required = {"subject", "x_label", "weight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Prepared weight family data must contain 'subject', 'x_label', and 'weight'. "
            f"Missing: {sorted(missing)}."
        )

    df = df.copy()
    df["subject"] = df["subject"].astype(str)
    df["x_label"] = df["x_label"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])
    if df.empty:
        return None

    x_order = list(prepared.x_order) if prepared.x_order is not None else pd.unique(df["x_label"]).tolist()
    df = df[df["x_label"].isin(x_order)].copy()
    if df.empty:
        return None

    if prepared.plot_kind == "line":
        summary = (
            df.groupby("x_label", as_index=False, observed=False)["weight"]
            .mean()
        )
        summary["x_label"] = pd.Categorical(summary["x_label"], categories=x_order, ordered=True)
        summary = summary.sort_values("x_label")
        if summary.empty:
            return None

        positions = np.arange(len(summary))
        fig, ax, created_fig = resolve_glmhmmt_single_axis(ax=ax, figsize=figsize)
        ax.plot(
            positions,
            summary["weight"],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(prepared.xlabel)
        ax.set_ylabel(prepared.ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(summary["x_label"].astype(str).tolist())
        if created_fig:
            fig.tight_layout()
        return fig

    subject_order = pd.unique(df["subject"]).tolist()
    per_feature_values: list[np.ndarray] = []
    subject_lines = np.full((len(subject_order), len(x_order)), np.nan, dtype=float)

    for feature_idx, x_label in enumerate(x_order):
        feature_df = df[df["x_label"] == x_label].copy()
        if feature_df.empty:
            per_feature_values.append(np.asarray([], dtype=float))
            continue
        by_subject = (
            feature_df.groupby("subject", observed=False)["weight"]
            .mean()
            .reindex(subject_order)
        )
        subject_lines[:, feature_idx] = by_subject.to_numpy(dtype=float)
        per_feature_values.append(by_subject.dropna().to_numpy(dtype=float))

    if not any(values.size for values in per_feature_values):
        return None

    fig, ax, created_fig = resolve_glmhmmt_single_axis(ax=ax, figsize=figsize)
    positions = np.arange(1, len(x_order) + 1)
    custom_boxplot(
        ax,
        per_feature_values,
        positions=positions,
        widths=0.55,
        median_colors="tab:blue",
        line_values=subject_lines if connect_subjects else None,
        line_color="#7A7A7A",
        line_alpha=0.15,
        line_linewidth=1.0,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(prepared.xlabel)
    ax.set_ylabel(prepared.ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(x_order))
    if created_fig:
        fig.tight_layout()
    return fig


def make_single_panel_figure(
    *,
    extra_right_legend: bool = False,
    figsize=(3.0, 3.0),
    ax: plt.Axes | None = None,
    **style,
):
    _ = extra_right_legend
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    apply_axis_style(ax, **style)
    return fig, ax


def resolve_single_axis(
    *,
    ax: plt.Axes | None = None,
    figsize=(3.0, 3.0),
    constrained_layout: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained_layout)
    else:
        fig = ax.figure
    return fig, ax


def resolve_axes(
    axes: Sequence[plt.Axes] | None = None,
    *,
    n_axes: int,
    figsize,
    squeeze: bool = False,
    **subplots_kwargs,
):
    if axes is None:
        fig, axes = plt.subplots(
            1,
            n_axes,
            figsize=figsize,
            squeeze=squeeze,
            **subplots_kwargs,
        )
        axes = np.atleast_1d(axes).ravel()
        return fig, axes

    axes = np.asarray(axes, dtype=object).ravel()
    if len(axes) < n_axes:
        raise ValueError(f"Expected at least {n_axes} axes, got {len(axes)}.")
    return axes[0].figure, axes


def plot_mean_over_data(
    df_like,
    *,
    x_col: str,
    y_col: str,
    subject_col: str = "subject",
    x_order: list | None = None,
    x_tick_labels: list | dict | None = None,
    xlabel: str,
    ylabel: str = "Accuracy",
    title: str,
    baseline: float,
    baseline_area: bool = True,
    color: str = "#2b7bba",
    invert_x: bool = False,
    show_baseline_ttest: bool = False,
    ax: plt.Axes | None = None,
    figsize=fig_size(n_cols=3),
    label: str | None = None,
    **style,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    if isinstance(df_like, pd.DataFrame):
        df = df_like.copy()
    elif hasattr(df_like, "to_pandas"):
        df = df_like.to_pandas().copy()
    else:
        df = pd.DataFrame(df_like).copy()

    if x_col not in df.columns:
        raise ValueError(f"Missing x column {x_col!r}.")
    if y_col not in df.columns:
        raise ValueError(f"Missing y column {y_col!r}.")

    df["_y"] = pd.to_numeric(df[y_col], errors="coerce")
    df = df[df[x_col].notna() & df["_y"].notna()].copy()

    if df.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
        ax.axis("off")
        apply_axis_style(ax, **style)
        return ax

    if subject_col in df.columns:
        subject_summary = (
            df.groupby([subject_col, x_col], observed=True)["_y"]
            .mean()
            .reset_index(name="subject_mean")
        )
        summary = (
            subject_summary.groupby(x_col, observed=True)["subject_mean"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )
    else:
        summary = (
            df.groupby(x_col, observed=True)["_y"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )

    if x_order is not None:
        summary = summary[summary[x_col].isin(x_order)].copy()
        if summary.empty:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            else:
                fig = ax.figure
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.axis("off")
            apply_axis_style(ax, **style)
            return ax
        summary[x_col] = pd.Categorical(summary[x_col], categories=x_order, ordered=True)
        summary = summary.sort_values(x_col)
        x = np.arange(len(summary), dtype=float)
        if x_tick_labels is None:
            tick_labels = [str(value) for value in summary[x_col]]
        elif isinstance(x_tick_labels, dict):
            tick_labels = [x_tick_labels.get(value, str(value)) for value in summary[x_col]]
        else:
            label_map = dict(zip(x_order, x_tick_labels, strict=False))
            tick_labels = [label_map.get(value, str(value)) for value in summary[x_col]]
    else:
        summary["_x_numeric"] = pd.to_numeric(summary[x_col], errors="coerce")
        summary = summary.dropna(subset=["_x_numeric"]).sort_values("_x_numeric")
        if summary.empty:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            else:
                fig = ax.figure
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.axis("off")
            apply_axis_style(ax, **style)
            return ax
        x = summary["_x_numeric"].to_numpy(dtype=float)
        tick_labels = []
        for val in x:
            if isinstance(x_tick_labels, dict) and val in x_tick_labels:
                tick_labels.append(x_tick_labels[val])
            elif isinstance(x_tick_labels, dict) and int(val) in x_tick_labels:
                tick_labels.append(x_tick_labels[int(val)])
            elif np.isclose(val, 0.1):
                tick_labels.append("0")
            else:
                tick_labels.append(f"{val:g}")


    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    ax.errorbar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["sem"].to_numpy(dtype=float),
        fmt="o-",
        color=color,
        ecolor=color,
        capsize=0,
        label=label,
    )

    ax.axhline(baseline, color="gray", ls="--")
    if baseline_area:
        ax.axhspan(0.0, baseline, color="gray", alpha=0.1, zorder=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, baseline, 1.0])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:.2f}".rstrip("0").rstrip("."))
    )
    ax.set_xticks(x, labels=tick_labels)

    if show_baseline_ttest:
        for xi, (_, row) in zip(x, summary.iterrows(), strict=False):
            if x_order is not None:
                values = subject_summary.loc[
                    subject_summary[x_col] == row[x_col], "subject_mean"
                ].dropna()
            else:
                values = subject_summary.loc[
                    pd.to_numeric(subject_summary[x_col], errors="coerce").eq(row["_x_numeric"]),
                    "subject_mean",
                ].dropna()
            if len(values) < 2:
                continue
            pvalue = float(ttest_1samp(values.to_numpy(dtype=float), popmean=baseline).pvalue)
            label = _significance_stars(pvalue)
            if not label:
                continue
            y = min(0.97, float(row["mean"]) + float(row["sem"]) + 0.05)
            ax.text(
                xi,
                y,
                label,
                ha="center",
                va="bottom",
                # fontsize=9,
                color="black",
            )

    if invert_x:
        # Only invert if not already inverted (track with a marker on the axis)
        if not getattr(ax, '_x_inverted_marker', False):
            ax.invert_xaxis()
            ax._x_inverted_marker = True

    # Add legend if labels are present
    if label is not None or ax.get_lines():
        ax.legend(frameon=False)

    return fig


COUNTERFACTUAL_PALETTE = {
    "Data": "#1f77b4",
    "Full fitted": "#111111",
    "Fixed bias": "#d55e00",
    "Fixed lapses": "#009e73",
}


def _plot_counterfactual_summary(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str,
    mean_col: str,
    lo_col: str,
    hi_col: str,
    order: Sequence[str],
) -> None:
    """Shared line/error plotting for the RB and lag-match simulation summaries."""
    required = {"scenario", x_col, mean_col, lo_col, hi_col}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        return

    for scenario in order:
        sub = df[df["scenario"] == scenario].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(x_col)
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[mean_col].to_numpy(dtype=float)
        lo = sub[lo_col].to_numpy(dtype=float)
        hi = sub[hi_col].to_numpy(dtype=float)
        color = COUNTERFACTUAL_PALETTE[scenario]
        if scenario == "Data":
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                label=scenario,
                zorder=4,
            )
        else:
            ax.plot(x, y, lw=1.8, marker="o", ms=3, color=color, label=scenario)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0.0)


def plot_action_trace_counterfactual_rb(summary, meta, *, ax: plt.Axes | None = None, figsize=(4.6, 3.4)):
    """Plot repetition bias from empirical data and parameter-fixed simulations."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)
    df = to_pandas_df(summary)
    _plot_counterfactual_summary(
        ax,
        df,
        x_col="_counterfactual_rb_x",
        mean_col="rb_mean",
        lo_col="rb_lo",
        hi_col="rb_hi",
        order=("Data", "Full fitted", "Fixed bias", "Fixed lapses"),
    )
    ax.axhline(float(meta.get("baseline", 0.5)), color="0.5", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel(meta.get("xlabel", "Task variable"))
    ax.set_ylabel("Rep. bias")
    ax.set_ylim(0.0, 1.0)
    if meta.get("xticks") is not None:
        ax.set_xticks(meta["xticks"])
        if meta.get("x_tick_labels") is not None:
            ax.set_xticklabels(meta["x_tick_labels"])
    if meta.get("invert_x", False):
        ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("")
    return fig, ax


def plot_action_trace_counterfactual_lag_match(
    lag_summary,
    meta,
    *,
    ax: plt.Axes | None = None,
    figsize=(4.6, 3.2),
):
    """Plot p(response_t = response_{t-L}) from lag 1 to the selected max lag."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)
    df = to_pandas_df(lag_summary)
    _plot_counterfactual_summary(
        ax,
        df,
        x_col="lag",
        mean_col="lag_match_mean",
        lo_col="lag_match_lo",
        hi_col="lag_match_hi",
        order=("Data", "Full fitted", "Fixed bias", "Fixed lapses"),
    )
    ax.axhline(float(meta.get("baseline", 0.5)), color="0.5", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("History lag")
    ax.set_ylabel(r"$p(\hat r_t = r_{t-L})$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.5, float(meta.get("max_history_lag", 10)) + 0.5)
    ax.set_xticks(np.arange(1, int(meta.get("max_history_lag", 10)) + 1))
    ax.set_title("Lag-match parameter-fixed simulation")
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


def plot_action_trace_counterfactual_subject_scatter(
    subject_scatter,
    meta=None,
    *,
    ax: plt.Axes | None = None,
    figsize=(3.6, 3.6),
):
    """Plot one animal per point: empirical RB vs full-fitted simulated RB."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)
    df = to_pandas_df(subject_scatter)
    required = {"empirical_rb", "full_fitted_rb"}
    if df.empty or not required.issubset(df.columns):
        ax.text(0.5, 0.5, "No valid subject data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig, ax

    x = df["empirical_rb"].to_numpy(dtype=float)
    y = df["full_fitted_rb"].to_numpy(dtype=float)
    yerr = None
    if {"full_fitted_rb_lo", "full_fitted_rb_hi"}.issubset(df.columns):
        lo = df["full_fitted_rb_lo"].to_numpy(dtype=float)
        hi = df["full_fitted_rb_hi"].to_numpy(dtype=float)
        yerr = np.vstack([np.clip(y - lo, 0.0, None), np.clip(hi - y, 0.0, None)])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=".",
        ms=5,
        color=COUNTERFACTUAL_PALETTE["Full fitted"],
        ecolor=COUNTERFACTUAL_PALETTE["Full fitted"],
        alpha=0.8,
        capsize=2,
        linewidth=1.0,
    )
    ax.plot([0, 1], [0, 1], color="0.45", lw=0.9, ls="--", zorder=0)
    ax.axhline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental RB")
    ax.set_ylabel("Full fitted RB")
    ax.set_title("Full model by animal")
    if len(df) > 1 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.2f}", ha="left", va="top", transform=ax.transAxes)
    return fig, ax


def plot_action_trace_parameter_fixed_rb(*args, **kwargs):
    """Alias for the parameter-fixed repetition-bias plot."""
    return plot_action_trace_counterfactual_rb(*args, **kwargs)


def plot_action_trace_parameter_fixed_lag_match(*args, **kwargs):
    """Alias for the parameter-fixed lag-match plot."""
    return plot_action_trace_counterfactual_lag_match(*args, **kwargs)


def plot_action_trace_parameter_fixed_subject_scatter(*args, **kwargs):
    """Alias for the parameter-fixed full-model animal scatter."""
    return plot_action_trace_counterfactual_subject_scatter(*args, **kwargs)


def add_shared_figure_legend(
    fig,
    *,
    source_ax,
    title: str | None = None,
    legend_ax: plt.Axes | None = None,
    bbox_x: float = 0.94,
    legend: bool = True,
) -> None:
    if not legend:
        return
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return
    if legend_ax is not None:
        legend_ax.axis("off")
        legend_ax.legend(
            handles,
            labels,
            title=title,
            loc="center left",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            labelspacing=0.35,
            handlelength=2.0,
        )
        return
    fig.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(bbox_x, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        labelspacing=0.35,
        handlelength=2.0,
    )


def _axes_from_plot_result(result):
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, plt.Axes):
                return item.figure, np.asarray([item], dtype=object)
            if isinstance(item, (list, tuple, np.ndarray)):
                axes = [ax for ax in np.asarray(item, dtype=object).ravel() if isinstance(ax, plt.Axes)]
                if axes:
                    return axes[0].figure, np.asarray(axes, dtype=object)
            if isinstance(item, plt.Figure):
                return item, np.asarray(item.axes, dtype=object)
    if isinstance(result, plt.Axes):
        return result.figure, np.asarray([result], dtype=object)
    if isinstance(result, plt.Figure):
        return result, np.asarray(result.axes, dtype=object)
    raise TypeError("Could not resolve matplotlib axes from plot result.")


def _axis_artist_snapshot(axes) -> dict[int, dict[str, set]]:
    return {
        id(ax): {
            "lines": set(ax.lines),
            "collections": set(ax.collections),
            "patches": set(ax.patches),
        }
        for ax in np.asarray(axes, dtype=object).ravel()
        if isinstance(ax, plt.Axes)
    }


def _style_axis_artists(ax, *, before: dict[str, set] | None, style: dict) -> None:
    color = style.get("color")
    linestyle = style.get("linestyle")
    linewidth = style.get("linewidth")
    alpha = style.get("alpha")
    marker = style.get("marker")

    new_lines = list(ax.lines) if before is None else [artist for artist in ax.lines if artist not in before["lines"]]
    for line in new_lines:
        if color is not None:
            line.set_color(color)
            line.set_markerfacecolor(color)
            line.set_markeredgecolor(color)
        if linestyle is not None and line.get_linestyle() not in {"None", "", " "}:
            line.set_linestyle(linestyle)
        if linewidth is not None:
            line.set_linewidth(linewidth)
        if alpha is not None:
            line.set_alpha(alpha)
        if marker is not None and line.get_marker() not in {None, "None", "", " "}:
            line.set_marker(marker)

    new_collections = (
        list(ax.collections)
        if before is None
        else [artist for artist in ax.collections if artist not in before["collections"]]
    )
    for collection in new_collections:
        if color is not None:
            try:
                collection.set_edgecolor(color)
                collection.set_facecolor(color)
            except Exception:
                pass
        if alpha is not None:
            collection.set_alpha(alpha)

    new_patches = list(ax.patches) if before is None else [artist for artist in ax.patches if artist not in before["patches"]]
    for patch in new_patches:
        if color is not None:
            patch.set_edgecolor(color)
            patch.set_facecolor(color)
        if alpha is not None:
            patch.set_alpha(alpha)


def overlay_plot_by_group(
    plot_fn,
    df_like,
    *,
    group_col: str,
    group_order: list | None = None,
    group_labels: dict | None = None,
    group_styles: dict | None = None,
    use_default_colors: bool = True,
    plot_kwargs: dict | None = None,
    axes_kwarg: str = "axes",
    legend_title: str | None = None,
    legend_loc: str = "upper right",
):
    """Call an existing axes-aware plot once per group and overlay the result.

    This keeps task plots unchanged: the wrapper filters the dataframe, reuses
    the axes from the first call, and styles only the artists added by each
    subsequent call.
    """
    df = to_pandas_df(df_like)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}.")
    df = df[df[group_col].notna()].copy()
    if df.empty:
        return None, []

    if group_order is None:
        group_order = list(pd.unique(df[group_col]))
    if group_labels is None:
        group_labels = {}

    default_colors = sns.color_palette("tab10", n_colors=max(1, len(group_order)))
    default_styles = {}
    for idx, value in enumerate(group_order):
        style = {"linestyle": "-"}
        if use_default_colors:
            style["color"] = default_colors[idx]
        default_styles[value] = style
    if group_styles is not None:
        for value, style in group_styles.items():
            default_styles.setdefault(value, {}).update(style)

    fig = None
    axes = None
    base_kwargs = dict(plot_kwargs or {})

    for group_value in group_order:
        sub = df[df[group_col] == group_value].copy()
        if sub.empty:
            continue

        kwargs = dict(base_kwargs)
        before = None
        if axes is not None:
            kwargs[axes_kwarg] = axes[0] if axes_kwarg == "ax" else axes
            before = _axis_artist_snapshot(axes)

        result = plot_fn(sub, **kwargs)
        if result is None:
            continue
        fig, axes = _axes_from_plot_result(result)
        style = default_styles.get(group_value, {})

        for ax in np.asarray(axes, dtype=object).ravel():
            if not isinstance(ax, plt.Axes):
                continue
            _style_axis_artists(
                ax,
                before=None if before is None else before.get(id(ax)),
                style=style,
            )
            if ax.legend_ is not None:
                ax.legend_.remove()

    if fig is None or axes is None:
        return None, []

    handles = []
    for group_value in group_order:
        style = default_styles.get(group_value, {})
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color", "black"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 2.0),
                marker=style.get("marker", None),
                label=group_labels.get(group_value, str(group_value)),
            )
        )
    fig.legend(handles=handles, title=legend_title, loc=legend_loc, frameon=False)
    fig.tight_layout()
    return fig, axes


def centered_numeric_group_palette(group_order: list) -> dict:
    numeric_order = [float(val) for val in group_order]
    negatives = [val for val in numeric_order if val < 0]
    positives = [val for val in numeric_order if val > 0]
    has_zero = any(np.isclose(val, 0.0) for val in numeric_order)

    palette = {}
    if negatives:
        neg_colors = list(
            reversed(sns.color_palette("Blues", len(negatives) + 2)[1:-1])
        )
        for value, color in zip(sorted(negatives), neg_colors, strict=False):
            palette[value] = color
    if has_zero:
        palette[0.0] = (0.45, 0.45, 0.45)
    if positives:
        pos_colors = sns.color_palette("Reds", len(positives) + 2)[1:-1]
        for value, color in zip(sorted(positives), pos_colors, strict=False):
            palette[value] = color
    return palette


def _apply_summary_axis_style(ax, *, meta, **style):
    ax.set_xlabel(style.get("xlabel", meta["xlabel"]))
    ax.set_ylabel(style.get("ylabel", meta["ylabel"]))
    ax.set_ylim(0.0, 1.0)

    if meta["baseline"] is not None:
        ax.axhline(meta["baseline"], color="gray", lw=0.8, ls="--", alpha=0.5)

    apply_axis_style(ax, **style)


def plot_simple_summary(
    summary_df,
    *,
    meta,
    ax: plt.Axes | None = None,
    figsize=(3.0, 3.0),
    legend: bool = True,
    **style,
):
    if summary_df is None or summary_df.empty:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    x = summary_df["x_center"].to_numpy(dtype=float)
    model_mean = summary_df["model_mean"].to_numpy(dtype=float)
    model_sem = summary_df["model_sem"].to_numpy(dtype=float)

    ax.plot(
        x,
        model_mean,
        color="black",
        linewidth=2.0,
        label="Model",
        zorder=3,
    )
    ax.fill_between(
        x,
        np.clip(model_mean - model_sem, 0.0, 1.0),
        np.clip(model_mean + model_sem, 0.0, 1.0),
        color="black",
        alpha=0.12,
        linewidth=0.0,
        zorder=2,
    )
    ax.errorbar(
        x,
        summary_df["data_mean"].to_numpy(dtype=float),
        yerr=summary_df["data_sem"].to_numpy(dtype=float),
        fmt="o",
        color="#2b7bba",
        ecolor="#2b7bba",
        elinewidth=1.0,
        capsize=3,
        label="Data",
        zorder=4,
    )
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    _apply_summary_axis_style(ax, meta=meta, **style)
    if legend:
        ax.legend(frameon=False, fontsize=8)
    elif ax.legend_ is not None:
        ax.legend_.remove()
    return ax


def plot_repeat_by_regressor_simple(
    plot_df,
    *,
    regressor_col: str,
    views: dict,
    is_mcdr: bool,
    baseline: float | None = None,
    xlabel: str | None = None,
    n_bins: int = 10,
    legend: bool = True,
    **style,
):
    df_pd = attach_repeat_choice_evidence(
        plot_df,
        views=views,
        is_mcdr=is_mcdr,
    )
    required_cols = {regressor_col, "_repeat_choice", "_p_repeat_model", "subject"}
    if df_pd.empty or not required_cols.issubset(df_pd.columns):
        return None

    df_pd = df_pd.copy()
    df_pd[regressor_col] = pd.to_numeric(df_pd[regressor_col], errors="coerce")
    df_pd["_repeat_choice"] = pd.to_numeric(df_pd["_repeat_choice"], errors="coerce")
    df_pd["_p_repeat_model"] = pd.to_numeric(df_pd["_p_repeat_model"], errors="coerce")
    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd["_repeat_choice"])
        & np.isfinite(df_pd["_p_repeat_model"])
    ].copy()
    if df_pd.empty:
        return None

    df_pd, bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=n_bins,
        quantiles=None,
    )
    if df_pd is None:
        return None

    summary = summarize_simple_curve(
        df_pd,
        subject_col="subject",
        reg_bin_col="_reg_bin",
        regressor_col=regressor_col,
        data_col="_repeat_choice",
        model_col="_p_repeat_model",
    )
    if summary.empty:
        return None

    if baseline is None:
        baseline = 1.0 / next(iter(views.values())).num_classes if views else 0.5
    meta = {
        "xlabel": xlabel or display_regressor_name(regressor_col),
        "ylabel": r"$p(\mathrm{repeat})$",
        "baseline": float(baseline),
        "xlim": padded_numeric_limits(bin_centers["x_center"], absolute_pad=0.25),
    }
    return plot_simple_summary(summary, meta=meta, legend=legend, **style)


def prepare_binned_accuracy_total_evidence_panels(
    plot_df,
    *,
    regressor_col: str,
    adapter,
    views: dict,
    is_mcdr: bool,
    baseline: float,
    n_bins: int = 10,
) -> tuple[list[dict] | None, str | None]:
    if adapter is None or views is None:
        raise ValueError("x_axis='total_evidence' requires adapter=... and views=....")

    df_pd = attach_total_fitted_evidence(
        plot_df,
        adapter=adapter,
        views=views,
        is_mcdr=is_mcdr,
    )
    required_cols = {
        regressor_col,
        "correct_bool",
        "_fitted_correct_prob",
        "_fitted_total_evidence",
        "subject",
    }
    if df_pd.empty or not required_cols.issubset(df_pd.columns):
        return None, None

    df_pd = df_pd.copy()
    for col in [regressor_col, "correct_bool", "_fitted_correct_prob", "_fitted_total_evidence"]:
        df_pd[col] = pd.to_numeric(df_pd[col], errors="coerce")
    df_pd = df_pd[
        np.isfinite(df_pd[regressor_col])
        & np.isfinite(df_pd["correct_bool"])
        & np.isfinite(df_pd["_fitted_correct_prob"])
        & np.isfinite(df_pd["_fitted_total_evidence"])
    ].copy()
    if df_pd.empty:
        return None, None

    df_pd, reg_bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col=regressor_col,
        max_bins=4,
        quantiles=None,
    )
    if df_pd is None:
        return None, None
    reg_bin_labels = reg_bin_centers["_reg_bin"].tolist()

    df_pd, evidence_bin_centers = attach_quantile_bin_column(
        df_pd,
        value_col="_fitted_total_evidence",
        bin_col="_evidence_bin",
        max_bins=n_bins,
        quantiles=None,
    )
    if df_pd is None:
        return None, None

    summary = summarize_grouped_panel(
        df_pd,
        line_group_col="_reg_bin",
        x_col="_evidence_bin",
        subject_col="subject",
        data_col="correct_bool",
        model_col="_fitted_correct_prob",
        line_order=reg_bin_labels,
    )
    if summary.empty:
        return None, None
    summary = summary.merge(
        evidence_bin_centers[["_evidence_bin", "x_center"]],
        on="_evidence_bin",
        how="left",
    ).sort_values(["_reg_bin", "x_center"])

    subject_summary = (
        df_pd.groupby(["_reg_bin", "subject", "_evidence_bin"], observed=True)
        .agg(
            data_mean=("correct_bool", "mean"),
            model_mean=("_fitted_correct_prob", "mean"),
            n_trials=("correct_bool", "count"),
        )
        .reset_index()
        .merge(evidence_bin_centers[["_evidence_bin", "x_center"]], on="_evidence_bin", how="left")
    )

    meta = {
        "xlabel": "Correct-vs-rest fitted evidence",
        "ylabel": "Accuracy",
        "legend_title": display_regressor_name(regressor_col),
        "baseline": baseline,
        "line_order": reg_bin_labels,
        "line_x_centers": dict(zip(reg_bin_centers["_reg_bin"], reg_bin_centers["x_center"], strict=False)),
        "x_col": "x_center",
        "fit_x_col": "x_center",
    }
    return [{"summary": summary, "subject_summary": subject_summary, "meta": meta}], display_regressor_name(regressor_col)


def fit_lapse_logistic_for_panel(
    panel: dict,
    *,
    fit_lapse_by_subject: bool = True,
    lapse_max: float = 0.4,
    share_lapse_logistic_core: bool = False,
    default_x_col: str,
    fit_x_limits: tuple[float, float] | None = None,
):
    summary = panel.get("summary")
    meta = panel.get("meta", {})
    if summary is None or summary.empty:
        return {}

    x_fit_col = meta.get("fit_x_col", meta.get("x_col", default_x_col))
    if x_fit_col not in summary.columns:
        return {}
    if pd.to_numeric(summary[x_fit_col], errors="coerce").notna().sum() < 2:
        return {}
    if fit_x_limits is not None:
        x_min, x_max = (float(v) for v in fit_x_limits)
        x_values = pd.to_numeric(summary[x_fit_col], errors="coerce")
        summary = summary.loc[x_values.between(x_min, x_max, inclusive="both")].copy()
        if summary.empty or x_values.loc[summary.index].nunique() < 2:
            return {}

    line_order = meta.get("line_order") or summary["_reg_bin"].dropna().unique().tolist()
    subject_summary = panel.get("subject_summary")
    if (
        fit_lapse_by_subject
        and subject_summary is not None
        and not subject_summary.empty
        and x_fit_col in subject_summary.columns
    ):
        if fit_x_limits is not None:
            x_min, x_max = (float(v) for v in fit_x_limits)
            subject_x_values = pd.to_numeric(subject_summary[x_fit_col], errors="coerce")
            subject_summary = subject_summary.loc[
                subject_x_values.between(x_min, x_max, inclusive="both")
            ].copy()
            if subject_summary.empty:
                return {}
        subject_fits = fit_lapse_logistic_by_subject_group(
            subject_summary,
            subject_col="subject",
            line_group_col="_reg_bin",
            x_col=x_fit_col,
            y_col="data_mean",
            weight_col="n_trials",
            line_order=line_order,
            lapse_max=lapse_max,
            shared_core=share_lapse_logistic_core,
        )
        if subject_fits:
            return subject_fits

    return fit_lapse_logistic_by_group(
        summary,
        line_group_col="_reg_bin",
        x_col=x_fit_col,
        y_col="md",
        weight_col="nd",
        line_order=line_order,
        lapse_max=lapse_max,
        shared_core=share_lapse_logistic_core,
    )


def plot_lapse_fit_parameter_panels(
    axes,
    fits: dict,
    *,
    line_order,
    meta: dict,
    regressor_label: str,
):
    axes = np.atleast_1d(axes)
    if len(axes) < 2:
        return

    lapse_ax, bias_ax = axes[:2]
    groups = [group for group in list(line_order or fits.keys()) if group in fits]
    if not groups:
        for ax in (bias_ax, lapse_ax):
            ax.set_axis_off()
        return

    center_map = meta.get("line_x_centers") or {}
    x_values = []
    use_numeric_centers = bool(center_map)
    for idx, group in enumerate(groups):
        value = center_map.get(group, center_map.get(str(group), np.nan))
        if use_numeric_centers and np.isfinite(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]):
            x_values.append(float(value))
        else:
            use_numeric_centers = False
            x_values.append(float(idx))

    if not use_numeric_centers:
        x_values = np.arange(len(groups), dtype=float)
        for ax in (bias_ax, lapse_ax):
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(group) for group in groups], rotation=0)

    lapse_left = np.asarray([fits[group].lapse_left for group in groups], dtype=float)
    lapse_right = np.asarray([fits[group].lapse_right for group in groups], dtype=float)
    bias = np.asarray([fits[group].bias for group in groups], dtype=float)
    y_values = np.concatenate([lapse_left, lapse_right, bias, np.asarray([0.0])])
    y_values = y_values[np.isfinite(y_values)]
    shared_ylim = None
    if y_values.size:
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if np.isclose(y_min, y_max):
            pad = 0.1 if np.isclose(y_min, 0.0) else 0.1 * abs(y_min)
        else:
            pad = 0.05 * (y_max - y_min)
        shared_ylim = (y_min - pad, y_max + pad)

    lapse_ax.plot(x_values, lapse_left, "-o", color="#2b7bba", lw=1.5, ms=4, label="left")
    lapse_ax.plot(x_values, lapse_right, "-o", color="#c43c39", lw=1.5, ms=4, label="right")
    apply_axis_style(
        lapse_ax,
        xlabel=regressor_label,
        ylabel="Lapse",
        title="Lapses",
    )
    lapse_ax.legend(frameon=False, fontsize=8)

    bias_ax.plot(x_values, bias, "-o", color="black", lw=1.5, ms=4)
    bias_ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    apply_axis_style(
        bias_ax,
        xlabel=regressor_label,
        ylabel="Psychometric bias",
        title="Bias",
    )
    if shared_ylim is not None:
        lapse_ax.set_ylim(shared_ylim)
        bias_ax.set_ylim(shared_ylim)


def plot_grouped_summary(
    ax,
    summary_df,
    *,
    line_group_col: str,
    x_col: str,
    meta,
    label_map: dict | None = None,
    palette: dict | None = None,
    legend: bool = True,
    **style,
):
    if summary_df is None or summary_df.empty:
        ax.set_axis_off()
        apply_axis_style(ax, **style)
        return ax

    summary_df = summary_df.copy()
    rename_map = {
        "data_mean": "md",
        "model_mean": "mm",
        "data_sem": "sem",
    }
    for src, dst in rename_map.items():
        if dst not in summary_df.columns and src in summary_df.columns:
            summary_df[dst] = summary_df[src]

    line_order = meta.get("line_order") or list(
        summary_df[line_group_col].dropna().unique()
    )
    default_palette = sns.color_palette("viridis", len(line_order))

    for group_value, default_color in zip(line_order, default_palette, strict=False):
        sub = summary_df[summary_df[line_group_col] == group_value].copy()
        if sub.empty:
            continue

        color = (
            palette.get(group_value, default_color)
            if palette is not None
            else default_color
        )
        label = (
            label_map.get(group_value, group_value)
            if label_map is not None
            else group_value
        )

        if meta.get("categorical_x", False):
            x_order = meta.get("x_order")
            if x_order is not None:
                x_pos_map = {str(value): idx for idx, value in enumerate(x_order)}
                xpos = sub[x_col].astype(str).map(x_pos_map).to_numpy(dtype=float)
            else:
                xpos = np.arange(len(sub), dtype=float)
        else:
            xpos = sub[x_col].to_numpy(dtype=float)

        ax.plot(
            xpos,
            sub["mm"].to_numpy(dtype=float),
            "-",
            color=color,
            lw=2.0,
            label=str(label),
        )
        ax.errorbar(
            xpos,
            sub["md"].to_numpy(dtype=float),
            yerr=sub["sem"].to_numpy(dtype=float),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            ms=5,
            capsize=3,
            zorder=5,
        )

    if meta.get("categorical_x", False) and meta.get("x_order") is not None:
        ax.set_xticks(np.arange(len(meta["x_order"]), dtype=float))
        ax.set_xticklabels(meta["x_tick_labels"] or meta["x_order"])
    elif meta.get("xticks") is not None:
        ax.set_xticks(meta["xticks"], labels=meta.get("x_tick_labels"))

    _apply_summary_axis_style(ax, meta=meta, **style)

    legend_kwargs = {
        "title": meta.get("legend_title"),
        "frameon": False,
    }
    if meta.get("legend_outside", False):
        legend_kwargs.update(
            {
                "loc": "upper left",
                "bbox_to_anchor": (1.0, 1.0),
                "borderaxespad": 0.0,
                "title_fontsize": 9,
                "labelspacing": 0.35,
                "handlelength": 2.0,
            }
        )
    if legend:
        ax.legend(**legend_kwargs)
    elif ax.legend_ is not None:
        ax.legend_.remove()
    return ax


def _summarize_regressor_by_regressor_magnitude(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_axis: str,
    subject_col: str,
    n_bins: int,
    use_abs_x: bool,
) -> pd.DataFrame | None:
    required = {x_axis, y_axis, subject_col}
    if not required.issubset(df.columns):
        return None

    df = df.copy()
    df[x_axis] = pd.to_numeric(df[x_axis], errors="coerce")
    df[y_axis] = pd.to_numeric(df[y_axis], errors="coerce")
    df["_x_magnitude"] = np.abs(df[x_axis]) if use_abs_x else df[x_axis]
    df = df[np.isfinite(df["_x_magnitude"]) & np.isfinite(df[y_axis])].copy()
    if df.empty:
        return None

    df, centers = attach_quantile_bin_column(
        df,
        value_col="_x_magnitude",
        bin_col="_x_bin",
        max_bins=n_bins,
        center_col="x_center",
        center_agg="median",
    )
    if df is None or centers.empty:
        return None

    subject_summary = (
        df.groupby([subject_col, "_x_bin"], observed=True)
        .agg(y_mean=(y_axis, "mean"))
        .reset_index()
        .merge(centers[["_x_bin", "x_center"]], on="_x_bin", how="left")
    )
    if subject_summary.empty:
        return None

    summary = (
        subject_summary.groupby("_x_bin", observed=True)
        .agg(
            x_center=("x_center", "mean"),
            y_mean=("y_mean", "mean"),
            y_std=("y_mean", "std"),
            n_subjects=(subject_col, "count"),
        )
        .reset_index()
        .sort_values("x_center")
    )
    summary["y_sem"] = summary["y_std"].fillna(0.0) / np.sqrt(summary["n_subjects"].clip(lower=1))
    return summary


def plot_regressor_net_impact(
    plot_df,
    *,
    x_axis: str,
    y_axis: str,
    subject_col: str = "subject",
    n_bins: int = 6,
    use_abs_x: bool = True,
    ax: plt.Axes | None = None,
    axes: Sequence[plt.Axes] | None = None,
    figsize=(3.4, 3.0),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = "#2b7bba",
    **style,
):
    """Plot one regressor's value across bins of another regressor's magnitude."""
    if axes is not None:
        axes = np.asarray(axes, dtype=object).ravel()
        if len(axes) == 0:
            raise ValueError("Expected at least one axis in `axes`.")
        ax = axes[0]
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize)

    df = to_pandas_df(plot_df)
    summary = _summarize_regressor_by_regressor_magnitude(
        df,
        x_axis=x_axis,
        y_axis=y_axis,
        subject_col=subject_col,
        n_bins=n_bins,
        use_abs_x=use_abs_x,
    )
    if summary is None or summary.empty:
        ax.text(0.5, 0.5, "No valid regressor data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        apply_axis_style(ax, **style)
        return ax

    x = summary["x_center"].to_numpy(dtype=float)
    y_mean = summary["y_mean"].to_numpy(dtype=float)
    y_sem = summary["y_sem"].to_numpy(dtype=float)

    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.errorbar(
        x,
        y_mean,
        yerr=y_sem,
        fmt="o-",
        color=color,
        ecolor=color,
        elinewidth=1.0,
        linewidth=2.0,
        markersize=5,
        capsize=3,
        zorder=3,
    )

    x_label = display_regressor_name(x_axis)
    y_label = display_regressor_name(y_axis)
    ax.set_xlabel(xlabel or (f"|{x_label}|" if use_abs_x else x_label))
    ax.set_ylabel(ylabel or y_label)
    if title is not None:
        ax.set_title(title)
    apply_axis_style(ax, **style)
    return ax


def plot_integration_map_panels(
    panels: list[dict],
    *,
    meta: dict,
    axes: Sequence[plt.Axes] | None = None,
    figsize=None,
    contour_levels: tuple[float, ...] = (0.15, 0.3, 0.5, 0.7, 0.85),
    colours=None,
    cmap: str | None = None,
    interpolation: str | None = None,
    cbar_ax: plt.Axes | None = None,
    show_colorbar: bool = True,
    colorbar_label: str | None = None,
    data_points_cutoff: float = 20.0,
    **style,
):
    if not panels:
        return None

    _ = cmap, interpolation
    if colours is None:
        colours = np.array([[103, 169, 221], [241, 135, 34]], dtype=float) / 255.0
    else:
        colours = np.asarray(colours, dtype=float)
        if colours.max(initial=0.0) > 1.0:
            colours = colours / 255.0
    if colours.shape != (2, 3):
        raise ValueError("colours must be a 2-by-3 RGB array.")

    n_panels = len(panels)
    if axes is None:
        ncols = n_panels + int(show_colorbar and cbar_ax is None)
        width_ratios = [1.0] * n_panels + ([0.08] if show_colorbar and cbar_ax is None else [])
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=figsize or (4 * n_panels + (0.35 if show_colorbar else 0.0), 4.0),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios},
        )
        axes = np.asarray(axes, dtype=object).ravel()
        if show_colorbar and cbar_ax is None:
            cbar_ax = axes[-1]
            axes = axes[:n_panels]
    else:
        axes = np.asarray(axes, dtype=object).ravel()
        fig = axes[0].figure

    for ax, panel in zip(axes, panels, strict=False):
        z = np.asarray(panel["map"], dtype=float)
        z_for_colour = np.nan_to_num(z, nan=0.0)
        rgb = colours[0] + z_for_colour[..., None] * (colours[1] - colours[0])
        intensity = np.minimum(
            np.nan_to_num(np.asarray(panel["n_datapoints"], dtype=float), nan=0.0)
            / float(data_points_cutoff),
            1.0,
        )
        rgb = 1.0 - (1.0 - rgb) * intensity[..., None]

        x_centers = np.asarray(panel["x_centers"], dtype=float)
        y_centers = np.asarray(panel["y_centers"], dtype=float)
        if x_centers.size == 0 or y_centers.size == 0:
            ax.set_axis_off()
            continue

        if x_centers.size == 1:
            x_extent = (float(panel["x_edges"][0]), float(panel["x_edges"][-1]))
        else:
            x_extent = (float(x_centers[0]), float(x_centers[-1]))
        if y_centers.size == 1:
            y_extent = (float(panel["y_edges"][0]), float(panel["y_edges"][-1]))
        else:
            y_extent = (float(y_centers[0]), float(y_centers[-1]))

        ax.imshow(
            np.transpose(rgb, (1, 0, 2)),
            extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        z_for_contour = np.nan_to_num(z, nan=0.0)
        finite = np.isfinite(z_for_contour)
        if finite.any():
            lo = float(np.nanmin(z_for_contour))
            hi = float(np.nanmax(z_for_contour))
            levels = list(contour_levels)
            mid_idx = int(round((len(levels) + 1) / 2.0)) - 1
            thick_level = levels[mid_idx] if 0 <= mid_idx < len(levels) else None
            thin_levels = [
                level
                for idx, level in enumerate(levels)
                if idx != mid_idx and lo < level < hi
            ]
            if thin_levels:
                ax.contour(
                    x_centers,
                    y_centers,
                    z_for_contour.T,
                    levels=thin_levels,
                    colors="black",
                    linewidths=0.5,
                )
            if thick_level is not None and lo < thick_level < hi:
                ax.contour(
                    x_centers,
                    y_centers,
                    z_for_contour.T,
                    levels=[thick_level],
                    colors="black",
                    linewidths=1.0,
                )
        ax.set_xlabel(meta["xlabel"])
        if meta.get("xticks") is not None:
            ax.set_xticks(meta["xticks"], labels=meta.get("x_tick_labels"))
        # ax.set_box_aspect(1)
        apply_axis_style(ax, **style)

    axes[0].set_ylabel(meta["ylabel"])
    if show_colorbar:
        color_map = LinearSegmentedColormap.from_list("integration_map", colours)
        scalar_mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=color_map)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            cax=cbar_ax,
            ax=None if cbar_ax is not None else axes,
        )
        colorbar.set_label(colorbar_label or meta.get("colorbar_label", r"$\mathit{p}(\mathrm{right})$"))
    return fig, axes

def plot_session_running_accuracy_repetition(
    prepared,
    *,
    ax: plt.Axes | None = None,
    figsize=(4.8, 3.0),
    accuracy_label: str = "Accuracy",
    repetition_label: str = "Repeating bias",
    **style,
):
    """Plot running accuracy and repetition probability across trials."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)

    trace = to_pandas_df(prepared.get("trace", pd.DataFrame()))
    required = {"trial_index", "running_accuracy", "running_repetition"}
    if trace.empty or not required.issubset(trace.columns):
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        apply_axis_style(ax, **style)
        return fig, ax

    ax.plot(
        trace["trial_index"],
        trace["running_accuracy"],
        lw=1.8,
        color="#2b7bba",
        label=accuracy_label,
    )
    ax.plot(
        trace["trial_index"],
        trace["running_repetition"],
        lw=1.8,
        color="#c43c39",
        label=repetition_label,
    )

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Running probability")
    ax.legend(frameon=False, fontsize=8)

    window = prepared.get("meta", {}).get("running_window")
    if window is not None:
        ax.set_title(f"Running behavior, window={window}")

    apply_axis_style(ax, **style)
    return fig, ax


def plot_session_behavior_autocorrelogram(
    prepared,
    *,
    ax: plt.Axes | None = None,
    figsize=(4.0, 3.0),
    **style,
):
    """Plot autocorrelogram of outcome and repetition vectors."""
    fig, ax = resolve_single_axis(ax=ax, figsize=figsize, constrained_layout=True)

    ac = to_pandas_df(prepared.get("autocorr", pd.DataFrame()))
    required = {"lag", "autocorr", "signal"}
    if ac.empty or not required.issubset(ac.columns):
        ax.text(0.5, 0.5, "No valid autocorrelation data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        apply_axis_style(ax, **style)
        return fig, ax

    palette = {
        "Outcome": "#2b7bba",
        "Repetition": "#c43c39",
    }

    for signal in ("Outcome", "Repetition"):
        sub = ac[ac["signal"] == signal].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("lag")
        ax.plot(
            sub["lag"],
            sub["autocorr"],
            lw=1.8,
            marker="o",
            ms=3,
            color=palette.get(signal, "black"),
            label=signal,
        )

    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Behavioral timescale")
    if ac["lag"].notna().any():
        ax.set_xlim(left=1.0, right=float(np.nanmax(ac["lag"])) if len(ac) else None)
    ax.legend(frameon=False, fontsize=8)

    apply_axis_style(ax, **style)
    return fig, ax


def plot_session_accuracy_repetition_timescale(
    prepared,
    *,
    axes: Sequence[plt.Axes] | None = None,
    figsize=(8.8, 3.0),
    running_style: dict | None = None,
    autocorr_style: dict | None = None,
):
    """Plot running behavior and behavioral autocorrelogram as a two-panel figure."""
    fig, axes = resolve_axes(
        axes=axes,
        n_axes=2,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    plot_session_running_accuracy_repetition(
        prepared,
        ax=axes[0],
        **(running_style or {}),
    )
    plot_session_behavior_autocorrelogram(
        prepared,
        ax=axes[1],
        **(autocorr_style or {}),
    )
    return fig, axes


def plot_corrected_behavior_autocorrelograms(
    prepared,
    *,
    axes: Sequence[plt.Axes] | None = None,
    figsize=(7.0, 3.0),
    model_autocorr=None,
    glm_autocorr=None,
    autocorr_col: str = "autocorr",
    sem_col: str | None = "autocorr_sem",
    model_autocorr_col: str | None = None,
    data_color: str = "#1f77b4",
    model_color: str = "black",
    data_label: str = "Data",
    model_label: str = "Fitted GLM",
    ylabel: str = "Corrected autocorrelation",
    signals: Sequence[str] = ("Outcome", "Repetition"),
    titles: dict[str, str] | None = None,
    **style,
):
    """Plot Tiffany-style autocorrelograms for outcomes and repetitions.

    Pass ``model_autocorr`` or ``glm_autocorr`` to overlay a fitted GLM simulation.
    ``autocorr_col``/``sem_col`` can be changed to plot raw autocorrelation or
    cross-session correction terms with the same style.
    """
    fig, axes = resolve_axes(
        axes=axes,
        n_axes=len(signals),
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )

    data = to_pandas_df(prepared.get("autocorr", pd.DataFrame()))
    model = model_autocorr if model_autocorr is not None else glm_autocorr
    if model is None:
        model = prepared.get("model_autocorr")
    model = to_pandas_df(model) if model is not None else pd.DataFrame()

    default_titles = {
        "Outcome": "Choice outcomes",
        "Repetition": "Repeated responses",
    }
    if titles is not None:
        default_titles.update(titles)

    for ax, signal in zip(axes, signals, strict=False):
        title = default_titles.get(signal, str(signal))
        sub = (
            data[data["signal"] == signal].copy()
            if "signal" in data.columns
            else pd.DataFrame()
        )
        if sub.empty or not {"lag", autocorr_col}.issubset(sub.columns):
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        sub = sub.sort_values("lag")
        yerr = sub[sem_col].to_numpy(dtype=float) if sem_col is not None and sem_col in sub.columns else None
        ax.errorbar(
            sub["lag"].to_numpy(dtype=float),
            sub[autocorr_col].to_numpy(dtype=float),
            yerr=yerr,
            fmt="o",
            ms=4,
            color=data_color,
            ecolor=data_color,
            elinewidth=1.0,
            capsize=2,
            label=data_label,
            zorder=4,
        )

        model_sub = (
            model[model["signal"] == signal].copy()
            if "signal" in model.columns
            else pd.DataFrame()
        )
        _model_col = model_autocorr_col or autocorr_col
        if _model_col not in model_sub.columns and "autocorr" in model_sub.columns:
            _model_col = "autocorr"
        if not model_sub.empty and {"lag", _model_col}.issubset(model_sub.columns):
            model_sub = model_sub.sort_values("lag")
            ax.plot(
                model_sub["lag"].to_numpy(dtype=float),
                model_sub[_model_col].to_numpy(dtype=float),
                color=model_color,
                lw=1.8,
                label=model_label,
                zorder=3,
            )

        ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        # ax.set_xlim(0, 25.5)
        apply_axis_style(ax, **style)

    return fig, axes


def draw_closed_loop_autocorrelogram_overlay_panel(
    ax: plt.Axes,
    signal: str,
    data_ac: pd.DataFrame,
    glm_ac: pd.DataFrame | None = None,
    glmhmm_ac: pd.DataFrame | None = None,
    *,
    colors: dict[str, str] | None = None,
    data_label: str = "Data",
    glm_label: str = "GLM",
    glmhmm_label: str = "GLM-HMM",
    ylabel: str = "Autocorrelation",
) -> None:
    colors = {
        "data": "tab:blue",
        "glm": "tab:gray",
        "glmhmm": "tab:red",
        **(colors or {}),
    }
    data_ac = to_pandas_df(data_ac)
    glm_ac = to_pandas_df(glm_ac) if glm_ac is not None else pd.DataFrame()
    glmhmm_ac = to_pandas_df(glmhmm_ac) if glmhmm_ac is not None else pd.DataFrame()

    data_sub = (
        data_ac[data_ac["signal"] == signal].sort_values("lag")
        if "signal" in data_ac.columns
        else pd.DataFrame()
    )
    if data_sub.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    yerr = (
        data_sub["autocorr_sem"].to_numpy(dtype=float)
        if "autocorr_sem" in data_sub.columns
        else None
    )
    ax.errorbar(
        data_sub["lag"],
        data_sub["autocorr"],
        yerr=yerr,
        fmt="o",
        capsize=0,
        ms=3,
        color=colors["data"],
        ecolor=colors["data"],
        label=data_label,
        zorder=4,
    )
    for label, model_ac, color in (
        (glm_label, glm_ac, colors["glm"]),
        (glmhmm_label, glmhmm_ac, colors["glmhmm"]),
    ):
        sub = (
            model_ac[model_ac["signal"] == signal].sort_values("lag")
            if "signal" in model_ac.columns
            else pd.DataFrame()
        )
        if sub.empty or "autocorr" not in sub.columns:
            continue
        ax.plot(sub["lag"], sub["autocorr"], color=color, label=label, zorder=3)

    ax.axhline(0.0, color="0.5", ls="--")
    ax.set_title("Outcomes" if signal == "Outcome" else "Repetitions")
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)
    if signal == "Repetition":
        ax.set_ylim(top=0.15)
    else:
        ax.set_ylim(top=0.05)
    ax.legend(frameon=False)


def plot_closed_loop_autocorrelogram_overlay(
    data_ac: pd.DataFrame,
    glm_ac: pd.DataFrame,
    glmhmm_ac: pd.DataFrame | None = None,
    *,
    axes: Sequence[plt.Axes] | None = None,
    figsize=None,
) -> plt.Figure:
    if figsize is None:
        figsize = fig_size(1, 2)
    fig, axes = resolve_axes(
        axes=axes,
        n_axes=2,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    for ax, signal in zip(axes, ("Outcome", "Repetition"), strict=True):
        draw_closed_loop_autocorrelogram_overlay_panel(
            ax,
            signal,
            data_ac,
            glm_ac,
            glmhmm_ac,
        )
    return fig


def plot_closed_loop_autocorrelograms_by_task(
    autocorrelograms_by_task: dict,
    *,
    task_order: Sequence[str] | None = None,
    task_labels: dict[str, str] | None = None,
    figsize=None,
) -> plt.Figure:
    task_order = list(task_order or autocorrelograms_by_task)
    task_labels = task_labels or {}
    if figsize is None:
        figsize = (7.0, 2.0 * max(1, len(task_order)))
    fig, axes = plt.subplots(
        len(task_order),
        2,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
        sharex=True,
    )
    for row_idx, task_name in enumerate(task_order):
        payload = autocorrelograms_by_task[task_name]
        for col_idx, signal in enumerate(("Outcome", "Repetition")):
            ax = axes[row_idx, col_idx]
            draw_closed_loop_autocorrelogram_overlay_panel(
                ax,
                signal,
                payload["data"]["autocorr"],
                payload.get("glm", {}).get("autocorr"),
                payload.get("glmhmm", {}).get("autocorr"),
            )
            if col_idx == 0:
                ax.set_ylabel(f"{task_labels.get(task_name, task_name)}\nAutocorrelation")
            else:
                ax.set_ylabel("")
            if row_idx < len(task_order) - 1:
                ax.set_xlabel("")
    return fig


def save_closed_loop_autocorrelogram_overlay_panels(
    data_ac: pd.DataFrame,
    glm_ac: pd.DataFrame,
    glmhmm_ac: pd.DataFrame | None,
    out_dir,
    *,
    stem_prefix: str,
) -> list:
    saved_paths = []
    for signal in ("Outcome", "Repetition"):
        fig, ax = plt.subplots(figsize=fig_size(2, 1), layout="constrained")
        draw_closed_loop_autocorrelogram_overlay_panel(
            ax,
            signal,
            data_ac,
            glm_ac,
            glmhmm_ac,
        )
        stem = out_dir / f"{stem_prefix}_closed_loop_autocorrelogram_{signal.lower()}"
        fig.savefig(stem.with_suffix(".png"), dpi=300)
        fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
        saved_paths.extend([stem.with_suffix(".png"), stem.with_suffix(".svg"), stem.with_suffix(".pdf")])
    return saved_paths


def psychometric_repeat(
    plot_df,
    ax=None,
    figsize=fig_size(n_cols=3),
    title="",
    color="tab:blue",
    *,
    session_col,
    trial_col,
    choice_col,
    stimulus_col,
    subject_col="subject",
    delay_col=None,
    difficulty_col=None,
    is_mcdr=False,
):
    df_pd = plot_df.to_pandas().copy() if hasattr(plot_df, "to_pandas") else pd.DataFrame(plot_df).copy()

    required_cols = [subject_col, session_col, trial_col, choice_col, stimulus_col]
    missing_cols = [column for column in required_cols if column not in df_pd.columns]
    optional_cols = [column for column in [delay_col, difficulty_col] if column is not None]
    missing_cols.extend(column for column in optional_cols if column not in df_pd.columns)
    if missing_cols:
        raise KeyError(f"psychometric_repeat missing dataframe columns: {', '.join(missing_cols)}")

    sort_cols = [subject_col, session_col, trial_col]
    plot_df = df_pd.sort_values(sort_cols, kind="stable").copy()
    plot_df["choice"] = pd.to_numeric(plot_df[choice_col], errors="coerce")
    plot_df["previous_choice"] = plot_df.groupby(
        [subject_col, session_col],
        observed=True,
    )["choice"].shift(1)
    plot_df["_repeat"] = (plot_df["choice"] == plot_df["previous_choice"]).astype(float)

    if is_mcdr:
        baseline = 1.0 / 3.0
    else:
        choice_values = set(plot_df["choice"].dropna().unique().tolist())
        if choice_values.issubset({-1.0, 1.0}):
            plot_df["previous_choice_sign"] = plot_df["previous_choice"]
        elif choice_values.issubset({0.0, 1.0}):
            plot_df["previous_choice_sign"] = (2.0 * plot_df["previous_choice"]) - 1.0
        else:
            raise ValueError("psychometric_repeat expects binary choices unless is_mcdr=True.")
        baseline = 0.5

    signed_stimulus = pd.to_numeric(plot_df[stimulus_col], errors="coerce")
    x_order = None
    x_tick_labels = None
    if delay_col is not None:
        x_values = (
            pd.to_numeric(plot_df[delay_col], errors="coerce").abs()
            * np.sign(signed_stimulus)
            * plot_df["previous_choice_sign"]
        )
        x_order = ["neg_0.1", "neg_1", "neg_3", "neg_10", "pos_10", "pos_3", "pos_1", "pos_0.1"]
        x_tick_labels = ["-0", "-1", "-3", "-10", "10", "3", "1", "0"]

        delay_magnitude = pd.Series(x_values, index=plot_df.index).abs()
        delay_label = np.where(
            np.isclose(delay_magnitude, 0.1),
            "0.1",
            delay_magnitude.round().astype("Int64").astype(str),
        )
        delay_side = np.where(x_values < 0, "neg", "pos")
        x_values = pd.Series(delay_side, index=plot_df.index) + "_" + pd.Series(delay_label, index=plot_df.index)
        xlabel = "Delay x choice$_{-1}$"
    elif is_mcdr:
        if difficulty_col is None:
            raise ValueError("psychometric_repeat requires difficulty_col when is_mcdr=True.")
        difficulty_labels = plot_df[difficulty_col].astype(str).map(
            {
                "VG": "VG",
                "DS": "DS",
                "DM": "DM",
                "DL": "DL",
            }
        )
        current_target = pd.to_numeric(plot_df[stimulus_col], errors="coerce")
        side_labels = np.where(current_target == plot_df["previous_choice"], "pos", "neg")
        x_values = pd.Series(side_labels, index=plot_df.index) + "_" + difficulty_labels
        x_order = ["neg_VG", "neg_DS", "neg_DM", "neg_DL", "pos_DL", "pos_DM", "pos_DS", "pos_VG"]
        x_tick_labels = ["-VG", "-DS", "-DM", "-DL", "DL", "DM", "DS", "VG"]
        xlabel = "Difficulty. x choice$_{-1}$"
    else:
        x_values = signed_stimulus * plot_df["previous_choice_sign"]
        x_values = pd.Series(x_values, index=plot_df.index).mask(lambda values: np.isclose(values, 0.0), 0.0)
        xlabel = "Stim. x choice$_{-1}$"
        x_tick_labels = [-20, -8, "", "", 0, "", "", 8, 20]

    plot_df["_repeat_x"] = x_values
    plot_df = plot_df.dropna(subset=["_repeat_x", "_repeat", "previous_choice"]).copy()

    return plot_mean_over_data(
        plot_df,
        x_col="_repeat_x",
        y_col="_repeat",
        subject_col=subject_col,
        x_order=x_order,
        x_tick_labels=x_tick_labels,
        xlabel=xlabel,
        ylabel=r"$p(\mathrm{repeat})$",
        title=title,
        baseline=baseline,
        baseline_area=False,
        color=color,
        ax=ax,
        figsize=figsize,
    )


__all__ = [
    "add_shared_figure_legend",
    "apply_axis_style",
    "centered_numeric_group_palette",
    "make_single_panel_figure",
    "draw_closed_loop_autocorrelogram_overlay_panel",
    "plot_empirical_accuracy_curve",
    "plot_action_trace_counterfactual_lag_match",
    "plot_action_trace_counterfactual_rb",
    "plot_action_trace_counterfactual_subject_scatter",
    "plot_action_trace_parameter_fixed_lag_match",
    "plot_action_trace_parameter_fixed_rb",
    "plot_action_trace_parameter_fixed_subject_scatter",
    "plot_closed_loop_autocorrelogram_overlay",
    "plot_closed_loop_autocorrelograms_by_task",
    "plot_corrected_behavior_autocorrelograms",
    "plot_session_accuracy_repetition_timescale",
    "plot_session_behavior_autocorrelogram",
    "plot_session_running_accuracy_repetition",
    "plot_prepared_weight_family",
    "plot_grouped_summary",
    "plot_integration_map_panels",
    "plot_regressor_net_impact",
    "plot_simple_summary",
    "resolve_axes",
    "resolve_single_axis",
    "save_closed_loop_autocorrelogram_overlay_panels",
]
