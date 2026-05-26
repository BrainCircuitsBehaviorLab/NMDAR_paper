# /// script
# [tool.marimo.opengraph]
# title = "Figure 1" 
# description = " Figure 1: Behavioral performance across tasks."
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from plot_saver import make_plot_saver
    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths
    from glmhmmt.tasks.fitted_regressors import FittedWeightRegressorSpec, mean_feature_weights_from_fit
    import os

    from src.utils import fig_size
    from src.process.common import attach_signed_delay_columns
    from src.plots.common import plot_mean_over_data

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")

    def save_fixed_bbox_pdf(fig, filename):
        fig.tight_layout()
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(filename)

    return (
        Path,
        attach_signed_delay_columns,
        fig_size,
        get_adapter,
        make_plot_saver,
        mo,
        pl,
        plot_mean_over_data,
        plt,
        save_fixed_bbox_pdf,
        sns,
    )


@app.cell
def _(Path, plt, sns):
    # Set style
    sns.set_theme(style='ticks', context='notebook')
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    return


@app.cell
def _(Path, make_plot_saver, mo):
    # Set paths
    data_path = Path(__file__).parents[1] / "data/processed"
    print(data_path)

    project_path = Path(__file__).resolve().parents[1]
    print(project_path)
    save_plot = make_plot_saver(
        mo,
        results_dir=project_path / "results",
        config_path=project_path / "config.toml",
        task_name="figure1",
        model_id="behavior",
    )
    return (data_path,)


@app.cell
def _(get_adapter):
    # Get adapters
    MCDR = get_adapter("MCDR")
    two_afc = get_adapter("2AFC")
    two_afc_delay = get_adapter("2AFC_delay")
    return MCDR, two_afc, two_afc_delay


@app.cell
def _(MCDR, data_path, pl, two_afc):
    # df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "alexis_combined.parquet"))
    df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "df_alexis_drug_combined.parquet"))  # With drug
    df_2AFC_delay = pl.read_parquet(data_path / "tiffany.parquet")
    df_MCDR = MCDR.subject_filter(pl.read_parquet(data_path / "MCDR_all.parquet"))
    # df_MCDR = df_MCDR.filter(pl.col("batch") == "11B")
    return df_2AFC, df_2AFC_delay, df_MCDR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavior plots
    """)
    return


@app.cell
def _(MCDR, two_afc, two_afc_delay):
    MCDR_plots = MCDR.get_plots()
    two_afc_plots = two_afc.get_plots()
    two_afc_delay_plots = two_afc_delay.get_plots()
    return MCDR_plots, two_afc_delay_plots, two_afc_plots


@app.cell
def _(
    attach_signed_delay_columns,
    df_2AFC_delay,
    fig_size,
    pl,
    plot_mean_over_data,
    plt,
    two_afc_delay_plots,
):
    # 2ADC

    # two_afc_delay_plots.plot_accuracy(df_2AFC_delay, figsize=fig_size(n_cols=2), title='')
    fig_, ax_ = plt.subplots(figsize=fig_size(n_cols=3), constrained_layout=True)
    two_afc_delay_plots.plot_accuracy(df_2AFC_delay.filter(pl.col("drug") == 'Saline'), ax=ax_, color="tab:gray", title="", label='Saline')
    two_afc_delay_plots.plot_accuracy(df_2AFC_delay.filter(pl.col("drug") == 'NR2B'), ax=ax_, color="tab:pink", title="", label='Drug')
    plt.savefig('acc_vs_delay.svg')
    plt.show()

    signed_delay_order = ["0L", "-1", "-3", "-10", "10", "3", "1", "0R"]
    signed_delay_tick_labels = ["-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"]
    df_2AFC_delay_signed = attach_signed_delay_columns(df_2AFC_delay.to_pandas())
    df_2AFC_delay_signed["p_right"] = (df_2AFC_delay_signed["choices"] > 0).astype(float)
    df_2AFC_delay_signed["_signed_delay_plot"] = df_2AFC_delay_signed[
        "_signed_delay_cat"
    ].astype(str)
    df_2AFC_delay_signed = df_2AFC_delay_signed[
        df_2AFC_delay_signed["_signed_delay_plot"].isin(signed_delay_order)
    ].copy()
    plot_mean_over_data(
        df_2AFC_delay_signed,
        x_col="_signed_delay_plot",
        x_order=signed_delay_order,
        x_tick_labels=signed_delay_tick_labels,
        y_col="p_right",
        xlabel="Signed delay (s)",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=0.5,
        color="tab:blue",
        figsize=fig_size(n_cols=3),
    )
    plt.savefig('p_right_vs_signed_delay.svg')
    plt.show()

    print(f"Number of subjects: {df_2AFC_delay['subject'].n_unique()}")
    # two_afc_delay_plots.plot_rb(df_2AFC_delay, figsize=fig_size(n_cols=3), title='')
    # plt.savefig('2ADC_rb.svg')
    # plt.show()

    fig_2ADC, ax_2ADC = plt.subplots(figsize = fig_size(n_cols=3), constrained_layout=True)
    two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "NR2B"), ax = ax_2ADC, title='', color = "tab:pink")
    two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "Saline"), ax = ax_2ADC, title='', color = "tab:gray")
    # two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "Rest"), ax = ax_2ADC, figsize=fig_size(n_cols=3), title='', show_baseline_ttest=True)
    plt.savefig('2ADC_rb.svg')
    plt.show()
    return


@app.cell
def _(df_2AFC, fig_size, pl, plot_mean_over_data, plt, two_afc_plots):
    # 2AFC

    # two_afc_plots.plot_accuracy(df_2AFC, figsize=fig_size(n_cols=2), title="")
    fig, ax = plt.subplots(figsize=fig_size(n_cols=3), constrained_layout=True)
    two_afc_plots.plot_accuracy(df_2AFC.filter(pl.col("Drug") == 0), ax=ax, color="tab:gray", title="")
    two_afc_plots.plot_accuracy(df_2AFC.filter(pl.col("Drug") == 1), ax=ax, color="tab:pink", title="")
    plt.savefig("acc_vs_ild.svg")
    plt.show()

    df_2AFC_p_right = df_2AFC.to_pandas().copy()
    df_2AFC_p_right["p_right"] = df_2AFC_p_right["Choice"].astype(float)
    plot_mean_over_data(
        df_2AFC_p_right,
        x_col="ILD",
        y_col="p_right",
        xlabel="ILD (dB)",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=0.5,
        color="tab:blue",
        figsize=fig_size(n_cols=3),
    )
    plt.gca().set_xticks(
        [-20, -8, -4, -2, 0, 2, 4, 8, 20],
        labels=["-20", "-8", "", "", "0", "", "", "8", "20"],
    )
    plt.savefig("p_right_vs_ild.svg")
    plt.show()

    print(f"Number of subjects: {df_2AFC['subject'].n_unique()}")
    fig_2AFC, ax_2AFC = plt.subplots(figsize=fig_size(n_cols=3), constrained_layout=True)
    two_afc_plots.plot_rb(df_2AFC.filter(pl.col("Drug") == 0), ax=ax_2AFC, title="", color="tab:gray")
    two_afc_plots.plot_rb(df_2AFC.filter(pl.col("Drug") == 1), ax=ax_2AFC, title="", color="tab:pink")
    # two_afc_plots.plot_rb(df_2AFC, ax=ax_2AFC, figsize=fig_size(3, 1.25), title="", show_baseline_ttest=True)
    plt.savefig("2AFC_rb.svg")
    plt.show()
    return


@app.cell
def _(MCDR_plots, df_MCDR, fig_size, pl, plt, save_fixed_bbox_pdf):
    palette_a = ['#230027', '#9C69A3', '#C698CB', '#C88FEC', '#EFD9F5']
    labels_a = ["Visual", "Easy", "Medium", "Hard"]
    order_a = ["VG", "SL", "SM", "SS"]

    df_MCDR_acc = df_MCDR.to_pandas().copy()
    df_MCDR_acc["ttype_plot"] = df_MCDR_acc["ttype_c"].replace({"DL": "SS", "DM": "SM", "DS": "SL"})
    accuracy_col = "correct_bool" if "correct_bool" in df_MCDR_acc.columns else "performance"
    subject_summary = (
        df_MCDR_acc[df_MCDR_acc["ttype_plot"].isin(order_a)]
        .groupby(["subject", "ttype_plot"], observed=True)[accuracy_col]
        .mean()
        .reset_index(name="subject_mean")
    )
    summary = (
        subject_summary.groupby("ttype_plot", observed=True)["subject_mean"]
        .agg(mean="mean", std="std", n="count")
        .reindex(order_a)
        .reset_index()
    )
    summary["sem"] = summary["std"].fillna(0.0) / summary["n"].clip(lower=1).pow(0.5)

    fig_MCDR_acc, ax_MCDR_acc = plt.subplots(figsize=fig_size(n_cols=2))
    x = list(range(len(order_a)))
    ax_MCDR_acc.plot(x, summary["mean"], color="tab:purple", linewidth=1.5, zorder=1)
    for xi, mean, sem, color in zip(x, summary["mean"], summary["sem"], palette_a, strict=False):
        ax_MCDR_acc.errorbar(
            xi,
            mean,
            yerr=sem,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=0,
            zorder=3,
        )
    ax_MCDR_acc.axhline(1 / 3, color="gray", ls="--")
    ax_MCDR_acc.axhspan(0.0, 1 / 3, color="gray", alpha=0.1, zorder=0)
    ax_MCDR_acc.set_xticks(x, labels=labels_a)
    ax_MCDR_acc.set_xlabel("Difficulty")
    ax_MCDR_acc.set_ylabel("Accuracy")
    ax_MCDR_acc.set_ylim(0.0, 1.0)
    ax_MCDR_acc.set_yticks([0.0, 1 / 3, 1.0])
    ax_MCDR_acc.set_yticklabels([0, 0.33, 1])
    plt.savefig('performance-3CDR.pdf')
    plt.show()

    fig_MCDR, ax_MCDR = plt.subplots(figsize = fig_size(3,1.25))
    # MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='', color = "tab:gray")
    # MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='',  color = "tab:pink")
    # MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='',  color = "tab:red")
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("batch") == "11B"),
        ax=ax_MCDR,
        figsize=fig_size(3, 1.25),
        title="",
        show_baseline_ttest=True,
    )
    # ax.set_ylim(0.2,0.6)

    save_fixed_bbox_pdf(fig_MCDR, 'MCDR_rb.pdf')
    plt.show()
    return


@app.cell
def _(df_MCDR, fig_size, plt):
    def _plot_mcdr_accuracy_by_category(df, *, x_col, order, labels, palette, filename, xlabel):
        accuracy_col = "correct_bool" if "correct_bool" in df.columns else "performance"
        subject_summary = (
            df[df[x_col].isin(order)]
            .groupby(["subject", x_col], observed=True)[accuracy_col]
            .mean()
            .reset_index(name="subject_mean")
        )
        summary = (
            subject_summary.groupby(x_col, observed=True)["subject_mean"]
            .agg(mean="mean", std="std", n="count")
            .reindex(order)
            .reset_index()
        )
        summary["sem"] = summary["std"].fillna(0.0) / summary["n"].clip(lower=1).pow(0.5)

        fig, ax = plt.subplots(figsize=fig_size(n_cols=2))
        x = list(range(len(order)))
        ax.plot(x, summary["mean"], color="tab:orange" if x_col == "stimd_c" else "tab:purple", linewidth=1.5, zorder=1)
        for xi, mean, sem, color in zip(x, summary["mean"], summary["sem"], palette, strict=False):
            ax.errorbar(
                xi,
                mean,
                yerr=sem,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=0,
                zorder=3,
            )
        ax.axhline(1 / 3, color="gray", ls="--")
        ax.axhspan(0.0, 1 / 3, color="gray", alpha=0.1, zorder=0)
        ax.set_xticks(x, labels=labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 1 / 3, 1.0])
        ax.set_yticklabels([0, 0.33, 1])
        plt.savefig(filename)
        plt.show()

    df = df_MCDR.to_pandas().copy()

    palette_b = ["#FFB74D", "#FB8C00", "#EF6C00"]
    labels_b = ["Short", "Med", "Long"]
    order_b = ["SS", "SM", "SL"]
    df_b = df[(df["ttype_c"] == "DS") & (df["stimd_c"].isin(order_b))]
    _plot_mcdr_accuracy_by_category(
        df_b,
        x_col="stimd_c",
        order=order_b,
        labels=labels_b,
        palette=palette_b,
        filename="MCDR_accuracy_stimd.pdf",
        xlabel="Stimulus duration",
    )

    palette_c = ["#5E2A7E", "#9C69A3", "#C698CB"]
    labels_c = ["Short", "Med", "Long"]
    order_c = ["DS", "DM", "DL"]
    df_c = df[df["stimd_c"] == "SS"]
    _plot_mcdr_accuracy_by_category(
        df_c,
        x_col="ttype_c",
        order=order_c,
        labels=labels_c,
        palette=palette_c,
        filename="MCDR_accuracy_ttype.pdf",
        xlabel="Delay duration",
    )
    return


@app.cell
def _(
    MCDR_plots,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    fig_size,
    pl,
    plt,
    two_afc_delay_plots,
    two_afc_plots,
):
    from matplotlib.lines import Line2D

    fig_rb_mosaic, axes_rb_mosaic = plt.subplot_mosaic(
        [["delay", "afc", "mcdr", "mcdr3"]],
        figsize=(12,3),
        layout="constrained",
    )

    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "NR2B"),
        ax=axes_rb_mosaic["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "Saline"),
        ax=axes_rb_mosaic["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "Rest"),
        ax=axes_rb_mosaic["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_mosaic["delay"].set_title("2AFC delay")

    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 1),
        ax=axes_rb_mosaic["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=axes_rb_mosaic["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=axes_rb_mosaic["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    axes_rb_mosaic["afc"].set_title("2AFC")

    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"),
        ax=axes_rb_mosaic["mcdr"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"),
        ax=axes_rb_mosaic["mcdr"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"),
        ax=axes_rb_mosaic["mcdr"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_mosaic["mcdr"].set_title("MCDR11")

    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "3B"),
        ax=axes_rb_mosaic["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "3B"),
        ax=axes_rb_mosaic["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "3B"),
        ax=axes_rb_mosaic["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_mosaic["mcdr3"].set_title("MCDR3B")

    for _ax in axes_rb_mosaic.values():
        _legend = _ax.get_legend()
        if _legend is not None:
            _legend.remove()

    fig_rb_mosaic.legend(
        handles=[
            Line2D([0], [0], color="tab:pink", marker="o", linewidth=1.5, label="Drug"),
            Line2D([0], [0], color="tab:gray", marker="o", linewidth=1.5, label="Saline"),
            Line2D([0], [0], color="tab:red", marker="o", linewidth=1.5, label="Rest"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
