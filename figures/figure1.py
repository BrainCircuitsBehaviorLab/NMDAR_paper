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
        np,
        pd,
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
def _(np, pd, plot_mean_over_data):
    def psychometric_repeat(
        plot_df,
        ax=None,
        figsize=(3.0, 3.0),
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
            x_tick_labels = ["0", "1", "3", "10", "10", "3", "1", "0"]

            delay_magnitude = pd.Series(x_values, index=plot_df.index).abs()
            delay_label = np.where(
                np.isclose(delay_magnitude, 0.1),
                "0.1",
                delay_magnitude.round().astype("Int64").astype(str),
            )
            delay_side = np.where(x_values < 0, "neg", "pos")
            x_values = pd.Series(delay_side, index=plot_df.index) + "_" + pd.Series(delay_label, index=plot_df.index)
            xlabel = "Delay signed by prev. choice"
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
            xlabel = "Dif. signed by prev. choice"
        else:
            x_values = signed_stimulus * plot_df["previous_choice_sign"]
            x_values = pd.Series(x_values, index=plot_df.index).mask(lambda values: np.isclose(values, 0.0), 0.0)
            xlabel = "Stim. signed by prev. choice"

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
            baseline_area=True,
            color=color,
            ax=ax,
            figsize=figsize,
        )

    return (psychometric_repeat,)


@app.cell
def _(
    attach_signed_delay_columns,
    df_2AFC_delay,
    fig_size,
    pl,
    plot_mean_over_data,
    plt,
    psychometric_repeat,
    save_fixed_bbox_pdf,
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

    fig_2ADC_repeat, ax_2ADC_repeat = plt.subplots(figsize=fig_size(2))
    psychometric_repeat(
        df_2AFC_delay.filter(pl.col("drug") == "NR2B"),
        ax=ax_2ADC_repeat,
        figsize=fig_size(2, 1.25),
        title="",
        color="tab:pink",
        session_col="session",
        trial_col="trial",
        choice_col="choices",
        stimulus_col="stim",
        delay_col="delays",
    )
    psychometric_repeat(
        df_2AFC_delay.filter(pl.col("drug") == "Saline"),
        ax=ax_2ADC_repeat,
        figsize=fig_size(2, 1.25),
        title="",
        color="tab:gray",
        session_col="session",
        trial_col="trial",
        choice_col="choices",
        stimulus_col="stim",
        delay_col="delays",
    )
    save_fixed_bbox_pdf(fig_2ADC_repeat, "2ADC_psychometric_repeat.pdf")
    plt.show()
    return


@app.cell
def _(
    df_2AFC,
    fig_size,
    pl,
    plot_mean_over_data,
    plt,
    psychometric_repeat,
    save_fixed_bbox_pdf,
    two_afc_plots,
):
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

    fig_2AFC_repeat, ax_2AFC_repeat = plt.subplots(figsize=fig_size(2))
    psychometric_repeat(
        df_2AFC.filter(pl.col("Drug") == 1),
        ax=ax_2AFC_repeat,
        figsize=fig_size(2, 1.25),
        title="",
        color="tab:pink",
        session_col="Session",
        trial_col="Trial",
        choice_col="Choice",
        stimulus_col="ILD",
    )
    psychometric_repeat(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=ax_2AFC_repeat,
        figsize=fig_size(2, 1.25),
        title="",
        color="tab:gray",
        session_col="Session",
        trial_col="Trial",
        choice_col="Choice",
        stimulus_col="ILD",
    )
    save_fixed_bbox_pdf(fig_2AFC_repeat, "2AFC_psychometric_repeat.pdf")
    plt.show()
    return


@app.cell
def _(
    MCDR_plots,
    df_MCDR,
    fig_size,
    pl,
    plt,
    psychometric_repeat,
    save_fixed_bbox_pdf,
):
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

    fig_MCDR, ax_MCDR = plt.subplots(figsize = fig_size(2))
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='', color = "tab:gray")
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='',  color = "tab:pink")
    # MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='',  color = "tab:red")
    # MCDR_plots.plot_rb(
    #     df_MCDR.filter(pl.col("batch") == "11B"),
    #     ax=ax_MCDR,
    #     figsize=fig_size(2),
    #     title="",
    #     show_baseline_ttest=True,
    # )
    # ax.set_ylim(0.2,0.6)

    save_fixed_bbox_pdf(fig_MCDR, 'MCDR_rb.pdf')
    plt.show()

    fig_MCDR_repeat, ax_MCDR_repeat = plt.subplots(figsize=fig_size(2))
    psychometric_repeat(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"),
        ax=ax_MCDR_repeat,
        figsize=fig_size(2, 1.25),
        title="",
        color="tab:gray",
        session_col="session",
        trial_col="trial",
        choice_col="response",
        stimulus_col="stimulus",
        difficulty_col="ttype_c",
        is_mcdr=True,
    )
    psychometric_repeat(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"),
        ax=ax_MCDR_repeat,
        figsize=fig_size(2, 1.25),
        title="",
        color="tab:pink",
        session_col="session",
        trial_col="trial",
        choice_col="response",
        stimulus_col="stimulus",
        difficulty_col="ttype_c",
        is_mcdr=True,
    )
    save_fixed_bbox_pdf(fig_MCDR_repeat, "MCDR_psychometric_repeat.pdf")
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
    return (Line2D,)


@app.cell
def _(
    Line2D,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    fig_size,
    np,
    pl,
    plt,
    psychometric_repeat,
):

    _panel_size = fig_size(3)
    fig_repeat_mosaic, axes_repeat_mosaic = plt.subplot_mosaic(
        [["delay", "afc", "mcdr"]],
        figsize=(fig_size(1,3)),
        layout="constrained",
    )

    psychometric_repeat(
        df_2AFC_delay.filter(pl.col("drug") == "NR2B"),
        ax=axes_repeat_mosaic["delay"],
        figsize=fig_size(3),
        title="",
        color="tab:pink",
        session_col="session",
        trial_col="trial",
        choice_col="choices",
        stimulus_col="stim",
        delay_col="delays",
    )
    psychometric_repeat(
        df_2AFC_delay.filter(pl.col("drug") == "Saline"),
        ax=axes_repeat_mosaic["delay"],
        figsize=fig_size(3),
        title="",
        color="tab:gray",
        session_col="session",
        trial_col="trial",
        choice_col="choices",
        stimulus_col="stim",
        delay_col="delays",
    )
    axes_repeat_mosaic["delay"].set_title("2AFC delay")

    psychometric_repeat(
        df_2AFC.filter(pl.col("Drug") == 1),
        ax=axes_repeat_mosaic["afc"],
        figsize=fig_size(3),
        title="",
        color="tab:pink",
        session_col="Session",
        trial_col="Trial",
        choice_col="Choice",
        stimulus_col="ILD",
    )
    psychometric_repeat(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=axes_repeat_mosaic["afc"],
        figsize=fig_size(3),
        title="",
        color="tab:gray",
        session_col="Session",
        trial_col="Trial",
        choice_col="Choice",
        stimulus_col="ILD",
    )
    axes_repeat_mosaic["afc"].set_title("2AFC")
    axes_repeat_mosaic["afc"].set_xticks(
        axes_repeat_mosaic["afc"].get_xticks(),
        [
            "" if any(np.isclose(_tick, _hidden_tick) for _hidden_tick in [-4, -2,2,4]) else _label.get_text()
            for _tick, _label in zip(
                axes_repeat_mosaic["afc"].get_xticks(),
                axes_repeat_mosaic["afc"].get_xticklabels(),
                strict=False,
            )
        ],
    )

    psychometric_repeat(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"),
        ax=axes_repeat_mosaic["mcdr"],
        figsize=fig_size(3),
        title="",
        color="tab:gray",
        session_col="session",
        trial_col="trial",
        choice_col="response",
        stimulus_col="stimulus",
        difficulty_col="ttype_c",
        is_mcdr=True,
    )
    psychometric_repeat(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"),
        ax=axes_repeat_mosaic["mcdr"],
        figsize=fig_size(3),
        title="",
        color="tab:pink",
        session_col="session",
        trial_col="trial",
        choice_col="response",
        stimulus_col="stimulus",
        difficulty_col="ttype_c",
        is_mcdr=True,
    )
    axes_repeat_mosaic["mcdr"].set_title("MCDR")

    for _ax in axes_repeat_mosaic.values():
        _legend = _ax.get_legend()
        if _legend is not None:
            _legend.remove()

    axes_repeat_mosaic["mcdr"].set_ylabel("")
    axes_repeat_mosaic["afc"].set_ylabel("")
    fig_repeat_mosaic.legend(
        handles=[
            Line2D([0], [0], color="tab:pink", marker="o", linewidth=1.5, label="Drug"),
            Line2D([0], [0], color="tab:gray", marker="o", linewidth=1.5, label="Saline"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 2AFC session raster
    """)
    return


@app.cell
def _(mo):
    pick_random_2afc_session = mo.ui.button(
        value=0,
        on_click=lambda value: (value or 0) + 1,
        label="Pick random session",
    )
    pick_random_2afc_session
    return (pick_random_2afc_session,)


@app.cell
def _(df_2AFC, mo, np, pd, pick_random_2afc_session, pl, plt, sns):
    def _repeat_counts_by_session(df):
        _pdf = (
            df.select(["subject", "Session", "Trial", "Choice"])
            .sort(["subject", "Session", "Trial"])
            .to_pandas()
        )
        _pdf["Choice"] = pd.to_numeric(_pdf["Choice"], errors="coerce")
        _pdf["_previous_choice"] = _pdf.groupby(["subject", "Session"], observed=True)["Choice"].shift(1)
        _pdf["_repeat_choice"] = (
            _pdf["Choice"].notna()
            & _pdf["_previous_choice"].notna()
            & (_pdf["Choice"] == _pdf["_previous_choice"])
        )
        return (
            _pdf.groupby(["subject", "Session"], observed=True)
            .agg(
                n_trials=("Choice", "size"),
                n_repetitions=("_repeat_choice", "sum"),
            )
            .reset_index()
            .sort_values(["subject", "Session"], kind="stable")
            .reset_index(drop=True)
        )

    def _selected_2afc_session(df, session_summary, click_count):
        _session_summary = session_summary[session_summary["n_trials"] > 1].reset_index(drop=True)
        mo.stop(_session_summary.empty, mo.md("No 2AFC sessions with at least two trials."))

        _rng = np.random.default_rng(int(click_count or 0))
        _row_idx = int(_rng.integers(0, len(_session_summary)))
        _row = _session_summary.iloc[_row_idx].to_dict()
        _session_df = (
            df.filter(
                (pl.col("subject").cast(pl.Utf8) == str(_row["subject"]))
                & (pl.col("Session").cast(pl.Utf8) == str(_row["Session"]))
            )
            .sort("Trial")
            .to_pandas()
            .reset_index(drop=True)
        )
        return _row, _session_df

    def _compute_repeat_choice(session_df):
        _choice = pd.to_numeric(session_df["Choice"], errors="coerce")
        _prev_choice = _choice.shift(1)
        return (_choice.notna() & _prev_choice.notna() & (_choice == _prev_choice)).to_numpy()

    def _plot_session_raster(session_meta, session_df, repeat_choice, session_summary, *, chunk_size=20):
        _n_trials = len(session_df)
        _trial_x = np.arange(_n_trials, dtype=float)
        _hit = pd.to_numeric(session_df["Hit"], errors="coerce")
        _repeat_correct = repeat_choice & (_hit.fillna(0).to_numpy(dtype=float) > 0)
        _repeat_incorrect = repeat_choice & ~_repeat_correct

        _fig, (_ax_raster, _ax_chunk, _ax_hist) = plt.subplots(
            3,
            1,
            figsize=(8,8),
            gridspec_kw={"height_ratios": [1.0, 1.2, 1.0]},
            sharex=False,
            layout="constrained",
        )

        _trial_colors = np.full(_n_trials, "tab:gray", dtype=object)
        _trial_colors[_repeat_incorrect] = "tab:blue"
        _trial_colors[_repeat_correct] = "#084594"

        _ax_raster.bar(
            _trial_x,
            np.ones(_n_trials),
            width=1.0,
            align="edge",
            color=_trial_colors,
            linewidth=0,
        )
        _ax_raster.set_xlim(0, _n_trials)
        _ax_raster.set_ylim(0, 1.0)
        _ax_raster.set_yticks([])
        _ax_raster.set_ylabel("Trials")
        _session_label = str(session_meta["Session"])
        _ax_raster.set_title(
            f"subject {session_meta['subject']} | {_session_label} | "
            f"{_n_trials} trials | {int(repeat_choice.sum())} repeats"
        )

        _legend_handles = [
            plt.Line2D([0], [0], color="tab:gray", lw=6, label="Alternate"),
            plt.Line2D([0], [0], color="tab:blue", lw=6, label="Repeat incorrect"),
            plt.Line2D([0], [0], color="#084594", lw=6, label="Repeat correct"),
        ]
        _ax_raster.legend(
            handles=_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
            frameon=False,
        )

        _chunk_idx = (_trial_x // chunk_size).astype(int)
        _n_chunks = int(_chunk_idx.max()) + 1 if _n_trials else 0
        _incorrect_repeat_counts = np.bincount(
            _chunk_idx,
            weights=_repeat_incorrect.astype(int),
            minlength=_n_chunks,
        )
        _chunk_centers = (np.arange(_n_chunks) * chunk_size) + (chunk_size / 2)
        _cumulative_incorrect_repeats = np.cumsum(_incorrect_repeat_counts)

        _ax_chunk.bar(
            _chunk_centers,
            _incorrect_repeat_counts,
            width=chunk_size * 0.8,
            color="tab:blue",
            label=f"Incorrect repeats per {chunk_size}-trial chunk",
        )
        _ax_cumulative = _ax_chunk.twinx()
        _ax_cumulative.plot(
            _chunk_centers,
            _cumulative_incorrect_repeats,
            color="#222222",
            marker="o",
            label="Cumulative incorrect repeats",
        )

        _ax_chunk.set_xlim(0, _n_trials)
        _ax_chunk.set_xlabel("Trial in session")
        _ax_chunk.set_ylabel("Incorrect repeat count")
        _ax_cumulative.set_ylabel("Cumulative incorrect repeats")
        _ax_chunk.set_title(f"Incorrect repeat-choice count by {chunk_size}-trial chunk")

        _max_count = max(1, int(np.nanmax(_incorrect_repeat_counts)) if len(_incorrect_repeat_counts) else 1)
        _ax_chunk.set_ylim(0, _max_count + 1)
        _ax_cumulative.set_ylim(0, max(1, int(_cumulative_incorrect_repeats[-1]) if len(_cumulative_incorrect_repeats) else 1) + 1)

        _handles, _labels = _ax_chunk.get_legend_handles_labels()
        _handles_2, _labels_2 = _ax_cumulative.get_legend_handles_labels()
        _ax_chunk.legend(
            _handles + _handles_2,
            _labels + _labels_2,
            loc="upper left",
            frameon=False,
        )

        _session_repetitions = pd.to_numeric(session_summary["n_repetitions"], errors="coerce").dropna()
        _selected_repetitions = int(session_meta["n_repetitions"])
        _hist_max = max(int(_session_repetitions.max()), _selected_repetitions, 1)
        _bin_width = max(5, int(np.ceil(_hist_max / 20)))
        _bins = np.arange(0, _hist_max + _bin_width + 1, _bin_width)
        _ax_hist.hist(
            _session_repetitions,
            bins=_bins,
            color="tab:green",
        )
        _ax_hist.axvline(
            _selected_repetitions,
            color="black",
            label="Selected session",
        )
        _ax_hist.set_xlabel("Repeat choices per session")
        _ax_hist.set_ylabel("Sessions")
        _ax_hist.set_title("Distribution of session repeat-choice counts")
        _ax_hist.legend(loc="upper right", frameon=False)

        sns.despine(ax=_ax_raster, left=True, bottom=True)
        sns.despine(ax=_ax_chunk)
        sns.despine(ax=_ax_cumulative, left=True, right=False)
        sns.despine(ax=_ax_hist)
        return _fig

    _session_repeat_summary = _repeat_counts_by_session(df_2AFC)
    _session_meta, _session_df = _selected_2afc_session(
        df_2AFC,
        _session_repeat_summary,
        pick_random_2afc_session.value,
    )
    _repeat_choice = _compute_repeat_choice(_session_df)
    _fig = _plot_session_raster(
        _session_meta,
        _session_df,
        _repeat_choice,
        _session_repeat_summary,
    )
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
