# /// script
# [tool.marimo.opengraph]
# title = "Figure 1" 
# description = " Figure 1: Behavioral performance across tasks."
# ///

import marimo

__generated_with = "0.23.9"
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
    import os

    from plot_saver import make_plot_saver
    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths
    from glmhmmt.tasks.fitted_regressors import FittedWeightRegressorSpec, mean_feature_weights_from_fit
    from src.process.common import attach_signed_delay_columns
    from src.plots.common import plot_mean_over_data, psychometric_repeat, fig_size

    return (
        Path,
        attach_signed_delay_columns,
        configure_paths,
        fig_size,
        get_adapter,
        make_plot_saver,
        mo,
        np,
        os,
        pd,
        pl,
        plot_mean_over_data,
        plt,
        psychometric_repeat,
        sns,
    )


@app.cell
def _(Path, fig_size, plt, sns):
    # Set style
    sns.set_theme(style='ticks', context='notebook')
    plt.style.use(Path(__file__).resolve().parents[1] / "paper.mplstyle")
    figsize = fig_size(n_cols=3)
    plt.rcParams["svg.fonttype"] = 'none'
    return (figsize,)


@app.cell
def _(Path, configure_paths, os):
    # Set paths
    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")

    data_path = Path(__file__).parents[1] / "data/processed"
    print(data_path)

    project_path = Path(__file__).resolve().parents[1]
    print(project_path)

    path_panels = project_path / "figures" / "panels1"
    print(path_panels)
    os.makedirs(path_panels, exist_ok=True)
    return data_path, path_panels, project_path


@app.cell
def _(make_plot_saver, mo, project_path):
    save_plot = make_plot_saver(
        mo,
        results_dir=project_path / "results",
        config_path=project_path / "config.toml",
        task_name="figure1",
        model_id="behavior",
    )
    return


@app.cell
def _(get_adapter):
    # Get adapters
    MCDR = get_adapter("MCDR")
    two_afc = get_adapter("2AFC")
    two_afc_delay = get_adapter("2AFC_delay")
    return MCDR, two_afc, two_afc_delay


@app.cell
def _(MCDR, data_path, pl, two_afc):
    # Import data
    # df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "alexis_combined.parquet"))
    df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "df_alexis_drug_combined.parquet"))  # With drug
    df_2AFC_delay = pl.read_parquet(data_path / "tiffany.parquet")
    df_MCDR = MCDR.subject_filter(pl.read_parquet(data_path / "MCDR_all.parquet"))

    # Compute r_right for MCDR p right
    df_MCDR = df_MCDR.with_columns((pl.col("r_c") == "R").cast(pl.Int8).alias("r_right"))
    df_MCDR = df_MCDR.with_columns(
        pl.when(pl.col("x_c") == "R")
        .then(pl.lit("pos_") + pl.col("ttype_c"))
        .otherwise(pl.lit("neg_") + pl.col("ttype_c"))
        .alias("signed_ttype_c")
    )
    # df_MCDR = df_MCDR.filter(pl.col("batch") == "11B")
    return df_2AFC, df_2AFC_delay, df_MCDR


@app.cell
def _(df_MCDR):
    df_MCDR
    return


@app.cell
def _(MCDR, two_afc, two_afc_delay):
    # Get plots
    MCDR_plots = MCDR.get_plots()
    two_afc_plots = two_afc.get_plots()
    two_afc_delay_plots = two_afc_delay.get_plots()
    return MCDR_plots, two_afc_delay_plots, two_afc_plots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2ADC
    Two-alternative delayed-response task
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy
    """)
    return


@app.cell
def _(df_2AFC_delay, figsize, path_panels, pl, plt, two_afc_delay_plots):
    plt.figure(figsize=figsize, constrained_layout=True)
    acc_2ADC = plt.gca()
    two_afc_delay_plots.plot_accuracy(df_2AFC_delay.filter(pl.col("drug") == 'Saline'), ax=acc_2ADC, color="tab:gray", title="", label='Saline')
    two_afc_delay_plots.plot_accuracy(df_2AFC_delay.filter(pl.col("drug") == 'NR2B'), ax=acc_2ADC, color="tab:pink", title="", label='Drug')
    plt.ylim(0.45, 1)
    plt.savefig(f'{path_panels}/acc_2ADC.svg')
    plt.savefig(f'{path_panels}/acc_2ADC.png')
    acc_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PC right
    """)
    return


@app.cell
def _(
    attach_signed_delay_columns,
    df_2AFC_delay,
    fig_size,
    path_panels,
    plot_mean_over_data,
    plt,
):
    signed_delay_order = ["0L", "-1", "-3", "-10", "10", "3", "1", "0R"]
    signed_delay_tick_labels = ["-0.1", "-1", "-3", "-10", "10", "3", "1", "0.1"]
    df_2AFC_delay_signed = attach_signed_delay_columns(df_2AFC_delay.to_pandas())
    df_2AFC_delay_signed["p_right"] = (df_2AFC_delay_signed["choices"] > 0).astype(float)
    df_2AFC_delay_signed["_signed_delay_plot"] = df_2AFC_delay_signed["_signed_delay_cat"].astype(str)
    df_2AFC_delay_signed = df_2AFC_delay_signed[df_2AFC_delay_signed["_signed_delay_plot"].isin(signed_delay_order)].copy()

    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    p_right_2ADC = plt.gca()

    plot_mean_over_data(
        df_2AFC_delay_signed[df_2AFC_delay_signed.drug=="Saline"],
        x_col="_signed_delay_plot",
        x_order=signed_delay_order,
        x_tick_labels=signed_delay_tick_labels,
        y_col="p_right",
        xlabel="Delay (s)",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=0.5,
        baseline_area=False,
        color="tab:gray",
        ax=p_right_2ADC,
    )

    plot_mean_over_data(
        df_2AFC_delay_signed[df_2AFC_delay_signed.drug=="NR2B"],
        x_col="_signed_delay_plot",
        x_order=signed_delay_order,
        x_tick_labels=signed_delay_tick_labels,
        y_col="p_right",
        xlabel="Delay (s)",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=0.5,
        baseline_area=False,
        color="tab:pink",
        ax=p_right_2ADC,
    )

    plt.savefig(f'{path_panels}/p_right_2ADC.svg')
    plt.savefig(f'{path_panels}/p_right_2ADC.pdf', transparent=True,)
    p_right_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Repeating bias
    """)
    return


@app.cell
def _(df_2AFC_delay, fig_size, path_panels, pl, plt, two_afc_delay_plots):
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    rb_2ADC = plt.gca()
    two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "NR2B"), ax = rb_2ADC, title='', color = "tab:pink")
    two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "Saline"), ax = rb_2ADC, title='', color = "tab:gray")
    # two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "Rest"), ax = rb_2ADC, title='', color = "k")
    plt.ylim(0.45, 1)
    plt.savefig(f'{path_panels}/rb_2ADC.svg')
    plt.savefig(f'{path_panels}/rb_2ADC.pdf', transparent=True,)
    rb_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PC repeat
    """)
    return


@app.cell
def _(df_2AFC_delay, figsize, path_panels, pl, plt, psychometric_repeat):
    plt.figure(figsize=figsize, constrained_layout=True)
    p_rep_2ADC = plt.gca()

    psychometric_repeat(
        df_2AFC_delay.filter(pl.col("drug") == "NR2B"),
        ax=p_rep_2ADC,
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
        ax=p_rep_2ADC,
        title="",
        color="tab:gray",
        session_col="session",
        trial_col="trial",
        choice_col="choices",
        stimulus_col="stim",
        delay_col="delays",
    )

    plt.savefig(f'{path_panels}/p_rep_2ADC.svg')
    plt.savefig(f'{path_panels}/p_rep_2ADC.png')
    p_rep_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2AFC
    Two-alternative forced-choice (Alexis)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy
    """)
    return


@app.cell
def _(df_2AFC, figsize, path_panels, pl, plt, two_afc_plots):
    plt.figure(figsize=figsize, constrained_layout=True)
    acc_2AFC = plt.gca()
    two_afc_plots.plot_accuracy(df_2AFC.filter(pl.col("Drug") == 0), ax=acc_2AFC, color="tab:gray", title="", label='Saline')
    two_afc_plots.plot_accuracy(df_2AFC.filter(pl.col("Drug") == 1), ax=acc_2AFC, color="tab:pink", title="", label='Drug')
    plt.ylim(0.45, 1)
    plt.savefig(f'{path_panels}/acc_2AFC.svg')
    plt.savefig(f'{path_panels}/acc_2AFC.png')
    acc_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PC right
    """)
    return


@app.cell
def _(df_2AFC, fig_size, path_panels, plot_mean_over_data, plt):
    df_2AFC_p_right = df_2AFC.to_pandas().copy()
    df_2AFC_p_right["p_right"] = df_2AFC_p_right["Choice"].astype(float)

    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    p_right_2AFC = plt.gca()

    plot_mean_over_data(
        df_2AFC_p_right[df_2AFC_p_right.Drug==0],
        x_col="ILD",
        y_col="p_right",
        xlabel="ILD (dB)",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=0.5,
        baseline_area=False,
        color="tab:gray",
        ax=p_right_2AFC
    )

    plot_mean_over_data(
        df_2AFC_p_right[df_2AFC_p_right.Drug==1],
        x_col="ILD",
        y_col="p_right",
        xlabel="ILD (dB)",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=0.5,
        baseline_area=False,
        color="tab:pink",
        ax=p_right_2AFC
    )

    plt.gca().set_xticks(
        [-20, -8, -4, -2, 0, 2, 4, 8, 20],
        labels=["-20", "-8", "", "", "0", "", "", "8", "20"],
    )

    plt.savefig(f'{path_panels}/p_right_2AFC.svg')
    plt.savefig(f'{path_panels}/p_right_2AFC.pdf', transparent=True,)
    p_right_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Repeating bias
    """)
    return


@app.cell
def _(df_2AFC, fig_size, path_panels, pl, plt, two_afc_plots):
    plt.figure(figsize=fig_size(2,1), constrained_layout=True)
    rb_2AFC = plt.gca()
    two_afc_plots.plot_rb(df_2AFC.filter(pl.col("Drug") == 0), ax=rb_2AFC, title="", color="tab:gray")
    two_afc_plots.plot_rb(df_2AFC.filter(pl.col("Drug") == 1), ax=rb_2AFC, title="", color="tab:pink")
    plt.ylim(0.45, 1)
    plt.savefig(f'{path_panels}/rb_2AFC.svg')
    plt.savefig(f'{path_panels}/rb_2AFC.pdf', transparent=True,)
    rb_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PC repeat
    """)
    return


@app.cell
def _(df_2AFC, figsize, path_panels, pl, plt, psychometric_repeat):
    plt.figure(figsize=figsize, constrained_layout=True)
    p_rep_2AFC = plt.gca()

    psychometric_repeat(
        df_2AFC.filter(pl.col("Drug") == 1),
        ax=p_rep_2AFC,
        title="",
        color="tab:pink",
        session_col="Session",
        trial_col="Trial",
        choice_col="Choice",
        stimulus_col="ILD",
    )
    psychometric_repeat(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=p_rep_2AFC,
        title="",
        color="tab:gray",
        session_col="Session",
        trial_col="Trial",
        choice_col="Choice",
        stimulus_col="ILD",
    )

    plt.gca().set_xticks([-20, -8, -4, -2, 0, 2, 4, 8, 20],
        labels=["-20", "-8", "", "", "0", "", "", "8", "20"])

    plt.savefig(f'{path_panels}/p_rep_2AFC.svg')
    plt.savefig(f'{path_panels}/p_rep_2AFC.png')
    p_rep_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MCDR
    Multiple-choice delayed-response task (Balma)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy
    """)
    return


@app.cell
def _(df_MCDR, figsize, path_panels, pl, plot_mean_over_data, plt):
    plt.figure(figsize=figsize, constrained_layout=True)
    acc_MCDR = plt.gca()

    plot_mean_over_data(
        df_MCDR.filter(pl.col("drug") == "saline"),
        x_col="ttype_c",
        y_col="performance",
        x_order=["VG", "DS", "DM", "DL"],
        x_tick_labels=["VG", "EZ", "Med.", "Hard"],
        xlabel="Difficulty",
        ylabel="Accuracy",
        title="",
        baseline=1/3,
        baseline_area=False,
        color="tab:gray",
        ax=acc_MCDR,
    )

    plot_mean_over_data(
        df_MCDR.filter(pl.col("drug") == "drug"),
        x_col="ttype_c",
        y_col="performance",
        x_order=["VG", "DS", "DM", "DL"],
        x_tick_labels=["VG", "Easy", "Med.", "Hard"],
        xlabel="Difficulty",
        ylabel="Accuracy",
        title="",
        baseline=1/3,
        baseline_area=False,
        color="tab:pink",
        ax=acc_MCDR,
    )
    plt.ylim(0.3, 1)
    plt.savefig(f'{path_panels}/acc_MCDR.svg')
    plt.savefig(f'{path_panels}/acc_MCDR.png')
    acc_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PC right
    """)
    return


@app.cell
def _():
    difficulties = ["VG", "DS", "DM", "DL"]

    signed_order =[
        "neg_VG", "neg_DS", "neg_DM", "neg_DL",
        "pos_DL", "pos_DM", "pos_DS", "pos_VG",
    ]

    signed_labels = [

        "-VG", "-DS", "-DM", "-DL",

        "DL", "DM", "DS", "VG",

    ]
    return signed_labels, signed_order


@app.cell
def _(
    df_MCDR,
    fig_size,
    path_panels,
    pl,
    plot_mean_over_data,
    plt,
    signed_labels,
    signed_order,
):
    plt.figure(figsize=fig_size(2), constrained_layout=True)
    p_right_3CDR = plt.gca()

    plot_mean_over_data(
        df_MCDR.filter(pl.col("drug") == "rest"),
        x_col="signed_ttype_c",
        y_col="r_right",
        x_order=signed_order,
        x_tick_labels=signed_labels,
        xlabel="Difficulty",
        ylabel=r"$p(\mathrm{right})$",
        title="",
        baseline=1/3,
        baseline_area=True,
        color="tab:blue",
        ax=p_right_3CDR,
    )

    # plot_mean_over_data(
    #     df_MCDR.filter(pl.col("drug") == "drug"),
    #     x_col="signed_ttype_c",
    #     y_col="r_right",
    #     x_order=signed_order,
    #     x_tick_labels=signed_labels,
    #     xlabel="Difficulty",
    #     ylabel=r"$p(\mathrm{right})$",
    #     title="",
    #     baseline=1/3,
    #     baseline_area=True,
    #     color="tab:pink",
    #     ax=p_right_3CDR,
    # )

    plt.savefig(f'{path_panels}/p_right_3CDR.svg')
    plt.savefig(f'{path_panels}/p_right_3CDR.png')
    p_right_3CDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Repeating bias
    """)
    return


@app.cell
def _(MCDR_plots, df_MCDR, figsize, path_panels, pl, plt):
    plt.figure(figsize=figsize, constrained_layout=True)
    rb_MCDR = plt.gca()
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"), ax=rb_MCDR, title='', color="tab:gray")
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"), ax=rb_MCDR, title='', color="tab:pink")
    # MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"), ax=rb_MCDR, , title='', color="tab:red")
    # MCDR_plots.plot_rb(df_MCDR.filter(pl.col("batch") == "11B"), ax=rb_MCDR, , title="", color='k')
    plt.ylim(0.3, 1)
    plt.savefig(f'{path_panels}/rb_MCDR.svg')
    plt.savefig(f'{path_panels}/rb_MCDR.png')
    rb_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PC repeat
    """)
    return


@app.cell
def _(df_MCDR, figsize, path_panels, pl, plt, psychometric_repeat):
    plt.figure(figsize=figsize, constrained_layout=True)
    p_rep_MCDR = plt.gca()

    psychometric_repeat(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"),
        ax=p_rep_MCDR,
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
        ax=p_rep_MCDR,
        title="",
        color="tab:pink",
        session_col="session",
        trial_col="trial",
        choice_col="response",
        stimulus_col="stimulus",
        difficulty_col="ttype_c",
        is_mcdr=True,
    )

    plt.savefig(f'{path_panels}/p_rep_MCDR.svg')
    plt.savefig(f'{path_panels}/p_rep_MCDR.png')
    p_rep_MCDR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example session raster (2AFC)
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


if __name__ == "__main__":
    app.run()
