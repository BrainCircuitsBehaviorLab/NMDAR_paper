import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from plot_saver import make_plot_saver
    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    sns.set_style("ticks")
    return Path, get_adapter, make_plot_saver, mo, pl, plt, sns


@app.cell
def _(Path):
    data_path = Path(__file__).parents[1] / "data/processed"
    data_path
    return (data_path,)


@app.cell
def _(Path, make_plot_saver, mo):
    project_path = Path(__file__).resolve().parents[1]
    save_plot = make_plot_saver(
        mo,
        results_dir=project_path / "results",
        config_path=project_path / "config.toml",
        task_name="figure1",
        model_id="behavior",
    )
    return (save_plot,)


@app.cell
def _(get_adapter):
    MCDR = get_adapter("MCDR")
    two_afc = get_adapter("2AFC")
    two_afc_delay = get_adapter("2AFC_delay")
    return MCDR, two_afc, two_afc_delay


@app.cell
def _(MCDR, data_path, pl, two_afc, two_afc_delay):
    df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "alexis_combined.parquet"))
    df_2AFC_delay = two_afc_delay.subject_filter(pl.read_parquet(data_path / "tiffany.parquet"))
    df_MCDR = MCDR.subject_filter(pl.read_parquet(data_path / "df_filtered.parquet"))
    return df_2AFC, df_2AFC_delay, df_MCDR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavior plots
    """)
    return


@app.cell
def _(
    MCDR,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    mo,
    save_plot,
    sns,
    two_afc,
    two_afc_delay,
):
    sns.set_context("paper")
    MCDR_plots = MCDR.get_plots()
    two_afc_plots = two_afc.get_plots()
    two_afc_delay_plots = two_afc_delay.get_plots()
    _figsize = (3.0, 3.0)
    _fig_mcdr = MCDR_plots.plot_accuracy_by_difficulty(df_MCDR, figsize=_figsize, title="MCDR")
    _fig_2afc = two_afc_plots.plot_accuracy_by_stimulus(df_2AFC, figsize=_figsize, title="2AFC")
    _fig_delay = two_afc_delay_plots.plot_accuracy_by_delay(df_2AFC_delay, figsize=_figsize, title="2AFC delay")
    mo.hstack(
        [
            mo.vstack(
                [
                    _fig_mcdr,
                    save_plot(_fig_mcdr, "MCDR accuracy", stem="mcdr_accuracy"),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    _fig_2afc,
                    save_plot(_fig_2afc, "2AFC accuracy", stem="two_afc_accuracy"),
                ],
                align="center",
            ),
            mo.vstack(
                [
                    _fig_delay,
                    save_plot(_fig_delay, "2AFC delay accuracy", stem="two_afc_delay_accuracy"),
                ],
                align="center",
            ),
        ],
        align="center",
    )
    return MCDR_plots, two_afc_delay_plots, two_afc_plots


@app.cell
def _(
    MCDR_plots,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    plt,
    sns,
    two_afc_delay_plots,
    two_afc_plots,
):
    fig, axs = plt.subplot_mosaic(
        [["mcdr"], ["two_afc"], ["delay"]],
        figsize=(2, 6),
        constrained_layout=True,
        sharey=True,
    )
    MCDR_plots.plot_accuracy_by_difficulty(df_MCDR, ax=axs["mcdr"], title="")
    two_afc_plots.plot_accuracy_by_stimulus(df_2AFC, ax=axs["two_afc"], title="")
    two_afc_delay_plots.plot_accuracy_by_delay(df_2AFC_delay, ax=axs["delay"], title="")
    sns.despine(fig=fig)
    fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
