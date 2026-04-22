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
    return Path, get_adapter, mo, pl, sns


@app.cell
def _(Path):
    data_path = Path(__file__).parents[1] / "data/processed"
    data_path
    return (data_path,)


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
def _(MCDR, df_2AFC, df_2AFC_delay, df_MCDR, mo, sns, two_afc, two_afc_delay):
    sns.set_context("paper")
    MCDR_plots = MCDR.get_plots()
    two_afc_plots = two_afc.get_plots()
    two_afc_delay_plots = two_afc_delay.get_plots()
    mo.hstack(
        [
            MCDR_plots.plot_accuracy_by_difficulty(df_MCDR),
            two_afc_plots.plot_accuracy_by_stimulus(df_2AFC),
            two_afc_delay_plots.plot_accuracy_by_delay(df_2AFC_delay),
        ],
        align="center",
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
