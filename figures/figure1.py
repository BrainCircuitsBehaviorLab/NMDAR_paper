import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from plot_saver import make_plot_saver
    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths
    import os

    from src.utils import fig_size

    return Path, fig_size, get_adapter, make_plot_saver, mo, pl, plt, sns


@app.cell
def _(sns):
    # Set style
    sns.set_theme(style='ticks', context='notebook')
    # style_path = os.path.expanduser('~/PycharmProjects/alexis_style.mplstyle')
    # plt.style.use(style_path)
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
def _(MCDR, two_afc, two_afc_delay):
    MCDR_plots = MCDR.get_plots()
    two_afc_plots = two_afc.get_plots()
    two_afc_delay_plots = two_afc_delay.get_plots()
    return MCDR_plots, two_afc_delay_plots, two_afc_plots


@app.cell
def _(df_2AFC_delay, fig_size, plt, two_afc_delay_plots):
    # 2ADC
    two_afc_delay_plots.plot_accuracy(df_2AFC_delay, figsize=fig_size(n_cols=3), title='')
    plt.savefig('acc_vs_delay.svg')
    plt.show()

    two_afc_delay_plots.plot_rb(df_2AFC_delay, figsize=fig_size(n_cols=3), title='')
    plt.savefig('2ADC_rb.svg')
    plt.show()
    return


@app.cell
def _(df_2AFC, fig_size, plt, two_afc_plots):
    # 2AFC
    two_afc_plots.plot_accuracy(df_2AFC, figsize=fig_size(n_cols=3), title='')
    plt.savefig('acc_vs_ild.svg')
    plt.show()

    two_afc_plots.plot_rb(df_2AFC, figsize=fig_size(n_cols=3), title='')
    plt.savefig('2AFC_rb.svg')
    plt.show()
    return


@app.cell
def _(MCDR_plots, df_MCDR, fig_size, plt):
    MCDR_plots.plot_accuracy(df_MCDR, figsize=fig_size(n_cols=3), title='')
    plt.savefig('acc_vs_difficulty.svg')
    plt.show()

    MCDR_plots.plot_rb(df_MCDR, figsize=fig_size(n_cols=3), title='')
    plt.savefig('MCDR_rb.svg')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
