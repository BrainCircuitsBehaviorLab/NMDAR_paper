import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    from glmhmmt.runtime import get_runtime_paths
    from licks import add_lick_data
    from pathlib import Path
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    paths = get_runtime_paths()
    DATA_PATH = Path("/Users/javierrodriguezmartinez/NMDAR_paper/data/")
    return DATA_PATH, Path, add_lick_data, paths, pd, pl, plt, sns


@app.cell
def _(DATA_PATH, Path, pl):
    data_path = Path(__file__).parents[1] / "data"
    print(data_path)

    df = pl.read_csv(
        DATA_PATH / "raw" / "tiffany_complete.csv",
        infer_schema_length=None,
        ignore_errors=False,
    )
    df = df.filter(
        pl.col("valids"),
        pl.col("block").is_null(),
        pl.col("coherences").is_null(),
        pl.col("coherences_left").is_null(),
        pl.col("coherences_right").is_null(),
        pl.col("prob") == "Random",
        pl.col("presented_delays") == "0.01,1.0,3.0,10.0",
        pl.col("drug") != "Ephys",
    )

    df = df.with_columns(
        pl.when(pl.col("delays") == 0.01)
        .then(0.1)
        .otherwise(pl.col("delays"))
        .alias("delays")
    )

    df = df.rename(
        {"session_name": "session", "trials": "trial", "repeat_choice": "repeat"}
    )
    df = df.drop(
        [
            "valids",
            "block",
            "coherences",
            "coherences_left",
            "coherences_right",
            df.columns[0],
        ]
    )
    df
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(DATA_PATH, pl):
    licks_df = pl.read_csv(
        DATA_PATH / "raw" / "licks_2ADC.csv",
        infer_schema_length=None,
        schema_overrides={
            "C_s": pl.String,
            "C_e": pl.String,
            "L_s": pl.String,
            "L_e": pl.String,
            "ResponseWindow_start": pl.String,
            "ResponseWindow_end": pl.String,
        },
    )

    # Ambiguous session/trial keys cannot be assigned safely to one behavioral trial.
    _duplicate_lick_keys = (
        licks_df.group_by("session_name", "trials")
        .len()
        .filter(pl.col("len") > 1)
        .select("session_name", "trials")
    )
    _unique_licks = licks_df.join(
        _duplicate_lick_keys,
        on=["session_name", "trials"],
        how="anti",
    )

    # Translate Tiffany columns to the input contract expected by add_lick_data.
    licks_contract_df = (
        _unique_licks.select(
            "session_name",
            "trials",
            "C_s",
            "L_s",
            "ResponseWindow_start",
            "ResponseWindow_end",
        )
        .with_columns(
            pl.col("L_s")
            .fill_null("")
            .str.split(",")
            .list.eval(pl.element().cast(pl.Float64, strict=False))
            .list.drop_nulls()
            .alias("Port1In"),
            pl.col("C_s")
            .fill_null("")
            .str.split(",")
            .list.eval(pl.element().cast(pl.Float64, strict=False))
            .list.drop_nulls()
            .alias("Port2In"),
            pl.col("ResponseWindow_start")
            .str.split(",")
            .list.first()
            .cast(pl.Float64, strict=False)
            .alias("RespWinStart"),
            pl.col("ResponseWindow_end")
            .str.split(",")
            .list.first()
            .cast(pl.Float64, strict=False)
            .alias("RespWinEnd"),
        )
        .with_columns(
            pl.col("RespWinStart").alias("StimStart"),
            (pl.col("RespWinEnd") - pl.col("RespWinStart")).alias("RespWinLen"),
            pl.col("RespWinEnd").is_null().cast(pl.Int8).alias("Miss"),
        )
        .select(
            pl.col("session_name").alias("session"),
            pl.col("trials").alias("trial"),
            "Port1In",
            "Port2In",
            "RespWinStart",
            "RespWinEnd",
            "StimStart",
            "RespWinLen",
            "Miss",
        )
    )

    return (licks_contract_df,)


@app.cell
def _(add_lick_data, df, licks_contract_df, paths, pd, pl):
    _final_df_pd = (
        df.join(
            licks_contract_df,
            on=["session", "trial"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("Port1In").fill_null(pl.lit([], dtype=pl.List(pl.Float64))),
            pl.col("Port2In").fill_null(pl.lit([], dtype=pl.List(pl.Float64))),
            pl.col("Miss").fill_null(1),
        )
        .to_pandas()
    )
    _final_df_pd = pd.concat(
        [
            add_lick_data(_session_df.reset_index(drop=True))
            for _, _session_df in _final_df_pd.groupby("session", sort=False)
        ],
        ignore_index=True,
    )

    final_df = pl.from_pandas(_final_df_pd).drop(
        "Port1In",
        "Port2In",
        "RespWinStart",
        "RespWinEnd",
        "StimStart",
        "RespWinLen",
        "Miss",
    )
    final_df = final_df.with_columns(
        pl.col("subject")
        .str.to_uppercase()
        .str.extract(r"^([NEC])", 1)
        .alias("batch")
    )

    _output_path = paths.DATA_PATH / "tiffany_complete.parquet"
    final_df.write_parquet(_output_path)
    print(f"Saved to {_output_path}")
    return (final_df,)


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(final_df, plt, sns):
    plt.figure()
    sns.histplot(data=final_df, x="nLicks", stat="count", edgecolor=None)
    plt.yscale("log")
    plt.axhline(1, color="r", alpha=0.5)
    sns.despine()
    plt.show()
    return


@app.cell
def _(final_df, plt, sns):
    plt.figure()
    sns.histplot(data=final_df, x="nLicks", stat="count", edgecolor=None)
    plt.yscale("log")
    plt.axhline(1, color="r", alpha=0.5)
    plt.xlim(0, 500)
    sns.despine()
    plt.show()
    return


@app.cell
def _(final_df, plt, sns):
    plt.figure()
    _ax = sns.kdeplot(
        data=final_df,
        x="RT",
        hue="batch",
        palette = "Set2",
        common_norm=False,
    )
    _ax.legend_.set_frame_on(False)
    plt.xlim(0,1)
    sns.despine()
    plt.show()
    return


@app.cell
def _(final_df, plt, sns):
    plt.figure()
    _ax = sns.kdeplot(
        data=final_df,
        x="ILI",
        hue="batch",
        palette = "Set2",
        common_norm=False,
    )
    _ax.legend_.set_frame_on(False)
    plt.xlim(0,0.5)
    sns.despine()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
