import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    from glmhmmt.runtime import get_runtime_paths
    from pathlib import Path
    paths = get_runtime_paths()
    DATA_PATH = Path("/Users/javierrodriguezmartinez/NMDAR_paper/data/")
    return DATA_PATH, Path, paths, pl


@app.cell
def _(DATA_PATH, Path, paths, pl):
    data_path = Path(__file__).parents[1] / "data"
    print(data_path)

    df = pl.read_csv(
        DATA_PATH / "raw" / "tiffany_complete.csv",
        infer_schema_length=None,
        ignore_errors=False,
    )
    df = df.filter(
        pl.col("valids") == True,
        pl.col("block").is_null(),
        pl.col("coherences").is_null(),
        pl.col("coherences_left").is_null(),
        pl.col("coherences_right").is_null(),
        pl.col("prob") == "Random",
        pl.col("presented_delays") == "0.01,1.0,3.0,10.0",
        pl.col("drug") != "Ephys",
    )

    df = df.with_columns(pl.when(pl.col("delays") == 0.01).then(0.1).otherwise(pl.col("delays")).alias("delays"))

    df = df.rename({"session_name": "session", "trials": "trial", "repeat_choice": "repeat"})
    df = df.drop(
        [
            "valids",
            "block",
            "coherences",
            "coherences_left",
            "coherences_right",
            df.columns[0]
        ]
    )
    df.write_parquet(paths.DATA_PATH / "tiffany_complete.parquet")
    print(paths.DATA_PATH / "tiffany_complete.parquet")
    df
    return (df,)


@app.cell
def _(df, pl):
    df.group_by("presented_delays").agg(
        [
            pl.len().alias("count"),
            pl.col("subject").n_unique().alias("n_subjects"),
        ]
    ).sort("count", descending=True)
    return


@app.cell
def _(paths, pl):
    prueba = pl.read_parquet(paths.DATA_PATH / "tiffany.parquet")
    return (prueba,)


@app.cell
def _(prueba):
    prueba
    return


@app.cell
def _():
    # df = pd.read_csv(paths.DATA_PATH / "simplified_dataset.csv")
    # df = df.drop(columns = ["block"])
    # # df["drug"] = df["drug"].str.strip().str.lower()
    # # df["drug"] = np.where(
    # #     df["drug"].str.lower().isin(["ephys", "rest"]),
    # #     df["drug"],
    # #     None
    # # )
    # df["drug"].unique()
    return


@app.cell
def _():
    # df
    return


@app.cell
def _():
    # import pyarrow as pa

    # bad_cols = []

    # for col in df.columns:
    #     try:
    #         pa.array(df[col], from_pandas=True)
    #     except Exception as e:
    #         bad_cols.append((col, str(e)))

    # bad_cols
    # col = bad_cols[0][0]

    # df[col].map(type).value_counts()
    return


@app.cell
def _():
    # df_pl = pl.from_pandas(df)
    return


@app.cell
def _():
    # df_pl.write_parquet("tiffany_complete.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
