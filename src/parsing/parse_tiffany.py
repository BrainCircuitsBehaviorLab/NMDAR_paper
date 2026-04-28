import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import numpy as np
    from glmhmmt.runtime import get_runtime_paths

    paths = get_runtime_paths()
    return paths, pd, pl


@app.cell
def _():
    # df = pl.from_pandas(pd.read_csv(paths.DATA_PATH / "tiffany.csv",index_col=0))
    # df.write_parquet(paths.DATA_PATH / "tiffany.parquet")
    # df
    return


@app.cell
def _(paths, pd):
    df = pd.read_csv(paths.DATA_PATH / "simplified_dataset.csv")
    df = df.drop(columns = ["block"])
    # df["drug"] = df["drug"].str.strip().str.lower()
    # df["drug"] = np.where(
    #     df["drug"].str.lower().isin(["ephys", "rest"]),
    #     df["drug"],
    #     None
    # )
    df["drug"].unique()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    import pyarrow as pa

    bad_cols = []

    for col in df.columns:
        try:
            pa.array(df[col], from_pandas=True)
        except Exception as e:
            bad_cols.append((col, str(e)))

    bad_cols
    col = bad_cols[0][0]

    df[col].map(type).value_counts()
    return


@app.cell
def _(df, pl):
    df_pl = pl.from_pandas(df)
    return (df_pl,)


@app.cell
def _(df_pl):
    df_pl.write_parquet("tiffany_complete.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
