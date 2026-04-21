import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from plot_saver import make_plot_saver
    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.views import get_state_color, build_views
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        select_subject_behavior_df,
    )

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    sns.set_style("ticks")
    paths = get_runtime_paths()
    return (
        build_trial_and_weights_df,
        build_views,
        get_adapter,
        load_fit_arrays,
        mo,
        paths,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading data
    """)
    return


@app.cell
def _(get_adapter):
    task_name = "2AFC"
    adapter = get_adapter(task_name)
    df_all = adapter.read_dataset()
    df_all = adapter.subject_filter(df_all)
    plots = adapter.get_plots()
    subjects= list(df_all["subject"].unique())
    return adapter, df_all, plots, subjects, task_name


@app.cell
def _(
    adapter,
    build_views,
    df_all,
    load_fit_arrays,
    mo,
    paths,
    subjects,
    task_name,
):
    alias_model = "one hot"
    OUT = paths.RESULTS / "fits" / task_name / "glm" / alias_model
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=subjects,
        emission_cols=None,
    )
    K = 1
    views = build_views(arrays_store, adapter, K, subjects)
    mo.md(f"Loaded {len(arrays_store)} subjects from `{alias_model}`")
    return (views,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Processing results dataframes
    """)
    return


@app.cell
def _(adapter, build_trial_and_weights_df, df_all, plots, trial_df_sel, views):
    trial_df, weights_df = build_trial_and_weights_df(
        df_all,
        views=views,
        adapter=adapter,
        min_session_length=1,
    )

    _choice_lag_cols = []
    for _view in views.values():
        for _feat in list(getattr(_view, "feat_names", []) or []):
            _feat = str(_feat)
            if _feat.startswith("choice_lag_") and _feat not in _choice_lag_cols:
                _choice_lag_cols.append(_feat)

    if not _choice_lag_cols:
        _choice_lag_cols = adapter.choice_lag_cols(trial_df_sel)

    plot_df_all = plots.prepare_predictions_df(trial_df)
    plot_df_all = plots.add_choice_lag_summary_regressor(
        plot_df_all,
        choice_lag_cols=_choice_lag_cols,
    )
    return plot_df_all, weights_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(plot_df_all, plots, views):
    _fig_all, _ = plots.plot_categorical_performance_all(
        plot_df_all,
        "glm",
        background_style="model",
        views=views, # To show per subject means
    )
    _fig_all
    return


@app.cell
def _(plot_df_all, plots):
    _fig_regressor = plots.plot_right_by_regressor_simple(
        plot_df_all,
        regressor_col= "choice_lag_one_hot_sum",
    )
    _fig_regressor
    return


@app.cell
def _(plot_df_all, plots):
    _fig_binned = plots.plot_binned_accuracy_figure(
        plot_df_all,
        regressor_col="choice_lag_one_hot_sum",
    )
    _fig_binned
    return


@app.cell
def _(plot_df_all, plots):
    _fig = plots.plot_right_by_regressor(
        plot_df_all,
        regressor_col="choice_lag_one_hot_sum",
        title=None,
    )
    _fig 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the weight_df of ther plot_df_all you can also make your own plots:
    """)
    return


@app.cell
def _(weights_df):
    df_agg = (
        weights_df.to_pandas()
        .groupby(["subject", "feature"])["weight"]
        .mean()
        .reset_index()
    )
    df_agg["weight"] = -df_agg["weight"] # flip sign
    df_stim = df_agg[df_agg["feature"].str.contains("stim", na=False)]
    df_bias = df_agg[df_agg["feature"].str.contains("bias", na=False)]
    df_action_trace = df_agg[df_agg["feature"].str.contains("choice", na=False)]
    return df_action_trace, df_bias, df_stim


@app.cell
def _(df_stim, plt, sns):
    fig = plt.figure(figsize=(5, 5))
    sns.boxplot(data=df_stim, x="feature", y="weight", order = ["stim_" + x for x in ["2", "4", "8", "20"]])
    sns.despine()
    plt.show()
    return


@app.cell
def _(df_action_trace, plt, sns):
    _fig = plt.figure(figsize=(5, 5))
    sns.boxplot(data=df_action_trace, x="feature", y="weight", order=[f"choice_lag_{x:02d}" for x in range(1, 16)], color="white", linecolor="#1f77b4")
    sns.despine()
    plt.ylim(0, 1)
    plt.xticks(
        range(0, 15),
        labels=range(1, 16),
    )
    plt.show()
    return


@app.cell
def _(df_bias, plt, sns):
    _fig = plt.figure(figsize=(5, 5))
    sns.lineplot(data=df_bias, x="feature", y="weight")
    sns.despine()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
