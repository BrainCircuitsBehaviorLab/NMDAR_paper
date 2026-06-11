import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import base64
    import io
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    # Custom package and plots

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.plots.emissions import _fold_three_choice_raw_weights as fold_three_choice
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import (
        add_choice_lag_summary_regressor,
        build_transition_chunk_drug_plot_data,
        build_transition_chunk_plot_data,
        prepare_closed_loop_model_autocorrelograms,
        prepare_corrected_behavior_autocorrelograms,
    )
    from src.plots.common import (
        animal_chunk_histogram,
        boxplot_STYLE,
        build_repetition_variance_by_drug_task,
        build_session_repetition_data,
        build_session_trial_outcomes_data,
        fig_size,
        pick_existing_column,
        plot_session_repetition_running_count,
        plot_session_response_raster,
        plot_session_trial_outcomes,
        two_afc_session_repeat_alternate_accuracy as build_two_afc_session_repeat_alternate_accuracy,
        two_afc_transition_chunk_lengths as build_two_afc_transition_chunk_lengths,
    )


    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    return (
        Path,
        add_choice_lag_summary_regressor,
        animal_chunk_histogram,
        build_transition_chunk_plot_data,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        load_fit_arrays,
        np,
        pd,
        pl,
        plt,
        prepare_predictions_df,
        sns,
    )


@app.cell
def _(Path, configure_paths, get_runtime_paths):
    ROOT = Path(__file__).resolve().parents[1]

    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    data_path = Path(__file__).parents[1] / "data/processed"
    print(data_path)

    project_path = Path(__file__).resolve().parents[1]
    print(project_path)

    path_panels = project_path / "figures" / "panels2"
    print(path_panels)
    return data_path, paths


@app.cell
def _(Path, plt, sns):
    sns.set_theme(style='ticks', context='notebook')
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    return


@app.cell
def _(get_adapter):
    task_names = ("2AFC_delay", "2AFC", "MCDR")
    model_name = "one hot"
    adapters = {_task_name: get_adapter(_task_name) for _task_name in task_names}
    plots_by_task = {
        _task_name: _adapter.get_plots()
        for _task_name, _adapter in adapters.items()
    }
    dfs = {
        _task_name: _adapter.subject_filter(_adapter.read_dataset())
        for _task_name, _adapter in adapters.items()
    }

    subjects_by_task = {
        _task_name: list(_df["subject"].unique())
        for _task_name, _df in dfs.items()
    }
    return adapters, dfs, model_name, subjects_by_task, task_names


@app.cell
def _(
    adapters,
    add_choice_lag_summary_regressor,
    build_trial_and_weights_df,
    build_views,
    dfs,
    load_fit_arrays,
    model_name,
    paths,
    prepare_predictions_df,
    subjects_by_task,
    task_names,
):
    trial_dfs, weight_dfs, views, plot_dfs = dict(), dict(), dict(), dict()
    for _task in task_names:
        _adapter = adapters[_task]
        _df_all = dfs[_task]
        _subjects = subjects_by_task[_task]
        _out = paths.RESULTS / "fits" / _task / "glm" / model_name
        _arrays_store, _ = load_fit_arrays(
            out_dir=_out,
            arrays_suffix="glm_arrays.npz",
            adapter=_adapter,
            df_all=_df_all,
            subjects=_subjects,
            emission_cols=None,
        )
        views[_task] = build_views(_arrays_store, _adapter, 1, subjects_by_task[_task])
        trial_dfs[_task], weight_dfs[_task] = build_trial_and_weights_df(_df_all, views=views[_task], adapter=_adapter, min_session_length=1)
        plot_dfs[_task] = prepare_predictions_df(_task, trial_dfs[_task])
        _choice_lag_cols = []
        for _view in views[_task].values():
            for _feature in list(getattr(_view, "feat_names", []) or []):
                _feature = str(_feature)
                if _feature.startswith("choice_lag_") and _feature not in _choice_lag_cols:
                    _choice_lag_cols.append(_feature)
        # plot_dfs[_task] = add_choice_lag_summary_regressor(plot_dfs[_task], choice_lag_cols=_adapter.choice_lag_cols(trial_dfs[_task]))
        plot_dfs[_task] = add_choice_lag_summary_regressor(plot_dfs[_task], choice_lag_cols=_choice_lag_cols)
    return (plot_dfs,)


@app.cell
def _(data_path, get_adapter, pl):
    print(data_path)
    two_afc = get_adapter("2AFC")
    df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "df_alexis_drug_combined.parquet"))  # With drug
    df_2AFC_licks = two_afc.subject_filter(pl.read_parquet(data_path / "alexis_drug_combined.parquet"))
    df_2AFC
    return df_2AFC, df_2AFC_licks


@app.cell
def _(df_2AFC_licks, pd):
    two_afc_repeat_run_trial_metrics = (
        df_2AFC_licks
        .select(["subject", "Session", "Trial", "Choice", "Hit", "ILI", "RT", "nLicks"])
        .to_pandas()
        .sort_values(["subject", "Session", "Trial"])
    )
    two_afc_repeat_run_trial_metrics["previous_response"] = (
        two_afc_repeat_run_trial_metrics
        .groupby(["subject", "Session"], observed=True)["Choice"]
        .shift(1)
    )
    _repeat = (
        two_afc_repeat_run_trial_metrics["Choice"]
        .eq(two_afc_repeat_run_trial_metrics["previous_response"])
        .fillna(False)
    )
    _alternating = (
        two_afc_repeat_run_trial_metrics["previous_response"].notna()
        & ~_repeat
    )

    def _run_length(_mask):
        _block_id = (~_mask).groupby(
            [
                two_afc_repeat_run_trial_metrics["subject"],
                two_afc_repeat_run_trial_metrics["Session"],
            ],
            observed=True,
        ).cumsum()
        return _mask.astype(int).groupby(
            [
                two_afc_repeat_run_trial_metrics["subject"],
                two_afc_repeat_run_trial_metrics["Session"],
                _block_id,
            ],
            observed=True,
        ).cumsum()

    two_afc_repeat_run_trial_metrics["current_repeat_length"] = _run_length(_repeat)
    two_afc_repeat_run_trial_metrics["current_alternating_length"] = _run_length(_alternating)

    def _metrics_for_transition(_transition, _length_col):
        _metrics = two_afc_repeat_run_trial_metrics[
            two_afc_repeat_run_trial_metrics[_length_col] > 0
        ].melt(
            id_vars=["subject", "Session", "Trial", "Hit", _length_col],
            value_vars=["ILI", "RT", "nLicks"],
            var_name="metric",
            value_name="value",
        )
        _metrics = _metrics.rename(columns={_length_col: "current_chunk_length"})
        _metrics["transition"] = _transition
        return _metrics

    two_afc_repeat_run_trial_metrics_long = pd.concat(
        [
            _metrics_for_transition("Repeating", "current_repeat_length"),
            _metrics_for_transition("Alternating", "current_alternating_length"),
        ],
        ignore_index=True,
    )
    two_afc_repeat_run_trial_metrics_long["value"] = pd.to_numeric(
        two_afc_repeat_run_trial_metrics_long["value"],
        errors="coerce",
    )
    two_afc_repeat_run_trial_metrics_long = two_afc_repeat_run_trial_metrics_long.dropna(
        subset=["value"]
    )
    two_afc_repeat_run_trial_metrics_long["outcome"] = (
        pd.to_numeric(two_afc_repeat_run_trial_metrics_long["Hit"], errors="coerce")
        .map({1.0: "Correct", 0.0: "Incorrect"})
    )
    return (two_afc_repeat_run_trial_metrics_long,)


@app.cell
def _(plt, sns, two_afc_repeat_run_trial_metrics_long):
    _metric_order = ["ILI", "RT", "nLicks"]
    _metric_labels = {
        "ILI": "ILI",
        "RT": "RT",
        "nLicks": "nLicks",
    }
    fig_2afc_repeat_run_trial_metrics, axes_2afc_repeat_run_trial_metrics = plt.subplots(
        2,
        len(_metric_order),
        figsize=(10, 5),
        sharex=True,
    )
    _max_chunk_length = min(
        10,
        int(two_afc_repeat_run_trial_metrics_long["current_chunk_length"].max()),
    )
    for _ax, _metric in zip(axes_2afc_repeat_run_trial_metrics[0], _metric_order, strict=False):
        _data = two_afc_repeat_run_trial_metrics_long[
            two_afc_repeat_run_trial_metrics_long["metric"] == _metric
        ]
        sns.lineplot(
            data=_data,
            x="current_chunk_length",
            y="value",
            hue="transition",
            estimator="mean",
            errorbar="se",
            palette={"Repeating": "tab:brown", "Alternating": "tab:purple"},
            ax=_ax,
        )
        _ax.set_xlim(1, _max_chunk_length)
        _ax.set_title(_metric_labels[_metric])
        _ax.set_xlabel("")
        _ax.set_ylabel(_metric_labels[_metric])
        if _ax.get_legend() is not None:
            _ax.get_legend().set_frame_on(False)
            _ax.get_legend().set_title("")

    for _ax, _metric in zip(axes_2afc_repeat_run_trial_metrics[1], _metric_order, strict=False):
        _data = two_afc_repeat_run_trial_metrics_long[
            (two_afc_repeat_run_trial_metrics_long["metric"] == _metric)
            & (two_afc_repeat_run_trial_metrics_long["transition"] == "Repeating")
            & two_afc_repeat_run_trial_metrics_long["outcome"].notna()
        ]
        sns.lineplot(
            data=_data,
            x="current_chunk_length",
            y="value",
            hue="outcome",
            estimator="mean",
            errorbar="se",
            palette={"Correct": "tab:green", "Incorrect": "tab:red"},
            ax=_ax,
        )
        _ax.set_xlim(1, 30)
        _ax.set_ylim(1, 10)
        _ax.set_title(f"{_metric_labels[_metric]} - repeating")
        _ax.set_xlabel("Block length")
        _ax.set_ylabel(_metric_labels[_metric])
        if _ax.get_legend() is not None:
            _ax.get_legend().set_frame_on(False)
            _ax.get_legend().set_title("")
    fig_2afc_repeat_run_trial_metrics.tight_layout()
    fig_2afc_repeat_run_trial_metrics
    return


@app.cell
def _(pd, plot_dfs, plt, sns, task_names):
    _task_labels = {"2AFC": "2AFC", "2AFC_delay": "2ADC", "MCDR": "MCDR"}

    def _repeat_outcome_proportions_for_task(_task_name):
        _trials = (
            plot_dfs[_task_name]
            .select(["subject", "session", "trial_idx", "response", "performance"])
            .to_pandas()
            .sort_values(["subject", "session", "trial_idx"])
        )
        _trials["previous_response"] = (
            _trials.groupby(["subject", "session"], observed=True)["response"].shift(1)
        )
        _repeat = _trials["response"].eq(_trials["previous_response"]).fillna(False)
        _block_id = (~_repeat).groupby(
            [_trials["subject"], _trials["session"]],
            observed=True,
        ).cumsum()
        _trials["current_repeat_length"] = _repeat.astype(int).groupby(
            [_trials["subject"], _trials["session"], _block_id],
            observed=True,
        ).cumsum()
        _trials["outcome"] = (
            pd.to_numeric(_trials["performance"], errors="coerce")
            .map({1.0: "Correct", 0.0: "Incorrect"})
        )
        _trials = _trials[
            (_trials["current_repeat_length"] > 0)
            & _trials["outcome"].notna()
        ]
        _proportions = (
            _trials
            .groupby(["current_repeat_length", "outcome"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        _proportions["proportion"] = (
            _proportions["count"]
            / _proportions.groupby("current_repeat_length", observed=True)["count"].transform("sum")
        )
        _proportions["task"] = _task_name
        _proportions["task_label"] = _task_labels.get(_task_name, _task_name)
        return _proportions

    repeat_run_outcome_proportions_by_task = pd.concat(
        [
            _repeat_outcome_proportions_for_task(_task_name)
            for _task_name in task_names
        ],
        ignore_index=True,
    )

    _task_order = ["2AFC", "2ADC", "MCDR"]
    fig_repeat_run_outcome_proportions_by_task, axes_repeat_run_outcome_proportions_by_task = plt.subplots(
        1,
        len(_task_order),
        figsize=(10, 3),
        sharex=True,
        sharey=True,
    )
    for _ax, _task_label in zip(axes_repeat_run_outcome_proportions_by_task, _task_order, strict=False):
        _data = repeat_run_outcome_proportions_by_task[
            repeat_run_outcome_proportions_by_task["task_label"] == _task_label
        ]
        sns.lineplot(
            data=_data,
            x="current_repeat_length",
            y="proportion",
            hue="outcome",
            marker="o",
            palette={"Correct": "tab:green", "Incorrect": "tab:red"},
            ax=_ax,
        )
        _ax.set_xlim(
            1,
            min(10, int(repeat_run_outcome_proportions_by_task["current_repeat_length"].max())),
        )
        _ax.set_ylim(0, 1)
        _ax.set_title(_task_label)
        _ax.set_xlabel("Block length")
        _ax.set_ylabel("Proportion")
        if _ax.get_legend() is not None:
            _ax.get_legend().set_frame_on(False)
            _ax.get_legend().set_title("")
    fig_repeat_run_outcome_proportions_by_task.tight_layout()
    fig_repeat_run_outcome_proportions_by_task
    return


@app.cell
def _(df_2AFC, pd):
    two_afc_repeat_alternate_trials_drug = (
        df_2AFC
        .select(["subject", "Session", "Trial", "Side", "Choice", "Drug", "Hit"])
        .to_pandas()
        .sort_values(["subject", "Session", "Trial"])
    )
    two_afc_repeat_alternate_trials_drug["previous_response"] = (
        two_afc_repeat_alternate_trials_drug
        .groupby(["subject", "Drug", "Session"], observed=True)["Choice"]
        .shift(1)
    )
    two_afc_repeat_alternate_trials_drug = two_afc_repeat_alternate_trials_drug.dropna(
        subset=["previous_response"]
    )
    two_afc_repeat_alternate_trials_drug["transition"] = (
        two_afc_repeat_alternate_trials_drug["Choice"]
        .eq(two_afc_repeat_alternate_trials_drug["previous_response"])
        .map({True: "repeating", False: "alternating"})
    )
    two_afc_repeat_alternate_trials_drug["transition_chunk"] = (
        two_afc_repeat_alternate_trials_drug
        .groupby(["subject", "Drug", "Session"], observed=True)["transition"]
        .transform(lambda transition: transition.ne(transition.shift()).cumsum())
    )

    two_afc_transition_chunk_lengths_drug = (
        two_afc_repeat_alternate_trials_drug
        .groupby(["subject", "Drug", "Session", "transition", "transition_chunk"], observed=True)
        .size()
        .rename("chunk_length")
        .reset_index()
    )
    two_afc_transition_chunk_counts_drug = (
        two_afc_transition_chunk_lengths_drug
        .groupby(["subject", "Drug", "transition", "chunk_length"], observed=True)
        .size()
        .rename("count")
        .reindex(
            pd.MultiIndex.from_product(
                [
                    two_afc_transition_chunk_lengths_drug["subject"].unique(),
                    two_afc_transition_chunk_lengths_drug["Drug"].unique(),
                    ["repeating", "alternating"],
                    range(1, int(two_afc_transition_chunk_lengths_drug["chunk_length"].max()) + 1),
                ],
                names=["subject", "Drug", "transition", "chunk_length"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    two_afc_transition_chunk_counts_drug["frequency"] = (
        two_afc_transition_chunk_counts_drug["count"]
        / two_afc_transition_chunk_counts_drug.groupby(["subject", "Drug", "transition"], observed=True)["count"].transform("sum")
    )
    two_afc_transition_chunk_frequency_drug = (
        two_afc_transition_chunk_counts_drug
        .groupby(["transition", "Drug", "chunk_length"], observed=True)["frequency"]
        .mean()
        .reset_index()
    )
    two_afc_session_transition_accuracy_drug = (
        two_afc_repeat_alternate_trials_drug
        .groupby(["subject", "Drug", "Session", "transition"], observed=True)["Hit"]
        .mean()
        .unstack("transition")
        .reset_index()
        .rename(
            columns={
                "alternating": "alternating_accuracy",
                "repeating": "repeating_accuracy",
            }
        )
    )
    two_afc_transition_chunk_lengths_drug["drug_label"] = (
        two_afc_transition_chunk_lengths_drug["Drug"]
        .map({0: "Saline", 1: "Drug"})
        .fillna(two_afc_transition_chunk_lengths_drug["Drug"])
    )
    return (two_afc_transition_chunk_lengths_drug,)


@app.cell
def _(
    animal_chunk_histogram,
    fig_size,
    plt,
    sns,
    two_afc_transition_chunk_lengths_drug,
):
    _drug_chunk_hist_ylabel = "Frequency"
    fig_2afc_repeating_chunk_lengths_drug, ax_2afc_repeating_chunk_lengths_drug = plt.subplots(
        figsize=fig_size(2, 1)
    )
    _hist_data = animal_chunk_histogram(
        two_afc_transition_chunk_lengths_drug[
            two_afc_transition_chunk_lengths_drug["transition"] == "repeating"
        ],
        group_cols=["drug_label"],
        stat="probability",
    )
    sns.histplot(
        data=_hist_data,
        x="chunk_length",
        hue="drug_label",
        weights="hist_weight",
        stat="count",
        common_norm=False,
        element="step",
        bins=range(1, int(two_afc_transition_chunk_lengths_drug["chunk_length"].max()) + 2),
        palette={"Saline": "tab:gray", "Drug": "tab:pink"},
        ax=ax_2afc_repeating_chunk_lengths_drug,
    )
    ax_2afc_repeating_chunk_lengths_drug.set_xlim(0, 100)
    # ax_2afc_repeating_chunk_lengths_drug.set_yscale("log")
    ax_2afc_repeating_chunk_lengths_drug.set_title("Repeating")
    ax_2afc_repeating_chunk_lengths_drug.set_xlabel("Chunk length")
    ax_2afc_repeating_chunk_lengths_drug.set_ylabel(_drug_chunk_hist_ylabel)
    ax_2afc_repeating_chunk_lengths_drug
    return


@app.cell
def _(
    animal_chunk_histogram,
    fig_size,
    np,
    plt,
    two_afc_transition_chunk_lengths_drug,
):
    _drug_palette = {"Saline": "tab:gray", "Drug": "tab:pink"}
    _drug_chunk_hist_ylabel = "Frequency"
    _max_chunk_length = 50
    _x = np.arange(1, _max_chunk_length + 1)
    _data = two_afc_transition_chunk_lengths_drug[
        two_afc_transition_chunk_lengths_drug["transition"] == "repeating"
    ]
    _hist_data = animal_chunk_histogram(
        _data,
        group_cols=["drug_label"],
        stat="probability",
    )

    fig_2afc_repeating_chunk_lengths_drug_lines, ax_2afc_repeating_chunk_lengths_drug_lines = plt.subplots(
        figsize=fig_size(2, 1)
    )
    for _drug_label, _color in _drug_palette.items():
        _drug_data = _hist_data[_hist_data["drug_label"] == _drug_label]
        if _drug_data.empty:
            continue
        _y = (
            _drug_data
            .groupby("chunk_length", observed=True)["hist_weight"]
            .sum()
            .reindex(_x, fill_value=0)
            .sort_index()
            .to_numpy(dtype=float)
        )
        ax_2afc_repeating_chunk_lengths_drug_lines.plot(
            _x,
            _y,
            color=_color,
            linewidth=1.5,
            label=_drug_label,
        )

    ax_2afc_repeating_chunk_lengths_drug_lines.set_xlim(0, 20)
    ax_2afc_repeating_chunk_lengths_drug_lines.set_yscale('log')
    ax_2afc_repeating_chunk_lengths_drug_lines.set_title("Repeating choices")
    ax_2afc_repeating_chunk_lengths_drug_lines.set_xlabel("Chunk length")
    ax_2afc_repeating_chunk_lengths_drug_lines.set_ylabel(_drug_chunk_hist_ylabel)
    ax_2afc_repeating_chunk_lengths_drug_lines.legend(frameon=False)
    ax_2afc_repeating_chunk_lengths_drug_lines
    return


@app.cell
def _(
    animal_chunk_histogram,
    fig_size,
    plt,
    sns,
    two_afc_transition_chunk_lengths_drug,
):
    _drug_chunk_hist_ylabel = "Frequency"
    fig_2afc_alternating_chunk_lengths_drug, ax_2afc_alternating_chunk_lengths_drug = plt.subplots(
        figsize=fig_size(2, 1)
    )
    _hist_data = animal_chunk_histogram(
        two_afc_transition_chunk_lengths_drug[
            two_afc_transition_chunk_lengths_drug["transition"] == "alternating"
        ],
        group_cols=["drug_label"],
        stat="probability",
    )
    sns.histplot(
        data=_hist_data,
        x="chunk_length",
        hue="drug_label",
        weights="hist_weight",
        stat="count",
        common_norm=False,
        element="step",
        bins=range(1, int(two_afc_transition_chunk_lengths_drug["chunk_length"].max()) + 2),
        palette={"Saline": "tab:gray", "Drug": "tab:pink"},
        ax=ax_2afc_alternating_chunk_lengths_drug,
    )
    ax_2afc_alternating_chunk_lengths_drug.set_xlim(0, 20)
    # ax_2afc_alternating_chunk_lengths_drug.set_yscale("log")
    ax_2afc_alternating_chunk_lengths_drug.set_title("Alternating")
    ax_2afc_alternating_chunk_lengths_drug.set_xlabel("Chunk length")
    ax_2afc_alternating_chunk_lengths_drug.set_ylabel(_drug_chunk_hist_ylabel)
    ax_2afc_alternating_chunk_lengths_drug
    return


@app.cell
def _(build_transition_chunk_plot_data, pl, plot_dfs, task_names):
    chunk_hist_stat = "count"  # Use "probability" for relative frequencies.
    chunk_hist_ylabel = {"count": "Count", "probability": "Frequency"}[chunk_hist_stat]
    transition_palette = {"repeating": "tab:brown", "alternating": "tab:purple"}
    transition_drug_palette = transition_palette
    transition_chunk_lengths_by_task, transition_chunk_plot_data, _transition_palette, transition_repeat_probabilities = (
        build_transition_chunk_plot_data(
            plot_dfs,
            task_names,
            stat=chunk_hist_stat,
            transition_palette=transition_palette,
        )
    )
    def _transition_trials(_value_col, _extra_cols=()):
        _previous_col = f"previous_{_value_col}"
        return (
            plot_dfs["2AFC"]
            .select(["subject", "session", "trial_idx", _value_col, *_extra_cols])
            .sort(["subject", "session", "trial_idx"])
            .with_columns(
                pl.col(_value_col)
                .shift()
                .over(["subject", "session"])
                .alias(_previous_col)
            )
            .drop_nulls(_previous_col)
            .with_columns(
                pl.when(pl.col(_value_col) == pl.col(_previous_col))
                .then(pl.lit("repeating"))
                .otherwise(pl.lit("alternating"))
                .alias("transition")
            )
            .with_columns(
                (
                    pl.col("transition")
                    != pl.col("transition").shift().over(["subject", "session"])
                )
                .fill_null(True)
                .cum_sum()
                .over(["subject", "session"])
                .alias("transition_chunk")
            )
        )

    def _chunk_lengths(_trials):
        return (
            _trials
            .group_by(["subject", "session", "transition", "transition_chunk"])
            .len(name="chunk_length")
            .to_pandas()
        )

    _two_afc_repeat_alternate_trials_pl = _transition_trials(
        "response",
        ["performance"],
    )
    two_afc_repeat_alternate_trials = _two_afc_repeat_alternate_trials_pl.to_pandas()
    two_afc_transition_chunk_lengths = _chunk_lengths(_two_afc_repeat_alternate_trials_pl)
    two_afc_session_transition_accuracy = (
        two_afc_repeat_alternate_trials
        .groupby(["subject", "session", "transition"], observed=True)["performance"]
        .mean()
        .unstack("transition")
        .reset_index()
        .rename(
            columns={
                "alternating": "alternating_accuracy",
                "repeating": "repeating_accuracy",
            }
        )
    )

    two_afc_stimulus_transition_chunk_lengths = _chunk_lengths(
        _transition_trials("stimulus")
    )
    return (
        chunk_hist_stat,
        chunk_hist_ylabel,
        two_afc_session_transition_accuracy,
        two_afc_stimulus_transition_chunk_lengths,
    )


@app.cell
def _(
    animal_chunk_histogram,
    chunk_hist_stat,
    chunk_hist_ylabel,
    fig_size,
    plt,
    sns,
    two_afc_stimulus_transition_chunk_lengths,
):
    fig_2afc_stimulus_transition_chunk_lengths, ax_2afc_stimulus_transition_chunk_lengths = plt.subplots(
        figsize=fig_size(2, 1)
    )
    _hist_data = animal_chunk_histogram(
        two_afc_stimulus_transition_chunk_lengths,
        group_cols=["transition"],
        stat=chunk_hist_stat,
    )
    sns.histplot(
        data=_hist_data,
        x="chunk_length",
        hue="transition",
        weights="hist_weight",
        stat="count",
        discrete=True,
        common_norm=False,
        palette={"repeating": "tab:brown", "alternating": "tab:purple"},
        alpha=0.75,
        element="step",
        fill=False,
        ax=ax_2afc_stimulus_transition_chunk_lengths,
    )

    ax_2afc_stimulus_transition_chunk_lengths.set_xlim(0, 50)
    ax_2afc_stimulus_transition_chunk_lengths.set_ylabel(chunk_hist_ylabel)
    ax_2afc_stimulus_transition_chunk_lengths.set_title("Stimulus")
    ax_2afc_stimulus_transition_chunk_lengths.set_xlabel("Chunk length")
    ax_2afc_stimulus_transition_chunk_lengths.get_legend().set_title(None)
    ax_2afc_stimulus_transition_chunk_lengths.get_legend().set_frame_on(False)
    ax_2afc_stimulus_transition_chunk_lengths.get_legend().set_loc("lower right")
    ax_2afc_stimulus_transition_chunk_lengths
    return


@app.cell
def _(sns, two_afc_session_transition_accuracy):
    joint_2afc_repeating_vs_alternating_accuracy = sns.jointplot(
        data=two_afc_session_transition_accuracy,
        x="repeating_accuracy",
        y="alternating_accuracy",
        color="tab:blue",
        kind="hist",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    fig_2afc_repeating_vs_alternating_accuracy = joint_2afc_repeating_vs_alternating_accuracy.fig
    ax_2afc_repeating_vs_alternating_accuracy = joint_2afc_repeating_vs_alternating_accuracy.ax_joint
    ax_2afc_repeating_vs_alternating_accuracy.plot([0, 1], [0, 1], color="tab:gray")
    ax_2afc_repeating_vs_alternating_accuracy.set_xlabel("Repeating accuracy")
    ax_2afc_repeating_vs_alternating_accuracy.set_ylabel("Alternating accuracy")
    fig_2afc_repeating_vs_alternating_accuracy.canvas.draw()
    _joint_position = ax_2afc_repeating_vs_alternating_accuracy.get_position()
    _marg_y_position = joint_2afc_repeating_vs_alternating_accuracy.ax_marg_y.get_position()
    _colorbar_ax = fig_2afc_repeating_vs_alternating_accuracy.add_axes(
        [_marg_y_position.x1 + 0.02, _joint_position.y0, 0.03, _joint_position.height]
    )
    fig_2afc_repeating_vs_alternating_accuracy.colorbar(
        joint_2afc_repeating_vs_alternating_accuracy.ax_joint.collections[0],
        cax=_colorbar_ax,
        label="Count",
    )
    fig_2afc_repeating_vs_alternating_accuracy
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
