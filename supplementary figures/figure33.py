# /// script
# [tool.marimo.opengraph]
# title = "Supplementary Figure 3 — Dwell time by cumulative reward"
# description = "Average engaged and disengaged dwell times before and after 50% of each session's total reward."
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3.3

    ## Description

    Average dwell times in the engaged and disengaged states, comparing the trials at or below versus above 50% of each session's total reward.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import ttest_rel
    from statannotations.Annotator import Annotator

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import MCDR as process_mcdr
    from src.process import two_adc as process_two_adc
    from src.process import two_afc as process_two_afc
    from src.process.common import glmhmmt_state_dwell_df
    from src.plots.common import BOXPLOT_STYLE, fig_size

    return (
        Annotator,
        BOXPLOT_STYLE,
        Path,
        build_trial_and_weights_df,
        build_views,
        configure_paths,
        fig_size,
        get_adapter,
        get_runtime_paths,
        glmhmmt_state_dwell_df,
        json,
        load_app_config,
        load_fit_arrays,
        mo,
        np,
        os,
        pd,
        plt,
        process_mcdr,
        process_two_adc,
        process_two_afc,
        sns,
        ttest_rel,
    )


@app.cell
def _(load_app_config, process_mcdr, process_two_adc, process_two_afc):
    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    return (prepare_predictions_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings
    """)
    return


@app.cell
def _():
    model_type = "glmhmmt"
    MODEL_BY_TASK = {
        "2AFC_delay": "param half-pure",
        "2AFC": "param half-pure",
    }
    task_names = tuple(MODEL_BY_TASK)
    return MODEL_BY_TASK, model_type, task_names


@app.cell
def _():
    mount_figure = True
    return (mount_figure,)


@app.cell
def _():
    panel_names = {
        "2AFC_delay": "S3a",
        "2AFC": "S3b",
    }
    return (panel_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paths
    """)
    return


@app.cell
def _(Path, configure_paths, get_runtime_paths, os):
    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    path_panels = ROOT / "supplementary figures" / "panels33"
    for panel_format in ("svg", "png"):
        os.makedirs(path_panels / panel_format, exist_ok=True)
    return ROOT, path_panels, paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Style
    """)
    return


@app.cell
def _(ROOT, plt, sns):
    sns.set_theme(style="ticks", context="paper")
    plt.style.use(ROOT / "paper.mplstyle")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["savefig.bbox"] = "standard"

    task_labels = {
        "2AFC_delay": "2ADC",
        "2AFC": "2AFC",
    }
    state_order = ["Engaged", "Disengaged"]
    reward_order = ["≤50% reward", ">50% reward"]
    reward_palette = {
        "≤50% reward": "tab:purple",
        ">50% reward": "tab:brown",
    }
    return reward_order, reward_palette, state_order, task_labels


@app.cell
def _(fig_size, mount_figure, plt):
    if mount_figure:
        fig, axd = plt.subplot_mosaic(
            [["dwell_time_by_reward_2ADC", "dwell_time_by_reward_2AFC"]],
            figsize=fig_size(1, 2),
            constrained_layout=True,
        )
    else:
        fig, axd = None, {}
    return axd, fig


@app.cell
def _(Annotator, pd):
    def add_subject_pair_lines(
        ax,
        df,
        *,
        x,
        y,
        order,
        hue,
        hue_order,
        subject_col="subject",
        offset=0.2,
    ):
        for x_position, x_value in enumerate(order):
            state_df = df[df[x] == x_value]
            paired = state_df.pivot_table(
                values=y,
                index=subject_col,
                columns=hue,
                aggfunc="first",
            )
            if not all(level in paired.columns for level in hue_order):
                continue
            for _, values in paired.dropna(subset=hue_order).iterrows():
                ax.plot(
                    [x_position - offset, x_position + offset],
                    [values[hue_order[0]], values[hue_order[1]]],
                    color="0.75",
                    linewidth=0.5,
                    zorder=0,
                )

    def add_paired_reward_annotation(
        ax,
        df,
        *,
        x,
        y,
        order,
        hue,
        hue_order,
        subject_col="subject",
    ):
        paired_frames = []
        pairs = []
        for x_value in order:
            state_df = df[df[x] == x_value]
            paired = state_df.pivot_table(
                values=y,
                index=subject_col,
                columns=hue,
                aggfunc="first",
            )
            if not all(level in paired.columns for level in hue_order):
                continue
            paired = paired.dropna(subset=hue_order)
            if len(paired) < 2:
                continue
            paired_subjects = set(paired.index.astype(str))
            paired_frames.append(
                state_df[state_df[subject_col].astype(str).isin(paired_subjects)]
            )
            pairs.append(((x_value, hue_order[0]), (x_value, hue_order[1])))

        if not pairs:
            return
        annotator = Annotator(
            ax,
            pairs,
            data=pd.concat(paired_frames, ignore_index=True),
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            line_height=0,
            verbose=False,
        ).apply_and_annotate()

    return add_paired_reward_annotation, add_subject_pair_lines


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load original data and fits
    """)
    return


@app.cell
def _(MODEL_BY_TASK, get_adapter):
    adapters = {
        task_name: get_adapter(task_name)
        for task_name in MODEL_BY_TASK
    }
    dfs = {
        task_name: adapter.subject_filter(adapter.read_dataset())
        for task_name, adapter in adapters.items()
    }
    return adapters, dfs


@app.cell
def _(MODEL_BY_TASK, adapters, dfs, json, model_type, paths):
    model_configs = {}
    for config_task_name, config_model_id in MODEL_BY_TASK.items():
        config_model_dir = (
            paths.RESULTS
            / "fits"
            / config_task_name
            / model_type
            / config_model_id
        )
        config_path = config_model_dir / "config.json"
        if config_path.exists():
            task_config = json.loads(config_path.read_text())
        else:
            task_config = {
                "task": config_task_name,
                "model_id": config_model_id,
                "subjects": list(dfs[config_task_name]["subject"].unique()),
                "K_list": [2],
                "emission_cols": None,
                "transition_cols": None,
            }
        task_config["model_id"] = config_model_id
        model_configs[config_task_name] = task_config

        config_adapter = adapters[config_task_name]
        for key in (
            "state_scoring_feature",
            "state_scoring_rule",
            "state_split_feature",
            "state_split_rule",
        ):
            if key in task_config:
                setattr(config_adapter, key, task_config[key] or None)
    return (model_configs,)


@app.cell
def _(
    adapters,
    build_trial_and_weights_df,
    build_views,
    dfs,
    load_fit_arrays,
    model_configs,
    model_type,
    paths,
    prepare_predictions_df,
    task_names,
):
    plot_dfs = {}
    model_load_report = []

    for fit_task_name in task_names:
        fit_config = model_configs[fit_task_name]
        fit_adapter = adapters[fit_task_name]
        all_trials = dfs[fit_task_name]
        fit_model_id = fit_config["model_id"]
        fit_model_dir = (
            paths.RESULTS
            / "fits"
            / fit_task_name
            / model_type
            / fit_model_id
        )
        n_states = int((fit_config.get("K_list") or [2])[0])
        subjects = [
            str(subject)
            for subject in (
                fit_config.get("subjects")
                or list(all_trials["subject"].unique())
            )
        ]

        arrays_store, _ = load_fit_arrays(
            out_dir=fit_model_dir,
            arrays_suffix=f"{model_type}_arrays.npz",
            adapter=fit_adapter,
            df_all=all_trials,
            subjects=subjects,
            emission_cols=fit_config.get("emission_cols") or None,
            transition_cols=fit_config.get("transition_cols") or None,
            k=n_states,
        )
        selected_subjects = [
            subject for subject in subjects if subject in arrays_store
        ]
        if not selected_subjects:
            model_load_report.append(
                f"{fit_task_name}: no arrays found in {fit_model_dir}"
            )
            continue

        selected_arrays = {
            subject: arrays_store[subject]
            for subject in selected_subjects
        }
        task_views = build_views(
            selected_arrays,
            fit_adapter,
            n_states,
            selected_subjects,
        )
        trial_df, _ = build_trial_and_weights_df(
            all_trials,
            views=task_views,
            adapter=fit_adapter,
            min_session_length=2,
        )
        plot_dfs[fit_task_name] = prepare_predictions_df(
            fit_task_name,
            trial_df,
        )
        model_load_report.append(
            f"{fit_task_name}: loaded {len(selected_subjects)} subjects from "
            f"{model_type}/{fit_model_id}"
        )

    active_task_names = tuple(plot_dfs)
    return active_task_names, model_load_report, plot_dfs


@app.cell
def _(mo, model_load_report):
    mo.md("\n".join(f"- {line}" for line in model_load_report))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dwell-time data
    """)
    return


@app.cell
def _(active_task_names, glmhmmt_state_dwell_df, np, pd, plot_dfs):
    reward_trial_dfs = {}
    reward_dwell_run_dfs = {}
    session_reward_dwell_dfs = {}
    subject_reward_dwell_dfs = {}

    for dwell_task_name in active_task_names:
        trials = plot_dfs[dwell_task_name].to_pandas().copy()
        trials = (
            trials.dropna(
                subset=["subject", "session", "trial_idx", "state_label"]
            )
            .sort_values(
                ["subject", "session", "trial_idx"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        trials["subject"] = trials["subject"].astype(str)
        trials["session"] = trials["session"].astype(str)
        session_groups = trials.groupby(
            ["subject", "session"],
            observed=True,
            sort=False,
        )
        trials["reward"] = trials["performance"].fillna(0).astype(float)
        trials["cumulative_reward"] = session_groups["reward"].cumsum()
        trials["session_total_reward"] = session_groups["reward"].transform(
            "sum"
        )
        trials = trials[trials["session_total_reward"] > 0].copy()
        trials["reward_segment"] = np.where(
            trials["cumulative_reward"] <= trials["session_total_reward"] / 2,
            "≤50% reward",
            ">50% reward",
        )
        trials["reward_segment_id"] = (
            trials["session"] + "::" + trials["reward_segment"]
        )
        reward_trial_dfs[dwell_task_name] = trials

        dwell_runs = glmhmmt_state_dwell_df(
            trials,
            session_col="reward_segment_id",
        ).merge(
            trials[
                ["subject", "session", "reward_segment", "reward_segment_id"]
            ].drop_duplicates(),
            on=["subject", "reward_segment_id"],
            how="left",
            validate="m:1",
        )
        reward_dwell_run_dfs[dwell_task_name] = dwell_runs

        session_dwell = (
            dwell_runs.groupby(
                ["subject", "session", "reward_segment", "state_label"],
                as_index=False,
                observed=True,
            )
            .agg(mean_dwell_trials=("dwell_trials", "mean"))
            .sort_values(
                ["state_label", "reward_segment", "subject", "session"]
            )
        )
        session_reward_dwell_dfs[dwell_task_name] = session_dwell
        subject_reward_dwell_dfs[dwell_task_name] = (
            session_dwell.groupby(
                ["subject", "reward_segment", "state_label"],
                as_index=False,
                observed=True,
            )
            .agg(
                mean_dwell_trials=("mean_dwell_trials", "mean"),
                n_sessions=("session", "nunique"),
            )
            .sort_values(["state_label", "reward_segment", "subject"])
        )

    dwell_summary = pd.concat(
        [
            dwell_df.assign(task=summary_task_name)
            for summary_task_name, dwell_df in subject_reward_dwell_dfs.items()
        ],
        ignore_index=True,
    )
    return dwell_summary, subject_reward_dwell_dfs


@app.cell
def _(dwell_summary):
    dwell_summary
    return


@app.cell
def _(
    panel_names,
    reward_order,
    state_order,
    subject_reward_dwell_dfs,
    task_labels,
    ttest_rel,
):
    tests = []
    for _task_name, _task_df in subject_reward_dwell_dfs.items():
        for _state in state_order:
            _paired = (
                _task_df[_task_df["state_label"] == _state]
                .pivot(
                    index="subject",
                    columns="reward_segment",
                    values="mean_dwell_trials",
                )
                .reindex(columns=reward_order)
                .dropna()
            )
            _result = ttest_rel(
                _paired[reward_order[0]],
                _paired[reward_order[1]],
            )
            tests.append(
                {
                    "panel": panel_names[_task_name],
                    "task": task_labels[_task_name],
                    "comparison": "≤50% vs. >50% of session reward",
                    "state": _state,
                    "test": "Paired t-test",
                    "n_subjects": len(_paired),
                    "statistic": _result.statistic,
                    "degrees_of_freedom": _result.df,
                    "p_value": _result.pvalue,
                }
            )
    return (tests,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average dwell time: by cumulative reward
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2ADC
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    add_paired_reward_annotation,
    add_subject_pair_lines,
    axd,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    reward_order,
    reward_palette,
    sns,
    state_order,
    subject_reward_dwell_dfs,
    task_labels,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    dwell_time_by_reward_2ADC = (
        plt.gca() if not mount_figure else axd["dwell_time_by_reward_2ADC"]
    )
    dwell_time_by_reward_2ADC.clear()
    sns.boxplot(
        data=subject_reward_dwell_dfs["2AFC_delay"],
        x="state_label",
        y="mean_dwell_trials",
        hue="reward_segment",
        order=state_order,
        hue_order=reward_order,
        palette=reward_palette,
        ax=dwell_time_by_reward_2ADC,
        **BOXPLOT_STYLE,
    )
    dwell_time_by_reward_2ADC.set_yscale("log")
    add_subject_pair_lines(
        dwell_time_by_reward_2ADC,
        subject_reward_dwell_dfs["2AFC_delay"],
        x="state_label",
        y="mean_dwell_trials",
        order=state_order,
        hue="reward_segment",
        hue_order=reward_order,
    )
    add_paired_reward_annotation(
        dwell_time_by_reward_2ADC,
        subject_reward_dwell_dfs["2AFC_delay"],
        x="state_label",
        y="mean_dwell_trials",
        order=state_order,
        hue="reward_segment",
        hue_order=reward_order,
    )
    dwell_time_by_reward_2ADC.set_title(task_labels["2AFC_delay"])
    dwell_time_by_reward_2ADC.set_xlabel("")
    dwell_time_by_reward_2ADC.set_ylabel("Mean dwell time (trials)")
    dwell_time_by_reward_2ADC.set_xticks([0, 1], ["Eng.", "Dis."])
    dwell_time_by_reward_2ADC.legend(frameon=False, title="")
    if not mount_figure:
        dwell_time_by_reward_2ADC.figure.savefig(
            path_panels / "svg" / "2AFC_delay_dwell_time_by_reward.svg"
        )
        dwell_time_by_reward_2ADC.figure.savefig(
            path_panels / "png" / "2AFC_delay_dwell_time_by_reward.png",
            dpi=300,
        )
    dwell_time_by_reward_2ADC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2AFC
    """)
    return


@app.cell
def _(
    BOXPLOT_STYLE,
    add_paired_reward_annotation,
    add_subject_pair_lines,
    axd,
    fig_size,
    mount_figure,
    path_panels,
    plt,
    reward_order,
    reward_palette,
    sns,
    state_order,
    subject_reward_dwell_dfs,
    task_labels,
):
    plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
    dwell_time_by_reward_2AFC = (
        plt.gca() if not mount_figure else axd["dwell_time_by_reward_2AFC"]
    )
    dwell_time_by_reward_2AFC.clear()
    sns.boxplot(
        data=subject_reward_dwell_dfs["2AFC"],
        x="state_label",
        y="mean_dwell_trials",
        hue="reward_segment",
        order=state_order,
        hue_order=reward_order,
        palette=reward_palette,
        ax=dwell_time_by_reward_2AFC,
        **BOXPLOT_STYLE,
    )
    dwell_time_by_reward_2AFC.set_yscale("log")
    add_subject_pair_lines(
        dwell_time_by_reward_2AFC,
        subject_reward_dwell_dfs["2AFC"],
        x="state_label",
        y="mean_dwell_trials",
        order=state_order,
        hue="reward_segment",
        hue_order=reward_order,
    )
    add_paired_reward_annotation(
        dwell_time_by_reward_2AFC,
        subject_reward_dwell_dfs["2AFC"],
        x="state_label",
        y="mean_dwell_trials",
        order=state_order,
        hue="reward_segment",
        hue_order=reward_order,
    )
    dwell_time_by_reward_2AFC.set_title(task_labels["2AFC"])
    dwell_time_by_reward_2AFC.set_xlabel("")
    dwell_time_by_reward_2AFC.set_ylabel("Mean dwell time (trials)")
    dwell_time_by_reward_2AFC.set_xticks([0, 1], ["Eng.", "Dis."])
    dwell_time_by_reward_2AFC.legend(frameon=False, title="")
    if not mount_figure:
        dwell_time_by_reward_2AFC.figure.savefig(
            path_panels / "svg" / "2AFC_dwell_time_by_reward.svg"
        )
        dwell_time_by_reward_2AFC.figure.savefig(
            path_panels / "png" / "2AFC_dwell_time_by_reward.png",
            dpi=300,
        )
    dwell_time_by_reward_2AFC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final figure
    """)
    return


@app.cell
def _(axd, fig, mount_figure, path_panels):
    if mount_figure:
        _legend = axd["dwell_time_by_reward_2AFC"].get_legend()
        if _legend is not None:
            _legend.remove()
        axd["dwell_time_by_reward_2AFC"].set_ylabel("")

        fig.align_labels()
        fig.savefig(path_panels / "supplementary_figure33.svg")
        fig.savefig(path_panels / "supplementary_figure33.png", dpi=300)
        fig.savefig(path_panels / "supplementary_figure33.pdf")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical tests
    """)
    return


@app.cell
def _(mo, path_panels, pd, tests):
    tests_df = pd.DataFrame(tests)
    tex_path = path_panels / "table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    tests_latex = tests_df.to_latex(
        buf=tex_path,
        index=False,
        float_format="%.3g",
        escape=True,
    )
    tests_latex = tests_df.to_latex(
        index=False,
        float_format="%.3g",
        escape=True,
    )
    mo.vstack(
        [
            tests_df,
            mo.md(f"```latex\n{tests_latex}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
