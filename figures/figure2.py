# /// script
# [tool.marimo.opengraph]
# title = "Figure 2" 
# description = " Figure 2: GLM model predictions."
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import io
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.image as mpimg
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    ROOT = Path(__file__).resolve().parents[1]

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import add_choice_lag_summary_regressor
    from src.plots.common import fig_size
    # from figure_layout_widget import FigureLayoutWidget

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=0.8)
    return (
        ROOT,
        add_choice_lag_summary_regressor,
        build_trial_and_weights_df,
        build_views,
        fig_size,
        get_adapter,
        load_fit_arrays,
        mo,
        mpatches,
        mpimg,
        np,
        paths,
        pd,
        pl,
        plt,
        prepare_predictions_df,
        re,
        sns,
    )


@app.cell
def _(ROOT, re):
    task_names = ("2AFC", "2AFC_delay", "MCDR")
    _hash_re = re.compile(r"^[A-Za-z0-9]{8}$")

    def saved_glm_model_names(task_name: str) -> list[str]:
        _fit_dir = ROOT / "results" / "fits" / task_name / "glm"
        if not _fit_dir.exists():
            return []
        return sorted(
            _item.name
            for _item in _fit_dir.iterdir()
            if _item.is_dir()
            and not _hash_re.fullmatch(_item.name)
            and any(_item.glob("*_glm_arrays.npz"))
        )

    model_names_by_task = {
        _task_name: saved_glm_model_names(_task_name)
        for _task_name in task_names
    }
    model_options = sorted(
        set.intersection(*[set(_names) for _names in model_names_by_task.values()])
    )
    if not model_options:
        model_options = sorted(
            set().union(*[set(_names) for _names in model_names_by_task.values()])
        )
    return model_options, task_names


@app.cell
def _(mo, model_options):
    mo.stop(not model_options, mo.md("No saved non-hash GLM models were found."))
    model_name = mo.ui.dropdown(
        options=model_options,
        value="one hot" if "one hot" in model_options else model_options[0],
        label="GLM model",
    )
    model_name
    return (model_name,)


@app.cell
def _(get_adapter, task_names):
    adapters = {_task_name: get_adapter(_task_name) for _task_name in task_names}
    # adapters = {"2AFC": get_adapter("2AFC")}
    # adapters["2AFC_delay"].get_plots()
    plots_by_task = {
        _task_name: _adapter.get_plots()
        for _task_name, _adapter in adapters.items()
    }
    adapters
    return adapters, plots_by_task


@app.cell
def _(adapters):
    dfs = {
        _task_name: _adapter.subject_filter(_adapter.read_dataset())
        for _task_name, _adapter in adapters.items()
    }
    # dfs["MCDR"] = dfs["MCDR"].filter(pl.col("batch") == "11B")

    subjects_by_task = {
        _task_name: list(_df["subject"].unique())
        for _task_name, _df in dfs.items()
    }
    # dfs["MCDR"]
    return dfs, subjects_by_task


@app.cell
def _(
    adapters,
    add_choice_lag_summary_regressor,
    build_trial_and_weights_df,
    build_views,
    dfs,
    load_fit_arrays,
    mo,
    model_name,
    paths,
    plots_by_task,
    prepare_predictions_df,
    subjects_by_task,
    task_names,
):
    def build_plot_payload(task_name: str) -> dict:
        _adapter = adapters[task_name]
        _df_all = dfs[task_name]
        _subjects = subjects_by_task[task_name]
        _out = paths.RESULTS / "fits" / task_name / "glm" / model_name.value
        _arrays_store, _ = load_fit_arrays(
            out_dir=_out,
            arrays_suffix="glm_arrays.npz",
            adapter=_adapter,
            df_all=_df_all,
            subjects=_subjects,
            emission_cols=None,
        )
        _selected_subjects = [
            _subject for _subject in _subjects if str(_subject) in _arrays_store
        ]
        mo.stop(
            not _selected_subjects,
            mo.md(f"No fitted subjects found for `{task_name}/glm/{model_name.value}`."),
        )
        _views = build_views(_arrays_store, _adapter, 1, _selected_subjects)
        _trial_df, _ = build_trial_and_weights_df(
            _df_all,
            views=_views,
            adapter=_adapter,
            min_session_length=1,
        )

        _choice_lag_cols = []
        for _view in _views.values():
            for _feature in list(getattr(_view, "feat_names", []) or []):
                _feature = str(_feature)
                if _feature.startswith("choice_lag_") and _feature not in _choice_lag_cols:
                    _choice_lag_cols.append(_feature)
        if not _choice_lag_cols:
            _choice_lag_cols = _adapter.choice_lag_cols(_trial_df)

        _plots = plots_by_task[task_name]
        _plot_df = prepare_predictions_df(task_name, _trial_df)
        _plot_df = add_choice_lag_summary_regressor(
            _plot_df,
            choice_lag_cols=_choice_lag_cols,
        )
        return {
            "adapter": _adapter,
            "plot_df": _plot_df,
            "plots": _plots,
            "views": _views,
        }

    plot_payloads = {
        _task_name: build_plot_payload(_task_name)
        for _task_name in task_names
    }
    mo.md(f"Loaded `{model_name.value}` for {', '.join(task_names)}.")
    return (plot_payloads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Session trial outcomes
    """)
    return


@app.cell
def _(mo, task_names):
    ui_trial_task = mo.ui.dropdown(
        options=list(task_names),
        value=task_names[0],
        label="Task",
    )
    return (ui_trial_task,)


@app.cell
def _(mo, plot_payloads, ui_trial_task):
    _plot_df = plot_payloads[ui_trial_task.value]["plot_df"]
    _subject_options = sorted(
        [subject for subject in _plot_df["subject"].unique().to_list() if subject is not None],
        key=lambda value: str(value),
    )
    mo.stop(not _subject_options, mo.md("No subjects available for the selected task."))
    ui_trial_subject = mo.ui.dropdown(
        options=_subject_options,
        value=_subject_options[0],
        label="Subject",
    )
    return (ui_trial_subject,)


@app.cell
def _(mo, pl, plot_payloads, ui_trial_subject, ui_trial_task):
    def _pick_existing_column(_df, _candidates):
        for _candidate in _candidates:
            if _candidate and _candidate in _df.columns:
                return _candidate
        return None

    _payload = plot_payloads[ui_trial_task.value]
    _adapter = _payload["adapter"]
    _plot_df = _payload["plot_df"]
    _session_col = _pick_existing_column(
        _plot_df,
        ["session", getattr(_adapter, "session_col", None), "Session"],
    )
    mo.stop(_session_col is None, mo.md("No session column found for the selected task."))

    _subject_df = _plot_df.filter(
        pl.col("subject").cast(pl.Utf8) == str(ui_trial_subject.value)
    )
    _session_options = sorted(
        [session for session in _subject_df[_session_col].unique().to_list() if session is not None],
        key=lambda value: str(value),
    )
    mo.stop(not _session_options, mo.md("No sessions available for the selected subject."))
    ui_trial_session = mo.ui.dropdown(
        options=_session_options,
        value=_session_options[0],
        label="Session",
    )
    return (ui_trial_session,)


@app.cell
def _(mo, ui_trial_session, ui_trial_subject, ui_trial_task):
    mo.hstack([ui_trial_task, ui_trial_subject, ui_trial_session], justify="start")
    return


@app.cell
def _(
    mo,
    np,
    pd,
    pl,
    plot_payloads,
    ui_trial_session,
    ui_trial_subject,
    ui_trial_task,
):
    def _pick_existing_column(_df, _candidates):
        for _candidate in _candidates:
            if _candidate and _candidate in _df.columns:
                return _candidate
        return None

    def _coerce_correct(_series):
        _numeric = pd.to_numeric(_series, errors="coerce")
        if _numeric.notna().any():
            return _numeric.fillna(0).astype(float).to_numpy() > 0
        return (
            _series.astype(str)
            .str.lower()
            .isin(["1", "true", "correct", "hit", "yes"])
            .to_numpy()
        )

    def _difficulty_values(_pdf, _task_name):
        def _numeric(_column, *, _abs=False):
            _values = pd.to_numeric(_pdf[_column], errors="coerce")
            if _abs:
                _values = _values.abs()
            return _values

        if _task_name == "2AFC" and "ILD" in _pdf.columns:
            return _numeric("ILD", _abs=True), "Difficulty (|ILD| dB)"

        if _task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            for _column in ["delay", "delays"]:
                if _column in _pdf.columns:
                    return _numeric(_column), "Difficulty (delay, s)"

        if _task_name == "MCDR":
            if "stimd_c" in _pdf.columns:
                return _pdf["stimd_c"].astype(str), "Difficulty"
            if "stimd_n" in _pdf.columns:
                return _numeric("stimd_n"), "Difficulty"

        for _column in ["difficulty", "stimd_n", "delay", "delays", "ILD", "stimulus"]:
            if _column in _pdf.columns:
                return _numeric(_column, _abs=_column in {"ILD", "stimulus"}), "Difficulty"

        raise ValueError("No difficulty-like column found for this task.")

    def _difficulty_labels(_difficulty):
        _difficulty = pd.Series(_difficulty).reset_index(drop=True)
        _numeric = pd.to_numeric(_difficulty, errors="coerce")
        if _numeric.notna().all():
            return _numeric.astype(float).map(lambda _value: f"{_value:g}")
        return _difficulty.astype(str)

    def build_session_trial_outcomes_data(_task_name, _subject, _session):
        _payload = plot_payloads[_task_name]
        _adapter = _payload["adapter"]
        _plot_df = _payload["plot_df"]
        _behavioral_cols = getattr(_adapter, "behavioral_cols", {}) or {}
        _session_col = _pick_existing_column(
            _plot_df,
            ["session", getattr(_adapter, "session_col", None), _behavioral_cols.get("session"), "Session"],
        )
        _trial_col = _pick_existing_column(
            _plot_df,
            ["trial", "trial_idx", _behavioral_cols.get("trial"), _behavioral_cols.get("trial_idx"), "Trial"],
        )
        _correct_col = _pick_existing_column(
            _plot_df,
            ["correct_bool", "performance", _behavioral_cols.get("performance"), "Hit", "hit"],
        )
        mo.stop(
            _session_col is None or _trial_col is None or _correct_col is None,
            mo.md("Session plot needs session, trial, and correctness columns."),
        )

        _session_df = _plot_df.filter(
            (pl.col("subject").cast(pl.Utf8) == str(_subject))
            & (pl.col(_session_col).cast(pl.Utf8) == str(_session))
        ).sort(_trial_col)
        mo.stop(_session_df.height == 0, mo.md("No trials for the selected subject/session."))

        _pdf = _session_df.to_pandas()
        _x = np.arange(len(_pdf), dtype=float)
        _xlabel = "Trial number (within session)"

        _difficulty, _ylabel = _difficulty_values(_pdf, _task_name)
        _difficulty = pd.Series(_difficulty)
        _valid = _difficulty.notna().to_numpy()
        mo.stop(not _valid.any(), mo.md("No valid difficulty values for this session."))

        _correct = _coerce_correct(_pdf[_correct_col])
        _x = _x[_valid]
        _difficulty_label = _difficulty_labels(_difficulty[_valid]).to_numpy()
        _correct = _correct[_valid]

        _edges = np.empty(len(_x) + 1, dtype=float)
        if len(_x) == 1:
            _edges[:] = [_x[0] - 0.5, _x[0] + 0.5]
        else:
            _midpoints = (_x[:-1] + _x[1:]) / 2.0
            _edges[1:-1] = _midpoints
            _edges[0] = _x[0] - (_midpoints[0] - _x[0])
            _edges[-1] = _x[-1] + (_x[-1] - _midpoints[-1])

        return (
            pd.DataFrame(
                {
                    "trial_x": _x,
                    "trial_left": _edges[:-1],
                    "trial_right": _edges[1:],
                    "difficulty_label": _difficulty_label,
                    "correct": _correct.astype(bool),
                }
            ),
            _xlabel,
            _ylabel,
        )

    session_trial_outcomes_data, session_trial_xlabel, _ = build_session_trial_outcomes_data(
        ui_trial_task.value,
        ui_trial_subject.value,
        ui_trial_session.value,
    )
    return session_trial_outcomes_data, session_trial_xlabel


@app.cell
def _(mo, session_trial_outcomes_data):
    _difficulty_options = session_trial_outcomes_data["difficulty_label"].drop_duplicates().to_list()
    _options = ["None", *_difficulty_options]
    ui_easy_difficulty = mo.ui.dropdown(
        options=_options,
        value="None",
        label="Easy difficulty",
    )
    ui_trial_tick_step = mo.ui.dropdown(
        options=[10, 20, 50, 100],
        value=20,
        label="Trial tick step",
    )
    mo.hstack([ui_easy_difficulty, ui_trial_tick_step], justify="start")
    return ui_easy_difficulty, ui_trial_tick_step


@app.cell
def _(
    mpatches,
    plt,
    session_trial_outcomes_data,
    session_trial_xlabel,
    sns,
    ui_easy_difficulty,
    ui_trial_tick_step,
):
    _df = session_trial_outcomes_data.copy().reset_index(drop=True)
    _has_easy_selection = ui_easy_difficulty.value != "None"
    _df["is_easy"] = (
        _has_easy_selection
        & (_df["difficulty_label"].astype(str) == str(ui_easy_difficulty.value))
    )
    _df["color"] = [
        "#006d2c" if _correct and _easy else
        "#2ca02c" if _correct else
        "#7f0000" if _easy else
        "#d62728"
        for _correct, _easy in zip(_df["correct"], _df["is_easy"], strict=False)
    ]

    _fig, _ax = plt.subplots(figsize=(6, 3), dpi=150)
    for _, _row in _df.iterrows():
        _ax.add_patch(
            mpatches.Rectangle(
                (_row["trial_left"], -0.38),
                _row["trial_right"] - _row["trial_left"],
                0.76,
                facecolor=_row["color"],
                alpha=0.75,
                linewidth=0,
            )
        )

    _ax.set_xlabel(session_trial_xlabel)
    _ax.set_xlim(_df["trial_left"].iloc[0], _df["trial_right"].iloc[-1])
    _last_trial = int(_df["trial_x"].max())
    _tick_step = int(ui_trial_tick_step.value)
    _xticks = list(range(0, _last_trial + 1, _tick_step))
    _ax.set_xticks(_xticks)
    _ax.set_ylim(-0.5, 0.5)
    _ax.set_yticks([])
    _ax.set_ylabel("")

    _legend_handles = [
        mpatches.Patch(color="#2ca02c", alpha=0.75, label="Correct"),
        mpatches.Patch(color="#d62728", alpha=0.75, label="Incorrect"),
    ]
    if _has_easy_selection:
        _legend_handles.extend(
            [
                mpatches.Patch(color="#006d2c", alpha=0.75, label=f"Easy correct ({ui_easy_difficulty.value})"),
                mpatches.Patch(color="#7f0000", alpha=0.75, label=f"Easy incorrect ({ui_easy_difficulty.value})"),
            ]
        )
    _ax.legend(
        handles=_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(_legend_handles),
        frameon=False,
    )
    sns.despine(ax=_ax, left=True)
    _fig.tight_layout()
    _fig.savefig("Error_trials_2AFC.pdf")
    _fig
    return


@app.cell
def _(ROOT, fig_size, mpimg, plot_payloads, plt, sns):
    _panel_width, _panel_height = fig_size(n_cols=3)

    fig, axd = plt.subplot_mosaic(
        [
            ["a", "a", "a"],
            ["b", "c", "d"],
            ["e", "f", "g"],
            ["h", "i", "j"],
        ],
        figsize=(_panel_width * 3, _panel_height * 4.8),
        constrained_layout=True,
        dpi=500,
        gridspec_kw={"height_ratios": [1.35, 1.0, 1.0, 1.0]},
    )

    _img = mpimg.imread(ROOT / "illustrations" / "glm.png")
    axd["a"].imshow(_img)
    axd["a"].axis("off")

    _panel_grid = [
        ("2AFC", ("b", "c", "d")),
        ("2AFC_delay", ("e", "f", "g")),
        ("MCDR", ("h", "i", "j")),
    ]
    for _task_name, (_repeat_key, _binned_key, _right_key) in _panel_grid:
        _payload = plot_payloads[_task_name]
        _plots = _payload["plots"]
        _plot_df = _payload["plot_df"]
        _views = _payload["views"]

        _plots.plot_repeat_by_repeat_evidence(
            _plot_df,
            views=_views,
            ax=axd[_repeat_key],
            legend=False,
            figsize=fig_size(n_cols=3),
            title="",
        )
        _plots.plot_binned_accuracy_figure(
            _plot_df,
            regressor_col="choice_lag_one_hot_sum",
            axes=[axd[_binned_key]],
            max_panels=1,
            legend=False,
            figsize=fig_size(n_cols=3),
        )
        _plots.plot_right_by_regressor(
            _plot_df,
            regressor_col="choice_lag_one_hot_sum",
            ax=axd[_right_key],
            title=None,
            legend=False,
            figsize=fig_size(n_cols=3),
        )
        axd[_repeat_key].set_ylabel(f"{_task_name}\n{axd[_repeat_key].get_ylabel()}")

    for _label, _ax in axd.items():
        _panel_label = "glm" if _label == "glm" else _label
        if _label != "a":
            _ax.text(
                -0.25,
                1.1,
                _panel_label,
                transform=_ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="right",
            )
            _ax.set_box_aspect(1)
            _ax.tick_params(axis="both", labelsize=6)
            _ax.xaxis.label.set_size(7)
            _ax.yaxis.label.set_size(7)
            _ax.title.set_size(7)
        else:
            _ax.text(
            -0.1,
            1.1,
            _panel_label,
            transform=_ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )

    sns.despine(fig=fig)
    fig.savefig("figure2.png")
    out_path = ROOT / "figures" / "__marimo__" / "assets" / "figure2" / "opengraph.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=300)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
