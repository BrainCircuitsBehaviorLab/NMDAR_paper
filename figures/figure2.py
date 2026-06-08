# /// script
# [tool.marimo.opengraph]
# title = "Figure 2" 
# description = " Figure 2: GLM model predictions."
# ///

import marimo

__generated_with = "0.23.8"
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
    sns.set_theme(style='ticks', context='notebook')
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
    return (
        Path,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example session repetition
    """)
    return


@app.cell
def _(mo):
    ui_repetition_window = mo.ui.dropdown(
        options=[10, 20, 50, 100],
        value=20,
        label="Repetition window (N trials)",
    )
    ui_repetition_window
    return (ui_repetition_window,)


@app.cell
def _(
    mo,
    np,
    pl,
    plot_payloads,
    ui_repetition_window,
    ui_trial_session,
    ui_trial_subject,
    ui_trial_task,
):
    def _pick_existing_column(_df, _candidates):
        for _candidate in _candidates:
            if _candidate and _candidate in _df.columns:
                return _candidate
        return None

    _payload = plot_payloads[ui_trial_task.value]
    _adapter = _payload["adapter"]
    _plot_df = _payload["plot_df"]
    _behavioral_cols = getattr(_adapter, "behavioral_cols", {}) or {}
    _session_col = _pick_existing_column(
        _plot_df,
        ["session", getattr(_adapter, "session_col", None), _behavioral_cols.get("session"), "Session"],
    )
    _trial_col = _pick_existing_column(
        _plot_df,
        ["trial_idx", "trial", _behavioral_cols.get("trial_idx"), _behavioral_cols.get("trial"), "Trial"],
    )
    _response_col = _pick_existing_column(
        _plot_df,
        ["response", _behavioral_cols.get("response"), "Choice", "choices"],
    )
    _stimulus_col = _pick_existing_column(
        _plot_df,
        ["stimulus", _behavioral_cols.get("stimulus"), "Side", "stim"],
    )
    mo.stop(
        _session_col is None or _trial_col is None or _response_col is None or _stimulus_col is None,
        mo.md("Example repetition plot needs session, trial, response, and stimulus columns."),
    )

    _session_df = _plot_df.filter(
        (pl.col("subject").cast(pl.Utf8) == str(ui_trial_subject.value))
        & (pl.col(_session_col).cast(pl.Utf8) == str(ui_trial_session.value))
    ).sort(_trial_col)
    mo.stop(_session_df.height == 0, mo.md("No trials for the selected subject/session."))

    session_repetition_data = (
        _session_df
        .select([_trial_col, _response_col, _stimulus_col])
        .to_pandas()
        .rename(
            columns={
                _trial_col: "trial",
                _response_col: "response",
                _stimulus_col: "stimulus",
            }
        )
        .reset_index(drop=True)
    )
    session_repetition_data["trial_x"] = np.arange(len(session_repetition_data))
    session_repetition_data["previous_response"] = session_repetition_data["response"].shift(1)
    session_repetition_data["previous_stimulus"] = session_repetition_data["stimulus"].shift(1)
    session_repetition_data["response_repeat"] = (
        session_repetition_data["response"].eq(session_repetition_data["previous_response"]).fillna(False)
    )
    session_repetition_data["stimulus_repeat"] = (
        session_repetition_data["stimulus"].eq(session_repetition_data["previous_stimulus"]).fillna(False)
    )
    _window = int(ui_repetition_window.value)
    session_repetition_data["response_repeat_window_count"] = (
        session_repetition_data["response_repeat"].astype(float).rolling(_window, min_periods=1).sum()
    )
    session_repetition_data["stimulus_repeat_window_count"] = (
        session_repetition_data["stimulus_repeat"].astype(float).rolling(_window, min_periods=1).sum()
    )
    return (session_repetition_data,)


@app.cell
def _(plt, session_repetition_data, sns):
    _response_labels = sorted(
        session_repetition_data["response"].dropna().unique(),
        key=lambda value: str(value),
    )
    _response_y = {value: index for index, value in enumerate(_response_labels)}
    _colors = sns.color_palette("tab10", n_colors=max(len(_response_labels), 1))

    fig_response_raster, ax_response_raster = plt.subplots(figsize=(6, 1.6), dpi=150)
    for _idx, _response in enumerate(_response_labels):
        _mask = session_repetition_data["response"].eq(_response)
        ax_response_raster.scatter(
            session_repetition_data.loc[_mask, "trial_x"],
            [_response_y[_response]] * int(_mask.sum()),
            s=10,
            color=_colors[_idx],
            label=str(_response),
        )

    ax_response_raster.set_xlabel("Trial number (within session)")
    ax_response_raster.set_ylabel("Response")
    ax_response_raster.set_yticks(list(_response_y.values()))
    ax_response_raster.set_yticklabels([str(label) for label in _response_labels])
    ax_response_raster.set_xlim(-0.5, len(session_repetition_data) - 0.5)
    if len(_response_labels) > 1:
        ax_response_raster.legend(title="Response", frameon=False, loc="upper right")
    sns.despine(ax=ax_response_raster)
    fig_response_raster.tight_layout()
    fig_response_raster
    return


@app.cell
def _(fig_size, plt, session_repetition_data, sns, ui_repetition_window):
    sns.set_context("paper")
    _window = int(ui_repetition_window.value)
    fig_session_repetition_running_count, ax_session_repetition_running_count = plt.subplots(
        figsize=fig_size(1,3),
        dpi=150,
    )
    ax_session_repetition_running_count.plot(
        session_repetition_data["trial_x"],
        session_repetition_data["response_repeat_window_count"],
        color="tab:brown",
        linewidth=1.5,
        label="Response repetition",
    )
    ax_session_repetition_running_count.plot(
        session_repetition_data["trial_x"],
        session_repetition_data["stimulus_repeat_window_count"],
        color="tab:blue",
        linewidth=1.5,
        label="Stimulus repetition",
    )
    ax_session_repetition_running_count.set_xlabel("Trial number (within session)")
    ax_session_repetition_running_count.set_ylabel(f"Repetitions / {_window} trials")
    ax_session_repetition_running_count.set_ylim(0, _window)
    ax_session_repetition_running_count.set_xlim(-0.5, len(session_repetition_data) - 0.5)
    ax_session_repetition_running_count.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax_session_repetition_running_count)
    fig_session_repetition_running_count.tight_layout()
    fig_session_repetition_running_count
    return


@app.cell
def _(Path, get_adapter, pl, ui_repetition_window):
    _window = int(ui_repetition_window.value)
    _data_path = Path(__file__).parents[1] / "data/processed/"
    _two_afc = get_adapter("2AFC")
    two_afc_drug_repetition_trials = (
        _two_afc
        .subject_filter(pl.read_parquet(_data_path / "alexis_drug_combined.parquet"))
        .select(["subject", "Session", "Trial", "Side", "Choice", "Drug"])
        .to_pandas()
        .sort_values(["subject", "Drug", "Session", "Trial"])
    )
    two_afc_drug_repetition_trials["drug_label"] = (
        two_afc_drug_repetition_trials["Drug"]
        .map({0: "Saline", 1: "Drug"})
        .fillna(two_afc_drug_repetition_trials["Drug"])
    )
    two_afc_drug_repetition_trials["previous_response"] = (
        two_afc_drug_repetition_trials
        .groupby(["subject", "Drug", "Session"], observed=True)["Choice"]
        .shift(1)
    )
    two_afc_drug_repetition_trials["previous_stimulus"] = (
        two_afc_drug_repetition_trials
        .groupby(["subject", "Drug", "Session"], observed=True)["Side"]
        .shift(1)
    )
    two_afc_drug_repetition_trials["response_repeat"] = (
        two_afc_drug_repetition_trials["Choice"]
        .eq(two_afc_drug_repetition_trials["previous_response"])
    )
    two_afc_drug_repetition_trials["stimulus_repeat"] = (
        two_afc_drug_repetition_trials["Side"]
        .eq(two_afc_drug_repetition_trials["previous_stimulus"])
    )

    def _rolling_repeat_fraction(_series):
        return _series.astype(float).rolling(_window, min_periods=_window).mean()

    two_afc_drug_repetition_trials["response_repeat_window_fraction"] = (
        two_afc_drug_repetition_trials
        .groupby(["subject", "Drug", "Session"], observed=True)["response_repeat"]
        .transform(_rolling_repeat_fraction)
    )
    two_afc_drug_repetition_trials["stimulus_repeat_window_fraction"] = (
        two_afc_drug_repetition_trials
        .groupby(["subject", "Drug", "Session"], observed=True)["stimulus_repeat"]
        .transform(_rolling_repeat_fraction)
    )
    two_afc_drug_repetition_variance = (
        two_afc_drug_repetition_trials
        .groupby(["subject", "Drug", "drug_label", "Session"], observed=True)
        .agg(
            response_repeat_variance=("response_repeat_window_fraction", "var"),
            stimulus_repeat_variance=("stimulus_repeat_window_fraction", "var"),
        )
        .reset_index()
    )
    two_afc_drug_repetition_variance_long = two_afc_drug_repetition_variance.melt(
        id_vars=["subject", "Drug", "drug_label", "Session"],
        value_vars=["response_repeat_variance", "stimulus_repeat_variance"],
        var_name="signal",
        value_name="variance",
    )
    two_afc_drug_repetition_variance_long["signal"] = (
        two_afc_drug_repetition_variance_long["signal"]
        .map(
            {
                "response_repeat_variance": "Response repetition",
                "stimulus_repeat_variance": "Stimulus repetition",
            }
        )
    )
    two_afc_drug_repetition_variance_long = two_afc_drug_repetition_variance_long.dropna(
        subset=["variance"]
    )
    stimulus_repeat_binomial_variance = 0.5 * (1 - 0.5) / _window
    return (
        stimulus_repeat_binomial_variance,
        two_afc_drug_repetition_variance_long,
    )


@app.cell
def _(
    fig_size,
    plt,
    sns,
    stimulus_repeat_binomial_variance,
    two_afc_drug_repetition_variance_long,
):
    fig_2afc_drug_repetition_variance, ax_2afc_drug_repetition_variance = plt.subplots(
        figsize=fig_size(2,1),
        dpi=150,
    )
    sns.boxplot(
        data=two_afc_drug_repetition_variance_long,
        x="signal",
        y="variance",
        hue="drug_label",
        palette={"Saline": "tab:gray", "Drug": "tab:pink"},
        showfliers=False,
        ax=ax_2afc_drug_repetition_variance,
        fill=False,
        whiskerprops = {"color": "gray"},
        boxprops = {"color": "gray"},
        medianprops = {"linewidth": 2},
        showcaps = False
    )
    # sns.stripplot(
    #     data=two_afc_drug_repetition_variance_long,
    #     x="signal",
    #     y="variance",
    #     hue="drug_label",
    #     palette={"Saline": "tab:gray", "Drug": "tab:pink"},
    #     dodge=True,
    #     alpha=0.35,
    #     size=2,
    #     ax=ax_2afc_drug_repetition_variance,
    # )
    ax_2afc_drug_repetition_variance.axhline(
        stimulus_repeat_binomial_variance,
        color="tab:blue",
        linestyle="--",
        label="Stim binomial",
    )
    ax_2afc_drug_repetition_variance.set_xlabel("")
    ax_2afc_drug_repetition_variance.set_ylabel("Variance of running fraction")
    _handles, _labels = ax_2afc_drug_repetition_variance.get_legend_handles_labels()
    _legend = dict(zip(_labels, _handles, strict=False))
    ax_2afc_drug_repetition_variance.legend(
        _legend.values(),
        _legend.keys(),
        frameon=False,
        loc="upper right",
    )
    sns.despine(ax=ax_2afc_drug_repetition_variance)
    fig_2afc_drug_repetition_variance.tight_layout()
    fig_2afc_drug_repetition_variance
    return


@app.cell
def _(pd, plot_payloads):
    two_afc_repeat_alternate_trials = (
        plot_payloads["2AFC"]["plot_df"]
        .select(["subject", "session", "trial_idx", "stimulus", "response", "performance"])
        .to_pandas()
        .sort_values(["subject", "session", "trial_idx"])
    )
    two_afc_repeat_alternate_trials["previous_response"] = (
        two_afc_repeat_alternate_trials
        .groupby(["subject", "session"], observed=True)["response"]
        .shift(1)
    )
    two_afc_repeat_alternate_trials = two_afc_repeat_alternate_trials.dropna(
        subset=["previous_response"]
    )
    two_afc_repeat_alternate_trials["transition"] = (
        two_afc_repeat_alternate_trials["response"]
        .eq(two_afc_repeat_alternate_trials["previous_response"])
        .map({True: "repeating", False: "alternating"})
    )
    two_afc_repeat_alternate_trials["transition_chunk"] = (
        two_afc_repeat_alternate_trials
        .groupby(["subject", "session"], observed=True)["transition"]
        .transform(lambda transition: transition.ne(transition.shift()).cumsum())
    )

    two_afc_transition_chunk_lengths = (
        two_afc_repeat_alternate_trials
        .groupby(["subject", "session", "transition", "transition_chunk"], observed=True)
        .size()
        .rename("chunk_length")
        .reset_index()
    )
    two_afc_transition_chunk_counts = (
        two_afc_transition_chunk_lengths
        .groupby(["subject", "transition", "chunk_length"], observed=True)
        .size()
        .rename("count")
        .reindex(
            pd.MultiIndex.from_product(
                [
                    two_afc_transition_chunk_lengths["subject"].unique(),
                    ["repeating", "alternating"],
                    range(1, int(two_afc_transition_chunk_lengths["chunk_length"].max()) + 1),
                ],
                names=["subject", "transition", "chunk_length"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    two_afc_transition_chunk_counts["frequency"] = (
        two_afc_transition_chunk_counts["count"]
        / two_afc_transition_chunk_counts.groupby(["subject", "transition"], observed=True)["count"].transform("sum")
    )
    two_afc_transition_chunk_frequency = (
        two_afc_transition_chunk_counts
        .groupby(["transition", "chunk_length"], observed=True)["frequency"]
        .mean()
        .reset_index()
    )
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

    two_afc_stimulus_repeat_alternate_trials = (
        plot_payloads["2AFC"]["plot_df"]
        .select(["subject", "session", "trial_idx", "stimulus"])
        .to_pandas()
        .sort_values(["subject", "session", "trial_idx"])
    )
    two_afc_stimulus_repeat_alternate_trials["previous_stimulus"] = (
        two_afc_stimulus_repeat_alternate_trials
        .groupby(["subject", "session"], observed=True)["stimulus"]
        .shift(1)
    )
    two_afc_stimulus_repeat_alternate_trials = two_afc_stimulus_repeat_alternate_trials.dropna(
        subset=["previous_stimulus"]
    )
    two_afc_stimulus_repeat_alternate_trials["transition"] = (
        two_afc_stimulus_repeat_alternate_trials["stimulus"]
        .eq(two_afc_stimulus_repeat_alternate_trials["previous_stimulus"])
        .map({True: "repeating", False: "alternating"})
    )
    two_afc_stimulus_repeat_alternate_trials["transition_chunk"] = (
        two_afc_stimulus_repeat_alternate_trials
        .groupby(["subject", "session"], observed=True)["transition"]
        .transform(lambda transition: transition.ne(transition.shift()).cumsum())
    )
    two_afc_stimulus_transition_chunk_lengths = (
        two_afc_stimulus_repeat_alternate_trials
        .groupby(["subject", "session", "transition", "transition_chunk"], observed=True)
        .size()
        .rename("chunk_length")
        .reset_index()
    )
    two_afc_stimulus_transition_chunk_counts = (
        two_afc_stimulus_transition_chunk_lengths
        .groupby(["subject", "transition", "chunk_length"], observed=True)
        .size()
        .rename("count")
        .reindex(
            pd.MultiIndex.from_product(
                [
                    two_afc_stimulus_transition_chunk_lengths["subject"].unique(),
                    ["repeating", "alternating"],
                    range(1, int(two_afc_stimulus_transition_chunk_lengths["chunk_length"].max()) + 1),
                ],
                names=["subject", "transition", "chunk_length"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    two_afc_stimulus_transition_chunk_counts["frequency"] = (
        two_afc_stimulus_transition_chunk_counts["count"]
        / two_afc_stimulus_transition_chunk_counts.groupby(["subject", "transition"], observed=True)["count"].transform("sum")
    )
    two_afc_stimulus_transition_chunk_frequency = (
        two_afc_stimulus_transition_chunk_counts
        .groupby(["transition", "chunk_length"], observed=True)["frequency"]
        .mean()
        .reset_index()
    )
    return (
        two_afc_repeat_alternate_trials,
        two_afc_session_transition_accuracy,
        two_afc_stimulus_transition_chunk_lengths,
        two_afc_transition_chunk_lengths,
    )


@app.cell
def _(two_afc_repeat_alternate_trials):
    two_afc_repeat_alternate_trials
    return


@app.cell
def _(two_afc_transition_chunk_lengths):
    two_afc_transition_chunk_lengths[two_afc_transition_chunk_lengths["transition"] == "alternating"]["chunk_length"].median()
    return


@app.cell
def _(two_afc_transition_chunk_lengths):
    two_afc_transition_chunk_lengths[two_afc_transition_chunk_lengths["transition"] == "repeating"]["chunk_length"].median()
    return


@app.cell
def _():
    chunk_hist_stat = "count"  # Use "probability" for relative frequencies.
    chunk_hist_ylabel = {"count": "Count", "probability": "Frequency"}[chunk_hist_stat]
    return chunk_hist_stat, chunk_hist_ylabel


@app.cell
def _(pd, plot_payloads, task_names):
    _task_labels = {"2AFC": "2AFC", "2AFC_delay": "2ADC", "MCDR": "MCDR"}

    def _transition_chunks_for_sequence(_plot_df, _task_name, _sequence_col, _sequence_label):
        _trials = (
            _plot_df
            .select(["subject", "session", "trial_idx", _sequence_col])
            .to_pandas()
            .dropna(subset=[_sequence_col])
            .sort_values(["subject", "session", "trial_idx"])
        )
        _trials["previous_value"] = (
            _trials.groupby(["subject", "session"], observed=True)[_sequence_col].shift(1)
        )
        _trials = _trials.dropna(subset=["previous_value"])
        _trials["transition"] = (
            _trials[_sequence_col]
            .eq(_trials["previous_value"])
            .map({True: "repeating", False: "alternating"})
        )
        _trials["transition_chunk"] = (
            _trials
            .groupby(["subject", "session"], observed=True)["transition"]
            .transform(lambda transition: transition.ne(transition.shift()).cumsum())
        )
        _chunks = (
            _trials
            .groupby(["subject", "session", "transition", "transition_chunk"], observed=True)
            .size()
            .rename("chunk_length")
            .reset_index()
        )
        _chunks["task"] = _task_name
        _chunks["task_label"] = _task_labels.get(_task_name, _task_name)
        _chunks["sequence"] = _sequence_label
        _p_repeat = (_trials["transition"] == "repeating").mean()
        return _chunks, {
            "task": _task_name,
            "task_label": _task_labels.get(_task_name, _task_name),
            "sequence": _sequence_label,
            "p_repeat": _p_repeat,
        }

    _chunk_frames = []
    _repeat_probability_rows = []
    for _task_name in task_names:
        _plot_df = plot_payloads[_task_name]["plot_df"]
        _available_columns = set(_plot_df.columns)
        for _sequence_col, _sequence_label in [
            ("response", "Choices"),
            ("stimulus", "Stimulus"),
        ]:
            if {"subject", "session", "trial_idx", _sequence_col}.issubset(_available_columns):
                _chunks, _repeat_probability = _transition_chunks_for_sequence(
                    _plot_df,
                    _task_name,
                    _sequence_col,
                    _sequence_label,
                )
                _chunk_frames.append(_chunks)
                _repeat_probability_rows.append(_repeat_probability)

    transition_chunk_lengths_by_task = pd.concat(_chunk_frames, ignore_index=True)
    transition_repeat_probabilities = pd.DataFrame(_repeat_probability_rows)
    return transition_chunk_lengths_by_task, transition_repeat_probabilities


@app.cell
def _(
    chunk_hist_stat,
    chunk_hist_ylabel,
    np,
    plt,
    transition_chunk_lengths_by_task,
    transition_repeat_probabilities,
):
    from matplotlib.lines import Line2D

    _task_order = ["2ADC", "2AFC", "MCDR"]
    _transition_palette = {"repeating": "tab:brown", "alternating": "tab:purple"}
    _max_chunk_length = 100

    fig_transition_chunks_by_task, axes_transition_chunks_by_task = plt.subplots(
        1,
        len(_task_order),
        figsize=(12,4),
        sharex=True,
        sharey=False,
    )

    def _geometric_chunk_probability(_chunk_lengths, _repeat_probability, _transition):
        _continue_probability = (
            _repeat_probability
            if _transition == "repeating"
            else 1.0 - _repeat_probability
        )
        return (1.0 - _continue_probability) * (_continue_probability ** (_chunk_lengths - 1))

    def _repeat_probability_for(_task_label, _sequence):
        _matches = transition_repeat_probabilities.loc[
            (transition_repeat_probabilities["task_label"] == _task_label)
            & (transition_repeat_probabilities["sequence"] == _sequence),
            "p_repeat",
        ]
        if _matches.empty:
            return None
        return float(_matches.iloc[0])

    for _ax, _task_label in zip(axes_transition_chunks_by_task, _task_order, strict=False):
        _data = transition_chunk_lengths_by_task[
            (transition_chunk_lengths_by_task["task_label"] == _task_label)
            & (transition_chunk_lengths_by_task["sequence"] == "Choices")
        ]
        if _data.empty:
            _ax.axis("off")
            continue

        _x = np.arange(1, _max_chunk_length + 1)
        _choice_probability = _repeat_probability_for(_task_label, "Choices")

        for _transition, _color in _transition_palette.items():
            _transition_data = _data[_data["transition"] == _transition]
            _transition_total = len(_transition_data)
            _animal_y = (
                _transition_data["chunk_length"]
                .value_counts()
                .reindex(_x, fill_value=0)
                .sort_index()
                .to_numpy(dtype=float)
            )
            if chunk_hist_stat == "probability" and _animal_y.sum() > 0:
                _animal_y = _animal_y / _animal_y.sum()
            _ax.plot(
                _x,
                _animal_y,
                color=_color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                zorder=3,
            )

            if _choice_probability is not None:
                _generated_y = _geometric_chunk_probability(
                    _x,
                    _choice_probability,
                    _transition,
                )
                if chunk_hist_stat == "count":
                    _generated_y = _generated_y * _transition_total
                _ax.plot(
                    _x,
                    _generated_y,
                    color=_color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=2,
                )

        _ax.set_xlim(0, 30)
        _ax.set_ylim(1,1e6)
        _ax.set_title(_task_label)
        _ax.set_xlabel("Chunk length")
        _ax.set_ylabel(chunk_hist_ylabel)
        _ax.set_yscale('log')
        if _ax.get_legend() is not None:
            _ax.get_legend().set_frame_on(False)
            _ax.get_legend().set_title("")

    _legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Animals"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Choice independent generated"),
        Line2D([0], [0], color="tab:brown", linestyle="-", linewidth=1.5, label="Repeating"),
        Line2D([0], [0], color="tab:purple", linestyle="-", linewidth=1.5, label="Alternating"),
    ]
    fig_transition_chunks_by_task.legend(
        handles=_legend_handles,
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig_transition_chunks_by_task.tight_layout()
    fig_transition_chunks_by_task
    return


@app.cell
def _(
    chunk_hist_stat,
    chunk_hist_ylabel,
    fig_size,
    plt,
    sns,
    two_afc_transition_chunk_lengths,
):
    fig_2afc_transition_chunk_lengths, ax_2afc_transition_chunk_lengths = plt.subplots(
        figsize=fig_size(2, 1)
    )
    sns.histplot(
        data=two_afc_transition_chunk_lengths,
        x="chunk_length",
        hue="transition",
        stat=chunk_hist_stat,
        discrete=True,
        common_norm=False,
        palette={"repeating": "tab:brown", "alternating": "tab:purple"},
        alpha=0.75,
        element="step",
        fill=False,
        ax=ax_2afc_transition_chunk_lengths,
    )

    ax_2afc_transition_chunk_lengths.set_xlim(0, 50)
    ax_2afc_transition_chunk_lengths.set_title("Choices")
    ax_2afc_transition_chunk_lengths.set_xlabel("Chunk length")
    ax_2afc_transition_chunk_lengths.set_ylabel(chunk_hist_ylabel)
    return


@app.cell
def _(Path, get_adapter, pl):
    data_path = Path(__file__).parents[1] / "data/processed/"
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
def _(pd, plot_payloads, plt, sns, task_names):
    _task_labels = {"2AFC": "2AFC", "2AFC_delay": "2ADC", "MCDR": "MCDR"}

    def _repeat_outcome_proportions_for_task(_task_name):
        _trials = (
            plot_payloads[_task_name]["plot_df"]
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
    chunk_hist_stat,
    chunk_hist_ylabel,
    fig_size,
    plt,
    sns,
    two_afc_transition_chunk_lengths_drug,
):
    fig_2afc_repeating_chunk_lengths_drug, ax_2afc_repeating_chunk_lengths_drug = plt.subplots(
        figsize=fig_size(2, 1)
    )
    sns.histplot(
        data=two_afc_transition_chunk_lengths_drug[
            two_afc_transition_chunk_lengths_drug["transition"] == "repeating"
        ],
        x="chunk_length",
        hue="drug_label",
        stat=chunk_hist_stat,
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
    ax_2afc_repeating_chunk_lengths_drug.set_ylabel(chunk_hist_ylabel)
    ax_2afc_repeating_chunk_lengths_drug
    return


@app.cell
def _(
    chunk_hist_stat,
    chunk_hist_ylabel,
    fig_size,
    np,
    plt,
    two_afc_transition_chunk_lengths_drug,
):
    _drug_palette = {"Saline": "tab:gray", "Drug": "tab:pink"}
    _max_chunk_length = 50
    _x = np.arange(1, _max_chunk_length + 1)
    _data = two_afc_transition_chunk_lengths_drug[
        two_afc_transition_chunk_lengths_drug["transition"] == "repeating"
    ]

    fig_2afc_repeating_chunk_lengths_drug_lines, ax_2afc_repeating_chunk_lengths_drug_lines = plt.subplots(
        figsize=fig_size(2, 1)
    )
    for _drug_label, _color in _drug_palette.items():
        _drug_data = _data[_data["drug_label"] == _drug_label]
        if _drug_data.empty:
            continue
        _counts = (
            _drug_data["chunk_length"]
            .value_counts()
            .reindex(_x, fill_value=0)
            .sort_index()
            .astype(float)
        )
        _y = _counts.to_numpy()
        if chunk_hist_stat == "probability" and _y.sum() > 0:
            _y = _y / _y.sum()
        ax_2afc_repeating_chunk_lengths_drug_lines.plot(
            _x,
            _y,
            color=_color,
            linewidth=1.5,
            label=_drug_label,
        )

    ax_2afc_repeating_chunk_lengths_drug_lines.set_xlim(0, 20)
    ax_2afc_repeating_chunk_lengths_drug_lines.set_title("Repeating choices")
    ax_2afc_repeating_chunk_lengths_drug_lines.set_xlabel("Chunk length")
    ax_2afc_repeating_chunk_lengths_drug_lines.set_ylabel(chunk_hist_ylabel)
    ax_2afc_repeating_chunk_lengths_drug_lines.legend(frameon=False)
    ax_2afc_repeating_chunk_lengths_drug_lines
    return


@app.cell
def _(
    chunk_hist_stat,
    chunk_hist_ylabel,
    fig_size,
    plt,
    sns,
    two_afc_transition_chunk_lengths_drug,
):
    fig_2afc_alternating_chunk_lengths_drug, ax_2afc_alternating_chunk_lengths_drug = plt.subplots(
        figsize=fig_size(2, 1)
    )
    sns.histplot(
        data=two_afc_transition_chunk_lengths_drug[
            two_afc_transition_chunk_lengths_drug["transition"] == "alternating"
        ],
        x="chunk_length",
        hue="drug_label",
        stat=chunk_hist_stat,
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
    ax_2afc_alternating_chunk_lengths_drug.set_ylabel(chunk_hist_ylabel)
    ax_2afc_alternating_chunk_lengths_drug
    return


@app.cell
def _(
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
    sns.histplot(
        data=two_afc_stimulus_transition_chunk_lengths,
        x="chunk_length",
        hue="transition",
        stat=chunk_hist_stat,
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
