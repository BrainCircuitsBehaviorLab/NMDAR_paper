import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import io
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
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
    from src.process import two_afc_delay as process_two_afc_delay
    from src.process.common import add_choice_lag_summary_regressor
    from figure_layout_widget import FigureLayoutWidget

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name == "2AFC_delay":
            return process_two_afc_delay.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=0.8)
    return (
        FigureLayoutWidget,
        ROOT,
        add_choice_lag_summary_regressor,
        base64,
        build_trial_and_weights_df,
        build_views,
        get_adapter,
        io,
        load_fit_arrays,
        mo,
        mpimg,
        np,
        paths,
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
    import polars as pl
    dfs["MCDR"] = dfs["MCDR"].filter(pl.col("batch") == "11B")

    subjects_by_task = {
        _task_name: list(_df["subject"].unique())
        for _task_name, _df in dfs.items()
    }
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


@app.cell
def _(np, plt):
    def fig_size(n_cols=1, ratio=None):
        """
        Get figure size for A4 page with n_cols columns and specified ratio (width/height).
        :param n_cols: Number of columns (0 for full page)
        :param ratio: Width/height ratio (None for default)
        :return:
        """

        if ratio is None:
            default_figsize = np.array(plt.rcParams["figure.figsize"])
            default_ratio = default_figsize[0] / default_figsize[1]
            ratio = default_ratio  # 4:3

        # All measurements are in inches
        A4_size = np.array((8.27, 11.69))  # A4 measurements
        margins = 2  # On both dimension
        size = A4_size - margins  # Effective size after margins removal (2 per dimension)
        width = size[0]
        height = size[1]

        # Full page (minus margins)
        if n_cols == 0:
            # Full A4 minus margins
            figsize = (width, height)
            if ratio == 1:  # Square
                figsize = (size[0], size[0])
            return figsize

        else:
            fig_width = width / n_cols
            fig_height = fig_width / ratio
            figsize = (fig_width, fig_height)
            return figsize

    return (fig_size,)


@app.cell
def _(fig_size, plot_payloads):
    def render_axis_plot(task_name: str, plot_key: str, ax):
        _payload = plot_payloads[task_name]
        _plots = _payload["plots"]
        _plot_df = _payload["plot_df"]
        _views = _payload["views"]

        if plot_key == "repeat_evidence":
            _plots.plot_repeat_by_repeat_evidence(
                _plot_df,
                views=_views,
                ax=ax,
                legend=False,
                figsize=fig_size(n_cols=3),
                title="",
            )
        elif plot_key == "binned_accuracy":
            _plots.plot_binned_accuracy_figure(
                _plot_df,
                regressor_col="choice_lag_one_hot_sum",
                axes=[ax],
                max_panels=1,
                legend=False,
                figsize=fig_size(n_cols=3),
            )
        else:
            _plots.plot_right_by_regressor(
                _plot_df,
                regressor_col="choice_lag_one_hot_sum",
                ax=ax,
                title=None,
                legend=False,
                figsize=fig_size(n_cols=3),
            )

    return (render_axis_plot,)


@app.cell
def _(FigureLayoutWidget, base64, io, mo, plt, render_axis_plot, task_names):
    _plot_options = []
    _plot_specs = (
        ("repeat_evidence", "Repeat evidence"),
        ("binned_accuracy", "Binned accuracy"),
        ("right_by_regressor", "Right by regressor"),
    )

    def _render_plot_png(task_name: str, plot_key: str) -> str:
        _fig, _ax = plt.subplots(figsize=(2.2, 1.7), constrained_layout=True, dpi=120)
        render_axis_plot(task_name, plot_key, _ax)
        _ax.tick_params(axis="both", labelsize=5)
        _ax.xaxis.label.set_size(6)
        _ax.yaxis.label.set_size(6)
        _ax.title.set_size(6)
        _buffer = io.BytesIO()
        _fig.savefig(_buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(_fig)
        return "data:image/png;base64," + base64.b64encode(_buffer.getvalue()).decode("ascii")

    _default_placements = []
    for _task_name in task_names:
        _placement_row = []
        for _plot_key, _plot_label in _plot_specs:
            _value = f"{_task_name}:{_plot_key}"
            _plot_options.append(
                {
                    "group": _task_name,
                    "label": _plot_label,
                    "value": _value,
                    "image": _render_plot_png(_task_name, _plot_key),
                    "code_template": f"render_axis_plot({_task_name!r}, {_plot_key!r}, {{axis}})",
                }
            )
            _placement_row.append(_value)
        _default_placements.append(_placement_row)

    layout_widget = mo.ui.anywidget(
        FigureLayoutWidget(
            rows=3,
            cols=3,
            placements=_default_placements,
            plot_options=_plot_options,
        )
    )
    layout_widget
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
