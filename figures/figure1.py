# /// script
# [tool.marimo.opengraph]
# title = "Figure 1" 
# description = " Figure 1: Behavioral performance across tasks."
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from plot_saver import make_plot_saver
    from glmhmmt.tasks import get_adapter
    from glmhmmt.runtime import configure_paths
    from glmhmmt.tasks.fitted_regressors import FittedWeightRegressorSpec, mean_feature_weights_from_fit
    import os

    from src.utils import fig_size

    configure_paths(config_path=Path(__file__).resolve().parents[1] / "config.toml")
    return (
        FittedWeightRegressorSpec,
        Path,
        fig_size,
        get_adapter,
        make_plot_saver,
        mean_feature_weights_from_fit,
        mo,
        np,
        pd,
        pl,
        plt,
        sns,
    )


@app.cell
def _(Path, plt, sns):
    # Set style
    sns.set_theme(style='ticks', context='notebook')
    # style_path = os.path.expanduser('~/PycharmProjects/alexis_style.mplstyle')
    plt.style.use(Path(__file__).resolve().parents[1] / "styles" / "paper.mplstyle")
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
    return data_path, save_plot


@app.cell
def _(get_adapter):
    # Get adapters
    MCDR = get_adapter("MCDR")
    two_afc = get_adapter("2AFC")
    two_afc_delay = get_adapter("2AFC_delay")
    return MCDR, two_afc, two_afc_delay


@app.cell
def _(MCDR, data_path, pl, two_afc):
    df_2AFC = two_afc.subject_filter(pl.read_parquet(data_path / "alexis_combined.parquet"))
    df_2AFC_delay = pl.read_parquet(data_path / "tiffany.parquet")
    df_MCDR = MCDR.subject_filter(pl.read_parquet(data_path / "MCDR_all.parquet"))
    # df_MCDR = df_MCDR.filter(pl.col("batch") == "11B")
    df_2AFC
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
def _(df_2AFC_delay, fig_size, pl, plt, two_afc_delay_plots):
    # 2ADC
    two_afc_delay_plots.plot_accuracy(df_2AFC_delay, figsize=fig_size(n_cols=3), title='')
    plt.savefig('acc_vs_delay.svg')
    plt.show()

    # two_afc_delay_plots.plot_rb(df_2AFC_delay, figsize=fig_size(n_cols=3), title='')
    # plt.savefig('2ADC_rb.svg')
    # plt.show()
    fig_2ADC, ax_2ADC = plt.subplots(figsize = fig_size(n_cols=3))
    two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "NR2B"), ax = ax_2ADC, figsize=fig_size(n_cols=3), title='', color = "tab:pink")
    two_afc_delay_plots.plot_rb(df_2AFC_delay.filter(pl.col("drug") == "Saline"), ax = ax_2ADC, figsize=fig_size(n_cols=3), title='', color = "tab:gray")
    plt.savefig('2ADC_rb.svg')
    plt.show()
    return


@app.cell
def _(df_2AFC):
    df_2AFC
    return


@app.cell
def _(df_2AFC, fig_size, pl, plt, two_afc_plots):
    # 2AFC
    two_afc_plots.plot_accuracy(df_2AFC, figsize=fig_size(n_cols=3), title='')
    plt.savefig('acc_vs_ild.svg')
    plt.show()

    fig_2AFC, ax_2AFC = plt.subplots(figsize = fig_size(n_cols=3))
    two_afc_plots.plot_rb(df_2AFC.filter(pl.col("Drug") == 1), ax = ax_2AFC, figsize=fig_size(n_cols=3), title='', color = "tab:pink")
    two_afc_plots.plot_rb(df_2AFC.filter(pl.col("Drug") == 0), ax = ax_2AFC, figsize=fig_size(n_cols=3), title='', color = "tab:gray")
    plt.savefig('2AFC_rb.svg')
    plt.show()
    return


@app.cell
def _(MCDR_plots, df_MCDR, fig_size, pl, plt):
    MCDR_plots.plot_accuracy(df_MCDR, figsize=fig_size(n_cols=3), title='')
    plt.savefig('acc_vs_difficulty.svg')
    plt.show()

    fig_MCDR, ax_MCDR = plt.subplots(figsize = fig_size(n_cols=3))
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='', color = "tab:gray")
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='',  color = "tab:pink")
    MCDR_plots.plot_rb(df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"), ax = ax_MCDR, figsize=fig_size(n_cols=3), title='',  color = "tab:red")
    # ax.set_ylim(0.2,0.6)

    plt.savefig('MCDR_rb.svg')
    plt.show()
    return


@app.cell
def _(
    MCDR_plots,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    fig_size,
    pl,
    plt,
    two_afc_delay_plots,
    two_afc_plots,
):
    from matplotlib.lines import Line2D

    fig_rb_mosaic, axes_rb_mosaic = plt.subplot_mosaic(
        [["delay", "afc", "mcdr", "mcdr3"]],
        figsize=(12,3),
        layout="constrained",
    )

    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "NR2B"),
        ax=axes_rb_mosaic["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "Saline"),
        ax=axes_rb_mosaic["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "Rest"),
        ax=axes_rb_mosaic["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_mosaic["delay"].set_title("2AFC delay")

    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 1),
        ax=axes_rb_mosaic["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=axes_rb_mosaic["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=axes_rb_mosaic["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    axes_rb_mosaic["afc"].set_title("2AFC")

    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"),
        ax=axes_rb_mosaic["mcdr"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"),
        ax=axes_rb_mosaic["mcdr"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"),
        ax=axes_rb_mosaic["mcdr"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_mosaic["mcdr"].set_title("MCDR11")

    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "3B"),
        ax=axes_rb_mosaic["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "3B"),
        ax=axes_rb_mosaic["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "3B"),
        ax=axes_rb_mosaic["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_mosaic["mcdr3"].set_title("MCDR3B")

    for _ax in axes_rb_mosaic.values():
        _legend = _ax.get_legend()
        if _legend is not None:
            _legend.remove()

    fig_rb_mosaic.legend(
        handles=[
            Line2D([0], [0], color="tab:pink", marker="o", linewidth=1.5, label="Drug"),
            Line2D([0], [0], color="tab:gray", marker="o", linewidth=1.5, label="Saline"),
            Line2D([0], [0], color="tab:red", marker="o", linewidth=1.5, label="Rest"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    plt.show()
    return (Line2D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Strategy cartoons
    """)
    return


@app.cell
def _(FittedWeightRegressorSpec, mean_feature_weights_from_fit, np, pd, pl):
    STRATEGY_PALETTE = {
        "Mixture": "#d55e00",
        "Additive": "#009e73",
    }
    _ONE_HOT_MODEL_ID = "one hot"
    _BINARY_CHOICE_LAG_NAMES = tuple(f"choice_lag_{_idx:02d}" for _idx in range(1, 16))
    _MCDR_SIDES = ("L", "C", "R")
    _MCDR_STIM_FEATURES = tuple(
        f"stim{_stim_idx}{_side}"
        for _stim_idx in range(1, 5)
        for _side in _MCDR_SIDES
    )

    def _cartoon_sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def _cartoon_softmax(logits):
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def _fitted_weight_spec(
        *,
        target_name,
        task,
        source_features=(),
        source_feature_prefixes=(),
        exclude_features=(),
        class_idx=0,
    ):
        return FittedWeightRegressorSpec(
            target_name=target_name,
            fit_task=task,
            fit_model_kind="glm",
            fit_model_id=_ONE_HOT_MODEL_ID,
            arrays_suffix="glm_arrays.npz",
            source_features=tuple(source_features),
            source_feature_prefixes=tuple(source_feature_prefixes),
            exclude_features=tuple(exclude_features),
            class_idx=int(class_idx),
        )

    def _fitted_weight_map(
        *,
        target_name,
        task,
        source_features=(),
        source_feature_prefixes=(),
        exclude_features=(),
        class_idx=0,
    ):
        return mean_feature_weights_from_fit(
            _fitted_weight_spec(
                target_name=target_name,
                task=task,
                source_features=source_features,
                source_feature_prefixes=source_feature_prefixes,
                exclude_features=exclude_features,
                class_idx=class_idx,
            )
        )

    def _weights_for_names(weight_map, names):
        return np.asarray([float(weight_map.get(_name, 0.0)) for _name in names], dtype=float)

    def _add_binary_empirical_lags(df, *, n_lags=15):
        out = df.sort_values(["subject", "session", "trial"]).copy()
        for _lag_idx in range(1, int(n_lags) + 1):
            out[f"empirical_choice_lag_{_lag_idx:02d}"] = (
                out.groupby(["subject", "session"], observed=True)["empirical_choice"]
                .shift(_lag_idx)
                .fillna(0.0)
                .astype(float)
            )
        return out

    def _add_mcdr_empirical_lags(df, *, n_lags=15):
        out = df.sort_values(["subject", "session", "trial"]).copy()
        for _lag_idx in range(1, int(n_lags) + 1):
            out[f"empirical_choice_lag_{_lag_idx:02d}"] = (
                out.groupby(["subject", "session"], observed=True)["empirical_choice"]
                .shift(_lag_idx)
                .fillna(-1)
                .astype(int)
            )
        return out

    def build_binary_sequence_template(
        df,
        *,
        signed_x_expr,
        choice_expr,
        subject_col,
        session_col,
        trial_col,
        x_label,
    ):
        out = (
            df.with_columns(
                pl.lit(x_label).alias("x_label"),
                signed_x_expr.alias("signed_x"),
            )
            .select(
                pl.col(subject_col).cast(pl.String).alias("subject"),
                pl.col(session_col).cast(pl.String).alias("session"),
                pl.col(trial_col).cast(pl.Int64).alias("trial"),
                pl.col("signed_x").cast(pl.Float64),
                choice_expr.cast(pl.Float64).alias("empirical_choice"),
                pl.col("x_label"),
            )
            .sort(["subject", "session", "trial"])
            .with_columns(pl.col("signed_x").abs().alias("rep_x"))
        )
        return _add_binary_empirical_lags(out.to_pandas())

    def build_mcdr_sequence_template(df):
        difficulty_labels = ["VG", "Easy", "Mid", "Hard"]
        out = (
            df.filter(pl.col("stimd_n") > 0)
            .select(
                pl.col("subject").cast(pl.String).alias("subject"),
                pl.col("session").cast(pl.String).alias("session"),
                pl.col("trial").cast(pl.Int64).alias("trial"),
                (4 - pl.col("stimd_n").cast(pl.Int64)).alias("difficulty_code"),
                pl.col("stimulus").cast(pl.Int64).alias("target"),
                pl.col("response").cast(pl.Int64).alias("empirical_choice"),
            )
            .sort(["subject", "session", "trial"])
        )
        out = out.with_columns(
            pl.col("difficulty_code")
            .replace_strict({0: "VG", 1: "Easy", 2: "Mid", 3: "Hard"})
            .alias("difficulty")
        )
        pdf = _add_mcdr_empirical_lags(out.to_pandas())
        pdf["rep_x"] = pd.Categorical(pdf["difficulty"], categories=difficulty_labels, ordered=True)
        return pdf

    def _load_binary_cartoon_weights(task):
        if task == "2AFC":
            return {
                "stim": _fitted_weight_map(
                    target_name="strategy_cartoon_2afc_stim",
                    task="2AFC",
                    source_feature_prefixes=("stim_",),
                    exclude_features=("stim_0",),
                ),
                "choice_lag": _fitted_weight_map(
                    target_name="strategy_cartoon_2afc_choice_lag",
                    task="2AFC",
                    source_features=_BINARY_CHOICE_LAG_NAMES,
                ),
            }
        if task == "2AFC delay":
            return {
                "stim": _fitted_weight_map(
                    target_name="strategy_cartoon_2afc_delay_stim_x_delay",
                    task="2AFC_delay",
                    source_feature_prefixes=("stim_x_delay_hot_",),
                ),
                "choice_lag": _fitted_weight_map(
                    target_name="strategy_cartoon_2afc_delay_choice_lag",
                    task="2AFC_delay",
                    source_features=_BINARY_CHOICE_LAG_NAMES,
                ),
            }
        raise ValueError(f"No fitted binary cartoon weights configured for task={task!r}.")

    def _load_mcdr_cartoon_weights():
        return {
            "stim_L": _fitted_weight_map(
                target_name="strategy_cartoon_mcdr_stim_L",
                task="MCDR",
                source_features=_MCDR_STIM_FEATURES,
                class_idx=0,
            ),
            "stim_R": _fitted_weight_map(
                target_name="strategy_cartoon_mcdr_stim_R",
                task="MCDR",
                source_features=_MCDR_STIM_FEATURES,
                class_idx=1,
            ),
            "choice_lag_L": _fitted_weight_map(
                target_name="strategy_cartoon_mcdr_choice_lag_L",
                task="MCDR",
                source_feature_prefixes=("choice_lag_",),
                class_idx=0,
            ),
            "choice_lag_R": _fitted_weight_map(
                target_name="strategy_cartoon_mcdr_choice_lag_R",
                task="MCDR",
                source_feature_prefixes=("choice_lag_",),
                class_idx=1,
            ),
        }

    def _linear_from_feature_dict(feature_dict, weight_map, *, n_rows):
        out = np.zeros(int(n_rows), dtype=float)
        for _name, _values in feature_dict.items():
            out += np.asarray(_values, dtype=float) * float(weight_map.get(_name, 0.0))
        return out

    def _cartoon_feature_token(value):
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    def _binary_stim_one_hot_design(signed_values, signed_levels, *, feature_prefix):
        signed_values = np.asarray(signed_values, dtype=float)
        abs_levels = sorted({abs(float(_level)) for _level in signed_levels})
        nonzero_abs = [float(_level) for _level in abs_levels if not np.isclose(_level, 0.0)]
        names = []
        columns = []
        if any(np.isclose(_level, 0.0) for _level in abs_levels):
            names.append("stim_0")
            columns.append(np.isclose(signed_values, 0.0).astype(float))
        for _abs_level in nonzero_abs:
            _col = np.select(
                [np.isclose(signed_values, _abs_level), np.isclose(signed_values, -_abs_level)],
                [1.0, -1.0],
                default=0.0,
            )
            names.append(f"{feature_prefix}_{_cartoon_feature_token(_abs_level)}")
            columns.append(_col.astype(float))
        return np.column_stack(columns), names

    def _normalize_history_mode(history_mode):
        value = str(history_mode).strip().lower().replace("-", "_").replace(" ", "_")
        if value in {"open", "open_loop"}:
            return "open_loop"
        if value in {"closed", "closed_loop"}:
            return "closed_loop"
        raise ValueError(f"Unknown history_mode={history_mode!r}; expected 'open_loop' or 'closed_loop'.")

    def _clean_binary_choice(value):
        if pd.isna(value):
            return 0.0
        value = float(value)
        if np.isclose(value, 0.0):
            return 0.0
        return 1.0 if value > 0.0 else -1.0

    def _clean_mcdr_choice(value):
        if pd.isna(value):
            return -1
        value = int(value)
        return value if 0 <= value <= 2 else -1

    def _sample_binary_task_cartoon(
        rng,
        *,
        task,
        template_df,
        mix_action=0.5,
        action_gain=1.0,
        history_mode="open_loop",
        n_lags=15,
    ):
        history_mode = _normalize_history_mode(history_mode)
        use_simulated_history = history_mode == "closed_loop"
        stim_feature_prefix = "stim_x_delay_hot" if "delay" in task.lower() else "stim"
        fitted_weights = _load_binary_cartoon_weights(task)
        choice_lag_names = [f"choice_lag_{_idx:02d}" for _idx in range(1, int(n_lags) + 1)]
        choice_weights = float(action_gain) * _weights_for_names(
            fitted_weights["choice_lag"],
            choice_lag_names,
        )
        signed_levels = sorted(template_df["signed_x"].dropna().unique().tolist())
        rows = []
        for (_subject, _session), _group in template_df.groupby(["subject", "session"], sort=False):
            _group = _group.sort_values("trial")
            for _model in ["Mixture", "Additive"]:
                _history = np.zeros(int(n_lags), dtype=float)
                _model_rows = []
                for _row in _group.itertuples(index=False):
                    if not use_simulated_history:
                        _history = np.asarray(
                            [
                                getattr(_row, f"empirical_choice_lag_{_lag_idx:02d}")
                                for _lag_idx in range(1, int(n_lags) + 1)
                            ],
                            dtype=float,
                        )
                    _stim_values = np.asarray([float(_row.signed_x)], dtype=float)
                    _stim_design, _stim_names = _binary_stim_one_hot_design(
                        _stim_values,
                        signed_levels,
                        feature_prefix=stim_feature_prefix,
                    )
                    _stim_weights = _weights_for_names(fitted_weights["stim"], _stim_names)
                    _stim_logit = float((_stim_design @ _stim_weights)[0])
                    _action_trace = float(_history.sum())
                    _action_logit = float(_history @ choice_weights)
                    _prev_choice = float(_history[0])
                    _p_stim = float(_cartoon_sigmoid(_stim_logit))
                    _p_action = float(_cartoon_sigmoid(_action_logit))
                    if _model == "Mixture":
                        _p_right = ((1.0 - mix_action) * _p_stim) + (mix_action * _p_action)
                    else:
                        _p_right = float(_cartoon_sigmoid(_stim_logit + _action_logit))
                    _choice_right = bool(rng.random() < _p_right)
                    _choice_signed = 1.0 if _choice_right else -1.0
                    _row_data = {
                        "task": task,
                        "subject": _subject,
                        "session": _session,
                        "model": _model,
                        "trial": int(_row.trial),
                        "signed_x": float(_row.signed_x),
                        "rep_x": float(_row.rep_x),
                        "x_label": _row.x_label,
                        "history_mode": history_mode,
                        "action_gain": float(action_gain),
                        "stim_logit": _stim_logit,
                        "action_logit": _action_logit,
                        "at": _action_trace,
                        "at_abs": abs(_action_trace),
                        "prev_choice": _prev_choice,
                        "choice": _choice_signed,
                        "choice_right": float(_choice_right),
                        "repeat": float(_choice_signed == _prev_choice),
                    }
                    _row_data.update(
                        {
                            _stim_name: float(_stim_design[0, _stim_idx])
                            for _stim_idx, _stim_name in enumerate(_stim_names)
                        }
                    )
                    _row_data.update(
                        {
                            f"choice_lag_{_lag_idx + 1:02d}": float(_history[_lag_idx])
                            for _lag_idx in range(int(n_lags))
                        }
                    )
                    _model_rows.append(_row_data)
                    if use_simulated_history:
                        _history = np.concatenate(([float(_choice_signed)], _history[:-1]))
                rows.append(pd.DataFrame(_model_rows))
        return pd.concat(rows, ignore_index=True)

    def _sample_mcdr_cartoon(
        rng,
        *,
        template_df,
        mix_action=0.35,
        action_gain=1.0,
        history_mode="open_loop",
        n_lags=15,
    ):
        history_mode = _normalize_history_mode(history_mode)
        use_simulated_history = history_mode == "closed_loop"
        difficulties = ["VG", "Easy", "Mid", "Hard"]
        sides = _MCDR_SIDES
        fitted_weights = _load_mcdr_cartoon_weights()
        rows = []
        for (_subject, _session), _group in template_df.groupby(["subject", "session"], sort=False):
            _group = _group.sort_values("trial")
            for _model in ["Mixture", "Additive"]:
                _history = np.full(int(n_lags), -1, dtype=int)
                _model_rows = []
                for _row in _group.itertuples(index=False):
                    if not use_simulated_history:
                        _history = np.asarray(
                            [
                                getattr(_row, f"empirical_choice_lag_{_lag_idx:02d}")
                                for _lag_idx in range(1, int(n_lags) + 1)
                            ],
                            dtype=int,
                        )
                    _difficulty = _row.difficulty
                    _difficulty_idx = int(_row.difficulty_code)
                    _target_idx = int(_row.target)
                    _stim_hot_row = {
                        f"stim{_stim_idx + 1}{_side}": float(
                            (_difficulty_idx == _stim_idx) and (_target_idx == _class_idx)
                        )
                        for _stim_idx in range(len(difficulties))
                        for _class_idx, _side in enumerate(sides)
                    }
                    _choice_lag_hot_row = {
                        f"choice_lag_{_lag_idx + 1:02d}{_side}": float(
                            (_history[_lag_idx] >= 0) and (_history[_lag_idx] == _class_idx)
                        )
                        for _lag_idx in range(int(n_lags))
                        for _class_idx, _side in enumerate(sides)
                    }
                    _stim_logits = np.zeros(3, dtype=float)
                    _stim_logits[0] = float(_linear_from_feature_dict(
                        _stim_hot_row,
                        fitted_weights["stim_L"],
                        n_rows=1,
                    )[0])
                    _stim_logits[2] = float(_linear_from_feature_dict(
                        _stim_hot_row,
                        fitted_weights["stim_R"],
                        n_rows=1,
                    )[0])
                    _action_logits = np.zeros(3, dtype=float)
                    _action_logits[0] = float(_linear_from_feature_dict(
                        _choice_lag_hot_row,
                        fitted_weights["choice_lag_L"],
                        n_rows=1,
                    )[0])
                    _action_logits[2] = float(_linear_from_feature_dict(
                        _choice_lag_hot_row,
                        fitted_weights["choice_lag_R"],
                        n_rows=1,
                    )[0])
                    _action_logits *= float(action_gain)
                    _prev_choice = int(_history[0]) if _history[0] >= 0 else -1
                    _repeat_action_trace = float(_action_logits[_prev_choice]) if _prev_choice >= 0 else 0.0
                    _stim_probs = _cartoon_softmax(_stim_logits[None, :])[0]
                    _action_probs = _cartoon_softmax(_action_logits[None, :])[0]
                    if _model == "Mixture":
                        _probs = ((1.0 - mix_action) * _stim_probs) + (mix_action * _action_probs)
                    else:
                        _probs = _cartoon_softmax((_stim_logits + _action_logits)[None, :])[0]
                    _choice = int(rng.choice(3, p=_probs))
                    _model_rows.append(
                        {
                            "task": "MCDR",
                            "subject": _subject,
                            "session": _session,
                            "model": _model,
                            "history_mode": history_mode,
                            "action_gain": float(action_gain),
                            "trial": int(_row.trial),
                            "difficulty": _difficulty,
                            "difficulty_code": _difficulty_idx,
                            "rep_x": _row.rep_x,
                            "at_abs": _repeat_action_trace,
                            "target": _target_idx,
                            "prev_choice": _prev_choice,
                            "choice": _choice,
                            "correct": float(_choice == _target_idx),
                            "repeat": float((_prev_choice >= 0) and (_choice == _prev_choice)),
                            **_stim_hot_row,
                            **_choice_lag_hot_row,
                        }
                    )
                    if use_simulated_history:
                        _history = np.concatenate(([int(_choice)], _history[:-1]))
                rows.append(pd.DataFrame(_model_rows))
        out = pd.concat(rows, ignore_index=True)
        out["rep_x"] = pd.Categorical(out["rep_x"], categories=difficulties, ordered=True)
        return out

    def simulate_strategy_cartoon_data(
        seed=11,
        *,
        mix_action=0.35,
        action_gain=1.0,
        history_mode="open_loop",
        template_2afc,
        template_2afc_delay,
        template_mcdr,
        tasks=("2AFC", "2ADC"),
    ):
        rng = np.random.default_rng(int(seed))
        mix_action = float(mix_action)
        action_gain = float(action_gain)
        history_mode = _normalize_history_mode(history_mode)
        selected_tasks = set(tasks)
        include_2afc_delay = bool({"2AFC delay", "2ADC"} & selected_tasks)
        two_afc = (
            _sample_binary_task_cartoon(
                rng,
                task="2AFC",
                template_df=template_2afc,
                mix_action=mix_action,
                action_gain=action_gain,
                history_mode=history_mode,
            )
            if "2AFC" in selected_tasks and template_2afc is not None
            else pd.DataFrame()
        )
        two_afc_delay = (
            _sample_binary_task_cartoon(
                rng,
                task="2AFC delay",
                template_df=template_2afc_delay,
                mix_action=mix_action,
                action_gain=action_gain,
                history_mode=history_mode,
            )
            if include_2afc_delay and template_2afc_delay is not None
            else pd.DataFrame()
        )
        mcdr = (
            _sample_mcdr_cartoon(
                rng,
                template_df=template_mcdr,
                mix_action=mix_action,
                action_gain=action_gain,
                history_mode=history_mode,
            )
            if "MCDR" in selected_tasks and template_mcdr is not None
            else pd.DataFrame()
        )
        return two_afc, two_afc_delay, mcdr

    def subject_curve(df, *, value_col, x_col, model_col="model", subject_col="subject"):
        subject = (
            df.groupby([model_col, subject_col, x_col], observed=True)[value_col]
            .mean()
            .reset_index(name="subject_mean")
        )
        agg = (
            subject.groupby([model_col, x_col], observed=True)["subject_mean"]
            .agg(mean="mean", sem=lambda _x: float(_x.std(ddof=1) / np.sqrt(max(len(_x), 1))))
            .reset_index()
        )
        agg["sem"] = agg["sem"].fillna(0.0)
        return agg

    def repetition_curve(
        df,
        *,
        x_col,
        model_col="model",
        subject_col="subject",
        choice_col="choice",
        prev_choice_col="prev_choice",
    ):
        work = df.dropna(subset=[model_col, subject_col, x_col, choice_col, prev_choice_col]).copy()
        choice_set = sorted(work[choice_col].dropna().unique())
        rows = []
        for (_model, _subject, _x_val), _group in work.groupby(
            [model_col, subject_col, x_col],
            observed=True,
            sort=False,
        ):
            _probs = []
            for _choice in choice_set:
                _side = _group[_group[prev_choice_col] == _choice]
                if _side.empty:
                    continue
                _probs.append((_side[choice_col] == _choice).mean())
            if _probs:
                rows.append(
                    {
                        model_col: _model,
                        subject_col: _subject,
                        x_col: _x_val,
                        "subject_mean": float(np.mean(_probs)),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=[model_col, x_col, "mean", "sem"])
        subject = pd.DataFrame(rows)
        agg = (
            subject.groupby([model_col, x_col], observed=True)["subject_mean"]
            .agg(mean="mean", sem=lambda _x: float(_x.std(ddof=1) / np.sqrt(max(len(_x), 1))))
            .reset_index()
        )
        agg["sem"] = agg["sem"].fillna(0.0)
        return agg

    def rep_x_labels(df, summary, *, x_col="rep_x"):
        for _source in (df, summary):
            if _source is None or x_col not in _source:
                continue
            _series = _source[x_col]
            _categories = getattr(getattr(_series, "dtype", None), "categories", None)
            if _categories is not None:
                return list(_categories)
            _values = _series.dropna().tolist()
            if _values:
                return list(dict.fromkeys(_values))
        return []

    def add_at_bins(df, *, n_bins=4):
        out = df.copy()
        labels = [f"Q{_idx + 1}" for _idx in range(int(n_bins))]
        out["at_bin"] = pd.qcut(out["at"], q=int(n_bins), labels=labels, duplicates="drop")
        return out

    def add_signed_x_bins(df, *, n_bins=None):
        out = df.copy()
        unique_signed = sorted(out["signed_x"].dropna().unique())
        if n_bins is None or len(unique_signed) <= int(n_bins):
            out["signed_x_bin"] = pd.Categorical(
                out["signed_x"],
                categories=unique_signed,
                ordered=True,
            )
        else:
            out["signed_x_bin"] = pd.qcut(out["signed_x"], q=int(n_bins), duplicates="drop")
        return out

    def add_action_trace_x_bins(df, *, n_bins=9):
        out = df.copy()
        out["at_x_bin"] = pd.qcut(out["at"], q=int(n_bins), duplicates="drop")
        centers = (
            out.groupby("at_x_bin", observed=True)["at"]
            .mean()
            .reset_index(name="at_x_center")
        )
        return out.merge(centers, on="at_x_bin", how="left")

    return (
        STRATEGY_PALETTE,
        add_action_trace_x_bins,
        add_at_bins,
        add_signed_x_bins,
        build_binary_sequence_template,
        build_mcdr_sequence_template,
        rep_x_labels,
        repetition_curve,
        simulate_strategy_cartoon_data,
    )


@app.cell
def _(
    build_binary_sequence_template,
    build_mcdr_sequence_template,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    pl,
    ui_strategy_tasks,
):
    _strategy_tasks = set(ui_strategy_tasks.value)
    strategy_template_2afc = (
        build_binary_sequence_template(
            df_2AFC,
            signed_x_expr=pl.col("ILD"),
            choice_expr=(2.0 * pl.col("Choice")) - 1.0,
            subject_col="subject",
            session_col="Session",
            trial_col="Trial",
            x_label="ILD (dB)",
        )
        if "2AFC" in _strategy_tasks
        else None
    )
    strategy_template_2afc_delay = (
        build_binary_sequence_template(
            df_2AFC_delay,
            signed_x_expr=pl.col("stim") * pl.col("delays"),
            choice_expr=pl.col("choices"),
            subject_col="subject",
            session_col="session",
            trial_col="trial",
            x_label="Signed delay (s)",
        )
        if "2ADC" in _strategy_tasks or "2AFC delay" in _strategy_tasks
        else None
    )
    strategy_template_mcdr = (
        build_mcdr_sequence_template(df_MCDR)
        if "MCDR" in _strategy_tasks
        else None
    )
    return (
        strategy_template_2afc,
        strategy_template_2afc_delay,
        strategy_template_mcdr,
    )


@app.cell
def _(plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def _draw_box(ax, xy, width, height, text, *, fc="#ffffff", ec="#222222"):
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.015",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )
        return patch

    def _draw_arrow(ax, start, end):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color="#333333",
            )
        )

    def make_strategy_model_cartoon():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), layout="constrained")
        for _ax in axes:
            _ax.set_xlim(0, 1)
            _ax.set_ylim(0, 1)
            _ax.axis("off")

        _ax = axes[0]
        _ax.set_title("Mixture policy", fontsize=10)
        _draw_box(_ax, (0.05, 0.65), 0.28, 0.18, "Stimulus\npolicy", fc="#eaf2fb")
        _draw_box(_ax, (0.05, 0.18), 0.28, 0.18, "Action-trace\npolicy", fc="#fdebd3")
        _draw_box(_ax, (0.45, 0.42), 0.18, 0.18, "Latent\nswitch", fc="#f3f3f3")
        _draw_box(_ax, (0.75, 0.42), 0.18, 0.18, "Choice", fc="#ffffff")
        _draw_arrow(_ax, (0.33, 0.74), (0.45, 0.54))
        _draw_arrow(_ax, (0.33, 0.27), (0.45, 0.46))
        _draw_arrow(_ax, (0.63, 0.51), (0.75, 0.51))
        _ax.text(0.35, 0.83, "some trials", fontsize=8, ha="center")
        _ax.text(0.35, 0.15, "other trials", fontsize=8, ha="center")

        _ax = axes[1]
        _ax.set_title("Additive GLM", fontsize=10)
        _draw_box(_ax, (0.05, 0.65), 0.28, 0.18, "Stimulus\nevidence", fc="#eaf2fb")
        _draw_box(_ax, (0.05, 0.18), 0.28, 0.18, "Action-trace\nevidence", fc="#fdebd3")
        _draw_box(_ax, (0.45, 0.42), 0.20, 0.18, "Weighted\nsum", fc="#edf5e9")
        _draw_box(_ax, (0.76, 0.42), 0.18, 0.18, "Choice", fc="#ffffff")
        _draw_arrow(_ax, (0.33, 0.74), (0.45, 0.54))
        _draw_arrow(_ax, (0.33, 0.27), (0.45, 0.46))
        _draw_arrow(_ax, (0.65, 0.51), (0.76, 0.51))
        _ax.text(0.36, 0.83, "always used", fontsize=8, ha="center")
        _ax.text(0.36, 0.15, "always used", fontsize=8, ha="center")
        return fig

    return (make_strategy_model_cartoon,)


@app.cell
def _(make_strategy_model_cartoon, mo, save_plot):
    strategy_model_cartoon = make_strategy_model_cartoon()
    mo.vstack(
        [
            strategy_model_cartoon,
            save_plot(
                strategy_model_cartoon,
                "strategy model cartoon",
                stem="strategy_model_cartoon",
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    ui_strategy_mix = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.35,
        label="Mixture action-trace probability",
    )
    ui_strategy_action_gain = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.05,
        value=1.0,
        label="Action-trace parameter gain",
    )
    ui_strategy_history_mode = mo.ui.dropdown(
        options=["open_loop", "closed_loop"],
        value="open_loop",
        label="Choice-history mode",
    )
    ui_strategy_tasks = mo.ui.multiselect(
        options=["2AFC", "2ADC", "MCDR"],
        value=["2AFC", "2ADC"],
        label="Strategy cartoon tasks",
    )
    mo.hstack(
        [
            ui_strategy_mix,
            ui_strategy_action_gain,
            ui_strategy_history_mode,
            ui_strategy_tasks,
        ],
        justify="start",
    )
    return (
        ui_strategy_action_gain,
        ui_strategy_history_mode,
        ui_strategy_mix,
        ui_strategy_tasks,
    )


@app.cell
def _(
    STRATEGY_PALETTE,
    mo,
    np,
    plt,
    rep_x_labels,
    repetition_curve,
    save_plot,
    simulate_strategy_cartoon_data,
    strategy_template_2afc,
    strategy_template_2afc_delay,
    strategy_template_mcdr,
    ui_strategy_action_gain,
    ui_strategy_history_mode,
    ui_strategy_mix,
    ui_strategy_tasks,
):
    _selected_tasks = set(ui_strategy_tasks.value)
    cartoon_2afc, cartoon_2afc_delay, cartoon_mcdr = simulate_strategy_cartoon_data(
        mix_action=float(ui_strategy_mix.value),
        action_gain=float(ui_strategy_action_gain.value),
        history_mode=ui_strategy_history_mode.value,
        template_2afc=strategy_template_2afc,
        template_2afc_delay=strategy_template_2afc_delay,
        template_mcdr=strategy_template_mcdr,
        tasks=tuple(_selected_tasks),
    )

    tasks = []
    if "2AFC" in _selected_tasks and not cartoon_2afc.empty:
        tasks.append(("2AFC", cartoon_2afc, "|ILD| (dB)", 0.5, True))
    if ({"2ADC", "2AFC delay"} & _selected_tasks) and not cartoon_2afc_delay.empty:
        tasks.append(("2ADC", cartoon_2afc_delay, "Delay (s)", 0.5, False))
    if "MCDR" in _selected_tasks and not cartoon_mcdr.empty:
        tasks.append(("MCDR", cartoon_mcdr, "Difficulty", 1 / 3, False))

    if not tasks:
        fig_rep, _ax = plt.subplots(figsize=(3.2, 2.1), layout="constrained")
        _ax.text(0.5, 0.5, "Select at least one task", ha="center", va="center")
        _ax.axis("off")
    else:
        fig_rep, _axes_rep = plt.subplots(
            1,
            len(tasks),
            figsize=(max(3.0, 2.8 * len(tasks)), 2.5),
            layout="constrained",
            squeeze=False,
        )
        _axes_rep = _axes_rep.ravel()
        for _ax, (_title, _df, _xlabel, _baseline, _invert) in zip(_axes_rep, tasks, strict=True):
            _summary = repetition_curve(_df, x_col="rep_x")
            for _model in ["Mixture", "Additive"]:
                _sub = _summary[_summary["model"] == _model].copy()
                _sub = _sub.sort_values("rep_x")
                _x = np.arange(len(_sub)) if _title == "MCDR" else _sub["rep_x"].to_numpy(dtype=float)
                _y = _sub["mean"].to_numpy(dtype=float)
                _sem = _sub["sem"].to_numpy(dtype=float)
                _ax.plot(_x, _y, marker="o", color=STRATEGY_PALETTE[_model], label=_model)
                _ax.fill_between(
                    _x,
                    np.clip(_y - _sem, 0.0, 1.0),
                    np.clip(_y + _sem, 0.0, 1.0),
                    color=STRATEGY_PALETTE[_model],
                    alpha=0.14,
                    linewidth=0.0,
                )
            _ax.axhline(_baseline, color="0.55", linestyle="--", linewidth=0.9)
            _ax.set_title(_title)
            _ax.set_xlabel(_xlabel)
            _ax.set_ylabel("Rep. bias")
            _ax.set_ylim(0.3, 1)
            if _title == "MCDR":
                _ax.set_ylim(0.2, 0.5)
                _labels = rep_x_labels(_df, _summary)
                _ax.set_xticks(np.arange(len(_labels)))
                _ax.set_xticklabels(_labels)
            elif _invert:
                _ax.invert_xaxis()
            if _ax is _axes_rep[-1]:
                _ax.legend(frameon=False, fontsize=8)
    mo.vstack(
        [
            fig_rep,
            save_plot(
                fig_rep,
                "strategy cartoon repetition bias",
                stem="strategy_cartoon_repetition_bias",
            ),
        ],
        align="center",
    )
    fig_rep
    return cartoon_2afc, cartoon_2afc_delay, cartoon_mcdr


@app.cell
def _(
    Line2D,
    MCDR_plots,
    STRATEGY_PALETTE,
    cartoon_2afc,
    cartoon_2afc_delay,
    cartoon_mcdr,
    df_2AFC,
    df_2AFC_delay,
    df_MCDR,
    fig_size,
    mo,
    np,
    pl,
    plt,
    rep_x_labels,
    repetition_curve,
    two_afc_delay_plots,
    two_afc_plots,
    ui_strategy_tasks,
):
    mo.stop(
        set(ui_strategy_tasks.value) != {"2AFC", "2ADC", "MCDR"},
        mo.md("Select 2AFC, 2ADC, and MCDR to render the full empirical/cartoon mosaic."),
    )

    fig_rb_combined, axes_rb_combined = plt.subplot_mosaic(
        [
            ["delay", "afc", "mcdr11", "mcdr3"],
            ["cartoon_delay", "cartoon_afc", "cartoon_mcdr", "cartoon_mcdr2"],
        ],
        figsize=(12, 5.8),
        layout="constrained",
    )

    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "NR2B"),
        ax=axes_rb_combined["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "Saline"),
        ax=axes_rb_combined["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    two_afc_delay_plots.plot_rb(
        df_2AFC_delay.filter(pl.col("drug") == "Rest"),
        ax=axes_rb_combined["delay"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_combined["delay"].set_title("2AFC delay")
    axes_rb_combined["delay"].set_xticks([0.1, 1, 3, 10])
    axes_rb_combined["delay"].set_xticklabels(["0.1", "1", "3", "10"])

    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 1),
        ax=axes_rb_combined["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    two_afc_plots.plot_rb(
        df_2AFC.filter(pl.col("Drug") == 0),
        ax=axes_rb_combined["afc"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    axes_rb_combined["afc"].set_title("2AFC")
    axes_rb_combined["afc"].set_xticks([0, 2, 4, 8, 20])
    axes_rb_combined["afc"].set_xticklabels(["0", "2", "4", "8", "20"])
    axes_rb_combined["afc"].invert_xaxis()

    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "11B"),
        ax=axes_rb_combined["mcdr11"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "11B"),
        ax=axes_rb_combined["mcdr11"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "11B"),
        ax=axes_rb_combined["mcdr11"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_combined["mcdr11"].set_title("MCDR11")
    axes_rb_combined["mcdr11"].set_xticks([0, 1, 2, 3])
    axes_rb_combined["mcdr11"].set_xticklabels(["VG", "Easy", "Mid", "Hard"])

    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "saline", pl.col("batch") == "3B"),
        ax=axes_rb_combined["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:gray",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "drug", pl.col("batch") == "3B"),
        ax=axes_rb_combined["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:pink",
    )
    MCDR_plots.plot_rb(
        df_MCDR.filter(pl.col("drug") == "rest", pl.col("batch") == "3B"),
        ax=axes_rb_combined["mcdr3"],
        figsize=fig_size(n_cols=3),
        title="",
        color="tab:red",
    )
    axes_rb_combined["mcdr3"].set_title("MCDR3B")
    axes_rb_combined["mcdr3"].set_xticks([0, 1, 2, 3])
    axes_rb_combined["mcdr3"].set_xticklabels(["VG", "Easy", "Mid", "Hard"])

    cartoon_tasks = [
        ("cartoon_delay", "2AFC delay cartoon", cartoon_2afc_delay, "Delay (s)", False),
        ("cartoon_afc", "2AFC cartoon", cartoon_2afc, "|ILD| (dB)", True),
        ("cartoon_mcdr", "MCDR cartoon", cartoon_mcdr, "Difficulty", False),
    ]
    for _key, _title, _df, _xlabel, _invert in cartoon_tasks:
        _ax = axes_rb_combined[_key]
        _summary = repetition_curve(_df, x_col="rep_x")
        for _model in ["Mixture", "Additive"]:
            _sub = _summary[_summary["model"] == _model].copy().sort_values("rep_x")
            _x = np.arange(len(_sub)) if _title == "MCDR cartoon" else _sub["rep_x"].to_numpy(dtype=float)
            _y = _sub["mean"].to_numpy(dtype=float)
            _sem = _sub["sem"].to_numpy(dtype=float)
            _ax.plot(_x, _y, marker="o", color=STRATEGY_PALETTE[_model], label=_model)
            _ax.fill_between(
                _x,
                np.clip(_y - _sem, 0.0, 1.0),
                np.clip(_y + _sem, 0.0, 1.0),
                color=STRATEGY_PALETTE[_model],
                alpha=0.14,
                linewidth=0.0,
            )
        _ax.axhline(1 / 3 if _title == "MCDR cartoon" else 0.5, color="0.55", linestyle="--", linewidth=0.9)
        _ax.set_title(_title)
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel("Rep. bias")
        _ax.set_ylim(0,1)
        if _title == "MCDR cartoon":
            _labels = rep_x_labels(_df, _summary)
            _ax.set_xticks(np.arange(len(_labels)))
            _ax.set_xticklabels(_labels)
            # _ax.set_ylim(0.2, 0.5)
        elif _title == "2AFC delay cartoon":
            _ax.set_xticks([0.1, 1, 3, 10])
            _ax.set_xticklabels(["0.1", "1", "3", "10"])
        elif _title == "2AFC cartoon":
            _ax.set_xticks([0, 2, 4, 8, 20])
            _ax.set_xticklabels(["0", "2", "4", "8", "20"])
            _ax.invert_xaxis()
        elif _invert:
            _ax.invert_xaxis()

    for _key, _ax in axes_rb_combined.items():
        _legend = _ax.get_legend()
        if _legend is not None:
            _legend.remove()

    fig_rb_combined.legend(
        handles=[
            Line2D([0], [0], color="tab:pink", marker="o", linewidth=1.5, label="Drug"),
            Line2D([0], [0], color="tab:gray", marker="o", linewidth=1.5, label="Saline"),
            Line2D([0], [0], color="tab:red", marker="o", linewidth=1.5, label="Rest"),
            Line2D([0], [0], color=STRATEGY_PALETTE["Mixture"], marker="o", linewidth=1.5, label="Mixture"),
            Line2D([0], [0], color=STRATEGY_PALETTE["Additive"], marker="o", linewidth=1.5, label="Additive"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        frameon=False,
    )
    fig_rb_combined
    return


@app.cell
def _(
    add_action_trace_x_bins,
    add_at_bins,
    add_signed_x_bins,
    cartoon_2afc,
    cartoon_2afc_delay,
    mo,
    np,
    pd,
    plt,
    save_plot,
    ui_strategy_tasks,
):
    mo.stop(
        not {"2AFC", "2ADC"}.issubset(set(ui_strategy_tasks.value)),
        mo.md("Select both 2AFC and 2ADC to render the binary psychometric cartoon panels."),
    )

    fig_psych, _axes_psych = plt.subplots(
        2,
        4,
        figsize=(10.5, 5.2),
        layout="constrained",
        sharey=True,
    )
    task_rows = [
        ("2AFC", cartoon_2afc, "Signed ILD (dB)", "Signed ILD bin"),
        ("2ADC", cartoon_2afc_delay, "Signed delay (s)", "Signed delay bin"),
    ]
    for _row_idx, (_task_name, _df_task, _stim_label, _bin_label) in enumerate(task_rows):
        _df_at = add_at_bins(_df_task, n_bins=4)
        _df_signed = add_action_trace_x_bins(add_signed_x_bins(_df_task), n_bins=9)
        for _model_idx, _model in enumerate(["Mixture", "Additive"]):
            _model_df_at = _df_at[_df_at["model"] == _model].copy()
            _model_df_signed = _df_signed[_df_signed["model"] == _model].copy()

            _ax = _axes_psych[_row_idx, _model_idx * 2]
            _psych = (
                _model_df_at.groupby(["at_bin", "signed_x"], observed=True)["choice_right"]
                .mean()
                .reset_index()
            )
            _delay_order = [-0.1, -1.0, -3.0, -10.0, 10.0, 3.0, 1.0, 0.1]
            _at_bins = list(_psych["at_bin"].dropna().unique())
            _colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(_at_bins)))
            for _at_bin, _color in zip(_at_bins, _colors, strict=False):
                _sub = _psych[_psych["at_bin"] == _at_bin].sort_values("signed_x")
                _x = _sub["signed_x"]
                if _task_name == "2ADC":
                    _sub["signed_x"] = pd.Categorical(
                        _sub["signed_x"],
                        categories=_delay_order,
                        ordered=True,
                    )
                    _sub = _sub.sort_values("signed_x")
                    _x = np.arange(len(_delay_order))
                _ax.plot(
                    _x,
                    _sub["choice_right"],
                    marker="o",
                    linewidth=1.1,
                    color=_color,
                    label=str(_at_bin),
                )
            _ax.axhline(0.5, color="0.65", linestyle="--", linewidth=0.8)
            if _task_name == "2ADC":
                _ax.axvline(3.5, color="0.65", linestyle="--", linewidth=0.8)
            else:
                _ax.axvline(0.0, color="0.65", linestyle="--", linewidth=0.8)
            _ax.set_title(f"{_task_name}: {_model}\nby action trace")
            _ax.set_xlabel(_stim_label)
            _ax.set_ylabel("p(right)")
            _ax.set_ylim(0.0, 1.0)
            if _task_name == "2ADC":
                _ax.set_xticks(np.arange(8))
                _ax.set_xticklabels(["0", "-1", "-3", "-10", "10", "3", "1", "0"])
                _ax.set_xlim(-0.4, 7.4)
            if _row_idx == 0 and _model_idx == 1:
                _ax.legend(title="AT bin", frameon=False, fontsize=7, title_fontsize=8)

            _ax = _axes_psych[_row_idx, _model_idx * 2 + 1]
            _at_curve = (
                _model_df_signed.groupby(
                    ["signed_x_bin", "at_x_bin", "at_x_center"],
                    observed=True,
                )["choice_right"]
                .mean()
                .reset_index()
            )
            _x_bins = list(_at_curve["signed_x_bin"].dropna().unique())
            _colors = plt.cm.magma(np.linspace(0.18, 0.85, len(_x_bins)))
            for _x_bin, _color in zip(_x_bins, _colors, strict=False):
                _sub = _at_curve[_at_curve["signed_x_bin"] == _x_bin].sort_values("at_x_center")
                _ax.plot(
                    _sub["at_x_center"],
                    _sub["choice_right"],
                    marker="o",
                    linewidth=1.2,
                    color=_color,
                    label=str(_x_bin),
                )
            _ax.axhline(0.5, color="0.65", linestyle="--", linewidth=0.8)
            _ax.axvline(0.0, color="0.65", linestyle="--", linewidth=0.8)
            _ax.set_title(f"{_task_name}: {_model}\nby stimulus strength")
            _ax.set_xlabel("Action trace")
            _ax.set_xlim(-15.5, 15.5)
            _ax.set_xticks([-10, 0, 10])
            _ax.set_ylim(0.0, 1.0)
            if _row_idx == 0 and _model_idx == 1:
                _ax.legend(title=_bin_label, frameon=False, fontsize=7, title_fontsize=8)

    for _ax in _axes_psych.ravel():
        _ax.label_outer()
    mo.vstack(
        [
            fig_psych,
            save_plot(
                fig_psych,
                "strategy cartoon psychometrics", 
                stem="strategy_cartoon_psychometrics",
            ),
        ],
        align="center",
    )
    fig_psych
    return


@app.cell
def _(
    cartoon_2afc,
    cartoon_2afc_delay,
    df_2AFC,
    df_2AFC_delay,
    mo,
    np,
    pd,
    plt,
    save_plot,
    ui_strategy_tasks,
):
    mo.stop(
        not {"2AFC", "2ADC"}.issubset(set(ui_strategy_tasks.value)),
        mo.md("Select both 2AFC and 2ADC to render the empirical/model psychometric overlay."),
    )

    def _as_pandas(_df):
        if hasattr(_df, "to_pandas"):
            return _df.to_pandas()
        return _df.copy()

    def _empirical_binary_psych_df(
        _df,
        *,
        subject_col,
        session_col,
        trial_col,
        signed_x_col=None,
        signed_x_expr=None,
        choice_col,
        choice_transform,
        lag_count=15,
    ):
        _out = _as_pandas(_df)
        if signed_x_expr is None:
            _out["signed_x"] = pd.to_numeric(_out[signed_x_col], errors="coerce")
        else:
            _out["signed_x"] = signed_x_expr(_out)
        _choice_signed = choice_transform(_out[choice_col])
        _out["choice_right"] = (_choice_signed > 0).astype(float)
        _out["choice_signed"] = _choice_signed
        _out = _out.sort_values([subject_col, session_col, trial_col]).copy()
        _out["at"] = 0.0
        for _, _idx in _out.groupby([subject_col, session_col], sort=False).groups.items():
            _choices = _out.loc[_idx, "choice_signed"].fillna(0.0).to_numpy(dtype=float)
            _at = np.zeros(len(_choices), dtype=float)
            for _trial_idx in range(len(_choices)):
                _start = max(0, _trial_idx - int(lag_count))
                _at[_trial_idx] = _choices[_start:_trial_idx].sum()
            _out.loc[_idx, "at"] = _at
        return _out[["signed_x", "choice_right", "at"]].dropna()

    def _ordered_delay_curve(_sub, _delay_order):
        _sub = _sub.copy()
        _sub["signed_x"] = pd.Categorical(
            _sub["signed_x"],
            categories=_delay_order,
            ordered=True,
        )
        _sub = _sub.sort_values("signed_x")
        _x = _sub["signed_x"].cat.codes.to_numpy(dtype=float)
        return _sub, _x

    def _shared_edges(_model_df, _data_df, *, n_bins):
        _values = pd.concat(
            [
                pd.to_numeric(_model_df["at"], errors="coerce"),
                pd.to_numeric(_data_df["at"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        if _values.nunique() <= 1:
            _center = float(_values.iloc[0]) if len(_values) else 0.0
            return np.array([_center - 0.5, _center + 0.5], dtype=float)
        _, _edges = pd.qcut(_values, q=int(n_bins), retbins=True, duplicates="drop")
        _edges = np.unique(_edges.astype(float))
        _edges[0] = -np.inf
        _edges[-1] = np.inf
        return _edges

    def _assign_at_bins(_df, _edges, *, prefix):
        _out = _df.copy()
        _labels = [f"{prefix}{_idx + 1}" for _idx in range(len(_edges) - 1)]
        return pd.cut(
            _out["at"],
            bins=_edges,
            labels=_labels,
            include_lowest=True,
            ordered=True,
        )

    def _with_shared_at_bins(_model_df, _data_df, *, n_bins=4):
        _edges = _shared_edges(_model_df, _data_df, n_bins=n_bins)
        _model_out = _model_df.copy()
        _data_out = _data_df.copy()
        _model_out["at_bin"] = _assign_at_bins(_model_out, _edges, prefix="Q")
        _data_out["at_bin"] = _assign_at_bins(_data_out, _edges, prefix="Q")
        return _model_out, _data_out

    def _with_shared_signed_x_bins(_model_df, _data_df):
        _categories = sorted(
            pd.concat([_model_df["signed_x"], _data_df["signed_x"]], ignore_index=True)
            .dropna()
            .unique()
        )
        _model_out = _model_df.copy()
        _data_out = _data_df.copy()
        _model_out["signed_x_bin"] = pd.Categorical(
            _model_out["signed_x"],
            categories=_categories,
            ordered=True,
        )
        _data_out["signed_x_bin"] = pd.Categorical(
            _data_out["signed_x"],
            categories=_categories,
            ordered=True,
        )
        return _model_out, _data_out

    def _with_shared_action_trace_bins(_model_df, _data_df, *, n_bins=9):
        _edges = _shared_edges(_model_df, _data_df, n_bins=n_bins)
        _model_out = _model_df.copy()
        _data_out = _data_df.copy()
        _model_out["at_x_bin"] = _assign_at_bins(_model_out, _edges, prefix="B")
        _data_out["at_x_bin"] = _assign_at_bins(_data_out, _edges, prefix="B")
        _pooled = pd.concat([_model_out[["at", "at_x_bin"]], _data_out[["at", "at_x_bin"]]])
        _centers = _pooled.groupby("at_x_bin", observed=True)["at"].mean()
        _model_out["at_x_center"] = _model_out["at_x_bin"].map(_centers).astype(float)
        _data_out["at_x_center"] = _data_out["at_x_bin"].map(_centers).astype(float)
        return _model_out, _data_out

    _empirical_2afc = _empirical_binary_psych_df(
        df_2AFC,
        subject_col="subject",
        session_col="Session",
        trial_col="Trial",
        signed_x_col="ILD",
        choice_col="Choice",
        choice_transform=lambda _choice: (2.0 * pd.to_numeric(_choice, errors="coerce")) - 1.0,
    )
    _empirical_2afc_delay = _empirical_binary_psych_df(
        df_2AFC_delay,
        subject_col="subject",
        session_col="session",
        trial_col="trial",
        signed_x_expr=lambda _df: pd.to_numeric(_df["stim"], errors="coerce")
        * pd.to_numeric(_df["delays"], errors="coerce"),
        choice_col="choices",
        choice_transform=lambda _choice: pd.to_numeric(_choice, errors="coerce"),
    )

    fig_psych_data, _axes_psych_data = plt.subplots(
        2,
        4,
        figsize=(10.5, 5.2),
        layout="constrained",
        sharey=True,
    )
    _task_rows = [
        ("2AFC", cartoon_2afc, _empirical_2afc, "Signed ILD (dB)", "Signed ILD bin"),
        (
            "2ADC",
            cartoon_2afc_delay,
            _empirical_2afc_delay,
            "Signed delay (s)",
            "Signed delay bin",
        ),
    ]
    _delay_order = [-0.1, -1.0, -3.0, -10.0, 10.0, 3.0, 1.0, 0.1]
    for _row_idx, (_task_name, _df_task, _df_data, _stim_label, _bin_label) in enumerate(_task_rows):
        _df_at, _data_at = _with_shared_at_bins(_df_task, _df_data, n_bins=4)
        _df_signed, _data_signed = _with_shared_signed_x_bins(_df_task, _df_data)
        _df_signed, _data_signed = _with_shared_action_trace_bins(_df_signed, _data_signed, n_bins=9)

        for _model_idx, _model in enumerate(["Mixture", "Additive"]):
            _model_df_at = _df_at[_df_at["model"] == _model].copy()
            _model_df_signed = _df_signed[_df_signed["model"] == _model].copy()

            _ax = _axes_psych_data[_row_idx, _model_idx * 2]
            _psych = (
                _model_df_at.groupby(["at_bin", "signed_x"], observed=True)["choice_right"]
                .mean()
                .reset_index()
            )
            _data_psych = (
                _data_at.groupby(["at_bin", "signed_x"], observed=True)["choice_right"]
                .mean()
                .reset_index()
            )
            _at_bins = list(_psych["at_bin"].dropna().unique())
            _colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(_at_bins)))
            for _at_bin, _color in zip(_at_bins, _colors, strict=False):
                _sub = _psych[_psych["at_bin"] == _at_bin].sort_values("signed_x")
                _x = _sub["signed_x"]
                if _task_name == "2ADC":
                    _sub, _x = _ordered_delay_curve(_sub, _delay_order)
                _ax.plot(
                    _x,
                    _sub["choice_right"],
                    linewidth=1.2,
                    color=_color,
                    label=str(_at_bin),
                )
                _data_sub = _data_psych[_data_psych["at_bin"] == _at_bin].sort_values("signed_x")
                _data_x = _data_sub["signed_x"]
                if _task_name == "2ADC":
                    _data_sub, _data_x = _ordered_delay_curve(_data_sub, _delay_order)
                _ax.scatter(
                    _data_x,
                    _data_sub["choice_right"],
                    s=22,
                    color=_color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            _ax.axhline(0.5, color="0.65", linestyle="--", linewidth=0.8)
            if _task_name == "2ADC":
                _ax.axvline(3.5, color="0.65", linestyle="--", linewidth=0.8)
            else:
                _ax.axvline(0.0, color="0.65", linestyle="--", linewidth=0.8)
            _ax.set_title(f"{_task_name}: {_model}\nby action trace")
            _ax.set_xlabel(_stim_label)
            _ax.set_ylabel("p(right)")
            _ax.set_ylim(0.0, 1.0)
            if _task_name == "2ADC":
                _ax.set_xticks(np.arange(8))
                _ax.set_xticklabels(["0", "-1", "-3", "-10", "10", "3", "1", "0"])
                _ax.set_xlim(-0.4, 7.4)
            if _row_idx == 0 and _model_idx == 1:
                _ax.legend(title="AT bin", frameon=False, fontsize=7, title_fontsize=8)

            _ax = _axes_psych_data[_row_idx, _model_idx * 2 + 1]
            _at_curve = (
                _model_df_signed.groupby(
                    ["signed_x_bin", "at_x_bin", "at_x_center"],
                    observed=True,
                )["choice_right"]
                .mean()
                .reset_index()
            )
            _data_at_curve = (
                _data_signed.groupby(
                    ["signed_x_bin", "at_x_bin", "at_x_center"],
                    observed=True,
                )["choice_right"]
                .mean()
                .reset_index()
            )
            _x_bins = list(_at_curve["signed_x_bin"].dropna().unique())
            _colors = plt.cm.magma(np.linspace(0.18, 0.85, len(_x_bins)))
            for _x_bin, _color in zip(_x_bins, _colors, strict=False):
                _sub = _at_curve[_at_curve["signed_x_bin"] == _x_bin].sort_values("at_x_center")
                _ax.plot(
                    _sub["at_x_center"],
                    _sub["choice_right"],
                    linewidth=1.2,
                    color=_color,
                    label=str(_x_bin),
                )
                _data_sub = _data_at_curve[_data_at_curve["signed_x_bin"] == _x_bin].sort_values("at_x_center")
                _ax.scatter(
                    _data_sub["at_x_center"],
                    _data_sub["choice_right"],
                    s=22,
                    color=_color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            _ax.axhline(0.5, color="0.65", linestyle="--", linewidth=0.8)
            _ax.axvline(0.0, color="0.65", linestyle="--", linewidth=0.8)
            _ax.set_title(f"{_task_name}: {_model}\nby stimulus strength")
            _ax.set_xlabel("Action trace")
            _ax.set_xlim(-15.5, 15.5)
            _ax.set_xticks([-10, 0, 10])
            _ax.set_ylim(0.0, 1.0)
            if _row_idx == 0 and _model_idx == 1:
                _ax.legend(title=_bin_label, frameon=False, fontsize=7, title_fontsize=8)

    for _ax in _axes_psych_data.ravel():
        _ax.label_outer()
    mo.vstack(
        [
            fig_psych_data,
            save_plot(
                fig_psych_data,
                "strategy cartoon psychometrics data overlay",
                stem="strategy_cartoon_psychometrics_data_overlay",
            ),
        ],
        align="center",
    )
    fig_psych_data
    return


if __name__ == "__main__":
    app.run()
