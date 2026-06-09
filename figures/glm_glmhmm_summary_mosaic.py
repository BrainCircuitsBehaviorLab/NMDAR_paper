# /// script
# [tool.marimo.opengraph]
# title = "GLM and GLM-HMM summary mosaics"
# description = "Stimulus weights, categorical performance, and previous-choice weights for selected GLM and GLM-HMM fits."
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from scipy.stats import ttest_1samp

    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
    )
    from glmhmmt.postprocess import build_emission_weights_df
    import glmhmmt.plots as model_plots
    from glmhmmt.runtime import configure_paths, get_runtime_paths, load_app_config
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.plots.common import fig_size, plot_prepared_weight_family
    from src.process import MCDR as process_mcdr
    from src.process import two_afc as process_two_afc
    from src.process import two_adc as process_two_adc
    from src.process.common import add_choice_lag_summary_regressor

    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    def prepare_predictions_df(task_name, df):
        if task_name == "MCDR":
            return process_mcdr.prepare_predictions_df(df, cfg=load_app_config())
        if task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            return process_two_adc.prepare_predictions_df(df)
        return process_two_afc.prepare_predictions_df(df)

    sns.set_style("ticks")
    sns.set_context("paper")
    plt.style.use(ROOT / "styles" / "paper.mplstyle")
    return (
        ROOT,
        add_choice_lag_summary_regressor,
        build_emission_weights_df,
        build_trial_and_weights_df,
        build_views,
        fig_size,
        get_adapter,
        load_fit_arrays,
        mo,
        model_plots,
        np,
        paths,
        pd,
        pl,
        plot_prepared_weight_family,
        plt,
        prepare_predictions_df,
        re,
        sns,
        ttest_1samp,
    )


@app.cell
def _(ROOT, re):
    task_specs = {
        "2AFC": "2AFC",
        "2AFC_delay": "2ADC",
        "MCDR": "3CDR",
    }
    task_names = tuple(task_specs)
    hash_re = re.compile(r"^[A-Za-z0-9]{8}$")

    def saved_model_names(task_name: str, model_kind: str) -> list[str]:
        fit_dir = ROOT / "results" / "fits" / task_name / model_kind
        if not fit_dir.exists():
            return []
        suffix = f"{model_kind}_arrays.npz"
        return sorted(
            item.name
            for item in fit_dir.iterdir()
            if item.is_dir()
            and not hash_re.fullmatch(item.name)
            and any(item.glob(f"*_{suffix}"))
        )

    return saved_model_names, task_names, task_specs


@app.cell
def _(mo, task_names):
    ui_task = mo.ui.dropdown(
        options=list(task_names),
        value="2AFC",
        label="Task",
    )
    return (ui_task,)


@app.cell
def _(mo, task_specs, ui_task):
    mo.vstack([ui_task, mo.md(f"Selected task label: **{task_specs[ui_task.value]}**")])
    return


@app.cell
def _(mo, saved_model_names, ui_task):
    glm_model_options = saved_model_names(ui_task.value, "glm")
    glmhmm_model_options = saved_model_names(ui_task.value, "glmhmm")
    mo.stop(
        not glm_model_options and not glmhmm_model_options,
        mo.md(f"No saved GLM or GLM-HMM models found for `{ui_task.value}`."),
    )
    return glm_model_options, glmhmm_model_options


@app.cell
def _(glm_model_options, glmhmm_model_options, mo):
    def _default_model(options: list[str], preferred: str) -> str:
        if preferred in options:
            return preferred
        if "one hot" in options:
            return "one hot"
        if "param" in options:
            return "param"
        return options[0]

    ui_glm_model = mo.ui.dropdown(
        options=glm_model_options or ["None"],
        value=_default_model(glm_model_options, "one hot") if glm_model_options else "None",
        label="GLM model",
    )
    ui_glmhmm_model = mo.ui.dropdown(
        options=glmhmm_model_options or ["None"],
        value=_default_model(glmhmm_model_options, "param") if glmhmm_model_options else "None",
        label="GLM-HMM model",
    )
    mo.hstack([ui_glm_model, ui_glmhmm_model], justify="start")
    return ui_glm_model, ui_glmhmm_model


@app.cell
def _(
    add_choice_lag_summary_regressor,
    build_emission_weights_df,
    build_trial_and_weights_df,
    build_views,
    get_adapter,
    load_fit_arrays,
    mo,
    paths,
    prepare_predictions_df,
    re,
    ui_task,
):
    def _selected_subjects(subjects, arrays_store) -> list[str]:
        selected = [str(subject) for subject in subjects if str(subject) in arrays_store]
        return selected or [str(subject) for subject in arrays_store]

    def _choice_lag_cols(adapter, trial_df, views) -> list[str]:
        cols = []
        for view in views.values():
            for feature in list(getattr(view, "feat_names", []) or []):
                feature = str(feature)
                if feature.startswith("choice_lag_") and feature not in cols:
                    cols.append(feature)
        if not cols and hasattr(adapter, "choice_lag_cols"):
            cols = adapter.choice_lag_cols(trial_df)
        return cols

    def _infer_fit_k(out_dir, model_kind: str) -> int | None:
        if model_kind == "glm" or not out_dir.exists():
            return None
        suffix = f"{model_kind}_arrays.npz"
        k_values = sorted(
            {
                int(match.group(1))
                for path in out_dir.glob(f"*_K*_{suffix}")
                for match in [re.search(r"_K(\d+)_", path.name)]
                if match is not None
            }
        )
        return k_values[0] if k_values else None

    def load_plot_payload(model_kind: str, model_name: str) -> dict | None:
        if model_name == "None":
            return None

        task_name = ui_task.value
        adapter = get_adapter(task_name)
        df_all = adapter.subject_filter(adapter.read_dataset())
        subjects = list(df_all["subject"].unique())
        out_dir = paths.RESULTS / "fits" / task_name / model_kind / model_name
        inferred_k = _infer_fit_k(out_dir, model_kind)
        arrays_store, _ = load_fit_arrays(
            out_dir=out_dir,
            arrays_suffix=f"{model_kind}_arrays.npz",
            adapter=adapter,
            df_all=df_all,
            subjects=subjects,
            emission_cols=None,
            k=inferred_k,
        )
        selected = _selected_subjects(subjects, arrays_store)
        mo.stop(
            not selected,
            mo.md(f"No fitted subjects found for `{task_name}/{model_kind}/{model_name}`."),
        )

        first_arrays = arrays_store[selected[0]]
        K = int(first_arrays["emission_weights"].shape[0])
        views = build_views(arrays_store, adapter, K, selected)
        views_sel = {subject: views[subject] for subject in selected}
        trial_df, weights_df = build_trial_and_weights_df(
            df_all,
            views=views,
            adapter=adapter,
            min_session_length=1,
        )
        mo.stop(trial_df.height == 0, mo.md(f"No trial data found for `{task_name}/{model_kind}/{model_name}`."))

        plot_df = prepare_predictions_df(task_name, trial_df)
        choice_lag_cols = _choice_lag_cols(adapter, trial_df, views)
        if choice_lag_cols:
            plot_df = add_choice_lag_summary_regressor(
                plot_df,
                choice_lag_cols=choice_lag_cols,
            )

        return {
            "adapter": adapter,
            "emission_weights_df": build_emission_weights_df(views_sel),
            "K": K,
            "model_kind": model_kind,
            "model_name": model_name,
            "plot_df": plot_df,
            "plots": adapter.get_plots(),
            "selected": selected,
            "task_name": task_name,
            "views": views,
            "weights_df": weights_df,
        }

    return (load_plot_payload,)


@app.cell
def _(load_plot_payload, mo, ui_glm_model):
    glm_payload = load_plot_payload("glm", ui_glm_model.value)
    mo.stop(glm_payload is None, mo.md("Select a GLM model."))
    mo.md(f"Loaded GLM `{glm_payload['task_name']}/glm/{glm_payload['model_name']}`.")
    return (glm_payload,)


@app.cell
def _(load_plot_payload, mo, ui_glmhmm_model):
    glmhmm_payload = load_plot_payload("glmhmm", ui_glmhmm_model.value)
    mo.stop(glmhmm_payload is None, mo.md("Select a GLM-HMM model."))
    mo.md(f"Loaded GLM-HMM `{glmhmm_payload['task_name']}/glmhmm/{glmhmm_payload['model_name']}`.")
    return (glmhmm_payload,)


@app.cell
def _(
    fig_size,
    np,
    pd,
    pl,
    plot_prepared_weight_family,
    plt,
    sns,
    ttest_1samp,
):
    def _glm_significance_stars(pvalue: float) -> str:
        if not np.isfinite(pvalue):
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        if pvalue < 0.05:
            return "*"
        return ""

    def _glm_annotate_choice_lag_ttests(
        ax: plt.Axes,
        panel_df: pd.DataFrame,
        lag_order: list[int],
        y: float = 1.0,
    ) -> None:
        for lag in lag_order:
            values = panel_df.loc[panel_df["lag"] == lag, "weight"].dropna().to_numpy(dtype=float)
            if values.size < 2:
                continue
            stars = _glm_significance_stars(float(ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue))
            ax.text(
                lag,
                y,
                stars,
                ha="center",
                va="bottom",
                fontsize=plt.rcParams["font.size"],
                color="black",
                clip_on=False,
            )

    def _glm_drop_categorical_facets(df):
        drop_cols = [col for col in ("condition", "experiment") if col in getattr(df, "columns", [])]
        if not drop_cols:
            return df
        if isinstance(df, pl.DataFrame):
            return df.drop(drop_cols)
        return pd.DataFrame(df).drop(columns=drop_cols, errors="ignore")

    def plot_glm_stim_weights(payload, ax: plt.Axes) -> None:
        adapter = payload["adapter"]
        weights_df = payload["weights_df"]
        task_name = payload["task_name"]
        prepared = adapter.prepare_weight_family_plot(weights_df, "stim_hot", variant="folded")
        plotted = plot_prepared_weight_family(
            prepared,
            figsize=fig_size(2, 1),
            ax=ax,
            connect_subjects=False,
        )
        if plotted is None:
            ax.text(0.5, 0.5, "No stimulus weights found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        if task_name == "2AFC":
            ax.set_xlabel("Stimulus level")
        elif task_name in {"2AFC_delay", "2ADC", "2ADC_DRUG", "2AFC_delay_DRUG"}:
            ax.set_xlabel("Delay level")
        # ax.set_title("Stimulus weights")
        upper = float(ax.get_ylim()[1])
        if np.isfinite(upper):
            ax.set_ylim(-0.25, upper)

    def plot_glm_categorical_panel(payload, ax: plt.Axes) -> None:
        plot_df = payload["plot_df"]
        plots = payload["plots"]
        task_name = payload["task_name"]
        if task_name == "MCDR":
            ax.set_axis_off()
            inset_axes = [
                ax.inset_axes([0.00, 0.10, 0.30, 0.82]),
                ax.inset_axes([0.35, 0.10, 0.30, 0.82]),
                ax.inset_axes([0.70, 0.10, 0.30, 0.82]),
            ]
            plots.plot_categorical_performance_all(
                plot_df,
                "GLM",
                background_style="model",
                axes=inset_axes,
                views=payload["views"],
            )
            for inset_ax in inset_axes:
                inset_ax.set_title("")
            return
        kwargs = {"views": payload["views"]} if task_name == "2AFC" else {}
        plots.plot_categorical_performance_all(
            _glm_drop_categorical_facets(plot_df),
            "GLM",
            background_style="model",
            axes=[ax],
            **kwargs,
        )
        # ax.set_title("Categorical performance")

    def _glm_choice_lag_outcome_df(payload) -> pd.DataFrame:
        weights_df = payload["weights_df"]
        if isinstance(weights_df, pl.DataFrame) and "subject" in weights_df.columns:
            weights_df = weights_df.filter(pl.col("subject").is_in(payload["selected"]))
        choice_df = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        choice_df = choice_df.copy()
        choice_df["feature"] = choice_df["feature"].astype(str)
        choice_df["weight"] = pd.to_numeric(choice_df["weight"], errors="coerce")
        parsed = choice_df["feature"].str.extract(r"^choice_lag_(corr|inc)_(\d+)$")
        choice_df["outcome_family"] = parsed[0].map({"corr": "Correct", "inc": "Incorrect"})
        choice_df["lag"] = pd.to_numeric(parsed[1], errors="coerce")
        choice_df = choice_df[
            choice_df["outcome_family"].isin(["Correct", "Incorrect"])
            & choice_df["lag"].between(1, 100)
            & np.isfinite(choice_df["weight"])
        ].copy()
        if choice_df.empty:
            return pd.DataFrame(columns=["feature", "weight", "outcome_family", "lag"])
        choice_df["lag"] = choice_df["lag"].astype(int)
        return choice_df

    def _glm_split_axis_columns(ax: plt.Axes, ncols: int = 2) -> list[plt.Axes]:
        fig = ax.figure
        subgrid = ax.get_subplotspec().subgridspec(1, ncols)
        ax.remove()
        axes = []
        for col in range(ncols):
            sharey = axes[0] if axes else None
            axes.append(fig.add_subplot(subgrid[0, col], sharey=sharey))
        ax._replacement_axes = axes
        return axes

    def _plot_glm_choice_lag_outcomes(payload, ax: plt.Axes) -> bool:
        choice_df = _glm_choice_lag_outcome_df(payload)
        if choice_df.empty:
            return False
        lag_order = sorted(choice_df["lag"].unique().tolist())
        outcome_axes = _glm_split_axis_columns(ax)
        for panel_ax, outcome_family in zip(outcome_axes, ("Correct", "Incorrect"), strict=False):
            panel_df = choice_df[choice_df["outcome_family"] == outcome_family].copy()
            sns.lineplot(
                data=panel_df,
                x="lag",
                y="weight",
                estimator="mean",
                errorbar="se",
                marker="o",
                color="tab:blue",
                ax=panel_ax,
            )
            panel_ax.axhline(0, color="black", linestyle="--", alpha=0.6)
            panel_ax.set_title(outcome_family)
            panel_ax.set_xlabel("Choice lag")
            panel_ax.set_ylabel("Weight" if outcome_family == "Correct" else "")
            panel_ax.set_xticks(range(5, max(lag_order) + 1, 5))
            _glm_annotate_choice_lag_ttests(panel_ax, panel_df, lag_order, y=1.5)
        y_lims = [panel_ax.get_ylim() for panel_ax in outcome_axes]
        y_min = min(bottom for bottom, _ in y_lims)
        y_max = max(top for _, top in y_lims)
        for panel_ax in outcome_axes:
            # panel_ax.set_ylim(y_min, y_max)
            panel_ax.set_ylim(-0.5,2)
        return True

    def plot_glm_choice_lag_lineplot(payload, ax: plt.Axes) -> None:
        if _plot_glm_choice_lag_outcomes(payload, ax):
            return

        weights_df = payload["weights_df"]
        choice_df = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        choice_df = choice_df.copy()
        choice_df["feature"] = choice_df["feature"].astype(str)
        choice_df["weight"] = pd.to_numeric(choice_df["weight"], errors="coerce")
        choice_df["lag"] = pd.to_numeric(
            choice_df["feature"].str.extract(r"^choice_lag_(\d+)$", expand=False),
            errors="coerce",
        )
        choice_df = choice_df[choice_df["lag"].between(1, 100) & np.isfinite(choice_df["weight"])].copy()
        if choice_df.empty:
            ax.text(0.5, 0.5, "No choice-lag weights found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        choice_df["lag"] = choice_df["lag"].astype(int)
        lag_order = sorted(choice_df["lag"].unique().tolist())
        sns.lineplot(
            data=choice_df,
            x="lag",
            y="weight",
            estimator="mean",
            errorbar="se",
            marker="o",
            markersize=3,
            linewidth=1.25,
            color="#1f77b4",
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Choice lag")
        ax.set_ylabel("Weight")
        ax.set_xticks(list(range(5, max(lag_order) + 1, 5)) or lag_order)
        ax.set_ylim(-0.5, 2)
        _glm_annotate_choice_lag_ttests(ax, choice_df, lag_order)

    return (
        plot_glm_categorical_panel,
        plot_glm_choice_lag_lineplot,
        plot_glm_stim_weights,
    )


@app.cell
def _(fig_size, model_plots, np, pd, pl, plt, sns, ttest_1samp):
    def _glmhmm_significance_stars(pvalue: float) -> str:
        if not np.isfinite(pvalue):
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        if pvalue < 0.05:
            return "*"
        return ""

    def _glmhmm_annotate_choice_lag_ttests(
        ax: plt.Axes,
        panel_df: pd.DataFrame,
        lag_order: list[int],
        y: float = 1.75,
    ) -> None:
        for lag in lag_order:
            values = panel_df.loc[panel_df["lag"] == lag, "weight"].dropna().to_numpy(dtype=float)
            if values.size < 2:
                continue
            stars = _glmhmm_significance_stars(float(ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue))
            if not stars:
                continue
            ax.text(
                lag,
                y,
                stars,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                clip_on=False,
            )

    def _put_legend_inside_panel(ax, *, loc="upper right", anchor=(0.98, 0.98)):
        legend = ax.get_legend()
        if legend is None:
            return
        handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
        labels = [text.get_text() for text in legend.get_texts()]
        title = legend.get_title().get_text()
        legend.remove()
        ax.legend(
            handles,
            labels,
            title=title or None,
            frameon=legend.get_frame_on(),
            loc=loc,
            bbox_to_anchor=anchor,
            borderaxespad=0.2,
        )

    def plot_glmhmm_stim_summary(payload, ax: plt.Axes) -> None:
        weights_df = payload.get("emission_weights_df", payload["weights_df"])
        preferred_feature_order = ["bias", "bias_param", "biasparam", "stim_param", "stim_x_delay_param"]
        feature_labels = {
            "stim_param": r"$\mathrm{Stim}_{\mathrm{param}}$",
            "stim_x_delay_param": r"$\mathrm{Stim}:\mathrm{Delay}_{\mathrm{param}}$",
            "bias": r"$\mathrm{Bias}$",
            "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
            "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
        }
        feature_labeler = lambda feature: feature_labels.get(str(feature), str(feature))
        summary_df = weights_df.filter(
            (pl.col("feature") == "stim_param") | (pl.col("feature") == "bias")
        )
        if summary_df.is_empty():
            summary_df = weights_df.filter(pl.col("feature").is_in(preferred_feature_order))
        if summary_df.is_empty():
            ax.text(0.5, 0.5, "No stim/bias weights found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        summary_pdf = summary_df.to_pandas() if hasattr(summary_df, "to_pandas") else pd.DataFrame(summary_df)
        available_features = pd.unique(summary_pdf["feature"].astype(str)).tolist()
        feature_order = [feature for feature in preferred_feature_order if feature in available_features]
        feature_order.extend(feature for feature in available_features if feature not in feature_order)
        model_plots.emission_weights_summary_boxplot(
            summary_df,
            K=payload["K"],
            connect_subjects=True,
            show_ttests=True,
            feature_order=feature_order,
            feature_labeler=feature_labeler,
            ax=ax,
            tick_rotation=0,
            figsize=fig_size(2, 1),
        )
        # ax.set_title("Stimulus/bias weights")
        _put_legend_inside_panel(ax, anchor=(0.98, 0.3))

    def plot_glmhmm_categorical_state_overlay(payload, ax: plt.Axes) -> None:
        plot_fn = getattr(payload["plots"], "plot_categorical_performance_state_overlay", None)
        if plot_fn is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No state overlay plot available", ha="center", va="center")
            return
        state_plot_kwargs = dict(
            background_style="model",
            show_weighted_points=True,
            show_data_smooth=True,
            show_model_smooth=True,
            model_line_mode="smooth",
            state_assignment_mode="map",
            figure_dpi=300,
        )
        plot_fn(
            df=payload["plot_df"],
            views=payload["views"],
            model_name=f"glmhmm K={payload['K']} -- all states",
            ax=ax,
            **state_plot_kwargs,
        )
        # ax.set_title("Categorical performance by state")

    def _glmhmm_choice_lag_df(payload) -> pd.DataFrame:
        weights_df = payload.get("emission_weights_df", payload["weights_df"])
        weights_pdf = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        required = {"subject", "state_label", "feature", "weight"}
        if weights_pdf.empty or not required.issubset(weights_pdf.columns):
            return pd.DataFrame(columns=["subject", "state_label", "feature", "weight", "lag"])
        choice_df = weights_pdf.copy()
        choice_df["subject"] = choice_df["subject"].astype(str)
        choice_df["state_label"] = choice_df["state_label"].astype(str)
        choice_df["feature"] = choice_df["feature"].astype(str)
        choice_df["weight"] = pd.to_numeric(choice_df["weight"], errors="coerce")
        choice_df["lag"] = pd.to_numeric(
            choice_df["feature"].str.extract(r"^choice_lag_(\d+)$", expand=False),
            errors="coerce",
        )
        choice_df = choice_df[
            choice_df["state_label"].isin(["Engaged", "Disengaged"])
            & choice_df["lag"].between(1, 100)
            & np.isfinite(choice_df["weight"])
        ].copy()
        choice_df["lag"] = choice_df["lag"].astype(int)
        return choice_df[choice_df["lag"].isin(range(1, 101))].copy()

    def _glmhmm_choice_lag_outcome_df(payload) -> pd.DataFrame:
        weights_df = payload.get("emission_weights_df", payload["weights_df"])
        weights_pdf = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        required = {"subject", "state_label", "feature", "weight"}
        if weights_pdf.empty or not required.issubset(weights_pdf.columns):
            return pd.DataFrame(columns=["subject", "state_label", "feature", "weight", "outcome_family", "lag"])
        choice_df = weights_pdf.copy()
        choice_df["subject"] = choice_df["subject"].astype(str)
        choice_df["state_label"] = choice_df["state_label"].astype(str)
        choice_df["feature"] = choice_df["feature"].astype(str)
        choice_df["weight"] = pd.to_numeric(choice_df["weight"], errors="coerce")
        parsed = choice_df["feature"].str.extract(r"^choice_lag_(corr|inc)_(\d+)$")
        choice_df["outcome_family"] = parsed[0].map({"corr": "Correct", "inc": "Incorrect"})
        choice_df["lag"] = pd.to_numeric(parsed[1], errors="coerce")
        choice_df = choice_df[
            choice_df["state_label"].isin(["Engaged", "Disengaged"])
            & choice_df["outcome_family"].isin(["Correct", "Incorrect"])
            & choice_df["lag"].between(1, 100)
            & np.isfinite(choice_df["weight"])
        ].copy()
        if choice_df.empty:
            return pd.DataFrame(columns=["subject", "state_label", "feature", "weight", "outcome_family", "lag"])
        choice_df["lag"] = choice_df["lag"].astype(int)
        return choice_df[choice_df["lag"].isin(range(1, 101))].copy()

    def _glmhmm_split_axis_columns(ax: plt.Axes, ncols: int = 2) -> list[plt.Axes]:
        fig = ax.figure
        subgrid = ax.get_subplotspec().subgridspec(1, ncols)
        ax.remove()
        axes = []
        for col in range(ncols):
            sharey = axes[0] if axes else None
            axes.append(fig.add_subplot(subgrid[0, col], sharey=sharey))
        ax._replacement_axes = axes
        return axes

    def _plot_glmhmm_choice_lag_outcomes(payload, ax: plt.Axes, state_label: str) -> bool:
        choice_df = _glmhmm_choice_lag_outcome_df(payload)
        state_df = choice_df[choice_df["state_label"] == state_label].copy()
        if state_df.empty:
            return False
        lag_order = list(range(1, 101))
        outcome_axes = _glmhmm_split_axis_columns(ax)
        color = {"Engaged": "tab:green", "Disengaged": "tab:gray"}.get(state_label, "#1f77b4")
        for panel_ax, outcome_family in zip(outcome_axes, ("Correct", "Incorrect"), strict=False):
            panel_df = state_df[state_df["outcome_family"] == outcome_family].copy()
            sns.lineplot(
                data=panel_df,
                x="lag",
                y="weight",
                estimator="mean",
                errorbar="se",
                marker="o",
                markersize=3,
                linewidth=1.25,
                color=color,
                ax=panel_ax,
            )
            panel_ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            panel_ax.set_title(f"{outcome_family} - {state_label}")
            panel_ax.set_xlabel("Choice lag")
            panel_ax.set_ylabel("Weight" if outcome_family == "Correct" else "")
            panel_ax.set_xticks(range(5, 101, 5))
            panel_ax.set_ylim(-0.5, 4)
            _glmhmm_annotate_choice_lag_ttests(panel_ax, panel_df, lag_order, y=3.75)
            if state_label == "Engaged":
                panel_ax.set_xlabel("")
                panel_ax.tick_params(axis="x", labelbottom=False)
        return True

    def plot_glmhmm_choice_lag_state(payload, ax: plt.Axes, state_label: str) -> None:
        if _plot_glmhmm_choice_lag_outcomes(payload, ax, state_label):
            return

        choice_df = _glmhmm_choice_lag_df(payload)
        state_df = choice_df[choice_df["state_label"] == state_label].copy()
        if state_df.empty:
            ax.text(0.5, 0.5, f"No {state_label} choice-lag weights", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        lag_order = list(range(1, 101))
        color = {"Engaged": "tab:green", "Disengaged": "tab:gray"}.get(state_label, "#1f77b4")
        sns.lineplot(
            data=state_df,
            x="lag",
            y="weight",
            estimator="mean",
            errorbar="se",
            marker="o",
            markersize=3,
            linewidth=1.25,
            color=color,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        # ax.set_title(state_label)
        ax.set_xlabel("Choice lag")
        ax.set_ylabel("Weight")
        ax.set_xticks(range(5, 101, 5))
        ax.set_ylim(-0.5, 2)
        _glmhmm_annotate_choice_lag_ttests(ax, state_df, lag_order, y=1.75)

    return (
        plot_glmhmm_categorical_state_overlay,
        plot_glmhmm_choice_lag_state,
        plot_glmhmm_stim_summary,
    )


@app.cell
def _(
    fig_size,
    glm_payload,
    mo,
    plot_glm_categorical_panel,
    plot_glm_choice_lag_lineplot,
    plot_glm_stim_weights,
    plt,
    sns,
):
    _stem = f"glm_summary_mosaic_{glm_payload['task_name']}_{glm_payload['model_name']}".replace("/", "_").replace(" ", "_")
    _fig_glm, _axd_glm = plt.subplot_mosaic(
        [["stim", "categorical"], ["choice_lag", "choice_lag"]],
        figsize=fig_size(1, 1),
        layout="constrained",
    )
    plot_glm_stim_weights(glm_payload, _axd_glm["stim"])
    plot_glm_categorical_panel(glm_payload, _axd_glm["categorical"])
    plot_glm_choice_lag_lineplot(glm_payload, _axd_glm["choice_lag"])
    for _key in ("stim", "categorical"):
        _axd_glm[_key].set_box_aspect(1)
    _fig_glm.canvas.draw()
    _label_grid_glm = {
        "stim": (0, 0, "a"),
        "categorical": (0, 1, "b"),
        "choice_lag": (1, 0, "c"),
    }
    _label_axes_glm = {
        _key: (getattr(_axd_glm[_key], "_replacement_axes", None) or [_axd_glm[_key]])[0]
        for _key in _label_grid_glm
    }
    _row_tops_glm = {
        _row: max(
            _label_axes_glm[_key].get_position().y1
            for _key, (_key_row, _, _) in _label_grid_glm.items()
            if _key_row == _row
        )
        for _row in {0, 1}
    }
    _col_lefts_glm = {
        _col: min(
            _label_axes_glm[_key].get_position().x0
            for _key, (_, _key_col, _) in _label_grid_glm.items()
            if _key_col == _col
        )
        for _col in {0, 1}
    }
    for _key, (_row, _col, _label) in _label_grid_glm.items():
        _fig_glm.text(
            _col_lefts_glm[_col] - 0.01,
            _row_tops_glm[_row] + 0.01,
            _label,
            transform=_fig_glm.transFigure,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
    sns.despine(fig=_fig_glm)
    _fig_glm.savefig(f"{_stem}.pdf")
    _fig_glm.savefig(f"{_stem}.png")
    mo.vstack([mo.md("## GLM summary mosaic"), _fig_glm], align="center")
    return


@app.cell
def _(
    fig_size,
    glmhmm_payload,
    mo,
    plot_glmhmm_categorical_state_overlay,
    plot_glmhmm_choice_lag_state,
    plot_glmhmm_stim_summary,
    plt,
    sns,
):
    _stem = f"glmhmm_summary_mosaic_{glmhmm_payload['task_name']}_{glmhmm_payload['model_name']}".replace("/", "_").replace(" ", "_")
    _fig_glmhmm, _axd_glmhmm = plt.subplot_mosaic(
        [
            ["stim", "categorical"],
            ["choice_engaged", "choice_engaged"],
            ["choice_disengaged", "choice_disengaged"],
        ],
        figsize=fig_size(1, 0.75),
        layout="constrained",
    )
    plot_glmhmm_stim_summary(glmhmm_payload, _axd_glmhmm["stim"])
    plot_glmhmm_categorical_state_overlay(glmhmm_payload, _axd_glmhmm["categorical"])
    plot_glmhmm_choice_lag_state(glmhmm_payload, _axd_glmhmm["choice_engaged"], "Engaged")
    plot_glmhmm_choice_lag_state(glmhmm_payload, _axd_glmhmm["choice_disengaged"], "Disengaged")
    _axd_glmhmm["choice_engaged"].set_xlabel("")
    _axd_glmhmm["choice_engaged"].tick_params(axis="x", labelbottom=False)
    for _key in ("stim", "categorical"):
        _axd_glmhmm[_key].set_box_aspect(1)
    _fig_glmhmm.canvas.draw()
    _label_grid_glmhmm = {
        "stim": (0, 0, "a"),
        "categorical": (0, 1, "b"),
        "choice_engaged": (1, 0, "c"),
        "choice_disengaged": (2, 0, "d"),
    }
    _label_axes_glmhmm = {
        _key: (getattr(_axd_glmhmm[_key], "_replacement_axes", None) or [_axd_glmhmm[_key]])[0]
        for _key in _label_grid_glmhmm
    }
    _row_tops_glmhmm = {
        _row: max(
            _label_axes_glmhmm[_key].get_position().y1
            for _key, (_key_row, _, _) in _label_grid_glmhmm.items()
            if _key_row == _row
        )
        for _row in {0, 1, 2}
    }
    _col_lefts_glmhmm = {
        _col: min(
            _label_axes_glmhmm[_key].get_position().x0
            for _key, (_, _key_col, _) in _label_grid_glmhmm.items()
            if _key_col == _col
        )
        for _col in {0, 1}
    }
    for _key, (_row, _col, _label) in _label_grid_glmhmm.items():
        _fig_glmhmm.text(
            _col_lefts_glmhmm[_col] - 0.01,
            _row_tops_glmhmm[_row] + 0.01,
            _label,
            transform=_fig_glmhmm.transFigure,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
    sns.despine(fig=_fig_glmhmm)
    _fig_glmhmm.savefig(f"{_stem}.pdf")
    _fig_glmhmm.savefig(f"{_stem}.png")
    mo.vstack([mo.md("## GLM-HMM summary mosaic"), _fig_glmhmm], align="center")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
