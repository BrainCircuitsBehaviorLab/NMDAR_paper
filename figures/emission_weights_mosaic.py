# /// script
# [tool.marimo.opengraph]
# title = "Emission weights mosaic"
# description = "GLM emission weights across 2ADC, 2AFC, and 3CDR tasks."
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
    import seaborn as sns
    from scipy.stats import ttest_1samp

    from glmhmmt.plots.common import custom_boxplot
    from glmhmmt.plots.emissions import (
        _fold_three_choice_raw_weights,
        _prepare_weights_df,
    )
    from glmhmmt.notebook_support.analysis_common import (
        load_fit_arrays,
    )
    from glmhmmt.postprocess import build_emission_weights_df
    from glmhmmt.runtime import configure_paths, get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from src.plots.common import fig_size, plot_prepared_weight_family

    ROOT = Path(__file__).resolve().parents[1]
    configure_paths(config_path=ROOT / "config.toml")
    paths = get_runtime_paths()

    sns.set_theme(style="ticks")
    sns.set_context("paper")
    plt.style.use(ROOT / "styles" / "paper.mplstyle")
    fold_three_choice_raw_weights = _fold_three_choice_raw_weights
    prepare_weights_df = _prepare_weights_df
    return (
        ROOT,
        build_emission_weights_df,
        build_views,
        custom_boxplot,
        fig_size,
        fold_three_choice_raw_weights,
        get_adapter,
        load_fit_arrays,
        mo,
        np,
        paths,
        pd,
        plot_prepared_weight_family,
        plt,
        prepare_weights_df,
        re,
        sns,
        ttest_1samp,
    )


@app.cell
def _(ROOT, re):
    task_specs = {
        "2AFC_delay": {"label": "2ADC", "stim_family": "stim_hot"},
        "2AFC": {"label": "2AFC", "stim_family": "stim_hot"},
        "MCDR": {"label": "3CDR", "stim_family": "stim_hot"},
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

    model_names_by_task = {
        task_name: saved_model_names(task_name, "glm")
        for task_name in task_names
    }
    return model_names_by_task, saved_model_names, task_names, task_specs


@app.cell
def _(mo, model_names_by_task, task_names, task_specs):
    missing_model_tasks = [
        task_specs[_task_name]["label"]
        for _task_name in task_names
        if not model_names_by_task[_task_name]
    ]
    mo.stop(
        bool(missing_model_tasks),
        mo.md(
            "No saved non-hash GLM models were found for: "
            + ", ".join(missing_model_tasks)
            + "."
        ),
    )

    model_selectors = {}
    preferred_defaults = {
        "2AFC_delay": "one hot",
        "2AFC": "one hot",
        "MCDR": "one hot 11",
    }

    for _task_name in task_names:
        options = model_names_by_task[_task_name]
        preferred = preferred_defaults.get(_task_name, "one hot")
        default_value = (
            preferred
            if preferred in options
            else "one hot"
            if "one hot" in options
            else options[0]
        )
        model_selectors[_task_name] = mo.ui.dropdown(
            options=options,
            value=default_value,
            label=f"{task_specs[_task_name]['label']} GLM",
        )

    mo.hstack([model_selectors[_task_name] for _task_name in task_names], justify="start")
    return (model_selectors,)


@app.cell
def _(
    build_emission_weights_df,
    build_views,
    get_adapter,
    load_fit_arrays,
    mo,
    model_selectors,
    paths,
    task_names,
    task_specs,
):
    def load_weight_payloads(
        model_selectors: dict,
        model_kind: str,
        *,
        require_all: bool = False,
    ) -> dict:
        payloads = {}
        for _task_name, _selector in model_selectors.items():
            if _selector.value == "None":
                continue
            _payload = build_weight_payload(
                _task_name,
                model_kind,
                _selector.value,
                require_subjects=require_all,
            )
            if _payload is not None:
                payloads[_task_name] = _payload
        return payloads

    def build_weight_payload(
        task_name: str,
        model_kind: str,
        model_name: str,
        *,
        require_subjects: bool = False,
    ) -> dict | None:
        adapter = get_adapter(task_name)
        df_all = adapter.subject_filter(adapter.read_dataset())
        subjects = list(df_all["subject"].unique())
        out_dir = paths.RESULTS / "fits" / task_name / model_kind / model_name

        arrays_store, _ = load_fit_arrays(
            out_dir=out_dir,
            arrays_suffix=f"{model_kind}_arrays.npz",
            adapter=adapter,
            df_all=df_all,
            subjects=subjects,
            emission_cols=None,
        )
        selected_subjects = [
            str(_subject) for _subject in subjects if str(_subject) in arrays_store
        ]
        if not selected_subjects:
            selected_subjects = list(arrays_store)
        if not selected_subjects:
            mo.stop(
                require_subjects,
                mo.md(f"No fitted subjects found for `{task_name}/{model_kind}/{model_name}`."),
            )
            return None

        first_arrays = arrays_store[selected_subjects[0]]
        K = int(first_arrays["emission_weights"].shape[0])
        views = build_views(arrays_store, adapter, K, selected_subjects)
        weights_df = build_emission_weights_df(views)
        return {
            "adapter": adapter,
            "K": K,
            "model_name": model_name,
            "model_kind": model_kind,
            "n_subjects": len(selected_subjects),
            "weights_df": weights_df,
        }

    weight_payloads = load_weight_payloads(model_selectors, "glm", require_all=True)
    mo.md(
        "Loaded GLMs: "
        + ", ".join(
            f"`{task_specs[_task_name]['label']}: {weight_payloads[_task_name]['model_name']}`"
            for _task_name in task_names
        )
        + "."
    )
    return load_weight_payloads, weight_payloads


@app.cell
def _(np, pd, ttest_1samp):
    def significance_stars(pvalue: float) -> str:
        if not pd.notna(pvalue) or pvalue >= 0.05:
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        return "*"

    def annotate_baseline_ttests(ax, prepared, *, expand_axis: bool = True) -> None:
        if prepared is None or prepared.x_order is None:
            return

        df = pd.DataFrame(prepared.data).copy()
        if df.empty:
            return
        df["x_label"] = df["x_label"].astype(str)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        if y_range <= 0:
            return

        y_values = []
        annotations = []
        for _x_pos, _x_label in enumerate(prepared.x_order, start=1):
            values = df.loc[df["x_label"] == str(_x_label), "weight"].dropna()
            if len(values) < 2:
                continue
            pvalue = ttest_1samp(values.to_numpy(dtype=float), 0.0).pvalue
            stars = significance_stars(float(pvalue))
            if not stars:
                continue
            q3 = float(np.nanpercentile(values.to_numpy(dtype=float), 95))
            y = max(q3, 0.0) + 0.001 * y_range
            if not expand_axis:
                y = min(y, ylim[1] - 0.4 * y_range)
            annotations.append((_x_pos, y, stars))
            y_values.append(y)

        if y_values and expand_axis:
            ax.set_ylim(ylim[0], max(ylim[1], max(y_values) + 0.08 * y_range))
        if y_values:
            for x_pos, y, stars in annotations:
                ax.text(x_pos, y, stars, ha="center", va="bottom")

    return (annotate_baseline_ttests,)


@app.cell
def _(
    ROOT,
    annotate_baseline_ttests,
    fig_size,
    plot_prepared_weight_family,
    plt,
    sns,
    task_names,
    task_specs,
    weight_payloads,
):
    fig, axd = plt.subplot_mosaic(
        [
            ["stim_2adc", "stim_2afc", "stim_3cdr"],
            ["choice_2adc", "choice_2afc", "choice_3cdr"],
        ],
        constrained_layout=True, figsize= fig_size(1,2)
    )

    axis_keys = {
        "2AFC_delay": ("stim_2adc", "choice_2adc"),
        "2AFC": ("stim_2afc", "choice_2afc"),
        "MCDR": ("stim_3cdr", "choice_3cdr"),
    }

    for _task_name in task_names:
        payload = weight_payloads[_task_name]
        adapter = payload["adapter"]
        weights_df = payload["weights_df"]
        stim_key, choice_key = axis_keys[_task_name]

        for _family_key, _axis_key in [
            (task_specs[_task_name]["stim_family"], stim_key),
            ("choice_lag", choice_key),
        ]:
            ax = axd[_axis_key]
            prepared = adapter.prepare_weight_family_plot(
                weights_df,
                _family_key,
                variant="folded",
            )
            if prepared is None:
                ax.text(0.5, 0.5, "No weights", ha="center", va="center")
                ax.axis("off")
                continue

            plot_prepared_weight_family(
                prepared,
                ax=ax,
                figsize=fig_size(n_cols=3,ratio=1),
                title="",
                connect_subjects=False,
            )
            if _family_key == "choice_lag":
                ax.set_ylim(-0.1, 1.1)
            annotate_baseline_ttests(
                ax,
                prepared,
                expand_axis=_family_key != "choice_lag",
            )

        axd[stim_key].set_title(task_specs[_task_name]["label"])

    for _key, ax in axd.items():
        ax.set_ylabel("Weight")
    axd["stim_2adc"].set_ylabel("Stimulus\nWeight")
    axd["choice_2adc"].set_ylabel("Previous choices\nWeight")

    sns.despine(fig=fig)
    fig.savefig("emission_weights_mosaic.pdf")
    fig.savefig("emission_weights_mosaic.png")

    out_path = ROOT / "figures" / "__marimo__" / "assets" / "emission_weights_mosaic" / "opengraph.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig
    return


@app.cell
def _(mo, saved_model_names, task_names, task_specs):
    def make_model_selectors(model_kind: str) -> dict:
        selectors = {}
        for _task_name in task_names:
            _options = saved_model_names(_task_name, model_kind)
            if not _options:
                continue
            selectors[_task_name] = mo.ui.dropdown(
                options=["None", *_options],
                value=_options[0],
                label=f"{task_specs[_task_name]['label']} {model_kind}",
            )
        return selectors

    return (make_model_selectors,)


@app.cell
def _(make_model_selectors, mo):
    glmhmm_model_selectors = make_model_selectors("glmhmm")
    mo.hstack(list(glmhmm_model_selectors.values()), justify="start")
    return (glmhmm_model_selectors,)


@app.cell
def _(glmhmm_model_selectors, load_weight_payloads, mo, task_specs):
    glmhmm_weight_payloads = load_weight_payloads(glmhmm_model_selectors, "glmhmm")
    mo.md(
        "Loaded GLMHMMs: "
        + (
            ", ".join(
                f"`{task_specs[_task_name]['label']}: {_payload['model_name']}`"
                for _task_name, _payload in glmhmm_weight_payloads.items()
            )
            if glmhmm_weight_payloads
            else "none."
        )
    )
    return (glmhmm_weight_payloads,)


@app.cell
def _(
    custom_boxplot,
    fig_size,
    fold_three_choice_raw_weights,
    np,
    pd,
    plt,
    prepare_weights_df,
    sns,
    task_specs,
):
    def plot_all_emission_weights(weights_df, ax, *, K: int | None = None):
        plot_weights = fold_three_choice_raw_weights(weights_df)
        if plot_weights is None:
            plot_weights = weights_df

        raw_df = plot_weights.to_pandas() if hasattr(plot_weights, "to_pandas") else pd.DataFrame(plot_weights)
        if raw_df.empty or "state_label" not in raw_df.columns:
            return False

        if "state_rank" in raw_df.columns:
            state_label_order = (
                raw_df[["state_label", "state_rank"]]
                .drop_duplicates()
                .assign(state_rank=lambda _df: pd.to_numeric(_df["state_rank"], errors="coerce"))
                .sort_values("state_rank")["state_label"]
                .astype(str)
                .tolist()
            )
        else:
            state_label_order = None

        feature_labels = {
            "stim_param": r"$\mathrm{Stim}_{\mathrm{param}}$",
            "stim_x_delay_param": r"$\mathrm{Stim}\times\mathrm{Delay}_{\mathrm{param}}$",
            "bias_param": r"$\mathrm{Bias}_{\mathrm{param}}$",
            "biasparam": r"$\mathrm{Bias}_{\mathrm{param}}$",
            "at_choice_param": r"$\mathrm{A}_t$",
            "choice_lag_param": r"$\mathrm{A}$",
        }
        feature_labeler = lambda feature: feature_labels.get(str(feature), str(feature))

        try:
            df, features, display_features, state_order, palette = prepare_weights_df(
                plot_weights,
                K=K,
                state_label_order=state_label_order,
                feature_labeler=feature_labeler,
            )
        except ValueError:
            return False

        grouped_values = []
        for state_label in state_order:
            grouped_values.append(
                [
                    df.loc[
                        (df["state_label"] == state_label)
                        & (df["feature"] == feature),
                        "weight",
                    ].to_numpy(dtype=float)
                    for feature in features
                ]
            )
        if not features or not state_order:
            return False

        n_features = len(features)
        n_states = len(state_order)
        group_width = 0.8
        hue_width = group_width / max(1, n_states)
        feature_positions = np.arange(1, n_features + 1, dtype=float)

        for state_idx, state_label in enumerate(state_order):
            positions = feature_positions + (state_idx - (n_states - 1) / 2.0) * hue_width
            custom_boxplot(
                ax,
                grouped_values[state_idx],
                positions=positions,
                widths=hue_width * 0.78,
                median_colors=palette[state_label],
            )
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set_xlabel("")
        ax.set_ylabel("Weight")
        ax.set_xticks(feature_positions)
        ax.set_xticklabels(display_features, rotation=0, ha="center")
        if n_states > 1:
            handles = [
                plt.Line2D([0], [0], color=palette[_state], linewidth=3, label=_state)
                for _state in state_order
            ]
            ax.legend(handles=handles, frameon=False)
        return True

    def plot_selected_emission_mosaic(weight_payloads: dict, filename_stem: str):
        selected_tasks = list(weight_payloads)
        if not selected_tasks:
            return None

        panel_width, panel_height = fig_size(3, 1.25)
        fig, axes = plt.subplots(
            1,
            len(selected_tasks),
            figsize=(panel_width * len(selected_tasks), panel_height),
            constrained_layout=True,
            squeeze=False,
        )

        for _axis, _task_name in zip(axes.ravel(), selected_tasks, strict=False):
            _payload = weight_payloads[_task_name]
            if not plot_all_emission_weights(_payload["weights_df"], _axis, K=_payload["K"]):
                _axis.text(0.5, 0.5, "No weights", ha="center", va="center")
                _axis.axis("off")
                continue
            _axis.set_title(task_specs[_task_name]["label"])

        for _axis in axes.ravel()[1:]:
            _axis.set_ylabel("")

        fig.supxlabel("Emission feature")
        sns.despine(fig=fig)
        fig.savefig(f"{filename_stem}.pdf")
        fig.savefig(f"{filename_stem}.png")
        return fig

    return (plot_selected_emission_mosaic,)


@app.cell
def _(glmhmm_weight_payloads, mo, plot_selected_emission_mosaic):
    mo.stop(not glmhmm_weight_payloads, mo.md("No GLMHMM models selected."))
    plot_selected_emission_mosaic(glmhmm_weight_payloads, "glmhmm_emission_weights_mosaic")
    return


@app.cell
def _(make_model_selectors, mo):
    glmhmmt_model_selectors = make_model_selectors("glmhmmt")
    mo.hstack(list(glmhmmt_model_selectors.values()), justify="start")
    return (glmhmmt_model_selectors,)


@app.cell
def _(glmhmmt_model_selectors, load_weight_payloads, mo, task_specs):
    glmhmmt_weight_payloads = load_weight_payloads(glmhmmt_model_selectors, "glmhmmt")
    mo.md(
        "Loaded GLMHMMTs: "
        + (
            ", ".join(
                f"`{task_specs[_task_name]['label']}: {_payload['model_name']}`"
                for _task_name, _payload in glmhmmt_weight_payloads.items()
            )
            if glmhmmt_weight_payloads
            else "none."
        )
    )
    return (glmhmmt_weight_payloads,)


@app.cell
def _(glmhmmt_weight_payloads, mo, plot_selected_emission_mosaic):
    mo.stop(not glmhmmt_weight_payloads, mo.md("No GLMHMMT models selected."))
    plot_selected_emission_mosaic(glmhmmt_weight_payloads, "glmhmmt_emission_weights_mosaic")
    return


if __name__ == "__main__":
    app.run()
