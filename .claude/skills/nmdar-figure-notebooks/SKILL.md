---
name: nmdar-figure-notebooks
description: Build and revise Marimo figure notebooks for the NMDAR paper. Use when editing files under figures/ such as figure1.py, figure2.py, or figure3.py, especially for GLM, GLM-HMM, or GLM-HMM-T figure panels that should use short cells, project helpers in src/process/common.py, seaborn/matplotlib plotting, dictionaries of task dataframes, and project style palettes.
---

# NMDAR Figure Notebooks

## Design Decisions

- Keep notebooks as paper-facing assembly files: imports, settings, paths, model selection, data loading, short plot cells, and optional figure mosaic.
- Put reusable preprocessing in `src/process/common.py`. Notebook cells should call helpers that return ready-to-plot pandas dataframes or small dictionaries.
- Prefer Polars expressions for notebook-side dataframe mutation and derived columns, for example `with_columns(pl.col(...))`. Use pandas only where seaborn/statannotations/helper APIs already return or require pandas dataframes.
- Plot directly with seaborn or simple matplotlib. Avoid notebook-local heavy payload construction unless it is one-off display code.
- Prefer explicit dictionaries keyed by task name: `adapters`, `dfs`, `subjects_by_task`, `arrays_by_task`, `views`, `trial_dfs`, `weight_dfs`, `plot_dfs`, and task-specific plot-data dictionaries.
- Define palettes and label maps near the Style cell. Plot cells should reuse `task_palette`, `state_palette`, `task_labels`, and `feature_labels` instead of redefining colors. For semantic state plots, use `state_palette = {"Engaged": "tab:green", "Disengaged": "tab:gray"}` unless the user explicitly asks for different colors.
- Use Markdown title cells for every major section and subplot family.
- Keep cells short. A cell should usually do one of: configure, load, preprocess, or draw one plot family.
- Do not use `mo.stop(...)` in figure notebooks unless the user explicitly asks for blocking reactive execution. Prefer returning typed empty dataframes and displaying a short `mo.md(...)` message in the plotting cell when data are unavailable.
- For split figure notebooks, make each task panel an explicit Marimo cell with its own Markdown subheading. Do not hide repeated plotting code behind notebook-local `plot_*_task` functions; keep the seaborn/matplotlib call, labels, and axis save/display code in the task cell.
- In split plot cells, avoid local boilerplate such as `_task_name`, `_df`, `_order`, `_hue_order`, and `mo.stop(...)`. Precompute plot orders, hue orders, task-label dataframes, and small summary dataframes in a single cell before the plot section, then reference dictionaries directly in each plot cell.
- Prefer one standalone plot per cell. When the user wants individual panels, do not use `plt.subplots`; create each panel with `plt.figure(figsize=figsize, constrained_layout=True)`, bind the axis with `panel_axis = plt.gca()`, save both `.svg` and `.png`, then return the axis.
- Save standalone panels into `figures/panelsN/<format>/` using `path_panels / stem.with_suffix(...)`.

## Figure Notebook Workflow

1. Read the closest existing figure notebooks first, usually `figures/figure1.py` and `figures/figure2.py`.
2. Mirror their Marimo structure: imports, settings, paths, style, load data and fits, derived plot data, plots, optional mounted figure.
3. Create a top-level model dictionary before loading fits. Use one model id for every task when possible; otherwise use task-specific ids:

```python
MODEL_BY_TASK = {
    "2AFC_delay": "model-id",
    "2AFC": "model-id",
    "MCDR": "model-id",
}
```

4. Load adapters and raw data into dictionaries, then load one fitted model per task from `results/fits/<task>/glmhmmt/<model_id>`.
5. Build `views`, `trial_dfs`, `weight_dfs`, and `plot_dfs` in one loading cell. Return all dictionaries needed by later cells.
6. Build derived plot data with helpers from `src/process/common.py`.
7. In plot cells, filter an already-prepared dataframe and call seaborn/matplotlib directly.
8. When the user asks for a split notebook, duplicate the short plotting cells per task, separated by Markdown cells such as `### 2ADC`, `### 2AFC`, and `### 3CDR`, rather than defining task-parameterized plot functions.
9. Before the plot section in split notebooks, define dictionaries such as `emission_orders`, `emission_hue_orders`, `transition_orders`, `psychometric_orders`, `accuracy_plot_dfs`, `occupancy_plot_dfs`, and summary tables needed by the plot cells.
10. Prefer axis-first plot cells that can mount into a figure mosaic:

```python
plt.figure(figsize=fig_size(1, 1), constrained_layout=True)
panel_axis = plt.gca() if not mount_figure else axd["panel_key"]
panel_axis.clear()

sns.lineplot(data=plot_df, x="x", y="y", ax=panel_axis)
panel_axis.set_xlabel("")

if not mount_figure:
    panel_axis.figure.savefig((path_panels / "panel_stem").with_suffix(f".{format}"))
panel_axis
```

Use `axis.figure.savefig(...)` rather than carrying a local `_fig` variable in split plot cells.

## GLM-HMM-T Figure 3 Conventions

- Use `glmhmmt.notebook_support.analysis_common.load_fit_arrays` with `arrays_suffix="glmhmmt_arrays.npz"` and `k=K`.
- Read model config from `config.json` when available to get `K`, `subjects`, `emission_cols`, and `transition_cols`.
- Set adapter state scoring attributes from config keys when present before calling `build_views`.
- Use `glmhmmt.postprocess` builders for model-derived payloads, but wrap any repeated dataframe shaping in `src/process/common.py`.
- Figure 3 should include initial cells for:
  - emission weights
  - transition weights
  - psychometrics by state
  - accuracy by state
  - occupancy by state
  - mean traces
  - 2AFC-only RT and lick plots by state
  - cumulative dwell-time probability
  - number of state switches per session, then averaged per animal
  - posteriors around a change

## Plotting Standards

- Use `sns.set_theme(style="ticks", context="notebook")` plus `styles/paper.mplstyle`.
- Set `plt.rcParams["svg.fonttype"] = "none"` and `plt.rcParams["savefig.bbox"] = "standard"`.
- Use `fig_size(n_cols, ratio)` for figure dimensions. The first argument is the number of A4-width columns occupied by the full figure, and the second is the width/height ratio. For a horizontally arranged multi-panel figure, prefer increasing the ratio instead of the column count; for example use `fig_size(1, 3)` for a one-column-wide figure with three horizontal subplots.
- Prefer `sns.boxplot`, `sns.pointplot`, `sns.lineplot`, `sns.ecdfplot`, and small `ax.errorbar` calls.
- For lineplots with markers, remove marker outlines (`markeredgewidth=0`, `markeredgecolor="none"`). For lineplots with filled error bands, remove band edges with `err_kws={"edgecolor": "none", "linewidth": 0}` or by clearing collection edge colors after plotting.
- Add `ax.axhline(...)` or `ax.axvline(...)` only when it communicates the baseline.
- Keep labels compact and paper-ready; avoid explanatory text inside plots. Prefer short tick labels that fit horizontally over rotated labels.
