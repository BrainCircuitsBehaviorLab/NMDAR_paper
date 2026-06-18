# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research analysis codebase for a neuroscience paper studying NMDAR antagonist effects on decision-making behavior. Fits GLM, GLM-HMM, and GLM-HMM-T (with transition regressors) models to behavioral data from multiple tasks, then generates publication figures.

## Environment setup

```bash
uv sync
uv pip install --reinstall --no-deps tfp-nightly==0.26.0.dev20260205
```

The `tfp-nightly` reinstall is required because both `tensorflow-probability` and `tfp-nightly` provide the same `tensorflow_probability` import; the nightly must be installed on top. Restart any running kernel after this step.

Python 3.11–3.13. Package manager is `uv`. Linter is `ruff`.

## Running notebooks and figures

Notebooks and figure scripts are **marimo** apps (`.py` files with `marimo.App()`). Run them with:

```bash
uv run marimo edit notebooks/glmhmmt_analysis.py
uv run marimo edit figures/figure1.py
```

## Batch plot generation

```bash
uv run python src/fit_and_plot.py                          # plots for FITS list in that file
uv run python src/fit_helper.py --task 2AFC --model-id param --model-kind glmhmm
```

`fit_helper.py` accepts `--fit-dir` for a direct path or `--task`/`--model-id`/`--model-kind` to resolve under `results/fits/<task>/<model_kind>/<model_id>`. Add `--fit` to refit before plotting.

## Architecture

### Config and runtime paths

`config.toml` at the project root defines `data_dir`, `results_dir`, adapter/plot plugin paths, palettes, and plot-saver settings. The `glmhmmt` library reads this via `configure_paths()` / `get_runtime_paths()` / `load_app_config()`. `PROJECT_ROOT` is located by searching parents for a directory containing both `config.toml` and `src/`.

### Task adapters (`src/process/`)

Each behavioral task has a **task adapter** module registered with `glmhmmt.tasks`. The adapters define dataset loading, design matrices, regressors, scoring, and task-specific prediction builders.

| Module | Task key | Description |
|---|---|---|
| `two_afc.py` | `2AFC` | Alexis human 2AFC auditory task |
| `two_afc_drug.py` | `2AFC_DRUG` | Drug/saline cohort of 2AFC |
| `two_adc.py` | `2AFC_delay` / `2ADC` | Tiffany 2AFC with delay |
| `two_adc_drug.py` | `2ADC_DRUG` / `2AFC_delay_DRUG` | Drug/saline cohort of 2ADC |
| `MCDR.py` | `MCDR` | 3-AFC rat task |
| `nuo_auditory.py` | `nuo_auditory` | Nuo auditory 2AFC |

Adapters expose `read_dataset()`, `subject_filter()`, `filter_condition_df()`, `get_plots()`, and design-matrix builders. They are the main extension point for new tasks.

### Plot modules (`src/plots/`)

`common.py` contains shared plotting utilities, figure-sizing (`fig_size()`), summary builders (psychometrics, repetition bias, integration maps), and overlay/group helpers. Task-specific modules (`two_afc.py`, `MCDR.py`, etc.) provide task-owned figure functions. The `glmhmmt.plots` package provides model-level plots (emission/transition weights, transition matrices, state diagnostics).

### Data pipeline (`src/process/common.py`)

Large shared module (~6000 lines) with:
- Payload/summary builders: `glmhmmt_state_*_df()` family for state-conditioned behavioral summaries
- Psychometric curve fitting: `fit_lapse_logistic_curve()` / `fit_lapse_logistic_by_group()` with 4-parameter lapse-logistic model
- 2D integration maps: `integration_map_2d()` / `prepare_right_integration_maps()`
- Regressor utilities: `display_regressor_name()`, `add_choice_lag_summary_regressor()`, quantile binning

### Model fitting

Models are fit via `glmhmmt.cli.fit_glmhmm` and `glmhmmt.cli.fit_glmhmmt`. Each fit is stored under `results/fits/<task>/<model_kind>/<model_id>/` with a `config.json` describing parameters and per-subject `*_arrays.npz` + `*_metrics.parquet` files.

### Figures (`figures/`)

Marimo apps assembling publication panels. `figure1.py` = behavioral performance, `figure2.py` = GLM model predictions, `figure3.py` = GLM-HMM-T states. Figures use `paper.mplstyle` (despine, no grid, 300 DPI, editable SVG text). Panels are exported to `figures/panels*/pdf/`.

### Data

Processed parquet files in `data/processed/`. Key datasets: `auditory_2AFC.parquet`, `df_alexis_drug_combined.parquet`, `tiffany.parquet`, `MCDR_all.parquet`. Raw data in `data/raw/`.

## Key conventions

- DataFrames: Polars (`pl.DataFrame`) is the primary format throughout adapters and postprocessing; conversion to pandas happens at plot boundaries via `to_pandas_df()`.
- Figure sizing: Use `fig_size(n_cols, ratio)` from `src/utils.py` or `src/plots/common.py` for A4-layout-aware sizing.
- State palette: Engaged = `tab:green`, Disengaged = `tab:gray`. K=4 palette defined in `config.toml`.
- Plot style: `sns.set_style("ticks")` + `paper.mplstyle`. Titles are cleaned off for publication via `clean_title()`.
- The `glmhmmt` package (v0.3.11) is the core modeling library providing HMM fitting, views, postprocessing, and plots.

## Working conventions for Claude Code

### Before making changes
- Always read the relevant adapter or plot module before editing it
- For `src/process/common.py`, search for existing utilities before writing new ones — it's ~6000 lines and likely already has what you need
- Check `config.toml` before hardcoding any path, palette, or parameter

### Code style
- Use Polars (`pl`) throughout; only convert to pandas at plot boundaries via `to_pandas_df()`
- Follow existing adapter structure when adding a new task: implement `read_dataset()`, `subject_filter()`, `filter_condition_df()`, `get_plots()`, and design-matrix builders
- Run `uv run ruff check` and `uv run ruff format` before considering any task done

### Data safety
- Never overwrite files under `data/raw/` — treat as read-only
- Never refit models without explicit instruction — fitting is slow and results are versioned
- When in doubt about a result path, resolve via `--task`/`--model-id`/`--model-kind` convention, not hardcoded strings

### Figures
- All figure scripts are marimo apps — do not convert to plain scripts
- Use `fig_size(n_cols, ratio)` for sizing, never hardcode figure dimensions
- Apply `paper.mplstyle` and `sns.set_style("ticks")` for all publication plots
- State colors: Engaged = `tab:green`, Disengaged = `tab:gray`

### When debugging
- Check `config.json` in the fit directory first for model parameters
- Confirm `configure_paths()` has been called before any path resolution
- For import errors with `tensorflow_probability`, re-run the `uv pip install --reinstall` step from Environment setup