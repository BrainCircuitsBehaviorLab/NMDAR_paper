# Codebase Documentation

## What this project does

This codebase analyzes how NMDAR antagonist drugs affect decision-making behavior. It takes raw behavioral trial data from multiple experimental tasks (human auditory discrimination, rat multi-choice), fits progressively richer statistical models (GLM, GLM-HMM, GLM-HMM-T), and produces publication-ready figures showing how subjects switch between cognitive states and how drugs alter those dynamics.

The pipeline flows in one direction:

```
Raw data --> Parsing --> Processed parquet --> Task adapters --> Model fitting --> Postprocessing --> Figures
```

---

## Main modules

### `config.toml` -- Central configuration

Defines all runtime paths (`data_dir`, `results_dir`), plugin discovery paths for adapters and plots, color palettes (state colors, K=4 palette), and plot-saver settings. Every other module reads this instead of hardcoding paths or parameters.

### `src/process/` -- Task adapters

Each file is a **task adapter** -- a self-contained module that knows how to load, filter, and build design matrices for one experimental task. They all follow the same interface: `read_dataset()`, `subject_filter()`, `filter_condition_df()`, `get_plots()`, plus design-matrix builders.

| File | Task | What it wraps |
|---|---|---|
| `two_afc.py` | `2AFC` | Alexis human 2AFC auditory discrimination |
| `two_afc_drug.py` | `2AFC_DRUG` | Drug vs. saline cohort of the above |
| `two_adc.py` | `2AFC_delay` / `2ADC` | Tiffany 2AFC with delay manipulation |
| `two_adc_drug.py` | `2ADC_DRUG` | Drug vs. saline cohort of the delay task |
| `MCDR.py` | `MCDR` | 3-AFC rat task |
| `nuo_auditory.py` | `nuo_auditory` | Nuo auditory 2AFC |

Adapters register themselves with the `glmhmmt.tasks` system so the fitting CLI and notebooks can look them up by task key string (e.g., `get_adapter("2AFC")`).

**Supporting files in `src/process/`:**

- **`common.py`** (~6000 lines) -- Shared data utilities used by all adapters: state-conditioned behavioral summaries (`glmhmmt_state_*_df()` family), psychometric curve fitting (4-parameter lapse-logistic), 2D integration maps, regressor display names, quantile binning, and Polars-to-pandas conversion (`to_pandas_df()`).
- **`design.py`** -- Small helpers for building design matrices: lagged regressors, one-hot encodings, level indicators, session dummies. Used inside each adapter's matrix builder.
- **`plot_payloads.py`** -- Shared trial-level payload builders for binary tasks. Adapters pass a `TaskPlotProfile` (which columns mean what, which axis to use) and this module produces standardized DataFrames ready for plotting.
- **`_choice_tau.py`** / **`_transition_params.py`** -- Private helpers for choice history EWMA decay and transition weight computations.

### `src/plots/` -- Plot modules

Task-specific plotting functions that take the payloads from `src/process/` and produce matplotlib figures.

| File | What it plots |
|---|---|
| `common.py` | Shared utilities: psychometric curves, boxplot styling, significance stars, axis formatting, overlay/group helpers |
| `two_afc.py` | 2AFC-specific figures (psychometrics by condition, integration maps) |
| `two_afc_drug.py` | Drug vs. saline comparison plots for 2AFC |
| `two_adc.py` / `two_adc_drug.py` | Same for the delay task |
| `MCDR.py` | Rat 3-AFC task plots |
| `nuo_auditory.py` | Nuo auditory plots |
| `mcdr_accuracy.py` | MCDR accuracy-specific analyses |

### `src/parsing/` -- Raw data parsers

Scripts that convert raw experimental files into the processed parquet files under `data/processed/`. Not run during normal analysis -- only when raw data changes.

- `parse_2AFC.py` -- Parses the Alexis 2AFC experiment
- `parse_tiffany.py` -- Parses Tiffany's delay experiment
- `parse_balma11.py` -- Parses Balma data
- `licks.py` -- Extracts lick timing data

### `src/fit_helper.py` -- Single-model fit + plot CLI

The workhorse script for fitting one model and generating its diagnostic plots. Accepts `--task`, `--model-id`, `--model-kind` (or `--fit-dir` for a direct path). Calls into `glmhmmt.cli.fit_glmhmm` or `glmhmmt.cli.fit_glmhmmt` for the actual fitting, then loads results, builds views, computes postprocessing payloads (emission weights, transition matrices, state accuracy), and renders all model-level plots.

### `src/fit_and_plot.py` -- Batch fit + plot runner

Iterates over a hardcoded `FITS` list of `(task, model_kind, model_id)` tuples and calls `fit_helper.save_model_plots()` for each. Used to regenerate all saved model plots in one go.

### `src/utils.py` -- Shared utilities

Currently contains `fig_size(n_cols, ratio)` which computes figure dimensions for A4 page layouts.

---

## External dependency: `glmhmmt` package

The `glmhmmt` library (v0.3.11) is the core modeling engine. This project does **not** contain the model fitting code itself -- it lives in the external package. Key submodules used:

| Submodule | Role |
|---|---|
| `glmhmmt.cli.fit_glmhmm` / `fit_glmhmmt` | CLI entry points for fitting GLM-HMM and GLM-HMM-T models |
| `glmhmmt.tasks` | Task adapter registry (`get_adapter()`, `TaskAdapter`, `_register`) |
| `glmhmmt.views` | `build_views()` -- constructs model views from fitted arrays |
| `glmhmmt.postprocess` | Builds structured payloads from model fits: emission weights, transition matrices, state accuracy, posterior counts |
| `glmhmmt.plots` | Model-level plot functions (emission/transition weights, transition matrices, state diagnostics) |
| `glmhmmt.runtime` | Path management: `configure_paths()`, `get_runtime_paths()`, `load_app_config()` |
| `glmhmmt.notebook_support` | Interactive widgets and helpers for marimo notebooks: `ModelManagerWidget`, `CoefficientEditorWidget`, `build_trial_and_weights_df()`, `load_fit_arrays()` |

---

## Figures (`figures/`)

All figure scripts are **marimo apps** (Python files with `marimo.App()`). They are run interactively with `uv run marimo edit figures/figure1.py`.

| File | Content |
|---|---|
| `figure1.py` | Behavioral performance across tasks -- psychometric curves, accuracy summaries |
| `figure2.py` | GLM model predictions -- fitted weights, model vs. data comparisons across tasks |
| `figure3.py` | GLM-HMM-T states -- state-dependent behavior, transition dynamics, drug effects |
| `figure3_short.py` | Condensed version of figure 3 |
| `emission_weights_mosaic.py` | Mosaic layout of emission weight comparisons |
| `glm_glmhmm_summary_mosaic.py` | Side-by-side GLM vs GLM-HMM summary |
| `autocorrelograms.py` | Choice autocorrelation analysis |
| `licks_analysis.py` | Lick timing analysis |
| `models_cartoon.py` | Schematic/cartoon of the model architectures |

Figures load adapters, call into `src/process/` for data prep, use `src/plots/` for rendering, and export panels to `figures/panels*/pdf/`.

---

## Notebooks (`notebooks/`)

Marimo apps for interactive model exploration and comparison:

| File | Purpose |
|---|---|
| `glmhmmt_analysis.py` | Main GLM-HMM-T analysis workbench -- load fits, inspect states, tweak coefficients |
| `glmhmm.py` | GLM-HMM exploration |
| `glm.py` | Basic GLM analysis |
| `glm_drug_comparison.py` | Drug vs. saline GLM comparison |
| `glmhmm_pairwise_comparison.py` | Pairwise model comparison |
| `glmhmmt_drug_transition_comparison.py` | Drug effects on state transitions |
| `glmgmmt_analysis.py` | GLM-GMMT variant analysis |
| `model_comparison.py` | Cross-model comparison metrics |

---

## Data flow

1. **Raw data** (`data/raw/`) is parsed once by `src/parsing/` scripts into **processed parquet files** (`data/processed/`).

2. **Task adapters** (`src/process/*.py`) load the parquet files, apply filters (experiment selection, subject exclusion), and build design matrices (stimulus regressors, choice history lags, session indicators).

3. **Model fitting** happens via the `glmhmmt` CLI. Each fit writes to `results/fits/<task>/<model_kind>/<model_id>/` with a `config.json` (parameters), per-subject `*_arrays.npz` (fitted weights, posteriors), and `*_metrics.parquet` (fit quality).

4. **Postprocessing** (`glmhmmt.postprocess` + `src/process/common.py`) takes the fitted arrays and builds structured payloads: emission weight DataFrames, transition matrices, state-conditioned behavioral summaries, psychometric fits per state.

5. **Plotting** (`src/plots/` + `glmhmmt.plots`) renders the payloads into matplotlib figures with publication styling (`paper.mplstyle`).

6. **Figures** (`figures/*.py`) assemble individual panels into composite publication figures and export them.

---

## How to run things

```bash
# Install dependencies
uv sync
uv pip install --reinstall --no-deps tfp-nightly==0.26.0.dev20260205

# Open a figure interactively
uv run marimo edit figures/figure1.py

# Open an analysis notebook
uv run marimo edit notebooks/glmhmmt_analysis.py

# Generate plots for all saved fits
uv run python src/fit_and_plot.py

# Fit + plot a single model
uv run python src/fit_helper.py --task 2AFC --model-id param --model-kind glmhmm

# Lint
uv run ruff check
uv run ruff format
```
