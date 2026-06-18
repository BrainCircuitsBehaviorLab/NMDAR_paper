# Codebase Diagnosis: Fragile, Hacky, and Confusing Parts

## 1. `src/process/common.py` is a 6200-line grab bag

This is the single biggest risk in the codebase. It contains:

- Data classes (`PreparedWeightFamilyPlot`, `LapseLogisticFit`, `TaskPlotColumns`)
- DataFrame summary builders (`glmhmmt_state_*_df()` -- 10+ functions)
- Psychometric curve fitting (`fit_lapse_logistic_curve` + 3 variants, ~300 lines)
- 2D integration maps (~200 lines)
- Counterfactual simulation engine (`build_action_trace_counterfactual`, ~270 lines)
- Closed-loop autocorrelogram simulation (~1200 lines, from line ~4300 to ~5600)
- GLM evaluation grids (`eval_glm_on_ild_grid`, `eval_glm_on_feature_grid`)
- State scoring/labeling (~250 lines)
- Repetition bias utilities
- Column resolution helpers
- Quantile binning

**Why it's fragile:** Any change risks breaking unrelated functionality. You can't reason about what depends on what. The autocorrelogram simulation block alone (lines ~4300-6228) is a self-contained subsystem that shares nothing with the psychometric fitting above it.

**Refactor recommendation:** Split into at least:
- `common/dataframes.py` -- summary builders (`glmhmmt_state_*_df()`)
- `common/psychometric.py` -- lapse-logistic fitting
- `common/integration.py` -- 2D integration maps
- `common/autocorrelogram.py` -- the entire closed-loop simulation pipeline
- `common/counterfactual.py` -- action-trace counterfactual analysis
- `common/columns.py` -- column resolution, regressor naming, quantile binning

---

## 2. Duplicated `fig_size()` -- two copies that disagree

`src/utils.py:6` and `src/plots/common.py:97` both define `fig_size()`, and they compute different results:

- `src/utils.py` uses **mm** internally: `A4_size = (210, 297)`, `margins = 50.8`
- `src/plots/common.py` uses **inches**: `A4_size = (8.27, 11.69)`, `margins = 2`

These produce slightly different outputs because `210mm / 25.4 = 8.2677...`, not `8.27`. Half the codebase imports from one, half from the other. Figure scripts use `src.utils.fig_size`, mosaic figures use `src.plots.common.fig_size`.

**Fix:** Delete one, import from the other everywhere.

---

## 3. Duplicated `pick_existing_column` / `pick_column` -- three copies

- `src/process/common.py:87` -- `pick_existing_column()` (handles nested sequences)
- `src/plots/common.py:132` -- `pick_existing_column()` (simpler, flat list only)
- `src/fit_helper.py:170` -- `pick_column()` (yet another variant)

All do the same thing: find the first column name from a candidate list that exists in a DataFrame. Three implementations means three places bugs can hide.

---

## 4. Copy-pasted `_build_selector_groups` fallback

The same 10-line fallback `_build_selector_groups` implementation is copy-pasted inside `try/except ImportError` blocks in three adapters:

- `src/process/two_adc.py:37`
- `src/process/nuo_auditory.py:27`
- `src/process/MCDR.py:26`

This was a compatibility shim for an older `glmhmmt` version. If you're on a version that has it, all three dead-code copies are never used. If you're on an old version, fixing a bug in the fallback means editing three files.

**Fix:** Either remove the `try/except` (if the current `glmhmmt` always exports it), or put the fallback in one shared helper.

---

## 5. `PROJECT_ROOT` discovery is repeated in every entry point

The same 7-line `PROJECT_ROOT = next(...)` pattern for finding the project root by walking parents looking for `config.toml + src/` appears in:

- `src/fit_helper.py:22`
- `src/fit_and_plot.py:9`
- `figures/autocorrelograms.py:35`
- `notebooks/glmhmm.py:13`

Plus each entry point does its own `sys.path.insert(0, ...)`. If the discovery heuristic breaks (e.g., running from a different working directory), it breaks differently per file.

**Fix:** Put it in `src/__init__.py` or a `src/_paths.py` module.

---

## 6. Task name strings are fragile implicit contracts

Task routing relies on bare string matching scattered throughout the code:

- `fit_helper.py:74`: `if task_name == "2AFC_DRUG":`
- `fit_helper.py:76`: `if task_name in {"2ADC_DRUG", "2AFC_delay_DRUG"}:`
- `fit_helper.py:80`: `if task_name == "MCDR":`
- `process/common.py:3766`: `raise ValueError("...implemented only for 2AFC and delay tasks.")`

The task adapter system uses registered keys, but the glue code bypasses it with hardcoded string checks. Adding a new task means hunting for these scattered `if task_name ==` checks.

**What to watch out for:** The `2ADC` task has **two** registered aliases (`"2ADC"` and `"2AFC_delay"`), plus the drug variant has two more (`"2ADC_DRUG"`, `"2AFC_delay_DRUG"`). If you use the wrong alias in a string check, things silently fall through.

---

## 7. `fit_helper.prepare_predictions_df` is a fragile dispatcher

```python
def prepare_predictions_df(task_name: str, df: pl.DataFrame, adapter) -> pd.DataFrame:
    adapter_module = sys.modules.get(adapter.__module__)
    prepare_fn = getattr(adapter_module, "prepare_predictions_df", None)
    if prepare_fn is None and task_name == "2AFC_DRUG":
        prepare_fn = importlib.import_module("src.process.two_afc").prepare_predictions_df
    if prepare_fn is None and task_name in {"2ADC_DRUG", "2AFC_delay_DRUG"}:
        prepare_fn = importlib.import_module("src.process.two_adc").prepare_predictions_df
    ...
    if task_name == "MCDR":
        return prepare_fn(df, cfg=load_app_config())
    return prepare_fn(df)
```

It uses `sys.modules` introspection, then falls back to hardcoded module names for drug tasks, then has a special calling convention for MCDR (extra `cfg` argument). This breaks silently if a module isn't imported yet or if a new task doesn't fit the pattern.

**Fix:** Make `prepare_predictions_df` part of the adapter interface.

---

## 8. Hardcoded paths in figure scripts

Despite `config.toml` defining `data_dir`, several figure scripts hardcode:
```python
data_path = Path(__file__).parents[1] / "data/processed"
```
(`figure1.py:67`, `figure2.py:146`, `licks_analysis.py:94`, `models_cartoon.py:51`)

If `data_dir` in `config.toml` ever changes, these won't follow.

---

## 9. The autocorrelogram simulation block (lines ~4300-6228)

This is ~1900 lines of tightly coupled simulation code living inside `common.py`. It includes:
- Closed-loop trial-by-trial simulation with state transitions
- Multiple choice-history value inference strategies
- Manual softmax + lapse application
- 3 separate `prepare_*_autocorrelograms` entry points with overlapping logic

This is the most complex code in the repo and the hardest to debug. A single bug in `closed_loop_autocorrelogram_x` (which reconstructs design matrix rows mid-simulation) silently corrupts all downstream analyses.

**Be careful:** These functions reach into fitted arrays by string column name (`"choice_lag_param"`, `"at_choice_param"`, `"prev_choice"`) and reconstruct model inputs manually. If the adapter changes its column naming, this code doesn't fail -- it just produces wrong numbers.

---

## 10. `src/plots/common.py` is also big (3077 lines) and mixes concerns

It has its own copy of `fig_size`, its own `pick_existing_column`, plotting functions, data transformation functions (`build_session_trial_outcomes_data`, `build_repetition_variance_by_drug_task`), and pure data summarization (`two_afc_repeat_alternate_trials`, `animal_chunk_histogram`). Data prep shouldn't live in a plotting module.

---

## Priority ranking for refactoring

| Priority | Item | Risk if untouched | Effort |
|---|---|---|---|
| 1 | Split `process/common.py` | High -- any edit is a minefield | Medium |
| 2 | Unify `fig_size` | Medium -- silent dimension mismatch | Low |
| 3 | Make `prepare_predictions_df` part of adapter interface | Medium -- breaks on new tasks | Low |
| 4 | Remove duplicated `pick_existing_column` / `_build_selector_groups` | Low -- just confusing | Low |
| 5 | Centralize `PROJECT_ROOT` | Low -- annoying, rarely breaks | Low |
| 6 | Extract autocorrelogram subsystem | High -- hardest to debug | High |

The biggest thing to be careful about day-to-day: **column name strings are the hidden API** of this codebase. Adapters define them, `common.py` consumes them by name, and there's no type checking or validation connecting the two. Renaming a regressor column in an adapter will silently break downstream summaries and simulations without any error.