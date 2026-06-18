# GUI Architecture

## How the GUI works

Everything runs as **marimo apps** -- Python files with `marimo.App()` where each `@app.cell` is a reactive cell. When a UI element's value changes, marimo automatically re-runs all downstream cells that depend on it. No custom event wiring needed.

---

## Native marimo building blocks

### Layout
- `mo.vstack()` / `mo.hstack()` -- stacking panels vertically/horizontally (used everywhere for arranging plots, controls, and save buttons)
- `mo.md()` -- markdown section headers and explanatory text

### Controls (all from `mo.ui.*`)
- `mo.ui.dropdown()` -- model selection, subject/session pickers, regressor pickers, scoring keys
- `mo.ui.checkbox()` -- toggle plot options (fit lapse logistic, show weighted dots, data smooth)
- `mo.ui.slider()` / `mo.ui.range_slider()` -- number of quantiles, simulation counts, lag ranges
- `mo.ui.radio()` -- psychometric background mode, state assignment mode, model line mode
- `mo.ui.multiselect()` -- subject filtering, model directory selection
- `mo.ui.run_button()` -- trigger expensive operations (run fit, run simulations, pick random session)
- `mo.ui.number()` -- seed values
- `mo.ui.dataframe()` -- display DataFrames as interactive tables
- `mo.ui.table()` -- display fit parameter tables
- `mo.ui.matplotlib()` -- wrap a matplotlib axis as interactive (used for emission summary with `debounce=True`)
- `mo.ui.dictionary()` -- group multiple dropdowns into a single reactive dict (model comparison picks)

### State management
- `mo.state()` -- explicit get/set state pairs for things like `last_fit_click` counter and `session_pick` (subject+session selection that persists across random-session button clicks)
- `mo.stop()` -- conditional cell halt with a fallback display (e.g., "No fitted arrays found -- run the fit first")

### Progress
- `mo.status.progress_bar()` -- shown during model fitting

---

## Custom / third-party widgets (3 total)

All three come from the `glmhmmt` package (or `wigglystuff`) and are bridged into marimo via `mo.ui.anywidget()`.

### 1. `ModelManagerWidget` (~900 lines of inline JS/CSS)

A custom panel for model configuration with:
- Task dropdown
- "New Fit" / "Load Existing" tabs
- Clickable chip grids for subjects, emission regressors, and transition regressors
- K slider, tau slider, lapse checkbox + max slider
- A scrollable table of saved fits (click to load)
- Custom alias text input + "RUN FIT" button
- Dark mode support

Used in `glmhmm.py`, `glmhmmt_analysis.py`, and `glm.py`. Its value is parsed into a `ModelCfg` dataclass via `ModelCfg.from_value()`.

The Python-side `_update_options()` method scans `results/fits/` for saved configs and queries the task adapter for available column options.

### 2. `CoefficientEditorWidget` (external JS/CSS assets)

An audio-equalizer-style UI for interactively editing emission weight coefficients:
- Vertical sliders per feature per choice channel (Left/Right)
- Per-feature live numeric readout while sliding
- Multi-channel layout with features side by side
- Smooth drag interaction (prevents re-render during drag)

Used in `glmhmmt_analysis.py` to tweak individual GLM weights per state/feature/side and see the psychometric curve update live.

There is also an older `CoefTweakerWidget` in `widgets.py` with similar vertical-slider functionality inlined as JS.

### 3. `TangleSlider` (from `wigglystuff`)

An inline draggable number rendered within text. Used in `glmhmm.py` for a posterior threshold control. Wrapped via `wrap_anywidget()`.

---

## `plot_saver` (third-party PyPI package, v0.1.7)

`make_plot_saver()` creates save buttons that appear next to each figure. Call `save_plot(fig, label, stem=...)` and it renders a marimo button. When clicked, it saves the figure to `results/` in the format specified by `config.toml`.

Used in every notebook and figure script -- the primary way plots get exported to disk.

---

## Can the custom widgets be replaced with native marimo?

### `ModelManagerWidget` -- Yes, fully replaceable

Native marimo equivalents:
- `mo.ui.dropdown` for task
- `mo.ui.tabs` for New/Load
- `mo.ui.multiselect` for subjects, emission cols, transition cols
- `mo.ui.slider` for K, tau, lapse_max
- `mo.ui.checkbox` for lapse toggle
- `mo.ui.dataframe` or `mo.ui.table` for saved fits
- `mo.ui.text` for alias, `mo.ui.run_button` for the fit trigger

The Python-side logic (scanning `results/fits/`, querying adapters for columns) would move into regular marimo cells.

**What you lose:** The chip-grid UI is nicer than a multiselect for quickly toggling 15+ regressors -- you can see all options at a glance and toggle individually. The whole thing is also a single cohesive panel rather than scattered widgets. But functionally, native marimo covers it completely.

### `CoefficientEditorWidget` -- No clean replacement

Marimo doesn't have vertical sliders (`mo.ui.slider` is horizontal only). You could fake it with one horizontal slider per feature per channel, but:
- You'd need ~20-30 individual sliders arranged in a grid
- You lose the equalizer visual metaphor that makes it intuitive to compare weights across features
- `mo.ui.slider` triggers a full cell re-run on each change, while the anywidget batches updates locally in the browser until mouseup

This one genuinely benefits from being a custom widget.

### `TangleSlider` -- Trivially replaceable

A `mo.ui.slider` or `mo.ui.number` does the same thing. The only difference is that `TangleSlider` renders as an underlined number you drag within a sentence -- a stylistic choice, not functional.

### Summary

| Widget | Replaceable? | Worth replacing? |
|---|---|---|
| `ModelManagerWidget` | Yes, fully | Maybe -- lose chip grid but gain simplicity and no JS maintenance |
| `CoefficientEditorWidget` | Not cleanly | No -- vertical slider grid with smooth drag has no native equivalent |
| `TangleSlider` | Yes, trivially | Yes -- `mo.ui.slider` does the same thing |