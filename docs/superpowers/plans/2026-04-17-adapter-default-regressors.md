# Adapter Default Regressors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update 2AFC, 2AFC_delay, and MCDR task adapters so shared UIs preselect the requested collapsed regressor families and MCDR exposes side-grouped bias, stimulus, and choice-lag selectors.

**Architecture:** Keep the source of truth in the task adapters under `NMDAR_paper/src/process/`. Add regression coverage for selector grouping and `default_emission_cols(...)`, then update each adapter to expand the requested collapsed families into the concrete feature columns consumed by fits and widgets.

**Tech Stack:** Python, unittest, Polars, pandas, adapter-driven notebook/model-manager UIs

---

### Task 1: Regression tests for adapter defaults and grouping

**Files:**
- Modify: `NMDAR_paper/tests/test_two_afc_glm.py`
- Modify: `NMDAR_paper/tests/test_mcdr_glm.py`
- Create: `NMDAR_paper/tests/test_two_afc_delay_glm.py`
- Test: `NMDAR_paper/tests/test_two_afc_glm.py`
- Test: `NMDAR_paper/tests/test_mcdr_glm.py`
- Test: `NMDAR_paper/tests/test_two_afc_delay_glm.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_default_emission_cols_expand_requested_hot_families(self) -> None:
    adapter = TwoAFCAdapter()
    ...

def test_default_emission_cols_expand_requested_delay_and_choice_lag_families(self) -> None:
    adapter = TwoAFCDelayAdapter()
    ...

def test_build_emission_groups_exposes_side_grouped_bias_and_choice_lag_families(self) -> None:
    groups = mcdr._build_emission_groups([...])
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest NMDAR_paper/tests/test_two_afc_glm.py NMDAR_paper/tests/test_two_afc_delay_glm.py NMDAR_paper/tests/test_mcdr_glm.py -v`
Expected: FAIL because the current adapters still return the old defaults and MCDR does not expose the requested collapsed families.

- [ ] **Step 3: Write minimal implementation**

```python
# Update default_emission_cols/build_emission_groups in:
# - NMDAR_paper/src/process/two_afc.py
# - NMDAR_paper/src/process/two_afc_delay.py
# - NMDAR_paper/src/process/MCDR.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest NMDAR_paper/tests/test_two_afc_glm.py NMDAR_paper/tests/test_two_afc_delay_glm.py NMDAR_paper/tests/test_mcdr_glm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C NMDAR_paper add \
  docs/superpowers/plans/2026-04-17-adapter-default-regressors.md \
  tests/test_two_afc_glm.py \
  tests/test_two_afc_delay_glm.py \
  tests/test_mcdr_glm.py \
  src/process/two_afc.py \
  src/process/two_afc_delay.py \
  src/process/MCDR.py
git -C NMDAR_paper commit -m "feat: update adapter default regressor families"
```
