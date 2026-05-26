# NMDAR paper

Code is **versioned**. Data is **referenced**. Results are **reproducible**.

## Install

Create or refresh the environment with `uv`, then reinstall `tfp-nightly` last.
Both `tensorflow-probability` and `tfp-nightly` provide the same
`tensorflow_probability` import package, so the nightly files must be installed
on top of the stable package.

```bash
uv sync
uv pip install --reinstall --no-deps tfp-nightly==0.26.0.dev20260205
```

After reinstalling, restart the marimo kernel and verify that the TensorFlow
Probability JAX bijector can be constructed:

```bash
uv run python - <<'PY'
from tensorflow_probability.substrates import jax as tfp

tfp.bijectors.SoftmaxCentered()
print("TFP JAX substrate OK")
PY
```
