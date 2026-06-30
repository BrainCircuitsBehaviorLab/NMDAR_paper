from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


def infer_cpu_budget(default: int = 1) -> int:
    """Infer the CPU thread budget granted to this process."""
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or default)


def resolve_threads_per_worker(n_workers: int, *, total_threads: int | None = None) -> int:
    """Split the available CPU threads across model-level workers."""
    workers = max(1, int(n_workers))
    budget = max(1, int(total_threads) if total_threads is not None else infer_cpu_budget())
    return max(1, budget // workers)


def _configure_worker_threads(threads_per_worker: int | None) -> None:
    if threads_per_worker is None:
        return
    threads = max(1, int(threads_per_worker))
    thread_value = str(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = thread_value
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={thread_value}"
    )


def fit_transition_model_job(job: dict[str, Any]) -> dict[str, Any]:
    """Fit one complete GLM-HMM-T transition-regressor model.

    The function intentionally imports glmhmmt inside the worker so XLA sees
    the per-worker thread environment before JAX is initialized.
    """
    _configure_worker_threads(job.get("threads_per_worker"))

    project_root = Path(job["project_root"]).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from glmhmmt.cli.fit_glmhmmt import main as fit_main
    from glmhmmt.runtime import configure_paths, get_runtime_paths

    configure_paths(config_path=project_root / "config.toml")
    paths = get_runtime_paths()

    spec = dict(job["spec"])
    task_name = str(job["task_name"])
    cv_mode = str(job["cv_mode"])
    cv_repeats = 5 if cv_mode != "none" else 0
    subjects = [str(subject) for subject in job["subjects"]]
    out_dir = paths.RESULTS / "fits" / task_name / "glmhmmt" / str(spec["model_id"])

    fit_main(
        subjects=subjects,
        K_list=[int(job["K"])],
        num_iters=int(job["num_iters"]),
        n_restarts=1 if cv_mode != "none" else int(job["n_restarts"]),
        base_seed=int(job.get("base_seed", 0)),
        out_dir=out_dir,
        tau=float(job["tau"]),
        emission_cols=list(spec["emission_cols"]),
        transition_cols=list(spec["transition_cols"]),
        frozen_emissions=spec["frozen_emissions"] or None,
        task=task_name,
        cv_mode=cv_mode,
        cv_repeats=cv_repeats,
        verbose=False,
        baseline_class_idx=int(job["baseline_class_idx"]),
    )

    return {
        "model_id": str(spec["model_id"]),
        "out_dir": str(out_dir),
        "n_subjects": len(subjects),
        "threads_per_worker": job.get("threads_per_worker"),
    }
