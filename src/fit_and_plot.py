from __future__ import annotations
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = next(
    (
        p
        for base in (Path.cwd(), Path(__file__).resolve())
        for p in (base, *base.parents)
        if (p / "config.toml").exists() and (p / "src").exists()
    ),
    Path.cwd(),
)
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glmhmmt.runtime import configure_paths, get_runtime_paths  # noqa: E402
from fit_helper import save_model_plots  # noqa: E402


# Saved drug/frozen fits to plot. Entries are (task, model family, model id).
# Keep missing experimental aliases commented out; active entries should be
# expected to exist under results/fits/<task>/<model_kind>/<model_id>.
FITS = [
    ("2AFC_DRUG", "glmhmm", "drug_param2"),
    ("2AFC_DRUG", "glmhmm", "saline_param2"),
    ("2AFC_DRUG", "glmhmm", "drug_param2_pure"),
    ("2AFC_DRUG", "glmhmm", "saline_param2_pure"),
    ("2AFC_DRUG", "glmhmmt", "base_model"),
    ("2AFC_DRUG", "glmhmmt", "base_model_pure"),
    # ("2ADC_DRUG", "glmhmm", "param_drug2_pure"),
    # ("2ADC_DRUG", "glmhmm", "param_saline2_pure"),
    # ("2ADC_DRUG", "glmhmmt", "base_param"),
    # ("2ADC_DRUG", "glmhmmt", "base_param pure"),
    # ("MCDR", "glmhmm", "drug_param"),
    # ("MCDR", "glmhmm", "drug_param_frozen"),
    # ("MCDR", "glmhmm", "param frozen"),
    # ("MCDR", "glmhmm", "param frozen_saline"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save plots for the selected drug/frozen GLM-HMM and GLM-HMM-T fits.",
    )
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "config.toml"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--fit-results-dir",
        default=None,
        help="Optional results directory to load saved fits from. Defaults to --results-dir/config results.",
    )
    parser.add_argument("--fit", action="store_true", help="Rerun each fit before plotting.")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default=None)
    parser.add_argument("--dpi", type=int, default=300)  # This is not the place for this, remove
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--scoring-key", default=None)
    parser.add_argument("--regressor", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        config_path=args.config_path,
    )
    paths = get_runtime_paths()
    fit_results_dir = (
        Path(args.fit_results_dir).expanduser().resolve()
        if args.fit_results_dir is not None
        else paths.RESULTS
    )
    plot_args = SimpleNamespace(
        fit=args.fit,
        K=args.K,
        subjects=args.subjects,
        scoring_key=args.scoring_key,
        regressor=args.regressor,
        format=args.format,
        dpi=args.dpi,
        num_iters=args.num_iters,
        n_restarts=args.n_restarts,
        seed=args.seed,
    )

    total_saved = 0
    skipped: list[str] = []
    failed: list[str] = []

    for task_name, model_kind, model_id in FITS:
        fit_dir = fit_results_dir / "fits" / task_name / model_kind / model_id
        label = f"{task_name}/{model_kind}/{model_id}"
        if not fit_dir.exists():
            print(f"SKIP missing fit: {label}")
            skipped.append(label)
            continue
        try:
            saved = save_model_plots(fit_dir, plot_args, model_kind=model_kind)
        except Exception as exc:
            print(f"FAIL {label}: {exc}")
            failed.append(label)
            continue
        total_saved += len(saved)
        print(f"OK {label}: saved {len(saved)} plot(s) to {saved[0].parent if saved else fit_dir}")

    print(f"Done. Saved {total_saved} plot(s). Skipped {len(skipped)}. Failed {len(failed)}.")
    if failed:
        print("Failed fits:")
        for label in failed:
            print(f"  {label}")


if __name__ == "__main__":
    main()
