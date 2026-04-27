"""One-command Colab runner for the irrigation Kaggle project.

Expected data files:
  data/train.csv
  data/test.csv
  data/sample_submission.csv

Example:
  python run_colab_pipeline.py
  python run_colab_pipeline.py --experiments lgbm_holdout,lgbm_scipy,blend_holdout
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "src" / "improved_pipeline.py"
LEGACY_PIPELINE = ROOT / "src" / "legacy_best_pipeline.py"

EXPERIMENTS = {
    "legacy_best": ["--legacy-best"],
    "histgbm_holdout": ["--holdout-only", "--model", "histgbm"],
    "lgbm_holdout": ["--holdout-only", "--model", "lgbm"],
    "lgbm_scipy": ["--holdout-only", "--model", "lgbm", "--use-scipy-threshold"],
    "lgbm_high_15": [
        "--holdout-only",
        "--model",
        "lgbm",
        "--high-weight-multiplier",
        "1.5",
    ],
    "blend_holdout": ["--holdout-only", "--model", "blend"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Colab Kaggle experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--experiments",
        default="legacy_best",
        help="Comma-separated experiment names. Available: "
        + ", ".join(EXPERIMENTS),
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the quick sample-row validation run.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=10000,
        help="Rows used for the smoke test.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run the smoke test and stop before full experiments.",
    )
    return parser.parse_args()


def check_inputs(data_dir: Path) -> None:
    missing = [
        name
        for name in ["train.csv", "test.csv", "sample_submission.csv"]
        if not (data_dir / name).exists()
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(
            f"Missing data files in {data_dir}: {missing_text}\n"
            "Copy the Kaggle CSVs into data/ before running."
        )

    if not PIPELINE.exists():
        raise SystemExit(f"Missing pipeline script: {PIPELINE}")
    if not LEGACY_PIPELINE.exists():
        raise SystemExit(f"Missing legacy pipeline script: {LEGACY_PIPELINE}")


def run_command(args: list[str]) -> None:
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def preserve_outputs(experiment: str, output_dir: Path) -> None:
    submissions = output_dir / "submissions"
    metrics = output_dir / "metrics"
    figures = output_dir / "figures"

    submission_src = submissions / "improved_submission.csv"
    if submission_src.exists():
        shutil.copy2(submission_src, submissions / f"{experiment}.csv")

    artifact_pairs = [
        (metrics / "improved_holdout_metrics.json", metrics / f"{experiment}_metrics.json"),
        (
            metrics / "improved_holdout_classification_report.csv",
            metrics / f"{experiment}_classification_report.csv",
        ),
        (
            metrics / "improved_holdout_confusion_matrix.csv",
            metrics / f"{experiment}_confusion_matrix.csv",
        ),
        (
            figures / "improved_holdout_confusion_matrix.png",
            figures / f"{experiment}_confusion_matrix.png",
        ),
    ]
    for src, dst in artifact_pairs:
        if src.exists():
            shutil.copy2(src, dst)


def preserve_legacy_outputs(output_dir: Path) -> None:
    submissions = output_dir / "submissions"

    submission_src = submissions / "legacy_best_submission.csv"
    if submission_src.exists():
        shutil.copy2(submission_src, submissions / "legacy_best.csv")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    check_inputs(data_dir)
    for folder in ["metrics", "figures", "submissions"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    if not args.skip_smoke_test:
        run_command(
            [
                sys.executable,
                str(PIPELINE),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--holdout-only",
                "--model",
                "lgbm",
                "--sample-rows",
                str(args.smoke_rows),
                "--skip-submission",
            ]
        )

    if args.smoke_only:
        print("\nSmoke test finished. No full experiments were run.")
        return

    selected = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(
            "Unknown experiments: "
            + ", ".join(unknown)
            + "\nAvailable: "
            + ", ".join(EXPERIMENTS)
        )

    for experiment in selected:
        if experiment == "legacy_best":
            run_command(
                [
                    sys.executable,
                    str(LEGACY_PIPELINE),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            preserve_legacy_outputs(output_dir)
        else:
            run_command(
                [
                    sys.executable,
                    str(PIPELINE),
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                    *EXPERIMENTS[experiment],
                ]
            )
            preserve_outputs(experiment, output_dir)

    print("\nFinished. Submit one of these files to Kaggle:")
    for csv_path in sorted((output_dir / "submissions").glob("*.csv")):
        print(f"  {csv_path}")


if __name__ == "__main__":
    main()
