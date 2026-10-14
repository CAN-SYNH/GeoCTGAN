from __future__ import annotations

import argparse
from pathlib import Path

from geoctgan.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic data evaluation")
    parser.add_argument("generation_run")
    parser.add_argument("--output-parent")
    parser.add_argument("--evaluation-dir")
    parser.add_argument("--da-boundaries")
    parser.add_argument("--marginalization-csv")
    parser.add_argument("--benchmark-manifest")
    args = parser.parse_args()
    output_parent = args.output_parent
    if output_parent is None and args.evaluation_dir is not None:
        output_parent = Path(args.evaluation_dir).parent
    if output_parent is None:
        parser.error("--output-parent or --evaluation-dir is required")
    run_evaluation(
        args.generation_run,
        output_parent=output_parent,
        evaluation_dir=args.evaluation_dir,
        da_boundaries=args.da_boundaries,
        marginalization_csv=args.marginalization_csv,
        benchmark_manifest=args.benchmark_manifest,
    )


if __name__ == "__main__":
    main()
