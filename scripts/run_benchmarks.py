#!/usr/bin/env python3
from __future__ import annotations

import argparse

from geoctgan.benchmarks import run_benchmark_generation
from geoctgan.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark analyses")
    parser.add_argument("config")
    parser.add_argument("train_parquet")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    run_benchmark_generation(cfg, args.train_parquet, args.output_dir)


if __name__ == "__main__":
    main()
