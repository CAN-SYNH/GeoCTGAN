#!/usr/bin/env python3
from __future__ import annotations

import argparse

from geoctgan import run_sensitivity_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-only configuration sensitivity")
    parser.add_argument("config")
    parser.add_argument(
        "--run-dir",
        help="Output directory",
    )
    args = parser.parse_args()
    run_sensitivity_pipeline(args.config, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
