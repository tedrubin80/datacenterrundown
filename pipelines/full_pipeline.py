#!/usr/bin/env python3
"""Full end-to-end pipeline: data generation -> training -> analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from pipelines.generate_data import main as generate_data
from pipelines.train_idea3 import main as train_idea3
from pipelines.train_idea5 import main as train_idea5


def main(seed: int = 42, rows: int = 10000):
    print("\n" + "#" * 60)
    print("# FULL PIPELINE: Climate-Driven Data Center TCO Analysis")
    print("#" * 60)

    print("\n\n>>> PHASE 1: DATA GENERATION <<<\n")
    generate_data(rows=rows, seed=seed)

    print("\n\n>>> PHASE 2: IDEA 3 - DYNAMIC TCO <<<\n")
    train_idea3(seed=seed)

    print("\n\n>>> PHASE 3: IDEA 5 - EXTREME WEATHER & INSURANCE <<<\n")
    train_idea5(seed=seed)

    print("\n" + "#" * 60)
    print("# ALL PIPELINES COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=10000)
    args = parser.parse_args()
    main(seed=args.seed, rows=args.rows)
