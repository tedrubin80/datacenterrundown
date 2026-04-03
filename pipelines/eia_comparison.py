#!/usr/bin/env python3
"""Compare paper TCO vs EIA-calibrated power costs."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
from src.data.location_profiles import load_locations
from src.tco.monte_carlo import run_all_locations

paper = load_locations(eia_calibrated=False)
calibrated = load_locations(eia_calibrated=True)

print("=" * 75)
print("EIA-CALIBRATED TCO vs PAPER ASSUMPTIONS (25yr, 5K sims)")
print("=" * 75)

paper_results = run_all_locations(paper, n_simulations=5000, horizon_years=25, seed=42)
cal_results = run_all_locations(calibrated, n_simulations=5000, horizon_years=25, seed=42)

print(f"\n  {'Location':<30} {'Paper TCO':>10} {'EIA TCO':>10} {'Delta':>10} {'Delta %':>8}")
print("  " + "-" * 70)

for key in sorted(paper_results, key=lambda k: paper_results[k].mean):
    p = paper_results[key].mean
    c = cal_results[key].mean
    delta = c - p
    pct = delta / p * 100
    marker = " *" if abs(pct) > 1 else ""
    print(f"  {paper[key].name:<30} {p:>9,.0f}M {c:>9,.0f}M {delta:>+9,.0f}M {pct:>+7.1f}%{marker}")

print("\n* = EIA-calibrated rate differs from paper assumption")
