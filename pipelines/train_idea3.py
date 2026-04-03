#!/usr/bin/env python3
"""Pipeline: Train Idea 3 models (Dynamic TCO under climate scenarios).

Steps:
1. Load climate projections
2. Train ensemble model (XGBoost + RF)
3. Optionally train Bayesian NN
4. Run dynamic Monte Carlo for all locations x scenarios
5. Compare with static baseline
6. Generate visualizations
"""

import sys
import os
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.climate_projections import generate_all_projections
from src.data.location_profiles import load_locations
from src.tco.monte_carlo import run_all_locations
from src.tco.dynamic_distributions import run_dynamic_mc
from src.models.idea3.trainer import train_ensemble, train_bayesian
from src.risk.metrics import compute_risk_metrics
from src.risk.scenario_comparator import (
    build_comparison_table,
    rank_locations,
    climate_cost_premium_table,
)
from src.visualization.tco_plots import (
    plot_tco_distributions,
    plot_static_vs_dynamic,
    plot_tco_trajectory,
)
from src.visualization.risk_heatmaps import plot_risk_heatmap, plot_ranking_changes
from src.visualization.climate_plots import plot_temperature_pathways, plot_pue_degradation

RESULTS_DIR = Path(__file__).parents[1] / "data" / "results"


def main(seed: int = 42, use_bnn: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IDEA 3: DYNAMIC TCO UNDER CLIMATE SCENARIOS")
    print("=" * 60)

    # Step 1: Generate climate projections
    print("\n[1/6] Generating climate projections...")
    projections = generate_all_projections(seed=seed)
    locations = load_locations()

    # Step 2: Run static Monte Carlo baseline (both horizons)
    print("\n[2/6] Running static Monte Carlo baseline (5,000 sims x 10 locations)...")
    static_results = run_all_locations(locations, n_simulations=5000, horizon_years=10, seed=seed)
    static_results_15 = run_all_locations(locations, n_simulations=5000, horizon_years=15, seed=seed)
    print("  10-year horizon:")
    for key, result in static_results.items():
        print(f"    {key}: mean={result.mean:.1f}M, CV={result.cv:.4f}")
    print("  15-year horizon:")
    for key, result in static_results_15.items():
        print(f"    {key}: mean={result.mean:.1f}M, CV={result.cv:.4f}")

    # Step 3: Physics-based climate shifts (no ML training needed)
    print("\n[3/6] Using physics-based climate shift model...")
    print("  PUE shift: ~0.004 per °C x humidity factor (Depoorter et al.)")
    print("  Power cost: scaled by scenario power_price_delta_pct")
    print("  Insurance: event_ratio^1.5 nonlinear scaling")

    # Step 4: Run dynamic Monte Carlo for all locations x scenarios x horizons
    for horizon in [10, 15]:
        print(f"\n[4/6] Running dynamic Monte Carlo ({horizon}-year horizon, 3 scenarios x 10 locations)...")
        dynamic_results = {}

        for loc_key, loc in locations.items():
            dynamic_results[loc_key] = {}
            for scenario in ["rcp26", "rcp45", "rcp85"]:
                loc_proj = projections[
                    (projections["location_key"] == loc_key) &
                    (projections["scenario"] == scenario)
                ]
                if len(loc_proj) == 0:
                    continue

                result = run_dynamic_mc(
                    location=loc,
                    climate_projections=loc_proj,
                    scenario=scenario,
                    n_simulations=5000,
                    horizon_years=horizon,
                    seed=seed,
                )
                dynamic_results[loc_key][scenario] = result.tco_distribution
                print(f"  {loc_key}/{scenario}: mean={result.mean:.1f}M, CV={result.cv:.4f}")

        # Store for the primary analysis (use 15yr as the main result)
        if horizon == 15:
            dynamic_results_15 = dynamic_results
        else:
            dynamic_results_10 = dynamic_results

    # Use 15yr as primary for analysis
    dynamic_results = dynamic_results_15

    # Step 5: Analysis
    print("\n[5/6] Computing risk metrics and comparisons...")

    # Build comparison table
    all_dists = {}
    for loc_key in locations:
        all_dists[loc_key] = {"static": static_results[loc_key].tco_distribution}
        all_dists[loc_key].update(dynamic_results.get(loc_key, {}))

    comparison = build_comparison_table(all_dists)
    comparison.to_csv(RESULTS_DIR / "idea3_comparison_table.csv", index=False)
    print(f"  Comparison table: {len(comparison)} rows")

    # Climate cost premium
    static_dists = {k: v.tco_distribution for k, v in static_results.items()}
    premium_table = climate_cost_premium_table(static_dists, dynamic_results)
    premium_table.to_csv(RESULTS_DIR / "idea3_climate_premium.csv", index=False)
    print("  Climate premium table saved")

    # Ranking analysis
    ranked = rank_locations(comparison)
    ranked.to_csv(RESULTS_DIR / "idea3_rankings.csv", index=False)

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")

    # Static TCO distributions
    static_dists_plot = {loc.name: static_results[k].tco_distribution
                         for k, loc in locations.items()}
    plot_tco_distributions(static_dists_plot,
                           output_path=str(RESULTS_DIR / "static_tco_distributions.png"))

    # Static vs Dynamic for key locations
    for loc_key in ["boden_sweden", "atlanta_georgia", "evanston_wyoming"]:
        if loc_key in dynamic_results:
            plot_static_vs_dynamic(
                static_results[loc_key].tco_distribution,
                dynamic_results[loc_key],
                locations[loc_key].name,
                output_path=str(RESULTS_DIR / f"static_vs_dynamic_{loc_key}.png"),
            )

    # Risk heatmap
    plot_risk_heatmap(comparison, metric="cv",
                      output_path=str(RESULTS_DIR / "risk_heatmap_cv.png"))
    plot_risk_heatmap(comparison, metric="mean_tco",
                      output_path=str(RESULTS_DIR / "risk_heatmap_mean.png"))

    # Climate plots
    for loc_key in ["boden_sweden", "atlanta_georgia", "johor_malaysia"]:
        plot_temperature_pathways(projections, loc_key,
                                 output_path=str(RESULTS_DIR / f"temp_pathways_{loc_key}.png"))

    plot_pue_degradation(projections, scenario="rcp85",
                         output_path=str(RESULTS_DIR / "pue_degradation_rcp85.png"))

    print("\n" + "=" * 60)
    print("IDEA 3 PIPELINE COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bnn", action="store_true", help="Also train Bayesian NN")
    args = parser.parse_args()
    main(seed=args.seed, use_bnn=args.bnn)
