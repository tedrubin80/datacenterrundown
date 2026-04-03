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

    # Step 2: Run static Monte Carlo baseline (all horizons)
    print("\n[2/6] Running static Monte Carlo baseline (5,000 sims x 10 locations)...")
    static_by_horizon = {}
    for h in [10, 15, 25]:
        static_by_horizon[h] = run_all_locations(locations, n_simulations=5000, horizon_years=h, seed=seed)
        print(f"  {h}-year horizon:")
        for key, result in static_by_horizon[h].items():
            print(f"    {key}: mean={result.mean:.1f}M, CV={result.cv:.4f}")
    static_results = static_by_horizon[10]

    # Step 3: Physics-based climate shifts (no ML training needed)
    print("\n[3/6] Using physics-based climate shift model...")
    print("  PUE shift: ~0.004 per °C x humidity factor (Depoorter et al.)")
    print("  Power cost: scaled by scenario power_price_delta_pct")
    print("  Insurance: event_ratio^1.5 nonlinear scaling")

    # Step 4: Run dynamic Monte Carlo for all locations x scenarios x horizons
    dynamic_by_horizon = {}
    for horizon in [10, 15, 25]:
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

        dynamic_by_horizon[horizon] = dynamic_results

    # Use 25yr as primary for analysis (shows full compounding)
    dynamic_results = dynamic_by_horizon[25]
    static_results = static_by_horizon[25]

    # Print horizon comparison for key locations
    print("\n  === HORIZON COMPARISON (RCP 8.5 climate premium) ===")
    print(f"  {'Location':<30} {'10yr Gap':>10} {'15yr Gap':>10} {'25yr Gap':>10}")
    print("  " + "-" * 62)
    for loc_key in ["boden_sweden", "atlanta_georgia", "johor_malaysia", "evanston_wyoming"]:
        gaps = []
        for h in [10, 15, 25]:
            s = static_by_horizon[h][loc_key].mean
            d85 = np.mean(dynamic_by_horizon[h][loc_key].get("rcp85", [s]))
            d26 = np.mean(dynamic_by_horizon[h][loc_key].get("rcp26", [s]))
            gaps.append(d85 - d26)
        print(f"  {locations[loc_key].name:<30} {gaps[0]:>+9.1f}M {gaps[1]:>+9.1f}M {gaps[2]:>+9.1f}M")

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
