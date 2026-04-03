#!/usr/bin/env python3
"""Pipeline: Combined Facility + Hardware TCO under climate scenarios.

Integrates:
- DCCore facility TCO (static + dynamic MC)
- PC Hardware Cost Addendum (4 tiers × 10 locations)
- Climate-adjusted hardware energy costs under RCP scenarios
- Apple ARM efficiency tier comparison
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

from src.data.location_profiles import load_locations
from src.data.climate_projections import generate_all_projections
from src.tco.monte_carlo import run_all_locations
from src.tco.dynamic_distributions import run_dynamic_mc
from src.tco.hardware_costs import (
    HARDWARE_TIERS,
    compute_hw_tco,
    compute_climate_adjusted_hw_tco,
    all_locations_all_tiers,
)
from src.risk.metrics import compute_risk_metrics

RESULTS_DIR = Path(__file__).parents[1] / "data" / "results"


def main(seed: int = 42):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMBINED TCO: FACILITY + HARDWARE UNDER CLIMATE SCENARIOS")
    print("=" * 70)

    locations = load_locations()
    projections = generate_all_projections(seed=seed)

    # ----------------------------------------------------------------
    # Step 1: Static hardware TCO (all tiers × all locations)
    # ----------------------------------------------------------------
    print("\n[1/5] Computing static hardware TCO (4 tiers × 10 locations)...")
    static_hw = all_locations_all_tiers(locations, n_racks=500, horizon_years=10)
    hw_df = pd.DataFrame(static_hw)
    hw_df.to_csv(RESULTS_DIR / "hardware_tco_all_tiers.csv", index=False)

    # Print summary for Standard AI tier
    print("\n  Standard AI Rack (H100) — 500 racks, 10yr:")
    ai_df = hw_df[hw_df["tier_key"] == "standard_ai"].sort_values("hw_tco_10yr_m")
    for _, row in ai_df.iterrows():
        print(f"    {row['location']:<30} HW={row['hw_tco_10yr_m']:>10,.1f}M  "
              f"Energy={row['hw_energy_10yr_m']:>8,.1f}M  "
              f"Combined={row['combined_tco_m']:>10,.1f}M")

    # ARM comparison
    print("\n  ARM Efficiency vs Standard AI — energy savings:")
    for loc_key, loc in sorted(locations.items(), key=lambda x: x[1].power_cost_mwh[1]):
        ai = hw_df[(hw_df["location_key"] == loc_key) & (hw_df["tier_key"] == "standard_ai")]
        arm = hw_df[(hw_df["location_key"] == loc_key) & (hw_df["tier_key"] == "arm_efficiency")]
        if len(ai) > 0 and len(arm) > 0:
            savings = ai.iloc[0]["hw_energy_10yr_m"] - arm.iloc[0]["hw_energy_10yr_m"]
            pct = savings / ai.iloc[0]["hw_energy_10yr_m"] * 100
            print(f"    {loc.name:<30} AI Energy={ai.iloc[0]['hw_energy_10yr_m']:>7.1f}M  "
                  f"ARM Energy={arm.iloc[0]['hw_energy_10yr_m']:>7.1f}M  "
                  f"Saved={savings:>7.1f}M ({pct:.0f}%)")

    # ----------------------------------------------------------------
    # Step 2: Static facility TCO baseline
    # ----------------------------------------------------------------
    print("\n[2/5] Running facility Monte Carlo (5K sims × 10 locations)...")
    facility_results = run_all_locations(locations, n_simulations=5000, horizon_years=10, seed=seed)

    # ----------------------------------------------------------------
    # Step 3: Climate-adjusted hardware TCO under RCP scenarios
    # ----------------------------------------------------------------
    print("\n[3/5] Computing climate-adjusted hardware TCO (3 scenarios × 10 locations)...")

    climate_hw_results = []
    for scenario in ["rcp26", "rcp45", "rcp85"]:
        for loc_key, loc in locations.items():
            loc_proj = projections[
                (projections["location_key"] == loc_key) &
                (projections["scenario"] == scenario)
            ].sort_values("year")

            if len(loc_proj) == 0:
                continue

            # Build year lookup
            climate_by_year = {}
            for _, row in loc_proj.iterrows():
                climate_by_year[int(row["year"])] = {
                    "projected_pue": row.get("projected_pue", loc.pue[1]),
                    "power_price_delta_pct": row.get("power_price_delta_pct", 0),
                }

            for tier_key, tier in HARDWARE_TIERS.items():
                hw = compute_climate_adjusted_hw_tco(
                    tier, loc, climate_by_year,
                    n_racks=500, horizon_years=15, start_year=2025,
                )

                # Also get facility dynamic TCO
                facility_dynamic = run_dynamic_mc(
                    loc, loc_proj, scenario=scenario,
                    n_simulations=2000, horizon_years=15, seed=seed,
                )

                total_energy = sum(y["energy"] for y in hw["yearly"])
                climate_hw_results.append({
                    "location": loc.name,
                    "location_key": loc_key,
                    "scenario": scenario,
                    "tier": tier.name,
                    "tier_key": tier_key,
                    "hw_tco_15yr_m": round(hw["total_hw_tco_m"], 1),
                    "hw_energy_15yr_m": round(total_energy, 1),
                    "facility_tco_15yr_m": round(facility_dynamic.mean, 1),
                    "combined_tco_15yr_m": round(hw["total_hw_tco_m"] + facility_dynamic.mean, 1),
                })

    climate_hw_df = pd.DataFrame(climate_hw_results)
    climate_hw_df.to_csv(RESULTS_DIR / "combined_tco_climate_adjusted.csv", index=False)

    # ----------------------------------------------------------------
    # Step 4: Summary comparisons
    # ----------------------------------------------------------------
    print("\n[4/5] Summary: Combined TCO (Facility + Hardware, 15yr, Standard AI)...")

    ai_climate = climate_hw_df[climate_hw_df["tier_key"] == "standard_ai"]
    print(f"\n  {'Location':<30} {'RCP2.6':>10} {'RCP4.5':>10} {'RCP8.5':>10} {'Gap':>10}")
    print("  " + "-" * 72)

    for loc_key in sorted(locations.keys(), key=lambda k: locations[k].power_cost_mwh[1]):
        loc = locations[loc_key]
        vals = {}
        for scenario in ["rcp26", "rcp45", "rcp85"]:
            row = ai_climate[
                (ai_climate["location_key"] == loc_key) &
                (ai_climate["scenario"] == scenario)
            ]
            if len(row) > 0:
                vals[scenario] = row.iloc[0]["combined_tco_15yr_m"]

        if len(vals) == 3:
            gap = vals["rcp85"] - vals["rcp26"]
            print(f"  {loc.name:<30} {vals['rcp26']:>9,.0f}M {vals['rcp45']:>9,.0f}M "
                  f"{vals['rcp85']:>9,.0f}M {gap:>+9,.0f}M")

    # ARM climate advantage
    print("\n  ARM Efficiency Climate Advantage (15yr, RCP 8.5 vs Standard AI):")
    print(f"  {'Location':<30} {'AI Combined':>12} {'ARM Combined':>12} {'Savings':>10}")
    print("  " + "-" * 66)

    for loc_key in sorted(locations.keys(), key=lambda k: locations[k].power_cost_mwh[1]):
        loc = locations[loc_key]
        ai_row = climate_hw_df[
            (climate_hw_df["location_key"] == loc_key) &
            (climate_hw_df["scenario"] == "rcp85") &
            (climate_hw_df["tier_key"] == "standard_ai")
        ]
        arm_row = climate_hw_df[
            (climate_hw_df["location_key"] == loc_key) &
            (climate_hw_df["scenario"] == "rcp85") &
            (climate_hw_df["tier_key"] == "arm_efficiency")
        ]
        if len(ai_row) > 0 and len(arm_row) > 0:
            ai_total = ai_row.iloc[0]["combined_tco_15yr_m"]
            arm_total = arm_row.iloc[0]["combined_tco_15yr_m"]
            savings = ai_total - arm_total
            print(f"  {loc.name:<30} {ai_total:>11,.0f}M {arm_total:>11,.0f}M {savings:>+9,.0f}M")

    # ----------------------------------------------------------------
    # Step 5: Climate impact on hardware energy alone
    # ----------------------------------------------------------------
    print("\n[5/5] Climate impact on hardware energy (Standard AI, 15yr)...")
    print(f"\n  {'Location':<30} {'Static':>10} {'RCP2.6':>10} {'RCP8.5':>10} {'Climate Δ':>10}")
    print("  " + "-" * 62)

    for loc_key in sorted(locations.keys(), key=lambda k: locations[k].power_cost_mwh[1]):
        loc = locations[loc_key]
        # Static hardware energy
        static_hw_result = compute_hw_tco(
            HARDWARE_TIERS["standard_ai"], loc, n_racks=500, horizon_years=15
        )
        static_energy = static_hw_result["total_energy_m"]

        r26 = ai_climate[
            (ai_climate["location_key"] == loc_key) & (ai_climate["scenario"] == "rcp26")
        ]
        r85 = ai_climate[
            (ai_climate["location_key"] == loc_key) & (ai_climate["scenario"] == "rcp85")
        ]

        if len(r26) > 0 and len(r85) > 0:
            e26 = r26.iloc[0]["hw_energy_15yr_m"]
            e85 = r85.iloc[0]["hw_energy_15yr_m"]
            delta = e85 - e26
            print(f"  {loc.name:<30} {static_energy:>9,.1f}M {e26:>9,.1f}M "
                  f"{e85:>9,.1f}M {delta:>+9,.1f}M")

    print("\n" + "=" * 70)
    print("COMBINED TCO PIPELINE COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
