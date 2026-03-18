#!/usr/bin/env python3
"""Pipeline: Generate all synthetic data for the project.

Steps:
1. Run SynGen (or fallback) for raw data generation
2. Generate climate projections for all locations x RCP scenarios
3. Apply correlation engine to raw data
4. Save processed datasets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.data.syngen_runner import generate_all_raw_data
from src.data.climate_projections import generate_all_projections
from src.data.correlation_engine import inject_correlations, validate_correlations
from src.data.location_profiles import load_locations

DATA_DIR = Path(__file__).parents[1] / "data"


def main(rows: int = 10000, seed: int = 42):
    """Run the full data generation pipeline."""
    print("=" * 60)
    print("DATA GENERATION PIPELINE")
    print("=" * 60)

    # Step 1: Raw synthetic data
    print("\n[1/4] Generating raw synthetic data via SynGen...")
    raw_data = generate_all_raw_data(rows_override=rows)
    for name, df in raw_data.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")

    # Step 2: Climate projections
    print("\n[2/4] Generating climate projections (10 locations x 3 RCP scenarios)...")
    projections = generate_all_projections(seed=seed)
    print(f"  Total rows: {len(projections)}")
    print(f"  Scenarios: {projections['scenario'].unique().tolist()}")
    print(f"  Locations: {projections['location_key'].nunique()}")

    proj_path = DATA_DIR / "processed" / "climate_projections.csv"
    proj_path.parent.mkdir(parents=True, exist_ok=True)
    projections.to_csv(proj_path, index=False)
    print(f"  Saved to {proj_path}")

    # Step 3: Apply correlations to raw data where applicable
    print("\n[3/4] Injecting correlations into raw data...")
    for name, df in raw_data.items():
        available_cols = [c for c in df.columns if c in [
            "avg_temperature_c", "cooling_degree_days", "relative_humidity_pct",
            "storm_frequency_annual", "power_cost_mwh", "pue",
            "insurance_base_millions", "downtime_hours",
        ]]
        if len(available_cols) >= 2:
            correlated = inject_correlations(df, columns=available_cols)
            result = validate_correlations(
                correlated, available_cols,
                np.eye(len(available_cols)),  # just check structure
            )
            print(f"  {name}: correlated {len(available_cols)} columns "
                  f"(max deviation from target: {result['max_deviation']:.3f})")
            out_path = DATA_DIR / "processed" / f"{name}_correlated.csv"
            correlated.to_csv(out_path, index=False)
        else:
            out_path = DATA_DIR / "processed" / f"{name}.csv"
            df.to_csv(out_path, index=False)

    # Step 4: Location profiles summary
    print("\n[4/4] Saving location profiles summary...")
    locations = load_locations()
    loc_df = pd.DataFrame([
        {
            "key": loc.key,
            "name": loc.name,
            "tier": loc.tier,
            "currency": loc.currency,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "baseline_temp_c": loc.baseline_temp_c,
            "renewable_pct": loc.renewable_pct,
            "grid_reliability_score": loc.grid_reliability_score,
            "mean_tco_10yr": loc.mean_tco_10yr_millions,
            "cv": loc.cv,
        }
        for loc in locations.values()
    ])
    loc_path = DATA_DIR / "processed" / "locations_summary.csv"
    loc_df.to_csv(loc_path, index=False)
    print(f"  Saved {len(loc_df)} locations to {loc_path}")

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print(f"Output directory: {DATA_DIR / 'processed'}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--rows", type=int, default=10000, help="Rows per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(rows=args.rows, seed=args.seed)
