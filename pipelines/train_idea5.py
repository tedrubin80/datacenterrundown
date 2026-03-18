#!/usr/bin/env python3
"""Pipeline: Train Idea 5 models (Extreme Weather & Insurance Impact).

Steps:
1. Load climate projections
2. Train survival model for time-to-outage
3. Train outage event classifier
4. Train insurance premium regressor
5. Generate predictions and analysis
6. Visualize results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.data.climate_projections import generate_all_projections
from src.data.location_profiles import load_locations
from src.models.idea5.trainer import (
    train_survival,
    train_classifier,
    train_insurance,
    _generate_event_data,
    _generate_insurance_targets,
)
from src.visualization.survival_plots import (
    plot_insurance_trajectories,
)

RESULTS_DIR = Path(__file__).parents[1] / "data" / "results"


def main(seed: int = 42):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IDEA 5: EXTREME WEATHER & INSURANCE IMPACT")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/6] Generating climate projections and event data...")
    projections = generate_all_projections(seed=seed)
    locations = load_locations()
    events_df = _generate_event_data(projections, seed=seed)
    print(f"  Climate projections: {len(projections)} rows")
    print(f"  Event records: {len(events_df)} rows")
    print(f"  Events with downtime: {(events_df['downtime_hours'] > 0).sum()}")

    # Step 2: Train survival model
    print("\n[2/6] Training survival model (Weibull AFT)...")
    survival_model, surv_metrics = train_survival(projections, seed=seed)
    print(f"  Concordance index: {surv_metrics.get('concordance', 'N/A')}")
    print(f"  Observations: {surv_metrics.get('n_observations', 0)}")
    print(f"  Events: {surv_metrics.get('n_events', 0)}")

    if survival_model.summary() is not None:
        summary = survival_model.summary()
        summary.to_csv(RESULTS_DIR / "idea5_survival_coefficients.csv")
        print("  Coefficient summary saved")

    # Step 3: Train outage classifier
    print("\n[3/6] Training outage event classifier...")
    classifier, cls_metrics = train_classifier(projections, seed=seed)
    if "error" not in cls_metrics:
        print(f"  Validation AUROC: {cls_metrics.get('val_auroc', 'N/A'):.4f}")
        print(f"  Validation AUPRC: {cls_metrics.get('val_auprc', 'N/A'):.4f}")

        # Feature importance
        fi = classifier.feature_importance()
        if fi:
            fi_df = pd.DataFrame(
                sorted(fi.items(), key=lambda x: -x[1]),
                columns=["feature", "importance"],
            )
            fi_df.to_csv(RESULTS_DIR / "idea5_classifier_feature_importance.csv", index=False)
            print("  Top features:")
            for _, row in fi_df.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
    else:
        print(f"  Skipped: {cls_metrics['error']}")

    # Step 4: Train insurance premium model
    print("\n[4/6] Training insurance premium regressor...")
    insurance_model, ins_metrics = train_insurance(projections, seed=seed)
    print(f"  Mean RMSE: {ins_metrics.get('mean_rmse', 'N/A')}")
    print(f"  Mean MAE: {ins_metrics.get('mean_mae', 'N/A')}")
    for key, val in ins_metrics.items():
        if key.startswith("q"):
            print(f"  {key}: {val:.4f}")

    # Step 5: Generate per-location predictions
    print("\n[5/6] Generating location-specific predictions...")

    insurance_results = []
    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days", "power_price_delta_pct",
    ]

    for loc_key, loc in locations.items():
        for scenario in ["rcp26", "rcp45", "rcp85"]:
            loc_proj = projections[
                (projections["location_key"] == loc_key) &
                (projections["scenario"] == scenario)
            ].sort_values("year")

            if len(loc_proj) == 0:
                continue

            # Pad missing feature cols with defaults
            ins_feature_cols = []
            for col in feature_cols:
                if col in loc_proj.columns:
                    ins_feature_cols.append(col)

            # Add dummy columns for event aggregates needed by insurance model
            for extra in ["n_events", "max_severity", "total_damage", "total_downtime"]:
                if extra not in loc_proj.columns:
                    loc_proj = loc_proj.copy()
                    loc_proj[extra] = 0
                ins_feature_cols.append(extra)

            X = loc_proj[ins_feature_cols].values
            premium_pred = insurance_model.predict(X)
            quantiles = insurance_model.predict_quantiles(X)

            for i, (_, row) in enumerate(loc_proj.iterrows()):
                rec = {
                    "location": loc_key,
                    "scenario": scenario,
                    "year": int(row["year"]),
                    "premium_predicted": round(premium_pred[i], 3),
                }
                for col in quantiles.columns:
                    rec[col] = round(quantiles[col].iloc[i], 3)
                insurance_results.append(rec)

    ins_results_df = pd.DataFrame(insurance_results)
    ins_results_df.to_csv(RESULTS_DIR / "idea5_insurance_predictions.csv", index=False)
    print(f"  Insurance predictions: {len(ins_results_df)} rows")

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")

    for loc_key in ["boden_sweden", "atlanta_georgia", "johor_malaysia"]:
        for scenario in ["rcp45", "rcp85"]:
            loc_ins = ins_results_df[
                (ins_results_df["location"] == loc_key) &
                (ins_results_df["scenario"] == scenario)
            ]
            if len(loc_ins) > 0:
                plot_insurance_trajectories(
                    loc_ins,
                    f"{locations[loc_key].name} ({scenario.upper()})",
                    output_path=str(RESULTS_DIR / f"insurance_trajectory_{loc_key}_{scenario}.png"),
                )

    print("\n" + "=" * 60)
    print("IDEA 5 PIPELINE COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
