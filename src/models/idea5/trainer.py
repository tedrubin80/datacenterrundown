"""Training orchestration for Idea 5 models."""

from typing import Optional

import numpy as np
import pandas as pd

from ...data.dataset import DataCenterDataset
from ...data.climate_projections import generate_all_projections
from ...data.location_profiles import load_locations
from .event_classifier import OutageClassifier
from .insurance_regressor import InsurancePremiumModel
from .survival_model import OutageSurvivalModel, prepare_survival_data


def _generate_event_data(climate_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic extreme weather event records from climate projections.

    Creates event records correlated with climate severity.
    """
    rng = np.random.default_rng(seed)
    records = []

    for _, row in climate_df.iterrows():
        n_events = int(row.get("extreme_event_freq", 2))
        for _ in range(n_events):
            severity = rng.beta(2, 5)  # skewed toward lower severity
            # Higher temps and more events increase severity
            temp_factor = max(0, row.get("avg_temp_c", 15) - 10) / 30
            severity = min(1.0, severity + temp_factor * 0.2)

            downtime = 0.0
            if severity > 0.5:
                downtime = rng.exponential(severity * 48)

            records.append({
                "location_key": row["location_key"],
                "year": int(row["year"]),
                "scenario": row.get("scenario", "rcp45"),
                "severity": round(severity, 3),
                "duration_hours": round(rng.exponential(24 * severity) + 1, 1),
                "downtime_hours": round(downtime, 1),
                "damage_cost_millions": round(max(0, rng.exponential(severity * 5)), 3),
                "grid_outage": bool(rng.random() < severity * 0.3),
            })

    return pd.DataFrame(records)


def _generate_insurance_targets(
    climate_df: pd.DataFrame,
    events_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate insurance premium targets correlated with climate and events."""
    rng = np.random.default_rng(seed)

    # Aggregate events per location-year
    event_agg = events_df.groupby(["location_key", "year"]).agg(
        n_events=("severity", "count"),
        max_severity=("severity", "max"),
        total_damage=("damage_cost_millions", "sum"),
        total_downtime=("downtime_hours", "sum"),
    ).reset_index()

    df = climate_df.merge(event_agg, on=["location_key", "year"], how="left")
    df[["n_events", "max_severity", "total_damage", "total_downtime"]] = \
        df[["n_events", "max_severity", "total_damage", "total_downtime"]].fillna(0)

    # Premium driven by: baseline risk + climate trend + event history
    locations = load_locations()
    base_premiums = {
        k: rng.uniform(1.0, 3.0) if v.tier == "nordic"
        else rng.uniform(3.0, 8.0)
        for k, v in locations.items()
    }

    premiums = []
    for _, row in df.iterrows():
        base = base_premiums.get(row["location_key"], 3.0)
        climate_factor = 1 + row.get("power_price_delta_pct", 0) / 200
        event_factor = 1 + row.get("total_damage", 0) * 0.05
        noise = rng.normal(1.0, 0.05)
        premium = base * climate_factor * event_factor * noise
        premiums.append(max(0.5, round(premium, 3)))

    df["annual_premium_millions"] = premiums
    return df


def train_survival(
    climate_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> tuple[OutageSurvivalModel, dict]:
    """Train survival model for outage prediction."""
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    events_df = _generate_event_data(climate_df, seed=seed)
    survival_df = prepare_survival_data(events_df, climate_df)

    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days",
    ]

    model = OutageSurvivalModel(model_type="weibull_aft")
    metrics = model.fit(
        survival_df,
        duration_col="time_to_outage_days",
        event_col="outage_occurred",
        feature_cols=feature_cols,
    )

    return model, metrics


def train_classifier(
    climate_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> tuple[OutageClassifier, dict]:
    """Train outage event classifier."""
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    events_df = _generate_event_data(climate_df, seed=seed)
    survival_df = prepare_survival_data(events_df, climate_df)

    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days",
    ]

    X = survival_df[feature_cols].values
    y = survival_df["outage_occurred"].values

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = OutageClassifier(seed=seed)
    metrics = model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

    return model, metrics


def train_insurance(
    climate_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> tuple[InsurancePremiumModel, dict]:
    """Train insurance premium prediction model."""
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    events_df = _generate_event_data(climate_df, seed=seed)
    insurance_df = _generate_insurance_targets(climate_df, events_df, seed=seed)

    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days", "power_price_delta_pct",
        "n_events", "max_severity", "total_damage", "total_downtime",
    ]

    valid = insurance_df.dropna(subset=feature_cols + ["annual_premium_millions"])
    X = valid[feature_cols].values
    y = valid["annual_premium_millions"].values

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = InsurancePremiumModel(seed=seed)
    metrics = model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)

    return model, metrics
