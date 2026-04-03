"""Training orchestration for Idea 5 models.

Uses real FEMA disaster data for US locations when available,
with synthetic fallback for international locations.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...data.dataset import DataCenterDataset
from ...data.climate_projections import generate_all_projections
from ...data.location_profiles import load_locations
from .event_classifier import OutageClassifier
from .insurance_regressor import InsurancePremiumModel
from .survival_model import OutageSurvivalModel, prepare_survival_data

FEMA_PATH = Path(__file__).parents[3] / "data" / "raw" / "fema" / "DisasterDeclarationsSummaries.csv"


def _load_real_fema_events() -> Optional[pd.DataFrame]:
    """Load real FEMA disaster data if available."""
    if not FEMA_PATH.exists():
        return None

    from ...data.load_fema import get_our_locations, compute_event_severity_proxy
    raw = get_our_locations()
    severity = compute_event_severity_proxy()
    return severity


def _build_event_data(
    climate_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Build event data using real FEMA for US, synthetic for international.

    For US locations: uses actual FEMA annual event counts and severity.
    For international: generates synthetic events from climate projections.
    """
    rng = np.random.default_rng(seed)
    fema_data = _load_real_fema_events()

    records = []
    for _, row in climate_df.iterrows():
        loc_key = row["location_key"]
        year = int(row["year"])

        # Try real FEMA data
        if fema_data is not None:
            fema_match = fema_data[
                (fema_data["location_key"] == loc_key) & (fema_data["year"] == year)
            ]
            if len(fema_match) > 0:
                fm = fema_match.iloc[0]
                records.append({
                    "location_key": loc_key,
                    "year": year,
                    "scenario": row.get("scenario", "historical"),
                    "n_events": int(fm["n_events"]),
                    "severity": float(fm["mean_severity"]),
                    "max_severity": float(fm["max_severity"]),
                    "total_duration_days": float(fm["total_duration_days"]),
                    "downtime_hours": float(fm["total_duration_days"]) * 0.5,  # estimate 50% of event duration causes some downtime
                    "source": "fema",
                })
                continue

            # For future years: extrapolate from recent FEMA trends
            recent = fema_data[
                (fema_data["location_key"] == loc_key) & (fema_data["year"] >= 2015)
            ]
            if len(recent) > 0:
                avg_events = recent["n_events"].mean()
                avg_severity = recent["mean_severity"].mean()
                avg_duration = recent["total_duration_days"].mean()

                # Scale by climate scenario event multiplier
                event_freq = row.get("extreme_event_freq", avg_events)
                baseline_freq = recent["n_events"].iloc[0] if len(recent) > 0 else avg_events
                scale = event_freq / max(1, baseline_freq) if baseline_freq > 0 else 1.0

                n_events = max(1, int(avg_events * scale + rng.normal(0, avg_events * 0.1)))
                severity = min(1.0, avg_severity * (1 + (scale - 1) * 0.3) + rng.normal(0, 0.03))
                duration = avg_duration * scale + rng.normal(0, avg_duration * 0.1)

                records.append({
                    "location_key": loc_key,
                    "year": year,
                    "scenario": row.get("scenario", "projected"),
                    "n_events": n_events,
                    "severity": round(max(0, severity), 3),
                    "max_severity": round(min(1.0, severity * 1.3), 3),
                    "total_duration_days": round(max(0, duration), 1),
                    "downtime_hours": round(max(0, duration * 0.5), 1),
                    "source": "fema_projected",
                })
                continue

        # Synthetic fallback for international locations
        n_events = int(row.get("extreme_event_freq", 2))
        temp_factor = max(0, row.get("avg_temp_c", 15) - 10) / 30
        severity = float(rng.beta(2, 5) + temp_factor * 0.2)
        severity = min(1.0, severity)
        downtime = rng.exponential(severity * 48) if severity > 0.5 else 0.0

        records.append({
            "location_key": loc_key,
            "year": year,
            "scenario": row.get("scenario", "synthetic"),
            "n_events": n_events,
            "severity": round(severity, 3),
            "max_severity": round(min(1.0, severity * 1.2), 3),
            "total_duration_days": round(rng.exponential(24 * severity) / 24, 1),
            "downtime_hours": round(downtime, 1),
            "source": "synthetic",
        })

    return pd.DataFrame(records)


def _build_insurance_targets(
    climate_df: pd.DataFrame,
    events_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Build insurance premium targets from event data.

    Premium model:
    - Base premium by tier (Nordic low, US moderate, traditional high)
    - Event frequency multiplier (real FEMA counts for US)
    - Climate trend multiplier (power price pressure)
    - Severity multiplier (nonlinear)
    """
    rng = np.random.default_rng(seed)
    locations = load_locations()

    # Base premiums by tier (millions/year for a 10MW facility)
    tier_base = {
        "nordic": 1.5,
        "us_secondary": 4.0,
        "traditional": 6.0,
        "emerging": 5.0,
    }

    df = climate_df.merge(events_df, on=["location_key", "year", "scenario"], how="left")
    df[["n_events", "severity", "downtime_hours"]] = \
        df[["n_events", "severity", "downtime_hours"]].fillna(0)

    premiums = []
    for _, row in df.iterrows():
        loc = locations.get(row["location_key"])
        base = tier_base.get(loc.tier if loc else "traditional", 4.0)

        # Event frequency factor: more events = higher premium (nonlinear)
        n_events = row.get("n_events", 0)
        # Normalize: US states have way more declarations (county-level)
        # Scale to per-facility risk
        if row.get("source", "") in ("fema", "fema_projected"):
            # FEMA counts are state-level, normalize by ~100 counties
            facility_events = n_events / 50
        else:
            facility_events = n_events

        event_factor = 1.0 + facility_events ** 0.7 * 0.3

        # Severity factor
        severity = row.get("severity", 0)
        severity_factor = 1.0 + severity * 0.5

        # Climate trend factor
        power_pct = row.get("power_price_delta_pct", 0)
        climate_factor = 1.0 + max(0, power_pct) / 100

        # Time factor: insurance industry raises rates over time
        year = row.get("year", 2025)
        time_factor = 1.0 + (year - 2025) * 0.008  # ~0.8% annual hardening

        premium = base * event_factor * severity_factor * climate_factor * time_factor
        premium *= rng.normal(1.0, 0.03)  # small noise
        premiums.append(max(0.5, round(premium, 3)))

    df["annual_premium_millions"] = premiums
    return df


def _prepare_survival_from_events(
    events_df: pd.DataFrame,
    climate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert event data to survival format for the survival model."""
    records = []
    for loc in events_df["location_key"].unique():
        loc_events = events_df[events_df["location_key"] == loc].sort_values("year")
        loc_climate = climate_df[climate_df["location_key"] == loc]

        rng = np.random.default_rng(42)
        for _, event_row in loc_events.iterrows():
            year = event_row["year"]
            climate_row = loc_climate[loc_climate["year"] == year]
            if len(climate_row) == 0:
                continue
            cr = climate_row.iloc[0]

            # Outage probability based on multiple factors (not just severity)
            # so classifier has a non-trivial learning task
            n_events = max(1, event_row.get("n_events", 1))
            severity = event_row.get("severity", 0)
            temp = cr.get("avg_temp_c", 15)
            outage_prob = (
                0.1                                    # base rate
                + min(0.3, n_events / 50)              # event frequency
                + severity * 0.2                       # severity
                + max(0, temp - 20) * 0.01             # heat stress
                + rng.uniform(-0.1, 0.1)               # noise
            )
            has_outage = rng.random() < np.clip(outage_prob, 0.05, 0.95)

            records.append({
                "location_key": loc,
                "year": year,
                "time_to_outage_days": max(1, int(365 / n_events)) if has_outage else 365,
                "outage_occurred": 1 if has_outage else 0,
                "avg_temp_c": cr.get("avg_temp_c", 0),
                "extreme_event_freq": cr.get("extreme_event_freq", 0),
                "projected_pue": cr.get("projected_pue", 1.1),
                "humidity_pct": cr.get("humidity_pct", 50),
                "cooling_degree_days": cr.get("cooling_degree_days", 0),
                "n_events": n_events,
                "severity": event_row.get("severity", 0),
                "source": event_row.get("source", "unknown"),
            })

    return pd.DataFrame(records)


# --- Public training functions ---

def train_survival(
    climate_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> tuple[OutageSurvivalModel, dict]:
    """Train survival model for outage prediction using real + synthetic data."""
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    events_df = _build_event_data(climate_df, seed=seed)
    survival_df = _prepare_survival_from_events(events_df, climate_df)

    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days", "n_events", "severity",
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
    """Train outage event classifier using real + synthetic data."""
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    events_df = _build_event_data(climate_df, seed=seed)
    survival_df = _prepare_survival_from_events(events_df, climate_df)

    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days", "n_events", "severity",
    ]

    X = survival_df[feature_cols].values
    y = survival_df["outage_occurred"].values

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
    """Train insurance premium model using real + synthetic event data."""
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    events_df = _build_event_data(climate_df, seed=seed)
    insurance_df = _build_insurance_targets(climate_df, events_df, seed=seed)

    feature_cols = [
        "avg_temp_c", "extreme_event_freq", "projected_pue",
        "humidity_pct", "cooling_degree_days", "power_price_delta_pct",
        "n_events", "severity", "downtime_hours",
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


# Keep old synthetic function for backward compat with notebooks
def _generate_event_data(climate_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Legacy synthetic event generator (use _build_event_data instead)."""
    return _build_event_data(climate_df, seed=seed)


def _generate_insurance_targets(
    climate_df: pd.DataFrame,
    events_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Legacy insurance target generator."""
    return _build_insurance_targets(climate_df, events_df, seed=seed)
