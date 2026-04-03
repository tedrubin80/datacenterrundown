"""Load and process FEMA disaster declarations for Idea 5 models."""

from pathlib import Path

import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "fema" / "DisasterDeclarationsSummaries.csv"

# Map state abbreviations to our location keys
STATE_TO_LOCATION = {
    "WY": "evanston_wyoming",
    "UT": "salt_lake_city_utah",
    "IA": "des_moines_iowa",
    "SC": "florence_south_carolina",
    "GA": "atlanta_georgia",
}

# Incident types relevant to datacenter risk
DC_RELEVANT_TYPES = [
    "Severe Storm", "Hurricane", "Flood", "Tornado", "Fire",
    "Severe Ice Storm", "Snowstorm", "Winter Storm", "Typhoon",
    "Coastal Storm", "Earthquake",
]


def load_fema(path: str = None) -> pd.DataFrame:
    """Load raw FEMA disaster declarations."""
    if path is None:
        path = DATA_PATH
    return pd.read_csv(path, parse_dates=["declarationDate", "incidentBeginDate", "incidentEndDate"])


def get_our_locations(df: pd.DataFrame = None) -> pd.DataFrame:
    """Filter to our 5 US datacenter states and add location_key."""
    if df is None:
        df = load_fema()

    filtered = df[df["state"].isin(STATE_TO_LOCATION.keys())].copy()
    filtered["location_key"] = filtered["state"].map(STATE_TO_LOCATION)
    return filtered


def annual_event_counts(df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute annual disaster event counts per location.

    Returns DataFrame with: location_key, year, total_events,
    severe_storm, hurricane, flood, tornado, fire, other.
    """
    if df is None:
        df = get_our_locations()

    df = df.copy()
    df["year"] = df["declarationDate"].dt.year

    # Total events per location-year
    totals = df.groupby(["location_key", "year"]).size().reset_index(name="total_events")

    # Breakdown by type
    for incident_type in ["Severe Storm", "Hurricane", "Flood", "Tornado", "Fire"]:
        col_name = incident_type.lower().replace(" ", "_")
        type_counts = (
            df[df["incidentType"] == incident_type]
            .groupby(["location_key", "year"])
            .size()
            .reset_index(name=col_name)
        )
        totals = totals.merge(type_counts, on=["location_key", "year"], how="left")

    totals = totals.fillna(0)
    return totals


def compute_event_severity_proxy(df: pd.DataFrame = None) -> pd.DataFrame:
    """Estimate event severity from duration and program declarations.

    Uses incident duration (begin to end date) and whether PA/IA programs
    were declared as a proxy for severity.
    """
    if df is None:
        df = get_our_locations()

    df = df.copy()
    df["year"] = df["declarationDate"].dt.year

    # Duration in days
    df["duration_days"] = (df["incidentEndDate"] - df["incidentBeginDate"]).dt.days
    df["duration_days"] = df["duration_days"].clip(lower=0).fillna(1)

    # Severity proxy: sum of programs declared (PA, IA, HM = higher severity)
    df["programs_declared"] = (
        df["paProgramDeclared"].fillna(0).astype(int) +
        df["iaProgramDeclared"].fillna(0).astype(int) +
        df["hmProgramDeclared"].fillna(0).astype(int)
    )

    # Composite severity: normalized 0-1
    df["severity_proxy"] = (
        df["duration_days"] / df["duration_days"].quantile(0.95) * 0.5 +
        df["programs_declared"] / 3 * 0.5
    ).clip(0, 1)

    # Aggregate per location-year
    annual = df.groupby(["location_key", "year"]).agg(
        n_events=("disasterNumber", "nunique"),
        mean_severity=("severity_proxy", "mean"),
        max_severity=("severity_proxy", "max"),
        total_duration_days=("duration_days", "sum"),
        mean_programs=("programs_declared", "mean"),
    ).reset_index()

    return annual


def event_trend_analysis(df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute rolling trends in event frequency per location.

    Returns DataFrame with 5-year rolling average of events.
    """
    counts = annual_event_counts(df)

    trends = []
    for loc in counts["location_key"].unique():
        loc_df = counts[counts["location_key"] == loc].sort_values("year").copy()
        loc_df["events_5yr_avg"] = loc_df["total_events"].rolling(5, min_periods=1).mean()
        loc_df["events_trend"] = loc_df["total_events"].rolling(5, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        )
        trends.append(loc_df)

    return pd.concat(trends, ignore_index=True)
