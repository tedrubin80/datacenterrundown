"""Generate climate projection time-series under RCP scenarios."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .location_profiles import LocationProfile, load_locations


def load_scenarios(config_path: str = None) -> dict:
    """Load RCP scenario definitions."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "climate_scenarios.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["scenarios"]


def _interpolate_yearly(by_year: dict, target_years: np.ndarray) -> np.ndarray:
    """Linearly interpolate between defined milestone years."""
    known_years = sorted(by_year.keys())
    known_vals = [by_year[y] for y in known_years]
    return np.interp(target_years, known_years, known_vals)


def project_climate(
    location: LocationProfile,
    scenario: dict,
    year_start: int = 2025,
    year_end: int = 2050,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate yearly climate projections for a location under a scenario.

    Returns DataFrame with columns:
        year, avg_temp_c, cooling_degree_days, humidity_pct,
        extreme_event_freq, pue_delta, power_price_delta_pct
    """
    rng = np.random.default_rng(seed)
    years = np.arange(year_start, year_end + 1)
    n = len(years)

    # Interpolate scenario curves
    temp_anomaly = _interpolate_yearly(scenario["temp_anomaly_by_year"], years)
    event_mult = _interpolate_yearly(scenario["extreme_event_multiplier_by_year"], years)
    cooling_inc = _interpolate_yearly(scenario["cooling_demand_increase_pct_by_year"], years)
    power_pres = _interpolate_yearly(scenario["power_price_pressure_pct_by_year"], years)

    # Apply to location baselines with noise
    avg_temp = location.baseline_temp_c + temp_anomaly + rng.normal(0, 0.3, n)
    cdd = location.cooling_degree_days * (1 + cooling_inc / 100) + rng.normal(0, 20, n)
    cdd = np.maximum(cdd, 0)
    humidity = location.baseline_humidity_pct + rng.normal(0, 2.0, n)
    humidity = np.clip(humidity, 15, 98)
    event_freq = location.extreme_event_freq_annual * event_mult + rng.poisson(0.5, n)
    event_freq = np.maximum(event_freq, 0).astype(int)

    # PUE degradation: hotter temps push PUE up
    pue_base = location.pue[1]  # mode value
    pue_delta = (temp_anomaly - temp_anomaly[0]) * 0.015  # ~0.015 PUE per degree C
    pue_delta += rng.normal(0, 0.005, n)

    return pd.DataFrame({
        "year": years,
        "location_key": location.key,
        "avg_temp_c": np.round(avg_temp, 2),
        "cooling_degree_days": np.round(cdd).astype(int),
        "humidity_pct": np.round(humidity, 1),
        "extreme_event_freq": event_freq,
        "projected_pue": np.round(pue_base + pue_delta, 4),
        "pue_delta": np.round(pue_delta, 4),
        "power_price_delta_pct": np.round(power_pres + rng.normal(0, 1.0, n), 2),
    })


def generate_all_projections(
    year_start: int = 2025,
    year_end: int = 2050,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate projections for all locations x all RCP scenarios."""
    locations = load_locations()
    scenarios = load_scenarios()
    frames = []

    for scenario_key, scenario in scenarios.items():
        for loc_key, loc in locations.items():
            df = project_climate(loc, scenario, year_start, year_end, seed=seed)
            df["scenario"] = scenario_key
            frames.append(df)

    return pd.concat(frames, ignore_index=True)
