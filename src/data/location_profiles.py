"""Location profiles from DCCore.pdf with parameter ranges."""

from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path

import yaml


@dataclass
class LocationProfile:
    """A datacenter location with its cost/climate parameter ranges."""

    key: str
    name: str
    tier: str
    currency: str
    latitude: float
    longitude: float
    capex_millions: Tuple[float, float, float]  # min, mode, max
    power_cost_mwh: Tuple[float, float, float]
    pue: Tuple[float, float, float]
    baseline_temp_c: float
    baseline_humidity_pct: float
    cooling_degree_days: int
    renewable_pct: float
    extreme_event_freq_annual: int
    grid_reliability_score: float
    mean_tco_10yr_millions: float
    cv: float

    @property
    def is_eur(self) -> bool:
        return self.currency == "EUR"


def load_locations(config_path: str = None) -> dict[str, LocationProfile]:
    """Load all location profiles from YAML config."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "locations.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    locations = {}
    for key, vals in raw["locations"].items():
        locations[key] = LocationProfile(
            key=key,
            name=vals["name"],
            tier=vals["tier"],
            currency=vals["currency"],
            latitude=vals["latitude"],
            longitude=vals["longitude"],
            capex_millions=tuple(vals["capex_millions"]),
            power_cost_mwh=tuple(vals["power_cost_mwh"]),
            pue=tuple(vals["pue"]),
            baseline_temp_c=vals["baseline_temp_c"],
            baseline_humidity_pct=vals["baseline_humidity_pct"],
            cooling_degree_days=vals["cooling_degree_days"],
            renewable_pct=vals["renewable_pct"],
            extreme_event_freq_annual=vals["extreme_event_freq_annual"],
            grid_reliability_score=vals["grid_reliability_score"],
            mean_tco_10yr_millions=vals["mean_tco_10yr_millions"],
            cv=vals["cv"],
        )
    return locations
