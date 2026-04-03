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


def load_locations(config_path: str = None, eia_calibrated: bool = False) -> dict[str, LocationProfile]:
    """Load all location profiles from YAML config.

    Args:
        config_path: Path to locations.yaml.
        eia_calibrated: If True, override US power costs with EIA-calibrated
            estimates from configs/eia_calibration.yaml.
    """
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "locations.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Load EIA calibration if requested
    eia_overrides = {}
    if eia_calibrated:
        eia_path = Path(__file__).parents[2] / "configs" / "eia_calibration.yaml"
        if eia_path.exists():
            with open(eia_path) as f:
                eia_raw = yaml.safe_load(f)
            for loc_key, vals in eia_raw.get("eia_calibrated_rates", {}).items():
                cal = vals["calibrated_mwh"]
                # Build new triangular: ±30% around calibrated midpoint
                eia_overrides[loc_key] = (cal * 0.7, cal, cal * 1.3)

    locations = {}
    for key, vals in raw["locations"].items():
        power_cost = tuple(vals["power_cost_mwh"])
        if eia_calibrated and key in eia_overrides:
            power_cost = eia_overrides[key]

        locations[key] = LocationProfile(
            key=key,
            name=vals["name"],
            tier=vals["tier"],
            currency=vals["currency"],
            latitude=vals["latitude"],
            longitude=vals["longitude"],
            capex_millions=tuple(vals["capex_millions"]),
            power_cost_mwh=power_cost,
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
