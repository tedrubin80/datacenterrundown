"""Static Monte Carlo simulation replicating DCCore.pdf baseline."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .components import TCOParams, compute_tco, sample_triangular
from ..data.location_profiles import LocationProfile


@dataclass
class MCResult:
    """Results from a Monte Carlo simulation run."""

    location_key: str
    n_simulations: int
    tco_distribution: np.ndarray
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    cv: float = 0.0

    def __post_init__(self):
        dist = self.tco_distribution
        self.mean = float(np.mean(dist))
        self.std = float(np.std(dist))
        self.median = float(np.median(dist))
        self.p5 = float(np.percentile(dist, 5))
        self.p95 = float(np.percentile(dist, 95))
        self.var_95 = self.p95 - self.mean
        self.cvar_95 = float(np.mean(dist[dist >= np.percentile(dist, 95)]))
        self.cv = self.std / self.mean if self.mean != 0 else 0.0


def run_static_mc(
    location: LocationProfile,
    n_simulations: int = 10000,
    horizon_years: int = 10,
    discount_rate: float = 0.07,
    seed: Optional[int] = 42,
) -> MCResult:
    """Run static Monte Carlo simulation for a location.

    Samples from triangular distributions for CapEx, power cost, PUE
    as defined in the paper.
    """
    rng = np.random.default_rng(seed)
    tco_values = np.zeros(n_simulations)

    for i in range(n_simulations):
        params = TCOParams(
            capex_millions=sample_triangular(location.capex_millions, rng),
            power_cost_mwh=sample_triangular(location.power_cost_mwh, rng),
            pue=sample_triangular(location.pue, rng),
            capacity_mw=10.0,
            utilization=rng.uniform(0.65, 0.90),
            staffing_annual_millions=rng.uniform(3.0, 8.0),
            maintenance_pct=rng.uniform(0.02, 0.04),
            insurance_annual_millions=rng.uniform(1.0, 5.0),
        )
        tco_values[i] = compute_tco(params, horizon_years, discount_rate)

    return MCResult(
        location_key=location.key,
        n_simulations=n_simulations,
        tco_distribution=tco_values,
    )


def run_all_locations(
    locations: dict[str, LocationProfile],
    n_simulations: int = 10000,
    horizon_years: int = 10,
    seed: int = 42,
) -> dict[str, MCResult]:
    """Run static MC for all locations. Returns dict of results."""
    results = {}
    for key, loc in locations.items():
        results[key] = run_static_mc(loc, n_simulations, horizon_years, seed=seed)
    return results
