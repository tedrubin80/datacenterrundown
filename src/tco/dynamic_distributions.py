"""Idea 3 Core: ML-shifted Monte Carlo distributions under climate scenarios.

Instead of static triangular distributions, the ML model predicts how
climate conditions at each year shift the distribution parameters for
power cost, PUE, and insurance. These shifted parameters feed into the
same TCO computation engine.
"""

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np
import pandas as pd

from .components import TCOParams, compute_annual_opex, sample_triangular
from .discount import npv
from ..data.location_profiles import LocationProfile


class DistributionPredictor(Protocol):
    """Interface for ML models that predict shifted distribution params."""

    def predict_shifts(
        self, climate_features: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """Given climate features for a year, return shifted (mean, std) per variable.

        Returns dict with keys: power_cost_shift, pue_shift, insurance_shift
        Each value is (mean_delta, std_scale_factor).
        """
        ...


@dataclass
class DynamicMCResult:
    """Results from climate-dynamic Monte Carlo."""

    location_key: str
    scenario: str
    n_simulations: int
    tco_distribution: np.ndarray
    yearly_tco: np.ndarray  # (n_simulations, horizon_years)
    mean: float = 0.0
    std: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    var_95: float = 0.0
    cv: float = 0.0

    def __post_init__(self):
        dist = self.tco_distribution
        self.mean = float(np.mean(dist))
        self.std = float(np.std(dist))
        self.p5 = float(np.percentile(dist, 5))
        self.p95 = float(np.percentile(dist, 95))
        self.var_95 = self.p95 - self.mean
        self.cv = self.std / self.mean if self.mean != 0 else 0.0


def run_dynamic_mc(
    location: LocationProfile,
    climate_projections: pd.DataFrame,
    model: DistributionPredictor,
    scenario: str,
    n_simulations: int = 10000,
    horizon_years: int = 10,
    discount_rate: float = 0.07,
    seed: Optional[int] = 42,
) -> DynamicMCResult:
    """Run Monte Carlo with ML-shifted distributions per year.

    For each simulation:
      1. Sample base CapEx from static distribution
      2. For each year t in horizon:
         a. Get climate features at year t from projections
         b. Ask ML model for distribution shifts
         c. Sample OpEx components from shifted distributions
         d. Discount and sum

    Args:
        location: Location profile with base parameter ranges.
        climate_projections: DataFrame with year-by-year climate vars for this location+scenario.
        model: ML model implementing DistributionPredictor protocol.
        scenario: RCP scenario name.
        n_simulations: Number of Monte Carlo iterations.
        horizon_years: TCO horizon.
        discount_rate: Discount rate for NPV.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    start_year = climate_projections["year"].min()

    tco_values = np.zeros(n_simulations)
    yearly_costs = np.zeros((n_simulations, horizon_years))

    # Location static features to append (matches training feature set)
    loc_static = np.array([
        location.latitude,
        location.longitude,
        location.baseline_temp_c,
        location.renewable_pct,
        location.grid_reliability_score,
    ])

    # Pre-compute climate features per year, augmented with location features
    climate_by_year = {}
    # Match training feature order: 6 climate + 5 location + 4 engineered = 15
    climate_cols = [
        "avg_temp_c", "cooling_degree_days", "humidity_pct",
        "extreme_event_freq", "projected_pue", "power_price_delta_pct",
    ]
    climate_cols = [c for c in climate_cols if c in climate_projections.columns]
    for _, row in climate_projections.iterrows():
        climate_feat = row[climate_cols].values.astype(float)
        # Append: years_from_start, temp_x_humidity, cdd_trend placeholder, event_freq_trend placeholder
        years_from_start = float(row["year"]) - start_year
        temp = row.get("avg_temp_c", 0)
        humidity = row.get("humidity_pct", 50)
        temp_x_humidity = temp * humidity / 100
        augmented = np.concatenate([
            climate_feat, loc_static,
            [years_from_start, temp_x_humidity, row.get("cooling_degree_days", 0), row.get("extreme_event_freq", 0)],
        ])
        climate_by_year[int(row["year"])] = augmented

    for i in range(n_simulations):
        # Static CapEx sample
        capex = sample_triangular(location.capex_millions, rng)
        total = capex

        for t in range(1, horizon_years + 1):
            year = start_year + t
            climate_feat = climate_by_year.get(year, climate_by_year.get(max(climate_by_year.keys())))

            # Get ML-predicted shifts
            shifts = model.predict_shifts(climate_feat.reshape(1, -1))

            # Shift power cost distribution
            pc_delta, pc_scale = shifts.get("power_cost_shift", (0.0, 1.0))
            base_power = sample_triangular(location.power_cost_mwh, rng)
            shifted_power = max(0.1, base_power + pc_delta + rng.normal(0, abs(pc_scale)))

            # Shift PUE distribution
            pue_delta, pue_scale = shifts.get("pue_shift", (0.0, 1.0))
            base_pue = sample_triangular(location.pue, rng)
            shifted_pue = max(1.0, base_pue + pue_delta + rng.normal(0, abs(pue_scale) * 0.01))

            # Shift insurance cost
            ins_delta, ins_scale = shifts.get("insurance_shift", (0.0, 1.0))
            base_insurance = rng.uniform(1.0, 5.0)
            shifted_insurance = max(0.1, base_insurance * ins_scale + ins_delta)

            params = TCOParams(
                capex_millions=capex,
                power_cost_mwh=shifted_power,
                pue=shifted_pue,
                capacity_mw=10.0,
                utilization=rng.uniform(0.65, 0.90),
                staffing_annual_millions=rng.uniform(3.0, 8.0),
                maintenance_pct=rng.uniform(0.02, 0.04),
                insurance_annual_millions=shifted_insurance,
            )

            annual_cost = compute_annual_opex(params)
            discounted = annual_cost / (1 + discount_rate) ** t
            yearly_costs[i, t - 1] = discounted
            total += discounted

        tco_values[i] = total

    return DynamicMCResult(
        location_key=location.key,
        scenario=scenario,
        n_simulations=n_simulations,
        tco_distribution=tco_values,
        yearly_tco=yearly_costs,
    )
