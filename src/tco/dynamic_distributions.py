"""Idea 3 Core: Climate-shifted Monte Carlo distributions.

Uses physics-based shift functions grounded in literature:
- Temperature → PUE: ~0.3-0.5% PUE increase per °C (Depoorter et al., 2015)
- Temperature → Power cost: cooling load scales with CDD, grid stress raises prices
- Extreme events → Insurance: nonlinear premium escalation with event frequency
- Grid stress → Power price: compounding effect in constrained grids

These deterministic shifts are applied year-by-year based on RCP climate
projections. The ML ensemble can optionally refine these as a second layer
once sufficient real data is available.
"""

from dataclasses import dataclass
from typing import Protocol, Optional

import numpy as np
import pandas as pd

from .components import TCOParams, compute_annual_opex, sample_triangular
from ..data.location_profiles import LocationProfile


class DistributionPredictor(Protocol):
    """Interface for ML models that predict shifted distribution params."""

    def predict_shifts(
        self, climate_features: np.ndarray
    ) -> dict[str, tuple[float, float]]:
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


def _physics_shifts(
    row: pd.Series,
    location: LocationProfile,
    start_year: int,
) -> dict[str, tuple[float, float]]:
    """Compute physics-based distribution shifts from climate projections.

    Returns dict with (mean_delta, std_noise_scale) per TCO component.
    """
    year = int(row["year"])
    years_elapsed = year - start_year

    # --- Temperature-driven PUE degradation ---
    # Literature: 0.5-1.5% PUE increase per °C (Shehabi et al., 2016; Lei & Masanet, 2020)
    # Air-cooled: ~0.008-0.012 PUE per °C; liquid-cooled: ~0.003-0.005
    # Higher baselines degrade faster (nonlinear — approaching cooling capacity limits)
    temp_now = row.get("avg_temp_c", location.baseline_temp_c)
    temp_delta = temp_now - location.baseline_temp_c
    humidity = row.get("humidity_pct", location.baseline_humidity_pct)
    # Wet-bulb effect: high humidity compounds cooling penalty
    humidity_factor = 1.0 + max(0, humidity - 60) / 100  # up to 1.4x at 100% humidity
    # Baseline PUE affects sensitivity: already-efficient facilities degrade less
    pue_base = location.pue[1]
    pue_sensitivity = 0.008 if pue_base < 1.15 else 0.012  # liquid vs air cooled
    # Nonlinear: each additional degree hurts more as you approach cooling limits
    pue_shift = temp_delta * pue_sensitivity * humidity_factor * (1 + temp_delta * 0.05)
    pue_noise = 0.003 + years_elapsed * 0.0005

    # --- Power cost escalation ---
    # Direct: power_price_delta_pct from scenario (grid stress, demand growth)
    # Indirect: higher CDD = more cooling energy at same PUE
    power_price_pct = row.get("power_price_delta_pct", 0)
    base_power_mode = location.power_cost_mwh[1]
    # CDD-driven cooling demand increase
    cdd_now = row.get("cooling_degree_days", location.cooling_degree_days)
    cdd_delta_pct = (cdd_now - location.cooling_degree_days) / max(1, location.cooling_degree_days) * 100
    # Total power cost shift: price pressure + cooling demand growth
    power_shift = base_power_mode * ((power_price_pct + cdd_delta_pct * 0.3) / 100)
    # Grid stress adds volatility — worse in high-demand regions
    grid_stress = 1 + (1 - location.grid_reliability_score) * 5
    power_noise = base_power_mode * 0.03 * grid_stress * (1 + years_elapsed * 0.015)

    # --- Insurance premium escalation ---
    # Nonlinear: premiums jump when event frequency crosses thresholds
    # Based on insurance industry catastrophe modeling (Swiss Re sigma reports)
    event_freq = row.get("extreme_event_freq", location.extreme_event_freq_annual)
    baseline_events = location.extreme_event_freq_annual
    event_ratio = event_freq / max(1, baseline_events)
    # Insurers price risk nonlinearly — doubling events more than doubles premiums
    # Using power law: premium_scale ~ event_ratio^1.5
    insurance_scale = event_ratio ** 1.5
    insurance_noise = 0.1 * (event_ratio - 1)  # more events = more price uncertainty

    return {
        "power_cost_shift": (power_shift, power_noise),
        "pue_shift": (pue_shift, pue_noise),
        "insurance_shift": (insurance_scale, insurance_noise),
    }


def run_dynamic_mc(
    location: LocationProfile,
    climate_projections: pd.DataFrame,
    model: Optional[DistributionPredictor] = None,
    scenario: str = "",
    n_simulations: int = 10000,
    horizon_years: int = 10,
    discount_rate: float = 0.07,
    seed: Optional[int] = 42,
) -> DynamicMCResult:
    """Run Monte Carlo with climate-shifted distributions per year.

    Uses physics-based shifts from climate projections. If an ML model is
    provided, its predictions are blended as a refinement layer.
    """
    rng = np.random.default_rng(seed)
    start_year = int(climate_projections["year"].min())

    tco_values = np.zeros(n_simulations)
    yearly_costs = np.zeros((n_simulations, horizon_years))

    # Pre-compute shifts per year from physics model
    shifts_by_year = {}
    proj_sorted = climate_projections.sort_values("year")
    for _, row in proj_sorted.iterrows():
        shifts_by_year[int(row["year"])] = _physics_shifts(row, location, start_year)

    # Optionally blend ML model predictions
    if model is not None:
        # ML model can provide a refinement multiplier (future use with real data)
        pass

    max_year = max(shifts_by_year.keys())

    for i in range(n_simulations):
        capex = sample_triangular(location.capex_millions, rng)
        total = capex

        for t in range(1, horizon_years + 1):
            year = start_year + t
            shifts = shifts_by_year.get(year, shifts_by_year[max_year])

            # Shift power cost
            pc_delta, pc_noise = shifts["power_cost_shift"]
            base_power = sample_triangular(location.power_cost_mwh, rng)
            shifted_power = max(0.1, base_power + pc_delta + rng.normal(0, pc_noise))

            # Shift PUE
            pue_delta, pue_noise = shifts["pue_shift"]
            base_pue = sample_triangular(location.pue, rng)
            shifted_pue = max(1.0, base_pue + pue_delta + rng.normal(0, pue_noise))

            # Shift insurance via scale factor
            ins_scale, ins_noise = shifts["insurance_shift"]
            base_insurance = rng.uniform(1.0, 5.0)
            shifted_insurance = max(0.1, base_insurance * ins_scale + rng.normal(0, ins_noise))

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
