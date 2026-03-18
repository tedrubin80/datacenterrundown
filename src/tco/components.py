"""TCO cost components matching DCCore.pdf formulas."""

import numpy as np
from dataclasses import dataclass


HOURS_PER_YEAR = 8760


@dataclass
class TCOParams:
    """Parameters for a single TCO simulation run."""

    capex_millions: float
    power_cost_mwh: float
    pue: float
    capacity_mw: float = 10.0
    utilization: float = 0.75
    staffing_annual_millions: float = 5.0
    maintenance_pct: float = 0.03
    insurance_annual_millions: float = 2.0
    tax_incentive_pct: float = 0.0


def compute_annual_power_cost(
    capacity_mw: float,
    utilization: float,
    pue: float,
    power_rate_mwh: float,
) -> float:
    """PowerCost_t = Capacity * Utilization * PUE * PowerRate * 8760.

    Returns cost in millions.
    """
    return (capacity_mw * utilization * pue * power_rate_mwh * HOURS_PER_YEAR) / 1e6


def compute_annual_opex(params: TCOParams) -> float:
    """Total annual operating expenditure in millions."""
    power = compute_annual_power_cost(
        params.capacity_mw, params.utilization, params.pue, params.power_cost_mwh
    )
    maintenance = params.capex_millions * params.maintenance_pct
    staffing = params.staffing_annual_millions
    insurance = params.insurance_annual_millions
    return power + maintenance + staffing + insurance


def compute_tco(
    params: TCOParams,
    horizon_years: int = 10,
    discount_rate: float = 0.07,
    inflation_rate: float = 0.025,
) -> float:
    """Discounted TCO over the horizon period.

    TCO = sum over t of (CapEx_t + OpEx_t) / (1+r)^t
    CapEx is incurred in year 0. OpEx grows with inflation.
    """
    # Year 0: CapEx (undiscounted)
    tco = params.capex_millions

    for t in range(1, horizon_years + 1):
        annual_opex = compute_annual_opex(params) * (1 + inflation_rate) ** t
        incentive_savings = annual_opex * (params.tax_incentive_pct / 100)
        net_opex = annual_opex - incentive_savings
        tco += net_opex / (1 + discount_rate) ** t

    return tco


def sample_triangular(tri_params: tuple, rng: np.random.Generator) -> float:
    """Sample from triangular distribution given (min, mode, max)."""
    lo, mode, hi = tri_params
    return float(rng.triangular(lo, mode, hi))
