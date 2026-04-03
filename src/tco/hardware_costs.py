"""Hardware cost layer for datacenter TCO.

Adds IT equipment costs (CapEx, refresh, maintenance, energy) to facility TCO.
Based on DCCore PC Hardware Cost Addendum (April 2026) and Apple ARM TCO Report.

4 hardware tiers:
- Standard AI Rack (H100): $3.5M/rack, 100kW, 4yr refresh, 12% maint
- Traditional PC Rack: $500K/rack, 20kW, 5yr refresh, 10% maint
- Hybrid GPU+CPU: $2.0M/rack, 60kW, 4yr refresh, 11% maint
- ARM Efficiency: $800K/rack equiv, 15kW, 5yr refresh, 8% maint (Apple silicon)

Reference facility: 100MW, 500 racks.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data.location_profiles import LocationProfile


@dataclass
class HardwareTier:
    """Hardware configuration parameters."""
    name: str
    rack_cost: float          # $ per rack
    power_kw: float           # kW per rack
    refresh_years: int        # hardware lifecycle
    maintenance_pct: float    # annual maintenance as % of hardware cost
    hw_inflation_pct: float   # annual hardware cost inflation
    resale_pct: float         # residual value at refresh time


# From DCCore Addendum + Apple ARM report
HARDWARE_TIERS = {
    "standard_ai": HardwareTier(
        name="Standard AI Rack (H100)",
        rack_cost=3_500_000,
        power_kw=100,
        refresh_years=4,
        maintenance_pct=0.12,
        hw_inflation_pct=0.05,
        resale_pct=0.05,       # GPUs depreciate fast
    ),
    "traditional_pc": HardwareTier(
        name="Traditional PC Rack",
        rack_cost=500_000,
        power_kw=20,
        refresh_years=5,
        maintenance_pct=0.10,
        hw_inflation_pct=0.05,
        resale_pct=0.10,
    ),
    "hybrid_gpu_cpu": HardwareTier(
        name="Hybrid GPU+CPU Cluster",
        rack_cost=2_000_000,
        power_kw=60,
        refresh_years=4,
        maintenance_pct=0.11,
        hw_inflation_pct=0.05,
        resale_pct=0.08,
    ),
    "arm_efficiency": HardwareTier(
        name="ARM Efficiency (Apple Silicon)",
        rack_cost=800_000,      # Mac Studio/Pro rack equivalent
        power_kw=15,            # ~50% less than x86 (Apple ARM report)
        refresh_years=5,
        maintenance_pct=0.08,   # lower maintenance (Forrester TEI)
        hw_inflation_pct=0.03,  # ARM pricing more stable
        resale_pct=0.30,        # 30% resale at 3yr (Apple ARM report)
    ),
}

HOURS_PER_YEAR = 8760


def compute_annual_hw_energy_cost(
    n_racks: int,
    power_kw: float,
    pue: float,
    power_rate_mwh: float,
) -> float:
    """Compute annual hardware-attributed energy cost in millions.

    Formula: Racks × kW × PUE × 8760 × Rate($/MWh) / 1e6
    """
    mw = n_racks * power_kw / 1000
    return mw * pue * HOURS_PER_YEAR * power_rate_mwh / 1e6


def compute_hw_year_cost(
    tier: HardwareTier,
    year: int,
    n_racks: int,
    pue: float,
    power_rate_mwh: float,
    power_inflation: float = 0.025,
) -> dict:
    """Compute hardware costs for a single year.

    Returns dict with: refresh, maintenance, energy, total (all in millions).
    """
    # Hardware refresh: occurs every refresh_years starting at that year
    refresh = 0.0
    if year > 0 and year % tier.refresh_years == 0:
        # Refresh cost = initial cost × inflation^year - resale value
        inflated_cost = tier.rack_cost * (1 + tier.hw_inflation_pct) ** year
        resale = tier.rack_cost * (1 + tier.hw_inflation_pct) ** (year - tier.refresh_years) * tier.resale_pct
        refresh = n_racks * (inflated_cost - resale) / 1e6

    # Maintenance: % of current hardware value
    current_hw_value = n_racks * tier.rack_cost * (1 + tier.hw_inflation_pct) ** year / 1e6
    maintenance = current_hw_value * tier.maintenance_pct

    # Energy: adjusted for power cost inflation
    adjusted_rate = power_rate_mwh * (1 + power_inflation) ** year
    energy = compute_annual_hw_energy_cost(n_racks, tier.power_kw, pue, adjusted_rate)

    return {
        "refresh": refresh,
        "maintenance": maintenance,
        "energy": energy,
        "total": refresh + maintenance + energy,
    }


def compute_hw_tco(
    tier: HardwareTier,
    location: LocationProfile,
    n_racks: int = 500,
    horizon_years: int = 10,
    discount_rate: float = 0.07,
) -> dict:
    """Compute full hardware TCO for a location and tier.

    Returns dict with: initial_capex, yearly_costs, total_tco,
    total_energy, total_refresh, total_maintenance (all in millions).
    """
    initial_capex = n_racks * tier.rack_cost / 1e6

    yearly = []
    total_refresh = 0.0
    total_maint = 0.0
    total_energy = 0.0
    discounted_total = initial_capex  # Year 0

    for year in range(1, horizon_years + 1):
        costs = compute_hw_year_cost(
            tier, year, n_racks,
            pue=location.pue[1],
            power_rate_mwh=location.power_cost_mwh[1],
        )
        discounted = costs["total"] / (1 + discount_rate) ** year
        yearly.append({**costs, "year": year, "discounted": discounted})
        discounted_total += discounted
        total_refresh += costs["refresh"]
        total_maint += costs["maintenance"]
        total_energy += costs["energy"]

    return {
        "tier": tier.name,
        "location": location.name,
        "initial_capex_m": initial_capex,
        "total_refresh_m": total_refresh,
        "total_maintenance_m": total_maint,
        "total_energy_m": total_energy,
        "total_hw_tco_m": discounted_total,
        "yearly": yearly,
    }


def compute_climate_adjusted_hw_tco(
    tier: HardwareTier,
    location: LocationProfile,
    climate_row_by_year: dict,
    n_racks: int = 500,
    horizon_years: int = 10,
    discount_rate: float = 0.07,
    start_year: int = 2025,
) -> dict:
    """Compute hardware TCO with climate-shifted PUE and power costs.

    climate_row_by_year: dict mapping year -> dict with projected_pue, power_price_delta_pct
    """
    initial_capex = n_racks * tier.rack_cost / 1e6
    yearly = []
    discounted_total = initial_capex

    for t in range(1, horizon_years + 1):
        year = start_year + t
        climate = climate_row_by_year.get(year, {})

        # Climate-adjusted PUE
        pue = climate.get("projected_pue", location.pue[1])

        # Climate-adjusted power rate
        base_rate = location.power_cost_mwh[1]
        price_delta_pct = climate.get("power_price_delta_pct", 0)
        adjusted_rate = base_rate * (1 + price_delta_pct / 100) * (1.025 ** t)

        # Energy cost with shifted PUE and rate
        energy = compute_annual_hw_energy_cost(n_racks, tier.power_kw, pue, adjusted_rate)

        # Refresh
        refresh = 0.0
        if t > 0 and t % tier.refresh_years == 0:
            inflated_cost = tier.rack_cost * (1 + tier.hw_inflation_pct) ** t
            resale = tier.rack_cost * (1 + tier.hw_inflation_pct) ** (t - tier.refresh_years) * tier.resale_pct
            refresh = n_racks * (inflated_cost - resale) / 1e6

        # Maintenance
        current_hw_value = n_racks * tier.rack_cost * (1 + tier.hw_inflation_pct) ** t / 1e6
        maintenance = current_hw_value * tier.maintenance_pct

        total = refresh + maintenance + energy
        discounted = total / (1 + discount_rate) ** t
        discounted_total += discounted

        yearly.append({
            "year": year,
            "refresh": refresh,
            "maintenance": maintenance,
            "energy": energy,
            "pue": pue,
            "power_rate": adjusted_rate,
            "total": total,
            "discounted": discounted,
        })

    return {
        "tier": tier.name,
        "location": location.name,
        "initial_capex_m": initial_capex,
        "total_hw_tco_m": discounted_total,
        "yearly": yearly,
    }


def compute_combined_tco(
    facility_tco: float,
    hw_tco: float,
) -> float:
    """Combined facility + hardware TCO."""
    return facility_tco + hw_tco


def all_locations_all_tiers(
    locations: dict,
    n_racks: int = 500,
    horizon_years: int = 10,
) -> list[dict]:
    """Compute hardware TCO for all locations × all tiers.

    Returns list of summary dicts for table building.
    """
    results = []
    for loc_key, loc in locations.items():
        for tier_key, tier in HARDWARE_TIERS.items():
            hw = compute_hw_tco(tier, loc, n_racks, horizon_years)
            results.append({
                "location": loc.name,
                "location_key": loc_key,
                "tier": tier.name,
                "tier_key": tier_key,
                "currency": loc.currency,
                "initial_capex_m": round(hw["initial_capex_m"], 1),
                "hw_energy_10yr_m": round(hw["total_energy_m"], 1),
                "hw_refresh_10yr_m": round(hw["total_refresh_m"], 1),
                "hw_maint_10yr_m": round(hw["total_maintenance_m"], 1),
                "hw_tco_10yr_m": round(hw["total_hw_tco_m"], 1),
                "facility_tco_m": loc.mean_tco_10yr_millions,
                "combined_tco_m": round(hw["total_hw_tco_m"] + loc.mean_tco_10yr_millions, 1),
            })
    return results
