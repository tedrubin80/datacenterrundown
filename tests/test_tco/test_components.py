"""Tests for TCO component calculations."""

import numpy as np
import pytest

from src.tco.components import (
    TCOParams,
    compute_annual_power_cost,
    compute_annual_opex,
    compute_tco,
    sample_triangular,
)


class TestPowerCost:
    def test_basic_calculation(self):
        # 10MW * 0.75 util * 1.1 PUE * $50/MWh * 8760h = $361.89M... in raw units
        cost = compute_annual_power_cost(10, 0.75, 1.1, 50)
        expected = 10 * 0.75 * 1.1 * 50 * 8760 / 1e6
        assert abs(cost - expected) < 0.01

    def test_zero_utilization(self):
        assert compute_annual_power_cost(10, 0.0, 1.1, 50) == 0.0

    def test_pue_impact(self):
        cost_low = compute_annual_power_cost(10, 0.75, 1.05, 50)
        cost_high = compute_annual_power_cost(10, 0.75, 1.45, 50)
        assert cost_high > cost_low


class TestTCO:
    def test_tco_positive(self):
        params = TCOParams(capex_millions=220, power_cost_mwh=22.5, pue=1.09)
        tco = compute_tco(params, horizon_years=10)
        assert tco > 220  # Must be at least CapEx

    def test_longer_horizon_costs_more(self):
        params = TCOParams(capex_millions=220, power_cost_mwh=22.5, pue=1.09)
        tco_10 = compute_tco(params, horizon_years=10)
        tco_15 = compute_tco(params, horizon_years=15)
        assert tco_15 > tco_10

    def test_incentive_reduces_tco(self):
        params_no_incentive = TCOParams(capex_millions=220, power_cost_mwh=22.5, pue=1.09)
        params_incentive = TCOParams(capex_millions=220, power_cost_mwh=22.5, pue=1.09,
                                     tax_incentive_pct=10.0)
        assert compute_tco(params_incentive) < compute_tco(params_no_incentive)


class TestSampling:
    def test_triangular_in_range(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            val = sample_triangular((10, 20, 30), rng)
            assert 10 <= val <= 30
