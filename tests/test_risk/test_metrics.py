"""Tests for risk metrics."""

import numpy as np
import pytest

from src.risk.metrics import compute_risk_metrics, risk_premium


class TestRiskMetrics:
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        dist = rng.normal(500, 40, size=100000)
        rm = compute_risk_metrics(dist)

        assert abs(rm.mean - 500) < 1.0
        assert abs(rm.std - 40) < 1.0
        assert rm.cv == pytest.approx(0.08, abs=0.01)
        assert rm.p95 > rm.mean
        assert rm.p5 < rm.mean
        assert rm.var_95 > 0
        assert rm.cvar_95 > rm.p95

    def test_low_variance(self):
        dist = np.full(1000, 100.0)
        rm = compute_risk_metrics(dist)
        assert rm.cv == 0.0
        assert rm.std == 0.0


class TestRiskPremium:
    def test_increased_risk(self):
        baseline = np.random.default_rng(42).normal(500, 40, 10000)
        shifted = np.random.default_rng(42).normal(550, 50, 10000)
        premium = risk_premium(baseline, shifted)

        assert premium["mean_increase"] > 0
        assert premium["mean_increase_pct"] > 0
