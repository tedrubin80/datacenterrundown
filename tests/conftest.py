"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.data.location_profiles import LocationProfile


@pytest.fixture
def sample_location():
    """Boden, Sweden test fixture."""
    return LocationProfile(
        key="boden_sweden",
        name="Boden, Sweden",
        tier="nordic",
        currency="EUR",
        latitude=66.0,
        longitude=21.7,
        capex_millions=(180, 220, 280),
        power_cost_mwh=(15.1, 22.5, 29.9),
        pue=(1.05, 1.09, 1.15),
        baseline_temp_c=1.5,
        baseline_humidity_pct=72.0,
        cooling_degree_days=10,
        renewable_pct=98,
        extreme_event_freq_annual=2,
        grid_reliability_score=0.95,
        mean_tco_10yr_millions=499.5,
        cv=0.073,
    )


@pytest.fixture
def sample_tco_distribution():
    """Pre-computed TCO distribution for testing."""
    rng = np.random.default_rng(42)
    return rng.normal(500, 40, size=10000)


@pytest.fixture
def sample_climate_df():
    """Small climate projection DataFrame for testing."""
    return pd.DataFrame({
        "year": list(range(2025, 2036)),
        "location_key": ["boden_sweden"] * 11,
        "scenario": ["rcp45"] * 11,
        "avg_temp_c": np.linspace(1.5, 3.0, 11),
        "cooling_degree_days": np.linspace(10, 25, 11).astype(int),
        "humidity_pct": [72.0] * 11,
        "extreme_event_freq": [2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5],
        "projected_pue": np.linspace(1.09, 1.12, 11),
        "pue_delta": np.linspace(0, 0.03, 11),
        "power_price_delta_pct": np.linspace(0, 10, 11),
    })
