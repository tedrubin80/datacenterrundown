"""Load and process EIA electricity price data for TCO calibration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

EIA_DIR = Path(__file__).parents[2] / "data" / "raw" / "eia"

STATE_TO_LOCATION = {
    "WY": "evanston_wyoming",
    "UT": "salt_lake_city_utah",
    "IA": "des_moines_iowa",
    "SC": "florence_south_carolina",
    "GA": "atlanta_georgia",
}


def load_retail_prices(path: str = None) -> pd.DataFrame:
    """Load EIA retail electricity prices."""
    if path is None:
        path = EIA_DIR / "retail_electricity_prices.json"
    if not path.exists():
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        # EIA reports in cents/kWh; convert to $/MWh
        df["price_usd_mwh"] = df["price"] * 10
    if "period" in df.columns:
        df["year"] = pd.to_numeric(df["period"], errors="coerce")
    return df


def load_industrial_prices(path: str = None) -> pd.DataFrame:
    """Load industrial prices (closer to datacenter rates)."""
    if path is None:
        path = EIA_DIR / "retail_prices_all_states_industrial.json"
    if not path.exists():
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["price_usd_mwh"] = df["price"] * 10
    if "period" in df.columns:
        df["year"] = pd.to_numeric(df["period"], errors="coerce")
    if "stateid" in df.columns:
        df["location_key"] = df["stateid"].map(STATE_TO_LOCATION)
    return df


def get_price_comparison() -> pd.DataFrame:
    """Compare real EIA prices vs paper assumptions for our locations."""
    from .location_profiles import load_locations

    prices = load_retail_prices()
    if prices.empty:
        return pd.DataFrame()

    locations = load_locations()
    rows = []

    for loc_key in STATE_TO_LOCATION.values():
        loc = locations.get(loc_key)
        if not loc:
            continue

        loc_prices = prices[prices.get("location_key") == loc_key]
        if loc_prices.empty:
            continue

        recent = loc_prices[loc_prices["year"] >= 2020]
        avg_real = recent["price_usd_mwh"].mean() if len(recent) > 0 else None
        trend = None
        if len(loc_prices) >= 3:
            sorted_p = loc_prices.sort_values("year")
            years = sorted_p["year"].values
            vals = sorted_p["price_usd_mwh"].values
            valid = ~np.isnan(vals)
            if valid.sum() >= 3:
                trend = float(np.polyfit(years[valid], vals[valid], 1)[0])

        rows.append({
            "location": loc.name,
            "location_key": loc_key,
            "paper_min_mwh": loc.power_cost_mwh[0],
            "paper_mode_mwh": loc.power_cost_mwh[1],
            "paper_max_mwh": loc.power_cost_mwh[2],
            "real_avg_mwh": round(avg_real, 2) if avg_real else None,
            "annual_trend_mwh": round(trend, 2) if trend else None,
            "n_years": len(loc_prices),
        })

    return pd.DataFrame(rows)


def get_price_trajectories() -> pd.DataFrame:
    """Get year-by-year price data for our locations."""
    prices = load_retail_prices()
    if prices.empty:
        return pd.DataFrame()

    result = prices[prices["location_key"].notna()][
        ["location_key", "year", "price_usd_mwh"]
    ].dropna().sort_values(["location_key", "year"])

    return result
