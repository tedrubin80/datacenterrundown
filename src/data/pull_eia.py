#!/usr/bin/env python3
"""Pull electricity price and generation data from EIA API v2.

Targets the 10 datacenter locations' states:
- Sweden/Norway/Iceland/Portugal/Malaysia: N/A (international)
- Wyoming, Utah, Iowa, South Carolina, Georgia: US state-level data
"""

import os
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

ENV_PATH = Path(__file__).parents[2] / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

API_KEY = os.environ.get("EIA_API_KEY", "")
BASE_URL = "https://api.eia.gov/v2"
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "eia"

# Map our locations to EIA state codes
LOCATION_STATES = {
    "evanston_wyoming": "WY",
    "salt_lake_city_utah": "UT",
    "des_moines_iowa": "IA",
    "florence_south_carolina": "SC",
    "atlanta_georgia": "GA",
}

# Also pull national-level data for context
ALL_STATES = list(LOCATION_STATES.values())


def _api_get(endpoint: str, params: dict = None) -> dict:
    """Make authenticated GET request to EIA API."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY

    url = f"{BASE_URL}{endpoint}?{urlencode(params, doseq=True)}"
    req = Request(url)
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def pull_retail_electricity_prices():
    """Pull retail electricity prices by state (commercial sector).

    This is the closest proxy for datacenter power costs.
    """
    print("\n--- Retail Electricity Prices (Commercial) ---")
    all_data = []

    for loc_name, state in LOCATION_STATES.items():
        print(f"  Pulling {state} ({loc_name})...")
        result = _api_get("/electricity/retail-sales/data/", {
            "frequency": "annual",
            "data[0]": "price",
            "facets[stateid][]": state,
            "facets[sectorid][]": "COM",  # Commercial sector
            "start": "2015",
            "end": "2025",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5000,
        })

        if "response" in result and "data" in result["response"]:
            rows = result["response"]["data"]
            for row in rows:
                row["location_key"] = loc_name
            all_data.extend(rows)
            print(f"    Got {len(rows)} records")
        else:
            print(f"    No data: {result.get('error', 'unknown')}")

        time.sleep(0.3)  # Rate limit respect

    _save(all_data, "retail_electricity_prices.json")
    return all_data


def pull_electricity_generation():
    """Pull electricity generation by fuel type per state.

    Shows renewable vs fossil fuel mix for each location.
    """
    print("\n--- Electricity Generation by Source ---")
    all_data = []

    for loc_name, state in LOCATION_STATES.items():
        print(f"  Pulling {state} ({loc_name})...")
        result = _api_get("/electricity/electric-power-operational-data/data/", {
            "frequency": "annual",
            "data[0]": "generation",
            "facets[location][]": state,
            "start": "2015",
            "end": "2025",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5000,
        })

        if "response" in result and "data" in result["response"]:
            rows = result["response"]["data"]
            for row in rows:
                row["location_key"] = loc_name
            all_data.extend(rows)
            print(f"    Got {len(rows)} records")
        else:
            print(f"    No data: {result.get('error', 'unknown')}")

        time.sleep(0.3)

    _save(all_data, "electricity_generation.json")
    return all_data


def pull_average_retail_prices_all_states():
    """Pull average retail prices for ALL US states for broader comparison."""
    print("\n--- Average Retail Prices (All States, Industrial) ---")
    result = _api_get("/electricity/retail-sales/data/", {
        "frequency": "annual",
        "data[0]": "price",
        "facets[sectorid][]": "IND",  # Industrial (closer to DC rates)
        "start": "2018",
        "end": "2025",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 5000,
    })

    if "response" in result and "data" in result["response"]:
        data = result["response"]["data"]
        print(f"  Got {len(data)} records across all states")
        _save(data, "retail_prices_all_states_industrial.json")
        return data
    else:
        print(f"  No data: {result.get('error', 'unknown')}")
        return []


def pull_consumption_data():
    """Pull electricity consumption by state."""
    print("\n--- Electricity Consumption ---")
    all_data = []

    for loc_name, state in LOCATION_STATES.items():
        print(f"  Pulling {state} ({loc_name})...")
        result = _api_get("/electricity/retail-sales/data/", {
            "frequency": "annual",
            "data[0]": "sales",
            "data[1]": "customers",
            "data[2]": "revenue",
            "facets[stateid][]": state,
            "start": "2015",
            "end": "2025",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5000,
        })

        if "response" in result and "data" in result["response"]:
            rows = result["response"]["data"]
            for row in rows:
                row["location_key"] = loc_name
            all_data.extend(rows)
            print(f"    Got {len(rows)} records")
        else:
            print(f"    No data: {result.get('error', 'unknown')}")

        time.sleep(0.3)

    _save(all_data, "electricity_consumption.json")
    return all_data


def _save(data: list, filename: str):
    """Save data to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved to {path}")


def main():
    if not API_KEY:
        print("ERROR: EIA_API_KEY not set")
        return

    print("=" * 60)
    print("EIA DATA PULL")
    print(f"States: {ALL_STATES}")
    print("=" * 60)

    pull_retail_electricity_prices()
    pull_electricity_generation()
    pull_average_retail_prices_all_states()
    pull_consumption_data()

    print("\n" + "=" * 60)
    print("EIA PULL COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
