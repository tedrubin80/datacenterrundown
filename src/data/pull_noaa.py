#!/usr/bin/env python3
"""Pull historical climate data from NOAA Climate Data Online (CDO) API v2.

Pulls temperature, precipitation, and extreme weather data for weather stations
near each of our 10 datacenter locations.
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

API_TOKEN = os.environ.get("NOAA_API_TOKEN", "")
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "noaa"

# FIPS codes and coordinates for our locations
# NOAA uses FIPS state codes for US locations
LOCATIONS = {
    "evanston_wyoming": {"fips": "FIPS:56", "lat": 41.3, "lon": -110.9, "country": "US"},
    "salt_lake_city_utah": {"fips": "FIPS:49", "lat": 40.8, "lon": -111.9, "country": "US"},
    "des_moines_iowa": {"fips": "FIPS:19", "lat": 41.6, "lon": -93.6, "country": "US"},
    "florence_south_carolina": {"fips": "FIPS:45", "lat": 34.2, "lon": -79.8, "country": "US"},
    "atlanta_georgia": {"fips": "FIPS:13", "lat": 33.7, "lon": -84.4, "country": "US"},
    "boden_sweden": {"lat": 66.0, "lon": 21.7, "country": "SW"},
    "kristiansand_norway": {"lat": 58.1, "lon": 8.0, "country": "NO"},
    "iceland_reykjanes": {"lat": 63.8, "lon": -22.7, "country": "IC"},
    "sines_portugal": {"lat": 37.9, "lon": -8.9, "country": "PO"},
    "johor_malaysia": {"lat": 1.5, "lon": 103.7, "country": "MY"},
}

# Rate limit: 5 requests/sec, 10K/day
REQUEST_DELAY = 0.25  # seconds between requests


def _api_get(endpoint: str, params: dict = None) -> dict:
    """Make authenticated GET to NOAA CDO API."""
    if params is None:
        params = {}

    url = f"{BASE_URL}{endpoint}"
    if params:
        url += f"?{urlencode(params, doseq=True)}"

    req = Request(url)
    req.add_header("token", API_TOKEN)
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def find_stations(location_key: str, loc_info: dict) -> list:
    """Find nearest weather stations for a location."""
    print(f"  Finding stations near {location_key}...")

    # Search by extent (bounding box around coordinates)
    lat, lon = loc_info["lat"], loc_info["lon"]
    extent = f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}"

    result = _api_get("/stations", {
        "datasetid": "GHCND",
        "extent": extent,
        "limit": 10,
        "sortfield": "datacoverage",
        "sortorder": "desc",
    })
    time.sleep(REQUEST_DELAY)

    if "results" in result:
        stations = result["results"]
        print(f"    Found {len(stations)} stations")
        return stations
    else:
        print(f"    No stations found: {result.get('error', 'unknown')}")
        return []


def pull_daily_climate_data(station_id: str, location_key: str, start: str, end: str) -> list:
    """Pull daily temperature and precipitation data for a station."""
    all_data = []
    offset = 1

    while True:
        result = _api_get("/data", {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": "TMAX,TMIN,TAVG,PRCP,SNOW,AWND",
            "startdate": start,
            "enddate": end,
            "units": "metric",
            "limit": 1000,
            "offset": offset,
        })
        time.sleep(REQUEST_DELAY)

        if "results" in result:
            batch = result["results"]
            for row in batch:
                row["location_key"] = location_key
            all_data.extend(batch)

            metadata = result.get("metadata", {}).get("resultset", {})
            total = metadata.get("count", 0)
            if offset + 1000 > total:
                break
            offset += 1000
        else:
            break

    return all_data


def pull_storm_events_by_state(fips: str, location_key: str) -> list:
    """Pull storm event summaries for US states."""
    print(f"  Pulling storm events for {location_key}...")

    result = _api_get("/data", {
        "datasetid": "GHCND",
        "locationid": fips,
        "datatypeid": "PRCP,SNOW,AWND",
        "startdate": "2020-01-01",
        "enddate": "2024-12-31",
        "units": "metric",
        "limit": 1000,
    })
    time.sleep(REQUEST_DELAY)

    if "results" in result:
        data = result["results"]
        for row in data:
            row["location_key"] = location_key
        print(f"    Got {len(data)} records")
        return data
    return []


def pull_all_locations():
    """Pull climate data for all 10 locations."""
    print("\n--- Historical Climate Data (2015-2024) ---")
    all_climate = []
    all_stations = {}

    for loc_key, loc_info in LOCATIONS.items():
        print(f"\n  === {loc_key} ===")

        stations = find_stations(loc_key, loc_info)
        if not stations:
            continue

        # Use the station with best data coverage
        best_station = stations[0]
        station_id = best_station["id"]
        all_stations[loc_key] = {
            "station_id": station_id,
            "name": best_station.get("name", ""),
            "coverage": best_station.get("datacoverage", 0),
        }
        print(f"    Using: {station_id} ({best_station.get('name', '')})")

        # Pull in yearly chunks to stay within API limits
        for year in range(2015, 2025):
            print(f"    Pulling {year}...", end=" ", flush=True)
            data = pull_daily_climate_data(
                station_id, loc_key,
                f"{year}-01-01", f"{year}-12-31"
            )
            all_climate.extend(data)
            print(f"{len(data)} records")

    return all_climate, all_stations


def _save(data, filename: str):
    """Save to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved to {path}")


def main():
    if not API_TOKEN:
        print("ERROR: NOAA_API_TOKEN not set")
        return

    print("=" * 60)
    print("NOAA CLIMATE DATA PULL")
    print(f"Locations: {len(LOCATIONS)}")
    print("=" * 60)

    # Pull climate data
    climate_data, stations = pull_all_locations()
    _save(climate_data, "historical_climate_daily.json")
    _save(stations, "selected_stations.json")
    print(f"\nTotal climate records: {len(climate_data)}")

    # Pull storm data for US locations
    print("\n--- Storm/Precipitation Events (US States) ---")
    storm_data = []
    for loc_key, loc_info in LOCATIONS.items():
        if "fips" in loc_info:
            data = pull_storm_events_by_state(loc_info["fips"], loc_key)
            storm_data.extend(data)

    _save(storm_data, "storm_events.json")
    print(f"Total storm records: {len(storm_data)}")

    print("\n" + "=" * 60)
    print("NOAA PULL COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
