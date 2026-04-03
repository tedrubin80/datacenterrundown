#!/usr/bin/env python3
"""Re-run only the failed pulls: FEMA declarations + NOAA extreme events."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# FEMA
print(">>> FEMA RE-PULL <<<")
from src.data.pull_fema import main as fema_main
fema_main()

# NOAA extreme events only (skip full climate re-pull)
print("\n>>> NOAA EXTREME EVENTS RE-PULL <<<")
import json
from src.data.pull_noaa import pull_extreme_precip_wind, _save, API_TOKEN

if not API_TOKEN:
    print("ERROR: NOAA_API_TOKEN not set")
else:
    stations_path = Path("data/raw/noaa/selected_stations.json")
    with open(stations_path) as f:
        stations = json.load(f)

    storm_data = []
    for loc_key, station_info in stations.items():
        data = pull_extreme_precip_wind(station_info["station_id"], loc_key)
        storm_data.extend(data)

    _save(storm_data, "extreme_events.json")
    print(f"Total extreme event records: {len(storm_data)}")

print("\n=== FIXES COMPLETE ===")
