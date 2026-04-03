#!/usr/bin/env python3
"""Pull disaster declaration data from OpenFEMA API.

No API key required. Pulls historical disaster data relevant to
datacenter risk assessment (Idea 5).
"""

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote

BASE_URL = "https://www.fema.gov/api/open/v2"
OUTPUT_DIR = Path(__file__).parents[2] / "data" / "raw" / "fema"

# Our US states
TARGET_STATES = {
    "evanston_wyoming": "Wyoming",
    "salt_lake_city_utah": "Utah",
    "des_moines_iowa": "Iowa",
    "florence_south_carolina": "South Carolina",
    "atlanta_georgia": "Georgia",
}

# Disaster types relevant to datacenters
RELEVANT_TYPES = [
    "Hurricane", "Severe Storm", "Tornado", "Flood", "Ice Storm",
    "Snow", "Fire", "Earthquake", "Severe Ice Storm", "Tropical Storm",
    "Coastal Storm", "Typhoon",
]


def _api_get(endpoint: str, params: str = "") -> dict:
    """Make GET request to OpenFEMA API."""
    url = f"{BASE_URL}{endpoint}"
    if params:
        url += f"?{quote(params, safe='=&$(),')}"

    req = Request(url)
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e)}


def pull_disaster_declarations():
    """Pull all disaster declarations for target states since 2000."""
    print("\n--- Disaster Declarations (2000-2025) ---")
    all_data = []

    for loc_key, state_name in TARGET_STATES.items():
        print(f"\n  {state_name} ({loc_key})...")
        skip = 0
        state_total = 0

        while True:
            filter_str = (
                f"$filter=state eq '{state_name}' and declarationDate ge '2000-01-01T00:00:00.000z'"
                f"&$top=1000&$skip={skip}"
                f"&$orderby=declarationDate desc"
            )

            result = _api_get("/DisasterDeclarationsSummaries", filter_str)
            time.sleep(0.3)

            if "DisasterDeclarationsSummaries" in result:
                records = result["DisasterDeclarationsSummaries"]
                if not records:
                    break

                for rec in records:
                    rec["location_key"] = loc_key

                all_data.extend(records)
                state_total += len(records)
                skip += 1000

                if len(records) < 1000:
                    break
            else:
                print(f"    Error or no data: {result.get('error', 'empty')}")
                break

        print(f"    Total: {state_total} declarations")

    _save(all_data, "disaster_declarations.json")
    return all_data


def pull_public_assistance():
    """Pull FEMA Public Assistance funded projects for damage cost data."""
    print("\n--- Public Assistance (Damage Costs) ---")
    all_data = []

    for loc_key, state_name in TARGET_STATES.items():
        print(f"\n  {state_name} ({loc_key})...")

        filter_str = (
            f"$filter=state eq '{state_name}' and declarationDate ge '2015-01-01T00:00:00.000z'"
            f"&$top=1000"
            f"&$select=disasterNumber,state,county,declarationDate,incidentType,"
            f"projectAmount,federalShareObligated,totalObligated"
            f"&$orderby=declarationDate desc"
        )

        result = _api_get("/PublicAssistanceFundedProjectsDetails", filter_str)
        time.sleep(0.3)

        if "PublicAssistanceFundedProjectsDetails" in result:
            records = result["PublicAssistanceFundedProjectsDetails"]
            for rec in records:
                rec["location_key"] = loc_key
            all_data.extend(records)
            print(f"    Got {len(records)} project records")
        else:
            print(f"    No data")

    _save(all_data, "public_assistance_projects.json")
    return all_data


def pull_hazard_mitigation():
    """Pull hazard mitigation grant data."""
    print("\n--- Hazard Mitigation Grants ---")
    all_data = []

    for loc_key, state_name in TARGET_STATES.items():
        print(f"\n  {state_name} ({loc_key})...")

        filter_str = (
            f"$filter=state eq '{state_name}'"
            f"&$top=1000"
            f"&$orderby=dateApproved desc"
        )

        result = _api_get("/HazardMitigationGrants", filter_str)
        time.sleep(0.3)

        if "HazardMitigationGrants" in result:
            records = result["HazardMitigationGrants"]
            for rec in records:
                rec["location_key"] = loc_key
            all_data.extend(records)
            print(f"    Got {len(records)} grants")
        else:
            print(f"    No data")

    _save(all_data, "hazard_mitigation_grants.json")
    return all_data


def _save(data: list, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved to {path} ({len(data)} records)")


def main():
    print("=" * 60)
    print("FEMA DISASTER DATA PULL (No API Key Required)")
    print(f"States: {list(TARGET_STATES.values())}")
    print("=" * 60)

    declarations = pull_disaster_declarations()
    assistance = pull_public_assistance()
    mitigation = pull_hazard_mitigation()

    # Summary
    print("\n" + "=" * 60)
    print("FEMA PULL COMPLETE")
    print(f"  Declarations: {len(declarations)}")
    print(f"  PA Projects:  {len(assistance)}")
    print(f"  HM Grants:    {len(mitigation)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
