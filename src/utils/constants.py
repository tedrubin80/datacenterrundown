"""Project-wide constants derived from DCCore.pdf."""

DISCOUNT_RATE = 0.07
INFLATION_RATE = 0.025
HOURS_PER_YEAR = 8760
VAR_CONFIDENCE = 0.95
EUR_USD_RATE = 1.10

LOCATION_IDS = {
    0: "boden_sweden",
    1: "kristiansand_norway",
    2: "iceland_reykjanes",
    3: "evanston_wyoming",
    4: "salt_lake_city_utah",
    5: "des_moines_iowa",
    6: "florence_south_carolina",
    7: "atlanta_georgia",
    8: "johor_malaysia",
    9: "sines_portugal",
}

LOCATION_TIERS = {
    "nordic": ["boden_sweden", "kristiansand_norway", "iceland_reykjanes"],
    "us_secondary": ["evanston_wyoming", "salt_lake_city_utah", "des_moines_iowa", "florence_south_carolina"],
    "traditional": ["atlanta_georgia"],
    "emerging": ["johor_malaysia", "sines_portugal"],
}

EVENT_TYPES = {
    0: "heatwave",
    1: "hurricane_typhoon",
    2: "flood",
    3: "ice_storm",
    4: "wildfire",
    5: "grid_failure",
}

RCP_SCENARIOS = ["rcp26", "rcp45", "rcp85"]
