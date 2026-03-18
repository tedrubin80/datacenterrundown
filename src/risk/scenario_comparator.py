"""Cross-scenario and cross-location comparison of risk-adjusted TCO."""

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import compute_risk_metrics, RiskMetrics


def build_comparison_table(
    results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Build a location x scenario comparison table.

    Args:
        results: Nested dict of {location: {scenario: tco_distribution}}.

    Returns:
        DataFrame with columns: location, scenario, mean, std, cv, var_95, cvar_95, p5, p95.
    """
    rows = []
    for loc, scenarios in results.items():
        for scenario, dist in scenarios.items():
            rm = compute_risk_metrics(dist)
            rows.append({
                "location": loc,
                "scenario": scenario,
                "mean_tco": round(rm.mean, 2),
                "std_tco": round(rm.std, 2),
                "cv": round(rm.cv, 4),
                "var_95": round(rm.var_95, 2),
                "cvar_95": round(rm.cvar_95, 2),
                "p5": round(rm.p5, 2),
                "p95": round(rm.p95, 2),
            })

    return pd.DataFrame(rows)


def rank_locations(
    comparison_df: pd.DataFrame,
    metric: str = "mean_tco",
    ascending: bool = True,
) -> pd.DataFrame:
    """Rank locations by a metric within each scenario.

    Returns DataFrame with added 'rank' column.
    """
    df = comparison_df.copy()
    df["rank"] = df.groupby("scenario")[metric].rank(ascending=ascending).astype(int)
    return df.sort_values(["scenario", "rank"])


def ranking_shift_analysis(
    comparison_df: pd.DataFrame,
    baseline_scenario: str = "static",
    metric: str = "mean_tco",
) -> pd.DataFrame:
    """Analyze how location rankings change from baseline to climate scenarios.

    Returns DataFrame showing rank changes per location per scenario.
    """
    ranked = rank_locations(comparison_df, metric=metric)

    baseline_ranks = ranked[ranked["scenario"] == baseline_scenario].set_index("location")["rank"]

    shifts = []
    for scenario in ranked["scenario"].unique():
        if scenario == baseline_scenario:
            continue
        scenario_ranks = ranked[ranked["scenario"] == scenario].set_index("location")["rank"]
        for loc in baseline_ranks.index:
            if loc in scenario_ranks.index:
                shifts.append({
                    "location": loc,
                    "scenario": scenario,
                    "baseline_rank": int(baseline_ranks[loc]),
                    "scenario_rank": int(scenario_ranks[loc]),
                    "rank_change": int(baseline_ranks[loc] - scenario_ranks[loc]),
                })

    return pd.DataFrame(shifts)


def climate_cost_premium_table(
    static_results: dict[str, np.ndarray],
    dynamic_results: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Compute climate cost premium: how much extra does each scenario cost?

    Args:
        static_results: {location: static_tco_distribution}
        dynamic_results: {location: {scenario: dynamic_tco_distribution}}

    Returns:
        DataFrame with absolute and percentage cost premiums.
    """
    rows = []
    for loc in static_results:
        static_mean = float(np.mean(static_results[loc]))
        for scenario, dist in dynamic_results.get(loc, {}).items():
            dynamic_mean = float(np.mean(dist))
            rows.append({
                "location": loc,
                "scenario": scenario,
                "static_mean_tco": round(static_mean, 2),
                "dynamic_mean_tco": round(dynamic_mean, 2),
                "climate_premium": round(dynamic_mean - static_mean, 2),
                "climate_premium_pct": round(
                    (dynamic_mean - static_mean) / static_mean * 100, 2
                ) if static_mean else 0,
            })

    return pd.DataFrame(rows)
