"""Climate scenario pathway visualizations."""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIO_COLORS = {"rcp26": "#2ecc71", "rcp45": "#f39c12", "rcp85": "#e74c3c"}
SCENARIO_LABELS = {
    "rcp26": "RCP 2.6 (Aggressive Mitigation)",
    "rcp45": "RCP 4.5 (Moderate Mitigation)",
    "rcp85": "RCP 8.5 (Business as Usual)",
}


def plot_temperature_pathways(
    projections_df: pd.DataFrame,
    location_key: str,
    output_path: Optional[str] = None,
):
    """Plot temperature trajectories for a location across RCP scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))
    loc_df = projections_df[projections_df["location_key"] == location_key]

    for scenario in loc_df["scenario"].unique():
        s_df = loc_df[loc_df["scenario"] == scenario].sort_values("year")
        color = SCENARIO_COLORS.get(scenario, "gray")
        label = SCENARIO_LABELS.get(scenario, scenario)
        ax.plot(s_df["year"], s_df["avg_temp_c"], color=color, label=label, linewidth=2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Temperature (°C)", fontsize=12)
    ax.set_title(f"Temperature Projections: {location_key}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pue_degradation(
    projections_df: pd.DataFrame,
    locations: list[str] = None,
    scenario: str = "rcp85",
    output_path: Optional[str] = None,
):
    """Plot PUE degradation over time for multiple locations under one scenario."""
    fig, ax = plt.subplots(figsize=(12, 7))
    s_df = projections_df[projections_df["scenario"] == scenario]

    if locations is None:
        locations = s_df["location_key"].unique()

    colors = plt.cm.tab10(np.linspace(0, 1, len(locations)))
    for loc, color in zip(locations, colors):
        loc_df = s_df[s_df["location_key"] == loc].sort_values("year")
        ax.plot(loc_df["year"], loc_df["projected_pue"], color=color, label=loc, linewidth=2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Projected PUE", fontsize=12)
    ax.set_title(f"PUE Degradation Under {SCENARIO_LABELS.get(scenario, scenario)}", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_extreme_events_trend(
    projections_df: pd.DataFrame,
    locations: list[str] = None,
    output_path: Optional[str] = None,
):
    """Plot extreme event frequency trends across scenarios (aggregated)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    if locations is None:
        locations = projections_df["location_key"].unique()[:5]

    for ax, scenario in zip(axes, ["rcp26", "rcp45", "rcp85"]):
        s_df = projections_df[projections_df["scenario"] == scenario]
        colors = plt.cm.tab10(np.linspace(0, 1, len(locations)))

        for loc, color in zip(locations, colors):
            loc_df = s_df[s_df["location_key"] == loc].sort_values("year")
            ax.plot(loc_df["year"], loc_df["extreme_event_freq"],
                    color=color, label=loc, linewidth=1.5, alpha=0.8)

        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=11)
        ax.set_xlabel("Year", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Extreme Events / Year", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Extreme Weather Event Frequency by Scenario", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
