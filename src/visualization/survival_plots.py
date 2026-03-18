"""Survival analysis and insurance visualizations for Idea 5."""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_survival_curves(
    survival_functions: dict[str, pd.DataFrame],
    title: str = "Datacenter Outage Survival Curves",
    output_path: Optional[str] = None,
):
    """Plot Kaplan-Meier-style survival curves per location.

    Args:
        survival_functions: Dict mapping location name to DataFrame
            with time index and survival probability columns.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(survival_functions)))

    for (name, sf), color in zip(survival_functions.items(), colors):
        if isinstance(sf, pd.DataFrame) and len(sf.columns) > 0:
            col = sf.columns[0]
            ax.plot(sf.index, sf[col], label=name, color=color, linewidth=2)

    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Survival Probability S(t)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="50% survival")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hazard_evolution(
    hazard_by_scenario: dict[str, np.ndarray],
    years: np.ndarray,
    location_name: str,
    output_path: Optional[str] = None,
):
    """Plot how hazard rate evolves over time under different climate scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))
    scenario_colors = {"rcp26": "#2ecc71", "rcp45": "#f39c12", "rcp85": "#e74c3c"}

    for scenario, hazard in hazard_by_scenario.items():
        color = scenario_colors.get(scenario, "blue")
        ax.plot(years, hazard, color=color, label=scenario.upper(), linewidth=2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual Outage Hazard Rate", fontsize=12)
    ax.set_title(f"Outage Hazard Evolution: {location_name}", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_insurance_trajectories(
    premium_df: pd.DataFrame,
    location_name: str,
    output_path: Optional[str] = None,
):
    """Plot insurance premium trajectories with quantile uncertainty bands.

    Args:
        premium_df: DataFrame with year, premium_predicted, and quantile columns.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    years = premium_df["year"]
    ax.plot(years, premium_df["premium_predicted"], "b-", linewidth=2, label="Predicted Premium")

    if "q0.05" in premium_df.columns and "q0.95" in premium_df.columns:
        ax.fill_between(years, premium_df["q0.05"], premium_df["q0.95"],
                        alpha=0.15, color="blue", label="90% PI")
    if "q0.25" in premium_df.columns and "q0.75" in premium_df.columns:
        ax.fill_between(years, premium_df["q0.25"], premium_df["q0.75"],
                        alpha=0.3, color="blue", label="50% PI")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual Premium (Millions)", fontsize=12)
    ax.set_title(f"Insurance Premium Trajectory: {location_name}", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
