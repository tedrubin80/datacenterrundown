"""TCO distribution and comparison visualizations."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


RESULTS_DIR = Path(__file__).parents[2] / "data" / "results"


def plot_tco_distributions(
    distributions: dict[str, np.ndarray],
    title: str = "TCO Distribution by Location",
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """Overlaid histograms of TCO distributions for multiple locations."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(distributions)))
    for (name, dist), color in zip(distributions.items(), colors):
        ax.hist(dist, bins=60, alpha=0.4, label=f"{name} (μ={np.mean(dist):.0f}M)", color=color)
        ax.axvline(np.mean(dist), color=color, linestyle="--", alpha=0.8)

    ax.set_xlabel("10-Year TCO (Millions)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_static_vs_dynamic(
    static_dist: np.ndarray,
    dynamic_dists: dict[str, np.ndarray],
    location_name: str,
    output_path: Optional[str] = None,
):
    """Compare static MC vs dynamic MC (per-RCP) for a single location."""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(static_dist, bins=60, alpha=0.5, label=f"Static MC (μ={np.mean(static_dist):.0f}M)",
            color="gray", edgecolor="black", linewidth=0.5)

    scenario_colors = {"rcp26": "#2ecc71", "rcp45": "#f39c12", "rcp85": "#e74c3c"}
    for scenario, dist in dynamic_dists.items():
        color = scenario_colors.get(scenario, "blue")
        ax.hist(dist, bins=60, alpha=0.4, label=f"{scenario.upper()} (μ={np.mean(dist):.0f}M)",
                color=color)

    ax.set_xlabel("10-Year TCO (Millions)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Static vs Climate-Dynamic TCO: {location_name}", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sensitivity_tornado(
    sensitivities: dict[str, float],
    title: str = "TCO Sensitivity Analysis",
    output_path: Optional[str] = None,
):
    """Tornado chart showing TCO sensitivity to each variable."""
    sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]))
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
    ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("TCO Change (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tco_trajectory(
    yearly_costs: dict[str, np.ndarray],
    location_name: str,
    output_path: Optional[str] = None,
):
    """Fan chart showing TCO accumulation over time with uncertainty bands."""
    fig, ax = plt.subplots(figsize=(12, 7))
    scenario_colors = {"rcp26": "#2ecc71", "rcp45": "#f39c12", "rcp85": "#e74c3c"}

    for scenario, costs in yearly_costs.items():
        # costs shape: (n_simulations, horizon_years)
        years = np.arange(1, costs.shape[1] + 1)
        cumulative = np.cumsum(costs, axis=1)
        median = np.median(cumulative, axis=0)
        p5 = np.percentile(cumulative, 5, axis=0)
        p95 = np.percentile(cumulative, 95, axis=0)

        color = scenario_colors.get(scenario, "blue")
        ax.plot(years, median, color=color, label=f"{scenario.upper()} median", linewidth=2)
        ax.fill_between(years, p5, p95, alpha=0.2, color=color)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative OpEx (Millions)", fontsize=12)
    ax.set_title(f"TCO Trajectory: {location_name}", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
