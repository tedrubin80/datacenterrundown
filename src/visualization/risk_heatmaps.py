"""Risk heatmap visualizations: location x scenario matrices."""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_risk_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = "cv",
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    cmap: str = "YlOrRd",
):
    """Heatmap of risk metric across locations and scenarios.

    Args:
        comparison_df: DataFrame with columns: location, scenario, and the metric.
        metric: Which metric column to plot.
        title: Plot title.
        output_path: Save path.
    """
    pivot = comparison_df.pivot(index="location", columns="scenario", values=metric)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color="white" if val > pivot.values.mean() else "black", fontsize=9)

    plt.colorbar(im, ax=ax, label=metric.upper())
    ax.set_title(title or f"Risk Heatmap: {metric.upper()} by Location and Scenario", fontsize=13)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ranking_changes(
    shift_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """Bump chart showing how location rankings change across scenarios."""
    fig, ax = plt.subplots(figsize=figsize)

    locations = shift_df["location"].unique()
    scenarios = sorted(shift_df["scenario"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(locations)))

    for loc, color in zip(locations, colors):
        loc_data = shift_df[shift_df["location"] == loc]
        baseline_rank = loc_data["baseline_rank"].iloc[0]

        x_vals = [0]
        y_vals = [baseline_rank]
        for i, scenario in enumerate(scenarios):
            row = loc_data[loc_data["scenario"] == scenario]
            if len(row) > 0:
                x_vals.append(i + 1)
                y_vals.append(row["scenario_rank"].iloc[0])

        ax.plot(x_vals, y_vals, "o-", color=color, label=loc, linewidth=2, markersize=8)

    ax.set_xticks(range(len(scenarios) + 1))
    ax.set_xticklabels(["Static Baseline"] + [s.upper() for s in scenarios], fontsize=10)
    ax.set_ylabel("Rank (1 = best)", fontsize=12)
    ax.set_title("Location Ranking Changes Across Climate Scenarios", fontsize=14)
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
