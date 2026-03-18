"""Risk metrics from DCCore.pdf: VaR, CVaR/ES, Coefficient of Variation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Complete risk profile for a TCO distribution."""

    mean: float
    std: float
    median: float
    var_95: float         # Value at Risk at 95%
    cvar_95: float        # Conditional VaR (Expected Shortfall)
    cv: float             # Coefficient of Variation
    p5: float
    p95: float
    iqr: float


def compute_risk_metrics(distribution: np.ndarray, alpha: float = 0.95) -> RiskMetrics:
    """Compute full risk metrics for a TCO distribution.

    Args:
        distribution: Array of simulated TCO values.
        alpha: Confidence level for VaR/CVaR (default 0.95).
    """
    mean = float(np.mean(distribution))
    std = float(np.std(distribution))
    median = float(np.median(distribution))
    p5 = float(np.percentile(distribution, 5))
    p95 = float(np.percentile(distribution, alpha * 100))
    var_95 = p95 - mean
    tail = distribution[distribution >= p95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else p95
    cv = std / mean if mean != 0 else 0.0
    iqr = float(np.percentile(distribution, 75) - np.percentile(distribution, 25))

    return RiskMetrics(
        mean=mean, std=std, median=median,
        var_95=var_95, cvar_95=cvar_95, cv=cv,
        p5=p5, p95=p95, iqr=iqr,
    )


def compare_scenarios(
    distributions: dict[str, np.ndarray],
    alpha: float = 0.95,
) -> dict[str, RiskMetrics]:
    """Compute risk metrics for multiple scenario distributions.

    Args:
        distributions: Dict mapping scenario name to TCO distribution array.

    Returns:
        Dict mapping scenario name to RiskMetrics.
    """
    return {name: compute_risk_metrics(dist, alpha) for name, dist in distributions.items()}


def risk_premium(baseline_dist: np.ndarray, shifted_dist: np.ndarray) -> dict:
    """Compute the additional risk premium from climate scenarios.

    Returns dict with absolute and percentage increases in mean TCO,
    VaR, and CVaR.
    """
    base = compute_risk_metrics(baseline_dist)
    shifted = compute_risk_metrics(shifted_dist)

    return {
        "mean_increase": shifted.mean - base.mean,
        "mean_increase_pct": (shifted.mean - base.mean) / base.mean * 100 if base.mean else 0,
        "var_increase": shifted.var_95 - base.var_95,
        "cvar_increase": shifted.cvar_95 - base.cvar_95,
        "cv_change": shifted.cv - base.cv,
    }
