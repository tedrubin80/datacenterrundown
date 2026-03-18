"""Inject realistic correlations into independently-generated synthetic data.

SynGen produces independent marginals. This module uses a Gaussian copula
to impose a target correlation structure while preserving marginal distributions.
"""

import numpy as np
import pandas as pd
from scipy import stats


# Target rank-correlation matrix for climate-financial variables
# Order: temp, cdd, humidity, extreme_events, power_cost, pue, insurance, downtime
DEFAULT_CORRELATION_MATRIX = np.array([
    # temp   cdd    humid  events power  pue    insur  down
    [1.00,  0.85,  0.20,  0.35,  0.40,  0.55,  0.30,  0.25],  # temperature
    [0.85,  1.00,  0.15,  0.30,  0.45,  0.60,  0.25,  0.20],  # cooling degree days
    [0.20,  0.15,  1.00,  0.15,  0.05,  0.20,  0.10,  0.10],  # humidity
    [0.35,  0.30,  0.15,  1.00,  0.20,  0.15,  0.65,  0.70],  # extreme events
    [0.40,  0.45,  0.05,  0.20,  1.00,  0.30,  0.20,  0.10],  # power cost
    [0.55,  0.60,  0.20,  0.15,  0.30,  1.00,  0.15,  0.15],  # PUE
    [0.30,  0.25,  0.10,  0.65,  0.20,  0.15,  1.00,  0.55],  # insurance
    [0.25,  0.20,  0.10,  0.70,  0.10,  0.15,  0.55,  1.00],  # downtime
])

CORRELATED_COLUMNS = [
    "avg_temperature_c",
    "cooling_degree_days",
    "relative_humidity_pct",
    "storm_frequency_annual",
    "power_cost_mwh",
    "pue",
    "insurance_base_millions",
    "downtime_hours",
]


def _to_uniform(series: pd.Series) -> np.ndarray:
    """Transform a series to uniform [0,1] via empirical CDF (rank transform)."""
    ranks = series.rank(method="average")
    n = len(series)
    return (ranks - 0.5) / n


def _from_uniform(uniform_vals: np.ndarray, original: pd.Series) -> np.ndarray:
    """Map uniform values back to original marginal distribution via quantile function."""
    sorted_orig = np.sort(original.values)
    n = len(sorted_orig)
    indices = np.clip((uniform_vals * n).astype(int), 0, n - 1)
    return sorted_orig[indices]


def inject_correlations(
    df: pd.DataFrame,
    columns: list[str] = None,
    corr_matrix: np.ndarray = None,
) -> pd.DataFrame:
    """Apply Gaussian copula to impose correlation structure.

    Args:
        df: DataFrame with independently generated columns.
        columns: Which columns to correlate. Must match corr_matrix dimensions.
        corr_matrix: Target Spearman rank correlation matrix.

    Returns:
        DataFrame with same marginals but correlated joint distribution.
    """
    if columns is None:
        columns = [c for c in CORRELATED_COLUMNS if c in df.columns]
    if corr_matrix is None:
        # Use sub-matrix matching available columns
        idx = [CORRELATED_COLUMNS.index(c) for c in columns if c in CORRELATED_COLUMNS]
        corr_matrix = DEFAULT_CORRELATION_MATRIX[np.ix_(idx, idx)]

    n_vars = len(columns)
    n_rows = len(df)
    result = df.copy()

    # Step 1: Transform marginals to uniform via rank CDF
    uniform = np.zeros((n_rows, n_vars))
    for i, col in enumerate(columns):
        uniform[:, i] = _to_uniform(df[col])

    # Step 2: Transform uniform to standard normal
    normal = stats.norm.ppf(np.clip(uniform, 1e-6, 1 - 1e-6))

    # Step 3: Apply Cholesky decomposition of target correlation
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, find nearest PD matrix
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(corr_matrix, 1.0)
        L = np.linalg.cholesky(corr_matrix)

    correlated_normal = (L @ normal.T).T

    # Step 4: Transform back to uniform
    correlated_uniform = stats.norm.cdf(correlated_normal)

    # Step 5: Map back to original marginals
    for i, col in enumerate(columns):
        result[col] = _from_uniform(correlated_uniform[:, i], df[col])

    return result


def validate_correlations(
    df: pd.DataFrame,
    columns: list[str],
    target_corr: np.ndarray,
    tolerance: float = 0.1,
) -> dict:
    """Check that achieved correlations are within tolerance of targets."""
    achieved = df[columns].corr(method="spearman").values
    max_diff = np.max(np.abs(achieved - target_corr))
    mean_diff = np.mean(np.abs(achieved - target_corr))
    return {
        "max_deviation": float(max_diff),
        "mean_deviation": float(mean_diff),
        "within_tolerance": bool(max_diff < tolerance),
        "achieved_corr": achieved,
    }
