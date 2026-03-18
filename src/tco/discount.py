"""Net present value and discounting utilities."""

import numpy as np
from typing import Sequence


def npv(cashflows: Sequence[float], rate: float = 0.07) -> float:
    """Compute net present value of a cashflow series.

    cashflows[0] is at t=0 (undiscounted).
    """
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / (1 + rate) ** t
    return total


def tco_npv(
    capex: float,
    annual_opex: Sequence[float],
    rate: float = 0.07,
) -> float:
    """Compute TCO as NPV of CapEx + annual OpEx stream."""
    cashflows = [capex] + list(annual_opex)
    return npv(cashflows, rate)


def real_to_nominal(real_rate: float, inflation: float) -> float:
    """Convert real discount rate to nominal (Fisher equation)."""
    return (1 + real_rate) * (1 + inflation) - 1


def annuity_factor(rate: float, years: int) -> float:
    """Present value annuity factor: (1 - (1+r)^-n) / r."""
    if rate == 0:
        return float(years)
    return (1 - (1 + rate) ** (-years)) / rate
