"""Idea 5: Survival analysis for datacenter outage time-to-event prediction.

Models time between extreme weather events that cause datacenter outages,
and how climate change shifts the hazard function over time.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    from lifelines import WeibullAFTFitter, CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


class OutageSurvivalModel:
    """Predicts time-to-outage using survival analysis.

    Supports both Cox Proportional Hazards and Weibull AFT models.
    Climate features serve as covariates that modify the hazard/survival function.
    """

    def __init__(self, model_type: str = "weibull_aft", penalizer: float = 0.01):
        self.model_type = model_type
        self.penalizer = penalizer
        self.model = None
        self.feature_names = None

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "time_to_outage_days",
        event_col: str = "outage_occurred",
        feature_cols: Optional[list[str]] = None,
    ) -> dict:
        """Fit survival model.

        Args:
            df: DataFrame with duration, event indicator, and covariates.
            duration_col: Column with time-to-event (or censoring time).
            event_col: Binary column (1=event observed, 0=censored).
            feature_cols: Covariate columns. If None, uses all except duration/event.

        Returns:
            Dict with concordance index and model summary stats.
        """
        if not HAS_LIFELINES:
            return {"error": "lifelines not installed", "concordance": 0.5}

        if feature_cols is None:
            feature_cols = [
                c for c in df.columns if c not in (duration_col, event_col)
            ]
        self.feature_names = feature_cols

        fit_df = df[feature_cols + [duration_col, event_col]].dropna()

        if self.model_type == "weibull_aft":
            self.model = WeibullAFTFitter(penalizer=self.penalizer)
            self.model.fit(fit_df, duration_col=duration_col, event_col=event_col)
        elif self.model_type == "cox_ph":
            self.model = CoxPHFitter(penalizer=self.penalizer)
            self.model.fit(fit_df, duration_col=duration_col, event_col=event_col)

        return {
            "concordance": float(self.model.concordance_index_),
            "n_observations": len(fit_df),
            "n_events": int(fit_df[event_col].sum()),
        }

    def predict_survival(
        self, df: pd.DataFrame, times: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict survival function S(t|X) for given covariates.

        Returns DataFrame with survival probabilities at each time point.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if times is None:
            times = np.arange(1, 366)

        if self.model_type == "weibull_aft":
            return self.model.predict_survival_function(df, times=times)
        else:
            return self.model.predict_survival_function(df, times=times)

    def predict_median_survival(self, df: pd.DataFrame) -> np.ndarray:
        """Predict median time-to-outage for each observation."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_median(df).values.flatten()

    def predict_hazard(
        self, df: pd.DataFrame, times: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Predict cumulative hazard function H(t|X)."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if times is None:
            times = np.arange(1, 366)
        return self.model.predict_cumulative_hazard(df, times=times)

    def summary(self) -> Optional[pd.DataFrame]:
        """Return model coefficient summary."""
        if self.model is None:
            return None
        return self.model.summary


def prepare_survival_data(
    events_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    location_key: Optional[str] = None,
) -> pd.DataFrame:
    """Prepare survival-format data from event records and climate projections.

    Converts discrete event records into time-to-event format with
    climate covariates.
    """
    if location_key:
        events_df = events_df[events_df["location_key"] == location_key]
        climate_df = climate_df[climate_df["location_key"] == location_key]

    records = []
    for loc in events_df["location_key"].unique():
        loc_events = events_df[events_df["location_key"] == loc].sort_values("year")
        loc_climate = climate_df[climate_df["location_key"] == loc]

        years = sorted(loc_climate["year"].unique())
        for i, year in enumerate(years):
            climate_row = loc_climate[loc_climate["year"] == year].iloc[0]
            year_events = loc_events[loc_events["year"] == year]

            if len(year_events) > 0 and year_events["downtime_hours"].sum() > 0:
                outage_occurred = 1
                time_to_outage = max(1, 365 // max(1, len(year_events)))
            else:
                outage_occurred = 0
                time_to_outage = 365  # censored at year end

            records.append({
                "location_key": loc,
                "year": year,
                "time_to_outage_days": time_to_outage,
                "outage_occurred": outage_occurred,
                "avg_temp_c": climate_row.get("avg_temp_c", 0),
                "extreme_event_freq": climate_row.get("extreme_event_freq", 0),
                "projected_pue": climate_row.get("projected_pue", 1.1),
                "humidity_pct": climate_row.get("humidity_pct", 50),
                "cooling_degree_days": climate_row.get("cooling_degree_days", 0),
            })

    return pd.DataFrame(records)
