"""Idea 5: Insurance premium prediction with quantile regression.

Models how climate change drives insurance cost escalation for
datacenter facilities, with uncertainty bands via quantile regression.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class InsurancePremiumModel:
    """Quantile regression for insurance premium prediction.

    Trains separate GBM models for different quantiles to produce
    prediction intervals (e.g., 5th, 25th, 50th, 75th, 95th percentile).
    """

    def __init__(
        self,
        quantiles: list[float] = None,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.08,
        seed: int = 42,
    ):
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.models: dict[float, GradientBoostingRegressor] = {}
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Train quantile regression models.

        Returns dict with metrics per quantile.
        """
        self.feature_names = feature_names
        metrics = {}

        for q in self.quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            self.models[q] = model

            if X_val is not None and y_val is not None:
                pred = model.predict(X_val)
                metrics[f"q{q:.2f}_mae"] = float(mean_absolute_error(y_val, pred))

        # Also train mean model for point predictions
        mean_model = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )
        mean_model.fit(X_train, y_train)
        self.models["mean"] = mean_model

        if X_val is not None and y_val is not None:
            mean_pred = mean_model.predict(X_val)
            metrics["mean_rmse"] = float(np.sqrt(mean_squared_error(y_val, mean_pred)))
            metrics["mean_mae"] = float(mean_absolute_error(y_val, mean_pred))

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction (median or mean)."""
        if "mean" in self.models:
            return self.models["mean"].predict(X)
        if 0.50 in self.models:
            return self.models[0.50].predict(X)
        raise RuntimeError("No model available for point prediction.")

    def predict_quantiles(self, X: np.ndarray) -> pd.DataFrame:
        """Predict all quantiles, returning a DataFrame.

        Columns are quantile values (e.g., 0.05, 0.25, 0.50, 0.75, 0.95).
        """
        results = {}
        for q in self.quantiles:
            if q in self.models:
                results[f"q{q:.2f}"] = self.models[q].predict(X)
        return pd.DataFrame(results)

    def predict_interval(
        self, X: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict lower and upper bounds of prediction interval."""
        lower = self.models[lower_q].predict(X) if lower_q in self.models else None
        upper = self.models[upper_q].predict(X) if upper_q in self.models else None
        return lower, upper

    def premium_trajectory(
        self,
        climate_trajectory: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Predict insurance premium trajectory over time with uncertainty bands.

        Args:
            climate_trajectory: DataFrame with year-by-year climate features.
            feature_cols: Which columns to use as features.

        Returns:
            DataFrame with year, predicted premium, and quantile bounds.
        """
        X = climate_trajectory[feature_cols].values
        quantile_preds = self.predict_quantiles(X)
        result = pd.DataFrame({"year": climate_trajectory["year"].values})
        result["premium_predicted"] = self.predict(X)
        for col in quantile_preds.columns:
            result[col] = quantile_preds[col].values
        return result
