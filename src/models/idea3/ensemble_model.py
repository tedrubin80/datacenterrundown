"""Idea 3: Ensemble model (XGBoost + Random Forest) for climate-to-TCO mapping.

Predicts how climate variables shift TCO component distributions.
Ensemble spread provides uncertainty estimates.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class EnsembleTCOModel:
    """Ensemble of gradient-boosted + random forest models for TCO prediction.

    Trains separate sub-models for each TCO shift target:
    - power_cost_shift (mean delta)
    - pue_shift (mean delta)
    - insurance_shift (scale factor)

    Ensemble disagreement provides epistemic uncertainty.
    """

    TARGET_NAMES = ["power_cost_shift", "pue_shift", "insurance_scale"]

    def __init__(
        self,
        n_estimators_xgb: int = 100,
        max_depth_xgb: int = 4,
        learning_rate_xgb: float = 0.1,
        n_estimators_rf: int = 100,
        max_depth_rf: int = 8,
        seed: int = 42,
    ):
        self.seed = seed
        self.models: dict[str, list[BaseEstimator]] = {}
        self.xgb_params = {
            "n_estimators": n_estimators_xgb,
            "max_depth": max_depth_xgb,
            "learning_rate": learning_rate_xgb,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
        }
        self.rf_params = {
            "n_estimators": n_estimators_rf,
            "max_depth": max_depth_rf,
            "min_samples_leaf": 5,
            "random_state": seed,
            "n_jobs": -1,
        }

    def fit(
        self,
        X: np.ndarray,
        y_dict: dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val_dict: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, float]:
        """Train ensemble for each target variable.

        Args:
            X: Feature matrix (n_samples, n_features).
            y_dict: Dict mapping target name to target array.
            X_val: Optional validation features.
            y_val_dict: Optional validation targets.

        Returns:
            Dict of validation R^2 scores per target.
        """
        scores = {}

        for target_name in self.TARGET_NAMES:
            if target_name not in y_dict:
                continue

            y = y_dict[target_name]
            members = []

            # XGBoost member
            if HAS_XGB:
                xgb_model = xgb.XGBRegressor(**self.xgb_params)
                fit_params = {}
                if X_val is not None and y_val_dict and target_name in y_val_dict:
                    fit_params["eval_set"] = [(X_val, y_val_dict[target_name])]
                    fit_params["verbose"] = False
                xgb_model.fit(X, y, **fit_params)
                members.append(xgb_model)

            # Random Forest member
            rf_model = RandomForestRegressor(**self.rf_params)
            rf_model.fit(X, y)
            members.append(rf_model)

            self.models[target_name] = members

            # Validation score
            if X_val is not None and y_val_dict and target_name in y_val_dict:
                preds = self._ensemble_predict(members, X_val)
                y_val = y_val_dict[target_name]
                ss_res = np.sum((y_val - preds) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                scores[target_name] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return scores

    def _ensemble_predict(
        self, members: list[BaseEstimator], X: np.ndarray
    ) -> np.ndarray:
        """Mean prediction across ensemble members."""
        preds = np.array([m.predict(X) for m in members])
        return np.mean(preds, axis=0)

    def _ensemble_std(
        self, members: list[BaseEstimator], X: np.ndarray
    ) -> np.ndarray:
        """Prediction std across ensemble members (epistemic uncertainty)."""
        preds = np.array([m.predict(X) for m in members])
        return np.std(preds, axis=0)

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict shift values for all targets.

        Returns dict mapping target name to predicted values.
        """
        results = {}
        for target_name, members in self.models.items():
            results[target_name] = self._ensemble_predict(members, X)
        return results

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Predict with uncertainty estimates.

        Returns dict mapping target name to (mean_pred, std_pred).
        """
        results = {}
        for target_name, members in self.models.items():
            mean = self._ensemble_predict(members, X)
            std = self._ensemble_std(members, X)
            results[target_name] = (mean, std)
        return results

    def predict_shifts(self, climate_features: np.ndarray) -> dict[str, tuple[float, float]]:
        """Implement DistributionPredictor protocol.

        Returns dict with (mean_delta, std_scale) per shift variable.
        """
        preds = self.predict_with_uncertainty(climate_features)
        shifts = {}

        if "power_cost_shift" in preds:
            mean, std = preds["power_cost_shift"]
            shifts["power_cost_shift"] = (float(mean[0]), float(std[0]))

        if "pue_shift" in preds:
            mean, std = preds["pue_shift"]
            shifts["pue_shift"] = (float(mean[0]), float(std[0]))

        if "insurance_scale" in preds:
            mean, std = preds["insurance_scale"]
            shifts["insurance_shift"] = (float(mean[0]), float(std[0]))

        return shifts

    def feature_importance(self) -> dict[str, np.ndarray]:
        """Get feature importance from the first XGBoost model per target."""
        importances = {}
        for target_name, members in self.models.items():
            for m in members:
                if hasattr(m, "feature_importances_"):
                    importances[target_name] = m.feature_importances_
                    break
        return importances
