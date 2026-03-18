"""Idea 5: Outage event classifier with explainability.

Predicts probability of datacenter outage given climate conditions.
Uses XGBoost with SHAP for feature importance explanations.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class OutageClassifier:
    """XGBoost classifier for predicting datacenter outage events.

    Binary classification: will an outage occur in the next period
    given current climate state?
    """

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        scale_pos_weight: float = 3.0,
        seed: int = 42,
    ):
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "scale_pos_weight": scale_pos_weight,
            "random_state": seed,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
        }
        self.model = None
        self.feature_names = None
        self.explainer = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """Train the outage classifier.

        Returns dict with training metrics.
        """
        if not HAS_XGB:
            return {"error": "xgboost not installed"}

        self.feature_names = feature_names
        self.model = xgb.XGBClassifier(**self.params)

        fit_params = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_params)

        metrics = {}
        if X_val is not None and y_val is not None:
            y_prob = self.model.predict_proba(X_val)[:, 1]
            y_pred = self.model.predict(X_val)
            metrics["val_auroc"] = float(roc_auc_score(y_val, y_prob))
            metrics["val_auprc"] = float(average_precision_score(y_val, y_prob))
            metrics["val_report"] = classification_report(
                y_val, y_pred, output_dict=True
            )

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict outage probability."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary outage label."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)

    def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Compute SHAP values for feature importance.

        Returns SHAP values array of shape (n_samples, n_features).
        """
        if not HAS_SHAP or self.model is None:
            return None

        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)

        return self.explainer.shap_values(X)

    def feature_importance(self) -> Optional[dict[str, float]]:
        """Get feature importance scores."""
        if self.model is None:
            return None
        importances = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importances))
        return dict(enumerate(importances))
