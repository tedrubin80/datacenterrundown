"""ML-ready dataset creation with feature engineering and splits."""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataCenterDataset:
    """Prepares and serves train/val/test data for both Idea 3 and 5."""

    # Features shared across models
    CLIMATE_FEATURES = [
        "avg_temp_c", "cooling_degree_days", "humidity_pct",
        "extreme_event_freq", "projected_pue", "power_price_delta_pct",
    ]
    LOCATION_FEATURES = [
        "latitude", "longitude", "baseline_temp_c", "renewable_pct",
        "grid_reliability_score",
    ]

    def __init__(
        self,
        climate_df: pd.DataFrame,
        locations_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.climate_df = climate_df
        self.locations_df = locations_df
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.scaler = StandardScaler()
        self._merged = None

    def prepare(self) -> pd.DataFrame:
        """Merge climate projections with location profiles and engineer features."""
        df = self.climate_df.copy()

        # Merge location static features
        if self.locations_df is not None:
            df = df.merge(self.locations_df, on="location_key", how="left")

        # Feature engineering
        df["years_from_start"] = df["year"] - df["year"].min()
        df["temp_x_humidity"] = df.get("avg_temp_c", 0) * df.get("humidity_pct", 0) / 100
        df["cdd_trend"] = df.groupby("location_key")["cooling_degree_days"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df["event_freq_trend"] = df.groupby("location_key")["extreme_event_freq"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        self._merged = df
        return df

    def get_feature_matrix(
        self,
        target_col: str,
        extra_features: list[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """Build feature matrix X and target vector y.

        Returns (X, y, feature_names).
        """
        if self._merged is None:
            self.prepare()

        feature_cols = []
        for col in self.CLIMATE_FEATURES + self.LOCATION_FEATURES:
            if col in self._merged.columns:
                feature_cols.append(col)

        # Add engineered features
        for col in ["years_from_start", "temp_x_humidity", "cdd_trend", "event_freq_trend"]:
            if col in self._merged.columns:
                feature_cols.append(col)

        if extra_features:
            feature_cols.extend([c for c in extra_features if c in self._merged.columns])

        feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate, preserve order
        valid = self._merged.dropna(subset=feature_cols + [target_col])

        X = valid[feature_cols].values
        y = valid[target_col].values
        return X, y, feature_cols

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale: bool = True,
    ) -> dict:
        """Split into train/val/test and optionally scale features.

        Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test, scaler.
        """
        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.val_ratio + test_ratio), random_state=self.seed
        )
        relative_val = self.val_ratio / (self.val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - relative_val), random_state=self.seed
        )

        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "scaler": self.scaler,
        }
