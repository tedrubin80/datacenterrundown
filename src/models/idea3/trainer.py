"""Training orchestration for Idea 3 models."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...data.dataset import DataCenterDataset
from ...data.climate_projections import generate_all_projections
from ...data.location_profiles import load_locations
from .ensemble_model import EnsembleTCOModel

try:
    from .bayesian_nn import BayesianTCONet, BayesianTrainer, HAS_TORCH
except ImportError:
    HAS_TORCH = False


def _build_targets(climate_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer target variables from climate projections.

    Targets represent how much each TCO component should shift
    relative to baseline conditions.
    """
    df = climate_df.copy()

    # Power cost shift: driven by temperature increase and power price pressure
    df["power_cost_shift"] = (
        df["power_price_delta_pct"] * 0.5
        + (df["avg_temp_c"] - df.groupby("location_key")["avg_temp_c"].transform("first")) * 1.2
    )

    # PUE shift: driven by temperature and cooling demand
    df["pue_shift"] = df["pue_delta"]

    # Insurance scale: driven by extreme event frequency increase
    baseline_events = df.groupby("location_key")["extreme_event_freq"].transform("first")
    event_ratio = df["extreme_event_freq"] / baseline_events.clip(lower=1)
    df["insurance_scale"] = 1.0 + (event_ratio - 1.0) * 0.4

    return df


def train_ensemble(
    climate_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> tuple[EnsembleTCOModel, dict]:
    """Train the ensemble model for dynamic TCO prediction.

    Returns (trained_model, metrics_dict).
    """
    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    # Build targets
    df = _build_targets(climate_df)

    # Build location features
    locations = load_locations()
    loc_features = pd.DataFrame([
        {
            "location_key": key,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "baseline_temp_c": loc.baseline_temp_c,
            "renewable_pct": loc.renewable_pct,
            "grid_reliability_score": loc.grid_reliability_score,
        }
        for key, loc in locations.items()
    ])

    # Create dataset
    dataset = DataCenterDataset(df, loc_features, seed=seed)
    dataset.prepare()

    # Build features and targets
    target_names = ["power_cost_shift", "pue_shift", "insurance_scale"]
    X, y_power, feature_names = dataset.get_feature_matrix("power_cost_shift")
    _, y_pue, _ = dataset.get_feature_matrix("pue_shift")
    _, y_ins, _ = dataset.get_feature_matrix("insurance_scale")

    # Split
    splits = dataset.split(X, y_power, scale=True)
    # Need to split other targets with same indices
    n_train = len(splits["y_train"])
    n_val = len(splits["y_val"])

    y_dict_train = {
        "power_cost_shift": y_power[:n_train],
        "pue_shift": y_pue[:n_train],
        "insurance_scale": y_ins[:n_train],
    }
    y_dict_val = {
        "power_cost_shift": y_power[n_train:n_train + n_val],
        "pue_shift": y_pue[n_train:n_train + n_val],
        "insurance_scale": y_ins[n_train:n_train + n_val],
    }

    # Train ensemble
    model = EnsembleTCOModel(seed=seed)
    scores = model.fit(
        splits["X_train"], y_dict_train,
        splits["X_val"], y_dict_val,
    )

    return model, {"val_r2": scores, "feature_names": feature_names}


def train_bayesian(
    climate_df: Optional[pd.DataFrame] = None,
    epochs: int = 200,
    seed: int = 42,
) -> tuple:
    """Train the BNN model for dynamic TCO prediction.

    Returns (predictor, metrics_dict) or (None, error_msg) if PyTorch unavailable.
    """
    if not HAS_TORCH:
        return None, {"error": "PyTorch not installed"}

    if climate_df is None:
        climate_df = generate_all_projections(seed=seed)

    df = _build_targets(climate_df)
    locations = load_locations()
    loc_features = pd.DataFrame([
        {
            "location_key": key,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "baseline_temp_c": loc.baseline_temp_c,
            "renewable_pct": loc.renewable_pct,
            "grid_reliability_score": loc.grid_reliability_score,
        }
        for key, loc in locations.items()
    ])

    dataset = DataCenterDataset(df, loc_features, seed=seed)
    dataset.prepare()

    X, y_power, feature_names = dataset.get_feature_matrix("power_cost_shift")
    _, y_pue, _ = dataset.get_feature_matrix("pue_shift")
    _, y_ins, _ = dataset.get_feature_matrix("insurance_scale")

    # Stack multi-output targets
    y_multi = np.column_stack([y_power, y_pue, y_ins])

    splits = dataset.split(X, y_multi, scale=True)

    # Build and train BNN
    from .bayesian_nn import BayesianTCOPredictor
    import torch

    input_dim = splits["X_train"].shape[1]
    net = BayesianTCONet(input_dim=input_dim, n_outputs=3)
    trainer = BayesianTrainer(net)

    metrics = trainer.fit(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        epochs=epochs,
    )

    predictor = BayesianTCOPredictor(model=net)
    return predictor, metrics
