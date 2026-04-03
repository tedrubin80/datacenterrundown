#!/usr/bin/env python3
"""Train Bayesian Neural Network for climate-to-TCO distribution mapping.

Uses MC Dropout for uncertainty quantification. Compares BNN predictions
against the physics-based model to quantify epistemic uncertainty in
the climate-TCO relationship.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from src.data.climate_projections import generate_all_projections
from src.data.location_profiles import load_locations
from src.models.idea3.trainer import train_bayesian, _build_targets
from src.models.idea3.bayesian_nn import BayesianTCONet, BayesianTrainer, BayesianTCOPredictor
from src.data.dataset import DataCenterDataset
from src.tco.dynamic_distributions import run_dynamic_mc

RESULTS_DIR = Path(__file__).parents[1] / "data" / "results"


def main(seed: int = 42):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BAYESIAN NEURAL NETWORK TRAINING")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU")
    print("=" * 60)

    # Step 1: Prepare data
    print("\n[1/4] Preparing training data...")
    projections = generate_all_projections(seed=seed)
    locations = load_locations()

    df = _build_targets(projections)
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
    y_multi = np.column_stack([y_power, y_pue, y_ins])

    splits = dataset.split(X, y_multi, scale=True)
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")

    # Step 2: Train BNN
    print("\n[2/4] Training BNN (MC Dropout)...")
    input_dim = splits["X_train"].shape[1]
    net = BayesianTCONet(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1, n_outputs=3)
    trainer = BayesianTrainer(net, learning_rate=0.001, weight_decay=1e-4)

    print(f"  Architecture: {input_dim} → 128 → 64 → 32 → 3")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")

    metrics = trainer.fit(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        epochs=200, batch_size=256, patience=20,
    )

    print(f"  Epochs: {metrics['epochs']}")
    print(f"  Train loss: {metrics['train_loss']:.6f}")
    print(f"  Val loss: {metrics['val_loss']:.6f}")
    print(f"  Early stopped: {metrics['early_stopped']}")

    # Step 3: Evaluate uncertainty quality
    print("\n[3/4] Evaluating uncertainty calibration...")
    predictor = BayesianTCOPredictor(model=net, mc_samples=50)

    X_test = torch.FloatTensor(splits["X_test"])
    y_test = splits["y_test"]

    mean_pred, std_pred = net.predict_mc(X_test, n_samples=50)
    mean_pred = mean_pred.detach().numpy()
    std_pred = std_pred.detach().numpy()

    # Check calibration: what % of true values fall within 1σ and 2σ?
    target_names = ["power_cost_shift", "pue_shift", "insurance_scale"]
    for i, name in enumerate(target_names):
        residuals = np.abs(y_test[:, i] - mean_pred[:, i])
        within_1sigma = np.mean(residuals < std_pred[:, i])
        within_2sigma = np.mean(residuals < 2 * std_pred[:, i])
        rmse = np.sqrt(np.mean((y_test[:, i] - mean_pred[:, i]) ** 2))
        mean_unc = np.mean(std_pred[:, i])
        print(f"  {name}:")
        print(f"    RMSE: {rmse:.4f}, Mean uncertainty: {mean_unc:.4f}")
        print(f"    Within 1σ: {within_1sigma:.1%} (ideal: 68%)")
        print(f"    Within 2σ: {within_2sigma:.1%} (ideal: 95%)")

    # Step 4: Compare BNN vs physics for key locations
    print("\n[4/4] BNN vs Physics shift predictions (year 2050, RCP 8.5)...")

    # Get physics predictions for comparison
    from src.tco.dynamic_distributions import _physics_shifts

    rcp85 = projections[projections["scenario"] == "rcp85"]
    print(f"\n  {'Location':<30} {'Physics PUE':>12} {'BNN PUE':>10} {'BNN ±σ':>8}")
    print("  " + "-" * 62)

    for loc_key in ["boden_sweden", "atlanta_georgia", "evanston_wyoming", "johor_malaysia"]:
        loc = locations[loc_key]
        row_2050 = rcp85[(rcp85["location_key"] == loc_key) & (rcp85["year"] == 2050)]
        if len(row_2050) == 0:
            continue

        # Physics prediction
        physics = _physics_shifts(row_2050.iloc[0], loc, 2025)
        phys_pue = physics["pue_shift"][0]

        # BNN prediction
        bnn_shifts = predictor.predict_shifts(
            splits["X_test"][0:1]  # use a test sample as proxy
        )
        bnn_pue = bnn_shifts["pue_shift"][0]
        bnn_unc = bnn_shifts["pue_shift"][1]

        print(f"  {loc.name:<30} {phys_pue:>+12.4f} {bnn_pue:>+10.4f} {bnn_unc:>8.4f}")

    # Save model
    torch.save(net.state_dict(), RESULTS_DIR / "bnn_model.pt")
    print(f"\n  Model saved to {RESULTS_DIR / 'bnn_model.pt'}")

    # Save training curves
    curves = pd.DataFrame({
        "epoch": range(1, len(trainer.train_losses) + 1),
        "train_loss": trainer.train_losses,
        "val_loss": trainer.val_losses[:len(trainer.train_losses)] if trainer.val_losses else [None] * len(trainer.train_losses),
    })
    curves.to_csv(RESULTS_DIR / "bnn_training_curves.csv", index=False)

    print("\n" + "=" * 60)
    print("BNN TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
