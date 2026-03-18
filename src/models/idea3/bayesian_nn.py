"""Idea 3: Bayesian Neural Network for climate-to-TCO distribution mapping.

Uses MC Dropout as a practical Bayesian approximation (Gal & Ghahramani, 2016).
Provides calibrated epistemic uncertainty over TCO shift predictions.
"""

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class BayesianTCONet(nn.Module):
        """Multi-output BNN with MC Dropout for uncertainty.

        Inputs: climate + location features
        Outputs: (power_cost_shift, pue_shift, insurance_scale)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int] = None,
            dropout_rate: float = 0.1,
            n_outputs: int = 3,
        ):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [128, 64, 32]

            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
                prev_dim = h_dim

            self.backbone = nn.Sequential(*layers)
            self.output_head = nn.Linear(prev_dim, n_outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass. Dropout stays active for MC sampling."""
            h = self.backbone(x)
            return self.output_head(h)

        def predict_mc(
            self, x: torch.Tensor, n_samples: int = 50
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """MC Dropout: run n_samples forward passes with dropout enabled.

            Returns (mean_predictions, std_predictions) each of shape (batch, n_outputs).
            """
            self.train()  # keep dropout active
            preds = torch.stack([self.forward(x) for _ in range(n_samples)])
            self.eval()
            return preds.mean(dim=0), preds.std(dim=0)


    class BayesianTrainer:
        """Training loop for BayesianTCONet."""

        def __init__(
            self,
            model: BayesianTCONet,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-4,
        ):
            self.model = model
            self.optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            self.criterion = nn.MSELoss()
            self.train_losses = []
            self.val_losses = []

        def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 256,
            patience: int = 20,
        ) -> dict:
            """Train with early stopping on validation loss.

            Args:
                X_train, y_train: Training data.
                X_val, y_val: Validation data.
                epochs: Max training epochs.
                batch_size: Mini-batch size.
                patience: Early stopping patience.

            Returns:
                Dict with final train/val loss and epochs trained.
            """
            train_ds = TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(epochs):
                # Training
                self.model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    self.optimizer.zero_grad()
                    pred = self.model(X_batch)
                    loss = self.criterion(pred, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item() * len(X_batch)

                train_loss = epoch_loss / len(train_ds)
                self.train_losses.append(train_loss)

                # Validation
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred = self.model(torch.FloatTensor(X_val))
                        val_loss = self.criterion(
                            val_pred, torch.FloatTensor(y_val)
                        ).item()
                    self.val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {
                            k: v.clone() for k, v in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self.model.load_state_dict(best_state)
                            return {
                                "epochs": epoch + 1,
                                "train_loss": train_loss,
                                "val_loss": best_val_loss,
                                "early_stopped": True,
                            }

            return {
                "epochs": epochs,
                "train_loss": self.train_losses[-1],
                "val_loss": self.val_losses[-1] if self.val_losses else None,
                "early_stopped": False,
            }


class BayesianTCOPredictor:
    """Wraps BayesianTCONet to implement DistributionPredictor protocol."""

    def __init__(self, model=None, mc_samples: int = 50):
        self.model = model
        self.mc_samples = mc_samples

    def predict_shifts(
        self, climate_features: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """Predict distribution shifts with uncertainty via MC Dropout."""
        if not HAS_TORCH or self.model is None:
            return {
                "power_cost_shift": (0.0, 1.0),
                "pue_shift": (0.0, 1.0),
                "insurance_shift": (0.0, 1.0),
            }

        x = torch.FloatTensor(climate_features)
        mean, std = self.model.predict_mc(x, self.mc_samples)
        mean = mean.detach().numpy()[0]
        std = std.detach().numpy()[0]

        return {
            "power_cost_shift": (float(mean[0]), float(std[0])),
            "pue_shift": (float(mean[1]), float(std[1])),
            "insurance_shift": (float(mean[2]), float(std[2])),
        }
