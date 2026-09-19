"""Fit and evaluate a small PyTorch 1D-CNN baseline on the DL dataset layer.

Reads data/dl_tracks.csv and the already-persisted data/dl_split.csv (does
NOT regenerate the split -- see src/generate_dl_split.py for that). Uses
src.dl_data.tracks_to_sequences() output unflattened -- (N, 29, 3) -- fits
per-channel scaling (src.dl_cnn_torch's independent NumPy reimplementation,
not TensorFlow's) and the PyTorch model on train only, and evaluates
validation only.

Test-set discipline: identical to src/train_dl_cnn.py -- the persisted test
split is validated for structural integrity (via lookup_split_indices) but
is never fit on or scored by the model here. It stays untouched until
Logistic Regression, TensorFlow, and PyTorch are all frozen; a later,
separate stage evaluates all three once each on the test split.

CPU-only: no CUDA/MPS device selection anywhere in this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dl_baseline import build_results_table, compute_classification_metrics, lookup_split_indices
from src.dl_cnn_torch import CNN_LEARNING_RATE, apply_channel_scaler, build_cnn_model, fit_channel_scaler, set_deterministic_seed
from src.dl_data import tracks_to_sequences

REPO_ROOT = Path(__file__).resolve().parent.parent
DL_TRACKS_PATH = REPO_ROOT / "data" / "dl_tracks.csv"  # read-only input
DL_SPLIT_PATH = REPO_ROOT / "data" / "dl_split.csv"  # read-only input
CNN_METRICS_PATH = REPO_ROOT / "results" / "dl" / "torch_cnn_metrics.csv"

# Arbitrary implementation constant, not a scientific parameter -- happy to
# change if you'd prefer a different one. Distinct from DL_SEED=2026,
# DL_SPLIT_SEED=2027, DL_BASELINE_RANDOM_STATE=2028, DL_TF_RANDOM_STATE=2029.
DL_TORCH_RANDOM_STATE = 2030
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 30
CNN_DECISION_THRESHOLD = 0.5


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> None:
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(1)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()


def build_and_train_model(X_train: np.ndarray, y_train: np.ndarray) -> nn.Module:
    """Build, seed, and fit the model on (X_train, y_train) for CNN_EPOCHS.

    Isolated as its own function (mirroring build_cnn_model()'s role in the
    TF script) so main()-plumbing tests can monkeypatch this single call to
    substitute a stub model/trainer without running real training.
    """
    set_deterministic_seed(DL_TORCH_RANDOM_STATE)
    model = build_cnn_model()
    set_deterministic_seed(DL_TORCH_RANDOM_STATE)  # reset immediately before training

    X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator().manual_seed(DL_TORCH_RANDOM_STATE)
    loader = DataLoader(dataset, batch_size=CNN_BATCH_SIZE, shuffle=True, generator=generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(CNN_EPOCHS):
        train_one_epoch(model, loader, optimizer, loss_fn)

    return model


def predict_probabilities(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Eval-mode forward pass + sigmoid -> P(injury=1), as a numpy array.

    Explicitly casts X to torch.float32 before the forward pass, just like
    the training path in build_and_train_model, rather than relying on an
    implicit dtype.
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


def main() -> None:
    tracks = pd.read_csv(DL_TRACKS_PATH)
    split_df = pd.read_csv(DL_SPLIT_PATH)

    X, y, track_ids = tracks_to_sequences(tracks)
    conditions_by_track = {
        track_id: ("injury" if label == 1 else "control") for track_id, label in zip(track_ids, y)
    }
    idx = lookup_split_indices(track_ids, conditions_by_track, split_df)
    # idx["test"] is validated above (as part of the persisted split's
    # integrity) but is never indexed into X/y below -- same test-split
    # discipline as src/train_dl_cnn.py.

    mean, scale = fit_channel_scaler(X[idx["train"]])  # fit on train ONLY
    X_train_scaled = apply_channel_scaler(X[idx["train"]], mean, scale)
    X_val_scaled = apply_channel_scaler(X[idx["val"]], mean, scale)  # apply only, never re-fit

    model = build_and_train_model(X_train_scaled, y[idx["train"]])

    y_val_prob = predict_probabilities(model, X_val_scaled)
    y_val_pred = (y_val_prob >= CNN_DECISION_THRESHOLD).astype(int)
    results = build_results_table(
        [
            compute_classification_metrics(
                y[idx["val"]],
                y_val_pred,
                y_val_prob,
                "val",
                DL_TORCH_RANDOM_STATE,
            )
        ]
    )

    CNN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(CNN_METRICS_PATH, index=False)
    print(f"Wrote {len(results)} rows to {CNN_METRICS_PATH}")
    print(
        "Note: this is a synthetic known-signal recovery benchmark, not a "
        "biological finding. It measures whether a small 1D CNN can "
        "recover the directional bias deliberately built into the injury "
        "condition's simulation (src/simulate.py). Test-split performance "
        "is intentionally not evaluated here -- it is reserved for a "
        "later, one-time comparison after Logistic Regression, "
        "TensorFlow, and PyTorch models are all frozen."
    )


if __name__ == "__main__":
    main()
