"""Pure-ish 1D-CNN baseline building blocks for the DL dataset layer (PyTorch).

Consumes tracks_to_sequences() output (N, 29, 3) directly -- no flattening,
preserving temporal order. The (N, 29, 3) -> (N, 3, 29) transpose required by
PyTorch's Conv1d (channels-first) happens ONLY inside forward(); every
function outside the model still speaks (N, 29, 3), matching src/dl_cnn.py's
TensorFlow counterpart so the two baselines stay directly comparable.

fit_channel_scaler()/apply_channel_scaler() are reimplemented independently
here (not imported from src.dl_cnn) so this PyTorch stage has zero
TensorFlow dependency -- src/dl_cnn.py imports tensorflow, and importing it
here just to reuse framework-agnostic NumPy helpers would pull that
dependency in unnecessarily. The numerical behavior is identical to the
TensorFlow implementation. No file I/O and no model fitting here (that
lives in src/train_dl_cnn_torch.py), mirroring the TF split between pure
building blocks and the I/O-and-fitting script.

This baseline is a synthetic known-signal recovery benchmark: it tests
whether a small 1D CNN can recover the directional-bias signal deliberately
built into the injury condition's simulation (see src/simulate.py's
directional_bias()). It is not a biological or causal claim.

CPU-only: no CUDA/MPS device selection is implemented. The dataset and
model are small enough that CPU execution keeps this stage simple and its
reproducibility contract clear.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from src.dl_baseline import N_CHANNELS, N_TRANSITIONS  # reused, not duplicated

CNN_INPUT_SHAPE: tuple[int, int] = (N_TRANSITIONS, N_CHANNELS)  # (29, 3), external/canonical
CNN_FILTERS: int = 16
CNN_KERNEL_SIZE: int = 3
CNN_CONV_PADDING: int = 1  # "same" for kernel_size=3, stride=1
CNN_LEARNING_RATE: float = 1e-3
# Arbitrary implementation constants, not scientific parameters -- matched to
# src/dl_cnn.py's TF architecture for a like-for-like comparison.


def set_deterministic_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and PyTorch, and request deterministic ops.

    Must be called explicitly by the caller (train_dl_cnn_torch.main(), or a
    test that needs reproducible training) -- importing this module has NO
    side effect on global state, mirroring dl_cnn.enable_deterministic_ops().
    CPU-only: no CUDA seeding, since this stage never selects a CUDA device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validate_sequence_shape(X: np.ndarray, name: str) -> None:
    if X.ndim != 3 or X.shape[1:] != (N_TRANSITIONS, N_CHANNELS):
        raise ValueError(
            f"Expected {name} with shape (N, {N_TRANSITIONS}, {N_CHANNELS}), got {X.shape}"
        )


def fit_channel_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean/population-std on TRAIN data only.

    Independent NumPy reimplementation of src.dl_cnn.fit_channel_scaler,
    numerically identical (same reshape-to-(-1,3), population std with
    ddof=0, zero-variance -> scale=1.0), kept separate so this module has no
    TensorFlow dependency. X_train has shape (N_train, 29, 3); it is
    reshaped to (N_train*29, 3) so each of the 3 channels is standardized
    independently across all train timesteps. Returns (mean, scale), each
    shape (3,).
    """
    _validate_sequence_shape(X_train, "X_train")

    flattened = X_train.reshape(-1, N_CHANNELS)
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def apply_channel_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a fitted per-channel scaler to X.

    Computes (X - mean) / scale, broadcasting mean/scale (each shape (3,))
    per-channel over X's shape (N, 29, 3). Shape-in == shape-out; never
    flattens.
    """
    _validate_sequence_shape(X, "X")
    return (X - mean) / scale


class Conv1dBaselineNet(nn.Module):
    """Conv1d -> ReLU -> Conv1d -> ReLU -> GlobalAveragePool(time) -> Linear(16,1).

    forward() accepts (N, 29, 3) -- the canonical external layout -- and
    performs the (N, 29, 3) -> (N, 3, 29) transpose internally, right before
    the first Conv1d, so nothing outside this class ever needs to know about
    PyTorch's channels-first convention. Returns a raw logit of shape (N, 1)
    -- no sigmoid; BCEWithLogitsLoss expects logits, not probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(N_CHANNELS, CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE, padding=CNN_CONV_PADDING)
        self.conv2 = nn.Conv1d(CNN_FILTERS, CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE, padding=CNN_CONV_PADDING)
        self.relu = nn.ReLU()
        self.head = nn.Linear(CNN_FILTERS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 29, 3) canonical -> (N, 3, 29) for Conv1d, at the boundary only.
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2)  # global average pool over time -> (N, 16)
        return self.head(x)  # (N, 1) raw logit


def build_cnn_model() -> Conv1dBaselineNet:
    """Factory, naming parity with src.dl_cnn.build_cnn_model().

    No parameters -- fully hard-coded, no hyperparameter search.
    """
    return Conv1dBaselineNet()
