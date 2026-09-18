"""Pure-ish 1D-CNN baseline building blocks for the DL dataset layer.

Consumes tracks_to_sequences() output (N, 29, 3) directly -- no
flattening, preserving temporal order -- and provides per-channel scaling
fit/apply helpers plus a small, explicit Keras Conv1D architecture. No
file I/O and no model fitting here (that lives in src/train_dl_cnn.py),
mirroring src/dl_baseline.py's split between pure building blocks and the
I/O-and-fitting script.

This baseline is a synthetic known-signal recovery benchmark: it tests
whether a small 1D CNN can recover the directional-bias signal
deliberately built into the injury condition's simulation (see
src/simulate.py's directional_bias()). It is not a biological or causal
claim (see src/dl_baseline.py for the same caveat applied to the
Logistic Regression baseline).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.dl_baseline import N_CHANNELS, N_TRANSITIONS  # reused, not duplicated

CNN_INPUT_SHAPE: tuple[int, int] = (N_TRANSITIONS, N_CHANNELS)
CNN_FILTERS: int = 16
CNN_KERNEL_SIZE: int = 3
CNN_CONV_PADDING: str = "same"
CNN_LEARNING_RATE: float = 1e-3
# Arbitrary implementation constants, not scientific parameters -- happy
# to change if you'd prefer different ones.


def enable_deterministic_ops() -> None:
    """Enable TensorFlow's deterministic op execution.

    Must be called explicitly by the caller (train_dl_cnn.main(), or a
    test that needs reproducible training) -- importing this module has
    NO side effect on global TensorFlow state.
    """
    tf.config.experimental.enable_op_determinism()


def _validate_sequence_shape(X: np.ndarray, name: str) -> None:
    if X.ndim != 3 or X.shape[1:] != (N_TRANSITIONS, N_CHANNELS):
        raise ValueError(
            f"Expected {name} with shape (N, {N_TRANSITIONS}, {N_CHANNELS}), got {X.shape}"
        )


def fit_channel_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean/population-std on TRAIN data only.

    X_train has shape (N_train, 29, 3); it is reshaped to (N_train*29, 3)
    so each of the 3 channels is standardized independently across all
    train timesteps (population std, ddof=0, matching sklearn's
    StandardScaler). Returns (mean, scale), each shape (3,); a
    zero-variance channel gets scale=1.0 to avoid divide-by-zero.
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


def build_cnn_model() -> keras.Model:
    """Conv1D -> ReLU -> Conv1D -> ReLU -> GlobalAveragePooling1D -> Dense(1, sigmoid).

    Compiled with Adam(CNN_LEARNING_RATE) and binary_crossentropy loss.
    No parameters -- fully hard-coded, no hyperparameter search.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=CNN_INPUT_SHAPE),
            keras.layers.Conv1D(CNN_FILTERS, CNN_KERNEL_SIZE, padding=CNN_CONV_PADDING, activation="relu"),
            keras.layers.Conv1D(CNN_FILTERS, CNN_KERNEL_SIZE, padding=CNN_CONV_PADDING, activation="relu"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    # metrics=["accuracy"] is for epoch-log visibility only; the persisted
    # metrics are computed independently via compute_classification_metrics,
    # same philosophy as the Logistic Regression baseline.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
