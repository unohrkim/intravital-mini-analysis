"""Deterministic tests for the TensorFlow 1D-CNN baseline stage.

Covers src/dl_cnn.py (per-channel scaling, model factory) and the main()
entry point of src/train_dl_cnn.py.

Real TensorFlow training is confined to a handful of tests on a tiny
synthetic (N=20-ish, 29, 3) dataset for 1-2 epochs. All tests that exercise
train_dl_cnn.main() substitute a lightweight stub model (via monkeypatch)
so the real 30-epoch, full-4000-track training never runs inside pytest --
that only happens via the explicit `uv run python -m src.train_dl_cnn` step.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import src.train_dl_cnn as train_dl_cnn
from src.dl_baseline import RESULTS_COLUMNS, lookup_split_indices
from src.dl_cnn import (
    N_CHANNELS,
    N_TRANSITIONS,
    apply_channel_scaler,
    build_cnn_model,
    enable_deterministic_ops,
    fit_channel_scaler,
)
from src.dl_data import tracks_to_sequences

REPO_ROOT = Path(__file__).resolve().parent.parent

# Read once at collection time (read-only -- never written to by any test)
# and reused across tests, mirroring the DL_TRACKS pattern in
# tests/test_dl_data.py and tests/test_dl_baseline.py.
DL_TRACKS = pd.read_csv(REPO_ROOT / "data" / "dl_tracks.csv")
DL_SPLIT = pd.read_csv(REPO_ROOT / "data" / "dl_split.csv")
X, Y, TRACK_IDS = tracks_to_sequences(DL_TRACKS)
CONDITIONS_BY_TRACK = {
    track_id: ("injury" if label == 1 else "control") for track_id, label in zip(TRACK_IDS, Y)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_synthetic_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_tiny = rng.normal(size=(n, N_TRANSITIONS, N_CHANNELS)).astype("float32")
    y_tiny = rng.integers(0, 2, size=n)
    return X_tiny, y_tiny


class _StubCnnModel:
    """Lightweight stand-in for a Keras model, used only by main()-plumbing
    tests so they never run real TensorFlow training."""

    def __init__(self) -> None:
        self.fit_calls: list[dict] = []

    def fit(self, X, y, validation_data=None, batch_size=None, epochs=None, verbose=None):
        self.fit_calls.append({"X": np.asarray(X), "y": np.asarray(y), "validation_data": validation_data})

    def predict(self, X, verbose=0):
        X = np.asarray(X)
        return np.full((X.shape[0], 1), 0.5)


# --- fit_channel_scaler / apply_channel_scaler (tiny synthetic data, no model) ---


def test_channel_scaler_matches_hand_computed_stats():
    X_tiny, _ = _tiny_synthetic_dataset(n=10, seed=1)

    mean, scale = fit_channel_scaler(X_tiny)

    flattened = X_tiny.reshape(-1, N_CHANNELS)
    expected_mean = flattened.mean(axis=0)
    expected_scale = flattened.std(axis=0, ddof=0)

    assert mean == pytest.approx(expected_mean)
    assert scale == pytest.approx(expected_scale)


def test_channel_scaler_fit_only_on_training_data():
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    mean, scale = fit_channel_scaler(X[idx["train"]])

    flattened_train = X[idx["train"]].reshape(-1, N_CHANNELS)
    expected_mean = flattened_train.mean(axis=0)
    expected_scale = flattened_train.std(axis=0, ddof=0)

    assert np.allclose(mean, expected_mean)
    assert np.allclose(scale, expected_scale)


def test_apply_channel_scaler_preserves_shape():
    X_tiny, _ = _tiny_synthetic_dataset(n=5, seed=2)
    mean, scale = fit_channel_scaler(X_tiny)

    scaled = apply_channel_scaler(X_tiny, mean, scale)

    assert scaled.shape == X_tiny.shape


def test_apply_channel_scaler_preserves_temporal_and_channel_order():
    X_tiny, _ = _tiny_synthetic_dataset(n=5, seed=3)
    mean, scale = fit_channel_scaler(X_tiny)

    scaled = apply_channel_scaler(X_tiny, mean, scale)

    for t in (0, 14, N_TRANSITIONS - 1):
        for c in range(N_CHANNELS):
            expected = (X_tiny[:, t, c] - mean[c]) / scale[c]
            assert scaled[:, t, c] == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad_shape",
    [
        (10, 87),  # 2-D
        (10, 3, 29),  # transposed axes
        (10, 1, 87),  # pre-flattened
    ],
)
def test_channel_scaler_rejects_malformed_shape(bad_shape):
    bad_X = np.zeros(bad_shape, dtype="float32")

    with pytest.raises(ValueError):
        fit_channel_scaler(bad_X)

    mean = np.zeros(N_CHANNELS)
    scale = np.ones(N_CHANNELS)
    with pytest.raises(ValueError):
        apply_channel_scaler(bad_X, mean, scale)


# --- build_cnn_model (tiny synthetic data, no or minimal training) ---


def test_build_cnn_model_output_shape():
    model = build_cnn_model()
    X_tiny, _ = _tiny_synthetic_dataset(n=4, seed=4)

    predictions = model.predict(X_tiny, verbose=0)

    assert predictions.shape == (4, 1)


def test_build_cnn_model_fresh_predictions_are_finite_and_in_unit_range():
    model = build_cnn_model()
    X_tiny, _ = _tiny_synthetic_dataset(n=6, seed=5)

    predictions = model.predict(X_tiny, verbose=0).reshape(-1)

    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_build_cnn_model_uses_binary_crossentropy_loss():
    model = build_cnn_model()
    assert model.loss == "binary_crossentropy"


def test_val_predicted_probabilities_are_within_unit_range():
    X_train_tiny, y_train_tiny = _tiny_synthetic_dataset(n=16, seed=6)
    X_val_tiny, _ = _tiny_synthetic_dataset(n=8, seed=7)

    mean, scale = fit_channel_scaler(X_train_tiny)
    X_train_scaled = apply_channel_scaler(X_train_tiny, mean, scale)
    X_val_scaled = apply_channel_scaler(X_val_tiny, mean, scale)

    tf.keras.utils.set_random_seed(123)
    model = build_cnn_model()
    model.fit(X_train_scaled, y_train_tiny, epochs=2, batch_size=8, verbose=0)

    y_val_prob = model.predict(X_val_scaled, verbose=0).reshape(-1)

    assert np.all(np.isfinite(y_val_prob))
    assert np.all(y_val_prob >= 0.0)
    assert np.all(y_val_prob <= 1.0)


def test_build_and_fit_is_deterministic_with_seed_reset():
    enable_deterministic_ops()

    X_train_tiny, y_train_tiny = _tiny_synthetic_dataset(n=16, seed=8)
    X_val_tiny, _ = _tiny_synthetic_dataset(n=8, seed=9)

    mean, scale = fit_channel_scaler(X_train_tiny)
    X_train_scaled = apply_channel_scaler(X_train_tiny, mean, scale)
    X_val_scaled = apply_channel_scaler(X_val_tiny, mean, scale)

    def _build_fit_predict() -> np.ndarray:
        tf.keras.utils.set_random_seed(2029)
        model = build_cnn_model()
        tf.keras.utils.set_random_seed(2029)  # reset immediately before fit
        model.fit(X_train_scaled, y_train_tiny, epochs=2, batch_size=8, verbose=0)
        return model.predict(X_val_scaled, verbose=0)

    first = _build_fit_predict()
    second = _build_fit_predict()

    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)


# --- split-lookup reuse (no model involved) ---


def test_lookup_split_indices_reused_from_dl_baseline():
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    assert len(idx["train"]) == 2800
    assert len(idx["val"]) == 600
    assert len(idx["test"]) == 600


# --- train_dl_cnn.main() plumbing (stub model -- no real TensorFlow training) ---


def test_train_dl_cnn_uses_persisted_split_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(train_dl_cnn, "build_cnn_model", lambda: _StubCnnModel())
    monkeypatch.setattr(train_dl_cnn, "CNN_METRICS_PATH", tmp_path / "tf_cnn_metrics.csv")

    train_dl_cnn.main()

    results = pd.read_csv(tmp_path / "tf_cnn_metrics.csv")
    assert results.loc[0, "n_samples"] == 600
    assert results.loc[0, "n_positive"] == 300
    assert results.loc[0, "n_negative"] == 300


def test_train_dl_cnn_never_flattens_input(tmp_path, monkeypatch):
    calls: list[np.ndarray] = []
    original_apply = train_dl_cnn.apply_channel_scaler

    def _spy_apply(X_arg, mean, scale):
        calls.append(np.asarray(X_arg))
        return original_apply(X_arg, mean, scale)

    monkeypatch.setattr(train_dl_cnn, "apply_channel_scaler", _spy_apply)
    monkeypatch.setattr(train_dl_cnn, "build_cnn_model", lambda: _StubCnnModel())
    monkeypatch.setattr(train_dl_cnn, "CNN_METRICS_PATH", tmp_path / "tf_cnn_metrics.csv")

    train_dl_cnn.main()

    assert len(calls) == 2  # train, val
    for call_array in calls:
        assert call_array.ndim == 3
        assert call_array.shape[1:] == (N_TRANSITIONS, N_CHANNELS)


def test_train_dl_cnn_never_touches_test_split(tmp_path, monkeypatch):
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)
    test_rows = X[idx["test"]]

    calls: list[np.ndarray] = []
    original_apply = train_dl_cnn.apply_channel_scaler

    def _spy_apply(X_arg, mean, scale):
        calls.append(np.asarray(X_arg))
        return original_apply(X_arg, mean, scale)

    monkeypatch.setattr(train_dl_cnn, "apply_channel_scaler", _spy_apply)
    monkeypatch.setattr(train_dl_cnn, "build_cnn_model", lambda: _StubCnnModel())
    monkeypatch.setattr(train_dl_cnn, "CNN_METRICS_PATH", tmp_path / "tf_cnn_metrics.csv")

    train_dl_cnn.main()

    # None of the (train, val) arrays fed into apply_channel_scaler equal
    # the test split's rows (checked by content, not just shape, since
    # val and test happen to have the same row count: 600 each).
    for call_array in calls:
        if call_array.shape == test_rows.shape:
            assert not np.array_equal(call_array, test_rows)

    results = pd.read_csv(tmp_path / "tf_cnn_metrics.csv")
    assert set(results["split"]) == {"val"}


def test_train_dl_cnn_main_writes_expected_metrics_without_mutating_real_files(tmp_path, monkeypatch):
    dl_tracks_hash_before = _sha256(REPO_ROOT / "data" / "dl_tracks.csv")
    dl_split_hash_before = _sha256(REPO_ROOT / "data" / "dl_split.csv")

    monkeypatch.setattr(train_dl_cnn, "build_cnn_model", lambda: _StubCnnModel())
    monkeypatch.setattr(train_dl_cnn, "CNN_METRICS_PATH", tmp_path / "tf_cnn_metrics.csv")

    train_dl_cnn.main()

    assert _sha256(REPO_ROOT / "data" / "dl_tracks.csv") == dl_tracks_hash_before
    assert _sha256(REPO_ROOT / "data" / "dl_split.csv") == dl_split_hash_before

    results = pd.read_csv(tmp_path / "tf_cnn_metrics.csv")
    assert len(results) == 1
    assert results.loc[0, "split"] == "val"
    assert list(results.columns) == RESULTS_COLUMNS
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= results.loc[0, metric] <= 1.0


def test_train_dl_cnn_main_is_deterministic_across_runs(tmp_path, monkeypatch):
    first_path = tmp_path / "tf_cnn_metrics_first.csv"
    second_path = tmp_path / "tf_cnn_metrics_second.csv"

    monkeypatch.setattr(train_dl_cnn, "build_cnn_model", lambda: _StubCnnModel())
    monkeypatch.setattr(train_dl_cnn, "CNN_METRICS_PATH", first_path)
    train_dl_cnn.main()

    monkeypatch.setattr(train_dl_cnn, "build_cnn_model", lambda: _StubCnnModel())
    monkeypatch.setattr(train_dl_cnn, "CNN_METRICS_PATH", second_path)
    train_dl_cnn.main()

    pd.testing.assert_frame_equal(pd.read_csv(first_path), pd.read_csv(second_path))
