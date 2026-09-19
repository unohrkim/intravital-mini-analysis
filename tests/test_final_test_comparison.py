"""Deterministic tests for the one-time final test-set comparison stage.

Covers src/final_test_comparison.py: the three evaluate_*() functions and
main()'s wiring.

Two tiers, mirroring the pattern established in tests/test_dl_cnn_torch.py:
- Per-model unit tests run real fitting/training on tiny synthetic data,
  with TF's epoch count and PyTorch's epoch count monkeypatched down to 2
  so nothing here trains for the real 30 epochs on the real 2800-track
  train split.
- main()-plumbing tests substitute all three evaluate_*() functions with a
  call-counting stub, so the real test-set evaluation of any model never
  actually runs inside pytest -- that only happens via the explicit
  `uv run python -m src.final_test_comparison` step.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import src.final_test_comparison as final_test_comparison
import src.train_dl_cnn_torch as train_dl_cnn_torch
from src.dl_baseline import N_CHANNELS, N_TRANSITIONS, flatten_sequences, lookup_split_indices
from src.dl_data import tracks_to_sequences
from src.final_test_comparison import FINAL_RESULTS_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent

# Read once at collection time (read-only -- never written to by any test),
# mirroring the DL_TRACKS pattern used across the other tests/test_dl_*.py files.
DL_TRACKS = pd.read_csv(REPO_ROOT / "data" / "dl_tracks.csv")
DL_SPLIT = pd.read_csv(REPO_ROOT / "data" / "dl_split.csv")
X, Y, TRACK_IDS = tracks_to_sequences(DL_TRACKS)
CONDITIONS_BY_TRACK = {
    track_id: ("injury" if label == 1 else "control") for track_id, label in zip(TRACK_IDS, Y)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_dataset(n_train: int, n_val: int, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Tiny synthetic (X, y, idx) with alternating labels.

    Labels alternate (0, 1, 0, 1, ...) rather than being drawn randomly, so
    every contiguous split (train/val/test) of length >= 2 is guaranteed to
    contain both classes -- avoiding a flaky roc_auc_score failure (which
    requires both classes present) on an unlucky random draw.
    """
    total = n_train + n_val + n_test
    rng = np.random.default_rng(seed)
    X_tiny = rng.normal(size=(total, N_TRANSITIONS, N_CHANNELS)).astype("float32")
    y_tiny = np.array([i % 2 for i in range(total)])
    idx = {
        "train": np.arange(0, n_train),
        "val": np.arange(n_train, n_train + n_val),
        "test": np.arange(n_train + n_val, total),
    }
    return X_tiny, y_tiny, idx


class _CallCounter:
    """Stub for an evaluate_*() function: records call count, returns a
    fixed-but-valid metrics dict without running any real model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0

    def __call__(self, X, y, idx):
        self.call_count += 1
        n_test = len(idx["test"])
        return {
            "model": self.model_name,
            "split": "test",
            "n_samples": n_test,
            "n_positive": n_test // 2,
            "n_negative": n_test - n_test // 2,
            "accuracy": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "f1": 0.6,
            "roc_auc": 0.6,
            "random_state": 999,
        }


# --- evaluate_logistic_regression (real sklearn fit, tiny data) ---


def test_evaluate_logistic_regression_fits_on_train_only_and_scores_test(monkeypatch):
    X_tiny, y_tiny, idx = _tiny_dataset(
        n_train=12,
        n_val=5,
        n_test=4,
        seed=1,
    )
    X_flat_tiny = flatten_sequences(X_tiny)

    fit_calls: list[np.ndarray] = []
    transform_calls: list[np.ndarray] = []

    original_fit = StandardScaler.fit
    original_transform = StandardScaler.transform

    def _spy_fit(self, X_arg, *args, **kwargs):
        fit_calls.append(np.asarray(X_arg).copy())
        return original_fit(self, X_arg, *args, **kwargs)

    def _spy_transform(self, X_arg, *args, **kwargs):
        transform_calls.append(np.asarray(X_arg).copy())
        return original_transform(self, X_arg, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", _spy_fit)
    monkeypatch.setattr(StandardScaler, "transform", _spy_transform)

    metrics = final_test_comparison.evaluate_logistic_regression(
        X_flat_tiny,
        y_tiny,
        idx,
    )

    assert len(fit_calls) == 1
    assert np.array_equal(
        fit_calls[0],
        X_flat_tiny[idx["train"]],
    )

    assert any(
        np.array_equal(call, X_flat_tiny[idx["test"]])
        for call in transform_calls
    )

    assert not any(
        np.array_equal(call, X_flat_tiny[idx["val"]])
        for call in fit_calls + transform_calls
    )

    assert metrics["model"] == "logistic_regression"
    assert metrics["split"] == "test"
    assert metrics["n_samples"] == 4
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[metric] <= 1.0


# --- evaluate_tf_cnn (real TF training, tiny data, epochs monkeypatched to 2) ---


def test_evaluate_tf_cnn_never_receives_validation_data(monkeypatch):
    monkeypatch.setattr(final_test_comparison, "TF_CNN_EPOCHS", 2)
    monkeypatch.setattr(final_test_comparison, "TF_CNN_BATCH_SIZE", 8)

    X_tiny, y_tiny, idx = _tiny_dataset(n_train=16, n_val=4, n_test=8, seed=2)

    fit_kwargs_seen: dict = {}
    real_build = final_test_comparison.build_tf_cnn_model

    def _wrapped_build():
        model = real_build()
        original_model_fit = model.fit

        def _spy_fit(*args, **kwargs):
            fit_kwargs_seen.update(kwargs)
            return original_model_fit(*args, **kwargs)

        model.fit = _spy_fit
        return model

    monkeypatch.setattr(final_test_comparison, "build_tf_cnn_model", _wrapped_build)

    metrics = final_test_comparison.evaluate_tf_cnn(X_tiny, y_tiny, idx)

    assert "validation_data" not in fit_kwargs_seen
    assert metrics["model"] == "tensorflow_cnn"
    assert metrics["split"] == "test"
    assert metrics["n_samples"] == 8
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[metric] <= 1.0


# --- evaluate_torch_cnn (real PyTorch training, tiny data, epochs monkeypatched to 2) ---


def test_evaluate_torch_cnn_fits_on_train_only_and_scores_test(monkeypatch):
    monkeypatch.setattr(train_dl_cnn_torch, "CNN_EPOCHS", 2)

    X_tiny, y_tiny, idx = _tiny_dataset(n_train=16, n_val=4, n_test=8, seed=3)

    calls: list[np.ndarray] = []
    original_build_and_train = final_test_comparison.build_and_train_model

    def _spy_build_and_train(X_train_arg, y_train_arg):
        calls.append(np.asarray(X_train_arg))
        return original_build_and_train(X_train_arg, y_train_arg)

    monkeypatch.setattr(final_test_comparison, "build_and_train_model", _spy_build_and_train)

    metrics = final_test_comparison.evaluate_torch_cnn(X_tiny, y_tiny, idx)

    assert len(calls) == 1
    assert calls[0].shape[0] == 16  # fit on train only
    assert metrics["model"] == "pytorch_cnn"
    assert metrics["split"] == "test"
    assert metrics["n_samples"] == 8
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[metric] <= 1.0


# --- split-lookup reuse (no model involved; real persisted split) ---


def test_final_comparison_uses_persisted_split_exactly():
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    assert len(idx["train"]) == 2800
    assert len(idx["val"]) == 600
    assert len(idx["test"]) == 600

    test_conditions = [CONDITIONS_BY_TRACK[TRACK_IDS[i]] for i in idx["test"]]
    assert test_conditions.count("control") == 300
    assert test_conditions.count("injury") == 300


# --- main() plumbing (all three evaluate_*() stubbed -- no real model runs) ---


def test_final_comparison_evaluates_each_model_exactly_once(tmp_path, monkeypatch):
    lr_stub = _CallCounter("logistic_regression")
    tf_stub = _CallCounter("tensorflow_cnn")
    torch_stub = _CallCounter("pytorch_cnn")

    monkeypatch.setattr(final_test_comparison, "evaluate_logistic_regression", lr_stub)
    monkeypatch.setattr(final_test_comparison, "evaluate_tf_cnn", tf_stub)
    monkeypatch.setattr(final_test_comparison, "evaluate_torch_cnn", torch_stub)
    monkeypatch.setattr(final_test_comparison, "FINAL_COMPARISON_PATH", tmp_path / "final_test_comparison.csv")

    final_test_comparison.main()

    assert lr_stub.call_count == 1
    assert tf_stub.call_count == 1
    assert torch_stub.call_count == 1


def test_final_comparison_writes_one_test_row_per_model(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test_comparison, "evaluate_logistic_regression", _CallCounter("logistic_regression"))
    monkeypatch.setattr(final_test_comparison, "evaluate_tf_cnn", _CallCounter("tensorflow_cnn"))
    monkeypatch.setattr(final_test_comparison, "evaluate_torch_cnn", _CallCounter("pytorch_cnn"))
    monkeypatch.setattr(final_test_comparison, "FINAL_COMPARISON_PATH", tmp_path / "final_test_comparison.csv")

    final_test_comparison.main()

    results = pd.read_csv(tmp_path / "final_test_comparison.csv")
    assert len(results) == 3
    assert set(results["split"]) == {"test"}
    assert sorted(results["model"]) == ["logistic_regression", "pytorch_cnn", "tensorflow_cnn"]
    assert list(results.columns) == FINAL_RESULTS_COLUMNS


def test_final_comparison_metrics_in_unit_range(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test_comparison, "evaluate_logistic_regression", _CallCounter("logistic_regression"))
    monkeypatch.setattr(final_test_comparison, "evaluate_tf_cnn", _CallCounter("tensorflow_cnn"))
    monkeypatch.setattr(final_test_comparison, "evaluate_torch_cnn", _CallCounter("pytorch_cnn"))
    monkeypatch.setattr(final_test_comparison, "FINAL_COMPARISON_PATH", tmp_path / "final_test_comparison.csv")

    final_test_comparison.main()

    results = pd.read_csv(tmp_path / "final_test_comparison.csv")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert results[metric].between(0.0, 1.0).all()


def test_final_comparison_does_not_mutate_real_files(tmp_path, monkeypatch):
    frozen_files = [
        REPO_ROOT / "data" / "dl_tracks.csv",
        REPO_ROOT / "data" / "dl_split.csv",
        REPO_ROOT / "src" / "dl_data.py",
        REPO_ROOT / "src" / "dl_baseline.py",
        REPO_ROOT / "src" / "train_dl_baseline.py",
        REPO_ROOT / "src" / "dl_cnn.py",
        REPO_ROOT / "src" / "train_dl_cnn.py",
        REPO_ROOT / "src" / "dl_cnn_torch.py",
        REPO_ROOT / "src" / "train_dl_cnn_torch.py",
    ]
    hashes_before = {path: _sha256(path) for path in frozen_files}

    monkeypatch.setattr(final_test_comparison, "evaluate_logistic_regression", _CallCounter("logistic_regression"))
    monkeypatch.setattr(final_test_comparison, "evaluate_tf_cnn", _CallCounter("tensorflow_cnn"))
    monkeypatch.setattr(final_test_comparison, "evaluate_torch_cnn", _CallCounter("pytorch_cnn"))
    monkeypatch.setattr(final_test_comparison, "FINAL_COMPARISON_PATH", tmp_path / "final_test_comparison.csv")

    final_test_comparison.main()

    for path, hash_before in hashes_before.items():
        assert _sha256(path) == hash_before
