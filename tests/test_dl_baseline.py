"""Deterministic tests for the logistic-regression baseline stage.

Covers src/dl_baseline.py (pure sequence-flattening, split-lookup, and
metrics building blocks) and the main() entry point of
src/train_dl_baseline.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import src.train_dl_baseline as train_dl_baseline
from src.dl_baseline import (
    RESULTS_COLUMNS,
    build_results_table,
    compute_classification_metrics,
    flatten_sequences,
    lookup_split_indices,
)
from src.dl_data import tracks_to_sequences
from src.train_dl_baseline import DL_BASELINE_RANDOM_STATE

REPO_ROOT = Path(__file__).resolve().parent.parent

# Read once at collection time (read-only -- never written to by any test)
# and reused across tests, mirroring the DL_TRACKS pattern in
# tests/test_dl_data.py.
DL_TRACKS = pd.read_csv(REPO_ROOT / "data" / "dl_tracks.csv")
DL_SPLIT = pd.read_csv(REPO_ROOT / "data" / "dl_split.csv")
X, Y, TRACK_IDS = tracks_to_sequences(DL_TRACKS)
CONDITIONS_BY_TRACK = {
    track_id: ("injury" if label == 1 else "control") for track_id, label in zip(TRACK_IDS, Y)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_df(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["track_id", "condition", "split"])


# --- flatten_sequences ---


def test_flatten_sequences_shape_is_87():
    X_flat = flatten_sequences(X)
    assert X_flat.shape == (X.shape[0], 87)


def test_flatten_sequences_preserves_channel_order():
    X_flat = flatten_sequences(X)
    for row in (0, 1, X.shape[0] - 1):
        assert np.array_equal(X_flat[row], X[row].reshape(-1))


@pytest.mark.parametrize(
    "bad_shape",
    [
        (10, 87),  # 2-D
        (10, 29, 2),  # wrong channel count
        (10, 3, 29),  # transposed axes -- still 87 elements/row
        (10, 1, 87),  # pre-flattened -- still 87 elements/row
    ],
)
def test_flatten_sequences_rejects_malformed_shape(bad_shape):
    bad_X = np.zeros(bad_shape)
    with pytest.raises(ValueError):
        flatten_sequences(bad_X)


# --- lookup_split_indices ---


def test_lookup_split_indices_uses_persisted_split_exactly():
    track_ids = ["C1", "C2", "I1", "I2"]
    conditions = {"C1": "control", "C2": "control", "I1": "injury", "I2": "injury"}
    split_df = _split_df(
        [
            ("C1", "control", "train"),
            ("C2", "control", "val"),
            ("I1", "injury", "train"),
            ("I2", "injury", "test"),
        ]
    )

    idx = lookup_split_indices(track_ids, conditions, split_df)

    assert idx["train"].tolist() == [0, 2]
    assert idx["val"].tolist() == [1]
    assert idx["test"].tolist() == [3]


def test_lookup_split_indices_no_overlap_between_splits():
    track_ids = ["C1", "C2", "C3", "I1", "I2", "I3"]
    conditions = {tid: ("injury" if tid.startswith("I") else "control") for tid in track_ids}
    split_df = _split_df(
        [
            ("C1", "control", "train"),
            ("C2", "control", "val"),
            ("C3", "control", "test"),
            ("I1", "injury", "train"),
            ("I2", "injury", "val"),
            ("I3", "injury", "test"),
        ]
    )

    idx = lookup_split_indices(track_ids, conditions, split_df)
    sets = {name: set(positions.tolist()) for name, positions in idx.items()}

    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    assert sets["train"] | sets["val"] | sets["test"] == set(range(len(track_ids)))


def test_lookup_split_indices_rejects_track_id_set_mismatch():
    # split_df's track_id set is {C1, C2, C4}; track_ids is {C1, C2, C3}.
    track_ids = ["C1", "C2", "C3"]
    conditions = {"C1": "control", "C2": "control", "C3": "control"}
    split_df = _split_df(
        [
            ("C1", "control", "train"),
            ("C2", "control", "val"),
            ("C4", "control", "test"),
        ]
    )

    with pytest.raises(ValueError):
        lookup_split_indices(track_ids, conditions, split_df)


def test_lookup_split_indices_rejects_condition_mismatch():
    track_ids = ["C1", "I1", "I2"]
    conditions = {"C1": "control", "I1": "injury", "I2": "injury"}
    split_df = _split_df(
        [
            ("C1", "control", "train"),
            ("I1", "control", "val"),  # should be "injury"
            ("I2", "injury", "test"),
        ]
    )

    with pytest.raises(ValueError):
        lookup_split_indices(track_ids, conditions, split_df)


def test_lookup_split_indices_rejects_duplicate_track_id_in_split_df():
    track_ids = ["C1", "I1"]
    conditions = {"C1": "control", "I1": "injury"}
    split_df = _split_df(
        [
            ("C1", "control", "train"),
            ("C1", "control", "val"),
            ("I1", "injury", "test"),
        ]
    )

    with pytest.raises(ValueError):
        lookup_split_indices(track_ids, conditions, split_df)


def test_lookup_split_indices_rejects_unexpected_split_value():
    track_ids = ["C1", "I1"]
    conditions = {"C1": "control", "I1": "injury"}
    split_df = _split_df(
        [
            ("C1", "control", "validation"),  # not "val"
            ("I1", "injury", "test"),
        ]
    )

    with pytest.raises(ValueError):
        lookup_split_indices(track_ids, conditions, split_df)


def test_lookup_split_indices_matches_real_dl_split_counts():
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    assert len(idx["train"]) == 2800
    assert len(idx["val"]) == 600
    assert len(idx["test"]) == 600

    sets = {name: set(positions.tolist()) for name, positions in idx.items()}
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    assert sets["train"] | sets["val"] | sets["test"] == set(range(len(TRACK_IDS)))


# --- compute_classification_metrics ---


def test_compute_classification_metrics_hand_computed():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, "val", 42)

    # Hand-verified confusion counts: TP=1 (idx1), TN=2 (idx0,idx3), FN=1 (idx2), FP=0.
    assert metrics["split"] == "val"
    assert metrics["random_state"] == 42
    assert metrics["n_samples"] == 4
    assert metrics["n_positive"] == 2
    assert metrics["n_negative"] == 2
    assert metrics["accuracy"] == pytest.approx(3 / 4)
    assert metrics["precision"] == pytest.approx(1 / 1)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["f1"] == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))
    # Independently recomputed via sklearn directly, not reusing internals of the function under test.
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_prob))


@pytest.mark.parametrize("bad_prob", [1.5, -0.1, np.nan, np.inf, -np.inf])
def test_compute_classification_metrics_rejects_invalid_probability(bad_prob):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_prob = np.array([0.5, bad_prob])

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_prob, "val", 1)


# --- build_results_table ---


def test_build_results_table_has_expected_columns_and_row_count():
    rows = [compute_classification_metrics(np.array([0, 1]), np.array([0, 1]), np.array([0.1, 0.9]), "val", 7)]

    table = build_results_table(rows)

    assert list(table.columns) == RESULTS_COLUMNS
    assert len(table) == 1


# --- end-to-end / integration ---


def test_label_alignment_with_track_ids():
    condition_from_tracks = DL_TRACKS.drop_duplicates("track_id").set_index("track_id")["condition"]
    condition_from_split = DL_SPLIT.set_index("track_id")["condition"]

    for track_id, label in zip(TRACK_IDS, Y):
        expected_condition = "injury" if label == 1 else "control"
        assert condition_from_tracks[track_id] == expected_condition
        assert condition_from_split[track_id] == expected_condition


def test_scaler_is_fit_only_on_training_data():
    X_flat = flatten_sequences(X)
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    scaler = StandardScaler().fit(X_flat[idx["train"]])

    expected_mean = np.mean(X_flat[idx["train"]], axis=0)
    expected_std = np.std(X_flat[idx["train"]], axis=0)

    assert np.allclose(scaler.mean_, expected_mean)
    assert np.allclose(scaler.scale_, expected_std)


def test_predicted_probabilities_are_within_unit_range():
    X_flat = flatten_sequences(X)
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    scaler = StandardScaler().fit(X_flat[idx["train"]])
    X_train_scaled = scaler.transform(X_flat[idx["train"]])
    X_val_scaled = scaler.transform(X_flat[idx["val"]])

    model = LogisticRegression(
        solver="lbfgs", l1_ratio=0, C=1.0, max_iter=1000, random_state=DL_BASELINE_RANDOM_STATE
    )
    model.fit(X_train_scaled, Y[idx["train"]])
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

    assert np.all(y_val_prob >= 0.0)
    assert np.all(y_val_prob <= 1.0)


def test_model_predictions_are_deterministic_across_runs():
    X_flat = flatten_sequences(X)
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)

    def _fit_and_predict_val():
        scaler = StandardScaler().fit(X_flat[idx["train"]])
        X_train_scaled = scaler.transform(X_flat[idx["train"]])
        X_val_scaled = scaler.transform(X_flat[idx["val"]])
        model = LogisticRegression(
            solver="lbfgs", l1_ratio=0, C=1.0, max_iter=1000, random_state=DL_BASELINE_RANDOM_STATE
        )
        model.fit(X_train_scaled, Y[idx["train"]])
        return model.predict_proba(X_val_scaled)[:, 1]

    first = _fit_and_predict_val()
    second = _fit_and_predict_val()
    np.testing.assert_array_equal(first, second)


def test_train_dl_baseline_never_touches_test_split(tmp_path, monkeypatch):
    X_flat = flatten_sequences(X)
    idx = lookup_split_indices(TRACK_IDS, CONDITIONS_BY_TRACK, DL_SPLIT)
    test_rows = X_flat[idx["test"]]

    transformed_inputs: list[np.ndarray] = []
    original_transform = StandardScaler.transform

    def _spy_transform(self, X_arg, *args, **kwargs):
        transformed_inputs.append(np.asarray(X_arg))
        return original_transform(self, X_arg, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "transform", _spy_transform)
    monkeypatch.setattr(train_dl_baseline, "BASELINE_METRICS_PATH", tmp_path / "baseline_metrics.csv")

    train_dl_baseline.main()

    # StandardScaler.transform is called once inside fit_transform (train)
    # and once explicitly (val); none of those calls' input arrays equal
    # the test split's rows (checked by content, not just shape, since
    # val and test happen to have the same row count: 600 each).
    for call_array in transformed_inputs:
        if call_array.shape == test_rows.shape:
            assert not np.array_equal(call_array, test_rows)

    results = pd.read_csv(tmp_path / "baseline_metrics.csv")
    assert set(results["split"]) == {"val"}


def test_train_dl_baseline_main_writes_expected_metrics_without_mutating_real_files(tmp_path, monkeypatch):
    dl_tracks_hash_before = _sha256(REPO_ROOT / "data" / "dl_tracks.csv")
    dl_split_hash_before = _sha256(REPO_ROOT / "data" / "dl_split.csv")

    monkeypatch.setattr(train_dl_baseline, "BASELINE_METRICS_PATH", tmp_path / "baseline_metrics.csv")
    train_dl_baseline.main()

    assert _sha256(REPO_ROOT / "data" / "dl_tracks.csv") == dl_tracks_hash_before
    assert _sha256(REPO_ROOT / "data" / "dl_split.csv") == dl_split_hash_before

    results = pd.read_csv(tmp_path / "baseline_metrics.csv")
    assert len(results) == 1
    assert results.loc[0, "split"] == "val"
    assert list(results.columns) == RESULTS_COLUMNS
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= results.loc[0, metric] <= 1.0


def test_train_dl_baseline_main_is_deterministic_across_runs(tmp_path, monkeypatch):
    first_path = tmp_path / "baseline_metrics_first.csv"
    second_path = tmp_path / "baseline_metrics_second.csv"

    monkeypatch.setattr(train_dl_baseline, "BASELINE_METRICS_PATH", first_path)
    train_dl_baseline.main()

    monkeypatch.setattr(train_dl_baseline, "BASELINE_METRICS_PATH", second_path)
    train_dl_baseline.main()

    pd.testing.assert_frame_equal(pd.read_csv(first_path), pd.read_csv(second_path))
