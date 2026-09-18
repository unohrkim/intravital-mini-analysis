"""Pure data-preparation functions for the DL dataset layer.

Converts a tidy tracking table into per-frame-transition model input
sequences, and produces a deterministic, stratified, track-level
train/val/test split. No file I/O: every function here takes in-memory
data and returns in-memory data, so it can be reused unchanged by future
TensorFlow and PyTorch loaders regardless of where the data comes from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("track_id", "condition", "frame", "x_um", "y_um")
CONDITIONS: tuple[str, str] = ("control", "injury")
CONDITION_LABELS: dict[str, int] = {"control": 0, "injury": 1}
N_FRAMES: int = 30
SINUSOID_X_UM: float = 100.0


def tracks_to_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert a tidy tracking table into per-transition model input sequences.

    For each track and each transition t -> t+1 (t = 0..28), computes three
    channels: dx_t = x_(t+1) - x_t, dy_t = y_(t+1) - y_t, and
    relative_x_t = (x_t - 100.0) / 100.0 (signed position relative to the
    sinusoid, from the pre-step x, consistent with how the injury bias in
    src.simulate.directional_bias() is computed from the pre-step x).

    Returns (X, y, track_ids): X has shape (N, 29, 3) with channel order
    [dx, dy, relative_x], y has shape (N,) with control=0/injury=1, and
    track_ids[i] names the track that X[i]/y[i] came from.
    """
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {missing_columns}")

    observed_conditions = set(df["condition"].unique())
    if observed_conditions != set(CONDITIONS):
        raise ValueError(
            f"condition column must contain exactly {set(CONDITIONS)}, got {observed_conditions}"
        )

    conditions_per_track = df.groupby("track_id")["condition"].nunique()
    if (conditions_per_track > 1).any():
        bad = conditions_per_track.loc[conditions_per_track > 1].index.tolist()
        raise ValueError(f"track_id(s) span more than one condition: {bad}")

    rows_per_track = df.groupby("track_id").size()
    if (rows_per_track != N_FRAMES).any():
        bad = rows_per_track.loc[rows_per_track != N_FRAMES].to_dict()
        raise ValueError(f"track_id(s) do not have exactly {N_FRAMES} rows: {bad}")

    df_sorted = df.sort_values(["track_id", "frame"], kind="stable").reset_index(drop=True)

    n_tracks = df_sorted["track_id"].nunique()
    expected_frames = np.tile(np.arange(N_FRAMES), n_tracks)
    actual_frames = df_sorted["frame"].to_numpy()
    frame_mismatch = (actual_frames != expected_frames).reshape(n_tracks, N_FRAMES).any(axis=1)
    if frame_mismatch.any():
        track_ids_sorted = df_sorted["track_id"].drop_duplicates().to_numpy()
        bad = track_ids_sorted[frame_mismatch].tolist()
        raise ValueError(f"track_id(s) do not have frames exactly 0..{N_FRAMES - 1}: {bad}")

    if not np.isfinite(df_sorted["x_um"]).all() or not np.isfinite(df_sorted["y_um"]).all():
        raise ValueError("x_um and y_um must be finite (no NaN, +inf, or -inf)")

    # groupby(sort=True) matches df_sorted's own track_id ordering, so this
    # gives exactly one condition per track_id, aligned with the reshape below.
    track_conditions = df_sorted.groupby("track_id", sort=True)["condition"].first()
    track_ids = track_conditions.index.tolist()
    conditions = track_conditions.tolist()

    x = df_sorted["x_um"].to_numpy().reshape(n_tracks, N_FRAMES)
    y_coord = df_sorted["y_um"].to_numpy().reshape(n_tracks, N_FRAMES)

    dx = np.diff(x, axis=1)
    dy = np.diff(y_coord, axis=1)
    relative_x = (x[:, :-1] - SINUSOID_X_UM) / SINUSOID_X_UM

    X = np.stack([dx, dy, relative_x], axis=-1)
    y = np.array([CONDITION_LABELS[c] for c in conditions], dtype=np.int64)

    return X, y, track_ids


def split_track_ids(
    track_ids,
    conditions,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Deterministic, condition-stratified train/val/test split at the track level.

    Returns a DataFrame with columns track_id, condition, split (split is
    one of "train", "val", "test"). No file I/O.
    """
    track_ids = list(track_ids)
    conditions = list(conditions)

    if len(track_ids) != len(conditions):
        raise ValueError(
            f"track_ids and conditions must be the same length, got {len(track_ids)} and {len(conditions)}"
        )

    if len(set(track_ids)) != len(track_ids):
        raise ValueError("track_ids must be unique")

    observed_conditions = set(conditions)
    if observed_conditions != set(CONDITIONS):
        raise ValueError(
            f"conditions must contain exactly {set(CONDITIONS)}, got {observed_conditions}"
        )

    fractions = (train_frac, val_frac, test_frac)
    if not all(f > 0 for f in fractions):
        raise ValueError("train_frac, val_frac, and test_frac must all be positive")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = np.random.default_rng(seed)
    tracks_by_condition: dict[str, list[str]] = {condition: [] for condition in CONDITIONS}
    for track_id, condition in zip(track_ids, conditions):
        tracks_by_condition[condition].append(track_id)

    rows: list[tuple[str, str, str]] = []
    for condition in CONDITIONS:
        ids = np.array(sorted(tracks_by_condition[condition]))
        shuffled = rng.permutation(ids)
        n = len(shuffled)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)

        splits = (
            ("train", shuffled[:n_train]),
            ("val", shuffled[n_train : n_train + n_val]),
            ("test", shuffled[n_train + n_val :]),
        )
        for split_name, split_ids in splits:
            rows.extend((track_id, condition, split_name) for track_id in split_ids)

    return pd.DataFrame(rows, columns=["track_id", "condition", "split"])
