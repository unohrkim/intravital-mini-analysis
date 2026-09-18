"""Pure logistic-regression baseline building blocks for the DL dataset layer.

Flattens tracks_to_sequences() output into per-track feature vectors, looks
up the already-persisted train/val/test split (data/dl_split.csv), and
computes classification metrics. No file I/O, and no sklearn model
construction here (StandardScaler/LogisticRegression fitting lives in
src/train_dl_baseline.py) so this stays a thin, directly testable layer.

This baseline is a synthetic known-signal recovery benchmark: it tests
whether a simple linear classifier can recover the directional-bias signal
deliberately built into the injury condition's simulation (see
src/simulate.py's directional_bias()). It is not a biological or causal
claim and does not validate the simulation model (see src/stats_analysis.py
for the same caveat applied to Step 3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

N_TRANSITIONS: int = 29
N_CHANNELS: int = 3
N_FLATTENED_FEATURES: int = N_TRANSITIONS * N_CHANNELS  # 87
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

RESULTS_COLUMNS: list[str] = [
    "split",
    "n_samples",
    "n_positive",
    "n_negative",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "random_state",
]


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten (N, 29, 3) per-track sequences into (N, 87) feature vectors.

    Row i is [dx_0, dy_0, relx_0, dx_1, dy_1, relx_1, ..., dx_28, dy_28,
    relx_28] (numpy's default C-order reshape).

    Validates the exact data contract, not just that the trailing
    dimensions multiply to N_FLATTENED_FEATURES (which would silently
    accept a wrong-shaped array such as (N, 3, 29) or (N, 1, 87)): raises
    ValueError unless X.ndim == 3 and X.shape[1:] == (N_TRANSITIONS,
    N_CHANNELS).
    """
    if X.ndim != 3 or X.shape[1:] != (N_TRANSITIONS, N_CHANNELS):
        raise ValueError(
            f"Expected X with shape (N, {N_TRANSITIONS}, {N_CHANNELS}), got {X.shape}"
        )

    return X.reshape(X.shape[0], N_FLATTENED_FEATURES)


def lookup_split_indices(
    track_ids: Sequence[str],
    conditions_by_track: Mapping[str, str],
    split_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Map "train"/"val"/"test" to integer positions into track_ids.

    The returned indices index directly into track_ids (and therefore into
    any array row-aligned with track_ids, such as X/y from
    tracks_to_sequences()). Does NOT call split_track_ids() -- only reads
    split_df's existing "split" column, i.e. it uses the persisted split
    exactly as-is rather than regenerating one.

    Validation, in order, each raising ValueError:
      1. split_df has columns track_id, condition, split.
      2. split_df["track_id"] values are unique.
      3. split_df["split"] values are exactly {"train", "val", "test"}.
      4. set(track_ids) == set(split_df["track_id"]) exactly (catches
         drift between tracks_to_sequences() output and the persisted
         split file).
      5. For every track_id, split_df's recorded condition matches
         conditions_by_track[track_id] (catches a stale/mismatched split
         file that assigns splits under a different track_id -> condition
         mapping than the current data/dl_tracks.csv).
    """
    required_columns = ("track_id", "condition", "split")
    missing_columns = [column for column in required_columns if column not in split_df.columns]
    if missing_columns:
        raise ValueError(f"split_df is missing required column(s): {missing_columns}")

    if split_df["track_id"].duplicated().any():
        duplicates = split_df.loc[split_df["track_id"].duplicated(), "track_id"].unique().tolist()
        raise ValueError(f"split_df has duplicate track_id(s): {duplicates}")

    observed_splits = set(split_df["split"].unique())
    if observed_splits != set(SPLIT_NAMES):
        raise ValueError(
            f"split_df['split'] must contain exactly {set(SPLIT_NAMES)}, got {observed_splits}"
        )

    track_id_set = set(track_ids)
    split_track_id_set = set(split_df["track_id"])
    if track_id_set != split_track_id_set:
        only_in_track_ids = sorted(track_id_set - split_track_id_set)[:10]
        only_in_split_df = sorted(split_track_id_set - track_id_set)[:10]
        raise ValueError(
            "track_ids and split_df['track_id'] must match exactly. "
            f"Only in track_ids (up to 10 shown): {only_in_track_ids}. "
            f"Only in split_df (up to 10 shown): {only_in_split_df}."
        )

    condition_by_track_in_split = split_df.set_index("track_id")["condition"]
    mismatched = [
        track_id
        for track_id in track_ids
        if condition_by_track_in_split[track_id] != conditions_by_track[track_id]
    ]
    if mismatched:
        raise ValueError(
            f"split_df condition disagrees with conditions_by_track for track_id(s): {mismatched[:10]}"
        )

    split_by_track = split_df.set_index("track_id")["split"]
    indices: dict[str, list[int]] = {split_name: [] for split_name in SPLIT_NAMES}
    for position, track_id in enumerate(track_ids):
        indices[split_by_track[track_id]].append(position)

    return {split_name: np.array(positions, dtype=int) for split_name, positions in indices.items()}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str,
    random_state: int,
) -> dict:
    """accuracy/precision/recall/f1/roc_auc for one split, keyed to match RESULTS_COLUMNS.

    y_prob is P(injury=1) per sample (the positive class, matching
    CONDITION_LABELS = {"control": 0, "injury": 1} in src/dl_data.py).
    precision/recall/f1 use sklearn's binary default pos_label=1 and
    zero_division=0. Raises ValueError if any y_prob value is not finite
    (NaN, +inf, -inf) or falls outside [0, 1] -- NaN comparisons against
    0.0 and 1.0 both evaluate to False, so an explicit finiteness check is
    required to actually catch NaN.
    """
    y_prob = np.asarray(y_prob)
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("y_prob must contain only finite values in [0, 1]")

    return {
        "split": split_name,
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "random_state": random_state,
    }


def build_results_table(rows: list[dict]) -> pd.DataFrame:
    """One row per dict (as produced by compute_classification_metrics), in RESULTS_COLUMNS order."""
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)
