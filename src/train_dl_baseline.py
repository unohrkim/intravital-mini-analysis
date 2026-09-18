"""Fit and evaluate a logistic-regression baseline on the DL dataset layer.

Reads data/dl_tracks.csv and the already-persisted data/dl_split.csv (does
NOT regenerate the split -- see src/generate_dl_split.py for that). Builds
flattened per-track feature vectors via src.dl_data.tracks_to_sequences()
and src.dl_baseline.flatten_sequences(), fits sklearn's StandardScaler and
LogisticRegression on the train split only, and evaluates validation only.

Test-set discipline: the persisted test split is validated for structural
integrity (via lookup_split_indices) but is never scored by the model here.
It stays untouched by any modeling decision until Logistic Regression,
TensorFlow, and PyTorch architectures/hyperparameters are all frozen using
train/validation data alone; a later, separate stage evaluates all three
once each on the test split for the final comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.dl_baseline import build_results_table, compute_classification_metrics, flatten_sequences, lookup_split_indices
from src.dl_data import tracks_to_sequences

REPO_ROOT = Path(__file__).resolve().parent.parent
DL_TRACKS_PATH = REPO_ROOT / "data" / "dl_tracks.csv"  # read-only input
DL_SPLIT_PATH = REPO_ROOT / "data" / "dl_split.csv"  # read-only input
BASELINE_METRICS_PATH = REPO_ROOT / "results" / "dl" / "baseline_metrics.csv"

# Arbitrary implementation constant, not a scientific parameter -- happy to
# change if you'd prefer a different one. Distinct from DL_SEED=2026 and
# DL_SPLIT_SEED=2027. (Not needed for lbfgs's own convergence, which is
# deterministic; set for self-documentation and in case the solver ever
# changes to one that does use randomness.)
DL_BASELINE_RANDOM_STATE = 2028


def main() -> None:
    tracks = pd.read_csv(DL_TRACKS_PATH)
    split_df = pd.read_csv(DL_SPLIT_PATH)

    X, y, track_ids = tracks_to_sequences(tracks)
    X_flat = flatten_sequences(X)
    conditions_by_track = {
        track_id: ("injury" if label == 1 else "control") for track_id, label in zip(track_ids, y)
    }
    idx = lookup_split_indices(track_ids, conditions_by_track, split_df)
    # idx["test"] is validated above (as part of the persisted split's
    # integrity) but is never indexed into X/y below: the test split stays
    # untouched by every modeling decision until Logistic Regression, TF,
    # and PyTorch are all frozen -- evaluating it is a later, separate stage.

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_flat[idx["train"]])  # fit on train ONLY

    model = LogisticRegression(
        solver="lbfgs",
        l1_ratio=0,
        C=1.0,
        max_iter=1000,
        random_state=DL_BASELINE_RANDOM_STATE,
    )  # explicit config, no hyperparameter search
    model.fit(X_train_scaled, y[idx["train"]])

    X_val_scaled = scaler.transform(X_flat[idx["val"]])  # transform only, never re-fit
    y_val = y[idx["val"]]
    y_val_pred = model.predict(X_val_scaled)
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    results = build_results_table(
        [compute_classification_metrics(y_val, y_val_pred, y_val_prob, "val", DL_BASELINE_RANDOM_STATE)]
    )

    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(BASELINE_METRICS_PATH, index=False)
    print(f"Wrote {len(results)} rows to {BASELINE_METRICS_PATH}")
    print(
        "Note: this is a synthetic known-signal recovery benchmark, not a "
        "biological finding. It measures whether a linear baseline can "
        "recover the directional bias deliberately built into the "
        "injury condition's simulation (src/simulate.py). Test-split "
        "performance is intentionally not evaluated here -- it is "
        "reserved for a later, one-time comparison after Logistic "
        "Regression, TensorFlow, and PyTorch models are all frozen."
    )


if __name__ == "__main__":
    main()
