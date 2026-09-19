"""One-time final held-out test comparison across the three frozen DL baselines.

Evaluates Logistic Regression, the TensorFlow 1D CNN, and the PyTorch 1D CNN
once each on data/dl_split.csv's persisted "test" split, using exactly the
architecture, preprocessing, training configuration, seeds, and decision
thresholds already frozen in src/train_dl_baseline.py, src/train_dl_cnn.py,
and src/train_dl_cnn_torch.py. None of those files are modified; every
constant and pure building block used here is imported from them.

"Frozen" means the architecture, preprocessing rules, training
configuration (optimizer, batch size, epochs), random seeds, and decision
threshold are fixed -- NOT that a fitted model artifact was saved to disk.
Logistic Regression and the TF CNN have no persisted weights (nothing was
pickled/saved after the validation-stage runs), so each model here is
deterministically refit from scratch on data/dl_split.csv's "train" split,
using its frozen configuration, immediately before this one-time test
evaluation. Determinism follows from the same seeding already used in the
frozen scripts (DL_BASELINE_RANDOM_STATE / DL_TF_RANDOM_STATE /
DL_TORCH_RANDOM_STATE), not from reloading saved weights.

The "val" split is loaded (as part of validating the persisted split's
structural integrity via lookup_split_indices) but is never indexed into
X/y anywhere in this module -- none of the three refits touch validation
data in any way. In particular, the TF CNN's frozen validation-stage script
(src/train_dl_cnn.py) passed validation_data=... to model.fit() only for
per-epoch logging visibility; there were no callbacks, no early stopping,
no checkpoint selection, and no other validation-dependent training
decision tied to it. Dropping validation_data entirely from the refit here
is therefore behaviorally equivalent for the trained weights (Keras does
not alter gradients/weights based on validation_data absent such
callbacks) while making "validation is not used at all in this stage"
explicit and easy to verify, rather than merely true-by-absence-of-side-
effect.

This is final synthetic held-out test performance, run exactly once. It is
still not biological evidence -- it measures whether each model recovers/
generalizes the directional-bias signal deliberately encoded into the
injury condition's simulation (see src/simulate.py's directional_bias()).
No hyperparameter search, threshold tuning, model selection, ablation, or
calibration follows this run; its output is treated as final once produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.dl_baseline import compute_classification_metrics, flatten_sequences, lookup_split_indices
from src.dl_cnn import (
    apply_channel_scaler as apply_tf_channel_scaler,
    build_cnn_model as build_tf_cnn_model,
    enable_deterministic_ops,
    fit_channel_scaler as fit_tf_channel_scaler,
)
from src.dl_cnn_torch import (
    apply_channel_scaler as apply_torch_channel_scaler,
    fit_channel_scaler as fit_torch_channel_scaler,
)
from src.dl_data import tracks_to_sequences
from src.train_dl_baseline import DL_BASELINE_RANDOM_STATE
from src.train_dl_cnn import (
    CNN_BATCH_SIZE as TF_CNN_BATCH_SIZE,
    CNN_DECISION_THRESHOLD as TF_CNN_DECISION_THRESHOLD,
    CNN_EPOCHS as TF_CNN_EPOCHS,
    DL_TF_RANDOM_STATE,
)
from src.train_dl_cnn_torch import (
    CNN_DECISION_THRESHOLD as TORCH_CNN_DECISION_THRESHOLD,
    DL_TORCH_RANDOM_STATE,
    build_and_train_model,
    predict_probabilities as torch_predict_probabilities,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DL_TRACKS_PATH = REPO_ROOT / "data" / "dl_tracks.csv"  # read-only input
DL_SPLIT_PATH = REPO_ROOT / "data" / "dl_split.csv"  # read-only input
FINAL_COMPARISON_PATH = REPO_ROOT / "results" / "dl" / "final_test_comparison.csv"

# Not RESULTS_COLUMNS from src.dl_baseline -- that list (and
# build_results_table(), which is keyed to it) has no "model" column, and
# is shared by the frozen val-only scripts, so it is not modified here.
FINAL_RESULTS_COLUMNS: list[str] = [
    "model",
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


def evaluate_logistic_regression(X_flat: np.ndarray, y: np.ndarray, idx: dict[str, np.ndarray]) -> dict:
    """Refit the frozen Logistic Regression config on train, score test once.

    Config copied verbatim from src/train_dl_baseline.py's main() (which
    inlines it rather than exposing a reusable function) -- solver, l1_ratio,
    C, max_iter, and random_state are unchanged; only the imported
    DL_BASELINE_RANDOM_STATE constant is reused directly, not duplicated.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_flat[idx["train"]])  # fit on train ONLY

    model = LogisticRegression(
        solver="lbfgs",
        l1_ratio=0,
        C=1.0,
        max_iter=1000,
        random_state=DL_BASELINE_RANDOM_STATE,
    )
    model.fit(X_train_scaled, y[idx["train"]])

    X_test_scaled = scaler.transform(X_flat[idx["test"]])  # transform only, never re-fit
    y_test = y[idx["test"]]
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob, "test", DL_BASELINE_RANDOM_STATE)
    metrics["model"] = "logistic_regression"
    return metrics


def evaluate_tf_cnn(X: np.ndarray, y: np.ndarray, idx: dict[str, np.ndarray]) -> dict:
    """Refit the frozen TF 1D-CNN config on train, score test once.

    Architecture (build_tf_cnn_model), scaler, batch size, epoch count, and
    random state are all imported unchanged from src/dl_cnn.py and
    src/train_dl_cnn.py. No validation_data is passed to fit() -- see the
    module docstring for why that is behaviorally equivalent to the frozen
    validation-stage script's training result.
    """
    mean, scale = fit_tf_channel_scaler(X[idx["train"]])  # fit on train ONLY
    X_train_scaled = apply_tf_channel_scaler(X[idx["train"]], mean, scale)
    X_test_scaled = apply_tf_channel_scaler(X[idx["test"]], mean, scale)

    enable_deterministic_ops()
    tf.keras.utils.set_random_seed(DL_TF_RANDOM_STATE)
    model = build_tf_cnn_model()
    tf.keras.utils.set_random_seed(DL_TF_RANDOM_STATE)  # reset immediately before fit
    model.fit(
        X_train_scaled,
        y[idx["train"]],
        batch_size=TF_CNN_BATCH_SIZE,
        epochs=TF_CNN_EPOCHS,
        verbose=2,
    )

    y_test_prob = model.predict(X_test_scaled, verbose=0).reshape(-1)
    y_test_pred = (y_test_prob >= TF_CNN_DECISION_THRESHOLD).astype(int)

    metrics = compute_classification_metrics(y[idx["test"]], y_test_pred, y_test_prob, "test", DL_TF_RANDOM_STATE)
    metrics["model"] = "tensorflow_cnn"
    return metrics


def evaluate_torch_cnn(X: np.ndarray, y: np.ndarray, idx: dict[str, np.ndarray]) -> dict:
    """Refit the frozen PyTorch 1D-CNN config on train, score test once.

    build_and_train_model is imported unchanged from
    src/train_dl_cnn_torch.py and reused verbatim -- unlike the LR/TF
    baselines, that function was already factored out during the PyTorch
    stage, so no orchestration glue needs to be reproduced here.
    """
    mean, scale = fit_torch_channel_scaler(X[idx["train"]])  # fit on train ONLY
    X_train_scaled = apply_torch_channel_scaler(X[idx["train"]], mean, scale)
    X_test_scaled = apply_torch_channel_scaler(X[idx["test"]], mean, scale)

    model = build_and_train_model(X_train_scaled, y[idx["train"]])

    y_test_prob = torch_predict_probabilities(model, X_test_scaled)
    y_test_pred = (y_test_prob >= TORCH_CNN_DECISION_THRESHOLD).astype(int)

    metrics = compute_classification_metrics(y[idx["test"]], y_test_pred, y_test_prob, "test", DL_TORCH_RANDOM_STATE)
    metrics["model"] = "pytorch_cnn"
    return metrics


def main() -> None:
    tracks = pd.read_csv(DL_TRACKS_PATH)
    split_df = pd.read_csv(DL_SPLIT_PATH)

    X, y, track_ids = tracks_to_sequences(tracks)
    X_flat = flatten_sequences(X)
    conditions_by_track = {
        track_id: ("injury" if label == 1 else "control") for track_id, label in zip(track_ids, y)
    }
    idx = lookup_split_indices(track_ids, conditions_by_track, split_df)
    # idx["val"] is validated above (as part of the persisted split's
    # structural integrity) but is never indexed into X/y anywhere below --
    # this stage evaluates train-only refits on test only.

    rows = [
        evaluate_logistic_regression(X_flat, y, idx),
        evaluate_tf_cnn(X, y, idx),
        evaluate_torch_cnn(X, y, idx),
    ]
    results = pd.DataFrame(rows, columns=FINAL_RESULTS_COLUMNS)

    FINAL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(FINAL_COMPARISON_PATH, index=False)
    print(f"Wrote {len(results)} rows to {FINAL_COMPARISON_PATH}")
    print(
        "Note: this is final synthetic held-out test performance, not "
        "biological evidence. It measures whether each model recovers/ "
        "generalizes the directional bias deliberately built into the "
        "injury condition's simulation (src/simulate.py). Each model was "
        "refit from scratch on the train split using its frozen "
        "configuration (no saved weights are reused); validation data was "
        "not used in any of the three refits. This evaluation runs once; "
        "no tuning follows it."
    )


if __name__ == "__main__":
    main()
