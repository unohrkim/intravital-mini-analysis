"""Deterministic tests for the DL dataset layer.

Covers src/dl_data.py (pure sequence conversion + split logic) and the
main() entry points of src/generate_dl_tracks.py and src/generate_dl_split.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.generate_dl_split as generate_dl_split
import src.generate_dl_tracks as generate_dl_tracks
from src.dl_data import (
    N_FRAMES,
    REQUIRED_COLUMNS,
    SINUSOID_X_UM,
    split_track_ids,
    tracks_to_sequences,
)
from src.generate_dl_split import DL_SPLIT_SEED
from src.generate_dl_tracks import DL_CELLS_PER_CONDITION, DL_SEED
from src.simulate import generate_tracks

REPO_ROOT = Path(__file__).resolve().parent.parent

# Computed once at collection time and reused across tests, since
# generate_tracks() is a pure, deterministic function.
DL_TRACKS = generate_tracks(cells_per_condition=DL_CELLS_PER_CONDITION, n_frames=30, seed=DL_SEED)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _two_condition_df(control_track_id: str = "C01", injury_track_id: str = "I01", seed: int = 0) -> pd.DataFrame:
    """A minimal well-formed table with one control and one injury track."""
    rng = np.random.default_rng(seed)
    frames = list(range(N_FRAMES))
    control = pd.DataFrame(
        {
            "track_id": control_track_id,
            "condition": "control",
            "frame": frames,
            "x_um": rng.uniform(0.0, 200.0, size=N_FRAMES),
            "y_um": rng.uniform(0.0, 200.0, size=N_FRAMES),
        }
    )
    injury = pd.DataFrame(
        {
            "track_id": injury_track_id,
            "condition": "injury",
            "frame": frames,
            "x_um": rng.uniform(0.0, 200.0, size=N_FRAMES),
            "y_um": rng.uniform(0.0, 200.0, size=N_FRAMES),
        }
    )
    return pd.concat([control, injury], ignore_index=True)


def _sample_track_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(n)]


# --- generation / structure ---


def test_dl_tracks_has_expected_track_and_row_counts():
    expected_track_count = 2 * DL_CELLS_PER_CONDITION
    expected_row_count = expected_track_count * N_FRAMES

    assert DL_TRACKS["track_id"].nunique() == expected_track_count
    assert len(DL_TRACKS) == expected_row_count


def test_dl_tracks_each_track_has_30_frames():
    assert (DL_TRACKS.groupby("track_id").size() == N_FRAMES).all()


def test_dl_tracks_labels_are_balanced():
    counts = DL_TRACKS.drop_duplicates("track_id")["condition"].value_counts()
    assert counts["control"] == DL_CELLS_PER_CONDITION
    assert counts["injury"] == DL_CELLS_PER_CONDITION


def test_dl_generation_is_deterministic_with_seed_2026():
    first = generate_tracks(cells_per_condition=DL_CELLS_PER_CONDITION, n_frames=30, seed=DL_SEED)
    second = generate_tracks(cells_per_condition=DL_CELLS_PER_CONDITION, n_frames=30, seed=DL_SEED)
    pdt.assert_frame_equal(first, second)


# --- tracks_to_sequences: happy path ---


def test_tracks_to_sequences_shape():
    X, y, track_ids = tracks_to_sequences(DL_TRACKS)
    expected_n = 2 * DL_CELLS_PER_CONDITION

    assert X.shape == (expected_n, N_FRAMES - 1, 3)
    assert y.shape == (expected_n,)
    assert len(track_ids) == expected_n


def test_tracks_to_sequences_labels_match_condition():
    X, y, track_ids = tracks_to_sequences(DL_TRACKS)
    condition_by_track = DL_TRACKS.drop_duplicates("track_id").set_index("track_id")["condition"]

    for track_id, label in zip(track_ids, y):
        expected_label = 0 if condition_by_track[track_id] == "control" else 1
        assert label == expected_label


def test_tracks_to_sequences_matches_manual_computation():
    track_id = "C01"
    track = DL_TRACKS.loc[DL_TRACKS["track_id"] == track_id].sort_values("frame")
    x = track["x_um"].to_numpy()
    y_coord = track["y_um"].to_numpy()

    expected_dx = np.diff(x)
    expected_dy = np.diff(y_coord)
    expected_relative_x = (x[:-1] - SINUSOID_X_UM) / SINUSOID_X_UM

    X, _y, track_ids = tracks_to_sequences(DL_TRACKS)
    row = track_ids.index(track_id)

    assert X[row, :, 0] == pytest.approx(expected_dx)
    assert X[row, :, 1] == pytest.approx(expected_dy)
    assert X[row, :, 2] == pytest.approx(expected_relative_x)


def test_relative_x_uses_pre_step_position():
    # x crosses the sinusoid (x=100) between frame 5 and frame 6.
    x_values = np.array([90.0] * 6 + [110.0] * (N_FRAMES - 6))
    y_values = np.linspace(50.0, 150.0, N_FRAMES)

    injury = pd.DataFrame(
        {
            "track_id": "I_CROSS",
            "condition": "injury",
            "frame": list(range(N_FRAMES)),
            "x_um": x_values,
            "y_um": y_values,
        }
    )
    rng = np.random.default_rng(1)
    control = pd.DataFrame(
        {
            "track_id": "C_CROSS",
            "condition": "control",
            "frame": list(range(N_FRAMES)),
            "x_um": rng.uniform(0.0, 200.0, size=N_FRAMES),
            "y_um": rng.uniform(0.0, 200.0, size=N_FRAMES),
        }
    )
    df = pd.concat([control, injury], ignore_index=True)

    X, _y, track_ids = tracks_to_sequences(df)
    row = track_ids.index("I_CROSS")

    # Transition index 5 is frame 5 -> frame 6: relative_x must use x_5 (=90),
    # not x_6 (=110).
    assert X[row, 5, 2] == pytest.approx(-0.1)
    assert X[row, 5, 2] != pytest.approx(0.1)


def test_tracks_to_sequences_is_invariant_to_row_order():
    shuffled = DL_TRACKS.sample(frac=1.0, random_state=42).reset_index(drop=True)

    X_original, y_original, track_ids_original = tracks_to_sequences(DL_TRACKS)
    X_shuffled, y_shuffled, track_ids_shuffled = tracks_to_sequences(shuffled)

    assert track_ids_shuffled == track_ids_original
    assert np.array_equal(y_shuffled, y_original)
    assert np.array_equal(X_shuffled, X_original)


# --- tracks_to_sequences: validation failures ---


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_rejects_missing_required_column(column):
    df = _two_condition_df().drop(columns=[column])
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


def test_rejects_unexpected_condition_value():
    df = _two_condition_df()
    df.loc[df["track_id"] == "I01", "condition"] = "unknown"
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


def test_rejects_when_only_one_condition_present():
    df = _two_condition_df()
    df = df.loc[df["condition"] == "control"].copy()
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


def test_rejects_track_id_spanning_two_conditions():
    df = _two_condition_df()
    df.loc[(df["track_id"] == "I01") & (df["frame"] == 0), "condition"] = "control"
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


@pytest.mark.parametrize("drop_last_row", [True, False])
def test_rejects_track_with_wrong_row_count(drop_last_row):
    df = _two_condition_df()
    if drop_last_row:
        # 29 rows for I01.
        drop_index = df.loc[df["track_id"] == "I01"].index[0]
        df = df.drop(index=drop_index)
    else:
        # 31 rows for I01 (duplicate of an existing row).
        extra_row = df.loc[(df["track_id"] == "I01") & (df["frame"] == 0)]
        df = pd.concat([df, extra_row], ignore_index=True)

    with pytest.raises(ValueError):
        tracks_to_sequences(df)


def test_rejects_track_with_duplicate_frame():
    df = _two_condition_df()
    # Frame 15 becomes a duplicate of frame 3; row count stays 30, but frame
    # 15 is now missing from the set.
    df.loc[(df["track_id"] == "I01") & (df["frame"] == 15), "frame"] = 3
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


def test_rejects_track_with_out_of_range_frame():
    df = _two_condition_df()
    # Frame 15 is replaced with 30; row count stays 30, but frame 15 is
    # missing and 30 is out of range.
    df.loc[(df["track_id"] == "I01") & (df["frame"] == 15), "frame"] = 30
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


def test_rejects_missing_x_or_y_value():
    df = _two_condition_df()
    df.loc[(df["track_id"] == "I01") & (df["frame"] == 0), "x_um"] = np.nan
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


@pytest.mark.parametrize("column", ["x_um", "y_um"])
@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_rejects_non_finite_coordinate(column, bad_value):
    df = _two_condition_df()
    df.loc[(df["track_id"] == "I01") & (df["frame"] == 0), column] = bad_value
    with pytest.raises(ValueError):
        tracks_to_sequences(df)


# --- split_track_ids: happy path ---


def test_split_is_deterministic_with_seed():
    track_ids = _sample_track_ids("C", 20) + _sample_track_ids("I", 20)
    conditions = ["control"] * 20 + ["injury"] * 20

    first = split_track_ids(track_ids, conditions, seed=123)
    second = split_track_ids(track_ids, conditions, seed=123)

    pdt.assert_frame_equal(first, second)


def test_split_has_no_track_id_overlap():
    track_ids = _sample_track_ids("C", 20) + _sample_track_ids("I", 20)
    conditions = ["control"] * 20 + ["injury"] * 20

    split_df = split_track_ids(track_ids, conditions, seed=123)
    train_ids = set(split_df.loc[split_df["split"] == "train", "track_id"])
    val_ids = set(split_df.loc[split_df["split"] == "val", "track_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "track_id"])

    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    assert train_ids | val_ids | test_ids == set(track_ids)


def test_split_sizes_and_stratification():
    track_ids = _sample_track_ids("C", 20) + _sample_track_ids("I", 20)
    conditions = ["control"] * 20 + ["injury"] * 20

    split_df = split_track_ids(track_ids, conditions, seed=123)
    counts = split_df.groupby(["condition", "split"]).size()

    for condition in ("control", "injury"):
        assert counts[condition, "train"] == 14
        assert counts[condition, "val"] == 3
        assert counts[condition, "test"] == 3


def test_split_sizes_match_required_counts_at_full_scale():
    track_conditions = DL_TRACKS[["track_id", "condition"]].drop_duplicates()
    split_df = split_track_ids(
        track_conditions["track_id"], track_conditions["condition"], seed=DL_SPLIT_SEED
    )
    counts = split_df.groupby(["condition", "split"]).size()

    for condition in ("control", "injury"):
        assert counts[condition, "train"] == 1400
        assert counts[condition, "val"] == 300
        assert counts[condition, "test"] == 300


# --- split_track_ids: validation failures ---


def test_split_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        split_track_ids(["C1", "C2"], ["control"], seed=1)


def test_split_rejects_duplicate_track_id():
    # Same track_id under two different conditions: drop_duplicates() on
    # (track_id, condition) pairs in generate_dl_split.py would NOT catch
    # this, since the two rows differ in their condition value.
    with pytest.raises(ValueError):
        split_track_ids(["C1", "C1"], ["control", "injury"], seed=1)


@pytest.mark.parametrize(
    "conditions",
    [
        ["control", "control"],  # only one of the two required conditions present
        ["control", "unknown"],  # an unexpected condition value present
    ],
    ids=["only-one-condition-present", "unexpected-condition-value"],
)
def test_split_rejects_unexpected_condition_value(conditions):
    with pytest.raises(ValueError):
        split_track_ids(["C1", "C2"], conditions, seed=1)


def test_split_rejects_fractions_not_summing_to_one():
    with pytest.raises(ValueError):
        split_track_ids(
            ["C1", "I1"], ["control", "injury"], seed=1, train_frac=0.5, val_frac=0.3, test_frac=0.3
        )


def test_split_rejects_non_positive_fraction():
    with pytest.raises(ValueError):
        split_track_ids(
            ["C1", "I1"], ["control", "injury"], seed=1, train_frac=0.7, val_frac=0.3, test_frac=0.0
        )


# --- main() integration tests: paths redirected to tmp_path, data/ untouched ---


def test_generate_dl_tracks_main_does_not_mutate_existing_outputs(tmp_path, monkeypatch):
    tracks_csv_hash_before = _sha256(REPO_ROOT / "data" / "tracks.csv")

    monkeypatch.setattr(generate_dl_tracks, "DL_TRACKS_PATH", tmp_path / "dl_tracks.csv")
    generate_dl_tracks.main()

    assert _sha256(REPO_ROOT / "data" / "tracks.csv") == tracks_csv_hash_before

    written = pd.read_csv(tmp_path / "dl_tracks.csv")
    expected_track_count = 2 * DL_CELLS_PER_CONDITION
    assert len(written) == expected_track_count * N_FRAMES
    assert written["track_id"].nunique() == expected_track_count
    assert list(written.columns) == ["track_id", "condition", "frame", "time_min", "x_um", "y_um"]


def test_generate_dl_split_main_writes_expected_split_csv(tmp_path, monkeypatch):
    tmp_tracks_path = tmp_path / "dl_tracks.csv"
    DL_TRACKS.to_csv(tmp_tracks_path, index=False)
    tmp_split_path = tmp_path / "dl_split.csv"

    monkeypatch.setattr(generate_dl_split, "DL_TRACKS_PATH", tmp_tracks_path)
    monkeypatch.setattr(generate_dl_split, "DL_SPLIT_PATH", tmp_split_path)

    generate_dl_split.main()

    split_df = pd.read_csv(tmp_split_path)
    expected_total = 2 * DL_CELLS_PER_CONDITION
    assert len(split_df) == expected_total
    assert list(split_df.columns) == ["track_id", "condition", "split"]
    assert split_df["track_id"].is_unique

    overall_counts = split_df["split"].value_counts()
    assert overall_counts["train"] == 2800
    assert overall_counts["val"] == 600
    assert overall_counts["test"] == 600

    per_condition_counts = split_df.groupby(["condition", "split"]).size()
    for condition in ("control", "injury"):
        assert per_condition_counts[condition, "train"] == 1400
        assert per_condition_counts[condition, "val"] == 300
        assert per_condition_counts[condition, "test"] == 300

    generate_dl_split.main()
    split_df_again = pd.read_csv(tmp_split_path)
    pdt.assert_frame_equal(split_df, split_df_again)
