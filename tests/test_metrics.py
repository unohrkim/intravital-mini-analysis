"""Deterministic tests for the metric functions in src/metrics.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.compute_metrics import TRACKS_PATH
from src.metrics import (
    ARREST_THRESHOLD_UM_MIN,
    approach_distance,
    arrest_coefficient,
    compute_frame_metrics,
    compute_step_metrics,
    compute_track_metrics,
    displacement,
    distance_to_sinusoid,
    mean_speed,
    path_length,
    persistence,
    step_distance,
    step_speed,
)


# ---------------------------------------------------------------------------
# Scalar metric functions
# ---------------------------------------------------------------------------


def test_step_distance_three_four_five_triangle():
    assert step_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


def test_step_distance_zero_displacement():
    assert step_distance(1.0, 1.0, 1.0, 1.0) == pytest.approx(0.0)


def test_step_speed_uses_frame_interval():
    assert step_speed(5.0, frame_interval_min=1) == pytest.approx(5.0)


def test_step_speed_zero_distance():
    assert step_speed(0.0) == pytest.approx(0.0)


def test_mean_speed_known_values():
    assert mean_speed([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_speed_single_value():
    assert mean_speed([5.0]) == pytest.approx(5.0)


def test_mean_speed_empty_is_nan():
    assert math.isnan(mean_speed([]))


def test_path_length_known_values():
    assert path_length([1.0, 2.0, 3.0]) == pytest.approx(6.0)


def test_path_length_empty_is_zero():
    assert path_length([]) == pytest.approx(0.0)


def test_displacement_known_values():
    assert displacement(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


def test_displacement_identical_start_and_end():
    assert displacement(2.0, 2.0, 2.0, 2.0) == pytest.approx(0.0)


def test_persistence_known_ratio():
    assert persistence(5.0, 10.0) == pytest.approx(0.5)


def test_persistence_zero_path_length_is_nan():
    assert math.isnan(persistence(0.0, 0.0))


def test_persistence_zero_displacement_nonzero_path_length():
    assert persistence(0.0, 10.0) == pytest.approx(0.0)


def test_arrest_coefficient_mixed_values():
    speeds = [1.0, 2.0, 3.0, 0.5]
    assert arrest_coefficient(speeds) == pytest.approx(0.5)


def test_arrest_coefficient_boundary_speed_is_not_arrested():
    assert arrest_coefficient([ARREST_THRESHOLD_UM_MIN]) == pytest.approx(0.0)


def test_arrest_coefficient_all_arrested():
    assert arrest_coefficient([0.1, 0.5, 1.9]) == pytest.approx(1.0)


def test_arrest_coefficient_none_arrested():
    assert arrest_coefficient([2.0, 3.0, 10.0]) == pytest.approx(0.0)


def test_arrest_coefficient_empty_is_nan():
    assert math.isnan(arrest_coefficient([]))


@pytest.mark.parametrize(
    ("x_um", "expected"),
    [
        (100.0, 0.0),
        (50.0, 50.0),
        (150.0, 50.0),
    ],
)
def test_distance_to_sinusoid(x_um, expected):
    assert distance_to_sinusoid(x_um) == pytest.approx(expected)


def test_approach_distance_toward_sinusoid_is_positive():
    assert approach_distance(50.0, 10.0) == pytest.approx(40.0)


def test_approach_distance_away_from_sinusoid_is_negative():
    assert approach_distance(10.0, 50.0) == pytest.approx(-40.0)


def test_approach_distance_no_change_is_zero():
    assert approach_distance(30.0, 30.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Orchestration functions
# ---------------------------------------------------------------------------


def _synthetic_tracks() -> pd.DataFrame:
    """Two tracks, deliberately out of frame order; T2 is a single-frame track."""
    rows = [
        ("T1", "control", 2, 2, 6.0, 8.0),
        ("T2", "injury", 0, 0, 50.0, 50.0),
        ("T1", "control", 0, 0, 0.0, 0.0),
        ("T1", "control", 1, 1, 3.0, 4.0),
    ]
    return pd.DataFrame(
        rows, columns=["track_id", "condition", "frame", "time_min", "x_um", "y_um"]
    )


def test_compute_frame_metrics_sorts_and_computes_distance():
    frame_metrics = compute_frame_metrics(_synthetic_tracks())

    assert list(frame_metrics.columns) == [
        "track_id",
        "condition",
        "frame",
        "time_min",
        "x_um",
        "y_um",
        "distance_to_sinusoid_um",
    ]
    assert len(frame_metrics) == 4
    assert list(zip(frame_metrics["track_id"], frame_metrics["frame"])) == [
        ("T1", 0),
        ("T1", 1),
        ("T1", 2),
        ("T2", 0),
    ]
    assert frame_metrics["distance_to_sinusoid_um"].tolist() == pytest.approx(
        [100.0, 97.0, 94.0, 50.0]
    )


def test_compute_step_metrics_sorts_and_computes_steps():
    step_metrics = compute_step_metrics(_synthetic_tracks())

    assert list(step_metrics.columns) == [
        "track_id",
        "condition",
        "frame_start",
        "frame_end",
        "time_min_start",
        "time_min_end",
        "step_distance_um",
        "step_speed_um_min",
        "is_arrested",
    ]

    # T1 has 2 steps (3 frames); T2 has 0 steps (1 frame).
    assert len(step_metrics) == 2
    assert step_metrics["step_distance_um"].tolist() == pytest.approx([5.0, 5.0])
    assert step_metrics["step_speed_um_min"].tolist() == pytest.approx([5.0, 5.0])
    assert step_metrics["is_arrested"].tolist() == [False, False]


def test_compute_track_metrics_multi_frame_track():
    track_metrics = compute_track_metrics(_synthetic_tracks())

    assert list(track_metrics.columns) == [
        "track_id",
        "condition",
        "n_frames",
        "mean_speed_um_min",
        "path_length_um",
        "displacement_um",
        "persistence",
        "arrest_coefficient",
        "initial_distance_to_sinusoid_um",
        "final_distance_to_sinusoid_um",
        "approach_distance_um",
    ]
    assert len(track_metrics) == 2

    t1 = track_metrics.loc[track_metrics["track_id"] == "T1"].iloc[0]
    assert t1["condition"] == "control"
    assert t1["n_frames"] == 3
    assert t1["mean_speed_um_min"] == pytest.approx(5.0)
    assert t1["path_length_um"] == pytest.approx(10.0)
    assert t1["displacement_um"] == pytest.approx(10.0)
    assert t1["persistence"] == pytest.approx(1.0)
    assert t1["arrest_coefficient"] == pytest.approx(0.0)
    assert t1["initial_distance_to_sinusoid_um"] == pytest.approx(100.0)
    assert t1["final_distance_to_sinusoid_um"] == pytest.approx(94.0)
    assert t1["approach_distance_um"] == pytest.approx(6.0)


def test_compute_track_metrics_single_frame_track_is_nan_where_undefined():
    track_metrics = compute_track_metrics(_synthetic_tracks())

    t2 = track_metrics.loc[track_metrics["track_id"] == "T2"].iloc[0]
    assert t2["condition"] == "injury"
    assert t2["n_frames"] == 1
    assert math.isnan(t2["mean_speed_um_min"])
    assert t2["path_length_um"] == pytest.approx(0.0)
    assert t2["displacement_um"] == pytest.approx(0.0)
    assert math.isnan(t2["persistence"])
    assert math.isnan(t2["arrest_coefficient"])
    assert t2["initial_distance_to_sinusoid_um"] == pytest.approx(50.0)
    assert t2["final_distance_to_sinusoid_um"] == pytest.approx(50.0)
    assert t2["approach_distance_um"] == pytest.approx(0.0)


def _tracks_with_frame_gap() -> pd.DataFrame:
    rows = [
        ("T1", "control", 0, 0, 0.0, 0.0),
        ("T1", "control", 2, 2, 6.0, 8.0),  # frame 1 missing -> diff of 2
    ]
    return pd.DataFrame(
        rows, columns=["track_id", "condition", "frame", "time_min", "x_um", "y_um"]
    )


def _tracks_with_duplicate_frame() -> pd.DataFrame:
    rows = [
        ("T1", "control", 0, 0, 0.0, 0.0),
        ("T1", "control", 0, 0, 1.0, 1.0),  # duplicate frame -> diff of 0
    ]
    return pd.DataFrame(
        rows, columns=["track_id", "condition", "frame", "time_min", "x_um", "y_um"]
    )


def test_compute_step_metrics_rejects_frame_gap():
    with pytest.raises(ValueError):
        compute_step_metrics(_tracks_with_frame_gap())


def test_compute_track_metrics_rejects_frame_gap():
    with pytest.raises(ValueError):
        compute_track_metrics(_tracks_with_frame_gap())


def test_compute_step_metrics_rejects_duplicate_frame():
    with pytest.raises(ValueError):
        compute_step_metrics(_tracks_with_duplicate_frame())


def test_compute_track_metrics_rejects_duplicate_frame():
    with pytest.raises(ValueError):
        compute_track_metrics(_tracks_with_duplicate_frame())


def _tracks_with_mixed_condition() -> pd.DataFrame:
    rows = [
        ("T1", "control", 0, 0, 0.0, 0.0),
        ("T1", "injury", 1, 1, 3.0, 4.0),  # same track_id, inconsistent condition
    ]
    return pd.DataFrame(
        rows, columns=["track_id", "condition", "frame", "time_min", "x_um", "y_um"]
    )


def test_compute_step_metrics_rejects_mixed_condition_track():
    with pytest.raises(ValueError):
        compute_step_metrics(_tracks_with_mixed_condition())


def test_compute_track_metrics_rejects_mixed_condition_track():
    with pytest.raises(ValueError):
        compute_track_metrics(_tracks_with_mixed_condition())


# ---------------------------------------------------------------------------
# Sanity check against the real, deterministic data/tracks.csv
# ---------------------------------------------------------------------------


def test_real_tracks_csv_produces_expected_shapes_and_no_nans():
    tracks = pd.read_csv(TRACKS_PATH)

    frame_metrics = compute_frame_metrics(tracks)
    step_metrics = compute_step_metrics(tracks)
    track_metrics = compute_track_metrics(tracks)

    assert len(frame_metrics) == 900
    assert len(step_metrics) == 870
    assert len(track_metrics) == 30

    assert not frame_metrics.isna().any().any()
    assert not step_metrics.isna().any().any()
    assert not track_metrics.isna().any().any()
