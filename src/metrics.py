"""Deterministic motility and sinusoid-migration metrics.

Implements the metric definitions in PROJECT_SPEC.md ("Deterministic
Analysis Metrics"): step distance/speed, mean speed, path length,
displacement, persistence, arrest coefficient, distance to sinusoid, and
approach distance. Pure functions only — no file I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.simulate import FRAME_INTERVAL_MIN, SINUSOID_X_UM

ARREST_THRESHOLD_UM_MIN: float = 2.0


def step_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    """Euclidean distance between two consecutive positions, in µm."""
    return math.hypot(x1 - x0, y1 - y0)


def step_speed(
    distance_um: float, frame_interval_min: float = FRAME_INTERVAL_MIN
) -> float:
    """Step distance converted to speed, in µm/min."""
    return distance_um / frame_interval_min


def mean_speed(step_speeds: Sequence[float]) -> float:
    """Arithmetic mean of a track's step speeds; NaN if there are none."""
    if len(step_speeds) == 0:
        return math.nan
    return float(np.mean(step_speeds))


def path_length(step_distances: Sequence[float]) -> float:
    """Sum of consecutive step distances for a track, in µm."""
    if len(step_distances) == 0:
        return 0.0
    return float(np.sum(step_distances))


def displacement(x0: float, y0: float, xn: float, yn: float) -> float:
    """Euclidean distance between a track's first and final positions, in µm."""
    return math.hypot(xn - x0, yn - y0)


def persistence(displacement_um: float, path_length_um: float) -> float:
    """displacement / path_length; NaN if path_length is zero (undefined ratio)."""
    if path_length_um == 0:
        return math.nan
    return displacement_um / path_length_um


def arrest_coefficient(
    step_speeds: Sequence[float], threshold: float = ARREST_THRESHOLD_UM_MIN
) -> float:
    """Fraction of a track's step speeds strictly below the arrest threshold.

    NaN if there are no valid step intervals (e.g. a single-frame track).
    """
    if len(step_speeds) == 0:
        return math.nan
    arrested = sum(1 for speed in step_speeds if speed < threshold)
    return arrested / len(step_speeds)


def distance_to_sinusoid(x_um, sinusoid_x_um: float = SINUSOID_X_UM):
    """abs(x_um - sinusoid_x_um); works for scalars or array-likes (e.g. a Series)."""
    return abs(x_um - sinusoid_x_um)


def approach_distance(initial_distance_um: float, final_distance_um: float) -> float:
    """Net movement toward the sinusoid: positive = toward, negative = away."""
    return initial_distance_um - final_distance_um


def _assert_consecutive_frames(frames: np.ndarray, track_id: str) -> None:
    """Raise ValueError unless sorted frames are consecutive (diff == 1 throughout).

    step_speed relies on the fixed FRAME_INTERVAL_MIN; a gap or duplicate frame
    would silently misapply that constant 1-minute interval to a step that
    doesn't actually span 1 minute.
    """
    diffs = np.diff(frames)
    if not np.all(diffs == 1):
        raise ValueError(
            f"track {track_id!r} has non-consecutive frames after sorting "
            f"(expected consecutive diffs of 1, got {diffs.tolist()})"
        )


def _assert_single_condition(conditions: pd.Series, track_id: str) -> str:
    """Raise ValueError unless every row for this track_id shares one condition.

    Returns that condition rather than silently taking the first value, which
    could otherwise mask inconsistent condition labels for a track_id.
    """
    unique_conditions = conditions.unique()
    if len(unique_conditions) != 1:
        raise ValueError(
            f"track {track_id!r} has inconsistent condition labels: "
            f"{sorted(unique_conditions.tolist())}"
        )
    return unique_conditions[0]


def compute_frame_metrics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-position metrics: distance_to_sinusoid for every recorded frame."""
    frame_metrics = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True).copy()
    frame_metrics["distance_to_sinusoid_um"] = distance_to_sinusoid(frame_metrics["x_um"])
    return frame_metrics[
        ["track_id", "condition", "frame", "time_min", "x_um", "y_um", "distance_to_sinusoid_um"]
    ]


def compute_step_metrics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-interval metrics: step_distance/step_speed/is_arrested between consecutive frames."""
    rows: list[tuple] = []

    for track_id, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("frame").reset_index(drop=True)
        condition = _assert_single_condition(ordered["condition"], track_id)
        _assert_consecutive_frames(ordered["frame"].to_numpy(), track_id)

        for i in range(len(ordered) - 1):
            x0, y0 = ordered.loc[i, "x_um"], ordered.loc[i, "y_um"]
            x1, y1 = ordered.loc[i + 1, "x_um"], ordered.loc[i + 1, "y_um"]

            distance = step_distance(x0, y0, x1, y1)
            speed = step_speed(distance)

            rows.append(
                (
                    track_id,
                    condition,
                    int(ordered.loc[i, "frame"]),
                    int(ordered.loc[i + 1, "frame"]),
                    int(ordered.loc[i, "time_min"]),
                    int(ordered.loc[i + 1, "time_min"]),
                    distance,
                    speed,
                    bool(speed < ARREST_THRESHOLD_UM_MIN),
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "track_id",
            "condition",
            "frame_start",
            "frame_end",
            "time_min_start",
            "time_min_end",
            "step_distance_um",
            "step_speed_um_min",
            "is_arrested",
        ],
    )


def compute_track_metrics(tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-track summary metrics, one row per track_id."""
    rows: list[tuple] = []

    for track_id, group in tracks.groupby("track_id", sort=True):
        ordered = group.sort_values("frame").reset_index(drop=True)
        condition = _assert_single_condition(ordered["condition"], track_id)
        n_frames = len(ordered)
        _assert_consecutive_frames(ordered["frame"].to_numpy(), track_id)

        x = ordered["x_um"].to_numpy()
        y = ordered["y_um"].to_numpy()

        step_distances = [
            step_distance(x[i], y[i], x[i + 1], y[i + 1]) for i in range(n_frames - 1)
        ]
        step_speeds = [step_speed(distance) for distance in step_distances]

        track_path_length = path_length(step_distances)
        track_displacement = displacement(x[0], y[0], x[-1], y[-1])
        track_persistence = persistence(track_displacement, track_path_length)
        track_mean_speed = mean_speed(step_speeds)
        track_arrest_coefficient = arrest_coefficient(step_speeds)

        initial_distance = distance_to_sinusoid(x[0])
        final_distance = distance_to_sinusoid(x[-1])
        track_approach_distance = approach_distance(initial_distance, final_distance)

        rows.append(
            (
                track_id,
                condition,
                n_frames,
                track_mean_speed,
                track_path_length,
                track_displacement,
                track_persistence,
                track_arrest_coefficient,
                initial_distance,
                final_distance,
                track_approach_distance,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
