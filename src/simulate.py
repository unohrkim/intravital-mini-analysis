"""Synthetic cell-track generation.

Implements the movement model defined in PROJECT_SPEC.md: independent
uniform initial positions, isotropic random-walk displacement, a constant
directional drift toward the simulated sinusoid for injury-condition cells,
and reflective field boundaries. Pure functions only — no file I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONDITIONS: tuple[str, str] = ("control", "injury")
CELLS_PER_CONDITION: int = 15
FRAMES_PER_CELL: int = 30
FRAME_INTERVAL_MIN: int = 1
FIELD_WIDTH_UM: float = 200.0
FIELD_HEIGHT_UM: float = 200.0
SINUSOID_X_UM: float = 100.0
RANDOM_SEED: int = 42
RANDOM_STEP_STD_UM: float = 3.0
INJURY_BIAS_UM: float = 0.8

_CONDITION_PREFIX: dict[str, str] = {"control": "C", "injury": "I"}


def reflect_coordinate(value: float, low: float, high: float) -> float:
    """Fold a coordinate back into [low, high] via triangle-wave reflection."""
    span = high - low
    if span <= 0:
        raise ValueError("high must be greater than low")

    offset = (value - low) % (2 * span)
    if offset > span:
        offset = 2 * span - offset
    return low + offset


def directional_bias(x_um: float, bias_magnitude: float = INJURY_BIAS_UM) -> float:
    """Constant drift toward the sinusoid at x = 100 um; zero exactly there."""
    if x_um < SINUSOID_X_UM:
        return bias_magnitude
    if x_um > SINUSOID_X_UM:
        return -bias_magnitude
    return 0.0


def sample_initial_position(rng: np.random.Generator) -> tuple[float, float]:
    """Draw (x, y) ~ Uniform(0, field size), independent of condition."""
    x = rng.uniform(0.0, FIELD_WIDTH_UM)
    y = rng.uniform(0.0, FIELD_HEIGHT_UM)
    return x, y


def simulate_track(
    rng: np.random.Generator,
    condition: str,
    n_frames: int = FRAMES_PER_CELL,
) -> list[tuple[int, float, float]]:
    """Simulate one cell's trajectory, returning [(frame, x_um, y_um), ...]."""
    if condition not in ("control", "injury"):
        raise ValueError(f"Unsupported condition: {condition!r}")

    x, y = sample_initial_position(rng)
    positions: list[tuple[int, float, float]] = [(0, x, y)]

    for frame in range(1, n_frames):
        dx = rng.normal(0.0, RANDOM_STEP_STD_UM)
        dy = rng.normal(0.0, RANDOM_STEP_STD_UM)
        if condition == "injury":
            dx += directional_bias(x)

        x = reflect_coordinate(x + dx, 0.0, FIELD_WIDTH_UM)
        y = reflect_coordinate(y + dy, 0.0, FIELD_HEIGHT_UM)
        positions.append((frame, x, y))

    return positions


def generate_tracks(
    conditions: tuple[str, ...] = CONDITIONS,
    cells_per_condition: int = CELLS_PER_CONDITION,
    n_frames: int = FRAMES_PER_CELL,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate the full tidy tracking table for all conditions and cells."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str, int, int, float, float]] = []

    for condition in conditions:
        prefix = _CONDITION_PREFIX[condition]
        for cell_index in range(cells_per_condition):
            track_id = f"{prefix}{cell_index + 1:02d}"
            for frame, x, y in simulate_track(rng, condition, n_frames):
                time_min = frame * FRAME_INTERVAL_MIN
                rows.append((track_id, condition, frame, time_min, x, y))

    return pd.DataFrame(
        rows,
        columns=["track_id", "condition", "frame", "time_min", "x_um", "y_um"],
    )
