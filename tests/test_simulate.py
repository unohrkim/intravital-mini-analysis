"""Deterministic tests for the synthetic movement model in src/simulate.py."""

from __future__ import annotations

import numpy as np
import pandas.testing as pdt
import pytest

from src.simulate import (
    CELLS_PER_CONDITION,
    FRAMES_PER_CELL,
    INJURY_BIAS_UM,
    directional_bias,
    generate_tracks,
    reflect_coordinate,
    sample_initial_position,
    simulate_track,
)


def test_generate_tracks_is_reproducible_with_seed_42():
    first = generate_tracks(seed=42)
    second = generate_tracks(seed=42)
    pdt.assert_frame_equal(first, second)


def test_expected_number_of_tracks_and_rows():
    tracks = generate_tracks()

    expected_track_count = 2 * CELLS_PER_CONDITION  # 2 conditions x 15 cells
    expected_row_count = expected_track_count * FRAMES_PER_CELL  # 30 tracks x 30 frames

    assert tracks["track_id"].nunique() == expected_track_count
    assert len(tracks) == expected_row_count
    assert (tracks.groupby("track_id").size() == FRAMES_PER_CELL).all()


def test_generate_tracks_returns_expected_columns_in_order():
    tracks = generate_tracks()

    assert list(tracks.columns) == [
        "track_id",
        "condition",
        "frame",
        "time_min",
        "x_um",
        "y_um",
    ]


def test_track_ids_follow_confirmed_convention():
    tracks = generate_tracks()

    control_ids = sorted(tracks.loc[tracks["condition"] == "control", "track_id"].unique())
    injury_ids = sorted(tracks.loc[tracks["condition"] == "injury", "track_id"].unique())

    assert control_ids == [f"C{i:02d}" for i in range(1, CELLS_PER_CONDITION + 1)]
    assert injury_ids == [f"I{i:02d}" for i in range(1, CELLS_PER_CONDITION + 1)]


def test_all_coordinates_within_field_boundaries():
    tracks = generate_tracks()

    assert tracks["x_um"].between(0, 200).all()
    assert tracks["y_um"].between(0, 200).all()


def test_sample_initial_position_is_reproducible_from_same_rng_state():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    assert sample_initial_position(rng_a) == sample_initial_position(rng_b)


def test_control_step_has_no_directional_bias():
    rng_reference = np.random.default_rng(0)
    x0, y0 = sample_initial_position(rng_reference)
    dx_random = rng_reference.normal(0.0, 3.0)
    dy_random = rng_reference.normal(0.0, 3.0)

    expected_x = reflect_coordinate(x0 + dx_random, 0.0, 200.0)
    expected_y = reflect_coordinate(y0 + dy_random, 0.0, 200.0)

    positions = simulate_track(np.random.default_rng(0), "control", n_frames=2)

    assert positions[1][1] == pytest.approx(expected_x)
    assert positions[1][2] == pytest.approx(expected_y)


def test_injury_step_applies_directional_bias_to_first_step():
    rng_reference = np.random.default_rng(0)
    x0, y0 = sample_initial_position(rng_reference)
    dx_random = rng_reference.normal(0.0, 3.0)
    dy_random = rng_reference.normal(0.0, 3.0)

    expected_x = reflect_coordinate(x0 + dx_random + directional_bias(x0), 0.0, 200.0)
    expected_x_without_bias = reflect_coordinate(x0 + dx_random, 0.0, 200.0)
    expected_y = reflect_coordinate(y0 + dy_random, 0.0, 200.0)

    positions = simulate_track(np.random.default_rng(0), "injury", n_frames=2)

    assert positions[1][1] == pytest.approx(expected_x)
    assert positions[1][2] == pytest.approx(expected_y)
    assert positions[1][1] != pytest.approx(expected_x_without_bias)


def test_injury_bias_direction():
    assert directional_bias(50.0) == INJURY_BIAS_UM
    assert directional_bias(150.0) == -INJURY_BIAS_UM


def test_injury_bias_is_zero_at_sinusoid():
    assert directional_bias(100.0) == 0.0


def test_simulate_track_rejects_unsupported_condition():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_track(rng, "not-a-condition")


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (-5.0, 0.0, 200.0, 5.0),
        (205.0, 0.0, 200.0, 195.0),
        (100.0, 0.0, 200.0, 100.0),
        (0.0, 0.0, 200.0, 0.0),
        (200.0, 0.0, 200.0, 200.0),
    ],
)
def test_reflect_coordinate(value, low, high, expected):
    assert reflect_coordinate(value, low, high) == pytest.approx(expected)
