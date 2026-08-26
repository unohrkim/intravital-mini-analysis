"""Deterministic tests for the synthetic movie rendering in src/render_movie.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import src.generate_movie as generate_movie
from src.render_movie import (
    BACKGROUND_INTENSITY,
    CELL_INTENSITY,
    CELL_MARKER_RADIUS_PX,
    IMAGE_HEIGHT_PX,
    IMAGE_WIDTH_PX,
    SINUSOID_INTENSITY,
    render_movie,
    um_to_pixel,
)
from src.simulate import FRAMES_PER_CELL

REPO_ROOT = Path(__file__).resolve().parent.parent


def _one_track_frame(x_um: float, y_um: float, frame: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": "C01",
                "condition": "control",
                "frame": frame,
                "time_min": frame,
                "x_um": x_um,
                "y_um": y_um,
            }
        ]
    )


def test_render_movie_frame_count():
    tracks = _one_track_frame(50.0, 50.0)
    movie = render_movie(tracks, n_frames=1)
    assert movie.shape[0] == 1

    full_tracks = pd.concat(
        [_one_track_frame(50.0, 50.0, frame=f) for f in range(FRAMES_PER_CELL)],
        ignore_index=True,
    )
    full_movie = render_movie(full_tracks)
    assert full_movie.shape[0] == FRAMES_PER_CELL


def test_render_movie_frame_dimensions():
    tracks = _one_track_frame(50.0, 50.0)
    movie = render_movie(tracks, n_frames=1)
    assert movie.shape[1:] == (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX)
    assert (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX) == (200, 200)


def test_render_movie_dtype_is_uint8():
    tracks = _one_track_frame(50.0, 50.0)
    movie = render_movie(tracks, n_frames=1)
    assert movie.dtype == np.uint8


def test_render_movie_is_deterministic():
    tracks = pd.concat(
        [_one_track_frame(50.0, 50.0, frame=f) for f in range(5)], ignore_index=True
    )
    first = render_movie(tracks, n_frames=5)
    second = render_movie(tracks, n_frames=5)
    assert np.array_equal(first, second)


def test_coordinate_to_pixel_consistency():
    x_um, y_um = 50.0, 150.0
    tracks = _one_track_frame(x_um, y_um)
    movie = render_movie(tracks, n_frames=1)
    frame = movie[0]

    x_px = um_to_pixel(x_um, IMAGE_WIDTH_PX)
    y_px = um_to_pixel(y_um, IMAGE_HEIGHT_PX)

    # x_um -> column, y_um -> row: frame[row, col], not swapped.
    assert frame[y_px, x_px] == CELL_INTENSITY

    # Full 5x5 neighborhood: every pixel inside the radius-2 disk is CELL_INTENSITY,
    # every pixel outside it is BACKGROUND_INTENSITY.
    for dr in range(-CELL_MARKER_RADIUS_PX, CELL_MARKER_RADIUS_PX + 1):
        for dc in range(-CELL_MARKER_RADIUS_PX, CELL_MARKER_RADIUS_PX + 1):
            row, col = y_px + dr, x_px + dc
            expected_in_disk = dr**2 + dc**2 <= CELL_MARKER_RADIUS_PX**2
            if expected_in_disk:
                assert frame[row, col] == CELL_INTENSITY
            else:
                assert frame[row, col] == BACKGROUND_INTENSITY


def test_sinusoid_placement_and_no_vertical_flip():
    # No cell overlapping column 100, so the full sinusoid column is intact.
    tracks = _one_track_frame(20.0, 50.0)
    movie = render_movie(tracks, n_frames=1)
    frame = movie[0]

    x_px_sinusoid = um_to_pixel(100.0, IMAGE_WIDTH_PX)
    assert x_px_sinusoid == 100
    assert np.all(frame[:, x_px_sinusoid] == SINUSOID_INTENSITY)

    other_column = frame[:, x_px_sinusoid - 10]
    assert not np.all(other_column == SINUSOID_INTENSITY)

    # y_um = 0 renders into row 0 (top row, no vertical flip).
    top_row_tracks = _one_track_frame(20.0, 0.0)
    top_row_movie = render_movie(top_row_tracks, n_frames=1)
    assert top_row_movie[0][0, um_to_pixel(20.0, IMAGE_WIDTH_PX)] == CELL_INTENSITY


def test_overlap_priority_background_then_sinusoid_then_cells():
    # Cell centered exactly on the sinusoid column.
    tracks = _one_track_frame(100.0, 50.0)
    movie = render_movie(tracks, n_frames=1)
    frame = movie[0]

    x_px, y_px = um_to_pixel(100.0, IMAGE_WIDTH_PX), um_to_pixel(50.0, IMAGE_HEIGHT_PX)
    assert x_px == 100

    # Inside the cell's disk footprint: cell wins over sinusoid.
    for dr in range(-CELL_MARKER_RADIUS_PX, CELL_MARKER_RADIUS_PX + 1):
        for dc in range(-CELL_MARKER_RADIUS_PX, CELL_MARKER_RADIUS_PX + 1):
            if dr**2 + dc**2 <= CELL_MARKER_RADIUS_PX**2:
                assert frame[y_px + dr, x_px + dc] == CELL_INTENSITY

    # Outside the disk footprint but still on the sinusoid column: sinusoid remains.
    far_row = y_px + CELL_MARKER_RADIUS_PX + 20
    assert far_row < IMAGE_HEIGHT_PX
    assert frame[far_row, x_px] == SINUSOID_INTENSITY


@pytest.mark.parametrize(
    ("coord_um", "size_px", "expected_px"),
    [
        (0.49, 200, 0),
        (0.50, 200, 1),
        (1.50, 200, 2),
        (200.0, 200, 199),
        (0.0, 200, 0),
        (-5.0, 200, 0),
    ],
)
def test_um_to_pixel_half_up_rounding_and_clipping(coord_um, size_px, expected_px):
    assert um_to_pixel(coord_um, size_px) == expected_px


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_movie_does_not_mutate_existing_outputs(tmp_path, monkeypatch):
    tracks_hash_before = _sha256(generate_movie.TRACKS_PATH)
    results_dir = REPO_ROOT / "results"
    results_hashes_before = {p: _sha256(p) for p in sorted(results_dir.glob("*.csv"))}

    # Redirect movie output to a temp location so this test doesn't touch data/.
    monkeypatch.setattr(
        generate_movie, "movie_path", lambda condition: tmp_path / f"synthetic_movie_{condition}.tif"
    )
    generate_movie.main()

    assert _sha256(generate_movie.TRACKS_PATH) == tracks_hash_before
    for path, hash_before in results_hashes_before.items():
        assert _sha256(path) == hash_before

    expected_filenames = {f"synthetic_movie_{condition}.tif" for condition in generate_movie.CONDITIONS}
    actual_filenames = {p.name for p in tmp_path.iterdir()}
    assert actual_filenames == expected_filenames

    for condition in generate_movie.CONDITIONS:
        movie = tifffile.imread(tmp_path / f"synthetic_movie_{condition}.tif")
        assert movie.shape == (FRAMES_PER_CELL, IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX)
        assert movie.dtype == np.uint8
