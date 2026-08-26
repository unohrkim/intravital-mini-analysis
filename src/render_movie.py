"""Deterministic rendering of synthetic tracking data into a TIFF movie.

Renders the coordinate tracks in data/tracks.csv into a simple synthetic
microscopy-style image stack. This is a visualization/reproducibility
artifact, not a new scientific model — no PSF, background noise, or image
noise model is applied. Pure functions only — no file I/O.

Coordinate convention:
    x_um maps to the pixel COLUMN index; y_um maps to the pixel ROW index.
    Array axis order is [frame, y, x], i.e. movie[frame, row, col] where
    `row` comes from y_um and `col` comes from x_um.
    y = 0 um corresponds to the TOP row (row index 0); no vertical flip
    is applied.

Draw order per frame is background -> sinusoid -> cells, so a cell marker
that overlaps the sinusoid column paints over it (final pixel is
CELL_INTENSITY, not SINUSOID_INTENSITY).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.simulate import FIELD_HEIGHT_UM, FIELD_WIDTH_UM, FRAMES_PER_CELL, SINUSOID_X_UM

PIXEL_SIZE_UM: float = 1.0
IMAGE_WIDTH_PX: int = int(FIELD_WIDTH_UM / PIXEL_SIZE_UM)
IMAGE_HEIGHT_PX: int = int(FIELD_HEIGHT_UM / PIXEL_SIZE_UM)

BACKGROUND_INTENSITY: int = 0
SINUSOID_INTENSITY: int = 80
CELL_INTENSITY: int = 255
CELL_MARKER_RADIUS_PX: int = 2

MOVIE_DTYPE = np.uint8


def um_to_pixel(coord_um: float, size_px: int, pixel_size_um: float = PIXEL_SIZE_UM) -> int:
    """Map a micron coordinate to the nearest pixel index, clipped to [0, size_px - 1].

    Uses an explicit half-up rounding rule (floor(x + 0.5)) rather than
    Python's built-in round(), which rounds halves to even: a coordinate
    exactly halfway between two pixels is mapped to the HIGHER pixel index.

    Clipping handles the reflected-boundary case where a coordinate lands
    exactly on the field edge (e.g. x_um = 200.0), which would otherwise
    round to an out-of-range index equal to size_px.
    """
    pixel = int(np.floor(coord_um / pixel_size_um + 0.5))
    return int(np.clip(pixel, 0, size_px - 1))


def draw_sinusoid(frame: np.ndarray) -> None:
    """Draw the sinusoid as a single vertical column at x = SINUSOID_X_UM, in place."""
    x_px = um_to_pixel(SINUSOID_X_UM, frame.shape[1])
    frame[:, x_px] = SINUSOID_INTENSITY


def draw_cell_marker(
    frame: np.ndarray,
    x_px: int,
    y_px: int,
    radius_px: int = CELL_MARKER_RADIUS_PX,
    intensity: int = CELL_INTENSITY,
) -> None:
    """Fill a hard-edged disk of the given radius centered at (x_px, y_px), in place.

    x_px indexes the column axis, y_px indexes the row axis. The disk is
    clipped to the frame's bounds.
    """
    height_px, width_px = frame.shape
    row_lo, row_hi = max(0, y_px - radius_px), min(height_px, y_px + radius_px + 1)
    col_lo, col_hi = max(0, x_px - radius_px), min(width_px, x_px + radius_px + 1)

    rows = np.arange(row_lo, row_hi)[:, None]
    cols = np.arange(col_lo, col_hi)[None, :]
    disk_mask = (rows - y_px) ** 2 + (cols - x_px) ** 2 <= radius_px**2

    region = frame[row_lo:row_hi, col_lo:col_hi]
    region[disk_mask] = intensity


def render_frame(
    positions: Sequence[tuple[float, float]],
    width_px: int = IMAGE_WIDTH_PX,
    height_px: int = IMAGE_HEIGHT_PX,
) -> np.ndarray:
    """Render one frame: background, then sinusoid, then cell markers in the given order.

    `positions` is a sequence of (x_um, y_um) pairs; callers must pre-sort
    cells (e.g. by track_id) for deterministic overlap behavior.
    """
    frame = np.full((height_px, width_px), BACKGROUND_INTENSITY, dtype=MOVIE_DTYPE)
    draw_sinusoid(frame)

    for x_um, y_um in positions:
        x_px = um_to_pixel(x_um, width_px)
        y_px = um_to_pixel(y_um, height_px)
        draw_cell_marker(frame, x_px, y_px)

    return frame


def render_movie(
    tracks: pd.DataFrame,
    n_frames: int = FRAMES_PER_CELL,
    width_px: int = IMAGE_WIDTH_PX,
    height_px: int = IMAGE_HEIGHT_PX,
) -> np.ndarray:
    """Render a full image stack from a tracks table, shape (n_frames, height_px, width_px).

    Operates on whatever subset of `tracks` is passed in — condition
    filtering is the caller's responsibility, so the same function renders
    either a single condition or a combined set.
    """
    movie = np.empty((n_frames, height_px, width_px), dtype=MOVIE_DTYPE)

    for frame_index in range(n_frames):
        frame_rows = tracks.loc[tracks["frame"] == frame_index].sort_values("track_id")
        positions = list(zip(frame_rows["x_um"], frame_rows["y_um"]))
        movie[frame_index] = render_frame(positions, width_px, height_px)

    return movie
