"""Deterministic tests for the Step 4B figures in src/visualize.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import src.generate_figures as generate_figures
from src.render_movie import CELL_INTENSITY, IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, um_to_pixel
from src.simulate import CONDITIONS, FRAMES_PER_CELL
from src.visualize import (
    OVERLAY_FRAME_INDEX,
    plot_approach_distance_distribution,
    plot_primary_effect_ci,
    plot_tiff_overlay,
    plot_trajectories,
    select_representative_track_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _canvas_array(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def _make_single_track_all_frames(
    track_id: str, condition: str, x_um: float, y_um: float
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "track_id": track_id,
                "condition": condition,
                "frame": frame,
                "time_min": frame,
                "x_um": x_um,
                "y_um": y_um,
            }
            for frame in range(FRAMES_PER_CELL)
        ]
    )


def _make_condition_comparison_row(**overrides) -> pd.DataFrame:
    base = {
        "metric": "approach_distance_um",
        "is_primary": True,
        "mean_difference": 24.7,
        "ci_lower": 14.6,
        "ci_upper": 34.5,
    }
    base.update(overrides)
    return pd.DataFrame([base])


# --- Figure 1: trajectories ------------------------------------------------


def test_select_representative_track_ids_first_5():
    tracks = pd.read_csv(REPO_ROOT / "data" / "tracks.csv")
    selected = select_representative_track_ids(tracks, n_per_condition=5)
    assert selected["control"] == [f"C{i:02d}" for i in range(1, 6)]
    assert selected["injury"] == [f"I{i:02d}" for i in range(1, 6)]


def test_plot_trajectories_is_deterministic():
    tracks = pd.read_csv(REPO_ROOT / "data" / "tracks.csv")
    first = _canvas_array(plot_trajectories(tracks))
    second = _canvas_array(plot_trajectories(tracks))
    assert np.array_equal(first, second)


def test_plot_trajectories_line_data_matches_source():
    tracks = pd.read_csv(REPO_ROOT / "data" / "tracks.csv")
    fig = plot_trajectories(tracks, n_per_condition=5)
    ax = fig.axes[0]

    c01 = tracks.loc[tracks["track_id"] == "C01"].sort_values("frame")
    matched = [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == len(c01)
        and np.allclose(line.get_xdata(), c01["x_um"].to_numpy())
        and np.allclose(line.get_ydata(), c01["y_um"].to_numpy())
    ]
    assert len(matched) == 1


# --- Figure 2: approach_distance_um distribution ----------------------------


def test_plot_approach_distance_distribution_matches_source_with_multiplicity():
    track_metrics = pd.read_csv(REPO_ROOT / "results" / "track_metrics.csv")
    fig = plot_approach_distance_distribution(track_metrics)
    ax = fig.axes[0]

    scatter_collections = [c for c in ax.collections if len(c.get_offsets()) > 0]
    assert len(scatter_collections) == len(CONDITIONS)

    for collection, condition in zip(scatter_collections, CONDITIONS):
        plotted_y = collection.get_offsets()[:, 1]
        expected_y = track_metrics.loc[
            track_metrics["condition"] == condition, "approach_distance_um"
        ].to_numpy(dtype=float)
        np.testing.assert_allclose(np.sort(plotted_y), np.sort(expected_y))


def test_plot_approach_distance_distribution_is_deterministic():
    track_metrics = pd.read_csv(REPO_ROOT / "results" / "track_metrics.csv")
    first = _canvas_array(plot_approach_distance_distribution(track_metrics))
    second = _canvas_array(plot_approach_distance_distribution(track_metrics))
    assert np.array_equal(first, second)


# --- Figure 3: primary effect CI --------------------------------------------


def test_plot_primary_effect_ci_uses_input_values():
    df = _make_condition_comparison_row()
    fig = plot_primary_effect_ci(df)
    ax = fig.axes[0]

    (container,) = ax.containers
    point_x = container.lines[0].get_xdata()[0]
    assert point_x == pytest.approx(24.7)

    (barlinecol,) = container.lines[2]
    (segment,) = barlinecol.get_segments()
    x_values = segment[:, 0]
    assert min(x_values) == pytest.approx(14.6)
    assert max(x_values) == pytest.approx(34.5)


def test_plot_primary_effect_ci_requires_exactly_one_primary_row():
    zero_primary = _make_condition_comparison_row(is_primary=False)
    with pytest.raises(ValueError):
        plot_primary_effect_ci(zero_primary)

    two_primary = pd.concat(
        [_make_condition_comparison_row(), _make_condition_comparison_row(metric="other")],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        plot_primary_effect_ci(two_primary)


def test_plot_primary_effect_ci_is_deterministic():
    df = _make_condition_comparison_row()
    first = _canvas_array(plot_primary_effect_ci(df))
    second = _canvas_array(plot_primary_effect_ci(df))
    assert np.array_equal(first, second)


# --- Figure 4: TIFF overlay --------------------------------------------------


def test_plot_tiff_overlay_has_two_subplots_with_expected_titles():
    dummy_frame = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), dtype=np.uint8)
    tracks = _make_single_track_all_frames("C01", "control", 50.0, 50.0)

    fig = plot_tiff_overlay(dummy_frame, dummy_frame, tracks, frame_index=OVERLAY_FRAME_INDEX)

    assert len(fig.axes) == 2
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        f"Control — frame {OVERLAY_FRAME_INDEX}",
        f"Injury — frame {OVERLAY_FRAME_INDEX}",
    ]


def test_plot_tiff_overlay_keeps_y_zero_at_top():
    # y=0 (top row) convention: y-axis limits must be reversed, i.e. the first
    # limit (bottom of the displayed axes range) is the larger value.
    dummy_frame = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), dtype=np.uint8)
    tracks = _make_single_track_all_frames("C01", "control", 50.0, 50.0)

    fig = plot_tiff_overlay(dummy_frame, dummy_frame, tracks, frame_index=OVERLAY_FRAME_INDEX)

    for ax in fig.axes:
        ylim_low, ylim_high = ax.get_ylim()
        assert ylim_low == pytest.approx(IMAGE_HEIGHT_PX - 0.5)
        assert ylim_high == pytest.approx(-0.5)
        assert ylim_low > ylim_high


def test_plot_tiff_overlay_is_deterministic():
    dummy_frame = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), dtype=np.uint8)
    tracks = _make_single_track_all_frames("C01", "control", 50.0, 50.0)

    first = _canvas_array(plot_tiff_overlay(dummy_frame, dummy_frame, tracks))
    second = _canvas_array(plot_tiff_overlay(dummy_frame, dummy_frame, tracks))
    assert np.array_equal(first, second)


def test_saved_tiff_matches_tracks_at_overlay_frame():
    tracks = pd.read_csv(REPO_ROOT / "data" / "tracks.csv")
    movies = {
        "control": tifffile.imread(REPO_ROOT / "data" / "synthetic_movie_control.tif"),
        "injury": tifffile.imread(REPO_ROOT / "data" / "synthetic_movie_injury.tif"),
    }

    for condition, movie in movies.items():
        frame = movie[OVERLAY_FRAME_INDEX]
        condition_tracks = tracks.loc[
            (tracks["condition"] == condition) & (tracks["frame"] == OVERLAY_FRAME_INDEX)
        ]
        assert len(condition_tracks) > 0

        for _, row in condition_tracks.iterrows():
            x_px = um_to_pixel(row["x_um"], IMAGE_WIDTH_PX)
            y_px = um_to_pixel(row["y_um"], IMAGE_HEIGHT_PX)
            assert frame[y_px, x_px] == CELL_INTENSITY


# --- generate_figures integration -------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_figures_does_not_mutate_existing_outputs(tmp_path, monkeypatch):
    source_paths = [
        generate_figures.TRACKS_PATH,
        generate_figures.TRACK_METRICS_PATH,
        generate_figures.CONDITION_COMPARISON_PATH,
        generate_figures.MOVIE_CONTROL_PATH,
        generate_figures.MOVIE_INJURY_PATH,
    ]
    hashes_before = {path: _sha256(path) for path in source_paths}

    monkeypatch.setattr(generate_figures, "figure_path", lambda name: tmp_path / name)
    generate_figures.main()

    for path, hash_before in hashes_before.items():
        assert _sha256(path) == hash_before

    expected_filenames = {
        "trajectories.png",
        "approach_distance_distribution.png",
        "primary_effect_ci.png",
        "tiff_overlay_frame29.png",
    }
    actual_filenames = {p.name for p in tmp_path.iterdir()}
    assert actual_filenames == expected_filenames

    for name in expected_filenames:
        content = (tmp_path / name).read_bytes()
        assert content.startswith(PNG_MAGIC)
        assert len(content) > 0
