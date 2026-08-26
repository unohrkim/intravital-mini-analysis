"""Deterministic publication-style figures summarizing the tracking pipeline.

Reads already-validated outputs (data/tracks.csv, results/track_metrics.csv,
results/condition_comparison.csv, the per-condition TIFF movies) and produces
Figure objects only — no file I/O, no new scientific computation. Pure
functions only, mirroring the src/render_movie.py split.

matplotlib.use("Agg") is set before importing pyplot so rendering is headless
and deterministic regardless of environment.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.render_movie import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, um_to_pixel  # noqa: E402
from src.simulate import CONDITIONS, FRAMES_PER_CELL, SINUSOID_X_UM  # noqa: E402

CONDITION_COLORS: dict[str, str] = {"control": "#1f77b4", "injury": "#d62728"}
FIGURE_DPI: int = 150
REPRESENTATIVE_N_PER_CONDITION: int = 5
OVERLAY_FRAME_INDEX: int = FRAMES_PER_CELL - 1  # 29
OVERLAY_COLOR: str = "lime"
POINT_OFFSET_SPACING: float = 0.06


def select_representative_track_ids(
    tracks: pd.DataFrame, n_per_condition: int = REPRESENTATIVE_N_PER_CONDITION
) -> dict[str, list[str]]:
    """First n_per_condition track_ids (sorted) for each condition. Deterministic."""
    selected: dict[str, list[str]] = {}
    for condition in CONDITIONS:
        ids = sorted(tracks.loc[tracks["condition"] == condition, "track_id"].unique())
        selected[condition] = ids[:n_per_condition]
    return selected


def plot_trajectories(
    tracks: pd.DataFrame, n_per_condition: int = REPRESENTATIVE_N_PER_CONDITION
) -> plt.Figure:
    """Representative control/injury trajectories (x_um vs y_um) with the sinusoid."""
    selected = select_representative_track_ids(tracks, n_per_condition)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axvline(SINUSOID_X_UM, color="black", linestyle="--", linewidth=1)

    for condition in CONDITIONS:
        for track_id in selected[condition]:
            track = tracks.loc[tracks["track_id"] == track_id].sort_values("frame")
            ax.plot(
                track["x_um"],
                track["y_um"],
                color=CONDITION_COLORS[condition],
                marker="o",
                markersize=2,
                linewidth=1,
            )

    legend_handles = [
        plt.Line2D([0], [0], color=CONDITION_COLORS[condition], label=condition)
        for condition in CONDITIONS
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linestyle="--", label=f"sinusoid (x={SINUSOID_X_UM:g} µm)")
    )
    ax.legend(handles=legend_handles, loc="upper right")

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Representative trajectories ({n_per_condition} per condition)")

    fig.tight_layout()
    return fig


def _condition_values_sorted_by_track_id(
    track_metrics: pd.DataFrame, condition: str, column: str
) -> np.ndarray:
    subset = track_metrics.loc[track_metrics["condition"] == condition].sort_values("track_id")
    return subset[column].to_numpy(dtype=float)


def plot_approach_distance_distribution(track_metrics: pd.DataFrame) -> plt.Figure:
    """Box plot + individual track points of approach_distance_um, per condition."""
    fig, ax = plt.subplots(figsize=(5, 5))

    positions = list(range(1, len(CONDITIONS) + 1))
    box_data = [
        _condition_values_sorted_by_track_id(track_metrics, condition, "approach_distance_um")
        for condition in CONDITIONS
    ]
    ax.boxplot(box_data, positions=positions, widths=0.5, showfliers=False)

    for position, condition, values in zip(positions, CONDITIONS, box_data):
        n = len(values)
        offsets = [(i - (n - 1) / 2) * POINT_OFFSET_SPACING for i in range(n)]
        x = [position + offset for offset in offsets]
        ax.scatter(x, values, color=CONDITION_COLORS[condition], zorder=3, s=15)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(CONDITIONS))
    ax.set_xlabel("condition")
    ax.set_ylabel("approach_distance_um (µm)")
    ax.set_title("Track-level approach distance toward the sinusoid")

    fig.tight_layout()
    return fig


def plot_primary_effect_ci(condition_comparison: pd.DataFrame) -> plt.Figure:
    """Mean difference (injury - control) with its 95% bootstrap CI, primary metric only."""
    primary_rows = condition_comparison.loc[condition_comparison["is_primary"].astype(bool)]
    if len(primary_rows) != 1:
        raise ValueError(
            f"expected exactly one is_primary=True row, got {len(primary_rows)}"
        )
    row = primary_rows.iloc[0]

    mean_diff = float(row["mean_difference"])
    ci_lower = float(row["ci_lower"])
    ci_upper = float(row["ci_upper"])
    metric = row["metric"]

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(
        [mean_diff],
        [0],
        xerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]],
        fmt="o",
        color="black",
        capsize=4,
    )
    ax.text(
        mean_diff,
        0.15,
        f"{mean_diff:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]",
        ha="center",
        va="bottom",
    )
    ax.set_yticks([0])
    ax.set_yticklabels([f"{metric} (primary)"])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("mean difference, injury − control (µm)")
    ax.set_title("Primary effect: mean difference with 95% bootstrap CI")

    fig.tight_layout()
    return fig


def plot_tiff_overlay(
    movie_frame_control: np.ndarray,
    movie_frame_injury: np.ndarray,
    tracks: pd.DataFrame,
    frame_index: int = OVERLAY_FRAME_INDEX,
) -> plt.Figure:
    """Each condition's frame-`frame_index` TIFF image with a trajectory overlay.

    Overlay coordinates are snapped through the same um_to_pixel used to render
    the TIFF (src.render_movie.um_to_pixel), and the image is displayed with a
    pixel-center-aligned extent, so this figure shows the RENDERED PIXEL-GRID
    correspondence between tracks.csv and the TIFF - not the raw continuous
    track coordinates.

    origin="upper" is passed explicitly (matching the TIFF rendering convention
    that y=0 is the top row) rather than relying on matplotlib's
    rcParams["image.origin"] default.
    """
    extent = [-0.5, IMAGE_WIDTH_PX - 0.5, IMAGE_HEIGHT_PX - 0.5, -0.5]
    frames = {"control": movie_frame_control, "injury": movie_frame_injury}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, condition in zip(axes, CONDITIONS):
        ax.imshow(frames[condition], cmap="gray", vmin=0, vmax=255, extent=extent, origin="upper")

        condition_tracks = tracks.loc[
            (tracks["condition"] == condition) & (tracks["frame"] <= frame_index)
        ]

        for track_id, track in condition_tracks.groupby("track_id", sort=True):
            track = track.sort_values("frame")
            x_px = [um_to_pixel(x, IMAGE_WIDTH_PX) for x in track["x_um"]]
            y_px = [um_to_pixel(y, IMAGE_HEIGHT_PX) for y in track["y_um"]]
            ax.plot(x_px, y_px, color=OVERLAY_COLOR, linewidth=0.6, alpha=0.7)
            ax.plot(
                x_px[-1],
                y_px[-1],
                marker="o",
                markerfacecolor="none",
                markeredgecolor=OVERLAY_COLOR,
                markersize=5,
            )

        ax.set_xlim(-0.5, IMAGE_WIDTH_PX - 0.5)
        ax.set_ylim(IMAGE_HEIGHT_PX - 0.5, -0.5)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(f"{condition.capitalize()} — frame {frame_index}")

    fig.tight_layout()
    return fig
