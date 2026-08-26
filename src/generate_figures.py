"""Generate the Step 4B figure set from the existing validated outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tifffile

from src.visualize import (
    FIGURE_DPI,
    OVERLAY_FRAME_INDEX,
    plot_approach_distance_distribution,
    plot_primary_effect_ci,
    plot_tiff_overlay,
    plot_trajectories,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKS_PATH = REPO_ROOT / "data" / "tracks.csv"
TRACK_METRICS_PATH = REPO_ROOT / "results" / "track_metrics.csv"
CONDITION_COMPARISON_PATH = REPO_ROOT / "results" / "condition_comparison.csv"
MOVIE_CONTROL_PATH = REPO_ROOT / "data" / "synthetic_movie_control.tif"
MOVIE_INJURY_PATH = REPO_ROOT / "data" / "synthetic_movie_injury.tif"
FIGURES_DIR = REPO_ROOT / "results" / "figures"


def figure_path(name: str) -> Path:
    return FIGURES_DIR / name


def main() -> None:
    tracks = pd.read_csv(TRACKS_PATH)
    track_metrics = pd.read_csv(TRACK_METRICS_PATH)
    condition_comparison = pd.read_csv(CONDITION_COMPARISON_PATH)
    movie_control = tifffile.imread(MOVIE_CONTROL_PATH)
    movie_injury = tifffile.imread(MOVIE_INJURY_PATH)

    figures = {
        "trajectories.png": plot_trajectories(tracks),
        "approach_distance_distribution.png": plot_approach_distance_distribution(track_metrics),
        "primary_effect_ci.png": plot_primary_effect_ci(condition_comparison),
        "tiff_overlay_frame29.png": plot_tiff_overlay(
            movie_control[OVERLAY_FRAME_INDEX], movie_injury[OVERLAY_FRAME_INDEX], tracks
        ),
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        output_path = figure_path(name)
        fig.savefig(output_path, dpi=FIGURE_DPI)
        plt.close(fig)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
