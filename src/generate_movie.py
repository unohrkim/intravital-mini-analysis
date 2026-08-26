"""Render data/tracks.csv into per-condition synthetic TIFF movies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tifffile

from src.render_movie import render_movie

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKS_PATH = REPO_ROOT / "data" / "tracks.csv"
CONDITIONS: tuple[str, ...] = ("control", "injury")


def movie_path(condition: str) -> Path:
    return REPO_ROOT / "data" / f"synthetic_movie_{condition}.tif"


def main() -> None:
    tracks = pd.read_csv(TRACKS_PATH)

    for condition in CONDITIONS:
        condition_tracks = tracks.loc[tracks["condition"] == condition]
        movie = render_movie(condition_tracks)

        output_path = movie_path(condition)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_path, movie, photometric="minisblack")

        print(f"Wrote {movie.shape[0]} frames to {output_path}")


if __name__ == "__main__":
    main()
