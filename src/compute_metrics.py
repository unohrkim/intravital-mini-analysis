"""Compute deterministic motility and sinusoid-migration metrics from data/tracks.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics import compute_frame_metrics, compute_step_metrics, compute_track_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKS_PATH = REPO_ROOT / "data" / "tracks.csv"
FRAME_METRICS_PATH = REPO_ROOT / "results" / "frame_metrics.csv"
STEP_METRICS_PATH = REPO_ROOT / "results" / "step_metrics.csv"
TRACK_METRICS_PATH = REPO_ROOT / "results" / "track_metrics.csv"


def main() -> None:
    tracks = pd.read_csv(TRACKS_PATH)

    frame_metrics = compute_frame_metrics(tracks)
    step_metrics = compute_step_metrics(tracks)
    track_metrics = compute_track_metrics(tracks)

    FRAME_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame_metrics.to_csv(FRAME_METRICS_PATH, index=False)
    step_metrics.to_csv(STEP_METRICS_PATH, index=False)
    track_metrics.to_csv(TRACK_METRICS_PATH, index=False)

    print(f"Wrote {len(frame_metrics)} rows to {FRAME_METRICS_PATH}")
    print(f"Wrote {len(step_metrics)} rows to {STEP_METRICS_PATH}")
    print(f"Wrote {len(track_metrics)} rows to {TRACK_METRICS_PATH}")


if __name__ == "__main__":
    main()
