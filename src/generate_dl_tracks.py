"""Write a larger, separately-seeded synthetic tracking table to data/dl_tracks.csv.

This is a deep-learning dataset layer built on top of the existing Step 1-4
pipeline: it reuses src.simulate.generate_tracks() unchanged (same movement
model, same boundaries) at a larger scale (2,000 tracks/condition) and with
a dedicated seed, so it does not affect data/tracks.csv or the Step 1-4
scientific outputs.
"""

from __future__ import annotations

from pathlib import Path

from src.simulate import generate_tracks

REPO_ROOT = Path(__file__).resolve().parent.parent
DL_TRACKS_PATH = REPO_ROOT / "data" / "dl_tracks.csv"

DL_CELLS_PER_CONDITION = 2000
DL_SEED = 2026


def main() -> None:
    tracks = generate_tracks(
        cells_per_condition=DL_CELLS_PER_CONDITION,
        n_frames=30,
        seed=DL_SEED,
    )
    DL_TRACKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(DL_TRACKS_PATH, index=False)
    print(f"Wrote {len(tracks)} rows to {DL_TRACKS_PATH}")


if __name__ == "__main__":
    main()
