"""Write a deterministic, condition-stratified train/val/test split to data/dl_split.csv.

Reads data/dl_tracks.csv, derives one (track_id, condition) pair per track,
and delegates the actual split logic to the pure src.dl_data.split_track_ids
so the same function can later be reused unchanged by TensorFlow and PyTorch
data loaders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dl_data import split_track_ids

REPO_ROOT = Path(__file__).resolve().parent.parent
DL_TRACKS_PATH = REPO_ROOT / "data" / "dl_tracks.csv"
DL_SPLIT_PATH = REPO_ROOT / "data" / "dl_split.csv"

# Dedicated split seed, distinct from DL_SEED=2026, to decouple partition
# randomness from trajectory-generation randomness.
DL_SPLIT_SEED = 2027


def main() -> None:
    tracks = pd.read_csv(DL_TRACKS_PATH)
    track_conditions = tracks[["track_id", "condition"]].drop_duplicates()

    split_df = split_track_ids(
        track_conditions["track_id"],
        track_conditions["condition"],
        seed=DL_SPLIT_SEED,
    )

    DL_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(DL_SPLIT_PATH, index=False)
    print(f"Wrote {len(split_df)} rows to {DL_SPLIT_PATH}")


if __name__ == "__main__":
    main()
