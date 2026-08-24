"""Write the synthetic tracking table to data/tracks.csv."""

from __future__ import annotations

from pathlib import Path

from src.simulate import generate_tracks

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tracks.csv"


def main() -> None:
    tracks = generate_tracks()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(tracks)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
