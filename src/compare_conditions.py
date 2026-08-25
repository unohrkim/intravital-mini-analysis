"""Compute the deterministic condition-level statistical comparison.

Reads results/track_metrics.csv (control vs. injury) and writes
results/condition_comparison.csv. See src/stats_analysis.py for the
scientific/statistical definitions and interpretation guardrails.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.stats_analysis import compare_conditions

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACK_METRICS_PATH = REPO_ROOT / "results" / "track_metrics.csv"
CONDITION_COMPARISON_PATH = REPO_ROOT / "results" / "condition_comparison.csv"


def main() -> None:
    track_metrics = pd.read_csv(TRACK_METRICS_PATH)
    comparison = compare_conditions(track_metrics)

    CONDITION_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CONDITION_COMPARISON_PATH, index=False)

    print(f"Wrote {len(comparison)} rows to {CONDITION_COMPARISON_PATH}")
    print(
        "Note: results describe differences observed in this synthetic dataset "
        "under fixed simulation parameters. They are not biological or causal "
        "claims and do not validate the simulation model."
    )


if __name__ == "__main__":
    main()
