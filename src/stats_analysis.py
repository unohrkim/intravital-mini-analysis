"""Deterministic condition-level statistical comparison (control vs. injury).

Implements the Step 3 analysis approved by the researcher: a two-sided
permutation test on the mean difference in `approach_distance_um` (the
pre-specified primary metric) plus the same machinery applied to five
secondary/exploratory metrics, with Hedges' g, a percentile bootstrap CI,
and BH-FDR correction across the secondary metrics only.

Results describe differences observed in this synthetic dataset under its
fixed simulation parameters. They are not biological or causal claims, and
a statistically detectable difference here only demonstrates that this
pipeline can recover the directional bias that was deliberately built into
the injury condition's simulation — it does not validate the simulation
model or say anything about real cell biology. No interpretive narrative is
generated here; that remains a human responsibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

from src.simulate import CONDITIONS

STATS_RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 100_000
N_BOOTSTRAP: int = 10_000
ALPHA: float = 0.05

PRIMARY_METRIC: str = "approach_distance_um"
SECONDARY_METRICS: tuple[str, ...] = (
    "mean_speed_um_min",
    "path_length_um",
    "displacement_um",
    "persistence",
    "arrest_coefficient",
)
ALL_METRICS: tuple[str, ...] = (PRIMARY_METRIC, *SECONDARY_METRICS)

OUTPUT_COLUMNS: list[str] = [
    "metric",
    "is_primary",
    "n_control",
    "mean_control",
    "sd_control",
    "median_control",
    "min_control",
    "max_control",
    "q25_control",
    "q75_control",
    "n_injury",
    "mean_injury",
    "sd_injury",
    "median_injury",
    "min_injury",
    "max_injury",
    "q25_injury",
    "q75_injury",
    "mean_difference",
    "hedges_g",
    "ci_lower",
    "ci_upper",
    "welch_t_statistic",
    "welch_p_value",
    "permutation_statistic",
    "permutation_p_value",
    "p_value_adjusted_bh",
    "n_permutations",
    "n_bootstrap",
    "random_seed",
]


def _validate_conditions(track_metrics: pd.DataFrame) -> None:
    """Require the observed condition labels to exactly equal the expected set.

    Both "control" and "injury" must be present; an unexpected label, a
    missing group, or a NaN label all raise ValueError. Does not require any
    particular count per group.
    """
    if track_metrics["condition"].isna().any():
        n_missing = int(track_metrics["condition"].isna().sum())
        raise ValueError(
            f"track_metrics condition column contains {n_missing} missing "
            "(NaN) value(s); every row must have a non-null condition label"
        )

    observed = set(track_metrics["condition"].unique())
    expected = set(CONDITIONS)
    if observed != expected:
        missing = expected - observed
        unexpected = observed - expected
        raise ValueError(
            f"track_metrics condition labels must exactly equal {sorted(expected)}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _validate_unique_track_ids(track_metrics: pd.DataFrame) -> None:
    """Require every track_id to appear exactly once in the input table."""
    if not track_metrics["track_id"].is_unique:
        duplicates = sorted(
            track_metrics.loc[track_metrics["track_id"].duplicated(), "track_id"]
            .unique()
            .tolist()
        )
        raise ValueError(f"duplicate track_id values found: {duplicates}")


def _extract_group_values(
    track_metrics: pd.DataFrame, metric: str, condition: str
) -> np.ndarray:
    """Values for one metric/condition, with NaNs excluded (metric-wise exclusion)."""
    values = track_metrics.loc[track_metrics["condition"] == condition, metric].to_numpy(
        dtype=float
    )
    return values[~np.isnan(values)]


def _validate_sufficient_n(
    values: np.ndarray, metric: str, condition: str, minimum: int = 2
) -> None:
    """Require at least `minimum` non-NaN observations to compute variance-based stats."""
    if len(values) < minimum:
        raise ValueError(
            f"metric {metric!r} has only {len(values)} non-NaN observation(s) "
            f"for condition {condition!r}; at least {minimum} are required"
        )


def descriptive_stats(values: Sequence[float]) -> dict:
    """n, mean, sample sd, median, min, max, and 25th/75th percentiles.

    NaNs in `values` are excluded before computing any statistic.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "sd": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "q25": math.nan,
            "q75": math.nan,
        }
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if n > 1 else math.nan,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def mean_difference(control: Sequence[float], injury: Sequence[float]) -> float:
    """mean(injury) - mean(control), in the metric's native unit."""
    return float(np.mean(injury) - np.mean(control))


def hedges_g(control: Sequence[float], injury: Sequence[float]) -> float:
    """Bias-corrected standardized effect size; positive means injury > control.

    Raises ValueError if either group has fewer than 2 observations (variance
    is undefined below that, regardless of the caller). NaN if the pooled
    standard deviation is zero (the standardized ratio is undefined), rather
    than raising or returning inf.
    """
    control_arr = np.asarray(control, dtype=float)
    injury_arr = np.asarray(injury, dtype=float)
    n1, n2 = len(control_arr), len(injury_arr)

    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"hedges_g requires at least 2 observations per group; "
            f"got n_control={n1}, n_injury={n2}"
        )

    var1 = np.var(control_arr, ddof=1)
    var2 = np.var(injury_arr, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return math.nan

    cohens_d = (np.mean(injury_arr) - np.mean(control_arr)) / pooled_sd
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(cohens_d * correction)


def _mean_difference_statistic(x: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    """Vectorized mean(y) - mean(x) along `axis`, for scipy's resampling machinery."""
    return np.mean(y, axis=axis) - np.mean(x, axis=axis)


def permutation_test_mean_difference(
    control: Sequence[float],
    injury: Sequence[float],
    n_resamples: int = N_PERMUTATIONS,
    seed: int = STATS_RANDOM_SEED,
) -> tuple[float, float]:
    """Two-sided permutation test on mean(injury) - mean(control).

    Relies only on exchangeability of the condition labels under the null,
    not on a normal-theory approximation. If the number of distinct label
    permutations is small enough, scipy performs an exact test; otherwise it
    falls back to a Monte Carlo approximation using `n_resamples` resamples
    seeded by `seed`.
    """
    result = stats.permutation_test(
        (control, injury),
        statistic=_mean_difference_statistic,
        vectorized=True,
        n_resamples=n_resamples,
        alternative="two-sided",
        random_state=seed,
    )
    return float(result.statistic), float(result.pvalue)


def bootstrap_ci_mean_difference(
    control: Sequence[float],
    injury: Sequence[float],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = STATS_RANDOM_SEED,
    confidence_level: float = 1 - ALPHA,
) -> tuple[float, float]:
    """95% percentile bootstrap CI for mean(injury) - mean(control).

    Resamples tracks with replacement independently within each condition.
    """
    result = stats.bootstrap(
        (control, injury),
        statistic=_mean_difference_statistic,
        vectorized=True,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=seed,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def adjust_pvalues_bh(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    return false_discovery_control(np.asarray(pvalues, dtype=float), method="bh")


def compare_conditions(track_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the condition_comparison table: one row per metric in ALL_METRICS.

    approach_distance_um (PRIMARY_METRIC) is the single pre-specified
    confirmatory comparison; SECONDARY_METRICS are exploratory and receive
    BH-FDR-adjusted p-values (the primary row's p_value_adjusted_bh is NaN,
    since it is a single uncorrected test).
    """
    _validate_conditions(track_metrics)
    _validate_unique_track_ids(track_metrics)

    rows: list[dict] = []
    secondary_row_indices: list[int] = []
    secondary_raw_pvalues: list[float] = []

    for metric in ALL_METRICS:
        control_values = _extract_group_values(track_metrics, metric, "control")
        injury_values = _extract_group_values(track_metrics, metric, "injury")
        _validate_sufficient_n(control_values, metric, "control")
        _validate_sufficient_n(injury_values, metric, "injury")

        control_desc = descriptive_stats(control_values)
        injury_desc = descriptive_stats(injury_values)

        diff = mean_difference(control_values, injury_values)
        g = hedges_g(control_values, injury_values)
        ci_lower, ci_upper = bootstrap_ci_mean_difference(control_values, injury_values)
        welch_stat, welch_p = stats.ttest_ind(injury_values, control_values, equal_var=False)
        perm_stat, perm_p = permutation_test_mean_difference(control_values, injury_values)

        is_primary = metric == PRIMARY_METRIC

        rows.append(
            {
                "metric": metric,
                "is_primary": is_primary,
                "n_control": control_desc["n"],
                "mean_control": control_desc["mean"],
                "sd_control": control_desc["sd"],
                "median_control": control_desc["median"],
                "min_control": control_desc["min"],
                "max_control": control_desc["max"],
                "q25_control": control_desc["q25"],
                "q75_control": control_desc["q75"],
                "n_injury": injury_desc["n"],
                "mean_injury": injury_desc["mean"],
                "sd_injury": injury_desc["sd"],
                "median_injury": injury_desc["median"],
                "min_injury": injury_desc["min"],
                "max_injury": injury_desc["max"],
                "q25_injury": injury_desc["q25"],
                "q75_injury": injury_desc["q75"],
                "mean_difference": diff,
                "hedges_g": g,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "welch_t_statistic": float(welch_stat),
                "welch_p_value": float(welch_p),
                "permutation_statistic": perm_stat,
                "permutation_p_value": perm_p,
                "p_value_adjusted_bh": math.nan,
                "n_permutations": N_PERMUTATIONS,
                "n_bootstrap": N_BOOTSTRAP,
                "random_seed": STATS_RANDOM_SEED,
            }
        )

        if not is_primary:
            secondary_row_indices.append(len(rows) - 1)
            secondary_raw_pvalues.append(perm_p)

    if secondary_raw_pvalues:
        adjusted = adjust_pvalues_bh(secondary_raw_pvalues)
        for row_index, adjusted_p in zip(secondary_row_indices, adjusted):
            rows[row_index]["p_value_adjusted_bh"] = float(adjusted_p)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
