"""Deterministic tests for src/stats_analysis.py."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from src.compare_conditions import TRACK_METRICS_PATH
from src.stats_analysis import (
    ALL_METRICS,
    OUTPUT_COLUMNS,
    PRIMARY_METRIC,
    STATS_RANDOM_SEED,
    _validate_conditions,
    _validate_sufficient_n,
    _validate_unique_track_ids,
    adjust_pvalues_bh,
    bootstrap_ci_mean_difference,
    compare_conditions,
    descriptive_stats,
    hedges_g,
    mean_difference,
    permutation_test_mean_difference,
)


# ---------------------------------------------------------------------------
# descriptive_stats
# ---------------------------------------------------------------------------


def test_descriptive_stats_hand_computed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = descriptive_stats(values)

    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["sd"] == pytest.approx(np.std(values, ddof=1))
    assert result["median"] == pytest.approx(3.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)
    assert result["q25"] == pytest.approx(2.0)
    assert result["q75"] == pytest.approx(4.0)


def test_descriptive_stats_excludes_nan():
    values = [1.0, 2.0, np.nan, 4.0]
    result = descriptive_stats(values)

    assert result["n"] == 3
    assert result["mean"] == pytest.approx((1.0 + 2.0 + 4.0) / 3)


def test_descriptive_stats_empty_is_all_nan():
    result = descriptive_stats([])

    assert result["n"] == 0
    assert math.isnan(result["mean"])
    assert math.isnan(result["sd"])
    assert math.isnan(result["median"])


# ---------------------------------------------------------------------------
# mean_difference
# ---------------------------------------------------------------------------


def test_mean_difference_known_values():
    assert mean_difference([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# hedges_g
# ---------------------------------------------------------------------------


def test_hedges_g_hand_computed_equal_group_sizes():
    control = np.array([1.0, 2.0, 3.0])
    injury = np.array([4.0, 5.0, 9.0])
    n1, n2 = 3, 3

    var1 = np.var(control, ddof=1)
    var2 = np.var(injury, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(injury) - np.mean(control)) / pooled_sd
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    expected_g = cohens_d * correction

    result = hedges_g(control, injury)
    assert result == pytest.approx(expected_g)
    # Equal group sizes do NOT reduce Hedges' g to plain Cohen's d.
    assert result != pytest.approx(cohens_d)


def test_hedges_g_sign_convention_flips_with_group_order():
    control = np.array([1.0, 2.0, 3.0])
    injury = np.array([4.0, 5.0, 9.0])

    g_forward = hedges_g(control, injury)
    g_swapped = hedges_g(injury, control)

    assert g_forward > 0  # injury > control here
    assert g_swapped == pytest.approx(-g_forward)


def test_hedges_g_zero_pooled_variance_is_nan():
    control = np.array([5.0, 5.0, 5.0])
    injury = np.array([9.0, 9.0])

    assert math.isnan(hedges_g(control, injury))


def test_hedges_g_raises_below_minimum_n():
    with pytest.raises(ValueError):
        hedges_g(np.array([1.0]), np.array([2.0, 3.0]))


# ---------------------------------------------------------------------------
# permutation_test_mean_difference
# ---------------------------------------------------------------------------


def test_permutation_test_matches_independent_exact_enumeration():
    control = np.array([1.0, 2.0, 5.0])
    injury = np.array([3.0, 6.0, 7.0])
    pooled = np.concatenate([control, injury])
    observed_diff = np.mean(injury) - np.mean(control)

    # Independently brute-force all C(6,3)=20 label splits, not reusing the
    # implementation under test.
    exact_diffs = []
    for combo in itertools.combinations(range(6), 3):
        idx_a = list(combo)
        idx_b = [i for i in range(6) if i not in idx_a]
        exact_diffs.append(np.mean(pooled[idx_b]) - np.mean(pooled[idx_a]))
    exact_diffs = np.array(exact_diffs)
    expected_p = np.mean(np.abs(exact_diffs) >= abs(observed_diff) - 1e-9)

    statistic, p_value = permutation_test_mean_difference(control, injury)

    assert statistic == pytest.approx(observed_diff)
    assert p_value == pytest.approx(expected_p, abs=1e-6)


def test_permutation_test_extreme_separation_gives_small_p():
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    injury = np.array([101.0, 102.0, 103.0, 104.0, 105.0])

    _, p_value = permutation_test_mean_difference(control, injury)

    assert p_value < 0.01


def test_permutation_test_identical_groups_gives_p_near_one():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    _, p_value = permutation_test_mean_difference(values, values.copy())

    assert p_value == pytest.approx(1.0)


def test_permutation_test_reproducible_with_fixed_seed():
    rng = np.random.default_rng(0)
    control = rng.normal(0, 1, size=15)
    injury = rng.normal(0.5, 1, size=15)

    result_a = permutation_test_mean_difference(
        control, injury, n_resamples=2000, seed=STATS_RANDOM_SEED
    )
    result_b = permutation_test_mean_difference(
        control, injury, n_resamples=2000, seed=STATS_RANDOM_SEED
    )

    assert result_a == result_b


# ---------------------------------------------------------------------------
# bootstrap_ci_mean_difference
# ---------------------------------------------------------------------------


def test_bootstrap_ci_reproducible_with_fixed_seed():
    rng = np.random.default_rng(1)
    control = rng.normal(0, 1, size=15)
    injury = rng.normal(0.5, 1, size=15)

    ci_a = bootstrap_ci_mean_difference(control, injury, n_resamples=500, seed=STATS_RANDOM_SEED)
    ci_b = bootstrap_ci_mean_difference(control, injury, n_resamples=500, seed=STATS_RANDOM_SEED)

    assert ci_a == ci_b


def test_bootstrap_ci_lower_le_upper():
    rng = np.random.default_rng(2)
    control = rng.normal(0, 1, size=15)
    injury = rng.normal(0.5, 1, size=15)

    ci_lower, ci_upper = bootstrap_ci_mean_difference(control, injury, n_resamples=500)

    assert ci_lower <= ci_upper


# ---------------------------------------------------------------------------
# adjust_pvalues_bh
# ---------------------------------------------------------------------------


def test_adjust_pvalues_bh_matches_hand_computed_example():
    raw = [0.005, 0.01, 0.03, 0.04, 0.2]
    expected = [0.025, 0.025, 0.05, 0.05, 0.2]

    adjusted = adjust_pvalues_bh(raw)

    assert adjusted.tolist() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_validate_conditions_rejects_unexpected_label():
    df = pd.DataFrame({"condition": ["control", "unknown"]})
    with pytest.raises(ValueError):
        _validate_conditions(df)


def test_validate_conditions_rejects_missing_group():
    df = pd.DataFrame({"condition": ["control", "control"]})
    with pytest.raises(ValueError):
        _validate_conditions(df)


def test_validate_conditions_rejects_nan_label():
    df = pd.DataFrame({"condition": ["control", np.nan]})
    with pytest.raises(ValueError):
        _validate_conditions(df)


def test_validate_conditions_accepts_both_groups_present():
    df = pd.DataFrame({"condition": ["control", "injury"]})
    _validate_conditions(df)  # should not raise


def test_validate_unique_track_ids_rejects_duplicate():
    df = pd.DataFrame({"track_id": ["A", "A", "B"]})
    with pytest.raises(ValueError):
        _validate_unique_track_ids(df)


def test_validate_sufficient_n_rejects_below_minimum():
    with pytest.raises(ValueError):
        _validate_sufficient_n(np.array([1.0]), metric="foo", condition="control")


def test_validate_sufficient_n_accepts_minimum():
    _validate_sufficient_n(np.array([1.0, 2.0]), metric="foo", condition="control")


# ---------------------------------------------------------------------------
# compare_conditions
# ---------------------------------------------------------------------------


def _make_track_metrics(rows: list[tuple]) -> pd.DataFrame:
    columns = ["track_id", "condition", *ALL_METRICS]
    return pd.DataFrame(rows, columns=columns)


def test_compare_conditions_raises_on_insufficient_n():
    rows = [
        ("C1", "control", 1.0, 1.0, 1.0, 1.0, 0.5, 0.1),  # only 1 control track
        ("I1", "injury", 2.0, 2.0, 2.0, 2.0, 0.5, 0.1),
        ("I2", "injury", 3.0, 3.0, 3.0, 3.0, 0.5, 0.1),
    ]
    df = _make_track_metrics(rows)

    with pytest.raises(ValueError):
        compare_conditions(df)


def test_compare_conditions_excludes_nan_metric_wise():
    rows = []
    for i in range(4):
        rows.append(
            (f"C{i}", "control", float(i), float(i) + 1, float(i) + 2, float(i) + 3, float(i) * 0.1, float(i) * 0.05)
        )
    for i in range(4):
        rows.append(
            (f"I{i}", "injury", float(i) + 10, float(i) + 11, float(i) + 12, float(i) + 13, float(i) * 0.1 + 0.5, float(i) * 0.05 + 0.2)
        )
    df = _make_track_metrics(rows)

    persistence_col = df.columns.get_loc("persistence")
    df.iloc[0, persistence_col] = np.nan  # one control track's persistence only

    result = compare_conditions(df)

    persistence_row = result.loc[result["metric"] == "persistence"].iloc[0]
    unaffected_row = result.loc[result["metric"] == PRIMARY_METRIC].iloc[0]

    assert persistence_row["n_control"] == 3
    assert unaffected_row["n_control"] == 4


# ---------------------------------------------------------------------------
# Integration: real, deterministic results/track_metrics.csv
# ---------------------------------------------------------------------------


def test_real_track_metrics_produces_expected_comparison_table():
    track_metrics = pd.read_csv(TRACK_METRICS_PATH)

    condition_counts = track_metrics.groupby("condition")["track_id"].nunique()
    assert condition_counts["control"] == 15
    assert condition_counts["injury"] == 15

    result = compare_conditions(track_metrics)

    assert len(result) == 6
    assert set(result["metric"]) == set(ALL_METRICS)
    assert (result["n_control"] == 15).all()
    assert (result["n_injury"] == 15).all()

    primary_row = result.loc[result["metric"] == PRIMARY_METRIC].iloc[0]
    assert bool(primary_row["is_primary"]) is True
    assert primary_row["mean_control"] == pytest.approx(0.691, abs=0.001)
    assert primary_row["mean_injury"] == pytest.approx(25.398, abs=0.001)

    assert result["permutation_p_value"].between(0, 1).all()
    assert result["welch_p_value"].between(0, 1).all()
    assert (result["ci_lower"] <= result["ci_upper"]).all()


def test_real_track_metrics_bh_correction_applies_only_to_secondary_metrics():
    track_metrics = pd.read_csv(TRACK_METRICS_PATH)
    result = compare_conditions(track_metrics)

    primary_row = result.loc[result["metric"] == PRIMARY_METRIC].iloc[0]
    secondary_rows = result.loc[result["metric"] != PRIMARY_METRIC]

    assert math.isnan(primary_row["p_value_adjusted_bh"])
    assert len(secondary_rows) == 5
    assert secondary_rows["p_value_adjusted_bh"].notna().all()
    assert secondary_rows["p_value_adjusted_bh"].between(0, 1).all()


def test_real_track_metrics_output_columns_match_schema():
    track_metrics = pd.read_csv(TRACK_METRICS_PATH)
    result = compare_conditions(track_metrics)

    assert list(result.columns) == OUTPUT_COLUMNS
