"""Tests for deterministic, synthetic-data eval-runner primitives used by the
frozen learned-Spawner comparison
(:mod:`qrokkun_env.frozen_spawner_eval_runner`).

These are pure statistics functions over already-computed elapsed-time
lists: no environment, no neural net, no checkpoint, and no training is
involved. Real-checkpoint end-to-end evaluation is out of scope for these
unit tests (see the preregistration journal entry).
"""

from __future__ import annotations

import math

import pytest

from qrokkun_env.frozen_spawner_eval_runner import (
    paired_difference_stats,
    restricted_survival_stats,
)


def test_restricted_survival_stats_basic():
    # cap = 10; one episode reaches the cap (censored), others are hit early.
    elapsed = [3.0, 5.0, 10.0, 7.0]
    stats = restricted_survival_stats(elapsed, cap=10.0)
    assert stats["restricted_mean"] == pytest.approx((3.0 + 5.0 + 10.0 + 7.0) / 4)
    assert stats["median"] == pytest.approx(6.0)
    assert stats["min"] == pytest.approx(3.0)
    assert stats["max"] == pytest.approx(10.0)
    assert stats["censor_count"] == 1
    assert stats["n"] == 4
    assert stats["per_seed_values"] == [3.0, 5.0, 10.0, 7.0]
    assert stats["hit_flags"] == [True, True, False, True]
    assert stats["hit_rate"] == pytest.approx(3 / 4)


def test_restricted_survival_stats_caps_values_above_cap():
    # elapsed values are never allowed to exceed the cap in the reported stat.
    elapsed = [12.0, 8.0]
    stats = restricted_survival_stats(elapsed, cap=10.0)
    assert stats["per_seed_values"] == [10.0, 8.0]
    assert stats["censor_count"] == 1
    assert stats["max"] == pytest.approx(10.0)


def test_restricted_survival_stats_empty_rejected():
    with pytest.raises(ValueError, match="empty"):
        restricted_survival_stats([], cap=10.0)


def test_restricted_survival_stats_non_positive_cap_rejected():
    with pytest.raises(ValueError, match="cap"):
        restricted_survival_stats([1.0], cap=0.0)


def test_early_death_rate_uses_threshold():
    elapsed = [1.0, 6.0, 10.0, 4.9]
    stats = restricted_survival_stats(elapsed, cap=10.0, early_death_threshold=5.0)
    # 1.0 and 4.9 are < 5.0 -> early deaths; 6.0 and 10.0 are not.
    assert stats["early_death_rate"] == pytest.approx(2 / 4)


def test_paired_difference_stats_basic():
    a = [3.0, 5.0, 7.0, 9.0]
    b = [4.0, 6.0, 6.0, 11.0]
    diffs = [b_i - a_i for a_i, b_i in zip(a, b)]  # [1, 1, -1, 2]
    stats = paired_difference_stats(a, b)
    assert stats["n"] == 4
    assert stats["per_seed_diff"] == diffs
    assert stats["mean_diff"] == pytest.approx(sum(diffs) / 4)
    assert stats["median_diff"] == pytest.approx(1.0)
    lo, hi = stats["ci95"]
    assert lo <= stats["mean_diff"] <= hi
    assert stats["improve_count"] + stats["worsen_count"] + stats["unchanged_count"] == 4


@pytest.mark.parametrize(
    ("a", "b", "direction", "classification"),
    [
        ([1.0, 1.0], [2.0, 2.0], "b_higher_than_a", "higher"),
        ([2.0, 2.0], [1.0, 1.0], "b_lower_than_a", "lower"),
        ([1.0, 2.0], [2.0, 1.0], "no_clear_direction", "indistinguishable"),
    ],
)
def test_paired_difference_preregistered_ci_classification(a, b, direction, classification):
    stats = paired_difference_stats(a, b)
    assert stats["direction"] == direction
    assert stats["classification"] == classification
    assert stats["classification_basis"] == "paired_mean_difference_95pct_ci"
    assert stats["classification_rule"] == "lower>0:higher; upper<0:lower; otherwise:indistinguishable"


def test_paired_difference_ci_boundary_at_zero_is_indistinguishable():
    stats = paired_difference_stats([1.0], [1.0])
    assert stats["ci95"] == (0.0, 0.0)
    assert stats["classification"] == "indistinguishable"


def test_paired_difference_stats_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length"):
        paired_difference_stats([1.0, 2.0], [1.0])


def test_paired_difference_stats_single_pair_has_no_finite_ci_variance():
    # n=1: pstdev of diffs is 0, so CI collapses to the single point (not NaN/inf).
    stats = paired_difference_stats([5.0], [7.0])
    assert stats["mean_diff"] == pytest.approx(2.0)
    lo, hi = stats["ci95"]
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_paired_difference_stats_empty_rejected():
    with pytest.raises(ValueError, match="empty"):
        paired_difference_stats([], [])
