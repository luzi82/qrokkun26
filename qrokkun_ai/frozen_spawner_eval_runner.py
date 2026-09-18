"""Deterministic eval-runner primitives for the frozen learned-Spawner
PlayerV5 comparison.

These are pure statistics functions over already-computed per-seed elapsed
survival times. They implement exactly the restricted metrics and paired
comparison contract fixed in
docs/journal/<...>_player_v5_frozen_spawner_eval_preregistration.txt:

* restricted mean/median/std/min/max of ``min(elapsed, cap)``, censor count,
  hit rate, early-death rate, and full per-seed values
  (:func:`restricted_survival_stats`);
* paired per-seed differences, paired mean/median difference, and a
  pre-specified paired 95% t-interval, with no CI even when there is only
  one seed (:func:`paired_difference_stats`).

Nothing here runs an environment, loads a checkpoint, or performs any
gradient step; these functions accept plain lists of floats, *or* lists of
structured per-seed outcome mappings (as returned by
``qrokkun_ai.frozen_spawner_eval_cli.run_frozen_spawner_episode``, i.e.
mappings with ``elapsed``/``hit``/``censored``/``termination_reason`` keys).
Wiring these to an actual environment/Player/Spawner rollout (via
``qrokkun_ai.eval_modes``/``qrokkun_ai.train.both_v4.run_episode``) is
deliberately left to a future, separately-run eval script — these unit
tests only exercise the statistics with synthetic data.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Sequence, Union


def restricted_survival_stats(
    elapsed: Sequence[Union[float, Mapping]],
    *,
    cap: float,
    early_death_threshold: float | None = None,
) -> dict:
    """Restricted-mean survival statistics: ``min(elapsed_i, cap)`` per seed.

    ``elapsed`` may be a plain sequence of floats (legacy contract: a hit
    flag is *derived* as ``capped_value < cap``), or a sequence of explicit
    per-seed outcome mappings each carrying ``elapsed`` and ``hit`` (as
    returned by ``run_frozen_spawner_episode``). The explicit-outcome form is
    required to correctly classify a hit that lands exactly on the frame
    cap: such an episode has ``capped_value == cap`` but ``hit=True``, which
    the derived ``value < cap`` rule alone cannot distinguish from a
    same-elapsed censor.

    Returns a dict with ``restricted_mean``, ``median``, ``std``, ``min``,
    ``max``, ``n``, ``censor_count`` (episodes that reached ``cap`` without
    being hit), ``hit_rate``, ``per_seed_values`` (the capped values), and
    ``hit_flags`` (True where the episode ended in a hit). If
    ``early_death_threshold`` is given, also reports ``early_death_rate``
    (fraction of seeds with capped value strictly below the threshold). When
    the input entries are outcome mappings carrying a ``termination_reason``,
    the per-seed reasons are also reported as ``termination_reasons``.
    """
    if not elapsed:
        raise ValueError("elapsed must be non-empty")
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    cap = float(cap)
    values = [float(e["elapsed"]) if isinstance(e, Mapping) else float(e) for e in elapsed]
    capped = [min(v, cap) for v in values]
    hit_flags = [
        bool(e["hit"]) if isinstance(e, Mapping) else (v < cap) for e, v in zip(elapsed, capped)
    ]
    n = len(capped)
    stats: dict = {
        "n": n,
        "restricted_mean": sum(capped) / n,
        "median": float(statistics.median(capped)),
        "std": float(statistics.pstdev(capped)) if n > 1 else 0.0,
        "min": min(capped),
        "max": max(capped),
        "censor_count": sum(1 for h in hit_flags if not h),
        "hit_rate": sum(1 for h in hit_flags if h) / n,
        "per_seed_values": capped,
        "hit_flags": hit_flags,
    }
    if early_death_threshold is not None:
        stats["early_death_rate"] = sum(1 for v in capped if v < early_death_threshold) / n
    if any(isinstance(e, Mapping) and "termination_reason" in e for e in elapsed):
        stats["termination_reasons"] = [
            e.get("termination_reason") if isinstance(e, Mapping) else None for e in elapsed
        ]
    return stats


def paired_difference_stats(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired ``b - a`` differences with a pre-specified paired 95% t-interval.

    Returns ``n``, ``per_seed_diff``, ``mean_diff``, ``median_diff``,
    ``std_diff``, ``ci95`` (a ``(lo, hi)`` tuple), and
    ``improve_count``/``worsen_count``/``unchanged_count`` (b > a / b < a /
    b == a). With a single pair the sample std is 0 and the interval
    collapses to the single point (finite, not NaN/inf).
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: len(a)={len(a)} != len(b)={len(b)}")
    if not a:
        raise ValueError("a/b must be non-empty")

    diffs = [float(bi) - float(ai) for ai, bi in zip(a, b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    median_diff = float(statistics.median(diffs))
    std_diff = float(statistics.stdev(diffs)) if n > 1 else 0.0

    if n > 1 and std_diff > 0.0:
        # Approximate paired 95% t-interval; t critical value depends only on
        # degrees of freedom (n - 1), fixed here via a lookup rather than a
        # scipy dependency (this module deliberately has none).
        t_crit = _t_critical_95(n - 1)
        half_width = t_crit * std_diff / math.sqrt(n)
    else:
        # n == 1, or all paired differences identical (std_diff == 0): the
        # interval collapses to the point estimate rather than being
        # undefined/NaN.
        half_width = 0.0
    ci95 = (mean_diff - half_width, mean_diff + half_width)

    # This is a preregistered descriptive diagnostic only.  It deliberately
    # does not decide promotion or pass/fail: a CI touching zero (including
    # exactly zero) is indistinguishable.
    lower, upper = ci95
    if lower > 0.0:
        direction, classification = "b_higher_than_a", "higher"
    elif upper < 0.0:
        direction, classification = "b_lower_than_a", "lower"
    else:
        direction, classification = "no_clear_direction", "indistinguishable"

    improve = sum(1 for d in diffs if d > 0.0)
    worsen = sum(1 for d in diffs if d < 0.0)
    unchanged = n - improve - worsen

    return {
        "n": n,
        "per_seed_diff": diffs,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "std_diff": std_diff,
        "ci95": ci95,
        "improve_count": improve,
        "worsen_count": worsen,
        "unchanged_count": unchanged,
        "direction": direction,
        "classification": classification,
        "classification_basis": "paired_mean_difference_95pct_ci",
        "classification_rule": "lower>0:higher; upper<0:lower; otherwise:indistinguishable",
    }


# Two-sided 95% t critical values keyed by degrees of freedom, for the paired
# interval above. Falls back to the normal z=1.96 for df beyond this table
# (matches the t distribution closely for df > 60).
_T_TABLE_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t_critical_95(df: int) -> float:
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    if df < 1:
        return _T_TABLE_95[1]
    if df > 30:
        return 1.96
    # df between table entries but not covered above (shouldn't happen given
    # the 1..30 range is dense); fall back to the nearest lower entry.
    return _T_TABLE_95[max(k for k in _T_TABLE_95 if k <= df)]
