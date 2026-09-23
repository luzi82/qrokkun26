#!/usr/bin/env python3
"""Pair analysis finalizer for a completed aux + control Phase 3 pair.

Run from the repository root with::

    PYTHONPATH=. python -m qrokkun_ai.v5.tools.phase3_pair_analysis \
        --aux-run-dir PATH --control-run-dir PATH \
        --updates-start START --updates-stop STOP --updates-step STEP \
        --eval-seed-start FIRST --eval-seed-count COUNT \
        --out analysis.json

Equivalently, pass ``--update N`` / ``--eval-seed S`` repeatedly, or point
``--config`` at a JSON file carrying ``updates`` and ``eval_seeds``.

The primary estimator is the pre-registered paired Student-t over PER-SEED
curve-averaged differences::

    d_s   = (1/|U|) * sum_{u in U} (elapsed_aux(u, s) - elapsed_control(u, s))
    dbar  = mean_s d_s
    CI    = dbar +- t_{1-alpha/2, n-1} * s_d / sqrt(n)

The matched update set ``U`` and the evaluation seed window are NEVER
defaulted: a particular experiment's ``U`` is a property of that
experiment's preregistration, not of this tool, so both must be stated
explicitly on the command line or in a config file.  The 35 checkpoints in a
``U`` are points on ONE training trajectory sharing one seed window; they are
never treated as independent replications, which is why inference is over the
``n`` evaluation seeds and never over ``|U|``.

This tool only reads finished run directories.  It never trains, never
evaluates an environment and never writes into a run directory unless asked.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from qrokkun_ai.v5.tools import phase3_run_provenance as prov

ANALYSIS_SCHEMA_VERSION = 1

CLASSIFICATION_SUPPORT = "support"
CLASSIFICATION_CONTRARY = "contrary"
CLASSIFICATION_INDETERMINATE = "indeterminate"
CLASSIFICATION_INVALID = "invalid"


class PairAnalysisError(RuntimeError):
    """The requested pair cannot be analysed as specified."""


# --------------------------------------------------------------------------- #
# Student-t quantile (computed here; never a hand-written approximation)
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    x = df / (df + t * t)
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: int) -> float:
    """Inverse CDF of Student's t, by bisection on the monotone CDF.

    Computing the quantile from the distribution keeps the analysis honest
    for any ``n`` and any confidence level, rather than pinning a constant
    that silently stops matching if either ever changes.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if p == 0.5:
        return 0.0
    low, high = -1e9, 1e9
    for _ in range(400):
        mid = 0.5 * (low + high)
        if student_t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# --------------------------------------------------------------------------- #
# the pre-registered primary estimator
# --------------------------------------------------------------------------- #
def valid_confidence(confidence: Any) -> bool:
    """Whether ``confidence`` names a two-sided interval that exists at all.

    Only a real number strictly inside ``(0, 1)`` has a two-sided Student-t
    quantile.  ``True`` is rejected with the other non-numbers: a boolean that
    silently reads as ``1.0`` would look like a 100% interval.
    """
    return (
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(confidence)
        and 0.0 < float(confidence) < 1.0
    )


def _exact_int(value: Any, *, label: str) -> int:
    """A real update index or seed: ``bool`` and ``float`` are not ints."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PairAnalysisError(f"{label} must be an exact int; got {value!r}")
    return value


def _exact_int_list(values: Any, *, label: str) -> list[int]:
    if not isinstance(values, list):
        raise PairAnalysisError(
            f"{label} must be a list of exact ints; got {type(values).__name__}"
        )
    return [_exact_int(item, label=label) for item in values]


def _canonical_index(value: Any) -> int | None:
    """An exact ``int``, or a canonical decimal digit string (``"0"``, ``"10"``).

    ``True``, ``10.0``, ``"01"``, ``"+1"`` and ``"1.0"`` are not indices: each
    of them would otherwise coerce onto a real seed and change the grid.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdecimal()
        and (value == "0" or not value.startswith("0"))
    ):
        return int(value)
    return None


def _require_index(value: Any, *, label: str) -> int:
    parsed = _canonical_index(value)
    if parsed is None:
        raise PairAnalysisError(
            f"{label} must be an exact int or a canonical decimal string; got {value!r}"
        )
    return parsed


def _real_finite(value: Any) -> float | None:
    """A finite ``int`` or ``float``. ``bool`` is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _require_elapsed(value: Any, *, label: str) -> float:
    parsed = _real_finite(value)
    if parsed is None:
        raise PairAnalysisError(
            f"{label} must be a finite real number; got {value!r}"
        )
    return parsed


def _index_elapsed(
    grid: Any, *, arm: str,
) -> tuple[dict[int, dict[int, float]], list[str], set[tuple[int, int]]]:
    """Copy ``grid`` into exact-int keys and finite elapsed values.

    Malformed keys and non-finite cells are reported and omitted from the
    copy, so the estimator can fail closed without calling ``statistics``
    on them.  ``rejected`` names cells that were present but unusable, so
    they are not also described as missing.
    """
    reasons: list[str] = []
    rejected: set[tuple[int, int]] = set()
    parsed: dict[int, dict[int, float]] = {}
    if not isinstance(grid, dict):
        return parsed, [f"{arm} elapsed grid is malformed"], rejected
    for update_key, block in grid.items():
        update = _canonical_index(update_key)
        if update is None:
            reasons.append(f"{arm} malformed update key {update_key!r}")
            continue
        if update in parsed:
            reasons.append(f"{arm} duplicate update key {update_key!r}")
            continue
        if not isinstance(block, dict):
            reasons.append(f"{arm} elapsed for update {update} is malformed")
            continue
        cells: dict[int, float] = {}
        for seed_key, value in block.items():
            seed = _canonical_index(seed_key)
            if seed is None:
                reasons.append(f"{arm} malformed seed key {seed_key!r}")
                continue
            if seed in cells:
                reasons.append(f"{arm} duplicate seed key {seed_key!r}")
                continue
            elapsed = _real_finite(value)
            if elapsed is None:
                reasons.append(
                    f"{arm} elapsed for update {update} seed {seed} is not a finite "
                    f"real number; got {value!r}"
                )
                rejected.add((update, seed))
                continue
            cells[seed] = elapsed
        parsed[update] = cells
    return parsed, reasons, rejected


def require_confidence(confidence: Any) -> float:
    """The confidence level for an analysis, or fail closed on an impossible one."""
    if not valid_confidence(confidence):
        raise PairAnalysisError(
            f"confidence must be a real number strictly between 0 and 1; got {confidence!r}"
        )
    return float(confidence)


def _censor_summary(
    updates: list[int],
    eval_seeds: list[int],
    censored: dict[int, dict[int, bool]] | None,
) -> dict[str, Any]:
    if censored is None:
        return {"status": prov.EVIDENCE_UNAVAILABLE, "count": None, "cells": []}
    cells = [
        {"update": update, "seed": seed}
        for update in updates
        for seed in eval_seeds
        if censored.get(update, {}).get(seed)
    ]
    return {"status": prov.EVIDENCE_OBSERVED, "count": len(cells), "cells": cells}


def paired_curve_difference(
    *,
    updates: list[int],
    eval_seeds: list[int],
    aux_elapsed: dict[int, dict[int, float]],
    control_elapsed: dict[int, dict[int, float]],
    aux_censored: dict[int, dict[int, bool]] | None = None,
    control_censored: dict[int, dict[int, bool]] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """The primary paired Student-t over per-seed curve-averaged differences.

    Inference is over the ``n`` evaluation seeds.  The ``|U|`` checkpoints are
    points on one trajectory and are averaged INTO each seed's ``d_s``; they
    are never counted as independent samples.

    Any missing cell of the ``|U|`` x ``n`` grid invalidates the whole pair.
    Dropping a seed or shrinking ``U`` to rescue an estimate is exactly the
    post-hoc freedom the preregistration forbids, so it fails closed instead.
    """
    updates = _exact_int_list(updates, label="updates")
    eval_seeds = _exact_int_list(eval_seeds, label="eval_seeds")
    aux_elapsed, aux_reasons, aux_rejected = _index_elapsed(aux_elapsed, arm="aux")
    control_elapsed, control_reasons, control_rejected = _index_elapsed(
        control_elapsed, arm="control",
    )
    reasons: list[str] = [*aux_reasons, *control_reasons]

    if not valid_confidence(confidence):
        reasons.append(
            f"confidence must be a real number strictly between 0 and 1; got {confidence!r}"
        )
    if not updates:
        reasons.append("the matched update set U is empty")
    if len(eval_seeds) < 2:
        reasons.append("a paired interval needs at least two evaluation seeds")
    if len(set(updates)) != len(updates):
        reasons.append("the matched update set U contains duplicates")
    if len(set(eval_seeds)) != len(eval_seeds):
        reasons.append("the evaluation seed window contains duplicates")

    for update in updates:
        for seed in eval_seeds:
            if (update, seed) not in aux_rejected and seed not in aux_elapsed.get(update, {}):
                reasons.append(f"aux missing elapsed for update {update} seed {seed}")
            if (update, seed) not in control_rejected and seed not in control_elapsed.get(update, {}):
                reasons.append(f"control missing elapsed for update {update} seed {seed}")

    censoring: dict[str, Any] = {
        "aux": _censor_summary(updates, eval_seeds, aux_censored),
        "control": _censor_summary(updates, eval_seeds, control_censored),
        "total_cells": len(updates) * len(eval_seeds),
    }
    censoring["status"] = (
        prov.EVIDENCE_OBSERVED
        if aux_censored is not None and control_censored is not None
        else prov.EVIDENCE_UNAVAILABLE
    )
    censoring["structurally_different"] = (
        censoring["status"] == prov.EVIDENCE_OBSERVED
        and censoring["aux"]["cells"] != censoring["control"]["cells"]
    )

    base: dict[str, Any] = {
        "estimator": "paired_student_t_over_per_seed_curve_difference",
        "updates": updates,
        "eval_seeds": eval_seeds,
        "n_updates": len(updates),
        "n": len(eval_seeds),
        "confidence": confidence,
        "censoring": censoring,
        "reasons": reasons,
    }

    if reasons:
        return {
            **base,
            "valid": False,
            "classification": CLASSIFICATION_INVALID,
            "per_seed_d_s": {},
            "dbar": None,
            "sample_sd": None,
            "df": None,
            "t_quantile": None,
            "standard_error": None,
            "ci_lower": None,
            "ci_upper": None,
            "provenance": prov.EVIDENCE_UNAVAILABLE,
        }

    per_seed = {
        str(seed): statistics.fmean(
            aux_elapsed[update][seed] - control_elapsed[update][seed] for update in updates
        )
        for seed in eval_seeds
    }
    values = [per_seed[str(seed)] for seed in eval_seeds]
    n = len(values)
    dbar = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    df = n - 1
    quantile = student_t_ppf(0.5 + confidence / 2.0, df)
    standard_error = sample_sd / math.sqrt(n)
    margin = quantile * standard_error
    lower, upper = dbar - margin, dbar + margin

    if lower > 0:
        classification = CLASSIFICATION_SUPPORT
    elif upper < 0:
        classification = CLASSIFICATION_CONTRARY
    else:
        classification = CLASSIFICATION_INDETERMINATE

    return {
        **base,
        "valid": True,
        "classification": classification,
        "per_seed_d_s": per_seed,
        "dbar": dbar,
        "sample_sd": sample_sd,
        "df": df,
        "t_quantile": quantile,
        "standard_error": standard_error,
        "ci_lower": lower,
        "ci_upper": upper,
        "provenance": prov.EVIDENCE_COMPUTED,
    }


# --------------------------------------------------------------------------- #
# reading a completed pair off disk
# --------------------------------------------------------------------------- #
AUX_UPDATE_PREFIX = "ppo_aux_update_"
CONTROL_UPDATE_PREFIX = "ppo_update_"

# The kinds an update-indexed pack may legitimately carry when it is used as a
# point on a paired curve.  A recovery pack is training state rather than an
# evaluated checkpoint, and ``final_alias`` is a copy of the last weights under
# a different name; neither is a member of a matched update set, so finding one
# at an update path means the pack is not what the analysis assumed.
PAIRABLE_CHECKPOINT_KINDS = (
    prov.CHECKPOINT_KIND_INITIAL_SNAPSHOT,
    prov.CHECKPOINT_KIND_DIAGNOSTIC_FULL_SNAPSHOT,
    prov.CHECKPOINT_KIND_PERIODIC_MODEL_ONLY,
    prov.CHECKPOINT_KIND_TERMINAL_FULL_SNAPSHOT,
)


def checkpoint_path(run_dir: Path, arm: str, update: int) -> Path:
    prefix = AUX_UPDATE_PREFIX if arm == "aux" else CONTROL_UPDATE_PREFIX
    return Path(run_dir) / f"{prefix}{int(update)}.pt"


def read_packed_evaluation(path: Path, *, arm: str, update: int) -> dict[str, Any]:
    """Per-seed deterministic evaluation as the checkpoint itself recorded it.

    The packed ``extra.eval_summary`` is primary evidence: it is what that
    exact weight state scored.  A pack without it fails closed rather than
    being substituted with an on-policy rollout mean from the progress
    journal, which measures a different thing entirely.
    """
    import torch

    path = Path(path)
    if not path.is_file():
        raise PairAnalysisError(f"{arm} checkpoint for update {update} is missing: {path}")
    try:
        packed = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - an unreadable pack is not analysable
        raise PairAnalysisError(f"{arm} checkpoint for update {update} is unreadable: {exc}") from exc

    extra = packed.get("extra") or {}
    if not isinstance(extra, dict):
        raise PairAnalysisError(
            f"{arm} checkpoint for update {update} extra is not a dict"
        )
    packed_update = extra.get("update")
    if isinstance(packed_update, bool) or not isinstance(packed_update, int) or packed_update != update:
        raise PairAnalysisError(
            f"{arm} checkpoint for update {update} records extra.update {packed_update!r}, "
            f"which is not the exact requested update"
        )
    summary = extra.get("eval_summary") or {}
    if not isinstance(summary, dict):
        raise PairAnalysisError(
            f"{arm} checkpoint for update {update} eval_summary is not a dict"
        )
    per_seed = summary.get("per_seed")
    if not isinstance(per_seed, dict) or not per_seed:
        raise PairAnalysisError(
            f"{arm} checkpoint for update {update} records no per-seed elapsed"
        )
    kind = extra.get("checkpoint_kind")
    if kind not in PAIRABLE_CHECKPOINT_KINDS:
        raise PairAnalysisError(
            f"{arm} checkpoint for update {update} carries checkpoint_kind {kind!r}; "
            f"a paired update must be one of {PAIRABLE_CHECKPOINT_KINDS}"
        )
    elapsed: dict[int, float] = {}
    for key, value in per_seed.items():
        seed = _require_index(key, label=f"{arm} per_seed key at update {update}")
        if seed in elapsed:
            raise PairAnalysisError(
                f"{arm} checkpoint for update {update} repeats seed {seed}"
            )
        elapsed[seed] = _require_elapsed(
            value, label=f"{arm} elapsed for update {update} seed {seed}",
        )
    if "per_seed_censored" not in summary:
        censored: dict[int, bool] | None = None
    else:
        raw_censored = summary["per_seed_censored"]
        if not isinstance(raw_censored, dict):
            raise PairAnalysisError(
                f"{arm} checkpoint for update {update} per_seed_censored must be a dict; "
                f"got {type(raw_censored).__name__}"
            )
        censored = {}
        for key, value in raw_censored.items():
            seed = _require_index(
                key, label=f"{arm} per_seed_censored key at update {update}",
            )
            if seed in censored:
                raise PairAnalysisError(
                    f"{arm} checkpoint for update {update} repeats censored seed {seed}"
                )
            if type(value) is not bool:
                raise PairAnalysisError(
                    f"{arm} checkpoint for update {update} per_seed_censored[{key!r}] "
                    f"must be a real bool; got {value!r}"
                )
            censored[seed] = value
        if set(censored) != set(elapsed):
            raise PairAnalysisError(
                f"{arm} checkpoint for update {update} per_seed_censored keys "
                f"{sorted(censored)} do not match per_seed keys {sorted(elapsed)}"
            )
    return {
        "elapsed": elapsed,
        "censored": censored,
        "checkpoint_kind": kind,
        "update": update,
    }


def _source_entry(run_dir: Path, arm: str, path: Path, update: int | None) -> dict[str, Any]:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    return {
        "arm": arm,
        "path": Path(path).relative_to(Path(run_dir)).as_posix(),
        "absolute_path": str(Path(path).resolve()),
        "role": prov.artifact_role(Path(path).name),
        "update": update,
        "sha256": file_sha256(path),
        "sha256_status": prov.EVIDENCE_COMPUTED,
    }


def _read_report(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "report.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PairAnalysisError(f"report.json is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise PairAnalysisError("report.json is malformed: expected a JSON object")
    return payload


def _snapshot_at(report: dict[str, Any] | None, update: int) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    for key in ("aux_arm", "ppo_arm", "arm"):
        arm_block = report.get(key)
        if not isinstance(arm_block, dict):
            continue
        for snapshot in arm_block.get("snapshots") or []:
            if isinstance(snapshot, dict) and snapshot.get("update") == update:
                return snapshot
    return None


def _teacher_secondary(
    aux_report: dict[str, Any] | None, control_report: dict[str, Any] | None, update: int,
) -> dict[str, Any]:
    """Terminal held-teacher agreement/KL for both arms, or explicit nulls.

    Agreement is "higher is closer to the teacher" and KL is "lower is closer
    to the teacher"; they are reported separately and never collapsed into
    one direction.
    """
    def _one(report: dict[str, Any] | None) -> dict[str, Any]:
        diagnostics = (_snapshot_at(report, update) or {}).get("teacher_diagnostics") or {}
        return {
            "agreement": diagnostics.get("agreement"),
            "teacher_to_student_kl": diagnostics.get("teacher_to_student_kl"),
        }

    aux_side, control_side = _one(aux_report), _one(control_report)
    complete = all(
        side[key] is not None
        for side in (aux_side, control_side)
        for key in ("agreement", "teacher_to_student_kl")
    )
    return {
        "update": update,
        "aux": aux_side,
        "control": control_side,
        "status": prov.EVIDENCE_OBSERVED if complete else prov.EVIDENCE_UNAVAILABLE,
        "note": (
            None if complete
            else "terminal teacher diagnostics are absent; never back-fill them "
                 "from survival or from another run"
        ),
    }


def _survival_summary(
    elapsed: dict[int, float], censored: dict[int, bool] | None, seeds: list[int],
) -> dict[str, Any]:
    """Terminal survival over the requested seeds, or an explicit non-summary.

    A checkpoint that never evaluated one of the requested seeds cannot be
    summarized over them.  That is reported as an unavailable record naming the
    absent seeds -- never a partial mean over whichever seeds happened to be
    present, and never a raised ``KeyError`` from the lookup.
    """
    missing = [seed for seed in seeds if seed not in elapsed]
    if missing:
        return {
            "status": prov.EVIDENCE_UNAVAILABLE,
            "missing_seeds": missing,
            "mean": None,
            "median": None,
            "pstdev": None,
            "n": None,
            "censored_count": None,
            "per_seed": {},
        }
    values = [elapsed[seed] for seed in seeds]
    return {
        "status": prov.EVIDENCE_OBSERVED,
        "missing_seeds": [],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values),
        "n": len(values),
            "censored_count": (
                None if censored is None else sum(1 for seed in seeds if censored.get(seed))
            ),
        "per_seed": {str(seed): elapsed[seed] for seed in seeds},
    }


def analyze_pair(
    *,
    aux_run_dir: Path,
    control_run_dir: Path,
    updates: list[int],
    eval_seeds: list[int],
    terminal_update: int | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Analyse one completed aux + control pair against an explicit ``U``.

    Nothing here is inferred from the experiment: ``updates`` and
    ``eval_seeds`` are supplied by the caller, and every artifact actually
    read is recorded with its SHA-256 so the analysis can be re-derived from
    exactly the same bytes.
    """
    aux_run_dir, control_run_dir = Path(aux_run_dir), Path(control_run_dir)
    updates = _exact_int_list(updates, label="updates")
    eval_seeds = _exact_int_list(eval_seeds, label="eval_seeds")
    confidence = require_confidence(confidence)
    if not updates:
        raise PairAnalysisError("an explicit matched update set U is required")
    terminal = (
        max(updates) if terminal_update is None
        else _exact_int(terminal_update, label="terminal_update")
    )

    sources: list[dict[str, Any]] = []
    read: dict[str, dict[int, dict[str, Any]]] = {"aux": {}, "control": {}}
    for arm, run_dir in (("aux", aux_run_dir), ("control", control_run_dir)):
        for update in sorted({*updates, terminal}):
            path = checkpoint_path(run_dir, arm, update)
            read[arm][update] = read_packed_evaluation(path, arm=arm, update=update)
            sources.append(_source_entry(run_dir, arm, path, update))

    # Two arms are only matched at an update if they wrote the SAME kind of
    # artifact there.  A diagnostic full snapshot against a model-only periodic
    # checkpoint is a schedule difference, not a treatment difference.
    for update in sorted({*updates, terminal}):
        aux_kind = read["aux"][update]["checkpoint_kind"]
        control_kind = read["control"][update]["checkpoint_kind"]
        if aux_kind != control_kind:
            raise PairAnalysisError(
                f"update {update} is not matched: aux wrote checkpoint_kind "
                f"{aux_kind!r} while control wrote {control_kind!r}"
            )

    def _censor_axis(arm: str, chosen: list[int]) -> dict[int, dict[int, bool]] | None:
        maps = [read[arm][item]["censored"] for item in chosen]
        if any(item is None for item in maps):
            return None
        return {item: read[arm][item]["censored"] for item in chosen}

    primary = paired_curve_difference(
        updates=updates,
        eval_seeds=eval_seeds,
        aux_elapsed={u: read["aux"][u]["elapsed"] for u in updates},
        control_elapsed={u: read["control"][u]["elapsed"] for u in updates},
        aux_censored=_censor_axis("aux", updates),
        control_censored=_censor_axis("control", updates),
        confidence=confidence,
    )

    terminal_paired = paired_curve_difference(
        updates=[terminal],
        eval_seeds=eval_seeds,
        aux_elapsed={terminal: read["aux"][terminal]["elapsed"]},
        control_elapsed={terminal: read["control"][terminal]["elapsed"]},
        aux_censored=_censor_axis("aux", [terminal]),
        control_censored=_censor_axis("control", [terminal]),
        confidence=confidence,
    )

    aux_report, control_report = _read_report(aux_run_dir), _read_report(control_run_dir)
    for arm, run_dir, report in (
        ("aux", aux_run_dir, aux_report), ("control", control_run_dir, control_report),
    ):
        if report is not None:
            sources.append(_source_entry(run_dir, arm, Path(run_dir) / "report.json", None))

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "written_hkt": prov.now_hkt().isoformat(),
        "configuration": {
            "aux_run_dir": str(aux_run_dir.resolve()),
            "control_run_dir": str(control_run_dir.resolve()),
            "updates": updates,
            "eval_seeds": eval_seeds,
            "terminal_update": terminal,
            "confidence": confidence,
        },
        "primary": primary,
        "secondary": {
            "terminal_survival": {
                "update": terminal,
                "paired": terminal_paired,
                "aux": _survival_summary(
                    read["aux"][terminal]["elapsed"], read["aux"][terminal]["censored"], eval_seeds,
                ),
                "control": _survival_summary(
                    read["control"][terminal]["elapsed"],
                    read["control"][terminal]["censored"],
                    eval_seeds,
                ),
                "note": "one checkpoint, never the primary endpoint",
            },
            "terminal_teacher": _teacher_secondary(aux_report, control_report, terminal),
        },
        "checkpoint_kinds": {
            arm: {str(u): read[arm][u]["checkpoint_kind"] for u in sorted(read[arm])}
            for arm in read
        },
        "source_artifacts": sources,
        "evidence_provenance": {
            "primary": primary["provenance"],
            "terminal_survival": terminal_paired["provenance"],
            "terminal_teacher": _teacher_secondary(aux_report, control_report, terminal)["status"],
            "source_artifacts": prov.EVIDENCE_COMPUTED,
            "configuration": prov.EVIDENCE_OBSERVED,
        },
        # This tool reports; it never authorizes an experimental checkpoint
        # for any production use.
        "no_promotion": True,
    }


# --------------------------------------------------------------------------- #
# CLI -- the matched update set and seed window are never defaulted
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aux-run-dir", type=Path, required=True)
    ap.add_argument("--control-run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="path of the analysis.json to write")
    ap.add_argument(
        "--config", type=Path,
        help="JSON file carrying 'updates' and/or 'eval_seeds' for this preregistration",
    )
    ap.add_argument(
        "--update", type=int, action="append", dest="update", metavar="N",
        help="one member of the matched update set U (repeatable)",
    )
    ap.add_argument("--updates-start", type=int, help="first member of the matched update set U")
    ap.add_argument("--updates-stop", type=int, help="last (inclusive) member of U")
    ap.add_argument("--updates-step", type=int, help="stride between members of U")
    ap.add_argument(
        "--eval-seed", type=int, action="append", dest="eval_seed", metavar="S",
        help="one evaluation seed (repeatable)",
    )
    ap.add_argument("--eval-seed-start", type=int, help="first evaluation seed")
    ap.add_argument("--eval-seed-count", type=int, help="number of consecutive evaluation seeds")
    ap.add_argument("--terminal-update", type=int, help="update used for the secondary outcomes")
    ap.add_argument("--confidence", type=float, default=0.95)
    return ap


def _config(args: argparse.Namespace) -> dict[str, Any]:
    path = getattr(args, "config", None)
    if path is None:
        return {}
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PairAnalysisError(f"config file is unreadable: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PairAnalysisError("config file must contain a JSON object")
    return loaded


def resolve_updates(args: argparse.Namespace) -> list[int]:
    """The matched update set U, which must be stated explicitly.

    There is no built-in U.  Which checkpoints are matched is a property of a
    specific preregistration, and silently defaulting to some previous
    experiment's set is exactly how an analysis ends up answering a question
    nobody asked.
    """
    listed = getattr(args, "update", None)
    range_requested = any(
        getattr(args, name) is not None
        for name in ("updates_start", "updates_stop", "updates_step")
    )
    if listed:
        updates = _exact_int_list(listed, label="updates")
    elif range_requested:
        if args.updates_start is None or args.updates_stop is None or args.updates_step is None:
            raise PairAnalysisError(
                "--updates-start, --updates-stop, and --updates-step are all required "
                "once any update-range flag is present"
            )
        start = _exact_int(args.updates_start, label="updates-start")
        stop = _exact_int(args.updates_stop, label="updates-stop")
        step = _exact_int(args.updates_step, label="updates-step")
        if step <= 0:
            raise PairAnalysisError("--updates-step must be positive")
        updates = list(range(start, stop + 1, step))
    else:
        config = _config(args)
        raw = config.get("updates")
        if "updates" not in config or (isinstance(raw, list) and not raw):
            raise PairAnalysisError(
                "an explicit matched update set U is required: pass --update/"
                "--updates-start with --updates-stop and --updates-step, or a --config "
                "file carrying 'updates'"
            )
        updates = _exact_int_list(raw, label="updates")
    if len(set(updates)) != len(updates):
        raise PairAnalysisError("the matched update set U contains duplicates")
    return sorted(updates)


def resolve_eval_seeds(args: argparse.Namespace) -> list[int]:
    """The evaluation seed window, which must also be stated explicitly."""
    listed = getattr(args, "eval_seed", None)
    range_requested = args.eval_seed_start is not None or args.eval_seed_count is not None
    if listed:
        seeds = _exact_int_list(listed, label="eval_seeds")
    elif range_requested:
        if args.eval_seed_start is None or args.eval_seed_count is None:
            raise PairAnalysisError(
                "--eval-seed-start and --eval-seed-count are both required "
                "once any eval-range flag is present"
            )
        start = _exact_int(args.eval_seed_start, label="eval-seed-start")
        count = _exact_int(args.eval_seed_count, label="eval-seed-count")
        if count <= 0:
            raise PairAnalysisError("--eval-seed-count must be positive")
        seeds = list(range(start, start + count))
    else:
        config = _config(args)
        raw = config.get("eval_seeds")
        if "eval_seeds" not in config or (isinstance(raw, list) and not raw):
            raise PairAnalysisError(
                "an explicit evaluation seed window is required: pass --eval-seed/"
                "--eval-seed-start with --eval-seed-count, or a --config file carrying "
                "'eval_seeds'"
            )
        seeds = _exact_int_list(raw, label="eval_seeds")
    if len(set(seeds)) != len(seeds):
        raise PairAnalysisError("the evaluation seed window contains duplicates")
    return sorted(seeds)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_pair(
        aux_run_dir=args.aux_run_dir,
        control_run_dir=args.control_run_dir,
        updates=resolve_updates(args),
        eval_seeds=resolve_eval_seeds(args),
        terminal_update=args.terminal_update,
        confidence=args.confidence,
    )
    prov.atomic_write_json(Path(args.out), analysis)
    print(
        json.dumps(
            {
                "classification": analysis["primary"]["classification"],
                "dbar": analysis["primary"]["dbar"],
                "ci_lower": analysis["primary"]["ci_lower"],
                "ci_upper": analysis["primary"]["ci_upper"],
                "out": str(Path(args.out).resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
