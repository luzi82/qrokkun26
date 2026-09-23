"""Contracts for the shared aux-vs-control pair analysis finalizer.

The pre-registered primary estimator is a paired Student-t over PER-SEED
curve-averaged differences.  It is implemented ONCE here and consumed by both
arms' completed run directories, and it never assumes any particular
experiment's matched update set or evaluation seed window: those must be
stated explicitly by the caller.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qrokkun_ai.v5.tools import phase3_pair_analysis as pair  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. the Student-t quantile is computed, never a hand-written constant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "df, expected",
    [
        (1, 12.706204736174698),
        (2, 4.302652729911275),
        (10, 2.228138851986273),
        (29, 2.045229642132703),
        (100, 1.9839715184496334),
    ],
)
def test_student_t_two_sided_95_quantiles_match_known_values(df: int, expected: float) -> None:
    assert pair.student_t_ppf(0.975, df) == pytest.approx(expected, abs=1e-9)


def test_student_t_quantile_is_symmetric_and_zero_at_the_median() -> None:
    assert pair.student_t_ppf(0.5, 7) == pytest.approx(0.0, abs=1e-12)
    assert pair.student_t_ppf(0.025, 29) == pytest.approx(-pair.student_t_ppf(0.975, 29), abs=1e-12)


def test_student_t_quantile_approaches_the_normal_for_large_df() -> None:
    assert pair.student_t_ppf(0.975, 2_000_000) == pytest.approx(1.959963984540054, abs=1e-5)


@pytest.mark.parametrize("p, df", [(0.0, 5), (1.0, 5), (0.5, 0), (0.5, -1)])
def test_student_t_quantile_rejects_an_undefined_request(p: float, df: int) -> None:
    with pytest.raises(ValueError):
        pair.student_t_ppf(p, df)


# --------------------------------------------------------------------------- #
# 2. the primary estimator: per-seed curve differences, paired t, CI, verdict
# --------------------------------------------------------------------------- #
def _curves(values: dict[int, dict[int, float]]) -> dict[int, dict[int, float]]:
    return values


def test_primary_estimator_reports_per_seed_d_s_dbar_sd_df_and_ci() -> None:
    result = pair.paired_curve_difference(
        updates=[10, 20],
        eval_seeds=[1, 2, 3],
        aux_elapsed=_curves({10: {1: 10.0, 2: 12.0, 3: 14.0}, 20: {1: 12.0, 2: 14.0, 3: 16.0}}),
        control_elapsed=_curves({10: {1: 9.0, 2: 12.0, 3: 15.0}, 20: {1: 11.0, 2: 14.0, 3: 17.0}}),
    )

    assert result["valid"] is True
    assert result["n_updates"] == 2
    assert result["per_seed_d_s"] == {"1": 1.0, "2": 0.0, "3": -1.0}
    assert result["dbar"] == pytest.approx(0.0)
    assert result["sample_sd"] == pytest.approx(1.0)
    assert result["n"] == 3
    assert result["df"] == 2
    assert result["confidence"] == 0.95
    assert result["t_quantile"] == pytest.approx(4.302652729911275, abs=1e-9)
    assert result["standard_error"] == pytest.approx(1.0 / math.sqrt(3))
    margin = 4.302652729911275 / math.sqrt(3)
    assert result["ci_lower"] == pytest.approx(-margin)
    assert result["ci_upper"] == pytest.approx(margin)
    assert result["classification"] == "indeterminate"
    assert result["provenance"] == "computed"


@pytest.mark.parametrize(
    "aux_values, classification",
    [
        ({1: 11.0, 2: 11.1, 3: 10.9}, "support"),
        ({1: 9.0, 2: 8.9, 3: 9.1}, "contrary"),
        ({1: 12.0, 2: 10.0, 3: 8.0}, "indeterminate"),
    ],
)
def test_primary_classification_follows_the_confidence_interval_only(
    aux_values: dict[int, float], classification: str,
) -> None:
    result = pair.paired_curve_difference(
        updates=[10],
        eval_seeds=[1, 2, 3],
        aux_elapsed={10: aux_values},
        control_elapsed={10: {1: 10.0, 2: 10.0, 3: 10.0}},
    )
    assert result["classification"] == classification
    if classification == "support":
        assert result["ci_lower"] > 0
    elif classification == "contrary":
        assert result["ci_upper"] < 0
    else:
        assert result["ci_lower"] <= 0 <= result["ci_upper"]


def test_d_s_is_the_curve_average_over_u_not_a_single_update_difference() -> None:
    result = pair.paired_curve_difference(
        updates=[1, 2, 3, 4],
        eval_seeds=[7, 8],
        aux_elapsed={
            1: {7: 10.0, 8: 5.0}, 2: {7: 20.0, 8: 5.0},
            3: {7: 30.0, 8: 5.0}, 4: {7: 40.0, 8: 5.0},
        },
        control_elapsed={
            1: {7: 10.0, 8: 5.0}, 2: {7: 10.0, 8: 5.0},
            3: {7: 10.0, 8: 5.0}, 4: {7: 10.0, 8: 5.0},
        },
    )
    # seed 7 averages (0 + 10 + 20 + 30) / 4 rather than reporting any single u
    assert result["per_seed_d_s"]["7"] == pytest.approx(15.0)
    assert result["per_seed_d_s"]["8"] == pytest.approx(0.0)
    assert result["n"] == 2 and result["n_updates"] == 4


@pytest.mark.parametrize(
    "missing_arm, expected_reason",
    [("aux", "aux missing"), ("control", "control missing")],
)
def test_a_missing_cell_makes_the_whole_pair_invalid(
    missing_arm: str, expected_reason: str,
) -> None:
    """Never drop a seed, never shrink U: an incomplete grid is invalid."""
    aux_elapsed = {10: {1: 1.0, 2: 2.0}, 20: {1: 1.0, 2: 2.0}}
    control_elapsed = {10: {1: 1.0, 2: 2.0}, 20: {1: 1.0, 2: 2.0}}
    if missing_arm == "aux":
        del aux_elapsed[20][2]
    else:
        del control_elapsed[20][2]

    result = pair.paired_curve_difference(
        updates=[10, 20], eval_seeds=[1, 2],
        aux_elapsed=aux_elapsed, control_elapsed=control_elapsed,
    )
    assert result["valid"] is False
    assert result["classification"] == "invalid"
    assert result["dbar"] is None
    assert result["ci_lower"] is None
    assert any(expected_reason in reason for reason in result["reasons"])
    assert result["provenance"] == "unavailable"


def test_a_single_seed_cannot_support_a_paired_interval() -> None:
    result = pair.paired_curve_difference(
        updates=[10], eval_seeds=[1], aux_elapsed={10: {1: 5.0}}, control_elapsed={10: {1: 1.0}},
    )
    assert result["valid"] is False
    assert result["classification"] == "invalid"
    assert any("at least two" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.5, 1.5, 100.0, float("nan"), float("inf"), None, "0.95", True],
)
def test_estimator_rejects_a_confidence_level_outside_the_open_unit_interval(
    confidence: Any,
) -> None:
    """A confidence level of 0, 1 or outside them has no two-sided Student-t
    quantile, so no interval exists.  The estimator must say so rather than
    reporting bounds derived from an undefined quantile."""
    result = pair.paired_curve_difference(
        updates=[10],
        eval_seeds=[1, 2, 3],
        aux_elapsed={10: {1: 11.0, 2: 11.0, 3: 11.0}},
        control_elapsed={10: {1: 10.0, 2: 10.0, 3: 10.0}},
        confidence=confidence,
    )
    assert result["valid"] is False
    assert result["classification"] == "invalid"
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert result["t_quantile"] is None
    assert result["dbar"] is None
    assert result["provenance"] == "unavailable"
    assert any("confidence" in reason for reason in result["reasons"])


def test_estimator_accepts_a_non_default_confidence_level() -> None:
    result = pair.paired_curve_difference(
        updates=[10],
        eval_seeds=[1, 2, 3],
        aux_elapsed={10: {1: 11.0, 2: 12.0, 3: 13.0}},
        control_elapsed={10: {1: 10.0, 2: 10.0, 3: 10.0}},
        confidence=0.99,
    )
    assert result["valid"] is True
    assert result["confidence"] == 0.99
    assert result["t_quantile"] == pytest.approx(pair.student_t_ppf(0.995, 2), abs=1e-9)


@pytest.mark.parametrize("arm", ["aux", "control"])
@pytest.mark.parametrize(
    "bad",
    [True, False, float("nan"), float("inf"), float("-inf"), "1.0", 1 + 0j],
)
def test_estimator_rejects_bool_nonfinite_and_malformed_elapsed_cells(
    arm: str, bad: Any,
) -> None:
    """A cell that is not a finite real number invalidates the pair.

    ``True`` must not become elapsed ``1.0``, and ``nan`` / ``inf`` must not
    escape into ``statistics`` as an uncaught error or a computed ``dbar``.
    """
    aux = {10: {1: 2.0, 2: 2.0}}
    control = {10: {1: 1.0, 2: 1.0}}
    (aux if arm == "aux" else control)[10][1] = bad
    result = pair.paired_curve_difference(
        updates=[10], eval_seeds=[1, 2], aux_elapsed=aux, control_elapsed=control,
    )
    assert result["valid"] is False
    assert result["classification"] == "invalid"
    assert result["dbar"] is None
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert result["provenance"] == "unavailable"
    assert result["reasons"]


@pytest.mark.parametrize(
    "cells",
    [
        {True: 1.0, 2: 2.0},
        {1.0: 1.0, 2: 2.0},
        {1: 1.0, 2: 2.0, "01": 9.0},
        {1: 1.0, 2: 2.0, "+1": 9.0},
        {1: 1.0, 2: 2.0, "1.0": 9.0},
        {1: 1.0, 2: 2.0, " 1": 9.0},
    ],
)
def test_estimator_rejects_malformed_seed_keys(cells: dict[Any, float]) -> None:
    """Seed keys are exact ints or canonical decimal strings. A bool, a float,
    or a non-canonical digit string must not be read as one of the seeds."""
    result = pair.paired_curve_difference(
        updates=[10], eval_seeds=[1, 2],
        aux_elapsed={10: cells},
        control_elapsed={10: {1: 1.0, 2: 1.0}},
    )
    assert result["valid"] is False
    assert result["classification"] == "invalid"
    assert result["dbar"] is None
    assert result["provenance"] == "unavailable"
    assert any("malformed" in reason for reason in result["reasons"])


def test_estimator_reports_censoring_per_arm_without_changing_the_estimate() -> None:
    kwargs: dict[str, Any] = dict(
        updates=[10, 20],
        eval_seeds=[1, 2],
        aux_elapsed={10: {1: 1.0, 2: 2.0}, 20: {1: 1.0, 2: 2.0}},
        control_elapsed={10: {1: 1.0, 2: 1.0}, 20: {1: 1.0, 2: 1.0}},
    )
    plain = pair.paired_curve_difference(**kwargs)
    censored = pair.paired_curve_difference(
        **kwargs,
        aux_censored={10: {1: False, 2: True}, 20: {1: False, 2: False}},
        control_censored={10: {1: False, 2: False}, 20: {1: False, 2: False}},
    )

    assert censored["dbar"] == plain["dbar"]
    assert censored["censoring"]["aux"]["count"] == 1
    assert censored["censoring"]["control"]["count"] == 0
    assert censored["censoring"]["aux"]["cells"] == [{"update": 10, "seed": 2}]
    assert censored["censoring"]["total_cells"] == 4
    assert censored["censoring"]["structurally_different"] is True
    assert plain["censoring"]["status"] == "unavailable"
    assert plain["censoring"]["structurally_different"] is False


def test_structurally_different_compares_ordered_cells_not_only_counts() -> None:
    """The same number of censored cells at different update/seed positions is
    a different paired grid. The flag follows the ordered cell identities."""
    kwargs: dict[str, Any] = dict(
        updates=[10, 20],
        eval_seeds=[1, 2],
        aux_elapsed={10: {1: 1.0, 2: 2.0}, 20: {1: 1.0, 2: 2.0}},
        control_elapsed={10: {1: 1.0, 2: 1.0}, 20: {1: 1.0, 2: 1.0}},
    )
    swapped = pair.paired_curve_difference(
        **kwargs,
        aux_censored={10: {1: True, 2: False}, 20: {1: False, 2: False}},
        control_censored={10: {1: False, 2: False}, 20: {1: False, 2: True}},
    )
    assert swapped["censoring"]["aux"]["count"] == 1
    assert swapped["censoring"]["control"]["count"] == 1
    assert swapped["censoring"]["aux"]["cells"] == [{"update": 10, "seed": 1}]
    assert swapped["censoring"]["control"]["cells"] == [{"update": 20, "seed": 2}]
    assert swapped["censoring"]["structurally_different"] is True
    assert swapped["dbar"] == pair.paired_curve_difference(**kwargs)["dbar"]

    same = pair.paired_curve_difference(
        **kwargs,
        aux_censored={10: {1: True, 2: False}, 20: {1: False, 2: False}},
        control_censored={10: {1: True, 2: False}, 20: {1: False, 2: False}},
    )
    assert same["censoring"]["aux"]["cells"] == same["censoring"]["control"]["cells"]
    assert same["censoring"]["structurally_different"] is False


# --------------------------------------------------------------------------- #
# 3. reading a completed pair off disk
# --------------------------------------------------------------------------- #
class _Omitted:
    """Sentinel for a packed field the test deliberately leaves unset."""


_OMITTED = _Omitted()


def _write_pack(
    path: Path, *, update: int, per_seed: dict[int, float],
    censored: dict[int, bool] | None = None, kind: str = "periodic_model_only",
    include_eval: bool = True, per_seed_payload: Any = None,
    censored_payload: Any = _OMITTED, include_censored: bool = True,
    packed_update: Any = _OMITTED,
) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import save_player_checkpoint
    from qrokkun_ai.v5.agents.player_ranked_topk import PlayerRankedTopK

    torch.manual_seed(0)
    extra: dict[str, Any] = {"checkpoint_kind": kind}
    if packed_update is not _OMITTED:
        extra["update"] = packed_update
    else:
        extra["update"] = update
    if include_eval:
        summary: dict[str, Any] = {
            "per_seed": (
                per_seed_payload if per_seed_payload is not None
                else {str(s): float(v) for s, v in per_seed.items()}
            ),
        }
        if include_censored:
            summary["per_seed_censored"] = (
                censored_payload if censored_payload is not _OMITTED
                else {str(s): bool((censored or {}).get(s, False)) for s in per_seed}
            )
        extra["eval_summary"] = summary
    save_player_checkpoint(
        PlayerRankedTopK(top_k=4, hidden=8), path,
        source_tool="tests", experimental=True, production_compatible=False, extra=extra,
    )


def _completed_pair(
    tmp_path: Path, *, updates: list[int] = [10, 20], seeds: list[int] = [1, 2, 3],
    aux_bonus: float = 1.0, aux_censored_cell: tuple[int, int] | None = None,
    terminal_teacher: bool = True,
) -> tuple[Path, Path]:
    aux_dir, control_dir = tmp_path / "aux", tmp_path / "control"
    aux_dir.mkdir()
    control_dir.mkdir()
    for update in updates:
        _write_pack(
            aux_dir / f"ppo_aux_update_{update}.pt", update=update,
            per_seed={s: 10.0 + s + aux_bonus for s in seeds},
            censored={
                s: aux_censored_cell == (update, s) for s in seeds
            },
        )
        _write_pack(
            control_dir / f"ppo_update_{update}.pt", update=update,
            per_seed={s: 10.0 + s for s in seeds},
        )
    terminal = max(updates)
    for directory, arm in ((aux_dir, "aux"), (control_dir, "control")):
        snapshot: dict[str, Any] = {"update": terminal}
        if terminal_teacher:
            snapshot["teacher_diagnostics"] = {
                "agreement": 0.66 if arm == "aux" else 0.13,
                "teacher_to_student_kl": 0.24 if arm == "aux" else 0.87,
            }
        (directory / "report.json").write_text(
            json.dumps({"status": "max_updates", "arm": arm, f"{arm}_arm": {"snapshots": [snapshot]}}
                       if arm == "aux" else
                       {"status": "max_updates", "arm": arm, "ppo_arm": {"snapshots": [snapshot]}},
                       indent=2, sort_keys=True)
        )
    return aux_dir, control_dir


def test_analyze_pair_reads_per_seed_elapsed_from_the_checkpoint_packs(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3],
    )
    primary = analysis["primary"]
    assert primary["valid"] is True
    # Every seed is exactly +1.0 s, so the interval is degenerate at 1.0.
    assert primary["per_seed_d_s"] == {"1": 1.0, "2": 1.0, "3": 1.0}
    assert primary["dbar"] == pytest.approx(1.0)
    assert primary["sample_sd"] == pytest.approx(0.0)
    assert primary["classification"] == "support"


def test_analyze_pair_fails_closed_when_a_pack_has_no_per_seed_elapsed(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    _write_pack(
        aux_dir / "ppo_aux_update_20.pt", update=20, per_seed={1: 1.0}, include_eval=False,
    )
    with pytest.raises(pair.PairAnalysisError, match="per-seed"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


def test_analyze_pair_fails_closed_on_a_missing_checkpoint(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    with pytest.raises(pair.PairAnalysisError, match="missing"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20, 30], eval_seeds=[1, 2, 3],
        )


def test_analyze_pair_fails_closed_when_a_pack_declares_no_checkpoint_kind(
    tmp_path: Path,
) -> None:
    """A pack that does not say what it is cannot be matched against another
    arm's pack: the reader would have to infer the kind from its absence."""
    aux_dir, control_dir = _completed_pair(tmp_path)
    _write_pack(
        aux_dir / "ppo_aux_update_20.pt", update=20,
        per_seed={1: 11.0, 2: 12.0, 3: 13.0}, kind=None,
    )
    with pytest.raises(pair.PairAnalysisError, match="checkpoint_kind"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


@pytest.mark.parametrize(
    "kind", ["final_alias", "recovery_current", "recovery_archive", "made_up_kind", ""],
)
def test_analyze_pair_fails_closed_on_an_unsupported_checkpoint_kind(
    kind: str, tmp_path: Path,
) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    _write_pack(
        aux_dir / "ppo_aux_update_20.pt", update=20,
        per_seed={1: 11.0, 2: 12.0, 3: 13.0}, kind=kind,
    )
    with pytest.raises(pair.PairAnalysisError, match="checkpoint_kind"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


@pytest.mark.parametrize("kind", list(pair.PAIRABLE_CHECKPOINT_KINDS))
def test_analyze_pair_accepts_every_pairable_kind_when_both_arms_agree(
    kind: str, tmp_path: Path,
) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    for path, prefix in (
        (aux_dir / "ppo_aux_update_20.pt", "aux"), (control_dir / "ppo_update_20.pt", "control"),
    ):
        _write_pack(
            path, update=20,
            per_seed={s: 10.0 + s + (1.0 if prefix == "aux" else 0.0) for s in (1, 2, 3)},
            kind=kind,
        )
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3],
    )
    assert analysis["primary"]["valid"] is True
    assert analysis["checkpoint_kinds"]["aux"]["20"] == kind


def test_analyze_pair_refuses_to_pair_two_arms_whose_kinds_disagree(tmp_path: Path) -> None:
    """A diagnostic full snapshot and a model-only periodic checkpoint are not
    the same kind of evidence; pairing them silently would compare artifacts
    produced by two different schedules."""
    aux_dir, control_dir = _completed_pair(tmp_path)
    _write_pack(
        aux_dir / "ppo_aux_update_20.pt", update=20,
        per_seed={1: 11.0, 2: 12.0, 3: 13.0}, kind="diagnostic_full_snapshot",
    )
    with pytest.raises(pair.PairAnalysisError, match="kind"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


def test_terminal_checkpoint_missing_a_requested_seed_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    """A terminal pack that never evaluated a requested seed must produce a
    controlled invalid secondary, never a raw ``KeyError`` out of the
    survival summary."""
    aux_dir, control_dir = _completed_pair(tmp_path)
    for path, bonus in ((aux_dir / "ppo_aux_update_30.pt", 1.0),
                        (control_dir / "ppo_update_30.pt", 0.0)):
        # Seed 3 was never evaluated at the terminal update.
        _write_pack(path, update=30, per_seed={1: 10.0 + bonus, 2: 11.0 + bonus})

    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3], terminal_update=30,
    )

    survival = analysis["secondary"]["terminal_survival"]
    for side in ("aux", "control"):
        assert survival[side]["status"] == "unavailable"
        assert survival[side]["missing_seeds"] == [3]
        assert survival[side]["mean"] is None
        assert survival[side]["median"] is None
        assert survival[side]["per_seed"] == {}
    assert survival["paired"]["valid"] is False
    assert survival["paired"]["classification"] == "invalid"
    assert analysis["evidence_provenance"]["terminal_survival"] == "unavailable"
    # A failed secondary never touches the primary endpoint.
    assert analysis["primary"]["valid"] is True
    assert analysis["primary"]["classification"] == "support"


def test_terminal_survival_declares_itself_observed_when_every_seed_is_present(
    tmp_path: Path,
) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3], terminal_update=20,
    )
    survival = analysis["secondary"]["terminal_survival"]
    assert survival["aux"]["status"] == "observed"
    assert survival["aux"]["missing_seeds"] == []
    assert survival["aux"]["n"] == 3


def test_analysis_records_censoring_and_the_hash_of_every_artifact_it_read(
    tmp_path: Path,
) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    aux_dir, control_dir = _completed_pair(tmp_path, aux_censored_cell=(20, 3))
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3],
    )

    assert analysis["primary"]["censoring"]["aux"]["count"] == 1
    assert analysis["primary"]["censoring"]["aux"]["cells"] == [{"update": 20, "seed": 3}]
    assert analysis["primary"]["censoring"]["control"]["count"] == 0
    assert analysis["primary"]["censoring"]["structurally_different"] is True

    sources = analysis["source_artifacts"]
    entries = {(entry["arm"], entry["path"]): entry for entry in sources}
    assert entries[("aux", "ppo_aux_update_20.pt")]["sha256"] == file_sha256(
        aux_dir / "ppo_aux_update_20.pt"
    )
    assert entries[("control", "ppo_update_10.pt")]["sha256"] == file_sha256(
        control_dir / "ppo_update_10.pt"
    )
    assert ("aux", "report.json") in entries
    assert all(entry["sha256_status"] == prov_evidence_computed() for entry in sources)


def prov_evidence_computed() -> str:
    from qrokkun_ai.v5.tools import phase3_run_provenance as prov

    return prov.EVIDENCE_COMPUTED


def test_analysis_reports_terminal_survival_and_teacher_secondaries(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3], terminal_update=20,
    )
    secondary = analysis["secondary"]

    survival = secondary["terminal_survival"]
    assert survival["update"] == 20
    assert survival["paired"]["n_updates"] == 1
    assert survival["paired"]["dbar"] == pytest.approx(1.0)
    assert survival["aux"]["mean"] == pytest.approx(statistics_mean([12.0, 13.0, 14.0]))
    assert survival["control"]["mean"] == pytest.approx(statistics_mean([11.0, 12.0, 13.0]))

    teacher = secondary["terminal_teacher"]
    assert teacher["aux"]["agreement"] == 0.66
    assert teacher["control"]["agreement"] == 0.13
    assert teacher["aux"]["teacher_to_student_kl"] == 0.24
    assert teacher["status"] == "observed"


def statistics_mean(values: list[float]) -> float:
    import statistics

    return statistics.fmean(values)


def test_terminal_teacher_secondary_fails_its_own_closure_when_absent(tmp_path: Path) -> None:
    """A missing teacher diagnostic is reported unavailable, never back-filled
    from survival or from another run's numbers."""
    aux_dir, control_dir = _completed_pair(tmp_path, terminal_teacher=False)
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3], terminal_update=20,
    )
    teacher = analysis["secondary"]["terminal_teacher"]
    assert teacher["status"] == "unavailable"
    assert teacher["aux"]["agreement"] is None
    assert teacher["control"]["agreement"] is None
    # The primary is untouched by a failed secondary.
    assert analysis["primary"]["classification"] == "support"


# --------------------------------------------------------------------------- #
# 4. the CLI never assumes any experiment's U or seed window
# --------------------------------------------------------------------------- #
def test_module_hardcodes_no_experiment_specific_update_set_or_seed_window() -> None:
    source = Path(pair.__file__).read_text()
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#") and "PYTHONPATH" not in line
    )
    for literal in ("3200", "10000", "3000", "3029", "130000", "seed 11"):
        assert literal not in code, literal


def test_cli_requires_an_explicit_matched_update_set(tmp_path: Path) -> None:
    args = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"),
         "--eval-seed-start", "1", "--eval-seed-count", "3"]
    )
    with pytest.raises(pair.PairAnalysisError, match="explicit"):
        pair.resolve_updates(args)


def test_cli_requires_an_explicit_eval_seed_window(tmp_path: Path) -> None:
    args = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"),
         "--update", "10", "--update", "20"]
    )
    with pytest.raises(pair.PairAnalysisError, match="explicit"):
        pair.resolve_eval_seeds(args)


def test_cli_update_range_and_repeated_flags_describe_the_same_set(tmp_path: Path) -> None:
    ranged = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"),
         "--updates-start", "10", "--updates-stop", "30", "--updates-step", "10"]
    )
    listed = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"),
         "--update", "10", "--update", "20", "--update", "30"]
    )
    assert pair.resolve_updates(ranged) == pair.resolve_updates(listed) == [10, 20, 30]


@pytest.mark.parametrize(
    "field, bad",
    [
        ("updates", True),
        ("updates", 10),
        ("updates", 1.0),
        ("updates", "10"),
        ("updates", {"0": 10}),
        ("eval_seeds", True),
        ("eval_seeds", 1),
        ("eval_seeds", 1.0),
        ("eval_seeds", "1"),
        ("updates", [True, 20]),
        ("updates", [1.0, 20]),
        ("updates", ["10", 20]),
        ("eval_seeds", [True, 2]),
        ("eval_seeds", [1.5, 2]),
        ("eval_seeds", ["1", 2]),
    ],
)
def test_config_updates_and_eval_seeds_accept_only_lists_of_exact_ints(
    field: str, bad: Any, tmp_path: Path,
) -> None:
    payload = {"updates": [10, 20], "eval_seeds": [1, 2, 3]}
    payload[field] = bad
    config = tmp_path / "prereg.json"
    config.write_text(json.dumps(payload))
    args = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"), "--config", str(config)]
    )
    with pytest.raises(pair.PairAnalysisError, match="exact int|list"):
        pair.resolve_updates(args)
        pair.resolve_eval_seeds(args)


@pytest.mark.parametrize(
    "updates, eval_seeds",
    [
        ([True, 20], [1, 2, 3]),
        ([1.0, 20], [1, 2, 3]),
        (["10", 20], [1, 2, 3]),
        ([10, 20], [True, 2, 3]),
        ([10, 20], [1.0, 2, 3]),
        ([10, 20], ["1", 2, 3]),
    ],
)
def test_api_updates_and_eval_seeds_reject_non_exact_ints(
    updates: list[Any], eval_seeds: list[Any], tmp_path: Path,
) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    with pytest.raises(pair.PairAnalysisError, match="exact int"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=updates, eval_seeds=eval_seeds,
        )
    with pytest.raises(pair.PairAnalysisError, match="exact int"):
        pair.paired_curve_difference(
            updates=updates, eval_seeds=eval_seeds,
            aux_elapsed={10: {1: 1.0}, 20: {1: 1.0}},
            control_elapsed={10: {1: 1.0}, 20: {1: 1.0}},
        )


def test_malformed_present_report_raises_and_cli_writes_nothing(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    (aux_dir / "report.json").write_text("{not json")
    before = (control_dir / "report.json").read_bytes()
    with pytest.raises(pair.PairAnalysisError, match="report.json"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )
    assert (control_dir / "report.json").read_bytes() == before

    out = tmp_path / "analysis.json"
    with pytest.raises(pair.PairAnalysisError, match="report.json"):
        pair.main(
            [
                "--aux-run-dir", str(aux_dir), "--control-run-dir", str(control_dir),
                "--updates-start", "10", "--updates-stop", "20", "--updates-step", "10",
                "--eval-seed-start", "1", "--eval-seed-count", "3",
                "--out", str(out),
            ]
        )
    assert not out.exists()


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"updates": [10, 20], "eval_seeds": [1.0, 2, 3]}, "exact int"),
        ({"updates": "10", "eval_seeds": [1, 2, 3]}, "list"),
    ],
)
def test_cli_rejects_a_malformed_config_and_writes_nothing(
    payload: dict[str, Any], message: str, tmp_path: Path,
) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    config = tmp_path / "prereg.json"
    config.write_text(json.dumps(payload))
    out = tmp_path / "analysis.json"
    with pytest.raises(pair.PairAnalysisError, match=message):
        pair.main(
            [
                "--aux-run-dir", str(aux_dir), "--control-run-dir", str(control_dir),
                "--config", str(config), "--out", str(out),
            ]
        )
    assert not out.exists()


def test_missing_report_stays_an_unavailable_teacher_secondary(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    (aux_dir / "report.json").unlink()
    (control_dir / "report.json").unlink()
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3], terminal_update=20,
    )
    teacher = analysis["secondary"]["terminal_teacher"]
    assert teacher["status"] == "unavailable"
    assert teacher["aux"]["agreement"] is None
    assert teacher["control"]["teacher_to_student_kl"] is None
    assert analysis["primary"]["classification"] == "support"


def test_cli_reads_the_update_set_and_seeds_from_a_config_file(tmp_path: Path) -> None:
    config = tmp_path / "prereg.json"
    config.write_text(json.dumps({"updates": [10, 20], "eval_seeds": [1, 2, 3]}))
    args = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"),
         "--config", str(config)]
    )
    assert pair.resolve_updates(args) == [10, 20]
    assert pair.resolve_eval_seeds(args) == [1, 2, 3]


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_analyze_pair_refuses_an_impossible_confidence_level(
    confidence: float, tmp_path: Path,
) -> None:
    """The analyze/CLI path raises instead of writing an analysis whose
    interval could never have been computed."""
    aux_dir, control_dir = _completed_pair(tmp_path)
    with pytest.raises(pair.PairAnalysisError, match="confidence"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3], confidence=confidence,
        )


def test_cli_refuses_an_impossible_confidence_level_and_writes_nothing(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    out = tmp_path / "analysis.json"
    with pytest.raises(pair.PairAnalysisError, match="confidence"):
        pair.main(
            [
                "--aux-run-dir", str(aux_dir), "--control-run-dir", str(control_dir),
                "--updates-start", "10", "--updates-stop", "20", "--updates-step", "10",
                "--eval-seed-start", "1", "--eval-seed-count", "3",
                "--confidence", "1.5", "--out", str(out),
            ]
        )
    assert not out.exists()


def _replace_aux_pack(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path]:
    aux_dir, control_dir = _completed_pair(tmp_path)
    _write_pack(aux_dir / "ppo_aux_update_20.pt", update=20, per_seed={1: 11.0, 2: 12.0, 3: 13.0}, **kwargs)
    return aux_dir, control_dir


@pytest.mark.parametrize(
    "per_seed_payload",
    [
        {True: 11.0, 2: 12.0, 3: 13.0},
        {1: True, 2: 12.0, 3: 13.0},
        {1: False, 2: 12.0, 3: 13.0},
        {1: float("nan"), 2: 12.0, 3: 13.0},
        {1: float("inf"), 2: 12.0, 3: 13.0},
        {1: float("-inf"), 2: 12.0, 3: 13.0},
        {"01": 11.0, "2": 12.0, "3": 13.0},
        {"+1": 11.0, "2": 12.0, "3": 13.0},
        {"1.0": 11.0, "2": 12.0, "3": 13.0},
        {1.0: 11.0, 2: 12.0, 3: 13.0},
        {"1": "11.0", "2": 12.0, "3": 13.0},
    ],
)
def test_reader_rejects_bool_nonfinite_and_malformed_per_seed_cells(
    per_seed_payload: dict[Any, Any], tmp_path: Path,
) -> None:
    aux_dir, control_dir = _replace_aux_pack(tmp_path, per_seed_payload=per_seed_payload)
    with pytest.raises(pair.PairAnalysisError):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


def test_reader_accepts_exact_int_and_canonical_decimal_seed_keys(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    _write_pack(
        aux_dir / "ppo_aux_update_20.pt", update=20, per_seed={},
        per_seed_payload={1: 12.0, "2": 13.0, "3": 14.0},
        censored_payload={1: False, "2": False, "3": False},
    )
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3],
    )
    assert analysis["primary"]["valid"] is True
    assert analysis["primary"]["dbar"] == pytest.approx(1.0)


def test_missing_per_seed_censored_stays_unavailable(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    for update in (10, 20):
        _write_pack(
            aux_dir / f"ppo_aux_update_{update}.pt", update=update,
            per_seed={s: 11.0 + s for s in (1, 2, 3)}, include_censored=False,
        )
    analysis = pair.analyze_pair(
        aux_run_dir=aux_dir, control_run_dir=control_dir,
        updates=[10, 20], eval_seeds=[1, 2, 3],
    )
    censoring = analysis["primary"]["censoring"]
    assert censoring["status"] == "unavailable"
    assert censoring["aux"]["status"] == "unavailable"
    assert censoring["aux"]["count"] is None
    assert analysis["primary"]["valid"] is True
    assert analysis["primary"]["dbar"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "censored_payload",
    [
        [],
        "false",
        None,
        {},
        {"1": 0, "2": False, "3": False},
        {"1": 1, "2": True, "3": False},
        {"1": False},
        {"1": False, "2": False, "3": False, "4": True},
        {"01": False, "2": False, "3": False},
        {"1": "false", "2": False, "3": False},
        {True: False, 2: False, 3: False},
    ],
)
def test_present_per_seed_censored_must_match_seeds_with_real_bools(
    censored_payload: Any, tmp_path: Path,
) -> None:
    aux_dir, control_dir = _replace_aux_pack(tmp_path, censored_payload=censored_payload)
    with pytest.raises(pair.PairAnalysisError):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


@pytest.mark.parametrize("packed_update", [99, True, 10.0, "10", None])
def test_packed_update_must_be_the_exact_filename_update(
    packed_update: Any, tmp_path: Path,
) -> None:
    aux_dir, control_dir = _replace_aux_pack(tmp_path, packed_update=packed_update)
    with pytest.raises(pair.PairAnalysisError, match="update"):
        pair.analyze_pair(
            aux_run_dir=aux_dir, control_run_dir=control_dir,
            updates=[10, 20], eval_seeds=[1, 2, 3],
        )


def test_cli_rejects_a_packed_update_mismatch_and_writes_nothing(tmp_path: Path) -> None:
    aux_dir, control_dir = _replace_aux_pack(tmp_path, packed_update=99)
    out = tmp_path / "analysis.json"
    with pytest.raises(pair.PairAnalysisError, match="update"):
        pair.main(
            [
                "--aux-run-dir", str(aux_dir), "--control-run-dir", str(control_dir),
                "--updates-start", "10", "--updates-stop", "20", "--updates-step", "10",
                "--eval-seed-start", "1", "--eval-seed-count", "3",
                "--out", str(out),
            ]
        )
    assert not out.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--updates-start", "0", "--updates-stop", "100", "--updates-step", "0",
         "--eval-seed-start", "1", "--eval-seed-count", "3"],
        ["--updates-start", "10", "--eval-seed-start", "1", "--eval-seed-count", "3"],
        ["--updates-stop", "20", "--eval-seed-start", "1", "--eval-seed-count", "3"],
        ["--updates-step", "10", "--eval-seed-start", "1", "--eval-seed-count", "3"],
        ["--updates-start", "10", "--updates-stop", "20",
         "--eval-seed-start", "1", "--eval-seed-count", "3"],
        ["--update", "10", "--update", "20", "--eval-seed-start", "5", "--eval-seed-count", "0"],
        ["--update", "10", "--update", "20", "--eval-seed-start", "5"],
        ["--update", "10", "--update", "20", "--eval-seed-count", "3"],
        ["--update", "10", "--update", "20", "--eval-seed-start", "1", "--eval-seed-count", "-2"],
        ["--updates-start", "10", "--updates-stop", "20", "--updates-step", "-5",
         "--eval-seed-start", "1", "--eval-seed-count", "3"],
    ],
)
def test_cli_range_flags_never_fall_back_to_config_and_write_nothing(
    extra: list[str], tmp_path: Path,
) -> None:
    """A present range flag owns that axis. Zero, negative, and partial ranges
    raise instead of silently selecting the config file's grid."""
    aux_dir, control_dir = _completed_pair(tmp_path)
    config = tmp_path / "prereg.json"
    config.write_text(json.dumps({"updates": [10, 20], "eval_seeds": [1, 2, 3]}))
    out = tmp_path / "analysis.json"
    with pytest.raises(pair.PairAnalysisError):
        pair.main(
            [
                "--aux-run-dir", str(aux_dir), "--control-run-dir", str(control_dir),
                "--config", str(config), *extra, "--out", str(out),
            ]
        )
    assert not out.exists()


def test_complete_range_flags_take_precedence_over_config(tmp_path: Path) -> None:
    config = tmp_path / "prereg.json"
    config.write_text(json.dumps({"updates": [10, 20], "eval_seeds": [1, 2, 3]}))
    args = pair.build_parser().parse_args(
        ["--aux-run-dir", str(tmp_path), "--control-run-dir", str(tmp_path),
         "--out", str(tmp_path / "analysis.json"), "--config", str(config),
         "--updates-start", "30", "--updates-stop", "30", "--updates-step", "10",
         "--eval-seed-start", "8", "--eval-seed-count", "2"]
    )
    assert pair.resolve_updates(args) == [30]
    assert pair.resolve_eval_seeds(args) == [8, 9]


def test_main_writes_analysis_json_for_a_completed_pair(tmp_path: Path) -> None:
    aux_dir, control_dir = _completed_pair(tmp_path)
    out = tmp_path / "analysis.json"
    exit_code = pair.main(
        [
            "--aux-run-dir", str(aux_dir), "--control-run-dir", str(control_dir),
            "--updates-start", "10", "--updates-stop", "20", "--updates-step", "10",
            "--eval-seed-start", "1", "--eval-seed-count", "3",
            "--terminal-update", "20", "--out", str(out),
        ]
    )
    assert exit_code == 0

    analysis = json.loads(out.read_text())
    assert analysis["schema_version"] == pair.ANALYSIS_SCHEMA_VERSION
    assert analysis["configuration"]["updates"] == [10, 20]
    assert analysis["configuration"]["eval_seeds"] == [1, 2, 3]
    assert analysis["configuration"]["aux_run_dir"] == str(aux_dir.resolve())
    assert analysis["primary"]["classification"] == "support"
    assert analysis["no_promotion"] is True
    assert analysis["evidence_provenance"]["primary"] == "computed"
