"""Small CPU-only contracts for Phase 3 long-run/resume support."""

from __future__ import annotations

import sys
import json
import signal
import datetime as dt
import argparse
from pathlib import Path
from typing import Any

import pytest
import torch

_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qrokkun_ai.v5.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_ai.v5.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4

from qrokkun_ai.v1.agents.player_v1 import PlayerV1

from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention as control
from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention_aux as aux
from qrokkun_ai.v5.tools import phase3_run_provenance as prov

_CURRENT_SCHEMA_VERSION = control.CURRENT_RUN_SCHEMA_VERSION

# A stubbed initial load still has to look like a real strict load: the input
# manifest records architecture/schema identity and fails closed without it.
_STUB_INIT_META = {
    "architecture": "player_ranked_topk",
    "architecture_version": 1,
    "schema_version": 1,
    "production_compatible": True,
    "experimental": False,
}


def _write_tiny_teacher(tmp_path: Path, *, hidden: int = 8) -> Path:
    """A real loadable V1 teacher, so teacher identity can be produced."""
    path = tmp_path / "teacher.pt"
    torch.save({"hidden": hidden, "state_dict": PlayerV1(hidden=hidden).state_dict()}, path)
    return path


def _current_contract(**fields: object) -> dict:
    return {"schema_version": _CURRENT_SCHEMA_VERSION, **fields}


@pytest.mark.parametrize(
    "existing_contract, error",
    [(None, "run.json is missing"), ({"wrong": True}, "schema_version")],
)
def test_resume_contract_rejection_precedes_initial_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_contract, error: str,
) -> None:
    """A bad resume must fail before the initial environment-evaluation gate."""
    args = argparse.Namespace(
        out_dir=tmp_path, run_dir=tmp_path, init_checkpoint=Path("init.pt"),
        teacher=Path("teacher.pt"), updates=1, episodes_per_update=1,
        max_frames=1, eval_seeds=[0], eval_max_steps=1,
        initial_gate_mean_min=0.0, initial_gate_median_min=0.0,
        data_episodes=1, data_max_steps=1, data_frames_cap=1,
        end_time=None, max_updates=None, effective_seed=1, resume=True,
    )
    monkeypatch.setattr(
        control, "load_initial_checkpoint",
        lambda *_: (__import__("torch").nn.Linear(1, 1), {"metadata": {}}),
    )
    monkeypatch.setattr(control, "checkpoint_provenance", lambda *_: {"file_sha256": "init", "state_dict_sha256": "state"})
    monkeypatch.setattr(control, "file_sha256", lambda *_: "teacher")
    monkeypatch.setattr(control, "current_git_commit", lambda *_: "commit")
    monkeypatch.setattr(control, "_git_dirty", lambda *_: False)
    if existing_contract is not None:
        (tmp_path / "run.json").write_text(json.dumps(existing_contract))
    monkeypatch.setattr(
        control, "evaluate_deterministic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("evaluation ran before contract rejection")),
    )
    recovery_called = False

    def unexpected_recovery(*_args, **_kwargs):
        nonlocal recovery_called
        recovery_called = True
        raise AssertionError("recovery ran before contract rejection")

    monkeypatch.setattr(control, "load_recovery", unexpected_recovery)

    with pytest.raises(control.RunStateError, match=error):
        control.run_experiment(args, __import__("torch").device("cpu"))
    assert recovery_called is False


def test_truncate_progress_journal_drops_stale_ahead_rows(tmp_path: Path) -> None:
    """Recovery is source of truth: ahead jsonl rows are dropped, then a fresh row is appended."""
    progress = tmp_path / "ppo_updates.jsonl"
    progress.write_text("".join(json.dumps({"update": n}) + "\n" for n in range(1, 38)))

    control.truncate_progress_journal_to(progress, 0)
    assert progress.read_text() == ""
    assert control.reconcile_progress_journal(progress, completed_update=0) == set()
    assert control.append_progress_row(progress, {"update": 1, "metric": 3}) is True
    assert [json.loads(line)["update"] for line in progress.read_text().splitlines()] == [1]


@pytest.mark.parametrize("module, filename", [
    (control, "ppo_updates.jsonl"),
    (aux, "ppo_aux_updates.jsonl"),
])
def test_truncate_then_append_continues_from_recovered_completed(
    tmp_path: Path, module, filename: str,
) -> None:
    progress = tmp_path / filename
    progress.write_text("".join(json.dumps({"update": n}) + "\n" for n in range(1, 68)))
    control.truncate_progress_journal_to(progress, 50)
    assert control.reconcile_progress_journal(progress, completed_update=50) == set(range(1, 51))
    assert module.append_progress_row(progress, {"update": 51}) is True
    assert [json.loads(line)["update"] for line in progress.read_text().splitlines()] == list(range(1, 52))


def test_reconcile_progress_journal_accepts_zero_lag_prefix(tmp_path: Path) -> None:
    progress = tmp_path / "ppo_updates.jsonl"
    progress.write_text("".join(json.dumps({"update": n}) + "\n" for n in range(1, 51)))
    assert control.reconcile_progress_journal(progress, completed_update=50) == set(range(1, 51))


def test_reconcile_progress_journal_fails_closed_on_gap(tmp_path: Path) -> None:
    progress = tmp_path / "ppo_updates.jsonl"
    progress.write_text(json.dumps({"update": 1}) + "\n" + json.dumps({"update": 3}) + "\n")
    with pytest.raises(control.RunStateError, match="contiguous"):
        control.reconcile_progress_journal(progress, completed_update=3)


@pytest.mark.parametrize("module", [control, aux])
def test_long_run_parser_accepts_run_directory_stop_controls_and_seed(module) -> None:
    args = module.build_parser().parse_args(
        [
            "--init-checkpoint", "init.pt", "--teacher", "teacher.pt",
            "--run-dir", "run", "--end-time", "20260915-2359",
            "--max-updates", "7", "--resume", "--seed", "19",
        ]
    )
    assert args.run_dir == Path("run")
    assert args.end_time.tzinfo is not None
    assert args.end_time.strftime("%Y%m%d-%H%M") == "20260915-2359"
    assert args.max_updates == 7
    assert args.resume is True
    assert args.seed == 19


@pytest.mark.parametrize("module", [control, aux])
@pytest.mark.parametrize(
    "argv, expected_max, expected_end",
    [
        (['--end-time', '20260915-2359'], 2_147_483_647, '2026-09-15T23:59:00+08:00'),
        (['--max-updates', '275'], 275, '2099-12-31T23:59:00+08:00'),
        ([], 200, '2099-12-31T23:59:00+08:00'),
        (
            ['--end-time', '20260915-2359', '--max-updates', '275'],
            275,
            '2026-09-15T23:59:00+08:00',
        ),
    ],
)
def test_stop_budget_defaults_are_effective_and_not_limited_by_ppo_plan(
    module, argv: list[str], expected_max: int, expected_end: str,
) -> None:
    """Parser/default handling resolves the two stop controls as one contract."""
    args = module.apply_mode_defaults(module.build_parser().parse_args([
        '--init-checkpoint', 'init.pt', '--teacher', 'teacher.pt', *argv,
    ]))
    assert args.effective_max_updates == expected_max
    assert args.effective_end_time.isoformat() == expected_end


def test_effective_stop_budget_controls_boundary_order_and_allows_past_200() -> None:
    deadline = control.parse_end_time('20260915-1200')
    assert control.boundary_stop_reason(
        completed=200, configured_updates=200, max_updates=2_147_483_647,
        end_time=control.FAR_FUTURE_END_TIME, now=deadline,
    ) is None
    assert control.boundary_stop_reason(
        completed=275, configured_updates=200, max_updates=275,
        end_time=deadline, now=deadline,
    ) == 'deadline'
    assert control.boundary_stop_reason(
        completed=275, configured_updates=200, max_updates=275,
        end_time=control.FAR_FUTURE_END_TIME, now=deadline,
    ) == 'max_updates'


def test_resume_may_reduce_max_updates_to_above_completed_and_audits_it(tmp_path: Path) -> None:
    """A lower target is allowed when it is still at or above completed updates."""
    contract = _current_contract(stop_args={
        "effective_max_updates": 200,
        "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": {}, "optimizer": {}, "completed_update": 40,
        "rng": control.capture_rng_state(),
    })
    reduced = {**contract, "stop_args": {**contract["stop_args"], "effective_max_updates": 50}}
    control.create_or_validate_run_contract(tmp_path, reduced, resume=True)
    rows = [json.loads(row) for row in (tmp_path / "stop_budget_amendments.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "stop_budget_extended"
    assert rows[0]["prior"]["effective_max_updates"] == 200
    assert rows[0]["new"]["effective_max_updates"] == 50
    assert rows[0]["completed_update"] == 40


def test_resume_may_set_max_updates_equal_to_completed_and_audits_it(tmp_path: Path) -> None:
    """A target equal to completed is allowed and recorded."""
    contract = _current_contract(stop_args={
        "effective_max_updates": 200,
        "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": {}, "optimizer": {}, "completed_update": 40,
        "rng": control.capture_rng_state(),
    })
    equal = {**contract, "stop_args": {**contract["stop_args"], "effective_max_updates": 40}}
    control.create_or_validate_run_contract(tmp_path, equal, resume=True)
    rows = [json.loads(row) for row in (tmp_path / "stop_budget_amendments.jsonl").read_text().splitlines()]
    assert rows[0]["new"]["effective_max_updates"] == 40
    assert rows[0]["completed_update"] == 40


def test_resume_rejects_max_updates_below_completed(tmp_path: Path) -> None:
    contract = _current_contract(stop_args={
        "effective_max_updates": 200,
        "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": {}, "optimizer": {}, "completed_update": 40,
        "rng": control.capture_rng_state(),
    })
    below = {**contract, "stop_args": {**contract["stop_args"], "effective_max_updates": 39}}
    with pytest.raises(control.RunStateError, match="below completed"):
        control.create_or_validate_run_contract(tmp_path, below, resume=True)
    assert not (tmp_path / "stop_budget_amendments.jsonl").exists()


def test_resume_may_move_deadline_earlier_when_still_strictly_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier deadline is allowed when it remains strictly after now."""
    now = dt.datetime(2026, 9, 15, 12, 0, tzinfo=control._HKT)
    monkeypatch.setattr(control, "_now_hkt", lambda: now)
    contract = _current_contract(stop_args={
        "effective_max_updates": 200,
        "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": {}, "optimizer": {}, "completed_update": 0,
        "rng": control.capture_rng_state(),
    })
    earlier = {**contract, "stop_args": {
        **contract["stop_args"], "effective_end_time_hkt": "2026-09-16T08:00:00+08:00",
    }}
    control.create_or_validate_run_contract(tmp_path, earlier, resume=True)
    rows = [json.loads(row) for row in (tmp_path / "stop_budget_amendments.jsonl").read_text().splitlines()]
    assert rows[0]["prior"]["effective_end_time_hkt"] == "2099-12-31T23:59:00+08:00"
    assert rows[0]["new"]["effective_end_time_hkt"] == "2026-09-16T08:00:00+08:00"


@pytest.mark.parametrize("end_hkt", [
    "2026-09-15T12:00:00+08:00",
    "2026-09-15T11:59:00+08:00",
])
def test_resume_rejects_deadline_at_or_before_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, end_hkt: str,
) -> None:
    now = dt.datetime(2026, 9, 15, 12, 0, tzinfo=control._HKT)
    monkeypatch.setattr(control, "_now_hkt", lambda: now)
    contract = _current_contract(stop_args={
        "effective_max_updates": 200,
        "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": {}, "optimizer": {}, "completed_update": 0,
        "rng": control.capture_rng_state(),
    })
    requested = {**contract, "stop_args": {**contract["stop_args"], "effective_end_time_hkt": end_hkt}}
    with pytest.raises(control.RunStateError, match="not in the future"):
        control.create_or_validate_run_contract(tmp_path, requested, resume=True)
    assert not (tmp_path / "stop_budget_amendments.jsonl").exists()


@pytest.mark.parametrize("end_hkt", [
    "2026-09-15T12:00:00+08:00",
    "2026-09-15T11:59:00+08:00",
])
def test_identical_expired_authorized_deadline_resume_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, end_hkt: str,
) -> None:
    """Idempotency applies only to a still-future authorized pair."""
    now = dt.datetime(2026, 9, 15, 12, 0, tzinfo=control._HKT)
    monkeypatch.setattr(control, "_now_hkt", lambda: now)
    contract = _current_contract(stop_args={
        "effective_max_updates": 200,
        "effective_end_time_hkt": end_hkt,
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    with pytest.raises(control.RunStateError, match="not in the future"):
        control.create_or_validate_run_contract(tmp_path, contract, resume=True)
    assert not (tmp_path / "stop_budget_amendments.jsonl").exists()


def test_identical_authorized_stop_budget_resume_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = dt.datetime(2026, 9, 15, 12, 0, tzinfo=control._HKT)
    monkeypatch.setattr(control, "_now_hkt", lambda: now)
    contract = _current_contract(tool="control", stop_args={
        "effective_max_updates": 200, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": {}, "optimizer": {}, "completed_update": 0,
        "rng": control.capture_rng_state(),
    })
    raised = {**contract, "stop_args": {**contract["stop_args"], "effective_max_updates": 275}}
    control.create_or_validate_run_contract(tmp_path, raised, resume=True)
    audit = tmp_path / "stop_budget_amendments.jsonl"
    assert len(audit.read_text().splitlines()) == 1
    control.create_or_validate_run_contract(tmp_path, raised, resume=True)
    assert len(audit.read_text().splitlines()) == 1


@pytest.mark.parametrize('changed', [
    {'tool': 'other'}, {'arm': 'other'}, {'inputs': {'checkpoint': 'other'}}, {'knobs': {'lr': 7}}, {'effective_seed': 2},
])
def test_stop_extensions_do_not_relax_other_contract_identity(tmp_path: Path, changed: dict) -> None:
    contract = _current_contract(tool='control', inputs={'checkpoint': 'a'}, knobs={'lr': 1}, effective_seed=1,
                                 stop_args={'effective_max_updates': 1, 'effective_end_time_hkt': '2026-09-15T06:45:00+08:00'})
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    requested = {**contract, **changed, 'stop_args': {**contract['stop_args'], 'effective_max_updates': 2}}
    with pytest.raises(control.RunStateError, match='mismatch'):
        control.create_or_validate_run_contract(tmp_path, requested, resume=True)


def test_malformed_or_noncontiguous_stop_amendment_fails_closed(tmp_path: Path) -> None:
    contract = _current_contract(stop_args={'effective_max_updates': 1, 'effective_end_time_hkt': '2099-12-31T23:59:00+08:00'})
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    audit = tmp_path / 'stop_budget_amendments.jsonl'
    audit.write_text('{bad json}\n')
    with pytest.raises(control.RunStateError, match='amendment'):
        control.create_or_validate_run_contract(tmp_path, contract, resume=True)
    audit.write_text(json.dumps({
        'event': 'stop_budget_extended', 'prior': contract['stop_args'], 'new': contract['stop_args'],
        'completed_update': 0, 'at_hkt': '2026-09-15T07:00:00+08:00', 'runtime_provenance': {},
    }) + '\n')
    with pytest.raises(control.RunStateError, match='amendment'):
        control.create_or_validate_run_contract(tmp_path, contract, resume=True)


def test_stop_amendments_must_be_timestamp_ordered_and_contiguous(tmp_path: Path) -> None:
    contract = _current_contract(stop_args={'effective_max_updates': 1, 'effective_end_time_hkt': '2099-12-31T23:59:00+08:00'})
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    first = {'effective_max_updates': 2, 'effective_end_time_hkt': '2099-12-31T23:59:00+08:00'}
    second = {'effective_max_updates': 3, 'effective_end_time_hkt': '2099-12-31T23:59:00+08:00'}
    rows = [
        {'event': 'stop_budget_extended', 'prior': contract['stop_args'], 'new': first,
         'completed_update': 0, 'at_hkt': '2026-09-15T08:00:00+08:00', 'runtime_provenance': {}},
        {'event': 'stop_budget_extended', 'prior': first, 'new': second,
         'completed_update': 0, 'at_hkt': '2026-09-15T07:00:00+08:00', 'runtime_provenance': {}},
    ]
    (tmp_path / 'stop_budget_amendments.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    with pytest.raises(control.RunStateError, match='amendment'):
        control.create_or_validate_run_contract(tmp_path, _current_contract(stop_args=second), resume=True)


@pytest.mark.parametrize("stop_args", [
    {"effective_max_updates": -1, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00"},
    {"effective_max_updates": True, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00"},
    {"effective_max_updates": 10, "effective_end_time_hkt": "not-a-deadline"},
    {"effective_max_updates": 10, "effective_end_time_hkt": "2026-09-16T08:00:00"},
])
def test_invalid_requested_stop_budget_fails_closed(tmp_path: Path, stop_args: dict) -> None:
    contract = _current_contract(stop_args={
        "effective_max_updates": 10, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    with pytest.raises(control.RunStateError, match="malformed"):
        control.create_or_validate_run_contract(
            tmp_path, {**contract, "stop_args": stop_args}, resume=True,
        )
    assert not (tmp_path / "stop_budget_amendments.jsonl").exists()


@pytest.mark.parametrize('bad_version', [None, 0, 1, 2, 3, True, 1.0])
def test_resume_requires_exact_integer_current_schema_version(tmp_path: Path, bad_version: object) -> None:
    """Versionless, v1, v2, v3, unsupported, bool, and float runs are archival-only."""
    existing = {'tool': 'control', 'stop_args': {
        'end_time_hkt': '2026-09-15T06:45:00+08:00', 'max_updates': None,
    }}
    if bad_version is not None:
        existing['schema_version'] = bad_version
    (tmp_path / 'run.json').write_text(json.dumps(existing))
    with pytest.raises(control.RunStateError, match='schema_version'):
        control.create_or_validate_run_contract(
            tmp_path,
            _current_contract(tool='control', stop_args={
                'effective_max_updates': 201,
                'effective_end_time_hkt': '2026-09-15T07:00:00+08:00',
            }),
            resume=True,
        )
    assert not (tmp_path / 'stop_budget_amendments.jsonl').exists()


def test_fresh_contract_writes_schema_version_and_exact_current_version_resumes(tmp_path: Path) -> None:
    contract = _current_contract(tool='control', stop_args={
        'effective_max_updates': 1, 'effective_end_time_hkt': '2099-12-31T23:59:00+08:00',
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    assert json.loads((tmp_path / 'run.json').read_text())['schema_version'] == _CURRENT_SCHEMA_VERSION
    control.create_or_validate_run_contract(tmp_path, contract, resume=True)


def test_fresh_contract_without_current_schema_version_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(control.RunStateError, match='schema_version'):
        control.create_or_validate_run_contract(tmp_path, {'tool': 'control'}, resume=False)
    assert not (tmp_path / 'run.json').exists()


def test_contract_is_immutable_and_recovery_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    contract = _current_contract(tool="control", inputs={"checkpoint": "a"}, no_promotion=True,
                                 stop_args={"effective_max_updates": 1,
                                            "effective_end_time_hkt": "2099-12-31T23:59:00+08:00"})
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    assert json.loads((tmp_path / "run.json").read_text()) == contract
    control.create_or_validate_run_contract(tmp_path, contract, resume=True)
    with pytest.raises(control.RunStateError, match="mismatch"):
        control.create_or_validate_run_contract(tmp_path, {**contract, "tool": "aux"}, resume=True)
    with pytest.raises(control.RunStateError, match="missing"):
        control.load_recovery(tmp_path / "recovery.pt", __import__("torch").device("cpu"))
    (tmp_path / "recovery.pt").write_bytes(b"not a torch checkpoint")
    with pytest.raises(control.RunStateError, match="malformed"):
        control.load_recovery(tmp_path / "recovery.pt", __import__("torch").device("cpu"))


def test_recovery_round_trip_and_status_history_are_append_only(tmp_path: Path) -> None:
    import torch

    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters())
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "model": net.state_dict(), "optimizer": opt.state_dict(), "completed_update": 2,
        "total_frames": 9, "rng": control.capture_rng_state(),
    })
    state = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    assert state["completed_update"] == 2
    control.append_run_status(tmp_path, {"event": "update_complete", "update": 1})
    control.append_run_status(tmp_path, {"event": "resumed", "completed_update": 1})
    rows = (tmp_path / "status.jsonl").read_text().splitlines()
    assert [json.loads(row)["event"] for row in rows] == ["update_complete", "resumed"]


def test_recovery_load_keeps_cpu_rng_state_when_resuming_to_cuda_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery deserialization must not relocate CPU RNG state to CUDA.

    The fake loader models ``torch.load(..., map_location=cuda)`` relocating
    the saved CPU ByteTensor.  The isolated fake setter then reproduces the
    real CPU-only ``torch.set_rng_state`` rejection without requiring CUDA.
    """
    cuda = torch.device("cuda")
    cpu_rng = torch.get_rng_state()
    relocated_rng = object()
    map_locations: list[object] = []
    real_set_rng_state = torch.set_rng_state

    def fake_load(_path, *, map_location, weights_only):
        assert weights_only is False
        map_locations.append(map_location)
        return {
            "model": {}, "optimizer": {}, "completed_update": 201,
            "rng": {
                "python": control.random.getstate(),
                "numpy": control.np.random.get_state(),
                "torch_cpu": relocated_rng if map_location == cuda else cpu_rng,
            },
        }

    def fake_set_rng_state(state):
        if state is relocated_rng:
            raise TypeError("RNG state must be a torch.ByteTensor")
        real_set_rng_state(state)

    recovery_path = tmp_path / "recovery.pt"
    recovery_path.touch()
    monkeypatch.setattr(control.torch, "load", fake_load)
    monkeypatch.setattr(control.torch, "set_rng_state", fake_set_rng_state)

    state = control.load_recovery(recovery_path, cuda)

    assert state["completed_update"] == 201
    assert map_locations == [torch.device("cpu")]
    assert state["rng"]["torch_cpu"].device.type == "cpu"
    assert state["rng"]["torch_cpu"].dtype == torch.uint8


def test_aux_resume_uses_the_shared_control_recovery_loader() -> None:
    assert aux.ret_mod is control
    assert aux.ret_mod.load_recovery is control.load_recovery


def test_stop_request_is_boundary_request_and_aux_recovery_has_alpha_generators(tmp_path: Path) -> None:
    import signal
    import torch

    stop = control.StopRequest()
    assert stop.requested is False
    stop.handler(signal.SIGINT, None)
    assert (stop.requested, stop.signal_name) == (True, "SIGINT")
    gen = aux.training_generator()
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters())
    control.atomic_save_recovery(tmp_path / "recovery.pt", {
        "arm": "aux", "model": net.state_dict(), "optimizer": opt.state_dict(),
        "completed_update": 1, "rng": control.capture_rng_state(), "alpha": 0.15,
        "calibration": {"alpha": 0.15}, "calib_seeds": [50000], "first_update_rollouts": [],
        "training_generator_state": gen.get_state(),
        "calibration_generator_state": aux.calibration_generator().get_state(),
        "diagnostic_generator_state": aux.diagnostic_generator().get_state(),
    })
    recovered = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    assert recovered["alpha"] == 0.15
    assert torch.equal(recovered["training_generator_state"], gen.get_state())


def test_max_update_and_hkt_deadline_have_explicit_boundary_reasons() -> None:
    deadline = control.parse_end_time("20260915-1200")
    assert control.boundary_stop_reason(
        completed=1, configured_updates=10, max_updates=1, end_time=None
    ) == "max_updates"
    assert control.boundary_stop_reason(
        completed=0, configured_updates=10, max_updates=8, end_time=deadline,
        now=deadline + dt.timedelta(seconds=1),
    ) == "deadline"


def _stub_rollout(seed: int = 0) -> control.Rollout:
    return control.Rollout(
        seed=seed,
        player=[__import__("numpy").zeros(PLAYER_FEAT_V4, dtype="float32")],
        bullets=[__import__("numpy").zeros((MAX_BULLETS_V4, BULLET_FEAT_V4), dtype="float32")],
        pad=[__import__("numpy").ones(MAX_BULLETS_V4, dtype="bool")],
        actions=[0], log_probs=[0.0], values=[0.0], rewards=[0.0], dones=[True],
        elapsed=1.0, censored=False,
    )


def _tiny_ranked(seed: int = 0) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=8, hidden=16)


def _arm_args(tmp_path: Path, *, max_updates: int, resume: bool = False) -> argparse.Namespace:
    end = dt.datetime(2099, 12, 31, 23, 59, tzinfo=control._HKT)
    return argparse.Namespace(
        updates=10, episodes_per_update=1, max_frames=4, eval_seeds=[0], eval_max_steps=4,
        out_dir=tmp_path, run_dir=tmp_path, resume=resume, seed=1, max_updates=max_updates,
        end_time=end, effective_max_updates=max_updates, effective_end_time=end,
    )


def _stub_control_loop(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    updates: list[int] = []

    def fake_collect(net, device, seed, max_frames):
        return _stub_rollout(seed)

    def fake_ppo(net, opt, rollouts, device):
        updates.append(1)
        return {"optimizer_steps": 1, "approx_kl": 0.0, "clip_fraction": 0.0,
                "explained_variance": 0.0, "entropy": 0.0, "policy_loss": 0.0,
                "value_loss": 0.0, "total_loss": 0.0, "n_samples": 1}

    monkeypatch.setattr(control, "collect_rollout", fake_collect)
    monkeypatch.setattr(control, "ppo_update", fake_ppo)
    monkeypatch.setattr(
        control, "evaluate_deterministic",
        lambda *a, **k: [{"seed": 0, "elapsed": 1.0, "censored": True}],
    )
    return updates


def _stub_aux_loop(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    updates: list[int] = []

    def fake_collect(net, device, seed, max_frames):
        return _stub_rollout(seed)

    def fake_aux_update(net, opt, rollouts, train_tensors, alpha, device, **kwargs):
        updates.append(1)
        alignment = {
            "g_ppo_norm": 1.0, "g_ret_norm": 1.0, "cosine_similarity": 0.0,
            "grad_ratio": 0.15, "g_ret_weighted_norm": 0.15, "alpha": float(alpha),
            "measurement": "pre_update_on_policy", "recomputed_post_update": False,
        }
        return {
            "optimizer_steps": 1, "approx_kl": 0.0, "clip_fraction": 0.0,
            "explained_variance": 0.0, "entropy": 0.0, "ppo_policy_loss": 0.0,
            "ppo_value_loss": 0.0, "total_loss": 0.0, "retention_hybrid_loss": 0.0,
            "retention_hard_ce": 0.0, "retention_soft_kl": 0.0, "retention_soft_ce": 0.0,
            "alpha": float(alpha), "g_ppo_norm": 1.0, "g_ret_norm": 1.0,
            "g_ret_weighted_norm": 0.15, "grad_ratio": 0.15, "cosine_similarity": 0.0,
            "grad_alignment": alignment, "n_samples": 1, "n_retention_samples": 1,
        }

    monkeypatch.setattr(aux, "collect_rollout", fake_collect)
    monkeypatch.setattr(aux, "ppo_aux_update", fake_aux_update)
    monkeypatch.setattr(
        aux, "calibrate_alpha",
        lambda *a, **k: {"alpha": 0.15, "g_ppo_norm": 1.0, "g_ret_norm": 1.0},
    )
    monkeypatch.setattr(
        aux, "evaluate_deterministic",
        lambda *a, **k: [{"seed": 0, "elapsed": 1.0, "censored": True}],
    )
    monkeypatch.setattr(aux, "teacher_diagnostics", lambda *a, **k: {"agreement": 1.0})
    return updates


def test_control_resume_loop_consumes_amended_authorized_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates = _stub_control_loop(monkeypatch)
    net = _tiny_ranked()
    contract = _current_contract(stop_args={
        "effective_max_updates": 10, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    first = control.run_ppo_arm(
        net, torch.device("cpu"), _arm_args(tmp_path, max_updates=2), [], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    assert first["completed_updates"] == 2
    reduced = {**contract, "stop_args": {**contract["stop_args"], "effective_max_updates": 2}}
    authorized = control.create_or_validate_run_contract(tmp_path, reduced, resume=True)
    args = _arm_args(tmp_path, max_updates=authorized["effective_max_updates"], resume=True)
    updates.clear()
    resumed = control.run_ppo_arm(
        net, torch.device("cpu"), args, [], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    assert updates == []
    assert resumed["completed_updates"] == 2
    assert resumed["effective_max_updates"] == 2


def test_aux_resume_loop_consumes_amended_authorized_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates = _stub_aux_loop(monkeypatch)
    net = _tiny_ranked()
    train = {
        "player": torch.zeros(8, PLAYER_FEAT_V4),
        "bullets": torch.zeros(8, MAX_BULLETS_V4, BULLET_FEAT_V4),
        "pad": torch.ones(8, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.zeros(8, 5),
        "elapsed": torch.zeros(8),
    }
    contract = _current_contract(stop_args={
        "effective_max_updates": 10, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    first = aux.run_aux_arm(
        net, torch.device("cpu"), _arm_args(tmp_path, max_updates=2), [], train, None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
        minibatch=4,
    )
    assert first["completed_updates"] == 2
    reduced = {**contract, "stop_args": {**contract["stop_args"], "effective_max_updates": 2}}
    authorized = control.create_or_validate_run_contract(tmp_path, reduced, resume=True)
    args = _arm_args(tmp_path, max_updates=authorized["effective_max_updates"], resume=True)
    updates.clear()
    resumed = aux.run_aux_arm(
        net, torch.device("cpu"), args, [], train, None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
        minibatch=4,
    )
    assert updates == []
    assert resumed["completed_updates"] == 2
    assert resumed["effective_max_updates"] == 2


def _resume_from_update_argv(*extra: str) -> list[str]:
    return ["--init-checkpoint", "init.pt", "--teacher", "teacher.pt", *extra]


@pytest.mark.parametrize("module", [control, aux])
def test_resume_from_update_parser_implies_resume_mode(module) -> None:
    """`--resume-from-update` is itself resume mode; explicit `--resume` is equivalent."""
    without = module.apply_mode_defaults(module.build_parser().parse_args(
        _resume_from_update_argv("--resume-from-update", "200", "--max-updates", "201"),
    ))
    with_flag = module.apply_mode_defaults(module.build_parser().parse_args(
        _resume_from_update_argv("--resume", "--resume-from-update", "200", "--max-updates", "201"),
    ))
    assert without.resume_from_update == with_flag.resume_from_update == 200
    assert without.resume is True and with_flag.resume is True
    assert control.resolve_resume_from_stop_budget(without, 200) == (
        control.resolve_resume_from_stop_budget(with_flag, 200)
    )
    assert without.effective_max_updates == 201


@pytest.mark.parametrize("module", [control, aux])
@pytest.mark.parametrize("n", [1, 25, 49, 199, 201])
def test_resume_from_update_rejects_non_positive_200_multiples(module, n: int) -> None:
    args = module.build_parser().parse_args(
        _resume_from_update_argv("--resume-from-update", str(n), "--max-updates", "400"),
    )
    with pytest.raises((ValueError, control.RunStateError), match="50"):
        control.resolve_resume_from_stop_budget(args, n)


@pytest.mark.parametrize("module", [control, aux])
def test_resume_from_update_rejects_missing_stop_flags(module) -> None:
    args = module.build_parser().parse_args(
        _resume_from_update_argv("--resume", "--resume-from-update", "200"),
    )
    with pytest.raises((ValueError, control.RunStateError), match="stop"):
        control.resolve_resume_from_stop_budget(args, 200)


@pytest.mark.parametrize("module", [control, aux])
def test_resume_from_update_rejects_max_updates_not_strictly_greater_than_n(module) -> None:
    args = module.build_parser().parse_args(
        _resume_from_update_argv("--resume", "--resume-from-update", "200", "--max-updates", "200"),
    )
    with pytest.raises((ValueError, control.RunStateError), match="max-updates"):
        control.resolve_resume_from_stop_budget(args, 200)


@pytest.mark.parametrize("module", [control, aux])
def test_resume_from_update_end_time_only_uses_unbounded_max_updates(module) -> None:
    args = module.apply_mode_defaults(module.build_parser().parse_args(
        _resume_from_update_argv("--resume-from-update", "200", "--end-time", "20991231-2359"),
    ))
    target, deadline = control.resolve_resume_from_stop_budget(args, 200)
    assert target == control.UNBOUNDED_MAX_UPDATES
    assert deadline == args.end_time


@pytest.mark.parametrize("module", [control, aux])
def test_resume_from_update_max_only_uses_far_future_end_time(module) -> None:
    args = module.apply_mode_defaults(module.build_parser().parse_args(
        _resume_from_update_argv("--resume-from-update", "200", "--max-updates", "400"),
    ))
    target, deadline = control.resolve_resume_from_stop_budget(args, 200)
    assert target == 400
    assert deadline == control.FAR_FUTURE_END_TIME


def test_new_control_contract_has_schema_version_5(tmp_path: Path) -> None:
    contract = _current_contract(tool="phase3_ranked_ppo_retention", stop_args={
        "effective_max_updates": 1, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    assert json.loads((tmp_path / "run.json").read_text())["schema_version"] == 5
    assert control.CURRENT_RUN_SCHEMA_VERSION == 5


def test_new_aux_contract_has_schema_version_5(tmp_path: Path) -> None:
    contract = _current_contract(tool="phase3_ranked_ppo_retention_aux", arm="aux", stop_args={
        "effective_max_updates": 1, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    })
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    assert json.loads((tmp_path / "run.json").read_text())["schema_version"] == 5
    assert aux.ret_mod.CURRENT_RUN_SCHEMA_VERSION == 5


def test_recovery_archives_every_50_including_zero_and_periodic_model_stays_200() -> None:
    assert control.RECOVERY_ARCHIVE_INTERVAL == 50
    assert control.PERIODIC_MODEL_CHECKPOINT_INTERVAL == 200


@pytest.mark.parametrize("completed, force, due", [
    (0, False, False),
    (1, False, False),
    (10, False, False),
    (25, False, False),
    (37, False, False),
    (37, True, True),
    (49, False, False),
    (50, False, False),
    (50, True, True),
    (199, False, False),
    (200, False, False),
    (200, True, True),
])
def test_latest_recovery_due_cadence(completed: int, force: bool, due: bool) -> None:
    assert control.latest_recovery_due(completed, force=force) is due


@pytest.mark.parametrize("completed, due", [
    (0, True),
    (1, False),
    (10, False),
    (25, False),
    (49, False),
    (50, True),
    (100, True),
    (199, False),
    (200, True),
])
def test_archived_recovery_due_every_50_including_zero(completed: int, due: bool) -> None:
    assert control.archived_recovery_due(completed) is due


def test_control_sparse_latest_recovery_and_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_control_loop(monkeypatch)
    saves: list[tuple[str, int]] = []
    real = control.atomic_save_recovery

    def spy(path, state, **kwargs):
        saves.append((Path(path).name, int(state["completed_update"])))
        return real(path, state, **kwargs)

    monkeypatch.setattr(control, "atomic_save_recovery", spy)
    control.run_ppo_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=50),
        [0, 10, 25, 50, 100, 200], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    latest = [n for name, n in saves if name == "recovery.pt"]
    archives = [n for name, n in saves if name.startswith("update_")]
    assert 0 not in latest
    assert 1 not in latest
    assert 10 not in latest
    assert 25 not in latest
    assert 49 not in latest
    assert latest[-1] == 50
    assert 0 in archives
    assert 50 in archives
    assert 10 not in archives
    assert 25 not in archives
    jsonl = [json.loads(line) for line in (tmp_path / "ppo_updates.jsonl").read_text().splitlines()]
    assert jsonl[-1]["update"] == 50
    for row in jsonl:
        assert row["total_wall_s"] == row["collect_wall_s"] + row["ppo_wall_s"]
        assert row["collect_wall_s"] >= 0.0
        assert row["ppo_wall_s"] >= 0.0


def test_control_terminal_force_saves_off_cadence_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_control_loop(monkeypatch)
    control.run_ppo_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=10),
        [0, 10, 25, 50, 100, 200], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    state = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    assert state["completed_update"] == 10
    assert (tmp_path / "recovery_archives" / "update_0.pt").is_file()
    assert not (tmp_path / "recovery_archives" / "update_10.pt").exists()


def test_unlink_latest_recovery_removes_only_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "recovery.pt"
    control.unlink_latest_recovery(path)
    path.write_bytes(b"leftover")
    control.unlink_latest_recovery(path)
    assert not path.exists()


def test_recovery_pt_is_absent_during_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    present_during_collect: list[bool] = []
    (tmp_path / "recovery.pt").write_bytes(b"stale-latest")

    def fake_collect(net, device, seed, max_frames):
        present_during_collect.append((tmp_path / "recovery.pt").is_file())
        return _stub_rollout(seed)

    _stub_control_loop(monkeypatch)
    monkeypatch.setattr(control, "collect_rollout", fake_collect)
    control.run_ppo_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=3),
        [], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    assert present_during_collect
    assert not any(present_during_collect)
    assert (tmp_path / "recovery.pt").is_file()
    assert control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))["completed_update"] == 3


def test_sigint_does_not_write_recovery_pt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[int, Any] = {}
    real_signal = signal.signal

    def capturing(sig, handler):
        captured[sig] = handler
        return real_signal(sig, handler)

    monkeypatch.setattr(signal, "signal", capturing)
    updates: list[int] = []

    def fake_ppo(net, opt, rollouts, device):
        updates.append(1)
        if len(updates) == 2:
            captured[signal.SIGINT](signal.SIGINT, None)
        return {
            "optimizer_steps": 1, "approx_kl": 0.0, "clip_fraction": 0.0,
            "explained_variance": 0.0, "entropy": 0.0, "policy_loss": 0.0,
            "value_loss": 0.0, "total_loss": 0.0, "n_samples": 1,
        }

    monkeypatch.setattr(control, "collect_rollout", lambda *a, **k: _stub_rollout(0))
    monkeypatch.setattr(control, "ppo_update", fake_ppo)
    monkeypatch.setattr(
        control, "evaluate_deterministic",
        lambda *a, **k: [{"seed": 0, "elapsed": 1.0, "censored": True}],
    )
    result = control.run_ppo_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=10),
        [], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    assert result["stop_reason"] == "interrupted"
    assert result["completed_updates"] == 2
    assert not (tmp_path / "recovery.pt").exists()
    assert (tmp_path / "recovery_archives" / "update_0.pt").is_file()
    with pytest.raises(control.RunStateError, match="recovery.pt is missing"):
        control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))


def test_aux_resume_reuses_cached_first_update_collect_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_aux_loop(monkeypatch)
    train = {
        "player": torch.zeros(8, PLAYER_FEAT_V4),
        "bullets": torch.zeros(8, MAX_BULLETS_V4, BULLET_FEAT_V4),
        "pad": torch.ones(8, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.zeros(8, 5),
        "elapsed": torch.zeros(8),
    }
    aux.run_aux_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=0),
        [], train, None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
        minibatch=4,
    )
    resumed = aux.run_aux_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=1, resume=True),
        [], train, None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
        minibatch=4,
    )
    assert resumed["completed_updates"] == 1
    row = json.loads((tmp_path / "ppo_aux_updates.jsonl").read_text().splitlines()[0])
    assert row["collect_wall_s"] == 0.0
    assert row["total_wall_s"] == row["collect_wall_s"] + row["ppo_wall_s"]


def _stub_control_experiment_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        control, "load_initial_checkpoint",
        lambda *_: (torch.nn.Linear(1, 1), _STUB_INIT_META),
    )
    monkeypatch.setattr(
        control, "checkpoint_provenance",
        lambda *_: {"file_sha256": "init", "state_dict_sha256": "state"},
    )
    monkeypatch.setattr(control, "file_sha256", lambda *_: "teacher")
    monkeypatch.setattr(control, "current_git_commit", lambda *_: "commit")
    monkeypatch.setattr(control, "_git_dirty", lambda *_: False)


def test_schema_v1_is_rejected_on_normal_resume_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        out_dir=tmp_path, run_dir=tmp_path, init_checkpoint=Path("init.pt"),
        teacher=Path("teacher.pt"), updates=1, episodes_per_update=1,
        max_frames=1, eval_seeds=[0], eval_max_steps=1,
        initial_gate_mean_min=0.0, initial_gate_median_min=0.0,
        data_episodes=1, data_max_steps=1, data_frames_cap=1,
        end_time=None, max_updates=201, effective_seed=1, resume=True,
        resume_from_update=None, effective_max_updates=201,
        effective_end_time=control.FAR_FUTURE_END_TIME,
    )
    _stub_control_experiment_identity(monkeypatch)
    (tmp_path / "run.json").write_text(json.dumps({"schema_version": 1, "tool": "control"}))
    monkeypatch.setattr(
        control, "evaluate_deterministic",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("evaluation ran before schema rejection")),
    )
    with pytest.raises(control.RunStateError, match="schema_version"):
        control.run_experiment(args, torch.device("cpu"))


def test_schema_v1_is_rejected_on_resume_from_update_before_rewind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        out_dir=tmp_path, run_dir=tmp_path, init_checkpoint=Path("init.pt"),
        teacher=Path("teacher.pt"), updates=1, episodes_per_update=1,
        max_frames=1, eval_seeds=[0], eval_max_steps=1,
        initial_gate_mean_min=0.0, initial_gate_median_min=0.0,
        data_episodes=1, data_max_steps=1, data_frames_cap=1,
        end_time=None, max_updates=201, effective_seed=1, resume=True,
        resume_from_update=200, effective_max_updates=201,
        effective_end_time=control.FAR_FUTURE_END_TIME,
    )
    _stub_control_experiment_identity(monkeypatch)
    (tmp_path / "run.json").write_text(json.dumps({"schema_version": 1, "tool": "control"}))
    (tmp_path / "recovery.pt").write_bytes(b"latest-recovery")
    archive = tmp_path / "recovery_archives" / "update_200.pt"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive-200")
    monkeypatch.setattr(
        control, "evaluate_deterministic",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("evaluation ran before schema rejection")),
    )
    with pytest.raises(control.RunStateError, match="schema_version"):
        control.run_experiment(args, torch.device("cpu"))
    assert (tmp_path / "recovery.pt").read_bytes() == b"latest-recovery"
    assert archive.read_bytes() == b"archive-200"


def test_archived_recovery_path_for_update_200(tmp_path: Path) -> None:
    assert control.archived_recovery_path(tmp_path, 200) == tmp_path / "recovery_archives" / "update_200.pt"


def test_update_199_creates_no_recovery_archive(tmp_path: Path) -> None:
    control.save_archived_recovery_if_due(tmp_path, {
        "completed_update": 199, "model": {}, "optimizer": {}, "rng": control.capture_rng_state(),
    })
    assert not (tmp_path / "recovery_archives").exists()


def test_update_200_archive_matches_full_recovery_payload(tmp_path: Path) -> None:
    state = {
        "format": 1, "arm": "control", "completed_update": 200,
        "model": {"w": 3}, "optimizer": {"p": 4}, "total_frames": 9,
        "optimizer_steps": 7, "snapshots": [{"update": 200}],
        "rng": {"python": 1},
    }
    control.atomic_save_recovery(tmp_path / "recovery.pt", state)
    control.save_archived_recovery_if_due(tmp_path, state)
    archive = tmp_path / "recovery_archives" / "update_200.pt"
    assert archive.is_file()
    loaded = torch.load(archive, map_location="cpu", weights_only=False)
    latest = torch.load(tmp_path / "recovery.pt", map_location="cpu", weights_only=False)
    # The payload is identical; only the declared kind distinguishes the two.
    assert loaded["checkpoint_kind"] == "recovery_archive"
    assert latest["checkpoint_kind"] == "recovery_current"
    assert {k: v for k, v in loaded.items() if k != "checkpoint_kind"} == state
    assert {k: v for k, v in latest.items() if k != "checkpoint_kind"} == state


def test_archive_at_400_does_not_remove_update_200(tmp_path: Path) -> None:
    control.save_archived_recovery_if_due(tmp_path, {"completed_update": 200, "marker": "a200"})
    control.save_archived_recovery_if_due(tmp_path, {"completed_update": 400, "marker": "a400"})
    first = torch.load(tmp_path / "recovery_archives" / "update_200.pt", map_location="cpu", weights_only=False)
    second = torch.load(tmp_path / "recovery_archives" / "update_400.pt", map_location="cpu", weights_only=False)
    assert first["marker"] == "a200"
    assert second["marker"] == "a400"


def test_control_update_200_archive_includes_scheduled_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_control_loop(monkeypatch)
    net = _tiny_ranked()
    control.run_ppo_arm(
        net, torch.device("cpu"), _arm_args(tmp_path, max_updates=200), [200], None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    archive = torch.load(
        tmp_path / "recovery_archives" / "update_200.pt", map_location="cpu", weights_only=False,
    )
    latest = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    assert archive["completed_update"] == latest["completed_update"] == 200
    assert any(item["update"] == 200 for item in archive["snapshots"])
    assert archive["snapshots"] == latest["snapshots"]
    assert set(archive) >= {"model", "optimizer", "completed_update", "total_frames", "optimizer_steps", "snapshots", "rng"}


def test_aux_update_200_archive_includes_aux_recovery_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_aux_loop(monkeypatch)
    net = _tiny_ranked()
    train = {
        "player": torch.zeros(8, PLAYER_FEAT_V4),
        "bullets": torch.zeros(8, MAX_BULLETS_V4, BULLET_FEAT_V4),
        "pad": torch.ones(8, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.zeros(8, 5),
        "elapsed": torch.zeros(8),
    }
    aux.run_aux_arm(
        net, torch.device("cpu"), _arm_args(tmp_path, max_updates=200), [200], train, None, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
        minibatch=4,
    )
    archive = torch.load(
        tmp_path / "recovery_archives" / "update_200.pt", map_location="cpu", weights_only=False,
    )
    latest = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    for key in (
        "alpha", "calibration", "training_generator_state",
        "calibration_generator_state", "diagnostic_generator_state",
        "model", "optimizer", "completed_update", "rng",
    ):
        assert key in archive
        assert key in latest
    assert archive["completed_update"] == 200
    assert any(item["update"] == 200 for item in archive["snapshots"])


def test_control_periodic_model_only_cadence_preserves_diagnostics_and_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_control_loop(monkeypatch)
    writes: list[str] = []
    evaluations: list[int] = []
    diagnostics: list[int] = []
    save = control.save_player_checkpoint

    def record_save(net, path, **kwargs):
        writes.append(Path(path).name)
        return save(net, path, **kwargs)

    monkeypatch.setattr(control, "save_player_checkpoint", record_save)
    monkeypatch.setattr(
        control, "evaluate_deterministic",
        lambda *_args, **_kwargs: evaluations.append(1) or [{"seed": 0, "elapsed": 1.0, "censored": False}],
    )
    monkeypatch.setattr(
        control, "teacher_diagnostics", lambda *_args, **_kwargs: diagnostics.append(1) or {"agreement": 1.0},
    )
    result = control.run_ppo_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=601), [200], {"held": torch.zeros(1)}, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    assert not (tmp_path / "ppo_update_199.pt").exists()
    assert writes.count("ppo_update_200.pt") == 1
    for update in (400, 600):
        assert (tmp_path / f"ppo_update_{update}.pt").is_file()
        assert (tmp_path / "recovery_archives" / f"update_{update}.pt").is_file()
    assert (tmp_path / "ppo_update_400.pt").is_file()
    assert [item["update"] for item in result["snapshots"]] == [200, 601]
    assert len(evaluations) == 5
    assert len(diagnostics) == 2


def test_control_terminal_400_is_periodic_model_only_not_a_diagnostic_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_control_loop(monkeypatch)
    diagnostics: list[int] = []
    monkeypatch.setattr(
        control, "teacher_diagnostics", lambda *_args, **_kwargs: diagnostics.append(1) or {"agreement": 1.0},
    )
    result = control.run_ppo_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=400), control.snapshot_schedule(400),
        {"held": torch.zeros(1)}, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={},
    )
    checkpoint = torch.load(tmp_path / "ppo_update_400.pt", map_location="cpu", weights_only=False)
    assert checkpoint["extra"]["checkpoint_kind"] == "periodic_model_only"
    assert all(item["update"] != 400 for item in result["snapshots"])
    assert len(diagnostics) == 6


def test_aux_periodic_model_only_cadence_preserves_diagnostics_and_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_aux_loop(monkeypatch)
    writes: list[str] = []
    evaluations: list[int] = []
    diagnostics: list[int] = []
    save = aux.save_player_checkpoint

    def record_save(net, path, **kwargs):
        writes.append(Path(path).name)
        return save(net, path, **kwargs)

    monkeypatch.setattr(aux, "save_player_checkpoint", record_save)
    monkeypatch.setattr(
        aux, "evaluate_deterministic",
        lambda *_args, **_kwargs: evaluations.append(1) or [{"seed": 0, "elapsed": 1.0, "censored": False}],
    )
    monkeypatch.setattr(
        aux, "teacher_diagnostics", lambda *_args, **_kwargs: diagnostics.append(1) or {"agreement": 1.0},
    )
    monkeypatch.setattr(
        aux, "grad_alignment", lambda *_args, **_kwargs: {
            "g_ppo_norm": 1.0, "g_ret_norm": 1.0, "cosine_similarity": 0.0,
        },
    )
    train = {
        "player": torch.zeros(8, PLAYER_FEAT_V4),
        "bullets": torch.zeros(8, MAX_BULLETS_V4, BULLET_FEAT_V4),
        "pad": torch.ones(8, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.zeros(8, 5),
        "elapsed": torch.zeros(8),
    }
    result = aux.run_aux_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=601), [200], train, train, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={}, minibatch=4,
    )
    assert not (tmp_path / "ppo_aux_update_199.pt").exists()
    assert writes.count("ppo_aux_update_200.pt") == 1
    for update in (400, 600):
        assert (tmp_path / f"ppo_aux_update_{update}.pt").is_file()
        assert (tmp_path / "recovery_archives" / f"update_{update}.pt").is_file()
    assert (tmp_path / "ppo_aux_update_400.pt").is_file()
    assert [item["update"] for item in result["snapshots"]] == [200, 601]
    assert len(evaluations) == 5
    assert len(diagnostics) == 2


def test_aux_terminal_400_is_periodic_model_only_not_a_diagnostic_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_aux_loop(monkeypatch)
    diagnostics: list[int] = []
    monkeypatch.setattr(
        aux, "teacher_diagnostics", lambda *_args, **_kwargs: diagnostics.append(1) or {"agreement": 1.0},
    )
    monkeypatch.setattr(
        aux, "grad_alignment", lambda *_args, **_kwargs: {
            "g_ppo_norm": 1.0, "g_ret_norm": 1.0, "cosine_similarity": 0.0,
        },
    )
    train = {
        "player": torch.zeros(8, PLAYER_FEAT_V4),
        "bullets": torch.zeros(8, MAX_BULLETS_V4, BULLET_FEAT_V4),
        "pad": torch.ones(8, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.zeros(8, 5),
        "elapsed": torch.zeros(8),
    }
    result = aux.run_aux_arm(
        _tiny_ranked(), torch.device("cpu"), _arm_args(tmp_path, max_updates=400), control.snapshot_schedule(400),
        train, train, tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d", ppo_knobs={}, minibatch=4,
    )
    checkpoint = torch.load(tmp_path / "ppo_aux_update_400.pt", map_location="cpu", weights_only=False)
    assert checkpoint["extra"]["checkpoint_kind"] == "periodic_model_only"
    assert all(item["update"] != 400 for item in result["snapshots"])
    assert len(diagnostics) == 6


def _synthetic_v2_run_through_400(tmp_path: Path) -> dict[str, Any]:
    """A v2 run with progress 1..400, archives 200/400, latest recovery 400, mixed amendments."""
    stop = {
        "effective_max_updates": 800, "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
    }
    contract = _current_contract(tool="control", stop_args=stop)
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    progress = tmp_path / "ppo_updates.jsonl"
    progress.write_text("".join(json.dumps({"update": n, "metric": n}) + "\n" for n in range(1, 401)))
    state_200 = {
        "format": 1, "arm": "control", "model": {"w": 200}, "optimizer": {"p": 200},
        "completed_update": 200, "total_frames": 200, "optimizer_steps": 200,
        "snapshots": [{"update": 200}], "rng": control.capture_rng_state(),
    }
    state_400 = {
        "format": 1, "arm": "control", "model": {"w": 400}, "optimizer": {"p": 400},
        "completed_update": 400, "total_frames": 400, "optimizer_steps": 400,
        "snapshots": [{"update": 200}, {"update": 400}], "rng": control.capture_rng_state(),
    }
    control.atomic_save_recovery(control.archived_recovery_path(tmp_path, 200), state_200)
    control.atomic_save_recovery(control.archived_recovery_path(tmp_path, 400), state_400)
    control.atomic_save_recovery(tmp_path / "recovery.pt", state_400)
    amendments = [
        {
            "event": "stop_budget_extended",
            "prior": stop,
            "new": {"effective_max_updates": 600, "effective_end_time_hkt": stop["effective_end_time_hkt"]},
            "completed_update": 100,
            "at_hkt": "2026-09-15T08:00:00+08:00",
            "runtime_provenance": {},
        },
        {
            "event": "stop_budget_extended",
            "prior": {"effective_max_updates": 600, "effective_end_time_hkt": stop["effective_end_time_hkt"]},
            "new": {"effective_max_updates": 800, "effective_end_time_hkt": stop["effective_end_time_hkt"]},
            "completed_update": 300,
            "at_hkt": "2026-09-15T09:00:00+08:00",
            "runtime_provenance": {},
        },
    ]
    (tmp_path / "stop_budget_amendments.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in amendments)
    )
    return {"state_200": state_200, "state_400": state_400, "amendments": amendments}


def test_rewind_to_archived_recovery_200_restores_and_prunes(tmp_path: Path) -> None:
    fixture = _synthetic_v2_run_through_400(tmp_path)
    control.rewind_run_to_archived_recovery(
        tmp_path, 200, progress_filename="ppo_updates.jsonl", device=torch.device("cpu"),
    )
    latest = torch.load(tmp_path / "recovery.pt", map_location="cpu", weights_only=False)
    assert latest["completed_update"] == 200
    assert latest["model"] == fixture["state_200"]["model"]
    updates = [json.loads(line)["update"] for line in (tmp_path / "ppo_updates.jsonl").read_text().splitlines()]
    assert updates == list(range(1, 201))
    assert (tmp_path / "recovery_archives" / "update_200.pt").is_file()
    assert not (tmp_path / "recovery_archives" / "update_400.pt").exists()
    kept = [json.loads(line) for line in (tmp_path / "stop_budget_amendments.jsonl").read_text().splitlines()]
    assert [row["completed_update"] for row in kept] == [100]
    events = [json.loads(line)["event"] for line in (tmp_path / "status.jsonl").read_text().splitlines()]
    assert events[-1] == "rewound_to_archived_recovery"


def test_rewind_missing_archive_raises_before_destructive_changes(tmp_path: Path) -> None:
    _synthetic_v2_run_through_400(tmp_path)
    control.archived_recovery_path(tmp_path, 200).unlink()
    progress_before = (tmp_path / "ppo_updates.jsonl").read_text()
    recovery_before = (tmp_path / "recovery.pt").read_bytes()
    amendments_before = (tmp_path / "stop_budget_amendments.jsonl").read_text()
    with pytest.raises(control.RunStateError, match="archive"):
        control.rewind_run_to_archived_recovery(
            tmp_path, 200, progress_filename="ppo_updates.jsonl", device=torch.device("cpu"),
        )
    assert (tmp_path / "ppo_updates.jsonl").read_text() == progress_before
    assert (tmp_path / "recovery.pt").read_bytes() == recovery_before
    assert (tmp_path / "stop_budget_amendments.jsonl").read_text() == amendments_before
    assert (tmp_path / "recovery_archives" / "update_400.pt").is_file()


@pytest.mark.parametrize("n", [1, 199])
def test_rewind_rejects_non_archive_n_before_destructive_changes(tmp_path: Path, n: int) -> None:
    _synthetic_v2_run_through_400(tmp_path)
    progress_before = (tmp_path / "ppo_updates.jsonl").read_text()
    recovery_before = (tmp_path / "recovery.pt").read_bytes()
    with pytest.raises(control.RunStateError, match="50"):
        control.rewind_run_to_archived_recovery(
            tmp_path, n, progress_filename="ppo_updates.jsonl", device=torch.device("cpu"),
        )
    assert (tmp_path / "ppo_updates.jsonl").read_text() == progress_before
    assert (tmp_path / "recovery.pt").read_bytes() == recovery_before
    assert (tmp_path / "recovery_archives" / "update_400.pt").is_file()


def test_rewind_rejects_archive_completed_update_mismatch_before_destructive_changes(tmp_path: Path) -> None:
    _synthetic_v2_run_through_400(tmp_path)
    wrong = {
        "format": 1, "arm": "control", "model": {}, "optimizer": {},
        "completed_update": 199, "rng": control.capture_rng_state(),
    }
    control.atomic_save_recovery(control.archived_recovery_path(tmp_path, 200), wrong)
    progress_before = (tmp_path / "ppo_updates.jsonl").read_text()
    recovery_before = (tmp_path / "recovery.pt").read_bytes()
    with pytest.raises(control.RunStateError, match="completed_update"):
        control.rewind_run_to_archived_recovery(
            tmp_path, 200, progress_filename="ppo_updates.jsonl", device=torch.device("cpu"),
        )
    assert (tmp_path / "ppo_updates.jsonl").read_text() == progress_before
    assert (tmp_path / "recovery.pt").read_bytes() == recovery_before
    assert (tmp_path / "recovery_archives" / "update_400.pt").is_file()


def _control_knobs(args: argparse.Namespace) -> dict[str, Any]:
    knobs = control.ppo_hyperparameters()
    knobs["updates"] = args.updates
    knobs["episodes_per_update"] = args.episodes_per_update
    knobs["max_frames_per_episode"] = args.max_frames
    knobs["eval_seeds"] = args.eval_seeds
    knobs["eval_max_steps"] = args.eval_max_steps
    knobs["initial_gate_mean_min"] = args.initial_gate_mean_min
    knobs["initial_gate_median_min"] = args.initial_gate_median_min
    knobs["data_episodes"] = args.data_episodes
    knobs["data_max_steps"] = args.data_max_steps
    knobs["data_frames_cap"] = args.data_frames_cap
    return knobs


def _aux_knobs(args: argparse.Namespace) -> dict[str, Any]:
    knobs = _control_knobs(args)
    knobs["target_retention_grad_ratio"] = aux.TARGET_RETENTION_GRAD_RATIO
    knobs["retention_objective"] = "phase2_ranked_multiseed.hybrid_loss"
    knobs["retention_hard_weight"] = aux.HYBRID_HARD_WEIGHT
    knobs["retention_soft_weight"] = aux.HYBRID_SOFT_WEIGHT
    knobs["retention_temperature"] = aux.HYBRID_TEMPERATURE
    knobs["retention_sampler_seed"] = aux.RETENTION_SAMPLER_SEED
    knobs["retention_calibration_seed"] = aux.RETENTION_CALIBRATION_SEED
    knobs["retention_diagnostic_seed"] = aux.RETENTION_DIAGNOSTIC_SEED
    knobs["calibration_grad_samples"] = aux.CALIBRATION_GRAD_SAMPLES
    return knobs


def _write_matching_run_json(
    tmp_path: Path, args: argparse.Namespace, *, tool: str, arm: str, knobs: dict[str, Any],
    historical_max: int = 800,
) -> None:
    init_prov = {"file_sha256": "init", "state_dict_sha256": "state"}
    teacher_prov = prov.contract_teacher_inputs(prov.teacher_identity(args.teacher))
    provenance = {
        "init_checkpoint": init_prov, "teacher": teacher_prov,
        "git_commit": "commit", "dirty": False,
        "torch_version": torch.__version__, "device": "cpu",
    }
    contract = {
        "format": 1, "schema_version": control.CURRENT_RUN_SCHEMA_VERSION,
        "tool": tool, "arm": arm,
        "inputs": {"init_checkpoint": init_prov, "teacher": teacher_prov},
        "provenance": provenance, "knobs": knobs,
        "stop_args": {
            "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
            "effective_max_updates": historical_max,
        },
        "effective_seed": args.effective_seed, "no_promotion": True,
    }
    (tmp_path / "run.json").write_text(json.dumps(contract, indent=2, sort_keys=True))


def _orch_args(tmp_path: Path, *, resume_from_update: int = 200, max_updates: int = 201) -> argparse.Namespace:
    return argparse.Namespace(
        updates=200, episodes_per_update=1, max_frames=4, eval_seeds=[0], eval_max_steps=4,
        out_dir=tmp_path, run_dir=tmp_path, resume=False, seed=1,
        max_updates=max_updates, end_time=None, resume_from_update=resume_from_update,
        effective_max_updates=max_updates, effective_end_time=control.FAR_FUTURE_END_TIME,
        initial_gate_mean_min=0.0, initial_gate_median_min=0.0,
        data_episodes=1, data_max_steps=1, data_frames_cap=1,
        init_checkpoint=Path("init.pt"), teacher=_write_tiny_teacher(tmp_path), effective_seed=1,
    )


def _stub_control_experiment(monkeypatch: pytest.MonkeyPatch, net: PlayerRankedTopK) -> list[str]:
    order: list[str] = []
    _stub_control_loop(monkeypatch)
    monkeypatch.setattr(control, "load_initial_checkpoint", lambda *_: (net, _STUB_INIT_META))
    monkeypatch.setattr(control, "checkpoint_provenance", lambda *_: {"file_sha256": "init", "state_dict_sha256": "state"})
    monkeypatch.setattr(control, "file_sha256", lambda *_: "teacher")
    monkeypatch.setattr(control, "current_git_commit", lambda *_: "commit")
    monkeypatch.setattr(control, "_git_dirty", lambda *_: False)
    monkeypatch.setattr(control, "load_teacher", lambda *_: net)
    monkeypatch.setattr(control, "collect_canonical_dataset", lambda *_a, **_k: {
        "hash": "d", "n_episodes": 1, "n_held_episodes": 0, "n_train_episodes": 1,
        "n_held_frames": 0, "n_train_frames": 1, "teacher_file_sha256": "teacher",
        "held_tensors": None,
    })
    monkeypatch.setattr(control, "run_frozen_arm", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(control, "build_retention_reference", lambda snap: {"update": snap.get("update")})
    monkeypatch.setattr(control, "evaluate_retention", lambda *_a, **_k: {})
    real_schema = control._require_current_schema_version
    real_rewind = control.rewind_run_to_archived_recovery

    def spy_schema(contract, expected=control.CURRENT_RUN_SCHEMA_VERSION):
        order.append("schema")
        # Retention resume must still be checked against the retention schema,
        # not whatever version a decoupled tool happens to use.
        assert expected == control.CURRENT_RUN_SCHEMA_VERSION
        return real_schema(contract, expected)

    def spy_rewind(*a, **k):
        order.append("rewind")
        return real_rewind(*a, **k)

    def spy_eval(*a, **k):
        order.append("eval")
        return [{"seed": 0, "elapsed": 1.0, "censored": True}]

    monkeypatch.setattr(control, "_require_current_schema_version", spy_schema)
    monkeypatch.setattr(control, "rewind_run_to_archived_recovery", spy_rewind)
    monkeypatch.setattr(control, "evaluate_deterministic", spy_eval)
    return order


def _stub_aux_experiment(monkeypatch: pytest.MonkeyPatch, net: PlayerRankedTopK) -> list[str]:
    order: list[str] = []
    _stub_aux_loop(monkeypatch)
    monkeypatch.setattr(aux, "load_initial_checkpoint", lambda *_: (net, _STUB_INIT_META))
    monkeypatch.setattr(aux, "checkpoint_provenance", lambda *_: {"file_sha256": "init", "state_dict_sha256": "state"})
    monkeypatch.setattr(aux, "file_sha256", lambda *_: "teacher")
    monkeypatch.setattr(aux, "current_git_commit", lambda *_: "commit")
    monkeypatch.setattr(aux, "_git_dirty", lambda *_: False)
    monkeypatch.setattr(aux, "load_teacher", lambda *_: net)
    monkeypatch.setattr(aux, "collect_aux_dataset", lambda *_a, **_k: {
        "hash": "d", "n_episodes": 1, "n_held_episodes": 0, "n_train_episodes": 1,
        "n_held_frames": 0, "n_train_frames": 1, "teacher_file_sha256": "teacher",
        "held_tensors": None,
        "train_tensors": {
            "player": torch.zeros(8, PLAYER_FEAT_V4),
            "bullets": torch.zeros(8, MAX_BULLETS_V4, BULLET_FEAT_V4),
            "pad": torch.ones(8, MAX_BULLETS_V4, dtype=torch.bool),
            "teacher_logits": torch.zeros(8, 5),
            "elapsed": torch.zeros(8),
        },
    })
    monkeypatch.setattr(aux, "build_retention_reference", lambda snap: {"update": snap.get("update")})
    monkeypatch.setattr(aux, "evaluate_retention", lambda *_a, **_k: {})
    real_schema = control._require_current_schema_version
    real_rewind = control.rewind_run_to_archived_recovery

    def spy_schema(contract, expected=control.CURRENT_RUN_SCHEMA_VERSION):
        order.append("schema")
        assert expected == control.CURRENT_RUN_SCHEMA_VERSION
        return real_schema(contract, expected)

    def spy_rewind(*a, **k):
        order.append("rewind")
        return real_rewind(*a, **k)

    def spy_eval(*a, **k):
        order.append("eval")
        return [{"seed": 0, "elapsed": 1.0, "censored": True}]

    monkeypatch.setattr(control, "_require_current_schema_version", spy_schema)
    monkeypatch.setattr(control, "rewind_run_to_archived_recovery", spy_rewind)
    monkeypatch.setattr(aux, "evaluate_deterministic", spy_eval)
    return order


def _control_archive_state(net: PlayerRankedTopK, completed: int) -> dict[str, Any]:
    opt = torch.optim.Adam(net.parameters(), lr=control.PPO_LR, eps=1e-8)
    return {
        "format": 1, "arm": "control", "model": net.state_dict(), "optimizer": opt.state_dict(),
        "completed_update": completed, "total_frames": completed, "optimizer_steps": completed,
        "snapshots": [{"update": u} for u in (200,) if u <= completed],
        "rng": control.capture_rng_state(),
    }


def test_control_resume_from_update_rewinds_then_runs_n_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = _tiny_ranked()
    fixture = _synthetic_v2_run_through_400(tmp_path)
    state_200 = _control_archive_state(net, 200)
    control.atomic_save_recovery(control.archived_recovery_path(tmp_path, 200), state_200)
    args = _orch_args(tmp_path)
    _write_matching_run_json(
        tmp_path, args, tool="phase3_ranked_ppo_retention", arm="control", knobs=_control_knobs(args),
    )
    order = _stub_control_experiment(monkeypatch, net)
    report = control.run_experiment(args, torch.device("cpu"))
    assert "schema" in order and "rewind" in order and "eval" in order
    assert order.index("schema") < order.index("rewind") < order.index("eval")
    assert report["ppo_arm"]["completed_updates"] == 201
    assert report["stop_budget"]["effective_max_updates"] == 201
    assert report["stop_budget"]["effective_max_updates"] != fixture["state_400"]["completed_update"]
    latest = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    assert latest["completed_update"] == 201
    rows = [json.loads(line) for line in (tmp_path / "stop_budget_amendments.jsonl").read_text().splitlines()]
    assert rows[-1]["new"]["effective_max_updates"] == 201


def test_aux_resume_from_update_rewinds_then_runs_n_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = _tiny_ranked()
    args = _orch_args(tmp_path)
    _synthetic_v2_run_through_400(tmp_path)
    progress = tmp_path / "ppo_aux_updates.jsonl"
    progress.write_text("".join(json.dumps({"update": n}) + "\n" for n in range(1, 401)))
    state_200 = {
        "format": 1, "arm": "aux", "model": net.state_dict(),
        "optimizer": torch.optim.Adam(net.parameters()).state_dict(),
        "completed_update": 200, "total_episodes": 200, "total_frames": 200,
        "optimizer_steps": 200, "snapshots": [{"update": 200}], "alpha": 0.15,
        "calibration": {"alpha": 0.15}, "calib_seeds": [50000],
        "first_update_rollouts": [_stub_rollout(50000)],
        "last_update_rollouts": [_stub_rollout(50000)],
        "last_grad_alignment": {"g_ppo_norm": 1.0},
        "training_generator_state": aux.training_generator().get_state(),
        "calibration_generator_state": aux.calibration_generator().get_state(),
        "diagnostic_generator_state": aux.diagnostic_generator().get_state(),
        "rng": control.capture_rng_state(),
    }
    control.atomic_save_recovery(control.archived_recovery_path(tmp_path, 200), state_200)
    control.atomic_save_recovery(tmp_path / "recovery.pt", {**state_200, "completed_update": 400})
    _write_matching_run_json(
        tmp_path, args, tool="phase3_ranked_ppo_retention_aux", arm="aux", knobs=_aux_knobs(args),
    )
    order = _stub_aux_experiment(monkeypatch, net)
    report = aux.run_experiment(args, torch.device("cpu"))
    assert "schema" in order and "rewind" in order and "eval" in order
    assert order.index("schema") < order.index("rewind") < order.index("eval")
    assert report["aux_arm"]["completed_updates"] == 201
    assert report["stop_budget"]["effective_max_updates"] == 201
    latest = control.load_recovery(tmp_path / "recovery.pt", torch.device("cpu"))
    assert latest["completed_update"] == 201


def test_resume_from_update_ignores_historical_stop_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = _tiny_ranked()
    args = _orch_args(tmp_path, max_updates=201)
    _synthetic_v2_run_through_400(tmp_path)
    control.atomic_save_recovery(
        control.archived_recovery_path(tmp_path, 200), _control_archive_state(net, 200),
    )
    _write_matching_run_json(
        tmp_path, args, tool="phase3_ranked_ppo_retention", arm="control",
        knobs=_control_knobs(args), historical_max=800,
    )
    _stub_control_experiment(monkeypatch, net)
    report = control.run_experiment(args, torch.device("cpu"))
    assert report["stop_budget"]["effective_max_updates"] == 201
    assert report["ppo_arm"]["effective_max_updates"] == 201
    assert json.loads((tmp_path / "run.json").read_text())["stop_args"]["effective_max_updates"] == 800
