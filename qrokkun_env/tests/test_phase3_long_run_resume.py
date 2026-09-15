"""Small CPU-only contracts for Phase 3 long-run/resume support."""

from __future__ import annotations

import sys
import json
import datetime as dt
import argparse
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4

from tools import phase3_ranked_ppo_retention as control
from tools import phase3_ranked_ppo_retention_aux as aux

_CURRENT_SCHEMA_VERSION = control.CURRENT_RUN_SCHEMA_VERSION


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


def test_progress_journal_recovers_a_crash_after_fsync_without_duplicate_row(tmp_path: Path) -> None:
    """A journal row durable before recovery is retried idempotently on resume."""
    progress = tmp_path / "ppo_updates.jsonl"
    progress.write_text(json.dumps({"update": 1, "metric": 3}) + "\n")

    pending = control.reconcile_progress_journal(progress, completed_update=0)
    assert pending == {1}
    assert control.append_progress_row(progress, {"update": 1, "metric": 3}) is False
    assert [json.loads(line)["update"] for line in progress.read_text().splitlines()] == [1]


@pytest.mark.parametrize("module, filename", [
    (control, "ppo_updates.jsonl"),
    (aux, "ppo_aux_updates.jsonl"),
])
def test_control_and_aux_progress_writers_are_idempotent_after_recovery_lag(
    tmp_path: Path, module, filename: str,
) -> None:
    """Both arms use the same append-only, recovery-aware progress protocol."""
    progress = tmp_path / filename
    progress.write_text(json.dumps({"update": 1}) + "\n")
    assert control.reconcile_progress_journal(progress, completed_update=0) == {1}
    assert module.append_progress_row(progress, {"update": 1}) is False
    assert module.append_progress_row(progress, {"update": 2}) is True
    assert [json.loads(line)["update"] for line in progress.read_text().splitlines()] == [1, 2]


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


@pytest.mark.parametrize('bad_version', [None, 0, 2, True, 1.0])
def test_resume_requires_exact_integer_current_schema_version(tmp_path: Path, bad_version: object) -> None:
    """Versionless, old, unsupported, bool, and float runs are archival-only."""
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
