"""Small CPU-only contracts for Phase 3 long-run/resume support."""

from __future__ import annotations

import sys
import json
import datetime as dt
import argparse
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import phase3_ranked_ppo_retention as control
from tools import phase3_ranked_ppo_retention_aux as aux


@pytest.mark.parametrize(
    "existing_contract, error",
    [(None, "run.json is missing"), ({"wrong": True}, "contract mismatch")],
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

    with pytest.raises(control.RunStateError, match=error):
        control.run_experiment(args, __import__("torch").device("cpu"))


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


def test_resume_contract_binds_effective_stop_budget(tmp_path: Path) -> None:
    """Changing a resolved default is as incompatible as changing a CLI value."""
    contract = {
        'stop_args': {
            'effective_max_updates': 2_147_483_647,
            'effective_end_time_hkt': '2026-09-15T23:59:00+08:00',
        },
    }
    control.create_or_validate_run_contract(tmp_path, contract, resume=False)
    changed = {
        'stop_args': {
            'effective_max_updates': 200,
            'effective_end_time_hkt': '2026-09-15T23:59:00+08:00',
        },
    }
    with pytest.raises(control.RunStateError, match='mismatch'):
        control.create_or_validate_run_contract(tmp_path, changed, resume=True)


def test_contract_is_immutable_and_recovery_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    contract = {"tool": "control", "inputs": {"checkpoint": "a"}, "no_promotion": True}
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
