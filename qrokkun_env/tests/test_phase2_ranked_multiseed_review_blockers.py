"""Regression tests for review blockers B1-B8 in tools/phase2_ranked_multiseed.py
and qrokkun_env/agents/player_checkpoints.py.

Each test below is a targeted RED-then-GREEN regression for one blocker:

* B1: per-seed epochs.jsonl with every epoch's held metrics/selection candidate.
* B2: args.held_agreement_min honored consistently (per-seed gate + aggregate gate).
* B3: canonical teacher file SHA-256 recorded in the aggregate report and every checkpoint.
* B4: complete checkpoint/report provenance (commit, runtime versions, actions,
      obs dims, dataset identity hash, selected epoch/held metrics).
* B5: production gate requires the exact canonical evaluation seed window and
      full n/per_seed coverage; timeouts/censoring are reported separately from
      death; quick mode never claims production_reproducibility_gate.
* B6: empty held data is rejected with a clear error before training/gating,
      never a bare TypeError.
* B7: hidden_info source ("auto_matched" vs "explicit_override") survives into
      the per-seed report even when run_multiseed resolves it once up front.
* B8: min-across-seed gate fields are unambiguously named, all_closed_loop_ran
      also reflects incomplete per-seed coverage, no args-is-None dead path,
      and gate per_seed keys serialize consistently as strings.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PAD_RADIUS, PLAYER_FEAT_V4  # noqa: E402
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.agents.player_v1 import PlayerV1  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402
from qrokkun_env.obs import OBS_DIM  # noqa: E402
from tools import phase2_ranked_multiseed as rm  # noqa: E402
from tools.phase2_distill_v1_to_v4 import Frame  # noqa: E402


def make_frames(n: int, *, episodes: int = 4, seed: int = 0) -> list[Frame]:
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    for i in range(n):
        n_live = int(rng.integers(0, 20))
        bullets = np.zeros((MAX_BULLETS_V4, BULLET_FEAT_V4), dtype=np.float32)
        pad = np.ones((MAX_BULLETS_V4,), dtype=np.bool_)
        bullets[:, 4] = PAD_RADIUS
        if n_live:
            bullets[:n_live] = rng.normal(size=(n_live, BULLET_FEAT_V4)).astype(np.float32)
            pad[:n_live] = False
        frames.append(
            Frame(
                player=rng.normal(size=(PLAYER_FEAT_V4,)).astype(np.float32),
                bullets=bullets,
                pad=pad,
                teacher_logits=rng.normal(size=(len(ACTIONS),)).astype(np.float32),
                elapsed=float(i % 60),
                episode=i % max(episodes, 1),
            )
        )
    return frames


def write_dummy_teacher(path: Path) -> bytes:
    data = b"not a real checkpoint but a stable byte string for hashing"
    path.write_bytes(data)
    return data


def write_real_teacher(path: Path, *, hidden: int = 8) -> None:
    torch.manual_seed(0)
    net = PlayerV1(hidden=hidden)
    torch.save(
        {
            "obs_dim": OBS_DIM,
            "actions": list(ACTIONS),
            "hidden": hidden,
            "state_dict": net.state_dict(),
        },
        path,
    )


def quick_args(tmp_path: Path, extra: list[str] | None = None):
    argv = [
        "--teacher",
        str(tmp_path / "fake_teacher.pt"),
        "--out-dir",
        str(tmp_path / "out"),
        "--quick",
        *(extra or []),
    ]
    args = rm.build_parser().parse_args(argv)
    return rm.apply_mode_defaults(args)


def full_args(tmp_path: Path, extra: list[str] | None = None):
    argv = [
        "--teacher",
        str(tmp_path / "fake_teacher.pt"),
        "--out-dir",
        str(tmp_path / "out"),
        *(extra or []),
    ]
    args = rm.build_parser().parse_args(argv)
    return rm.apply_mode_defaults(args)


# --------------------------------------------------------------------------- #
# B1: per-seed epochs.jsonl
# --------------------------------------------------------------------------- #
def test_train_one_seed_writes_epochs_jsonl_with_one_record_per_epoch(tmp_path: Path) -> None:
    train_frames = make_frames(24, episodes=4, seed=11)
    held_frames = make_frames(8, episodes=4, seed=12)
    args = quick_args(tmp_path, extra=["--epochs", "3", "--batch-size", "6", "--hidden", "24"])
    device = torch.device("cpu")

    result = rm.train_one_seed(
        seed=4, args=args, train_frames=train_frames, held_frames=held_frames, device=device, out_dir=tmp_path / "s4"
    )

    epochs_path = Path(result["checkpoint"]).parent / "epochs.jsonl"
    assert epochs_path.exists()
    lines = epochs_path.read_text().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    for i, rec in enumerate(records, start=1):
        assert rec["epoch"] == i
        assert "held_metrics" in rec and "agreement" in rec["held_metrics"]
        assert "candidate" in rec
        assert "selected" in rec and isinstance(rec["selected"], bool)
    # at least one epoch must be recorded as selected (the checkpointed one)
    assert any(rec["selected"] for rec in records)


# --------------------------------------------------------------------------- #
# B2: args.held_agreement_min honored consistently
# --------------------------------------------------------------------------- #
def test_seed_passes_closed_loop_gate_honors_explicit_threshold() -> None:
    assert rm.seed_passes_closed_loop_gate({"agreement": 0.70}, threshold=0.65) is True
    assert rm.seed_passes_closed_loop_gate({"agreement": 0.70}, threshold=0.95) is False


def test_run_closed_loop_stage_honors_args_held_agreement_min(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_closed_loop(net, device, seeds, max_steps):
        calls.append(1)
        return [{"elapsed": 10.0, "censored": False} for _ in seeds]

    monkeypatch.setattr(rm, "closed_loop_for_seed", fake_closed_loop)
    args = quick_args(tmp_path, extra=["--held-agreement-min", "0.95"])
    seed_reports = {
        4: {"seed": 4, "held_metrics": {"agreement": 0.70}, "checkpoint": None, "net": PlayerRankedTopK(top_k=8, hidden=8)},
    }
    out = rm.run_closed_loop_stage(seed_reports, args, torch.device("cpu"))
    assert len(calls) == 0  # 0.70 < 0.95 -- must not clear the raised bar
    assert out[4]["closed_loop_ran"] is False


def test_evaluate_aggregate_gate_honors_args_held_agreement_min(tmp_path: Path) -> None:
    from qrokkun_env.tests.test_phase2_ranked_multiseed import _seed_report

    args = quick_args(tmp_path, extra=["--held-agreement-min", "0.95", "--closed-loop-num-seeds", "5"])
    reports = {
        4: _seed_report(4, 0.70, True, [30.0] * 5),
        5: _seed_report(5, 0.80, True, [40.0] * 5),
        6: _seed_report(6, 0.90, True, [50.0] * 5),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    # every held agreement (0.70/0.80/0.90) is below the raised 0.95 bar.
    assert gate["all_held_pass"] is False
    assert gate["gate_pass"] is False


# --------------------------------------------------------------------------- #
# B3: canonical teacher file SHA-256 in the aggregate report and every checkpoint
# --------------------------------------------------------------------------- #
def test_run_multiseed_report_records_teacher_sha256(tmp_path: Path, monkeypatch) -> None:
    teacher_path = tmp_path / "fake_teacher.pt"
    teacher_bytes = write_dummy_teacher(teacher_path)
    expected_hash = hashlib.sha256(teacher_bytes).hexdigest()

    train_frames = make_frames(24, episodes=4, seed=31)
    held_frames = make_frames(8, episodes=4, seed=32)
    args = quick_args(tmp_path, extra=["--epochs", "1", "--batch-size", "6", "--hidden", "24"])
    device = torch.device("cpu")

    def fake_closed_loop(net, device, seeds, max_steps):
        return [{"elapsed": 30.0, "censored": False} for _ in seeds]

    monkeypatch.setattr(rm, "closed_loop_for_seed", fake_closed_loop)

    report = rm.run_multiseed(args, train_frames, held_frames, device)

    assert report["teacher_sha256"] == expected_hash
    for seed in (4, 5, 6):
        ckpt = torch.load(report["seeds"][seed]["checkpoint"], map_location="cpu", weights_only=False)
        assert ckpt["extra"]["teacher_sha256"] == expected_hash


# --------------------------------------------------------------------------- #
# B4: complete checkpoint/report provenance
# --------------------------------------------------------------------------- #
def test_pack_player_checkpoint_records_runtime_provenance() -> None:
    from qrokkun_env.agents import player_checkpoints as pc

    net = PlayerRankedTopK(top_k=8, hidden=16)
    ckpt = pc.pack_player_checkpoint(net, source_commit="deadbeef", source_tool="unit-test")
    src = ckpt["source"]
    assert src["python_version"] == __import__("platform").python_version()
    assert src["numpy_version"] == np.__version__
    assert src["device"] == "cpu"


def test_run_multiseed_report_records_full_runtime_and_dataset_provenance(tmp_path: Path, monkeypatch) -> None:
    write_dummy_teacher(tmp_path / "fake_teacher.pt")
    train_frames = make_frames(24, episodes=4, seed=41)
    held_frames = make_frames(8, episodes=4, seed=42)
    args = quick_args(tmp_path, extra=["--epochs", "1", "--batch-size", "6", "--hidden", "24"])
    device = torch.device("cpu")

    def fake_closed_loop(net, device, seeds, max_steps):
        return [{"elapsed": 30.0, "censored": False} for _ in seeds]

    monkeypatch.setattr(rm, "closed_loop_for_seed", fake_closed_loop)
    report = rm.run_multiseed(args, train_frames, held_frames, device)

    assert report["python_version"] == __import__("platform").python_version()
    assert report["numpy_version"] == np.__version__
    assert report["device"] == str(device)
    assert report["actions"] == list(ACTIONS)
    assert report["observation"]["player_feat"] == PLAYER_FEAT_V4
    assert report["observation"]["bullet_feat"] == BULLET_FEAT_V4
    assert report["observation"]["max_bullets"] == MAX_BULLETS_V4
    assert report["dataset_identity"]["hash"]

    for seed in (4, 5, 6):
        rep = report["seeds"][seed]
        assert rep["selected_epoch"] is not None
        assert rep["held_metrics"] is not None
        ckpt = torch.load(rep["checkpoint"], map_location="cpu", weights_only=False)
        assert ckpt["extra"]["dataset_identity_hash"] == report["dataset_identity"]["hash"]
        assert ckpt["extra"]["selected_epoch"] == rep["selected_epoch"]
        assert ckpt["extra"]["selected_held_metrics"]["agreement"] == pytest.approx(
            rep["held_metrics"]["agreement"]
        )


# --------------------------------------------------------------------------- #
# B5: canonical seed window, coverage, timeout/censoring vs death, run vs
#     canonical gate
# --------------------------------------------------------------------------- #
def test_closed_loop_for_seed_reports_censored_timeouts_separately_from_death() -> None:
    net = PlayerRankedTopK(top_k=8, hidden=8)
    device = torch.device("cpu")
    # max_steps=1 all but guarantees the scripted env has not finished yet.
    results = rm.closed_loop_for_seed(net, device, seeds=[3000], max_steps=1)
    assert len(results) == 1
    assert set(results[0]) == {"elapsed", "censored"}
    assert isinstance(results[0]["censored"], bool)


def test_evaluate_aggregate_gate_distinguishes_run_gate_from_canonical_gate(tmp_path: Path) -> None:
    from qrokkun_env.tests.test_phase2_ranked_multiseed import _seed_report

    # quick mode's own (non-canonical) window -- gate_pass may be True for the
    # configured run, but it must never be reported as the canonical
    # production_reproducibility_gate.
    args = quick_args(tmp_path, extra=["--closed-loop-num-seeds", "5"])
    reports = {
        4: _seed_report(4, 0.70, True, [30.0] * 5),
        5: _seed_report(5, 0.80, True, [40.0] * 5),
        6: _seed_report(6, 0.90, True, [50.0] * 5),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert gate["gate_pass"] is True
    assert gate["is_canonical_seed_window"] is False
    assert gate["production_reproducibility_gate"] is False

    full = full_args(tmp_path, extra=["--closed-loop-num-seeds", str(rm.CLOSED_LOOP_NUM_SEEDS)])
    gate_full = rm.evaluate_aggregate_gate(reports, full)
    assert gate_full["is_canonical_seed_window"] is False  # reports only have 5 closed-loop times, not 30

    reports_full = {
        4: _seed_report(4, 0.70, True, [30.0] * rm.CLOSED_LOOP_NUM_SEEDS, eval_seed_start=3000),
        5: _seed_report(5, 0.80, True, [40.0] * rm.CLOSED_LOOP_NUM_SEEDS, eval_seed_start=3000),
        6: _seed_report(6, 0.90, True, [50.0] * rm.CLOSED_LOOP_NUM_SEEDS, eval_seed_start=3000),
    }
    canonical_args = full_args(tmp_path)
    gate_canonical = rm.evaluate_aggregate_gate(reports_full, canonical_args)
    assert gate_canonical["is_canonical_seed_window"] is True
    assert gate_canonical["production_reproducibility_gate"] is True


def test_evaluate_aggregate_gate_requires_complete_per_seed_coverage(tmp_path: Path) -> None:
    from qrokkun_env.tests.test_phase2_ranked_multiseed import _seed_report

    args = quick_args(tmp_path, extra=["--closed-loop-num-seeds", "5"])
    reports = {
        4: _seed_report(4, 0.70, True, [30.0] * 3),  # only 3 of the configured 5 -- incomplete coverage
        5: _seed_report(5, 0.80, True, [40.0] * 5),
        6: _seed_report(6, 0.90, True, [50.0] * 5),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert gate["all_closed_loop_ran"] is False
    assert gate["gate_pass"] is False


# --------------------------------------------------------------------------- #
# B6: empty held data rejected cleanly, no TypeError
# --------------------------------------------------------------------------- #
def test_train_one_seed_rejects_empty_held_frames_cleanly(tmp_path: Path) -> None:
    train_frames = make_frames(10, episodes=2, seed=51)
    args = quick_args(tmp_path, extra=["--epochs", "1"])
    with pytest.raises(ValueError, match="held"):
        rm.train_one_seed(
            seed=4, args=args, train_frames=train_frames, held_frames=[], device=torch.device("cpu"), out_dir=tmp_path / "s4"
        )


def test_run_multiseed_rejects_empty_held_frames_cleanly(tmp_path: Path) -> None:
    train_frames = make_frames(10, episodes=2, seed=52)
    args = quick_args(tmp_path, extra=["--epochs", "1"])
    with pytest.raises(ValueError, match="held"):
        rm.run_multiseed(args, train_frames, [], torch.device("cpu"))


# --------------------------------------------------------------------------- #
# B7: hidden_info source survives run_multiseed's single resolution
# --------------------------------------------------------------------------- #
def test_run_multiseed_preserves_auto_matched_hidden_source_per_seed(tmp_path: Path, monkeypatch) -> None:
    write_dummy_teacher(tmp_path / "fake_teacher.pt")
    train_frames = make_frames(24, episodes=4, seed=61)
    held_frames = make_frames(8, episodes=4, seed=62)
    # No --hidden override: run_multiseed's own resolve_hidden call must be
    # auto_matched, and that source must reach every per-seed report too.
    args = quick_args(tmp_path, extra=["--epochs", "1", "--batch-size", "6"])
    device = torch.device("cpu")

    def fake_closed_loop(net, device, seeds, max_steps):
        return [{"elapsed": 30.0, "censored": False} for _ in seeds]

    monkeypatch.setattr(rm, "closed_loop_for_seed", fake_closed_loop)
    report = rm.run_multiseed(args, train_frames, held_frames, device)

    assert report["params"]["hidden_info"]["source"] == "auto_matched"
    for seed in (4, 5, 6):
        assert report["seeds"][seed]["hidden_info"]["source"] == "auto_matched"


# --------------------------------------------------------------------------- #
# B8: field naming, coverage-aware all_closed_loop_ran, no args=None dead
#     path, str seed keys
# --------------------------------------------------------------------------- #
def test_gate_min_across_seed_fields_are_clearly_named(tmp_path: Path) -> None:
    from qrokkun_env.tests.test_phase2_ranked_multiseed import _seed_report

    args = quick_args(tmp_path, extra=["--closed-loop-num-seeds", "5"])
    reports = {
        4: _seed_report(4, 0.70, True, [30.0] * 5),
        5: _seed_report(5, 0.80, True, [40.0] * 5),
        6: _seed_report(6, 0.90, True, [50.0] * 5),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert "min_seed_closed_loop_mean" in gate
    assert "min_seed_closed_loop_median" in gate
    assert "closed_loop_mean" not in gate
    assert "closed_loop_median" not in gate


def test_gate_per_seed_keys_serialize_consistently_as_strings(tmp_path: Path) -> None:
    from qrokkun_env.tests.test_phase2_ranked_multiseed import _seed_report

    args = quick_args(tmp_path, extra=["--closed-loop-num-seeds", "5"])
    reports = {
        4: _seed_report(4, 0.70, True, [30.0] * 5),
        5: _seed_report(5, 0.80, True, [40.0] * 5),
        6: _seed_report(6, 0.90, True, [50.0] * 5),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert set(gate["per_seed"]) == {"4", "5", "6"}
    assert all(isinstance(k, str) for k in gate["per_seed"])
    json.dumps(gate)  # must never rely on json.dumps' implicit int-key coercion


def test_run_closed_loop_stage_has_no_args_none_dead_path() -> None:
    import inspect

    source = inspect.getsource(rm.run_closed_loop_stage)
    assert "args is None" not in source


# --------------------------------------------------------------------------- #
# Real quick-mode end-to-end CLI test (no monkeypatching of main)
# --------------------------------------------------------------------------- #
def test_quick_cli_end_to_end_with_real_teacher_checkpoint(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.pt"
    write_real_teacher(teacher_path, hidden=8)
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "-m",
        "tools.phase2_ranked_multiseed",
        "--teacher",
        str(teacher_path),
        "--out-dir",
        str(out_dir),
        "--quick",
        "--train-seeds",
        "4",
        "--epochs",
        "1",
        "--collect-episodes",
        "2",
        "--held-agreement-min",
        "0.0",
    ]
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr

    report_path = out_dir / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    assert report["teacher_sha256"] == hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    assert report["params"]["held_agreement_min"] == pytest.approx(0.0)
    assert "production_reproducibility_gate" in report["gate"]
    # quick mode's window is never the canonical one -- it must never claim
    # the production reproducibility gate.
    assert report["gate"]["is_canonical_seed_window"] is False
    assert report["gate"]["production_reproducibility_gate"] is False
    assert set(report["gate"]["per_seed"]) == {"4"}

    seed4_dir = out_dir / "seed4"
    epochs_path = seed4_dir / "epochs.jsonl"
    assert epochs_path.exists()
    lines = epochs_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["epoch"] == 1
    assert "held_metrics" in rec

    ckpt_path = seed4_dir / "seed4.pt"
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["extra"]["teacher_sha256"] == report["teacher_sha256"]
    assert ckpt["extra"]["dataset_identity_hash"] == report["dataset_identity"]["hash"]
    assert ckpt["source"]["python_version"]
    assert ckpt["source"]["numpy_version"]
