"""Tests for tools/phase2_ranked_multiseed.py (Phase 2 multi-seed ranked top-k
runner contract).

STRICT TDD RED PHASE: tools/phase2_ranked_multiseed.py does not exist yet.
Every test below therefore fails at collection with the same
``ModuleNotFoundError`` for ``tools.phase2_ranked_multiseed`` -- that is the
expected RED signal. No production code is added by this file.

Contract under specification:

* Collect the canonical V1 teacher dataset exactly ONCE (data seed 4; 240
  episodes; episode seeds starting at 20000; up to 4200 steps (70s) per
  episode; frames capped at 700/episode; 12% held-out by episode) and reuse
  the *exact same* train/held ``Frame`` objects for every train seed in
  ``TRAIN_SEEDS = (4, 5, 6)``.
* Model: ``qrokkun_env.agents.player_ranked_topk.PlayerRankedTopK`` with
  ``top_k=8`` and a hidden width auto-matched near a target trainable
  parameter count of 269050 (never silently reusing an unrelated --hidden).
* Objective: hybrid loss = 0.5 * hard (teacher-argmax) cross-entropy + 0.5 *
  soft cross-entropy at temperature 1.0.
* Training: 80 epochs, batch size 1024, lr 3e-4 (quick mode shrinks all of
  this for fast tests).
* Each train seed gets an independent, deterministic model initialization
  (same seed -> identical init; different seed -> different init) while the
  optimizer step count / batch-order policy is identical (equal) across
  seeds -- only the model init differs.
* Checkpoint selection per seed: held agreement, then lower held
  teacher->student KL, then latest -- ties are broken the same way as
  ``tools.phase2_capacity_controls.should_replace_selected``.
* Checkpoints are packed/validated through the STRICT production
  ``qrokkun_env.agents.player_checkpoints`` provenance path (never the
  loose experimental dict packing used by the other phase2 tools).
* Held-out metrics are reported per seed with bullet-density buckets.
* Closed-loop (deterministic argmax, built-in scripted env only, seeds
  3000..3029 inclusive, up to 4200 steps) is run for a seed ONLY IF that
  seed's held agreement >= 0.65.
* Aggregate gate (explicit pass/fail detail): every seed's held agreement
  >= 0.65; every seed that passed the held gate actually ran its closed
  loop; the combined closed-loop mean >= 25s and median >= 25s; and the
  mean of the per-seed train-seed means >= 30s.
* No PPO, no learned Spawner, no random-fraction control, no NAS, no
  promotion anywhere in this module.
* CLI supports --quick for a fast smoke mode.
* The report records shared-dataset identity/hash, all resolved
  hyperparameters, optimizer step counts, wall-clock runtime, git commit,
  torch version, and teacher checkpoint provenance.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qrokkun_env.agents.obs_v4 import (  # noqa: E402
    BULLET_FEAT_V4,
    MAX_BULLETS_V4,
    PAD_RADIUS,
    PLAYER_FEAT_V4,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402

# This import is expected to fail with ModuleNotFoundError -- that failure IS
# the RED signal this test file exists to pin down. Every test in this file
# depends on it, so every test fails the same way until the module exists.
from tools import phase2_ranked_multiseed as rm  # noqa: E402
from tools.phase2_distill_v1_to_v4 import Frame  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
# 1. pre-registered constants (the runner contract itself)
# --------------------------------------------------------------------------- #
def test_train_seeds_are_exactly_four_five_six() -> None:
    assert tuple(rm.TRAIN_SEEDS) == (4, 5, 6)


def test_hybrid_loss_weights_and_temperature() -> None:
    assert rm.HYBRID_HARD_WEIGHT == pytest.approx(0.5)
    assert rm.HYBRID_SOFT_WEIGHT == pytest.approx(0.5)
    assert rm.HYBRID_TEMPERATURE == pytest.approx(1.0)


def test_held_agreement_gate_threshold_is_065() -> None:
    assert rm.HELD_AGREEMENT_GATE_MIN == pytest.approx(0.65)


def test_closed_loop_seed_window_is_3000_through_3029() -> None:
    assert rm.CLOSED_LOOP_SEED_START == 3000
    assert rm.CLOSED_LOOP_NUM_SEEDS == 30
    seeds = rm.closed_loop_seed_list()
    assert seeds == list(range(3000, 3030))


def test_closed_loop_max_steps_is_4200() -> None:
    assert rm.CLOSED_LOOP_MAX_STEPS == 4200


def test_aggregate_thresholds() -> None:
    assert rm.AGGREGATE_CLOSED_LOOP_MEAN_MIN == pytest.approx(25.0)
    assert rm.AGGREGATE_CLOSED_LOOP_MEDIAN_MIN == pytest.approx(25.0)
    assert rm.AGGREGATE_TRAIN_SEED_MEAN_MIN == pytest.approx(30.0)


def test_target_hidden_param_count_near_269050() -> None:
    assert abs(rm.TARGET_HIDDEN_PARAM_COUNT - 269050) <= 1


def test_default_top_k_is_8() -> None:
    assert rm.DEFAULT_TOP_K == 8


# --------------------------------------------------------------------------- #
# 2. CLI / mode defaults
# --------------------------------------------------------------------------- #
def test_full_mode_defaults_match_pre_registered_spec(tmp_path: Path) -> None:
    args = full_args(tmp_path)
    assert args.data_seed == 4
    assert tuple(args.train_seeds) == (4, 5, 6)
    assert args.collect_episodes == 240
    assert args.collect_seed_start == 20000
    assert args.max_steps == 4200
    assert args.frames_per_episode_cap == 700
    assert args.held_out_frac == pytest.approx(0.12)
    assert args.top_k == 8
    assert args.epochs == 80
    assert args.batch_size == 1024
    assert args.lr == pytest.approx(3e-4)
    assert args.held_agreement_min == pytest.approx(0.65)
    assert args.closed_loop_seed_start == 3000
    assert args.closed_loop_num_seeds == 30
    assert args.closed_loop_max_steps == 4200


def test_quick_mode_shrinks_everything_for_a_fast_test(tmp_path: Path) -> None:
    q = quick_args(tmp_path)
    f = full_args(tmp_path)
    assert q.collect_episodes < f.collect_episodes
    assert q.epochs < f.epochs
    assert q.batch_size <= f.batch_size
    assert q.max_steps < f.max_steps
    assert q.closed_loop_num_seeds < f.closed_loop_num_seeds
    assert q.closed_loop_max_steps < f.closed_loop_max_steps
    # quick mode must not silently change the pre-registered seeds/weights
    assert tuple(q.train_seeds) == (4, 5, 6)
    assert q.data_seed == 4


def test_explicit_cli_override_beats_mode_defaults(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--epochs", "3", "--hidden", "48"])
    assert args.epochs == 3
    assert args.hidden == 48


# --------------------------------------------------------------------------- #
# 3. hybrid loss
# --------------------------------------------------------------------------- #
def test_hybrid_loss_matches_manual_half_hard_half_soft_ce() -> None:
    torch.manual_seed(0)
    student = torch.randn(16, len(ACTIONS), requires_grad=False)
    teacher = torch.randn(16, len(ACTIONS))

    hard = torch.nn.functional.cross_entropy(student, teacher.argmax(-1))
    teacher_probs = torch.nn.functional.softmax(teacher / 1.0, dim=-1)
    student_logp = torch.nn.functional.log_softmax(student, dim=-1)
    soft = -(teacher_probs * student_logp).sum(-1).mean()
    expected = 0.5 * hard + 0.5 * soft

    got = rm.hybrid_loss(student, teacher)
    assert torch.allclose(got, expected, atol=1e-6)


def test_hybrid_loss_is_the_arm_objective_used_for_training() -> None:
    # A perfectly-matching student (huge correct logit) should have much
    # lower hybrid loss than a random one.
    teacher = torch.zeros(4, len(ACTIONS))
    teacher[:, 0] = 10.0
    good_student = torch.full((4, len(ACTIONS)), -10.0)
    good_student[:, 0] = 10.0
    bad_student = torch.zeros(4, len(ACTIONS))
    bad_student[:, -1] = 10.0
    assert rm.hybrid_loss(good_student, teacher) < rm.hybrid_loss(bad_student, teacher)


# --------------------------------------------------------------------------- #
# 4. hidden width auto-match near the target parameter count
# --------------------------------------------------------------------------- #
def test_resolve_hidden_auto_matches_target_param_count(tmp_path: Path) -> None:
    args = full_args(tmp_path)
    hidden, info = rm.resolve_hidden(args)
    assert isinstance(hidden, int) and hidden > 0
    net = PlayerRankedTopK(top_k=args.top_k, hidden=hidden)
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert param_count == info["param_count"]
    assert info["target"] == rm.TARGET_HIDDEN_PARAM_COUNT
    # "near" -- within a small relative tolerance, never an arbitrary hidden.
    assert info["rel_error"] <= 0.02
    assert info["source"] == "auto_matched"


def test_resolve_hidden_explicit_override_is_never_silently_replaced(tmp_path: Path) -> None:
    args = full_args(tmp_path, extra=["--hidden", "17"])
    hidden, info = rm.resolve_hidden(args)
    assert hidden == 17
    assert info["source"] == "explicit_override"


# --------------------------------------------------------------------------- #
# 5. shared dataset collected exactly once and reused verbatim per seed
# --------------------------------------------------------------------------- #
def test_shared_dataset_is_collected_once_and_reused_by_identity(tmp_path: Path, monkeypatch) -> None:
    args = quick_args(tmp_path)
    train_frames = make_frames(20, episodes=6, seed=1)
    held_frames = make_frames(6, episodes=6, seed=2)
    calls: list[int] = []

    def fake_collect(*_a, **_k):
        calls.append(1)
        return train_frames, held_frames

    monkeypatch.setattr(rm, "collect_dataset", fake_collect)
    device = torch.device("cpu")

    got_train, got_held, identity = rm.collect_shared_dataset(teacher=None, device=device, args=args)

    assert len(calls) == 1
    assert got_train is train_frames
    assert got_held is held_frames
    assert isinstance(identity, dict)
    assert isinstance(identity.get("hash"), str) and len(identity["hash"]) == 64
    assert identity["n_train"] == len(train_frames)
    assert identity["n_held"] == len(held_frames)

    # Exercising each train seed must reuse the exact SAME Frame objects
    # (identity, not just equal content) -- never re-collected per seed.
    for seed in rm.TRAIN_SEEDS:
        t2, h2 = rm.frames_for_seed(seed, got_train, got_held)
        assert t2 is got_train
        assert h2 is got_held


def test_dataset_identity_hash_is_deterministic_and_content_sensitive() -> None:
    frames_a = make_frames(10, episodes=3, seed=7)
    frames_b = make_frames(10, episodes=3, seed=7)
    frames_c = make_frames(10, episodes=3, seed=8)

    id_a1 = rm.dataset_identity(frames_a, frames_a[:2])
    id_a2 = rm.dataset_identity(frames_a, frames_a[:2])
    id_b = rm.dataset_identity(frames_b, frames_b[:2])
    id_c = rm.dataset_identity(frames_c, frames_c[:2])

    assert id_a1["hash"] == id_a2["hash"]
    # different rng seed for content -> should very likely (deterministically
    # here since RNG differs) produce a different hash
    assert id_a1["hash"] != id_c["hash"]
    assert isinstance(id_b["hash"], str)


# --------------------------------------------------------------------------- #
# 6. independent deterministic model init, equal optimizer steps/batch order
# --------------------------------------------------------------------------- #
def test_model_init_is_deterministic_per_seed_and_differs_across_seeds() -> None:
    net_a1 = rm.init_model(seed=4, top_k=8, hidden=32)
    net_a2 = rm.init_model(seed=4, top_k=8, hidden=32)
    net_b = rm.init_model(seed=5, top_k=8, hidden=32)

    sd_a1 = net_a1.state_dict()
    sd_a2 = net_a2.state_dict()
    sd_b = net_b.state_dict()

    for k in sd_a1:
        assert torch.equal(sd_a1[k], sd_a2[k])
    assert any(not torch.equal(sd_a1[k], sd_b[k]) for k in sd_a1)


def test_batch_order_policy_is_identical_across_train_seeds() -> None:
    # Only the model init should vary by seed; the batch-order / optimizer
    # step schedule must be identical (same generator/policy) for every
    # train seed so the arms are truly matched.
    order_4 = rm.make_batch_order(n=37, batch_size=8, args=None, seed_for_model=4)
    order_5 = rm.make_batch_order(n=37, batch_size=8, args=None, seed_for_model=5)
    for a, b in zip(order_4, order_5):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 7. per-seed training: selection rule, optimizer steps, checkpoint provenance
# --------------------------------------------------------------------------- #
def test_train_one_seed_uses_capacity_controls_selection_rule() -> None:
    from tools.phase2_capacity_controls import should_replace_selected

    assert rm.should_replace_selected is should_replace_selected


def test_train_one_seed_runs_tiny_synthetic_job_and_produces_strict_checkpoint(tmp_path: Path) -> None:
    from qrokkun_env.agents import player_checkpoints as pc

    train_frames = make_frames(24, episodes=4, seed=11)
    held_frames = make_frames(8, episodes=4, seed=12)
    args = quick_args(tmp_path, extra=["--epochs", "2", "--batch-size", "6", "--hidden", "24"])
    device = torch.device("cpu")

    result = rm.train_one_seed(
        seed=4,
        args=args,
        train_frames=train_frames,
        held_frames=held_frames,
        device=device,
        out_dir=tmp_path / "seed4",
    )

    assert result["seed"] == 4
    assert result["epochs_run"] == 2
    assert result["optimizer_steps"] > 0
    assert "held_metrics" in result and "agreement" in result["held_metrics"]
    assert "by_bullet_bucket" in result["held_metrics"]

    ckpt_path = Path(result["checkpoint"])
    assert ckpt_path.exists()
    net, meta = pc.load_ranked_top_k_checkpoint(ckpt_path, device)
    assert isinstance(net, PlayerRankedTopK)
    assert meta["production_compatible"] is True
    assert meta["architecture"] == "player_ranked_topk"
    assert meta["extra"]["seed"] == 4


def test_two_seeds_trained_on_identical_frames_yield_different_but_valid_checkpoints(tmp_path: Path) -> None:
    train_frames = make_frames(24, episodes=4, seed=21)
    held_frames = make_frames(8, episodes=4, seed=22)
    args = quick_args(tmp_path, extra=["--epochs", "2", "--batch-size", "6", "--hidden", "24"])
    device = torch.device("cpu")

    r4 = rm.train_one_seed(
        seed=4, args=args, train_frames=train_frames, held_frames=held_frames, device=device, out_dir=tmp_path / "s4"
    )
    r5 = rm.train_one_seed(
        seed=5, args=args, train_frames=train_frames, held_frames=held_frames, device=device, out_dir=tmp_path / "s5"
    )
    assert r4["optimizer_steps"] == r5["optimizer_steps"]

    sd4 = torch.load(r4["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    sd5 = torch.load(r5["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    assert any(not torch.equal(sd4[k], sd5[k]) for k in sd4)


# --------------------------------------------------------------------------- #
# 8. per-seed closed-loop gating (held agreement >= 0.65)
# --------------------------------------------------------------------------- #
def test_seed_passes_closed_loop_gate_uses_held_agreement_threshold() -> None:
    assert rm.seed_passes_closed_loop_gate({"agreement": 0.65}) is True
    assert rm.seed_passes_closed_loop_gate({"agreement": 0.649999}) is False
    assert rm.seed_passes_closed_loop_gate({"agreement": 0.9}) is True
    assert rm.seed_passes_closed_loop_gate({"agreement": 0.0}) is False


def test_closed_loop_only_runs_for_seeds_that_pass_the_held_gate(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_closed_loop(net, device, seeds, max_steps):
        calls.append(1)
        return [10.0 for _ in seeds]

    monkeypatch.setattr(rm, "closed_loop_for_seed", fake_closed_loop)

    seed_reports = {
        4: {"seed": 4, "held_metrics": {"agreement": 0.9}, "checkpoint": None, "net": PlayerRankedTopK(top_k=8, hidden=8)},
        5: {"seed": 5, "held_metrics": {"agreement": 0.1}, "checkpoint": None, "net": PlayerRankedTopK(top_k=8, hidden=8)},
    }
    args = quick_args(tmp_path)

    out = rm.run_closed_loop_stage(seed_reports, args, torch.device("cpu"))

    assert len(calls) == 1  # only seed 4 (agreement 0.9 >= 0.65) triggers the closed loop
    assert out[4]["closed_loop_ran"] is True
    assert out[5]["closed_loop_ran"] is False
    assert "closed_loop" in out[4]
    assert out[5].get("closed_loop") is None


# --------------------------------------------------------------------------- #
# 9. aggregate gate: explicit pass/fail detail
# --------------------------------------------------------------------------- #
def _seed_report(
    seed: int,
    agreement: float,
    ran: bool,
    times: list[float] | None,
    *,
    eval_seed_start: int = 9000,
) -> dict[str, Any]:
    rep: dict[str, Any] = {
        "seed": seed,
        "held_metrics": {"agreement": agreement},
        "closed_loop_ran": ran,
    }
    if times is not None:
        rep["closed_loop"] = {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "per_seed": {str(eval_seed_start + i): t for i, t in enumerate(times)},
            "n": len(times),
        }
    return rep


def test_aggregate_gate_passes_when_every_condition_is_met(tmp_path: Path) -> None:
    args = quick_args(tmp_path)
    reports = {
        4: _seed_report(4, 0.70, True, [30.0] * args.closed_loop_num_seeds),
        5: _seed_report(5, 0.80, True, [40.0] * args.closed_loop_num_seeds),
        6: _seed_report(6, 0.90, True, [50.0] * args.closed_loop_num_seeds),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert gate["gate_pass"] is True
    assert gate["all_held_pass"] is True
    assert gate["all_closed_loop_ran"] is True
    assert gate["min_seed_closed_loop_mean"] >= 25.0
    assert gate["min_seed_closed_loop_median"] >= 25.0
    assert gate["train_seed_mean_of_means"] >= 30.0
    assert set(gate["per_seed"]) == {"4", "5", "6"}
    for seed in (4, 5, 6):
        assert gate["per_seed"][str(seed)]["held_pass"] is True


def test_aggregate_gate_fails_when_one_seed_misses_held_threshold(tmp_path: Path) -> None:
    args = quick_args(tmp_path)
    reports = {
        4: _seed_report(4, 0.40, False, None),  # below 0.65 -> closed loop never ran
        5: _seed_report(5, 0.80, True, [40.0] * 5),
        6: _seed_report(6, 0.90, True, [50.0] * 5),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert gate["gate_pass"] is False
    assert gate["all_held_pass"] is False
    assert gate["per_seed"]["4"]["held_pass"] is False
    assert "reason" in gate and "4" in gate["reason"]


def test_aggregate_gate_fails_when_closed_loop_mean_or_median_below_25(tmp_path: Path) -> None:
    args = quick_args(tmp_path)
    reports = {
        4: _seed_report(4, 0.70, True, [10.0] * args.closed_loop_num_seeds),  # too slow
        5: _seed_report(5, 0.80, True, [40.0] * args.closed_loop_num_seeds),
        6: _seed_report(6, 0.90, True, [50.0] * args.closed_loop_num_seeds),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert gate["gate_pass"] is False
    assert gate["closed_loop_mean_pass"] is False or gate["closed_loop_median_pass"] is False


def test_aggregate_gate_fails_when_mean_of_train_seed_means_below_30(tmp_path: Path) -> None:
    args = quick_args(tmp_path)
    reports = {
        4: _seed_report(4, 0.70, True, [26.0] * args.closed_loop_num_seeds),
        5: _seed_report(5, 0.70, True, [26.0] * args.closed_loop_num_seeds),
        6: _seed_report(6, 0.70, True, [26.0] * args.closed_loop_num_seeds),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    # combined mean/median (26.0) clears 25, but mean-of-per-seed-means
    # (26.0) misses the stricter 30.0 bar.
    assert gate["train_seed_mean_of_means"] == pytest.approx(26.0)
    assert gate["train_seed_mean_of_means_pass"] is False
    assert gate["gate_pass"] is False


def test_aggregate_gate_requires_closed_loop_to_have_actually_run_for_every_passing_seed(tmp_path: Path) -> None:
    args = quick_args(tmp_path)
    # Held agreement passes for all three, but seed 6's closed loop was
    # (incorrectly) skipped -- the aggregate gate must still fail closed.
    reports = {
        4: _seed_report(4, 0.70, True, [40.0] * args.closed_loop_num_seeds),
        5: _seed_report(5, 0.80, True, [40.0] * args.closed_loop_num_seeds),
        6: _seed_report(6, 0.90, False, None),
    }
    gate = rm.evaluate_aggregate_gate(reports, args)
    assert gate["all_closed_loop_ran"] is False
    assert gate["gate_pass"] is False


# --------------------------------------------------------------------------- #
# 10. no PPO / learned Spawner / random-fraction / NAS / promotion anywhere
# --------------------------------------------------------------------------- #
def test_module_never_references_forbidden_mechanisms() -> None:
    import inspect

    source = inspect.getsource(rm)
    forbidden = ["PPO", "learned_spawner", "LearnedSpawner", "random_fraction", "nas_", "promote_to_production", "NAS"]
    lowered = source
    for term in forbidden:
        assert term not in lowered, f"forbidden mechanism {term!r} referenced in tools/phase2_ranked_multiseed.py"


# --------------------------------------------------------------------------- #
# 11. full report: provenance, hyperparameters, steps, runtime
# --------------------------------------------------------------------------- #
def test_run_multiseed_report_carries_full_provenance(tmp_path: Path, monkeypatch) -> None:
    train_frames = make_frames(24, episodes=4, seed=31)
    held_frames = make_frames(8, episodes=4, seed=32)
    args = quick_args(tmp_path, extra=["--epochs", "1", "--batch-size", "6", "--hidden", "24"])
    Path(args.teacher).write_bytes(b"synthetic teacher provenance")
    device = torch.device("cpu")

    def fake_closed_loop(net, device, seeds, max_steps):
        return [{"elapsed": 30.0, "censored": False} for _ in seeds]

    monkeypatch.setattr(rm, "closed_loop_for_seed", fake_closed_loop)

    report = rm.run_multiseed(args, train_frames, held_frames, device)

    assert set(report["seeds"]) == {4, 5, 6}
    assert "dataset_identity" in report
    assert report["dataset_identity"]["n_train"] == len(train_frames)
    assert report["dataset_identity"]["n_held"] == len(held_frames)
    assert "params" in report
    assert report["params"]["epochs"] == args.epochs
    assert report["params"]["batch_size"] == args.batch_size
    assert report["params"]["lr"] == pytest.approx(args.lr)
    assert report["params"]["hidden"] == args.hidden
    assert report["params"]["top_k"] == args.top_k
    assert "runtime_seconds" in report and report["runtime_seconds"] >= 0
    assert "git_commit" in report  # may be None outside a git checkout, but key must exist
    assert "torch_version" in report and report["torch_version"] == torch.__version__
    assert "teacher" in report and str(args.teacher) in report["teacher"]
    for seed in (4, 5, 6):
        assert report["seeds"][seed]["optimizer_steps"] > 0
    assert "gate" in report


# --------------------------------------------------------------------------- #
# 12. sanity: importing this module never touches PPO/NAS/random-fraction/etc.
# --------------------------------------------------------------------------- #
def test_module_exposes_the_documented_public_contract() -> None:
    required_names = {
        "TRAIN_SEEDS",
        "HYBRID_HARD_WEIGHT",
        "HYBRID_SOFT_WEIGHT",
        "HYBRID_TEMPERATURE",
        "HELD_AGREEMENT_GATE_MIN",
        "CLOSED_LOOP_SEED_START",
        "CLOSED_LOOP_NUM_SEEDS",
        "CLOSED_LOOP_MAX_STEPS",
        "AGGREGATE_CLOSED_LOOP_MEAN_MIN",
        "AGGREGATE_CLOSED_LOOP_MEDIAN_MIN",
        "AGGREGATE_TRAIN_SEED_MEAN_MIN",
        "TARGET_HIDDEN_PARAM_COUNT",
        "DEFAULT_TOP_K",
        "build_parser",
        "apply_mode_defaults",
        "hybrid_loss",
        "resolve_hidden",
        "collect_shared_dataset",
        "collect_dataset",
        "frames_for_seed",
        "dataset_identity",
        "init_model",
        "make_batch_order",
        "should_replace_selected",
        "train_one_seed",
        "seed_passes_closed_loop_gate",
        "closed_loop_for_seed",
        "closed_loop_seed_list",
        "run_closed_loop_stage",
        "evaluate_aggregate_gate",
        "run_multiseed",
        "main",
    }
    missing = required_names - set(dir(rm))
    assert not missing, f"tools/phase2_ranked_multiseed.py is missing: {sorted(missing)}"
