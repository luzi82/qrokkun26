"""Tests for tools/phase2_matched_hybrid.py (Phase 2 matched architecture/objective
experiment with a gated closed-loop evaluation).

Never touches the NAS, never loads a real teacher checkpoint, and only runs a
closed-loop episode when explicitly exercising the gated closed-loop stage on
tiny synthetic seeds/models -- everything else stays a fast unit suite on
synthetic frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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
from qrokkun_env.agents.player_v4 import PlayerV4  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402
from tools import phase2_matched_hybrid as mh  # noqa: E402
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


def quick_args(tmp_path: Path, arms: str = "all", extra: list[str] | None = None):
    argv = [
        "--teacher",
        str(tmp_path / "fake_teacher.pt"),
        "--out-dir",
        str(tmp_path / "out"),
        "--arms",
        arms,
        "--quick",
        *(extra or []),
    ]
    args = mh.build_parser().parse_args(argv)
    return mh.apply_mode_defaults(args)


# --------------------------------------------------------------------------- #
# 1. arm catalogue
# --------------------------------------------------------------------------- #
def test_all_six_arms_are_declared_exactly() -> None:
    required = {
        mh.ARM_ATTN64_HARD,
        mh.ARM_ATTN64_SOFT_T1,
        mh.ARM_ATTN64_HYBRID,
        mh.ARM_ATTN8_HYBRID,
        mh.ARM_FLAT8_HARD,
        mh.ARM_FLAT8_HYBRID,
    }
    assert set(mh.ALL_ARMS) == required
    assert set(mh.ARM_SPECS) == required


def test_arm_specs_have_expected_model_objective_and_mask() -> None:
    specs = mh.ARM_SPECS
    assert specs[mh.ARM_ATTN64_HARD].model == mh.MODEL_ATTN64
    assert specs[mh.ARM_ATTN64_HARD].objective == mh.OBJ_HARD
    assert specs[mh.ARM_ATTN64_HARD].top_k_mask is False

    assert specs[mh.ARM_ATTN64_SOFT_T1].model == mh.MODEL_ATTN64
    assert specs[mh.ARM_ATTN64_SOFT_T1].objective == mh.OBJ_SOFT_T1
    assert specs[mh.ARM_ATTN64_SOFT_T1].top_k_mask is False

    assert specs[mh.ARM_ATTN64_HYBRID].model == mh.MODEL_ATTN64
    assert specs[mh.ARM_ATTN64_HYBRID].objective == mh.OBJ_HYBRID
    assert specs[mh.ARM_ATTN64_HYBRID].top_k_mask is False

    assert specs[mh.ARM_ATTN8_HYBRID].model == mh.MODEL_ATTN64
    assert specs[mh.ARM_ATTN8_HYBRID].objective == mh.OBJ_HYBRID
    assert specs[mh.ARM_ATTN8_HYBRID].top_k_mask is True

    assert specs[mh.ARM_FLAT8_HARD].model == mh.MODEL_FLAT8
    assert specs[mh.ARM_FLAT8_HARD].objective == mh.OBJ_HARD

    assert specs[mh.ARM_FLAT8_HYBRID].model == mh.MODEL_FLAT8
    assert specs[mh.ARM_FLAT8_HYBRID].objective == mh.OBJ_HYBRID


def test_resolve_arms_accepts_all_csv_and_rejects_unknown() -> None:
    assert mh.resolve_arms("all") == list(mh.ALL_ARMS)
    assert mh.resolve_arms(f"{mh.ARM_FLAT8_HARD},{mh.ARM_ATTN64_HARD}") == [mh.ARM_FLAT8_HARD, mh.ARM_ATTN64_HARD]
    with pytest.raises(SystemExit):
        mh.resolve_arms("nope")


# --------------------------------------------------------------------------- #
# 2. hybrid loss == 0.5 hard CE + 0.5 soft CE at T=1
# --------------------------------------------------------------------------- #
def test_hybrid_loss_equals_half_hard_plus_half_soft_t1() -> None:
    torch.manual_seed(0)
    student_logits = torch.randn(16, len(ACTIONS))
    teacher_logits = torch.randn(16, len(ACTIONS))

    hard = torch.nn.functional.cross_entropy(student_logits, teacher_logits.argmax(-1))
    teacher_probs = torch.nn.functional.softmax(teacher_logits / 1.0, dim=-1)
    student_logp = torch.nn.functional.log_softmax(student_logits, dim=-1)
    soft = -(teacher_probs * student_logp).sum(-1).mean()
    expected = 0.5 * hard + 0.5 * soft

    spec = mh.ARM_SPECS[mh.ARM_ATTN64_HYBRID]
    got = mh.arm_loss(spec, student_logits, teacher_logits)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_hard_objective_ignores_teacher_soft_distribution() -> None:
    torch.manual_seed(1)
    student_logits = torch.randn(8, len(ACTIONS))
    teacher_logits = torch.randn(8, len(ACTIONS))
    spec = mh.ARM_SPECS[mh.ARM_ATTN64_HARD]
    expected = torch.nn.functional.cross_entropy(student_logits, teacher_logits.argmax(-1))
    got = mh.arm_loss(spec, student_logits, teacher_logits)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_soft_t1_objective_uses_temperature_one() -> None:
    torch.manual_seed(2)
    student_logits = torch.randn(8, len(ACTIONS))
    teacher_logits = torch.randn(8, len(ACTIONS))
    spec = mh.ARM_SPECS[mh.ARM_ATTN64_SOFT_T1]
    teacher_probs = torch.nn.functional.softmax(teacher_logits / mh.SOFT_T1_TEMPERATURE, dim=-1)
    student_logp = torch.nn.functional.log_softmax(student_logits, dim=-1)
    expected = -(teacher_probs * student_logp).sum(-1).mean()
    got = mh.arm_loss(spec, student_logits, teacher_logits)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)
    assert mh.SOFT_T1_TEMPERATURE == 1.0


# --------------------------------------------------------------------------- #
# 3. flat8 parameter matching (<=2% of PlayerV4) -- auto-chosen, not global hidden
# --------------------------------------------------------------------------- #
def test_count_trainable_params_matches_manual_sum() -> None:
    net = PlayerV4(d_model=16, hidden=16)
    expected = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert mh.count_trainable_params(net) == expected


def test_flat8_param_count_is_monotonic_in_hidden() -> None:
    lo = mh.flat8_param_count(hidden=8, top_k=8)
    hi = mh.flat8_param_count(hidden=64, top_k=8)
    assert hi > lo


def test_choose_flat_hidden_matches_target_within_2_percent() -> None:
    target = mh.count_trainable_params(PlayerV4(d_model=128, hidden=256))
    result = mh.choose_flat_hidden(target=target, top_k=8)
    assert result["within_2pct"] is True
    assert result["rel_error"] <= 0.02
    achieved = mh.flat8_param_count(hidden=result["hidden"], top_k=8)
    assert achieved == result["param_count"]


def test_choose_flat_hidden_is_close_to_brute_force_best() -> None:
    target = mh.count_trainable_params(PlayerV4(d_model=16, hidden=16))
    result = mh.choose_flat_hidden(target=target, top_k=8, lo=1, hi=256)
    brute_best_hidden = min(range(1, 257), key=lambda h: abs(mh.flat8_param_count(h, 8) - target))
    brute_best = mh.flat8_param_count(brute_best_hidden, 8)
    assert abs(result["param_count"] - target) == abs(brute_best - target)


def test_flat8_hidden_is_auto_chosen_not_global_hidden(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--d-model", "128", "--hidden", "256"])
    flat_hidden, info = mh.resolve_flat8_hidden(args)
    assert flat_hidden != args.hidden
    assert info["target"] == mh.count_trainable_params(PlayerV4(d_model=args.d_model, hidden=args.hidden))
    assert info["hidden"] == flat_hidden


# --------------------------------------------------------------------------- #
# 4. make_model dispatch (attn64 vs flat8, including top-8 mask forward)
# --------------------------------------------------------------------------- #
def test_make_model_dispatches_attn64_and_flat8(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--d-model", "16", "--hidden", "16"])
    args.flat_hidden, _info = mh.resolve_flat8_hidden(args)
    attn_net = mh.make_model(mh.ARM_SPECS[mh.ARM_ATTN64_HARD], args, torch.device("cpu"))
    flat_net = mh.make_model(mh.ARM_SPECS[mh.ARM_FLAT8_HARD], args, torch.device("cpu"))
    assert isinstance(attn_net, PlayerV4)
    assert isinstance(flat_net, mh.FlatTop8MLP)
    assert flat_net.top_k == args.top_k


def test_forward_arm_applies_top_k_mask_only_for_masked_arms() -> None:
    torch.manual_seed(0)
    net = PlayerV4(d_model=16, hidden=16)
    player = torch.randn(4, PLAYER_FEAT_V4)
    bullets = torch.randn(4, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(4, MAX_BULLETS_V4, dtype=torch.bool)
    unmasked_spec = mh.ARM_SPECS[mh.ARM_ATTN64_HARD]
    masked_spec = mh.ARM_SPECS[mh.ARM_ATTN8_HYBRID]
    dist_u, _ = mh.forward_arm(net, unmasked_spec, player, bullets, pad, top_k=8)
    dist_m, _ = mh.forward_arm(net, masked_spec, player, bullets, pad, top_k=8)
    assert not torch.allclose(dist_u.logits, dist_m.logits)


# --------------------------------------------------------------------------- #
# 5. train_arm: identical epochs/batches/seed/optimizer_steps across arms,
#    param_count reporting, frame-level-only selection
# --------------------------------------------------------------------------- #
def test_train_arm_reports_param_count_and_optimizer_steps(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--epochs", "2", "--batch-size", "8"])
    args.flat_hidden, _info = mh.resolve_flat8_hidden(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_frames = make_frames(24, seed=9)
    held_frames = make_frames(12, seed=10)
    result = mh.train_arm(
        mh.ARM_SPECS[mh.ARM_ATTN64_HARD], args, train_frames, held_frames, torch.device("cpu"), out_dir
    )
    expected_steps = 2 * math.ceil(24 / 8)
    assert result["optimizer_steps"] == expected_steps
    assert result["param_count"] == mh.count_trainable_params(PlayerV4(d_model=args.d_model, hidden=args.hidden))


def test_all_non_tiny_arms_share_epochs_batches_seed_and_optimizer_steps(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--epochs", "2", "--batch-size", "8"])
    train_frames = make_frames(24, seed=9)
    held_frames = make_frames(12, seed=10)
    report = mh.run_experiment(args, train_frames, held_frames, torch.device("cpu"))
    steps = {name: arm["optimizer_steps"] for name, arm in report["arms"].items()}
    assert len(set(steps.values())) == 1
    epochs_run = {arm["epochs_run"] for arm in report["arms"].values()}
    assert epochs_run == {2}
    for arm in report["arms"].values():
        assert arm["final"]["train"]["n"] == 24
        assert arm["final"]["held"]["n"] == 12


def test_flat8_arms_use_the_same_auto_matched_hidden(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms=f"{mh.ARM_FLAT8_HARD},{mh.ARM_FLAT8_HYBRID}", extra=["--epochs", "1", "--batch-size", "8"])
    report = mh.run_experiment(args, make_frames(20, seed=1), make_frames(8, seed=2), torch.device("cpu"))
    hard_params = report["arms"][mh.ARM_FLAT8_HARD]["param_count"]
    hybrid_params = report["arms"][mh.ARM_FLAT8_HYBRID]["param_count"]
    assert hard_params == hybrid_params
    assert report["flat8_match"]["hidden"] > 0
    assert report["flat8_match"]["within_2pct"] is True


def test_selection_never_uses_closed_loop_or_episode_outcome() -> None:
    src = (_REPO_ROOT / "tools" / "phase2_matched_hybrid.py").read_text()
    marker = "# closed-loop stage begins here"
    assert marker in src
    pre_gate_src = src.split(marker, 1)[0]
    for banned in ("SELECTION_SEEDS", "survival", "PPO(", "learned Spawner net", "episode_outcome"):
        assert banned not in pre_gate_src, banned


# --------------------------------------------------------------------------- #
# 6. gate evaluation (frame-level only, evaluated BEFORE any closed loop)
# --------------------------------------------------------------------------- #
def _fake_held(agreement: float, bucket_agreements: dict[str, float] | None = None) -> dict:
    buckets = bucket_agreements or {"1-7": 0.6, "8-19": 0.5, "20+": 0.45}
    by_bullet = {name: {"n": 10, "agreement": a} for name, a in buckets.items()}
    by_bullet["0"] = {"n": 0, "agreement": None}
    return {"agreement": agreement, "n": 100, "by_bullet_bucket": by_bullet}


def _fake_report(chosen_agreement: float, baseline_agreement: float, bucket_agreements=None) -> dict:
    return {
        "arms": {
            mh.ARM_ATTN64_SOFT_T1: {"final": {"held": _fake_held(baseline_agreement)}},
            mh.ARM_ATTN64_HYBRID: {"final": {"held": _fake_held(chosen_agreement, bucket_agreements)}},
            mh.ARM_ATTN64_HARD: {"final": {"held": _fake_held(0.1)}},
        }
    }


def test_gate_passes_when_all_thresholds_met() -> None:
    report = _fake_report(chosen_agreement=0.60, baseline_agreement=0.50)
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    assert gate["gate_pass"] is True
    assert gate["chosen_arm"] == mh.ARM_ATTN64_HYBRID
    assert gate["closed_loop_ran"] is False  # not yet run -- gate eval only decides whether it *may* run


def test_gate_fails_on_low_agreement() -> None:
    report = _fake_report(chosen_agreement=0.30, baseline_agreement=0.50)
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    assert gate["gate_pass"] is False
    assert gate["thresholds"]["held_agreement_min"]["pass"] is False


def test_gate_fails_on_insufficient_improvement_over_soft_t1() -> None:
    report = _fake_report(chosen_agreement=0.51, baseline_agreement=0.50)
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    assert gate["gate_pass"] is False
    assert gate["thresholds"]["improvement_over_attn64_soft_t1_min"]["pass"] is False


def test_gate_fails_on_weak_bullet_bucket() -> None:
    report = _fake_report(chosen_agreement=0.60, baseline_agreement=0.50, bucket_agreements={"1-7": 0.6, "8-19": 0.2, "20+": 0.45})
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    assert gate["gate_pass"] is False
    assert gate["thresholds"]["bullet_bucket_min_agreement"]["pass"] is False
    assert gate["thresholds"]["bullet_bucket_min_agreement"]["buckets"]["8-19"]["pass"] is False


def test_gate_report_includes_every_threshold_observed_and_reason() -> None:
    report = _fake_report(chosen_agreement=0.60, baseline_agreement=0.50)
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    for key in ("held_agreement_min", "improvement_over_attn64_soft_t1_min", "bullet_bucket_min_agreement"):
        t = gate["thresholds"][key]
        assert "threshold" in t and "pass" in t
    assert "chosen_arm" in gate
    assert "closed_loop_ran" in gate
    assert isinstance(gate["reason"], str) and gate["reason"]


def test_gate_zero_bucket_is_ignored_when_empty() -> None:
    report = _fake_report(chosen_agreement=0.60, baseline_agreement=0.50)
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    assert "0" not in gate["thresholds"]["bullet_bucket_min_agreement"]["buckets"]


def test_gate_uses_selected_checkpoint_metrics_not_final_epoch() -> None:
    report = _fake_report(chosen_agreement=0.60, baseline_agreement=0.50)
    for arm in report["arms"].values():
        arm["selected_metrics"] = arm["final"]["held"]
    # Final-epoch metrics are deliberately misleading. The gate must describe
    # the checkpoint it will actually load, not a different epoch.
    report["arms"][mh.ARM_ATTN64_HYBRID]["final"]["held"] = _fake_held(0.10)
    report["arms"][mh.ARM_ATTN64_SOFT_T1]["final"]["held"] = _fake_held(0.90)
    args = argparse.Namespace(gate_agreement_min=0.50, gate_improvement_min=0.03, gate_bullet_bucket_min=0.40)
    gate = mh.evaluate_gate(report, args)
    assert gate["chosen_arm"] == mh.ARM_ATTN64_HYBRID
    assert gate["thresholds"]["held_agreement_min"]["observed"] == pytest.approx(0.60)
    assert gate["gate_pass"] is True


# --------------------------------------------------------------------------- #
# 7. gated closed-loop orchestration -- only runs the scripted env when the
#    pre-registered gate passes; never on gate failure.
# --------------------------------------------------------------------------- #
def test_run_gated_closed_loop_skips_env_when_gate_fails(tmp_path: Path) -> None:
    # Random synthetic teacher/student logits give chance-level agreement,
    # which is far below the default gate thresholds -- the gate must fail
    # and the closed loop must never be attempted.
    args = quick_args(tmp_path, arms=f"{mh.ARM_ATTN64_SOFT_T1},{mh.ARM_ATTN64_HYBRID}")
    train_frames = make_frames(20, seed=1)
    held_frames = make_frames(8, seed=2)
    report = mh.run_experiment(args, train_frames, held_frames, torch.device("cpu"))
    gate = mh.run_gated_closed_loop(report, args, torch.device("cpu"))
    assert gate["gate_pass"] is False
    assert gate["closed_loop_ran"] is False
    assert "closed_loop" not in gate


def test_run_gated_closed_loop_runs_scripted_seeds_when_gate_passes(tmp_path: Path, monkeypatch) -> None:
    args = quick_args(tmp_path, arms=f"{mh.ARM_ATTN64_SOFT_T1},{mh.ARM_ATTN64_HYBRID}")
    train_frames = make_frames(20, seed=1)
    held_frames = make_frames(8, seed=2)
    report = mh.run_experiment(args, train_frames, held_frames, torch.device("cpu"))

    forced_gate = {
        "chosen_arm": mh.ARM_ATTN64_HYBRID,
        "gate_pass": True,
        "thresholds": {},
        "reason": "forced pass for test",
        "closed_loop_ran": False,
    }
    monkeypatch.setattr(mh, "evaluate_gate", lambda _report, _args: dict(forced_gate))

    gate = mh.run_gated_closed_loop(report, args, torch.device("cpu"))
    assert gate["gate_pass"] is True
    assert gate["closed_loop_ran"] is True
    assert gate["closed_loop"]["n"] == args.closed_loop_num_seeds
    assert gate["closed_loop_seeds"] == list(
        range(args.closed_loop_seed_start, args.closed_loop_seed_start + args.closed_loop_num_seeds)
    )
    assert set(gate["closed_loop"]) == {"mean", "median", "std", "min", "max", "n", "per_seed"}
    assert set(gate["closed_loop"]["per_seed"]) == {str(seed) for seed in gate["closed_loop_seeds"]}
    values = list(gate["closed_loop"]["per_seed"].values())
    assert gate["closed_loop"]["min"] == min(values)
    assert gate["closed_loop"]["max"] == max(values)


# --------------------------------------------------------------------------- #
# 8. report serialization / quick end-to-end
# --------------------------------------------------------------------------- #
def test_final_report_is_json_serializable_after_gate_and_closed_loop(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms=f"{mh.ARM_ATTN64_SOFT_T1},{mh.ARM_ATTN64_HYBRID}")
    train_frames = make_frames(20, seed=1)
    held_frames = make_frames(8, seed=2)
    report = mh.run_experiment(args, train_frames, held_frames, torch.device("cpu"))
    gate = mh.run_gated_closed_loop(report, args, torch.device("cpu"))
    report["gate"] = gate
    report_path = Path(args.out_dir) / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    reloaded = json.loads(report_path.read_text())
    assert reloaded["gate"]["gate_pass"] is False
    assert reloaded["arms"][mh.ARM_ATTN64_HYBRID]["param_count"] > 0


def test_quick_end_to_end_run_experiment_and_gate_is_fast_and_deterministic(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms="all")
    train_frames = make_frames(16, seed=3)
    held_frames = make_frames(6, seed=4)
    report = mh.run_experiment(args, train_frames, held_frames, torch.device("cpu"))
    gate = mh.run_gated_closed_loop(report, args, torch.device("cpu"))
    assert set(report["arms"]) == set(mh.ALL_ARMS)
    assert gate["closed_loop_ran"] is False
