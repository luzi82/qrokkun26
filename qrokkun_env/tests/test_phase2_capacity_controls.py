"""Tests for tools/phase2_capacity_controls.py (Phase 2 capacity-control harness).

These tests never touch the NAS, never load a real teacher checkpoint and never
run a closed-loop episode: every training path is exercised on tiny synthetic
frame sets so the whole file stays a fast unit suite.
"""

from __future__ import annotations

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

from qrokkun_env.agents.encoder_v4 import BulletSetEncoder  # noqa: E402
from qrokkun_env.agents.obs_v4 import (  # noqa: E402
    BULLET_FEAT_V4,
    MAX_BULLETS_V4,
    PAD_RADIUS,
    PLAYER_FEAT_V4,
)
from qrokkun_env.agents.player_v4 import PlayerV4  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402
from tools import phase2_capacity_controls as cc  # noqa: E402
from tools.phase2_distill_v1_to_v4 import Frame  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_frames(n: int, *, episodes: int = 2, seed: int = 0) -> list[Frame]:
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


def quick_args(tmp_path: Path, arms: str = "tiny_overfit", extra: list[str] | None = None):
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
    args = cc.build_parser().parse_args(argv)
    return cc.apply_mode_defaults(args)


# --------------------------------------------------------------------------- #
# 1. arm catalogue / experiment contract
# --------------------------------------------------------------------------- #
def test_all_required_controls_are_declared_as_arms() -> None:
    required = {
        cc.ARM_TINY_OVERFIT,
        cc.ARM_TOP8_MASK,
        cc.ARM_FLAT_TOP8_MLP,
        cc.ARM_OBJ_HARD,
        cc.ARM_OBJ_SOFT_T1,
        cc.ARM_OBJ_SOFT_LOWT,
    }
    assert required <= set(cc.ALL_ARMS)
    assert set(cc.ARM_SPECS) == set(cc.ALL_ARMS)
    for name, spec in cc.ARM_SPECS.items():
        assert spec.name == name
        assert spec.model in (cc.MODEL_PLAYER_V4, cc.MODEL_FLAT_TOP8_MLP)
        assert spec.label_mode in (cc.LABEL_HARD, cc.LABEL_SOFT)
        assert spec.description


def test_objective_controls_span_hard_soft_t1_and_low_temperature() -> None:
    hard = cc.ARM_SPECS[cc.ARM_OBJ_HARD]
    soft1 = cc.ARM_SPECS[cc.ARM_OBJ_SOFT_T1]
    low = cc.ARM_SPECS[cc.ARM_OBJ_SOFT_LOWT]
    assert hard.label_mode == cc.LABEL_HARD
    assert (soft1.label_mode, soft1.temperature) == (cc.LABEL_SOFT, 1.0)
    assert low.label_mode == cc.LABEL_SOFT and low.temperature is None
    for spec in (hard, soft1, low):
        assert spec.model == cc.MODEL_PLAYER_V4 and not spec.top_k_mask


def test_tiny_and_positive_control_arms_use_hard_labels() -> None:
    assert cc.ARM_SPECS[cc.ARM_TINY_OVERFIT].label_mode == cc.LABEL_HARD
    assert cc.ARM_SPECS[cc.ARM_FLAT_TOP8_MLP].label_mode == cc.LABEL_HARD
    assert cc.ARM_SPECS[cc.ARM_TOP8_MASK].top_k_mask is True
    assert cc.ARM_SPECS[cc.ARM_FLAT_TOP8_MLP].model == cc.MODEL_FLAT_TOP8_MLP


def test_low_temperature_resolves_from_cli(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--low-temperature", "0.25"])
    assert cc.resolve_temperature(cc.ARM_SPECS[cc.ARM_OBJ_SOFT_LOWT], args) == pytest.approx(0.25)
    assert cc.resolve_temperature(cc.ARM_SPECS[cc.ARM_OBJ_SOFT_T1], args) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. sampling contract: uniform, no bucket weighting, tiny arm has no held set
# --------------------------------------------------------------------------- #
def test_sampling_is_uniform_for_every_arm(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms="all")
    frames = make_frames(64, seed=1)
    for spec in cc.ARM_SPECS.values():
        assert cc.sampling_weights(spec, frames, args) is None, spec.name
    assert cc.SAMPLING_POLICY == "uniform"


def test_uniform_batch_indices_cover_every_frame_once_and_are_deterministic() -> None:
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = cc.uniform_batch_indices(10, 4, g1)
    b = cc.uniform_batch_indices(10, 4, g2)
    assert [t.tolist() for t in a] == [t.tolist() for t in b]
    flat = sorted(int(i) for t in a for i in t)
    assert flat == list(range(10))
    assert [len(t) for t in a] == [4, 4, 2]


def test_tiny_arm_truncates_frames_and_drops_held_out(tmp_path: Path) -> None:
    args = quick_args(tmp_path, extra=["--tiny-frames", "9"])
    train, held = cc.resolve_arm_data(cc.ARM_SPECS[cc.ARM_TINY_OVERFIT], make_frames(50, seed=2), make_frames(20, seed=3), args)
    assert len(train) == 9
    assert held == []


def test_non_tiny_arms_keep_the_disjoint_held_set(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms="all")
    train_in, held_in = make_frames(30, seed=4), make_frames(11, seed=5)
    for name in (cc.ARM_TOP8_MASK, cc.ARM_FLAT_TOP8_MLP, cc.ARM_OBJ_HARD):
        train, held = cc.resolve_arm_data(cc.ARM_SPECS[name], train_in, held_in, args)
        assert len(train) == 30 and len(held) == 11


def test_split_frames_by_episode_is_disjoint() -> None:
    frames = make_frames(60, episodes=6, seed=6)
    train, held = cc.split_frames_by_episode(frames, held_frac=0.34, seed=11)
    train_eps = {f.episode for f in train}
    held_eps = {f.episode for f in held}
    assert held_eps and not (train_eps & held_eps)
    assert len(train) + len(held) == len(frames)
    again = cc.split_frames_by_episode(frames, held_frac=0.34, seed=11)[1]
    assert [f.episode for f in again] == [f.episode for f in held]


# --------------------------------------------------------------------------- #
# 3. top-8 mask transform
# --------------------------------------------------------------------------- #
def test_apply_top_k_mask_pads_slots_at_or_after_k() -> None:
    bullets = torch.randn(3, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(3, MAX_BULLETS_V4, dtype=torch.bool)
    orig = bullets.clone()
    mb, mp = cc.apply_top_k_mask(bullets, pad, 8)
    assert bool(mp[:, 8:].all()) and not bool(mp[:, :8].any())
    assert torch.equal(mb[:, :8], orig[:, :8])
    assert torch.allclose(mb[:, 8:, 4], torch.full((3, MAX_BULLETS_V4 - 8), PAD_RADIUS))
    assert float(mb[:, 8:, :4].abs().sum()) == 0.0
    # input tensors must not be mutated in place (train/eval reuse the same cache)
    assert torch.equal(bullets, orig) and not bool(pad.any())


def test_apply_top_k_mask_keeps_existing_pad_flags() -> None:
    bullets = torch.randn(1, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(1, MAX_BULLETS_V4, dtype=torch.bool)
    pad[0, 3] = True
    _mb, mp = cc.apply_top_k_mask(bullets, pad, 8)
    assert bool(mp[0, 3])


def test_transform_batch_is_identity_for_unmasked_arms() -> None:
    bullets = torch.randn(2, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(2, MAX_BULLETS_V4, dtype=torch.bool)
    tb, tp = cc.transform_batch(cc.ARM_SPECS[cc.ARM_OBJ_HARD], bullets, pad, 8)
    assert torch.equal(tb, bullets) and torch.equal(tp, pad)
    mb, mp = cc.transform_batch(cc.ARM_SPECS[cc.ARM_TOP8_MASK], bullets, pad, 8)
    assert bool(mp[:, 8:].all()) and not torch.equal(mb, bullets)


def test_masked_arms_are_blind_to_slots_beyond_k() -> None:
    """Train and eval both go through transform_batch, so tail edits are no-ops."""
    torch.manual_seed(0)
    spec = cc.ARM_SPECS[cc.ARM_TOP8_MASK]
    net = PlayerV4(d_model=16, hidden=16)
    net.eval()
    player = torch.randn(2, PLAYER_FEAT_V4)
    bullets = torch.randn(2, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(2, MAX_BULLETS_V4, dtype=torch.bool)
    edited = bullets.clone()
    edited[:, 8:] += 5.0
    with torch.no_grad():
        a = cc.forward_arm(net, spec, player, bullets, pad, 8)[0].logits
        b = cc.forward_arm(net, spec, player, edited, pad, 8)[0].logits
    assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. flat top-8 MLP positive control
# --------------------------------------------------------------------------- #
def test_flat_top8_mlp_input_dim_keeps_player_bullets_and_live_mask() -> None:
    net = cc.FlatTop8MLP(top_k=8, hidden=32)
    assert net.in_dim == PLAYER_FEAT_V4 + 8 * BULLET_FEAT_V4 + 8


def test_flat_top8_mlp_forward_matches_player_v4_signature() -> None:
    net = cc.FlatTop8MLP(top_k=8, hidden=32)
    player = torch.randn(5, PLAYER_FEAT_V4)
    bullets = torch.randn(5, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(5, MAX_BULLETS_V4, dtype=torch.bool)
    dist, value = net(player, bullets, pad)
    assert dist.logits.shape == (5, len(ACTIONS))
    assert value.shape == (5,)
    assert torch.isfinite(dist.logits).all()


def test_flat_top8_mlp_features_carry_player_bullets_and_mask() -> None:
    net = cc.FlatTop8MLP(top_k=8, hidden=32)
    player = torch.randn(1, PLAYER_FEAT_V4)
    bullets = torch.randn(1, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(1, MAX_BULLETS_V4, dtype=torch.bool)
    feats = net.features(player, bullets, pad)
    assert feats.shape == (1, net.in_dim)
    assert torch.allclose(feats[0, :PLAYER_FEAT_V4], player[0])
    body = feats[0, PLAYER_FEAT_V4 : PLAYER_FEAT_V4 + 8 * BULLET_FEAT_V4]
    assert torch.allclose(body, bullets[0, :8].reshape(-1))
    assert torch.allclose(feats[0, -8:], torch.ones(8))

    pad2 = pad.clone()
    pad2[0, 2] = True
    feats2 = net.features(player, bullets, pad2)
    assert float(feats2[0, -8 + 2]) == 0.0
    dead = feats2[0, PLAYER_FEAT_V4 + 2 * BULLET_FEAT_V4 : PLAYER_FEAT_V4 + 3 * BULLET_FEAT_V4]
    assert float(dead.abs().sum()) == 0.0  # padded slot features zeroed, mask carries liveness
    assert not torch.equal(feats, feats2)


def test_flat_top8_mlp_ignores_slots_beyond_top_k() -> None:
    net = cc.FlatTop8MLP(top_k=8, hidden=32)
    player = torch.randn(1, PLAYER_FEAT_V4)
    bullets = torch.randn(1, MAX_BULLETS_V4, BULLET_FEAT_V4)
    pad = torch.zeros(1, MAX_BULLETS_V4, dtype=torch.bool)
    other = bullets.clone()
    other[:, 8:] += 3.0
    assert torch.equal(net.features(player, bullets, pad), net.features(player, other, pad))


def test_make_model_dispatches_on_spec(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms="all")
    device = torch.device("cpu")
    assert isinstance(cc.make_model(cc.ARM_SPECS[cc.ARM_OBJ_HARD], args, device), PlayerV4)
    assert isinstance(cc.make_model(cc.ARM_SPECS[cc.ARM_TOP8_MASK], args, device), PlayerV4)
    assert isinstance(cc.make_model(cc.ARM_SPECS[cc.ARM_FLAT_TOP8_MLP], args, device), cc.FlatTop8MLP)


# --------------------------------------------------------------------------- #
# 5. objective controls
# --------------------------------------------------------------------------- #
def test_hard_loss_targets_teacher_argmax() -> None:
    student = torch.randn(6, len(ACTIONS))
    teacher = torch.randn(6, len(ACTIONS))
    expected = torch.nn.functional.cross_entropy(student, teacher.argmax(-1))
    got = cc.arm_loss(cc.ARM_SPECS[cc.ARM_OBJ_HARD], student, teacher, 1.0)
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_soft_loss_at_low_temperature_approaches_hard_loss() -> None:
    student = torch.randn(32, len(ACTIONS))
    teacher = torch.randn(32, len(ACTIONS)) * 3.0
    hard = cc.arm_loss(cc.ARM_SPECS[cc.ARM_OBJ_HARD], student, teacher, 1.0).item()
    t1 = cc.arm_loss(cc.ARM_SPECS[cc.ARM_OBJ_SOFT_T1], student, teacher, 1.0).item()
    low = cc.arm_loss(cc.ARM_SPECS[cc.ARM_OBJ_SOFT_LOWT], student, teacher, 0.05).item()
    assert abs(low - hard) < abs(t1 - hard)


def test_tiny_arm_uses_hard_labels_not_soft(tmp_path: Path) -> None:
    student = torch.randn(8, len(ACTIONS))
    teacher = torch.randn(8, len(ACTIONS))
    hard = cc.arm_loss(cc.ARM_SPECS[cc.ARM_OBJ_HARD], student, teacher, 1.0)
    tiny = cc.arm_loss(cc.ARM_SPECS[cc.ARM_TINY_OVERFIT], student, teacher, 1.0)
    assert tiny.item() == pytest.approx(hard.item(), rel=1e-6)


# --------------------------------------------------------------------------- #
# 6. metrics: agreement / CE / entropies / KL, by elapsed and live-bullet bucket
# --------------------------------------------------------------------------- #
def test_compute_metrics_reports_all_required_quantities() -> None:
    torch.manual_seed(3)
    n = 40
    student = torch.randn(n, len(ACTIONS))
    teacher = torch.randn(n, len(ACTIONS))
    elapsed = torch.linspace(0.0, 60.0, n)
    live = torch.randint(0, 40, (n,))
    m = cc.compute_metrics(student, teacher, elapsed, live)
    for key in cc.CORE_METRIC_KEYS:
        assert key in m
    assert m["n"] == n
    assert 0.0 <= m["agreement"] <= 1.0
    assert m["by_elapsed_bucket"] and m["by_bullet_bucket"]
    for sub in list(m["by_elapsed_bucket"].values()) + list(m["by_bullet_bucket"].values()):
        for key in cc.CORE_METRIC_KEYS:
            assert key in sub
    assert sum(s["n"] for s in m["by_elapsed_bucket"].values()) == n
    assert sum(s["n"] for s in m["by_bullet_bucket"].values()) == n
    assert set(m["by_elapsed_bucket"]) <= set(cc.ELAPSED_BUCKETS)
    assert set(m["by_bullet_bucket"]) <= set(cc.BULLET_BUCKETS)


def test_kl_equals_soft_ce_minus_teacher_entropy() -> None:
    torch.manual_seed(4)
    student = torch.randn(16, len(ACTIONS))
    teacher = torch.randn(16, len(ACTIONS))
    m = cc.compute_metrics(student, teacher, torch.zeros(16), torch.zeros(16, dtype=torch.long))
    assert m["teacher_to_student_kl"] == pytest.approx(m["soft_ce"] - m["teacher_entropy"], abs=1e-5)
    assert m["teacher_to_student_kl"] >= -1e-6


def test_perfect_student_has_agreement_one_and_zero_kl() -> None:
    teacher = torch.randn(12, len(ACTIONS)) * 2.0
    m = cc.compute_metrics(teacher.clone(), teacher, torch.zeros(12), torch.zeros(12, dtype=torch.long))
    assert m["agreement"] == pytest.approx(1.0)
    assert m["teacher_to_student_kl"] == pytest.approx(0.0, abs=1e-6)
    assert m["student_entropy"] == pytest.approx(m["teacher_entropy"], abs=1e-5)


def test_empty_metrics_are_reported_as_none() -> None:
    m = cc.compute_metrics(
        torch.zeros(0, len(ACTIONS)),
        torch.zeros(0, len(ACTIONS)),
        torch.zeros(0),
        torch.zeros(0, dtype=torch.long),
    )
    assert m["n"] == 0 and m["agreement"] is None


def test_live_bullet_buckets_are_ordered_and_cover_the_range() -> None:
    assert cc.live_bullet_bucket(0) == cc.BULLET_BUCKETS[0]
    assert cc.live_bullet_bucket(1) != cc.live_bullet_bucket(0)
    assert cc.live_bullet_bucket(7) == cc.live_bullet_bucket(1)
    assert cc.live_bullet_bucket(8) != cc.live_bullet_bucket(7)
    assert cc.live_bullet_bucket(MAX_BULLETS_V4) == cc.BULLET_BUCKETS[-1]
    seen = [cc.live_bullet_bucket(i) for i in range(MAX_BULLETS_V4 + 1)]
    assert set(seen) == set(cc.BULLET_BUCKETS)


def test_live_counts_from_pad_counts_unpadded_slots() -> None:
    pad = torch.ones(2, MAX_BULLETS_V4, dtype=torch.bool)
    pad[0, :5] = False
    counts = cc.live_counts_from_pad(pad)
    assert counts.tolist() == [5, 0]


def test_bullet_buckets_use_raw_live_count_even_for_masked_arms(tmp_path: Path) -> None:
    """The top-8 arm must still be reported against the true bullet density."""
    frames = make_frames(24, seed=8)
    tensors = cc.frames_to_tensors(frames, torch.device("cpu"))
    raw = cc.live_counts_from_pad(tensors["pad"])
    net = PlayerV4(d_model=16, hidden=16)
    m = cc.model_metrics(net, cc.ARM_SPECS[cc.ARM_TOP8_MASK], tensors, top_k=8, batch_size=8)
    expected = {}
    for c in raw.tolist():
        expected[cc.live_bullet_bucket(int(c))] = expected.get(cc.live_bullet_bucket(int(c)), 0) + 1
    assert {k: v["n"] for k, v in m["by_bullet_bucket"].items()} == expected


# --------------------------------------------------------------------------- #
# 7. selection / acceptance reporting
# --------------------------------------------------------------------------- #
def test_selection_prefers_agreement_then_kl_then_latest() -> None:
    lo = {"agreement": 0.40, "teacher_to_student_kl": 0.5}
    hi = {"agreement": 0.55, "teacher_to_student_kl": 9.0}
    assert cc.should_replace_selected(hi, lo) is True
    assert cc.should_replace_selected(lo, hi) is False
    same_a_lo_kl = {"agreement": 0.40, "teacher_to_student_kl": 0.1}
    assert cc.should_replace_selected(same_a_lo_kl, lo) is True
    assert cc.should_replace_selected(lo, same_a_lo_kl) is False
    # exact tie -> retain the latest candidate
    assert cc.should_replace_selected(dict(lo), lo) is True
    assert cc.should_replace_selected(lo, None) is True


def test_selection_never_uses_closed_loop_survival() -> None:
    src = (_REPO_ROOT / "tools" / "phase2_capacity_controls.py").read_text()
    for banned in ("closed_loop_times", "SELECTION_SEEDS", "survival", "PPO(", "Spawner"):
        assert banned not in src, banned


def test_acceptance_is_reported_not_asserted() -> None:
    met = cc.acceptance_report(0.97, 0.95)
    missed = cc.acceptance_report(0.42, 0.95)
    assert met["met"] is True and missed["met"] is False
    assert met["target"] == pytest.approx(0.95)
    assert missed["observed"] == pytest.approx(0.42)
    assert "note" in missed and missed["verdict"] != "architecture_failure"
    assert cc.acceptance_report(None, 0.95)["met"] is None


# --------------------------------------------------------------------------- #
# 8. all-pad finiteness regression (encoder + PlayerV4, forward and backward)
# --------------------------------------------------------------------------- #
def _all_pad_batch(batch: int = 3):
    player = torch.randn(batch, PLAYER_FEAT_V4)
    bullets = torch.zeros(batch, MAX_BULLETS_V4, BULLET_FEAT_V4)
    bullets[..., 4] = PAD_RADIUS
    pad = torch.ones(batch, MAX_BULLETS_V4, dtype=torch.bool)
    return player, bullets, pad


@pytest.mark.parametrize("train_mode", [False, True])
def test_all_pad_bullet_set_encoder_forward_and_backward_are_finite(train_mode: bool) -> None:
    torch.manual_seed(0)
    enc = BulletSetEncoder(d_model=32)
    enc.train(train_mode)
    player, bullets, pad = _all_pad_batch()
    out = enc(player, bullets, pad)
    assert torch.isfinite(out).all(), "all-pad encoder forward produced non-finite values"
    out.sum().backward()
    bad = [n for n, p in enc.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"all-pad encoder backward produced non-finite grads: {bad}"


@pytest.mark.parametrize("train_mode", [False, True])
def test_all_pad_player_v4_forward_and_backward_are_finite(train_mode: bool) -> None:
    torch.manual_seed(0)
    net = PlayerV4(d_model=32, hidden=32)
    net.train(train_mode)
    player, bullets, pad = _all_pad_batch()
    dist, value = net(player, bullets, pad)
    assert torch.isfinite(dist.logits).all(), "all-pad PlayerV4 logits are non-finite"
    assert torch.isfinite(value).all(), "all-pad PlayerV4 value is non-finite"
    (dist.logits.sum() + value.sum()).backward()
    bad = [n for n, p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"all-pad PlayerV4 backward produced non-finite grads: {bad}"


def test_mixed_all_pad_and_live_batch_gradients_are_finite() -> None:
    torch.manual_seed(0)
    net = PlayerV4(d_model=32, hidden=32)
    player, bullets, pad = _all_pad_batch(4)
    bullets[0, :3] = torch.randn(3, BULLET_FEAT_V4)
    pad[0, :3] = False
    dist, value = net(player, bullets, pad)
    loss = torch.nn.functional.cross_entropy(dist.logits, torch.zeros(4, dtype=torch.long)) + value.pow(2).mean()
    loss.backward()
    bad = [n for n, p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"mixed all-pad batch produced non-finite grads: {bad}"


# --------------------------------------------------------------------------- #
# 9. CLI / quick mode / end-to-end report
# --------------------------------------------------------------------------- #
def test_resolve_arms_accepts_all_csv_and_rejects_unknown() -> None:
    assert cc.resolve_arms("all") == list(cc.ALL_ARMS)
    assert cc.resolve_arms(f"{cc.ARM_TOP8_MASK},{cc.ARM_OBJ_HARD}") == [cc.ARM_TOP8_MASK, cc.ARM_OBJ_HARD]
    with pytest.raises(SystemExit):
        cc.resolve_arms("nope")


def test_quick_mode_shrinks_defaults_but_respects_explicit_flags(tmp_path: Path) -> None:
    full = cc.apply_mode_defaults(
        cc.build_parser().parse_args(["--teacher", "t.pt", "--out-dir", str(tmp_path)])
    )
    quick = quick_args(tmp_path, arms="all")
    assert quick.quick is True and full.quick is False
    assert quick.collect_episodes < full.collect_episodes
    assert quick.epochs < full.epochs
    assert quick.max_steps < full.max_steps
    assert quick.tiny_frames <= full.tiny_frames
    explicit = cc.apply_mode_defaults(
        cc.build_parser().parse_args(["--teacher", "t.pt", "--out-dir", str(tmp_path), "--quick", "--epochs", "7"])
    )
    assert explicit.epochs == 7


def test_default_arms_is_all_and_selection_is_explicit(tmp_path: Path) -> None:
    args = cc.apply_mode_defaults(cc.build_parser().parse_args(["--teacher", "t.pt", "--out-dir", str(tmp_path)]))
    assert cc.resolve_arms(args.arms) == list(cc.ALL_ARMS)
    one = quick_args(tmp_path, arms=cc.ARM_FLAT_TOP8_MLP)
    assert cc.resolve_arms(one.arms) == [cc.ARM_FLAT_TOP8_MLP]


def test_train_arm_writes_checkpoint_and_reports_train_and_held(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms=cc.ARM_OBJ_HARD, extra=["--epochs", "2", "--batch-size", "8"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "epochs.jsonl"
    result = cc.train_arm(
        cc.ARM_SPECS[cc.ARM_OBJ_HARD],
        args,
        make_frames(24, seed=9),
        make_frames(12, seed=10),
        torch.device("cpu"),
        out_dir,
        log_path=log_path,
    )
    assert result["epochs_run"] == 2
    assert len(result["epoch_history"]) == 2
    assert result["final"]["train"]["n"] == 24
    assert result["final"]["held"]["n"] == 12
    assert result["selected"]["epoch"] in (1, 2)
    ckpt_path = Path(result["checkpoint"])
    latest_path = Path(result["latest_checkpoint"])
    assert ckpt_path.exists() and latest_path.exists()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ck["arm"] == cc.ARM_OBJ_HARD
    assert ck["ckpt_role"] == "player"
    assert ck["label_mode"] == cc.LABEL_HARD
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2 and lines[0]["arm"] == cc.ARM_OBJ_HARD


def test_tiny_arm_report_has_no_held_metrics_and_an_acceptance_block(tmp_path: Path) -> None:
    args = quick_args(tmp_path, arms=cc.ARM_TINY_OVERFIT, extra=["--epochs", "2", "--batch-size", "8", "--tiny-frames", "16"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = cc.train_arm(
        cc.ARM_SPECS[cc.ARM_TINY_OVERFIT],
        args,
        make_frames(40, seed=11),
        make_frames(10, seed=12),
        torch.device("cpu"),
        out_dir,
    )
    assert result["final"]["held"] is None
    assert result["final"]["train"]["n"] == 16
    acc = result["acceptance"]
    assert acc["target"] == pytest.approx(args.tiny_target_agreement)
    assert acc["metric"] == "train_agreement"
    assert isinstance(acc["met"], bool)


def test_run_experiment_writes_machine_readable_report(tmp_path: Path) -> None:
    arms = f"{cc.ARM_TINY_OVERFIT},{cc.ARM_FLAT_TOP8_MLP},{cc.ARM_TOP8_MASK}"
    args = quick_args(tmp_path, arms=arms, extra=["--epochs", "1", "--batch-size", "8", "--tiny-frames", "12"])
    report = cc.run_experiment(args, make_frames(32, seed=13), make_frames(10, seed=14), torch.device("cpu"))
    report_path = Path(report["report_path"])
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text())
    assert on_disk["seed"] == args.seed
    assert on_disk["sampling"] == cc.SAMPLING_POLICY
    assert set(on_disk["arms"]) == set(cc.resolve_arms(arms))
    for name, arm in on_disk["arms"].items():
        assert arm["spec"]["name"] == name
        assert "train" in arm["final"] and "held" in arm["final"]
        assert Path(arm["checkpoint"]).exists()
    assert on_disk["selection"]["criteria"] == ["held_agreement", "held_kl", "latest"]
    assert on_disk["dataset"]["train_frames"] == 32
    assert on_disk["dataset"]["held_frames"] == 10


def test_run_experiment_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    frames = make_frames(24, seed=15)
    held = make_frames(8, seed=16)
    outs = []
    for i in range(2):
        args = quick_args(tmp_path / f"run{i}", arms=cc.ARM_OBJ_HARD, extra=["--epochs", "1", "--batch-size", "8"])
        args.out_dir = str(tmp_path / f"run{i}")
        rep = cc.run_experiment(args, frames, held, torch.device("cpu"))
        outs.append(rep["arms"][cc.ARM_OBJ_HARD]["final"]["held"]["hard_ce"])
    assert outs[0] == pytest.approx(outs[1], rel=1e-9)
    assert not math.isnan(outs[0])
