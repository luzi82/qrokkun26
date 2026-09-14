"""Tests for tools/phase3_ranked_ppo_retention_aux.py.

Round-1 one-knob follow-up to the Phase 3 BC-retention control: unregularized
scripted PPO washed the PlayerV5 BC skill, so this arm adds a BC-retention
auxiliary loss scaled by a single pre-registered ``alpha``. EVERY other knob
stays locked to ``tools.phase3_ranked_ppo_retention``.

Contract under specification
============================

* Locked knobs (lr/gamma/lambda/clip/entropy/value/epochs/minibatch/
  episodes-per-update/updates/max-grad-norm/rollout seed start/eval seed
  window/snapshot schedule) are the SAME OBJECTS as phase3's -- never
  re-declared, never CLI-overridable.
* The retention objective is EXACTLY
  ``phase2_ranked_multiseed.hybrid_loss`` (0.5 hard CE + 0.5 soft at T=1).
* ``alpha`` is calibrated ONCE at the starting checkpoint, before any
  optimizer step, as ``0.15 * ||g_ppo|| / (||g_ret|| + 1e-8)`` where
  ``g_ppo`` is the gradient of the PPO policy loss ONLY (no value, no
  entropy) and ``g_ret`` the gradient of ``hybrid_loss`` on one retention
  minibatch drawn from the teacher TRAIN split. It is then frozen.
* Teacher frames never enter advantage/old_log_prob/GAE/ratio; the held-out
  teacher split is diagnostics/gates only and is never read by the update.
* Snapshots are experimental (``experimental=True``,
  ``production_compatible=False``); promotion is ALWAYS false.
"""

from __future__ import annotations

import argparse
import ast
import inspect
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
    PLAYER_FEAT_V4,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.agents.player_v1 import PlayerV1  # noqa: E402
from qrokkun_env.agents.player_checkpoints import (  # noqa: E402
    CheckpointError,
    save_player_checkpoint,
)
from qrokkun_env.env import ACTIONS  # noqa: E402
from tools import phase2_ranked_multiseed as ms  # noqa: E402
from tools import phase3_ranked_ppo_retention as ret  # noqa: E402

# RED until the tool exists.
from tools import phase3_ranked_ppo_retention_aux as aux  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. every PPO knob stays locked to phase3
# --------------------------------------------------------------------------- #
def test_locked_knobs_are_identical_to_phase3() -> None:
    assert aux.PPO_LR == ret.PPO_LR
    assert aux.PPO_GAMMA == ret.PPO_GAMMA
    assert aux.PPO_LAMBDA == ret.PPO_LAMBDA
    assert aux.PPO_CLIP == ret.PPO_CLIP
    assert aux.PPO_ENTROPY_COEF == ret.PPO_ENTROPY_COEF
    assert aux.PPO_VALUE_COEF == ret.PPO_VALUE_COEF
    assert aux.PPO_EPOCHS == ret.PPO_EPOCHS
    assert aux.PPO_MINIBATCH == ret.PPO_MINIBATCH
    assert aux.EPISODES_PER_UPDATE == ret.EPISODES_PER_UPDATE
    assert aux.PPO_MAX_GRAD_NORM == ret.PPO_MAX_GRAD_NORM
    assert aux.PPO_UPDATES == ret.PPO_UPDATES
    assert aux.PPO_MAX_FRAMES == ret.PPO_MAX_FRAMES
    assert aux.PPO_ROLLOUT_SEED_START == ret.PPO_ROLLOUT_SEED_START
    assert aux.PPO_TORCH_SEED == ret.PPO_TORCH_SEED
    assert aux.EVAL_SEED_START == ret.EVAL_SEED_START
    assert aux.EVAL_SEED_COUNT == ret.EVAL_SEED_COUNT
    assert aux.EVAL_MAX_STEPS == ret.EVAL_MAX_STEPS
    assert aux.SNAPSHOT_UPDATES == ret.SNAPSHOT_UPDATES
    assert aux.DATA_SEED == ret.DATA_SEED
    assert aux.RETENTION_FRACTION == ret.RETENTION_FRACTION
    assert aux.MAX_HELD_AGREEMENT_DROP == ret.MAX_HELD_AGREEMENT_DROP
    assert aux.HELD_OUT_FRAC == ret.HELD_OUT_FRAC
    assert aux.FRAMES_PER_EPISODE_CAP == ret.FRAMES_PER_EPISODE_CAP
    # schedules/helpers are the phase3 functions themselves, not copies
    assert aux.rollout_seed_schedule is ret.rollout_seed_schedule
    assert aux.eval_seed_list is ret.eval_seed_list
    assert aux.snapshot_schedule is ret.snapshot_schedule
    assert aux.evaluate_retention is ret.evaluate_retention
    assert aux.load_initial_checkpoint is ret.load_initial_checkpoint
    assert aux.collect_canonical_dataset is ret.collect_canonical_dataset
    assert aux.teacher_diagnostics is ret.teacher_diagnostics
    assert aux.evaluate_deterministic is ret.evaluate_deterministic
    assert aux.collect_rollout is ret.collect_rollout
    assert aux.rollouts_to_batch is ret.rollouts_to_batch


# --------------------------------------------------------------------------- #
# 2. the retention objective is exactly phase2's hybrid loss
# --------------------------------------------------------------------------- #
def test_retention_objective_is_phase2_hybrid_loss() -> None:
    assert aux.hybrid_loss is ms.hybrid_loss
    assert aux.HYBRID_HARD_WEIGHT == ms.HYBRID_HARD_WEIGHT == 0.5
    assert aux.HYBRID_SOFT_WEIGHT == ms.HYBRID_SOFT_WEIGHT == 0.5
    assert aux.HYBRID_TEMPERATURE == ms.HYBRID_TEMPERATURE == 1.0

    torch.manual_seed(0)
    student = torch.randn(7, 5)
    teacher = torch.randn(7, 5)
    assert torch.allclose(
        aux.retention_loss(student, teacher), ms.hybrid_loss(student, teacher)
    )


# --------------------------------------------------------------------------- #
# 3. the single new pre-registered knob
# --------------------------------------------------------------------------- #
def test_target_retention_grad_ratio_is_preregistered_midpoint() -> None:
    assert aux.TARGET_RETENTION_GRAD_RATIO == 0.15
    # midpoint of the agreed 10-20% band
    assert 0.10 <= aux.TARGET_RETENTION_GRAD_RATIO <= 0.20


def test_retention_loss_components_decompose_the_hybrid_loss() -> None:
    torch.manual_seed(1)
    student = torch.randn(11, 5)
    teacher = torch.randn(11, 5)
    parts = aux.retention_loss_components(student, teacher)
    assert torch.allclose(
        torch.tensor(parts["hybrid"]), ms.hybrid_loss(student, teacher).detach()
    )
    recomposed = (
        ms.HYBRID_HARD_WEIGHT * parts["hard_ce"] + ms.HYBRID_SOFT_WEIGHT * parts["soft_ce"]
    )
    assert recomposed == pytest.approx(parts["hybrid"], rel=1e-6)
    # soft cross-entropy minus the teacher's own entropy is the soft KL
    teacher_probs = torch.softmax(teacher, dim=-1)
    teacher_entropy = float(
        -(teacher_probs * torch.log_softmax(teacher, dim=-1)).sum(-1).mean().item()
    )
    assert parts["soft_kl"] == pytest.approx(parts["soft_ce"] - teacher_entropy, rel=1e-5)


# --------------------------------------------------------------------------- #
# helpers (CPU-light: tiny nets, synthetic frames, never 240 real episodes)
# --------------------------------------------------------------------------- #
def _tiny_net(seed: int = 0, hidden: int = 8, top_k: int = 4) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def _teacher_tensors(n: int = 12, seed: int = 3, marker: float | None = None) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    player = torch.randn(n, PLAYER_FEAT_V4, generator=g)
    if marker is not None:
        player[:, 0] = marker
    return {
        "player": player,
        "bullets": torch.randn(n, MAX_BULLETS_V4, BULLET_FEAT_V4, generator=g),
        "pad": torch.zeros(n, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.randn(n, len(ACTIONS), generator=g),
        "elapsed": torch.rand(n, generator=g) * 30.0,
    }


def _fake_rollouts(n_frames: int = 6, n_rollouts: int = 2, seed: int = 5, marker: float = 0.0) -> list:
    rng = np.random.default_rng(seed)
    rollouts = []
    for i in range(n_rollouts):
        player = rng.standard_normal((n_frames, PLAYER_FEAT_V4)).astype(np.float32)
        player[:, 0] = marker
        rollouts.append(
            ret.Rollout(
                seed=i,
                player=[p for p in player],
                bullets=[
                    rng.standard_normal((MAX_BULLETS_V4, BULLET_FEAT_V4)).astype(np.float32)
                    for _ in range(n_frames)
                ],
                pad=[np.zeros(MAX_BULLETS_V4, dtype=np.bool_) for _ in range(n_frames)],
                actions=[int(a) for a in rng.integers(0, len(ACTIONS), n_frames)],
                log_probs=[float(x) for x in rng.standard_normal(n_frames)],
                values=[float(x) for x in rng.standard_normal(n_frames)],
                rewards=[float(x) for x in rng.standard_normal(n_frames)],
                dones=[False] * (n_frames - 1) + [True],
                elapsed=float(n_frames) / 60.0,
                censored=False,
            )
        )
    return rollouts


# --------------------------------------------------------------------------- #
# 4. retention minibatches come from the teacher TRAIN split only
# --------------------------------------------------------------------------- #
def test_sample_retention_minibatch_is_deterministic_and_sized() -> None:
    train = _teacher_tensors(n=40)
    gen_a = torch.Generator().manual_seed(aux.RETENTION_SAMPLER_SEED)
    gen_b = torch.Generator().manual_seed(aux.RETENTION_SAMPLER_SEED)
    mb_a = aux.sample_retention_minibatch(train, torch.device("cpu"), gen_a, size=8)
    mb_b = aux.sample_retention_minibatch(train, torch.device("cpu"), gen_b, size=8)
    for key in ("player", "bullets", "pad", "teacher_logits"):
        assert mb_a[key].shape[0] == 8
        assert torch.equal(mb_a[key], mb_b[key])
    # every sampled row is an actual train row
    matches = (train["player"].unsqueeze(0) == mb_a["player"].unsqueeze(1)).all(-1).any(-1)
    assert bool(matches.all())


def test_sample_retention_minibatch_handles_split_smaller_than_minibatch() -> None:
    train = _teacher_tensors(n=5)
    gen = torch.Generator().manual_seed(0)
    mb = aux.sample_retention_minibatch(train, torch.device("cpu"), gen, size=64)
    assert mb["player"].shape[0] == 5


# --------------------------------------------------------------------------- #
# 5. gradient alignment: policy-loss-only grad vs hybrid-loss grad, no step
# --------------------------------------------------------------------------- #
def test_grad_alignment_reports_norms_and_cosine_without_stepping() -> None:
    net = _tiny_net()
    device = torch.device("cpu")
    before = [p.detach().clone() for p in net.parameters()]
    rollouts = _fake_rollouts()
    train = _teacher_tensors(n=20)

    info = aux.grad_alignment(
        net, rollouts, train, device, generator=aux.retention_generator(), minibatch=8
    )

    assert info["g_ppo_norm"] > 0.0
    assert info["g_ret_norm"] > 0.0
    assert -1.0 - 1e-6 <= info["cosine_similarity"] <= 1.0 + 1e-6
    assert info["n_retention_samples"] == 8
    # g_ppo and g_ret must be measured on the SAME (minibatch) scale, never
    # a full-rollout PPO gradient vs a minibatch-sized retention gradient.
    assert info["n_samples"] == 8
    assert info["n_samples"] == info["n_retention_samples"]
    # no optimizer step and no leftover dirty .grad on the live net
    for p, b in zip(net.parameters(), before):
        assert torch.equal(p.detach(), b)
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_grad_alignment_fails_closed_for_empty_ppo_rollouts() -> None:
    with pytest.raises(ValueError, match="non-empty PPO rollout batch"):
        aux.grad_alignment(
            _tiny_net(),
            [],
            _teacher_tensors(n=4),
            torch.device("cpu"),
            generator=aux.retention_generator(),
            minibatch=4,
        )


def test_grad_alignment_ppo_grad_excludes_value_and_entropy_terms() -> None:
    net = _tiny_net(seed=2)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(seed=7)
    train = _teacher_tensors(n=20, seed=8)
    seed = 4242
    info = aux.grad_alignment(
        net,
        rollouts,
        train,
        device,
        generator=torch.Generator().manual_seed(seed),
        minibatch=8,
    )

    params = [p for p in net.parameters() if p.requires_grad]
    batch = ret.rollouts_to_batch(rollouts, device)
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    n_frames = int(batch["player"].shape[0])
    # grad_alignment draws its PPO subsample as the FIRST op on its
    # generator: reproduce that with a fresh generator of the same seed.
    index = torch.randperm(n_frames, generator=torch.Generator().manual_seed(seed))[:8]
    dist, _value = net(
        batch["player"][index], batch["bullets"][index], batch["pad"][index]
    )
    logp = dist.log_prob(batch["actions"][index])
    ratio = torch.exp(logp - batch["old_log_probs"][index])
    policy_loss = -torch.min(
        ratio * adv[index], torch.clamp(ratio, 1 - ret.PPO_CLIP, 1 + ret.PPO_CLIP) * adv[index]
    ).mean()
    grads = torch.autograd.grad(policy_loss, params, allow_unused=True)
    expected = torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(params, grads)
        ]
    )
    assert info["g_ppo_norm"] == pytest.approx(float(expected.norm().item()), rel=1e-6)


def test_grad_alignment_uses_matched_minibatch_scale() -> None:
    """The probe's g_ppo and g_ret must be measured on equal-sized draws,
    never a full-rollout PPO gradient against a minibatch-sized retention
    gradient. Uses an explicit small ``minibatch`` (CPU-light); the default
    is still asserted to be ``PPO_MINIBATCH``."""
    net = _tiny_net(seed=9)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=10, n_rollouts=3, seed=21)  # 30 frames total
    train = _teacher_tensors(n=40, seed=22)
    minibatch = 8

    info = aux.grad_alignment(
        net, rollouts, train, device, generator=torch.Generator().manual_seed(0),
        minibatch=minibatch,
    )

    total_frames = sum(len(r.actions) for r in rollouts)
    assert total_frames > minibatch  # the mismatch this guards against
    assert info["n_samples"] == minibatch
    assert info["n_retention_samples"] == minibatch
    assert info["n_samples"] == info["n_retention_samples"]
    assert (
        inspect.signature(aux.grad_alignment).parameters["minibatch"].default
        == aux.PPO_MINIBATCH
    )


def test_grad_alignment_matches_sizes_when_rollout_is_shorter_than_minibatch() -> None:
    """Short-rollout edge: when the PPO batch has FEWER frames than the
    requested minibatch, g_ppo is necessarily measured on all available
    frames -- the retention draw must shrink to match, never stay at the
    full requested minibatch size."""
    net = _tiny_net(seed=11)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=3, n_rollouts=1, seed=25)  # 3 frames total
    train = _teacher_tensors(n=40, seed=26)  # plenty of retention rows
    minibatch = 8

    total_frames = sum(len(r.actions) for r in rollouts)
    assert total_frames < minibatch  # the edge this guards against

    info = aux.grad_alignment(
        net, rollouts, train, device,
        generator=torch.Generator().manual_seed(1), minibatch=minibatch,
    )

    assert info["n_samples"] == total_frames
    assert info["n_retention_samples"] == total_frames
    assert info["n_samples"] == info["n_retention_samples"]


def test_grad_alignment_matches_sizes_when_retention_pool_is_smaller_than_minibatch() -> None:
    """Small-retention-pool edge: when the teacher TRAIN split has FEWER
    rows than the requested minibatch, g_ret is necessarily measured on all
    available retention rows -- the PPO draw must shrink to match, never
    stay at the full requested minibatch size (even though the rollout
    batch itself has plenty of frames)."""
    net = _tiny_net(seed=12)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=10, n_rollouts=1, seed=27)  # 10 frames total
    train = _teacher_tensors(n=3, seed=28)  # only 3 retention rows
    minibatch = 8

    total_frames = sum(len(r.actions) for r in rollouts)
    assert total_frames > minibatch
    assert len(train["player"]) < minibatch  # the edge this guards against

    info = aux.grad_alignment(
        net, rollouts, train, device,
        generator=torch.Generator().manual_seed(2), minibatch=minibatch,
    )

    assert info["n_samples"] == len(train["player"])
    assert info["n_retention_samples"] == len(train["player"])
    assert info["n_samples"] == info["n_retention_samples"]


def test_snapshot_grad_ratio_is_matched_minibatch_scale() -> None:
    """Snapshot/ppo_aux_update grad_alignment fields correspond to equal
    g_ppo/g_ret sample counts, even though the rollout batch is larger."""
    device = torch.device("cpu")
    net = _tiny_net(seed=10)
    opt = torch.optim.Adam(net.parameters(), lr=aux.PPO_LR, eps=1e-8)
    rollouts = _fake_rollouts(n_frames=10, n_rollouts=3, seed=23)
    train = _teacher_tensors(n=40, seed=24)
    minibatch = 8

    total_frames = sum(len(r.actions) for r in rollouts)
    assert total_frames > minibatch

    # the alignment probe itself (as used for snapshots)
    alignment = aux.grad_alignment(
        net, rollouts, train, device,
        generator=aux.diagnostic_generator(), minibatch=minibatch,
    )
    assert alignment["n_samples"] == alignment["n_retention_samples"] == minibatch

    # and the same holds for the diagnostics embedded in ppo_aux_update
    metrics = aux.ppo_aux_update(
        net, opt, rollouts, train, 0.1, device,
        generator=aux.training_generator(),
        diagnostics_generator=aux.diagnostic_generator(),
        minibatch=minibatch,
    )
    # ppo_aux_update's top-level n_samples/n_retention_samples describe the
    # full training loop (rollout batch / summed retention minibatches
    # across epochs), not the alignment probe -- but the g_ppo/g_ret pair
    # baked into grad_ratio/cosine_similarity must still be equal-count,
    # which is only true if grad_alignment (above) used matched sizes.
    assert math.isfinite(metrics["grad_ratio"])
    assert math.isfinite(metrics["cosine_similarity"])


def test_calibration_and_diagnostics_do_not_advance_global_torch_rng() -> None:
    """calibrate_alpha and grad_alignment must draw exclusively from the
    generators passed to them, never the default (global) CPU RNG stream.

    We seed the default generator, draw a marker token, then draw a second
    "expected next" token WITHOUT calling into the aux module -- this is
    what the default stream produces if nothing touches it. Separately, we
    reseed, re-draw the (identical) marker, invoke calibrate_alpha/
    grad_alignment, and draw again: if those calls never advanced the
    default generator, this second draw must equal the untouched one.
    """
    device = torch.device("cpu")
    # build the net/data first: _tiny_net seeds the default RNG internally,
    # so do that BEFORE establishing our own control point below.
    net = _tiny_net(seed=30)
    rollouts = _fake_rollouts(n_frames=8, n_rollouts=2, seed=31)
    train = _teacher_tensors(n=24, seed=32)

    torch.manual_seed(0)
    marker = torch.rand(4)
    expected_next = torch.rand(4)  # what the default stream yields untouched

    torch.manual_seed(0)
    assert torch.equal(torch.rand(4), marker)

    aux.calibrate_alpha(net, rollouts, train, device, minibatch=6)
    aux.grad_alignment(
        net, rollouts, train, device, generator=aux.diagnostic_generator(), minibatch=6
    )

    probe_next = torch.rand(4)
    assert torch.equal(probe_next, expected_next)


# --------------------------------------------------------------------------- #
# 6. alpha calibration: one frozen, positive, finite, repeatable value
# --------------------------------------------------------------------------- #
def test_calibrate_alpha_matches_each_draw_when_rollout_is_shorter_than_minibatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every calibration gradient pair uses the available PPO-frame count."""
    net = _tiny_net(seed=34)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=3, n_rollouts=1, seed=35)
    train = _teacher_tensors(n=20, seed=36)
    minibatch = 8
    ppo_draw_sizes: list[int] = []
    retention_draw_sizes: list[int] = []

    real_policy_loss_only = aux.policy_loss_only
    real_sample_retention_minibatch = aux.sample_retention_minibatch

    def record_ppo_draw(*args, **kwargs):
        ppo_draw_sizes.append(int(kwargs["index"].shape[0]))
        return real_policy_loss_only(*args, **kwargs)

    def record_retention_draw(*args, **kwargs):
        batch = real_sample_retention_minibatch(*args, **kwargs)
        retention_draw_sizes.append(int(batch["player"].shape[0]))
        return batch

    monkeypatch.setattr(aux, "policy_loss_only", record_ppo_draw)
    monkeypatch.setattr(aux, "sample_retention_minibatch", record_retention_draw)

    calibration_draws = 2
    result = aux.calibrate_alpha(
        net,
        rollouts,
        train,
        device,
        minibatch=minibatch,
        n_minibatches=calibration_draws,
        generator=torch.Generator().manual_seed(37),
    )

    assert sum(len(r.actions) for r in rollouts) < minibatch
    assert ppo_draw_sizes == retention_draw_sizes == [3, 3]
    assert result["n_samples"] == result["n_retention_samples"] == 6


def test_calibrate_alpha_matches_each_draw_when_retention_pool_is_smaller_than_minibatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every calibration gradient pair uses the available retention-frame count."""
    net = _tiny_net(seed=38)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=10, n_rollouts=1, seed=39)
    train = _teacher_tensors(n=3, seed=40)
    minibatch = 8
    ppo_draw_sizes: list[int] = []
    retention_draw_sizes: list[int] = []

    real_policy_loss_only = aux.policy_loss_only
    real_sample_retention_minibatch = aux.sample_retention_minibatch

    def record_ppo_draw(*args, **kwargs):
        ppo_draw_sizes.append(int(kwargs["index"].shape[0]))
        return real_policy_loss_only(*args, **kwargs)

    def record_retention_draw(*args, **kwargs):
        batch = real_sample_retention_minibatch(*args, **kwargs)
        retention_draw_sizes.append(int(batch["player"].shape[0]))
        return batch

    monkeypatch.setattr(aux, "policy_loss_only", record_ppo_draw)
    monkeypatch.setattr(aux, "sample_retention_minibatch", record_retention_draw)

    calibration_draws = 2
    result = aux.calibrate_alpha(
        net,
        rollouts,
        train,
        device,
        minibatch=minibatch,
        n_minibatches=calibration_draws,
        generator=torch.Generator().manual_seed(41),
    )

    assert sum(len(r.actions) for r in rollouts) > minibatch
    assert len(train["player"]) < minibatch
    assert ppo_draw_sizes == retention_draw_sizes == [3, 3]
    assert result["n_samples"] == result["n_retention_samples"] == 6


def test_calibrate_alpha_fails_closed_for_an_empty_retention_pool() -> None:
    """An auxiliary run without TRAIN teacher frames is invalid, rather than
    a zero-sized gradient calibration that can produce a NaN alpha."""
    with pytest.raises(ValueError, match="non-empty retention TRAIN split"):
        aux.calibrate_alpha(
            _tiny_net(seed=42),
            _fake_rollouts(n_frames=8, n_rollouts=1, seed=43),
            _teacher_tensors(n=0, seed=44),
            torch.device("cpu"),
            minibatch=4,
        )


def test_calibrate_alpha_fails_closed_for_empty_ppo_rollouts() -> None:
    with pytest.raises(ValueError, match="non-empty PPO rollout batch"):
        aux.calibrate_alpha(
            _tiny_net(),
            [],
            _teacher_tensors(n=4),
            torch.device("cpu"),
            minibatch=4,
        )


def test_calibrate_alpha_uses_the_preregistered_formula() -> None:
    net = _tiny_net(seed=4)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(seed=11)
    train = _teacher_tensors(n=24, seed=12)

    cal = aux.calibrate_alpha(net, rollouts, train, device, minibatch=8)

    assert cal["target_ratio"] == aux.TARGET_RETENTION_GRAD_RATIO
    expected = aux.TARGET_RETENTION_GRAD_RATIO * (
        cal["g_ppo_norm"] / (cal["g_ret_norm"] + 1e-8)
    )
    assert cal["alpha"] == pytest.approx(expected, rel=1e-9)
    assert cal["alpha"] > 0.0
    assert math.isfinite(cal["alpha"])
    assert "cosine_similarity" in cal
    # the weighted retention gradient sits at the pre-registered ratio
    assert cal["g_ret_weighted_norm"] / (cal["g_ppo_norm"] + 1e-8) == pytest.approx(
        aux.TARGET_RETENTION_GRAD_RATIO, rel=1e-5
    )


def test_calibrate_alpha_is_repeatable_and_takes_no_step() -> None:
    device = torch.device("cpu")
    rollouts = _fake_rollouts(seed=11)
    train = _teacher_tensors(n=24, seed=12)

    net_a = _tiny_net(seed=4)
    before = [p.detach().clone() for p in net_a.parameters()]
    cal_a = aux.calibrate_alpha(net_a, rollouts, train, device, minibatch=8)
    cal_b = aux.calibrate_alpha(net_a, rollouts, train, device, minibatch=8)

    assert cal_a["alpha"] == pytest.approx(cal_b["alpha"], rel=1e-12)
    for p, b in zip(net_a.parameters(), before):
        assert torch.equal(p.detach(), b)


def test_frozen_alpha_state_never_retunes() -> None:
    state = aux.FrozenAlpha(alpha=0.25, calibration={"alpha": 0.25})
    assert state.alpha == 0.25
    with pytest.raises(AttributeError):
        state.alpha = 0.5


# --------------------------------------------------------------------------- #
# 7. the update rule: PPO on rollouts + alpha * hybrid on train-split frames
# --------------------------------------------------------------------------- #
class _SpyNet(torch.nn.Module):
    """Wraps a net and records the player features of every forward call."""

    def __init__(self, net: PlayerRankedTopK) -> None:
        super().__init__()
        self.net = net
        self.calls: list[torch.Tensor] = []

    @property
    def top_k(self) -> int:
        return self.net.top_k

    @property
    def hidden(self) -> int:
        return self.net.hidden

    def forward(self, player, bullets, pad):  # type: ignore[override]
        self.calls.append(player.detach().clone())
        return self.net(player, bullets, pad)


_TEACHER_MARKER = 777.0


def _run_one_aux_update(alpha: float = 0.3, minibatch: int = 4):
    device = torch.device("cpu")
    spy = _SpyNet(_tiny_net(seed=6))
    opt = torch.optim.Adam(spy.parameters(), lr=aux.PPO_LR, eps=1e-8)
    rollouts = _fake_rollouts(n_frames=6, n_rollouts=2, seed=13, marker=0.0)
    train = _teacher_tensors(n=16, seed=14, marker=_TEACHER_MARKER)
    metrics = aux.ppo_aux_update(
        spy, opt, rollouts, train, alpha, device, generator=aux.retention_generator(),
        minibatch=minibatch,
    )
    return spy, rollouts, train, metrics


def test_ppo_aux_update_logs_the_full_schema() -> None:
    _spy, rollouts, _train, metrics = _run_one_aux_update()
    required = {
        "ppo_policy_loss",
        "ppo_value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "retention_hybrid_loss",
        "retention_hard_ce",
        "retention_soft_kl",
        "g_ppo_norm",
        "g_ret_norm",
        "g_ret_weighted_norm",
        "grad_ratio",
        "cosine_similarity",
        "alpha",
        "n_samples",
        "optimizer_steps",
        "n_retention_samples",
    }
    assert required <= set(metrics)
    assert metrics["alpha"] == 0.3
    assert metrics["n_samples"] == sum(len(r.actions) for r in rollouts)
    assert metrics["optimizer_steps"] > 0
    assert metrics["g_ret_weighted_norm"] == pytest.approx(
        metrics["alpha"] * metrics["g_ret_norm"], rel=1e-9
    )
    assert metrics["grad_ratio"] == pytest.approx(
        metrics["g_ret_weighted_norm"] / (metrics["g_ppo_norm"] + 1e-8), rel=1e-9
    )
    json.dumps(metrics)  # jsonl-serializable


def test_teacher_train_frames_never_enter_the_ppo_batch() -> None:
    spy, rollouts, _train, metrics = _run_one_aux_update()
    n_rollout_frames = sum(len(r.actions) for r in rollouts)

    ppo_frames = 0
    retention_calls = 0
    for player in spy.calls:
        is_teacher = player[:, 0] == _TEACHER_MARKER
        # a forward is either wholly PPO data or wholly teacher data
        assert bool(is_teacher.all()) or not bool(is_teacher.any())
        if bool(is_teacher.all()):
            retention_calls += 1
        else:
            ppo_frames += int(player.shape[0])

    # PPO sees exactly the rollout frames, once per epoch (plus the
    # diagnostic policy-gradient pass on a matched-size minibatch subset)
    diagnostic_ppo_frames = min(4, n_rollout_frames)  # default minibatch used by _run_one_aux_update
    assert ppo_frames == n_rollout_frames * aux.PPO_EPOCHS + diagnostic_ppo_frames
    assert retention_calls == metrics["optimizer_steps"] + 1  # +1 diagnostic pass
    assert metrics["n_samples"] == n_rollout_frames


def test_ppo_aux_update_never_receives_held_out_tensors() -> None:
    sig = inspect.signature(aux.ppo_aux_update)
    assert not [name for name in sig.parameters if "held" in name]
    src = inspect.getsource(aux.ppo_aux_update)
    assert "held" not in src.replace("held-out teacher", "")


def test_alpha_zero_reproduces_plain_ppo_direction() -> None:
    """With alpha=0 the retention term contributes nothing to the weights."""
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=6, n_rollouts=2, seed=13)
    train = _teacher_tensors(n=16, seed=14, marker=_TEACHER_MARKER)

    def _weights(alpha: float) -> list[torch.Tensor]:
        net = _tiny_net(seed=6)
        opt = torch.optim.Adam(net.parameters(), lr=aux.PPO_LR, eps=1e-8)
        torch.manual_seed(0)
        aux.ppo_aux_update(
            net, opt, rollouts, train, alpha, device,
            generator=aux.retention_generator(), minibatch=4,
        )
        return [p.detach().clone() for p in net.parameters()]

    zero = _weights(0.0)
    nonzero = _weights(0.5)
    assert any(not torch.equal(a, b) for a, b in zip(zero, nonzero))


# --------------------------------------------------------------------------- #
# 7b. RNG discipline: one collection per window, three unshared generators,
#     matched calibration minibatch sizes, alpha=0 bitwise PPO equivalence
# --------------------------------------------------------------------------- #
def _fake_rollout(seed: int, n_frames: int = 6, marker: float = 0.0):
    roll = _fake_rollouts(n_frames=n_frames, n_rollouts=1, seed=seed, marker=marker)[0]
    roll.seed = seed
    return roll


def test_calibration_window_is_collected_once_and_is_the_update_one_batch() -> None:
    """A) The calibration rollouts ARE the update-1 rollouts: one collection,
    one action-sampling stream, no discarded window."""
    device = torch.device("cpu")
    net = _tiny_net(seed=6)

    torch.manual_seed(aux.PPO_TORCH_SEED)
    calib, first_update = aux.calibration_and_first_update_rollouts(
        net, device, episodes_per_update=1, max_frames=8
    )

    assert calib is first_update
    for a, b in zip(calib, first_update):
        assert a.actions == b.actions
        assert torch.equal(
            torch.tensor(a.log_probs), torch.tensor(b.log_probs)
        )
    assert [r.seed for r in calib] == aux.rollout_seed_schedule(0, episodes_per_update=1)


def test_run_aux_arm_never_collects_the_first_window_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A) Snapshot 0 / calibration must not re-collect, and the first
    ``ppo_aux_update`` must be handed the calibration rollout list itself."""
    collected: list[int] = []

    def fake_collect(net, device, seed, max_frames):
        collected.append(int(seed))
        return _fake_rollout(int(seed))

    monkeypatch.setattr(aux, "collect_rollout", fake_collect)

    seen_batches: list[list] = []
    real_update = aux.ppo_aux_update

    def spy_update(net, opt, rollouts, *a, **kw):
        seen_batches.append(rollouts)
        return real_update(net, opt, rollouts, *a, **kw)

    monkeypatch.setattr(aux, "ppo_aux_update", spy_update)

    arm, args = _tiny_arm(tmp_path)

    expected_seeds = aux.rollout_seed_schedule(
        0, episodes_per_update=1
    ) + aux.rollout_seed_schedule(1, episodes_per_update=1)
    assert collected == expected_seeds  # each window collected exactly once
    assert len(seen_batches) == args.updates
    assert [r.seed for r in seen_batches[0]] == aux.rollout_seed_schedule(
        0, episodes_per_update=1
    )
    # D/4) accounting counts the calibration window once, as update 1
    assert arm["total_episodes"] == args.updates * args.episodes_per_update
    assert arm["total_frames"] == 6 * args.updates


def test_diagnostic_draws_never_advance_the_training_retention_stream() -> None:
    """B) Snapshot/diagnostic gradient probes use their own generator, so the
    training retention minibatches are unchanged by the snapshot schedule."""
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=6, n_rollouts=2, seed=13, marker=0.0)
    train = _teacher_tensors(n=16, seed=14, marker=_TEACHER_MARKER)

    def _training_rows(with_diagnostics_between: bool) -> list[torch.Tensor]:
        spy = _SpyNet(_tiny_net(seed=6))
        opt = torch.optim.Adam(spy.parameters(), lr=aux.PPO_LR, eps=1e-8)
        gen = aux.training_generator()
        if with_diagnostics_between:
            aux.grad_alignment(
                spy.net, rollouts, train, device,
                generator=aux.diagnostic_generator(), minibatch=4,
            )
        torch.manual_seed(0)
        aux.ppo_aux_update(
            spy, opt, rollouts, train, 0.3, device,
            generator=gen, diagnostics_generator=aux.diagnostic_generator(),
            minibatch=4,
        )
        teacher_calls = [c for c in spy.calls if bool((c[:, 0] == _TEACHER_MARKER).all())]
        return teacher_calls[1:]  # [0] is the in-update diagnostic probe

    plain = _training_rows(False)
    with_diag = _training_rows(True)
    assert plain and len(plain) == len(with_diag)
    for a, b in zip(plain, with_diag):
        assert torch.equal(a, b)

    # the training stream is exactly the pristine retention sampler stream
    expected_gen = aux.training_generator()
    for rows in plain:
        expected = aux.sample_retention_minibatch(train, device, expected_gen, size=4)
        assert torch.equal(rows, expected["player"])


def test_the_three_retention_generators_are_independent_streams() -> None:
    """B) Calibration / diagnostic / training generators are distinct."""
    seeds = {
        aux.RETENTION_SAMPLER_SEED,
        aux.RETENTION_CALIBRATION_SEED,
        aux.RETENTION_DIAGNOSTIC_SEED,
    }
    assert len(seeds) == 3
    gens = [aux.training_generator(), aux.calibration_generator(), aux.diagnostic_generator()]
    assert len({g.initial_seed() for g in gens}) == 3
    # sharing one generator between training and diagnostics is refused
    shared = aux.training_generator()
    with pytest.raises(ValueError):
        aux.ppo_aux_update(
            _tiny_net(seed=6),
            torch.optim.Adam(_tiny_net(seed=6).parameters()),
            _fake_rollouts(n_frames=6, n_rollouts=1, seed=13),
            _teacher_tensors(n=16, seed=14),
            0.1,
            torch.device("cpu"),
            generator=shared,
            diagnostics_generator=shared,
            minibatch=4,
        )


def test_calibration_uses_matched_minibatch_sizes_for_both_gradients() -> None:
    """C) ``g_ppo`` and ``g_ret`` are measured on the SAME minibatch size."""
    net = _tiny_net(seed=4)
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=8, n_rollouts=2, seed=11)
    train = _teacher_tensors(n=24, seed=12)

    cal = aux.calibrate_alpha(net, rollouts, train, device, minibatch=4)

    n = aux.CALIBRATION_GRAD_SAMPLES
    assert n >= 1
    assert cal["calibration_minibatches"] == n
    assert cal["minibatch_size"] == 4
    assert cal["n_samples"] == cal["n_retention_samples"] == n * 4
    # defaults: both gradients use PPO_MINIBATCH frames per draw
    assert (
        inspect.signature(aux.calibrate_alpha).parameters["minibatch"].default
        == aux.PPO_MINIBATCH
    )
    # and the formula still divides the two matched-size mean norms
    assert cal["alpha"] == pytest.approx(
        aux.TARGET_RETENTION_GRAD_RATIO * cal["g_ppo_norm"] / (cal["g_ret_norm"] + 1e-8),
        rel=1e-9,
    )


def test_alpha_zero_is_bitwise_identical_to_a_ppo_only_step() -> None:
    """D) With alpha=0 the retention term cannot move a single weight."""
    device = torch.device("cpu")
    rollouts = _fake_rollouts(n_frames=6, n_rollouts=2, seed=13)
    train = _teacher_tensors(n=16, seed=14, marker=_TEACHER_MARKER)
    minibatch = 4

    net = _tiny_net(seed=6)
    opt = torch.optim.Adam(net.parameters(), lr=aux.PPO_LR, eps=1e-8)
    torch.manual_seed(0)
    aux.ppo_aux_update(
        net, opt, rollouts, train, 0.0, device,
        generator=aux.training_generator(),
        diagnostics_generator=aux.diagnostic_generator(),
        minibatch=minibatch,
    )

    ref = _tiny_net(seed=6)
    ref_opt = torch.optim.Adam(ref.parameters(), lr=aux.PPO_LR, eps=1e-8)
    batch = ret.rollouts_to_batch(rollouts, device)
    advantages = aux.normalized_advantages(batch)
    n_samples = int(batch["player"].shape[0])
    torch.manual_seed(0)
    for _epoch in range(aux.PPO_EPOCHS):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, minibatch):
            mb = perm[start : start + minibatch]
            dist, value = ref(batch["player"][mb], batch["bullets"][mb], batch["pad"][mb])
            log_prob = dist.log_prob(batch["actions"][mb])
            ratio = torch.exp(log_prob - batch["old_log_probs"][mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - aux.PPO_CLIP, 1 + aux.PPO_CLIP) * advantages[mb]
            loss = (
                -torch.min(surr1, surr2).mean()
                + aux.PPO_VALUE_COEF * torch.nn.functional.mse_loss(value, batch["returns"][mb])
                - aux.PPO_ENTROPY_COEF * dist.entropy().mean()
            )
            ref_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), aux.PPO_MAX_GRAD_NORM)
            ref_opt.step()

    got, want = net.state_dict(), ref.state_dict()
    assert set(got) == set(want)
    for key in want:
        assert torch.equal(got[key], want[key]), key


# --------------------------------------------------------------------------- #
# 8. the arm: frozen alpha, jsonl schema, experimental snapshots
# --------------------------------------------------------------------------- #
def _quick_args(out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        updates=2,
        episodes_per_update=1,
        max_frames=12,
        eval_seeds=[3000],
        eval_max_steps=12,
        out_dir=out_dir,
    )


def _tiny_arm(tmp_path: Path):
    device = torch.device("cpu")
    init_net = _tiny_net(seed=9)
    train = _teacher_tensors(n=16, seed=15)
    held = _teacher_tensors(n=8, seed=16)
    args = _quick_args(tmp_path)
    arm = aux.run_aux_arm(
        init_net,
        device,
        args,
        [0, 2],
        train,
        held,
        tmp_path,
        parent_state_dict_sha256="parent-sd",
        parent_file_sha256="parent-file",
        dataset_hash="dataset-hash",
        ppo_knobs={"updates": 2},
        minibatch=4,
    )
    return arm, args


def test_run_aux_arm_freezes_alpha_and_logs_grad_fields(tmp_path: Path) -> None:
    arm, _args = _tiny_arm(tmp_path)

    cal = arm["alpha_calibration"]
    assert cal["alpha"] > 0.0
    assert cal["target_ratio"] == aux.TARGET_RETENTION_GRAD_RATIO
    assert arm["alpha"] == cal["alpha"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "ppo_aux_updates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["update"] for r in rows] == [1, 2]
    required = {
        "update",
        "ppo_policy_loss",
        "ppo_value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "retention_hybrid_loss",
        "retention_hard_ce",
        "retention_soft_kl",
        "g_ppo_norm",
        "g_ret_norm",
        "g_ret_weighted_norm",
        "grad_ratio",
        "cosine_similarity",
        "alpha",
        "n_samples",
        "optimizer_steps",
        "n_retention_samples",
    }
    for row in rows:
        assert required <= set(row)
        # alpha is frozen for the whole run: never retuned from eval/survival
        assert row["alpha"] == cal["alpha"]


def test_aux_snapshots_are_experimental_and_carry_agreement(tmp_path: Path) -> None:
    arm, _args = _tiny_arm(tmp_path)

    assert [s["update"] for s in arm["snapshots"]] == [0, 2]
    for snap in arm["snapshots"]:
        assert snap["teacher_diagnostics"]["agreement"] is not None
        assert "mean" in snap["evaluation"] and "median" in snap["evaluation"]
        assert snap["grad_alignment"]["cosine_similarity"] is not None
        assert snap["alpha"] == arm["alpha"]
        ckpt = torch.load(snap["checkpoint"], map_location="cpu", weights_only=False)
        assert ckpt["experimental"] is True
        assert ckpt["production_compatible"] is False
        assert ckpt["extra"]["alpha"] == arm["alpha"]
        assert ckpt["extra"]["target_retention_grad_ratio"] == aux.TARGET_RETENTION_GRAD_RATIO

    final = torch.load(arm["final_checkpoint"], map_location="cpu", weights_only=False)
    assert final["experimental"] is True
    assert final["production_compatible"] is False


def test_update_snapshots_store_the_pre_update_on_policy_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot taken after update N must report the alignment measured
    BEFORE update N's optimizer steps, on the rollouts that were on-policy
    for those weights -- the value ``ppo_aux_update`` already returned. It
    must not be recomputed afterwards at post-update weights against the now
    stale/off-policy rollout batch.
    """
    ga_calls: list[int] = []
    real_grad_alignment = aux.grad_alignment

    def counting_grad_alignment(*a, **kw):
        ga_calls.append(1)
        return real_grad_alignment(*a, **kw)

    monkeypatch.setattr(aux, "grad_alignment", counting_grad_alignment)

    returned: list[dict] = []
    real_update = aux.ppo_aux_update

    def spy_update(*a, **kw):
        metrics = real_update(*a, **kw)
        returned.append(metrics)
        return metrics

    monkeypatch.setattr(aux, "ppo_aux_update", spy_update)

    arm, args = _tiny_arm(tmp_path)  # updates=2, snapshots at [0, 2]

    snaps = {s["update"]: s for s in arm["snapshots"]}
    assert set(snaps) == {0, 2}

    # the update-2 snapshot reuses update 2's own pre-update probe verbatim
    update2 = returned[1]
    align2 = snaps[2]["grad_alignment"]
    for key in ("g_ppo_norm", "g_ret_norm", "cosine_similarity", "grad_ratio",
                "g_ret_weighted_norm", "alpha"):
        assert align2[key] == update2[key], key

    # ... and no extra probe is run after the optimizer steps: exactly one
    # probe for snapshot 0 plus one inside each ppo_aux_update.
    assert len(ga_calls) == 1 + args.updates

    # measurement semantics are recorded explicitly, not left implicit
    assert align2["measurement"] == "pre_update_on_policy"
    assert align2["measured_before_update"] == 2
    assert align2["recomputed_post_update"] is False
    assert snaps[0]["grad_alignment"]["measurement"] == "initial_on_policy"
    assert snaps[0]["grad_alignment"]["recomputed_post_update"] is False

    # the same semantics survive into the written snapshot checkpoints' report
    rows = [
        json.loads(line)
        for line in (tmp_path / "ppo_aux_updates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[1]["grad_alignment"]["measurement"] == "pre_update_on_policy"


def test_aux_arm_snapshots_feed_reused_retention_gate(tmp_path: Path) -> None:
    arm, _args = _tiny_arm(tmp_path)
    reference = ret.build_retention_reference(arm["snapshots"][0])
    verdict = ret.evaluate_retention(reference, arm["snapshots"])
    assert verdict["promotion"] is False
    assert "retention_pass" in verdict


# --------------------------------------------------------------------------- #
# 9. dataset: retention trains on the TRAIN split, diagnostics use held-out
# --------------------------------------------------------------------------- #
def test_collect_aux_dataset_materializes_train_and_held_tensors() -> None:
    torch.manual_seed(0)
    teacher = PlayerV1()
    teacher.eval()
    device = torch.device("cpu")

    data = aux.collect_aux_dataset(
        teacher, device, episodes=4, max_steps=25, frames_cap=25
    )

    for key in ("train_tensors", "held_tensors", "hash", "n_train_frames", "n_held_frames"):
        assert key in data
    train, held = data["train_tensors"], data["held_tensors"]
    assert train["player"].shape[0] == data["n_train_frames"] > 0
    assert held["player"].shape[0] == data["n_held_frames"]
    assert set(train) >= {"player", "bullets", "pad", "teacher_logits", "elapsed"}
    # the split is by episode, so the two splits partition the collection
    assert data["n_train_episodes"] + data["n_held_episodes"] == data["n_episodes"]
    assert data["n_held_episodes"] >= 1


# --------------------------------------------------------------------------- #
# 10. CLI: no flag may unlock a PPO knob
# --------------------------------------------------------------------------- #
def test_parser_exposes_only_the_allowed_flags() -> None:
    parser = aux.build_parser()
    options = {opt for action in parser._actions for opt in action.option_strings}
    assert options == {
        "-h",
        "--help",
        "--init-checkpoint",
        "--teacher",
        "--out-dir",
        "--device",
        "--quick",
    }
    help_text = parser.format_help()
    for banned in ("--alpha", "--entropy", "--gamma", "--lr", "--clip", "--lam",
                   "--minibatch", "--epochs", "--updates", "--episodes"):
        assert banned not in help_text


def test_apply_mode_defaults_locks_full_run_and_shrinks_quick_run() -> None:
    full = aux.apply_mode_defaults(argparse.Namespace(quick=False))
    assert full.updates == aux.PPO_UPDATES
    assert full.episodes_per_update == aux.EPISODES_PER_UPDATE
    assert full.max_frames == aux.PPO_MAX_FRAMES
    assert full.eval_seeds == aux.eval_seed_list()
    assert full.eval_max_steps == aux.EVAL_MAX_STEPS

    quick = aux.apply_mode_defaults(argparse.Namespace(quick=True))
    assert quick.updates < aux.PPO_UPDATES
    assert quick.episodes_per_update < aux.EPISODES_PER_UPDATE
    # quick mode never changes the alpha formula or the retention math
    assert aux.TARGET_RETENTION_GRAD_RATIO == 0.15
    assert aux.RETENTION_FRACTION == ret.RETENTION_FRACTION


def test_module_imports_without_training_side_effects() -> None:
    tree = ast.parse(Path(aux.__file__).read_text())
    # no module-level statement may execute training: only imports,
    # assignments, defs/classes and the ``__main__`` guard are allowed.
    for node in tree.body:
        assert isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.FunctionDef,
                ast.ClassDef,
                ast.If,
            ),
        )
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant)  # docstrings only
        if isinstance(node, ast.If):
            assert ast.unparse(node.test) == "__name__ == '__main__'"


def test_no_promotion_anywhere_and_snapshots_are_experimental() -> None:
    src = Path(aux.__file__).read_text()
    assert "production_compatible=True" not in src
    assert src.count("production_compatible=False") == src.count("save_player_checkpoint(")
    assert "promote" not in src.lower().replace("promotion", "")


def test_load_initial_checkpoint_is_phase3s_and_refuses_experimental(tmp_path: Path) -> None:
    net = _tiny_net()
    path = tmp_path / "experimental.pt"
    save_player_checkpoint(
        net, path, source_tool="tests", experimental=True, production_compatible=False
    )
    with pytest.raises(CheckpointError):
        aux.load_initial_checkpoint(path, torch.device("cpu"))

    ok_path = tmp_path / "prod.pt"
    save_player_checkpoint(net, ok_path, source_tool="tests")
    loaded, meta = aux.load_initial_checkpoint(ok_path, torch.device("cpu"))
    assert meta["production_compatible"] is True
    assert meta["experimental"] is False
    assert isinstance(loaded, PlayerRankedTopK)


def test_evaluate_retention_is_reused_and_fails_closed_without_agreement() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.8}
    snapshots = [
        {"update": 0, "evaluation": {"mean": 40.0, "median": 40.0},
         "teacher_diagnostics": {"agreement": 0.8}},
        {"update": 2, "evaluation": {"mean": 40.0, "median": 40.0},
         "teacher_diagnostics": {"agreement": None}},
    ]
    verdict = aux.evaluate_retention(reference, snapshots)
    assert verdict["retention_pass"] is False
    assert verdict["promotion"] is False
    assert "missing held teacher agreement" in verdict["reason"]


# --------------------------------------------------------------------------- #
# 11. quick end-to-end: report schema, alpha frozen, promotion always false
# --------------------------------------------------------------------------- #
def test_quick_end_to_end_writes_report_with_alpha_and_no_promotion(tmp_path: Path) -> None:
    torch.manual_seed(0)
    init_path = tmp_path / "init.pt"
    save_player_checkpoint(_tiny_net(seed=21), init_path, source_tool="tests")

    teacher = PlayerV1(hidden=16)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"hidden": 16, "state_dict": teacher.state_dict()}, teacher_path)

    out_dir = tmp_path / "run"
    args = aux.apply_mode_defaults(
        argparse.Namespace(
            init_checkpoint=init_path,
            teacher=teacher_path,
            out_dir=out_dir,
            device="cpu",
            quick=True,
        )
    )
    report = aux.run_experiment(args, torch.device("cpu"))

    assert report["status"] == "completed"
    assert report["arm_ran"] is True
    assert report["retention"]["promotion"] is False
    alpha = report["alpha_calibration"]["alpha"]
    assert alpha > 0.0 and math.isfinite(alpha)
    assert report["knobs"]["target_retention_grad_ratio"] == aux.TARGET_RETENTION_GRAD_RATIO
    assert report["knobs"]["retention_objective"] == "phase2_ranked_multiseed.hybrid_loss"
    assert report["aux_arm"]["alpha"] == alpha
    assert report["dataset"]["n_train_frames"] > 0

    on_disk = json.loads((out_dir / "report.json").read_text())
    assert on_disk["status"] == "completed"
    assert on_disk["retention"]["promotion"] is False

    rows = [
        json.loads(line)
        for line in (out_dir / "ppo_aux_updates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows and all(r["alpha"] == alpha for r in rows)
    assert all(r["n_retention_samples"] > 0 for r in rows)

    for snap in report["aux_arm"]["snapshots"]:
        ckpt = torch.load(snap["checkpoint"], map_location="cpu", weights_only=False)
        assert ckpt["experimental"] is True and ckpt["production_compatible"] is False
