#!/usr/bin/env python3
"""Phase 3 round-1 BC-retention auxiliary PPO for ``player_ranked_topk``.

Run from the repository root with::

    PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention_aux \
        --init-checkpoint PATH --teacher PATH --out-dir PATH --device cuda

This is the ONE-KNOB follow-up to :mod:`tools.phase3_ranked_ppo_retention`,
whose unregularized scripted PPO washed the PlayerV5 BC skill. Round 1 adds a
single new knob: a BC-retention auxiliary loss scaled by a pre-registered
``alpha``. Every other knob is imported verbatim from the phase 3 control and
is never re-declared or made CLI-overridable here.

Snapshots written by this arm are experimental, non-production artifacts;
nothing here is ever marked production-compatible and promotion is always
false.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_env.agents.player_checkpoints import (
    current_git_commit,
    file_sha256,
    save_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

from tools import phase3_ranked_ppo_retention as ret_mod
from tools.phase2_ranked_multiseed import (
    HYBRID_HARD_WEIGHT,
    HYBRID_SOFT_WEIGHT,
    HYBRID_TEMPERATURE,
    hybrid_loss,
)
from tools.phase3_ranked_ppo_retention import (
    COLLECT_EPISODES,
    COLLECT_MAX_STEPS,
    COLLECT_SEED_START,
    DATA_SEED,
    EPISODES_PER_UPDATE,
    EVAL_MAX_STEPS,
    EVAL_SEED_COUNT,
    EVAL_SEED_START,
    FRAMES_PER_EPISODE_CAP,
    HELD_OUT_FRAC,
    MAX_HELD_AGREEMENT_DROP,
    PPO_CLIP,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_LAMBDA,
    PPO_LR,
    PPO_MAX_FRAMES,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH,
    PPO_ROLLOUT_SEED_START,
    PPO_TORCH_SEED,
    PPO_UPDATES,
    PPO_VALUE_COEF,
    RETENTION_FRACTION,
    SNAPSHOT_UPDATES,
    collect_canonical_dataset,
    collect_rollout,
    evaluate_deterministic,
    evaluate_retention,
    eval_seed_list,
    load_initial_checkpoint,
    rollout_censor_stats,
    rollout_seed_schedule,
    rollouts_to_batch,
    snapshot_schedule,
    summarize_evaluation,
    teacher_diagnostics,
    _REPO_ROOT,
    _git_dirty,
    _write_report,
    build_retention_reference,
    checkpoint_provenance,
    evaluate_initial_gate,
    load_teacher,
    ppo_hyperparameters,
)

# --------------------------------------------------------------------------- #
# the ONLY new knob: pre-registered target ratio between the weighted
# retention gradient and the PPO policy gradient at the starting checkpoint
# (midpoint of the agreed 10-20% band). Calibrated once, then frozen; never
# retuned from eval/survival, never exposed as a CLI flag.
# --------------------------------------------------------------------------- #
TARGET_RETENTION_GRAD_RATIO = 0.15

# Fixed seeds for the THREE never-shared retention-sampling streams, so the
# retention draws are reproducible and independent of the PPO minibatch
# permutation stream -- and of each other:
#   * training    -- the retention minibatches consumed by optimizer steps
#   * calibration -- the alpha calibration draws (once, before any step)
#   * diagnostic  -- grad_alignment probes at snapshots / per-update logging
# Keeping them separate means changing the snapshot schedule (or adding any
# diagnostic probe) can never shift a later training retention minibatch.
RETENTION_SAMPLER_SEED = 12345
RETENTION_CALIBRATION_SEED = 22345
RETENTION_DIAGNOSTIC_SEED = 32345

# Number of matched-size (PPO minibatch, retention minibatch) gradient pairs
# averaged by the alpha calibration.
CALIBRATION_GRAD_SAMPLES = 8

_GRAD_EPS = 1e-8


def retention_generator(seed: int = RETENTION_SAMPLER_SEED) -> torch.Generator:
    """Fresh, deterministically seeded sampler for retention minibatches."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def training_generator() -> torch.Generator:
    """Retention sampler for the optimizer steps inside ``ppo_aux_update``."""
    return retention_generator(RETENTION_SAMPLER_SEED)


def calibration_generator() -> torch.Generator:
    """Retention/PPO minibatch sampler used by ``calibrate_alpha`` only."""
    return retention_generator(RETENTION_CALIBRATION_SEED)


def diagnostic_generator() -> torch.Generator:
    """Retention sampler used by ``grad_alignment`` probes only."""
    return retention_generator(RETENTION_DIAGNOSTIC_SEED)


def sample_retention_minibatch(
    train_tensors: dict[str, torch.Tensor],
    device: torch.device,
    generator: torch.Generator,
    size: int = PPO_MINIBATCH,
) -> dict[str, torch.Tensor]:
    """Draw one retention minibatch from the teacher TRAIN split.

    Held-out teacher frames are never passed here: they are diagnostics/gate
    material only. Sampling is independent of the PPO minibatch permutation,
    so teacher frames never influence advantages/ratios.
    """
    n = int(train_tensors["player"].shape[0])
    k = min(int(size), n)
    idx = torch.randperm(n, generator=generator)[:k]
    return {
        "player": train_tensors["player"][idx].to(device),
        "bullets": train_tensors["bullets"][idx].to(device),
        "pad": train_tensors["pad"][idx].to(device),
        "teacher_logits": train_tensors["teacher_logits"][idx].to(device),
    }


def retention_loss(student_logits, teacher_logits):
    """The BC-retention objective: EXACTLY phase 2's hybrid loss (0.5 hard CE
    + 0.5 soft at T=1). Never a locally invented hard/soft mix."""
    return hybrid_loss(student_logits, teacher_logits)


@torch.no_grad()
def retention_loss_components(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> dict[str, float]:
    """Reporting-only decomposition of :func:`retention_loss`.

    ``hybrid`` is the objective actually optimized (phase 2's ``hybrid_loss``);
    ``hard_ce``/``soft_ce`` are its two weighted terms, and ``soft_kl`` is the
    teacher->student KL, i.e. ``soft_ce`` minus the teacher's own entropy.
    """
    hard_ce = float(F.cross_entropy(student_logits, teacher_logits.argmax(-1)).item())
    teacher_probs = F.softmax(teacher_logits / HYBRID_TEMPERATURE, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    soft_ce = float(-(teacher_probs * student_logp).sum(-1).mean().item())
    teacher_entropy = float(
        -(teacher_probs * F.log_softmax(teacher_logits / HYBRID_TEMPERATURE, dim=-1))
        .sum(-1)
        .mean()
        .item()
    )
    return {
        "hybrid": float(hybrid_loss(student_logits, teacher_logits).item()),
        "hard_ce": hard_ce,
        "soft_ce": soft_ce,
        "soft_kl": soft_ce - teacher_entropy,
    }


# --------------------------------------------------------------------------- #
# gradient-ratio machinery (alpha calibration + per-update diagnostics)
# --------------------------------------------------------------------------- #
def trainable_parameters(net: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Gradient of ``loss`` w.r.t. ``params`` as one concatenated vector.

    Uses ``torch.autograd.grad`` so the live ``.grad`` buffers of the network
    are never touched: computing diagnostics can never leak into an
    optimizer step.
    """
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    return torch.cat(
        [(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)]
    )


def normalized_advantages(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Advantage normalization exactly as in phase3's ``ppo_update``."""
    advantages = batch["advantages"]
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def policy_loss_only(
    net, batch: dict[str, torch.Tensor], advantages: torch.Tensor, index: torch.Tensor | None = None
) -> torch.Tensor:
    """The clipped PPO POLICY loss alone -- no value term, no entropy bonus."""
    if index is None:
        player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
        actions, old_log_probs, adv = batch["actions"], batch["old_log_probs"], advantages
    else:
        player, bullets, pad = batch["player"][index], batch["bullets"][index], batch["pad"][index]
        actions, old_log_probs, adv = (
            batch["actions"][index],
            batch["old_log_probs"][index],
            advantages[index],
        )
    dist, _value = net(player, bullets, pad)
    log_prob = dist.log_prob(actions)
    ratio = torch.exp(log_prob - old_log_probs)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv
    return -torch.min(surr1, surr2).mean()


def grad_alignment(
    net,
    rollouts: list,
    train_tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    generator: torch.Generator,
    minibatch: int = PPO_MINIBATCH,
) -> dict[str, Any]:
    """Norms of (and cosine between) the PPO policy gradient and the
    retention gradient at the CURRENT weights.

    Both gradients are measured on the SAME minibatch size
    (``PPO_MINIBATCH`` by default) -- exactly the scale ``calibrate_alpha``
    uses -- never a full-rollout PPO gradient against a differently-sized
    retention gradient, which would make ``grad_ratio``/``cosine_similarity``
    compare two mismatched quantities. ``g_ppo`` is drawn from a
    ``torch.randperm`` subset of the rollout batch and ``g_ret`` from
    :func:`sample_retention_minibatch`, both fed by the same ``generator``
    stream (mirroring ``calibrate_alpha``). When the rollout batch holds fewer
    frames than ``minibatch``, BOTH draws shrink to the number of available
    PPO frames so the two gradients stay matched-size. Takes no optimizer step
    and leaves ``.grad`` buffers untouched.
    """
    params = trainable_parameters(net)

    batch = rollouts_to_batch(rollouts, device)
    advantages = normalized_advantages(batch)
    n_frames = int(batch["player"].shape[0])
    ppo_k = min(int(minibatch), n_frames)
    index = torch.randperm(n_frames, generator=generator)[:ppo_k].to(device)
    g_ppo = flat_grad(policy_loss_only(net, batch, advantages, index=index), params)

    mb = sample_retention_minibatch(train_tensors, device, generator, size=ppo_k)
    dist, _value = net(mb["player"], mb["bullets"], mb["pad"])
    g_ret = flat_grad(retention_loss(dist.logits, mb["teacher_logits"]), params)

    g_ppo_norm = float(g_ppo.norm().item())
    g_ret_norm = float(g_ret.norm().item())
    cosine = float(
        (g_ppo @ g_ret / (g_ppo.norm() * g_ret.norm() + _GRAD_EPS)).item()
    )
    return {
        "g_ppo_norm": g_ppo_norm,
        "g_ret_norm": g_ret_norm,
        "cosine_similarity": cosine,
        "n_samples": int(index.shape[0]),
        "n_retention_samples": int(mb["player"].shape[0]),
    }


@dataclass(frozen=True)
class FrozenAlpha:
    """The single new knob, frozen after calibration.

    Immutable on purpose: alpha is pre-registered from the gradient-ratio
    formula at the starting checkpoint and is NEVER retuned from evaluation
    or survival metrics during the run.
    """

    alpha: float
    calibration: dict[str, Any]


def calibrate_alpha(
    net,
    rollouts: list,
    train_tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    target_ratio: float = TARGET_RETENTION_GRAD_RATIO,
    minibatch: int = PPO_MINIBATCH,
    n_minibatches: int = CALIBRATION_GRAD_SAMPLES,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Calibrate alpha ONCE at the starting checkpoint, before any optimizer
    step::

        alpha = target_ratio * mean||g_ppo|| / (mean||g_ret|| + 1e-8)

    ``g_ppo`` is the gradient of the PPO policy loss only (no value, no
    entropy) and ``g_ret`` the gradient of the hybrid retention loss on a
    train-split retention minibatch. Both are measured on the SAME minibatch
    size (``PPO_MINIBATCH``) -- the shapes an optimizer step actually sees --
    never a full-batch PPO gradient against a 512-frame retention gradient,
    which would compare two differently-scaled quantities. ``n_minibatches``
    matched pairs are drawn and their norms averaged. Deterministic for fixed
    weights/rollouts/tensors.
    """
    gen = calibration_generator() if generator is None else generator
    params = trainable_parameters(net)

    batch = rollouts_to_batch(rollouts, device)
    advantages = normalized_advantages(batch)
    n_frames = int(batch["player"].shape[0])
    ppo_k = min(int(minibatch), n_frames)

    ppo_norms: list[float] = []
    ret_norms: list[float] = []
    cosines: list[float] = []
    n_ppo_samples = 0
    n_ret_samples = 0

    for _ in range(int(n_minibatches)):
        index = torch.randperm(n_frames, generator=gen)[:ppo_k].to(device)
        g_ppo = flat_grad(policy_loss_only(net, batch, advantages, index=index), params)

        mb = sample_retention_minibatch(train_tensors, device, gen, size=ppo_k)
        dist, _value = net(mb["player"], mb["bullets"], mb["pad"])
        g_ret = flat_grad(retention_loss(dist.logits, mb["teacher_logits"]), params)

        ppo_norms.append(float(g_ppo.norm().item()))
        ret_norms.append(float(g_ret.norm().item()))
        cosines.append(
            float((g_ppo @ g_ret / (g_ppo.norm() * g_ret.norm() + _GRAD_EPS)).item())
        )
        n_ppo_samples += int(index.shape[0])
        n_ret_samples += int(mb["player"].shape[0])

    g_ppo_norm = statistics.fmean(ppo_norms)
    g_ret_norm = statistics.fmean(ret_norms)
    alpha = float(target_ratio * (g_ppo_norm / (g_ret_norm + _GRAD_EPS)))
    result: dict[str, Any] = {
        "g_ppo_norm": g_ppo_norm,
        "g_ret_norm": g_ret_norm,
        "cosine_similarity": statistics.fmean(cosines),
        "n_samples": n_ppo_samples,
        "n_retention_samples": n_ret_samples,
        "calibration_minibatches": int(n_minibatches),
        "minibatch_size": int(minibatch),
        "alpha": alpha,
        "target_ratio": float(target_ratio),
    }
    result["g_ret_weighted_norm"] = alpha * g_ret_norm
    result["grad_ratio"] = result["g_ret_weighted_norm"] / (g_ppo_norm + _GRAD_EPS)
    return result


# --------------------------------------------------------------------------- #
# the update rule: locked PPO objective + alpha * hybrid retention loss
# --------------------------------------------------------------------------- #
def ppo_aux_update(
    net,
    opt: torch.optim.Optimizer,
    rollouts: list,
    train_tensors: dict[str, torch.Tensor],
    alpha: float,
    device: torch.device,
    *,
    generator: torch.Generator,
    diagnostics_generator: torch.Generator | None = None,
    minibatch: int = PPO_MINIBATCH,
) -> dict[str, Any]:
    """One PPO update with the BC-retention auxiliary term::

        total = policy_loss + value_coef*value_loss - entropy_coef*entropy
                + alpha * hybrid_loss(student_logits, teacher_logits)

    The PPO minibatch comes from on-policy scripted rollouts only; the
    retention minibatch is drawn independently from the teacher TRAIN split
    at every optimizer step. Teacher frames never contribute to advantages,
    old log-probs, GAE or the PPO ratio. Optimizer, clipping, epochs and
    minibatch size are identical to phase3's ``ppo_update``.

    ``generator`` feeds the training retention minibatches ONLY; the
    per-update gradient-alignment probe draws from ``diagnostics_generator``,
    a separate stream, so logging/diagnostics can never shift the training
    retention samples.
    """
    if diagnostics_generator is None:
        diagnostics_generator = diagnostic_generator()
    if diagnostics_generator is generator:
        raise ValueError(
            "the diagnostic and training retention generators must be distinct "
            "streams: sharing them lets diagnostics shift training samples"
        )

    diagnostics = grad_alignment(
        net, rollouts, train_tensors, device,
        generator=diagnostics_generator, minibatch=minibatch,
    )

    batch = rollouts_to_batch(rollouts, device)
    player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
    actions, old_log_probs = batch["actions"], batch["old_log_probs"]
    old_values, returns = batch["values"], batch["returns"]
    advantages = normalized_advantages(batch)

    n_samples = int(player.shape[0])
    optimizer_steps = 0
    kl_sum = 0.0
    clip_sum = 0.0
    entropy_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    total_loss_sum = 0.0
    retention_sum = 0.0
    retention_hard_sum = 0.0
    retention_soft_kl_sum = 0.0
    retention_soft_ce_sum = 0.0
    retention_frames = 0
    seen = 0

    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, minibatch):
            mb = perm[start : start + minibatch]
            dist, value = net(player[mb], bullets[mb], pad[mb])
            log_prob = dist.log_prob(actions[mb])
            ratio = torch.exp(log_prob - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns[mb])
            entropy = dist.entropy().mean()

            ret_mb = sample_retention_minibatch(
                train_tensors, device, generator, size=minibatch
            )
            ret_dist, _ret_value = net(ret_mb["player"], ret_mb["bullets"], ret_mb["pad"])
            retention = retention_loss(ret_dist.logits, ret_mb["teacher_logits"])

            loss = (
                policy_loss
                + PPO_VALUE_COEF * value_loss
                - PPO_ENTROPY_COEF * entropy
                + alpha * retention
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), PPO_MAX_GRAD_NORM)
            opt.step()
            optimizer_steps += 1

            with torch.no_grad():
                approx_kl = float((old_log_probs[mb] - log_prob).mean().item())
                clip_fraction = float((torch.abs(ratio - 1.0) > PPO_CLIP).float().mean().item())
                parts = retention_loss_components(
                    ret_dist.logits.detach(), ret_mb["teacher_logits"]
                )
            bs = int(mb.shape[0])
            kl_sum += approx_kl * bs
            clip_sum += clip_fraction * bs
            entropy_sum += float(entropy.item()) * bs
            policy_loss_sum += float(policy_loss.item()) * bs
            value_loss_sum += float(value_loss.item()) * bs
            total_loss_sum += float(loss.item()) * bs
            seen += bs

            ret_bs = int(ret_mb["player"].shape[0])
            retention_sum += parts["hybrid"] * ret_bs
            retention_hard_sum += parts["hard_ce"] * ret_bs
            retention_soft_kl_sum += parts["soft_kl"] * ret_bs
            retention_soft_ce_sum += parts["soft_ce"] * ret_bs
            retention_frames += ret_bs

    with torch.no_grad():
        ret_var = torch.var(returns)
        explained_variance = float(1.0 - torch.var(returns - old_values) / (ret_var + 1e-8))

    denom = max(seen, 1)
    ret_denom = max(retention_frames, 1)
    g_ret_weighted_norm = float(alpha) * diagnostics["g_ret_norm"]
    # The alignment probe is taken BEFORE any optimizer step of this update, on
    # the rollout batch that is on-policy for those weights. It is returned
    # verbatim so callers (snapshots/logs) can record it instead of recomputing
    # it afterwards at post-update weights against a now stale rollout batch.
    alignment = dict(diagnostics)
    alignment["alpha"] = float(alpha)
    alignment["g_ret_weighted_norm"] = g_ret_weighted_norm
    alignment["grad_ratio"] = g_ret_weighted_norm / (diagnostics["g_ppo_norm"] + _GRAD_EPS)
    alignment["measurement"] = "pre_update_on_policy"
    alignment["recomputed_post_update"] = False
    return {
        "approx_kl": kl_sum / denom,
        "clip_fraction": clip_sum / denom,
        "explained_variance": explained_variance,
        "entropy": entropy_sum / denom,
        "ppo_policy_loss": policy_loss_sum / denom,
        "ppo_value_loss": value_loss_sum / denom,
        "total_loss": total_loss_sum / denom,
        "retention_hybrid_loss": retention_sum / ret_denom,
        "retention_hard_ce": retention_hard_sum / ret_denom,
        "retention_soft_kl": retention_soft_kl_sum / ret_denom,
        "retention_soft_ce": retention_soft_ce_sum / ret_denom,
        "alpha": float(alpha),
        "g_ppo_norm": diagnostics["g_ppo_norm"],
        "g_ret_norm": diagnostics["g_ret_norm"],
        "g_ret_weighted_norm": g_ret_weighted_norm,
        "grad_ratio": g_ret_weighted_norm / (diagnostics["g_ppo_norm"] + _GRAD_EPS),
        "cosine_similarity": diagnostics["cosine_similarity"],
        "grad_alignment": alignment,
        "optimizer_steps": optimizer_steps,
        "n_samples": n_samples,
        "n_retention_samples": retention_frames,
    }


# --------------------------------------------------------------------------- #
# rollout collection: the calibration window IS the update-1 window
# --------------------------------------------------------------------------- #
def collect_rollout_window(
    net, device: torch.device, seeds: list[int], *, max_frames: int
) -> list:
    """Collect one rollout per seed with the phase3 collector."""
    return [collect_rollout(net, device, seed, max_frames=max_frames) for seed in seeds]


def calibration_and_first_update_rollouts(
    net,
    device: torch.device,
    *,
    episodes_per_update: int = EPISODES_PER_UPDATE,
    max_frames: int = PPO_MAX_FRAMES,
) -> tuple[list, list]:
    """Collect the update-1 rollout window EXACTLY ONCE and return it twice.

    Alpha calibration and the first PPO update share one and the same batch:
    collecting the window a second time for update 1 would both burn a second
    action-sampling stream from the global torch RNG (so update 1 would train
    on different actions than the ones alpha was calibrated on) and pay for
    the same env seeds twice. The returned objects are the same list.
    """
    seeds = rollout_seed_schedule(0, episodes_per_update=episodes_per_update)
    rollouts = collect_rollout_window(net, device, seeds, max_frames=max_frames)
    return rollouts, rollouts


# --------------------------------------------------------------------------- #
# canonical teacher dataset (collected exactly once, via the phase3 helper)
# --------------------------------------------------------------------------- #
def collect_aux_dataset(teacher, device: torch.device, **kwargs: Any) -> dict[str, Any]:
    """The phase3 canonical dataset, plus the TRAIN split tensors.

    Collection itself is never reimplemented here: it delegates to
    ``phase3_ranked_ppo_retention.collect_canonical_dataset`` (same data
    seed, same episode/frame caps, same episode-level split, same identity
    hash). The retention term trains on the train split only; the held-out
    split stays reserved for agreement diagnostics and gates.
    """
    return collect_canonical_dataset(teacher, device, include_train_tensors=True, **kwargs)


# --------------------------------------------------------------------------- #
# aux PPO arm: calibrate alpha once, then train with the locked knobs
# --------------------------------------------------------------------------- #
def run_aux_arm(
    init_net,
    device: torch.device,
    args: argparse.Namespace,
    snapshot_updates: list[int],
    train_tensors: dict[str, torch.Tensor],
    diagnostic_tensors: dict[str, torch.Tensor] | None,
    out_dir: Path,
    *,
    parent_state_dict_sha256: str,
    parent_file_sha256: str,
    dataset_hash: str,
    ppo_knobs: dict[str, Any],
    minibatch: int = PPO_MINIBATCH,
) -> dict[str, Any]:
    """Train the auxiliary-retention PPO arm from a fresh clone of the loaded
    initial state (the init checkpoint is never mutated).

    ``alpha`` is calibrated once, at the starting checkpoint, BEFORE any
    optimizer step, from the update-1 rollout batch -- collected exactly once
    with the same collector and rollout-seed schedule as the phase3 control,
    then reused verbatim as the update-1 PPO batch, so nothing is collected
    twice and no rollout is discarded. Alpha is frozen for the whole run.
    Three never-shared retention streams are used (calibration, diagnostics,
    training), so snapshots/diagnostics can never shift training samples.
    ``diagnostic_tensors`` are the out-of-sample teacher frames used for
    agreement diagnostics/gates only -- they are never passed to the update.
    Every checkpoint written here is experimental and non-production;
    promotion is always false.
    """
    net = PlayerRankedTopK(top_k=init_net.top_k, hidden=init_net.hidden).to(device)
    net.load_state_dict(init_net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR, eps=1e-8)
    torch.manual_seed(PPO_TORCH_SEED)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "ppo_aux_updates.jsonl"

    generator = training_generator()
    calib_gen = calibration_generator()
    diag_gen = diagnostic_generator()

    # ---- alpha calibration (no optimizer step has happened yet) ---------- #
    # The window is collected ONCE and is reused verbatim as the update-1 PPO
    # batch: alpha is calibrated on exactly the rollouts update 1 trains on.
    calib_rollouts, first_update_rollouts = calibration_and_first_update_rollouts(
        net,
        device,
        episodes_per_update=args.episodes_per_update,
        max_frames=args.max_frames,
    )
    calib_seeds = [r.seed for r in calib_rollouts]
    calibration = calibrate_alpha(
        net,
        calib_rollouts,
        train_tensors,
        device,
        minibatch=minibatch,
        generator=calib_gen,
    )
    calibration["rollout_seed_window"] = calib_seeds
    frozen = FrozenAlpha(alpha=float(calibration["alpha"]), calibration=calibration)
    alpha = frozen.alpha

    snapshots: list[dict[str, Any]] = []
    total_episodes = 0
    total_frames = 0
    optimizer_steps = 0
    rollout_seed_window = calib_seeds

    def _pack_extra(update: int, eval_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "update": update,
            "parent_state_dict_sha256": parent_state_dict_sha256,
            "parent_file_sha256": parent_file_sha256,
            "dataset_hash": dataset_hash,
            "eval_summary": eval_summary,
            "ppo_knobs": ppo_knobs,
            "rollout_seed_window": rollout_seed_window,
            "eval_seed_window": args.eval_seeds,
            "alpha": alpha,
            "target_retention_grad_ratio": TARGET_RETENTION_GRAD_RATIO,
            "alpha_calibration": calibration,
            "retention_objective": "phase2_ranked_multiseed.hybrid_loss",
        }

    def _snapshot(update: int, rollouts: list, alignment: dict[str, Any] | None = None) -> None:
        eval_results = evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
        eval_summary = summarize_evaluation(eval_results)
        if diagnostic_tensors is not None:
            diag = teacher_diagnostics(net, diagnostic_tensors, device)
        else:
            diag = {"agreement": None}
        if alignment is None:
            # Snapshot 0 only: no update has happened yet, so the probe at the
            # CURRENT (initial) weights is itself pre-update and on-policy for
            # the calibration/update-1 window.
            alignment = grad_alignment(
                net, rollouts, train_tensors, device, generator=diag_gen, minibatch=minibatch
            )
            alignment["alpha"] = alpha
            alignment["g_ret_weighted_norm"] = alpha * alignment["g_ret_norm"]
            alignment["grad_ratio"] = alignment["g_ret_weighted_norm"] / (
                alignment["g_ppo_norm"] + _GRAD_EPS
            )
            alignment["measurement"] = "initial_on_policy"
            alignment["measured_before_update"] = 1
            alignment["recomputed_post_update"] = False
        else:
            # Post-update snapshots reuse the probe ``ppo_aux_update`` already
            # took BEFORE its optimizer steps. Recomputing it here would
            # measure post-update weights against the pre-update (now
            # off-policy) rollout batch -- a different quantity.
            alignment = dict(alignment)
        ckpt_path = out_dir / f"ppo_aux_update_{update}.pt"
        save_player_checkpoint(
            net,
            ckpt_path,
            source_tool="phase3_ranked_ppo_retention_aux",
            experimental=True,
            production_compatible=False,
            extra=_pack_extra(update, eval_summary),
        )
        snapshots.append(
            {
                "update": update,
                "evaluation": eval_summary,
                "teacher_diagnostics": diag,
                "grad_alignment": alignment,
                "alpha": alpha,
                "checkpoint": str(ckpt_path),
                "state_dict_sha256": state_dict_sha256(net.state_dict()),
            }
        )

    if 0 in snapshot_updates:
        # Snapshot 0 only INSPECTS the already-collected window: it never
        # collects again and never samples an action.
        _snapshot(0, calib_rollouts)

    with jsonl_path.open("w") as jsonl_f:
        for update in range(1, args.updates + 1):
            if update == 1:
                rollouts = first_update_rollouts
                seeds = calib_seeds
            else:
                seeds = rollout_seed_schedule(
                    update - 1, episodes_per_update=args.episodes_per_update
                )
                rollouts = collect_rollout_window(
                    net, device, seeds, max_frames=args.max_frames
                )
            rollout_seed_window = seeds
            metrics = ppo_aux_update(
                net,
                opt,
                rollouts,
                train_tensors,
                alpha,
                device,
                generator=generator,
                diagnostics_generator=diag_gen,
                minibatch=minibatch,
            )
            optimizer_steps += int(metrics["optimizer_steps"])
            metrics["grad_alignment"]["measured_before_update"] = update
            total_episodes += len(rollouts)
            total_frames += sum(len(r.actions) for r in rollouts)
            row = dict(metrics)
            row["update"] = update
            row["scripted_survival_mean"] = statistics.mean([r.elapsed for r in rollouts])
            row["rollout_censoring"] = rollout_censor_stats(rollouts)
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()
            if update in snapshot_updates:
                _snapshot(update, rollouts, alignment=metrics["grad_alignment"])

    final_eval_summary = summarize_evaluation(
        evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
    )
    final_path = out_dir / "ppo_aux_final.pt"
    save_player_checkpoint(
        net,
        final_path,
        source_tool="phase3_ranked_ppo_retention_aux",
        experimental=True,
        production_compatible=False,
        extra=_pack_extra(args.updates, final_eval_summary),
    )

    return {
        "alpha": alpha,
        "alpha_calibration": calibration,
        "optimizer_steps": optimizer_steps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "snapshots": snapshots,
        "final_checkpoint": str(final_path),
        "updates_jsonl": str(jsonl_path),
    }


# --------------------------------------------------------------------------- #
# CLI / experiment orchestration
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The CLI intentionally exposes NO knob that could unlock a locked PPO
    hyperparameter (or alpha): the whole point of round 1 is that exactly one
    new knob exists and it is pre-registered in code, not on the command
    line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase3_ranked_ppo_retention_aux"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in the update budget / seed windows / gate thresholds, exactly as
    the phase3 control does. Quick mode shrinks the budget for smoke tests and
    relaxes the initial gate only -- it never changes the alpha formula or the
    retention math."""
    return ret_mod.apply_mode_defaults(args)


def run_experiment(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Strict, fail-closed orchestration of the auxiliary-retention arm.

    Nothing runs before the pre-registered initial gate passes; alpha is
    calibrated once at the starting checkpoint and frozen; the retention
    verdict reuses phase3's ``evaluate_retention`` and promotion is always
    false.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_net, init_meta = load_initial_checkpoint(args.init_checkpoint, device)
    init_prov = checkpoint_provenance(args.init_checkpoint, init_meta)
    provenance = {
        "init_checkpoint": init_prov,
        "teacher": {"file_sha256": file_sha256(args.teacher)},
        "git_commit": current_git_commit(_REPO_ROOT),
        "dirty": _git_dirty(_REPO_ROOT),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    knobs = ppo_hyperparameters()
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
    knobs["target_retention_grad_ratio"] = TARGET_RETENTION_GRAD_RATIO
    knobs["retention_objective"] = "phase2_ranked_multiseed.hybrid_loss"
    knobs["retention_hard_weight"] = HYBRID_HARD_WEIGHT
    knobs["retention_soft_weight"] = HYBRID_SOFT_WEIGHT
    knobs["retention_temperature"] = HYBRID_TEMPERATURE
    knobs["retention_sampler_seed"] = RETENTION_SAMPLER_SEED
    knobs["retention_calibration_seed"] = RETENTION_CALIBRATION_SEED
    knobs["retention_diagnostic_seed"] = RETENTION_DIAGNOSTIC_SEED
    knobs["calibration_grad_samples"] = CALIBRATION_GRAD_SAMPLES

    gate_summary = summarize_evaluation(
        evaluate_deterministic(init_net, device, args.eval_seeds, args.eval_max_steps)
    )
    initial_gate = evaluate_initial_gate(
        gate_summary,
        mean_min=args.initial_gate_mean_min,
        median_min=args.initial_gate_median_min,
    )

    report: dict[str, Any] = {
        "status": None,
        "arm_ran": False,
        "initial_gate": initial_gate,
        "provenance": provenance,
        "knobs": knobs,
        "dataset": None,
        "alpha_calibration": None,
        "aux_arm": None,
        "retention": None,
    }

    if not initial_gate["gate_pass"]:
        report["status"] = "failed_closed"
        _write_report(out_dir, report)
        return report

    teacher = load_teacher(args.teacher, device)
    dataset = collect_aux_dataset(
        teacher,
        device,
        episodes=args.data_episodes,
        seed_start=COLLECT_SEED_START,
        max_steps=args.data_max_steps,
        frames_cap=args.data_frames_cap,
        held_frac=HELD_OUT_FRAC,
        teacher_path=args.teacher,
    )
    report["dataset"] = {
        "hash": dataset["hash"],
        "n_episodes": dataset["n_episodes"],
        "n_held_episodes": dataset["n_held_episodes"],
        "n_train_episodes": dataset["n_train_episodes"],
        "n_held_frames": dataset["n_held_frames"],
        "n_train_frames": dataset["n_train_frames"],
        "teacher_file_sha256": dataset["teacher_file_sha256"],
    }

    snapshot_updates = snapshot_schedule(args.updates)
    aux_arm = run_aux_arm(
        init_net,
        device,
        args,
        snapshot_updates,
        dataset["train_tensors"],
        dataset["held_tensors"],
        out_dir,
        parent_state_dict_sha256=init_prov["state_dict_sha256"],
        parent_file_sha256=init_prov["file_sha256"],
        dataset_hash=dataset["hash"],
        ppo_knobs=knobs,
    )
    report["aux_arm"] = {k: v for k, v in aux_arm.items()}
    report["alpha_calibration"] = aux_arm["alpha_calibration"]

    reference = build_retention_reference(aux_arm["snapshots"][0])
    report["retention"] = evaluate_retention(reference, aux_arm["snapshots"])
    report["retention_reference"] = reference

    report["status"] = "completed"
    report["arm_ran"] = True
    _write_report(out_dir, report)
    return report


def main() -> None:
    args = apply_mode_defaults(build_parser().parse_args())
    device = torch.device(args.device)
    report = run_experiment(args, device)
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm_ran": report["arm_ran"],
                "alpha": (report["alpha_calibration"] or {}).get("alpha"),
            }
        )
    )


__all__ = [
    "FrozenAlpha",
    "apply_mode_defaults",
    "build_parser",
    "main",
    "run_experiment",
    "collect_aux_dataset",
    "run_aux_arm",
    "calibrate_alpha",
    "flat_grad",
    "ppo_aux_update",
    "grad_alignment",
    "normalized_advantages",
    "policy_loss_only",
    "trainable_parameters",
    "RETENTION_SAMPLER_SEED",
    "RETENTION_CALIBRATION_SEED",
    "RETENTION_DIAGNOSTIC_SEED",
    "CALIBRATION_GRAD_SAMPLES",
    "TARGET_RETENTION_GRAD_RATIO",
    "retention_generator",
    "training_generator",
    "calibration_generator",
    "diagnostic_generator",
    "collect_rollout_window",
    "calibration_and_first_update_rollouts",
    "sample_retention_minibatch",
    "retention_loss_components",
    "HYBRID_HARD_WEIGHT",
    "HYBRID_SOFT_WEIGHT",
    "HYBRID_TEMPERATURE",
    "hybrid_loss",
    "retention_loss",
    "COLLECT_EPISODES",
    "COLLECT_MAX_STEPS",
    "COLLECT_SEED_START",
    "DATA_SEED",
    "EPISODES_PER_UPDATE",
    "EVAL_MAX_STEPS",
    "EVAL_SEED_COUNT",
    "EVAL_SEED_START",
    "FRAMES_PER_EPISODE_CAP",
    "HELD_OUT_FRAC",
    "MAX_HELD_AGREEMENT_DROP",
    "PPO_CLIP",
    "PPO_ENTROPY_COEF",
    "PPO_EPOCHS",
    "PPO_GAMMA",
    "PPO_LAMBDA",
    "PPO_LR",
    "PPO_MAX_FRAMES",
    "PPO_MAX_GRAD_NORM",
    "PPO_MINIBATCH",
    "PPO_ROLLOUT_SEED_START",
    "PPO_TORCH_SEED",
    "PPO_UPDATES",
    "PPO_VALUE_COEF",
    "RETENTION_FRACTION",
    "SNAPSHOT_UPDATES",
    "collect_canonical_dataset",
    "collect_rollout",
    "evaluate_deterministic",
    "evaluate_retention",
    "eval_seed_list",
    "load_initial_checkpoint",
    "rollout_seed_schedule",
    "rollouts_to_batch",
    "snapshot_schedule",
    "teacher_diagnostics",
]


if __name__ == "__main__":
    main()
