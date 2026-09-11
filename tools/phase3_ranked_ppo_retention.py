#!/usr/bin/env python3
"""Phase 3 BC-retention control for ``player_ranked_topk``.

Run from the repository root with::

    PYTHONPATH=. python -m tools.phase3_ranked_ppo_retention \
        --init-checkpoint PATH --teacher PATH --out-dir PATH --device cuda

The experiment compares a frozen/no-update control with scripted-only PPO
starting from the exact same BC checkpoint. PPO snapshots are experimental
and are never marked production-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4, encode_obs
from qrokkun_env.agents.player_checkpoints import (
    CheckpointError,
    current_git_commit,
    file_sha256,
    load_ranked_top_k_checkpoint,
    save_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_env.agents.player_v1 import PlayerV1
from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.train import player_v1 as scripted_ppo

from tools.phase2_distill_v1_to_v4 import collect_dataset, frames_to_tensors
from tools.phase2_ranked_multiseed import dataset_identity

_REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# canonical dataset constants (pre-registered, collected exactly once)
# --------------------------------------------------------------------------- #
DATA_SEED = 4
COLLECT_EPISODES = 240
COLLECT_SEED_START = 20000
COLLECT_MAX_STEPS = 4200
FRAMES_PER_EPISODE_CAP = 700
HELD_OUT_FRAC = 0.12

# --------------------------------------------------------------------------- #
# PPO hyperparameters -- copied verbatim from the current scripted-Player PPO
# CLI defaults (qrokkun_env/train/player_v1.py), never reinvented.
# --------------------------------------------------------------------------- #
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_ENTROPY_COEF = 0.02
PPO_VALUE_COEF = 0.5
PPO_EPOCHS = 4
PPO_MINIBATCH = 512
EPISODES_PER_UPDATE = 8
PPO_MAX_GRAD_NORM = 1.0

# reuse the project's GAE/reward implementations verbatim -- never reinvented.
gae = scripted_ppo.gae
shaped_reward = scripted_ppo.shaped_reward

# --------------------------------------------------------------------------- #
# update budget / rollout / evaluation / snapshot schedules
# --------------------------------------------------------------------------- #
PPO_UPDATES = 200
PPO_MAX_FRAMES = 4200
PPO_ROLLOUT_SEED_START = 50000
PPO_TORCH_SEED = 0

EVAL_SEED_START = 3000
EVAL_SEED_COUNT = 30
EVAL_MAX_STEPS = 4200

SNAPSHOT_UPDATES: tuple[int, ...] = (0, 10, 25, 50, 100, 200)

# --------------------------------------------------------------------------- #
# pre-registered interpretation (retention, never promotion of a checkpoint)
# --------------------------------------------------------------------------- #
INITIAL_GATE_MEAN_MIN = 25.0
INITIAL_GATE_MEDIAN_MIN = 25.0
RETENTION_FRACTION = 0.80
MAX_HELD_AGREEMENT_DROP = 0.05


def ppo_hyperparameters() -> dict[str, Any]:
    """Every PPO knob used by the control, as a JSON-serializable dict."""
    return {
        "lr": PPO_LR,
        "gamma": PPO_GAMMA,
        "lam": PPO_LAMBDA,
        "clip": PPO_CLIP,
        "entropy_coef": PPO_ENTROPY_COEF,
        "value_coef": PPO_VALUE_COEF,
        "ppo_epochs": PPO_EPOCHS,
        "minibatch": PPO_MINIBATCH,
        "max_grad_norm": PPO_MAX_GRAD_NORM,
        "optimizer": "adam",
        "adam_eps": 1e-8,
        "advantage_normalization": True,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "max_frames_per_episode": PPO_MAX_FRAMES,
        "updates": PPO_UPDATES,
        "rollout_seed_start": PPO_ROLLOUT_SEED_START,
        "torch_seed": PPO_TORCH_SEED,
        "value_bootstrap": False,
        "truncation_treated_as_terminal": True,
    }


def rollout_seed_schedule(update: int, episodes_per_update: int = EPISODES_PER_UPDATE) -> list[int]:
    """Fixed, declared, deterministic per-update rollout seeds.

    Seeds never repeat across updates and never overlap the evaluation seed
    window, since the rollout seed space starts well above it. This is the
    single declared helper for rollout seeds -- callers (including
    ``run_ppo_arm``) must call it rather than reimplementing the arithmetic
    inline, so an actual (e.g. quick-mode) ``episodes_per_update`` always
    stays consistent with the seeds actually used.
    """
    start = PPO_ROLLOUT_SEED_START + update * episodes_per_update
    return list(range(start, start + episodes_per_update))


def eval_seed_list() -> list[int]:
    """Canonical deterministic evaluation seed window."""
    return list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SEED_COUNT))


def snapshot_schedule(final_update: int) -> list[int]:
    """Pre-registered snapshot updates, filtered to ``final_update`` with the
    final update always included."""
    updates = {u for u in SNAPSHOT_UPDATES if u <= final_update}
    updates.add(final_update)
    return sorted(updates)


# --------------------------------------------------------------------------- #
# strict checkpoint loading / provenance
# --------------------------------------------------------------------------- #
def load_initial_checkpoint(
    path: Path | str, device: torch.device
) -> tuple[PlayerRankedTopK, dict[str, Any]]:
    """Strictly load the initial ``player_ranked_topk`` production checkpoint.

    Delegates to :mod:`qrokkun_env.agents.player_checkpoints`, which refuses
    legacy PlayerV1/PlayerV4 dict checkpoints, unknown architectures, and
    tampered state dicts by raising ``CheckpointError``.
    """
    net, meta = load_ranked_top_k_checkpoint(path, device, eval_mode=True)
    if meta.get("production_compatible") is not True or meta.get("experimental") is not False:
        raise CheckpointError(
            "initial checkpoint must be explicitly production-compatible "
            "(production_compatible=True, experimental=False)"
        )
    return net, meta


def checkpoint_provenance(path: Path | str, meta: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable provenance record for the loaded checkpoint."""
    return {
        "file_sha256": file_sha256(path),
        "state_dict_sha256": meta["state_dict_sha256"],
        "actions": list(ACTIONS),
        "observation": {
            "player_feat": PLAYER_FEAT_V4,
            "bullet_feat": BULLET_FEAT_V4,
            "max_bullets": MAX_BULLETS_V4,
        },
        "architecture": meta["architecture"],
        "schema_verified": True,
    }


# --------------------------------------------------------------------------- #
# rollout container (populated by collect_rollout, below -- future slice)
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    seed: int
    player: list[np.ndarray] = field(default_factory=list)
    bullets: list[np.ndarray] = field(default_factory=list)
    pad: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    elapsed: float = 0.0
    censored: bool = True


# --------------------------------------------------------------------------- #
# on-policy sampled collection (never argmax; deterministic seeded reset only)
# --------------------------------------------------------------------------- #
def collect_rollout(net: PlayerRankedTopK, device: torch.device, seed: int, max_frames: int) -> Rollout:
    """Collect one on-policy rollout from a deterministically seeded env reset.

    Actions are SAMPLED from the current policy (never argmax) so the batch
    reflects the policy actually being optimized. Old log-probs, values and
    dones are recorded at collection time for the PPO ratio/clip terms.
    """
    env = Qrokkun26Env(seed=seed)
    env.reset(seed=seed)
    player_l: list[np.ndarray] = []
    bullets_l: list[np.ndarray] = []
    pad_l: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    elapsed = env.elapsed
    for _ in range(max_frames):
        player, bullets, pad = encode_obs(env)
        p_t = torch.tensor(player, dtype=torch.float32, device=device).unsqueeze(0)
        b_t = torch.tensor(bullets, dtype=torch.float32, device=device).unsqueeze(0)
        m_t = torch.tensor(pad, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, value = net(p_t, b_t, m_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        player_l.append(player)
        bullets_l.append(bullets)
        pad_l.append(pad)
        actions.append(int(action.item()))
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        _obs, base, done, info = env.step(int(action.item()))
        rewards.append(shaped_reward(env, float(base), bool(done)))
        dones.append(bool(done))
        elapsed = float(info.get("elapsed", env.elapsed))
        if done:
            break
    censored = (not dones[-1]) if dones else True
    return Rollout(
        seed=seed,
        player=player_l,
        bullets=bullets_l,
        pad=pad_l,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        elapsed=elapsed,
        censored=censored,
    )


# --------------------------------------------------------------------------- #
# GAE / batch construction (per-episode boundaries; never bleed across them)
# --------------------------------------------------------------------------- #
def compute_gae_for_rollouts(
    rollouts: list[Rollout], gamma: float, lam: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages/returns computed per rollout (episode) then concatenated,
    so advantages never bleed across episode boundaries."""
    adv_parts: list[torch.Tensor] = []
    ret_parts: list[torch.Tensor] = []
    for r in rollouts:
        adv, ret_t = gae(r.rewards, r.values, r.dones, gamma, lam, device)
        adv_parts.append(adv)
        ret_parts.append(ret_t)
    return torch.cat(adv_parts), torch.cat(ret_parts)


def rollouts_to_batch(rollouts: list[Rollout], device: torch.device) -> dict[str, torch.Tensor]:
    """Flatten rollouts into a PPO batch. Contains only on-policy rollout
    data -- teacher frames/logits NEVER enter a PPO batch."""
    player_l: list[np.ndarray] = []
    bullets_l: list[np.ndarray] = []
    pad_l: list[np.ndarray] = []
    actions_l: list[int] = []
    old_lp_l: list[float] = []
    values_l: list[float] = []
    for r in rollouts:
        player_l.extend(r.player)
        bullets_l.extend(r.bullets)
        pad_l.extend(r.pad)
        actions_l.extend(r.actions)
        old_lp_l.extend(r.log_probs)
        values_l.extend(r.values)
    advantages, returns = compute_gae_for_rollouts(rollouts, PPO_GAMMA, PPO_LAMBDA, device)
    return {
        "player": torch.tensor(np.stack(player_l), dtype=torch.float32, device=device),
        "bullets": torch.tensor(np.stack(bullets_l), dtype=torch.float32, device=device),
        "pad": torch.tensor(np.stack(pad_l), dtype=torch.bool, device=device),
        "actions": torch.tensor(actions_l, dtype=torch.int64, device=device),
        "old_log_probs": torch.tensor(old_lp_l, dtype=torch.float32, device=device),
        "values": torch.tensor(values_l, dtype=torch.float32, device=device),
        "advantages": advantages,
        "returns": returns,
    }


def rollout_censor_stats(rollouts: list[Rollout]) -> dict[str, Any]:
    """How many/what fraction of this update's rollouts were censored
    (timed out at ``max_frames`` without a true terminal ``done``). The
    reused ``player_v1.gae`` always appends a terminal bootstrap value of
    zero regardless of whether the rollout actually terminated, so a
    censored (truncated) rollout is treated identically to a terminated
    one -- this never changes PPO semantics, it only makes that fact
    honestly observable per update."""
    n = len(rollouts)
    censor_count = sum(1 for r in rollouts if r.censored)
    return {
        "censor_count": censor_count,
        "censor_rate": censor_count / n if n else 0.0,
        "n_rollouts": n,
    }


def ppo_update(
    net: PlayerRankedTopK, opt: torch.optim.Optimizer, rollouts: list[Rollout], device: torch.device
) -> dict[str, Any]:
    """Clipped PPO update over the on-policy rollout batch. Reuses the
    module-level hyperparameters/GAE, never reinventing them."""
    batch = rollouts_to_batch(rollouts, device)
    player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
    actions, old_log_probs = batch["actions"], batch["old_log_probs"]
    old_values, returns = batch["values"], batch["returns"]
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n_samples = int(player.shape[0])
    optimizer_steps = 0
    kl_sum = 0.0
    clip_sum = 0.0
    entropy_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    total_loss_sum = 0.0
    seen = 0

    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, PPO_MINIBATCH):
            mb = perm[start : start + PPO_MINIBATCH]
            dist, value = net(player[mb], bullets[mb], pad[mb])
            log_prob = dist.log_prob(actions[mb])
            ratio = torch.exp(log_prob - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), PPO_MAX_GRAD_NORM)
            opt.step()
            optimizer_steps += 1

            with torch.no_grad():
                approx_kl = float((old_log_probs[mb] - log_prob).mean().item())
                clip_fraction = float((torch.abs(ratio - 1.0) > PPO_CLIP).float().mean().item())
            bs = int(mb.shape[0])
            kl_sum += approx_kl * bs
            clip_sum += clip_fraction * bs
            entropy_sum += float(entropy.item()) * bs
            policy_loss_sum += float(policy_loss.item()) * bs
            value_loss_sum += float(value_loss.item()) * bs
            total_loss_sum += float(loss.item()) * bs
            seen += bs

    with torch.no_grad():
        ret_var = torch.var(returns)
        explained_variance = float(1.0 - torch.var(returns - old_values) / (ret_var + 1e-8))

    denom = max(seen, 1)
    return {
        "approx_kl": kl_sum / denom,
        "clip_fraction": clip_sum / denom,
        "explained_variance": explained_variance,
        "entropy": entropy_sum / denom,
        "policy_loss": policy_loss_sum / denom,
        "value_loss": value_loss_sum / denom,
        "total_loss": total_loss_sum / denom,
        "optimizer_steps": optimizer_steps,
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------- #
# deterministic evaluation + censoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_deterministic(
    net: PlayerRankedTopK, device: torch.device, seeds: list[int], max_steps: int
) -> list[dict[str, Any]]:
    """Deterministic argmax evaluation on the given seeds; reproducible."""
    was_training = net.training
    net.eval()
    results: list[dict[str, Any]] = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        elapsed = env.elapsed
        done = False
        for _ in range(max_steps):
            player, bullets, pad = encode_obs(env)
            p_t = torch.tensor(player, dtype=torch.float32, device=device).unsqueeze(0)
            b_t = torch.tensor(bullets, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.tensor(pad, dtype=torch.bool, device=device).unsqueeze(0)
            dist, _value = net(p_t, b_t, m_t)
            action = int(dist.logits.argmax(-1).item())
            _obs, _r, done, info = env.step(action)
            elapsed = float(info.get("elapsed", env.elapsed))
            if done:
                break
        results.append({"seed": seed, "elapsed": elapsed, "censored": not done})
    if was_training:
        net.train()
    return results


def summarize_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/median/pstdev/min/max plus explicit per-seed censoring."""
    elapsed = [float(r["elapsed"]) for r in results]
    n = len(results)
    censored_count = sum(1 for r in results if r["censored"])
    return {
        "mean": statistics.mean(elapsed),
        "median": statistics.median(elapsed),
        "pstdev": statistics.pstdev(elapsed),
        "min": min(elapsed),
        "max": max(elapsed),
        "n": n,
        "censored_count": censored_count,
        "censor_rate": censored_count / n if n else 0.0,
        "per_seed": {str(r["seed"]): float(r["elapsed"]) for r in results},
        "per_seed_censored": {str(r["seed"]): bool(r["censored"]) for r in results},
    }


# --------------------------------------------------------------------------- #
# teacher diagnostics (held frames only -- NEVER PPO rollout data)
# --------------------------------------------------------------------------- #
def _bucket_mean(values: list[float], agree: list[float], edges: list[tuple[float, float]], labels: list[str]) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = {label: [] for label in labels}
    for value, a in zip(values, agree):
        for (lo, hi), label in zip(edges, labels):
            if lo <= value < hi:
                buckets[label].append(a)
                break
    return {label: (sum(vals) / len(vals) if vals else None) for label, vals in buckets.items()}


@torch.no_grad()
def teacher_diagnostics(net: PlayerRankedTopK, tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    """Compare the student policy against held teacher logits ONLY -- these
    frames never enter a PPO batch/update."""
    was_training = net.training
    net.eval()
    player = tensors["player"].to(device)
    bullets = tensors["bullets"].to(device)
    pad = tensors["pad"].to(device)
    teacher_logits = tensors["teacher_logits"].to(device)
    elapsed = tensors["elapsed"].to(device)

    dist, _value = net(player, bullets, pad)
    student_logits = dist.logits

    student_argmax = student_logits.argmax(-1)
    teacher_argmax = teacher_logits.argmax(-1)
    agree_t = (student_argmax == teacher_argmax).float()
    agreement = float(agree_t.mean().item())

    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    kl = (teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs)).sum(-1)
    teacher_to_student_kl = float(kl.mean().item())

    n_live = (~pad).sum(dim=-1).float()

    elapsed_edges = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, float("inf"))]
    elapsed_labels = ["0-15", "15-30", "30-45", "45-60", "60+"]
    bullet_edges = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, float("inf"))]
    bullet_labels = ["0-2", "3-5", "6-9", "10+"]

    agree_list = agree_t.cpu().tolist()
    by_elapsed_bucket = _bucket_mean(elapsed.cpu().tolist(), agree_list, elapsed_edges, elapsed_labels)
    by_bullet_bucket = _bucket_mean(n_live.cpu().tolist(), agree_list, bullet_edges, bullet_labels)

    if was_training:
        net.train()

    return {
        "agreement": agreement,
        "teacher_to_student_kl": teacher_to_student_kl,
        "by_elapsed_bucket": by_elapsed_bucket,
        "by_bullet_bucket": by_bullet_bucket,
    }


# --------------------------------------------------------------------------- #
# frozen arm: zero optimizer steps, bit-identical weights, never mutated
# --------------------------------------------------------------------------- #
def run_frozen_arm(
    net: PlayerRankedTopK,
    device: torch.device,
    snapshots: list[int],
    eval_seeds: list[int],
    eval_max_steps: int,
) -> dict[str, Any]:
    """Evaluate ``net`` once (it is never touched) and report the same
    evaluation at every requested snapshot update; zero optimizer steps."""
    before = state_dict_sha256(net.state_dict())
    results = evaluate_deterministic(net, device, eval_seeds, eval_max_steps)
    summary = summarize_evaluation(results)
    snapshot_records = [{"update": update, "evaluation": summary} for update in snapshots]
    after = state_dict_sha256(net.state_dict())
    return {
        "optimizer_steps": 0,
        "state_dict_sha256_initial": before,
        "state_dict_sha256_final": after,
        "state_dict_sha256_by_update": {str(update): before for update in snapshots},
        "state_dict_bit_identical": before == after,
        "metrics_bit_identical": True,
        "snapshots": snapshot_records,
    }


# --------------------------------------------------------------------------- #
# pre-registered gate math (retention, never promotion)
# --------------------------------------------------------------------------- #
def evaluate_initial_gate(
    summary: dict[str, Any],
    *,
    mean_min: float = INITIAL_GATE_MEAN_MIN,
    median_min: float = INITIAL_GATE_MEDIAN_MIN,
) -> dict[str, Any]:
    """Initial gate: mean >= mean_min AND median >= median_min (inclusive)."""
    mean_pass = float(summary["mean"]) >= mean_min
    median_pass = float(summary["median"]) >= median_min
    failing = []
    if not mean_pass:
        failing.append("mean")
    if not median_pass:
        failing.append("median")
    gate_pass = mean_pass and median_pass
    reason = "" if gate_pass else f"initial gate failed: {', '.join(failing)} below minimum"
    return {
        "gate_pass": gate_pass,
        "mean_pass": mean_pass,
        "median_pass": median_pass,
        "mean_min": mean_min,
        "median_min": median_min,
        "reason": reason,
    }


class MissingHeldAgreementError(ValueError):
    """Raised when the pre-PPO reference lacks a usable agreement value."""


def build_retention_reference(initial_snapshot: dict[str, Any]) -> dict[str, float]:
    agreement = initial_snapshot.get("teacher_diagnostics", {}).get("agreement")
    if agreement is None:
        raise MissingHeldAgreementError("initial snapshot is missing held teacher agreement")
    evaluation = initial_snapshot["evaluation"]
    return {
        "mean": float(evaluation["mean"]),
        "median": float(evaluation["median"]),
        "held_agreement": float(agreement),
    }


def evaluate_retention(reference: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Final retention passes only if the FINAL PPO mean and median are each
    >= 80% of the reference AND held teacher agreement declined by no more
    than 0.05 absolute. The first snapshot where either threshold breaks is
    always reported. This is a retention gate only -- there is no promotion
    of any checkpoint anywhere."""
    ordered = sorted(snapshots, key=lambda s: s["update"])
    mean_threshold = RETENTION_FRACTION * float(reference["mean"])
    median_threshold = RETENTION_FRACTION * float(reference["median"])
    agreement_floor = float(reference["held_agreement"]) - MAX_HELD_AGREEMENT_DROP

    first_breaking_snapshot = None
    missing_agreement_updates: list[int] = []
    for s in ordered:
        ev = s["evaluation"]
        agreement = s.get("teacher_diagnostics", {}).get("agreement")
        if agreement is None:
            missing_agreement_updates.append(s["update"])
        breaks = (
            ev["mean"] < mean_threshold
            or ev["median"] < median_threshold
            or agreement is None
            or agreement < agreement_floor
        )
        if breaks and first_breaking_snapshot is None:
            first_breaking_snapshot = s["update"]

    final = ordered[-1]
    final_mean_pass = final["evaluation"]["mean"] >= mean_threshold
    final_median_pass = final["evaluation"]["median"] >= median_threshold
    final_agreement = final.get("teacher_diagnostics", {}).get("agreement")
    if final_agreement is None:
        final_agreement_drop = None
    else:
        final_agreement_drop = float(reference["held_agreement"]) - float(final_agreement)
    agreement_pass = (
        final_agreement is not None
        and not missing_agreement_updates
        and final_agreement >= agreement_floor
    )

    retention_pass = final_mean_pass and final_median_pass and agreement_pass
    failing = []
    if not final_mean_pass:
        failing.append("mean below threshold")
    if not final_median_pass:
        failing.append("median below threshold")
    if missing_agreement_updates:
        updates = ", ".join(str(update) for update in missing_agreement_updates)
        failing.append(f"missing held teacher agreement at update(s): {updates}")
    elif not agreement_pass:
        failing.append("agreement below threshold")
    reason = "" if retention_pass else f"retention failed: {', '.join(failing)}"

    return {
        "retention_pass": retention_pass,
        "mean_threshold": mean_threshold,
        "median_threshold": median_threshold,
        "agreement_floor": agreement_floor,
        "first_breaking_snapshot": first_breaking_snapshot,
        "final_mean_pass": final_mean_pass,
        "final_median_pass": final_median_pass,
        "agreement_pass": agreement_pass,
        "final_agreement_drop": final_agreement_drop,
        "reason": reason,
        "promotion": False,
    }


# --------------------------------------------------------------------------- #
# canonical teacher dataset (collected exactly once locally; held frames only
# ever feed teacher-agreement diagnostics -- never a PPO batch)
# --------------------------------------------------------------------------- #
def load_teacher(path: Path | str, device: torch.device) -> PlayerV1:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = PlayerV1(hidden=int(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.to(device)
    net.eval()
    return net


def collect_canonical_dataset(
    teacher: PlayerV1,
    device: torch.device,
    *,
    episodes: int = COLLECT_EPISODES,
    seed_start: int = COLLECT_SEED_START,
    max_steps: int = COLLECT_MAX_STEPS,
    frames_cap: int = FRAMES_PER_EPISODE_CAP,
    held_frac: float = HELD_OUT_FRAC,
    teacher_path: Path | str | None = None,
) -> dict[str, Any]:
    """Recollect the canonical V1 teacher dataset locally (once) using the
    EXACT Phase2 ``collect_dataset``/``collect_episode`` contract -- full
    episodes up to ``max_steps``, then even-stride cap subsampling over the
    WHOLE episode (never a truncation to the first ``frames_cap`` frames),
    split by episode into train/held with the shared ``DATA_SEED``. This is
    never reimplemented here; it delegates to the Phase2 helpers verbatim.
    Only held-episode frames are materialized into tensors, since train
    frames are never used by this harness (no teacher data ever enters a
    PPO batch). The identity hash is Phase2's own
    ``phase2_ranked_multiseed.dataset_identity`` over the actual Frame
    content (player/bullets/pad/teacher_logits/elapsed/episode) -- never a
    hash of collection parameters alone."""
    rng = random.Random(DATA_SEED)
    train_frames, held_frames = collect_dataset(
        teacher,
        device,
        n_episodes=episodes,
        seed_start=seed_start,
        max_steps=max_steps,
        frames_cap=frames_cap,
        held_out_frac=held_frac,
        rng=rng,
    )
    identity = dataset_identity(train_frames, held_frames)
    held_tensors = frames_to_tensors(held_frames, device)

    result: dict[str, Any] = {
        "hash": identity["hash"],
        "n_episodes": episodes,
        "n_held_episodes": len({f.episode for f in held_frames}),
        "n_train_episodes": len({f.episode for f in train_frames}),
        "n_held_frames": identity["n_held"],
        "n_train_frames": identity["n_train"],
        "held_tensors": held_tensors,
    }
    if teacher_path is not None:
        result["teacher_file_sha256"] = file_sha256(teacher_path)
    return result


# --------------------------------------------------------------------------- #
# PPO arm: on-policy training with the pre-registered snapshot schedule
# --------------------------------------------------------------------------- #
def run_ppo_arm(
    init_net: PlayerRankedTopK,
    device: torch.device,
    args: argparse.Namespace,
    snapshot_updates: list[int],
    held_tensors: dict[str, torch.Tensor] | None,
    out_dir: Path,
    *,
    parent_state_dict_sha256: str,
    parent_file_sha256: str,
    dataset_hash: str,
    ppo_knobs: dict[str, Any],
) -> dict[str, Any]:
    """Train the paired PPO arm from the exact same loaded initial state as
    the frozen arm (a fresh clone, so the frozen arm/init checkpoint are
    never mutated).

    Every snapshot/final checkpoint written here is an experimental,
    non-production artifact: it is packed with ``experimental=True`` /
    ``production_compatible=False`` and an ``extra`` provenance block
    threaded straight from ``run_experiment`` (the parent init checkpoint's
    state-dict/file hashes, the canonical dataset hash, and the actual
    per-run PPO knobs -- never reimplemented or reinvented locally), plus
    the eval summary at that update and the rollout/eval seed windows used
    to produce it.
    """
    net = PlayerRankedTopK(top_k=init_net.top_k, hidden=init_net.hidden).to(device)
    net.load_state_dict(init_net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR, eps=1e-8)
    torch.manual_seed(PPO_TORCH_SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "ppo_updates.jsonl"
    snapshots: list[dict[str, Any]] = []
    total_frames = 0
    optimizer_steps = 0
    # Seed window for the rollout that produced (or is about to produce, for
    # the update-0 snapshot) the current net weights.
    rollout_seed_window = rollout_seed_schedule(0, episodes_per_update=args.episodes_per_update)

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
        }

    def _snapshot(update: int) -> None:
        eval_results = evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
        eval_summary = summarize_evaluation(eval_results)
        if held_tensors is not None:
            diag = teacher_diagnostics(net, held_tensors, device)
        else:
            diag = {"agreement": None}
        ckpt_path = out_dir / f"ppo_update_{update}.pt"
        save_player_checkpoint(
            net,
            ckpt_path,
            source_tool="phase3_ranked_ppo_retention",
            experimental=True,
            production_compatible=False,
            extra=_pack_extra(update, eval_summary),
        )
        snapshots.append(
            {
                "update": update,
                "evaluation": eval_summary,
                "teacher_diagnostics": diag,
                "checkpoint": str(ckpt_path),
                "state_dict_sha256": state_dict_sha256(net.state_dict()),
            }
        )

    if 0 in snapshot_updates:
        _snapshot(0)

    with jsonl_path.open("w") as jsonl_f:
        for update in range(1, args.updates + 1):
            seeds = rollout_seed_schedule(update - 1, episodes_per_update=args.episodes_per_update)
            rollout_seed_window = seeds
            rollouts = [collect_rollout(net, device, seed, max_frames=args.max_frames) for seed in seeds]
            metrics = ppo_update(net, opt, rollouts, device)
            optimizer_steps += int(metrics["optimizer_steps"])
            total_frames += sum(len(r.actions) for r in rollouts)
            row = dict(metrics)
            row["update"] = update
            row["scripted_survival_mean"] = statistics.mean([r.elapsed for r in rollouts])
            row["rollout_censoring"] = rollout_censor_stats(rollouts)
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()
            if update in snapshot_updates:
                _snapshot(update)

    final_eval_summary = summarize_evaluation(evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps))
    final_path = out_dir / "ppo_final.pt"
    save_player_checkpoint(
        net,
        final_path,
        source_tool="phase3_ranked_ppo_retention",
        experimental=True,
        production_compatible=False,
        extra=_pack_extra(args.updates, final_eval_summary),
    )

    return {
        "optimizer_steps": optimizer_steps,
        "total_episodes": args.updates * args.episodes_per_update,
        "total_frames": total_frames,
        "snapshots": snapshots,
        "final_checkpoint": str(final_path),
    }


# --------------------------------------------------------------------------- #
# CLI / experiment orchestration
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase3_ranked_ppo_retention"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in the update budget / seed windows / gate thresholds. Non-quick
    runs use the pre-registered production defaults (200 updates etc); quick
    mode shrinks every knob so a control can be smoke-tested fast, and
    relaxes the initial gate (never the retention math itself)."""
    if getattr(args, "quick", False):
        args.updates = 2
        args.episodes_per_update = 2
        args.max_frames = 30
        args.eval_seeds = eval_seed_list()[:3]
        args.eval_max_steps = 40
        args.data_episodes = 6
        args.data_max_steps = 40
        args.data_frames_cap = 40
        args.initial_gate_mean_min = 0.0
        args.initial_gate_median_min = 0.0
    else:
        args.updates = PPO_UPDATES
        args.episodes_per_update = EPISODES_PER_UPDATE
        args.max_frames = PPO_MAX_FRAMES
        args.eval_seeds = eval_seed_list()
        args.eval_max_steps = EVAL_MAX_STEPS
        args.data_episodes = COLLECT_EPISODES
        args.data_max_steps = COLLECT_MAX_STEPS
        args.data_frames_cap = FRAMES_PER_EPISODE_CAP
        args.initial_gate_mean_min = INITIAL_GATE_MEAN_MIN
        args.initial_gate_median_min = INITIAL_GATE_MEDIAN_MIN
    return args


def _git_dirty(repo_root: Path | None = None) -> bool | None:
    root = repo_root if repo_root is not None else _REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def run_experiment(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Strict, fail-closed orchestration of the paired frozen/ppo control.
    No env rollouts and no PPO ever run before the pre-registered initial
    gate passes; no checkpoint promotion happens anywhere."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_net, init_meta = load_initial_checkpoint(args.init_checkpoint, device)
    init_prov = checkpoint_provenance(args.init_checkpoint, init_meta)
    teacher_prov = {"file_sha256": file_sha256(args.teacher)}
    provenance = {
        "init_checkpoint": init_prov,
        "teacher": teacher_prov,
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

    gate_results = evaluate_deterministic(init_net, device, args.eval_seeds, args.eval_max_steps)
    gate_summary = summarize_evaluation(gate_results)
    initial_gate = evaluate_initial_gate(
        gate_summary, mean_min=args.initial_gate_mean_min, median_min=args.initial_gate_median_min
    )

    report: dict[str, Any] = {
        "status": None,
        "control_ran": False,
        "initial_gate": initial_gate,
        "provenance": provenance,
        "knobs": knobs,
        "dataset": None,
        "frozen_arm": None,
        "ppo_arm": None,
        "retention": None,
    }

    if not initial_gate["gate_pass"]:
        report["status"] = "failed_closed"
        _write_report(out_dir, report)
        return report

    teacher = load_teacher(args.teacher, device)
    dataset = collect_canonical_dataset(
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

    frozen_arm = run_frozen_arm(init_net, device, snapshot_updates, args.eval_seeds, args.eval_max_steps)
    report["frozen_arm"] = frozen_arm

    ppo_arm = run_ppo_arm(
        init_net,
        device,
        args,
        snapshot_updates,
        dataset["held_tensors"],
        out_dir,
        parent_state_dict_sha256=init_prov["state_dict_sha256"],
        parent_file_sha256=init_prov["file_sha256"],
        dataset_hash=dataset["hash"],
        ppo_knobs=knobs,
    )
    report["ppo_arm"] = ppo_arm

    initial_snapshot = ppo_arm["snapshots"][0]
    reference = build_retention_reference(initial_snapshot)
    retention = evaluate_retention(reference, ppo_arm["snapshots"])
    report["retention"] = retention

    report["status"] = "completed"
    report["control_ran"] = True
    _write_report(out_dir, report)
    return report


def main() -> None:
    args = apply_mode_defaults(build_parser().parse_args())
    device = torch.device(args.device)
    report = run_experiment(args, device)
    print(json.dumps({"status": report["status"], "control_ran": report["control_ran"]}))


if __name__ == "__main__":
    main()
