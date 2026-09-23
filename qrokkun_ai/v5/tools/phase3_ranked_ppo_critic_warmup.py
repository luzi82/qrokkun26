#!/usr/bin/env python3
"""Phase 3 critic-warmup control for ``player_ranked_topk``.

Run from the repository root with::

    PYTHONPATH=. python -m qrokkun_ai.v5.tools.phase3_ranked_ppo_critic_warmup \
        --arm warmup --init-checkpoint PATH --teacher PATH --run-dir RUN --device cuda

The only treatment is value-head-only warmup on a frozen BC ``body``/``policy``
before unregularized scripted PPO. Snapshots are experimental; promotion is
always false. New runs use this harness's own ``WARMUP_RUN_SCHEMA_VERSION``,
which is validated explicitly and moves only when THIS contract changes.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_ai.v5.agents.obs_v4 import encode_obs
from qrokkun_ai.v5.agents.player_checkpoints import (
    current_git_commit,
    file_sha256,
    save_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_ai.v5.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_env.env import Qrokkun26Env

from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention as ret_mod
from qrokkun_ai.v5.tools.phase3_ranked_ppo_retention import (
    COLLECT_EPISODES,
    COLLECT_MAX_STEPS,
    COLLECT_SEED_START,
    EPISODES_PER_UPDATE,
    EVAL_MAX_STEPS,
    EVAL_SEED_COUNT,
    EVAL_SEED_START,
    FRAMES_PER_EPISODE_CAP,
    HELD_OUT_FRAC,
    INITIAL_GATE_MEAN_MIN,
    INITIAL_GATE_MEDIAN_MIN,
    PERIODIC_MODEL_CHECKPOINT_INTERVAL,
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
    PPO_VALUE_COEF,
    collect_canonical_dataset,
    collect_rollout,
    evaluate_deterministic,
    evaluate_initial_gate,
    eval_seed_list,
    load_initial_checkpoint,
    load_teacher,
    ppo_hyperparameters,
    ppo_update,
    rollout_censor_stats,
    rollout_seed_schedule,
    shaped_reward,
    summarize_evaluation,
    teacher_diagnostics,
    _REPO_ROOT,
    _git_dirty,
    _write_report,
    checkpoint_provenance,
)

PPO_UPDATES = 10
SNAPSHOT_UPDATES: tuple[int, ...] = (0, 10)
WARMUP_UPDATES = 10
WARMUP_SEED_START = 40000
HELD_VALUE_FIT_SEED_START = 41000
HELD_VALUE_FIT_COUNT = 30
WARMUP_GENERATOR_SEED = 4242
HELD_FIT_GENERATOR_SEED = 4343
TOOL_NAME = "phase3_ranked_ppo_critic_warmup"

# This harness keeps its own run-contract version rather than following the
# retention arms'.  Its contract records the teacher by file hash only, so the
# retention bump to 5 -- which binds the teacher's state-dict hash and
# architecture -- describes nothing warmup writes.  Following that bump would
# have refused every schema-4 warmup run.json already on disk while changing
# nothing about what warmup actually records.  A future warmup contract change
# bumps THIS constant, and the shared validators are told which version to
# require rather than assuming the retention one.
WARMUP_RUN_SCHEMA_VERSION = 4


def require_matching_warmup_run_contract(
    run_dir: Path, contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate an existing warmup ``run.json`` against the warmup schema."""
    return ret_mod.require_matching_current_run_contract(
        run_dir, contract, expected_schema_version=WARMUP_RUN_SCHEMA_VERSION,
    )


def create_or_validate_warmup_run_contract(
    run_dir: Path, contract: dict[str, Any], *, resume: bool,
) -> dict[str, Any] | None:
    """Create or validate the warmup run contract at the warmup schema version."""
    return ret_mod.create_or_validate_run_contract(
        run_dir, contract, resume=resume,
        expected_schema_version=WARMUP_RUN_SCHEMA_VERSION,
    )


def snapshot_schedule(final_update: int) -> list[int]:
    updates = {u for u in SNAPSHOT_UPDATES if u <= final_update}
    updates.add(final_update)
    return sorted(updates)


def warmup_seed_schedule(update: int, episodes_per_update: int = EPISODES_PER_UPDATE) -> list[int]:
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("warmup update must be a 1-based positive int")
    start = WARMUP_SEED_START + (update - 1) * episodes_per_update
    return list(range(start, start + episodes_per_update))


def held_value_fit_seeds() -> list[int]:
    return list(range(HELD_VALUE_FIT_SEED_START, HELD_VALUE_FIT_SEED_START + HELD_VALUE_FIT_COUNT))


def freeze_actor(net: PlayerRankedTopK) -> None:
    for module in (net.body, net.policy):
        for param in module.parameters():
            param.requires_grad_(False)


def unfreeze_actor(net: PlayerRankedTopK) -> None:
    for param in net.parameters():
        param.requires_grad_(True)


def value_optimizer(net: PlayerRankedTopK) -> torch.optim.Adam:
    freeze_actor(net)
    return torch.optim.Adam(net.value.parameters(), lr=PPO_LR, eps=1e-8)


def actor_state_snapshot(net: PlayerRankedTopK) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in net.state_dict().items()
        if name.startswith("body.") or name.startswith("policy.")
    }


def assert_actor_unchanged(net: PlayerRankedTopK, before: dict[str, torch.Tensor]) -> None:
    after = actor_state_snapshot(net)
    if after.keys() != before.keys():
        raise RuntimeError("actor parameter set changed during critic warmup")
    for name, tensor in before.items():
        if not torch.equal(tensor, after[name]):
            raise RuntimeError(f"actor parameter {name} changed during critic warmup")


@torch.no_grad()
def held_policy_outputs(
    net: PlayerRankedTopK, tensors: dict[str, torch.Tensor], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = net.training
    net.eval()
    dist, _value = net(
        tensors["player"].to(device),
        tensors["bullets"].to(device),
        tensors["pad"].to(device),
    )
    logits = dist.logits.detach().cpu()
    argmax = logits.argmax(-1)
    if was_training:
        net.train()
    return logits, argmax


def monte_carlo_returns(rewards: list[float], gamma: float) -> list[float]:
    remaining = 0.0
    out: list[float] = []
    for reward in reversed(rewards):
        remaining = float(reward) + gamma * remaining
        out.append(remaining)
    out.reverse()
    return out


def _usable_warmup_rollouts(rollouts: list[ret_mod.Rollout]) -> tuple[list[ret_mod.Rollout], int]:
    usable = [rollout for rollout in rollouts if not rollout.censored and rollout.player]
    dropped = len(rollouts) - len(usable)
    return usable, dropped


def warmup_value_update(
    net: PlayerRankedTopK,
    opt: torch.optim.Optimizer,
    rollouts: list[ret_mod.Rollout],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    usable, dropped = _usable_warmup_rollouts(rollouts)
    if not usable:
        return {"optimizer_steps": 0, "n_samples": 0, "n_dropped_censored": dropped, "value_loss": None}
    player_l: list[np.ndarray] = []
    bullets_l: list[np.ndarray] = []
    pad_l: list[np.ndarray] = []
    returns_l: list[float] = []
    for rollout in usable:
        player_l.extend(rollout.player)
        bullets_l.extend(rollout.bullets)
        pad_l.extend(rollout.pad)
        returns_l.extend(monte_carlo_returns(rollout.rewards, PPO_GAMMA))
    player = torch.tensor(np.stack(player_l), dtype=torch.float32, device=device)
    bullets = torch.tensor(np.stack(bullets_l), dtype=torch.float32, device=device)
    pad = torch.tensor(np.stack(pad_l), dtype=torch.bool, device=device)
    returns = torch.tensor(returns_l, dtype=torch.float32, device=device)
    n_samples = int(player.shape[0])
    optimizer_steps = 0
    loss_sum = 0.0
    seen = 0
    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(n_samples, device=device, generator=generator)
        for start in range(0, n_samples, PPO_MINIBATCH):
            mb = perm[start : start + PPO_MINIBATCH]
            _dist, value = net(player[mb], bullets[mb], pad[mb])
            loss = F.mse_loss(value, returns[mb])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.value.parameters(), PPO_MAX_GRAD_NORM)
            opt.step()
            optimizer_steps += 1
            bs = int(mb.shape[0])
            loss_sum += float(loss.item()) * bs
            seen += bs
    return {
        "optimizer_steps": optimizer_steps,
        "n_samples": n_samples,
        "n_dropped_censored": dropped,
        "value_loss": loss_sum / max(seen, 1),
    }


def seed_ppo_phase(seed: int = PPO_TORCH_SEED) -> None:
    effective = int(seed)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def isolated_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _sample_action(dist: torch.distributions.Categorical, generator: torch.Generator) -> torch.Tensor:
    probs = dist.probs
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, 1, replacement=True, generator=generator)
    return idx.reshape(probs.shape[:-1])


def collect_rollout_with_generator(
    net: PlayerRankedTopK,
    device: torch.device,
    seed: int,
    max_frames: int,
    generator: torch.Generator,
) -> ret_mod.Rollout:
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
        action = _sample_action(dist, generator)
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
    return ret_mod.Rollout(
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


def collect_held_value_fit_pack(
    net: PlayerRankedTopK,
    device: torch.device,
    max_frames: int,
    generator: torch.Generator,
    *,
    n_seeds: int = HELD_VALUE_FIT_COUNT,
) -> dict[str, Any]:
    player_l: list[np.ndarray] = []
    bullets_l: list[np.ndarray] = []
    pad_l: list[np.ndarray] = []
    returns_l: list[float] = []
    used: list[int] = []
    dropped = 0
    for seed in held_value_fit_seeds()[: int(n_seeds)]:
        rollout = collect_rollout_with_generator(net, device, seed, max_frames, generator)
        if rollout.censored or not rollout.player:
            dropped += 1
            continue
        player_l.extend(rollout.player)
        bullets_l.extend(rollout.bullets)
        pad_l.extend(rollout.pad)
        returns_l.extend(monte_carlo_returns(rollout.rewards, PPO_GAMMA))
        used.append(seed)
    if not player_l:
        raise ret_mod.RunStateError("held value-fit pack is empty after dropping censored episodes")
    return {
        "player": np.stack(player_l),
        "bullets": np.stack(bullets_l),
        "pad": np.stack(pad_l),
        "returns": np.asarray(returns_l, dtype=np.float32),
        "seeds": used,
        "n_dropped_censored": dropped,
    }


@torch.no_grad()
def held_value_fit_ev(
    net: PlayerRankedTopK, pack: dict[str, Any], device: torch.device,
) -> float:
    was_training = net.training
    net.eval()
    player = torch.as_tensor(pack["player"], dtype=torch.float32, device=device)
    bullets = torch.as_tensor(pack["bullets"], dtype=torch.float32, device=device)
    pad = torch.as_tensor(pack["pad"], dtype=torch.bool, device=device)
    returns = torch.as_tensor(pack["returns"], dtype=torch.float32, device=device)
    _dist, values = net(player, bullets, pad)
    ret_var = torch.var(returns)
    ev = float(1.0 - torch.var(returns - values) / (ret_var + 1e-8))
    if was_training:
        net.train()
    return ev


def _snapshot_agreement(report: dict[str, Any], update: int) -> float | None:
    for item in report.get("snapshots") or []:
        if item.get("update") == update:
            agreement = (item.get("teacher_diagnostics") or {}).get("agreement")
            return None if agreement is None else float(agreement)
    return None


def compare_arms(warmup_report: dict[str, Any], direct_report: dict[str, Any]) -> dict[str, Any]:
    warmup_ev = float(warmup_report["ppo_entry_held_ev"])
    direct_ev = float(direct_report["ppo_entry_held_ev"])
    mechanism_pass = warmup_ev > direct_ev and warmup_ev > 0.0
    warmup_ag = _snapshot_agreement(warmup_report, PPO_UPDATES)
    direct_ag = _snapshot_agreement(direct_report, PPO_UPDATES)
    failing: list[str] = []
    if not mechanism_pass:
        failing.append("held value-fit mechanism")
    if warmup_ag is None or direct_ag is None:
        failing.append("missing held teacher agreement")
        primary_pass = False
    else:
        primary_pass = mechanism_pass and warmup_ag > direct_ag
        if mechanism_pass and not primary_pass:
            failing.append("u10 held agreement")
    return {
        "mechanism_pass": mechanism_pass,
        "primary_pass": primary_pass,
        "promotion": False,
        "warmup_held_ev": warmup_ev,
        "direct_held_ev": direct_ev,
        "warmup_u10_agreement": warmup_ag,
        "direct_u10_agreement": direct_ag,
        "reason": "" if primary_pass else f"critic-warmup comparison failed: {', '.join(failing)}",
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=("direct", "warmup"))
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase3_ranked_ppo_critic_warmup"))
    ap.add_argument("--run-dir", type=Path, help="resumable run directory (defaults to --out-dir)")
    ap.add_argument("--end-time", type=ret_mod.parse_end_time, help="HKT deadline: YYYYMMDD-HHMM")
    ap.add_argument("--max-updates", type=int, help="maximum completed PPO updates for this run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from-update", type=int, metavar="N")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "quick", False):
        args.updates = 2
        args.warmup_updates = 1
        args.episodes_per_update = 2
        args.max_frames = 30
        args.eval_seeds = eval_seed_list()[:3]
        args.eval_max_steps = 40
        args.data_episodes = 6
        args.data_max_steps = 40
        args.data_frames_cap = 40
        args.held_value_fit_count = 3
        args.initial_gate_mean_min = 0.0
        args.initial_gate_median_min = 0.0
    else:
        args.updates = PPO_UPDATES
        args.warmup_updates = WARMUP_UPDATES
        args.episodes_per_update = EPISODES_PER_UPDATE
        args.max_frames = PPO_MAX_FRAMES
        args.eval_seeds = eval_seed_list()
        args.eval_max_steps = EVAL_MAX_STEPS
        args.data_episodes = COLLECT_EPISODES
        args.data_max_steps = COLLECT_MAX_STEPS
        args.data_frames_cap = FRAMES_PER_EPISODE_CAP
        args.held_value_fit_count = HELD_VALUE_FIT_COUNT
        args.initial_gate_mean_min = INITIAL_GATE_MEAN_MIN
        args.initial_gate_median_min = INITIAL_GATE_MEDIAN_MIN
    if getattr(args, "max_updates", None) is not None and args.max_updates < 0:
        raise ValueError("--max-updates must be non-negative")
    selected = getattr(args, "resume_from_update", None)
    if selected is not None:
        args.effective_max_updates, args.effective_end_time = ret_mod.resolve_resume_from_stop_budget(
            args, selected,
        )
    else:
        args.effective_max_updates, args.effective_end_time = ret_mod.resolve_stop_budget(
            getattr(args, "max_updates", None), getattr(args, "end_time", None),
            default_max_updates=args.updates,
        )
    if getattr(args, "run_dir", None) is not None:
        args.run_dir = Path(args.run_dir)
    elif hasattr(args, "out_dir"):
        args.run_dir = Path(args.out_dir)
    else:
        args.run_dir = None
    seed = getattr(args, "seed", None)
    args.effective_seed = PPO_TORCH_SEED if seed is None else ret_mod.configure_seed(seed)
    return args


def _pack_extra(
    update: int,
    eval_summary: dict[str, Any],
    *,
    parent_state_dict_sha256: str,
    parent_file_sha256: str,
    dataset_hash: str,
    knobs: dict[str, Any],
    rollout_seed_window: list[int],
    eval_seeds: list[int],
    arm: str,
) -> dict[str, Any]:
    return {
        "update": update,
        "arm": arm,
        "parent_state_dict_sha256": parent_state_dict_sha256,
        "parent_file_sha256": parent_file_sha256,
        "dataset_hash": dataset_hash,
        "eval_summary": eval_summary,
        "ppo_knobs": knobs,
        "rollout_seed_window": rollout_seed_window,
        "eval_seed_window": eval_seeds,
    }


def run_warmup_phase(
    net: PlayerRankedTopK,
    device: torch.device,
    args: argparse.Namespace,
    held_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    before_actor = actor_state_snapshot(net)
    before_logits, before_argmax = held_policy_outputs(net, held_tensors, device)
    opt = value_optimizer(net)
    generator = isolated_generator(WARMUP_GENERATOR_SEED, device)
    total_frames = 0
    optimizer_steps = 0
    dropped = 0
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for update in range(1, args.warmup_updates + 1):
        seeds = warmup_seed_schedule(update, episodes_per_update=args.episodes_per_update)
        rollouts = [
            collect_rollout_with_generator(net, device, seed, args.max_frames, generator)
            for seed in seeds
        ]
        metrics = warmup_value_update(net, opt, rollouts, device, generator)
        optimizer_steps += int(metrics["optimizer_steps"])
        total_frames += sum(len(rollout.actions) for rollout in rollouts)
        dropped += int(metrics["n_dropped_censored"])
        rows.append({"update": update, "seeds": seeds, **metrics})
    assert_actor_unchanged(net, before_actor)
    after_logits, after_argmax = held_policy_outputs(net, held_tensors, device)
    if not torch.equal(before_logits, after_logits) or not torch.equal(before_argmax, after_argmax):
        raise RuntimeError("held policy logits changed during critic warmup")
    unfreeze_actor(net)
    return {
        "updates": args.warmup_updates,
        "episodes": args.warmup_updates * args.episodes_per_update,
        "total_frames": total_frames,
        "optimizer_steps": optimizer_steps,
        "n_dropped_censored": dropped,
        "wall_s": time.perf_counter() - wall_start,
        "rows": rows,
        "actor_unchanged": True,
        "held_policy_unchanged": True,
    }


def run_ppo_phase(
    net: PlayerRankedTopK,
    device: torch.device,
    args: argparse.Namespace,
    snapshot_updates: list[int],
    held_tensors: dict[str, torch.Tensor] | None,
    out_dir: Path,
    *,
    arm: str,
    parent_state_dict_sha256: str,
    parent_file_sha256: str,
    dataset_hash: str,
    knobs: dict[str, Any],
) -> dict[str, Any]:
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR, eps=1e-8)
    seed_ppo_phase(PPO_TORCH_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "ppo_updates.jsonl"
    snapshots: list[dict[str, Any]] = []
    total_frames = 0
    optimizer_steps = 0
    rollout_seed_window = rollout_seed_schedule(0, episodes_per_update=args.episodes_per_update)

    def _snapshot(update: int) -> None:
        eval_results = evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
        eval_summary = summarize_evaluation(eval_results)
        diag = teacher_diagnostics(net, held_tensors, device) if held_tensors is not None else {"agreement": None}
        ckpt_path = out_dir / f"ppo_update_{update}.pt"
        save_player_checkpoint(
            net,
            ckpt_path,
            source_tool=TOOL_NAME,
            experimental=True,
            production_compatible=False,
            extra=_pack_extra(
                update, eval_summary,
                parent_state_dict_sha256=parent_state_dict_sha256,
                parent_file_sha256=parent_file_sha256,
                dataset_hash=dataset_hash,
                knobs=knobs,
                rollout_seed_window=rollout_seed_window,
                eval_seeds=args.eval_seeds,
                arm=arm,
            ),
        )
        snapshots.append({
            "update": update,
            "evaluation": eval_summary,
            "teacher_diagnostics": diag,
            "checkpoint": str(ckpt_path),
            "state_dict_sha256": state_dict_sha256(net.state_dict()),
        })

    recovery_path = out_dir / "recovery.pt"
    completed = 0
    if getattr(args, "resume", False):
        recovery = ret_mod.load_recovery(recovery_path, device)
        try:
            net.load_state_dict(recovery["model"], strict=True)
            opt.load_state_dict(recovery["optimizer"])
            completed = recovery["completed_update"]
            total_frames = int(recovery["total_frames"])
            optimizer_steps = int(recovery["optimizer_steps"])
            snapshots = list(recovery.get("snapshots", []))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ret_mod.RunStateError("resume recovery state is malformed") from exc
        ret_mod.append_run_status(out_dir, {"event": "resumed", "completed_update": completed})
        ret_mod.truncate_progress_journal_to(jsonl_path, completed)
    else:
        if 0 in snapshot_updates:
            _snapshot(0)
        ret_mod.reconcile_progress_journal(jsonl_path, completed_update=completed)

    def _save_boundary(force: bool = False) -> None:
        state = {
            "format": 1, "arm": arm, "model": net.state_dict(),
            "optimizer": opt.state_dict(), "completed_update": completed,
            "total_frames": total_frames, "optimizer_steps": optimizer_steps,
            "snapshots": snapshots, "rng": ret_mod.capture_rng_state(),
        }
        if ret_mod.latest_recovery_due(completed, force=force):
            ret_mod.atomic_save_recovery(recovery_path, state)
        ret_mod.save_archived_recovery_if_due(out_dir, state)

    if not getattr(args, "resume", False):
        _save_boundary()
    ret_mod.unlink_latest_recovery(recovery_path)
    stop = ret_mod.StopRequest()
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_handlers:
        signal.signal(sig, stop.handler)
    stop_reason: str | None = None
    max_updates, end_time = ret_mod.effective_stop_budget(args)
    max_updates_explicit = getattr(args, "max_updates", None) is not None
    try:
        for update in range(completed + 1, max_updates + 1):
            requested_reason = ret_mod.boundary_stop_reason(
                completed=completed, configured_updates=args.updates,
                max_updates=max_updates, end_time=end_time, interrupted=stop.requested,
                max_updates_explicit=max_updates_explicit,
            )
            if requested_reason is not None:
                stop_reason = requested_reason
                break
            seeds = rollout_seed_schedule(update - 1, episodes_per_update=args.episodes_per_update)
            rollout_seed_window = seeds
            collect_start = time.perf_counter()
            rollouts = [collect_rollout(net, device, seed, max_frames=args.max_frames) for seed in seeds]
            collect_wall_s = time.perf_counter() - collect_start
            ppo_start = time.perf_counter()
            metrics = ppo_update(net, opt, rollouts, device)
            ppo_wall_s = time.perf_counter() - ppo_start
            optimizer_steps += int(metrics["optimizer_steps"])
            total_frames += sum(len(rollout.actions) for rollout in rollouts)
            row = dict(metrics)
            row["update"] = update
            row["scripted_survival_mean"] = statistics.mean([rollout.elapsed for rollout in rollouts])
            row["rollout_censoring"] = rollout_censor_stats(rollouts)
            row["collect_wall_s"] = collect_wall_s
            row["ppo_wall_s"] = ppo_wall_s
            row["total_wall_s"] = collect_wall_s + ppo_wall_s
            ret_mod.append_progress_row(jsonl_path, row)
            completed = update
            _save_boundary()
            ret_mod.append_run_status(out_dir, {
                "event": "update_complete", "update": update, "total_frames": total_frames,
            })
            if update in snapshot_updates and update <= PERIODIC_MODEL_CHECKPOINT_INTERVAL:
                _snapshot(update)
                _save_boundary()
            if stop.requested:
                stop_reason = "interrupted"
                ret_mod.unlink_latest_recovery(recovery_path)
                break
            requested_reason = ret_mod.boundary_stop_reason(
                completed=completed, configured_updates=args.updates,
                max_updates=max_updates, end_time=end_time, interrupted=False,
                max_updates_explicit=max_updates_explicit,
            )
            if requested_reason is not None:
                stop_reason = requested_reason
                break
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)

    if stop_reason is None:
        stop_reason = ret_mod.boundary_stop_reason(
            completed=completed, configured_updates=args.updates, max_updates=max_updates,
            end_time=end_time, max_updates_explicit=max_updates_explicit,
        ) or "max_updates"
    ret_mod.append_run_status(out_dir, {"event": stop_reason, "completed_update": completed})
    if completed and not any(item["update"] == completed for item in snapshots):
        _snapshot(completed)
    if stop_reason == "interrupted":
        ret_mod.unlink_latest_recovery(recovery_path)
    else:
        _save_boundary(force=True)

    final_eval_summary = summarize_evaluation(
        evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
    )
    final_path = out_dir / "ppo_final.pt"
    save_player_checkpoint(
        net, final_path, source_tool=TOOL_NAME, experimental=True, production_compatible=False,
        extra=_pack_extra(
            completed, final_eval_summary,
            parent_state_dict_sha256=parent_state_dict_sha256,
            parent_file_sha256=parent_file_sha256,
            dataset_hash=dataset_hash,
            knobs=knobs,
            rollout_seed_window=rollout_seed_window,
            eval_seeds=args.eval_seeds,
            arm=arm,
        ),
    )
    return {
        "optimizer_steps": optimizer_steps,
        "total_episodes": completed * args.episodes_per_update,
        "total_frames": total_frames,
        "snapshots": snapshots,
        "final_checkpoint": str(final_path),
        "completed_updates": completed,
        "stop_reason": stop_reason,
        "promotion": False,
    }


def run_experiment(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    out_dir = Path(getattr(args, "run_dir", args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_max_updates, effective_end_time = ret_mod.effective_stop_budget(args)
    init_net, init_meta = load_initial_checkpoint(args.init_checkpoint, device)
    init_prov = checkpoint_provenance(args.init_checkpoint, init_meta)
    try:
        teacher_prov = {"file_sha256": file_sha256(args.teacher)}
    except OSError:
        teacher_prov = {"file_sha256": None}
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
    knobs["warmup_updates"] = args.warmup_updates
    knobs["warmup_seed_start"] = WARMUP_SEED_START
    knobs["held_value_fit_seed_start"] = HELD_VALUE_FIT_SEED_START
    knobs["truncation_in_warmup"] = "drop_censored_episode"
    knobs["value_bootstrap"] = False
    contract = {
        "format": 1,
        "schema_version": WARMUP_RUN_SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "arm": args.arm,
        "inputs": {"init_checkpoint": init_prov, "teacher": teacher_prov},
        "provenance": provenance,
        "knobs": knobs,
        "stop_args": {
            "effective_end_time_hkt": effective_end_time.isoformat(),
            "effective_max_updates": effective_max_updates,
        },
        "effective_seed": getattr(args, "effective_seed", PPO_TORCH_SEED),
        "no_promotion": True,
    }
    selected = getattr(args, "resume_from_update", None)
    if selected is not None:
        resolved_max, resolved_end = ret_mod.resolve_resume_from_stop_budget(args, selected)
        args.effective_max_updates = resolved_max
        args.effective_end_time = resolved_end
        effective_max_updates, effective_end_time = resolved_max, resolved_end
        contract["stop_args"] = {
            "effective_end_time_hkt": resolved_end.isoformat(),
            "effective_max_updates": resolved_max,
        }
        require_matching_warmup_run_contract(out_dir, contract)
        ret_mod.rewind_run_to_archived_recovery(
            out_dir, selected, progress_filename="ppo_updates.jsonl", device=device,
        )
        authorized = create_or_validate_warmup_run_contract(out_dir, contract, resume=True)
        args.effective_max_updates = authorized["effective_max_updates"]
        args.effective_end_time = ret_mod.dt.datetime.fromisoformat(authorized["effective_end_time_hkt"])
        effective_max_updates, effective_end_time = args.effective_max_updates, args.effective_end_time
    elif getattr(args, "resume", False):
        authorized = create_or_validate_warmup_run_contract(out_dir, contract, resume=True)
        args.effective_max_updates = authorized["effective_max_updates"]
        args.effective_end_time = ret_mod.dt.datetime.fromisoformat(authorized["effective_end_time_hkt"])
        effective_max_updates, effective_end_time = args.effective_max_updates, args.effective_end_time

    gate_summary = summarize_evaluation(
        evaluate_deterministic(init_net, device, args.eval_seeds, args.eval_max_steps)
    )
    initial_gate = evaluate_initial_gate(
        gate_summary, mean_min=args.initial_gate_mean_min, median_min=args.initial_gate_median_min,
    )
    report: dict[str, Any] = {
        "status": None,
        "arm": args.arm,
        "arm_ran": False,
        "initial_gate": initial_gate,
        "provenance": provenance,
        "knobs": knobs,
        "promotion": False,
        "ppo_entry_held_ev": None,
        "warmup": None,
        "snapshots": None,
    }
    if not initial_gate["gate_pass"] or teacher_prov["file_sha256"] is None:
        report["status"] = "failed_closed"
        _write_report(out_dir, report)
        return report
    if not getattr(args, "resume", False):
        create_or_validate_warmup_run_contract(out_dir, contract, resume=False)
    ret_mod.append_run_status(out_dir, {
        "event": "resume_requested" if getattr(args, "resume", False) else "started",
        "arm": args.arm,
    })

    teacher = load_teacher(args.teacher, device)
    dataset = collect_canonical_dataset(
        teacher, device,
        episodes=args.data_episodes, seed_start=COLLECT_SEED_START,
        max_steps=args.data_max_steps, frames_cap=args.data_frames_cap,
        held_frac=HELD_OUT_FRAC, teacher_path=args.teacher,
    )
    report["dataset"] = {
        "hash": dataset["hash"],
        "n_held_frames": dataset["n_held_frames"],
        "n_train_frames": dataset["n_train_frames"],
    }

    net = PlayerRankedTopK(top_k=init_net.top_k, hidden=init_net.hidden).to(device)
    net.load_state_dict(init_net.state_dict())
    pack_path = out_dir / "held_value_fit_pack.pt"
    if getattr(args, "resume", False) and pack_path.is_file():
        pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    else:
        fit_gen = isolated_generator(HELD_FIT_GENERATOR_SEED, device)
        pack = collect_held_value_fit_pack(
            net, device, args.max_frames, fit_gen, n_seeds=args.held_value_fit_count,
        )
        torch.save(pack, pack_path)

    warmup_account = {
        "updates": 0, "episodes": 0, "total_frames": 0, "optimizer_steps": 0,
        "n_dropped_censored": 0, "wall_s": 0.0, "actor_unchanged": True,
        "held_policy_unchanged": True,
    }
    if args.arm == "warmup":
        warmup_account = run_warmup_phase(net, device, args, dataset["held_tensors"])
    report["warmup"] = warmup_account
    report["ppo_entry_held_ev"] = held_value_fit_ev(net, pack, device)

    ppo = run_ppo_phase(
        net, device, args, snapshot_schedule(args.updates), dataset["held_tensors"], out_dir,
        arm=args.arm,
        parent_state_dict_sha256=init_prov["state_dict_sha256"],
        parent_file_sha256=init_prov["file_sha256"],
        dataset_hash=dataset["hash"],
        knobs=knobs,
    )
    report["snapshots"] = ppo["snapshots"]
    report["ppo"] = ppo
    report["status"] = ppo["stop_reason"]
    report["arm_ran"] = True
    report["promotion"] = False
    _write_report(out_dir, report)
    return report


def main() -> None:
    args = apply_mode_defaults(build_parser().parse_args())
    report = run_experiment(args, torch.device(args.device))
    print(json.dumps({
        "status": report["status"], "arm": report["arm"], "arm_ran": report["arm_ran"],
        "promotion": False,
    }))


if __name__ == "__main__":
    main()
