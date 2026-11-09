#!/usr/bin/env python3
"""PlayerV21 lockstep vectorized trainer & evaluator (Zero-Inertia Physics, 120s Cap)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_ai.v21.agents.player_v21 import PlayerV21, save_v21_checkpoint
from qrokkun_ai.v21.obs_v21 import (
    DEFAULT_HORIZON_V21,
    DEFAULT_TOP_K_V21,
    SAFETY_THRESHOLD_PX,
    compute_kinematic_expert_targets,
    encode_obs_v21,
)
from qrokkun_env import constants as C
from qrokkun_env.gv20 import Gv20Env

MAX_STEPS_V21 = 7200  # 120.0 seconds at 60 Hz
VAL_SEEDS_V21: list[int] = list(range(2000, 2030))
PAIRED_EVAL_SEEDS_V21: list[int] = list(range(3000, 3030))
ROLLOUT_SEED_START_V21 = 100000

PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_MINIBATCH = 512
PPO_MAX_GRAD_NORM = 1.0
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3


def shaped_reward_v21(env: Gv20Env, base: float, done: bool) -> float:
    """Zero-inertia per-step survival reward + clearance shaping + true-death penalty."""
    r = float(base)
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.003 * min(math.sqrt(d2) / 80.0, 1.0)
    if done and env.dead:
        r -= 1.0
    return r


@dataclass
class RolloutV21:
    seed: int
    player: list[np.ndarray] = field(default_factory=list)
    bullets: list[np.ndarray] = field(default_factory=list)
    pad: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    expert_actions: list[int] = field(default_factory=list)
    expert_logits: list[np.ndarray] = field(default_factory=list)
    elapsed: float = 0.0
    terminated: bool = False
    truncated: bool = False
    censored: bool = False
    bootstrap_value: float = 0.0


@torch.no_grad()
def eval_player_v21(
    net: PlayerV21,
    device: torch.device,
    seeds: list[int] = PAIRED_EVAL_SEEDS_V21,
    max_steps: int = MAX_STEPS_V21,
) -> dict[str, Any]:
    """Deterministic lockstep vectorized evaluation across `seeds` up to `max_steps` (120s cap)."""
    was_train = net.training
    net.eval()

    envs = [Gv20Env(seed=int(s)) for s in seeds]
    for e, s in zip(envs, seeds):
        e.reset(seed=int(s))

    active = list(range(len(envs)))
    times = [0.0] * len(envs)
    terminated_flags = [False] * len(envs)
    truncated_flags = [False] * len(envs)

    for _ in range(int(max_steps)):
        if not active:
            break
        p_l, b_l, m_l = [], [], []
        for idx in active:
            p, b, m = encode_obs_v21(envs[idx])
            p_l.append(p)
            b_l.append(b)
            m_l.append(m)
        p_t = torch.tensor(np.stack(p_l), dtype=torch.float32, device=device)
        b_t = torch.tensor(np.stack(b_l), dtype=torch.float32, device=device)
        m_t = torch.tensor(np.stack(m_l), dtype=torch.bool, device=device)
        dist = net.forward_policy(p_t, b_t, m_t)
        actions = dist.logits.argmax(dim=-1).cpu().tolist()

        still_active = []
        for idx, a in zip(active, actions):
            _, _, done, info = envs[idx].step(int(a))
            if done:
                times[idx] = float(info.get("elapsed", envs[idx].elapsed))
                terminated_flags[idx] = True
                truncated_flags[idx] = False
            else:
                still_active.append(idx)
        active = still_active

    # Remaining active environments reached max_steps without dying -> truncated / censored
    for idx in active:
        times[idx] = float(envs[idx].elapsed)
        terminated_flags[idx] = False
        truncated_flags[idx] = True

    if was_train:
        net.train()

    censors = sum(1 for flag in truncated_flags if flag)
    return {
        "n": len(seeds),
        "max_steps": int(max_steps),
        "cap_seconds": float(max_steps) * C.DT,
        "mean": float(np.mean(times)) if times else 0.0,
        "median": float(np.median(times)) if times else 0.0,
        "std": float(np.std(times)) if times else 0.0,
        "min": float(np.min(times)) if times else 0.0,
        "max": float(np.max(times)) if times else 0.0,
        "censors": int(censors),
        "per_seed": [
            {
                "seed": int(s),
                "elapsed": float(t),
                "terminated": bool(term),
                "truncated": bool(trunc),
                "censored": bool(trunc),
            }
            for s, t, term, trunc in zip(seeds, times, terminated_flags, truncated_flags)
        ],
    }


@torch.no_grad()
def collect_rollouts_v21(
    net: PlayerV21,
    device: torch.device,
    seeds: list[int],
    *,
    max_steps: int = MAX_STEPS_V21,
) -> list[RolloutV21]:
    """Collect strictly on-policy lockstep vectorized rollouts with exact cap-boundary death precedence and V(s_T) bootstrap."""
    was_train = net.training
    net.eval()

    envs = [Gv20Env(seed=int(s)) for s in seeds]
    for e, s in zip(envs, seeds):
        e.reset(seed=int(s))

    rollouts = [RolloutV21(seed=int(s)) for s in seeds]
    active = list(range(len(envs)))

    for _ in range(int(max_steps)):
        if not active:
            break
        p_l, b_l, m_l = [], [], []
        for idx in active:
            p, b, m = encode_obs_v21(envs[idx])
            p_l.append(p)
            b_l.append(b)
            m_l.append(m)
        p_t = torch.tensor(np.stack(p_l), dtype=torch.float32, device=device)
        b_t = torch.tensor(np.stack(b_l), dtype=torch.float32, device=device)
        m_t = torch.tensor(np.stack(m_l), dtype=torch.bool, device=device)

        dist, val_t = net(p_t, b_t, m_t)
        exp_act_t, exp_logits_t = compute_kinematic_expert_targets(
            p_t,
            b_t,
            m_t,
            base_logits=None,
            horizon=net.lookahead_horizon,
            top_k=net.top_k,
            safety_margin=SAFETY_THRESHOLD_PX,
        )

        # Strictly on-policy action sampling so old_log_probs match behavior policy!
        exec_acts = dist.sample()
        lp_t = dist.log_prob(exec_acts)

        acts = exec_acts.cpu().tolist()
        lps = lp_t.cpu().tolist()
        vals = val_t.cpu().tolist()
        exp_acts = exp_act_t.cpu().tolist()
        exp_logs = exp_logits_t.cpu().numpy()

        still_active = []
        for i_local, idx in enumerate(active):
            env = envs[idx]
            ro = rollouts[idx]
            a = int(acts[i_local])
            ro.player.append(p_l[i_local])
            ro.bullets.append(b_l[i_local])
            ro.pad.append(m_l[i_local])
            ro.actions.append(a)
            ro.log_probs.append(float(lps[i_local]))
            ro.values.append(float(vals[i_local]))
            ro.expert_actions.append(int(exp_acts[i_local]))
            ro.expert_logits.append(exp_logs[i_local])

            _, base, done, info = env.step(a)
            ro.rewards.append(shaped_reward_v21(env, float(base), bool(done)))
            ro.dones.append(bool(done))
            ro.elapsed = float(info.get("elapsed", env.elapsed))
            if done:
                ro.terminated = True
                ro.truncated = False
                ro.censored = False
                ro.bootstrap_value = 0.0
            else:
                still_active.append(idx)
        active = still_active

    # Any rollout still active after max_steps did NOT die on the final step -> truncated/censored!
    if active:
        p_l, b_l, m_l = [], [], []
        for idx in active:
            p, b, m = encode_obs_v21(envs[idx])
            p_l.append(p)
            b_l.append(b)
            m_l.append(m)
        p_t = torch.tensor(np.stack(p_l), dtype=torch.float32, device=device)
        b_t = torch.tensor(np.stack(b_l), dtype=torch.float32, device=device)
        m_t = torch.tensor(np.stack(m_l), dtype=torch.bool, device=device)
        last_vals = net.forward_value(p_t, b_t, m_t).cpu().tolist()
        for idx, lv in zip(active, last_vals):
            ro = rollouts[idx]
            ro.terminated = False
            ro.truncated = True
            ro.censored = True
            ro.bootstrap_value = float(lv)

    if was_train:
        net.train()
    return rollouts


def rollouts_to_batch_v21(
    rollouts: list[RolloutV21],
    gamma: float,
    lam: float,
    device: torch.device,
    *,
    reward_scale: float = 0.05,
) -> dict[str, torch.Tensor]:
    player_l, bullets_l, pad_l = [], [], []
    actions_l, old_lp_l, values_l = [], [], []
    exp_act_l, exp_logits_l = [], []
    adv_parts, ret_parts = [], []

    for ro in rollouts:
        if not ro.actions:
            continue
        player_l.extend(ro.player)
        bullets_l.extend(ro.bullets)
        pad_l.extend(ro.pad)
        actions_l.extend(ro.actions)
        old_lp_l.extend(ro.log_probs)
        values_l.extend(ro.values)
        exp_act_l.extend(ro.expert_actions)
        exp_logits_l.extend(ro.expert_logits)

        rews = [r * reward_scale for r in ro.rewards]
        tail_val = float(ro.bootstrap_value) if ro.truncated else 0.0
        vals = [float(v) for v in ro.values] + [tail_val]
        running = 0.0
        adv_list = [0.0] * len(rews)
        for t in range(len(rews) - 1, -1, -1):
            mask = 0.0 if ro.dones[t] else 1.0
            delta = rews[t] + gamma * vals[t + 1] * mask - vals[t]
            running = delta + gamma * lam * mask * running
            adv_list[t] = running
        adv_t = torch.tensor(adv_list, dtype=torch.float32, device=device)
        ret_t = adv_t + torch.tensor(vals[:-1], dtype=torch.float32, device=device)
        adv_parts.append(adv_t)
        ret_parts.append(ret_t)

    return {
        "player": torch.tensor(np.stack(player_l), dtype=torch.float32, device=device),
        "bullets": torch.tensor(np.stack(bullets_l), dtype=torch.float32, device=device),
        "pad": torch.tensor(np.stack(pad_l), dtype=torch.bool, device=device),
        "actions": torch.tensor(actions_l, dtype=torch.int64, device=device),
        "old_log_probs": torch.tensor(old_lp_l, dtype=torch.float32, device=device),
        "values": torch.tensor(values_l, dtype=torch.float32, device=device),
        "advantages": torch.cat(adv_parts),
        "returns": torch.cat(ret_parts),
        "expert_actions": torch.tensor(exp_act_l, dtype=torch.int64, device=device),
        "expert_logits": torch.tensor(np.stack(exp_logits_l), dtype=torch.float32, device=device),
    }


def train_step_v21(
    net: PlayerV21,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    dagger_weight: float = 1.0,
    ppo_weight: float = 0.3,
    entropy_coef: float = 0.005,
    minibatch: int = PPO_MINIBATCH,
) -> dict[str, float]:
    net.train()
    player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
    actions, old_lp = batch["actions"], batch["old_log_probs"]
    old_values, returns = batch["values"], batch["returns"]
    raw_adv = batch["advantages"]
    advantages = (raw_adv - raw_adv.mean()) / (raw_adv.std(unbiased=False) + 1e-8)
    exp_acts = batch["expert_actions"]
    exp_logits = batch["expert_logits"]

    actor_params = net.actor_parameters()
    critic_params = net.critic_parameters()

    n_samples = int(player.shape[0])
    pl_sum, vl_sum, dl_sum, kl_sum, agr_sum, seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n_samples, device=player.device)
        for start in range(0, n_samples, minibatch):
            mb = perm[start : start + minibatch]
            dist, v_pred = net(player[mb], bullets[mb], pad[mb])
            log_prob = dist.log_prob(actions[mb])
            ratio = torch.exp(log_prob - old_lp[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            hard_ce = F.cross_entropy(dist.logits, exp_acts[mb])
            t_probs = F.softmax(exp_logits[mb], dim=-1)
            s_logp = F.log_softmax(dist.logits, dim=-1)
            soft_ce = -(t_probs * s_logp).sum(-1).mean()
            dagger_loss = 0.65 * hard_ce + 0.35 * soft_ce

            actor_loss = ppo_weight * policy_loss + dagger_weight * dagger_loss - entropy_coef * entropy
            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor_params, PPO_MAX_GRAD_NORM)
            opt_actor.step()

            value_loss = F.mse_loss(v_pred, returns[mb])
            opt_critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic_params, PPO_MAX_GRAD_NORM)
            opt_critic.step()

            with torch.no_grad():
                approx_kl = float((old_lp[mb] - log_prob).mean().item())
                agr = float((dist.logits.argmax(dim=-1) == exp_acts[mb]).float().sum().item())
            bs = int(mb.shape[0])
            pl_sum += float(policy_loss.item()) * bs
            vl_sum += float(value_loss.item()) * bs
            dl_sum += float(dagger_loss.item()) * bs
            kl_sum += approx_kl * bs
            agr_sum += agr
            seen += bs

    with torch.no_grad():
        ret_var = torch.var(returns)
        ev = float(1.0 - torch.var(returns - old_values) / (ret_var + 1e-8))

    denom = max(seen, 1)
    return {
        "policy_loss": pl_sum / denom,
        "value_loss": vl_sum / denom,
        "dagger_loss": dl_sum / denom,
        "approx_kl": kl_sum / denom,
        "expert_agreement": agr_sum / denom,
        "explained_variance": ev,
        "n_samples": n_samples,
    }


def run_training_v21(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.quick else args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "progress.jsonl"

    updates = 2 if args.quick else int(args.updates)
    episodes_per_update = 2 if args.quick else int(args.episodes_per_update)
    max_steps = 120 if args.quick else int(args.max_steps)
    val_seeds = VAL_SEEDS_V21[:3] if args.quick else VAL_SEEDS_V21
    eval_seeds = PAIRED_EVAL_SEEDS_V21[:3] if args.quick else PAIRED_EVAL_SEEDS_V21

    net = PlayerV21(
        top_k=int(args.top_k),
        hidden=int(args.hidden),
        lookahead_horizon=int(args.horizon),
    ).to(device)
    opt_actor = torch.optim.Adam(net.actor_parameters(), lr=ACTOR_LR, eps=1e-8)
    opt_critic = torch.optim.Adam(net.critic_parameters(), lr=CRITIC_LR, eps=1e-8)

    val_0 = eval_player_v21(net, device, seeds=val_seeds, max_steps=max_steps)
    best_val_mean = val_0["mean"]
    best_update = 0
    best_ckpt_path = out_dir / "best_player_v21.pt"
    latest_ckpt_path = out_dir / "latest_player_v21.pt"
    save_v21_checkpoint(net, best_ckpt_path, extra={"update": 0, "val_eval": val_0})

    snapshots = [{"update": 0, "val_eval": val_0}]
    t0 = time.perf_counter()
    print(
        f"[PlayerV21] u=0 | val_mean={val_0['mean']:.2f}s (med={val_0['median']:.2f}s, cens={val_0['censors']}/{val_0['n']})",
        flush=True,
    )

    with log_path.open("w") as logf:
        for u in range(1, updates + 1):
            u_t0 = time.perf_counter()
            seeds = list(
                range(
                    ROLLOUT_SEED_START_V21 + (u - 1) * episodes_per_update,
                    ROLLOUT_SEED_START_V21 + u * episodes_per_update,
                )
            )
            rollouts = collect_rollouts_v21(
                net, device, seeds, max_steps=max_steps
            )
            batch = rollouts_to_batch_v21(
                rollouts, gamma=args.gamma, lam=args.lam, device=device, reward_scale=0.05
            )
            metrics = train_step_v21(
                net,
                opt_actor,
                opt_critic,
                batch,
                dagger_weight=args.dagger_weight,
                ppo_weight=args.ppo_weight,
                entropy_coef=args.entropy_coef,
            )
            surv_mean = statistics.mean(r.elapsed for r in rollouts)
            row = {
                "update": u,
                "rollout_surv_mean": surv_mean,
                **metrics,
                "wall_s": time.perf_counter() - u_t0,
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()

            if args.quick or u in (10, 25, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000) or u == updates:
                val_res = eval_player_v21(net, device, seeds=val_seeds, max_steps=max_steps)
                eval_res = eval_player_v21(net, device, seeds=eval_seeds, max_steps=max_steps)
                snapshots.append({"update": u, "val_eval": val_res, "paired_eval": eval_res})
                save_v21_checkpoint(
                    net, latest_ckpt_path, extra={"update": u, "val_eval": val_res, "paired_eval": eval_res}
                )
                if val_res["mean"] >= best_val_mean:
                    best_val_mean = val_res["mean"]
                    best_update = u
                    save_v21_checkpoint(
                        net, best_ckpt_path, extra={"update": u, "val_eval": val_res, "paired_eval": eval_res}
                    )
                el_m = (time.perf_counter() - t0) / 60.0
                print(
                    f"[PlayerV21] u={u:4d}/{updates} ({el_m:5.1f}m) | "
                    f"val={val_res['mean']:5.2f}s | paired_eval={eval_res['mean']:5.2f}s "
                    f"(med={eval_res['median']:5.2f}s, cens={eval_res['censors']}/{eval_res['n']}) | "
                    f"best_val={best_val_mean:5.2f}s@u{best_update} | agr={metrics['expert_agreement']:.3f} | EV={metrics['explained_variance']:.3f}",
                    flush=True,
                )

    report = {
        "architecture": net.architecture,
        "architecture_version": net.architecture_version,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "max_steps": max_steps,
        "cap_seconds": float(max_steps) * C.DT,
        "best_val_mean": best_val_mean,
        "best_update": best_update,
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
        "snapshots": snapshots,
        "total_wall_s": time.perf_counter() - t0,
    }
    with (out_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--episodes-per-update", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_V21)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_V21)
    ap.add_argument("--hidden", type=int, default=472)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_V21)
    ap.add_argument("--gamma", type=float, default=0.9995)
    ap.add_argument("--lam", type=float, default=0.999)
    ap.add_argument("--dagger-weight", type=float, default=1.0)
    ap.add_argument("--ppo-weight", type=float, default=0.3)
    ap.add_argument("--entropy-coef", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/player_v21"))
    ap.add_argument("--quick", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    run_training_v21(args)


if __name__ == "__main__":
    main()
