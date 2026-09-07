"""Canonical net: qrokkun_env.agents.player_v1 (long AC run)"""

#!/usr/bin/env python3
"""Time-boxed CPU training: tiny actor-critic vs scripted spawner.

Goal: beat the flee-nearest rule baseline on held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.sanity import run_episode
from qrokkun_env.train_player import PlayerMLP, evaluate


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


def shaped_reward(env: Qrokkun26Env, base: float, done: bool) -> float:
    """dt survival + tiny clear-space bonus; death penalty."""
    r = base
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        # ~0..1 bonus when far from nearest bullet (in field units).
        r += 0.002 * min(math.sqrt(d2) / 80.0, 1.0)
    if done and env.dead:
        r -= 1.0
    return r


@dataclass
class Batch:
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)


def collect(env: Qrokkun26Env, net: ActorCritic, max_steps: int) -> tuple[Batch, float]:
    batch = Batch()
    env.reset()
    for _ in range(max_steps):
        obs = torch.tensor(vectorize(env), dtype=torch.float32)
        dist, value = net(obs)
        action = dist.sample()
        batch.log_probs.append(dist.log_prob(action))
        batch.values.append(value)
        batch.entropies.append(dist.entropy())
        _o, base, done, info = env.step(int(action.item()))
        batch.rewards.append(shaped_reward(env, float(base), done))
        batch.dones.append(done)
        if done:
            return batch, float(info.get("elapsed", env.elapsed))
    return batch, env.elapsed


def gae_advantages(
    rewards: list[float],
    values: list[torch.Tensor],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    vals = [float(v.detach()) for v in values] + [0.0]
    adv = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * vals[t + 1] * mask - vals[t]
        gae = delta + gamma * lam * mask * gae
        adv.append(gae)
    adv.reverse()
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = adv_t + torch.tensor(vals[:-1], dtype=torch.float32)
    return adv_t, ret_t


def eval_ac(net: ActorCritic, seeds: range, max_steps: int) -> float:
    # Wrap as PlayerMLP-compatible for greedy eval via logits path.
    net.eval()
    times: list[float] = []
    with torch.no_grad():
        for seed in seeds:
            env = Qrokkun26Env(seed=seed)
            env.reset(seed=seed)
            for _ in range(max_steps):
                obs = torch.tensor(vectorize(env), dtype=torch.float32)
                dist, _v = net(obs)
                action = int(dist.probs.argmax().item())
                _o, _r, done, info = env.step(action)
                if done:
                    times.append(float(info["elapsed"]))
                    break
            else:
                times.append(env.elapsed)
    net.train()
    return sum(times) / max(len(times), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=60 * 60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("dist/player_ac_long.pt"))
    ap.add_argument("--log", type=Path, default=Path("dist/train_long.jsonl"))
    ap.add_argument("--status", type=Path, default=Path("dist/train_long_status.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    net = ActorCritic(hidden=args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    flee_ref = sum(run_episode("flee", seed=s).elapsed for s in range(200, 220)) / 20
    idle_ref = sum(run_episode("idle", seed=s).elapsed for s in range(200, 220)) / 20
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"params={n_params} hidden={args.hidden} hours={args.hours} "
        f"flee_ref={flee_ref:.2f}s idle_ref={idle_ref:.2f}s"
    )

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    best = -1.0
    ep = 0
    beat_flee = False

    with args.log.open("w") as logf:
        while time.time() < deadline:
            env = Qrokkun26Env(seed=args.seed + ep)
            batch, survived = collect(env, net, args.max_steps)
            adv, ret = gae_advantages(
                batch.rewards, batch.values, batch.dones, args.gamma, args.lam
            )
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            pg = torch.stack([-lp * A for lp, A in zip(batch.log_probs, adv)]).mean()
            vloss = torch.stack(
                [(v - R) ** 2 for v, R in zip(batch.values, ret)]
            ).mean()
            entropy = torch.stack(batch.entropies).mean()
            loss = pg + args.value_coef * vloss - args.entropy * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            row: dict = {
                "ep": ep,
                "survived": survived,
                "steps": len(batch.rewards),
                "loss": float(loss.item()),
                "elapsed_wall": time.time() - t0,
            }
            if ep % 50 == 0:
                ev = eval_ac(net, range(300, 320), args.max_steps)
                row["eval_mean_s"] = ev
                row["flee_ref"] = flee_ref
                row["idle_ref"] = idle_ref
                improved = ev > best
                if improved:
                    best = ev
                    torch.save(
                        {
                            "state_dict": net.state_dict(),
                            "hidden": args.hidden,
                            "obs_dim": OBS_DIM,
                            "actions": list(ACTIONS),
                            "eval_mean_s": ev,
                            "episode": ep,
                            "algo": "actor-critic-gae",
                        },
                        args.out,
                    )
                status = {
                    "ep": ep,
                    "best_eval_s": best,
                    "last_eval_s": ev,
                    "flee_ref_s": flee_ref,
                    "idle_ref_s": idle_ref,
                    "beat_flee": best > flee_ref,
                    "wall_hours": (time.time() - t0) / 3600,
                    "params": n_params,
                    "checkpoint": str(args.out),
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"ep={ep:5d} surv={survived:6.2f}s eval={ev:.2f}s "
                    f"best={best:.2f}s flee={flee_ref:.2f}s "
                    f"wall={(time.time()-t0)/60:.1f}m"
                )
                if best > flee_ref and not beat_flee:
                    beat_flee = True
                    print(f"BEAT_FLEE at ep={ep} best={best:.2f}s")
            logf.write(json.dumps(row) + "\n")
            if ep % 50 == 0:
                logf.flush()
            ep += 1

    # Final eval
    ev = eval_ac(net, range(300, 340), args.max_steps)
    if ev > best:
        best = ev
        torch.save(
            {
                "state_dict": net.state_dict(),
                "hidden": args.hidden,
                "obs_dim": OBS_DIM,
                "actions": list(ACTIONS),
                "eval_mean_s": ev,
                "episode": ep,
                "algo": "actor-critic-gae",
            },
            args.out,
        )
    status = {
        "ep": ep,
        "best_eval_s": best,
        "last_eval_s": ev,
        "flee_ref_s": flee_ref,
        "idle_ref_s": idle_ref,
        "beat_flee": best > flee_ref,
        "wall_hours": (time.time() - t0) / 3600,
        "params": n_params,
        "checkpoint": str(args.out),
        "done": True,
    }
    args.status.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
