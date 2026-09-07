"""Canonical net: qrokkun_env.agents.player_v0"""

#!/usr/bin/env python3
"""Train a tiny player MLP vs the scripted spawner (CPU-friendly REINFORCE)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.sanity import run_episode


class PlayerMLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(x))


@dataclass
class Rollout:
    log_probs: list[torch.Tensor]
    rewards: list[float]
    entropies: list[torch.Tensor]


def collect_episode(env: Qrokkun26Env, policy: PlayerMLP, max_steps: int) -> Rollout:
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    entropies: list[torch.Tensor] = []
    env.reset()
    for _ in range(max_steps):
        obs = torch.tensor(vectorize(env), dtype=torch.float32)
        dist = policy(obs)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        _o, reward, done, _info = env.step(int(action.item()))
        rewards.append(float(reward))
        if done:
            break
    return Rollout(log_probs=log_probs, rewards=rewards, entropies=entropies)


def discount_returns(rewards: list[float], gamma: float) -> list[float]:
    G = 0.0
    out: list[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def evaluate(policy: PlayerMLP, seeds: range, max_steps: int) -> float:
    policy.eval()
    times: list[float] = []
    with torch.no_grad():
        for seed in seeds:
            env = Qrokkun26Env(seed=seed)
            env.reset(seed=seed)
            for _ in range(max_steps):
                obs = torch.tensor(vectorize(env), dtype=torch.float32)
                action = int(policy(obs).probs.argmax().item())
                _o, _r, done, info = env.step(action)
                if done:
                    times.append(float(info["elapsed"]))
                    break
            else:
                times.append(env.elapsed)
    policy.train()
    return sum(times) / max(len(times), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=60 * 45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("dist/player_mlp.pt"))
    ap.add_argument("--log", type=Path, default=Path("dist/train_player_log.jsonl"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    policy = PlayerMLP(hidden=args.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"params={n_params} obs_dim={OBS_DIM} actions={len(ACTIONS)} device=cpu")

    t0 = time.time()
    best_eval = -1.0
    with args.log.open("w") as logf:
        for ep in range(args.episodes):
            env = Qrokkun26Env(seed=args.seed + ep)
            rollout = collect_episode(env, policy, args.max_steps)
            returns = discount_returns(rollout.rewards, args.gamma)
            r_t = torch.tensor(returns, dtype=torch.float32)
            # EMA baseline for advantage.
            ep_return = float(r_t[0]) if r_t.numel() else 0.0
            if not hasattr(main, "_baseline"):
                main._baseline = ep_return
            main._baseline = 0.9 * main._baseline + 0.1 * ep_return
            adv = r_t - main._baseline
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            pg = torch.stack([-lp * A for lp, A in zip(rollout.log_probs, adv)]).sum()
            entropy = torch.stack(rollout.entropies).mean()
            loss = pg - args.entropy * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            survived = sum(rollout.rewards)
            row = {
                "ep": ep,
                "survived": survived,
                "steps": len(rollout.rewards),
                "loss": float(loss.item()),
            }
            if ep % 20 == 0 or ep == args.episodes - 1:
                ev = evaluate(policy, range(100, 110), args.max_steps)
                row["eval_mean_s"] = ev
                if ev > best_eval:
                    best_eval = ev
                    torch.save(
                        {
                            "state_dict": policy.state_dict(),
                            "hidden": args.hidden,
                            "obs_dim": OBS_DIM,
                            "actions": list(ACTIONS),
                            "eval_mean_s": ev,
                            "episode": ep,
                        },
                        args.out,
                    )
                print(
                    f"ep={ep:4d} survived={survived:6.2f}s steps={len(rollout.rewards):4d} "
                    f"eval={ev:.2f}s best={best_eval:.2f}s"
                )
            logf.write(json.dumps(row) + "\n")
            logf.flush()

    # Baselines for context
    idle_t = sum(run_episode("idle", seed=s).elapsed for s in range(100, 110)) / 10
    flee_t = sum(run_episode("flee", seed=s).elapsed for s in range(100, 110)) / 10
    print(
        f"done in {time.time()-t0:.1f}s  checkpoint={args.out}  "
        f"best_eval={best_eval:.2f}s  idle={idle_t:.2f}s flee={flee_t:.2f}s"
    )


if __name__ == "__main__":
    main()
