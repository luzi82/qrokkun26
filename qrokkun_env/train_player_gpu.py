#!/usr/bin/env python3
"""Canonical net: qrokkun_env.agents.player_v1

GPU player training: flee BC pretrain + PPO vs scripted spawner.
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
import torch.nn.functional as F
from torch.distributions import Categorical

from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.policies import FleeNearestBullet
from qrokkun_env.agents.player_v1 import PlayerV1
from qrokkun_env.sanity import run_episode


ActorCritic = PlayerV1


@dataclass
class Traj:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)


def collect_traj(env: Qrokkun26Env, net: ActorCritic, device: torch.device, max_steps: int) -> tuple[Traj, float]:
    traj = Traj()
    env.reset()
    for _ in range(max_steps):
        obs = vectorize(env)
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        dist, value = net(x)
        action = dist.sample()
        traj.obs.append(obs)
        traj.actions.append(int(action.item()))
        traj.log_probs.append(dist.log_prob(action).detach())
        traj.values.append(value.detach())
        _o, base, done, info = env.step(int(action.item()))
        traj.rewards.append(shaped_reward(env, float(base), done))
        traj.dones.append(done)
        if done:
            return traj, float(info.get("elapsed", env.elapsed))
    return traj, env.elapsed


def gae(rewards, values, dones, gamma: float, lam: float, device: torch.device):
    vals = [float(v) for v in values] + [0.0]
    adv = []
    gae_v = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * vals[t + 1] * mask - vals[t]
        gae_v = delta + gamma * lam * mask * gae_v
        adv.append(gae_v)
    adv.reverse()
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    ret_t = adv_t + torch.tensor(vals[:-1], dtype=torch.float32, device=device)
    return adv_t, ret_t


@torch.no_grad()
def eval_mean(net: ActorCritic, device: torch.device, seeds: range, max_steps: int) -> float:
    net.eval()
    times = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        for _ in range(max_steps):
            x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
            dist, _v = net(x)
            a = int(dist.probs.argmax().item())
            _o, _r, done, info = env.step(a)
            if done:
                times.append(float(info["elapsed"]))
                break
        else:
            times.append(env.elapsed)
    net.train()
    return sum(times) / max(len(times), 1)


def ppo_update(net, opt, trajs, device, clip, epochs, minibatch, entropy_coef, value_coef, gamma, lam):
    obs, acts, old_lp, rews, vals, dones = [], [], [], [], [], []
    for tr in trajs:
        obs.extend(tr.obs)
        acts.extend(tr.actions)
        old_lp.extend([float(x) for x in tr.log_probs])
        rews.extend(tr.rewards)
        vals.extend(tr.values)
        dones.extend(tr.dones)
    # Compute advantages per trajectory then concat
    adv_all, ret_all, old_lp_t, act_t, obs_t = [], [], [], [], []
    idx = 0
    for tr in trajs:
        n = len(tr.rewards)
        adv, ret = gae(tr.rewards, tr.values, tr.dones, gamma, lam, device)
        adv_all.append(adv)
        ret_all.append(ret)
        old_lp_t.append(torch.tensor([float(x) for x in tr.log_probs], device=device))
        act_t.append(torch.tensor(tr.actions, dtype=torch.int64, device=device))
        obs_t.append(torch.tensor(tr.obs, dtype=torch.float32, device=device))
        idx += n
    adv = torch.cat(adv_all)
    ret = torch.cat(ret_all)
    old_lp = torch.cat(old_lp_t)
    acts = torch.cat(act_t)
    obs = torch.cat(obs_t)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, minibatch):
            mb = perm[start : start + minibatch]
            dist, value = net(obs[mb])
            lp = dist.log_prob(acts[mb])
            ratio = torch.exp(lp - old_lp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            pg = -torch.min(surr1, surr2).mean()
            vloss = F.mse_loss(value, ret[mb])
            ent = dist.entropy().mean()
            loss = pg + value_coef * vloss - entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bc-steps", type=int, default=400)
    ap.add_argument("--bc-batch", type=int, default=64)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=60 * 90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/player_gpu.pt"))
    ap.add_argument("--status", type=Path, default=Path("runs/player_gpu_status.json"))
    ap.add_argument("--log", type=Path, default=Path("runs/player_gpu.jsonl"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    net = ActorCritic(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)

    flee_ref = sum(run_episode("flee", seed=s).elapsed for s in range(200, 230)) / 30
    idle_ref = sum(run_episode("idle", seed=s).elapsed for s in range(200, 230)) / 30
    n_params = sum(p.numel() for p in net.parameters())
    print(f"device={device} params={n_params} flee_ref={flee_ref:.2f} idle_ref={idle_ref:.2f}", flush=True)

    print("BC pretrain…", flush=True)
    bc_loss = bc_pretrain(net, device, args.bc_steps, args.bc_batch, lr=1e-3)
    ev0 = eval_mean(net, device, range(300, 320), args.max_steps)
    print(f"BC done loss={bc_loss:.4f} eval={ev0:.2f}s", flush=True)

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    best = ev0
    torch.save({"state_dict": net.state_dict(), "hidden": args.hidden, "eval_mean_s": ev0, "stage": "bc"}, args.out)
    update = 0
    beat = best > flee_ref

    with args.log.open("w") as logf:
        while time.time() < deadline:
            trajs = []
            survs = []
            for i in range(args.rollouts):
                env = Qrokkun26Env(seed=args.seed + update * 100 + i)
                tr, surv = collect_traj(env, net, device, args.max_steps)
                trajs.append(tr)
                survs.append(surv)
            ppo_update(
                net, opt, trajs, device, args.clip, args.ppo_epochs, args.minibatch,
                args.entropy, args.value_coef, args.gamma, args.lam,
            )
            row = {
                "update": update,
                "surv_mean": sum(survs) / len(survs),
                "wall_h": (time.time() - t0) / 3600,
            }
            if update % 5 == 0:
                ev = eval_mean(net, device, range(300, 330), args.max_steps)
                row["eval_mean_s"] = ev
                row["flee_ref"] = flee_ref
                if ev > best:
                    best = ev
                    torch.save(
                        {
                            "state_dict": net.state_dict(),
                            "hidden": args.hidden,
                            "obs_dim": OBS_DIM,
                            "actions": list(ACTIONS),
                            "eval_mean_s": ev,
                            "update": update,
                            "algo": "bc+ppo",
                        },
                        args.out,
                    )
                status = {
                    "update": update,
                    "best_eval_s": best,
                    "last_eval_s": ev,
                    "flee_ref_s": flee_ref,
                    "idle_ref_s": idle_ref,
                    "beat_flee": best > flee_ref,
                    "wall_hours": (time.time() - t0) / 3600,
                    "params": n_params,
                    "device": str(device),
                    "checkpoint": str(args.out),
                    "bc_eval_s": ev0,
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"upd={update:5d} surv={row['surv_mean']:6.2f}s eval={ev:.2f}s "
                    f"best={best:.2f}s flee={flee_ref:.2f}s wall={status['wall_hours']:.2f}h",
                    flush=True,
                )
                if best > flee_ref and not beat:
                    beat = True
                    print(f"BEAT_FLEE at update={update} best={best:.2f}s", flush=True)
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            update += 1

    ev = eval_mean(net, device, range(300, 360), args.max_steps)
    if ev > best:
        best = ev
        torch.save(
            {
                "state_dict": net.state_dict(),
                "hidden": args.hidden,
                "obs_dim": OBS_DIM,
                "actions": list(ACTIONS),
                "eval_mean_s": ev,
                "update": update,
                "algo": "bc+ppo",
            },
            args.out,
        )
    status = {
        "update": update,
        "best_eval_s": best,
        "last_eval_s": ev,
        "flee_ref_s": flee_ref,
        "idle_ref_s": idle_ref,
        "beat_flee": best > flee_ref,
        "wall_hours": (time.time() - t0) / 3600,
        "params": n_params,
        "device": str(device),
        "checkpoint": str(args.out),
        "bc_eval_s": ev0,
        "done": True,
    }
    args.status.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
