"""Canonical nets: agents.player_v1 + agents.spawner_v1 (both_v1)"""

#!/usr/bin/env python3
"""Simultaneous player+spawner PPO; keep baseline ckpts for comparison."""

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

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Qrokkun26Env, _move_toward, _spawn_interval
from qrokkun_env.godot_rng import f32
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.train_spawner_gpu import (
    SPAWNER_ACTIONS,
    SPAWNER_OBS_DIM,
    SpawnerAC,
    spawn_from_action,
    spawner_vectorize,
)


class PlayerAC(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


@dataclass
class Traj:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)


def gae(rewards, values, dones, gamma, lam, device):
    vals = [float(v) for v in values] + [0.0]
    adv, gae_v = [], 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * vals[t + 1] * mask - vals[t]
        gae_v = delta + gamma * lam * mask * gae_v
        adv.append(gae_v)
    adv.reverse()
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    ret_t = adv_t + torch.tensor(vals[:-1], dtype=torch.float32, device=device)
    return adv_t, ret_t


def ppo_update(net, opt, trajs, device, clip, epochs, minibatch, entropy_coef, value_coef, gamma, lam):
    advs, rets, old_lps, acts, obss = [], [], [], [], []
    for tr in trajs:
        if not tr.rewards:
            continue
        adv, ret = gae(tr.rewards, tr.values, tr.dones, gamma, lam, device)
        advs.append(adv)
        rets.append(ret)
        old_lps.append(torch.tensor([float(x) for x in tr.log_probs], device=device))
        acts.append(torch.tensor(tr.actions, dtype=torch.int64, device=device))
        obss.append(torch.tensor(tr.obs, dtype=torch.float32, device=device))
    if not advs:
        return 0
    adv = torch.cat(advs)
    ret = torch.cat(rets)
    old_lp = torch.cat(old_lps)
    acts_t = torch.cat(acts)
    obs = torch.cat(obss)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, minibatch):
            mb = perm[start : start + minibatch]
            dist, value = net(obs[mb])
            lp = dist.log_prob(acts_t[mb])
            ratio = torch.exp(lp - old_lp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            loss = (
                -torch.min(surr1, surr2).mean()
                + value_coef * F.mse_loss(value, ret[mb])
                - entropy_coef * dist.entropy().mean()
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return N


def shaped_player_r(env: Qrokkun26Env, done: bool) -> float:
    r = C.DT  # survive
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.002 * min(math.sqrt(d2) / 80.0, 1.0)
    if done and env.dead:
        r -= 1.0
    return r


def shaped_spawner_r(env: Qrokkun26Env, done: bool, is_spawn_step: bool) -> float:
    r = -0.02 if is_spawn_step else 0.0
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.01 * (1.0 - min(math.sqrt(d2) / 100.0, 1.0))
    if done and env.dead:
        r += 5.0
    elif done and not env.dead:
        r -= 2.0
    return r


@torch.no_grad()
def act_player(net, env, device, sample: bool, temp: float = 1.0):
    x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
    dist, value = net(x)
    if temp != 1.0:
        dist = Categorical(logits=dist.logits / temp)
    a = dist.sample() if sample else dist.probs.argmax()
    return int(a.item()), float(dist.log_prob(a if sample else a).item()), float(value.item()), vectorize(env)


@torch.no_grad()
def act_spawner(net, env, device, sample: bool, temp: float = 1.0):
    sobs = spawner_vectorize(env)
    x = torch.tensor(sobs, dtype=torch.float32, device=device)
    dist, value = net(x)
    if temp != 1.0:
        dist = Categorical(logits=dist.logits / temp)
    a = dist.sample() if sample else dist.probs.argmax()
    return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), sobs


def run_episode(
    env: Qrokkun26Env,
    player: PlayerAC,
    spawner: SpawnerAC,
    device: torch.device,
    max_steps: int,
    sample: bool,
    train_player: bool,
    train_spawner: bool,
    temp_p: float,
    temp_s: float,
) -> tuple[Traj, Traj, float]:
    ptraj, straj = Traj(), Traj()
    env.reset()
    for _ in range(max_steps):
        env.elapsed += env.dt
        env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed)
        spawns = 0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            a, lp, v, sobs = act_spawner(spawner, env, device, sample and train_spawner, temp_s)
            spawn_from_action(env, a)
            spawns += 1
            if train_spawner:
                straj.obs.append(sobs)
                straj.actions.append(a)
                straj.log_probs.append(lp)
                straj.values.append(v)
                straj.rewards.append(shaped_spawner_r(env, False, True))
                straj.dones.append(False)
            if spawns == 1 and env.elapsed > 18.0 and env.rng.randf() < 0.16:
                env.spawn_acc += interval
            interval = _spawn_interval(env.elapsed)

        a, lp, v, pobs = act_player(player, env, device, sample and train_player, temp_p)
        if train_player:
            ptraj.obs.append(pobs)
            ptraj.actions.append(a)
            ptraj.log_probs.append(lp)
            ptraj.values.append(v)
            ptraj.rewards.append(0.0)  # fill after move
            ptraj.dones.append(False)

        dx, dy = ACTION_TO_DIR[ACTIONS[a]]
        if dx != 0.0 or dy != 0.0:
            n = math.hypot(dx, dy)
            dx, dy = dx / n, dy / n
            env.pvx, env.pvy = _move_toward(
                env.pvx, env.pvy, dx * C.PLAYER_MAX_SPEED, dy * C.PLAYER_MAX_SPEED, C.PLAYER_ACCEL * env.dt
            )
        else:
            env.pvx = env.pvy = 0.0
        env.px = f32(env.px + f32(env.pvx * env.dt))
        env.py = f32(env.py + f32(env.pvy * env.dt))
        env.px = f32(min(max(env.px, C.FIELD_X + C.PLAYER_MARGIN), C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN))
        env.py = f32(min(max(env.py, C.FIELD_Y + C.PLAYER_MARGIN), C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN))
        env._integrate_bullets()
        done = env._check_hit()
        if done:
            env.dead = True
        if train_player and ptraj.rewards:
            ptraj.rewards[-1] = shaped_player_r(env, done)
            ptraj.dones[-1] = done
        if done:
            if train_spawner and straj.rewards:
                straj.rewards[-1] += shaped_spawner_r(env, True, False)
                straj.dones[-1] = True
            return ptraj, straj, env.elapsed
    if train_player and ptraj.rewards:
        ptraj.dones[-1] = True
    if train_spawner and straj.rewards:
        straj.rewards[-1] += shaped_spawner_r(env, True, False)
        straj.dones[-1] = True
    return ptraj, straj, env.elapsed


@torch.no_grad()
def eval_pair(player, spawner, device, seeds, max_steps) -> float:
    times = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        _p, _s, t = run_episode(env, player, spawner, device, max_steps, False, False, False, 1.0, 1.0)
        times.append(t)
    return sum(times) / max(len(times), 1)


def load_player(path: Path, device) -> PlayerAC:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = PlayerAC(hidden=int(ck.get("hidden", 256))).to(device)
    net.load_state_dict(ck["state_dict"])
    return net


def load_spawner(path: Path, device) -> SpawnerAC:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = SpawnerAC(hidden=int(ck.get("hidden", 256))).to(device)
    net.load_state_dict(ck["state_dict"])
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.8)
    ap.add_argument("--baseline-dir", type=Path, default=Path("runs/baseline_v1"))
    ap.add_argument("--out-player", type=Path, default=Path("runs/both_player.pt"))
    ap.add_argument("--out-spawner", type=Path, default=Path("runs/both_spawner.pt"))
    ap.add_argument("--status", type=Path, default=Path("runs/both_status.json"))
    ap.add_argument("--compare", type=Path, default=Path("runs/both_vs_baseline.json"))
    ap.add_argument("--log", type=Path, default=Path("runs/both_gpu.jsonl"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--mix-baseline", type=float, default=0.35, help="fraction of eps vs frozen baseline opponent")
    ap.add_argument("--temp-p", type=float, default=1.15)
    ap.add_argument("--temp-s", type=float, default=1.25)
    ap.add_argument("--entropy-p", type=float, default=0.04)
    ap.add_argument("--entropy-s", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=60 * 60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    base_p = args.baseline_dir / "player_gpu.pt"
    base_s = args.baseline_dir / "spawner_gpu.pt"
    # Train copies initialized from baseline
    player = load_player(base_p, device)
    spawner = load_spawner(base_s, device)
    # Frozen baselines for mix + final compare
    base_player = load_player(base_p, device).eval()
    base_spawner = load_spawner(base_s, device).eval()
    for p in list(base_player.parameters()) + list(base_spawner.parameters()):
        p.requires_grad_(False)

    opt_p = torch.optim.Adam(player.parameters(), lr=args.lr)
    opt_s = torch.optim.Adam(spawner.parameters(), lr=args.lr)
    args.out_player.parent.mkdir(parents=True, exist_ok=True)

    # Baseline self-play survival (old vs old)
    base_vs_base = eval_pair(base_player, base_spawner, device, range(600, 620), args.max_steps)
    print(f"device={device} baseline_vs_baseline_surv={base_vs_base:.2f}s", flush=True)

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    best_cross = -1.0  # new player vs base spawner (higher better for player progress)
    update = 0

    with args.log.open("w") as logf:
        while time.time() < deadline:
            ptrajs, strajs, survs = [], [], []
            for i in range(args.rollouts):
                env = Qrokkun26Env(seed=args.seed + update * 100 + i)
                mix = (i / max(args.rollouts - 1, 1)) < args.mix_baseline
                # Half mixes: train player vs frozen base spawner; train spawner vs frozen base player
                if mix and i % 2 == 0:
                    p_net, s_net = player, base_spawner
                    tp, ts = True, False
                elif mix:
                    p_net, s_net = base_player, spawner
                    tp, ts = False, True
                else:
                    p_net, s_net = player, spawner
                    tp, ts = True, True
                pt, st, surv = run_episode(
                    env, p_net, s_net, device, args.max_steps, True, tp, ts, args.temp_p, args.temp_s
                )
                if tp:
                    ptrajs.append(pt)
                if ts:
                    strajs.append(st)
                survs.append(surv)

            ppo_update(
                player, opt_p, ptrajs, device, args.clip, args.ppo_epochs, args.minibatch,
                args.entropy_p, 0.5, args.gamma, args.lam,
            )
            ppo_update(
                spawner, opt_s, strajs, device, args.clip, args.ppo_epochs, args.minibatch,
                args.entropy_s, 0.5, args.gamma, args.lam,
            )

            row = {"update": update, "surv_mean": sum(survs) / len(survs), "wall_h": (time.time() - t0) / 3600}
            if update % 4 == 0:
                # Cross evaluations
                new_p_vs_base_s = eval_pair(player, base_spawner, device, range(700, 715), args.max_steps)
                base_p_vs_new_s = eval_pair(base_player, spawner, device, range(700, 715), args.max_steps)
                new_vs_new = eval_pair(player, spawner, device, range(800, 815), args.max_steps)
                row.update(
                    {
                        "newP_vs_baseS": new_p_vs_base_s,
                        "baseP_vs_newS": base_p_vs_new_s,
                        "new_vs_new": new_vs_new,
                        "base_vs_base": base_vs_base,
                    }
                )
                improved = new_p_vs_base_s > best_cross
                if improved:
                    best_cross = new_p_vs_base_s
                torch.save(
                    {"state_dict": player.state_dict(), "hidden": args.hidden, "algo": "both-player",
                     "eval_newP_vs_baseS": new_p_vs_base_s, "update": update},
                    args.out_player,
                )
                torch.save(
                    {"state_dict": spawner.state_dict(), "hidden": args.hidden, "algo": "both-spawner",
                     "eval_baseP_vs_newS": base_p_vs_new_s, "n_actions": SPAWNER_ACTIONS, "update": update},
                    args.out_spawner,
                )
                status = {
                    "update": update,
                    "wall_hours": (time.time() - t0) / 3600,
                    "base_vs_base": base_vs_base,
                    "newP_vs_baseS": new_p_vs_base_s,
                    "baseP_vs_newS": base_p_vs_new_s,
                    "new_vs_new": new_vs_new,
                    "best_newP_vs_baseS": best_cross,
                    "out_player": str(args.out_player),
                    "out_spawner": str(args.out_spawner),
                    "done": False,
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"upd={update:5d} train_surv={row['surv_mean']:5.2f} "
                    f"newP|baseS={new_p_vs_base_s:.2f} baseP|newS={base_p_vs_new_s:.2f} "
                    f"new|new={new_vs_new:.2f} base|base={base_vs_base:.2f} "
                    f"wall={status['wall_hours']:.2f}h",
                    flush=True,
                )
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            update += 1

    # Final comparison suite
    seeds = range(900, 940)
    cmp = {
        "base_vs_base": eval_pair(base_player, base_spawner, device, seeds, args.max_steps),
        "new_vs_new": eval_pair(player, spawner, device, seeds, args.max_steps),
        "newP_vs_baseS": eval_pair(player, base_spawner, device, seeds, args.max_steps),
        "baseP_vs_newS": eval_pair(base_player, spawner, device, seeds, args.max_steps),
        "newP_vs_newS": eval_pair(player, spawner, device, seeds, args.max_steps),
        "n_seeds": 40,
        "wall_hours": (time.time() - t0) / 3600,
        "updates": update,
        "baseline_dir": str(args.baseline_dir),
        "out_player": str(args.out_player),
        "out_spawner": str(args.out_spawner),
        "notes": {
            "higher_surv_better_for_player": True,
            "lower_surv_when_spawner_is_stronger": True,
        },
    }
    torch.save(
        {"state_dict": player.state_dict(), "hidden": args.hidden, "algo": "both-player",
         "eval_newP_vs_baseS": cmp["newP_vs_baseS"], "update": update},
        args.out_player,
    )
    torch.save(
        {"state_dict": spawner.state_dict(), "hidden": args.hidden, "algo": "both-spawner",
         "eval_baseP_vs_newS": cmp["baseP_vs_newS"], "n_actions": SPAWNER_ACTIONS, "update": update},
        args.out_spawner,
    )
    args.compare.write_text(json.dumps(cmp, indent=2) + "\n")
    status = {**cmp, "done": True}
    args.status.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(cmp, indent=2), flush=True)


if __name__ == "__main__":
    main()
