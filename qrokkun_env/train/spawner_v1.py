#!/usr/bin/env python3
"""Canonical nets: agents.player_v1 + agents.spawner_v1

Train spawner PPO vs a frozen player policy (GB10).
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

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Bullet, Qrokkun26Env, _bullet_speed, _spawn_interval, _move_toward
from qrokkun_env.godot_rng import f32
from qrokkun_env.obs import OBS_DIM, vectorize

from qrokkun_env.agents.player_v1 import PlayerV1, OBS_DIM_V1
from qrokkun_env.agents.spawner_v1 import (
    SPAWNER_ACTIONS,
    SPAWNER_OBS_DIM,
    SpawnerV1,
    decode_spawner,
    spawn_from_action,
    vectorize_spawner,
)
from qrokkun_env.obs import OBS_DIM, vectorize

PlayerAC = PlayerV1
SpawnerAC = SpawnerV1
spawner_vectorize = vectorize_spawner

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
        return
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


@torch.no_grad()
def player_act(player: PlayerAC, env: Qrokkun26Env, device: torch.device) -> int:
    x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
    dist, _value = player(x)
    return int(dist.probs.argmax().item())


def run_episode(
    env: Qrokkun26Env,
    player: PlayerAC,
    spawner: SpawnerAC,
    device: torch.device,
    max_steps: int,
    train_spawner: bool,
) -> tuple[Traj, float]:
    """Spawner gets reward for killing faster; decisions only on spawn frames."""
    traj = Traj()
    env.reset()

    for _ in range(max_steps):
        # Advance time / decide whether spawn is due (mirror interval, without auto-spawn).
        env.elapsed += env.dt
        env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed)
        spawns_this_frame = 0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            sobs = spawner_vectorize(env)
            x = torch.tensor(sobs, dtype=torch.float32, device=device)
            dist, value = spawner(x)
            if train_spawner:
                action = dist.sample()
                logp = dist.log_prob(action)
            else:
                action = dist.probs.argmax()
                logp = dist.log_prob(action)
            spawn_from_action(env, int(action.item()))
            spawns_this_frame += 1
            if train_spawner:
                traj.obs.append(sobs)
                traj.actions.append(int(action.item()))
                traj.log_probs.append(logp.detach())
                traj.values.append(value.detach())
                r = -0.02
                if env.bullets:
                    d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
                    r += 0.01 * (1.0 - min(math.sqrt(d2) / 100.0, 1.0))
                traj.rewards.append(r)
                traj.dones.append(False)
            # Match scripted double-spawn chance after first in a burst.
            if spawns_this_frame == 1 and env.elapsed > 18.0 and env.rng.randf() < 0.16:
                pass  # allow loop to continue if acc still high; also force one extra
                env.spawn_acc += interval  # schedule immediate extra spawn
            interval = _spawn_interval(env.elapsed)

        # Player move
        pa = player_act(player, env, device)
        dx, dy = ACTION_TO_DIR[ACTIONS[pa]]
        if dx != 0.0 or dy != 0.0:
            n = math.hypot(dx, dy)
            dx, dy = dx / n, dy / n
            tx, ty = dx * C.PLAYER_MAX_SPEED, dy * C.PLAYER_MAX_SPEED
            env.pvx, env.pvy = _move_toward(env.pvx, env.pvy, tx, ty, C.PLAYER_ACCEL * env.dt)
        else:
            env.pvx = env.pvy = 0.0
        env.px = f32(env.px + f32(env.pvx * env.dt))
        env.py = f32(env.py + f32(env.pvy * env.dt))
        env.px = f32(min(max(env.px, C.FIELD_X + C.PLAYER_MARGIN), C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN))
        env.py = f32(min(max(env.py, C.FIELD_Y + C.PLAYER_MARGIN), C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN))
        env._integrate_bullets()
        if env._check_hit():
            env.dead = True
            if train_spawner and traj.rewards:
                traj.rewards[-1] += 5.0  # kill bonus on last spawn decision
                traj.dones[-1] = True
            return traj, env.elapsed
    if train_spawner and traj.rewards:
        traj.rewards[-1] -= 2.0  # timeout / player survived long
        traj.dones[-1] = True
    return traj, env.elapsed


@torch.no_grad()
def eval_spawner(player, spawner, device, seeds, max_steps) -> float:
    """Lower survival time = stronger spawner."""
    times = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        # use GodotRNG for the optional double-spawn coin flip only
        _tr, t = run_episode(env, player, spawner, device, max_steps, train_spawner=False)
        times.append(t)
    return sum(times) / max(len(times), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=4.8)
    ap.add_argument("--player-ckpt", type=Path, default=Path("runs/player_gpu.pt"))
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=60 * 90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/spawner_gpu.pt"))
    ap.add_argument("--status", type=Path, default=Path("runs/spawner_gpu_status.json"))
    ap.add_argument("--log", type=Path, default=Path("runs/spawner_gpu.jsonl"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load frozen player
    ck = torch.load(args.player_ckpt, map_location=device, weights_only=False)
    player = PlayerAC(hidden=int(ck.get("hidden", 256))).to(device)
    player.load_state_dict(ck["state_dict"])
    player.eval()
    for p in player.parameters():
        p.requires_grad_(False)

    spawner = SpawnerAC(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(spawner.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Baseline: scripted spawner survival vs this player (from player training status)
    base = eval_spawner(player, spawner, device, range(400, 420), args.max_steps)
    print(f"device={device} spawner_actions={SPAWNER_ACTIONS} init_eval_surv={base:.2f}s", flush=True)

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    best = base  # lower is better for spawner
    update = 0

    with args.log.open("w") as logf:
        while time.time() < deadline:
            trajs, survs = [], []
            for i in range(args.rollouts):
                env = Qrokkun26Env(seed=args.seed + update * 100 + i)
                tr, surv = run_episode(env, player, spawner, device, args.max_steps, True)
                trajs.append(tr)
                survs.append(surv)
            ppo_update(
                spawner, opt, trajs, device, args.clip, args.ppo_epochs, args.minibatch,
                args.entropy, args.value_coef, args.gamma, args.lam,
            )
            row = {"update": update, "surv_mean": sum(survs) / len(survs), "wall_h": (time.time() - t0) / 3600}
            if update % 5 == 0:
                ev = eval_spawner(player, spawner, device, range(500, 520), args.max_steps)
                row["eval_surv_s"] = ev
                improved = ev < best
                if improved:
                    best = ev
                    torch.save(
                        {
                            "state_dict": spawner.state_dict(),
                            "hidden": args.hidden,
                            "eval_surv_s": ev,
                            "update": update,
                            "n_actions": SPAWNER_ACTIONS,
                            "algo": "spawner-ppo-vs-frozen-player",
                        },
                        args.out,
                    )
                status = {
                    "update": update,
                    "best_surv_s": best,
                    "last_surv_s": ev,
                    "init_surv_s": base,
                    "lower_is_better": True,
                    "beat_init": best < base - 0.5,
                    "wall_hours": (time.time() - t0) / 3600,
                    "checkpoint": str(args.out),
                    "player_ckpt": str(args.player_ckpt),
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"upd={update:5d} surv={row['surv_mean']:6.2f}s eval={ev:.2f}s "
                    f"best={best:.2f}s init={base:.2f}s wall={status['wall_hours']:.2f}h",
                    flush=True,
                )
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            update += 1

    ev = eval_spawner(player, spawner, device, range(500, 540), args.max_steps)
    if ev < best:
        best = ev
        torch.save(
            {
                "state_dict": spawner.state_dict(),
                "hidden": args.hidden,
                "eval_surv_s": ev,
                "update": update,
                "n_actions": SPAWNER_ACTIONS,
                "algo": "spawner-ppo-vs-frozen-player",
            },
            args.out,
        )
    status = {
        "update": update,
        "best_surv_s": best,
        "last_surv_s": ev,
        "init_surv_s": base,
        "lower_is_better": True,
        "beat_init": best < base - 0.5,
        "wall_hours": (time.time() - t0) / 3600,
        "checkpoint": str(args.out),
        "player_ckpt": str(args.player_ckpt),
        "done": True,
    }
    args.status.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
