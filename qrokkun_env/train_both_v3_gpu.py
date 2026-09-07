#!/usr/bin/env python3
"""train-both v3: diversity (RNG+pool+entropy), spawner chooses bullet kind. No early-kill penalty."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Bullet, Qrokkun26Env, _bullet_speed, _move_toward, _spawn_interval
from qrokkun_env.godot_rng import f32
from qrokkun_env.obs_rich import OBS_DIM_RICH, vectorize_rich
from qrokkun_env.policies import FleeNearestBullet

# edge(4) * along(8) * aim(9) * kind(4)
N_EDGE, N_ALONG, N_AIM, N_KIND = 4, 8, 9, 4
SPAWNER_ACTIONS = N_EDGE * N_ALONG * N_AIM * N_KIND  # 1152
OBS_DIM = OBS_DIM_RICH  # 390: player+spawn_acc + 64 bullets*(dx,dy,vx,vy,r,kind)


def decode_spawner(a: int) -> tuple[int, int, int, int]:
    a = int(a)
    kind = a % N_KIND
    a //= N_KIND
    aim = a % N_AIM
    a //= N_AIM
    along = a % N_ALONG
    edge = a // N_ALONG
    return edge, along, aim, kind


def spawn_from_action(env: Qrokkun26Env, action: int, *, rng_jitter: bool = True) -> None:
    """Learned spawn with optional env RNG jitter for diversity."""
    edge, along, aim, kind = decode_spawner(action)
    fx0, fy0 = C.FIELD_X, C.FIELD_Y
    fx1, fy1 = C.FIELD_X + C.FIELD_W, C.FIELD_Y + C.FIELD_H
    t = (along + 0.5) / N_ALONG
    if rng_jitter:
        t = min(max(t + env.rng.randf_range(-0.04, 0.04), 0.02), 0.98)
    if edge == 0:
        pos = (fx0 + (fx1 - fx0) * t, fy0 - 8.0)
    elif edge == 1:
        pos = (fx0 + (fx1 - fx0) * t, fy1 + 8.0)
    elif edge == 2:
        pos = (fx0 - 8.0, fy0 + (fy1 - fy0) * t)
    else:
        pos = (fx1 + 8.0, fy0 + (fy1 - fy0) * t)

    lead = 0.15
    tx = env.px + env.pvx * lead
    ty = env.py + env.pvy * lead
    if aim == 0:
        dx, dy = tx - pos[0], ty - pos[1]
    else:
        ang = (aim - 1) * (math.tau / 8.0)
        miss = 18.0
        if rng_jitter:
            miss = env.rng.randf_range(12.0, 26.0)
            ang += env.rng.randf_range(-0.25, 0.25)
        dx = (tx + math.cos(ang) * miss) - pos[0]
        dy = (ty + math.sin(ang) * miss) - pos[1]
    if rng_jitter and aim == 0:
        jitter = env.rng.randf_range(-0.2, 0.2)
        cos_j, sin_j = math.cos(jitter), math.sin(jitter)
        dx, dy = dx * cos_j - dy * sin_j, dx * sin_j + dy * cos_j

    speed = _bullet_speed(env.elapsed)
    if rng_jitter:
        speed *= env.rng.randf_range(0.92, 1.08)
    # kind-specific speed tweaks (match scripted spirit)
    if kind == 2:
        speed *= 0.62
    elif kind == 3:
        speed *= 1.15
    n = math.hypot(dx, dy) or 1.0
    vx, vy = f32(dx / n * speed), f32(dy / n * speed)
    env.bullets.append(Bullet(f32(pos[0]), f32(pos[1]), vx, vy, kind, C.BULLET_RADIUS[kind]))


class PlayerAC(nn.Module):
    def __init__(self, hidden: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_RICH, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


class SpawnerAC(nn.Module):
    def __init__(self, hidden: int = 512, n_actions: int = SPAWNER_ACTIONS) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_RICH, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.policy = nn.Linear(hidden, n_actions)
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
    packs = []
    for tr in trajs:
        if not tr.rewards:
            continue
        adv, ret = gae(tr.rewards, tr.values, tr.dones, gamma, lam, device)
        packs.append((
            torch.tensor(tr.obs, dtype=torch.float32, device=device),
            torch.tensor(tr.actions, dtype=torch.int64, device=device),
            torch.tensor([float(x) for x in tr.log_probs], device=device),
            adv, ret,
        ))
    if not packs:
        return 0
    obs = torch.cat([p[0] for p in packs])
    acts = torch.cat([p[1] for p in packs])
    old_lp = torch.cat([p[2] for p in packs])
    adv = torch.cat([p[3] for p in packs])
    ret = torch.cat([p[4] for p in packs])
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
    r = C.DT
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.002 * min(math.sqrt(d2) / 80.0, 1.0)
    if done and env.dead:
        r -= 1.0
    return r


def shaped_spawner_r(env: Qrokkun26Env, done: bool, is_spawn_step: bool) -> float:
    # No early-one-shot penalty — full kill bonus anytime.
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
def act_player_net(net, env, device, sample: bool, temp: float):
    x = torch.tensor(vectorize_rich(env), dtype=torch.float32, device=device)
    dist, value = net(x)
    if temp != 1.0:
        dist = Categorical(logits=dist.logits / max(temp, 1e-6))
    a = dist.sample() if sample else dist.probs.argmax()
    return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), vectorize_rich(env)


@torch.no_grad()
def act_spawner_net(net, env, device, sample: bool, temp: float):
    sobs = vectorize_rich(env)
    x = torch.tensor(sobs, dtype=torch.float32, device=device)
    dist, value = net(x)
    if temp != 1.0:
        dist = Categorical(logits=dist.logits / max(temp, 1e-6))
    a = dist.sample() if sample else dist.probs.argmax()
    return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), sobs


def apply_player_action(env: Qrokkun26Env, a: int) -> None:
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


def run_episode(
    env: Qrokkun26Env,
    player,  # PlayerAC | None | "flee"
    spawner,  # SpawnerAC | None ("scripted")
    device: torch.device,
    max_steps: int,
    *,
    sample: bool,
    train_player: bool,
    train_spawner: bool,
    temp_p: float,
    temp_s: float,
    rng_jitter: bool,
) -> tuple[Traj, Traj, float]:
    ptraj, straj = Traj(), Traj()
    env.reset()
    flee = FleeNearestBullet() if player == "flee" else None

    for _ in range(max_steps):
        if spawner is None:
            # scripted: use env.step for spawn+physics after choosing player action
            if player == "flee":
                a = ACTIONS.index(flee.act(env))
                lp = v = 0.0
                pobs = vectorize_rich(env)
            else:
                a, lp, v, pobs = act_player_net(player, env, device, sample and train_player, temp_p)
            if train_player:
                ptraj.obs.append(pobs)
                ptraj.actions.append(a)
                ptraj.log_probs.append(lp)
                ptraj.values.append(v)
                ptraj.rewards.append(0.0)
                ptraj.dones.append(False)
            _o, _r, done, _ = env.step(a)
            if train_player and ptraj.rewards:
                ptraj.rewards[-1] = shaped_player_r(env, done)
                ptraj.dones[-1] = done
            if done:
                return ptraj, straj, env.elapsed
            continue

        # learned spawner path
        env.elapsed += env.dt
        env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed)
        spawns = 0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            a, lp, v, sobs = act_spawner_net(spawner, env, device, sample and train_spawner, temp_s)
            spawn_from_action(env, a, rng_jitter=rng_jitter)
            spawns += 1
            if train_spawner:
                straj.obs.append(sobs)
                straj.actions.append(a)
                straj.log_probs.append(lp)
                straj.values.append(v)
                straj.rewards.append(shaped_spawner_r(env, False, True))
                straj.dones.append(False)
            # earlier double-spawn chance for diversity (from 8s, not only 18s)
            thr = 8.0
            p_double = 0.20 if env.elapsed > 18.0 else (0.10 if env.elapsed > thr else 0.0)
            if spawns == 1 and p_double > 0 and env.rng.randf() < p_double:
                env.spawn_acc += interval
            interval = _spawn_interval(env.elapsed)

        if player == "flee":
            a = ACTIONS.index(flee.act(env))
            lp = v = 0.0
            pobs = vectorize_rich(env)
        else:
            a, lp, v, pobs = act_player_net(player, env, device, sample and train_player, temp_p)
        if train_player:
            ptraj.obs.append(pobs)
            ptraj.actions.append(a)
            ptraj.log_probs.append(lp)
            ptraj.values.append(v)
            ptraj.rewards.append(0.0)
            ptraj.dones.append(False)

        apply_player_action(env, a)
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
def eval_pair(player, spawner, device, seeds, max_steps, rng_jitter=False) -> float:
    times = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        _p, _s, t = run_episode(
            env, player, spawner, device, max_steps,
            sample=False, train_player=False, train_spawner=False,
            temp_p=1.0, temp_s=1.0, rng_jitter=rng_jitter,
        )
        times.append(t)
    return sum(times) / max(len(times), 1)


def load_player(path: Path, device) -> PlayerAC:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = PlayerAC(hidden=int(ck.get("hidden", 256))).to(device)
    net.load_state_dict(ck["state_dict"])
    return net


def freeze(net: nn.Module) -> nn.Module:
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.4)
    ap.add_argument("--out-player", type=Path, default=Path("runs/both_v3_player.pt"))
    ap.add_argument("--out-spawner", type=Path, default=Path("runs/both_v3_spawner.pt"))
    ap.add_argument("--status", type=Path, default=Path("runs/both_v3_status.json"))
    ap.add_argument("--compare", type=Path, default=Path("runs/both_v3_compare.json"))
    ap.add_argument("--log", type=Path, default=Path("runs/both_v3.jsonl"))
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--temp-p", type=float, default=1.2)
    ap.add_argument("--temp-s", type=float, default=1.3)
    ap.add_argument("--entropy-p", type=float, default=0.05)
    ap.add_argument("--entropy-s", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=60 * 70)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Obs dim changed (64 bullets) → fresh nets; pool = flee + scripted only until new ckpts exist.
    player = PlayerAC(hidden=args.hidden).to(device)
    spawner = SpawnerAC(hidden=args.hidden, n_actions=SPAWNER_ACTIONS).to(device)
    pool_players = [("flee", "flee")]

    opt_p = torch.optim.Adam(player.parameters(), lr=args.lr)
    opt_s = torch.optim.Adam(spawner.parameters(), lr=args.lr)
    args.out_player.parent.mkdir(parents=True, exist_ok=True)

    # Anchors
    flee_vs_scripted = eval_pair("flee", None, device, range(2000, 2020), args.max_steps)
    print(f"device={device} obs={OBS_DIM_RICH} spawner_actions={SPAWNER_ACTIONS} flee|scripted={flee_vs_scripted:.2f}s", flush=True)

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    update = 0
    best_vs_scripted = -1.0

    modes = [
        "self",          # newP x newS
        "self",
        "p_vs_scripted", # train player vs scripted
        "s_vs_flee",     # train spawner vs flee
        "s_vs_pool",     # train spawner vs frozen pool player
        "p_vs_self",     # already covered
    ]

    with args.log.open("w") as logf:
        while time.time() < deadline:
            ptrajs, strajs, survs = [], [], []
            for i in range(args.rollouts):
                env = Qrokkun26Env(seed=args.seed + update * 200 + i)
                mode = modes[i % len(modes)]
                if mode == "self":
                    p_net, s_net, tp, ts = player, spawner, True, True
                elif mode == "p_vs_scripted":
                    p_net, s_net, tp, ts = player, None, True, False
                elif mode == "s_vs_flee":
                    p_net, s_net, tp, ts = "flee", spawner, False, True
                elif mode == "s_vs_pool":
                    name, p_net = random.choice(pool_players)
                    s_net, tp, ts = spawner, False, True
                else:
                    p_net, s_net, tp, ts = player, spawner, True, True

                pt, st, surv = run_episode(
                    env, p_net, s_net, device, args.max_steps,
                    sample=True, train_player=tp, train_spawner=ts,
                    temp_p=args.temp_p, temp_s=args.temp_s, rng_jitter=True,
                )
                if tp:
                    ptrajs.append(pt)
                if ts:
                    strajs.append(st)
                survs.append(surv)

            ppo_update(player, opt_p, ptrajs, device, args.clip, args.ppo_epochs, args.minibatch,
                       args.entropy_p, 0.5, args.gamma, args.lam)
            ppo_update(spawner, opt_s, strajs, device, args.clip, args.ppo_epochs, args.minibatch,
                       args.entropy_s, 0.5, args.gamma, args.lam)

            row = {"update": update, "surv_mean": sum(survs) / len(survs), "wall_h": (time.time() - t0) / 3600}
            if update % 5 == 0:
                vs_scripted = eval_pair(player, None, device, range(2100, 2115), args.max_steps)
                new_vs_new = eval_pair(player, spawner, device, range(2200, 2215), args.max_steps)
                flee_vs_new = eval_pair("flee", spawner, device, range(2300, 2315), args.max_steps)
                row.update({"newP|scripted": vs_scripted, "new|new": new_vs_new, "flee|newS": flee_vs_new})
                if vs_scripted > best_vs_scripted:
                    best_vs_scripted = vs_scripted
                torch.save({"state_dict": player.state_dict(), "hidden": args.hidden, "algo": "both-v2-player",
                            "eval_vs_scripted": vs_scripted, "update": update}, args.out_player)
                torch.save({"state_dict": spawner.state_dict(), "hidden": args.hidden, "algo": "both-v2-spawner",
                            "n_actions": SPAWNER_ACTIONS, "eval_flee_vs_newS": flee_vs_new, "update": update},
                           args.out_spawner)
                status = {
                    "update": update,
                    "wall_hours": (time.time() - t0) / 3600,
                    "newP_vs_scripted": vs_scripted,
                    "best_newP_vs_scripted": best_vs_scripted,
                    "new_vs_new": new_vs_new,
                    "flee_vs_newS": flee_vs_new,
                    "flee_vs_scripted_ref": flee_vs_scripted,
                    "spawner_actions": SPAWNER_ACTIONS,
                    "done": False,
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"upd={update:5d} train={row['surv_mean']:5.2f} "
                    f"newP|scripted={vs_scripted:.2f} new|new={new_vs_new:.2f} "
                    f"flee|newS={flee_vs_new:.2f} (flee|scripted={flee_vs_scripted:.2f}) "
                    f"wall={status['wall_hours']:.2f}h",
                    flush=True,
                )
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            update += 1

    seeds = range(3000, 3040)
    cmp = {
        "newP_vs_scripted": eval_pair(player, None, device, seeds, args.max_steps),
        "new_vs_new": eval_pair(player, spawner, device, seeds, args.max_steps),
        "flee_vs_newS": eval_pair("flee", spawner, device, seeds, args.max_steps),
        "flee_vs_scripted": eval_pair("flee", None, device, seeds, args.max_steps),
        "n_seeds": 40,
        "wall_hours": (time.time() - t0) / 3600,
        "updates": update,
        "spawner_actions": SPAWNER_ACTIONS,
        "notes": "v3: rich 64-bullet obs for both, kind control, env jitter, flee/scripted pool, no early-kill penalty",
    }
    torch.save({"state_dict": player.state_dict(), "hidden": args.hidden, "algo": "both-v2-player",
                "eval_vs_scripted": cmp["newP_vs_scripted"], "update": update}, args.out_player)
    torch.save({"state_dict": spawner.state_dict(), "hidden": args.hidden, "algo": "both-v2-spawner",
                "n_actions": SPAWNER_ACTIONS, "eval_flee_vs_newS": cmp["flee_vs_newS"], "update": update},
               args.out_spawner)
    args.compare.write_text(json.dumps(cmp, indent=2) + "\n")
    args.status.write_text(json.dumps({**cmp, "done": True}, indent=2) + "\n")
    print(json.dumps(cmp, indent=2), flush=True)


if __name__ == "__main__":
    main()
