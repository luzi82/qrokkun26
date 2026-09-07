#!/usr/bin/env python3
"""train-both v4: masked attention obs, continuous spawner (birth+aim) + kind.

Canonical nets: agents.player_v4 + agents.spawner_v4.
No offline replay; diversity via sample/entropy, env jitter, flee/scripted pool.
"""

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

from qrokkun_env import constants as C
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import AIM_SCALE, SpawnerV4, spawn_continuous
from qrokkun_env.agents.obs_v4 import encode_obs
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Qrokkun26Env, _move_toward, _spawn_interval
from qrokkun_env.godot_rng import f32
from qrokkun_env.policies import FleeNearestBullet

PlayerAC = PlayerV4
SpawnerAC = SpawnerV4


@dataclass
class Traj:
    # store flat lists; player uses int actions; spawner stores cont+kind
    player: list = field(default_factory=list)
    bullets: list = field(default_factory=list)
    pad: list = field(default_factory=list)
    actions: list = field(default_factory=list)  # player int OR spawner dict
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


def shaped_player_r(env: Qrokkun26Env, done: bool) -> float:
    r = C.DT
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.002 * min(math.sqrt(d2) / 80.0, 1.0)
    if done and env.dead:
        r -= 1.0
    return r


def shaped_spawner_r(env: Qrokkun26Env, done: bool, is_spawn: bool) -> float:
    r = -0.02 if is_spawn else 0.0
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.01 * (1.0 - min(math.sqrt(d2) / 100.0, 1.0))
    if done and env.dead:
        r += 5.0
    elif done and not env.dead:
        r -= 2.0
    return r


def _pack_obs(env, device):
    p, b, m = encode_obs(env)
    return (
        torch.tensor(p, dtype=torch.float32, device=device),
        torch.tensor(b, dtype=torch.float32, device=device),
        torch.tensor(m, dtype=torch.bool, device=device),
        p, b, m,
    )


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


@torch.no_grad()
def act_player(net, env, device, sample: bool, temp: float = 1.0):
    pt, bt, mt, p, b, m = _pack_obs(env, device)
    dist, value = net(pt.unsqueeze(0), bt.unsqueeze(0), mt.unsqueeze(0))
    if temp != 1.0:
        from torch.distributions import Categorical
        dist = Categorical(logits=dist.logits / max(temp, 1e-6))
    a = dist.sample() if sample else dist.probs.argmax(dim=-1)
    return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), p, b, m


@torch.no_grad()
def act_spawner(net, env, device, sample: bool, temp: float = 1.0):
    from torch.distributions import Categorical, Normal
    pt, bt, mt, p, b, m = _pack_obs(env, device)
    birth, aim, kind, value = net(pt.unsqueeze(0), bt.unsqueeze(0), mt.unsqueeze(0))
    # Temperature scales continuous exploration std and kind logits.
    t = max(float(temp), 1e-6)
    birth = Normal(birth.mean, birth.stddev * t)
    aim = Normal(aim.mean, aim.stddev * t)
    kind = Categorical(logits=kind.logits / t)
    if sample:
        bv = birth.sample()
        av = aim.sample()
        kv = kind.sample()
    else:
        bv = birth.mean
        av = aim.mean
        kv = kind.probs.argmax(dim=-1)
    lp = birth.log_prob(bv).sum(-1) + aim.log_prob(av).sum(-1) + kind.log_prob(kv)
    action = {
        "birth": (float(bv[0, 0]), float(bv[0, 1])),
        "aim": (float(av[0, 0]), float(av[0, 1])),
        "kind": int(kv.item()),
    }
    return action, float(lp.item()), float(value.item()), p, b, m


def run_episode(
    env, player, spawner, device, max_steps, *, sample, train_player, train_spawner, temp_p, temp_s, rng_jitter
):
    ptraj, straj = Traj(), Traj()
    env.reset()
    flee = FleeNearestBullet() if player == "flee" else None

    for _ in range(max_steps):
        if spawner is None:
            if player == "flee":
                a = ACTIONS.index(flee.act(env))
                lp = v = 0.0
                p, b, m = encode_obs(env)
            else:
                a, lp, v, p, b, m = act_player(player, env, device, sample, temp_p)
            if train_player:
                ptraj.player.append(p); ptraj.bullets.append(b); ptraj.pad.append(m)
                ptraj.actions.append(a); ptraj.log_probs.append(lp); ptraj.values.append(v)
                ptraj.rewards.append(0.0); ptraj.dones.append(False)
            _o, _r, done, _ = env.step(a)
            if train_player and ptraj.rewards:
                ptraj.rewards[-1] = shaped_player_r(env, done)
                ptraj.dones[-1] = done
            if done:
                return ptraj, straj, env.elapsed
            continue

        env.elapsed += env.dt
        env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed)
        spawns = 0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            act, lp, v, p, b, m = act_spawner(spawner, env, device, sample, temp_s)
            spawn_continuous(env, act["birth"], act["aim"], act["kind"], rng_jitter=rng_jitter)
            spawns += 1
            if train_spawner:
                straj.player.append(p); straj.bullets.append(b); straj.pad.append(m)
                straj.actions.append(act); straj.log_probs.append(lp); straj.values.append(v)
                straj.rewards.append(shaped_spawner_r(env, False, True)); straj.dones.append(False)
            thr, p_double = 8.0, 0.0
            if env.elapsed > 18.0:
                p_double = 0.20
            elif env.elapsed > thr:
                p_double = 0.10
            if spawns == 1 and p_double > 0 and env.rng.randf() < p_double:
                env.spawn_acc += interval
            interval = _spawn_interval(env.elapsed)

        if player == "flee":
            a = ACTIONS.index(flee.act(env))
            lp = v = 0.0
            p, b, m = encode_obs(env)
        else:
            a, lp, v, p, b, m = act_player(player, env, device, sample, temp_p)
        if train_player:
            ptraj.player.append(p); ptraj.bullets.append(b); ptraj.pad.append(m)
            ptraj.actions.append(a); ptraj.log_probs.append(lp); ptraj.values.append(v)
            ptraj.rewards.append(0.0); ptraj.dones.append(False)

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


def ppo_update_player(net, opt, trajs, device, clip, epochs, minibatch, entropy_coef, value_coef, gamma, lam):
    packs = []
    for tr in trajs:
        if not tr.rewards:
            continue
        adv, ret = gae(tr.rewards, tr.values, tr.dones, gamma, lam, device)
        packs.append((
            torch.tensor(tr.player, dtype=torch.float32, device=device),
            torch.tensor(tr.bullets, dtype=torch.float32, device=device),
            torch.tensor(tr.pad, dtype=torch.bool, device=device),
            torch.tensor(tr.actions, dtype=torch.int64, device=device),
            torch.tensor([float(x) for x in tr.log_probs], device=device),
            adv, ret,
        ))
    if not packs:
        return 0
    P = torch.cat([p[0] for p in packs]); B = torch.cat([p[1] for p in packs])
    M = torch.cat([p[2] for p in packs]); A = torch.cat([p[3] for p in packs])
    old = torch.cat([p[4] for p in packs])
    adv = torch.cat([p[5] for p in packs]); ret = torch.cat([p[6] for p in packs])
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, minibatch):
            mb = perm[s : s + minibatch]
            dist, value = net(P[mb], B[mb], M[mb])
            lp = dist.log_prob(A[mb])
            ratio = torch.exp(lp - old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            loss = -torch.min(surr1, surr2).mean() + value_coef * F.mse_loss(value, ret[mb]) - entropy_coef * dist.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return N


def ppo_update_spawner(net, opt, trajs, device, clip, epochs, minibatch, entropy_coef, value_coef, gamma, lam):
    packs = []
    for tr in trajs:
        if not tr.rewards:
            continue
        adv, ret = gae(tr.rewards, tr.values, tr.dones, gamma, lam, device)
        births = torch.tensor([a["birth"] for a in tr.actions], dtype=torch.float32, device=device)
        aims = torch.tensor([a["aim"] for a in tr.actions], dtype=torch.float32, device=device)
        kinds = torch.tensor([a["kind"] for a in tr.actions], dtype=torch.int64, device=device)
        packs.append((
            torch.tensor(tr.player, dtype=torch.float32, device=device),
            torch.tensor(tr.bullets, dtype=torch.float32, device=device),
            torch.tensor(tr.pad, dtype=torch.bool, device=device),
            births, aims, kinds,
            torch.tensor([float(x) for x in tr.log_probs], device=device),
            adv, ret,
        ))
    if not packs:
        return 0
    P = torch.cat([p[0] for p in packs]); B = torch.cat([p[1] for p in packs]); M = torch.cat([p[2] for p in packs])
    birth_a = torch.cat([p[3] for p in packs]); aim_a = torch.cat([p[4] for p in packs]); kind_a = torch.cat([p[5] for p in packs])
    old = torch.cat([p[6] for p in packs]); adv = torch.cat([p[7] for p in packs]); ret = torch.cat([p[8] for p in packs])
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, minibatch):
            mb = perm[s : s + minibatch]
            birth, aim, kind, value = net(P[mb], B[mb], M[mb])
            lp = birth.log_prob(birth_a[mb]).sum(-1) + aim.log_prob(aim_a[mb]).sum(-1) + kind.log_prob(kind_a[mb])
            ent = birth.entropy().sum(-1).mean() + aim.entropy().sum(-1).mean() + kind.entropy().mean()
            ratio = torch.exp(lp - old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            loss = -torch.min(surr1, surr2).mean() + value_coef * F.mse_loss(value, ret[mb]) - entropy_coef * ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    return N


@torch.no_grad()
def eval_pair(
    player,
    spawner,
    device,
    seeds,
    max_steps,
    *,
    sample_policy: bool = False,
    rng_jitter: bool = False,
) -> float:
    """Mean survival over seeds. Defaults = det_policy+det_env (v4.2 compat)."""
    from qrokkun_env.eval_modes import eval_survival_times

    times = eval_survival_times(
        run_episode,
        lambda seed: Qrokkun26Env(seed=seed),
        player,
        spawner,
        device,
        seeds,
        max_steps,
        sample_policy=sample_policy,
        rng_jitter=rng_jitter,
    )
    return sum(times) / max(len(times), 1)


@torch.no_grad()
def eval_pair_stats(
    player,
    spawner,
    device,
    seeds,
    max_steps,
    *,
    sample_policy: bool = False,
    rng_jitter: bool = False,
) -> dict:
    """Mean/median/std/n (+ times) for paired-seed reports."""
    from qrokkun_env.eval_modes import eval_survival_times, summarize_times

    times = eval_survival_times(
        run_episode,
        lambda seed: Qrokkun26Env(seed=seed),
        player,
        spawner,
        device,
        seeds,
        max_steps,
        sample_policy=sample_policy,
        rng_jitter=rng_jitter,
    )
    return summarize_times(times)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--out-player", type=Path, default=Path("runs/both_v4_player.pt"))
    ap.add_argument("--out-spawner", type=Path, default=Path("runs/both_v4_spawner.pt"))
    ap.add_argument("--status", type=Path, default=Path("runs/both_v4_status.json"))
    ap.add_argument("--compare", type=Path, default=Path("runs/both_v4_compare.json"))
    ap.add_argument("--log", type=Path, default=Path("runs/both_v4.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument(
        "--temp-p",
        type=float,
        default=1.0,
        help="Player action temperature (default 1.0). Exploration uses sampling+entropy, not temp!=1. Non-1.0 is rejected.",
    )
    ap.add_argument(
        "--temp-s",
        type=float,
        default=1.0,
        help="Spawner action temperature (default 1.0). Exploration uses sampling+entropy, not temp!=1. Non-1.0 is rejected.",
    )
    ap.add_argument("--entropy-p", type=float, default=0.04)
    ap.add_argument("--entropy-s", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=60 * 70)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument(
        "--corner-probe",
        type=Path,
        nargs="?",
        const=Path("runs/both_v4_corner_probe.json"),
        default=None,
        help="If set, run locked-P corner probe at end and write JSON (default path if flag alone).",
    )
    return ap


def reject_non_unit_temp(args: argparse.Namespace) -> None:
    """v4.2: keep CLI names but refuse any temperature other than 1.0."""
    if float(args.temp_p) != 1.0 or float(args.temp_s) != 1.0:
        raise SystemExit(
            "error: non-1.0 temperature is not supported "
            f"(got --temp-p={args.temp_p}, --temp-s={args.temp_s}). "
            "Exploration uses sampling+entropy, not temp!=1."
        )


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    reject_non_unit_temp(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    player = PlayerV4(d_model=args.d_model, hidden=args.hidden).to(device)
    spawner = SpawnerV4(d_model=args.d_model, hidden=args.hidden).to(device)
    opt_p = torch.optim.Adam(player.parameters(), lr=args.lr)
    opt_s = torch.optim.Adam(spawner.parameters(), lr=args.lr)
    args.out_player.parent.mkdir(parents=True, exist_ok=True)

    flee_vs_scripted = eval_pair("flee", None, device, range(2000, 2015), args.max_steps)
    print(
        f"device={device} aim_scale={AIM_SCALE} flee|scripted={flee_vs_scripted:.2f}s "
        f"player_params={sum(p.numel() for p in player.parameters())} "
        f"spawner_params={sum(p.numel() for p in spawner.parameters())}",
        flush=True,
    )

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    update = 0
    best_vs_scripted = -1.0
    modes = ["self", "self", "p_vs_scripted", "s_vs_flee", "self", "s_vs_flee"]

    with args.log.open("w") as logf:
        while time.time() < deadline:
            ptrajs, strajs, survs = [], [], []
            for i in range(args.rollouts):
                env = Qrokkun26Env(seed=args.seed + update * 200 + i)
                mode = modes[i % len(modes)]
                if mode == "p_vs_scripted":
                    p_net, s_net, tp, ts = player, None, True, False
                elif mode == "s_vs_flee":
                    p_net, s_net, tp, ts = "flee", spawner, False, True
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

            ppo_update_player(player, opt_p, ptrajs, device, args.clip, args.ppo_epochs, args.minibatch,
                              args.entropy_p, 0.5, args.gamma, args.lam)
            ppo_update_spawner(spawner, opt_s, strajs, device, args.clip, args.ppo_epochs, args.minibatch,
                               args.entropy_s, 0.5, args.gamma, args.lam)

            row = {"update": update, "surv_mean": sum(survs) / len(survs), "wall_h": (time.time() - t0) / 3600}
            if update % 5 == 0:
                # Mode 1 (det_det) — old keys; Mode 2 (det_stoch) for new×new / flee×newS.
                vs_scripted = eval_pair(player, None, device, range(2100, 2112), args.max_steps)
                new_vs_new = eval_pair(player, spawner, device, range(2200, 2212), args.max_steps)
                flee_vs_new = eval_pair("flee", spawner, device, range(2300, 2312), args.max_steps)
                new_vs_new_ds = eval_pair(
                    player, spawner, device, range(2200, 2212), args.max_steps,
                    sample_policy=False, rng_jitter=True,
                )
                flee_vs_new_ds = eval_pair(
                    "flee", spawner, device, range(2300, 2312), args.max_steps,
                    sample_policy=False, rng_jitter=True,
                )
                row.update({
                    "newP|scripted": vs_scripted,
                    "new|new": new_vs_new,
                    "flee|newS": flee_vs_new,
                    "new|new_det_stoch": new_vs_new_ds,
                    "flee|newS_det_stoch": flee_vs_new_ds,
                })
                if vs_scripted > best_vs_scripted:
                    best_vs_scripted = vs_scripted
                torch.save({"state_dict": player.state_dict(), "d_model": args.d_model, "hidden": args.hidden,
                            "algo": "both-v4-player", "eval_vs_scripted": vs_scripted, "update": update}, args.out_player)
                torch.save({"state_dict": spawner.state_dict(), "d_model": args.d_model, "hidden": args.hidden,
                            "algo": "both-v4-spawner", "aim_scale": AIM_SCALE, "eval_flee_vs_newS": flee_vs_new_ds,
                            "update": update}, args.out_spawner)
                status = {
                    "update": update,
                    "wall_hours": (time.time() - t0) / 3600,
                    # Compat keys = det_policy+det_env
                    "newP_vs_scripted": vs_scripted,
                    "best_newP_vs_scripted": best_vs_scripted,
                    "new_vs_new": new_vs_new,
                    "flee_vs_newS": flee_vs_new,
                    "flee_vs_scripted_ref": flee_vs_scripted,
                    # Distinct mode-suffixed fields (v4.3)
                    "newP_vs_scripted_det_det": vs_scripted,
                    "new_vs_new_det_det": new_vs_new,
                    "new_vs_new_det_stoch": new_vs_new_ds,
                    "flee_vs_newS_det_det": flee_vs_new,
                    "flee_vs_newS_det_stoch": flee_vs_new_ds,
                    "aim_scale": AIM_SCALE,
                    "eval_note": "new×new is observation only; prefer newP|scripted + flee|newS_det_stoch",
                    "done": False,
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"upd={update:5d} train={row['surv_mean']:5.2f} "
                    f"newP|scripted={vs_scripted:.2f} new|new={new_vs_new:.2f}/{new_vs_new_ds:.2f} "
                    f"flee|newS={flee_vs_new:.2f}/{flee_vs_new_ds:.2f} "
                    f"(flee|scripted={flee_vs_scripted:.2f}) "
                    f"wall={status['wall_hours']:.2f}h",
                    flush=True,
                )
            logf.write(json.dumps(row) + "\n"); logf.flush()
            update += 1

    from qrokkun_env.eval_modes import PAIRED_EVAL_SEEDS, metric_key

    seeds = PAIRED_EVAL_SEEDS
    newP_dd = eval_pair_stats(player, None, device, seeds, args.max_steps)
    new_dd = eval_pair_stats(player, spawner, device, seeds, args.max_steps)
    new_ds = eval_pair_stats(
        player, spawner, device, seeds, args.max_steps, sample_policy=False, rng_jitter=True,
    )
    flee_dd = eval_pair_stats("flee", spawner, device, seeds, args.max_steps)
    flee_ds = eval_pair_stats(
        "flee", spawner, device, seeds, args.max_steps, sample_policy=False, rng_jitter=True,
    )
    flee_sc = eval_pair_stats("flee", None, device, seeds, args.max_steps)
    cmp = {
        # Compat means (det_det)
        "newP_vs_scripted": newP_dd["mean"],
        "new_vs_new": new_dd["mean"],
        "flee_vs_newS": flee_dd["mean"],
        "flee_vs_scripted": flee_sc["mean"],
        # Distinct mode fields
        metric_key("newP_vs_scripted", False, False): newP_dd["mean"],
        metric_key("new_vs_new", False, False): new_dd["mean"],
        metric_key("new_vs_new", False, True): new_ds["mean"],
        metric_key("flee_vs_newS", False, False): flee_dd["mean"],
        metric_key("flee_vs_newS", False, True): flee_ds["mean"],
        metric_key("flee_vs_scripted", False, False): flee_sc["mean"],
        "by_mode": {
            "newP_vs_scripted_det_det": newP_dd,
            "new_vs_new_det_det": new_dd,
            "new_vs_new_det_stoch": new_ds,
            "flee_vs_newS_det_det": flee_dd,
            "flee_vs_newS_det_stoch": flee_ds,
            "flee_vs_scripted_det_det": flee_sc,
        },
        "paired_seeds": list(seeds),
        "n_seeds": len(seeds),
        "wall_hours": (time.time() - t0) / 3600,
        "updates": update,
        "aim_scale": AIM_SCALE,
        "notes": "v4.3 eval modes; new×new not sole selection metric; paired seeds",
    }
    # Drop bulky times from compare JSON by_mode for readability (keep summary stats).
    for _k, block in cmp["by_mode"].items():
        block.pop("times", None)
    torch.save({"state_dict": player.state_dict(), "d_model": args.d_model, "hidden": args.hidden,
                "algo": "both-v4-player", "eval_vs_scripted": cmp["newP_vs_scripted"], "update": update}, args.out_player)
    torch.save({"state_dict": spawner.state_dict(), "d_model": args.d_model, "hidden": args.hidden,
                "algo": "both-v4-spawner", "aim_scale": AIM_SCALE,
                "eval_flee_vs_newS": cmp[metric_key("flee_vs_newS", False, True)],
                "update": update}, args.out_spawner)
    args.compare.write_text(json.dumps(cmp, indent=2) + "\n")
    args.status.write_text(json.dumps({**cmp, "done": True}, indent=2) + "\n")
    print(json.dumps(cmp, indent=2), flush=True)

    if args.corner_probe is not None:
        from qrokkun_env.corner_probe import run_corner_probe

        probe = run_corner_probe(spawner, device, max_steps=min(args.max_steps, 60 * 40))
        args.corner_probe.parent.mkdir(parents=True, exist_ok=True)
        args.corner_probe.write_text(json.dumps(probe, indent=2) + "\n")
        print(f"corner_probe -> {args.corner_probe}", flush=True)


if __name__ == "__main__":
    main()
