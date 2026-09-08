#!/usr/bin/env python3
"""train-both v4: masked attention obs, continuous spawner (birth+aim) + kind.

Canonical nets: agents.player_v4 + agents.spawner_v4.
No offline replay; diversity via sample/entropy, env jitter, flee/scripted pool.

v4.4: terminated vs truncated + final-value bootstrap (last_value into GAE).
v4.5: per-transition delta_frames; gamma_t = gamma_frame ** delta_frames.
v4.6: PPO diagnostics + latest/best/snapshot ckpts (P/S separate selection).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qrokkun_env.train.checkpoints_v4 import (
    CheckpointManager,
    default_best_path,
    pack_player_ckpt,
    pack_spawner_ckpt,
    player_selection_key,
    player_selection_score,
    spawner_selection_key,
    spawner_selection_score,
)

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
    # v4.4: true death only (truncation keeps False so GAE can bootstrap)
    terminateds: list = field(default_factory=list)
    # v4.5: frames until next decision / episode end (Player usually 1)
    delta_frames: list = field(default_factory=list)
    last_value: float = 0.0  # v4.4: 0 if terminated else V(final_obs)
    terminated: bool = False
    truncated: bool = False


def gae(
    rewards,
    values,
    terminateds,
    gamma,
    lam,
    device,
    *,
    last_value: float = 0.0,
    delta_frames=None,
):
    """GAE with optional final-value bootstrap and variable-time discount.

    gamma / lam are **per-frame**. For each step t:
      gamma_t = gamma ** delta_frames_t
      lambda_t = lam ** delta_frames_t
    terminateds[t] True → no bootstrap through that step (true death).
    Truncation: terminateds[-1]=False and last_value=V(final_obs).
    """
    n = len(rewards)
    if delta_frames is None:
        dfs = [1] * n
    else:
        dfs = [max(int(d), 0) for d in delta_frames]
        if len(dfs) != n:
            raise ValueError(f"delta_frames length {len(dfs)} != rewards length {n}")
    if len(terminateds) != n:
        raise ValueError(f"terminateds length {len(terminateds)} != rewards length {n}")
    vals = [float(v) for v in values] + [float(last_value)]
    adv, gae_v = [], 0.0
    for t in reversed(range(n)):
        mask = 0.0 if terminateds[t] else 1.0
        g_t = gamma ** dfs[t]
        l_t = lam ** dfs[t]
        delta = rewards[t] + g_t * vals[t + 1] * mask - vals[t]
        gae_v = delta + g_t * l_t * mask * gae_v
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


def shaped_spawner_r(env: Qrokkun26Env, terminated: bool, is_spawn: bool) -> float:
    """Spawner shaping. Terminal +5 only on true death — never on time-limit truncation."""
    r = -0.02 if is_spawn else 0.0
    if env.bullets:
        d2 = min((b.x - env.px) ** 2 + (b.y - env.py) ** 2 for b in env.bullets)
        r += 0.01 * (1.0 - min(math.sqrt(d2) / 100.0, 1.0))
    if terminated and env.dead:
        r += 5.0
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


@torch.no_grad()
def player_value(net, env, device) -> float:
    pt, bt, mt, _, _, _ = _pack_obs(env, device)
    _dist, value = net(pt.unsqueeze(0), bt.unsqueeze(0), mt.unsqueeze(0))
    return float(value.item())


@torch.no_grad()
def spawner_value(net, env, device) -> float:
    pt, bt, mt, _, _, _ = _pack_obs(env, device)
    _birth, _aim, _kind, value = net(pt.unsqueeze(0), bt.unsqueeze(0), mt.unsqueeze(0))
    return float(value.item())


def _finalize_player_end(ptraj: Traj, env, player, device, *, terminated: bool) -> None:
    if not ptraj.rewards:
        return
    if terminated:
        ptraj.terminateds[-1] = True
        ptraj.terminated = True
        ptraj.truncated = False
        ptraj.last_value = 0.0
    else:
        ptraj.terminateds[-1] = False
        ptraj.terminated = False
        ptraj.truncated = True
        if player not in (None, "flee"):
            ptraj.last_value = player_value(player, env, device)
        else:
            ptraj.last_value = 0.0


def _finalize_spawner_end(straj: Traj, env, spawner, device, *, terminated: bool, frames_since_spawn: int) -> None:
    if not straj.rewards:
        return
    straj.delta_frames[-1] = max(int(frames_since_spawn), 0)
    if terminated:
        straj.rewards[-1] += shaped_spawner_r(env, True, False)
        straj.terminateds[-1] = True
        straj.terminated = True
        straj.truncated = False
        straj.last_value = 0.0
    else:
        # Time-limit: bootstrap V(final); do NOT apply fail-to-kill penalty.
        straj.terminateds[-1] = False
        straj.terminated = False
        straj.truncated = True
        if spawner is not None:
            straj.last_value = spawner_value(spawner, env, device)
        else:
            straj.last_value = 0.0


def run_episode(
    env, player, spawner, device, max_steps, *, sample, train_player, train_spawner, temp_p, temp_s, rng_jitter
):
    ptraj, straj = Traj(), Traj()
    env.reset()
    flee = FleeNearestBullet() if player == "flee" else None
    frames_since_spawn = 0

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
                ptraj.rewards.append(0.0); ptraj.terminateds.append(False); ptraj.delta_frames.append(1)
            _o, _r, done, _ = env.step(a)
            if train_player and ptraj.rewards:
                ptraj.rewards[-1] = shaped_player_r(env, done)
            if done:
                if train_player:
                    _finalize_player_end(ptraj, env, player, device, terminated=True)
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
                if straj.rewards:
                    straj.delta_frames[-1] = frames_since_spawn
                straj.player.append(p); straj.bullets.append(b); straj.pad.append(m)
                straj.actions.append(act); straj.log_probs.append(lp); straj.values.append(v)
                straj.rewards.append(shaped_spawner_r(env, False, True))
                straj.terminateds.append(False)
                straj.delta_frames.append(0)  # filled at next spawn or episode end
                frames_since_spawn = 0
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
            ptraj.rewards.append(0.0); ptraj.terminateds.append(False); ptraj.delta_frames.append(1)

        apply_player_action(env, a)
        env._integrate_bullets()
        done = env._check_hit()
        if done:
            env.dead = True
        if train_player and ptraj.rewards:
            ptraj.rewards[-1] = shaped_player_r(env, done)
        # One physics frame elapsed after any spawn decisions this step.
        frames_since_spawn += 1
        if done:
            if train_player:
                _finalize_player_end(ptraj, env, player, device, terminated=True)
            if train_spawner:
                _finalize_spawner_end(
                    straj, env, spawner, device, terminated=True, frames_since_spawn=frames_since_spawn
                )
            return ptraj, straj, env.elapsed

    # max_steps time-limit → truncated (bootstrap), not terminated.
    if train_player:
        _finalize_player_end(ptraj, env, player, device, terminated=False)
    if train_spawner:
        _finalize_spawner_end(
            straj, env, spawner, device, terminated=False, frames_since_spawn=frames_since_spawn
        )
    return ptraj, straj, env.elapsed


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true); 0 if y_true is constant."""
    y_pred = y_pred.detach().float().view(-1)
    y_true = y_true.detach().float().view(-1)
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y.item()) < 1e-8:
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred, unbiased=False) / (var_y + 1e-8)).item())


def traj_end_rates(trajs) -> dict[str, float]:
    """Fraction of non-empty trajs that terminated vs truncated."""
    nonempty = [t for t in trajs if t.rewards]
    n = len(nonempty)
    if n == 0:
        return {"termination_rate": 0.0, "truncation_rate": 0.0, "n_traj": 0}
    n_term = sum(1 for t in nonempty if t.terminated)
    n_trunc = sum(1 for t in nonempty if t.truncated)
    return {
        "termination_rate": n_term / n,
        "truncation_rate": n_trunc / n,
        "n_traj": n,
    }


def ppo_update_player(net, opt, trajs, device, clip, epochs, minibatch, entropy_coef, value_coef, gamma, lam):
    """PPO update for Player. Returns (n_samples, diagnostics dict)."""
    packs = []
    for tr in trajs:
        if not tr.rewards:
            continue
        adv, ret = gae(
            tr.rewards,
            tr.values,
            tr.terminateds,
            gamma,
            lam,
            device,
            last_value=tr.last_value,
            delta_frames=tr.delta_frames,
        )
        packs.append((
            torch.tensor(tr.player, dtype=torch.float32, device=device),
            torch.tensor(tr.bullets, dtype=torch.float32, device=device),
            torch.tensor(tr.pad, dtype=torch.bool, device=device),
            torch.tensor(tr.actions, dtype=torch.int64, device=device),
            torch.tensor([float(x) for x in tr.log_probs], device=device),
            adv, ret,
        ))
    empty: dict[str, Any] = {
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "ratio_mean": 1.0,
        "ratio_std": 0.0,
        "entropy": 0.0,
        "explained_variance": 0.0,
        "n": 0,
    }
    if not packs:
        return 0, empty
    P = torch.cat([p[0] for p in packs]); B = torch.cat([p[1] for p in packs])
    M = torch.cat([p[2] for p in packs]); A = torch.cat([p[3] for p in packs])
    old = torch.cat([p[4] for p in packs])
    adv = torch.cat([p[5] for p in packs]); ret = torch.cat([p[6] for p in packs])
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = P.shape[0]

    # Sanity diagnostics at update start (before any optimizer step).
    with torch.no_grad():
        dist0, value0 = net(P, B, M)
        lp0 = dist0.log_prob(A)
        ratio0 = torch.exp(lp0 - old)
        ratio_mean = float(ratio0.mean().item())
        ratio_std = float(ratio0.std(unbiased=False).item())
        entropy0 = float(dist0.entropy().mean().item())
        ev0 = explained_variance(value0, ret)

    kl_acc = []
    clip_acc = []
    ent_acc = []
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, minibatch):
            mb = perm[s : s + minibatch]
            dist, value = net(P[mb], B[mb], M[mb])
            lp = dist.log_prob(A[mb])
            ratio = torch.exp(lp - old[mb])
            with torch.no_grad():
                log_ratio = lp - old[mb]
                # Schulman approx KL; also clipfrac
                kl_acc.append(float(((ratio - 1.0) - log_ratio).mean().item()))
                clip_acc.append(float(((ratio - 1.0).abs() > clip).float().mean().item()))
                ent_acc.append(float(dist.entropy().mean().item()))
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            loss = -torch.min(surr1, surr2).mean() + value_coef * F.mse_loss(value, ret[mb]) - entropy_coef * dist.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    diag = {
        "approx_kl": sum(kl_acc) / max(len(kl_acc), 1),
        "clipfrac": sum(clip_acc) / max(len(clip_acc), 1),
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "entropy": entropy0 if not ent_acc else sum(ent_acc) / len(ent_acc),
        "explained_variance": ev0,
        "n": int(N),
    }
    return N, diag


def ppo_update_spawner(net, opt, trajs, device, clip, epochs, minibatch, entropy_coef, value_coef, gamma, lam):
    """PPO update for Spawner. Returns (n_samples, diagnostics dict)."""
    packs = []
    for tr in trajs:
        if not tr.rewards:
            continue
        adv, ret = gae(
            tr.rewards,
            tr.values,
            tr.terminateds,
            gamma,
            lam,
            device,
            last_value=tr.last_value,
            delta_frames=tr.delta_frames,
        )
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
    empty: dict[str, Any] = {
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "ratio_mean": 1.0,
        "ratio_std": 0.0,
        "entropy": 0.0,
        "explained_variance": 0.0,
        "n": 0,
    }
    if not packs:
        return 0, empty
    P = torch.cat([p[0] for p in packs]); B = torch.cat([p[1] for p in packs]); M = torch.cat([p[2] for p in packs])
    birth_a = torch.cat([p[3] for p in packs]); aim_a = torch.cat([p[4] for p in packs]); kind_a = torch.cat([p[5] for p in packs])
    old = torch.cat([p[6] for p in packs]); adv = torch.cat([p[7] for p in packs]); ret = torch.cat([p[8] for p in packs])
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    N = P.shape[0]

    with torch.no_grad():
        birth0, aim0, kind0, value0 = net(P, B, M)
        lp0 = birth0.log_prob(birth_a).sum(-1) + aim0.log_prob(aim_a).sum(-1) + kind0.log_prob(kind_a)
        ratio0 = torch.exp(lp0 - old)
        ratio_mean = float(ratio0.mean().item())
        ratio_std = float(ratio0.std(unbiased=False).item())
        entropy0 = float(
            (birth0.entropy().sum(-1).mean() + aim0.entropy().sum(-1).mean() + kind0.entropy().mean()).item()
        )
        ev0 = explained_variance(value0, ret)

    kl_acc = []
    clip_acc = []
    ent_acc = []
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, minibatch):
            mb = perm[s : s + minibatch]
            birth, aim, kind, value = net(P[mb], B[mb], M[mb])
            lp = birth.log_prob(birth_a[mb]).sum(-1) + aim.log_prob(aim_a[mb]).sum(-1) + kind.log_prob(kind_a[mb])
            ent = birth.entropy().sum(-1).mean() + aim.entropy().sum(-1).mean() + kind.entropy().mean()
            ratio = torch.exp(lp - old[mb])
            with torch.no_grad():
                log_ratio = lp - old[mb]
                kl_acc.append(float(((ratio - 1.0) - log_ratio).mean().item()))
                clip_acc.append(float(((ratio - 1.0).abs() > clip).float().mean().item()))
                ent_acc.append(float(ent.item()))
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            loss = -torch.min(surr1, surr2).mean() + value_coef * F.mse_loss(value, ret[mb]) - entropy_coef * ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    diag = {
        "approx_kl": sum(kl_acc) / max(len(kl_acc), 1),
        "clipfrac": sum(clip_acc) / max(len(clip_acc), 1),
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "entropy": entropy0 if not ent_acc else sum(ent_acc) / len(ent_acc),
        "explained_variance": ev0,
        "n": int(N),
    }
    return N, diag


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
    from qrokkun_env.eval_modes import eval_survival_times, validate_eval_mode

    validate_eval_mode(sample_policy, rng_jitter)

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
    from qrokkun_env.eval_modes import eval_survival_times, summarize_times, validate_eval_mode

    validate_eval_mode(sample_policy, rng_jitter)

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
    ap.add_argument("--gamma", type=float, default=0.99, help="Per-frame discount; GAE uses gamma**delta_frames (v4.5).")
    ap.add_argument("--lam", type=float, default=0.95, help="Per-frame GAE lambda; uses lam**delta_frames (v4.5).")
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
    ap.add_argument(
        "--out-player-best",
        type=Path,
        default=None,
        help="Best player ckpt (default: <out-player>_best.pt). Selection: newP×scripted.",
    )
    ap.add_argument(
        "--out-spawner-best",
        type=Path,
        default=None,
        help="Best spawner ckpt (default: <out-spawner>_best.pt). Selection: flee×newS_det_stoch.",
    )
    ap.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="If >0, copy latest weights to snapshot-dir every N updates (0=off).",
    )
    ap.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("runs/snapshots"),
        help="Directory for periodic snapshot copies (v4.6).",
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
    out_player_best = args.out_player_best or default_best_path(args.out_player)
    out_spawner_best = args.out_spawner_best or default_best_path(args.out_spawner)
    ckpt = CheckpointManager(
        latest_player=args.out_player,
        latest_spawner=args.out_spawner,
        best_player=out_player_best,
        best_spawner=out_spawner_best,
        snapshot_dir=args.snapshot_dir if args.snapshot_every > 0 else None,
        snapshot_every=args.snapshot_every,
    )

    flee_vs_scripted = eval_pair("flee", None, device, range(2000, 2015), args.max_steps)
    print(
        f"device={device} aim_scale={AIM_SCALE} flee|scripted={flee_vs_scripted:.2f}s "
        f"player_params={sum(p.numel() for p in player.parameters())} "
        f"spawner_params={sum(p.numel() for p in spawner.parameters())} "
        f"best_p={out_player_best} best_s={out_spawner_best}",
        flush=True,
    )

    t0 = time.time()
    deadline = t0 + args.hours * 3600
    update = 0
    best_vs_scripted = -1.0
    best_flee_vs_new_ds = float("inf")
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

            _n_p, diag_p = ppo_update_player(
                player, opt_p, ptrajs, device, args.clip, args.ppo_epochs, args.minibatch,
                args.entropy_p, 0.5, args.gamma, args.lam,
            )
            _n_s, diag_s = ppo_update_spawner(
                spawner, opt_s, strajs, device, args.clip, args.ppo_epochs, args.minibatch,
                args.entropy_s, 0.5, args.gamma, args.lam,
            )
            end_p = traj_end_rates(ptrajs)
            end_s = traj_end_rates(strajs)

            row = {
                "update": update,
                "surv_mean": sum(survs) / len(survs),
                "wall_h": (time.time() - t0) / 3600,
                "ppo_p": diag_p,
                "ppo_s": diag_s,
                "term_rate_p": end_p["termination_rate"],
                "trunc_rate_p": end_p["truncation_rate"],
                "term_rate_s": end_s["termination_rate"],
                "trunc_rate_s": end_s["truncation_rate"],
            }
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
                eval_metrics = {
                    "newP_vs_scripted": vs_scripted,
                    "newP|scripted": vs_scripted,
                    "new_vs_new": new_vs_new,
                    "new_vs_new_det_det": new_vs_new,
                    "new_vs_new_det_stoch": new_vs_new_ds,
                    "flee_vs_newS": flee_vs_new,
                    "flee_vs_newS_det_det": flee_vs_new,
                    "flee_vs_newS_det_stoch": flee_vs_new_ds,
                    "flee|newS_det_stoch": flee_vs_new_ds,
                }
                # Selection scores (separate objectives; NOT new×new_det_det).
                p_sel = player_selection_score(eval_metrics)
                s_sel = spawner_selection_score(eval_metrics)
                p_key = player_selection_key(eval_metrics)
                s_key = spawner_selection_key(eval_metrics)
                p_payload = pack_player_ckpt(
                    player,
                    d_model=args.d_model,
                    hidden=args.hidden,
                    update=update,
                    eval_metrics=eval_metrics,
                    selection_score_value=p_sel,
                    selection_key=p_key,
                )
                s_payload = pack_spawner_ckpt(
                    spawner,
                    d_model=args.d_model,
                    hidden=args.hidden,
                    update=update,
                    aim_scale=AIM_SCALE,
                    eval_metrics=eval_metrics,
                    selection_score_value=s_sel,
                    selection_key=s_key,
                )
                ckpt.save_latest(p_payload, s_payload)
                saved_best_p = ckpt.maybe_save_best_player(p_sel, p_payload)
                saved_best_s = ckpt.maybe_save_best_spawner(s_sel, s_payload)
                if saved_best_s:
                    best_flee_vs_new_ds = s_sel
                ckpt.maybe_snapshot(update, p_payload, s_payload)
                status = {
                    "update": update,
                    "wall_hours": (time.time() - t0) / 3600,
                    # Compat keys = det_policy+det_env
                    "newP_vs_scripted": vs_scripted,
                    "best_newP_vs_scripted": best_vs_scripted,
                    "best_player_selection": ckpt.best_player_score,
                    "best_player_selection_key": p_key,
                    "best_spawner_selection": ckpt.best_spawner_score if ckpt.best_spawner_score < float("inf") else None,
                    "best_spawner_selection_key": s_key,
                    "best_flee_vs_newS_det_stoch": None if best_flee_vs_new_ds == float("inf") else best_flee_vs_new_ds,
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
                    "ppo_p": diag_p,
                    "ppo_s": diag_s,
                    "term_rate_p": end_p["termination_rate"],
                    "trunc_rate_p": end_p["truncation_rate"],
                    "term_rate_s": end_s["termination_rate"],
                    "trunc_rate_s": end_s["truncation_rate"],
                    "ckpt_latest_player": str(args.out_player),
                    "ckpt_latest_spawner": str(args.out_spawner),
                    "ckpt_best_player": str(out_player_best),
                    "ckpt_best_spawner": str(out_spawner_best),
                    "saved_best_player": saved_best_p,
                    "saved_best_spawner": saved_best_s,
                    "eval_note": (
                        "v4.6: best_player←newP×scripted; best_spawner←flee×newS_det_stoch (minimize); "
                        "new×new observation only — not sole selection metric"
                    ),
                    "done": False,
                }
                args.status.write_text(json.dumps(status, indent=2) + "\n")
                print(
                    f"upd={update:5d} train={row['surv_mean']:5.2f} "
                    f"newP|scripted={vs_scripted:.2f} new|new={new_vs_new:.2f}/{new_vs_new_ds:.2f} "
                    f"flee|newS={flee_vs_new:.2f}/{flee_vs_new_ds:.2f} "
                    f"(flee|scripted={flee_vs_scripted:.2f}) "
                    f"kl_p={diag_p.get('approx_kl', 0):.4f} kl_s={diag_s.get('approx_kl', 0):.4f} "
                    f"wall={status['wall_hours']:.2f}h"
                    f"{' [bestP]' if saved_best_p else ''}"
                    f"{' [bestS]' if saved_best_s else ''}",
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
    # Mode 3 (stoch_stoch): deploy-distribution sanity; fork_rng per episode.
    new_ss = eval_pair_stats(
        player, spawner, device, seeds, args.max_steps, sample_policy=True, rng_jitter=True,
    )
    flee_ss = eval_pair_stats(
        "flee", spawner, device, seeds, args.max_steps, sample_policy=True, rng_jitter=True,
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
        metric_key("new_vs_new", True, True): new_ss["mean"],
        metric_key("flee_vs_newS", False, False): flee_dd["mean"],
        metric_key("flee_vs_newS", False, True): flee_ds["mean"],
        metric_key("flee_vs_newS", True, True): flee_ss["mean"],
        metric_key("flee_vs_scripted", False, False): flee_sc["mean"],
        "by_mode": {
            "newP_vs_scripted_det_det": newP_dd,
            "new_vs_new_det_det": new_dd,
            "new_vs_new_det_stoch": new_ds,
            "new_vs_new_stoch_stoch": new_ss,
            "flee_vs_newS_det_det": flee_dd,
            "flee_vs_newS_det_stoch": flee_ds,
            "flee_vs_newS_stoch_stoch": flee_ss,
            "flee_vs_scripted_det_det": flee_sc,
        },
        "paired_seeds": list(seeds),
        "n_seeds": len(seeds),
        "wall_hours": (time.time() - t0) / 3600,
        "updates": update,
        "aim_scale": AIM_SCALE,
        "notes": "v4.6; new×new not sole selection metric; paired seeds",
    }
    # Drop bulky times from compare JSON by_mode for readability (keep summary stats).
    for _k, block in cmp["by_mode"].items():
        block.pop("times", None)
    final_metrics = {
        "newP_vs_scripted": cmp["newP_vs_scripted"],
        "newP|scripted": cmp["newP_vs_scripted"],
        "new_vs_new": cmp["new_vs_new"],
        "new_vs_new_det_det": cmp["new_vs_new"],
        "new_vs_new_det_stoch": cmp[metric_key("new_vs_new", False, True)],
        "flee_vs_newS": cmp["flee_vs_newS"],
        "flee_vs_newS_det_det": cmp["flee_vs_newS"],
        "flee_vs_newS_det_stoch": cmp[metric_key("flee_vs_newS", False, True)],
        "flee|newS_det_stoch": cmp[metric_key("flee_vs_newS", False, True)],
    }
    p_sel = player_selection_score(final_metrics)
    s_sel = spawner_selection_score(final_metrics)
    p_payload = pack_player_ckpt(
        player,
        d_model=args.d_model,
        hidden=args.hidden,
        update=update,
        eval_metrics=final_metrics,
        selection_score_value=p_sel,
        selection_key=player_selection_key(final_metrics),
    )
    s_payload = pack_spawner_ckpt(
        spawner,
        d_model=args.d_model,
        hidden=args.hidden,
        update=update,
        aim_scale=AIM_SCALE,
        eval_metrics=final_metrics,
        selection_score_value=s_sel,
        selection_key=spawner_selection_key(final_metrics),
    )
    ckpt.save_latest(p_payload, s_payload)
    ckpt.maybe_save_best_player(p_sel, p_payload)
    ckpt.maybe_save_best_spawner(s_sel, s_payload)
    cmp["selection"] = {
        "best_player_key": player_selection_key(final_metrics),
        "best_player_score": ckpt.best_player_score,
        "best_spawner_key": spawner_selection_key(final_metrics),
        "best_spawner_score": None if ckpt.best_spawner_score == float("inf") else ckpt.best_spawner_score,
        "note": "new×new_det_det is observation only; not used as sole P/S selector",
        "ckpt_best_player": str(out_player_best),
        "ckpt_best_spawner": str(out_spawner_best),
    }
    cmp["notes"] = (
        "v4.6 eval modes + separate best P/S selection; new×new not sole selection metric; paired seeds"
    )
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
