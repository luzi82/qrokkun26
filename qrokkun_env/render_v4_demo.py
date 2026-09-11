#!/usr/bin/env python3
"""Render v4 matchups to mp4: newP×scripted, new×new, flee×newS."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from qrokkun_env import constants as C
from qrokkun_env.agents.obs_v4 import MAX_BULLETS_V4, encode_obs, order_bullets_v4
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4, spawn_continuous
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Qrokkun26Env, _move_toward, _spawn_interval
from qrokkun_env.godot_rng import f32
from qrokkun_env.policies import FleeNearestBullet
from qrokkun_env.render_player_demo import BULLET_COLORS, load_sprite, paste_centered

# 4-head cross-attention overlay: one distinct outline color per head.
HEAD_COLORS = [(255, 70, 70), (70, 170, 255), (255, 215, 70), (100, 235, 150)]
ATTN_BASE_R = 3.0  # world px, before scale
ATTN_GROW = 14.0  # world px added at attn weight == 1.0
ATTN_EPS = 0.03  # skip drawing near-zero weights


@torch.no_grad()
def render_attn(player: PlayerV4 | None, spawner: SpawnerV4 | None, env: Qrokkun26Env):
    """Cross-attention [nhead, K] for the current env state, from whichever net is live.

    Uses Player's encoder if a learned Player is in the matchup, else falls back to
    Spawner's encoder (flee x newS). Returns None if neither net is present.
    """
    net = player if player is not None else spawner
    if net is None:
        return None
    p, b, m = encode_obs(env)
    pt = torch.tensor(p).unsqueeze(0)
    bt = torch.tensor(b).unsqueeze(0)
    mt = torch.tensor(m).unsqueeze(0)
    if isinstance(net, PlayerV4):
        _dist, _v, attn = net.forward_with_attn(pt, bt, mt)
    else:
        _birth, _aim, _kind, _v, attn = net.forward_with_attn(pt, bt, mt)
    return attn[0].numpy()  # [nhead, K]


@torch.no_grad()
def player_act(net: PlayerV4 | None, env: Qrokkun26Env, mode: str, flee: FleeNearestBullet) -> int:
    if mode == "flee":
        return ACTIONS.index(flee.act(env))
    p, b, m = encode_obs(env)
    dist, _v = net(
        torch.tensor(p).unsqueeze(0),
        torch.tensor(b).unsqueeze(0),
        torch.tensor(m).unsqueeze(0),
    )
    return int(dist.probs.argmax().item())


@torch.no_grad()
def spawner_act(net: SpawnerV4, env: Qrokkun26Env) -> dict:
    p, b, m = encode_obs(env)
    birth, aim, kind, _v = net(
        torch.tensor(p).unsqueeze(0),
        torch.tensor(b).unsqueeze(0),
        torch.tensor(m).unsqueeze(0),
    )
    return {
        "birth": (float(birth.mean[0, 0]), float(birth.mean[0, 1])),
        "aim": (float(aim.mean[0, 0]), float(aim.mean[0, 1])),
        "kind": int(kind.probs.argmax().item()),
    }


def apply_player(env: Qrokkun26Env, a: int) -> None:
    dx, dy = ACTION_TO_DIR[ACTIONS[a]]
    if dx or dy:
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


def render(env, scale, ps, bs, last, title, ordered_bullets=None, attn=None):
    W, H = int(C.VIEW_W) * scale, int(C.VIEW_H) * scale
    img = Image.new("RGBA", (W, H), (18, 18, 28, 255))
    d = ImageDraw.Draw(img)
    fx0, fy0 = int(C.FIELD_X * scale), int(C.FIELD_Y * scale)
    fx1, fy1 = int((C.FIELD_X + C.FIELD_W) * scale), int((C.FIELD_Y + C.FIELD_H) * scale)
    d.rectangle((fx0, fy0, fx1, fy1), fill=(28, 32, 48, 255), outline=(90, 100, 140, 255))
    for b in env.bullets:
        paste_centered(img, bs.get(b.kind, bs[0]), b.x, b.y, scale)
    if attn is not None and ordered_bullets:
        nhead = attn.shape[0]
        for idx, b in enumerate(ordered_bullets):
            if idx >= attn.shape[1]:
                break
            cx, cy = b.x * scale, b.y * scale
            for h in range(nhead):
                w = float(attn[h, idx])
                if w < ATTN_EPS:
                    continue
                r = (ATTN_BASE_R + ATTN_GROW * w) * scale
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=HEAD_COLORS[h % len(HEAD_COLORS)], width=1)
    paste_centered(img, ps.get(last, ps["idle"]), env.px, env.py, scale)
    d.rectangle((0, 0, W, 14), fill=(10, 10, 16, 230))
    d.text((6, 2), f"{title}  t={env.elapsed:5.2f}s  bullets={len(env.bullets):3d}", fill=(220, 230, 255, 255))
    if attn is not None:
        lx = W - 150
        for h, color in enumerate(HEAD_COLORS):
            hx = lx + h * 38
            d.ellipse((hx, 3, hx + 8, 11), outline=color, width=2)
            d.text((hx + 11, 2), f"H{h}", fill=color)
    return img.convert("RGB")


def run_match(
    *,
    player_mode: str,
    spawner_mode: str,
    player: PlayerV4 | None,
    spawner: SpawnerV4 | None,
    seed: int,
    max_seconds: float,
    scale: int,
    assets: Path,
    out: Path,
    title: str,
) -> float:
    env = Qrokkun26Env(seed=seed)
    env.reset(seed=seed)
    flee = FleeNearestBullet()
    player_map = {
        "idle": "player.png", "n": "player_n.png", "ne": "player_ne.png", "e": "player_e.png",
        "se": "player_se.png", "s": "player_s.png", "sw": "player_sw.png", "w": "player_w.png", "nw": "player_nw.png",
    }
    ps = {k: load_sprite(assets / fn, int(C.PLAYER_RADIUS), (240, 240, 250)) for k, fn in player_map.items()}
    bf = {0: "bullet_small.png", 1: "bullet_med.png", 2: "bullet_big.png", 3: "bullet_lime.png"}
    bs = {k: load_sprite(assets / fn, int(C.BULLET_RADIUS[k]), BULLET_COLORS[k]) for k, fn in bf.items()}

    frames = out.with_suffix("").parent / f"_frames_{out.stem}"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)

    last = "idle"
    max_steps = int(max_seconds / C.DT)
    n = 0
    for i in range(max_steps):
        attn = render_attn(player, spawner, env)
        ordered = order_bullets_v4(env, MAX_BULLETS_V4) if attn is not None else None
        render(env, scale, ps, bs, last, title, ordered, attn).save(frames / f"f{i:06d}.png")
        if spawner_mode == "scripted":
            a = player_act(player, env, player_mode, flee)
            last = ACTIONS[a]
            _o, _r, done, _ = env.step(a)
            n = i + 1
            if done:
                hit_attn = render_attn(player, spawner, env)
                hit_ordered = order_bullets_v4(env, MAX_BULLETS_V4) if hit_attn is not None else None
                for j in range(30):
                    render(env, scale, ps, bs, last, title + "  HIT", hit_ordered, hit_attn).save(
                        frames / f"f{i+1+j:06d}.png"
                    )
                break
            continue

        env.elapsed += env.dt
        env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed)
        spawns = 0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            act = spawner_act(spawner, env)
            spawn_continuous(env, act["birth"], act["aim"], act["kind"], rng_jitter=False)
            spawns += 1
            p_double = 0.20 if env.elapsed > 18.0 else (0.10 if env.elapsed > 8.0 else 0.0)
            if spawns == 1 and p_double > 0 and env.rng.randf() < p_double:
                env.spawn_acc += interval
            interval = _spawn_interval(env.elapsed)

        a = player_act(player, env, player_mode, flee)
        last = ACTIONS[a]
        apply_player(env, a)
        env._integrate_bullets()
        n = i + 1
        if env._check_hit():
            env.dead = True
            hit_attn = render_attn(player, spawner, env)
            hit_ordered = order_bullets_v4(env, MAX_BULLETS_V4) if hit_attn is not None else None
            for j in range(30):
                render(env, scale, ps, bs, last, title + "  HIT", hit_ordered, hit_attn).save(
                    frames / f"f{i+1+j:06d}.png"
                )
            break

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-framerate", "60", "-i", str(frames / "f%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out),
        ]
    )
    shutil.rmtree(frames, ignore_errors=True)
    print(f"wrote {out} elapsed={env.elapsed:.2f}s dead={env.dead} frames={n}", flush=True)
    return env.elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", type=Path, default=Path("runs/both_v4_player.pt"))
    ap.add_argument("--spawner", type=Path, default=Path("runs/both_v4_spawner.pt"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=45.0)
    ap.add_argument("--outdir", type=Path, default=Path("dist"))
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    args = ap.parse_args()

    pck = torch.load(args.player, map_location="cpu", weights_only=False)
    sck = torch.load(args.spawner, map_location="cpu", weights_only=False)
    player = PlayerV4(d_model=int(pck.get("d_model", 128)), hidden=int(pck.get("hidden", 256)))
    player.load_state_dict(pck["state_dict"])
    player.eval()
    spawner = SpawnerV4(d_model=int(sck.get("d_model", 128)), hidden=int(sck.get("hidden", 256)))
    spawner.load_state_dict(sck["state_dict"])
    spawner.eval()

    jobs = [
        ("newP_x_scripted", "new", "scripted", player, None),
        ("new_x_new", "new", "new", player, spawner),
        ("flee_x_newS", "flee", "new", None, spawner),
    ]
    for stem, pm, sm, pnet, snet in jobs:
        run_match(
            player_mode=pm,
            spawner_mode=sm,
            player=pnet,
            spawner=snet,
            seed=args.seed,
            max_seconds=args.max_seconds,
            scale=args.scale,
            assets=args.assets,
            out=args.outdir / f"v4_{stem}.mp4",
            title=stem.replace("_", " "),
        )


if __name__ == "__main__":
    main()
