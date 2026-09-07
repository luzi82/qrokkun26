#!/usr/bin/env python3
"""Render best player vs best learned spawner to mp4."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR, Qrokkun26Env, _move_toward, _spawn_interval
from qrokkun_env.godot_rng import f32
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.render_player_demo import (
    PlayerAC,
    load_sprite,
    paste_centered,
    BULLET_COLORS,
)
from qrokkun_env.train_spawner_gpu import (
    SpawnerAC,
    SPAWNER_OBS_DIM,
    spawn_from_action,
    spawner_vectorize,
)


@torch.no_grad()
def player_act(net: PlayerAC, env: Qrokkun26Env, device: torch.device) -> int:
    x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
    return int(net(x).probs.argmax().item())


@torch.no_grad()
def spawner_act(net: SpawnerAC, env: Qrokkun26Env, device: torch.device) -> int:
    x = torch.tensor(spawner_vectorize(env), dtype=torch.float32, device=device)
    dist, _v = net(x)
    return int(dist.probs.argmax().item())


def render_frame(env, scale, player_sprites, bullet_sprites, last_action, title: str) -> Image.Image:
    W, H = int(C.VIEW_W) * scale, int(C.VIEW_H) * scale
    img = Image.new("RGBA", (W, H), (18, 18, 28, 255))
    d = ImageDraw.Draw(img)
    fx0, fy0 = int(C.FIELD_X * scale), int(C.FIELD_Y * scale)
    fx1, fy1 = int((C.FIELD_X + C.FIELD_W) * scale), int((C.FIELD_Y + C.FIELD_H) * scale)
    d.rectangle((fx0, fy0, fx1, fy1), fill=(28, 32, 48, 255), outline=(90, 100, 140, 255))
    for b in env.bullets:
        spr = bullet_sprites.get(b.kind, bullet_sprites[0])
        paste_centered(img, spr, b.x, b.y, scale)
    key = last_action if last_action in player_sprites else "idle"
    paste_centered(img, player_sprites[key], env.px, env.py, scale)
    d.rectangle((0, 0, W, max(16 * scale // 2, 14)), fill=(10, 10, 16, 230))
    d.text((6, 2), f"{title}  t={env.elapsed:5.2f}s  bullets={len(env.bullets):3d}  act={last_action}", fill=(220, 230, 255, 255))
    return img.convert("RGB")


def step_dual(env: Qrokkun26Env, player: PlayerAC, spawner: SpawnerAC, device: torch.device) -> tuple[str, bool]:
    """One physics frame: learned spawn then player move (mirrors train_spawner_gpu)."""
    env.elapsed += env.dt
    env.spawn_acc += env.dt
    interval = _spawn_interval(env.elapsed)
    spawns = 0
    while env.spawn_acc >= interval:
        env.spawn_acc -= interval
        spawn_from_action(env, spawner_act(spawner, env, device))
        spawns += 1
        if spawns == 1 and env.elapsed > 18.0 and env.rng.randf() < 0.16:
            env.spawn_acc += interval
        interval = _spawn_interval(env.elapsed)

    pa = player_act(player, env, device)
    last = ACTIONS[pa]
    dx, dy = ACTION_TO_DIR[last]
    if dx != 0.0 or dy != 0.0:
        n = (dx * dx + dy * dy) ** 0.5
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
        return last, True
    return last, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", type=Path, default=Path("runs/player_gpu.pt"))
    ap.add_argument("--spawner", type=Path, default=Path("runs/spawner_gpu.pt"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path("dist/player_vs_spawner_best.mp4"))
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    pck = torch.load(args.player, map_location=device, weights_only=False)
    sck = torch.load(args.spawner, map_location=device, weights_only=False)
    player = PlayerAC(hidden=int(pck.get("hidden", 256))).to(device)
    player.load_state_dict(pck["state_dict"])
    player.eval()
    spawner = SpawnerAC(hidden=int(sck.get("hidden", 256))).to(device)
    spawner.load_state_dict(sck["state_dict"])
    spawner.eval()

    player_map = {
        "idle": "player.png", "n": "player_n.png", "ne": "player_ne.png", "e": "player_e.png",
        "se": "player_se.png", "s": "player_s.png", "sw": "player_sw.png", "w": "player_w.png", "nw": "player_nw.png",
    }
    player_sprites = {k: load_sprite(args.assets / fn, int(C.PLAYER_RADIUS), (240, 240, 250)) for k, fn in player_map.items()}
    bullet_files = {0: "bullet_small.png", 1: "bullet_med.png", 2: "bullet_big.png", 3: "bullet_lime.png"}
    bullet_sprites = {k: load_sprite(args.assets / fn, int(C.BULLET_RADIUS[k]), BULLET_COLORS[k]) for k, fn in bullet_files.items()}

    env = Qrokkun26Env(seed=args.seed)
    env.reset(seed=args.seed)
    frames_dir = args.out.with_suffix("").parent / "_vs_demo_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    title = "player vs learned spawner"
    last = "idle"
    max_steps = int(args.max_seconds / C.DT)
    n = 0
    for i in range(max_steps):
        frame = render_frame(env, args.scale, player_sprites, bullet_sprites, last, title)
        frame.save(frames_dir / f"f{i:06d}.png")
        last, done = step_dual(env, player, spawner, device)
        n = i + 1
        if done:
            for j in range(30):
                frame = render_frame(env, args.scale, player_sprites, bullet_sprites, last, title + "  HIT")
                frame.save(frames_dir / f"f{i + 1 + j:06d}.png")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", "60", "-i", str(frames_dir / "f%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.out),
    ])
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"wrote {args.out} frames={n} elapsed={env.elapsed:.2f}s dead={env.dead} "
          f"player_eval={pck.get('eval_mean_s')} spawner_best={sck.get('eval_surv_s')}", flush=True)


if __name__ == "__main__":
    main()
