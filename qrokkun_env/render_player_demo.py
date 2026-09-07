#!/usr/bin/env python3
"""Render a trained player vs scripted spawner to PNG frames + mp4."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.distributions import Categorical

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.obs import OBS_DIM, vectorize

BULLET_COLORS = {
    0: (220, 80, 80),
    1: (80, 200, 120),
    2: (120, 160, 255),
    3: (240, 200, 60),
}


class PlayerAC(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.policy(self.body(x)))


def load_player(path: Path, device: torch.device) -> PlayerAC:
    ck = torch.load(path, map_location=device, weights_only=False)
    hidden = int(ck.get("hidden", 256))
    net = PlayerAC(hidden=hidden).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


@torch.no_grad()
def act(net: PlayerAC, env: Qrokkun26Env, device: torch.device, greedy: bool) -> int:
    x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
    dist = net(x)
    return int(dist.probs.argmax().item() if greedy else dist.sample().item())


def load_sprite(path: Path, fallback_r: int, color: tuple[int, int, int]) -> Image.Image:
    if path.is_file():
        im = Image.open(path).convert("RGBA")
        return im
    s = max(fallback_r * 2, 8)
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((0, 0, s - 1, s - 1), fill=color + (255,))
    return im


def paste_centered(canvas: Image.Image, sprite: Image.Image, x: float, y: float, scale: int) -> None:
    w, h = sprite.size
    sw, sh = w * scale, h * scale
    if scale != 1:
        sprite = sprite.resize((sw, sh), Image.NEAREST)
    canvas.alpha_composite(sprite, (int(round(x * scale - sw / 2)), int(round(y * scale - sh / 2))))


def render_frame(
    env: Qrokkun26Env,
    scale: int,
    player_sprites: dict[str, Image.Image],
    bullet_sprites: dict[int, Image.Image],
    last_action: str,
) -> Image.Image:
    W, H = int(C.VIEW_W) * scale, int(C.VIEW_H) * scale
    img = Image.new("RGBA", (W, H), (18, 18, 28, 255))
    d = ImageDraw.Draw(img)
    # field
    fx0, fy0 = int(C.FIELD_X * scale), int(C.FIELD_Y * scale)
    fx1, fy1 = int((C.FIELD_X + C.FIELD_W) * scale), int((C.FIELD_Y + C.FIELD_H) * scale)
    d.rectangle((fx0, fy0, fx1, fy1), fill=(28, 32, 48, 255), outline=(90, 100, 140, 255))
    for b in env.bullets:
        spr = bullet_sprites.get(b.kind, bullet_sprites[0])
        paste_centered(img, spr, b.x, b.y, scale)
    key = last_action if last_action in player_sprites else "idle"
    paste_centered(img, player_sprites[key], env.px, env.py, scale)
    # HUD
    d.rectangle((0, 0, W, max(16 * scale // 2, 14)), fill=(10, 10, 16, 230))
    text = f"t={env.elapsed:5.1f}s  bullets={len(env.bullets):3d}  act={last_action}"
    d.text((6, 2), text, fill=(220, 230, 255, 255))
    return img.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("runs/player_gpu.pt"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("dist/player_demo_scripted_spawner.mp4"))
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_player(args.ckpt, device)
    env = Qrokkun26Env(seed=args.seed)
    env.reset(seed=args.seed)

    player_map = {
        "idle": "player.png",
        "n": "player_n.png",
        "ne": "player_ne.png",
        "e": "player_e.png",
        "se": "player_se.png",
        "s": "player_s.png",
        "sw": "player_sw.png",
        "w": "player_w.png",
        "nw": "player_nw.png",
    }
    player_sprites = {
        k: load_sprite(args.assets / fn, int(C.PLAYER_RADIUS), (240, 240, 250))
        for k, fn in player_map.items()
    }
    bullet_files = {
        0: "bullet_small.png",
        1: "bullet_med.png",
        2: "bullet_big.png",
        3: "bullet_lime.png",
    }
    bullet_sprites = {
        k: load_sprite(args.assets / fn, int(C.BULLET_RADIUS[k]), BULLET_COLORS[k])
        for k, fn in bullet_files.items()
    }

    frames_dir = args.out.with_suffix("").parent / "_player_demo_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    last_action = "idle"
    max_steps = int(args.max_seconds / C.DT)
    n = 0
    for i in range(max_steps):
        frame = render_frame(env, args.scale, player_sprites, bullet_sprites, last_action)
        frame.save(frames_dir / f"f{i:06d}.png")
        a = act(net, env, device, args.greedy)
        last_action = ACTIONS[a]
        _o, _r, done, _info = env.step(a)
        n = i + 1
        if done:
            # hold last frame ~0.5s
            for j in range(30):
                frame.save(frames_dir / f"f{i + 1 + j:06d}.png")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", "60",
        "-i", str(frames_dir / "f%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(args.out),
    ]
    subprocess.check_call(cmd)
    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"wrote {args.out} frames={n} elapsed={env.elapsed:.2f}s dead={env.dead}", flush=True)


if __name__ == "__main__":
    main()
