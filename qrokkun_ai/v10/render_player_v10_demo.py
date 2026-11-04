#!/usr/bin/env python3
"""Fixed-contract, diagnostic-only PlayerV10 scripted-spawner MP4 renderer (120s Cap, Zero-Inertia Physics)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from qrokkun_ai.v10.agents.player_v10 import (
    PlayerV10,
    argmax_action_v10,
    load_v10_checkpoint,
)
from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, Qrokkun26Env

FIXED_SEED = 3000
FIXED_FRAME_CAP = 7200  # 120.0s cap at 60 Hz
HIT_ANIM_FRAMES = 30  # 0.50s hit animation tail at 60 Hz

BULLET_SPRITE_FILES = {
    0: "bullet_small.png",
    1: "bullet_med.png",
    2: "bullet_big.png",
    3: "bullet_lime.png",
}


class DemoRenderError(RuntimeError):
    """A fixed-contract diagnostic render cannot safely proceed."""


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bullet_sprites(assets: Path) -> dict[int, Image.Image]:
    """Load the exact Godot sprite for every canonical bullet kind."""
    sprites: dict[int, Image.Image] = {}
    for kind, filename in BULLET_SPRITE_FILES.items():
        path = assets / filename
        if not path.is_file():
            raise DemoRenderError(
                f"missing required bullet sprite for kind {kind}: {path}"
            )
        with Image.open(path) as source:
            sprites[kind] = source.convert("RGBA")
    return sprites


def load_demo_player(
    checkpoint: Path | str, device: torch.device | str = "cpu"
) -> tuple[PlayerV10, dict[str, object]]:
    """Load only a production-schema PlayerV10 checkpoint, strictly."""
    return load_v10_checkpoint(checkpoint, device=torch.device(device), eval_mode=True)


def deterministic_player_action(
    net: PlayerV10, env: Qrokkun26Env, device: torch.device | str
) -> int:
    """Return the PlayerV10 policy's fixed argmax action for this observation."""
    return argmax_action_v10(net, env, torch.device(device))


def _run_loaded_rollout(
    net: PlayerV10,
    device: torch.device | str,
    env: Qrokkun26Env,
    *,
    seed: int = FIXED_SEED,
    on_frame: Callable[[Qrokkun26Env, str, int], None] | None = None,
) -> dict[str, object]:
    dev = torch.device(device)
    for frame in range(1, FIXED_FRAME_CAP + 1):
        action = deterministic_player_action(net, env, dev)
        _obs, _reward, done, info = env.step(action)
        if on_frame is not None:
            on_frame(env, ACTIONS[action], frame)
        if done:
            return {
                "seed": seed,
                "elapsed": float(info["elapsed"]),
                "frames": frame,
                "hit": True,
                "censored": False,
                "termination_reason": "hit",
            }
    return {
        "seed": seed,
        "elapsed": float(env.elapsed),
        "frames": FIXED_FRAME_CAP,
        "hit": False,
        "censored": True,
        "termination_reason": "frame_cap",
    }


def _load_sprite(
    path: Path, fallback_radius: int, color: tuple[int, int, int]
) -> Image.Image:
    if path.is_file():
        return Image.open(path).convert("RGBA")
    size = max(fallback_radius * 2, 8)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(image).ellipse((0, 0, size - 1, size - 1), fill=color + (255,))
    return image


def _paste_centered(
    canvas: Image.Image, sprite: Image.Image, x: float, y: float, scale: int
) -> None:
    width, height = sprite.size
    if scale != 1:
        sprite = sprite.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    width, height = sprite.size
    canvas.alpha_composite(
        sprite, (round(x * scale - width / 2), round(y * scale - height / 2))
    )


def render_frame(
    env: Qrokkun26Env,
    action: str,
    assets: Path,
    scale: int,
    *,
    bullet_sprites: dict[int, Image.Image],
    player_sprites: dict[str, Image.Image] | None = None,
    seed: int | None = None,
    hit_step: int | None = None,
) -> Image.Image:
    """Draw one post-step canonical-environment frame (with optional hit animation)."""
    import math
    from PIL import ImageDraw

    width, height = int(C.VIEW_W) * scale, int(C.VIEW_H) * scale
    image = Image.new("RGBA", (width, height), (18, 18, 28, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            C.FIELD_X * scale,
            C.FIELD_Y * scale,
            (C.FIELD_X + C.FIELD_W) * scale,
            (C.FIELD_Y + C.FIELD_H) * scale,
        ),
        fill=(28, 32, 48, 255),
        outline=(90, 100, 140, 255),
    )
    for bullet in env.bullets:
        _paste_centered(image, bullet_sprites[bullet.kind], bullet.x, bullet.y, scale)
    if player_sprites is not None and action in player_sprites:
        p_sprite = player_sprites[action]
    else:
        player_files = {
            "idle": "player.png",
            **{name: f"player_{name}.png" for name in ACTIONS if name != "idle"},
        }
        p_sprite = _load_sprite(
            assets / player_files[action], int(C.PLAYER_RADIUS), (240, 240, 250)
        )

    if hit_step is None:
        _paste_centered(
            image,
            p_sprite,
            env.px,
            env.py,
            scale,
        )
        status_suffix = f"act={action}"
        hud_col = (220, 230, 255, 255)
    else:
        p = min(max(float(hit_step) / float(max(HIT_ANIM_FRAMES - 1, 1)), 0.0), 1.0)
        # Screen flash (mirrors main.gd flash Color(1.0, 0.25, 0.35) over first 11 frames)
        if hit_step < 12:
            flash_alpha = int((1.0 - hit_step / 12.0) * 115)
            flash_layer = Image.new("RGBA", (width, height), (255, 64, 89, flash_alpha))
            image.alpha_composite(flash_layer)
            draw = ImageDraw.Draw(image)

        # Jittering, blinking tinted player sprite
        shake = (1.0 - p) * 2.5
        jx = env.px + math.sin(p * 55.0) * shake
        jy = env.py + math.cos(p * 43.0) * shake
        blink = (hit_step // 2) % 2 == 0
        tint_rgb = (255, 64, 89) if blink else (255, 230, 95)
        r_ch, g_ch, b_ch, a_ch = p_sprite.split()
        tinted = Image.merge(
            "RGBA",
            (
                r_ch.point(lambda v: (v * tint_rgb[0]) // 255),
                g_ch.point(lambda v: (v * tint_rgb[1]) // 255),
                b_ch.point(lambda v: (v * tint_rgb[2]) // 255),
                a_ch,
            ),
        )
        _paste_centered(image, tinted, jx, jy, scale)

        # Expanding dual shockwave rings + 8 radial sparks around (env.px, env.py)
        cx_px, cy_px = env.px * scale, env.py * scale
        r1 = (4.0 + 22.0 * p) * scale
        draw.ellipse(
            (cx_px - r1, cy_px - r1, cx_px + r1, cy_px + r1),
            outline=(255, 80, 95, 255),
            width=max(2, scale),
        )
        if p > 0.10:
            p2 = min((p - 0.10) / 0.90, 1.0)
            r2 = (3.0 + 14.0 * p2) * scale
            draw.ellipse(
                (cx_px - r2, cy_px - r2, cx_px + r2, cy_px + r2),
                outline=(255, 225, 90, 255),
                width=max(1, scale - 1),
            )
        for k_ray in range(8):
            ang = k_ray * (math.pi / 4.0) + p * 0.6
            d_in = (3.0 + 10.0 * p) * scale
            d_out = (6.0 + 18.0 * p) * scale
            x0 = cx_px + math.cos(ang) * d_in
            y0 = cy_px + math.sin(ang) * d_in
            x1 = cx_px + math.cos(ang) * d_out
            y1 = cy_px + math.sin(ang) * d_out
            draw.line((x0, y0, x1, y1), fill=(255, 210, 110, 255), width=max(1, scale - 1))

        status_suffix = f"act={action}  [HIT!]"
        hud_col = (255, 110, 125, 255)

    seed_prefix = f"seed={seed}  " if seed is not None else ""
    draw.text(
        (6, 2),
        f"{seed_prefix}t={env.elapsed:5.1f}s  bullets={len(env.bullets):3d}  {status_suffix}",
        fill=hud_col,
    )
    return image.convert("RGB")


def _sidecar_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def render_demo(
    checkpoint: Path | str,
    output: Path | str,
    *,
    assets: Path | str = Path("assets"),
    scale: int = 3,
    device: str = "cpu",
    seed: int = FIXED_SEED,
) -> dict[str, object]:
    """Render the fixed demo MP4 and an identity sidecar, refusing overwrite."""
    import tempfile

    checkpoint, output, assets = Path(checkpoint), Path(output), Path(assets)
    sidecar = _sidecar_path(output)
    frames_dir = output.parent / f".{output.stem}.player_v10_frames"
    if output.exists() or sidecar.exists() or frames_dir.exists():
        raise DemoRenderError(
            "refusing to overwrite output, sidecar, or frame directory"
        )
    if scale <= 0:
        raise DemoRenderError("scale must be positive")
    bullet_sprites = load_bullet_sprites(assets)
    player_files = {
        "idle": "player.png",
        **{name: f"player_{name}.png" for name in ACTIONS if name != "idle"},
    }
    player_sprites = {
        act: _load_sprite(assets / fname, int(C.PLAYER_RADIUS), (240, 240, 250))
        for act, fname in player_files.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Use local TMPDIR for transient PNG frames when output is on slow network filesystem
    local_frames_dir = Path(tempfile.mkdtemp(prefix=f"{output.stem}_v10_frames_"))
    try:
        net, meta = load_demo_player(checkpoint, device=device)
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)

        last_action_str = "idle"

        def write_frame(env: Qrokkun26Env, action: str, frame: int) -> None:
            nonlocal last_action_str
            last_action_str = action
            render_frame(
                env,
                action,
                assets,
                scale,
                bullet_sprites=bullet_sprites,
                player_sprites=player_sprites,
                seed=seed,
                hit_step=None,
            ).save(local_frames_dir / f"f{frame - 1:06d}.png", compress_level=1)

        result = _run_loaded_rollout(net, device, env, seed=seed, on_frame=write_frame)
        if result.get("hit"):
            base_idx = int(result["frames"])
            for h_idx in range(HIT_ANIM_FRAMES):
                render_frame(
                    env,
                    last_action_str,
                    assets,
                    scale,
                    bullet_sprites=bullet_sprites,
                    player_sprites=player_sprites,
                    seed=seed,
                    hit_step=h_idx,
                ).save(local_frames_dir / f"f{base_idx + h_idx:06d}.png", compress_level=1)
        command = [
            "ffmpeg",
            "-framerate",
            "60",
            "-i",
            str(local_frames_dir / "f%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError) as exc:
            output.unlink(missing_ok=True)
            raise DemoRenderError(f"ffmpeg encoding failed: {exc}") from exc
        metadata = {
            "kind": "player_v10_scripted_spawner_diagnostic_demo",
            "experimental": True,
            "promotion": "forbidden",
            "checkpoint_selection": "none",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "file_sha256": file_sha256(checkpoint),
                "state_dict_sha256": meta["state_dict_sha256"],
            },
            "result": result,
        }
        try:
            sidecar.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            sidecar.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            raise
        return metadata
    finally:
        shutil.rmtree(local_frames_dir, ignore_errors=True)
        shutil.rmtree(frames_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Explicit preselected PlayerV10 checkpoint",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="New MP4 destination (must not already exist)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Scripted-spawner seed for this one diagnostic rollout (default: 3000)",
    )
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = render_demo(
        args.ckpt,
        args.out,
        assets=args.assets,
        scale=args.scale,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
