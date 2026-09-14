#!/usr/bin/env python3
"""Fixed-contract, diagnostic-only PlayerV5 scripted-spawner MP4 renderer.

This module deliberately has no checkpoint discovery or selection logic.  Its
single checkpoint argument is an already-selected input, validated by the
production ranked-top-k loader before it can drive the canonical environment.
It is not a frozen-Spawner evaluation runner and its output is experimental
diagnostic media only; it must never be used for promotion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from qrokkun_env import constants as C
from qrokkun_env.agents.player_checkpoints import file_sha256, load_ranked_top_k_checkpoint
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK, argmax_action
from qrokkun_env.env import ACTIONS, Qrokkun26Env

FIXED_SEED = 3000
FIXED_FRAME_CAP = 4200


class DemoRenderError(RuntimeError):
    """A fixed-contract diagnostic render cannot safely proceed."""


def load_demo_player(checkpoint: Path | str, device: torch.device | str = "cpu") -> tuple[PlayerRankedTopK, dict[str, object]]:
    """Load only a production-schema PlayerRankedTopK checkpoint, strictly."""
    return load_ranked_top_k_checkpoint(checkpoint, device=torch.device(device), eval_mode=True)


def deterministic_player_action(net: PlayerRankedTopK, env: Qrokkun26Env, device: torch.device | str) -> int:
    """Return the ranked-top-k policy's fixed argmax action for this observation."""
    return argmax_action(net, env, torch.device(device))


def run_demo_rollout(
    checkpoint: Path | str,
    *,
    device: torch.device | str = "cpu",
    on_frame: Callable[[Qrokkun26Env, str, int], None] | None = None,
) -> dict[str, object]:
    """Run one fixed scripted-spawner rollout, optionally emitting post-step frames.

    ``Qrokkun26Env.step`` is the canonical float32 Player/update-order path.
    The action is deliberately argmax-only, and the public API offers no seed,
    cap, or action override.
    """
    dev = torch.device(device)
    net, _meta = load_demo_player(checkpoint, device=dev)
    env = Qrokkun26Env(seed=FIXED_SEED)
    env.reset(seed=FIXED_SEED)
    return _run_loaded_rollout(net, dev, env, on_frame=on_frame)


def _run_loaded_rollout(
    net: PlayerRankedTopK,
    device: torch.device | str,
    env: Qrokkun26Env,
    *,
    on_frame: Callable[[Qrokkun26Env, str, int], None] | None = None,
) -> dict[str, object]:
    """Execute the fixed rollout with a strict-loaded net and canonical env.

    This small internal seam lets the behavioral tests force a genuine env
    collision without replacing environment physics or checkpoint validation.
    """
    dev = torch.device(device)
    for frame in range(1, FIXED_FRAME_CAP + 1):
        action = deterministic_player_action(net, env, dev)
        _obs, _reward, done, info = env.step(action)
        if on_frame is not None:
            on_frame(env, ACTIONS[action], frame)
        if done:
            return {
                "seed": FIXED_SEED,
                "elapsed": float(info["elapsed"]),
                "frames": frame,
                "hit": True,
                "censored": False,
                "termination_reason": "hit",
            }
    return {
        "seed": FIXED_SEED,
        "elapsed": float(env.elapsed),
        "frames": FIXED_FRAME_CAP,
        "hit": False,
        "censored": True,
        "termination_reason": "frame_cap",
    }


def _load_sprite(path: Path, fallback_radius: int, color: tuple[int, int, int]) -> Image.Image:
    if path.is_file():
        from PIL import ImageDraw

        return Image.open(path).convert("RGBA")
    size = max(fallback_radius * 2, 8)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    from PIL import ImageDraw

    ImageDraw.Draw(image).ellipse((0, 0, size - 1, size - 1), fill=color + (255,))
    return image


def _paste_centered(canvas: Image.Image, sprite: Image.Image, x: float, y: float, scale: int) -> None:
    width, height = sprite.size
    if scale != 1:
        sprite = sprite.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    width, height = sprite.size
    canvas.alpha_composite(sprite, (round(x * scale - width / 2), round(y * scale - height / 2)))


def render_frame(env: Qrokkun26Env, action: str, assets: Path, scale: int) -> Image.Image:
    """Draw one post-step canonical-environment frame without legacy renderer state."""
    from PIL import ImageDraw

    width, height = int(C.VIEW_W) * scale, int(C.VIEW_H) * scale
    image = Image.new("RGBA", (width, height), (18, 18, 28, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (C.FIELD_X * scale, C.FIELD_Y * scale, (C.FIELD_X + C.FIELD_W) * scale, (C.FIELD_Y + C.FIELD_H) * scale),
        fill=(28, 32, 48, 255), outline=(90, 100, 140, 255),
    )
    colors = ((220, 80, 80), (80, 200, 120), (120, 160, 255), (240, 200, 60))
    for bullet in env.bullets:
        _paste_centered(image, _load_sprite(assets / f"bullet_{bullet.kind}.png", int(bullet.radius), colors[bullet.kind]), bullet.x, bullet.y, scale)
    player_files = {"idle": "player.png", **{name: f"player_{name}.png" for name in ACTIONS if name != "idle"}}
    _paste_centered(image, _load_sprite(assets / player_files[action], int(C.PLAYER_RADIUS), (240, 240, 250)), env.px, env.py, scale)
    draw.text((6, 2), f"t={env.elapsed:5.1f}s  bullets={len(env.bullets):3d}  act={action}", fill=(220, 230, 255, 255))
    return image.convert("RGB")


def _sidecar_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def render_demo(checkpoint: Path | str, output: Path | str, *, assets: Path | str = Path("assets"), scale: int = 3, device: str = "cpu") -> dict[str, object]:
    """Render the fixed demo MP4 and an identity sidecar, refusing overwrite."""
    checkpoint, output, assets = Path(checkpoint), Path(output), Path(assets)
    sidecar = _sidecar_path(output)
    frames_dir = output.parent / f".{output.stem}.player_v5_frames"
    if output.exists() or sidecar.exists() or frames_dir.exists():
        raise DemoRenderError("refusing to overwrite output, sidecar, or frame directory")
    if scale <= 0:
        raise DemoRenderError("scale must be positive")
    frames_dir.mkdir(parents=True)
    try:
        # One strict load supplies both the rollout weights and sidecar identity.
        # There is intentionally no directory scan, ranking, or checkpoint choice.
        net, meta = load_demo_player(checkpoint, device=device)
        env = Qrokkun26Env(seed=FIXED_SEED)
        env.reset(seed=FIXED_SEED)

        def write_frame(env: Qrokkun26Env, action: str, frame: int) -> None:
            render_frame(env, action, assets, scale).save(frames_dir / f"f{frame - 1:06d}.png")

        result = _run_loaded_rollout(net, device, env, on_frame=write_frame)
        command = ["ffmpeg", "-framerate", "60", "-i", str(frames_dir / "f%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(output)]
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            # The destination was absent at entry, so a partial file can only
            # belong to this failed attempt and is safe to remove.
            output.unlink(missing_ok=True)
            raise DemoRenderError(f"ffmpeg encoding failed: {exc}") from exc
        metadata = {
            "kind": "player_v5_scripted_spawner_diagnostic_demo",
            "experimental": True,
            "promotion": "forbidden",
            "checkpoint_selection": "none",
            "checkpoint": {"path": str(checkpoint.resolve()), "file_sha256": file_sha256(checkpoint), "state_dict_sha256": meta["state_dict_sha256"]},
            "result": result,
        }
        try:
            sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            # The MP4 belongs to this attempt until its identity sidecar is
            # durably persisted.  Remove both targets so no-overwrite permits
            # a clean retry, including after a partial sidecar write.
            sidecar.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            raise
        return metadata
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path, help="Explicit preselected PlayerRankedTopK checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="New MP4 destination (must not already exist)")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = render_demo(args.ckpt, args.out, assets=args.assets, scale=args.scale, device=args.device)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
