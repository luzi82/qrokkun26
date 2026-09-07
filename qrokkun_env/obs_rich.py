"""Rich observation: player state + nearest K bullets (shared by player & spawner)."""

from __future__ import annotations

from qrokkun_env import constants as C
from qrokkun_env.env import Qrokkun26Env

MAX_BULLETS_RICH = 64
# px,py,pvx,pvy,elapsed,spawn_acc + K*(dx,dy,vx,vy,r,kind)
FEATS_PER_BULLET = 6
OBS_DIM_RICH = 6 + MAX_BULLETS_RICH * FEATS_PER_BULLET  # 390


def vectorize_rich(env: Qrokkun26Env, max_bullets: int = MAX_BULLETS_RICH) -> list[float]:
    px = (env.px - (C.FIELD_X + C.FIELD_W * 0.5)) / (C.FIELD_W * 0.5)
    py = (env.py - (C.FIELD_Y + C.FIELD_H * 0.5)) / (C.FIELD_H * 0.5)
    pvx = env.pvx / C.PLAYER_MAX_SPEED
    pvy = env.pvy / C.PLAYER_MAX_SPEED
    elapsed = min(env.elapsed / 60.0, 2.0)
    spawn_acc = env.spawn_acc
    out: list[float] = [px, py, pvx, pvy, elapsed, spawn_acc]
    ordered = sorted(
        env.bullets,
        key=lambda b: (b.x - env.px) ** 2 + (b.y - env.py) ** 2,
    )[:max_bullets]
    for b in ordered:
        out.extend(
            [
                (b.x - env.px) / C.FIELD_W,
                (b.y - env.py) / C.FIELD_H,
                b.vx / 125.0,
                b.vy / 125.0,
                b.radius / 4.0,
                b.kind / 3.0,
            ]
        )
    dim = 6 + max_bullets * FEATS_PER_BULLET
    while len(out) < dim:
        # pad empty slots: zeros; radius/kind sentinel-ish already 0
        out.append(0.0)
    return out[:dim]
