"""Observation vector for the player policy."""

from __future__ import annotations

import math

from qrokkun_env import constants as C
from qrokkun_env.env import Qrokkun26Env

# player(4) + elapsed(1) + up to K bullets * (dx,dy,vx,vy,r) + pad mask via r<0 already in env
MAX_BULLETS = 8
OBS_DIM = 4 + 1 + MAX_BULLETS * 5  # 45


def vectorize(env: Qrokkun26Env) -> list[float]:
    """Compact relative obs for a tiny MLP (not the padded gym dict)."""
    # Normalize roughly to ~[-1,1] / seconds scale.
    px = (env.px - (C.FIELD_X + C.FIELD_W * 0.5)) / (C.FIELD_W * 0.5)
    py = (env.py - (C.FIELD_Y + C.FIELD_H * 0.5)) / (C.FIELD_H * 0.5)
    pvx = env.pvx / C.PLAYER_MAX_SPEED
    pvy = env.pvy / C.PLAYER_MAX_SPEED
    elapsed = min(env.elapsed / 60.0, 2.0)
    out = [px, py, pvx, pvy, elapsed]
    ordered = sorted(
        env.bullets,
        key=lambda b: (b.x - env.px) ** 2 + (b.y - env.py) ** 2,
    )[:MAX_BULLETS]
    for b in ordered:
        out.extend(
            [
                (b.x - env.px) / C.FIELD_W,
                (b.y - env.py) / C.FIELD_H,
                b.vx / 125.0,
                b.vy / 125.0,
                b.radius / 4.0,
            ]
        )
    while len(out) < OBS_DIM:
        out.append(0.0)
    return out[:OBS_DIM]
