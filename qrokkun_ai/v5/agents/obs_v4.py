"""v4 observations: player state + up to 64 bullets with r<0 pad + kind one-hot + mask."""

from __future__ import annotations

import numpy as np

from qrokkun_env import constants as C
from qrokkun_env.env import Qrokkun26Env

MAX_BULLETS_V4 = 64
N_KIND = 4
# dx, dy, vx, vy, r, kind0..kind3
BULLET_FEAT_V4 = 5 + N_KIND
PLAYER_FEAT_V4 = 6  # px,py,pvx,pvy,elapsed,spawn_acc
PAD_RADIUS = -1.0


def _player_feats(env: Qrokkun26Env) -> list[float]:
    return [
        (env.px - (C.FIELD_X + C.FIELD_W * 0.5)) / (C.FIELD_W * 0.5),
        (env.py - (C.FIELD_Y + C.FIELD_H * 0.5)) / (C.FIELD_H * 0.5),
        env.pvx / C.PLAYER_MAX_SPEED,
        env.pvy / C.PLAYER_MAX_SPEED,
        min(env.elapsed / 60.0, 2.0),
        env.spawn_acc,
    ]


def _bullet_feat(env: Qrokkun26Env, b) -> list[float]:
    onehot = [0.0] * N_KIND
    k = int(b.kind)
    if 0 <= k < N_KIND:
        onehot[k] = 1.0
    return [
        (b.x - env.px) / C.FIELD_W,
        (b.y - env.py) / C.FIELD_H,
        b.vx / 125.0,
        b.vy / 125.0,
        b.radius / 4.0,
        *onehot,
    ]


def _pad_bullet() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, PAD_RADIUS, 0.0, 0.0, 0.0, 0.0]


def order_bullets_v4(env: Qrokkun26Env, max_bullets: int = MAX_BULLETS_V4) -> list:
    """Return bullets nearest-first (same order used by encode_obs), truncated to max_bullets."""
    return sorted(
        env.bullets,
        key=lambda b: (b.x - env.px) ** 2 + (b.y - env.py) ** 2,
    )[:max_bullets]


def encode_obs(env: Qrokkun26Env, max_bullets: int = MAX_BULLETS_V4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (player[P], bullets[K,F], pad_mask[K]) with pad_mask True = empty slot."""
    player = np.asarray(_player_feats(env), dtype=np.float32)
    ordered = order_bullets_v4(env, max_bullets)
    bullets = np.zeros((max_bullets, BULLET_FEAT_V4), dtype=np.float32)
    pad = np.ones((max_bullets,), dtype=np.bool_)
    for i, b in enumerate(ordered):
        bullets[i] = np.asarray(_bullet_feat(env, b), dtype=np.float32)
        pad[i] = False
    for i in range(len(ordered), max_bullets):
        bullets[i] = np.asarray(_pad_bullet(), dtype=np.float32)
        pad[i] = True
    return player, bullets, pad
