"""Spawner v3 — 1152-way discrete + rich 390-dim obs (shared with player_v3).

From `train_both_v3_gpu.py`. Checkpoint: both_v3_spawner.pt
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env import constants as C
from qrokkun_env.env import Bullet, Qrokkun26Env, _bullet_speed
from qrokkun_env.godot_rng import f32
from qrokkun_env.obs_rich import OBS_DIM_RICH, vectorize_rich

N_EDGE, N_ALONG, N_AIM, N_KIND = 4, 8, 9, 4
SPAWNER_ACTIONS = N_EDGE * N_ALONG * N_AIM * N_KIND  # 1152
OBS_DIM_V3 = OBS_DIM_RICH
vectorize = vectorize_rich


def decode_spawner(a: int) -> tuple[int, int, int, int]:
    a = int(a)
    kind = a % N_KIND
    a //= N_KIND
    aim = a % N_AIM
    a //= N_AIM
    along = a % N_ALONG
    edge = a // N_ALONG
    return edge, along, aim, kind


def spawn_from_action(env: Qrokkun26Env, action: int, *, rng_jitter: bool = True) -> None:
    edge, along, aim, kind = decode_spawner(action)
    fx0, fy0 = C.FIELD_X, C.FIELD_Y
    fx1, fy1 = C.FIELD_X + C.FIELD_W, C.FIELD_Y + C.FIELD_H
    t = (along + 0.5) / N_ALONG
    if rng_jitter:
        t = min(max(t + env.rng.randf_range(-0.04, 0.04), 0.02), 0.98)
    if edge == 0:
        pos = (fx0 + (fx1 - fx0) * t, fy0 - 8.0)
    elif edge == 1:
        pos = (fx0 + (fx1 - fx0) * t, fy1 + 8.0)
    elif edge == 2:
        pos = (fx0 - 8.0, fy0 + (fy1 - fy0) * t)
    else:
        pos = (fx1 + 8.0, fy0 + (fy1 - fy0) * t)

    lead = 0.15
    tx = env.px + env.pvx * lead
    ty = env.py + env.pvy * lead
    if aim == 0:
        dx, dy = tx - pos[0], ty - pos[1]
        if rng_jitter:
            jitter = env.rng.randf_range(-0.2, 0.2)
            cos_j, sin_j = math.cos(jitter), math.sin(jitter)
            dx, dy = dx * cos_j - dy * sin_j, dx * sin_j + dy * cos_j
    else:
        ang = (aim - 1) * (math.tau / 8.0)
        miss = 18.0
        if rng_jitter:
            miss = env.rng.randf_range(12.0, 26.0)
            ang += env.rng.randf_range(-0.25, 0.25)
        dx = (tx + math.cos(ang) * miss) - pos[0]
        dy = (ty + math.sin(ang) * miss) - pos[1]

    speed = _bullet_speed(env.elapsed)
    if rng_jitter:
        speed *= env.rng.randf_range(0.92, 1.08)
    if kind == 2:
        speed *= 0.62
    elif kind == 3:
        speed *= 1.15
    n = math.hypot(dx, dy) or 1.0
    vx, vy = f32(dx / n * speed), f32(dy / n * speed)
    env.bullets.append(Bullet(f32(pos[0]), f32(pos[1]), vx, vy, kind, C.BULLET_RADIUS[kind]))


class SpawnerV3(nn.Module):
    def __init__(self, hidden: int = 512, n_actions: int = SPAWNER_ACTIONS) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM_V3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


SpawnerAC = SpawnerV3
