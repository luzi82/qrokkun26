"""Spawner v1 — 288-way discrete (edge×along×aim), 8-dim obs, fixed medium kind.

From `train_spawner_gpu.py` / both_v1 era. Checkpoint: baseline_v1/spawner_gpu.pt
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from qrokkun_env import constants as C
from qrokkun_env.env import Bullet, Qrokkun26Env, _bullet_speed
from qrokkun_env.godot_rng import f32

N_EDGE, N_ALONG, N_AIM = 4, 8, 9
SPAWNER_ACTIONS = N_EDGE * N_ALONG * N_AIM  # 288
SPAWNER_OBS_DIM = 8


def decode_spawner(a: int) -> tuple[int, int, int]:
    aim = a % N_AIM
    a //= N_AIM
    along = a % N_ALONG
    edge = a // N_ALONG
    return edge, along, aim


def vectorize_spawner(env: Qrokkun26Env) -> list[float]:
    return [
        (env.px - (C.FIELD_X + C.FIELD_W * 0.5)) / (C.FIELD_W * 0.5),
        (env.py - (C.FIELD_Y + C.FIELD_H * 0.5)) / (C.FIELD_H * 0.5),
        env.pvx / C.PLAYER_MAX_SPEED,
        env.pvy / C.PLAYER_MAX_SPEED,
        min(env.elapsed / 60.0, 2.0),
        env.spawn_acc,
        min(len(env.bullets) / 40.0, 1.0),
        _bullet_speed(env.elapsed) / 125.0,
    ]


def spawn_from_action(env: Qrokkun26Env, action: int) -> None:
    edge, along, aim = decode_spawner(int(action))
    fx0, fy0 = C.FIELD_X, C.FIELD_Y
    fx1, fy1 = C.FIELD_X + C.FIELD_W, C.FIELD_Y + C.FIELD_H
    t = (along + 0.5) / N_ALONG
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
    else:
        ang = (aim - 1) * (math.tau / 8.0)
        miss = 18.0
        dx = (tx + math.cos(ang) * miss) - pos[0]
        dy = (ty + math.sin(ang) * miss) - pos[1]
    speed = _bullet_speed(env.elapsed)
    n = math.hypot(dx, dy) or 1.0
    vx, vy = f32(dx / n * speed), f32(dy / n * speed)
    kind = 1
    env.bullets.append(Bullet(f32(pos[0]), f32(pos[1]), vx, vy, kind, C.BULLET_RADIUS[kind]))


class SpawnerV1(nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(SPAWNER_OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, SPAWNER_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        h = self.body(x)
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


SpawnerAC = SpawnerV1
spawner_vectorize = vectorize_spawner
