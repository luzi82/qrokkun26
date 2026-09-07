"""Spawner v4 — continuous birth dir (2) + tanh aim offset (2) + kind (4).

Birth: unit direction from field center intersects outer edge (+8px).
Aim: target = player_pos + AIM_SCALE * tanh(raw_offset).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from qrokkun_env import constants as C
from qrokkun_env.agents.encoder_v4 import BulletSetEncoder
from qrokkun_env.agents.obs_v4 import N_KIND, encode_obs
from qrokkun_env.env import Bullet, Qrokkun26Env, _bullet_speed
from qrokkun_env.godot_rng import f32

AIM_SCALE = 40.0  # px after tanh
N_CONT = 4  # birth_x, birth_y, aim_dx, aim_dy


def _ray_edge_spawn(dx: float, dy: float) -> tuple[float, float]:
    """From field center along (dx,dy), hit expanded rect (field + 8px)."""
    cx = C.FIELD_X + C.FIELD_W * 0.5
    cy = C.FIELD_Y + C.FIELD_H * 0.5
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    # expanded bounds
    x0, y0 = C.FIELD_X - 8.0, C.FIELD_Y - 8.0
    x1, y1 = C.FIELD_X + C.FIELD_W + 8.0, C.FIELD_Y + C.FIELD_H + 8.0
    # distances to each side along ray (positive only)
    cands: list[tuple[float, float, float]] = []
    if ux > 1e-9:
        t = (x1 - cx) / ux
        y = cy + t * uy
        if y0 <= y <= y1 and t > 0:
            cands.append((t, x1, y))
    if ux < -1e-9:
        t = (x0 - cx) / ux
        y = cy + t * uy
        if y0 <= y <= y1 and t > 0:
            cands.append((t, x0, y))
    if uy > 1e-9:
        t = (y1 - cy) / uy
        x = cx + t * ux
        if x0 <= x <= x1 and t > 0:
            cands.append((t, x, y1))
    if uy < -1e-9:
        t = (y0 - cy) / uy
        x = cx + t * ux
        if x0 <= x <= x1 and t > 0:
            cands.append((t, x, y0))
    if not cands:
        return cx, y0  # fallback top
    _t, x, y = min(cands, key=lambda z: z[0])
    return x, y


def spawn_continuous(
    env: Qrokkun26Env,
    birth_xy: tuple[float, float],
    aim_raw: tuple[float, float],
    kind: int,
    *,
    rng_jitter: bool = True,
) -> None:
    bx, by = birth_xy
    if abs(bx) + abs(by) < 1e-6:
        bx, by = 0.0, -1.0
    if rng_jitter:
        bx += env.rng.randf_range(-0.05, 0.05)
        by += env.rng.randf_range(-0.05, 0.05)
    sx, sy = _ray_edge_spawn(bx, by)
    ox = math.tanh(aim_raw[0]) * AIM_SCALE
    oy = math.tanh(aim_raw[1]) * AIM_SCALE
    if rng_jitter:
        ox += env.rng.randf_range(-2.0, 2.0)
        oy += env.rng.randf_range(-2.0, 2.0)
    tx = env.px + ox
    ty = env.py + oy
    dx, dy = tx - sx, ty - sy
    speed = _bullet_speed(env.elapsed)
    if rng_jitter:
        speed *= env.rng.randf_range(0.92, 1.08)
    kind = int(kind) % N_KIND
    if kind == 2:
        speed *= 0.62
    elif kind == 3:
        speed *= 1.15
    n = math.hypot(dx, dy) or 1.0
    vx, vy = f32(dx / n * speed), f32(dy / n * speed)
    env.bullets.append(Bullet(f32(sx), f32(sy), vx, vy, kind, C.BULLET_RADIUS[kind]))


class SpawnerV4(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = BulletSetEncoder(d_model=d_model)
        self.body = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.birth_mean = nn.Linear(hidden, 2)
        self.aim_mean = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((N_CONT,), -0.5))
        self.kind = nn.Linear(hidden, N_KIND)
        self.value = nn.Linear(hidden, 1)

    def forward(
        self, player: torch.Tensor, bullets: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[Normal, Normal, Categorical, torch.Tensor]:
        h = self.body(self.encoder(player, bullets, pad_mask))
        std = self.log_std.clamp(-3.0, 1.0).exp().expand(h.shape[0], -1)
        birth = Normal(self.birth_mean(h), std[:, 0:2])
        aim = Normal(self.aim_mean(h), std[:, 2:4])
        kind = Categorical(logits=self.kind(h))
        value = self.value(h).squeeze(-1)
        return birth, aim, kind, value


def obs_tensors(env, device: torch.device):
    p, b, m = encode_obs(env)
    return (
        torch.tensor(p, dtype=torch.float32, device=device),
        torch.tensor(b, dtype=torch.float32, device=device),
        torch.tensor(m, dtype=torch.bool, device=device),
    )


SpawnerAC = SpawnerV4
