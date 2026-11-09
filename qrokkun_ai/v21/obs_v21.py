"""v21 observations & zero-inertia discrete-step kinematic clearance features (120s cap)."""

from __future__ import annotations

import math
import numpy as np
import torch

from qrokkun_env import constants as C
from qrokkun_env.env import ACTIONS, ACTION_TO_DIR
from qrokkun_env.gv20 import Gv20Env

MAX_BULLETS_V21 = 64
BULLET_FEAT_V21 = 4
PLAYER_FEAT_V21 = 6  # norm_x, norm_y, left_wall_dist, right_wall_dist, norm_elapsed_120s, spawn_acc
DEFAULT_TOP_K_V21 = 12
DEFAULT_HORIZON_V21 = 12
SAFETY_THRESHOLD_PX = 2.0


def _player_feats_v21(env: Gv20Env) -> list[float]:
    half_w = C.FIELD_W * 0.5
    half_h = C.FIELD_H * 0.5
    cx = C.FIELD_X + half_w
    cy = C.FIELD_Y + half_h
    left_min = C.FIELD_X + C.PLAYER_MARGIN
    right_max = C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN
    return [
        (env.px - cx) / half_w,
        (env.py - cy) / half_h,
        (env.px - left_min) / C.FIELD_W,
        (right_max - env.px) / C.FIELD_W,
        min(env.elapsed / 120.0, 1.0),
        float(env.spawn_acc),
    ]


def _bullet_feat_v21(env: Gv20Env, b) -> list[float]:
    return [(b.x-env.px)/C.FIELD_W, (b.y-env.py)/C.FIELD_H, b.vx/200, b.vy/200]


def _pad_bullet_v21():
    return [0.0, 0.0, 2.0, 2.0]


def order_bullets_v21(env: Gv20Env, max_bullets: int = MAX_BULLETS_V21) -> list:
    """Return bullets nearest-first by Euclidean distance squared, truncated to max_bullets."""
    return sorted(
        env.bullets,
        key=lambda b: (b.x - env.px) ** 2 + (b.y - env.py) ** 2,
    )[:max_bullets]


def encode_obs_v21(
    env: Gv20Env, max_bullets: int = MAX_BULLETS_V21
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (player[6], bullets[64, 4], pad_mask[64]) with pad_mask True = empty slot."""
    player = np.asarray(_player_feats_v21(env), dtype=np.float32)
    max_bullets = min(max_bullets, MAX_BULLETS_V21)
    ordered = order_bullets_v21(env, max_bullets)
    bullets = np.zeros((max_bullets, BULLET_FEAT_V21), dtype=np.float32)
    pad = np.ones((max_bullets,), dtype=np.bool_)
    for i, b in enumerate(ordered):
        bullets[i] = np.asarray(_bullet_feat_v21(env, b), dtype=np.float32)
        pad[i] = False
    for i in range(len(ordered), max_bullets):
        bullets[i] = np.asarray(_pad_bullet_v21(), dtype=np.float32)
        pad[i] = True
    return player, bullets, pad


def build_unit_dirs_v21(device: torch.device) -> torch.Tensor:
    """Return [9, 2] unit direction vectors for ACTIONS under zero-inertia physics."""
    dirs = []
    for a in ACTIONS:
        dx, dy = ACTION_TO_DIR[a]
        if dx != 0.0 or dy != 0.0:
            n = math.hypot(dx, dy)
            dx, dy = dx / n, dy / n
        dirs.append((dx, dy))
    return torch.tensor(dirs, dtype=torch.float32, device=device)


@torch.no_grad()
def batch_kinematic_clearances_v21(
    player: torch.Tensor,
    bullets: torch.Tensor,
    pad: torch.Tensor,
    horizon: int = DEFAULT_HORIZON_V21,
    top_k: int = DEFAULT_TOP_K_V21,
) -> torch.Tensor:
    """Pure PyTorch batched discrete-step clearance over `1..horizon` frames under ZERO-INERTIA physics.

    Args:
        player:  [N, 6] float32
        bullets: [N, K_max, 4] float32
        pad:     [N, K_max] bool (True = empty/padded slot)
        horizon: number of discrete lookahead frames (default 12)
        top_k:   number of nearest bullet slots to evaluate (default 12)

    Returns:
        min_clear: [N, 9] float32 minimum signed clearance in pixels (distance - (r_player + r_bullet)).
                   Empty/all-padded bullet sets return +999.0 for all 9 actions.
    """
    device = player.device
    n_batch = int(player.shape[0])
    k_use = min(int(top_k), int(bullets.shape[1]))
    unit_dirs = build_unit_dirs_v21(device)  # [9, 2]

    half_w = C.FIELD_W * 0.5
    half_h = C.FIELD_H * 0.5
    cx = C.FIELD_X + half_w
    cy = C.FIELD_Y + half_h

    px0 = player[:, 0] * half_w + cx  # [N]
    py0 = player[:, 1] * half_h + cy  # [N]

    b = bullets[:, :k_use, :]  # [N, K, 4]
    live_mask = ~pad[:, :k_use]  # [N, K] True = live bullet
    bx = b[:, :, 0] * C.FIELD_W + px0.unsqueeze(1)  # [N, K]
    by = b[:, :, 1] * C.FIELD_H + py0.unsqueeze(1)  # [N, K]
    bvx = b[:, :, 2] * 200.0  # [N, K]
    bvy = b[:, :, 3] * 200.0  # [N, K]
    br = torch.full_like(b[:, :, 0], 2.6)  # [N, K]

    # Under zero-inertia physics, velocity is instantaneously dir * PLAYER_MAX_SPEED!
    vx_a = unit_dirs[:, 0].unsqueeze(0) * C.PLAYER_MAX_SPEED  # [1, 9]
    vy_a = unit_dirs[:, 1].unsqueeze(0) * C.PLAYER_MAX_SPEED  # [1, 9]

    px = px0.unsqueeze(1).expand(n_batch, len(ACTIONS)).clone()  # [N, 9]
    py = py0.unsqueeze(1).expand(n_batch, len(ACTIONS)).clone()  # [N, 9]

    x_min = C.FIELD_X + C.PLAYER_MARGIN
    x_max = C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN
    y_min = C.FIELD_Y + C.PLAYER_MARGIN
    y_max = C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN

    min_clear = torch.full((n_batch, len(ACTIONS)), 999.0, dtype=torch.float32, device=device)
    if k_use == 0 or int(horizon) <= 0:
        return min_clear

    for _ in range(int(horizon)):
        px = torch.clamp(px + vx_a * C.DT, x_min, x_max)
        py = torch.clamp(py + vy_a * C.DT, y_min, y_max)

        bx = bx + bvx * C.DT
        by = by + bvy * C.DT

        dx = bx.unsqueeze(1) - px.unsqueeze(2)  # [N, 9, K]
        dy = by.unsqueeze(1) - py.unsqueeze(2)  # [N, 9, K]
        d = torch.hypot(dx, dy) - (C.PLAYER_RADIUS + br.unsqueeze(1))
        d = torch.where(live_mask.unsqueeze(1), d, torch.full_like(d, 999.0))
        step_min = d.min(dim=2).values  # [N, 9]
        min_clear = torch.minimum(min_clear, step_min)

    return min_clear


@torch.no_grad()
def compute_kinematic_expert_targets(
    player: torch.Tensor,
    bullets: torch.Tensor,
    pad: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    horizon: int = DEFAULT_HORIZON_V21,
    top_k: int = DEFAULT_TOP_K_V21,
    safety_margin: float = SAFETY_THRESHOLD_PX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute zero-inertia kinematic expert target actions [N] and target logits [N, 9]."""
    device = player.device
    clearances = batch_kinematic_clearances_v21(
        player, bullets, pad, horizon=horizon, top_k=top_k
    )  # [N, 9]
    unit_dirs = build_unit_dirs_v21(device)  # [9, 2]

    if base_logits is None:
        # Gentle center-seeking preference + clearance bonus when away from center
        # player[:, 0:2] is normalized offset from center in [-1, 1]
        norm_x = player[:, 0].unsqueeze(1)  # [N, 1]
        norm_y = player[:, 1].unsqueeze(1)  # [N, 1]
        ux = unit_dirs[:, 0].unsqueeze(0)  # [1, 9]
        uy = unit_dirs[:, 1].unsqueeze(0)  # [1, 9]
        # Dot product towards center (-norm_x, -norm_y)
        center_pull = -(norm_x * ux + norm_y * uy) * 1.5
        cl_bonus = torch.clamp(clearances, min=-10.0, max=28.0) * 0.25
        base = center_pull + cl_bonus
    else:
        base = base_logits + torch.clamp(clearances, min=0.0, max=20.0) * 0.10

    deficit = torch.clamp(safety_margin - clearances, min=0.0)
    target_logits = base - 8.0 * deficit

    safe_mask = clearances >= safety_margin  # [N, 9]
    masked_base = torch.where(safe_mask, base, torch.full_like(base, -1e9))
    has_safe = safe_mask.any(dim=-1)
    best_safe_act = masked_base.argmax(dim=-1)
    most_clear_act = clearances.argmax(dim=-1)
    target_actions = torch.where(has_safe, best_safe_act, most_clear_act)
    return target_actions, target_logits
