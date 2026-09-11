"""v4.7: prepare_initial_state — thin near-center random-Player reset.

No torch required: only touches qrokkun_env.env / qrokkun_env.reset_modes.
"""

from __future__ import annotations

import math
import random

from qrokkun_env import constants as C
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.reset_modes import RANDOM_PLAYER_RADIUS, prepare_initial_state


def test_prepare_initial_state_empty_field_and_zero_velocity() -> None:
    from qrokkun_env.env import Bullet

    env = Qrokkun26Env(seed=1)
    # Pollute state first to prove prepare_initial_state resets it.
    env.reset(seed=1)
    env.bullets.append(Bullet(1.0, 1.0, 2.0, 2.0, 0, 3.0))
    env.pvx, env.pvy = 5.0, -3.0
    env.elapsed = 12.0
    env.dead = True

    obs = prepare_initial_state(env, seed=1, rng=random.Random(0))

    assert env.bullets == []
    assert obs["bullets"]  # padded slots still present
    assert env.pvx == 0.0 and env.pvy == 0.0
    assert env.dead is False
    assert env.elapsed == 0.0


def test_prepare_initial_state_position_near_center_not_exact() -> None:
    env = Qrokkun26Env(seed=2)
    cx = C.FIELD_X + C.FIELD_W * 0.5
    cy = C.FIELD_Y + C.FIELD_H * 0.5

    saw_off_center = False
    for i in range(20):
        prepare_initial_state(env, seed=2, rng=random.Random(i))
        d = math.hypot(env.px - cx, env.py - cy)
        assert d <= RANDOM_PLAYER_RADIUS + 1e-6
        assert C.FIELD_X + C.PLAYER_MARGIN <= env.px <= C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN
        assert C.FIELD_Y + C.PLAYER_MARGIN <= env.py <= C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN
        if d > 1e-6:
            saw_off_center = True
    assert saw_off_center  # random, not pinned to exact center every draw


def test_prepare_initial_state_does_not_spawn_bullets() -> None:
    env = Qrokkun26Env(seed=3)
    for i in range(10):
        prepare_initial_state(env, seed=3, rng=random.Random(i))
        assert env.bullets == []
        obs = env.observe()
        assert obs["bullet_count"] == 0
