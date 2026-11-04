"""Smoke tests for Qrokkun26Env."""

from __future__ import annotations

import math
import struct

from qrokkun_env import ACTIONS, Qrokkun26Env
from qrokkun_env import constants as C
from qrokkun_env.env import _spawn_interval
from qrokkun_env.godot_rng import f32


def test_actions_count():
    assert len(ACTIONS) == 9


def test_reset_centers_player():
    env = Qrokkun26Env(seed=1)
    obs = env.reset()
    assert abs(obs["player"][0] - (C.FIELD_X + C.FIELD_W * 0.5)) < 1e-6
    assert abs(obs["player"][1] - (C.FIELD_Y + C.FIELD_H * 0.5)) < 1e-6
    assert obs["dead"] is False
    assert obs["elapsed"] == 0.0


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_instantaneous_velocity_no_inertia():
    """One frame reaches full speed in every direction; reverse has no coast."""
    diag = f32(102.0 / math.sqrt(2.0))
    # Godot Vector2(±1, ±1).normalized() * 102 and f32((±1/hypot(1,1))*102)
    # share IEEE-754 binary32 ±72.12489318847656 (0x42903ff2 / 0xc2903ff2).
    assert diag == f32((1.0 / math.hypot(1.0, 1.0)) * 102.0)
    assert _f32_bits(diag) == 0x42903FF2
    assert _f32_bits(f32(-diag)) == 0xC2903FF2
    expected = {
        "n": (0.0, -102.0),
        "ne": (diag, f32(-diag)),
        "e": (102.0, 0.0),
        "se": (diag, diag),
        "s": (0.0, 102.0),
        "sw": (f32(-diag), diag),
        "w": (-102.0, 0.0),
        "nw": (f32(-diag), f32(-diag)),
    }
    assert len(expected) == 8
    for action, (ex, ey) in expected.items():
        env = Qrokkun26Env(seed=0)
        env.reset()
        assert env.pvx == 0.0 and env.pvy == 0.0
        env.step(action)
        assert env.pvx == ex, (action, env.pvx, ex)
        assert env.pvy == ey, (action, env.pvy, ey)
        assert _f32_bits(env.pvx) == _f32_bits(ex)
        assert _f32_bits(env.pvy) == _f32_bits(ey)

    env = Qrokkun26Env(seed=0)
    env.reset()
    env.step("e")
    assert env.pvx == 102.0
    env.step("w")
    assert env.pvx == -102.0
    assert env.pvy == 0.0


def test_idle_snap_stop():
    env = Qrokkun26Env(seed=2)
    env.reset()
    env.step("e")
    assert env.pvx != 0.0 or env.pvy != 0.0
    env.step("idle")
    assert env.pvx == 0.0 and env.pvy == 0.0


def test_spawn_interval_decreases():
    assert _spawn_interval(0.0) > _spawn_interval(30.0) > _spawn_interval(90.0)


def test_seeded_run_is_deterministic():
    def run(seed: int) -> list[float]:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        times = []
        for i in range(180):
            action = ACTIONS[i % len(ACTIONS)]
            obs, reward, done, info = env.step(action)
            times.append(info["elapsed"])
            if done:
                break
        return times

    assert run(42) == run(42)


def test_hit_terminates():
    env = Qrokkun26Env(seed=0)
    env.reset()
    # Force a bullet on top of the player.
    from qrokkun_env.env import Bullet

    env.bullets = [Bullet(env.px, env.py, 0.0, 0.0, 0, 2.0)]
    obs, reward, done, info = env.step("idle")
    assert done is True
    assert env.dead is True
    assert reward == C.DT


def test_survives_a_few_seconds_moving():
    env = Qrokkun26Env(seed=7)
    env.reset(seed=7)
    done = False
    for _ in range(120):  # 2 seconds
        obs, reward, done, info = env.step("n")
        if done:
            break
    # Not asserting survival — only that API stays healthy.
    assert "elapsed" in info
    assert math.isfinite(obs["player"][0])


def test_observation_padding_is_not_a_fake_bullet():
    """Empty / padded bullet slots must not look like a live bullet at (0,0).

    observe() currently zero-fills the flat bullet vector, so an empty field
    reads as 32 bullets at the origin with radius 0 — bad for a small MLP.
    Prefer a mask channel, or a sentinel (e.g. radius < 0) on padded slots.
    """
    env = Qrokkun26Env(seed=0)
    env.reset()
    env.bullets.clear()
    obs = env.observe()
    assert obs["bullet_count"] == 0
    if "bullet_mask" in obs:
        assert sum(obs["bullet_mask"]) == 0
        return
    # No mask: every padded slot's radius (index 4 in each 6-float record) must
    # be a sentinel, not a real non-negative hit radius.
    for i in range(0, len(obs["bullets"]), 6):
        radius = obs["bullets"][i + 4]
        assert radius < 0.0, (i, obs["bullets"][i : i + 6])

def test_frame_order_spawn_before_player_move():
    """Godot: Main spawn uses pre-move player pos; player moves after."""
    env = Qrokkun26Env(seed=0)
    env.reset(seed=0)
    env.spawn_acc = 100.0  # guarantee at least one spawn this frame
    before = (env.px, env.py)

    # Capture aim target by wrapping _spawn_one.
    aims: list[tuple[float, float]] = []
    real = env._spawn_one

    def wrapped():
        # At spawn time player must still be at `before`.
        aims.append((env.px, env.py))
        return real()

    env._spawn_one = wrapped  # type: ignore[method-assign]
    env.step("e")
    assert aims, "expected spawn"
    assert aims[0] == before
    assert (env.px, env.py) != before

