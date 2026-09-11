"""v4.7: fraction=0 must never trigger a random reset over a simulated
training schedule; prepare_initial_state must not even be called.

No torch required: only exercises qrokkun_env.reset_modes.resolve_mode /
reset_kind_for_mode against the default BASE_LOOP mix.
"""

from __future__ import annotations

import random

from qrokkun_env.reset_modes import (
    BASE_LOOP,
    RESET_NORMAL,
    reset_kind_for_mode,
    resolve_mode,
)


def test_fraction_zero_never_uses_random_reset() -> None:
    rng = random.Random(123)
    calls = {"prepare_initial_state": 0}

    def fake_prepare_initial_state(*_a, **_kw):
        calls["prepare_initial_state"] += 1

    # Simulate ~50 updates x 16 rollouts, mirroring both_v4.main()'s loop.
    for update in range(50):
        for i in range(16):
            base = BASE_LOOP[i % len(BASE_LOOP)]
            mode = resolve_mode(base, 0.0, rng)
            reset_kind = reset_kind_for_mode(mode)
            assert reset_kind == RESET_NORMAL
            if reset_kind != RESET_NORMAL:
                fake_prepare_initial_state()

    assert calls["prepare_initial_state"] == 0


def test_fraction_zero_preserves_default_mode_mix() -> None:
    rng = random.Random(7)
    seen = set()
    for i in range(len(BASE_LOOP)):
        mode = resolve_mode(BASE_LOOP[i], 0.0, rng)
        seen.add(mode)
    assert seen == {"self_normal", "p_vs_scripted_normal", "s_vs_flee_normal"}


def test_positive_fraction_can_trigger_random_reset_eventually() -> None:
    rng = random.Random(1)
    kinds = {
        reset_kind_for_mode(resolve_mode("self", 1.0, rng)) for _ in range(20)
    }
    # fraction=1.0 on an eligible base mode must always randomize.
    assert kinds == {"random_player"}


def test_s_vs_flee_always_normal_regardless_of_fraction() -> None:
    rng = random.Random(2)
    for frac in (0.0, 0.3, 1.0):
        for _ in range(10):
            mode = resolve_mode("s_vs_flee", frac, rng)
            assert mode == "s_vs_flee_normal"
            assert reset_kind_for_mode(mode) == RESET_NORMAL
