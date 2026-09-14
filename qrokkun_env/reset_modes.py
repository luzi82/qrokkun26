"""v4.7 reset-mode plumbing: thin near-center random-Player reset + explicit
training-mode split (mechanism only — see docs/journal/1788798436_grokbot.txt
v4.7 section). Enabling ``random_fraction > 0`` as a new training baseline is
v4.8 and out of scope here; this module only provides the mechanism, default off.

Modes
-----
self_normal            — self-play (P vs learned S), normal reset. Trains both.
self_random_player      — self-play, near-center random-Player reset. Trains
                          Player ONLY; Spawner trajectory is never packed.
p_vs_scripted_normal    — Player vs scripted Spawner, normal reset. Trains
                          Player only (no learned Spawner in play).
p_vs_scripted_random    — same matchup, near-center random-Player reset.
s_vs_flee_normal        — learned Spawner vs scripted Flee Player. ALWAYS
                          normal reset regardless of ``random_fraction``.

No module here spawns bullets, moves the field center, or performs any
dynamics burn-in — that is explicitly deferred to v5.
"""

from __future__ import annotations

import math
import random as _random

from qrokkun_env import constants as C
from qrokkun_env.env import Qrokkun26Env

# Small radius (px) inside the field for the random-Player start; well within
# the playable bounds (min half-extent minus player margin) for any field size.
RANDOM_PLAYER_RADIUS = 40.0

MODE_SELF_NORMAL = "self_normal"
MODE_SELF_RANDOM_PLAYER = "self_random_player"
MODE_P_VS_SCRIPTED_NORMAL = "p_vs_scripted_normal"
MODE_P_VS_SCRIPTED_RANDOM = "p_vs_scripted_random"
MODE_S_VS_FLEE_NORMAL = "s_vs_flee_normal"

ALL_MODES = (
    MODE_SELF_NORMAL,
    MODE_SELF_RANDOM_PLAYER,
    MODE_P_VS_SCRIPTED_NORMAL,
    MODE_P_VS_SCRIPTED_RANDOM,
    MODE_S_VS_FLEE_NORMAL,
)

# Modes that must ALWAYS use normal reset, irrespective of random_fraction.
ALWAYS_NORMAL_MODES = frozenset({
    MODE_SELF_NORMAL,
    MODE_P_VS_SCRIPTED_NORMAL,
    MODE_S_VS_FLEE_NORMAL,
})

# Modes eligible to be swapped to a random-Player reset when random_fraction>0.
RANDOMIZABLE_MODES = frozenset({MODE_SELF_RANDOM_PLAYER, MODE_P_VS_SCRIPTED_RANDOM})

# Base training loop (v4.6 mix, unchanged): default proportions preserved.
BASE_LOOP = ("self", "self", "p_vs_scripted", "s_vs_flee", "self", "s_vs_flee")

RESET_NORMAL = "normal"
RESET_RANDOM_PLAYER = "random_player"


def is_random_mode(mode: str) -> bool:
    return mode in RANDOMIZABLE_MODES


def trains_spawner(mode: str) -> bool:
    """Whether this mode's Spawner trajectory should be packed for PPO.

    Only modes with a learned Spawner in play AND always-normal reset train
    the Spawner: self_normal, s_vs_flee_normal. self_random_player trains
    Player only (Spawner traj is not packed even though the learned Spawner
    still plays the opponent role in-episode).
    """
    return mode in (MODE_SELF_NORMAL, MODE_S_VS_FLEE_NORMAL)


def trains_player(mode: str) -> bool:
    """Whether this mode's Player trajectory should be packed for PPO."""
    return mode != MODE_S_VS_FLEE_NORMAL


def uses_scripted_spawner(mode: str) -> bool:
    """True if the opponent Spawner this episode is the scripted one (no net)."""
    return mode in (MODE_P_VS_SCRIPTED_NORMAL, MODE_P_VS_SCRIPTED_RANDOM)


def uses_flee_player(mode: str) -> bool:
    return mode == MODE_S_VS_FLEE_NORMAL


def resolve_mode(base: str, random_fraction: float, rng: _random.Random | None = None) -> str:
    """Map a BASE_LOOP entry ("self" / "p_vs_scripted" / "s_vs_flee") to a
    concrete v4.7 mode name, deciding normal vs random-Player reset.

    fraction<=0 always returns the *_normal variant (env.reset() path, as
    today) — prepare_initial_state is never called. s_vs_flee is ALWAYS
    s_vs_flee_normal regardless of fraction.
    """
    if base == "s_vs_flee":
        return MODE_S_VS_FLEE_NORMAL
    if base == "self":
        normal, randomv = MODE_SELF_NORMAL, MODE_SELF_RANDOM_PLAYER
    elif base == "p_vs_scripted":
        normal, randomv = MODE_P_VS_SCRIPTED_NORMAL, MODE_P_VS_SCRIPTED_RANDOM
    else:
        raise ValueError(f"unknown base loop mode: {base!r}")
    if random_fraction is None or float(random_fraction) <= 0.0:
        return normal
    r = rng if rng is not None else _random
    return randomv if r.random() < float(random_fraction) else normal


def reset_kind_for_mode(mode: str) -> str:
    """'normal' or 'random_player' reset kind implied by a resolved mode name."""
    if mode in ALWAYS_NORMAL_MODES:
        return RESET_NORMAL
    if mode in RANDOMIZABLE_MODES:
        return RESET_RANDOM_PLAYER
    raise ValueError(f"unknown mode: {mode!r}")


def prepare_initial_state(
    env: Qrokkun26Env,
    *,
    seed: int | None = None,
    radius: float = RANDOM_PLAYER_RADIUS,
    rng: _random.Random | None = None,
) -> dict:
    """v4.7 thin initial-state prep: near-center random Player position, an
    EMPTY field (no bullets), pvx=pvy=0. Does NOT spawn bullets and does NOT
    run any dynamics burn-in (v5).

    Reuses Qrokkun26Env.reset() for elapsed/spawn_acc/rng/bullets/dead/velocity
    reset (identical to normal reset), then overrides only the Player position
    to a uniformly random point within ``radius`` of the field center.
    """
    env.reset(seed=seed)
    r = rng if rng is not None else _random
    cx = C.FIELD_X + C.FIELD_W * 0.5
    cy = C.FIELD_Y + C.FIELD_H * 0.5
    max_r = max(
        min(
            float(radius),
            C.FIELD_W * 0.5 - C.PLAYER_MARGIN,
            C.FIELD_H * 0.5 - C.PLAYER_MARGIN,
        ),
        0.0,
    )
    ang = r.uniform(0.0, 2.0 * math.pi)
    rad = r.uniform(0.0, max_r)
    env.px = cx + math.cos(ang) * rad
    env.py = cy + math.sin(ang) * rad
    env.pvx = 0.0
    env.pvy = 0.0
    return env.observe()
