"""Opt-in Python Gv36: integer player lattice, inherited Gv20 world.

Use ``env = Gv36Env(seed=7); env.reset(position=(160, 126))``.
Only DT=1/60 is supported. P coordinates are cached Python floats rounded once
with f32, evaluated in float64 as (A + B / sqrt(2)) / 10, where
A=10*origin+17*cardinal and B=17*diagonal. This canonical operation order
makes equivalent integer keys share caches; cache reads never evaluate it.
Counts accumulate signed steps, never float positions. Clamp tests the raw
proposal, resets only out-of-bounds axes to integer wall origins, then commits.
Cardinal counts are rebased to 0..9 with divmod(count, 10), adding 17
times the quotient to the algebraic origin; canonical keys stay unchanged.
The immutable P snapshot includes velocity but is NOT a world checkpoint:
search must separately preserve clocks, bullets, RNGs, S and event stream.
No integer features are added to observations. Moving trajectories need not
match Gv20. S and bullet algorithms remain inherited floating point physics.
"""
from dataclasses import dataclass
import math

from qrokkun_env import constants as C
from qrokkun_env.godot_rng import f32
from qrokkun_env.gv20 import Gv20Env


@dataclass(frozen=True)
class PlayerState:
    x0: int
    x1: int
    x2: int
    y0: int
    y1: int
    y2: int
    px: float
    py: float
    pvx: float
    pvy: float


class Gv36Env(Gv20Env):
    def __init__(self, seed=None, max_bullets_obs=32, dt=C.DT):
        self.dt = dt
        self.set_player_position(160, 126)
        # The legacy dataclass writes px/py defaults. Only during this call
        # may those assignments be ignored; the prepared lattice stays valid.
        self._legacy_position_write = True
        try:
            super().__init__(seed=seed, max_bullets_obs=max_bullets_obs, dt=dt)
        finally:
            self._legacy_position_write = False

    @property
    def dt(self):
        return C.DT

    @dt.setter
    def dt(self, value):
        if value != C.DT:
            raise ValueError('Gv36 requires DT=1/60')

    @property
    def px(self):
        return self._player.px

    @px.setter
    def px(self, value):
        if not getattr(self, '_legacy_position_write', False):
            raise AttributeError('use set_player_position or restore_player_state')

    @property
    def py(self):
        return self._player.py

    @py.setter
    def py(self, value):
        if not getattr(self, '_legacy_position_write', False):
            raise AttributeError('use set_player_position or restore_player_state')

    x0 = property(lambda self: self._player.x0)
    x1 = property(lambda self: self._player.x1)
    x2 = property(lambda self: self._player.x2)
    y0 = property(lambda self: self._player.y0)
    y1 = property(lambda self: self._player.y1)
    y2 = property(lambda self: self._player.y2)

    @property
    def position_key(self):
        return (10*self.x0+17*self.x1, 17*self.x2,
                10*self.y0+17*self.y1, 17*self.y2)

    @staticmethod
    def _coordinate(origin, cardinal, diagonal):
        return (10*origin + 17*cardinal + (17*diagonal)/math.sqrt(2)) / 10

    @staticmethod
    def _normalize_axis(origin, cardinal, diagonal):
        quotient, cardinal = divmod(cardinal, 10)
        return origin + 17*quotient, cardinal, diagonal

    def set_player_position(self, x, y):
        if type(x) is not int or type(y) is not int:
            raise ValueError('player origins must be Python integers')
        if not (C.FIELD_X+C.PLAYER_MARGIN <= x <= C.FIELD_X+C.FIELD_W-C.PLAYER_MARGIN
                and C.FIELD_Y+C.PLAYER_MARGIN <= y <= C.FIELD_Y+C.FIELD_H-C.PLAYER_MARGIN):
            raise ValueError('player position outside legal rectangle')
        self._player = PlayerState(x, 0, 0, y, 0, 0, f32(x), f32(y), 0., 0.)
        self.pvx = self.pvy = 0.

    def reset(self, seed=None, *, position=(160, 126)):
        self.set_player_position(*position)
        self._legacy_position_write = True
        try:
            return super().reset(seed)
        finally:
            self._legacy_position_write = False

    def get_player_state(self):
        """Immutable P-only snapshot; no world/RNG checkpoint is implied."""
        p = self._player
        return PlayerState(p.x0, p.x1, p.x2, p.y0, p.y1, p.y2,
                           p.px, p.py, self.pvx, self.pvy)

    def restore_player_state(self, state):
        counts = (state.x0, state.x1, state.x2, state.y0, state.y1, state.y2)
        if any(type(v) is not int for v in counts):
            raise ValueError('player lattice requires Python integers')
        x = self._coordinate(*counts[:3])
        y = self._coordinate(*counts[3:])
        if not (C.FIELD_X+C.PLAYER_MARGIN <= x <= C.FIELD_X+C.FIELD_W-C.PLAYER_MARGIN
                and C.FIELD_Y+C.PLAYER_MARGIN <= y <= C.FIELD_Y+C.FIELD_H-C.PLAYER_MARGIN):
            raise ValueError('player position outside legal rectangle')
        if (state.px, state.py) != (f32(x), f32(y)):
            raise ValueError('inconsistent player cache')
        if not all(math.isfinite(v) and f32(v) == v for v in (state.pvx, state.pvy)):
            raise ValueError('velocity must be finite float32')
        counts = (self._normalize_axis(*counts[:3])
                  + self._normalize_axis(*counts[3:]))
        self._player = PlayerState(*counts, state.px, state.py, state.pvx, state.pvy)
        self.pvx, self.pvy = state.pvx, state.pvy

    def _integrate_player(self, dx, dy):
        diagonal = bool(dx and dy)
        axes = []
        for origin, cardinal, diag, direction, lower, upper in (
            (self.x0, self.x1, self.x2, dx, C.FIELD_X+C.PLAYER_MARGIN,
             C.FIELD_X+C.FIELD_W-C.PLAYER_MARGIN),
            (self.y0, self.y1, self.y2, dy, C.FIELD_Y+C.PLAYER_MARGIN,
             C.FIELD_Y+C.FIELD_H-C.PLAYER_MARGIN),
        ):
            sign = (direction > 0) - (direction < 0)
            cardinal += sign if not diagonal else 0
            diag += sign if diagonal else 0
            raw = self._coordinate(origin, cardinal, diag)
            if raw < lower or raw > upper:
                origin, cardinal, diag = int(lower if raw < lower else upper), 0, 0
                raw = float(origin)
            axes.append((*self._normalize_axis(origin, cardinal, diag), f32(raw)))
        x, y = axes
        vx, vy = f32(dx*C.PLAYER_MAX_SPEED), f32(dy*C.PLAYER_MAX_SPEED)
        self._player = PlayerState(*x[:3], *y[:3], x[3], y[3], vx, vy)
        self.pvx, self.pvy = vx, vy
