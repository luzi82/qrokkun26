"""Qrokkun26Env — gym-style reset/step with scripted spawner (Godot parity)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from qrokkun_env import constants as C
from qrokkun_env.godot_rng import GodotRNG, f32

# Discrete actions matching keyboard 8-way + idle.
ACTIONS = (
    "idle",
    "n",
    "ne",
    "e",
    "se",
    "s",
    "sw",
    "w",
    "nw",
)
ACTION_TO_DIR = {
    "idle": (0.0, 0.0),
    "n": (0.0, -1.0),
    "ne": (1.0, -1.0),
    "e": (1.0, 0.0),
    "se": (1.0, 1.0),
    "s": (0.0, 1.0),
    "sw": (-1.0, 1.0),
    "w": (-1.0, 0.0),
    "nw": (-1.0, -1.0),
}


def _spawn_interval(elapsed: float) -> float:
    t = elapsed
    if t < 8.0:
        return 0.48 + (0.32 - 0.48) * (t / 8.0)
    if t < 20.0:
        return 0.32 + (0.20 - 0.32) * ((t - 8.0) / 12.0)
    if t < 35.0:
        return 0.20 + (0.12 - 0.20) * ((t - 20.0) / 15.0)
    if t < 55.0:
        return 0.12 + (0.075 - 0.12) * ((t - 35.0) / 20.0)
    if t < 80.0:
        return 0.075 + (0.048 - 0.075) * ((t - 55.0) / 25.0)
    return 0.042


def _bullet_speed(elapsed: float) -> float:
    return 46.0 + (125.0 - 46.0) * min(max(elapsed / 70.0, 0.0), 1.0)


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    kind: int
    radius: float
    # Godot: Area2D added mid-physics does not run _physics_process that frame.
    moved: bool = False


@dataclass
class Qrokkun26Env:
    """Scripted-spawner environment. Player action is one of ACTIONS (or int index)."""

    seed: int | None = None
    max_bullets_obs: int = 32
    dt: float = C.DT
    rng: GodotRNG = field(init=False)
    elapsed: float = 0.0
    spawn_acc: float = C.SPAWN_ACC_START
    px: float = 0.0
    py: float = 0.0
    pvx: float = 0.0
    pvy: float = 0.0
    dead: bool = False
    bullets: list[Bullet] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = GodotRNG(0 if self.seed is None else int(self.seed))

    # -- API -----------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed
            self.rng = GodotRNG(int(seed))
        self.elapsed = 0.0
        self.spawn_acc = C.SPAWN_ACC_START
        self.px = C.FIELD_X + C.FIELD_W * 0.5
        self.py = C.FIELD_Y + C.FIELD_H * 0.5
        self.pvx = 0.0
        self.pvy = 0.0
        self.dead = False
        self.bullets.clear()
        return self.observe()

    def step(self, action: int | str) -> tuple[dict, float, bool, dict]:
        if self.dead:
            return self.observe(), 0.0, True, {"elapsed": self.elapsed}

        # Godot tree order: Main (elapsed + spawn) before Player, then Bullets.
        # Aiming therefore uses the pre-move player position this frame.
        dx, dy = self._parse_action(action)
        self.elapsed += self.dt
        self._spawn_step()
        self._integrate_player(dx, dy)
        self._integrate_bullets()
        hit = self._check_hit()
        if hit:
            self.dead = True
            obs = self.observe()
            return obs, self.dt, True, {"elapsed": self.elapsed, "hit": True}
        return self.observe(), self.dt, False, {"elapsed": self.elapsed}

    # Padded observation slots use radius < 0 so they are not live hits at (0,0).
    PAD_RADIUS = -1.0

    def observe(self) -> dict:
        # Nearest bullets by distance to player (stable for RL).
        ordered = sorted(
            self.bullets,
            key=lambda b: (b.x - self.px) ** 2 + (b.y - self.py) ** 2,
        )[: self.max_bullets_obs]
        bullet_flat: list[float] = []
        for b in ordered:
            bullet_flat.extend([b.x, b.y, b.vx, b.vy, b.radius, float(b.kind)])
        while len(bullet_flat) < self.max_bullets_obs * 6:
            # x,y,vx,vy,radius,kind — radius sentinel marks unused slot.
            bullet_flat.extend([0.0, 0.0, 0.0, 0.0, self.PAD_RADIUS, -1.0])
        return {
            "player": [self.px, self.py, self.pvx, self.pvy],
            "elapsed": self.elapsed,
            "dead": self.dead,
            "bullet_count": len(self.bullets),
            "bullets": bullet_flat,
        }

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _parse_action(action: int | str) -> tuple[float, float]:
        if isinstance(action, int):
            if not 0 <= action < len(ACTIONS):
                raise ValueError(f"action index out of range: {action}")
            action = ACTIONS[action]
        if action not in ACTION_TO_DIR:
            raise ValueError(f"unknown action: {action!r}")
        dx, dy = ACTION_TO_DIR[action]
        if dx != 0.0 or dy != 0.0:
            n = math.hypot(dx, dy)
            dx, dy = dx / n, dy / n
        return dx, dy

    def _integrate_player(self, dx: float, dy: float) -> None:
        if dx == 0.0 and dy == 0.0:
            self.pvx = 0.0
            self.pvy = 0.0
        else:
            self.pvx = f32(dx * C.PLAYER_MAX_SPEED)
            self.pvy = f32(dy * C.PLAYER_MAX_SPEED)
        self.px = f32(self.px + f32(self.pvx * self.dt))
        self.py = f32(self.py + f32(self.pvy * self.dt))
        self.px = f32(
            min(
                max(self.px, C.FIELD_X + C.PLAYER_MARGIN),
                C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN,
            )
        )
        self.py = f32(
            min(
                max(self.py, C.FIELD_Y + C.PLAYER_MARGIN),
                C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN,
            )
        )

    def _spawn_step(self) -> None:
        self.spawn_acc += self.dt
        interval = _spawn_interval(self.elapsed)
        while self.spawn_acc >= interval:
            self.spawn_acc -= interval
            self._spawn_one()
            if self.elapsed > 18.0 and self.rng.randf() < 0.16:
                self._spawn_one()
            interval = _spawn_interval(self.elapsed)

    def _spawn_one(self) -> None:
        edge = self.rng.randi_range(0, 3)
        fx0, fy0 = C.FIELD_X, C.FIELD_Y
        fx1, fy1 = C.FIELD_X + C.FIELD_W, C.FIELD_Y + C.FIELD_H
        if edge == 0:
            pos = (self.rng.randf_range(fx0, fx1), fy0 - 8.0)
        elif edge == 1:
            pos = (self.rng.randf_range(fx0, fx1), fy1 + 8.0)
        elif edge == 2:
            pos = (fx0 - 8.0, self.rng.randf_range(fy0, fy1))
        else:
            pos = (fx1 + 8.0, self.rng.randf_range(fy0, fy1))

        speed = _bullet_speed(self.elapsed) * self.rng.randf_range(0.88, 1.14)
        aim_p = min(max(0.22 + self.elapsed * 0.006, 0.22), 0.55)
        if self.rng.randf() < aim_p:
            miss = self.rng.randf_range(8.0, 34.0)
            ang = f32(self.rng.randf() * math.tau)
            tx = f32(self.px + f32(math.cos(ang) * miss))
            ty = f32(self.py + f32(math.sin(ang) * miss))
            dx, dy = f32(tx - pos[0]), f32(ty - pos[1])
        else:
            cx = fx0 + C.FIELD_W * 0.5
            cy = fy0 + C.FIELD_H * 0.5
            dx, dy = cx - pos[0], cy - pos[1]
            n = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n, dy / n
            jitter = math.radians(self.rng.randf_range(-50.0, 50.0))
            cos_j, sin_j = math.cos(jitter), math.sin(jitter)
            dx, dy = dx * cos_j - dy * sin_j, dx * sin_j + dy * cos_j
        n = f32(math.hypot(dx, dy) or 1.0)
        vx, vy = f32(dx / n * speed), f32(dy / n * speed)

        roll = self.rng.randf()
        if self.elapsed > 25.0 and roll < 0.12:
            kind = 2
            vx = f32(vx * 0.62)
            vy = f32(vy * 0.62)
        elif roll < 0.18:
            kind = 3
            vx = f32(vx * 1.15)
            vy = f32(vy * 1.15)
        elif roll < 0.55:
            kind = 1
        else:
            kind = 0
        self.bullets.append(
            Bullet(f32(pos[0]), f32(pos[1]), vx, vy, kind, C.BULLET_RADIUS[kind])
        )

    def _integrate_bullets(self) -> None:
        x0, x1, y0, y1 = C.BULLET_CULL
        alive: list[Bullet] = []
        for b in self.bullets:
            if b.moved:
                b.x = f32(b.x + f32(b.vx * self.dt))
                b.y = f32(b.y + f32(b.vy * self.dt))
            else:
                b.moved = True
            if x0 <= b.x <= x1 and y0 <= b.y <= y1:
                alive.append(b)
        self.bullets = alive

    def _check_hit(self) -> bool:
        pr = C.PLAYER_RADIUS
        for b in self.bullets:
            if (b.x - self.px) ** 2 + (b.y - self.py) ** 2 <= (pr + b.radius) ** 2:
                return True
        return False
