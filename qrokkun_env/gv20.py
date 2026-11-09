"""Gv20 rules. The legacy environment and its physics remain unchanged."""
from dataclasses import dataclass
import math
import random

from qrokkun_env import constants as C
from qrokkun_env.env import Qrokkun26Env, _spawn_interval

AIM_SIGMA = 13.2
BULLET_RADIUS = 2.6


def spawn_frames(stop):
    """Replay the old accumulator, shifting its first event to frame zero."""
    elapsed, acc, frame, first = 0.0, C.SPAWN_ACC_START, 0, None
    while first is None or frame - first < stop:
        frame += 1
        elapsed += C.DT
        acc += C.DT
        interval = _spawn_interval(elapsed)
        while acc >= interval:
            acc -= interval
            if first is None:
                first = frame
            if frame - first < stop:
                yield frame - first
            interval = _spawn_interval(elapsed)


def ray_position(theta):
    x, y = math.cos(theta), math.sin(theta)
    distance = min(160 / abs(x) if x else math.inf,
                   114 / abs(y) if y else math.inf)
    return 160 + distance*x, 126 + distance*y


def aimed_velocity(spawn, player, delta, speed):
    x, y = player[0]+delta[0]-spawn[0], player[1]+delta[1]-spawn[1]
    n = math.hypot(x, y)
    if n < 1e-12:
        return speed, 0.0
    return speed*x/n, speed*y/n


def learned_speed(a):
    return 115 + 85*a


def learned_aim_delta(s, direction):
    radius = math.hypot(C.FIELD_W, C.FIELD_H)*(math.tanh(s)+1)/2
    x, y = direction
    norm = math.hypot(x,y)
    if norm < 1e-12:
        x,y,norm = 1.0,0.0,1.0
    return radius*x/norm, radius*y/norm


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = BULLET_RADIUS
    moved: bool = False


class Gv20Env(Qrokkun26Env):
    """60 Hz player environment with independent player, S, and field streams."""
    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        episode_seed = 0 if self.seed is None else int(self.seed)
        self.rng = random.Random(f'{episode_seed}:player')
        self.s_rng = random.Random(f'{episode_seed}:S')
        self.field_rng = random.Random(f'{episode_seed}:field')
        self.frame = 0
        self.s_history = []
        self.field_history = []
        self._events = self._event_stream()
        self._next_event = next(self._events)
        super().reset()
        self._spawn_step()
        return self.observe()

    @staticmethod
    def _event_stream():
        # Infinite replay with bounded clock state; no RNG and no double events.
        elapsed, acc, frame, first = 0.0, C.SPAWN_ACC_START, 0, None
        while True:
            frame += 1
            elapsed += C.DT
            acc += C.DT
            interval = _spawn_interval(elapsed)
            while acc >= interval:
                acc -= interval
                if first is None:
                    first = frame
                yield frame-first, acc
                interval = _spawn_interval(elapsed)

    def step(self, action):
        if not self.dead:
            self.frame += 1
        return super().step(action)

    def _spawn_step(self):
        if self.frame == self._next_event[0]:
            self.spawn_acc = self._next_event[1]
            self._spawn_one()
            self._next_event = next(self._events)
        if self.frame > 0 and self.frame % 60 == 0:
            speeds = [s for f,s in self.s_history if self.frame-60 <= f < self.frame]
            if speeds:
                speed = sum(speeds)/len(speeds)
                pos = ray_position(self.field_rng.uniform(0, math.tau))
                self._add_bullet(pos, (0,0), speed)
                self.field_history.append((self.frame, speed))

    def _add_bullet(self, pos, delta, speed):
        vx,vy = aimed_velocity(pos, (self.px,self.py), delta, speed)
        self.bullets.append(Bullet(*pos, vx, vy))

    def _spawn_one(self):
        pos = ray_position(self.s_rng.uniform(0, math.tau))
        delta = self.s_rng.gauss(0,AIM_SIGMA), self.s_rng.gauss(0,AIM_SIGMA)
        speed = self.s_rng.uniform(30,200)
        self._add_bullet(pos,delta,speed)
        self.s_history.append((self.frame,speed))

    def observe(self):
        ordered = sorted(self.bullets, key=lambda b: (b.x-self.px)**2+(b.y-self.py)**2)[:64]
        rows = [[(b.x-self.px)/C.FIELD_W, (b.y-self.py)/C.FIELD_H,
                 b.vx/200, b.vy/200] for b in ordered]
        return {'player':[self.px,self.py,self.pvx,self.pvy], 'elapsed':self.elapsed,
                'dead':self.dead, 'bullet_count':len(self.bullets),
                'bullets':rows + [[0.,0.,2.,2.] for _ in range(64-len(rows))],
                'pad_mask':[False]*len(rows)+[True]*(64-len(rows))}
