"""Actual Gv36 physics with an independent TRAIN40 birth calendar."""
from qrokkun_env.gv36 import Gv36Env
from qrokkun_env.gv20 import ray_position
from qrokkun_env.env import _spawn_interval
from qrokkun_env.constants import DT
import math

class Gv39Env(Gv36Env):
    def __init__(self, *args, difficulty_offset=40, **kwargs):
        if difficulty_offset not in (0, 40):
            raise ValueError('only VAL0 and TRAIN40 are defined')
        self.difficulty_offset = difficulty_offset
        super().__init__(*args, **kwargs)

    @property
    def difficulty_elapsed(self):
        return self.difficulty_offset + self.elapsed

    def _event_stream(self):
        if not self.difficulty_offset:
            yield from super()._event_stream()
            return
        yield 0, 0.0
        frame, elapsed, acc = 0, 0.0, 0.0
        while True:
            frame += 1
            elapsed += DT
            acc += DT
            interval = _spawn_interval(40 + elapsed)
            while acc >= interval:
                acc -= interval
                yield frame, acc
                interval = _spawn_interval(40 + elapsed)

    def _spawn_step(self):
        if not self.difficulty_offset:
            return super()._spawn_step()
        # Keep the observable residual current, including frames without births.
        if self.frame == 0:
            self.spawn_acc = 0.0
        else:
            self.spawn_acc += DT
        while self.frame == self._next_event[0]:
            self.spawn_acc = self._next_event[1]
            self._spawn_one()
            self._next_event = next(self._events)
        if self.frame > 0 and self.frame % 60 == 0:
            speeds = [s for f,s in self.s_history if self.frame-60 <= f < self.frame]
            if speeds:
                speed = sum(speeds)/len(speeds)
                self._add_bullet(ray_position(self.field_rng.uniform(0, math.tau)), (0,0), speed)
                self.field_history.append((self.frame, speed))
