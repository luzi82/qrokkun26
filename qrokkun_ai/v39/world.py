"""Explicit dual-clock views and full-world forks, without model copying."""
import copy
from qrokkun_env.gv39 import Gv39Env
from qrokkun_ai.v24.core import LearnedEnv as LearnedS

class View:
    __slots__ = ('_env', '_kind')
    def __init__(self, env, kind):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_kind', kind)
    def __setattr__(self, key, value):
        raise AttributeError('observation views are read only')
    def __getattr__(self, key):
        env = self._env
        if key == 'frame' and self._kind == 'S':
            return env.frame + env.difficulty_offset * 60
        if key == 'elapsed' and self._kind == 'P':
            return env.difficulty_elapsed
        return getattr(env, key)

class LearnedEnv(Gv39Env):
    def __init__(self, callback, **kwargs):
        self.callback = callback
        super().__init__(**kwargs)
    def _spawn_one(self):
        # Existing S decoder and observation body; only the clock is viewed.
        callback = self.callback
        def shifted(obs):
            obs[4] = (self.frame + self.difficulty_offset * 60) / 7200
            return callback(obs)
        self.callback = shifted
        try:
            return LearnedS._spawn_one(self)
        finally:
            self.callback = callback

class CaptureEnv(LearnedEnv):
    def __init__(self, callback, **kwargs):
        self.capture = None
        super().__init__(callback, **kwargs)
    def step(self, action):
        if self.capture is not None:
            self.capture.before(self, action)
        return super().step(action)

def snapshot(env):
    return {k:copy.deepcopy(v) for k,v in vars(env).items()
            if k not in ('callback', '_events', 'capture')}

def restore(state, callback):
    env = LearnedEnv.__new__(LearnedEnv)
    env.__dict__.update(copy.deepcopy(state))
    env.callback = callback
    env._events = env._event_stream()
    while next(env._events) != env._next_event:
        pass
    return env
