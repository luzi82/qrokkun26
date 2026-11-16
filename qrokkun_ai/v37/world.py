"""Torch-free learned Gv36 and explicit full-world forks."""
import copy
from qrokkun_env.gv36 import Gv36Env

class LearnedEnv(Gv36Env):
    def __init__(self, callback, **kwargs):
        self.callback = callback
        Gv36Env.__init__(self, **kwargs)

    def _spawn_one(self):
        from qrokkun_ai.v24.core import LearnedEnv as LearnedS
        return LearnedS._spawn_one(self)


def snapshot(env):
    # Never deepcopy a live callback, model or Python generator.
    return {k: copy.deepcopy(v) for k, v in vars(env).items()
            if k not in ('callback', '_events', 'capture')}


def restore(state, callback):
    env = LearnedEnv.__new__(LearnedEnv)
    env.__dict__.update(copy.deepcopy(state))
    env.callback = callback
    env._events = env._event_stream()
    while next(env._events) != env._next_event:
        pass
    return env
