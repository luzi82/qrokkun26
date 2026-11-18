import pytest
from qrokkun_env.gv39 import Gv39Env
from qrokkun_env.gv36 import Gv36Env
from qrokkun_env.constants import DT

def test_hard_calendar_not_rebased_legacy():
    env = Gv39Env(seed=17)
    env.reset()
    assert env.frame == 0 and env.elapsed == 0
    assert len(env.s_history) == 1
    assert env.spawn_acc == 0
    assert env._next_event[0] == 7
    assert env.difficulty_elapsed == 40

def test_zero_offset_delegates_bit_path():
    a, b = Gv39Env(seed=17, difficulty_offset=0), Gv36Env(seed=17)
    assert a.reset() == b.reset()
    for _ in range(65):
        assert a.step(0) == b.step(0)
        assert a.s_history == b.s_history
        assert a.field_history == b.field_history
        assert a.spawn_acc == b.spawn_acc

def test_actual_reward_clock():
    e = Gv39Env(seed=1); e.reset()
    _, reward, _, _ = e.step(0)
    assert reward == DT and e.frame == 1 and e.elapsed == DT
    assert e.difficulty_elapsed == 40 + DT
