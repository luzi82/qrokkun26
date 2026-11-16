from qrokkun_env.gv36 import Gv36Env


def test_reset_center():
    env = Gv36Env(seed=7)
    obs = env.reset()
    assert obs['player'] == [160., 126., 0., 0.]
    assert env.position_key == (1600, 0, 1260, 0)
    assert (env.x0, env.x1, env.x2, env.y0, env.y1, env.y2) == (160, 0, 0, 126, 0, 0)

import copy
from dataclasses import replace
import math
import pytest
from qrokkun_env import constants as C
from qrokkun_env.gv20 import Gv20Env, Bullet, aimed_velocity
from qrokkun_env.godot_rng import f32


def move(env, action):
    env._integrate_player(*env._parse_action(action))


def test_actions_and_velocity():
    e = Gv36Env()
    for a in range(9):
        e.reset()
        move(e, a)
        dx, dy = e._parse_action(a)
        diagonal = dx != 0 and dy != 0
        cx, cy = (0, 0) if diagonal else (int(dx), int(dy))
        assert (e.x0, e.x1, e.y0, e.y1) == (
            160+17*(cx//10), cx % 10, 126+17*(cy//10), cy % 10)
        assert (e.px, e.py) == (
            f32(e._coordinate(160, cx, int(math.copysign(1, dx)) if diagonal else 0)),
            f32(e._coordinate(126, cy, int(math.copysign(1, dy)) if diagonal else 0)))
        assert (e.x2, e.y2) == ((int(math.copysign(1, dx)), int(math.copysign(1, dy))) if diagonal else (0, 0))
        assert (e.pvx, e.pvy) == (f32(dx*102), f32(dy*102))


def test_timestep():
    with pytest.raises(ValueError):
        Gv36Env(dt=.02)
    with pytest.raises(ValueError):
        Gv36Env().dt = .02


def test_reset_integer_and_cleanup():
    e = Gv36Env()
    move(e, 'ne')
    e.reset(position=(30, 40))
    assert (e.x0, e.y0, e.x2, e.y2, e.px, e.py) == (30, 40, 0, 0, 30., 40.)
    with pytest.raises(ValueError):
        e.reset(position=(30.5, 40))


def test_multiset_order():
    a, b = Gv36Env(), Gv36Env()
    actions = ['e', 'ne', 'sw', 's', 'se', 'nw', 'w']*7
    for action in actions:
        move(a, action)
    for action in reversed(actions):
        move(b, action)
    assert a.position_key == b.position_key
    assert (a.px, a.py) == (b.px, b.py)


def test_cancellation():
    e = Gv36Env()
    for a in ['ne']*31+['sw']*31:
        move(e, a)
    assert e.position_key == (1600, 0, 1260, 0)
    assert (e.px, e.py) == (160., 126.)


def test_canonical_equivalence():
    e = Gv36Env()
    s = e.get_player_state()
    e.restore_player_state(replace(s, x0=143, x1=10))
    assert e.position_key == (1600, 0, 1260, 0)
    assert e.px == 160.


def test_axis_clamp():
    e = Gv36Env()
    e.reset(position=(306, 100))
    move(e, 'se')
    assert (e.x0, e.x1, e.x2) == (306, 0, 0)
    assert (e.y0, e.y1, e.y2) == (100, 0, 1)


def test_corner_clamp():
    e = Gv36Env()
    e.reset(position=(14, 26))
    move(e, 'nw')
    assert e.position_key == (140, 0, 260, 0)


def test_raw_inside_rounded_boundary():
    e = Gv36Env()
    # Pell approximation: 665857/470832 > sqrt(2). Negative residual
    # is inside the right wall yet rounds to that wall in float32.
    x0, x1, x2 = -11319263, 0, 9416640
    raw = e._coordinate(x0, x1, x2)
    assert raw < 306 and f32(raw) == 306.
    s = replace(e.get_player_state(), x0=x0, x1=x1, x2=x2, px=f32(raw))
    e.restore_player_state(s)
    move(e, 'idle')
    assert (e.x0, e.x2) == (x0, x2)


def test_exact_boundary_no_reset():
    e = Gv36Env()
    e.restore_player_state(replace(e.get_player_state(), x0=289, x1=10, px=306.))
    # Keep a nontrivial other axis so an unnecessary whole-player reset fails.
    move(e, 's')
    before = e.get_player_state()
    move(e, 'idle')
    assert (e.x0, e.x1, e.x2) == (306, 0, 0)
    assert replace(e.get_player_state(), pvx=before.pvx, pvy=before.pvy) == before
    assert (e.pvx, e.pvy) == (0., 0.)


def test_cache_readonly_and_observation(monkeypatch):
    e = Gv36Env(); e.reset()
    move(e, 'ne')
    xy = e.px, e.py
    def fail(*args):
        raise AssertionError('coordinate evaluated on read')
    monkeypatch.setattr(e, '_coordinate', fail)
    for name in ('px', 'py', 'x1'):
        with pytest.raises(AttributeError):
            setattr(e, name, 12)
    obs = e.observe()
    assert len(obs['bullets']) == 64 and len(obs['player']) == 4
    assert set(obs) == set(Gv20Env().reset())
    e.bullets = [Bullet(xy[0], xy[1], 0, 0)]
    assert e._check_hit()
    e._add_bullet((8., 20.), (0, 0), 100)
    assert (e.bullets[-1].vx, e.bullets[-1].vy) == aimed_velocity((8.,20.), xy, (0,0),100)


def test_snapshot_restore_future_and_clone():
    e = Gv36Env(); move(e, 'ne')
    s = e.get_player_state()
    clone = copy.deepcopy(e)
    for a in ['s', 'w', 'nw']:
        move(e, a)
    end = e.get_player_state()
    e.restore_player_state(s)
    for a in ['s', 'w', 'nw']:
        move(e, a)
    assert e.get_player_state() == end
    assert clone.get_player_state() == s
    with pytest.raises(ValueError):
        e.restore_player_state(replace(s, px=s.px+1))
    assert e.get_player_state() == end


def test_restore_validates_integers_and_bounds():
    e = Gv36Env(); s = e.get_player_state()
    for bad in [replace(s, x1=1.), replace(s, x0=0, px=0.)]:
        with pytest.raises(ValueError):
            e.restore_player_state(bad)
        assert e.get_player_state() == s


def test_inherited_bullet_rules():
    e = Gv36Env()
    b = Bullet(100, 100, 60, 0)
    edge = Bullet(C.BULLET_CULL[1], 100, 60, 0, moved=True)
    e.bullets = [b, edge]
    e._integrate_bullets()
    assert e.bullets == [b] and b.x == 100 and b.moved
    e._integrate_bullets()
    assert b.x == 101
    b.x, b.y = e.px+8, e.py
    b.radius = 4
    assert e._check_hit()
    b.x += .001
    assert not e._check_hit()


def test_stationary_differential():
    a, b = Gv20Env(seed=0), Gv36Env(seed=0)
    assert a.reset() == b.reset()
    for _ in range(60):
        assert a.step('idle') == b.step('idle')
        assert a.bullets == b.bullets
        for name in ('rng', 's_rng', 'field_rng'):
            assert getattr(a, name).getstate() == getattr(b, name).getstate()
        assert (a.frame, a.spawn_acc, a._next_event) == (b.frame, b.spawn_acc, b._next_event)
    assert b.frame == 60 and b.field_history


def test_spawn_reads_custom_reset_and_pre_move_cache(monkeypatch):
    class Probe(Gv36Env):
        def _spawn_one(self):
            coordinate = self._coordinate
            def fail(*args):
                raise AssertionError('spawn evaluated player formula')
            self._coordinate = fail
            try:
                self.aim = self.px, self.py
                super()._spawn_one()
            finally:
                self._coordinate = coordinate
    e = Probe(seed=2)
    e.reset(position=(30, 40))
    assert e.aim == (30., 40.)
    move(e, 'ne'); before = e.px, e.py
    e._next_event = (1, 0.)
    e.step('e')
    assert e.aim == before and e.px != before[0]
