from dataclasses import replace

from qrokkun_env.gv36 import Gv36Env


def test_restore_negative_count_preserves_cache_and_key():
    e = Gv36Env()
    old = replace(e.get_player_state(), x1=-1, px=158.3000030517578)
    e.restore_player_state(old)
    s = e.get_player_state()
    assert 0 <= s.x1 <= 9
    assert (s.x0, s.x1, s.x2) == (143, 9, 0)
    assert (s.px, s.py, s.pvx, s.pvy) == (old.px, old.py, old.pvx, old.pvy)
    assert e.position_key == (1583, 0, 1260, 0)


def test_carries_axes_diagonal_and_equivalent_formula():
    from qrokkun_env.godot_rng import f32
    e = Gv36Env()
    for counts in [(143, 10, 3, 160, -21, -2),
                   (126, 20, 3, 109, 9, -2)]:
        x0, x1, x2, y0, y1, y2 = counts
        x, y = e._coordinate(*counts[:3]), e._coordinate(*counts[3:])
        old = replace(e.get_player_state(), x0=x0, x1=x1, x2=x2,
                      y0=y0, y1=y1, y2=y2, px=f32(x), py=f32(y), pvx=102.)
        e.restore_player_state(old)
        s = e.get_player_state()
        assert (s.x0, s.x1, s.x2) == (160, 0, 3)
        assert (s.y0, s.y1, s.y2) == (109, 9, -2)
        assert e._coordinate(s.x0, s.x1, s.x2) == x
        assert e._coordinate(s.y0, s.y1, s.y2) == y
        assert (s.px, s.py, s.pvx, s.pvy) == (old.px, old.py, old.pvx, old.pvy)


def test_outside_origin_actual_bounds_and_axis_clamp():
    from qrokkun_env.godot_rng import f32
    e = Gv36Env()
    raw = e._coordinate(-3, 10, 0)
    e.restore_player_state(replace(e.get_player_state(), x0=-3, x1=10, px=f32(raw)))
    assert (e.x0, e.x1, e.px) == (14, 0, 14.)
    # A legal actual position can have an origin below the rectangle.
    raw = e._coordinate(-3, 9, 2)
    e.restore_player_state(replace(e.get_player_state(), x0=-3, x1=9, x2=2, px=f32(raw)))
    e._integrate_player(0, 1)
    assert (e.x0, e.x1, e.x2) == (-3, 9, 2)
    e._integrate_player(-1, 0)
    assert (e.x0, e.x1, e.x2, e.px) == (14, 0, 0, 14.)
    assert (e.y1, e.pvy) == (1, 0.)


def test_old_snapshot_roundtrip_clone_and_invalid_restore():
    import copy
    import pytest
    e = Gv36Env()
    old = replace(e.get_player_state(), x0=143, x1=10)
    e.restore_player_state(old)
    normalized = e.get_player_state()
    clone = copy.deepcopy(e)
    e._integrate_player(-1, 0)
    assert clone.get_player_state() == normalized
    e.restore_player_state(normalized)
    assert e.get_player_state() == normalized
    for bad in (replace(old, x1=10.), replace(old, px=159.),
                replace(old, pvx=float('nan')), replace(old, x0=1000, px=1717.)):
        with pytest.raises(ValueError):
            e.restore_player_state(bad)
        assert e.get_player_state() == normalized
