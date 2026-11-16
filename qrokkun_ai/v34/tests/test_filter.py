from types import SimpleNamespace as NS
from qrokkun_env.gv20 import Bullet

def test_fast_receding_unreachable():
    from qrokkun_ai.v34.filter import harmless
    e=NS(px=160.,py=126.,dt=1/60)
    assert harmless(e,Bullet(200.,126.,200.,0.,moved=True))


def test_rectangle_outward_lane():
    from qrokkun_ai.v34.filter import harmless
    # P may already stand at its rightmost legal centre: bullet remains beyond
    # the collision radius and moves vertically, so radial disc alone is loose.
    e=NS(px=306.,py=126.,dt=1/60)
    assert harmless(e,Bullet(320.,126.,0.,10.,moved=True))


def test_cull_before_intercept_and_offscreen_inbound():
    from qrokkun_ai.v34.filter import harmless
    e=NS(px=200.,py=126.,dt=1/60)
    assert harmless(e,Bullet(300.,126.,50.,0.,moved=True))
    assert not harmless(e,Bullet(-10.,126.,200.,0.,moved=True))
    assert not harmless(e,Bullet(230.,126.,10.,0.,moved=True))


def test_birth_delay_and_tangent_retain():
    from qrokkun_ai.v34.filter import harmless
    e=NS(px=160.,py=126.,dt=1/60)
    assert harmless(e,Bullet(168.,126.,200.,0.,moved=True))
    assert not harmless(e,Bullet(168.,126.,200.,0.,moved=False))
    assert not harmless(e,Bullet(166.6,126.,200.,0.,moved=True))


def test_nonfinite_uncertain_retain():
    from qrokkun_ai.v34.filter import harmless
    e=NS(px=160.,py=126.,dt=1/60)
    assert not harmless(e,Bullet(200.,126.,float('nan'),0.,moved=True))
    assert not harmless(e,Bullet(200.,126.,0.,0.,moved=True))
