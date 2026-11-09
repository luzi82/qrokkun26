import math
import random
import pytest


def test_schedule_reset_and_field_windows():
    from qrokkun_env.gv20 import Gv20Env, spawn_frames
    assert list(spawn_frames(105))[:5] == [0, 27, 53, 79, 104]
    e = Gv20Env(seed=42)
    e.reset()
    assert (e.px, e.py) == (160, 126)
    assert len(e.bullets) == 1
    assert e.s_history[0][0] == 0
    e._check_hit = lambda: False
    for k in range(1, 181):
        before = len(e.field_history)
        e.step(0)
        assert len(e.field_history) - before == int(k % 60 == 0)
        if k % 60 == 0:
            speeds = [s for f, s in e.s_history if k-60 <= f < k]
            assert e.field_history[-1][1] == pytest.approx(sum(speeds)/len(speeds))
    assert all(30 <= s <= 200 for _, s in e.s_history)
    assert all(b.radius == 2.6 and not hasattr(b, 'kind') for b in e.bullets)
    b = e.bullets[0]
    v = b.vx, b.vy
    e.step(0)
    assert (b.vx, b.vy) == v


def test_geometry_and_aim():
    from qrokkun_env.gv20 import ray_position, aimed_velocity, learned_aim_delta, learned_speed, AIM_SIGMA
    assert AIM_SIGMA == 13.2
    for theta, pos in [(0, (320,126)), (math.pi/2,(160,240)), (math.pi,(0,126))]:
        assert ray_position(theta) == pytest.approx(pos)
    spawn, player = (0,12), (160,126)
    vx, vy = aimed_velocity(spawn, player, (0,0), 200)
    miss = abs((player[0]-spawn[0])*vy-(player[1]-spawn[1])*vx)/200
    assert miss <= 6.6
    vx, vy = aimed_velocity(spawn, player, (-1140,1600), 200)
    assert abs(160*vy-114*vx)/200 > 6.6
    assert aimed_velocity(player, player, (0,0), 30) == (30,0)
    for s in [-10, 0, 10]:
        delta = learned_aim_delta(s, (3,4))
        assert math.hypot(*delta) == pytest.approx(math.hypot(304,212)*(math.tanh(s)+1)/2)
        assert math.hypot(*delta) <= math.hypot(304,212)
    assert learned_aim_delta(0,(0,0)) == (math.hypot(304,212)/2, 0)
    assert [learned_speed(a) for a in [-1,0,1]] == [30,115,200]


def test_random_streams_and_unbounded_scripted_delta():
    from qrokkun_env.gv20 import Gv20Env
    a, b = Gv20Env(seed=17), Gv20Env(seed=17)
    a.reset(); b.reset()
    a._check_hit = b._check_hit = lambda: False
    for _ in range(180):
        for _ in range(13): b.rng.random()
        a.step(0); b.step(0)
    assert a.bullets == b.bullets
    assert a.s_history == b.s_history and a.field_history == b.field_history
    class Fixed:
        def uniform(self, low, high): return 0 if low == 0 else 100
        def gauss(self, mu, sigma): return 1000
    a.s_rng = Fixed()
    a._spawn_one()
    bullet = a.bullets[-1]
    assert bullet.vy > 0  # unclipped target is below right-hand spawn


def test_observation_and_network():
    import torch
    from qrokkun_env.gv20 import Gv20Env, Bullet
    from qrokkun_ai.v21.obs_v21 import encode_obs_v21, batch_kinematic_clearances_v21
    from qrokkun_ai.v21.agents.player_v21 import PlayerV21
    e = Gv20Env(); e.reset()
    e.bullets = [Bullet(161,126,200,0), Bullet(159,126,-200,0)]
    p,b,m = encode_obs_v21(e)
    assert b.shape == (64,4) and m.shape == (64,)
    assert list(b[:2,2]) == [1,-1] and not m[:2].any() and m[2:].all()
    assert (b[2:,2:] != 0).all()
    pt,bt,mt = torch.tensor(p)[None],torch.tensor(b)[None],torch.tensor(m)[None]
    net = PlayerV21(hidden=16)
    dist,val = net(pt,bt,mt)
    assert dist.logits.shape == (1,9) and torch.isfinite(val).all()
    assert batch_kinematic_clearances_v21(pt,bt,mt,horizon=1).shape == (1,9)
    e.bullets *= 40
    assert encode_obs_v21(e)[1].shape == (64,4)


def test_trainer_defaults_and_checkpoint_boundary(tmp_path):
    from qrokkun_ai.v10.train.train_player_v10 import build_parser as old_parser
    from qrokkun_ai.v21.train.train_player_v21 import build_parser, Gv20Env
    from qrokkun_ai.v10.agents.player_v10 import PlayerV10, save_v10_checkpoint
    from qrokkun_ai.v21.agents.player_v21 import load_v21_checkpoint, CheckpointV21Error
    old, new = old_parser().parse_args([]), build_parser().parse_args([])
    for key in ['gamma','lam','dagger_weight','ppo_weight','entropy_coef','episodes_per_update','max_steps','top_k','horizon']:
        assert getattr(old,key) == getattr(new,key)
    path = tmp_path/'old.pt'
    save_v10_checkpoint(PlayerV10(hidden=8), path)
    with pytest.raises(CheckpointV21Error, match='architecture'): load_v21_checkpoint(path)


def test_old_env_keeps_multiple_kinds():
    from qrokkun_env.env import Qrokkun26Env
    from qrokkun_ai.v10.obs_v10 import BULLET_FEAT_V10
    assert BULLET_FEAT_V10 == 9
    e = Qrokkun26Env(seed=17); e.reset()
    e._check_hit = lambda: False
    kinds = set()
    for _ in range(7200):
        e.step(0)
        kinds.update(b.kind for b in e.bullets)
    assert len(kinds) > 1
