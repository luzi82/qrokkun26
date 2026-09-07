from qrokkun_env.agents.obs_v4 import PAD_RADIUS, encode_obs
from qrokkun_env.agents.spawner_v4 import _ray_edge_spawn, spawn_continuous
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.train import both_v4
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4


def test_pad_uses_negative_radius() -> None:
    env = Qrokkun26Env(seed=0)
    env.reset()
    _p, b, m = encode_obs(env)
    assert bool(m.all())
    assert float(b[0, 4]) == PAD_RADIUS


def test_ray_hits_outside_field() -> None:
    x, y = _ray_edge_spawn(0.0, -1.0)
    assert y < 20.0  # above field top (FIELD_Y=20)


def test_both_v4_aliases() -> None:
    assert both_v4.PlayerAC is PlayerV4
    assert both_v4.SpawnerAC is SpawnerV4


def test_spawn_kind_onehot() -> None:
    env = Qrokkun26Env(seed=3)
    env.reset()
    spawn_continuous(env, (1.0, 0.0), (0.5, -0.2), kind=2, rng_jitter=False)
    _p, b, m = encode_obs(env)
    assert not bool(m[0])
    assert float(b[0, 5 + 2]) == 1.0  # kind2 onehot
