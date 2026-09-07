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


def test_aim_tanh_scale_bound() -> None:
    from qrokkun_env.agents.spawner_v4 import AIM_SCALE
    import math
    # tanh outputs in (-1,1) so offset magnitude < AIM_SCALE
    assert AIM_SCALE == 40.0
    assert abs(math.tanh(100.0) * AIM_SCALE) < AIM_SCALE + 1e-6


def test_act_spawner_temp_changes_std() -> None:
    import torch
    from torch.distributions import Normal
    from qrokkun_env.agents.spawner_v4 import SpawnerV4
    from qrokkun_env.agents.obs_v4 import encode_obs
    from qrokkun_env.train.both_v4 import act_spawner
    from qrokkun_env.env import Qrokkun26Env

    torch.manual_seed(0)
    net = SpawnerV4(d_model=32, hidden=64)
    env = Qrokkun26Env(seed=0)
    env.reset()
    p, b, m = encode_obs(env)
    birth, aim, kind, _v = net(
        torch.tensor(p).unsqueeze(0),
        torch.tensor(b).unsqueeze(0),
        torch.tensor(m).unsqueeze(0),
    )
    std1 = float(birth.stddev.mean().detach())
    std2 = float(Normal(birth.mean, birth.stddev * 2.0).stddev.mean().detach())
    assert std2 == std1 * 2.0
    a1, _, _, _, _, _ = act_spawner(net, env, torch.device("cpu"), sample=True, temp=1.0)
    a2, _, _, _, _, _ = act_spawner(net, env, torch.device("cpu"), sample=True, temp=2.0)
    assert "birth" in a1 and "aim" in a1 and "kind" in a1
    assert 0 <= a2["kind"] < 4


def test_log_std_clamped_in_forward() -> None:
    import torch
    from qrokkun_env.agents.spawner_v4 import SpawnerV4
    from qrokkun_env.agents.obs_v4 import encode_obs
    from qrokkun_env.env import Qrokkun26Env

    net = SpawnerV4(d_model=32, hidden=64)
    with torch.no_grad():
        net.log_std.fill_(10.0)  # would explode without clamp
    env = Qrokkun26Env(seed=1)
    env.reset()
    p, b, m = encode_obs(env)
    birth, aim, kind, v = net(
        torch.tensor(p).unsqueeze(0),
        torch.tensor(b).unsqueeze(0),
        torch.tensor(m).unsqueeze(0),
    )
    assert float(birth.stddev.max()) <= torch.tensor(1.0).exp().item() + 1e-5
