"""PPO likelihood ratio regression: stored vs recomputed log_prob at temp=1.0."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.train import both_v4
from qrokkun_env.train.both_v4 import act_player, act_spawner


def test_temp_argparse_defaults_are_one() -> None:
    src = inspect.getsource(both_v4.main)
    # Both player and spawner temps default to 1.0 (exploration via sampling+entropy).
    assert '"--temp-p"' in src and '"--temp-s"' in src
    assert "default=1.0" in src
    assert "default=1.15" not in src
    assert "default=1.2" not in src
    assert "sampling+entropy" in src


def test_player_ppo_ratio_before_update_is_one() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = PlayerV4(d_model=32, hidden=64).to(device)
    net.eval()

    stored_lps = []
    actions = []
    players = []
    bullets = []
    pads = []

    for seed in range(4):
        env = Qrokkun26Env(seed=seed)
        env.reset()
        a, lp, _v, p, b, m = act_player(net, env, device, sample=True, temp=1.0)
        stored_lps.append(lp)
        actions.append(a)
        players.append(p)
        bullets.append(b)
        pads.append(m)

    P = torch.tensor(np.array(players), dtype=torch.float32, device=device)
    B = torch.tensor(np.array(bullets), dtype=torch.float32, device=device)
    M = torch.tensor(np.array(pads), dtype=torch.bool, device=device)
    A = torch.tensor(actions, dtype=torch.int64, device=device)
    old = torch.tensor(stored_lps, dtype=torch.float32, device=device)

    # Recompute like ppo_update_player (no temp scaling) before any optimizer step.
    with torch.no_grad():
        dist, _value = net(P, B, M)
        lp = dist.log_prob(A)
        ratio = torch.exp(lp - old)

    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), ratio.tolist()


def test_spawner_ppo_ratio_before_update_is_one() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    net = SpawnerV4(d_model=32, hidden=64).to(device)
    net.eval()

    stored_lps = []
    births = []
    aims = []
    kinds = []
    players = []
    bullets = []
    pads = []

    for seed in range(4):
        env = Qrokkun26Env(seed=seed)
        env.reset()
        act, lp, _v, p, b, m = act_spawner(net, env, device, sample=True, temp=1.0)
        stored_lps.append(lp)
        births.append(act["birth"])
        aims.append(act["aim"])
        kinds.append(act["kind"])
        players.append(p)
        bullets.append(b)
        pads.append(m)

    P = torch.tensor(np.array(players), dtype=torch.float32, device=device)
    B = torch.tensor(np.array(bullets), dtype=torch.float32, device=device)
    M = torch.tensor(np.array(pads), dtype=torch.bool, device=device)
    birth_a = torch.tensor(births, dtype=torch.float32, device=device)
    aim_a = torch.tensor(aims, dtype=torch.float32, device=device)
    kind_a = torch.tensor(kinds, dtype=torch.int64, device=device)
    old = torch.tensor(stored_lps, dtype=torch.float32, device=device)

    # Recompute like ppo_update_spawner (no temp scaling) before any optimizer step.
    with torch.no_grad():
        birth, aim, kind, _value = net(P, B, M)
        lp = (
            birth.log_prob(birth_a).sum(-1)
            + aim.log_prob(aim_a).sum(-1)
            + kind.log_prob(kind_a)
        )
        ratio = torch.exp(lp - old)

    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), ratio.tolist()
