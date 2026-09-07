"""Runtime regressions from agents wiring."""

from __future__ import annotations

from dataclasses import fields

import torch

from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.train.player_v0 import PlayerMLP, Rollout, collect_episode
from qrokkun_env.train.player_v1 import ActorCritic, bc_pretrain, shaped_reward
from qrokkun_env.train.spawner_v1 import PlayerAC, player_act


def test_train_player_rollout_is_dataclass() -> None:
    names = {f.name for f in fields(Rollout)}
    assert names == {"log_probs", "rewards", "entropies"}
    env = Qrokkun26Env(seed=0)
    pol = PlayerMLP()
    roll = collect_episode(env, pol, max_steps=5)
    assert isinstance(roll, Rollout)
    assert len(roll.rewards) >= 1


def test_train_player_gpu_helpers_exist() -> None:
    env = Qrokkun26Env(seed=1)
    env.reset(seed=1)
    r = shaped_reward(env, 0.1, done=False)
    assert isinstance(r, float)
    net = ActorCritic(hidden=32)
    loss = bc_pretrain(net, torch.device("cpu"), steps=1, batch=2, lr=1e-3)
    assert loss >= 0.0


def test_spawner_player_act_unpacks_tuple() -> None:
    env = Qrokkun26Env(seed=2)
    env.reset(seed=2)
    player = PlayerAC(hidden=32)
    a = player_act(player, env, torch.device("cpu"))
    assert isinstance(a, int)
    assert 0 <= a < 9


def test_shims_still_import() -> None:
    import qrokkun_env.train_both_v3_gpu as shim
    from qrokkun_env.agents.player_v3 import PlayerV3

    assert shim.PlayerAC is PlayerV3
