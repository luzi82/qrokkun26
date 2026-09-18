"""Smoke tests for player MLP training pieces (no long train)."""

from __future__ import annotations

from dataclasses import fields

import pytest

torch = pytest.importorskip("torch")

from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_ai.v0.train.player_v0 import (
    PlayerMLP,
    Rollout,
    collect_episode,
    discount_returns,
)


def test_train_player_rollout_is_dataclass() -> None:
    names = {field.name for field in fields(Rollout)}
    assert names == {"log_probs", "rewards", "entropies"}
    env = Qrokkun26Env(seed=0)
    policy = PlayerMLP()
    rollout = collect_episode(env, policy, max_steps=5)
    assert isinstance(rollout, Rollout)
    assert len(rollout.rewards) >= 1


def test_policy_forward_and_one_episode():
    torch.manual_seed(0)
    policy = PlayerMLP(hidden=32)
    env = Qrokkun26Env(seed=1)
    env.reset(seed=1)
    rollout = collect_episode(env, policy, max_steps=30)
    assert len(rollout.rewards) >= 1
    assert len(rollout.log_probs) == len(rollout.rewards)
    G = discount_returns(rollout.rewards, 0.99)
    assert len(G) == len(rollout.rewards)
    loss = torch.stack([-lp * g for lp, g in zip(rollout.log_probs, G)]).sum()
    loss.backward()
    assert policy.body[0].weight.grad is not None
    assert policy.policy.out_features == len(ACTIONS)


def test_rollout_stores_categorical_entropy():
    """Entropy bonus must use Categorical.entropy(), not -log π(a_t).

    -mean(log_prob(sample)) is a biased stand-in: its gradient pushes the
    sampled action's logit the opposite way from maximizing entropy when the
    policy is peaked. collect_episode should keep per-step dist.entropy().
    """
    torch.manual_seed(0)
    policy = PlayerMLP(hidden=32)
    env = Qrokkun26Env(seed=2)
    env.reset(seed=2)
    rollout = collect_episode(env, policy, max_steps=16)
    assert hasattr(rollout, "entropies"), "Rollout missing entropies"
    assert len(rollout.entropies) == len(rollout.rewards)
    for H, lp in zip(rollout.entropies, rollout.log_probs):
        assert torch.is_tensor(H)
        # For a Categorical, entropy is not identically -log π(a).
        assert not torch.allclose(H, -lp), (float(H), float(-lp))
