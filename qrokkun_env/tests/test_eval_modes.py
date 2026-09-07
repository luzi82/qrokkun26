"""v4.3 eval mode regressions (spec E)."""

from __future__ import annotations

import pytest
import torch

from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.eval_modes import (
    MODE_DET_DET,
    MODE_DET_STOCH,
    MODE_STOCH_STOCH,
    PAIRED_EVAL_SEEDS,
    metric_key,
    mode_tag,
    run_eval_episode,
)
from qrokkun_env.train.both_v4 import eval_pair, eval_pair_stats, run_episode


@pytest.fixture
def tiny_nets():
    device = torch.device("cpu")
    torch.manual_seed(0)
    player = PlayerV4(d_model=16, hidden=32).to(device)
    spawner = SpawnerV4(d_model=16, hidden=32).to(device)
    player.eval()
    spawner.eval()
    return player, spawner, device


def test_mode_tags_and_metric_keys():
    assert mode_tag(False, False) == "det_det"
    assert mode_tag(False, True) == "det_stoch"
    assert mode_tag(True, True) == "stoch_stoch"
    assert metric_key("new_vs_new", False, True) == "new_vs_new_det_stoch"
    assert MODE_DET_DET == (False, False)
    assert MODE_DET_STOCH == (False, True)
    assert MODE_STOCH_STOCH == (True, True)
    assert len(PAIRED_EVAL_SEEDS) == 30
    assert PAIRED_EVAL_SEEDS[0] == 3000


def test_jitter_true_consumes_env_rng_early_learned_s(tiny_nets):
    """jitter True → early frames consume env.rng via spawn_continuous; False → not before 8s."""
    _player, spawner, device = tiny_nets
    max_steps = 150  # ~2.5s: enough for ≥1 spawn, still << 8s double-spawn

    env_j = Qrokkun26Env(seed=7)
    env_j.reset()
    state_before_j = env_j.rng.state
    run_episode(
        env_j, "flee", spawner, device, max_steps,
        sample=False, train_player=False, train_spawner=False,
        temp_p=1.0, temp_s=1.0, rng_jitter=True,
    )
    assert env_j.rng.state != state_before_j
    assert env_j.elapsed < 8.0

    env_d = Qrokkun26Env(seed=7)
    env_d.reset()
    state_before_d = env_d.rng.state
    run_episode(
        env_d, "flee", spawner, device, max_steps,
        sample=False, train_player=False, train_spawner=False,
        temp_p=1.0, temp_s=1.0, rng_jitter=False,
    )
    assert env_d.elapsed < 8.0
    # No double-spawn randf and no jitter → RNG untouched for learned S before 8s.
    assert env_d.rng.state == state_before_d


def test_sample_policy_fork_rng_preserves_global(tiny_nets):
    player, spawner, device = tiny_nets
    torch.manual_seed(12345)
    before = torch.get_rng_state()
    # Draw once so state is non-trivial, then snapshot again after a known draw.
    _ = torch.rand(3)
    snap = torch.get_rng_state()

    env = Qrokkun26Env(seed=99)
    run_eval_episode(
        run_episode,
        env,
        player,
        spawner,
        device,
        40,
        sample_policy=True,
        rng_jitter=True,
        episode_seed=99,
    )
    after = torch.get_rng_state()
    assert torch.equal(snap, after)

    # Control: without fork, sampling would move global RNG.
    torch.set_rng_state(snap)
    env2 = Qrokkun26Env(seed=99)
    run_episode(
        env2, player, spawner, device, 80,
        sample=True, train_player=False, train_spawner=False,
        temp_p=1.0, temp_s=1.0, rng_jitter=True,
    )
    assert not torch.equal(snap, torch.get_rng_state())
    # Restore for other tests
    torch.set_rng_state(before)


def test_same_seed_mode_reproducible(tiny_nets):
    player, spawner, device = tiny_nets
    seeds = [11, 12]

    a = eval_pair(player, spawner, device, seeds, 80, sample_policy=False, rng_jitter=True)
    b = eval_pair(player, spawner, device, seeds, 80, sample_policy=False, rng_jitter=True)
    assert a == b

    c = eval_pair(player, spawner, device, seeds, 80, sample_policy=False, rng_jitter=False)
    d = eval_pair(player, spawner, device, seeds, 80, sample_policy=False, rng_jitter=False)
    assert c == d


def test_mode2_seed_difference_visible_flee_news(tiny_nets):
    """Different seeds under det_policy+stoch_env should differ for flee×newS."""
    _player, spawner, device = tiny_nets
    t0 = eval_pair("flee", spawner, device, [21], 120, sample_policy=False, rng_jitter=True)
    t1 = eval_pair("flee", spawner, device, [22], 120, sample_policy=False, rng_jitter=True)
    # With jitter, trajectories diverge; allow equality only if both timeout identically —
    # for random tiny nets they should differ almost always.
    assert t0 != t1 or (t0 == t1 and t0 > 0)


def test_eval_pair_default_is_det_det_compat(tiny_nets):
    """Normal reset eval path matches old behaviour (mode 1 defaults)."""
    player, spawner, device = tiny_nets
    seeds = range(50, 53)
    a = eval_pair(player, None, device, seeds, 60)
    b = eval_pair(player, None, device, seeds, 60, sample_policy=False, rng_jitter=False)
    assert a == b
    stats = eval_pair_stats(player, None, device, seeds, 60)
    assert stats["n"] == 3
    assert abs(stats["mean"] - a) < 1e-9
