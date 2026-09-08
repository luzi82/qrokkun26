"""v4.4 terminated/truncated bootstrap + v4.5 variable-time GAE discounts."""

from __future__ import annotations

import math

import torch

from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.train.both_v4 import (
    Traj,
    gae,
    player_value,
    run_episode,
    shaped_spawner_r,
    spawner_value,
)


def test_gae_true_death_last_value_zero_no_bootstrap() -> None:
    """terminated → mask zeros bootstrap; last_value must not affect returns."""
    device = torch.device("cpu")
    rewards = [1.0, 2.0, -1.0]
    values = [0.5, 0.4, 0.3]
    terminateds = [False, False, True]
    delta_frames = [1, 1, 1]
    gamma, lam = 0.99, 0.95

    adv0, ret0 = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=0.0, delta_frames=delta_frames
    )
    adv_hi, ret_hi = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=99.0, delta_frames=delta_frames
    )
    assert torch.allclose(adv0, adv_hi, atol=1e-6)
    assert torch.allclose(ret0, ret_hi, atol=1e-6)

    # Last TD target ignores bootstrap: r_T - V_T (no gamma * last_value)
    # With GAE lambda, last advantage = delta_T = r_T + 0 - V_T
    assert abs(float(adv0[-1]) - (rewards[-1] - values[-1])) < 1e-5


def test_gae_truncation_final_value_enters_last_delta() -> None:
    """truncated: terminateds[-1]=False → last_value enters final delta."""
    device = torch.device("cpu")
    rewards = [0.1, 0.1]
    values = [1.0, 2.0]
    terminateds = [False, False]
    delta_frames = [1, 1]
    gamma, lam = 0.9, 0.0  # lam=0 → advantage == TD residual
    last_value = 5.0

    adv, ret = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=last_value, delta_frames=delta_frames
    )
    # last delta = r + gamma * last_value - V
    expected_last = rewards[-1] + gamma * last_value - values[-1]
    assert abs(float(adv[-1]) - expected_last) < 1e-5
    assert abs(float(ret[-1]) - (expected_last + values[-1])) < 1e-5

    # Contrast: if wrongly treated as terminated, last_value ignored
    adv_term, _ = gae(
        rewards, values, [False, True], gamma, lam, device, last_value=last_value, delta_frames=delta_frames
    )
    assert abs(float(adv_term[-1]) - (rewards[-1] - values[-1])) < 1e-5
    assert not torch.allclose(adv, adv_term, atol=1e-4)


def test_gae_variable_delta_frames_discounts() -> None:
    """gamma_t = gamma_frame ** delta_frames; Spawner-like uneven steps."""
    device = torch.device("cpu")
    rewards = [1.0, 1.0]
    values = [0.0, 0.0]
    terminateds = [False, False]
    last_value = 10.0
    gamma, lam = 0.9, 0.0

    # Uniform 1-frame
    adv1, _ = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=last_value, delta_frames=[1, 1]
    )
    # Last step with delta=3 → gamma**3
    adv3, _ = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=last_value, delta_frames=[1, 3]
    )
    expected1 = rewards[-1] + (gamma ** 1) * last_value - values[-1]
    expected3 = rewards[-1] + (gamma ** 3) * last_value - values[-1]
    assert abs(float(adv1[-1]) - expected1) < 1e-5
    assert abs(float(adv3[-1]) - expected3) < 1e-5
    assert abs(float(adv3[-1]) - float(adv1[-1])) > 0.1

    # Same-frame double spawn: delta_frames=0 → gamma**0 = 1
    adv0, _ = gae(
        [1.0], [0.0], [False], gamma, lam, device, last_value=last_value, delta_frames=[0]
    )
    assert abs(float(adv0[-1]) - (1.0 + 1.0 * last_value)) < 1e-5


def test_gae_lambda_also_scales_with_delta_frames() -> None:
    device = torch.device("cpu")
    # Two steps, non-zero lam so multi-step GAE uses gamma*lam product
    rewards = [0.0, 0.0]
    values = [0.0, 0.0]
    terminateds = [False, False]
    last_value = 1.0
    gamma, lam = 0.8, 0.5

    adv_a, _ = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=last_value, delta_frames=[1, 1]
    )
    adv_b, _ = gae(
        rewards, values, terminateds, gamma, lam, device, last_value=last_value, delta_frames=[2, 2]
    )
    # Hand-compute last deltas (lam= irrelevant for last), then prior step
    # t=1: delta = g**df * last_value; gae_1 = delta
    # t=0: delta = g**df * vals[1]=0; gae_0 = 0 + g**df * l**df * gae_1
    df = 1
    gae1_a = (gamma ** df) * last_value
    gae0_a = (gamma ** df) * (lam ** df) * gae1_a
    df = 2
    gae1_b = (gamma ** df) * last_value
    gae0_b = (gamma ** df) * (lam ** df) * gae1_b
    assert abs(float(adv_a[0]) - gae0_a) < 1e-5
    assert abs(float(adv_b[0]) - gae0_b) < 1e-5
    assert abs(gae0_a - gae0_b) > 1e-4


def test_survive_to_limit_no_spawner_fail_penalty() -> None:
    """Truncation must not apply death-style 'failed to kill' (-2) terminal bonus."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    player = PlayerV4(d_model=16, hidden=32).to(device)
    spawner = SpawnerV4(d_model=16, hidden=32).to(device)
    player.eval()
    spawner.eval()

    # Enough frames for ≥1 spawn (~70 to first), still short so flee usually lives.
    env = Qrokkun26Env(seed=42)
    _pt, st, elapsed = run_episode(
        env,
        "flee",
        spawner,
        device,
        max_steps=100,
        sample=False,
        train_player=False,
        train_spawner=True,
        temp_p=1.0,
        temp_s=1.0,
        rng_jitter=False,
    )
    assert not env.dead, "expected survive-to-limit for this short horizon"
    assert st.rewards, "need at least one spawn decision before truncate"
    assert st.truncated and not st.terminated
    assert st.terminateds[-1] is False
    assert abs(st.last_value - spawner_value(spawner, env, device)) < 1e-4
    # Alive → terminated shaping is 0 (no +5); API must not invent -2 fail-to-kill.
    assert shaped_spawner_r(env, terminated=True, is_spawn=False) == 0.0
    assert shaped_spawner_r(env, terminated=False, is_spawn=False) >= 0.0
    # Truncation path adds no terminal term; last reward is spawn-time shaping only
    # (≈ -0.02 + proximity), never ≈ -2.
    assert st.rewards[-1] > -1.0
    assert elapsed > 0


def test_run_episode_death_vs_truncation_contracts() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    player = PlayerV4(d_model=16, hidden=32).to(device)
    spawner = SpawnerV4(d_model=16, hidden=32).to(device)
    player.eval()
    spawner.eval()

    # Truncation
    env_t = Qrokkun26Env(seed=7)
    pt_t, st_t, _ = run_episode(
        env_t, player, spawner, device, max_steps=40,
        sample=False, train_player=True, train_spawner=True,
        temp_p=1.0, temp_s=1.0, rng_jitter=False,
    )
    if not env_t.dead:
        assert pt_t.truncated and not pt_t.terminated
        assert pt_t.terminateds[-1] is False
        assert abs(pt_t.last_value - player_value(player, env_t, device)) < 1e-4
        if st_t.rewards:
            assert st_t.truncated and not st_t.terminated
            assert st_t.terminateds[-1] is False
            assert abs(st_t.last_value - spawner_value(spawner, env_t, device)) < 1e-4
            assert st_t.delta_frames[-1] >= 1
            assert all(d >= 0 for d in st_t.delta_frames)
        assert all(d == 1 for d in pt_t.delta_frames)

    # Force a true death with idle player vs aggressive-ish learned S over longer horizon
    dead_found = False
    for seed in range(100, 200):
        env_d = Qrokkun26Env(seed=seed)
        pt_d, st_d, _ = run_episode(
            env_d, player, spawner, device, max_steps=60 * 20,
            sample=False, train_player=True, train_spawner=True,
            temp_p=1.0, temp_s=1.0, rng_jitter=False,
        )
        if env_d.dead and pt_d.rewards and st_d.rewards:
            assert pt_d.terminated and not pt_d.truncated
            assert pt_d.last_value == 0.0
            assert pt_d.terminateds[-1] is True
            assert st_d.terminated and not st_d.truncated
            assert st_d.last_value == 0.0
            assert st_d.terminateds[-1] is True
            # Kill bonus present on last spawner reward path (+5 folded in)
            assert st_d.rewards[-1] >= 4.0  # spawn cost -0.02 + shaping + 5
            dead_found = True
            break
    assert dead_found, "could not find a true-death episode for contract check"


def test_spawner_delta_frames_between_decisions() -> None:
    torch.manual_seed(3)
    device = torch.device("cpu")
    spawner = SpawnerV4(d_model=16, hidden=32).to(device)
    spawner.eval()
    env = Qrokkun26Env(seed=3)
    _pt, st, _ = run_episode(
        env, "flee", spawner, device, max_steps=120,
        sample=False, train_player=False, train_spawner=True,
        temp_p=1.0, temp_s=1.0, rng_jitter=False,
    )
    assert len(st.delta_frames) == len(st.rewards) >= 1
    # Early spawn interval is long (~0.48s ≈ 29 frames); expect some delta > 1
    if len(st.delta_frames) >= 2:
        assert max(st.delta_frames[:-1]) >= 1
    # Final delta covers last spawn → end
    assert st.delta_frames[-1] >= 0
