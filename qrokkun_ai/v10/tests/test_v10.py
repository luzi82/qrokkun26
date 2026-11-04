"""Unit tests for PlayerV10 (Zero-Inertia Physics, 120s Cap, Decoupled Actor-Critic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qrokkun_ai.v10.agents.player_v10 import (
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    CheckpointV10Error,
    PlayerV10,
    load_v10_checkpoint,
    save_v10_checkpoint,
)
from qrokkun_ai.v10.obs_v10 import (
    BULLET_FEAT_V10,
    MAX_BULLETS_V10,
    PLAYER_FEAT_V10,
    batch_kinematic_clearances_v10,
    compute_kinematic_expert_targets,
    encode_obs_v10,
)
from qrokkun_ai.v10.train.train_player_v10 import (
    MAX_STEPS_V10,
    build_parser,
    collect_rollouts_v10,
    run_training_v10,
)
from qrokkun_env import constants as C
from qrokkun_env.env import Bullet, Qrokkun26Env


def test_encode_obs_v10_shapes_and_ranges() -> None:
    env = Qrokkun26Env(seed=0)
    env.reset(seed=0)
    env.elapsed = 60.0
    p, b, m = encode_obs_v10(env)
    assert p.shape == (PLAYER_FEAT_V10,)
    assert b.shape == (MAX_BULLETS_V10, BULLET_FEAT_V10)
    assert m.shape == (MAX_BULLETS_V10,)
    assert p.dtype == np.float32
    assert b.dtype == np.float32
    assert m.dtype == np.bool_
    # At center, norm_x == 0, norm_y == 0, elapsed/120 == 0.5
    assert abs(float(p[0])) < 1e-6
    assert abs(float(p[1])) < 1e-6
    assert abs(float(p[4]) - 0.5) < 1e-6
    assert m.all()


def test_batch_kinematic_clearances_empty_and_approaching() -> None:
    env = Qrokkun26Env(seed=0)
    env.reset(seed=0)
    p, b, m = encode_obs_v10(env)
    p_t = torch.tensor(p).unsqueeze(0)
    b_t = torch.tensor(b).unsqueeze(0)
    m_t = torch.tensor(m).unsqueeze(0)

    # Empty field -> all 9 actions have finite clearance +999.0
    cl_empty = batch_kinematic_clearances_v10(p_t, b_t, m_t, horizon=10, top_k=12)
    assert cl_empty.shape == (1, 9)
    assert torch.allclose(cl_empty, torch.full((1, 9), 999.0))

    # Place a stationary bullet directly on top of player -> idle clearance < 0
    env.bullets = [Bullet(env.px, env.py, 0.0, 0.0, 0, 2.0, moved=True)]
    p2, b2, m2 = encode_obs_v10(env)
    cl_hit = batch_kinematic_clearances_v10(
        torch.tensor(p2).unsqueeze(0),
        torch.tensor(b2).unsqueeze(0),
        torch.tensor(m2).unsqueeze(0),
        horizon=4,
        top_k=12,
    )
    # Action 0 ("idle") stays at (px, py), so distance is 0, clearance is -(4.0 + 2.0) = -6.0
    assert float(cl_hit[0, 0].item()) < -5.0
    # Moving away (e.g. "e" = index 3) has strictly larger minimum clearance than staying idle
    assert float(cl_hit[0, 3].item()) > float(cl_hit[0, 0].item())

    target_act, target_logits = compute_kinematic_expert_targets(
        torch.tensor(p2).unsqueeze(0),
        torch.tensor(b2).unsqueeze(0),
        torch.tensor(m2).unsqueeze(0),
        horizon=4,
        top_k=12,
    )
    assert target_act.shape == (1,)
    assert target_logits.shape == (1, 9)
    assert int(target_act.item()) != 0  # Expert never chooses idle when bullet is on player


def test_player_v10_decoupled_gradient_isolation() -> None:
    net = PlayerV10(top_k=12, hidden=64, lookahead_horizon=6)
    assert net.in_dim == 6 + 12 * 9 + 12 + 18

    player = torch.zeros((4, PLAYER_FEAT_V10))
    bullets = torch.zeros((4, MAX_BULLETS_V10, BULLET_FEAT_V10))
    pad = torch.ones((4, MAX_BULLETS_V10), dtype=torch.bool)

    # Backward through value ONLY must leave all actor gradients None
    _dist, val = net(player, bullets, pad)
    v_loss = F.mse_loss(val, torch.ones_like(val))
    v_loss.backward()
    for p in net.actor_parameters():
        assert p.grad is None
    for p in net.critic_parameters():
        assert p.grad is not None

    net.zero_grad(set_to_none=True)
    # Backward through policy ONLY must leave all critic gradients None
    dist, _val = net(player, bullets, pad)
    p_loss = -dist.log_prob(torch.zeros(4, dtype=torch.int64)).mean()
    p_loss.backward()
    for p in net.critic_parameters():
        assert p.grad is None
    for p in net.actor_parameters():
        assert p.grad is not None


def test_checkpoint_v10_roundtrip_and_strict_validation(tmp_path: Path) -> None:
    net = PlayerV10(top_k=12, hidden=64, lookahead_horizon=8)
    ckpt_path = tmp_path / "player_v10.pt"
    save_v10_checkpoint(net, ckpt_path, extra={"update": 5})

    loaded, meta = load_v10_checkpoint(ckpt_path, device="cpu")
    assert loaded.top_k == 12
    assert loaded.hidden == 64
    assert loaded.lookahead_horizon == 8
    assert meta["architecture"] == ARCHITECTURE
    assert meta["architecture_version"] == ARCHITECTURE_VERSION
    assert meta["extra"]["update"] == 5
    for k, v in net.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])

    # Reject boolean architecture_version
    bad = torch.load(ckpt_path, weights_only=False)
    bad["architecture_version"] = True
    bad_path = tmp_path / "bad_bool.pt"
    torch.save(bad, bad_path)
    with pytest.raises(CheckpointV10Error, match="architecture_version"):
        load_v10_checkpoint(bad_path)

    # Reject tampered weights
    tampered = torch.load(ckpt_path, weights_only=False)
    tampered["state_dict"]["policy_head.bias"] = tampered["state_dict"]["policy_head.bias"] + 1.0
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(CheckpointV10Error, match="state_dict_sha256"):
        load_v10_checkpoint(tampered_path)


def test_cap_boundary_death_precedence_and_quick_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert MAX_STEPS_V10 == 7200
    net = PlayerV10(top_k=8, hidden=32, lookahead_horizon=4)
    # 1) 2-step cap rollout where player does not die -> truncated=True, censored=True, terminated=False
    ro_list = collect_rollouts_v10(net, torch.device("cpu"), [0], max_steps=2)
    assert len(ro_list) == 1
    assert ro_list[0].truncated is True
    assert ro_list[0].censored is True
    assert ro_list[0].terminated is False

    # 2) Collision on the exact final capped step (step 2 of max_steps=2) -> terminated=True, truncated=False, censored=False, bootstrap_value=0.0
    from qrokkun_ai.v10.train import train_player_v10 as t_mod

    class FinalStepDeathEnv(Qrokkun26Env):
        def step(self, action):
            obs, reward, done, info = super().step(action)
            if self.elapsed >= 2 * self.dt:
                self.dead = True
                done = True
            return obs, reward, done, info

    monkeypatch.setattr(t_mod, "Qrokkun26Env", FinalStepDeathEnv)
    ro_death = collect_rollouts_v10(net, torch.device("cpu"), [0], max_steps=2)[0]
    assert ro_death.terminated is True
    assert ro_death.truncated is False
    assert ro_death.censored is False
    assert ro_death.bootstrap_value == 0.0
    assert ro_death.rewards[-1] < 0.0  # received -1.0 death penalty
    monkeypatch.undo()

    # 3) Quick trainer smoke run
    args = build_parser().parse_args(
        [
            "--quick",
            "--device",
            "cpu",
            "--hidden",
            "32",
            "--top-k",
            "8",
            "--horizon",
            "4",
            "--out-dir",
            str(tmp_path / "smoke_run"),
        ]
    )
    report = run_training_v10(args)
    assert report["updates"] == 2
    assert (tmp_path / "smoke_run" / "best_player_v10.pt").is_file()
    assert (tmp_path / "smoke_run" / "report.json").is_file()
