"""CPU contracts for PlayerV5 critic-warmup (schema v4).

The only treatment is value-head-only warmup on a frozen BC body/policy
before unregularized scripted PPO. PPO knobs stay locked to the Phase 3
control except the 10-update / snapshot-(0, 10) budget.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qrokkun_ai.v5.agents.obs_v4 import (  # noqa: E402
    BULLET_FEAT_V4,
    MAX_BULLETS_V4,
    PLAYER_FEAT_V4,
)
from qrokkun_ai.v5.agents.player_checkpoints import (  # noqa: E402
    pack_player_checkpoint,
)
from qrokkun_ai.v5.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402
from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention as ret  # noqa: E402
from qrokkun_ai.v5.tools import phase3_ranked_ppo_critic_warmup as warmup  # noqa: E402


def _tiny_net(seed: int = 0, hidden: int = 8, top_k: int = 4) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def _fake_rollout(*, n_frames: int, censored: bool, seed: int = 0, reward: float = 1.0) -> ret.Rollout:
    player = [np.zeros(PLAYER_FEAT_V4, dtype=np.float32) for _ in range(n_frames)]
    bullets = [np.zeros((MAX_BULLETS_V4, BULLET_FEAT_V4), dtype=np.float32) for _ in range(n_frames)]
    pad = [np.ones(MAX_BULLETS_V4, dtype=np.bool_) for _ in range(n_frames)]
    dones = [False] * n_frames
    if not censored:
        dones[-1] = True
    return ret.Rollout(
        seed=seed,
        player=player,
        bullets=bullets,
        pad=pad,
        actions=[0] * n_frames,
        log_probs=[0.0] * n_frames,
        values=[0.0] * n_frames,
        rewards=[reward] * n_frames,
        dones=dones,
        elapsed=float(n_frames) / 60.0,
        censored=censored,
    )


def _held_tensors(n: int = 6, seed: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "player": torch.randn(n, PLAYER_FEAT_V4, generator=g),
        "bullets": torch.randn(n, MAX_BULLETS_V4, BULLET_FEAT_V4, generator=g),
        "pad": torch.zeros(n, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.randn(n, len(ACTIONS), generator=g),
        "elapsed": torch.rand(n, generator=g) * 30.0,
    }


def test_locked_knobs_match_control_except_update_budget() -> None:
    assert warmup.PPO_LR is ret.PPO_LR or warmup.PPO_LR == ret.PPO_LR
    assert warmup.PPO_GAMMA == ret.PPO_GAMMA
    assert warmup.PPO_LAMBDA == ret.PPO_LAMBDA
    assert warmup.PPO_CLIP == ret.PPO_CLIP
    assert warmup.PPO_ENTROPY_COEF == ret.PPO_ENTROPY_COEF
    assert warmup.PPO_VALUE_COEF == ret.PPO_VALUE_COEF
    assert warmup.PPO_EPOCHS == ret.PPO_EPOCHS
    assert warmup.PPO_MINIBATCH == ret.PPO_MINIBATCH
    assert warmup.EPISODES_PER_UPDATE == ret.EPISODES_PER_UPDATE
    assert warmup.PPO_MAX_GRAD_NORM == ret.PPO_MAX_GRAD_NORM
    assert warmup.PPO_MAX_FRAMES == ret.PPO_MAX_FRAMES
    assert warmup.PPO_ROLLOUT_SEED_START == ret.PPO_ROLLOUT_SEED_START
    assert warmup.PPO_TORCH_SEED == ret.PPO_TORCH_SEED
    assert warmup.EVAL_SEED_START == ret.EVAL_SEED_START
    assert warmup.EVAL_SEED_COUNT == ret.EVAL_SEED_COUNT
    assert warmup.shaped_reward is ret.shaped_reward
    assert warmup.collect_rollout is ret.collect_rollout
    assert warmup.ppo_update is ret.ppo_update
    assert warmup.evaluate_deterministic is ret.evaluate_deterministic
    assert warmup.CURRENT_RUN_SCHEMA_VERSION is ret.CURRENT_RUN_SCHEMA_VERSION
    assert warmup.PPO_UPDATES == 10
    assert warmup.SNAPSHOT_UPDATES == (0, 10)
    assert warmup.WARMUP_UPDATES == 10
    assert warmup.PPO_UPDATES != ret.PPO_UPDATES


def test_schema_version_is_shared_four() -> None:
    assert ret.CURRENT_RUN_SCHEMA_VERSION == 4
    assert warmup.CURRENT_RUN_SCHEMA_VERSION == 4


def test_warmup_optimizer_only_receives_value_parameters() -> None:
    net = _tiny_net()
    opt = warmup.value_optimizer(net)
    opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert opt_ids == {id(p) for p in net.value.parameters()}
    assert all(not p.requires_grad for p in net.body.parameters())
    assert all(not p.requires_grad for p in net.policy.parameters())
    assert all(p.requires_grad for p in net.value.parameters())


def test_warmup_leaves_actor_and_held_policy_bitwise_unchanged() -> None:
    device = torch.device("cpu")
    net = _tiny_net(seed=1)
    held = _held_tensors()
    before_actor = warmup.actor_state_snapshot(net)
    before_logits, before_argmax = warmup.held_policy_outputs(net, held, device)
    opt = warmup.value_optimizer(net)
    metrics = warmup.warmup_value_update(
        net, opt, [_fake_rollout(n_frames=8, censored=False, reward=1.0)], device,
    )
    assert metrics["optimizer_steps"] > 0
    warmup.assert_actor_unchanged(net, before_actor)
    after_logits, after_argmax = warmup.held_policy_outputs(net, held, device)
    assert torch.equal(before_logits, after_logits)
    assert torch.equal(before_argmax, after_argmax)


def test_truncated_rollouts_are_dropped_from_warmup_mse() -> None:
    device = torch.device("cpu")
    net = _tiny_net(seed=2)
    before = {k: v.detach().clone() for k, v in net.value.state_dict().items()}
    opt = warmup.value_optimizer(net)
    metrics = warmup.warmup_value_update(
        net, opt, [_fake_rollout(n_frames=5, censored=True, seed=1)], device,
    )
    assert metrics["optimizer_steps"] == 0
    assert metrics["n_samples"] == 0
    assert metrics["n_dropped_censored"] == 1
    after = net.value.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_monte_carlo_returns_have_no_bootstrap_on_true_death() -> None:
    rewards = [1.0, 2.0, 3.0]
    got = warmup.monte_carlo_returns(rewards, gamma=0.5)
    expected = [
        1.0 + 0.5 * (2.0 + 0.5 * 3.0),
        2.0 + 0.5 * 3.0,
        3.0,
    ]
    assert got == pytest.approx(expected)


def test_seed_windows_are_disjoint() -> None:
    warmup_seeds = [seed for update in range(1, warmup.WARMUP_UPDATES + 1)
                    for seed in warmup.warmup_seed_schedule(update)]
    ppo_seeds = [seed for update in range(warmup.PPO_UPDATES)
                 for seed in ret.rollout_seed_schedule(update)]
    eval_seeds = ret.eval_seed_list()
    held_fit = warmup.held_value_fit_seeds()
    assert warmup_seeds == list(range(40000, 40080))
    assert ppo_seeds == list(range(50000, 50080))
    assert held_fit == list(range(41000, 41030))
    windows = (set(warmup_seeds), set(ppo_seeds), set(eval_seeds), set(held_fit))
    for i, left in enumerate(windows):
        for right in list(windows)[i + 1 :]:
            assert left.isdisjoint(right)


def test_ppo_phase_reseed_matches_after_divergent_global_rng() -> None:
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    warmup.seed_ppo_phase()
    state_a = (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.get_rng_state().clone(),
    )
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    warmup.seed_ppo_phase()
    state_b = (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.get_rng_state().clone(),
    )
    assert state_a[0] == state_b[0]
    assert np.array_equal(state_a[1], state_b[1])
    assert torch.equal(state_a[2], state_b[2])


def test_new_run_json_writes_schema_four_and_rejects_three(tmp_path: Path) -> None:
    contract = {
        "format": 1,
        "schema_version": warmup.CURRENT_RUN_SCHEMA_VERSION,
        "tool": "phase3_ranked_ppo_critic_warmup",
        "arm": "direct",
        "stop_args": {
            "effective_max_updates": 1,
            "effective_end_time_hkt": "2099-12-31T23:59:00+08:00",
        },
        "no_promotion": True,
    }
    ret.create_or_validate_run_contract(tmp_path, contract, resume=False)
    written = json.loads((tmp_path / "run.json").read_text())
    assert written["schema_version"] == 4
    for bad in (3, 2, 1, 4.0, True, None):
        other = tmp_path / f"bad-{bad}"
        other.mkdir()
        stale = dict(contract)
        stale["schema_version"] = bad
        (other / "run.json").write_text(json.dumps(stale))
        with pytest.raises(ret.RunStateError, match="schema_version"):
            ret.create_or_validate_run_contract(other, contract, resume=True)


def test_packed_warmup_checkpoint_is_experimental_non_production() -> None:
    net = _tiny_net()
    ckpt = pack_player_checkpoint(
        net,
        source_tool="phase3_ranked_ppo_critic_warmup",
        experimental=True,
        production_compatible=False,
        extra={"update": 10},
    )
    assert ckpt["schema_version"] == 1
    assert ckpt["experimental"] is True
    assert ckpt["production_compatible"] is False


def test_compare_arms_mechanism_and_primary_without_promotion() -> None:
    warmup_report = {
        "ppo_entry_held_ev": 0.4,
        "snapshots": [
            {"update": 0, "teacher_diagnostics": {"agreement": 0.70}},
            {"update": 10, "teacher_diagnostics": {"agreement": 0.66}},
        ],
    }
    direct_report = {
        "ppo_entry_held_ev": -0.3,
        "snapshots": [
            {"update": 0, "teacher_diagnostics": {"agreement": 0.70}},
            {"update": 10, "teacher_diagnostics": {"agreement": 0.20}},
        ],
    }
    verdict = warmup.compare_arms(warmup_report, direct_report)
    assert verdict["mechanism_pass"] is True
    assert verdict["primary_pass"] is True
    assert verdict["promotion"] is False
    assert verdict["reason"] == ""


def test_compare_arms_fails_closed_on_missing_held_agreement() -> None:
    warmup_report = {
        "ppo_entry_held_ev": 0.5,
        "snapshots": [{"update": 10, "teacher_diagnostics": {}}],
    }
    direct_report = {
        "ppo_entry_held_ev": 0.1,
        "snapshots": [{"update": 10, "teacher_diagnostics": {"agreement": 0.2}}],
    }
    verdict = warmup.compare_arms(warmup_report, direct_report)
    assert verdict["mechanism_pass"] is True
    assert verdict["primary_pass"] is False
    assert verdict["promotion"] is False
    assert "agreement" in verdict["reason"]


def test_compare_arms_requires_positive_warmup_held_ev() -> None:
    verdict = warmup.compare_arms(
        {
            "ppo_entry_held_ev": -0.01,
            "snapshots": [{"update": 10, "teacher_diagnostics": {"agreement": 0.7}}],
        },
        {
            "ppo_entry_held_ev": -0.4,
            "snapshots": [{"update": 10, "teacher_diagnostics": {"agreement": 0.1}}],
        },
    )
    assert verdict["mechanism_pass"] is False
    assert verdict["primary_pass"] is False
    assert verdict["promotion"] is False
