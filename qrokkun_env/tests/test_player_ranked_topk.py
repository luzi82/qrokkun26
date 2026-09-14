"""Tests for the production ranked top-k Player architecture.

Pure unit tests on synthetic tensors: no env rollouts, no checkpoints from the
NAS, no GPU work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qrokkun_env.agents.obs_v4 import (  # noqa: E402
    BULLET_FEAT_V4,
    MAX_BULLETS_V4,
    PAD_RADIUS,
    PLAYER_FEAT_V4,
)
from qrokkun_env.env import ACTIONS  # noqa: E402


def make_batch(batch: int = 5, live: int | list[int] = 3, seed: int = 0):
    """Synthetic (player, bullets, pad) batch in the nearest-first v4 layout."""
    gen = torch.Generator().manual_seed(seed)
    player = torch.randn(batch, PLAYER_FEAT_V4, generator=gen)
    bullets = torch.zeros(batch, MAX_BULLETS_V4, BULLET_FEAT_V4)
    bullets[:, :, 4] = PAD_RADIUS
    pad = torch.ones(batch, MAX_BULLETS_V4, dtype=torch.bool)
    lives = [live] * batch if isinstance(live, int) else list(live)
    for i, n_live in enumerate(lives):
        if n_live:
            bullets[i, :n_live] = torch.randn(n_live, BULLET_FEAT_V4, generator=gen)
            pad[i, :n_live] = False
    return player, bullets, pad


def test_forward_returns_categorical_and_value_with_expected_shapes():
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=32)
    player, bullets, pad = make_batch(batch=5, live=3)
    dist, value = net(player, bullets, pad)

    assert isinstance(dist, torch.distributions.Categorical)
    assert tuple(dist.logits.shape) == (5, len(ACTIONS))
    assert tuple(value.shape) == (5,)


def test_only_nearest_ranked_top_k_slots_affect_output():
    """Slots beyond top_k are ignored; changing slot k..K-1 cannot move logits."""
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=32)
    player, bullets, pad = make_batch(batch=3, live=20, seed=1)
    dist_a, value_a = net(player, bullets, pad)

    far = bullets.clone()
    far_pad = pad.clone()
    far[:, 8:] = torch.randn_like(far[:, 8:])
    far_pad[:, 8:] = True
    dist_b, value_b = net(player, far, far_pad)

    assert torch.allclose(dist_a.logits, dist_b.logits, atol=1e-6)
    assert torch.allclose(value_a, value_b, atol=1e-6)


def test_padded_slot_contents_are_masked_out():
    """Garbage in a padded (non-live) slot inside the top-k window is zeroed."""
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=32)
    player, bullets, pad = make_batch(batch=2, live=3, seed=2)
    dirty = bullets.clone()
    dirty[:, 3:8] = 99.0  # slots marked pad=True must not leak in
    dist_a, _ = net(player, bullets, pad)
    dist_b, _ = net(player, dirty, pad)

    assert torch.allclose(dist_a.logits, dist_b.logits, atol=1e-6)


def test_hidden_width_and_top_k_are_configurable():
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = PlayerRankedTopK(top_k=4, hidden=17)
    assert net.top_k == 4
    assert net.hidden == 17
    assert net.in_dim == PLAYER_FEAT_V4 + 4 * BULLET_FEAT_V4 + 4
    assert net.body[0].in_features == net.in_dim
    assert net.body[0].out_features == 17
    wide = PlayerRankedTopK(top_k=8, hidden=472)
    narrow = PlayerRankedTopK(top_k=8, hidden=64)
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in narrow.parameters())


@pytest.mark.parametrize("lives", [[0, 0, 0], [0, 5, 64], [64, 64, 64]])
def test_all_pad_and_mixed_batches_are_finite_forward_and_backward(lives):
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=32)
    player, bullets, pad = make_batch(batch=len(lives), live=lives, seed=3)
    dist, value = net(player, bullets, pad)

    assert torch.isfinite(dist.logits).all()
    assert torch.isfinite(value).all()

    loss = dist.logits.square().mean() + value.square().mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_architecture_metadata_is_exposed():
    from qrokkun_env.agents.player_ranked_topk import ARCHITECTURE, PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=64)
    meta = net.architecture_metadata()

    assert meta["architecture"] == ARCHITECTURE == "player_ranked_topk"
    assert meta["architecture_version"] >= 1
    assert meta["top_k"] == 8
    assert meta["hidden"] == 64
    assert meta["actions"] == list(ACTIONS)
    assert meta["player_feat"] == PLAYER_FEAT_V4
    assert meta["bullet_feat"] == BULLET_FEAT_V4
    assert meta["max_bullets"] == MAX_BULLETS_V4
    assert meta["param_count"] == sum(p.numel() for p in net.parameters() if p.requires_grad)


def test_module_is_exported_from_agents_package():
    import qrokkun_env.agents as agents

    assert "player_ranked_topk" in agents.__all__
    assert "player_checkpoints" in agents.__all__


def test_obs_tensors_and_argmax_action_drive_inference_from_env():
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK, argmax_action, obs_tensors
    from qrokkun_env.env import Qrokkun26Env

    env = Qrokkun26Env(seed=123)
    env.reset(seed=123)
    net = PlayerRankedTopK(top_k=8, hidden=16)

    player, bullets, pad = obs_tensors(env, torch.device("cpu"))
    assert tuple(player.shape) == (1, PLAYER_FEAT_V4)
    assert tuple(bullets.shape) == (1, MAX_BULLETS_V4, BULLET_FEAT_V4)
    assert tuple(pad.shape) == (1, MAX_BULLETS_V4)
    assert pad.dtype is torch.bool

    action = argmax_action(net, env, torch.device("cpu"))
    assert isinstance(action, int)
    assert 0 <= action < len(ACTIONS)
    # deterministic: same env state -> same action
    assert argmax_action(net, env, torch.device("cpu")) == action
