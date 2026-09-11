"""Tests for the strict production checkpoint factory/loader for the ranked
top-k Player architecture (qrokkun_env/agents/player_checkpoints.py).

These tests never read a real NAS checkpoint; everything is packed in tmp_path.
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
    PLAYER_FEAT_V4,
)
from qrokkun_env.agents.player_v4 import PlayerV4  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402
from qrokkun_env.tests.test_player_ranked_topk import make_batch  # noqa: E402


def test_pack_checkpoint_carries_production_architecture_metadata():
    from qrokkun_env.agents import player_checkpoints as pc
    from qrokkun_env.agents.player_ranked_topk import ARCHITECTURE, PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=32)
    ckpt = pc.pack_player_checkpoint(net, source_commit="deadbeef", source_tool="unit-test")

    assert ckpt["architecture"] == ARCHITECTURE
    assert ckpt["architecture_version"] == net.architecture_version
    assert ckpt["schema_version"] == pc.CKPT_SCHEMA_VERSION
    assert ckpt["ckpt_role"] == "player"
    assert ckpt["production_compatible"] is True
    assert ckpt["experimental"] is False
    assert ckpt["top_k"] == 8
    assert ckpt["hidden"] == 32
    assert ckpt["actions"] == list(ACTIONS)
    assert ckpt["n_actions"] == len(ACTIONS)
    assert ckpt["observation"] == {
        "player_feat": PLAYER_FEAT_V4,
        "bullet_feat": BULLET_FEAT_V4,
        "max_bullets": MAX_BULLETS_V4,
        "in_dim": net.in_dim,
    }
    assert ckpt["param_count"] == sum(p.numel() for p in net.parameters())
    assert ckpt["source"]["commit"] == "deadbeef"
    assert ckpt["source"]["tool"] == "unit-test"
    assert isinstance(ckpt["state_dict_sha256"], str) and len(ckpt["state_dict_sha256"]) == 64


def test_save_load_round_trip_restores_identical_behaviour(tmp_path: Path):
    from qrokkun_env.agents import player_checkpoints as pc
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    torch.manual_seed(0)
    net = PlayerRankedTopK(top_k=8, hidden=24)
    path = tmp_path / "ranked.pt"
    pc.save_player_checkpoint(net, path, source_commit="abc123", source_tool="unit-test")

    loaded, meta = pc.load_player_checkpoint(path, device=torch.device("cpu"))

    assert isinstance(loaded, PlayerRankedTopK)
    assert (loaded.top_k, loaded.hidden) == (8, 24)
    assert meta["architecture"] == "player_ranked_topk"
    player, bullets, pad = make_batch(batch=4, live=6, seed=11)
    net.eval()
    loaded.eval()
    with torch.no_grad():
        a, va = net(player, bullets, pad)
        b, vb = loaded(player, bullets, pad)
    assert torch.allclose(a.logits, b.logits, atol=0)
    assert torch.allclose(va, vb, atol=0)


def test_loader_rejects_legacy_playerv4_checkpoint(tmp_path: Path):
    """A PlayerV4-style checkpoint without architecture metadata must never be
    silently loaded as the ranked top-k architecture."""
    from qrokkun_env.agents import player_checkpoints as pc

    v4 = PlayerV4(d_model=16, hidden=16)
    path = tmp_path / "v4_legacy.pt"
    torch.save({"state_dict": v4.state_dict(), "actions": list(ACTIONS), "hidden": 16}, path)

    with pytest.raises(pc.CheckpointArchitectureError):
        pc.load_player_checkpoint(path, device=torch.device("cpu"))
    with pytest.raises(pc.CheckpointArchitectureError):
        pc.load_ranked_top_k_checkpoint(path, device=torch.device("cpu"))


def test_ranked_checkpoint_is_never_loadable_as_playerv4(tmp_path: Path):
    from qrokkun_env.agents import player_checkpoints as pc
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = PlayerRankedTopK(top_k=8, hidden=24)
    path = tmp_path / "ranked.pt"
    pc.save_player_checkpoint(net, path)

    # explicit expectation mismatch is refused by the factory ...
    with pytest.raises(pc.CheckpointArchitectureError):
        pc.load_player_checkpoint(path, device=torch.device("cpu"), expected_architecture="player_v4")

    # ... and the raw weights are structurally incompatible with PlayerV4 too.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError):
        PlayerV4(d_model=16, hidden=24).load_state_dict(raw["state_dict"])


@pytest.mark.parametrize(
    "mutate, err_attr",
    [
        (lambda c: c.update(architecture="totally_unknown_arch"), "CheckpointArchitectureError"),
        (lambda c: c.update(schema_version=999), "CheckpointSchemaError"),
        (lambda c: c.update(actions=["left", "right"]), "CheckpointSchemaError"),
        (lambda c: c.update(observation={**c["observation"], "max_bullets": 7}), "CheckpointSchemaError"),
        (lambda c: c.update(top_k=None), "CheckpointSchemaError"),
        (lambda c: c.pop("state_dict"), "CheckpointSchemaError"),
        (lambda c: c.update(state_dict_sha256="0" * 64), "CheckpointSchemaError"),
    ],
)
def test_strict_validation_rejects_corrupted_fields(tmp_path: Path, mutate, err_attr):
    from qrokkun_env.agents import player_checkpoints as pc
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    ckpt = pc.pack_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16))
    mutate(ckpt)
    path = tmp_path / "bad.pt"
    torch.save(ckpt, path)

    with pytest.raises(getattr(pc, err_attr)):
        pc.load_player_checkpoint(path, device=torch.device("cpu"))


def test_build_from_metadata_factory_is_keyed_by_architecture():
    from qrokkun_env.agents import player_checkpoints as pc
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK

    net = pc.build_player_from_metadata({"architecture": "player_ranked_topk", "top_k": 8, "hidden": 33})
    assert isinstance(net, PlayerRankedTopK)
    assert (net.top_k, net.hidden) == (8, 33)

    with pytest.raises(pc.CheckpointArchitectureError):
        pc.build_player_from_metadata({"architecture": "player_v4", "hidden": 33})
    with pytest.raises(pc.CheckpointArchitectureError):
        pc.build_player_from_metadata({"top_k": 8, "hidden": 33})
