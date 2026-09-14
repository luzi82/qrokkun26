"""Behavioral tests for the fixed-contract PlayerV5 diagnostic renderer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_rollout_loads_ranked_checkpoint_and_reports_fixed_censor_contract(tmp_path: Path):
    """A schema-valid synthetic checkpoint drives the real scripted rollout."""
    from qrokkun_env.agents.player_checkpoints import save_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.render_player_v5_demo import FIXED_FRAME_CAP, FIXED_SEED, run_demo_rollout

    checkpoint = tmp_path / "player_update_200.pt"
    save_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), checkpoint, source_tool="test")

    states = []
    result = run_demo_rollout(
        checkpoint,
        device="cpu",
        on_frame=lambda env, _action, _frame: states.append((env.px, env.py, env.pvx, env.pvy)),
    )

    assert result["seed"] == FIXED_SEED == 3000
    assert result["frames"] <= FIXED_FRAME_CAP == 4200
    assert set(result) == {"seed", "elapsed", "frames", "hit", "censored", "termination_reason"}
    assert result["hit"] is (result["termination_reason"] == "hit")
    assert result["censored"] is (result["termination_reason"] == "frame_cap")
    from qrokkun_env.godot_rng import f32
    assert states and all(value == f32(value) for state in states for value in state)


def test_fixed_parser_has_no_seed_cap_or_action_override():
    from qrokkun_env.render_player_v5_demo import build_parser

    options = {action.dest for action in build_parser()._actions}
    assert {"ckpt", "out", "assets", "scale", "device"} <= options
    assert not ({"seed", "cap", "max_frames", "action", "greedy"} & options)


def test_loader_contract_rejects_wrong_architecture_and_hash_and_action_is_deterministic(tmp_path: Path):
    from qrokkun_env.agents.player_checkpoints import CheckpointArchitectureError, CheckpointSchemaError, pack_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.env import Qrokkun26Env
    from qrokkun_env.render_player_v5_demo import deterministic_player_action, load_demo_player

    net = PlayerRankedTopK(top_k=8, hidden=16)
    good = tmp_path / "player_update_200.pt"
    torch.save(pack_player_checkpoint(net, source_tool="test"), good)
    loaded, metadata = load_demo_player(good, "cpu")
    env = Qrokkun26Env(seed=3000)
    env.reset(seed=3000)
    assert metadata["architecture"] == "player_ranked_topk"
    assert deterministic_player_action(loaded, env, "cpu") == deterministic_player_action(loaded, env, "cpu")

    wrong_arch = pack_player_checkpoint(net, source_tool="test")
    wrong_arch["architecture"] = "player_v4"
    wrong_path = tmp_path / "wrong_arch.pt"
    torch.save(wrong_arch, wrong_path)
    with pytest.raises(CheckpointArchitectureError):
        load_demo_player(wrong_path, "cpu")

    wrong_hash = pack_player_checkpoint(net, source_tool="test")
    wrong_hash["state_dict_sha256"] = "0" * 64
    wrong_hash_path = tmp_path / "wrong_hash.pt"
    torch.save(wrong_hash, wrong_hash_path)
    with pytest.raises(CheckpointSchemaError):
        load_demo_player(wrong_hash_path, "cpu")

    wrong_schema = pack_player_checkpoint(net, source_tool="test")
    wrong_schema["schema_version"] = -1
    wrong_schema_path = tmp_path / "wrong_schema.pt"
    torch.save(wrong_schema, wrong_schema_path)
    with pytest.raises(CheckpointSchemaError):
        load_demo_player(wrong_schema_path, "cpu")


@pytest.mark.parametrize("architecture_version", [None, 999, True, 1.0])
def test_public_ranked_renderer_loader_rejects_missing_or_wrong_architecture_version(
    tmp_path: Path, architecture_version: int | float | None
):
    """A hash-valid checkpoint cannot declare an incompatible network version."""
    from qrokkun_env.agents.player_checkpoints import CheckpointSchemaError, pack_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.render_player_v5_demo import load_demo_player

    checkpoint = pack_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), source_tool="test")
    if architecture_version is None:
        checkpoint.pop("architecture_version")
    else:
        checkpoint["architecture_version"] = architecture_version
    path = tmp_path / f"bad-version-{architecture_version}.pt"
    torch.save(checkpoint, path)

    with pytest.raises(CheckpointSchemaError, match="architecture_version"):
        load_demo_player(path, "cpu")


def test_post_step_final_frame_and_hit_vs_censor_are_unambiguous(tmp_path: Path, monkeypatch):
    """Use the real canonical environment, including a real collision check."""
    from qrokkun_env.agents.player_checkpoints import load_ranked_top_k_checkpoint, save_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.env import Bullet, Qrokkun26Env
    from qrokkun_env.render_player_v5_demo import FIXED_SEED, _run_loaded_rollout

    checkpoint = tmp_path / "player_update_200.pt"
    save_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), checkpoint, source_tool="test")
    net, _meta = load_ranked_top_k_checkpoint(checkpoint)

    hit_env = Qrokkun26Env(seed=FIXED_SEED)
    hit_env.reset(seed=FIXED_SEED)
    hit_env.bullets.append(Bullet(hit_env.px, hit_env.py, 0.0, 0.0, 0, 999.0, moved=True))
    seen = []
    hit = _run_loaded_rollout(net, "cpu", hit_env, on_frame=lambda env, _action, frame: seen.append((frame, env.dead)))
    assert hit["termination_reason"] == "hit"
    assert hit["hit"] is True and hit["censored"] is False
    assert seen == [(1, True)], "the final frame is emitted after the hit update"

    censor_env = Qrokkun26Env(seed=FIXED_SEED)
    censor_env.reset(seed=FIXED_SEED)
    monkeypatch.setattr("qrokkun_env.render_player_v5_demo.FIXED_FRAME_CAP", 1)
    censored = _run_loaded_rollout(net, "cpu", censor_env)
    assert censored["frames"] == 1
    assert censored["termination_reason"] == "frame_cap"
    assert censored["hit"] is False and censored["censored"] is True


def test_collision_on_cap_frame_is_a_hit_not_censor(tmp_path: Path, monkeypatch):
    """A real canonical collision takes precedence over the exact cap boundary."""
    from qrokkun_env.agents.player_checkpoints import load_ranked_top_k_checkpoint, save_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.env import Bullet, Qrokkun26Env
    from qrokkun_env.render_player_v5_demo import FIXED_SEED, _run_loaded_rollout

    monkeypatch.setattr("qrokkun_env.render_player_v5_demo.FIXED_FRAME_CAP", 1)
    checkpoint = tmp_path / "player_update_200.pt"
    save_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), checkpoint, source_tool="test")
    net, _meta = load_ranked_top_k_checkpoint(checkpoint)
    env = Qrokkun26Env(seed=FIXED_SEED)
    env.reset(seed=FIXED_SEED)
    env.bullets.append(Bullet(env.px, env.py, 0.0, 0.0, 0, 999.0, moved=True))

    result = _run_loaded_rollout(net, "cpu", env)

    assert result["frames"] == 1
    assert result["hit"] is True
    assert result["censored"] is False
    assert result["termination_reason"] == "hit"


def test_renderer_writes_identity_sidecar_and_refuses_any_overwrite(tmp_path: Path, monkeypatch):
    """Only ffmpeg is substituted; loading and the actual rollout stay real."""
    from qrokkun_env.agents.player_checkpoints import file_sha256, save_player_checkpoint, state_dict_sha256
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.render_player_v5_demo import DemoRenderError, render_demo

    checkpoint = tmp_path / "player_update_200.pt"
    packed = save_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), checkpoint, source_tool="test")
    output = tmp_path / "demo.mp4"
    calls = []

    def fake_ffmpeg(command, check):
        calls.append(command)
        assert check is True
        assert "-y" not in command
        Path(command[-1]).write_bytes(b"synthetic mp4")

    monkeypatch.setattr("qrokkun_env.render_player_v5_demo.subprocess.run", fake_ffmpeg)
    metadata = render_demo(checkpoint, output, assets=Path("assets"), scale=1)

    sidecar = output.with_suffix(".mp4.json")
    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert calls and output.is_file()
    assert persisted == metadata
    assert metadata["experimental"] is True
    assert metadata["promotion"] == "forbidden"
    assert metadata["checkpoint_selection"] == "none"
    assert metadata["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "file_sha256": file_sha256(checkpoint),
        "state_dict_sha256": state_dict_sha256(packed["state_dict"]),
    }
    assert not (tmp_path / ".demo.player_v5_frames").exists()
    with pytest.raises(DemoRenderError):
        render_demo(checkpoint, output, assets=Path("assets"), scale=1)


def test_renderer_accepts_experimental_update_200_and_keeps_diagnostic_sidecar(tmp_path: Path, monkeypatch):
    """Experimental/non-production checkpoints remain renderable diagnostics."""
    from qrokkun_env.agents.player_checkpoints import save_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.render_player_v5_demo import render_demo

    checkpoint = tmp_path / "player_update_200.pt"
    save_player_checkpoint(
        PlayerRankedTopK(top_k=8, hidden=16),
        checkpoint,
        source_tool="test",
        experimental=True,
        production_compatible=False,
    )
    output = tmp_path / "experimental.mp4"

    def fake_ffmpeg(command, check):
        assert check is True
        Path(command[-1]).write_bytes(b"synthetic mp4")

    monkeypatch.setattr("qrokkun_env.render_player_v5_demo.subprocess.run", fake_ffmpeg)
    metadata = render_demo(checkpoint, output, assets=Path("assets"), scale=1)

    assert output.is_file()
    assert json.loads(output.with_suffix(".mp4.json").read_text(encoding="utf-8")) == metadata
    assert metadata["experimental"] is True
    assert metadata["promotion"] == "forbidden"


def test_ffmpeg_failure_is_closed_and_frame_directory_is_cleaned(tmp_path: Path, monkeypatch):
    from qrokkun_env.agents.player_checkpoints import save_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.render_player_v5_demo import DemoRenderError, render_demo

    checkpoint = tmp_path / "player_update_200.pt"
    save_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), checkpoint, source_tool="test")
    output = tmp_path / "failed.mp4"

    def failing_ffmpeg(_command, check):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    import subprocess

    monkeypatch.setattr("qrokkun_env.render_player_v5_demo.subprocess.run", failing_ffmpeg)
    with pytest.raises(DemoRenderError, match="ffmpeg"):
        render_demo(checkpoint, output, assets=Path("assets"), scale=1)
    assert not output.exists()
    assert not output.with_suffix(".mp4.json").exists()
    assert not (tmp_path / ".failed.player_v5_frames").exists()


def test_sidecar_write_failure_removes_successful_mp4_and_artifacts(tmp_path: Path, monkeypatch):
    """A failed sidecar commit cannot leave an MP4 that blocks a clean retry."""
    from qrokkun_env.agents.player_checkpoints import save_player_checkpoint
    from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
    from qrokkun_env.render_player_v5_demo import render_demo

    checkpoint = tmp_path / "player_update_200.pt"
    save_player_checkpoint(PlayerRankedTopK(top_k=8, hidden=16), checkpoint, source_tool="test")
    output = tmp_path / "sidecar-failure.mp4"
    sidecar = output.with_suffix(".mp4.json")

    def fake_ffmpeg(command, check):
        assert check is True
        Path(command[-1]).write_bytes(b"synthetic mp4")

    original_write_text = Path.write_text

    def fail_sidecar_write(path, *args, **kwargs):
        if path == sidecar:
            raise OSError("sidecar storage unavailable")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr("qrokkun_env.render_player_v5_demo.subprocess.run", fake_ffmpeg)
    monkeypatch.setattr(Path, "write_text", fail_sidecar_write)
    with pytest.raises(OSError, match="sidecar storage unavailable"):
        render_demo(checkpoint, output, assets=Path("assets"), scale=1)

    assert not output.exists()
    assert not sidecar.exists()
    assert not (tmp_path / ".sidecar-failure.player_v5_frames").exists()

    monkeypatch.setattr(Path, "write_text", original_write_text)
    metadata = render_demo(checkpoint, output, assets=Path("assets"), scale=1)

    assert output.is_file()
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == metadata
