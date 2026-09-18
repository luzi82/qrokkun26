"""Core sim stays torch-free; moved AI paths must not exist under qrokkun_env."""

from __future__ import annotations

import ast
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]


def test_moved_ai_paths_are_gone() -> None:
    assert not (ENV_ROOT / "agents").exists()
    assert not (ENV_ROOT / "train").exists()
    assert not (ENV_ROOT / "checkpoints").exists()
    for name in (
        "obs.py",
        "obs_rich.py",
        "reset_modes.py",
        "eval_modes.py",
        "corner_probe.py",
        "frozen_spawner_eval_cli.py",
        "frozen_spawner_eval_runner.py",
        "frozen_spawner_seed_package.py",
        "render_player_demo.py",
        "render_player_v5_demo.py",
        "render_v3_demo.py",
        "render_v4_demo.py",
        "render_vs_demo.py",
        "train_player.py",
        "train_player_gpu.py",
        "train_player_long.py",
        "train_spawner_gpu.py",
        "train_both_gpu.py",
        "train_both_v2_gpu.py",
        "train_both_v3_gpu.py",
        "train_both_v4_gpu.py",
    ):
        assert not (ENV_ROOT / name).exists(), name
    assert (ENV_ROOT / "policies.py").is_file()
    assert (ENV_ROOT / "sanity.py").is_file()


def test_qrokkun_env_does_not_import_torch_or_qrokkun_ai() -> None:
    offenders: list[str] = []
    for path in sorted(p for p in ENV_ROOT.rglob("*.py") if "__pycache__" not in p.parts):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "torch" or name.startswith("torch.") or name == "qrokkun_ai" or name.startswith("qrokkun_ai."):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []
