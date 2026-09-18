from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest
import qrokkun_ai

TRACKED_CHECKPOINT_PATHS = (
    "qrokkun_ai/v0/checkpoints/player_mlp_cpu.pt",
    "qrokkun_ai/v1/checkpoints/player_ac_long_cpu.pt",
)


VERSIONED_IMPORTS = (
    "qrokkun_ai.v0.agents.player_v0",
    "qrokkun_ai.v1.agents.player_v1",
    "qrokkun_ai.v1.agents.spawner_v1",
    "qrokkun_ai.v2.agents.player_v2",
    "qrokkun_ai.v2.agents.spawner_v2",
    "qrokkun_ai.v3.agents.player_v3",
    "qrokkun_ai.v3.agents.spawner_v3",
    "qrokkun_ai.v4.agents.player_v4",
    "qrokkun_ai.v4.agents.spawner_v4",
    "qrokkun_ai.v5.agents.player_ranked_topk",
)


def test_only_version_packages_are_runtime_children() -> None:
    root = Path(qrokkun_ai.__file__).resolve().parent
    assert {path.name for path in root.iterdir()} >= {
        "__init__.py", "v0", "v1", "v2", "v3", "v4", "v5",
    }
    forbidden = {"agents", "train", "tools", "tests"}
    assert not (forbidden & {path.name for path in root.iterdir()})


@pytest.mark.parametrize("module_name", VERSIONED_IMPORTS)
def test_versioned_model_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(
    "module_name",
    ("qrokkun_ai.agents.player_v1", "qrokkun_ai.train.both_v4", "qrokkun_ai.train_both_v4_gpu"),
)
def test_legacy_imports_are_not_supported(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_bundled_checkpoint_binaries_are_git_tracked() -> None:
    repo_root = Path(qrokkun_ai.__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *TRACKED_CHECKPOINT_PATHS],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
