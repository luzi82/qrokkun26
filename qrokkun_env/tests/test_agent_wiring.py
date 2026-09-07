"""Lock train scripts to agents/ modules and valid shebangs."""

from __future__ import annotations

from pathlib import Path

import qrokkun_env.train.both_v3 as both_v3
from qrokkun_env.agents.player_v3 import PlayerV3
from qrokkun_env.agents.spawner_v3 import SpawnerV3

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = sorted(ROOT.glob("train_*.py")) + sorted((ROOT / "train").glob("*.py"))
TRAIN_SCRIPTS = [p for p in TRAIN_SCRIPTS if p.name != "__init__.py"]


def test_both_v3_uses_agent_classes() -> None:
    assert both_v3.PlayerAC is PlayerV3
    assert both_v3.SpawnerAC is SpawnerV3


def test_train_scripts_shebang_is_first_line() -> None:
    assert TRAIN_SCRIPTS, "expected train scripts"
    for path in TRAIN_SCRIPTS:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "#!/usr/bin/env python3", f"{path}: first line is {first!r}"
