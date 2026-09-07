"""Lock train scripts to agents/ modules and valid shebangs."""

from __future__ import annotations

from pathlib import Path

import qrokkun_env.train_both_v3_gpu as both_v3
from qrokkun_env.agents.player_v3 import PlayerV3
from qrokkun_env.agents.spawner_v3 import SpawnerV3

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = sorted(ROOT.glob("train_*.py"))


def test_both_v3_uses_agent_classes() -> None:
    assert both_v3.PlayerAC is PlayerV3
    assert both_v3.SpawnerAC is SpawnerV3


def test_train_scripts_shebang_is_first_line() -> None:
    assert TRAIN_SCRIPTS, "expected train_*.py under qrokkun_env"
    for path in TRAIN_SCRIPTS:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == "#!/usr/bin/env python3", f"{path.name}: first line is {first!r}"
