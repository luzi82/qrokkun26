from __future__ import annotations

from pathlib import Path

import qrokkun_ai


def test_package_is_sibling_of_qrokkun_env() -> None:
    root = Path(qrokkun_ai.__file__).resolve().parent
    assert root.name == "qrokkun_ai"
    assert (root.parent / "qrokkun_env").is_dir()
    assert root.parent == Path(__file__).resolve().parents[2]


def test_import_player_v4() -> None:
    from qrokkun_ai.agents.player_v4 import PlayerV4  # noqa: F401


def test_import_obs_and_modes() -> None:
    from qrokkun_ai.obs import vectorize  # noqa: F401
    from qrokkun_ai.reset_modes import prepare_initial_state  # noqa: F401
    from qrokkun_ai.eval_modes import validate_eval_mode  # noqa: F401
    from qrokkun_env.policies import FleeNearestBullet  # noqa: F401
