"""v4.7: Spawner trajectories must not be packed/trained on
self_random_player or p_vs_scripted_normal/random — only self_normal and
s_vs_flee_normal train the Spawner. Player trains on every mode except
s_vs_flee_normal.

No torch required: this only exercises the mode-classification helpers in
qrokkun_env.reset_modes, which mirror how qrokkun_env.train.both_v4.main()
decides train_player / train_spawner per rollout.
"""

from __future__ import annotations

from qrokkun_env.reset_modes import (
    MODE_P_VS_SCRIPTED_NORMAL,
    MODE_P_VS_SCRIPTED_RANDOM,
    MODE_S_VS_FLEE_NORMAL,
    MODE_SELF_NORMAL,
    MODE_SELF_RANDOM_PLAYER,
    trains_player,
    trains_spawner,
)


def test_self_random_player_trains_player_only() -> None:
    assert trains_player(MODE_SELF_RANDOM_PLAYER) is True
    assert trains_spawner(MODE_SELF_RANDOM_PLAYER) is False


def test_p_vs_scripted_never_trains_spawner() -> None:
    assert trains_spawner(MODE_P_VS_SCRIPTED_NORMAL) is False
    assert trains_spawner(MODE_P_VS_SCRIPTED_RANDOM) is False
    assert trains_player(MODE_P_VS_SCRIPTED_NORMAL) is True
    assert trains_player(MODE_P_VS_SCRIPTED_RANDOM) is True


def test_self_normal_and_s_vs_flee_normal_train_spawner() -> None:
    assert trains_spawner(MODE_SELF_NORMAL) is True
    assert trains_spawner(MODE_S_VS_FLEE_NORMAL) is True


def test_s_vs_flee_normal_does_not_train_player() -> None:
    assert trains_player(MODE_S_VS_FLEE_NORMAL) is False


def test_spawner_batch_filter_keeps_only_normal_reset_trajs() -> None:
    """Mirrors the defensive filter in both_v4.main(): even if a Spawner traj
    were ever produced under a non-normal reset_mode, the PPO batch must drop
    it."""
    from dataclasses import dataclass

    @dataclass
    class FakeTraj:
        reset_mode: str
        rewards: list

    trajs = [
        FakeTraj("normal", [1.0]),
        FakeTraj("random_player", [1.0]),
        FakeTraj("normal", [1.0]),
    ]
    normal_only = [t for t in trajs if t.reset_mode == "normal"]
    assert len(normal_only) == 2
    assert all(t.reset_mode == "normal" for t in normal_only)
