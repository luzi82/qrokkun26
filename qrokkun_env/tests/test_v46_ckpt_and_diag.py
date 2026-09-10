"""v4.6: PPO diagnostics + separate best P/S checkpoint selection."""

from __future__ import annotations

from pathlib import Path

import torch

from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.train.both_v4 import (
    explained_variance,
    ppo_update_player,
    ppo_update_spawner,
    run_episode,
    snapshot_gate_payloads,
    traj_end_rates,
)
from qrokkun_env.train.checkpoints_v4 import (
    FORBIDDEN_SOLE_SELECTOR_KEYS,
    PLAYER_SELECTION_KEYS,
    SPAWNER_SELECTION_KEYS,
    CheckpointManager,
    default_best_path,
    pack_player_ckpt,
    pack_spawner_ckpt,
    player_selection_key,
    player_selection_score,
    spawner_selection_key,
    spawner_selection_score,
)


def test_selection_keys_exclude_newxnew_det_det() -> None:
    assert "new_vs_new_det_det" not in PLAYER_SELECTION_KEYS
    assert "new_vs_new_det_det" not in SPAWNER_SELECTION_KEYS
    assert "new_vs_new_det_det" in FORBIDDEN_SOLE_SELECTOR_KEYS
    for k in PLAYER_SELECTION_KEYS + SPAWNER_SELECTION_KEYS:
        assert k not in FORBIDDEN_SOLE_SELECTOR_KEYS


def test_player_selection_ignores_huge_newxnew() -> None:
    metrics = {
        "new_vs_new_det_det": 999.0,
        "new_vs_new": 999.0,
        "newP_vs_scripted": 7.5,
    }
    assert player_selection_score(metrics) == 7.5
    assert player_selection_key(metrics) == "newP_vs_scripted"


def test_player_selection_prefers_det_stoch_key() -> None:
    metrics = {
        "newP_vs_scripted": 5.0,
        "newP_vs_scripted_det_stoch": 6.0,
        "new_vs_new_det_det": 100.0,
    }
    assert player_selection_score(metrics) == 6.0
    assert player_selection_key(metrics) == "newP_vs_scripted_det_stoch"


def test_spawner_selection_uses_flee_det_stoch_not_newxnew() -> None:
    metrics = {
        "new_vs_new_det_det": 0.01,  # would look "good" if wrongly minimized
        "flee_vs_newS": 8.0,
        "flee_vs_newS_det_stoch": 3.5,
    }
    assert spawner_selection_score(metrics) == 3.5
    assert spawner_selection_key(metrics) == "flee_vs_newS_det_stoch"


def test_best_ckpt_path_updates_when_metric_improves(tmp_path: Path) -> None:
    latest_p = tmp_path / "both_v4_player.pt"
    latest_s = tmp_path / "both_v4_spawner.pt"
    best_p = default_best_path(latest_p)
    best_s = default_best_path(latest_s)
    assert best_p.name == "both_v4_player_best.pt"

    mgr = CheckpointManager(
        latest_player=latest_p,
        latest_spawner=latest_s,
        best_player=best_p,
        best_spawner=best_s,
        snapshot_dir=tmp_path / "snapshots",
        snapshot_every=5,
    )

    player = PlayerV4(d_model=32, hidden=32)
    spawner = SpawnerV4(d_model=32, hidden=32)

    p0 = pack_player_ckpt(player, d_model=32, hidden=32, update=0)
    s0 = pack_spawner_ckpt(spawner, d_model=32, hidden=32, update=0, aim_scale=40.0)
    mgr.save_latest(p0, s0)
    assert latest_p.exists() and latest_s.exists()

    assert mgr.maybe_save_best_player(1.0, {**p0, "mark": "p1"})
    assert best_p.exists()
    loaded = torch.load(best_p, map_location="cpu", weights_only=False)
    assert loaded["mark"] == "p1"
    assert loaded["ckpt_kind"] == "best_player"

    # No improve → path contents unchanged
    assert not mgr.maybe_save_best_player(0.5, {**p0, "mark": "p_worse"})
    loaded2 = torch.load(best_p, map_location="cpu", weights_only=False)
    assert loaded2["mark"] == "p1"

    assert mgr.maybe_save_best_player(2.0, {**p0, "mark": "p2"})
    loaded3 = torch.load(best_p, map_location="cpu", weights_only=False)
    assert loaded3["mark"] == "p2"

    # Spawner: lower flee×newS is better
    assert mgr.maybe_save_best_spawner(5.0, {**s0, "mark": "s1"})
    assert best_s.exists()
    assert not mgr.maybe_save_best_spawner(6.0, {**s0, "mark": "s_worse"})
    assert torch.load(best_s, map_location="cpu", weights_only=False)["mark"] == "s1"
    assert mgr.maybe_save_best_spawner(4.0, {**s0, "mark": "s2"})
    assert torch.load(best_s, map_location="cpu", weights_only=False)["mark"] == "s2"

    # Snapshot cadence
    assert not mgr.maybe_snapshot(1, p0, s0)
    assert mgr.maybe_snapshot(5, p0, s0)
    assert (tmp_path / "snapshots" / "player_update_5.pt").exists()
    assert (tmp_path / "snapshots" / "spawner_update_5.pt").exists()


def test_newxnew_alone_does_not_drive_selection() -> None:
    """Sole new×new_det_det must not be used as P or S selector."""
    only_new = {"new_vs_new_det_det": 70.0, "new_vs_new": 70.0}
    try:
        player_selection_score(only_new)
        raised_p = False
    except KeyError:
        raised_p = True
    try:
        spawner_selection_score(only_new)
        raised_s = False
    except KeyError:
        raised_s = True
    assert raised_p and raised_s


def test_explained_variance_and_traj_rates() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.clone()
    assert abs(explained_variance(y_pred, y_true) - 1.0) < 1e-5
    assert explained_variance(torch.zeros(4), y_true) < 0.5

    from qrokkun_env.train.both_v4 import Traj

    t_term = Traj()
    t_term.rewards = [1.0]
    t_term.terminated = True
    t_trunc = Traj()
    t_trunc.rewards = [1.0]
    t_trunc.truncated = True
    rates = traj_end_rates([t_term, t_trunc, Traj()])
    assert rates["n_traj"] == 2
    assert abs(rates["termination_rate"] - 0.5) < 1e-9
    assert abs(rates["truncation_rate"] - 0.5) < 1e-9


def test_snapshot_every_3_fires_independent_of_eval_cadence(tmp_path: Path) -> None:
    """Regression for the v4.6 cadence bug: --snapshot-every 3 must fire at
    updates 0, 3, 6, 9, ... even though eval only runs on update % 5 == 0.
    If maybe_snapshot were only reachable inside the `update % 5 == 0` eval
    gate, updates 3, 6, and 9 (all not multiples of 5) would never snapshot.
    """
    latest_p = tmp_path / "p.pt"
    latest_s = tmp_path / "s.pt"
    mgr = CheckpointManager(
        latest_player=latest_p,
        latest_spawner=latest_s,
        best_player=default_best_path(latest_p),
        best_spawner=default_best_path(latest_s),
        snapshot_dir=tmp_path / "snapshots",
        snapshot_every=3,
    )
    player = PlayerV4(d_model=16, hidden=16)
    spawner = SpawnerV4(d_model=16, hidden=16)

    fired = []
    for update in range(10):
        eval_ran = update % 5 == 0  # mirrors the training loop's eval gate
        eval_p_payload = eval_s_payload = None
        if eval_ran:
            eval_p_payload = pack_player_ckpt(player, d_model=16, hidden=16, update=update)
            eval_s_payload = pack_spawner_ckpt(
                spawner, d_model=16, hidden=16, update=update, aim_scale=40.0,
            )
        p_payload, s_payload = snapshot_gate_payloads(
            update, eval_ran, eval_p_payload, eval_s_payload, player, spawner,
            d_model=16, hidden=16, aim_scale=40.0,
        )
        if mgr.maybe_snapshot(update, p_payload, s_payload):
            fired.append(update)

    # Multiples of 3 in [0, 10): includes 3, 6, 9 which are NOT multiples of 5.
    assert fired == [0, 3, 6, 9]
    for u in (3, 6, 9):
        assert (tmp_path / "snapshots" / f"player_update_{u}.pt").exists()
        assert (tmp_path / "snapshots" / f"spawner_update_{u}.pt").exists()


def test_snapshot_gate_reuses_eval_payload_on_eval_tick() -> None:
    player = PlayerV4(d_model=16, hidden=16)
    spawner = SpawnerV4(d_model=16, hidden=16)
    eval_p = pack_player_ckpt(
        player, d_model=16, hidden=16, update=5, eval_metrics={"newP_vs_scripted": 3.0},
    )
    eval_s = pack_spawner_ckpt(
        spawner, d_model=16, hidden=16, update=5, aim_scale=40.0,
        eval_metrics={"flee_vs_newS_det_stoch": 1.0},
    )
    p_payload, s_payload = snapshot_gate_payloads(
        5, True, eval_p, eval_s, player, spawner, d_model=16, hidden=16, aim_scale=40.0,
    )
    assert p_payload is eval_p and s_payload is eval_s
    assert "eval" in p_payload and "eval" in s_payload

    # Non-eval tick: freshly built payloads without eval metrics.
    p2, s2 = snapshot_gate_payloads(
        6, False, None, None, player, spawner, d_model=16, hidden=16, aim_scale=40.0,
    )
    assert "eval" not in p2 and "eval" not in s2
    assert p2["update"] == 6 and s2["update"] == 6


def test_ppo_player_diag_ratio_near_one_at_start() -> None:
    device = torch.device("cpu")
    player = PlayerV4(d_model=32, hidden=32)
    opt = torch.optim.Adam(player.parameters(), lr=1e-3)
    env = Qrokkun26Env(seed=11)
    pt, _st, _surv = run_episode(
        env,
        player,
        None,
        device,
        max_steps=40,
        sample=True,
        train_player=True,
        train_spawner=False,
        temp_p=1.0,
        temp_s=1.0,
        rng_jitter=False,
    )
    n, diag = ppo_update_player(
        player, opt, [pt], device, clip=0.2, epochs=1, minibatch=64,
        entropy_coef=0.01, value_coef=0.5, gamma=0.99, lam=0.95,
    )
    assert n > 0
    assert "approx_kl" in diag and "clipfrac" in diag
    assert "ratio_mean" in diag and "ratio_std" in diag
    assert "entropy" in diag and "explained_variance" in diag
    # Sanity at update start: ratio ~ 1 before first step (diag captures pre-step)
    assert abs(diag["ratio_mean"] - 1.0) < 0.05


def test_ppo_spawner_diag_ratio_near_one_at_start() -> None:
    device = torch.device("cpu")
    player = PlayerV4(d_model=32, hidden=32)
    spawner = SpawnerV4(d_model=32, hidden=32)
    opt = torch.optim.Adam(spawner.parameters(), lr=1e-3)
    env = Qrokkun26Env(seed=11)
    _pt, st, _surv = run_episode(
        env,
        player,
        spawner,
        device,
        max_steps=120,
        sample=True,
        train_player=False,
        train_spawner=True,
        temp_p=1.0,
        temp_s=1.0,
        rng_jitter=False,
    )
    n, diag = ppo_update_spawner(
        spawner, opt, [st], device, clip=0.2, epochs=1, minibatch=64,
        entropy_coef=0.01, value_coef=0.5, gamma=0.99, lam=0.95,
    )
    assert n > 0
    assert "approx_kl" in diag and "clipfrac" in diag
    assert "ratio_mean" in diag and "ratio_std" in diag
    assert "entropy" in diag and "explained_variance" in diag
    # Sanity at update start: ratio ~ 1 before first step (diag captures pre-step)
    assert abs(diag["ratio_mean"] - 1.0) < 0.05
