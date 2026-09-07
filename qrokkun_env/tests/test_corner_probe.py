"""v4.3 locked-P corner probe tests (spec E + Code reviewer lock assertion)."""

from __future__ import annotations

import pytest
import torch

from qrokkun_env import constants as C
from qrokkun_env.agents.spawner_v4 import SpawnerV4
from qrokkun_env.corner_probe import (
    probe_positions,
    run_corner_probe,
    run_locked_probe_episode,
)
from qrokkun_env.env import Qrokkun26Env


@pytest.fixture
def tiny_spawner():
    device = torch.device("cpu")
    torch.manual_seed(1)
    net = SpawnerV4(d_model=16, hidden=32).to(device)
    net.eval()
    return net, device


def test_nine_positions_in_bounds_and_bottom_right():
    pos = probe_positions()
    assert len(pos) == 9
    assert "bottom_right" in pos
    assert set(pos) >= {
        "center",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "top_mid",
        "bottom_mid",
        "left_mid",
        "right_mid",
    }
    x0 = C.FIELD_X + C.PLAYER_MARGIN
    y0 = C.FIELD_Y + C.PLAYER_MARGIN
    x1 = C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN
    y1 = C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN
    for name, (px, py) in pos.items():
        assert x0 - 1e-6 <= px <= x1 + 1e-6, name
        assert y0 - 1e-6 <= py <= y1 + 1e-6, name
    brx, bry = pos["bottom_right"]
    assert abs(brx - x1) < 1e-9
    assert abs(bry - y1) < 1e-9


def test_player_locked_during_probe(tiny_spawner):
    spawner, device = tiny_spawner
    pos = probe_positions()
    px, py = pos["bottom_right"]
    env = Qrokkun26Env(seed=5)
    # Monkey-patch: record every px,py change attempt by wrapping lock check mid-run
    result = run_locked_probe_episode(
        env, spawner, device, px, py, max_steps=90,
        sample_policy=False, rng_jitter=False, episode_seed=5,
    )
    assert result["player_locked"] is True
    assert len(result["positions_unique"]) == 1
    ux, uy = result["positions_unique"][0]
    assert abs(ux - px) < 1e-6 and abs(uy - py) < 1e-6
    # Explicit: final env coords match pinned corner
    assert abs(env.px - px) < 1e-6
    assert abs(env.py - py) < 1e-6
    assert env.pvx == 0.0 and env.pvy == 0.0


def test_corner_probe_schema_stable(tiny_spawner):
    spawner, device = tiny_spawner
    report = run_corner_probe(
        spawner,
        device,
        seeds=[4100, 4101],
        max_steps=60,
        sample_policy=False,
        rng_jitter=False,
    )
    assert report["schema"] == "both_v4_corner_probe.v1"
    assert report["n_positions"] == 9
    assert "bottom_right" in report
    assert report["bottom_right"]["name"] == "bottom_right"
    assert "note" in report and "new×new" in report["note"]
    for block in report["positions"]:
        assert set(block) >= {
            "name",
            "px",
            "py",
            "n_trials",
            "hit_rate",
            "first_hit_mean",
            "closest_approach_mean",
            "player_locked",
            "birth_aim_stats",
            "trials",
        }
        assert block["player_locked"] is True
        assert 0.0 <= block["hit_rate"] <= 1.0


def test_bottom_right_case_always_runs(tiny_spawner):
    spawner, device = tiny_spawner
    report = run_corner_probe(spawner, device, seeds=[4200], max_steps=45)
    names = [p["name"] for p in report["positions"]]
    assert names.count("bottom_right") == 1
    assert report["bottom_right"] is not None
    assert report["bottom_right"]["n_trials"] == 1
