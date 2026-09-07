"""PPO likelihood ratio regression: stored vs recomputed log_prob at temp=1.0."""

from __future__ import annotations

import pytest
import numpy as np
import torch

from qrokkun_env import constants as C
from qrokkun_env.agents.obs_v4 import encode_obs
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.agents.spawner_v4 import SpawnerV4, spawn_continuous
from qrokkun_env.env import Qrokkun26Env
from qrokkun_env.train.both_v4 import (
    act_player,
    act_spawner,
    build_parser,
    reject_non_unit_temp,
    run_episode,
)


def test_temp_argparse_defaults_are_one() -> None:
    args = build_parser().parse_args([])
    assert args.temp_p == 1.0
    assert args.temp_s == 1.0


@pytest.mark.parametrize("flag", ["--temp-p", "--temp-s"])
def test_temp_not_one_exits_with_clear_error(flag: str) -> None:
    args = build_parser().parse_args([flag, "1.15"])
    with pytest.raises(SystemExit, match=r"not supported") as ei:
        reject_non_unit_temp(args)
    msg = str(ei.value)
    assert "1.15" in msg
    assert "--temp-p" in msg and "--temp-s" in msg


def test_temp_one_is_accepted() -> None:
    args = build_parser().parse_args(["--temp-p", "1.0", "--temp-s", "1.0"])
    reject_non_unit_temp(args)
    assert args.temp_p == 1.0 and args.temp_s == 1.0


def _live_kinds(bullets: np.ndarray, pad: np.ndarray) -> set[int]:
    kinds: set[int] = set()
    for i, is_pad in enumerate(pad):
        if is_pad:
            continue
        kinds.add(int(np.argmax(bullets[i, 5:9])))
    return kinds


def _scripted_burn(env: Qrokkun26Env, n_steps: int, action: str) -> None:
    for _ in range(n_steps):
        if env.dead:
            break
        env.step(action)


def _player_ratio(net: PlayerV4, stored_lps, actions, players, bullets, pads, device):
    P = torch.tensor(np.array(players), dtype=torch.float32, device=device)
    B = torch.tensor(np.array(bullets), dtype=torch.float32, device=device)
    M = torch.tensor(np.array(pads), dtype=torch.bool, device=device)
    A = torch.tensor(actions, dtype=torch.int64, device=device)
    old = torch.tensor(stored_lps, dtype=torch.float32, device=device)
    with torch.no_grad():
        dist, _value = net(P, B, M)
        lp = dist.log_prob(A)
        return torch.exp(lp - old)


def _spawner_ratio(net: SpawnerV4, stored_lps, births, aims, kinds, players, bullets, pads, device):
    P = torch.tensor(np.array(players), dtype=torch.float32, device=device)
    B = torch.tensor(np.array(bullets), dtype=torch.float32, device=device)
    M = torch.tensor(np.array(pads), dtype=torch.bool, device=device)
    birth_a = torch.tensor(births, dtype=torch.float32, device=device)
    aim_a = torch.tensor(aims, dtype=torch.float32, device=device)
    kind_a = torch.tensor(kinds, dtype=torch.int64, device=device)
    old = torch.tensor(stored_lps, dtype=torch.float32, device=device)
    with torch.no_grad():
        birth, aim, kind, _value = net(P, B, M)
        lp = (
            birth.log_prob(birth_a).sum(-1)
            + aim.log_prob(aim_a).sum(-1)
            + kind.log_prob(kind_a)
        )
        return torch.exp(lp - old)


def test_player_ppo_ratio_before_update_is_one() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = PlayerV4(d_model=32, hidden=64).to(device)
    net.eval()

    stored_lps: list[float] = []
    actions: list[int] = []
    players: list = []
    bullets: list = []
    pads: list = []
    px_seen: list[float] = []
    pvx_seen: list[float] = []
    nonempty = 0
    multi_kind = 0

    # Scripted burn then sample (non-empty attention + moved player).
    for seed, action, n in ((0, "e", 110), (1, "nw", 120), (2, "s", 100)):
        env = Qrokkun26Env(seed=seed)
        env.reset()
        _scripted_burn(env, n, action)
        if env.dead:
            env.reset()
            _scripted_burn(env, 80, "idle")
        a, lp, _v, p, b, m = act_player(net, env, device, sample=True, temp=1.0)
        stored_lps.append(lp)
        actions.append(a)
        players.append(p)
        bullets.append(b)
        pads.append(m)
        px_seen.append(float(env.px))
        pvx_seen.append(float(env.pvx))
        if not bool(m.all()):
            nonempty += 1
        if len(_live_kinds(b, m)) >= 2:
            multi_kind += 1

    # Legal off-center position/velocity (not default center + 0 vel).
    env = Qrokkun26Env(seed=3)
    env.reset()
    env.px = C.FIELD_X + C.PLAYER_MARGIN + 48.0
    env.py = C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN - 36.0
    env.pvx = 55.0
    env.pvy = -28.0
    spawn_continuous(env, (1.0, 0.0), (0.2, -0.1), kind=0, rng_jitter=False)
    spawn_continuous(env, (0.0, 1.0), (-0.3, 0.4), kind=2, rng_jitter=False)
    spawn_continuous(env, (-1.0, 0.2), (0.0, 0.0), kind=3, rng_jitter=False)
    a, lp, _v, p, b, m = act_player(net, env, device, sample=True, temp=1.0)
    stored_lps.append(lp)
    actions.append(a)
    players.append(p)
    bullets.append(b)
    pads.append(m)
    px_seen.append(float(env.px))
    pvx_seen.append(float(env.pvx))
    if not bool(m.all()):
        nonempty += 1
    if len(_live_kinds(b, m)) >= 2:
        multi_kind += 1

    assert nonempty >= 1, "need non-empty bullet attention"
    assert multi_kind >= 1, "need at least one obs with multiple bullet kinds"
    assert max(px_seen) - min(px_seen) > 1.0
    assert any(abs(v) > 1.0 for v in pvx_seen)

    ratio = _player_ratio(net, stored_lps, actions, players, bullets, pads, device)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), ratio.tolist()


def test_spawner_ppo_ratio_nonempty_obs_is_one() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    net = SpawnerV4(d_model=32, hidden=64).to(device)
    net.eval()

    stored_lps: list[float] = []
    births: list = []
    aims: list = []
    kinds: list = []
    players: list = []
    bullets: list = []
    pads: list = []
    nonempty = 0

    for seed, action, n in ((4, "e", 110), (5, "n", 130), (6, "sw", 100), (7, "idle", 140)):
        env = Qrokkun26Env(seed=seed)
        env.reset()
        _scripted_burn(env, n, action)
        if env.dead or not env.bullets:
            env.reset()
            spawn_continuous(env, (0.8, -0.4), (0.1, 0.2), kind=1, rng_jitter=False)
            spawn_continuous(env, (-0.5, 0.9), (-0.2, 0.0), kind=0, rng_jitter=False)
        act, lp, _v, p, b, m = act_spawner(net, env, device, sample=True, temp=1.0)
        stored_lps.append(lp)
        births.append(act["birth"])
        aims.append(act["aim"])
        kinds.append(act["kind"])
        players.append(p)
        bullets.append(b)
        pads.append(m)
        if not bool(m.all()):
            nonempty += 1

    assert nonempty >= 1, "spawner path needs non-empty obs"

    ratio = _spawner_ratio(
        net, stored_lps, births, aims, kinds, players, bullets, pads, device
    )
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), ratio.tolist()


def test_run_episode_ratio_before_optimizer_is_one() -> None:
    """Short integration: pack traj like PPO update, recompute log_prob before opt.step."""
    torch.manual_seed(2)
    device = torch.device("cpu")
    player = PlayerV4(d_model=32, hidden=64).to(device)
    spawner = SpawnerV4(d_model=32, hidden=64).to(device)
    player.eval()
    spawner.eval()

    ptraj = straj = None
    for seed in range(20, 40):
        env = Qrokkun26Env(seed=seed)
        pt, st, _t = run_episode(
            env,
            player,
            spawner,
            device,
            max_steps=180,
            sample=True,
            train_player=True,
            train_spawner=True,
            temp_p=1.0,
            temp_s=1.0,
            rng_jitter=False,
        )
        if pt.rewards and st.rewards:
            ptraj, straj = pt, st
            break
    assert ptraj is not None and straj is not None, "need packed P and S trajs"
    assert any(not bool(np.asarray(m).all()) for m in ptraj.pad), "P traj should see bullets"
    assert any(not bool(np.asarray(m).all()) for m in straj.pad), "S traj should see bullets"

    p_ratio = _player_ratio(
        player,
        ptraj.log_probs,
        ptraj.actions,
        ptraj.player,
        ptraj.bullets,
        ptraj.pad,
        device,
    )
    assert torch.allclose(p_ratio, torch.ones_like(p_ratio), atol=1e-5), p_ratio.tolist()

    s_ratio = _spawner_ratio(
        spawner,
        straj.log_probs,
        [a["birth"] for a in straj.actions],
        [a["aim"] for a in straj.actions],
        [a["kind"] for a in straj.actions],
        straj.player,
        straj.bullets,
        straj.pad,
        device,
    )
    assert torch.allclose(s_ratio, torch.ones_like(s_ratio), atol=1e-5), s_ratio.tolist()
