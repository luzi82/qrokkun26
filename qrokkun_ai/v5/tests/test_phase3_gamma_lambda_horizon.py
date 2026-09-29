"""Gamma/lambda horizon facade and corrected post-step bootstrap.

Formal 10k training is out of scope. These tests use tiny synthetic fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qrokkun_ai.v1.agents.player_v1 import PlayerV1
from qrokkun_ai.v5.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4, encode_obs
from qrokkun_ai.v5.agents.player_checkpoints import load_player_checkpoint, save_player_checkpoint
from qrokkun_ai.v5.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_ai.v5.tools import phase3_gamma_lambda_horizon as horizon
from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention as ret
from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention_aux as aux


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {opt for action in parser._actions for opt in action.option_strings}


def test_parser_requires_named_arm_and_leaves_legacy_parser_unchanged() -> None:
    from qrokkun_ai.v5.tools.phase3_gamma_lambda_horizon import build_parser

    base = ["--init-checkpoint", "init.pt", "--teacher", "teacher.pt"]
    legacy = aux.build_parser()
    legacy_before = _option_strings(legacy)
    assert "--horizon-arm" not in legacy_before

    control = build_parser().parse_args([*base, "--horizon-arm", "control"])
    long_arm = build_parser().parse_args([*base, "--horizon-arm", "long"])
    assert control.horizon_arm == "control"
    assert long_arm.horizon_arm == "long"
    assert _option_strings(legacy) == legacy_before

    fresh_legacy = aux.build_parser()
    assert "--horizon-arm" not in _option_strings(fresh_legacy)
    with pytest.raises(SystemExit):
        fresh_legacy.parse_args([*base, "--horizon-arm", "control"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(base)
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--horizon-arm", "other"])


def test_horizon_config_is_locked_and_legacy_is_unchanged() -> None:
    assert aux.horizon_config(None) == (ret.PPO_GAMMA, ret.PPO_LAMBDA, False)
    assert aux.horizon_config("control") == (0.99, 0.95, True)
    assert aux.horizon_config("long") == (0.9995, 0.999, True)
    assert aux.horizon_config("control")[0] * aux.horizon_config("control")[1] == pytest.approx(0.9405)
    assert aux.horizon_config("long")[0] * aux.horizon_config("long")[1] == pytest.approx(0.9985005)
    with pytest.raises(ValueError):
        aux.horizon_config("other")
    legacy = ret.ppo_hyperparameters()
    assert legacy["gamma"] == 0.99
    assert legacy["lam"] == 0.95
    assert legacy["value_bootstrap"] is False
    assert legacy["truncation_treated_as_terminal"] is True
    assert "horizon_arm" not in legacy
    assert "bootstrap_source" not in legacy


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    init = tmp_path / "init.pt"
    torch.manual_seed(0)
    save_player_checkpoint(PlayerRankedTopK(top_k=4, hidden=8), init, source_tool="tests")
    teacher = tmp_path / "teacher.pt"
    torch.save({"hidden": 8, "state_dict": PlayerV1(hidden=8).state_dict()}, teacher)
    return init, teacher


def _teacher_tensors(n: int = 4) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return {
        "player": torch.randn(n, PLAYER_FEAT_V4, generator=g),
        "bullets": torch.randn(n, MAX_BULLETS_V4, BULLET_FEAT_V4, generator=g),
        "pad": torch.zeros(n, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.randn(n, len(ACTIONS), generator=g),
        "elapsed": torch.rand(n, generator=g),
    }


def _fake_rollout(seed: int) -> ret.Rollout:
    rng = np.random.default_rng(seed)
    n = 2
    return ret.Rollout(
        seed=int(seed),
        player=[rng.standard_normal(PLAYER_FEAT_V4).astype(np.float32) for _ in range(n)],
        bullets=[
            rng.standard_normal((MAX_BULLETS_V4, BULLET_FEAT_V4)).astype(np.float32) for _ in range(n)
        ],
        pad=[np.zeros(MAX_BULLETS_V4, dtype=np.bool_) for _ in range(n)],
        actions=[0, 1],
        log_probs=[-0.1, -0.2],
        values=[0.2, 0.3],
        rewards=[0.0, 0.1],
        dones=[False, True],
        elapsed=1.0,
        censored=False,
    )


def _install_training_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[int]]:
    boundaries = {"eval": [], "collect": []}

    def fake_eval(_net, _device, seeds, _max_steps):
        boundaries["eval"].append(list(seeds))
        return [{"seed": int(seed), "elapsed": 40.0, "censored": False} for seed in seeds]

    def fake_dataset(_teacher, _device, **_kwargs):
        tensors = _teacher_tensors()
        return {
            "hash": "synthetic-dataset",
            "n_episodes": 1,
            "n_held_episodes": 1,
            "n_train_episodes": 1,
            "n_held_frames": 4,
            "n_train_frames": 4,
            "teacher_file_sha256": "teacher-sha",
            "train_tensors": tensors,
            "held_tensors": tensors,
        }

    def fake_collect(_net, _device, seed, max_frames, **_kwargs):
        boundaries["collect"].append(int(seed))
        return _fake_rollout(int(seed))

    monkeypatch.setattr(aux, "evaluate_deterministic", fake_eval)
    monkeypatch.setattr(aux, "collect_aux_dataset", fake_dataset)
    monkeypatch.setattr(aux, "collect_rollout", fake_collect)
    return boundaries["eval"], boundaries["collect"]


def _legacy_oracle(args: argparse.Namespace) -> dict:
    knobs = ret.ppo_hyperparameters()
    knobs["updates"] = args.updates
    knobs["episodes_per_update"] = args.episodes_per_update
    knobs["max_frames_per_episode"] = args.max_frames
    knobs["eval_seeds"] = args.eval_seeds
    knobs["eval_max_steps"] = args.eval_max_steps
    knobs["initial_gate_mean_min"] = args.initial_gate_mean_min
    knobs["initial_gate_median_min"] = args.initial_gate_median_min
    knobs["data_episodes"] = args.data_episodes
    knobs["data_max_steps"] = args.data_max_steps
    knobs["data_frames_cap"] = args.data_frames_cap
    knobs["target_retention_grad_ratio"] = aux.TARGET_RETENTION_GRAD_RATIO
    knobs["retention_objective"] = "phase2_ranked_multiseed.hybrid_loss"
    knobs["retention_hard_weight"] = aux.HYBRID_HARD_WEIGHT
    knobs["retention_soft_weight"] = aux.HYBRID_SOFT_WEIGHT
    knobs["retention_temperature"] = aux.HYBRID_TEMPERATURE
    knobs["retention_sampler_seed"] = aux.RETENTION_SAMPLER_SEED
    knobs["retention_calibration_seed"] = aux.RETENTION_CALIBRATION_SEED
    knobs["retention_diagnostic_seed"] = aux.RETENTION_DIAGNOSTIC_SEED
    knobs["calibration_grad_samples"] = aux.CALIBRATION_GRAD_SAMPLES
    if getattr(args, "rollout_seed_start", None) is not None:
        knobs["rollout_seed_start"] = int(args.rollout_seed_start)
    if hasattr(args, "terminal_teacher_diagnostics"):
        knobs["terminal_teacher_diagnostics"] = bool(args.terminal_teacher_diagnostics)
    return knobs


def _cli_args(parser: argparse.ArgumentParser, run_dir: Path, init: Path, teacher: Path, *, arm: str | None, resume: bool) -> argparse.Namespace:
    argv = [
        "--init-checkpoint", str(init),
        "--teacher", str(teacher),
        "--run-dir", str(run_dir),
        "--device", "cpu",
        "--seed", "17",
        "--max-updates", "1",
        "--rollout-seed-start", "210000",
    ]
    if arm is not None:
        argv.extend(["--horizon-arm", arm])
    if resume:
        argv.append("--resume")
    from qrokkun_ai.v5.tools.phase3_gamma_lambda_horizon import build_parser

    chosen = build_parser() if arm is not None else parser
    return aux.apply_mode_defaults(chosen.parse_args(argv))


def test_horizon_contract_knobs_and_resume_mismatch_before_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    eval_calls, collect_calls = _install_training_stubs(monkeypatch)
    with pytest.raises(ValueError, match="horizon"):
        aux.run_experiment(argparse.Namespace(horizon_arm="sideways"), device)
    assert eval_calls == []
    assert collect_calls == []

    init, teacher = _write_inputs(tmp_path)
    legacy_dir = tmp_path / "legacy"
    legacy_args = _cli_args(aux.build_parser(), legacy_dir, init, teacher, arm=None, resume=False)
    legacy_report = aux.run_experiment(legacy_args, device)
    assert legacy_report["knobs"] == _legacy_oracle(legacy_args)
    legacy_saved = json.loads((legacy_dir / "run.json").read_text())["knobs"]
    assert legacy_saved == json.loads(json.dumps(_legacy_oracle(legacy_args)))
    assert "horizon_arm" not in legacy_saved
    assert "bootstrap_source" not in legacy_saved

    control_dir = tmp_path / "control"
    control_args = _cli_args(aux.build_parser(), control_dir, init, teacher, arm="control", resume=False)
    control_report = aux.run_experiment(control_args, device)
    saved = json.loads((control_dir / "run.json").read_text())
    knobs = saved["knobs"]
    assert knobs["horizon_arm"] == "control"
    assert knobs["gamma"] == 0.99
    assert knobs["lam"] == 0.95
    assert knobs["value_bootstrap"] is True
    assert knobs["truncation_treated_as_terminal"] is False
    assert knobs["bootstrap_source"] == aux.BOOTSTRAP_SOURCE_POST_STEP
    assert control_report["knobs"] == knobs
    _net, meta = load_player_checkpoint(control_dir / "ppo_aux_update_0.pt", "cpu")
    assert meta["extra"]["ppo_knobs"] == knobs
    assert meta["experimental"] is True
    assert meta["production_compatible"] is False

    original = (control_dir / "run.json").read_text()

    def _reject(arm: str, mutate) -> None:
        payload = json.loads(original)
        mutate(payload)
        (control_dir / "run.json").write_text(json.dumps(payload))
        n_eval, n_collect = len(eval_calls), len(collect_calls)
        args = _cli_args(aux.build_parser(), control_dir, init, teacher, arm=arm, resume=True)
        with pytest.raises(ret.RunStateError, match="contract mismatch"):
            aux.run_experiment(args, device)
        assert len(eval_calls) == n_eval
        assert len(collect_calls) == n_collect

    _reject("long", lambda _payload: None)
    _reject("control", lambda payload: payload["knobs"].__setitem__("gamma", 0.5))
    _reject("control", lambda payload: payload["knobs"].__setitem__("lam", 0.5))
    _reject("control", lambda payload: payload["knobs"].__setitem__("value_bootstrap", False))


class _TinyEnv:
    """Deterministic one-attribute stand-in. ``die_on_frame`` is 1-based."""

    def __init__(self, seed: int = 0, die_on_frame: int | None = None) -> None:
        self.seed = seed
        self.die_on_frame = die_on_frame
        self.frame = 0
        self.px = 200.0
        self.py = 300.0
        self.pvx = 0.0
        self.pvy = 0.0
        self.elapsed = 0.0
        self.spawn_acc = 0.0
        self.bullets: list = []
        self.dead = False

    def reset(self, seed: int | None = None):
        self.frame = 0
        self.elapsed = 0.0
        self.dead = False
        self.px = 200.0
        return None

    def step(self, _action: int):
        self.frame += 1
        self.elapsed = self.frame / 60.0
        self.px += 3.0
        done = self.die_on_frame is not None and self.frame == self.die_on_frame
        self.dead = bool(done)
        return None, 0.25, done, {"elapsed": self.elapsed}


class _ProbeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.samples = 0

    def forward(self, player, bullets, pad):
        logits = torch.zeros(player.shape[0], len(ACTIONS), device=player.device)
        dist = Categorical(logits=logits + self.anchor * 0)
        sample = dist.sample

        def counted(*args, **kwargs):
            self.samples += 1
            return sample(*args, **kwargs)

        dist.sample = counted
        value = player[:, 4] + self.anchor * 0
        return dist, value


def test_post_step_bootstrap_uses_successor_value_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ret, "Qrokkun26Env", lambda seed=0: _TinyEnv(seed=seed))
    net = _ProbeNet()
    device = torch.device("cpu")
    rollout = ret.collect_rollout(net, device, seed=3, max_frames=2, bootstrap_truncation=True)
    replay = _TinyEnv(seed=3)
    replay.reset(seed=3)
    for action in rollout.actions:
        replay.step(action)
    player, bullets, pad = encode_obs(replay)
    with torch.no_grad():
        _dist, successor = net(
            torch.tensor(player, dtype=torch.float32).unsqueeze(0),
            torch.tensor(bullets, dtype=torch.float32).unsqueeze(0),
            torch.tensor(pad, dtype=torch.bool).unsqueeze(0),
        )
    assert rollout.censored is True
    assert rollout.bootstrap_value == pytest.approx(float(successor.item()))
    assert rollout.bootstrap_value != pytest.approx(rollout.values[-1])
    assert net.samples == len(rollout.actions)
    assert rollout.rewards == [0.25, 0.25]


def test_terminal_on_cap_has_zero_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ret, "Qrokkun26Env", lambda seed=0: _TinyEnv(seed=seed, die_on_frame=1))
    net = _ProbeNet()
    forwards = {"n": 0}
    forward = net.forward

    def counting(player, bullets, pad):
        forwards["n"] += 1
        return forward(player, bullets, pad)

    net.forward = counting
    rollout = ret.collect_rollout(net, torch.device("cpu"), seed=4, max_frames=1, bootstrap_truncation=True)
    assert rollout.censored is False
    assert rollout.dones == [True]
    assert rollout.bootstrap_value == 0.0
    assert forwards["n"] == 1
    assert net.samples == 1


def test_legacy_collector_keeps_zero_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ret, "Qrokkun26Env", lambda seed=0: _TinyEnv(seed=seed))
    net = _ProbeNet()
    rollout = ret.collect_rollout(net, torch.device("cpu"), seed=5, max_frames=2)
    assert rollout.censored is True
    assert rollout.bootstrap_value == 0.0
    assert net.samples == len(rollout.actions)


def _analytic_gae(rewards, values, dones, bootstrap, censored, gamma, lam):
    extended = [float(v) for v in values] + [float(bootstrap) if censored else 0.0]
    running = 0.0
    adv = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * extended[t + 1] * mask - extended[t]
        running = delta + gamma * lam * mask * running
        adv[t] = running
    adv_t = torch.tensor(adv, dtype=torch.float32)
    ret_t = adv_t + torch.tensor(extended[:-1], dtype=torch.float32)
    return adv_t, ret_t


def _gae_rollout(**kwargs) -> ret.Rollout:
    n = len(kwargs["rewards"])
    base = dict(
        seed=0,
        player=[np.zeros(PLAYER_FEAT_V4, dtype=np.float32) for _ in range(n)],
        bullets=[np.zeros((MAX_BULLETS_V4, BULLET_FEAT_V4), dtype=np.float32) for _ in range(n)],
        pad=[np.ones(MAX_BULLETS_V4, dtype=np.bool_) for _ in range(n)],
        actions=[0] * n,
        log_probs=[0.0] * n,
        elapsed=0.1,
    )
    base.update(kwargs)
    return ret.Rollout(**base)


def test_bootstrapped_gae_two_steps_and_terminal_mask() -> None:
    device = torch.device("cpu")
    censored = _gae_rollout(
        values=[0.5, 0.25],
        rewards=[1.0, 2.0],
        dones=[False, False],
        censored=True,
        bootstrap_value=4.0,
    )
    terminal = _gae_rollout(
        values=[0.4, 0.1],
        rewards=[0.5, 1.5],
        dones=[False, True],
        censored=False,
        bootstrap_value=9.0,
        seed=1,
    )
    for gamma, lam in ((0.99, 0.95), (0.9995, 0.999)):
        adv, ret_t = ret.compute_gae_for_rollouts(
            [censored, terminal], gamma, lam, device, bootstrap_truncation=True,
        )
        a1, t1 = _analytic_gae(censored.rewards, censored.values, censored.dones, 4.0, True, gamma, lam)
        a2, t2 = _analytic_gae(terminal.rewards, terminal.values, terminal.dones, 9.0, False, gamma, lam)
        assert torch.equal(adv, torch.cat([a1, a2]).to(device))
        assert torch.equal(ret_t, torch.cat([t1, t2]).to(device))
        # The nonzero tail must move the censored return, and the planted 9.0 must not.
        legacy_adv, _legacy_ret = ret.compute_gae_for_rollouts(
            [censored], gamma, lam, device, bootstrap_truncation=False,
        )
        assert not torch.equal(adv[:2], legacy_adv)
        assert float(t2[-1]) == pytest.approx(1.5 - 0.1 + 0.1)

    batch = ret.rollouts_to_batch(
        [censored, terminal], device, gamma=0.9995, lam=0.999, bootstrap_truncation=True,
    )
    assert "teacher_logits" not in batch
    assert not hasattr(censored, "teacher_logits")
    assert batch["returns"].shape[0] == 4
    assert batch["advantages"].dtype == torch.float32


def test_legacy_gae_matches_v1() -> None:
    from qrokkun_ai.v1.train import player_v1 as scripted_ppo

    device = torch.device("cpu")
    first = _gae_rollout(
        values=[0.5, 0.4, 0.3], rewards=[1.0, 1.0, 1.0], dones=[False, False, False], censored=True,
    )
    second = _gae_rollout(
        values=[0.1, 0.2], rewards=[2.0, 2.0], dones=[False, True], censored=False, seed=2,
        bootstrap_value=7.0,
    )
    adv, ret_t = ret.compute_gae_for_rollouts(
        [first, second], ret.PPO_GAMMA, ret.PPO_LAMBDA, device,
    )
    parts = [
        scripted_ppo.gae(r.rewards, r.values, r.dones, ret.PPO_GAMMA, ret.PPO_LAMBDA, device)
        for r in (first, second)
    ]
    assert torch.equal(adv, torch.cat([p[0] for p in parts]))
    assert torch.equal(ret_t, torch.cat([p[1] for p in parts]))
    batch = ret.rollouts_to_batch([first, second], device)
    assert torch.equal(batch["advantages"], adv)
    assert torch.equal(batch["returns"], ret_t)
    assert "teacher_logits" not in batch


def _censored_bootstrap_rollout() -> ret.Rollout:
    return _gae_rollout(
        values=[0.5, 0.25],
        rewards=[1.0, 2.0],
        dones=[False, False],
        censored=True,
        bootstrap_value=4.0,
    )


def test_both_horizon_arms_calibrate_and_update_on_own_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    rollouts = [_censored_bootstrap_rollout()]
    train = _teacher_tensors()
    calls: list[dict] = []
    returns: dict[str, torch.Tensor] = {}
    real_batch = aux.rollouts_to_batch

    def spy(rollouts_arg, device_arg, *args, **kwargs):
        batch = real_batch(rollouts_arg, device_arg, *args, **kwargs)
        calls.append(dict(kwargs))
        returns.setdefault(str(kwargs.get("gamma")), batch["returns"].detach().clone())
        return batch

    monkeypatch.setattr(aux, "rollouts_to_batch", spy)
    torch.manual_seed(0)
    seed_net = PlayerRankedTopK(top_k=4, hidden=8)
    alphas = {}
    for arm, gamma, lam in (("control", 0.99, 0.95), ("long", 0.9995, 0.999)):
        net = PlayerRankedTopK(top_k=4, hidden=8)
        net.load_state_dict(seed_net.state_dict())
        first = aux.calibrate_alpha(
            net, rollouts, train, device, minibatch=4, n_minibatches=2, horizon_arm=arm,
        )
        second = aux.calibrate_alpha(
            net, rollouts, train, device, minibatch=4, n_minibatches=2, horizon_arm=arm,
        )
        assert second["alpha"] == first["alpha"]
        assert first["target_ratio"] == 0.15
        alphas[arm] = first["alpha"]
        opt = torch.optim.Adam(net.parameters(), lr=aux.PPO_LR, eps=1e-8)
        aux.ppo_aux_update(
            net, opt, rollouts, train, first["alpha"], device,
            generator=aux.training_generator(),
            diagnostics_generator=aux.diagnostic_generator(),
            minibatch=4,
            horizon_arm=arm,
        )
        arm_calls = [call for call in calls if call.get("gamma") == gamma]
        assert arm_calls
        assert all(call["lam"] == lam and call["bootstrap_truncation"] is True for call in arm_calls)
    assert alphas["control"] != alphas["long"]
    assert not torch.equal(returns["0.99"], returns["0.9995"])


def test_update_one_reuses_calibration_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    collect_kwargs: list[dict] = []

    def fake_collect(_net, _device, seed, max_frames, **kwargs):
        collect_kwargs.append(dict(kwargs))
        return _fake_rollout(int(seed))

    monkeypatch.setattr(aux, "collect_rollout", fake_collect)
    net = PlayerRankedTopK(top_k=4, hidden=8)
    calib, first = aux.calibration_and_first_update_rollouts(
        net, device, episodes_per_update=8, max_frames=2,
        rollout_seed_start=210000, horizon_arm="long",
    )
    assert calib is first
    assert [roll.seed for roll in calib] == list(range(210000, 210008))
    assert collect_kwargs == [{"bootstrap_truncation": True}] * 8

    seen: dict[str, object] = {}
    real_calibrate = aux.calibrate_alpha
    real_update = aux.ppo_aux_update

    def spy_calibrate(net_arg, rollouts_arg, *args, **kwargs):
        seen["calibration_rollouts"] = rollouts_arg
        seen["calibration_arm"] = kwargs.get("horizon_arm")
        return real_calibrate(net_arg, rollouts_arg, *args, **kwargs)

    def spy_update(net_arg, opt, rollouts_arg, *args, **kwargs):
        seen.setdefault("update_rollouts", []).append(rollouts_arg)
        seen.setdefault("update_arms", []).append(kwargs.get("horizon_arm"))
        return real_update(net_arg, opt, rollouts_arg, *args, **kwargs)

    monkeypatch.setattr(aux, "calibrate_alpha", spy_calibrate)
    monkeypatch.setattr(aux, "ppo_aux_update", spy_update)
    monkeypatch.setattr(
        aux, "evaluate_deterministic",
        lambda _net, _device, seeds, _steps: [
            {"seed": int(seed), "elapsed": 30.0, "censored": False} for seed in seeds
        ],
    )
    args = argparse.Namespace(
        updates=1, episodes_per_update=8, max_frames=2, eval_seeds=[3000, 3001],
        eval_max_steps=1, out_dir=tmp_path, seed=17, rollout_seed_start=210000,
        max_updates=1, terminal_teacher_diagnostics=False,
    )
    aux.run_aux_arm(
        net, device, args, [0, 1], _teacher_tensors(), _teacher_tensors(), tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d",
        ppo_knobs={"horizon_arm": "control"}, minibatch=4, horizon_arm="control",
    )
    assert seen["calibration_arm"] == "control"
    assert seen["calibration_rollouts"] is seen["update_rollouts"][0]
    assert [roll.seed for roll in seen["calibration_rollouts"]] == list(range(210000, 210008))
    assert seen["update_arms"] == ["control"]


def test_later_rollouts_and_diagnostics_use_selected_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    collect_kwargs: list[dict] = []

    def fake_collect(_net, _device, seed, max_frames, **kwargs):
        collect_kwargs.append({"seed": int(seed), **kwargs})
        return _fake_rollout(int(seed))

    batch_kwargs: list[dict] = []
    real_batch = aux.rollouts_to_batch

    def spy_batch(rollouts_arg, device_arg, *args, **kwargs):
        batch_kwargs.append(dict(kwargs))
        return real_batch(rollouts_arg, device_arg, *args, **kwargs)

    monkeypatch.setattr(aux, "collect_rollout", fake_collect)
    monkeypatch.setattr(aux, "rollouts_to_batch", spy_batch)
    monkeypatch.setattr(
        aux, "evaluate_deterministic",
        lambda _net, _device, seeds, _steps: [
            {"seed": int(seed), "elapsed": 30.0, "censored": False} for seed in seeds
        ],
    )
    args = argparse.Namespace(
        updates=2, episodes_per_update=1, max_frames=2, eval_seeds=[3000, 3001],
        eval_max_steps=1, out_dir=tmp_path, seed=None, rollout_seed_start=210000,
        max_updates=2,
    )
    aux.run_aux_arm(
        PlayerRankedTopK(top_k=4, hidden=8), device, args, [0, 2],
        _teacher_tensors(), _teacher_tensors(), tmp_path,
        parent_state_dict_sha256="p", parent_file_sha256="f", dataset_hash="d",
        ppo_knobs={"horizon_arm": "long"}, minibatch=4, horizon_arm="long",
    )
    assert [row["seed"] for row in collect_kwargs] == [210000, 210001]
    assert all(row.get("bootstrap_truncation") is True for row in collect_kwargs)
    assert batch_kwargs
    assert all(
        call == {"gamma": 0.9995, "lam": 0.999, "bootstrap_truncation": True}
        for call in batch_kwargs
    )


def test_legacy_aux_update_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cpu")
    batch_kwargs: list[dict] = []
    real_batch = aux.rollouts_to_batch

    def spy_batch(rollouts_arg, device_arg, *args, **kwargs):
        batch_kwargs.append(dict(kwargs))
        return real_batch(rollouts_arg, device_arg, *args, **kwargs)

    monkeypatch.setattr(aux, "rollouts_to_batch", spy_batch)
    rollouts = [_fake_rollout(1)]
    train = _teacher_tensors()
    net = PlayerRankedTopK(top_k=4, hidden=8)
    opt = torch.optim.Adam(net.parameters(), lr=aux.PPO_LR, eps=1e-8)
    aux.ppo_aux_update(
        net, opt, rollouts, train, 0.1, device,
        generator=aux.training_generator(),
        diagnostics_generator=aux.diagnostic_generator(),
        minibatch=4,
    )
    assert batch_kwargs
    assert all(call == {} for call in batch_kwargs)
    collect_kwargs: list[dict] = []

    def fake_collect(_net, _device, seed, max_frames):
        collect_kwargs.append({"seed": seed})
        return _fake_rollout(int(seed))

    monkeypatch.setattr(aux, "collect_rollout", fake_collect)
    window = aux.collect_rollout_window(net, device, [7], max_frames=2)
    assert collect_kwargs == [{"seed": 7}]
    assert window[0].seed == 7


def _horizon_argv(init: Path, teacher: Path, run_dir: Path, arm: str, *extra: str) -> list[str]:
    return [
        "phase3_gamma_lambda_horizon",
        "--init-checkpoint", str(init),
        "--teacher", str(teacher),
        "--run-dir", str(run_dir),
        "--device", "cpu",
        "--seed", "17",
        "--max-updates", "1",
        "--rollout-seed-start", "210000",
        "--horizon-arm", arm,
        *extra,
    ]


def test_horizon_cli_contract_preflight_and_no_rollout_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    init, teacher = _write_inputs(tmp_path)
    _eval_calls, collect_calls = _install_training_stubs(monkeypatch)
    monkeypatch.setattr(sys, "argv", _horizon_argv(init, teacher, tmp_path / "bad", "sideways"))
    with pytest.raises(SystemExit):
        horizon.main()
    assert collect_calls == []

    monkeypatch.setattr(sys, "argv", _horizon_argv(init, teacher, tmp_path / "control", "control"))
    horizon.main()
    control_knobs = json.loads((tmp_path / "control" / "run.json").read_text())["knobs"]
    monkeypatch.setattr(sys, "argv", _horizon_argv(init, teacher, tmp_path / "long", "long"))
    horizon.main()
    long_knobs = json.loads((tmp_path / "long" / "run.json").read_text())["knobs"]
    assert control_knobs["horizon_arm"] == "control"
    assert long_knobs["horizon_arm"] == "long"
    assert control_knobs["gamma"] == 0.99 and control_knobs["lam"] == 0.95
    assert long_knobs["gamma"] == 0.9995 and long_knobs["lam"] == 0.999
    for key in (
        "value_bootstrap", "truncation_treated_as_terminal", "bootstrap_source",
        "lr", "clip", "entropy_coef", "value_coef", "ppo_epochs", "minibatch",
        "max_grad_norm", "adam_eps", "advantage_normalization",
        "target_retention_grad_ratio", "eval_seeds",
    ):
        assert control_knobs[key] == long_knobs[key]
    assert control_knobs["value_bootstrap"] is True
    assert control_knobs["truncation_treated_as_terminal"] is False
    assert control_knobs["bootstrap_source"] == "post_step_V(s')"
    assert control_knobs["eval_seeds"] == list(range(3000, 3030))
    before = len(collect_calls)
    monkeypatch.setattr(
        sys, "argv", _horizon_argv(init, teacher, tmp_path / "control", "long", "--resume"),
    )
    with pytest.raises(ret.RunStateError, match="contract mismatch"):
        horizon.main()
    assert len(collect_calls) == before
    payload = json.loads((tmp_path / "control" / "run.json").read_text())
    payload["knobs"]["gamma"] = 0.5
    (tmp_path / "control" / "run.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys, "argv", _horizon_argv(init, teacher, tmp_path / "control", "control", "--resume"),
    )
    with pytest.raises(ret.RunStateError, match="contract mismatch"):
        horizon.main()
    assert len(collect_calls) == before
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["arm_ran"] is True


def test_fresh_horizon_checkpoint_metadata_and_no_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    init, teacher = _write_inputs(tmp_path)
    _install_training_stubs(monkeypatch)
    run_dir = tmp_path / "long"
    monkeypatch.setattr(sys, "argv", _horizon_argv(init, teacher, run_dir, "long"))
    horizon.main()
    report = json.loads((run_dir / "report.json").read_text())
    contract = json.loads((run_dir / "run.json").read_text())
    assert report["arm_ran"] is True
    assert report["status"] == "max_updates"
    assert report["aux_arm"]["completed_updates"] == 1
    assert report["aux_arm"]["completed_updates"] != 10000
    assert report["stop_budget"]["effective_max_updates"] == 1
    assert report["retention"]["promotion"] is False
    assert contract["no_promotion"] is True
    knobs = contract["knobs"]
    assert knobs["eval_seeds"] == list(range(3000, 3030))
    assert knobs["gamma"] == 0.9995 and knobs["lam"] == 0.999
    _net, meta = load_player_checkpoint(run_dir / "ppo_aux_final.pt", "cpu")
    assert meta["experimental"] is True
    assert meta["production_compatible"] is False
    assert meta["extra"]["ppo_knobs"] == knobs
    assert meta["extra"]["eval_seed_window"] == list(range(3000, 3030))
    assert meta["extra"]["alpha"] != 0.38932983697041385
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == {"status": "max_updates", "arm_ran": True}


def test_horizon_poststep_forward_does_not_shift_rng() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)
    net = PlayerRankedTopK(top_k=4, hidden=8)
    torch.manual_seed(11)
    boot = ret.collect_rollout(net, device, seed=9, max_frames=4, bootstrap_truncation=True)
    state_after_bootstrap = torch.get_rng_state().clone()
    torch.manual_seed(11)
    plain = ret.collect_rollout(net, device, seed=9, max_frames=4)
    assert boot.actions == plain.actions
    assert torch.equal(state_after_bootstrap, torch.get_rng_state())
    assert boot.censored is True
    assert plain.bootstrap_value == 0.0
    replay = Qrokkun26Env(seed=9)
    replay.reset(seed=9)
    for action in boot.actions:
        replay.step(action)
    player, bullets, pad = encode_obs(replay)
    with torch.no_grad():
        _dist, successor = net(
            torch.tensor(player, dtype=torch.float32).unsqueeze(0),
            torch.tensor(bullets, dtype=torch.float32).unsqueeze(0),
            torch.tensor(pad, dtype=torch.bool).unsqueeze(0),
        )
    assert boot.bootstrap_value == pytest.approx(float(successor.item()))
    assert boot.bootstrap_value != pytest.approx(boot.values[-1])
