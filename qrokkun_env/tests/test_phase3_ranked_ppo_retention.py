"""Tests for tools/phase3_ranked_ppo_retention.py (Phase 3 BC-retention control).

STRICT TDD RED PHASE: ``tools/phase3_ranked_ppo_retention.py`` does not exist
yet, so every test below fails at collection with the same
``ModuleNotFoundError`` for ``tools.phase3_ranked_ppo_retention`` -- that is
the expected RED signal. No production code is added by this file.

Contract under specification
============================

* Input is a STRICT ``--init-checkpoint``: it must be a production
  ``player_ranked_topk`` checkpoint loaded through
  ``qrokkun_env.agents.player_checkpoints`` (legacy PlayerV1/PlayerV4 dict
  checkpoints, unknown architectures and tampered state dicts are refused).
  The harness verifies and reports the source checkpoint file SHA-256, the
  recorded state-dict SHA-256, the action list and the observation schema.
* The canonical V1 teacher dataset is recollected exactly ONCE locally
  (data seed 4; 240 episodes from seed 20000; up to 4200 frames/episode;
  700 frames/episode cap; 12% held out by episode) and its identity hash is
  verified/reported. Held frames are used ONLY for teacher-agreement
  diagnostics and NEVER as PPO rollout data.
* Two paired arms start from the exact same loaded initial state:
  ``frozen`` (zero optimizer steps, never mutated) and ``ppo`` (the current
  scripted-Player PPO algorithm from ``qrokkun_env.train.player_v1``, reused
  verbatim where contract-compatible, with the same hyperparameters).
* PPO rollouts reset the built-in scripted env on deterministic seeds from a
  fixed declared schedule, but ACTIONS ARE SAMPLED from the current policy
  during collection (never argmax), storing old log-probs, values and dones.
  GAE is computed per episode so advantages never bleed across boundaries.
* 200 PPO updates by default, 8 complete episodes per update, 4200 frames
  per episode cap; snapshots/evaluations at updates 0, 10, 25, 50, 100, 200.
* Evaluation is deterministic argmax on seeds 3000..3029 for up to 4200
  frames and reports mean/median/pstdev/min/max plus explicit censoring.
* Pre-registered interpretation (retention, never promotion): the initial
  gate requires mean >= 25s AND median >= 25s. Only if it passes does the
  control run at all. Final retention passes only if the final PPO mean and
  median are each >= 80% of the frozen/initial reference AND held teacher
  agreement declined by no more than 0.05 absolute; the first snapshot where
  either threshold breaks is always reported.
* No learned Spawner, no self-play, no random resets, no NAS, no teacher
  data inside PPO batches, and no checkpoint promotion anywhere.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qrokkun_env.agents.obs_v4 import (  # noqa: E402
    BULLET_FEAT_V4,
    MAX_BULLETS_V4,
    PLAYER_FEAT_V4,
    encode_obs,
)
from qrokkun_env.agents.player_checkpoints import (  # noqa: E402
    CheckpointError,
    file_sha256,
    load_ranked_top_k_checkpoint,
    save_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.env import ACTIONS, Qrokkun26Env  # noqa: E402
from qrokkun_env.train import player_v1 as scripted_ppo  # noqa: E402

# This import is expected to fail with ModuleNotFoundError until the tool
# exists -- that failure IS the RED signal this file pins down.
from tools import phase3_ranked_ppo_retention as ret  # noqa: E402

TEACHER_CKPT = _REPO_ROOT / "artifacts" / "nas_tmp_runs" / "player_gpu.pt"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tiny_net(seed: int = 0, hidden: int = 16, top_k: int = 8) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def _write_ranked_ckpt(tmp_path: Path, name: str = "init.pt", **kwargs: Any) -> tuple[Path, PlayerRankedTopK]:
    net = _tiny_net(**kwargs)
    path = tmp_path / name
    save_player_checkpoint(net, path, source_tool="tests")
    return path, net


def _fake_rollout(rewards: list[float], dones: list[bool], values: list[float]) -> Any:
    n = len(rewards)
    return ret.Rollout(
        seed=0,
        player=[np.zeros(PLAYER_FEAT_V4, dtype=np.float32) for _ in range(n)],
        bullets=[np.zeros((MAX_BULLETS_V4, BULLET_FEAT_V4), dtype=np.float32) for _ in range(n)],
        pad=[np.ones(MAX_BULLETS_V4, dtype=np.bool_) for _ in range(n)],
        actions=[0] * n,
        log_probs=[0.0] * n,
        values=list(values),
        rewards=list(rewards),
        dones=list(dones),
        elapsed=float(n) / 60.0,
        censored=not dones[-1] if dones else True,
    )


def _snapshot_record(update: int, mean: float, median: float, agreement: float) -> dict[str, Any]:
    return {
        "update": update,
        "evaluation": {"mean": mean, "median": median},
        "teacher_diagnostics": {"agreement": agreement},
    }


# --------------------------------------------------------------------------- #
# 1. strict checkpoint loading / provenance
# --------------------------------------------------------------------------- #
def test_load_initial_checkpoint_accepts_ranked_topk(tmp_path: Path) -> None:
    path, net = _write_ranked_ckpt(tmp_path)
    loaded, meta = ret.load_initial_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, PlayerRankedTopK)
    assert meta["architecture"] == "player_ranked_topk"
    assert state_dict_sha256(loaded.state_dict()) == state_dict_sha256(net.state_dict())


def test_load_initial_checkpoint_refuses_legacy_playerv1_dict(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy_v1.pt"
    torch.save(
        {"state_dict": {"body.0.weight": torch.zeros(4, 4)}, "hidden": 256, "actions": list(ACTIONS)},
        legacy,
    )
    with pytest.raises(CheckpointError):
        ret.load_initial_checkpoint(legacy, torch.device("cpu"))


def test_load_initial_checkpoint_refuses_other_architecture(tmp_path: Path) -> None:
    path, net = _write_ranked_ckpt(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["architecture"] = "player_v4"
    bad = tmp_path / "bad_arch.pt"
    torch.save(ckpt, bad)
    with pytest.raises(CheckpointError):
        ret.load_initial_checkpoint(bad, torch.device("cpu"))


def test_load_initial_checkpoint_refuses_tampered_state_dict(tmp_path: Path) -> None:
    path, _net = _write_ranked_ckpt(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key = sorted(ckpt["state_dict"])[0]
    ckpt["state_dict"][key] = ckpt["state_dict"][key] + 1.0
    bad = tmp_path / "tampered.pt"
    torch.save(ckpt, bad)
    with pytest.raises(CheckpointError):
        ret.load_initial_checkpoint(bad, torch.device("cpu"))


def test_checkpoint_provenance_reports_hashes_and_schema(tmp_path: Path) -> None:
    path, net = _write_ranked_ckpt(tmp_path)
    _loaded, meta = ret.load_initial_checkpoint(path, torch.device("cpu"))
    prov = ret.checkpoint_provenance(path, meta)
    assert prov["file_sha256"] == file_sha256(path)
    assert prov["state_dict_sha256"] == state_dict_sha256(net.state_dict())
    assert prov["actions"] == list(ACTIONS)
    assert prov["observation"]["player_feat"] == PLAYER_FEAT_V4
    assert prov["observation"]["bullet_feat"] == BULLET_FEAT_V4
    assert prov["observation"]["max_bullets"] == MAX_BULLETS_V4
    assert prov["architecture"] == "player_ranked_topk"
    assert prov["schema_verified"] is True
    json.dumps(prov)  # provenance must stay JSON-serializable


# --------------------------------------------------------------------------- #
# 2. pre-registered dataset / seed schedules / hyperparameters
# --------------------------------------------------------------------------- #
def test_canonical_dataset_constants() -> None:
    assert ret.DATA_SEED == 4
    assert ret.COLLECT_EPISODES == 240
    assert ret.COLLECT_SEED_START == 20000
    assert ret.COLLECT_MAX_STEPS == 4200
    assert ret.FRAMES_PER_EPISODE_CAP == 700
    assert ret.HELD_OUT_FRAC == pytest.approx(0.12)


def test_ppo_hyperparameters_match_current_scripted_player_ppo() -> None:
    """Hyperparameters are copied from the current scripted-Player PPO CLI
    defaults (qrokkun_env/train/player_v1.py), never reinvented."""
    src = Path(scripted_ppo.__file__).read_text()

    def default_of(flag: str) -> float:
        m = re.search(rf'--{flag}",\s*type=(?:float|int),\s*default=([0-9.e-]+)\)', src)
        assert m is not None, f"could not find default for --{flag} in {scripted_ppo.__file__}"
        return float(m.group(1))

    assert ret.PPO_LR == default_of("lr")
    assert ret.PPO_GAMMA == default_of("gamma")
    assert ret.PPO_LAMBDA == default_of("lam")
    assert ret.PPO_CLIP == default_of("clip")
    assert ret.PPO_ENTROPY_COEF == default_of("entropy")
    assert ret.PPO_VALUE_COEF == default_of("value-coef")
    assert ret.PPO_EPOCHS == int(default_of("ppo-epochs"))
    assert ret.PPO_MINIBATCH == int(default_of("minibatch"))
    assert ret.EPISODES_PER_UPDATE == int(default_of("rollouts"))
    assert ret.PPO_MAX_GRAD_NORM == 1.0


def test_reuses_project_gae_and_reward_verbatim() -> None:
    assert ret.gae is scripted_ppo.gae
    assert ret.shaped_reward is scripted_ppo.shaped_reward


def test_ppo_hyperparameters_dict_records_every_knob() -> None:
    knobs = ret.ppo_hyperparameters()
    for key in (
        "lr",
        "gamma",
        "lam",
        "clip",
        "entropy_coef",
        "value_coef",
        "ppo_epochs",
        "minibatch",
        "max_grad_norm",
        "optimizer",
        "adam_eps",
        "advantage_normalization",
        "episodes_per_update",
        "max_frames_per_episode",
        "updates",
        "rollout_seed_start",
        "torch_seed",
        "value_bootstrap",
    ):
        assert key in knobs, key
    assert knobs["optimizer"] == "adam"
    json.dumps(knobs)


def test_rollout_seed_schedule_is_fixed_declared_and_deterministic() -> None:
    s0 = ret.rollout_seed_schedule(0)
    s1 = ret.rollout_seed_schedule(1)
    assert len(s0) == ret.EPISODES_PER_UPDATE
    assert s0 == ret.rollout_seed_schedule(0)  # deterministic
    assert set(s0).isdisjoint(s1)  # no reuse across updates
    assert s0[0] == ret.PPO_ROLLOUT_SEED_START
    assert all(isinstance(s, int) for s in s0)
    # never overlaps the evaluation seed window
    assert set(s0).isdisjoint(set(ret.eval_seed_list()))


def test_eval_seed_window_is_canonical() -> None:
    seeds = ret.eval_seed_list()
    assert seeds == list(range(3000, 3030))
    assert ret.EVAL_MAX_STEPS == 4200


def test_snapshot_schedule_is_preregistered_and_filtered() -> None:
    assert ret.SNAPSHOT_UPDATES == (0, 10, 25, 50, 100, 200)
    assert ret.snapshot_schedule(200) == [0, 10, 25, 50, 100, 200]
    assert ret.snapshot_schedule(30) == [0, 10, 25, 30]  # final update always included
    assert ret.snapshot_schedule(2) == [0, 2]
    assert ret.snapshot_schedule(0) == [0]


def test_default_update_budget() -> None:
    assert ret.PPO_UPDATES == 200
    assert ret.EPISODES_PER_UPDATE == 8
    assert ret.PPO_MAX_FRAMES == 4200


# --------------------------------------------------------------------------- #
# 3. on-policy sampled collection
# --------------------------------------------------------------------------- #
def test_collect_rollout_samples_actions_and_stores_old_log_probs_values_dones() -> None:
    net = _tiny_net(seed=3)
    device = torch.device("cpu")
    torch.manual_seed(0)
    roll = ret.collect_rollout(net, device, seed=50000, max_frames=80)

    n = len(roll.actions)
    assert n > 0
    assert len(roll.log_probs) == n
    assert len(roll.values) == n
    assert len(roll.dones) == n
    assert len(roll.rewards) == n
    assert len(roll.player) == n
    assert roll.seed == 50000
    # only the final transition may be terminal
    assert not any(roll.dones[:-1])

    argmax_actions = []
    with torch.no_grad():
        for i in range(n):
            p = torch.tensor(roll.player[i], dtype=torch.float32).unsqueeze(0)
            b = torch.tensor(roll.bullets[i], dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(roll.pad[i], dtype=torch.bool).unsqueeze(0)
            dist, value = net(p, b, m)
            lp = float(dist.log_prob(torch.tensor([roll.actions[i]])).item())
            assert lp == pytest.approx(roll.log_probs[i], abs=1e-5)
            assert float(value.item()) == pytest.approx(roll.values[i], abs=1e-5)
            argmax_actions.append(int(dist.logits.argmax(-1).item()))
    # sampled, not deterministic argmax
    assert any(a != g for a, g in zip(roll.actions, argmax_actions))


def test_collect_rollout_is_deterministic_env_reset_no_random_reset() -> None:
    net = _tiny_net(seed=3)
    device = torch.device("cpu")
    torch.manual_seed(7)
    a = ret.collect_rollout(net, device, seed=50000, max_frames=40)
    torch.manual_seed(7)
    b = ret.collect_rollout(net, device, seed=50000, max_frames=40)
    assert a.actions == b.actions
    assert a.rewards == pytest.approx(b.rewards)
    assert np.allclose(np.stack(a.player), np.stack(b.player))


def test_collect_rollout_marks_censored_episodes() -> None:
    net = _tiny_net(seed=3)
    device = torch.device("cpu")
    torch.manual_seed(0)
    roll = ret.collect_rollout(net, device, seed=50001, max_frames=5)
    assert len(roll.actions) == 5
    assert roll.censored is True
    assert roll.dones[-1] is False


# --------------------------------------------------------------------------- #
# 4. GAE boundaries / batch construction
# --------------------------------------------------------------------------- #
def test_gae_is_computed_per_episode_and_never_crosses_boundaries() -> None:
    device = torch.device("cpu")
    r1 = _fake_rollout([1.0, 1.0, 1.0], [False, False, False], [0.5, 0.4, 0.3])
    r2 = _fake_rollout([2.0, 2.0], [False, True], [0.1, 0.2])
    adv, ret_t = ret.compute_gae_for_rollouts([r1, r2], ret.PPO_GAMMA, ret.PPO_LAMBDA, device)

    a1, t1 = scripted_ppo.gae(r1.rewards, r1.values, r1.dones, ret.PPO_GAMMA, ret.PPO_LAMBDA, device)
    a2, t2 = scripted_ppo.gae(r2.rewards, r2.values, r2.dones, ret.PPO_GAMMA, ret.PPO_LAMBDA, device)
    assert torch.allclose(adv, torch.cat([a1, a2]))
    assert torch.allclose(ret_t, torch.cat([t1, t2]))

    # a naive concatenated computation would bleed r2's value into r1's tail
    joined, _ = scripted_ppo.gae(
        r1.rewards + r2.rewards, r1.values + r2.values, r1.dones + r2.dones, ret.PPO_GAMMA, ret.PPO_LAMBDA, device
    )
    assert not torch.allclose(joined, adv)


def test_rollouts_to_batch_contains_only_on_policy_rollout_data() -> None:
    device = torch.device("cpu")
    r1 = _fake_rollout([1.0, 1.0], [False, True], [0.5, 0.4])
    r2 = _fake_rollout([1.0], [True], [0.2])
    batch = ret.rollouts_to_batch([r1, r2], device)
    assert int(batch["player"].shape[0]) == 3
    assert int(batch["actions"].shape[0]) == 3
    assert int(batch["old_log_probs"].shape[0]) == 3
    assert int(batch["advantages"].shape[0]) == 3
    assert int(batch["returns"].shape[0]) == 3
    assert "teacher_logits" not in batch  # teacher data NEVER enters PPO batches


def test_ppo_update_reports_kl_clipfrac_ev_entropy_losses_and_steps() -> None:
    net = _tiny_net(seed=5)
    device = torch.device("cpu")
    opt = torch.optim.Adam(net.parameters(), lr=ret.PPO_LR)
    torch.manual_seed(0)
    rollouts = [ret.collect_rollout(net, device, seed=50000 + i, max_frames=30) for i in range(2)]
    metrics = ret.ppo_update(net, opt, rollouts, device)
    for key in (
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "entropy",
        "policy_loss",
        "value_loss",
        "total_loss",
        "optimizer_steps",
        "n_samples",
    ):
        assert key in metrics, key
    assert metrics["optimizer_steps"] > 0
    assert metrics["n_samples"] == sum(len(r.actions) for r in rollouts)
    json.dumps(metrics)


# --------------------------------------------------------------------------- #
# 5. deterministic evaluation + censoring
# --------------------------------------------------------------------------- #
def test_evaluate_deterministic_is_argmax_and_reproducible() -> None:
    net = _tiny_net(seed=11)
    device = torch.device("cpu")
    seeds = [3000, 3001]
    a = ret.evaluate_deterministic(net, device, seeds, max_steps=120)
    b = ret.evaluate_deterministic(net, device, seeds, max_steps=120)
    assert a == b
    assert [r["seed"] for r in a] == seeds
    assert all(set(r) >= {"seed", "elapsed", "censored"} for r in a)


def test_summarize_evaluation_reports_stats_and_censoring() -> None:
    results = [
        {"seed": 3000, "elapsed": 10.0, "censored": False},
        {"seed": 3001, "elapsed": 20.0, "censored": False},
        {"seed": 3002, "elapsed": 70.0, "censored": True},
    ]
    s = ret.summarize_evaluation(results)
    assert s["mean"] == pytest.approx(statistics.mean([10.0, 20.0, 70.0]))
    assert s["median"] == pytest.approx(20.0)
    assert s["pstdev"] == pytest.approx(statistics.pstdev([10.0, 20.0, 70.0]))
    assert s["min"] == pytest.approx(10.0)
    assert s["max"] == pytest.approx(70.0)
    assert s["n"] == 3
    assert s["censored_count"] == 1
    assert s["censor_rate"] == pytest.approx(1 / 3)
    assert s["per_seed"]["3002"] == pytest.approx(70.0)
    assert s["per_seed_censored"]["3002"] is True
    json.dumps(s)


# --------------------------------------------------------------------------- #
# 6. teacher diagnostics from held frames only
# --------------------------------------------------------------------------- #
def test_teacher_diagnostics_report_agreement_kl_and_buckets() -> None:
    device = torch.device("cpu")
    net = _tiny_net(seed=2)
    env = Qrokkun26Env(seed=1234)
    env.reset(seed=1234)
    players, bullets, pads, elapsed = [], [], [], []
    for _ in range(12):
        p, b, m = encode_obs(env)
        players.append(p)
        bullets.append(b)
        pads.append(m)
        elapsed.append(float(env.elapsed))
        env.step(0)
    tensors = {
        "player": torch.tensor(np.stack(players), dtype=torch.float32),
        "bullets": torch.tensor(np.stack(bullets), dtype=torch.float32),
        "pad": torch.tensor(np.stack(pads), dtype=torch.bool),
        "teacher_logits": torch.randn(12, len(ACTIONS)),
        "elapsed": torch.tensor(elapsed, dtype=torch.float32),
    }
    diag = ret.teacher_diagnostics(net, tensors, device)
    assert 0.0 <= diag["agreement"] <= 1.0
    assert "teacher_to_student_kl" in diag
    assert "by_elapsed_bucket" in diag
    assert "by_bullet_bucket" in diag
    json.dumps(diag)


# --------------------------------------------------------------------------- #
# 7. frozen arm never mutates anything
# --------------------------------------------------------------------------- #
def test_frozen_arm_is_bit_identical_and_takes_zero_optimizer_steps() -> None:
    device = torch.device("cpu")
    net = _tiny_net(seed=9)
    before = state_dict_sha256(net.state_dict())
    report = ret.run_frozen_arm(net, device, snapshots=[0, 1, 2], eval_seeds=[3000, 3001], eval_max_steps=60)
    after = state_dict_sha256(net.state_dict())

    assert report["optimizer_steps"] == 0
    assert before == after
    assert report["state_dict_sha256_initial"] == before
    assert report["state_dict_sha256_final"] == before
    assert set(report["state_dict_sha256_by_update"].values()) == {before}
    assert report["state_dict_bit_identical"] is True
    assert report["metrics_bit_identical"] is True
    summaries = [json.dumps(s["evaluation"], sort_keys=True) for s in report["snapshots"]]
    assert len(set(summaries)) == 1  # identical metrics at every snapshot
    assert [s["update"] for s in report["snapshots"]] == [0, 1, 2]
    json.dumps(report)


# --------------------------------------------------------------------------- #
# 8. gate math (pre-registered interpretation, never promotion)
# --------------------------------------------------------------------------- #
def test_initial_gate_thresholds() -> None:
    assert ret.INITIAL_GATE_MEAN_MIN == 25.0
    assert ret.INITIAL_GATE_MEDIAN_MIN == 25.0
    assert ret.RETENTION_FRACTION == pytest.approx(0.80)
    assert ret.MAX_HELD_AGREEMENT_DROP == pytest.approx(0.05)


def test_evaluate_initial_gate_pass_and_fail() -> None:
    ok = ret.evaluate_initial_gate({"mean": 30.0, "median": 26.0})
    assert ok["gate_pass"] is True
    bad_mean = ret.evaluate_initial_gate({"mean": 24.9, "median": 30.0})
    assert bad_mean["gate_pass"] is False
    assert "mean" in bad_mean["reason"]
    bad_median = ret.evaluate_initial_gate({"mean": 30.0, "median": 24.9})
    assert bad_median["gate_pass"] is False
    assert "median" in bad_median["reason"]
    edge = ret.evaluate_initial_gate({"mean": 25.0, "median": 25.0})
    assert edge["gate_pass"] is True  # >= is inclusive


def test_evaluate_retention_passes_when_all_thresholds_hold() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.70}
    snapshots = [
        _snapshot_record(0, 40.0, 40.0, 0.70),
        _snapshot_record(10, 36.0, 36.0, 0.68),
        _snapshot_record(20, 34.0, 34.0, 0.66),
    ]
    out = ret.evaluate_retention(reference, snapshots)
    assert out["retention_pass"] is True
    assert out["mean_threshold"] == pytest.approx(32.0)
    assert out["median_threshold"] == pytest.approx(32.0)
    assert out["agreement_floor"] == pytest.approx(0.65)
    assert out["first_breaking_snapshot"] is None
    assert out["promotion"] is False


def test_evaluate_retention_fails_on_final_mean_and_reports_first_break() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.70}
    snapshots = [
        _snapshot_record(0, 40.0, 40.0, 0.70),
        _snapshot_record(10, 33.0, 40.0, 0.70),
        _snapshot_record(25, 20.0, 40.0, 0.70),  # mean breaks here
        _snapshot_record(50, 18.0, 40.0, 0.70),
    ]
    out = ret.evaluate_retention(reference, snapshots)
    assert out["retention_pass"] is False
    assert out["final_mean_pass"] is False
    assert out["final_median_pass"] is True
    assert out["first_breaking_snapshot"] == 25
    assert "mean" in out["reason"]


def test_evaluate_retention_fails_on_teacher_agreement_drop() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.70}
    snapshots = [
        _snapshot_record(0, 40.0, 40.0, 0.70),
        _snapshot_record(10, 39.0, 39.0, 0.60),  # 0.10 absolute drop > 0.05
    ]
    out = ret.evaluate_retention(reference, snapshots)
    assert out["retention_pass"] is False
    assert out["agreement_pass"] is False
    assert out["final_agreement_drop"] == pytest.approx(0.10)
    assert out["first_breaking_snapshot"] == 10
    assert "agreement" in out["reason"]


def test_evaluate_retention_edge_is_inclusive() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.70}
    snapshots = [_snapshot_record(0, 40.0, 40.0, 0.70), _snapshot_record(10, 32.0, 32.0, 0.65)]
    out = ret.evaluate_retention(reference, snapshots)
    assert out["retention_pass"] is True
    assert out["first_breaking_snapshot"] is None


# --------------------------------------------------------------------------- #
# 9. fail-closed: no env rollouts / no PPO when the initial gate fails
# --------------------------------------------------------------------------- #
def test_run_experiment_fails_closed_before_ppo_when_initial_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt, _net = _write_ranked_ckpt(tmp_path)
    out_dir = tmp_path / "out_failclosed"
    args = ret.apply_mode_defaults(
        ret.build_parser().parse_args(
            [
                "--init-checkpoint",
                str(ckpt),
                "--teacher",
                str(TEACHER_CKPT),
                "--out-dir",
                str(out_dir),
                "--quick",
            ]
        )
    )
    # quick mode relaxes the gate; force the pre-registered production gate back
    args.initial_gate_mean_min = ret.INITIAL_GATE_MEAN_MIN
    args.initial_gate_median_min = ret.INITIAL_GATE_MEDIAN_MIN

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("PPO/rollout code must not run when the initial gate fails")

    monkeypatch.setattr(ret, "collect_rollout", _boom)
    monkeypatch.setattr(ret, "ppo_update", _boom)
    monkeypatch.setattr(ret, "run_ppo_arm", _boom)
    monkeypatch.setattr(ret, "run_frozen_arm", _boom)

    report = ret.run_experiment(args, torch.device("cpu"))
    assert report["initial_gate"]["gate_pass"] is False
    assert report["control_ran"] is False
    assert report["ppo_arm"] is None
    assert report["frozen_arm"] is None
    assert report["retention"] is None
    assert "fail" in report["status"] or report["status"] == "failed_closed"
    assert report["initial_gate"]["reason"]
    json.dumps(report)


# --------------------------------------------------------------------------- #
# 10. quick end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not TEACHER_CKPT.exists(), reason="teacher checkpoint not available locally")
def test_quick_end_to_end_runs_both_arms_and_writes_artifacts(tmp_path: Path) -> None:
    ckpt, _net = _write_ranked_ckpt(tmp_path)
    out_dir = tmp_path / "out_e2e"
    args = ret.apply_mode_defaults(
        ret.build_parser().parse_args(
            [
                "--init-checkpoint",
                str(ckpt),
                "--teacher",
                str(TEACHER_CKPT),
                "--out-dir",
                str(out_dir),
                "--quick",
            ]
        )
    )
    report = ret.run_experiment(args, torch.device("cpu"))

    assert report["control_ran"] is True
    assert report["initial_gate"]["gate_pass"] is True
    assert report["frozen_arm"]["optimizer_steps"] == 0
    assert report["ppo_arm"]["optimizer_steps"] > 0
    assert report["ppo_arm"]["total_episodes"] == args.updates * args.episodes_per_update
    assert report["ppo_arm"]["total_frames"] > 0
    assert report["retention"]["promotion"] is False
    assert report["dataset"]["hash"]
    assert report["dataset"]["n_held_frames"] > 0
    assert report["provenance"]["init_checkpoint"]["file_sha256"] == file_sha256(ckpt)
    assert report["provenance"]["teacher"]["file_sha256"] == file_sha256(TEACHER_CKPT)
    assert report["provenance"]["git_commit"] is not None
    assert "dirty" in report["provenance"]
    assert report["provenance"]["torch_version"] == torch.__version__
    assert report["provenance"]["device"] == "cpu"

    # frozen arm never moved
    assert report["frozen_arm"]["state_dict_bit_identical"] is True
    assert report["frozen_arm"]["state_dict_sha256_initial"] == report["provenance"]["init_checkpoint"][
        "state_dict_sha256"
    ]

    # artifacts on disk
    report_path = out_dir / "report.json"
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text())
    assert on_disk["control_ran"] is True

    jsonl = out_dir / "ppo_updates.jsonl"
    assert jsonl.exists()
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == args.updates
    assert rows[0]["update"] == 1
    for row in rows:
        assert "approx_kl" in row and "clip_fraction" in row and "explained_variance" in row
        assert "scripted_survival_mean" in row

    # strict per-snapshot + final PPO checkpoints exist and remain reloadable
    # by the architecture loader, but can never be reused as production init.
    snap_updates = [s["update"] for s in report["ppo_arm"]["snapshots"]]
    assert snap_updates == ret.snapshot_schedule(args.updates)
    for snap in report["ppo_arm"]["snapshots"]:
        path = Path(snap["checkpoint"])
        assert path.exists()
        loaded, meta = load_ranked_top_k_checkpoint(path, torch.device("cpu"))
        assert isinstance(loaded, PlayerRankedTopK)
        assert meta["experimental"] is True
        assert meta["production_compatible"] is False
        with pytest.raises(CheckpointError):
            ret.load_initial_checkpoint(path, torch.device("cpu"))
    final_ckpt = Path(report["ppo_arm"]["final_checkpoint"])
    assert final_ckpt.exists()
    load_ranked_top_k_checkpoint(final_ckpt, torch.device("cpu"))
    with pytest.raises(CheckpointError):
        ret.load_initial_checkpoint(final_ckpt, torch.device("cpu"))


def test_report_is_json_serializable_without_tensors(tmp_path: Path) -> None:
    ckpt, _net = _write_ranked_ckpt(tmp_path)
    out_dir = tmp_path / "out_json"
    args = ret.apply_mode_defaults(
        ret.build_parser().parse_args(
            [
                "--init-checkpoint",
                str(ckpt),
                "--teacher",
                str(TEACHER_CKPT),
                "--out-dir",
                str(out_dir),
                "--quick",
            ]
        )
    )
    report = ret.run_experiment(args, torch.device("cpu"))
    text = json.dumps(report)  # must not raise
    assert "tensor(" not in text
    assert isinstance(report["knobs"], dict)
    assert report["knobs"]["updates"] == args.updates


# --------------------------------------------------------------------------- #
# 11. forbidden mechanisms
# --------------------------------------------------------------------------- #
def test_module_never_uses_learned_spawner_selfplay_nas_or_promotion() -> None:
    src = Path(ret.__file__).read_text()
    lowered = src.lower()
    for banned in ("spawner_v1", "spawner_v2", "spawner_v3", "spawner_v4", "train_spawner"):
        assert banned not in lowered, banned
    for token in ("self_play", "selfplay", "random_fraction", "--random-fraction", "nas_search"):
        assert token not in lowered, token
    # promotion is explicitly disclaimed, never performed
    assert "promote" not in lowered.replace("never promote", "").replace("no promotion", "")
    # env resets are always seeded (no random resets)
    assert re.search(r"\.reset\(\s*\)", src) is None
    for call in re.findall(r"\.reset\(([^)]*)\)", src):
        assert "seed" in call, call


def test_teacher_frames_are_never_used_as_ppo_rollouts() -> None:
    src = Path(ret.__file__).read_text()
    ppo_section = src[src.index("def ppo_update") : src.index("def ppo_update") + 4000]
    assert "teacher" not in ppo_section.lower()
    collect_section = src[src.index("def collect_rollout") : src.index("def collect_rollout") + 3000]
    assert "teacher" not in collect_section.lower()
