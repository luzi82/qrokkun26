from __future__ import annotations

import copy

import json

import multiprocessing as mp

import os

import signal

import numpy as np

import pytest

import torch

from qrokkun_ai.v21.agents.player_v21 import PlayerV21

from qrokkun_ai.v21.obs_v21 import encode_obs_v21

from qrokkun_ai.v21.train import train_player_v21 as v21

from qrokkun_ai.v24.core import LearnedEnv, Spawner

from qrokkun_ai.v30 import compare as v30_compare, core, run as v30_run, supervise as v30_sup

def _tiny_models(seed: int = 17, hidden: int = 8) -> tuple[PlayerV21, Spawner]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return PlayerV21(hidden=hidden), Spawner().requires_grad_(False).eval()

def test_encode_obs_fast_exact_parity():
    _, s = _tiny_models()
    gen = torch.Generator(device="cpu").manual_seed(25_100_000)
    env = LearnedEnv(lambda obs: core.sample_spawner_action(s, obs, gen), seed=100_000)
    env.reset(seed=100_000)
    for act in (0, 1, 3, 5, 7, 2, 4, 6, 8):
        a = encode_obs_v21(env)
        b = core.encode_obs_fast(env)
        for x, y in zip(a, b):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert np.array_equal(x.view(np.uint8), y.view(np.uint8))
        env.step(act)

def test_spawner_action_only_exact_parity_and_rng_state():
    _, s = _tiny_models()
    x = np.linspace(-0.5, 0.5, 325, dtype=np.float32)
    g1 = torch.Generator(device="cpu").manual_seed(123456)
    g2 = torch.Generator(device="cpu").manual_seed(123456)
    for det in (False, True):
        a_ref, _ = s.sample(torch.from_numpy(x).unsqueeze(0), g1, deterministic=det)
        a_fast = core.sample_spawner_action(s, x, g2, deterministic=det)
        assert np.array_equal(a_ref[0].numpy().view(np.uint8), a_fast.view(np.uint8))
        assert torch.equal(g1.get_state(), g2.get_state())

def test_assemble_batch_matches_v21_gae_and_teacher():
    p, s = _tiny_models()
    trajs, _ = core.collect_serial(p, s, [100_000, 100_001], cap=4, validation=False)
    batch_v30 = core.assemble_batch(trajs, p, "cpu")

    ros: list[v21.RolloutV21] = []
    for ep in trajs:
        ro = v21.RolloutV21(seed=ep.seed)
        for t in range(ep.frames):
            ro.player.append(ep.player[t])
            ro.bullets.append(ep.bullets[t])
            ro.pad.append(ep.pad[t])
            ro.actions.append(int(ep.actions[t]))
            ro.log_probs.append(float(ep.log_probs[t]))
            ro.values.append(float(ep.values[t]))
            ro.rewards.append(float(ep.rewards[t]))
            ro.dones.append(bool(ep.dones[t]))
        pt = torch.from_numpy(ep.player)
        bt = torch.from_numpy(ep.bullets)
        mt = torch.from_numpy(ep.pad)
        ea, el = core.compute_kinematic_expert_targets(pt, bt, mt, horizon=p.lookahead_horizon, top_k=p.top_k)
        ro.expert_actions = [int(v.item()) for v in ea]
        ro.expert_logits = [row.numpy() for row in el]
        ro.terminated = ep.terminated
        ro.censored = ep.censored
        ro.truncated = ep.truncated
        ro.bootstrap_value = ep.bootstrap_value
        ros.append(ro)

    batch_ref = v21.rollouts_to_batch_v21(ros, 0.9995, 0.999, torch.device("cpu"), reward_scale=0.05)
    for k in batch_ref:
        assert torch.equal(
            batch_v30[k].contiguous().view(torch.uint8),
            batch_ref[k].contiguous().view(torch.uint8),
        ), f"mismatch in {k}"

def test_cap_boundary_collision_vs_censor():
    from qrokkun_env.gv20 import Bullet

    p, s = _tiny_models()
    # Normal 2-frame survival -> censored=True, terminated=False, non-zero bootstrap path exercised
    meta_surv, _ = core.run_episode(p, s, 100_000, cap=2, validation=False)
    assert meta_surv["censored"] is True and meta_surv["terminated"] is False and meta_surv["truncated"] is True

    # Force real collision on exact cap=1 frame -> terminated=True, censored=False, bootstrap=0.0
    orig_reset = LearnedEnv.reset

    def hit_on_frame_1(self, seed=None):
        out = orig_reset(self, seed=seed)
        self.bullets.append(Bullet(self.px, self.py, 0.0, 0.0))
        return out

    LearnedEnv.reset = hit_on_frame_1
    try:
        meta_hit, _ = core.run_episode(p, s, 100_000, cap=1, validation=False)
    finally:
        LearnedEnv.reset = orig_reset
    assert meta_hit["frames"] == 1
    assert meta_hit["terminated"] is True and meta_hit["censored"] is False and meta_hit["bootstrap_value"] == 0.0

def test_persistent_pool_worker_reuse_and_serial_parity():
    p, s = _tiny_models()
    q = copy.deepcopy(p)
    pool = core.EpisodePool(p, s, max_slots=3, max_cap=5, workers=2, pin_cores="off")
    try:
        init_pids = {w["pid"] for w in pool.worker_init_info}
        assert len(init_pids) == 2

        for ver, val in [(0, True), (0, False), (1, False)]:
            if ver == 1:
                with torch.no_grad():
                    for param in p.parameters():
                        param.add_(0.01)
                q.load_state_dict(p.state_dict())
            pool.sync_weights(p.state_dict(), policy_version=ver)
            m_trajs = pool.collect([100_000, 100_001, 100_002], cap=4, validation=val)
            c_trajs, _ = core.collect_serial(q, s, [100_000, 100_001, 100_002], cap=4, validation=val)
            for a, b in zip(m_trajs, c_trajs):
                assert a.summary() == b.summary()
                for field, _, _ in core.TRAJ_FIELDS:
                    assert np.array_equal(
                        getattr(a, field).view(np.uint8),
                        getattr(b, field).view(np.uint8),
                    )
        stats = pool.shutdown_and_stats()
        assert {w["pid"] for w in stats} == init_pids
        assert sum(int(w["tasks"]) for w in stats) == 9
    finally:
        pool.close()
    assert not mp.active_children()

def test_weight_diff_edge_cases():
    assert core.weight_diff({"x": torch.tensor([1.0])}, {"x": torch.tensor([1.0])}) == []
    assert core.weight_diff({"x": torch.tensor([0.0])}, {"x": torch.tensor([-0.0])}) == ["x"]
    assert core.weight_diff({}, {"x": torch.tensor([1.0])}) == ["x"]
    with pytest.raises(FloatingPointError):
        core.finite({"x": torch.tensor([float("nan")])})

def test_cooperative_sigterm_stops_and_reaps_pool(tmp_path, monkeypatch):
    out = tmp_path / "stop_m"
    orig_recv = core.EpisodePool._recv_one
    shm_names = []
    ready_results = []

    def trigger_stop(self, conn, proc, stop_check):
        if not shm_names:
            shm_names.extend([self.w_shm.name, self.t_shm.name])
            assert conn.poll(15), "worker did not reach startup readiness"
            os.kill(os.getpid(), signal.SIGTERM)
        result = orig_recv(self, conn, proc, stop_check)
        ready_results.append(result)
        return result

    monkeypatch.setattr(core.EpisodePool, "_recv_one", trigger_stop)
    rc = v30_run.cli([
        "--arm", "Pv30m", "--out-dir", str(out), "--source-commit", "c" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--pin-cores", "off", "--updates", "1", "--episodes", "2", "--cap", "4", "--validation", "2", "--hidden", "8", "--workers", "2",
    ])
    assert rc == 130
    st = json.loads((out / "status.json").read_text())
    assert st["status"] == "stopped"
    assert not mp.active_children()
    from multiprocessing.shared_memory import SharedMemory
    assert len(shm_names) == 2
    for name in shm_names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
    assert ready_results == []

def test_resume_rejects_completed_run_and_manifest_mismatch(tmp_path):
    out = tmp_path / "done_run"
    args = [
        "--arm", "Pv30c", "--out-dir", str(out), "--source-commit", "f" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--updates", "1", "--episodes", "1", "--cap", "2", "--validation", "1", "--hidden", "8",
    ]
    assert v30_run.main(args) == 0
    with pytest.raises(ValueError, match="already complete"):
        v30_run.main([*args, "--resume"])
    with pytest.raises(ValueError, match="manifest mismatch"):
        v30_run.main([*args, "--seed", "19", "--resume"])

def test_source_digest_excludes_tests_and_docs():
    d = core.source_digest()
    assert len(d) == 64