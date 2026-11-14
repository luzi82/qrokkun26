"""Proportional unit & integration tests for qrokkun_ai.v28 (within AGENTS.md budget)."""
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
from qrokkun_ai.v28 import compare as v28_compare, core, run as v28_run, supervise as v28_sup


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
    batch_v28 = core.assemble_batch(trajs, p, "cpu")

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
            batch_v28[k].contiguous().view(torch.uint8),
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


def test_schedules_and_six_digit_integer_sorting(tmp_path):
    assert core.schedule_for(5) == (1, [0, 1, 2, 3, 4, 5])
    assert core.schedule_for(100) == (10, list(range(0, 101, 10)))
    assert core.schedule_for(1000) == (100, list(range(0, 1001, 100)))
    with pytest.raises(ValueError):
        core.schedule_for(7, nonformal=False)

    for u in (0, 5, 900, 2000, 10000, 100000):
        name = v28_run.ckpt_name(u)
        assert len(name) == len("checkpoint-000000.pt")
        (tmp_path / name).touch()
    assert v28_run.discover_checkpoints(tmp_path) == [0, 5, 900, 2000, 10000, 100000]


def test_cli_validation_in_process():
    a = v28_run.parser().parse_args(["--arm", "Pv28c", "--out-dir", "unused", "--source-commit", "a" * 40, "--updates", "100"])
    cad, sched = v28_run.validate(a)
    assert cad == 10 and sched[-1] == 100

    for bad in (
        ["--arm", "Pv28c", "--out-dir", "x", "--source-commit", "bad"],
        ["--arm", "Pv28m", "--out-dir", "x", "--source-commit", "a" * 40, "--updates", "6"],
        ["--arm", "Pv28m", "--out-dir", "x", "--source-commit", "a" * 40, "--cap", "10"],
        ["--arm", "Pv28c", "--out-dir", "x", "--source-commit", "a" * 40, "--synthetic"],
        ["--arm", "Pv28c", "--out-dir", "x", "--source-commit", "a" * 40, "--device", "cpu"],
    ):
        with pytest.raises(ValueError):
            v28_run.validate(v28_run.parser().parse_args(bad))


def test_weight_diff_edge_cases():
    assert core.weight_diff({"x": torch.tensor([1.0])}, {"x": torch.tensor([1.0])}) == []
    assert core.weight_diff({"x": torch.tensor([0.0])}, {"x": torch.tensor([-0.0])}) == ["x"]
    assert core.weight_diff({}, {"x": torch.tensor([1.0])}) == ["x"]
    with pytest.raises(FloatingPointError):
        core.finite({"x": torch.tensor([float("nan")])})


def test_paired_run_terminal_fastpath_and_binary_search_localization(tmp_path, monkeypatch):
    pair = tmp_path / "pair"
    common = [
        "--source-commit", "a" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--updates", "4", "--checkpoint-every", "1",
        "--episodes", "2", "--cap", "3", "--validation", "2", "--hidden", "8",
    ]
    assert v28_run.main(["--arm", "Pv28c", "--out-dir", str(pair / "Pv28c"), *common]) == 0
    assert v28_run.main(["--arm", "Pv28m", "--out-dir", str(pair / "Pv28m"), "--workers", "2", *common]) == 0

    # Verify terminal-first fast path loads ONLY terminal checkpoint (u=4) on Pv28c and Pv28m!
    loaded_files: list[str] = []
    real_load = torch.load

    def spy_load(path, *args, **kwargs):
        loaded_files.append(os.path.basename(str(path)))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    assert v28_compare.compare(pair) == 0
    monkeypatch.setattr(torch, "load", real_load)

    comp = json.loads((pair / "comparison.json").read_text())
    assert comp["result"] == "WEIGHTPASS"
    assert comp["checked_updates"] == [4]
    assert comp["binary_search"] is None
    assert loaded_files == ["checkpoint-000004.pt", "checkpoint-000004.pt"]

    # Perturb Pv28m checkpoints from u=2..4 onward and verify binary search pinpoints (1, 2)!
    for u in (2, 3, 4):
        p_ckpt = pair / "Pv28m" / v28_run.ckpt_name(u)
        data = torch.load(p_ckpt, map_location="cpu", weights_only=False)
        data["model"]["policy_head.bias"] = data["model"]["policy_head.bias"] + 1e-4
        torch.save(data, p_ckpt)

    assert v28_compare.compare(pair) == 2
    comp_fail = json.loads((pair / "comparison.json").read_text())
    assert comp_fail["result"] == "WEIGHTFAIL"
    bs = comp_fail["binary_search"]
    assert bs["last_equal_update"] == 1
    assert bs["first_diverged_update"] == 2
    assert bs["first_diverged_tensors"] == ["policy_head.bias"]


def test_exact_resume_bitwise_identical_to_uninterrupted(tmp_path):
    base = [
        "--arm", "Pv28c", "--source-commit", "b" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--episodes", "2", "--cap", "3", "--validation", "2", "--hidden", "8",
    ]
    full_dir = tmp_path / "full"
    res_dir = tmp_path / "resumed"

    assert v28_run.main([*base, "--out-dir", str(full_dir), "--updates", "4", "--checkpoint-every", "2"]) == 0

    # Run first 2 updates into res_dir, rewrite manifest target to u=4 (simulating interruption after u=2 save/eval), then --resume!
    assert v28_run.main([*base, "--out-dir", str(res_dir), "--updates", "2", "--checkpoint-every", "2"]) == 0
    manifest_p = res_dir / "manifest.json"
    m = json.loads(manifest_p.read_text())
    m["semantic"]["updates"] = 4
    m["semantic"]["checkpoint_boundaries"] = [0, 2, 4]
    m["semantic"]["validation_boundaries"] = [0, 2, 4]
    manifest_p.write_text(json.dumps(m, indent=2))
    (res_dir / "status.json").write_text(json.dumps({"status": "stopped", "completed_updates": 2}))
    (res_dir / "closed.json").unlink()

    assert v28_run.main([*base, "--out-dir", str(res_dir), "--updates", "4", "--checkpoint-every", "2", "--resume"]) == 0

    c_full = torch.load(full_dir / "checkpoint-000004.pt", map_location="cpu", weights_only=False)
    c_res = torch.load(res_dir / "checkpoint-000004.pt", map_location="cpu", weights_only=False)
    assert core.weight_diff(c_full["model"], c_res["model"]) == []
    assert torch.equal(c_full["training_rng"], c_res["training_rng"])
    lines = (res_dir / "updates.jsonl").read_text().strip().splitlines()
    assert [json.loads(x)["update"] for x in lines] == [1, 2, 3, 4]


def test_cooperative_sigterm_stops_and_reaps_pool(tmp_path, monkeypatch):
    out = tmp_path / "stop_m"
    orig_collect = core.EpisodePool.collect

    def trigger_stop(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return orig_collect(self, *args, **kwargs)

    monkeypatch.setattr(core.EpisodePool, "collect", trigger_stop)
    rc = v28_run.cli([
        "--arm", "Pv28m", "--out-dir", str(out), "--source-commit", "c" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--updates", "1", "--episodes", "2", "--cap", "4", "--validation", "2", "--hidden", "8", "--workers", "2",
    ])
    assert rc == 130
    st = json.loads((out / "status.json").read_text())
    assert st["status"] == "stopped"
    assert not mp.active_children()


def test_supervise_subprocess_smoke(tmp_path):
    root = tmp_path / "sup_pair"
    base = [
        "--out-dir", str(root), "--source-commit", "d" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--updates", "1", "--episodes", "1", "--cap", "2", "--validation", "1", "--hidden", "8", "--workers", "1",
    ]
    rc = v28_sup.cli(base)
    assert rc == 0
    ps = json.loads((root / "pair-status.json").read_text())
    assert ps["status"] == "complete_equal"
    assert ps["exitcodes"] == {"Pv28c": 0, "Pv28m": 0, "compare": 0}

    # Resuming an already-complete pair must fail closed
    assert v28_sup.cli([*base, "--resume"]) != 0

    # If Pv28c finished and Pv28m was interrupted before completion, --resume with mismatched --source-commit must fail closed!
    (root / "pair-status.json").write_text(json.dumps({"status": "stopped", "phase": "Pv28m"}))
    (root / "Pv28m" / "status.json").write_text(json.dumps({"status": "stopped", "completed_updates": 0}))
    mismatched = [x if x != "d" * 40 else "9" * 40 for x in base]
    assert v28_sup.cli([*mismatched, "--resume"]) != 0


def test_binary_search_u0_divergence_and_terminal_only_divergence(tmp_path):
    pair = tmp_path / "bs_edges"
    common = [
        "--source-commit", "e" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--updates", "3", "--checkpoint-every", "1",
        "--episodes", "1", "--cap", "2", "--validation", "1", "--hidden", "8",
    ]
    assert v28_run.main(["--arm", "Pv28c", "--out-dir", str(pair / "Pv28c"), *common]) == 0
    assert v28_run.main(["--arm", "Pv28m", "--out-dir", str(pair / "Pv28m"), "--workers", "1", *common]) == 0

    # Case 1: Only terminal u=3 diverges -> last_equal=2, first_diverged=3
    p3 = pair / "Pv28m" / "checkpoint-000003.pt"
    d3 = torch.load(p3, map_location="cpu", weights_only=False)
    d3["model"]["value_head.bias"] = d3["model"]["value_head.bias"] + 1.0
    torch.save(d3, p3)
    assert v28_compare.compare(pair) == 2
    bs1 = json.loads((pair / "comparison.json").read_text())["binary_search"]
    assert (bs1["last_equal_update"], bs1["first_diverged_update"]) == (2, 3)

    # Case 2: Even u=0 diverges -> last_equal=None, first_diverged=0
    p0 = pair / "Pv28m" / "checkpoint-000000.pt"
    d0 = torch.load(p0, map_location="cpu", weights_only=False)
    d0["model"]["policy_head.bias"] = d0["model"]["policy_head.bias"] + 1.0
    torch.save(d0, p0)
    assert v28_compare.compare(pair) == 2
    bs2 = json.loads((pair / "comparison.json").read_text())["binary_search"]
    assert (bs2["last_equal_update"], bs2["first_diverged_update"]) == (None, 0)


def test_resume_rejects_completed_run_and_manifest_mismatch(tmp_path):
    out = tmp_path / "done_run"
    args = [
        "--arm", "Pv28c", "--out-dir", str(out), "--source-commit", "f" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--updates", "1", "--episodes", "1", "--cap", "2", "--validation", "1", "--hidden", "8",
    ]
    assert v28_run.main(args) == 0
    with pytest.raises(ValueError, match="already complete"):
        v28_run.main([*args, "--resume"])
    with pytest.raises(ValueError, match="manifest mismatch"):
        v28_run.main([*args, "--seed", "19", "--resume"])


def test_resume_recovers_missing_boundary_evaluation(tmp_path):
    out = tmp_path / "missing_eval"
    base = [
        "--arm", "Pv28c", "--out-dir", str(out), "--source-commit", "1" * 40,
        "--nonformal", "--synthetic", "--device", "cpu",
        "--episodes", "1", "--cap", "2", "--validation", "1", "--hidden", "8",
    ]
    assert v28_run.main([*base, "--updates", "1", "--checkpoint-every", "1"]) == 0
    # Simulate interruption right after checkpoint-000001.pt write before evaluation-000001.json write
    (out / "evaluation-000001.json").unlink()
    m = json.loads((out / "manifest.json").read_text())
    m["semantic"]["updates"] = 2
    m["semantic"]["checkpoint_boundaries"] = [0, 1, 2]
    m["semantic"]["validation_boundaries"] = [0, 1, 2]
    (out / "manifest.json").write_text(json.dumps(m))
    (out / "status.json").write_text(json.dumps({"status": "stopped", "completed_updates": 1}))
    (out / "closed.json").unlink()

    assert v28_run.main([*base, "--updates", "2", "--checkpoint-every", "1", "--resume"]) == 0
    assert (out / "evaluation-000001.json").exists()
    assert (out / "evaluation-000002.json").exists()


def test_sorted_affinity_cpus_and_worker_resolution():
    ranked = core.sorted_affinity_cpus()
    assert len(ranked) >= 1
    assert v28_run.resolve_workers("Pv28c", 12, 16, 8) == 0
    assert v28_run.resolve_workers("Pv28m", 12, 16, 6) == 6
    assert v28_run.resolve_workers("Pv28m", 12, 16, 0) >= 1


def test_source_digest_excludes_tests_and_docs():
    d = core.source_digest()
    assert len(d) == 64


def test_disk_estimate_scales_with_scheduled_checkpoints():
    _, s5 = core.schedule_for(5)
    _, s1000 = core.schedule_for(1000)
    assert len(s5) == 6 and len(s1000) == 11
    # Sparse cadence keeps u1000 estimate bounded under 512 MiB instead of 10+ GiB!
    assert core.estimate_bytes(472, len(s1000)) < 512 * 1024**2
