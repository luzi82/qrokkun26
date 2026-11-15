from __future__ import annotations

import copy
import json
import os

import pytest
import torch

from qrokkun_ai.v21.agents.player_v21 import PlayerV21
from qrokkun_ai.v24.core import Spawner
from qrokkun_ai.v30 import core as v30_core
from qrokkun_ai.v31 import compare, core, run


def test_v31_train_exact_0ulp_parity_with_v30():
    torch.manual_seed(17)
    p30 = PlayerV21(hidden=16)
    p31 = copy.deepcopy(p30)
    s = Spawner().requires_grad_(False).eval()
    eps, _ = core.collect_serial(p30, s, [100_000, 100_001], 6)
    batch30 = v30_core.assemble_batch(eps, p30, "cpu")
    batch31 = core.assemble_batch(eps, p31, "cpu")

    oa30, oc30 = v30_core.optimizers(p30)
    oa31, oc31 = core.optimizers(p31)
    g30 = torch.Generator(device="cpu").manual_seed(27_000_017)
    g31 = torch.Generator(device="cpu").manual_seed(27_000_017)

    m30 = v30_core.train(p30, oa30, oc30, batch30, generator=g30, minibatch=4)
    m31 = core.train(p31, oa31, oc31, batch31, generator=g31, minibatch=4)

    assert core.weight_diff(p30.state_dict(), p31.state_dict()) == []
    assert torch.equal(g30.get_state(), g31.get_state())
    assert m31["hoisted_features"] is True and m31["deferred_host_sync"] is True
    for k in ("policy_loss", "value_loss", "dagger_loss", "approx_kl", "expert_agreement", "explained_variance"):
        assert m30[k] == pytest.approx(m31[k], rel=0.0, abs=1e-15)


def test_v31_train_hoists_features_once_and_rejects_nonfinite(monkeypatch):
    torch.manual_seed(17)
    p = PlayerV21(hidden=8)
    s = Spawner().requires_grad_(False).eval()
    eps, _ = core.collect_serial(p, s, [100_000], 6)
    batch = core.assemble_batch(eps, p, "cpu")
    oa, oc = core.optimizers(p)

    feat_calls = []
    orig_features = p.features

    def count_features(*args, **kwargs):
        feat_calls.append(args[0].shape[0])
        return orig_features(*args, **kwargs)

    monkeypatch.setattr(p, "features", count_features)
    core.train(p, oa, oc, batch, generator=torch.Generator().manual_seed(1), minibatch=2)
    # 4 epochs x 3 minibatches = 12 steps, but features() must be called exactly ONCE on all 6 frames.
    assert feat_calls == [6]

    bad_batch = {k: v.clone() for k, v in batch.items()}
    bad_batch["returns"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        core.train(p, oa, oc, bad_batch, generator=torch.Generator().manual_seed(1), minibatch=2)


def test_learner_affinity_selection_and_restoration(tmp_path, monkeypatch):
    orig = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    target_cpu = min(orig) if orig else 0
    monkeypatch.setattr(core, "sorted_affinity_cpus", lambda: [(target_cpu, 3_900_000)])
    out = tmp_path / "aff_run"
    rc = run.main([
        "--arm", "Pv31c", "--out-dir", str(out), "--source-commit", "b" * 40,
        "--nonformal", "--synthetic", "--device", "cpu", "--pin-cores", "on",
        "--updates", "1", "--episodes", "1", "--cap", "2", "--validation", "1", "--hidden", "8",
    ])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["learner_pinned_cpu"] == target_cpu
    if orig is not None:
        assert set(os.sched_getaffinity(0)) == orig


def test_cross_version_pv30c_reference_compare_and_invariant_guard(tmp_path):
    cand_dir = tmp_path / "Pv31m"
    common = [
        "--source-commit", "d" * 40, "--nonformal", "--synthetic", "--device", "cpu",
        "--pin-cores", "off", "--updates", "1", "--episodes", "1", "--cap", "2",
        "--validation", "1", "--hidden", "8",
    ]
    assert run.main(["--arm", "Pv31m", "--workers", "1", "--out-dir", str(cand_dir), *common]) == 0

    # Construct a matching pv30_manifest_v1 / pv30_checkpoint_v1 reference arm directory from identical weights.
    ref_dir = tmp_path / "Pv30c_ref"
    ref_dir.mkdir()
    cand_manifest = json.loads((cand_dir / "manifest.json").read_text())
    ref_sem = copy.deepcopy(cand_manifest["semantic"])
    ref_sem["schema"] = "pv30_manifest_v1"
    ref_sem["source_commit"] = "5" * 40
    ref_sem["source_tree_sha"] = "e" * 64
    ref_sem["runtime"]["learner"].pop("hoisted_features", None)
    ref_sem["runtime"]["learner"].pop("deferred_host_sync", None)
    ref_sem["runtime"]["learner"].update(
        ppo_backend="cuda_actor_critic_streams",
        cuda_streams_enabled=True,
        host_dispatch_workers=2,
        kernel_overlap_measured=False,
    )
    (ref_dir / "manifest.json").write_text(json.dumps({"arm": "Pv30c", "semantic": ref_sem, "workers": 0}))
    (ref_dir / "status.json").write_text(json.dumps({"status": "complete", "completed_updates": 1}))
    (ref_dir / "closed.json").write_text((cand_dir / "closed.json").read_text())
    for u in (0, 1):
        ck = torch.load(cand_dir / run.ckpt_name(u), map_location="cpu", weights_only=False)
        ck["schema"] = "pv30_checkpoint_v1"
        ck["arm"] = "Pv30c"
        torch.save(ck, ref_dir / run.ckpt_name(u))

    out_cmp = tmp_path / "cmp_out"
    assert compare.compare(out_cmp, reference_dir=ref_dir, candidate_dir=cand_dir) == 0
    rep = json.loads((out_cmp / "comparison.json").read_text())
    assert rep["mode"] == "reference"
    assert rep["result"] == "WEIGHTPASS"
    assert rep["reference_arm"] == "Pv30c" and rep["candidate_arm"] == "Pv31m"

    # Perturb terminal weight by 1 ULP -> WEIGHTFAIL (exit 2)
    ck1 = torch.load(ref_dir / run.ckpt_name(1), map_location="cpu", weights_only=False)
    ck1["model"]["policy_head.bias"][0] = torch.nextafter(
        ck1["model"]["policy_head.bias"][0], torch.tensor(float("inf"))
    )
    torch.save(ck1, ref_dir / run.ckpt_name(1))
    assert compare.compare(out_cmp, reference_dir=ref_dir, candidate_dir=cand_dir) == 2

    # Perturb episodes invariant (e.g. 3 vs 1) -> hard error (CLI exit 3, NOT WEIGHTFAIL 2)
    bad_sem = copy.deepcopy(ref_sem)
    bad_sem["episodes"] = 3
    (ref_dir / "manifest.json").write_text(json.dumps({"arm": "Pv30c", "semantic": bad_sem, "workers": 0}))
    assert compare.cli([str(out_cmp), "--reference-dir", str(ref_dir), "--candidate-dir", str(cand_dir)]) == 3


def test_recv_one_preserves_received_worker_error_during_stop():
    pool = core.EpisodePool.__new__(core.EpisodePool)
    pool.stop_event = type("Ev", (), {"is_set": lambda self: True, "set": lambda self: None})()
    conn = type("Conn", (), {"poll": lambda self, _t: True, "recv": lambda self: ("error", "WORKER_BOOM")})()
    proc = type("Proc", (), {"pid": 42, "exitcode": 1})()
    with pytest.raises(InterruptedError, match=r"(?s)owner stop.*WORKER_BOOM"):
        pool._recv_one(conn, proc, lambda: False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA path verified on GB10")
def test_v31_cuda_hoisted_path():
    core.configure_runtime("cuda")
    torch.manual_seed(17)
    p = PlayerV21(hidden=8)
    s = Spawner().requires_grad_(False).eval()
    eps, _ = core.collect_serial(p, s, [100_000], 4)
    p.to("cuda")
    batch = core.assemble_batch(eps, p, "cuda")
    oa, oc = core.optimizers(p)
    m = core.train(p, oa, oc, batch, generator=torch.Generator(device="cuda").manual_seed(17), minibatch=2)
    assert m["ppo_backend"] == "cuda_hoisted_single_stream"
