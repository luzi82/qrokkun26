"""Offline linear raw-byte comparison supporting Pv31c/Pv31m pairs and Pv30c reference comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from qrokkun_ai.v26.cli import boundary
from .core import finite, weight_diff
from .run import CKPT_SCHEMA, SCHEMA, ckpt_name, discover_checkpoints, json_write

ARMS = ("Pv31c", "Pv31m")
REF_MANIFEST_TO_CKPT = {
    SCHEMA: CKPT_SCHEMA,
    "pv30_manifest_v1": "pv30_checkpoint_v1",
}
ACCEPTED_CKPT_SCHEMAS = frozenset(REF_MANIFEST_TO_CKPT.values())

SHARED_SEMANTIC_KEYS = (
    "updates",
    "checkpoint_cadence",
    "checkpoint_boundaries",
    "validation_cadence",
    "validation_boundaries",
    "episodes",
    "cap",
    "seed",
    "validation",
    "validation_seeds",
    "hidden",
    "synthetic",
    "nonformal",
    "hyperparameters",
    "initial_player_state_sha",
    "spawner_state_sha",
    "inputs",
    "fresh_adam",
    "P_offset",
    "S_train_offset",
    "S_validation_offset",
    "permutation_seed",
)

LEARNER_BACKEND_KEYS = frozenset(
    {
        "ppo_backend",
        "cuda_streams_enabled",
        "host_dispatch_workers",
        "kernel_overlap_measured",
        "hoisted_features",
        "deferred_host_sync",
    }
)

RATIO_KEYS = (
    "pool_init_s",
    "pool_shutdown_s",
    "weight_sync_s",
    "pure_rollout_s",
    "rollout_s",
    "collector_s",
    "batch_teacher_gae_s",
    "ppo_s",
    "validation_s",
    "checkpoint_s",
    "total_s",
)


def _load_model_state(arm_dir: Path, u: int) -> dict[str, torch.Tensor]:
    path = arm_dir / ckpt_name(u)
    if not path.exists():
        raise FileNotFoundError(f"missing expected checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") not in ACCEPTED_CKPT_SCHEMAS or int(ckpt.get("update", -1)) != int(u):
        raise ValueError(f"checkpoint metadata mismatch in {path}")
    manifest_path = arm_dir / "manifest.json"
    if manifest_path.exists():
        m_schema = json.loads(manifest_path.read_text()).get("semantic", {}).get("schema")
        if m_schema in REF_MANIFEST_TO_CKPT and ckpt.get("schema") != REF_MANIFEST_TO_CKPT[m_schema]:
            raise ValueError(f"checkpoint schema {ckpt.get('schema')!r} does not match manifest {m_schema!r}")
    model = ckpt.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError(f"missing model dictionary in {path}")
    from qrokkun_ai.v21.agents.player_v21 import PlayerV21

    hidden = model.get("policy_body.0.weight")
    if hidden is None or hidden.ndim != 2:
        raise ValueError("missing actor architecture")
    with torch.random.fork_rng(devices=[]):
        expected = PlayerV21(hidden=hidden.shape[0]).state_dict()
    if set(model) != set(expected) or any(
        model[k].shape != v.shape or model[k].dtype != v.dtype for k, v in expected.items()
    ):
        raise ValueError("incomplete actor/critic state")
    finite(model)
    return model


def _verify_reference_invariants(ref_sem: dict[str, Any], cand_sem: dict[str, Any]) -> None:
    ref_schema = ref_sem.get("schema")
    cand_schema = cand_sem.get("schema")
    if ref_schema not in REF_MANIFEST_TO_CKPT:
        raise ValueError(f"unsupported reference manifest schema: {ref_schema!r}")
    if cand_schema != SCHEMA:
        raise ValueError(f"unexpected candidate manifest schema: {cand_schema!r}")
    for key in SHARED_SEMANTIC_KEYS:
        if ref_sem.get(key) != cand_sem.get(key):
            raise ValueError(f"semantic invariant mismatch on {key}: {ref_sem.get(key)!r} != {cand_sem.get(key)!r}")
    ref_rt = ref_sem.get("runtime", {})
    cand_rt = cand_sem.get("runtime", {})
    if ref_rt.get("rollout") != cand_rt.get("rollout"):
        raise ValueError("rollout runtime mismatch between reference and candidate")
    ref_learner = {k: v for k, v in ref_rt.get("learner", {}).items() if k not in LEARNER_BACKEND_KEYS}
    cand_learner = {k: v for k, v in cand_rt.get("learner", {}).items() if k not in LEARNER_BACKEND_KEYS}
    if ref_learner != cand_learner:
        raise ValueError(f"learner runtime mismatch: {ref_learner!r} != {cand_learner!r}")


def compare(
    root: Path,
    stop_check: Callable[[], bool] = lambda: False,
    *,
    reference_dir: Path | None = None,
    candidate_dir: Path | None = None,
) -> int:
    def check_stop():
        if stop_check():
            raise InterruptedError("owner stop")

    check_stop()
    if (reference_dir is None) != (candidate_dir is None):
        raise ValueError("--reference-dir and --candidate-dir must be supplied together")

    root = Path(root)
    if reference_dir is None:
        mode = "paired"
        arm_dirs = [root / ARMS[0], root / ARMS[1]]
        manifests = [json.loads((d / "manifest.json").read_text()) for d in arm_dirs]
        if manifests[0].get("arm") != "Pv31c" or manifests[1].get("arm") != "Pv31m":
            raise ValueError("unexpected arm identities in manifests")
        if manifests[0]["semantic"] != manifests[1]["semantic"]:
            raise ValueError("source/input/runtime semantic mismatch between Pv31c and Pv31m")
        ref_arm, cand_arm = ARMS
    else:
        assert candidate_dir is not None
        mode = "reference"
        arm_dirs = [Path(reference_dir), Path(candidate_dir)]
        manifests = [json.loads((d / "manifest.json").read_text()) for d in arm_dirs]
        ref_arm = str(manifests[0].get("arm", ""))
        cand_arm = str(manifests[1].get("arm", ""))
        if ref_arm not in ("Pv30c", "Pv31c") or cand_arm != "Pv31m":
            raise ValueError(f"unexpected reference/candidate arms: {ref_arm!r} vs {cand_arm!r}")
        _verify_reference_invariants(manifests[0]["semantic"], manifests[1]["semantic"])
        root.mkdir(parents=True, exist_ok=True)

    sem = manifests[1]["semantic"]
    updates = int(sem["updates"])
    expected_schedule = sorted(int(x) for x in sem["checkpoint_boundaries"])
    if not expected_schedule or expected_schedule[0] != 0 or expected_schedule[-1] != updates:
        raise ValueError(f"invalid checkpoint schedule in manifest: {expected_schedule}")

    arm_labels = (ref_arm, cand_arm)
    closures: dict[str, Any] = {}
    for label, arm_dir in zip(arm_labels, arm_dirs):
        check_stop()
        status = json.loads((arm_dir / "status.json").read_text())
        if status.get("status") != "complete" or int(status.get("completed_updates", -1)) != updates:
            raise RuntimeError(f"incomplete arm {label}: {status}")
        found = discover_checkpoints(arm_dir)
        check_stop()
        missing = sorted(set(expected_schedule) - set(found))
        if missing:
            raise FileNotFoundError(f"{label} missing scheduled checkpoints: {missing}")
        closures[label] = json.loads((arm_dir / "closed.json").read_text())

    differences = {}
    for u in expected_schedule:
        check_stop()
        states = []
        for arm_dir in arm_dirs:
            check_stop()
            states.append(_load_model_state(arm_dir, u))
            check_stop()
        for state in states:
            finite(state)
        differences[u] = weight_diff(*states)
    term_diff = differences[updates]
    checked_updates = expected_schedule
    result, code = ("WEIGHTFAIL", 2) if term_diff else ("WEIGHTPASS", 0)

    timings = {label: closures[label]["timings"] for label in arm_labels}
    ratios: dict[str, float | None] = {}
    for k in RATIO_KEYS:
        tc = float(timings[ref_arm].get(k, 0.0))
        tm = float(timings[cand_arm].get(k, 0.0))
        ratios[k] = (tc / tm) if tm > 0.0 else None

    report: dict[str, Any] = {
        "mode": mode,
        "reference_arm": ref_arm,
        "candidate_arm": cand_arm,
        "result": result,
        "exitcode": code,
        "updates": updates,
        "schedule": expected_schedule,
        "terminal_update": updates,
        "terminal_differences": term_diff,
        "checked_updates": checked_updates,
        "differences_by_update": differences,
        "first_diverged_update": next((u for u in expected_schedule if differences[u]), None),
        "primary_weight_verdict": "terminal named actor+critic raw bits only",
        "statistical_ci": False,
        "workers": {ref_arm: manifests[0]["workers"], cand_arm: manifests[1]["workers"]},
        "provenance": {
            ref_arm: {
                "schema": manifests[0]["semantic"].get("schema"),
                "source_commit": manifests[0]["semantic"].get("source_commit"),
                "source_tree_sha": manifests[0]["semantic"].get("source_tree_sha"),
                "ppo_backend": manifests[0]["semantic"].get("runtime", {}).get("learner", {}).get("ppo_backend"),
            },
            cand_arm: {
                "schema": manifests[1]["semantic"].get("schema"),
                "source_commit": manifests[1]["semantic"].get("source_commit"),
                "source_tree_sha": manifests[1]["semantic"].get("source_tree_sha"),
                "ppo_backend": manifests[1]["semantic"].get("runtime", {}).get("learner", {}).get("ppo_backend"),
            },
        },
        "timings": timings,
        "serial_over_multi_wall_ratio": ratios,
        "speed_is_separate": True,
    }
    check_stop()
    json_write(root / "comparison.json", report)
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--reference-dir", default=None)
    p.add_argument("--candidate-dir", default=None)
    a = p.parse_args(argv)
    return compare(
        Path(a.directory),
        reference_dir=Path(a.reference_dir) if a.reference_dir else None,
        candidate_dir=Path(a.candidate_dir) if a.candidate_dir else None,
    )


def cli(argv: list[str] | None = None) -> int:
    return boundary(main, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
