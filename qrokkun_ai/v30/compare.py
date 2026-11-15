"""Offline linear raw-byte comparison with terminal-only verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

from qrokkun_ai.v26.cli import boundary
from .core import weight_diff
from .run import CKPT_SCHEMA, ckpt_name, discover_checkpoints, json_write

ARMS = ("Pv30c", "Pv30m")
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
    if ckpt.get("schema") != CKPT_SCHEMA or int(ckpt.get("update", -1)) != int(u):
        raise ValueError(f"checkpoint metadata mismatch in {path}")
    model = ckpt.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError(f"missing model dictionary in {path}")
    # Resolve all named actor and critic tensors; malformed/incomplete states are failures.
    from qrokkun_ai.v21.agents.player_v21 import PlayerV21
    from .core import finite
    hidden = model.get("policy_body.0.weight")
    if hidden is None or hidden.ndim != 2:
        raise ValueError("missing actor architecture")
    with torch.random.fork_rng(devices=[]):
        expected = PlayerV21(hidden=hidden.shape[0]).state_dict()
    if set(model) != set(expected) or any(model[k].shape != v.shape or model[k].dtype != v.dtype for k, v in expected.items()):
        raise ValueError("incomplete actor/critic state")
    finite(model)
    return model


def compare(root: Path, stop_check: Callable[[], bool] = lambda: False) -> int:
    def check_stop():
        if stop_check():
            raise InterruptedError("owner stop")

    check_stop()
    root = Path(root)
    manifests = [json.loads((root / arm / "manifest.json").read_text()) for arm in ARMS]
    if manifests[0].get("arm") != "Pv30c" or manifests[1].get("arm") != "Pv30m":
        raise ValueError("unexpected arm identities in manifests")
    if manifests[0]["semantic"] != manifests[1]["semantic"]:
        raise ValueError("source/input/runtime semantic mismatch between Pv30c and Pv30m")

    sem = manifests[0]["semantic"]
    updates = int(sem["updates"])
    expected_schedule = sorted(int(x) for x in sem["checkpoint_boundaries"])
    if not expected_schedule or expected_schedule[0] != 0 or expected_schedule[-1] != updates:
        raise ValueError(f"invalid checkpoint schedule in manifest: {expected_schedule}")

    closures = {}
    for arm in ARMS:
        check_stop()
        arm_dir = root / arm
        status = json.loads((arm_dir / "status.json").read_text())
        if status.get("status") != "complete" or int(status.get("completed_updates", -1)) != updates:
            raise RuntimeError(f"incomplete arm {arm}: {status}")
        found = discover_checkpoints(arm_dir)
        check_stop()
        missing = sorted(set(expected_schedule) - set(found))
        if missing:
            raise FileNotFoundError(f"{arm} missing scheduled checkpoints: {missing}")
        closures[arm] = json.loads((arm_dir / "closed.json").read_text())

    # Equality need not be monotone: inspect every checkpoint in order.
    differences = {}
    for u in expected_schedule:
        check_stop()
        states = []
        for arm in ARMS:
            check_stop()
            states.append(_load_model_state(root / arm, u))
            check_stop()
        for state in states:
            from .core import finite
            finite(state)
        differences[u] = weight_diff(*states)
    term_diff = differences[updates]
    checked_updates = expected_schedule
    result, code = ("WEIGHTFAIL", 2) if term_diff else ("WEIGHTPASS", 0)

    timings = {arm: closures[arm]["timings"] for arm in ARMS}
    ratios: dict[str, float | None] = {}
    for k in RATIO_KEYS:
        tc = float(timings["Pv30c"].get(k, 0.0))
        tm = float(timings["Pv30m"].get(k, 0.0))
        ratios[k] = (tc / tm) if tm > 0.0 else None

    report = {
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
        "workers": {"Pv30c": manifests[0]["workers"], "Pv30m": manifests[1]["workers"]},
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
    return compare(Path(p.parse_args(argv).directory))


def cli(argv: list[str] | None = None) -> int:
    return boundary(main, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
