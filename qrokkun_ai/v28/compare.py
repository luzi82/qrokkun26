"""Offline terminal-first raw-byte checkpoint comparison with binary-search localization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qrokkun_ai.v26.cli import boundary
from .core import weight_diff
from .run import CKPT_SCHEMA, ckpt_name, discover_checkpoints, json_write

ARMS = ("Pv28c", "Pv28m")
RATIO_KEYS = (
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
    return model


def compare(root: Path) -> int:
    root = Path(root)
    manifests = [json.loads((root / arm / "manifest.json").read_text()) for arm in ARMS]
    if manifests[0].get("arm") != "Pv28c" or manifests[1].get("arm") != "Pv28m":
        raise ValueError("unexpected arm identities in manifests")
    if manifests[0]["semantic"] != manifests[1]["semantic"]:
        raise ValueError("source/input/runtime semantic mismatch between Pv28c and Pv28m")

    sem = manifests[0]["semantic"]
    updates = int(sem["updates"])
    expected_schedule = sorted(int(x) for x in sem["checkpoint_boundaries"])
    if not expected_schedule or expected_schedule[0] != 0 or expected_schedule[-1] != updates:
        raise ValueError(f"invalid checkpoint schedule in manifest: {expected_schedule}")

    closures = {}
    for arm in ARMS:
        arm_dir = root / arm
        status = json.loads((arm_dir / "status.json").read_text())
        if status.get("status") != "complete" or int(status.get("completed_updates", -1)) != updates:
            raise RuntimeError(f"incomplete arm {arm}: {status}")
        found = discover_checkpoints(arm_dir)
        missing = sorted(set(expected_schedule) - set(found))
        if missing:
            raise FileNotFoundError(f"{arm} missing scheduled checkpoints: {missing}")
        closures[arm] = json.loads((arm_dir / "closed.json").read_text())

    # Step 1: Terminal-first check (avoids loading intermediate checkpoints when equal).
    c_term = _load_model_state(root / "Pv28c", updates)
    m_term = _load_model_state(root / "Pv28m", updates)
    term_diff = weight_diff(c_term, m_term)
    checked_updates = [updates]
    binary_search_report = None

    if not term_diff:
        result = "WEIGHTPASS"
        code = 0
    else:
        result = "WEIGHTFAIL"
        code = 2
        # Step 2: Binary search over integer-sorted schedule to locate earliest diverged milestone.
        diffs_by_u: dict[int, list[str]] = {updates: term_diff}

        def diff_at(u: int) -> list[str]:
            if u not in diffs_by_u:
                checked_updates.append(u)
                diffs_by_u[u] = weight_diff(
                    _load_model_state(root / "Pv28c", u),
                    _load_model_state(root / "Pv28m", u),
                )
            return diffs_by_u[u]

        d0 = diff_at(expected_schedule[0])
        if d0:
            last_equal = None
            first_div = expected_schedule[0]
            first_tensors = d0
        else:
            lo = 0
            hi = len(expected_schedule) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                u_mid = expected_schedule[mid]
                if diff_at(u_mid):
                    hi = mid
                else:
                    lo = mid
            last_equal = expected_schedule[lo]
            first_div = expected_schedule[hi]
            first_tensors = diff_at(first_div)

        binary_search_report = {
            "schedule": expected_schedule,
            "checked_updates": checked_updates,
            "last_equal_update": last_equal,
            "first_diverged_update": first_div,
            "first_diverged_tensors": first_tensors,
        }

    timings = {arm: closures[arm]["timings"] for arm in ARMS}
    ratios: dict[str, float | None] = {}
    for k in RATIO_KEYS:
        tc = float(timings["Pv28c"].get(k, 0.0))
        tm = float(timings["Pv28m"].get(k, 0.0))
        ratios[k] = (tc / tm) if tm > 0.0 else None

    report = {
        "result": result,
        "exitcode": code,
        "updates": updates,
        "schedule": expected_schedule,
        "terminal_update": updates,
        "terminal_differences": term_diff,
        "checked_updates": checked_updates,
        "binary_search": binary_search_report,
        "workers": {"Pv28c": manifests[0]["workers"], "Pv28m": manifests[1]["workers"]},
        "timings": timings,
        "serial_over_multi_wall_ratio": ratios,
        "speed_is_separate": True,
    }
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
