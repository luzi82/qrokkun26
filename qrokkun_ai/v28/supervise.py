"""Sequential Pv28c -> Pv28m subprocess supervisor followed by offline comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v26.cli import boundary
from . import core
from .compare import ARMS, compare
from .run import add_shared_args, json_write, now_iso, validate


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_shared_args(p)
    return p


def _build_child_cmd(a: argparse.Namespace, arm: str, arm_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "qrokkun_ai.v28.run",
        "--arm",
        arm,
        "--out-dir",
        str(arm_dir),
        "--source-commit",
        a.source_commit,
        "--player-checkpoint",
        a.player_checkpoint,
        "--spawner-checkpoint",
        a.spawner_checkpoint,
        "--device",
        a.device,
        "--updates",
        str(a.updates),
        "--episodes",
        str(a.episodes),
        "--cap",
        str(a.cap),
        "--seed",
        str(a.seed),
        "--validation",
        str(a.validation),
        "--hidden",
        str(a.hidden),
        "--pin-cores",
        a.pin_cores,
    ]
    if arm == "Pv28m" and a.workers > 0:
        cmd.extend(["--workers", str(a.workers)])
    if a.checkpoint_every is not None:
        cmd.extend(["--checkpoint-every", str(a.checkpoint_every)])
    if a.resume and arm_dir.exists() and (arm_dir / "manifest.json").exists():
        cmd.append("--resume")
    if a.nonformal:
        cmd.append("--nonformal")
    if a.synthetic:
        cmd.append("--synthetic")
    return cmd


def _verify_completed_arm_identity(
    a: argparse.Namespace,
    cadence: int,
    boundaries: list[int],
    sem: dict[str, Any],
) -> None:
    expected_inputs = (
        {}
        if a.synthetic
        else {
            "player": file_hash(a.player_checkpoint),
            "spawner": file_hash(a.spawner_checkpoint),
        }
    )
    checks = {
        "source_commit": a.source_commit,
        "source_tree_sha": core.source_digest(),
        "updates": int(a.updates),
        "checkpoint_cadence": int(cadence),
        "checkpoint_boundaries": boundaries,
        "episodes": int(a.episodes),
        "cap": int(a.cap),
        "seed": int(a.seed),
        "validation": int(a.validation),
        "hidden": int(a.hidden),
        "synthetic": bool(a.synthetic),
        "nonformal": bool(a.nonformal),
        "inputs": expected_inputs,
    }
    for key, val in checks.items():
        if sem.get(key) != val:
            raise ValueError(f"completed arm manifest mismatch on --resume for {key}: {sem.get(key)!r} != {val!r}")
    if sem.get("runtime", {}).get("learner", {}).get("device") != a.device:
        raise ValueError("completed arm device mismatch on --resume")


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    cadence, boundaries = validate(a)
    root = Path(a.out_dir)
    if a.resume:
        if not root.exists():
            raise FileNotFoundError(f"--resume requires existing paired directory: {root}")
        ps_path = root / "pair-status.json"
        if ps_path.exists():
            ps = json.loads(ps_path.read_text())
            if ps.get("status") in ("complete_equal", "complete_different"):
                raise ValueError(f"paired run at {root} is already complete ({ps.get('status')})")
    else:
        root.mkdir(parents=True, exist_ok=False)

    stopped = [False]
    active_proc: list[subprocess.Popen | None] = [None]
    handlers: dict[int, Any] = {}

    def on_stop(sig, frame):
        stopped[0] = True
        proc = active_proc[0]
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                pass

    exitcodes: dict[str, int] = {}
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, on_stop)

        expected_sem = None
        for arm in ARMS:
            if stopped[0]:
                json_write(root / "pair-status.json", {"status": "stopped", "phase": arm, "exitcodes": exitcodes, "updated_at": now_iso()})
                raise InterruptedError("owner stop")

            arm_dir = root / arm
            if a.resume and arm_dir.exists() and (arm_dir / "status.json").exists():
                st = json.loads((arm_dir / "status.json").read_text())
                if st.get("status") == "complete" and int(st.get("completed_updates", -1)) == a.updates:
                    sem = json.loads((arm_dir / "manifest.json").read_text())["semantic"]
                    _verify_completed_arm_identity(a, cadence, boundaries, sem)
                    if expected_sem is not None and sem != expected_sem:
                        raise ValueError("semantic mismatch across resumed arms")
                    expected_sem = sem
                    exitcodes[arm] = 0
                    continue

            json_write(root / "pair-status.json", {"status": "running", "phase": arm, "exitcodes": exitcodes, "updated_at": now_iso()})
            cmd = _build_child_cmd(a, arm, arm_dir)
            with (root / f"{arm}.log").open("a", encoding="utf-8") as log_fh:
                proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
                active_proc[0] = proc
                while proc.poll() is None:
                    if stopped[0]:
                        try:
                            proc.send_signal(signal.SIGTERM)
                        except OSError:
                            pass
                        try:
                            proc.wait(timeout=10.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        break
                    time.sleep(0.05)
                rc = proc.wait()
                active_proc[0] = None

            exitcodes[arm] = int(rc)
            if stopped[0] or rc == 130:
                json_write(root / "pair-status.json", {"status": "stopped", "phase": arm, "exitcodes": exitcodes, "updated_at": now_iso()})
                raise InterruptedError("owner stop")
            if rc != 0:
                json_write(root / "pair-status.json", {"status": "failed", "phase": arm, "exitcodes": exitcodes, "updated_at": now_iso()})
                return int(rc)

            sem = json.loads((arm_dir / "manifest.json").read_text())["semantic"]
            if expected_sem is not None and sem != expected_sem:
                raise ValueError("semantic mismatch between Pv28c and Pv28m manifests")
            expected_sem = sem

        if stopped[0]:
            raise InterruptedError("owner stop")

        json_write(root / "pair-status.json", {"status": "running", "phase": "compare", "exitcodes": exitcodes, "updated_at": now_iso()})
        cmp_code = compare(root)
        exitcodes["compare"] = int(cmp_code)
        json_write(
            root / "pair-status.json",
            {
                "status": "complete_equal" if cmp_code == 0 else "complete_different",
                "phase": "complete",
                "exitcodes": exitcodes,
                "updated_at": now_iso(),
            },
        )
        return int(cmp_code)
    finally:
        proc = active_proc[0]
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def cli(argv: list[str] | None = None) -> int:
    return boundary(main, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
