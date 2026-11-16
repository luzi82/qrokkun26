"""Pv34m random initialization matched control long run; no resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import time
from typing import Any

import numpy as np
import torch

from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v24.core import Spawner, file_hash
from qrokkun_ai.v25.core import P_PATH, S_PATH, VALIDATION
from qrokkun_ai.v26.cli import boundary
from qrokkun_ai.v26.run import HYPER, S_STATE
from qrokkun_ai.v31 import core
from .adapter import EpisodePool, runtime
from .initialization import random_player, locked_spawner, initialization_metadata
from .reference import load_reference
from .protocol import schedule, better

SCHEMA = "pv34_manifest_v1"
CKPT_SCHEMA = "pv34_checkpoint_v1"
CKPT_RE = re.compile(r"^checkpoint-(\d+)\.pt$")

TIMING_KEYS = (
    "pool_init_s",
    "weight_sync_s",
    "rollout_s",
    "pure_rollout_s",
    "batch_teacher_gae_s",
    "ppo_s",
    "validation_s",
    "checkpoint_s",
    "pool_shutdown_s",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def ckpt_name(u: int) -> str:
    return f"checkpoint-{int(u):06d}.pt"


def eval_name(u: int) -> str:
    return f"evaluation-{int(u):06d}.json"


def json_write(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def discover_checkpoints(out_dir: Path) -> list[int]:
    found: list[int] = []
    for p in out_dir.iterdir():
        m = CKPT_RE.match(p.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(set(found))


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--player-checkpoint", default=P_PATH)
    p.add_argument("--spawner-checkpoint", default=S_PATH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--cap", type=int, default=7200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--validation", type=int, default=16)
    p.add_argument("--hidden", type=int, default=472)
    p.add_argument("--workers", type=int, default=10, help="Pv31m worker count")
    p.add_argument("--pin-cores", choices=["auto", "on", "off"], default="on")
    p.add_argument("--validation-every", type=int, default=10)
    p.add_argument("--nonformal", action="store_true")
    p.add_argument("--synthetic", action="store_true")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "final"])
    p.add_argument("--control-reference-dir", default="/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261116-0728-pv33m-random-init-u1000")
    p.set_defaults(arm="Pv34m")
    add_shared_args(p)
    return p


def validate(a: argparse.Namespace) -> tuple[int, list[int]]:
    if a.command == "final" and a.nonformal:
        raise ValueError("nonformal has no final")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", a.source_commit):
        raise ValueError(f"--source-commit must be a 40-char hex SHA, got {a.source_commit!r}")
    if a.updates <= 0 or (not a.nonformal and a.updates != 1000):
        raise ValueError("formal requires 1000 updates")
    cadence, boundaries = 10, schedule(a.updates)
    for k, formal_val, hi in (
        ("episodes", 20, None),
        ("cap", 7200, 7200),
        ("seed", 17, None),
        ("validation", 16, 16),
        ("hidden", 472, None),
    ):
        n = getattr(a, k)
        if n <= 0 or (hi is not None and n > hi) or (not a.nonformal and n != formal_val):
            raise ValueError(f"invalid --{k}={n} (formal requires {formal_val})")
    if a.workers <= 0:
        raise ValueError(f"--workers must be >= 0, got {a.workers}")
    if a.synthetic and not a.nonformal:
        raise ValueError("--synthetic requires --nonformal")
    if not a.nonformal and a.device != "cuda":
        raise ValueError(f"formal run requires --device cuda, got {a.device!r}")
    if not a.synthetic and a.hidden != 472:
        raise ValueError("locked Pv21 architecture requires --hidden 472")
    validation_schedule(a)
    if not a.nonformal and (a.workers != 10 or a.pin_cores != "on"):
        raise ValueError("formal requires fresh launch, workers=10 and pin-cores=on")
    return cadence, boundaries


def validation_schedule(a: argparse.Namespace) -> tuple[int, list[int]]:
    cadence = a.validation_every
    if cadence <= 0 or (not a.nonformal and cadence != 10):
        raise ValueError("formal validation cadence must be 10")
    return cadence, sorted({0, *range(cadence, a.updates + 1, cadence), a.updates})


def _train_main(argv: list[str] | None = None, expected_semantic: dict[str, object] | None = None) -> int:
    a = parser().parse_args(argv)
    cadence, boundaries = validate(a)
    boundary_set = set(boundaries)
    val_cadence, val_boundaries = validation_schedule(a)
    val_set = set(val_boundaries)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)

    started_wall = time.perf_counter()
    stopped = [False]
    handlers: dict[int, Any] = {}
    pool: EpisodePool | None = None
    orig_affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    init_cpu_budget = core.cpu_budget()
    completed = 0
    last_ckpt = -1
    workers_n = 0
    best_update, best_total, latest_validated = 0, -1, -1
    errors = []
    original_threads = torch.get_num_threads()

    def write_status(state: str, phase: str, cur_u: int, error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "status": state,
            "arm": a.arm,
            "phase": phase,
            "completed_updates": int(completed),
            "current_update": int(cur_u),
            "target_updates": int(a.updates),
            "last_checkpoint": int(last_ckpt),
            "best_update": best_update,
            "latest_validated_update": latest_validated,
            "owner_stop": bool(stopped[0]),
            "errors": list(errors),
            "workers": int(workers_n),
            "updated_at": now_iso(),
        }
        if error is not None:
            payload["error"] = error
        json_write(out / "status.json", payload)

    def on_stop(sig, frame):
        stopped[0] = True
        if pool is not None:
            pool.stop_event.set()

    def is_stopped() -> bool:
        return bool(stopped[0])

    def check_stop() -> None:
        if stopped[0]:
            raise InterruptedError("owner stop")

    def check_disk(remaining_ckpts: int) -> None:
        check_stop()
        needed = core.estimate_bytes(a.hidden, max(1, remaining_ckpts))
        free = shutil.disk_usage(out).free
        if free < needed:
            raise OSError(f"checkpoint free space guard at {out}: free={free} < needed={needed}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, on_stop)

        check_disk(len(boundaries))
        check_stop()
        rt_info: dict[str, Any] = core.configure_runtime(a.device)
        rt_info["learner"].update(
            python=platform.python_version(),
            platform=platform.platform(),
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0) if (a.device.startswith("cuda") and torch.cuda.is_available()) else None,
        )

        # CPU initialization is isolated, independent of all earlier constructors.
        p = random_player(a.hidden).to(a.device)
        s_cpu = Spawner().requires_grad_(False).eval() if a.synthetic else locked_spawner(a.spawner_checkpoint)
        warm = None if a.nonformal else load_reference(Path(a.control_reference_dir), is_stopped)

        check_stop()
        p_cpu = PlayerV21(hidden=a.hidden).eval()
        p_cpu.load_state_dict({k: v.detach().cpu() for k, v in p.state_dict().items()})
        p_cpu.requires_grad_(False)

        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if a.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)

        frozen_s_sha = state_dict_sha256(s_cpu.state_dict())
        initial_p_sha = state_dict_sha256(p.state_dict())
        if not a.synthetic and initial_p_sha != '49a7c9426a1054fc000caf359fc1a0a3c90db49adfe6a53c3275350a9b5a7942':
            raise ValueError('Pv33 matched u0 raw state mismatch')
        oa, oc = core.optimizers(p)
        g = torch.Generator(device=a.device).manual_seed(core.PERM_BASE + a.seed)

        semantic: dict[str, object] = {
            "schema": SCHEMA,
            "observation_filter": "v34 conservative impossible collision; all live candidates before stable distance top64",
            "initialization": initialization_metadata(initial_p_sha),
            "control_reference": warm["identity"] if warm else None,
            "source_commit": a.source_commit,
            "source_tree_sha": core.source_digest(),
            "player_mode_training": "stochastic",
            "player_mode_evaluation": "argmax",
            "spawner_mode": "stochastic",
            "updates": int(a.updates),
            "checkpoint_cadence": int(cadence),
            "checkpoint_boundaries": boundaries,
            "validation_cadence": val_cadence,
            "validation_boundaries": val_boundaries,
            "episodes": int(a.episodes),
            "cap": int(a.cap),
            "seed": int(a.seed),
            "validation": int(a.validation),
            "validation_seeds": list(VALIDATION[: a.validation]),
            "hidden": int(a.hidden),
            "synthetic": bool(a.synthetic),
            "nonformal": bool(a.nonformal),
            "runtime": rt_info,
            "hyperparameters": HYPER,
            "initial_player_state_sha": initial_p_sha,
            "spawner_state_sha": frozen_s_sha,
            "inputs": {}
            if a.synthetic
            else {
                
                "spawner": file_hash(a.spawner_checkpoint),
            },
            "fresh_adam": True,
            "lineage_update": 0,
            "final_seeds": list(range(800000, 800064)),
            "final_window": "previously used Pv25 benchmark; not fresh heldout",
            "reference_sha": __import__("qrokkun_ai.v25.core", fromlist=["REFERENCE_SHA"]).REFERENCE_SHA,
            "P_offset": core.P_OFFSET,
            "S_train_offset": core.S_TRAIN_OFFSET,
            "S_validation_offset": core.S_VAL_OFFSET,
            "permutation_seed": core.PERM_BASE + int(a.seed),
        }

        if expected_semantic is not None and semantic != expected_semantic:
            raise ValueError("source/input/runtime semantic mismatch between paired arms")

        workers_n = a.workers
        totals: dict[str, float | int] = {k: 0.0 for k in TIMING_KEYS}
        totals["frames"] = 0
        prior_elapsed_s = 0.0
        start_update = 1

        check_stop()
        if True:
            t0 = time.perf_counter()
            pool = EpisodePool(
                p_template=p_cpu,
                s_cpu=s_cpu,
                max_slots=max(a.episodes, a.validation),
                max_cap=a.cap,
                workers=workers_n,
                pin_cores=a.pin_cores,
                stop_check=is_stopped,
            )
            totals["pool_init_s"] = float(totals["pool_init_s"]) + (time.perf_counter() - t0)

        learner_cpu = core.select_learner_cpu(a.pin_cores, strict=not a.nonformal)
        if learner_cpu is not None and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {int(learner_cpu)})
            except OSError as exc:
                if a.pin_cores == "on":
                    raise RuntimeError("learner affinity failed") from exc
                learner_cpu = None

        def sync_policy_version(ver: int) -> None:
            t0 = time.perf_counter()
            cpu_sd = {k: v.detach().cpu().contiguous() for k, v in p.state_dict().items()}
            if pool is not None:
                pool.sync_weights(cpu_sd, policy_version=ver, stop_check=is_stopped)
            else:
                p_cpu.load_state_dict(cpu_sd, strict=True)
                p_cpu.eval()
            totals["weight_sync_s"] = float(totals["weight_sync_s"]) + (time.perf_counter() - t0)

        def save_checkpoint(u: int) -> None:
            nonlocal last_ckpt
            remaining = sum(1 for b in boundaries if b >= u)
            check_disk(remaining)
            t0 = time.perf_counter()
            sync_cuda(a.device)
            core.finite(p.state_dict())
            snap_timings = dict(totals)
            snap_timings["elapsed_wall_s"] = prior_elapsed_s + (time.perf_counter() - started_wall)
            payload = {
                "schema": CKPT_SCHEMA,
                "initialization": initialization_metadata(initial_p_sha),
                "arm": a.arm,
                "update": int(u),
                "model": {k: v.detach().cpu() for k, v in p.state_dict().items()},
                "actor": oa.state_dict(),
                "critic": oc.state_dict(),
                "training_rng": g.get_state().cpu(),
                "rng_metadata": {"python": random.getstate(), "numpy": np.random.get_state(),
                                 "torch_cpu": torch.get_rng_state(),
                                 "torch_cuda": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else []},
                "completed_updates": int(u),
                "timings": snap_timings,
            }
            tmp = out / "checkpoint.tmp"
            torch.save(payload, tmp)
            tmp.replace(out / ckpt_name(u))
            totals["checkpoint_s"] = float(totals["checkpoint_s"]) + (time.perf_counter() - t0)
            last_ckpt = int(u)

        def run_validation(u: int) -> None:
            nonlocal best_update, best_total, latest_validated
            write_status("running", "validation", u)
            t0 = time.perf_counter()
            val_seeds = list(VALIDATION[: a.validation])
            if pool is not None:
                trajs = pool.collect(val_seeds, a.cap, validation=True, stop_check=is_stopped)
            else:
                trajs, _ = core.collect_serial(p_cpu, s_cpu, val_seeds, a.cap, validation=True, stop_check=is_stopped)
            dt = time.perf_counter() - t0
            totals["validation_s"] = float(totals["validation_s"]) + dt
            elapseds = [ep.elapsed for ep in trajs]
            records = [{**ep.summary(), "cap_frames": a.cap} for ep in trajs]
            total = sum(ep.frames for ep in trajs)
            if better(total, best_total):
                best_update, best_total = u, total
            latest_validated = u
            check_stop()
            for label, update in (("initial", 0), ("best", best_update), ("latest", u), *(([("terminal", u)]) if u == a.updates else [])):
                json_write(out / (label + ".json"), {"update": update, "checkpoint": ckpt_name(update)})
            json_write(
                out / eval_name(u),
                {
                    "update": int(u),
                    "mean_elapsed": float(sum(elapseds) / len(elapseds)),
                    "total_frames": int(sum(ep.frames for ep in trajs)),
                    "wall_s": dt,
                    "episodes": records,
                },
            )

        check_disk(len(boundaries))
        manifest = {
            "arm": a.arm,
            "semantic": semantic,
            "cpu_budget": init_cpu_budget,
            "workers": int(workers_n),
            "pin_cores": a.pin_cores,
            "pinned_cpus": pool.pinned_cpus if pool is not None else [],
            "learner_pinned_cpu": learner_cpu,
        }
        json_write(out / "manifest.json", manifest)
        sync_policy_version(0)
        save_checkpoint(0)
        run_validation(0)

        updates_log = out / "updates.jsonl"
        for u in range(start_update, a.updates + 1):
            check_stop()
            write_status("running", "rollout", u)
            seeds = list(range(100_000 + (u - 1) * a.episodes, 100_000 + u * a.episodes))

            t_roll = time.perf_counter()
            if pool is not None:
                trajs = pool.collect(seeds, a.cap, validation=False, stop_check=is_stopped)
            else:
                trajs, _ = core.collect_serial(p_cpu, s_cpu, seeds, a.cap, validation=False, stop_check=is_stopped)
            dt_roll = time.perf_counter() - t_roll
            pure_roll = pool.last_pure_rollout_s if pool is not None else sum(ep.rollout_s for ep in trajs)
            totals["pure_rollout_s"] += pure_roll
            totals["rollout_s"] = float(totals["rollout_s"]) + dt_roll

            write_status("running", "ppo", u)
            sync_cuda(a.device)
            t_batch = time.perf_counter()
            batch = core.assemble_batch(trajs, p, a.device)
            sync_cuda(a.device)
            dt_batch = time.perf_counter() - t_batch
            totals["batch_teacher_gae_s"] = float(totals["batch_teacher_gae_s"]) + dt_batch

            t_ppo = time.perf_counter()
            metrics = core.train(p, oa, oc, batch, generator=g, stop_check=is_stopped)
            sync_cuda(a.device)
            dt_ppo = time.perf_counter() - t_ppo
            totals["ppo_s"] = float(totals["ppo_s"]) + dt_ppo
            del batch

            if state_dict_sha256(s_cpu.state_dict()) != frozen_s_sha:
                raise RuntimeError("Spawner state mutated")

            frames = int(sum(ep.frames for ep in trajs))
            totals["frames"] = int(totals["frames"]) + frames

            sync_policy_version(u)
            if u in boundary_set:
                save_checkpoint(u)
            completed = u

            rec = {
                "update": int(u),
                "frames": frames,
                "mean_elapsed": float(sum(ep.elapsed for ep in trajs) / len(trajs)),
                "pure_rollout_s": pure_roll,
                "rollout_s": dt_roll,
                "batch_teacher_gae_s": dt_batch,
                "ppo_s": dt_ppo,
                "rollout_fps": frames / max(dt_roll, 1e-9),
                "metrics": metrics,
                "episodes": [ep.summary() for ep in trajs],
            }
            with updates_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, allow_nan=False) + "\n")

            if u in val_set:
                run_validation(u)
            write_status("running", "idle", u)

        check_stop()
        worker_stats: list[dict[str, object]] = []
        if pool is not None:
            t_shut = time.perf_counter()
            worker_stats = pool.shutdown_and_stats()
            totals["pool_shutdown_s"] = float(totals["pool_shutdown_s"]) + (time.perf_counter() - t_shut)
            pool = None

        if core.source_digest() != semantic["source_tree_sha"]:
            raise RuntimeError("production source changed during run")
        check_stop()

        total_wall_s = prior_elapsed_s + (time.perf_counter() - started_wall)
        collector_total_s = (
            float(totals["pool_init_s"])
            + float(totals["weight_sync_s"])
            + float(totals["rollout_s"])
            + float(totals["batch_teacher_gae_s"])
            + float(totals["pool_shutdown_s"])
        )
        totals_out = dict(totals)
        totals_out["frames_per_pure_rollout_s"] = int(totals["frames"]) / max(float(totals["pure_rollout_s"]), 1e-9)
        totals_out["collector_s"] = collector_total_s
        totals_out["total_s"] = total_wall_s
        totals_out["frames_per_rollout_s"] = int(totals["frames"]) / max(float(totals["rollout_s"]), 1e-9)
        totals_out["frames_per_collector_s"] = int(totals["frames"]) / max(collector_total_s, 1e-9)
        totals_out["frames_per_total_s"] = int(totals["frames"]) / max(total_wall_s, 1e-9)

        json_write(
            out / "closed.json",
            {
                "arm": a.arm,
                "control_reference": warm["identity"] if warm else None,
                "schema": SCHEMA,
                "formal": not a.nonformal,
                "source_tree_sha": semantic["source_tree_sha"],
                "manifest_sha": file_hash(out / "manifest.json"),
                "best_update": best_update,
                "latest_validated_update": latest_validated,
                "checkpoint_hashes": {ckpt_name(v): file_hash(out / ckpt_name(v)) for v in sorted({0, best_update, a.updates})},
                "completed_updates": int(completed),
                "saved_checkpoints": discover_checkpoints(out),
                "timings": totals_out,
                "workers": worker_stats,
            },
        )
        write_status("complete", "complete", completed)
        check_stop()
        return 0
    except BaseException as exc:
        errors.append(repr(exc))
        if isinstance(exc, (InterruptedError, KeyboardInterrupt)):
            if out.exists():
                write_status("stopped", "stopped", completed, error=repr(exc))
            if stopped[0]:
                raise InterruptedError("owner stop") from exc
            raise
        if out.exists():
            write_status("failed", "failed", completed, error=repr(exc))
        raise
    finally:
        for cleanup in (
            lambda: pool.close() if pool is not None else None,
            lambda: os.sched_setaffinity(0, orig_affinity) if orig_affinity is not None else None,
            lambda: torch.set_num_threads(original_threads),
        ):
            try:
                cleanup()
            except BaseException as exc:
                errors.append("cleanup: " + repr(exc))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if errors or stopped[0]:
            write_status("stopped" if stopped[0] else "failed", "cleanup", completed)
            if not __import__('sys').exc_info()[0]:
                raise RuntimeError(str(errors) if errors else "owner stop")


def train_main(argv=None, expected_semantic=None):
    with runtime():
        return _train_main(argv, expected_semantic)


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == "final":
        from .final import execute
        return execute(a)
    return train_main(argv)


def cli(argv: list[str] | None = None) -> int:
    return boundary(main, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
