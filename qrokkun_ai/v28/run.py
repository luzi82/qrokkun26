"""Pv28c / Pv28m resumable episode-parallel runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from qrokkun_ai.v25.run import inputs
from qrokkun_ai.v26.cli import boundary
from qrokkun_ai.v26.run import HYPER, S_STATE
from . import core

SCHEMA = "pv28_manifest_v1"
CKPT_SCHEMA = "pv28_checkpoint_v1"
CKPT_RE = re.compile(r"^checkpoint-(\d+)\.pt$")

TIMING_KEYS = (
    "pool_init_s",
    "weight_sync_s",
    "rollout_s",
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
    p.add_argument("--updates", type=int, default=5)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--cap", type=int, default=7200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--validation", type=int, default=16)
    p.add_argument("--hidden", type=int, default=472)
    p.add_argument("--workers", type=int, default=0, help="Pv28m worker count (0 = auto)")
    p.add_argument("--pin-cores", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--nonformal", action="store_true")
    p.add_argument("--synthetic", action="store_true")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["Pv28c", "Pv28m"], required=True)
    add_shared_args(p)
    return p


def validate(a: argparse.Namespace) -> tuple[int, list[int]]:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", a.source_commit):
        raise ValueError(f"--source-commit must be a 40-char hex SHA, got {a.source_commit!r}")
    cadence, boundaries = core.schedule_for(a.updates, nonformal=a.nonformal, checkpoint_every=a.checkpoint_every)
    for k, formal_val, hi in (
        ("episodes", 12, None),
        ("cap", 7200, 7200),
        ("seed", 17, None),
        ("validation", 16, 16),
        ("hidden", 472, None),
    ):
        n = getattr(a, k)
        if n <= 0 or (hi is not None and n > hi) or (not a.nonformal and n != formal_val):
            raise ValueError(f"invalid --{k}={n} (formal requires {formal_val})")
    if a.workers < 0:
        raise ValueError(f"--workers must be >= 0, got {a.workers}")
    if a.synthetic and not a.nonformal:
        raise ValueError("--synthetic requires --nonformal")
    if not a.nonformal and a.device != "cuda":
        raise ValueError(f"formal run requires --device cuda, got {a.device!r}")
    if not a.synthetic and a.hidden != 472:
        raise ValueError("locked Pv21 architecture requires --hidden 472")
    return cadence, boundaries


def resolve_workers(arm: str, episodes: int, validation: int, requested: int) -> int:
    if arm == "Pv28c":
        return 0
    if requested > 0:
        return int(requested)
    budget = core.cpu_budget()[0]
    return max(1, min(max(int(episodes), int(validation)), budget))


def truncate_updates_log(log_path: Path, keep_up_to: int) -> None:
    if not log_path.exists():
        return
    kept: list[str] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if int(rec["update"]) <= keep_up_to:
            kept.append(json.dumps(rec, allow_nan=False))
    tmp = log_path.with_suffix(".jsonl.tmp")
    tmp.write_text(("\n".join(kept) + "\n") if kept else "")
    tmp.replace(log_path)


def main(argv: list[str] | None = None, expected_semantic: dict[str, object] | None = None) -> int:
    a = parser().parse_args(argv)
    cadence, boundaries = validate(a)
    boundary_set = set(boundaries)

    out = Path(a.out_dir)
    if a.resume:
        if not out.exists() or not (out / "manifest.json").exists():
            raise FileNotFoundError(f"--resume requires existing run directory with manifest.json: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)

    started_wall = time.perf_counter()
    stopped = [False]
    handlers: dict[int, Any] = {}
    pool: core.EpisodePool | None = None
    completed = 0
    last_ckpt = -1
    workers_n = 0

    def write_status(state: str, phase: str, cur_u: int, error: str | None = None) -> None:
        payload: dict[str, Any] = {
            "status": state,
            "arm": a.arm,
            "phase": phase,
            "completed_updates": int(completed),
            "current_update": int(cur_u),
            "target_updates": int(a.updates),
            "last_checkpoint": int(last_ckpt),
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

        rt_info: dict[str, Any] = core.configure_runtime(a.device)
        rt_info["learner"].update(
            python=platform.python_version(),
            platform=platform.platform(),
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0) if (a.device.startswith("cuda") and torch.cuda.is_available()) else None,
        )

        # Construct/load models BEFORE seeding global RNGs so model constructor draws never leak.
        if a.synthetic:
            torch.manual_seed(a.seed)
            p_cpu_init = PlayerV21(hidden=a.hidden)
            s_cpu = Spawner().requires_grad_(False).eval()
            p = PlayerV21(hidden=a.hidden).to(a.device)
            p.load_state_dict(p_cpu_init.state_dict())
        else:
            p, s_cpu, _ = inputs(a.player_checkpoint, a.spawner_checkpoint, a.device)
            if state_dict_sha256(s_cpu.state_dict()) != S_STATE:
                raise ValueError("locked Sv24 state hash mismatch")

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
        oa, oc = core.optimizers(p)
        g = torch.Generator(device=a.device).manual_seed(core.PERM_BASE + a.seed)

        semantic: dict[str, object] = {
            "schema": SCHEMA,
            "source_commit": a.source_commit,
            "source_tree_sha": core.source_digest(),
            "updates": int(a.updates),
            "checkpoint_cadence": int(cadence),
            "checkpoint_boundaries": boundaries,
            "validation_boundaries": boundaries,
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
                "player": file_hash(a.player_checkpoint),
                "spawner": file_hash(a.spawner_checkpoint),
            },
            "fresh_adam": True,
            "P_offset": core.P_OFFSET,
            "S_train_offset": core.S_TRAIN_OFFSET,
            "S_validation_offset": core.S_VAL_OFFSET,
            "permutation_seed": core.PERM_BASE + int(a.seed),
        }

        if expected_semantic is not None and semantic != expected_semantic:
            raise ValueError("source/input/runtime semantic mismatch between paired arms")

        workers_n = resolve_workers(a.arm, a.episodes, a.validation, a.workers)
        totals: dict[str, float | int] = {k: 0.0 for k in TIMING_KEYS}
        totals["frames"] = 0
        prior_elapsed_s = 0.0
        start_update = 1

        if a.arm == "Pv28m":
            t0 = time.perf_counter()
            pool = core.EpisodePool(
                p_template=p_cpu,
                s_cpu=s_cpu,
                max_slots=max(a.episodes, a.validation),
                max_cap=a.cap,
                workers=workers_n,
                pin_cores=a.pin_cores,
            )
            totals["pool_init_s"] = float(totals["pool_init_s"]) + (time.perf_counter() - t0)

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
                "arm": a.arm,
                "update": int(u),
                "model": {k: v.detach().cpu() for k, v in p.state_dict().items()},
                "actor": oa.state_dict(),
                "critic": oc.state_dict(),
                "training_rng": g.get_state().cpu(),
                "completed_updates": int(u),
                "timings": snap_timings,
            }
            tmp = out / "checkpoint.tmp"
            torch.save(payload, tmp)
            tmp.replace(out / ckpt_name(u))
            totals["checkpoint_s"] = float(totals["checkpoint_s"]) + (time.perf_counter() - t0)
            last_ckpt = int(u)

        def run_validation(u: int) -> None:
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
            json_write(
                out / eval_name(u),
                {
                    "update": int(u),
                    "mean_elapsed": float(sum(elapseds) / len(elapseds)),
                    "total_frames": int(sum(ep.frames for ep in trajs)),
                    "wall_s": dt,
                    "episodes": [ep.summary() for ep in trajs],
                },
            )

        if a.resume:
            prev_manifest = json.loads((out / "manifest.json").read_text())
            if prev_manifest.get("arm") != a.arm or prev_manifest.get("semantic") != semantic:
                raise ValueError("manifest mismatch on --resume")
            if (out / "status.json").exists():
                prev_status = json.loads((out / "status.json").read_text())
                if prev_status.get("status") == "complete" and int(prev_status.get("completed_updates", -1)) == a.updates:
                    raise ValueError(f"run at {out} is already complete at u={a.updates}")
            found_ckpts = discover_checkpoints(out)
            if not found_ckpts:
                raise FileNotFoundError(f"no saved checkpoints found to resume in {out}")
            u_res = found_ckpts[-1]
            ckpt = torch.load(out / ckpt_name(u_res), map_location="cpu", weights_only=False)
            if ckpt.get("schema") != CKPT_SCHEMA or int(ckpt.get("update", -1)) != u_res:
                raise ValueError(f"invalid checkpoint metadata in {ckpt_name(u_res)}")
            p.load_state_dict({k: v.to(a.device) for k, v in ckpt["model"].items()}, strict=True)
            core.restore_optimizer(oa, ckpt["actor"], a.device)
            core.restore_optimizer(oc, ckpt["critic"], a.device)
            g.set_state(ckpt["training_rng"].cpu())
            saved_t = ckpt.get("timings", {})
            for k in TIMING_KEYS:
                if k in saved_t:
                    totals[k] = float(saved_t[k]) + (float(totals[k]) if k == "pool_init_s" else 0.0)
            totals["frames"] = int(saved_t.get("frames", 0))
            prior_elapsed_s = float(saved_t.get("elapsed_wall_s", 0.0))
            completed = u_res
            last_ckpt = u_res
            start_update = u_res + 1
            truncate_updates_log(out / "updates.jsonl", u_res)
            sync_policy_version(u_res)
            if u_res in boundary_set and not (out / eval_name(u_res)).exists():
                run_validation(u_res)
        else:
            check_disk(len(boundaries))
            manifest = {
                "arm": a.arm,
                "semantic": semantic,
                "cpu_budget": core.cpu_budget(),
                "workers": int(workers_n),
                "pin_cores": a.pin_cores,
                "pinned_cpus": pool.pinned_cpus if pool is not None else [],
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
                "rollout_s": dt_roll,
                "batch_teacher_gae_s": dt_batch,
                "ppo_s": dt_ppo,
                "rollout_fps": frames / max(dt_roll, 1e-9),
                "metrics": metrics,
                "episodes": [ep.summary() for ep in trajs],
            }
            with updates_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, allow_nan=False) + "\n")

            if u in boundary_set:
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
        totals_out["collector_s"] = collector_total_s
        totals_out["total_s"] = total_wall_s
        totals_out["frames_per_rollout_s"] = int(totals["frames"]) / max(float(totals["rollout_s"]), 1e-9)
        totals_out["frames_per_collector_s"] = int(totals["frames"]) / max(collector_total_s, 1e-9)
        totals_out["frames_per_total_s"] = int(totals["frames"]) / max(total_wall_s, 1e-9)

        json_write(
            out / "closed.json",
            {
                "arm": a.arm,
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
        if stopped[0] or isinstance(exc, (InterruptedError, KeyboardInterrupt)):
            if out.exists():
                write_status("stopped", "stopped", completed, error=repr(exc))
            if stopped[0]:
                raise InterruptedError("owner stop") from exc
            raise
        if out.exists():
            write_status("failed", "failed", completed, error=repr(exc))
        raise
    finally:
        if pool is not None:
            pool.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def cli(argv: list[str] | None = None) -> int:
    return boundary(main, argv)


if __name__ == "__main__":
    raise SystemExit(cli())
