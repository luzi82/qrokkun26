#!/usr/bin/env python3
"""Phase 3 BC-retention control for ``player_ranked_topk``.

Run from the repository root with::

    PYTHONPATH=. python -m qrokkun_ai.tools.phase3_ranked_ppo_retention \
        --init-checkpoint PATH --teacher PATH --out-dir PATH --device cuda

The experiment compares a frozen/no-update control with scripted-only PPO
starting from the exact same BC checkpoint. PPO snapshots are experimental
and are never marked production-compatible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import random
import signal
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_ai.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4, encode_obs
from qrokkun_ai.agents.player_checkpoints import (
    CheckpointError,
    current_git_commit,
    file_sha256,
    load_ranked_top_k_checkpoint,
    save_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_ai.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_ai.agents.player_v1 import PlayerV1
from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_ai.train import player_v1 as scripted_ppo

from qrokkun_ai.tools.phase2_distill_v1_to_v4 import collect_dataset, frames_to_tensors
from qrokkun_ai.tools.phase2_ranked_multiseed import dataset_identity

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HKT = ZoneInfo("Asia/Hong_Kong")
UNBOUNDED_MAX_UPDATES = 2_147_483_647
FAR_FUTURE_END_TIME = dt.datetime(2099, 12, 31, 23, 59, tzinfo=_HKT)
CURRENT_RUN_SCHEMA_VERSION = 2
RECOVERY_ARCHIVE_INTERVAL = 200
RECOVERY_ARCHIVE_DIRNAME = "recovery_archives"


def _now_hkt() -> dt.datetime:
    return dt.datetime.now(_HKT)


class RunStateError(RuntimeError):
    """A resumable run is absent, corrupt, or incompatible."""


def parse_end_time(value: str) -> dt.datetime:
    """Parse the documented ``YYYYMMDD-HHMM`` wall-clock deadline as HKT."""
    try:
        return dt.datetime.strptime(value, "%Y%m%d-%H%M").replace(tzinfo=_HKT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("end time must be YYYYMMDD-HHMM (HKT)") from exc


def resolve_stop_budget(
    max_updates: int | None, end_time: dt.datetime | None, *, default_max_updates: int,
) -> tuple[int, dt.datetime]:
    """Resolve the paired stop budget without changing the public CLI."""
    return (
        UNBOUNDED_MAX_UPDATES if max_updates is None and end_time is not None
        else default_max_updates if max_updates is None else max_updates,
        FAR_FUTURE_END_TIME if end_time is None else end_time,
    )


def effective_stop_budget(args: argparse.Namespace) -> tuple[int, dt.datetime]:
    """Return and cache the immutable budget used by a run and its resume."""
    if not hasattr(args, "effective_max_updates") or not hasattr(args, "effective_end_time"):
        effective_max, effective_end = resolve_stop_budget(
            getattr(args, "max_updates", None), getattr(args, "end_time", None),
            default_max_updates=args.updates,
        )
        args.effective_max_updates = effective_max
        args.effective_end_time = effective_end
    return args.effective_max_updates, args.effective_end_time


def resolve_resume_from_stop_budget(
    args: argparse.Namespace, selected_update: int, *, now: dt.datetime | None = None,
) -> tuple[int, dt.datetime]:
    """Treat selected archive N as resume mode and resolve this invocation's stop budget."""
    args.resume = True
    if (
        isinstance(selected_update, bool)
        or not isinstance(selected_update, int)
        or selected_update <= 0
        or selected_update % RECOVERY_ARCHIVE_INTERVAL != 0
    ):
        raise ValueError("--resume-from-update must be a positive multiple of 200")
    max_updates = getattr(args, "max_updates", None)
    end_time = getattr(args, "end_time", None)
    if max_updates is None and end_time is None:
        raise ValueError("--resume-from-update requires a stop flag (--max-updates and/or --end-time)")
    if max_updates is not None and max_updates <= selected_update:
        raise ValueError("--max-updates must be strictly greater than --resume-from-update")
    current = _now_hkt() if now is None else now
    if end_time is not None and end_time <= current:
        raise ValueError("--end-time must be strictly later than the current HKT time")
    target = UNBOUNDED_MAX_UPDATES if max_updates is None else max_updates
    deadline = FAR_FUTURE_END_TIME if end_time is None else end_time
    args.effective_max_updates = target
    args.effective_end_time = deadline
    return target, deadline


def _atomic_replace(path: Path, write: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_replace(path, lambda tmp: tmp.write_text(json.dumps(value, indent=2, sort_keys=True)))


def append_run_status(run_dir: Path, event: dict[str, Any]) -> None:
    """Append-only human/machine-readable lifecycle history."""
    row = {"at_hkt": dt.datetime.now(_HKT).isoformat(), **event}
    with (run_dir / "status.jsonl").open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def reconcile_progress_journal(path: Path, *, completed_update: int) -> set[int]:
    """Validate durable progress against recovery, allowing one pending retry.

    Progress is intentionally flushed before its recovery boundary.  A crash
    in that narrow interval leaves one durable row whose model state must be
    replayed from recovery.  The row is retained and the replay is made
    idempotent; anything other than that single-row lag fails closed.
    """
    if completed_update < 0:
        raise RunStateError("recovery completed_update is negative")
    updates: set[int] = set()
    if path.exists():
        try:
            lines = path.read_text().splitlines()
            for number, line in enumerate(lines, start=1):
                row = json.loads(line)
                update = row.get("update")
                if isinstance(update, bool) or not isinstance(update, int) or update < 1:
                    raise ValueError(f"invalid update on line {number}")
                if update in updates:
                    raise ValueError(f"duplicate update {update}")
                updates.add(update)
        except (OSError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise RunStateError("progress journal is malformed") from exc
    if updates != set(range(1, (max(updates) if updates else 0) + 1)):
        raise RunStateError("progress journal updates are not contiguous")
    journal_completed = max(updates, default=0)
    if journal_completed not in {completed_update, completed_update + 1}:
        raise RunStateError("progress journal and recovery boundary disagree")
    return updates


def append_progress_row(path: Path, row: dict[str, Any]) -> bool:
    """Durably append one update row, unless a recovery replay already has it."""
    update = row.get("update")
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise RunStateError("progress row has invalid update")
    existing = reconcile_progress_journal(path, completed_update=update - 1)
    if update in existing:
        return False
    if existing != set(range(1, update)):
        raise RunStateError("progress row is out of order")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as progress:
        progress.write(json.dumps(row) + "\n")
        progress.flush()
        os.fsync(progress.fileno())
    return True


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RunStateError("malformed recovery RNG state") from exc


def atomic_save_recovery(path: Path, state: dict[str, Any]) -> None:
    _atomic_replace(path, lambda tmp: torch.save(state, tmp))


def archived_recovery_path(run_dir: Path, update: int) -> Path:
    return run_dir / RECOVERY_ARCHIVE_DIRNAME / f"update_{update}.pt"


def save_archived_recovery_if_due(run_dir: Path, state: dict[str, Any]) -> None:
    update = state["completed_update"]
    if update > 0 and update % RECOVERY_ARCHIVE_INTERVAL == 0:
        atomic_save_recovery(archived_recovery_path(run_dir, update), state)


def load_archived_recovery(run_dir: Path, update: int, device: torch.device) -> dict[str, Any]:
    """Load a schema-v2 archived full recovery; do not restore RNG as a side effect."""
    if (
        isinstance(update, bool)
        or not isinstance(update, int)
        or update <= 0
        or update % RECOVERY_ARCHIVE_INTERVAL != 0
    ):
        raise RunStateError("--resume-from-update must be a positive multiple of 200")
    path = archived_recovery_path(run_dir, update)
    if not path.is_file():
        raise RunStateError("selected recovery archive is missing")
    try:
        state = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        if not isinstance(state, dict) or not isinstance(state.get("completed_update"), int):
            raise ValueError("missing completed_update")
    except (OSError, EOFError, RuntimeError, TypeError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        raise RunStateError("selected recovery archive is malformed") from exc
    if state["completed_update"] != update:
        raise RunStateError("selected recovery archive completed_update does not match N")
    return state


def _atomic_rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_replace(path, lambda tmp: tmp.write_text(payload))


def rewind_run_to_archived_recovery(
    run_dir: Path, update: int, *, progress_filename: str, device: torch.device,
) -> None:
    """Restore archive N as latest recovery and destroy same-run history after N."""
    state = load_archived_recovery(run_dir, update, device)
    atomic_save_recovery(run_dir / "recovery.pt", state)
    progress_path = run_dir / progress_filename
    if progress_path.exists():
        try:
            kept: list[dict[str, Any]] = []
            seen: set[int] = set()
            for number, line in enumerate(progress_path.read_text().splitlines(), start=1):
                row = json.loads(line)
                row_update = row.get("update")
                if isinstance(row_update, bool) or not isinstance(row_update, int) or row_update < 1:
                    raise ValueError(f"invalid update on line {number}")
                if row_update in seen:
                    raise ValueError(f"duplicate update {row_update}")
                seen.add(row_update)
                if row_update <= update:
                    kept.append(row)
        except (OSError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise RunStateError("progress journal is malformed") from exc
        _atomic_rewrite_jsonl(progress_path, kept)
    archive_dir = run_dir / RECOVERY_ARCHIVE_DIRNAME
    if archive_dir.is_dir():
        for archive in archive_dir.iterdir():
            name = archive.name
            if not name.startswith("update_") or not name.endswith(".pt"):
                continue
            try:
                parsed = int(name[len("update_"):-len(".pt")])
            except ValueError:
                continue
            if parsed > update:
                archive.unlink()
    amendments_path = run_dir / _STOP_AMENDMENTS
    if amendments_path.exists():
        try:
            kept_amendments: list[dict[str, Any]] = []
            for line in amendments_path.read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                completed = row.get("completed_update")
                if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
                    raise ValueError("bad completed_update")
                if completed <= update:
                    kept_amendments.append(row)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RunStateError("stop budget amendment log is malformed") from exc
        _atomic_rewrite_jsonl(amendments_path, kept_amendments)
    append_run_status(run_dir, {"event": "rewound_to_archived_recovery", "completed_update": update})



def load_recovery(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise RunStateError("resume requested but recovery.pt is missing")
    try:
        # Keep the captured CPU RNG ByteTensor on CPU: torch.set_rng_state
        # rejects CUDA tensors.  The normal subsequent model/optimizer
        # load_state_dict calls place their tensors for ``device``.
        state = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        if not isinstance(state, dict) or not isinstance(state.get("completed_update"), int):
            raise ValueError("missing completed_update")
        if not isinstance(state.get("model"), dict) or not isinstance(state.get("optimizer"), dict):
            raise ValueError("missing model or optimizer")
        restore_rng_state(state["rng"])
        return state
    except (OSError, EOFError, RuntimeError, TypeError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        raise RunStateError("resume recovery state is malformed") from exc


_STOP_AMENDMENTS = "stop_budget_amendments.jsonl"
def _stop_budget(stop_args: Any) -> dict[str, Any]:
    """Validate and normalize a persisted/requested effective stop budget."""
    if not isinstance(stop_args, dict):
        raise RunStateError("resume stop budget is malformed")
    max_key = "effective_max_updates"
    end_key = "effective_end_time_hkt"
    if set(stop_args) != {max_key, end_key}:
        raise RunStateError("resume stop budget is malformed")
    maximum = stop_args[max_key]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise RunStateError("resume stop budget is malformed")
    end_text = stop_args[end_key]
    if not isinstance(end_text, str):
        raise RunStateError("resume stop budget is malformed")
    try:
        end = dt.datetime.fromisoformat(end_text)
    except ValueError as exc:
        raise RunStateError("resume stop budget is malformed") from exc
    if end.tzinfo is None or end.utcoffset() != dt.timedelta(hours=8):
        raise RunStateError("resume stop budget is malformed")
    # Keep the serialized form canonical so a textual alias cannot split an
    # otherwise contiguous audit chain.
    if end.isoformat() != end_text:
        raise RunStateError("resume stop budget is malformed")
    return {"effective_max_updates": maximum, "effective_end_time_hkt": end_text}


def _without_stop(contract: dict[str, Any]) -> dict[str, Any]:
    reduced = dict(contract)
    reduced.pop("stop_args", None)
    return reduced


def _require_current_schema_version(contract: dict[str, Any]) -> None:
    schema_version = contract.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) \
            or schema_version != CURRENT_RUN_SCHEMA_VERSION:
        raise RunStateError("run.json schema_version is unsupported")


def _recovery_completed_update(run_dir: Path) -> int:
    """Read the recorded completed update needed for stop-budget comparison."""
    state = torch.load(run_dir / "recovery.pt", map_location=torch.device("cpu"), weights_only=False)
    return state["completed_update"]


def _read_authorized_stop_budget(run_dir: Path, original: dict[str, Any]) -> dict[str, Any]:
    """Resolve original budget plus a strictly contiguous amendment chain."""
    authorized = _stop_budget(original.get("stop_args"))
    path = run_dir / _STOP_AMENDMENTS
    if not path.exists():
        return authorized
    try:
        lines = path.read_text().splitlines()
        previous_at: dt.datetime | None = None
        for number, line in enumerate(lines, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "event", "prior", "new", "at_hkt", "runtime_provenance", "completed_update",
            } or row["event"] != "stop_budget_extended" or not isinstance(row["runtime_provenance"], dict):
                raise ValueError(f"bad row {number}")
            completed = row["completed_update"]
            if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
                raise ValueError(f"bad completed_update {number}")
            at_hkt = row["at_hkt"]
            if not isinstance(at_hkt, str):
                raise ValueError(f"bad timestamp {number}")
            _stop_budget({"effective_max_updates": 0, "effective_end_time_hkt": at_hkt})
            recorded_at = dt.datetime.fromisoformat(at_hkt)
            if previous_at is not None and recorded_at < previous_at:
                raise ValueError(f"out-of-order timestamp {number}")
            prior = _stop_budget(row["prior"])
            new = _stop_budget(row["new"])
            if prior != authorized or new == prior:
                raise ValueError(f"non-contiguous row {number}")
            authorized = new
            previous_at = recorded_at
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RunStateError) as exc:
        raise RunStateError("stop budget amendment log is malformed") from exc
    return authorized


def _append_stop_budget_amendment(
    run_dir: Path, *, prior: dict[str, Any], new: dict[str, Any],
    completed_update: int, runtime_provenance: Any,
) -> None:
    if not isinstance(runtime_provenance, dict):
        raise RunStateError("resume runtime provenance is malformed")
    row = {
        "event": "stop_budget_extended", "prior": prior, "new": new,
        "completed_update": completed_update,
        "at_hkt": _now_hkt().isoformat(), "runtime_provenance": runtime_provenance,
    }
    path = run_dir / _STOP_AMENDMENTS
    with path.open("a") as audit:
        audit.write(json.dumps(row, sort_keys=True) + "\n")
        audit.flush()
        os.fsync(audit.fileno())


def require_matching_current_run_contract(run_dir: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on schema/identity before any resume mutation."""
    path = run_dir / "run.json"
    if not path.is_file():
        raise RunStateError("resume requested but run.json is missing")
    try:
        existing = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunStateError("resume run.json is malformed") from exc
    if not isinstance(existing, dict) or not isinstance(contract, dict):
        raise RunStateError("resume immutable run contract mismatch")
    _require_current_schema_version(existing)
    if _without_stop(existing) != _without_stop(contract):
        raise RunStateError("resume immutable run contract mismatch")
    return existing


def create_or_validate_run_contract(run_dir: Path, contract: dict[str, Any], *, resume: bool) -> dict[str, Any] | None:
    """Lock experiment identity; append audited stop-budget revisions."""
    path = run_dir / "run.json"
    if resume:
        existing = require_matching_current_run_contract(run_dir, contract)
        authorized = _read_authorized_stop_budget(run_dir, existing)
        requested = _stop_budget(contract.get("stop_args"))
        requested_end = dt.datetime.fromisoformat(requested["effective_end_time_hkt"])
        if requested_end <= _now_hkt():
            raise RunStateError("resume stop budget deadline is not in the future")
        if requested != authorized:
            completed = _recovery_completed_update(run_dir)
            if requested["effective_max_updates"] < completed:
                raise RunStateError("resume stop budget target is below completed update")
            _append_stop_budget_amendment(
                run_dir, prior=authorized, new=requested, completed_update=completed,
                runtime_provenance=contract.get("provenance", {}),
            )
            authorized = requested
        return authorized
    if path.exists():
        raise RunStateError("run directory already has run.json; use --resume")
    if not isinstance(contract, dict):
        raise RunStateError("run.json schema_version is unsupported")
    _require_current_schema_version(contract)
    atomic_write_json(path, contract)


class StopRequest:
    """Signal handler that asks the update loop to stop at its next boundary."""
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name


def configure_seed(seed: int | None) -> int:
    """Seed only when requested; preserve historical default streams otherwise."""
    effective = PPO_TORCH_SEED if seed is None else int(seed)
    if seed is not None:
        random.seed(effective)
        np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective


def boundary_stop_reason(
    *, completed: int, configured_updates: int, max_updates: int | None,
    end_time: dt.datetime | None, now: dt.datetime | None = None, interrupted: bool = False,
    max_updates_explicit: bool = True,
) -> str | None:
    """Return a boundary-only stop reason; deadline wins before a new update."""
    if interrupted:
        return "interrupted"
    if end_time is not None and (now or dt.datetime.now(_HKT)) >= end_time:
        return "deadline"
    limit = configured_updates if max_updates is None else max_updates
    if completed >= limit:
        return "max_updates" if max_updates_explicit else "completed"
    return None

# --------------------------------------------------------------------------- #
# canonical dataset constants (pre-registered, collected exactly once)
# --------------------------------------------------------------------------- #
DATA_SEED = 4
COLLECT_EPISODES = 240
COLLECT_SEED_START = 20000
COLLECT_MAX_STEPS = 4200
FRAMES_PER_EPISODE_CAP = 700
HELD_OUT_FRAC = 0.12

# --------------------------------------------------------------------------- #
# PPO hyperparameters -- copied verbatim from the current scripted-Player PPO
# CLI defaults (qrokkun_ai/train/player_v1.py), never reinvented.
# --------------------------------------------------------------------------- #
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_ENTROPY_COEF = 0.02
PPO_VALUE_COEF = 0.5
PPO_EPOCHS = 4
PPO_MINIBATCH = 512
EPISODES_PER_UPDATE = 8
PPO_MAX_GRAD_NORM = 1.0

# reuse the project's GAE/reward implementations verbatim -- never reinvented.
gae = scripted_ppo.gae
shaped_reward = scripted_ppo.shaped_reward

# --------------------------------------------------------------------------- #
# update budget / rollout / evaluation / snapshot schedules
# --------------------------------------------------------------------------- #
PPO_UPDATES = 200
PPO_MAX_FRAMES = 4200
PPO_ROLLOUT_SEED_START = 50000
PPO_TORCH_SEED = 0

EVAL_SEED_START = 3000
EVAL_SEED_COUNT = 30
EVAL_MAX_STEPS = 4200

SNAPSHOT_UPDATES: tuple[int, ...] = (0, 10, 25, 50, 100, 200)

# --------------------------------------------------------------------------- #
# pre-registered interpretation (retention, never promotion of a checkpoint)
# --------------------------------------------------------------------------- #
INITIAL_GATE_MEAN_MIN = 25.0
INITIAL_GATE_MEDIAN_MIN = 25.0
RETENTION_FRACTION = 0.80
MAX_HELD_AGREEMENT_DROP = 0.05


def ppo_hyperparameters() -> dict[str, Any]:
    """Every PPO knob used by the control, as a JSON-serializable dict."""
    return {
        "lr": PPO_LR,
        "gamma": PPO_GAMMA,
        "lam": PPO_LAMBDA,
        "clip": PPO_CLIP,
        "entropy_coef": PPO_ENTROPY_COEF,
        "value_coef": PPO_VALUE_COEF,
        "ppo_epochs": PPO_EPOCHS,
        "minibatch": PPO_MINIBATCH,
        "max_grad_norm": PPO_MAX_GRAD_NORM,
        "optimizer": "adam",
        "adam_eps": 1e-8,
        "advantage_normalization": True,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "max_frames_per_episode": PPO_MAX_FRAMES,
        "updates": PPO_UPDATES,
        "rollout_seed_start": PPO_ROLLOUT_SEED_START,
        "torch_seed": PPO_TORCH_SEED,
        "value_bootstrap": False,
        "truncation_treated_as_terminal": True,
    }


def rollout_seed_schedule(update: int, episodes_per_update: int = EPISODES_PER_UPDATE) -> list[int]:
    """Fixed, declared, deterministic per-update rollout seeds.

    Seeds never repeat across updates and never overlap the evaluation seed
    window, since the rollout seed space starts well above it. This is the
    single declared helper for rollout seeds -- callers (including
    ``run_ppo_arm``) must call it rather than reimplementing the arithmetic
    inline, so an actual (e.g. quick-mode) ``episodes_per_update`` always
    stays consistent with the seeds actually used.
    """
    start = PPO_ROLLOUT_SEED_START + update * episodes_per_update
    return list(range(start, start + episodes_per_update))


def eval_seed_list() -> list[int]:
    """Canonical deterministic evaluation seed window."""
    return list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SEED_COUNT))


def snapshot_schedule(final_update: int) -> list[int]:
    """Pre-registered snapshot updates, filtered to ``final_update`` with the
    final update always included."""
    updates = {u for u in SNAPSHOT_UPDATES if u <= final_update}
    updates.add(final_update)
    return sorted(updates)


# --------------------------------------------------------------------------- #
# strict checkpoint loading / provenance
# --------------------------------------------------------------------------- #
def load_initial_checkpoint(
    path: Path | str, device: torch.device
) -> tuple[PlayerRankedTopK, dict[str, Any]]:
    """Strictly load the initial ``player_ranked_topk`` production checkpoint.

    Delegates to :mod:`qrokkun_ai.agents.player_checkpoints`, which refuses
    legacy PlayerV1/PlayerV4 dict checkpoints, unknown architectures, and
    tampered state dicts by raising ``CheckpointError``.
    """
    net, meta = load_ranked_top_k_checkpoint(path, device, eval_mode=True)
    if meta.get("production_compatible") is not True or meta.get("experimental") is not False:
        raise CheckpointError(
            "initial checkpoint must be explicitly production-compatible "
            "(production_compatible=True, experimental=False)"
        )
    return net, meta


def checkpoint_provenance(path: Path | str, meta: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable provenance record for the loaded checkpoint."""
    return {
        "file_sha256": file_sha256(path),
        "state_dict_sha256": meta["state_dict_sha256"],
        "actions": list(ACTIONS),
        "observation": {
            "player_feat": PLAYER_FEAT_V4,
            "bullet_feat": BULLET_FEAT_V4,
            "max_bullets": MAX_BULLETS_V4,
        },
        "architecture": meta["architecture"],
        "schema_verified": True,
    }


# --------------------------------------------------------------------------- #
# rollout container (populated by collect_rollout, below -- future slice)
# --------------------------------------------------------------------------- #
@dataclass
class Rollout:
    seed: int
    player: list[np.ndarray] = field(default_factory=list)
    bullets: list[np.ndarray] = field(default_factory=list)
    pad: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    elapsed: float = 0.0
    censored: bool = True


# --------------------------------------------------------------------------- #
# on-policy sampled collection (never argmax; deterministic seeded reset only)
# --------------------------------------------------------------------------- #
def collect_rollout(net: PlayerRankedTopK, device: torch.device, seed: int, max_frames: int) -> Rollout:
    """Collect one on-policy rollout from a deterministically seeded env reset.

    Actions are SAMPLED from the current policy (never argmax) so the batch
    reflects the policy actually being optimized. Old log-probs, values and
    dones are recorded at collection time for the PPO ratio/clip terms.
    """
    env = Qrokkun26Env(seed=seed)
    env.reset(seed=seed)
    player_l: list[np.ndarray] = []
    bullets_l: list[np.ndarray] = []
    pad_l: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    elapsed = env.elapsed
    for _ in range(max_frames):
        player, bullets, pad = encode_obs(env)
        p_t = torch.tensor(player, dtype=torch.float32, device=device).unsqueeze(0)
        b_t = torch.tensor(bullets, dtype=torch.float32, device=device).unsqueeze(0)
        m_t = torch.tensor(pad, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            dist, value = net(p_t, b_t, m_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        player_l.append(player)
        bullets_l.append(bullets)
        pad_l.append(pad)
        actions.append(int(action.item()))
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))
        _obs, base, done, info = env.step(int(action.item()))
        rewards.append(shaped_reward(env, float(base), bool(done)))
        dones.append(bool(done))
        elapsed = float(info.get("elapsed", env.elapsed))
        if done:
            break
    censored = (not dones[-1]) if dones else True
    return Rollout(
        seed=seed,
        player=player_l,
        bullets=bullets_l,
        pad=pad_l,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        elapsed=elapsed,
        censored=censored,
    )


# --------------------------------------------------------------------------- #
# GAE / batch construction (per-episode boundaries; never bleed across them)
# --------------------------------------------------------------------------- #
def compute_gae_for_rollouts(
    rollouts: list[Rollout], gamma: float, lam: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages/returns computed per rollout (episode) then concatenated,
    so advantages never bleed across episode boundaries."""
    adv_parts: list[torch.Tensor] = []
    ret_parts: list[torch.Tensor] = []
    for r in rollouts:
        adv, ret_t = gae(r.rewards, r.values, r.dones, gamma, lam, device)
        adv_parts.append(adv)
        ret_parts.append(ret_t)
    return torch.cat(adv_parts), torch.cat(ret_parts)


def rollouts_to_batch(rollouts: list[Rollout], device: torch.device) -> dict[str, torch.Tensor]:
    """Flatten rollouts into a PPO batch. Contains only on-policy rollout
    data -- teacher frames/logits NEVER enter a PPO batch."""
    player_l: list[np.ndarray] = []
    bullets_l: list[np.ndarray] = []
    pad_l: list[np.ndarray] = []
    actions_l: list[int] = []
    old_lp_l: list[float] = []
    values_l: list[float] = []
    for r in rollouts:
        player_l.extend(r.player)
        bullets_l.extend(r.bullets)
        pad_l.extend(r.pad)
        actions_l.extend(r.actions)
        old_lp_l.extend(r.log_probs)
        values_l.extend(r.values)
    advantages, returns = compute_gae_for_rollouts(rollouts, PPO_GAMMA, PPO_LAMBDA, device)
    return {
        "player": torch.tensor(np.stack(player_l), dtype=torch.float32, device=device),
        "bullets": torch.tensor(np.stack(bullets_l), dtype=torch.float32, device=device),
        "pad": torch.tensor(np.stack(pad_l), dtype=torch.bool, device=device),
        "actions": torch.tensor(actions_l, dtype=torch.int64, device=device),
        "old_log_probs": torch.tensor(old_lp_l, dtype=torch.float32, device=device),
        "values": torch.tensor(values_l, dtype=torch.float32, device=device),
        "advantages": advantages,
        "returns": returns,
    }


def rollout_censor_stats(rollouts: list[Rollout]) -> dict[str, Any]:
    """How many/what fraction of this update's rollouts were censored
    (timed out at ``max_frames`` without a true terminal ``done``). The
    reused ``player_v1.gae`` always appends a terminal bootstrap value of
    zero regardless of whether the rollout actually terminated, so a
    censored (truncated) rollout is treated identically to a terminated
    one -- this never changes PPO semantics, it only makes that fact
    honestly observable per update."""
    n = len(rollouts)
    censor_count = sum(1 for r in rollouts if r.censored)
    return {
        "censor_count": censor_count,
        "censor_rate": censor_count / n if n else 0.0,
        "n_rollouts": n,
    }


def ppo_update(
    net: PlayerRankedTopK, opt: torch.optim.Optimizer, rollouts: list[Rollout], device: torch.device
) -> dict[str, Any]:
    """Clipped PPO update over the on-policy rollout batch. Reuses the
    module-level hyperparameters/GAE, never reinventing them."""
    batch = rollouts_to_batch(rollouts, device)
    player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
    actions, old_log_probs = batch["actions"], batch["old_log_probs"]
    old_values, returns = batch["values"], batch["returns"]
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n_samples = int(player.shape[0])
    optimizer_steps = 0
    kl_sum = 0.0
    clip_sum = 0.0
    entropy_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    total_loss_sum = 0.0
    seen = 0

    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, PPO_MINIBATCH):
            mb = perm[start : start + PPO_MINIBATCH]
            dist, value = net(player[mb], bullets[mb], pad[mb])
            log_prob = dist.log_prob(actions[mb])
            ratio = torch.exp(log_prob - old_log_probs[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), PPO_MAX_GRAD_NORM)
            opt.step()
            optimizer_steps += 1

            with torch.no_grad():
                approx_kl = float((old_log_probs[mb] - log_prob).mean().item())
                clip_fraction = float((torch.abs(ratio - 1.0) > PPO_CLIP).float().mean().item())
            bs = int(mb.shape[0])
            kl_sum += approx_kl * bs
            clip_sum += clip_fraction * bs
            entropy_sum += float(entropy.item()) * bs
            policy_loss_sum += float(policy_loss.item()) * bs
            value_loss_sum += float(value_loss.item()) * bs
            total_loss_sum += float(loss.item()) * bs
            seen += bs

    with torch.no_grad():
        ret_var = torch.var(returns)
        explained_variance = float(1.0 - torch.var(returns - old_values) / (ret_var + 1e-8))

    denom = max(seen, 1)
    return {
        "approx_kl": kl_sum / denom,
        "clip_fraction": clip_sum / denom,
        "explained_variance": explained_variance,
        "entropy": entropy_sum / denom,
        "policy_loss": policy_loss_sum / denom,
        "value_loss": value_loss_sum / denom,
        "total_loss": total_loss_sum / denom,
        "optimizer_steps": optimizer_steps,
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------- #
# deterministic evaluation + censoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_deterministic(
    net: PlayerRankedTopK, device: torch.device, seeds: list[int], max_steps: int
) -> list[dict[str, Any]]:
    """Deterministic argmax evaluation on the given seeds; reproducible."""
    was_training = net.training
    net.eval()
    results: list[dict[str, Any]] = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        elapsed = env.elapsed
        done = False
        for _ in range(max_steps):
            player, bullets, pad = encode_obs(env)
            p_t = torch.tensor(player, dtype=torch.float32, device=device).unsqueeze(0)
            b_t = torch.tensor(bullets, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.tensor(pad, dtype=torch.bool, device=device).unsqueeze(0)
            dist, _value = net(p_t, b_t, m_t)
            action = int(dist.logits.argmax(-1).item())
            _obs, _r, done, info = env.step(action)
            elapsed = float(info.get("elapsed", env.elapsed))
            if done:
                break
        results.append({"seed": seed, "elapsed": elapsed, "censored": not done})
    if was_training:
        net.train()
    return results


def summarize_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/median/pstdev/min/max plus explicit per-seed censoring."""
    elapsed = [float(r["elapsed"]) for r in results]
    n = len(results)
    censored_count = sum(1 for r in results if r["censored"])
    return {
        "mean": statistics.mean(elapsed),
        "median": statistics.median(elapsed),
        "pstdev": statistics.pstdev(elapsed),
        "min": min(elapsed),
        "max": max(elapsed),
        "n": n,
        "censored_count": censored_count,
        "censor_rate": censored_count / n if n else 0.0,
        "per_seed": {str(r["seed"]): float(r["elapsed"]) for r in results},
        "per_seed_censored": {str(r["seed"]): bool(r["censored"]) for r in results},
    }


# --------------------------------------------------------------------------- #
# teacher diagnostics (held frames only -- NEVER PPO rollout data)
# --------------------------------------------------------------------------- #
def _bucket_mean(values: list[float], agree: list[float], edges: list[tuple[float, float]], labels: list[str]) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = {label: [] for label in labels}
    for value, a in zip(values, agree):
        for (lo, hi), label in zip(edges, labels):
            if lo <= value < hi:
                buckets[label].append(a)
                break
    return {label: (sum(vals) / len(vals) if vals else None) for label, vals in buckets.items()}


@torch.no_grad()
def teacher_diagnostics(net: PlayerRankedTopK, tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    """Compare the student policy against held teacher logits ONLY -- these
    frames never enter a PPO batch/update."""
    was_training = net.training
    net.eval()
    player = tensors["player"].to(device)
    bullets = tensors["bullets"].to(device)
    pad = tensors["pad"].to(device)
    teacher_logits = tensors["teacher_logits"].to(device)
    elapsed = tensors["elapsed"].to(device)

    dist, _value = net(player, bullets, pad)
    student_logits = dist.logits

    student_argmax = student_logits.argmax(-1)
    teacher_argmax = teacher_logits.argmax(-1)
    agree_t = (student_argmax == teacher_argmax).float()
    agreement = float(agree_t.mean().item())

    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    kl = (teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs)).sum(-1)
    teacher_to_student_kl = float(kl.mean().item())

    n_live = (~pad).sum(dim=-1).float()

    elapsed_edges = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, float("inf"))]
    elapsed_labels = ["0-15", "15-30", "30-45", "45-60", "60+"]
    bullet_edges = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, float("inf"))]
    bullet_labels = ["0-2", "3-5", "6-9", "10+"]

    agree_list = agree_t.cpu().tolist()
    by_elapsed_bucket = _bucket_mean(elapsed.cpu().tolist(), agree_list, elapsed_edges, elapsed_labels)
    by_bullet_bucket = _bucket_mean(n_live.cpu().tolist(), agree_list, bullet_edges, bullet_labels)

    if was_training:
        net.train()

    return {
        "agreement": agreement,
        "teacher_to_student_kl": teacher_to_student_kl,
        "by_elapsed_bucket": by_elapsed_bucket,
        "by_bullet_bucket": by_bullet_bucket,
    }


# --------------------------------------------------------------------------- #
# frozen arm: zero optimizer steps, bit-identical weights, never mutated
# --------------------------------------------------------------------------- #
def run_frozen_arm(
    net: PlayerRankedTopK,
    device: torch.device,
    snapshots: list[int],
    eval_seeds: list[int],
    eval_max_steps: int,
) -> dict[str, Any]:
    """Evaluate ``net`` once (it is never touched) and report the same
    evaluation at every requested snapshot update; zero optimizer steps."""
    before = state_dict_sha256(net.state_dict())
    results = evaluate_deterministic(net, device, eval_seeds, eval_max_steps)
    summary = summarize_evaluation(results)
    snapshot_records = [{"update": update, "evaluation": summary} for update in snapshots]
    after = state_dict_sha256(net.state_dict())
    return {
        "optimizer_steps": 0,
        "state_dict_sha256_initial": before,
        "state_dict_sha256_final": after,
        "state_dict_sha256_by_update": {str(update): before for update in snapshots},
        "state_dict_bit_identical": before == after,
        "metrics_bit_identical": True,
        "snapshots": snapshot_records,
    }


# --------------------------------------------------------------------------- #
# pre-registered gate math (retention, never promotion)
# --------------------------------------------------------------------------- #
def evaluate_initial_gate(
    summary: dict[str, Any],
    *,
    mean_min: float = INITIAL_GATE_MEAN_MIN,
    median_min: float = INITIAL_GATE_MEDIAN_MIN,
) -> dict[str, Any]:
    """Initial gate: mean >= mean_min AND median >= median_min (inclusive)."""
    mean_pass = float(summary["mean"]) >= mean_min
    median_pass = float(summary["median"]) >= median_min
    failing = []
    if not mean_pass:
        failing.append("mean")
    if not median_pass:
        failing.append("median")
    gate_pass = mean_pass and median_pass
    reason = "" if gate_pass else f"initial gate failed: {', '.join(failing)} below minimum"
    return {
        "gate_pass": gate_pass,
        "mean_pass": mean_pass,
        "median_pass": median_pass,
        "mean_min": mean_min,
        "median_min": median_min,
        "reason": reason,
    }


class MissingHeldAgreementError(ValueError):
    """Raised when the pre-PPO reference lacks a usable agreement value."""


def build_retention_reference(initial_snapshot: dict[str, Any]) -> dict[str, float]:
    agreement = initial_snapshot.get("teacher_diagnostics", {}).get("agreement")
    if agreement is None:
        raise MissingHeldAgreementError("initial snapshot is missing held teacher agreement")
    evaluation = initial_snapshot["evaluation"]
    return {
        "mean": float(evaluation["mean"]),
        "median": float(evaluation["median"]),
        "held_agreement": float(agreement),
    }


def evaluate_retention(reference: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Final retention passes only if the FINAL PPO mean and median are each
    >= 80% of the reference AND held teacher agreement declined by no more
    than 0.05 absolute. The first snapshot where either threshold breaks is
    always reported. This is a retention gate only -- there is no promotion
    of any checkpoint anywhere."""
    ordered = sorted(snapshots, key=lambda s: s["update"])
    mean_threshold = RETENTION_FRACTION * float(reference["mean"])
    median_threshold = RETENTION_FRACTION * float(reference["median"])
    agreement_floor = float(reference["held_agreement"]) - MAX_HELD_AGREEMENT_DROP

    first_breaking_snapshot = None
    missing_agreement_updates: list[int] = []
    for s in ordered:
        ev = s["evaluation"]
        agreement = s.get("teacher_diagnostics", {}).get("agreement")
        if agreement is None:
            missing_agreement_updates.append(s["update"])
        breaks = (
            ev["mean"] < mean_threshold
            or ev["median"] < median_threshold
            or agreement is None
            or agreement < agreement_floor
        )
        if breaks and first_breaking_snapshot is None:
            first_breaking_snapshot = s["update"]

    final = ordered[-1]
    final_mean_pass = final["evaluation"]["mean"] >= mean_threshold
    final_median_pass = final["evaluation"]["median"] >= median_threshold
    final_agreement = final.get("teacher_diagnostics", {}).get("agreement")
    if final_agreement is None:
        final_agreement_drop = None
    else:
        final_agreement_drop = float(reference["held_agreement"]) - float(final_agreement)
    agreement_pass = (
        final_agreement is not None
        and not missing_agreement_updates
        and final_agreement >= agreement_floor
    )

    retention_pass = final_mean_pass and final_median_pass and agreement_pass
    failing = []
    if not final_mean_pass:
        failing.append("mean below threshold")
    if not final_median_pass:
        failing.append("median below threshold")
    if missing_agreement_updates:
        updates = ", ".join(str(update) for update in missing_agreement_updates)
        failing.append(f"missing held teacher agreement at update(s): {updates}")
    elif not agreement_pass:
        failing.append("agreement below threshold")
    reason = "" if retention_pass else f"retention failed: {', '.join(failing)}"

    return {
        "retention_pass": retention_pass,
        "mean_threshold": mean_threshold,
        "median_threshold": median_threshold,
        "agreement_floor": agreement_floor,
        "first_breaking_snapshot": first_breaking_snapshot,
        "final_mean_pass": final_mean_pass,
        "final_median_pass": final_median_pass,
        "agreement_pass": agreement_pass,
        "final_agreement_drop": final_agreement_drop,
        "reason": reason,
        "promotion": False,
    }


# --------------------------------------------------------------------------- #
# canonical teacher dataset (collected exactly once locally; held frames only
# ever feed teacher-agreement diagnostics -- never a PPO batch)
# --------------------------------------------------------------------------- #
def load_teacher(path: Path | str, device: torch.device) -> PlayerV1:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = PlayerV1(hidden=int(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.to(device)
    net.eval()
    return net


def collect_canonical_dataset(
    teacher: PlayerV1,
    device: torch.device,
    *,
    episodes: int = COLLECT_EPISODES,
    seed_start: int = COLLECT_SEED_START,
    max_steps: int = COLLECT_MAX_STEPS,
    frames_cap: int = FRAMES_PER_EPISODE_CAP,
    held_frac: float = HELD_OUT_FRAC,
    teacher_path: Path | str | None = None,
    include_train_tensors: bool = False,
) -> dict[str, Any]:
    """Recollect the canonical V1 teacher dataset locally (once) using the
    EXACT Phase2 ``collect_dataset``/``collect_episode`` contract -- full
    episodes up to ``max_steps``, then even-stride cap subsampling over the
    WHOLE episode (never a truncation to the first ``frames_cap`` frames),
    split by episode into train/held with the shared ``DATA_SEED``. This is
    never reimplemented here; it delegates to the Phase2 helpers verbatim.
    Only held-episode frames are materialized into tensors, since train
    frames are never used by this harness (no teacher data ever enters a
    PPO batch). The identity hash is Phase2's own
    ``phase2_ranked_multiseed.dataset_identity`` over the actual Frame
    content (player/bullets/pad/teacher_logits/elapsed/episode) -- never a
    hash of collection parameters alone.

    ``include_train_tensors`` additionally materializes the TRAIN split
    tensors. This control never uses them (no teacher data ever enters a PPO
    batch here); they exist for the round-1 auxiliary-retention arm, which
    trains its retention term on the train split only."""
    rng = random.Random(DATA_SEED)
    train_frames, held_frames = collect_dataset(
        teacher,
        device,
        n_episodes=episodes,
        seed_start=seed_start,
        max_steps=max_steps,
        frames_cap=frames_cap,
        held_out_frac=held_frac,
        rng=rng,
    )
    identity = dataset_identity(train_frames, held_frames)
    held_tensors = frames_to_tensors(held_frames, device)

    result: dict[str, Any] = {
        "hash": identity["hash"],
        "n_episodes": episodes,
        "n_held_episodes": len({f.episode for f in held_frames}),
        "n_train_episodes": len({f.episode for f in train_frames}),
        "n_held_frames": identity["n_held"],
        "n_train_frames": identity["n_train"],
        "held_tensors": held_tensors,
    }
    if teacher_path is not None:
        result["teacher_file_sha256"] = file_sha256(teacher_path)
    if include_train_tensors:
        result["train_tensors"] = frames_to_tensors(train_frames, device)
    return result


# --------------------------------------------------------------------------- #
# PPO arm: on-policy training with the pre-registered snapshot schedule
# --------------------------------------------------------------------------- #
def run_ppo_arm(
    init_net: PlayerRankedTopK,
    device: torch.device,
    args: argparse.Namespace,
    snapshot_updates: list[int],
    held_tensors: dict[str, torch.Tensor] | None,
    out_dir: Path,
    *,
    parent_state_dict_sha256: str,
    parent_file_sha256: str,
    dataset_hash: str,
    ppo_knobs: dict[str, Any],
) -> dict[str, Any]:
    """Train the paired PPO arm from the exact same loaded initial state as
    the frozen arm (a fresh clone, so the frozen arm/init checkpoint are
    never mutated).

    Every snapshot/final checkpoint written here is an experimental,
    non-production artifact: it is packed with ``experimental=True`` /
    ``production_compatible=False`` and an ``extra`` provenance block
    threaded straight from ``run_experiment`` (the parent init checkpoint's
    state-dict/file hashes, the canonical dataset hash, and the actual
    per-run PPO knobs -- never reimplemented or reinvented locally), plus
    the eval summary at that update and the rollout/eval seed windows used
    to produce it.
    """
    net = PlayerRankedTopK(top_k=init_net.top_k, hidden=init_net.hidden).to(device)
    net.load_state_dict(init_net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR, eps=1e-8)
    # Preserve the historical default timing: the arm seed comes after its
    # clone/optimizer construction.  An explicit CLI seed intentionally has
    # already been applied before construction by ``apply_mode_defaults``.
    if getattr(args, "seed", None) is None:
        torch.manual_seed(PPO_TORCH_SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "ppo_updates.jsonl"
    snapshots: list[dict[str, Any]] = []
    total_frames = 0
    optimizer_steps = 0
    # Seed window for the rollout that produced (or is about to produce, for
    # the update-0 snapshot) the current net weights.
    rollout_seed_window = rollout_seed_schedule(0, episodes_per_update=args.episodes_per_update)

    def _pack_extra(update: int, eval_summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "update": update,
            "parent_state_dict_sha256": parent_state_dict_sha256,
            "parent_file_sha256": parent_file_sha256,
            "dataset_hash": dataset_hash,
            "eval_summary": eval_summary,
            "ppo_knobs": ppo_knobs,
            "rollout_seed_window": rollout_seed_window,
            "eval_seed_window": args.eval_seeds,
        }

    def _snapshot(update: int) -> None:
        eval_results = evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
        eval_summary = summarize_evaluation(eval_results)
        if held_tensors is not None:
            diag = teacher_diagnostics(net, held_tensors, device)
        else:
            diag = {"agreement": None}
        ckpt_path = out_dir / f"ppo_update_{update}.pt"
        save_player_checkpoint(
            net,
            ckpt_path,
            source_tool="phase3_ranked_ppo_retention",
            experimental=True,
            production_compatible=False,
            extra=_pack_extra(update, eval_summary),
        )
        snapshots.append(
            {
                "update": update,
                "evaluation": eval_summary,
                "teacher_diagnostics": diag,
                "checkpoint": str(ckpt_path),
                "state_dict_sha256": state_dict_sha256(net.state_dict()),
            }
        )

    def _save_periodic_model_checkpoint(update: int) -> None:
        if update <= 0 or update % RECOVERY_ARCHIVE_INTERVAL != 0:
            return
        if update in snapshot_updates and update <= RECOVERY_ARCHIVE_INTERVAL:
            return
        evaluation = summarize_evaluation(
            evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps)
        )
        save_player_checkpoint(
            net,
            out_dir / f"ppo_update_{update}.pt",
            source_tool="phase3_ranked_ppo_retention",
            experimental=True,
            production_compatible=False,
            extra={
                **_pack_extra(update, evaluation),
                "checkpoint_kind": "periodic_model_only",
            },
        )

    recovery_path = out_dir / "recovery.pt"
    completed = 0
    if getattr(args, "resume", False):
        recovery = load_recovery(recovery_path, device)
        try:
            net.load_state_dict(recovery["model"], strict=True)
            opt.load_state_dict(recovery["optimizer"])
            completed = recovery["completed_update"]
            total_frames = int(recovery["total_frames"])
            optimizer_steps = int(recovery["optimizer_steps"])
            snapshots = list(recovery.get("snapshots", []))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RunStateError("resume recovery state is malformed") from exc
        append_run_status(out_dir, {"event": "resumed", "completed_update": completed})
    elif 0 in snapshot_updates:
        _snapshot(0)

    # A durable journal may be one update ahead when a crash landed between
    # its fsync and the atomic recovery replacement.  That update is replayed
    # from recovery without adding a second history row.
    reconcile_progress_journal(jsonl_path, completed_update=completed)

    def _save_boundary() -> None:
        state = {
            "format": 1, "arm": "control", "model": net.state_dict(),
            "optimizer": opt.state_dict(), "completed_update": completed,
            "total_frames": total_frames, "optimizer_steps": optimizer_steps,
            "snapshots": snapshots, "rng": capture_rng_state(),
        }
        atomic_save_recovery(recovery_path, state)
        save_archived_recovery_if_due(out_dir, state)

    if not getattr(args, "resume", False):
        _save_boundary()
    stop = StopRequest()
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in old_handlers:
        signal.signal(s, stop.handler)
    stop_reason: str | None = None
    max_updates, end_time = effective_stop_budget(args)
    max_updates_explicit = getattr(args, "max_updates", None) is not None
    try:
        for update in range(completed + 1, max_updates + 1):
                # A signal only requests an orderly boundary; none is claimed
                # until this whole update has been optimized and checkpointed.
                requested_reason = boundary_stop_reason(
                    completed=completed, configured_updates=args.updates,
                    max_updates=max_updates, end_time=end_time, interrupted=stop.requested,
                    max_updates_explicit=max_updates_explicit,
                )
                if requested_reason is not None:
                    stop_reason = requested_reason
                    break
                seeds = rollout_seed_schedule(update - 1, episodes_per_update=args.episodes_per_update)
                rollout_seed_window = seeds
                rollouts = [collect_rollout(net, device, seed, max_frames=args.max_frames) for seed in seeds]
                metrics = ppo_update(net, opt, rollouts, device)
                optimizer_steps += int(metrics["optimizer_steps"])
                total_frames += sum(len(r.actions) for r in rollouts)
                row = dict(metrics)
                row["update"] = update
                row["scripted_survival_mean"] = statistics.mean([r.elapsed for r in rollouts])
                row["rollout_censoring"] = rollout_censor_stats(rollouts)
                append_progress_row(jsonl_path, row)
                completed = update
                _save_boundary()
                append_run_status(out_dir, {
                    "event": "update_complete", "update": update, "total_frames": total_frames,
                    "effective_max_updates": max_updates, "effective_end_time_hkt": end_time.isoformat(),
                })
                if update in snapshot_updates and update <= RECOVERY_ARCHIVE_INTERVAL:
                    _snapshot(update)
                    _save_boundary()
                _save_periodic_model_checkpoint(update)
                if stop.requested:
                    stop_reason = "interrupted"
                    break
                requested_reason = boundary_stop_reason(
                    completed=completed, configured_updates=args.updates,
                    max_updates=max_updates, end_time=end_time,
                    interrupted=False, max_updates_explicit=max_updates_explicit,
                )
                if requested_reason is not None:
                    stop_reason = requested_reason
                    break
    finally:
        for s, previous in old_handlers.items():
            signal.signal(s, previous)

    if stop_reason is None:
        stop_reason = boundary_stop_reason(
            completed=completed, configured_updates=args.updates, max_updates=max_updates,
            end_time=end_time, max_updates_explicit=max_updates_explicit,
        ) or "max_updates"
    append_run_status(out_dir, {
        "event": stop_reason, "completed_update": completed,
        "effective_max_updates": max_updates, "effective_end_time_hkt": end_time.isoformat(),
    })
    if (
        completed
        and not any(item["update"] == completed for item in snapshots)
        and not (completed > RECOVERY_ARCHIVE_INTERVAL and completed % RECOVERY_ARCHIVE_INTERVAL == 0)
    ):
        _snapshot(completed)
        _save_boundary()

    final_eval_summary = summarize_evaluation(evaluate_deterministic(net, device, args.eval_seeds, args.eval_max_steps))
    final_path = out_dir / "ppo_final.pt"
    save_player_checkpoint(
        net,
        final_path,
        source_tool="phase3_ranked_ppo_retention",
        experimental=True,
        production_compatible=False,
        extra=_pack_extra(completed, final_eval_summary),
    )

    return {
        "optimizer_steps": optimizer_steps,
        "total_episodes": completed * args.episodes_per_update,
        "total_frames": total_frames,
        "snapshots": snapshots,
        "final_checkpoint": str(final_path),
        "completed_updates": completed,
        "stop_reason": stop_reason,
        "effective_max_updates": max_updates,
        "effective_end_time_hkt": end_time.isoformat(),
    }


# --------------------------------------------------------------------------- #
# CLI / experiment orchestration
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase3_ranked_ppo_retention"))
    ap.add_argument("--run-dir", type=Path, help="resumable run directory (defaults to --out-dir)")
    ap.add_argument("--end-time", type=parse_end_time, help="HKT deadline: YYYYMMDD-HHMM")
    ap.add_argument("--max-updates", type=int, help="maximum completed updates for this run")
    ap.add_argument("--resume", action="store_true", help="resume only from a matching recovery boundary")
    ap.add_argument(
        "--resume-from-update", type=int, metavar="N",
        help="resume exactly from archived full recovery update N (positive multiple of 200)",
    )
    ap.add_argument("--seed", type=int, help="optional explicit RNG seed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in the update budget / seed windows / gate thresholds. Non-quick
    runs use the pre-registered production defaults (200 updates etc); quick
    mode shrinks every knob so a control can be smoke-tested fast, and
    relaxes the initial gate (never the retention math itself)."""
    if getattr(args, "quick", False):
        args.updates = 2
        args.episodes_per_update = 2
        args.max_frames = 30
        args.eval_seeds = eval_seed_list()[:3]
        args.eval_max_steps = 40
        args.data_episodes = 6
        args.data_max_steps = 40
        args.data_frames_cap = 40
        args.initial_gate_mean_min = 0.0
        args.initial_gate_median_min = 0.0
    else:
        args.updates = PPO_UPDATES
        args.episodes_per_update = EPISODES_PER_UPDATE
        args.max_frames = PPO_MAX_FRAMES
        args.eval_seeds = eval_seed_list()
        args.eval_max_steps = EVAL_MAX_STEPS
        args.data_episodes = COLLECT_EPISODES
        args.data_max_steps = COLLECT_MAX_STEPS
        args.data_frames_cap = FRAMES_PER_EPISODE_CAP
        args.initial_gate_mean_min = INITIAL_GATE_MEAN_MIN
        args.initial_gate_median_min = INITIAL_GATE_MEDIAN_MIN
    if getattr(args, "max_updates", None) is not None and args.max_updates < 0:
        raise ValueError("--max-updates must be non-negative")
    selected = getattr(args, "resume_from_update", None)
    if selected is not None:
        args.effective_max_updates, args.effective_end_time = resolve_resume_from_stop_budget(
            args, selected,
        )
    else:
        args.effective_max_updates, args.effective_end_time = resolve_stop_budget(
            getattr(args, "max_updates", None), getattr(args, "end_time", None),
            default_max_updates=args.updates,
        )
    if getattr(args, "run_dir", None) is not None:
        args.run_dir = Path(args.run_dir)
    elif hasattr(args, "out_dir"):
        args.run_dir = Path(args.out_dir)
    else:
        args.run_dir = None
    seed = getattr(args, "seed", None)
    # With no --seed, retain the historical arm-local torch seeding timing.
    # Explicit seeds intentionally apply before any network construction.
    args.effective_seed = PPO_TORCH_SEED if seed is None else configure_seed(seed)
    return args


def _git_dirty(repo_root: Path | None = None) -> bool | None:
    root = repo_root if repo_root is not None else _REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def run_experiment(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Strict, fail-closed orchestration of the paired frozen/ppo control.
    No env rollouts and no PPO ever run before the pre-registered initial
    gate passes; no checkpoint promotion happens anywhere."""
    out_dir = Path(getattr(args, "run_dir", args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_max_updates, effective_end_time = effective_stop_budget(args)

    init_net, init_meta = load_initial_checkpoint(args.init_checkpoint, device)
    init_prov = checkpoint_provenance(args.init_checkpoint, init_meta)
    try:
        teacher_prov = {"file_sha256": file_sha256(args.teacher)}
    except OSError:
        teacher_prov = {"file_sha256": None}
    provenance = {
        "init_checkpoint": init_prov,
        "teacher": teacher_prov,
        "git_commit": current_git_commit(_REPO_ROOT),
        "dirty": _git_dirty(_REPO_ROOT),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    knobs = ppo_hyperparameters()
    knobs["updates"] = args.updates
    knobs["episodes_per_update"] = args.episodes_per_update
    knobs["max_frames_per_episode"] = args.max_frames
    knobs["eval_seeds"] = args.eval_seeds
    knobs["eval_max_steps"] = args.eval_max_steps
    knobs["initial_gate_mean_min"] = args.initial_gate_mean_min
    knobs["initial_gate_median_min"] = args.initial_gate_median_min
    knobs["data_episodes"] = args.data_episodes
    knobs["data_max_steps"] = args.data_max_steps
    knobs["data_frames_cap"] = args.data_frames_cap

    contract = {
        "format": 1, "schema_version": CURRENT_RUN_SCHEMA_VERSION,
        "tool": "phase3_ranked_ppo_retention", "arm": "control",
        "inputs": {"init_checkpoint": init_prov, "teacher": teacher_prov}, "provenance": provenance,
        "knobs": knobs,
        "stop_args": {
            "effective_end_time_hkt": effective_end_time.isoformat(),
            "effective_max_updates": effective_max_updates,
        },
        "effective_seed": getattr(args, "effective_seed", PPO_TORCH_SEED), "no_promotion": True,
    }
    # A resume is tied to an immutable contract.  Validate it before the
    # initial gate, which evaluates the environment, while retaining the
    # historical new-run gate-before-run-creation behaviour.
    selected = getattr(args, "resume_from_update", None)
    if selected is not None:
        resolved_max, resolved_end = resolve_resume_from_stop_budget(args, selected)
        args.effective_max_updates = resolved_max
        args.effective_end_time = resolved_end
        effective_max_updates, effective_end_time = resolved_max, resolved_end
        contract["stop_args"] = {
            "effective_end_time_hkt": resolved_end.isoformat(),
            "effective_max_updates": resolved_max,
        }
        require_matching_current_run_contract(out_dir, contract)
        rewind_run_to_archived_recovery(
            out_dir, selected, progress_filename="ppo_updates.jsonl", device=device,
        )
        authorized = create_or_validate_run_contract(out_dir, contract, resume=True)
        args.effective_max_updates = authorized["effective_max_updates"]
        args.effective_end_time = dt.datetime.fromisoformat(authorized["effective_end_time_hkt"])
        effective_max_updates, effective_end_time = args.effective_max_updates, args.effective_end_time
    elif getattr(args, "resume", False):
        authorized = create_or_validate_run_contract(out_dir, contract, resume=True)
        args.effective_max_updates = authorized["effective_max_updates"]
        args.effective_end_time = dt.datetime.fromisoformat(authorized["effective_end_time_hkt"])
        effective_max_updates, effective_end_time = args.effective_max_updates, args.effective_end_time

    gate_results = evaluate_deterministic(init_net, device, args.eval_seeds, args.eval_max_steps)
    gate_summary = summarize_evaluation(gate_results)
    initial_gate = evaluate_initial_gate(
        gate_summary, mean_min=args.initial_gate_mean_min, median_min=args.initial_gate_median_min
    )

    report: dict[str, Any] = {
        "status": None,
        "control_ran": False,
        "initial_gate": initial_gate,
        "provenance": provenance,
        "knobs": knobs,
        "dataset": None,
        "frozen_arm": None,
        "ppo_arm": None,
        "retention": None,
        "stop_budget": {
            "effective_max_updates": effective_max_updates,
            "effective_end_time_hkt": effective_end_time.isoformat(),
        },
    }

    if not initial_gate["gate_pass"]:
        report["status"] = "failed_closed"
        _write_report(out_dir, report)
        return report

    if teacher_prov["file_sha256"] is None:
        report["status"] = "failed_closed"
        _write_report(out_dir, report)
        return report

    if not getattr(args, "resume", False):
        create_or_validate_run_contract(out_dir, contract, resume=False)
    append_run_status(out_dir, {
        "event": "resume_requested" if getattr(args, "resume", False) else "started",
        "effective_max_updates": effective_max_updates,
        "effective_end_time_hkt": effective_end_time.isoformat(),
    })

    teacher = load_teacher(args.teacher, device)
    dataset = collect_canonical_dataset(
        teacher,
        device,
        episodes=args.data_episodes,
        seed_start=COLLECT_SEED_START,
        max_steps=args.data_max_steps,
        frames_cap=args.data_frames_cap,
        held_frac=HELD_OUT_FRAC,
        teacher_path=args.teacher,
    )
    report["dataset"] = {
        "hash": dataset["hash"],
        "n_episodes": dataset["n_episodes"],
        "n_held_episodes": dataset["n_held_episodes"],
        "n_train_episodes": dataset["n_train_episodes"],
        "n_held_frames": dataset["n_held_frames"],
        "n_train_frames": dataset["n_train_frames"],
        "teacher_file_sha256": dataset["teacher_file_sha256"],
    }

    snapshot_updates = snapshot_schedule(args.updates)

    frozen_arm = run_frozen_arm(init_net, device, snapshot_updates, args.eval_seeds, args.eval_max_steps)
    report["frozen_arm"] = frozen_arm

    ppo_arm = run_ppo_arm(
        init_net,
        device,
        args,
        snapshot_updates,
        dataset["held_tensors"],
        out_dir,
        parent_state_dict_sha256=init_prov["state_dict_sha256"],
        parent_file_sha256=init_prov["file_sha256"],
        dataset_hash=dataset["hash"],
        ppo_knobs=knobs,
    )
    report["ppo_arm"] = ppo_arm

    initial_snapshot = ppo_arm["snapshots"][0]
    reference = build_retention_reference(initial_snapshot)
    retention = evaluate_retention(reference, ppo_arm["snapshots"])
    report["retention"] = retention

    report["status"] = ppo_arm["stop_reason"]
    report["control_ran"] = True
    _write_report(out_dir, report)
    return report


def main() -> None:
    args = apply_mode_defaults(build_parser().parse_args())
    device = torch.device(args.device)
    report = run_experiment(args, device)
    print(json.dumps({"status": report["status"], "control_ran": report["control_ran"]}))


if __name__ == "__main__":
    main()
