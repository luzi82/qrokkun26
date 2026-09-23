#!/usr/bin/env python3
"""Shared launch/input/artifact provenance for the Phase 3 retention arms.

Both Phase 3 arms -- the control (:mod:`phase3_ranked_ppo_retention`) and the
auxiliary-retention arm (:mod:`phase3_ranked_ppo_retention_aux`) -- record the
same launch snapshot, input identity, artifact inventory and report
completeness verdict.  Every one of those records is produced HERE, by one
implementation called from both arms, so the two arms cannot drift apart into
subtly different evidence.

Nothing in this module ever serializes the process environment: a launch
record captures the command line, the resolved working directory and the
runtime/hardware identity, never ``os.environ``.  Values that a command line
can plausibly carry as a credential are redacted by
:func:`sanitize_argv` before anything is written.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# The single definition of the experiment wall clock, shared by both arms.
HKT = ZoneInfo("Asia/Hong_Kong")

LAUNCH_SCHEMA_VERSION = 1

REDACTED = "<redacted>"

# Evidence provenance vocabulary shared by every generated reporting record.
#
# ``observed``     -- read directly from the live process, host or filesystem
#                     at the moment the record was written.
# ``computed``     -- produced by hashing or summarizing actual artifact
#                     content (a file SHA-256, a paired t statistic).
# ``derived``      -- reconstructed from a locked deterministic rule rather
#                     than read from a recorded value (a rollout seed window
#                     rebuilt from the schedule).
# ``unavailable``  -- could not be produced here; the value is explicitly
#                     ``None`` and is never guessed or back-filled.
EVIDENCE_OBSERVED = "observed"
EVIDENCE_COMPUTED = "computed"
EVIDENCE_DERIVED = "derived"
EVIDENCE_UNAVAILABLE = "unavailable"
EVIDENCE_STATUSES = (
    EVIDENCE_OBSERVED,
    EVIDENCE_COMPUTED,
    EVIDENCE_DERIVED,
    EVIDENCE_UNAVAILABLE,
)

# Flag/variable name components that make the following value a credential.
# Matching is component-wise (``--api-key`` -> {"api", "key"}) so an ordinary
# experiment flag such as ``--rollout-seed-start`` can never match.
_SECRET_NAME_COMPONENTS = frozenset(
    {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "cred",
        "credential",
        "credentials",
        "key",
        "keys",
        "passwd",
        "password",
        "pwd",
        "secret",
        "secrets",
        "session",
        "signature",
        "token",
        "tokens",
    }
)

_NAME_VALUE_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)=(?P<value>.*)$", re.DOTALL)
_URL_USERINFO_RE = re.compile(r"(?P<prefix>[A-Za-z][A-Za-z0-9+.\-]*://[^/\s:@]+:)(?P<secret>[^/\s@]*)@")


def dumps_canonical(value: Any) -> str:
    """Deterministic JSON text used for hashing and for redaction assertions."""
    return json.dumps(value, sort_keys=True, default=str)


def now_hkt() -> dt.datetime:
    return dt.datetime.now(HKT)


def atomic_replace(path: Path, write: Any) -> None:
    """Write via a same-directory temporary file, then atomically rename it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_replace(path, lambda tmp: tmp.write_text(json.dumps(value, indent=2, sort_keys=True)))


def _name_components(name: str) -> set[str]:
    return {part for part in re.split(r"[-_.]+", name.lstrip("-").lower()) if part}


def is_secret_name(name: str) -> bool:
    """Whether a flag/variable name makes its value a credential."""
    return bool(_name_components(name) & _SECRET_NAME_COMPONENTS)


def _redact_url_userinfo(token: str) -> str:
    return _URL_USERINFO_RE.sub(lambda m: f"{m.group('prefix')}{REDACTED}@", token)


def _looks_like_flag(token: str) -> bool:
    return token.startswith("-") and token != "-"


def sanitize_argv(argv: list[str] | tuple[str, ...]) -> list[str]:
    """Return ``argv`` with credential-bearing VALUES replaced by :data:`REDACTED`.

    Flag names, their order and every non-credential value are preserved
    verbatim so the recorded command line stays an auditable reconstruction of
    the launch.  Three shapes are covered: ``--flag VALUE``, ``--flag=VALUE``
    and ``NAME=VALUE``; in addition any URL userinfo password is redacted
    wherever it appears.  A secret-named flag that carries no value (the next
    token is itself a flag) never swallows that next token.
    """
    sanitized: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next and not _looks_like_flag(token):
            sanitized.append(REDACTED)
            redact_next = False
            continue
        redact_next = False
        if _looks_like_flag(token):
            name, sep, _value = token.partition("=")
            if is_secret_name(name):
                sanitized.append(f"{name}={REDACTED}" if sep else token)
                redact_next = not sep
                continue
        else:
            match = _NAME_VALUE_RE.match(token)
            if match is not None and is_secret_name(match.group("name")):
                sanitized.append(f"{match.group('name')}={REDACTED}")
                continue
        sanitized.append(_redact_url_userinfo(token))
    return sanitized


def _probe(call: Any) -> Any:
    """Run a hardware/driver query that is allowed to be absent or to fail."""
    try:
        return call()
    except Exception:  # noqa: BLE001 - any driver/runtime failure means "unavailable"
        return None


def _cuda_snapshot(torch: Any) -> dict[str, Any]:
    """CUDA build/runtime/driver identity, probed only where it is safe.

    ``build_version`` is a property of the installed wheel and is readable on
    any host.  Runtime and driver versions are only queried once
    ``torch.cuda.is_available()`` has said there is something to query, so a
    CPU-only host never initializes a CUDA context.
    """
    build_version = _probe(lambda: torch.version.cuda)
    available = bool(_probe(torch.cuda.is_available))
    snapshot: dict[str, Any] = {
        "available": available,
        "build_version": build_version,
        "runtime_version": None,
        "driver_version": None,
        "cudnn_version": None,
        "status": EVIDENCE_UNAVAILABLE,
    }
    if not available:
        return snapshot
    snapshot["runtime_version"] = _probe(lambda: torch._C._cuda_getCompiledVersion())
    if snapshot["runtime_version"] is None:
        snapshot["runtime_version"] = build_version
    snapshot["driver_version"] = _probe(lambda: torch._C._cuda_getDriverVersion())
    if snapshot["driver_version"] is None:
        snapshot["driver_version"] = _probe(lambda: torch.cuda.driver_version())
    snapshot["cudnn_version"] = _probe(lambda: torch.backends.cudnn.version())
    snapshot["status"] = EVIDENCE_OBSERVED
    return snapshot


def _gpu_snapshot(torch: Any) -> dict[str, Any]:
    """Model/count/properties of the visible GPUs, or explicit unavailability."""
    if not _probe(torch.cuda.is_available):
        return {"count": 0, "devices": [], "status": EVIDENCE_UNAVAILABLE}
    count = _probe(torch.cuda.device_count)
    if not count:
        return {"count": 0, "devices": [], "status": EVIDENCE_UNAVAILABLE}
    devices: list[dict[str, Any]] = []
    for index in range(int(count)):
        props = _probe(lambda index=index: torch.cuda.get_device_properties(index))
        devices.append(
            {
                "index": index,
                "name": _probe(lambda index=index: torch.cuda.get_device_name(index)),
                "total_memory_bytes": getattr(props, "total_memory", None),
                "capability": (
                    None
                    if props is None
                    else f"{getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}"
                ),
                "multi_processor_count": getattr(props, "multi_processor_count", None),
            }
        )
    return {"count": int(count), "devices": devices, "status": EVIDENCE_OBSERVED}


def runtime_snapshot(*, device_request: str | None = None) -> dict[str, Any]:
    """Host/platform/interpreter/accelerator identity observed at this launch.

    The process environment is never included: a launch record carries the
    command line, the resolved working directory and the runtime identity,
    and secrets live in neither of the latter two.  Every accelerator probe
    degrades to an explicit ``unavailable`` status rather than raising, so a
    CPU-only host produces a complete record with honest nulls.
    """
    import platform
    import socket

    import numpy as np
    import torch

    return {
        "host": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "device_request": None if device_request is None else str(device_request),
        "cuda": _cuda_snapshot(torch),
        "gpu": _gpu_snapshot(torch),
    }


# --------------------------------------------------------------------------- #
# launch.json -- one append-only record per invocation of either arm
# --------------------------------------------------------------------------- #
LAUNCH_FILENAME = "launch.json"

LAUNCH_MODE_FRESH = "fresh"
LAUNCH_MODE_RESUME = "resume"
LAUNCH_MODE_REWIND = "rewind"


def launch_lineage(*, resume: bool, resume_from_update: int | None) -> dict[str, Any]:
    """Classify how this invocation relates to earlier ones.

    ``--resume-from-update N`` also sets ``resume``, so the rewind case is
    decided first: a rewind destroys same-run history after N and is never
    reported as a plain resume.
    """
    if resume_from_update is not None:
        mode = LAUNCH_MODE_REWIND
    elif resume:
        mode = LAUNCH_MODE_RESUME
    else:
        mode = LAUNCH_MODE_FRESH
    return {
        "mode": mode,
        "resume_from_update": None if resume_from_update is None else int(resume_from_update),
    }


class LaunchRecordError(RuntimeError):
    """An existing ``launch.json`` cannot be read as a launch history."""


def read_launch_records(run_dir: Path) -> list[dict[str, Any]]:
    """Existing launch records, or an empty list when the file does not exist.

    An unreadable, wrongly shaped or non-contiguously sequenced history is
    NOT an empty history: treating it as one would renumber the next record to
    sequence 1 and overwrite the only copy of how the earlier launches
    happened.  Only the genuine no-file case is empty; everything else fails
    closed here, before any byte is written.
    """
    path = Path(run_dir) / LAUNCH_FILENAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchRecordError(f"{LAUNCH_FILENAME} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaunchRecordError(f"{LAUNCH_FILENAME} must contain a JSON object")
    schema_version = payload.get("schema_version")
    # ``True == 1`` and ``1.0 == 1``; only a built-in int is schema 1.
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) \
            or schema_version != LAUNCH_SCHEMA_VERSION:
        raise LaunchRecordError(
            f"{LAUNCH_FILENAME} schema_version is unsupported: {schema_version!r}"
        )
    launches = payload.get("launches")
    if not isinstance(launches, list):
        raise LaunchRecordError(f"{LAUNCH_FILENAME} carries no launches list")
    for position, entry in enumerate(launches, start=1):
        if not isinstance(entry, dict):
            raise LaunchRecordError(f"{LAUNCH_FILENAME} launch {position} is not an object")
        sequence = entry.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != position:
            raise LaunchRecordError(
                f"{LAUNCH_FILENAME} launch sequence is not contiguous 1..N at position "
                f"{position}: {sequence!r}"
            )
    return list(launches)


def write_launch_record(
    run_dir: Path,
    *,
    arm: str,
    tool: str,
    argv: list[str] | tuple[str, ...],
    device_request: str | None,
    lineage: dict[str, Any],
    git_commit: str | None,
    git_dirty: bool | None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Append this invocation's launch snapshot to ``launch.json``.

    Called by both arms before any evaluation, dataset collection or rollout,
    so a run that dies in its first minute still has a full record of how it
    was started.  The command line is recorded sanitized; the process
    environment is never recorded at all.
    """
    run_dir = Path(run_dir)
    started = now_hkt() if now is None else now
    existing = read_launch_records(run_dir)
    record: dict[str, Any] = {
        "sequence": len(existing) + 1,
        "arm": arm,
        "tool": tool,
        "argv": sanitize_argv(list(argv)),
        "executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "started_hkt": started.isoformat(),
        "started_utc": started.astimezone(dt.timezone.utc).isoformat(),
        "runtime": runtime_snapshot(device_request=device_request),
        "git": {"commit": git_commit, "dirty": git_dirty},
        "lineage": {
            **lineage,
            "previous_launch_sequence": existing[-1]["sequence"] if existing else None,
        },
        "evidence_provenance": {
            "argv": EVIDENCE_OBSERVED,
            "cwd": EVIDENCE_OBSERVED,
            "started_hkt": EVIDENCE_OBSERVED,
            "runtime": EVIDENCE_OBSERVED,
            "git": EVIDENCE_OBSERVED if git_commit is not None else EVIDENCE_UNAVAILABLE,
            "lineage": EVIDENCE_DERIVED,
        },
    }
    atomic_write_json(
        run_dir / LAUNCH_FILENAME,
        {"schema_version": LAUNCH_SCHEMA_VERSION, "launches": [*existing, record]},
    )
    return record


def record_launch(
    run_dir: Path,
    *,
    arm: str,
    tool: str,
    args: Any,
    device: Any,
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict[str, Any]:
    """The single entry point both arms call to write their launch snapshot.

    ``args.argv`` is honoured when present (the CLI sets it) so the recorded
    command line is the real one rather than whatever happens to be in
    ``sys.argv`` for an embedded caller.
    """
    argv = getattr(args, "argv", None)
    return write_launch_record(
        run_dir,
        arm=arm,
        tool=tool,
        argv=list(sys.argv) if argv is None else list(argv),
        device_request=str(device),
        lineage=launch_lineage(
            resume=bool(getattr(args, "resume", False)),
            resume_from_update=getattr(args, "resume_from_update", None),
        ),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


# --------------------------------------------------------------------------- #
# input_manifest.json -- what this run actually read, by path AND by content
# --------------------------------------------------------------------------- #
INPUT_MANIFEST_SCHEMA_VERSION = 1
INPUT_MANIFEST_FILENAME = "input_manifest.json"

# Identity that must exist before any experiment work happens.  A run that
# cannot say WHICH file it read, and prove it by content hash, has no
# reconstructable provenance at all.
REQUIRED_INPUT_FIELDS = (
    "inputs.init_checkpoint.path",
    "inputs.init_checkpoint.file_sha256",
    "inputs.init_checkpoint.state_dict_sha256",
    "inputs.init_checkpoint.architecture",
    "inputs.teacher.path",
    "inputs.teacher.file_sha256",
    "inputs.teacher.state_dict_sha256",
    "inputs.teacher.architecture",
)


def resolved_path(path: Any) -> str | None:
    """The absolute on-disk location of an input, or ``None`` if unresolvable."""
    if path is None:
        return None
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        return None


def init_checkpoint_identity(path: Any, meta: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Identity of the initial Player checkpoint: where it is and what it is.

    ``meta``/``provenance`` come from the arm's already-strict load, so this
    never re-reads or re-validates the checkpoint; it only records the
    resolved path alongside the identity that load already established.
    """
    absolute = resolved_path(path)
    record: dict[str, Any] = {
        "role": "init_checkpoint",
        "path": absolute,
        "path_exists": bool(absolute) and Path(absolute).is_file(),
        "file_sha256": provenance.get("file_sha256"),
        "state_dict_sha256": provenance.get("state_dict_sha256"),
        "architecture": meta.get("architecture"),
        "architecture_version": meta.get("architecture_version"),
        "checkpoint_schema_version": meta.get("schema_version"),
        "production_compatible": meta.get("production_compatible"),
        "experimental": meta.get("experimental"),
        "actions": provenance.get("actions"),
        "observation": provenance.get("observation"),
    }
    return record


def teacher_identity(path: Any) -> dict[str, Any]:
    """Identity of the V1 teacher checkpoint, including its state-dict hash.

    The teacher is a legacy ``{"hidden", "state_dict"}`` artifact rather than
    a packed Player checkpoint.  ``architecture`` is ``player_v1`` only after
    those weights strict-load into a ``PlayerV1`` of the declared hidden
    size; key presence is not identity.  The state-dict hash is recorded only
    for weights that loaded, so a contract never pins a dict this harness
    could not actually run.  An unreadable, malformed, or non-V1 teacher
    yields explicit nulls: the caller fails closed on them.
    """
    from qrokkun_ai.v1.agents.player_v1 import PlayerV1
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256, state_dict_sha256

    absolute = resolved_path(path)
    record: dict[str, Any] = {
        "role": "teacher",
        "path": absolute,
        "path_exists": bool(absolute) and Path(absolute).is_file(),
        "file_sha256": None,
        "state_dict_sha256": None,
        "architecture": None,
        "hidden": None,
    }
    if not record["path_exists"]:
        return record
    try:
        record["file_sha256"] = file_sha256(absolute)
    except OSError:
        return record
    try:
        import torch

        raw = torch.load(absolute, map_location="cpu", weights_only=True)
        hidden = raw["hidden"]
        # ``bool`` is an ``int`` subclass; accepting it would build a hidden
        # size of 1 and could strict-load a coincidentally matching tensor.
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            return record
        state = raw["state_dict"]
        PlayerV1(hidden=hidden).load_state_dict(state, strict=True)
        record["state_dict_sha256"] = state_dict_sha256(state)
        record["hidden"] = hidden
        record["architecture"] = "player_v1"
    except Exception:  # noqa: BLE001 - any teacher that will not load is "no identity"
        pass
    return record


def _lookup(manifest: dict[str, Any], dotted: str) -> Any:
    node: Any = manifest
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _identity_provenance(record: dict[str, Any], required: tuple[str, ...]) -> str:
    return (
        EVIDENCE_COMPUTED
        if all(record.get(field) is not None for field in required)
        else EVIDENCE_UNAVAILABLE
    )


def build_input_manifest(
    *,
    arm: str,
    tool: str,
    init_path: Any,
    init_meta: dict[str, Any],
    init_provenance: dict[str, Any],
    teacher_path: Any,
) -> dict[str, Any]:
    """The full record of what this run read, with an explicit completeness verdict."""
    init = init_checkpoint_identity(init_path, init_meta, init_provenance)
    teacher = teacher_identity(teacher_path)
    init["provenance"] = _identity_provenance(
        init, ("path", "file_sha256", "state_dict_sha256", "architecture")
    )
    teacher["provenance"] = _identity_provenance(
        teacher, ("path", "file_sha256", "state_dict_sha256", "architecture")
    )
    manifest: dict[str, Any] = {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "arm": arm,
        "tool": tool,
        "written_hkt": now_hkt().isoformat(),
        "inputs": {"init_checkpoint": init, "teacher": teacher},
        "dataset": {"status": EVIDENCE_UNAVAILABLE, "provenance": EVIDENCE_UNAVAILABLE},
        "required_fields": list(REQUIRED_INPUT_FIELDS),
    }
    missing = [field for field in REQUIRED_INPUT_FIELDS if _lookup(manifest, field) is None]
    manifest["missing"] = missing
    manifest["complete"] = not missing
    return manifest


def write_input_manifest(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    atomic_write_json(Path(run_dir) / INPUT_MANIFEST_FILENAME, manifest)
    return manifest


_INPUT_IDENTITY_FIELDS = ("file_sha256", "state_dict_sha256", "architecture")


def load_input_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Read an existing input manifest without rewriting it.

    A missing file is "no prior identity". A file that is present but not a
    JSON object fails closed: a resume must not treat a damaged manifest as
    permission to replace it.
    """
    path = Path(run_dir) / INPUT_MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("input_manifest.json is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("input_manifest.json is unreadable")
    return payload


def verify_resumed_input_identity(
    manifest: dict[str, Any], prior: dict[str, Any] | None,
) -> None:
    """The identity rebuilt in memory must agree with the manifest on disk.

    Nulls are not a disagreement: a failed rebuild is reported by the
    completeness verdict, and a field the prior manifest never recorded is
    not evidence that the inputs changed. A present value that differs is.
    """
    if not isinstance(prior, dict):
        return
    new_inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    old_inputs = prior.get("inputs") if isinstance(prior.get("inputs"), dict) else {}
    for role in ("init_checkpoint", "teacher"):
        new = new_inputs.get(role) if isinstance(new_inputs.get(role), dict) else {}
        old = old_inputs.get(role) if isinstance(old_inputs.get(role), dict) else {}
        for field in _INPUT_IDENTITY_FIELDS:
            old_value = old.get(field)
            new_value = new.get(field)
            if old_value is None or new_value is None:
                continue
            if old_value != new_value:
                raise ValueError(
                    f"resumed {role} {field} does not match input_manifest.json"
                )


def prior_dataset_identity(prior: dict[str, Any] | None) -> dict[str, Any] | None:
    """The dataset a previous manifest actually identified, if it did."""
    if not isinstance(prior, dict):
        return None
    dataset = prior.get("dataset")
    if not isinstance(dataset, dict):
        return None
    digest = dataset.get("hash")
    if not isinstance(digest, str) or not digest:
        return None
    return dict(dataset)


def report_dataset_from_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The report's dataset block for a manifest that already knows the hash.

    Provenance stays on the manifest. The report keeps the identity fields
    the collector recorded, so the two hashes are the same value.
    """
    carried = prior_dataset_identity(manifest)
    if carried is None:
        return None
    return {key: value for key, value in carried.items() if key != "provenance"}


def stage_input_manifest(
    run_dir: Path, manifest: dict[str, Any], *, resume: bool,
) -> dict[str, Any]:
    """Record a fresh run immediately; keep a resume's manifest on disk.

    A fresh launch has nothing to preserve, so the failed-closed identity is
    written before the gate and a retry in the same directory still sees it.
    A resume or rewind rebuilds and checks that identity in memory, carries
    the prior dataset when one was recorded, and does not touch
    ``input_manifest.json``. The caller writes the refreshed manifest only
    once the dataset is known again, or when it intentionally finalizes a
    failed report that still carries that dataset.
    """
    if resume:
        prior = load_input_manifest(run_dir)
        verify_resumed_input_identity(manifest, prior)
        carried = prior_dataset_identity(prior)
        if carried is not None:
            manifest["dataset"] = carried
        return manifest
    return write_input_manifest(run_dir, manifest)


def _staged_dataset_hash(manifest: dict[str, Any]) -> str | None:
    """A dataset hash the staged manifest already observed or computed.

    Only those two provenances pin the identity. A hash that was never
    established (``unavailable``, or no hash at all) is not a prior dataset
    for a resume to disagree with.
    """
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        return None
    if dataset.get("provenance") not in (EVIDENCE_OBSERVED, EVIDENCE_COMPUTED):
        return None
    digest = dataset.get("hash")
    if not isinstance(digest, str) or not digest:
        return None
    return digest


def attach_dataset_identity(
    run_dir: Path, manifest: dict[str, Any], dataset: dict[str, Any],
) -> dict[str, Any]:
    """Record the canonical dataset identity once collection has produced it.

    The dataset is recollected rather than read from disk, so its identity is
    only knowable after collection; it is added to the already-written
    manifest instead of being guessed up front.

    On resume or rewind the staged manifest may already carry an observed or
    computed hash. Recollection has to reproduce that hash. A mismatch raises
    before this function writes ``input_manifest.json``, so the caller never
    reaches training or a new checkpoint with a different dataset identity.
    """
    staged = _staged_dataset_hash(manifest)
    collected = dataset.get("hash") if isinstance(dataset, dict) else None
    if staged is not None and collected != staged:
        raise ValueError(
            "recollected dataset hash does not match the staged dataset identity"
        )
    manifest["dataset"] = {**dataset, "provenance": EVIDENCE_COMPUTED}
    return write_input_manifest(run_dir, manifest)


# --------------------------------------------------------------------------- #
# checkpoint kinds -- every pack says what it is, nothing is inferred
# --------------------------------------------------------------------------- #
# A reader must never have to deduce "this was a full diagnostic snapshot"
# from the ABSENCE of a ``checkpoint_kind`` key, which is what earlier runs
# forced.  Every checkpoint and every recovery pack carries its kind.
CHECKPOINT_KIND_INITIAL_SNAPSHOT = "initial_snapshot"
CHECKPOINT_KIND_DIAGNOSTIC_FULL_SNAPSHOT = "diagnostic_full_snapshot"
CHECKPOINT_KIND_PERIODIC_MODEL_ONLY = "periodic_model_only"
CHECKPOINT_KIND_TERMINAL_FULL_SNAPSHOT = "terminal_full_snapshot"
CHECKPOINT_KIND_FINAL_ALIAS = "final_alias"
CHECKPOINT_KIND_RECOVERY_CURRENT = "recovery_current"
CHECKPOINT_KIND_RECOVERY_ARCHIVE = "recovery_archive"

CHECKPOINT_KINDS = (
    CHECKPOINT_KIND_INITIAL_SNAPSHOT,
    CHECKPOINT_KIND_DIAGNOSTIC_FULL_SNAPSHOT,
    CHECKPOINT_KIND_PERIODIC_MODEL_ONLY,
    CHECKPOINT_KIND_TERMINAL_FULL_SNAPSHOT,
    CHECKPOINT_KIND_FINAL_ALIAS,
    CHECKPOINT_KIND_RECOVERY_CURRENT,
    CHECKPOINT_KIND_RECOVERY_ARCHIVE,
)


def require_checkpoint_kind(kind: str) -> str:
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"unknown checkpoint_kind {kind!r}; expected one of {CHECKPOINT_KINDS}")
    return kind


def scheduled_snapshot_kind(update: int) -> str:
    """The kind of a snapshot taken by the in-loop schedule."""
    return (
        CHECKPOINT_KIND_INITIAL_SNAPSHOT
        if int(update) == 0
        else CHECKPOINT_KIND_DIAGNOSTIC_FULL_SNAPSHOT
    )


# --------------------------------------------------------------------------- #
# calibration.json -- the auxiliary alpha calibration, or an explicit "no such
# thing in this arm"
# --------------------------------------------------------------------------- #
CALIBRATION_FILENAME = "calibration.json"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_CALIBRATED = "calibrated"


def calibration_record(
    *, arm: str, calibration: dict[str, Any] | None = None, reason: str | None = None,
) -> dict[str, Any]:
    """The calibration evidence for one arm.

    The control has no auxiliary term and therefore no alpha.  It records
    ``not_applicable`` explicitly instead of omitting the field, so a reader
    never has to infer the difference between "no auxiliary term" and "the
    calibration was not written down".
    """
    if calibration is None:
        return {
            "arm": arm,
            "status": STATUS_NOT_APPLICABLE,
            "calibration": None,
            "reason": reason or "this arm has no BC-retention auxiliary term to calibrate",
            "provenance": EVIDENCE_OBSERVED,
        }
    # A present alpha that is not a finite number is not a calibration.  The
    # record keeps no usable payload, so a NaN or a boolean cannot be read
    # back as the alpha that was fitted.
    if (
        isinstance(calibration, dict)
        and "alpha" in calibration
        and not _finite_alpha(calibration.get("alpha"))
    ):
        return {
            "arm": arm,
            "status": EVIDENCE_UNAVAILABLE,
            "calibration": None,
            "reason": reason or "alpha is not a finite number",
            "provenance": EVIDENCE_UNAVAILABLE,
        }
    return {
        "arm": arm,
        "status": STATUS_CALIBRATED,
        "calibration": calibration,
        "reason": None,
        "provenance": EVIDENCE_COMPUTED,
    }


def _finite_alpha(value: Any) -> bool:
    """True for a real finite number. Booleans are ints, and they are not alphas."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def write_calibration_record(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    atomic_write_json(Path(run_dir) / CALIBRATION_FILENAME, record)
    return record


_CALIBRATION_RECORD_KEYS = frozenset({"arm", "status", "calibration", "reason", "provenance"})


def _require_calibration_record(payload: Any, *, arm: str) -> None:
    """Reject anything that is not this arm's shared calibration wrapper.

    A missing file is handled by the caller. A present file that is
    unreadable, the wrong shape, or another arm's record is a failure, not
    evidence that no calibration exists.
    """
    if not isinstance(payload, dict) or set(payload) != _CALIBRATION_RECORD_KEYS:
        raise ValueError("calibration.json is not a calibration record")
    if payload.get("arm") != arm:
        raise ValueError("calibration.json arm does not match")
    status = payload.get("status")
    calibration = payload.get("calibration")
    reason = payload.get("reason")
    provenance = payload.get("provenance")
    if status == STATUS_NOT_APPLICABLE:
        valid = (
            calibration is None
            and isinstance(reason, str)
            and bool(reason)
            and provenance == EVIDENCE_OBSERVED
        )
    elif status == EVIDENCE_UNAVAILABLE:
        valid = (
            calibration is None
            and isinstance(reason, str)
            and bool(reason)
            and provenance == EVIDENCE_UNAVAILABLE
        )
    elif status == STATUS_CALIBRATED:
        valid = (
            isinstance(calibration, dict)
            and _finite_alpha(calibration.get("alpha"))
            and reason is None
            and provenance == EVIDENCE_COMPUTED
        )
    else:
        valid = False
    if not valid:
        raise ValueError("calibration.json is not a calibration record")


def load_retained_calibration_record(
    run_dir: Path, *, arm: str,
) -> dict[str, Any] | None:
    """Read ``calibration.json`` without rewriting it.

    No file means this launch has no retained calibration. A file that is
    present must be this arm's shared calibration-record wrapper; otherwise
    the call fails and the bytes on disk stay the caller's to keep.
    """
    path = Path(run_dir) / CALIBRATION_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("calibration.json is unreadable") from exc
    _require_calibration_record(payload, arm=arm)
    return payload


def rollout_seed_record(seeds: list[int] | tuple[int, ...]) -> dict[str, Any]:
    """The per-update rollout seed evidence both arms append to their journal.

    Callers pass the seeds the collected rollouts actually carry, not the
    seeds the schedule says they should carry, so a divergence between the
    two is visible in the journal instead of being reconstructable only under
    the assumption that nothing diverged.
    """
    values = [int(seed) for seed in seeds]
    return {
        "rollout_seeds": values,
        "rollout_seed_window": {
            "start": values[0] if values else None,
            "end_inclusive": values[-1] if values else None,
            "count": len(values),
        },
    }


def contract_teacher_inputs(identity: dict[str, Any]) -> dict[str, Any]:
    """The teacher identity fields bound into the immutable run contract.

    Content identity only: the resolved path is deliberately NOT part of the
    immutable contract, so a run stays resumable from a different mount while
    the bytes it trained against remain pinned.
    """
    return {
        "file_sha256": identity.get("file_sha256"),
        "state_dict_sha256": identity.get("state_dict_sha256"),
        "architecture": identity.get("architecture"),
    }


# --------------------------------------------------------------------------- #
# artifact_manifest.json -- everything the run left on disk, by role and hash
# --------------------------------------------------------------------------- #
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"

NOT_COMPUTED = "not_computed"

# A long run writes hundreds of full recovery archives totalling tens of GiB.
# Hashing them all on every finalization would read the whole run back from
# disk, so by default they are inventoried exactly -- real count, real sizes,
# real update indices -- and explicitly marked as unhashed rather than being
# silently dropped from the manifest.
DEFAULT_MAX_HASH_BYTES = 512 * 1024 * 1024

_UPDATE_NAME_RE = re.compile(r"(?:^|_)update_(?P<update>\d+)$")

# The progress journal each arm retains.  Evidence measurement and the
# artifact manifest both use these names, so a report cannot claim a journal
# the manifest did not hash.
PROGRESS_JOURNAL_FILENAMES = {
    "control": "ppo_updates.jsonl",
    "aux": "ppo_aux_updates.jsonl",
}


def artifact_role(relative_path: str) -> str:
    """Classify a run artifact by its filename, using one shared vocabulary."""
    path = Path(relative_path)
    name = path.name
    if path.parent.name == "recovery_archives":
        return CHECKPOINT_KIND_RECOVERY_ARCHIVE
    known = {
        "report.json": "report",
        LAUNCH_FILENAME: "launch",
        INPUT_MANIFEST_FILENAME: "input_manifest",
        CALIBRATION_FILENAME: "calibration",
        "analysis.json": "pair_analysis",
        "run.json": "run_contract",
        "status.jsonl": "status_journal",
        "stop_budget_amendments.jsonl": "stop_budget_amendments",
        "recovery.pt": CHECKPOINT_KIND_RECOVERY_CURRENT,
        "ppo_final.pt": "final_checkpoint",
        "ppo_aux_final.pt": "final_checkpoint",
        **{name: "progress_journal" for name in PROGRESS_JOURNAL_FILENAMES.values()},
    }
    if name in known:
        return known[name]
    if _UPDATE_NAME_RE.search(path.stem) and name.endswith(".pt"):
        return "update_checkpoint"
    return "other"


def artifact_update(relative_path: str) -> int | None:
    """The update index an artifact belongs to, or ``None`` when it has none."""
    match = _UPDATE_NAME_RE.search(Path(relative_path).stem)
    return int(match.group("update")) if match else None


def build_artifact_manifest(
    run_dir: Path,
    *,
    arm: str,
    hash_recovery_archives: bool = False,
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
) -> dict[str, Any]:
    """Inventory every file in ``run_dir`` with role, update, size and hash.

    Each entry gets a real SHA-256 or an explicit ``not_computed`` status with
    the reason it was skipped -- never a missing field that leaves a reader
    guessing whether the hash was unavailable or simply never attempted.
    """
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    run_dir = Path(run_dir)
    artifacts: list[dict[str, Any]] = []
    archive_updates: list[int] = []
    archive_bytes = 0

    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative == ARTIFACT_MANIFEST_FILENAME or path.name.startswith("."):
            continue
        role = artifact_role(relative)
        size = path.stat().st_size
        update = artifact_update(relative)
        is_archive = role == CHECKPOINT_KIND_RECOVERY_ARCHIVE
        if is_archive:
            archive_bytes += size
            if update is not None:
                archive_updates.append(update)

        reason: str | None = None
        if is_archive and not hash_recovery_archives:
            reason = "recovery_archive_hashing_disabled"
        elif size > max_hash_bytes:
            reason = "exceeds_max_hash_bytes"

        if reason is None:
            try:
                digest: str | None = file_sha256(path)
                status = EVIDENCE_COMPUTED
            except OSError:
                digest, status, reason = None, NOT_COMPUTED, "unreadable"
        else:
            digest, status = None, NOT_COMPUTED

        artifacts.append(
            {
                "path": relative,
                "role": role,
                "update": update,
                "size_bytes": size,
                "sha256": digest,
                "sha256_status": status,
                "not_computed_reason": reason,
            }
        )

    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "arm": arm,
        "written_hkt": now_hkt().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(entry["size_bytes"] for entry in artifacts),
        "hash_policy": {
            "hash_recovery_archives": bool(hash_recovery_archives),
            "max_hash_bytes": int(max_hash_bytes),
        },
        "recovery_archives": {
            "count": len([a for a in artifacts if a["role"] == CHECKPOINT_KIND_RECOVERY_ARCHIVE]),
            "total_size_bytes": archive_bytes,
            "updates": sorted(archive_updates),
            "hashed": bool(hash_recovery_archives),
        },
        "artifacts": artifacts,
        "self_excluded": ARTIFACT_MANIFEST_FILENAME,
        "provenance": EVIDENCE_COMPUTED,
    }


def write_artifact_manifest(
    run_dir: Path,
    *,
    arm: str,
    hash_recovery_archives: bool = False,
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
) -> dict[str, Any]:
    manifest = build_artifact_manifest(
        run_dir, arm=arm, hash_recovery_archives=hash_recovery_archives,
        max_hash_bytes=max_hash_bytes,
    )
    atomic_write_json(Path(run_dir) / ARTIFACT_MANIFEST_FILENAME, manifest)
    return manifest


def finalize_run_reporting(
    run_dir: Path,
    *,
    arm: str,
    report: dict[str, Any],
    write_report: Any,
    hash_recovery_archives: bool = False,
    max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES,
) -> dict[str, Any]:
    """Finalize one arm's reporting: write ``report.json``, then inventory the run.

    Ordering matters and is deliberate.  The report carries only a REFERENCE
    to the artifact manifest (its filename and hash policy), never the
    manifest's counts or its own hash; the manifest is then written last so
    it can hash the final ``report.json``.  Writing them the other way round
    would make one of the two hashes stale the moment it was recorded.
    """
    report["artifact_manifest"] = {
        "path": ARTIFACT_MANIFEST_FILENAME,
        "hash_policy": {
            "hash_recovery_archives": bool(hash_recovery_archives),
            "max_hash_bytes": int(max_hash_bytes),
        },
        "provenance": EVIDENCE_COMPUTED,
    }
    report["evidence_provenance"] = report_evidence_provenance(
        report, run_dir=run_dir, arm=arm,
    )
    report["report_completeness"] = evaluate_report_completeness(report, arm=arm)
    write_report(run_dir, report)
    return write_artifact_manifest(
        run_dir, arm=arm, hash_recovery_archives=hash_recovery_archives,
        max_hash_bytes=max_hash_bytes,
    )


# --------------------------------------------------------------------------- #
# evidence provenance + report completeness
# --------------------------------------------------------------------------- #
def evidence(value: Any, status: str, *, note: str | None = None) -> dict[str, Any]:
    """Tag one reported value with how it was obtained.

    ``unavailable`` must carry a null value: a record that says "unavailable"
    while still holding a number lets a guess masquerade as a measurement.
    """
    if status not in EVIDENCE_STATUSES:
        raise ValueError(
            f"evidence provenance must be one of {EVIDENCE_STATUSES}; got {status!r}"
        )
    if status == EVIDENCE_UNAVAILABLE and value is not None:
        raise ValueError("an unavailable evidence record must carry a null value")
    return {"value": value, "provenance": status, "note": note}


REPORT_COMPLETENESS_SCHEMA_VERSION = 1

# The provenance a finished report must carry to be called complete.  These
# are dotted paths into the report itself, which embeds its own launch record,
# input manifest and artifact manifest reference.
REQUIRED_REPORT_FIELDS = (
    "status",
    # The pre-registered gate and the retention outcome.  A report that lacks
    # either verdict is not a complete account of the experiment, including a
    # failed-closed run that never produced them.
    "initial_gate.gate_pass",
    "retention.retention_pass",
    "provenance",
    "knobs",
    "evidence_provenance",
    "alpha_calibration",
    "stop_budget.effective_max_updates",
    "stop_budget.effective_end_time_hkt",
    "launch.argv",
    "launch.cwd",
    "launch.started_hkt",
    "launch.lineage.mode",
    "launch.git.commit",
    "launch.runtime.host",
    "launch.runtime.platform.system",
    "launch.runtime.python_version",
    "launch.runtime.numpy_version",
    "launch.runtime.torch_version",
    "launch.runtime.device_request",
    "launch.runtime.cuda.available",
    "launch.runtime.gpu.status",
    "input_manifest.inputs.init_checkpoint.path",
    "input_manifest.inputs.init_checkpoint.file_sha256",
    "input_manifest.inputs.init_checkpoint.state_dict_sha256",
    "input_manifest.inputs.init_checkpoint.architecture",
    "input_manifest.inputs.teacher.path",
    "input_manifest.inputs.teacher.file_sha256",
    "input_manifest.inputs.teacher.state_dict_sha256",
    "input_manifest.inputs.teacher.architecture",
    "input_manifest.dataset.hash",
    "artifact_manifest.path",
)


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _journal_row_is_observed(row: Any) -> bool:
    """A retained update row: exact-int update >= 1 and non-empty exact-int seeds."""
    if not isinstance(row, dict):
        return False
    update = row.get("update")
    seeds = row.get("rollout_seeds")
    if not _exact_int(update) or update < 1:
        return False
    if not isinstance(seeds, list) or not seeds:
        return False
    return all(_exact_int(seed) for seed in seeds)


def measure_progress_journal_evidence(path: Path) -> str:
    """Observe ``path`` itself. Malformed or empty journals are not a seed window.

    A file is observed when it contains at least one valid update row.  Lines
    that are not that row are ignored; they are never rewritten into one.
    An unreadable file is unavailable.
    """
    try:
        text = Path(path).read_text()
    except (OSError, UnicodeError):
        return EVIDENCE_UNAVAILABLE
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _journal_row_is_observed(row):
            return EVIDENCE_OBSERVED
    return EVIDENCE_UNAVAILABLE


def progress_journal_filename(arm: str) -> str | None:
    """The journal filename :func:`build_artifact_manifest` hashes for ``arm``."""
    return PROGRESS_JOURNAL_FILENAMES.get(arm)


def _progress_evidence(
    *,
    run_dir: Path | None,
    arm: str | None,
    progress_journal: Path | None,
) -> str:
    """Measure the retained journal from ``run_dir`` + ``arm`` or an explicit path.

    With neither, there is no journal to read.  The arm result's
    ``completed_updates`` counter is not a substitute for the file.
    """
    if progress_journal is not None:
        return measure_progress_journal_evidence(progress_journal)
    if run_dir is not None and arm is not None:
        name = progress_journal_filename(arm)
        if name is None:
            return EVIDENCE_UNAVAILABLE
        return measure_progress_journal_evidence(Path(run_dir) / name)
    return EVIDENCE_UNAVAILABLE


def _reported_alpha(record: dict[str, Any]) -> Any:
    calibration = record.get("calibration")
    if not isinstance(calibration, dict):
        return None
    return calibration.get("alpha")


def report_evidence_provenance(
    report: dict[str, Any],
    *,
    run_dir: Path | None = None,
    arm: str | None = None,
    progress_journal: Path | None = None,
) -> dict[str, str]:
    """How each family of reported evidence in ``report`` was obtained.

    Progress and the rollout seed window are the same retained journal: pass
    ``run_dir`` and ``arm`` (the file the artifact manifest hashes) or an
    explicit ``progress_journal`` path that was already measured.  A missing
    journal is unavailable.  Alpha is computed only for a finite numeric
    value; a non-finite, non-numeric, or boolean alpha is unavailable.
    """
    # Both arms report through :func:`calibration_record`, so the alpha lives
    # under ``calibration``.  A wrapper that claims "calibrated" while carrying
    # no finite alpha is not evidence of a calibration and stays unavailable.
    record = report.get("alpha_calibration") or {}
    if not isinstance(record, dict):
        record = {}
    if record.get("status") == STATUS_NOT_APPLICABLE:
        calibration_status = STATUS_NOT_APPLICABLE
    elif _finite_alpha(_reported_alpha(record)):
        calibration_status = EVIDENCE_COMPUTED
    else:
        calibration_status = EVIDENCE_UNAVAILABLE
    manifest = report.get("input_manifest") or {}
    progress = _progress_evidence(
        run_dir=run_dir, arm=arm, progress_journal=progress_journal,
    )
    return {
        # Read straight off the live process at launch.
        "launch": EVIDENCE_OBSERVED if report.get("launch") else EVIDENCE_UNAVAILABLE,
        # Content hashes over the real input files.
        "input_manifest": EVIDENCE_COMPUTED if manifest.get("complete") else EVIDENCE_UNAVAILABLE,
        "dataset": EVIDENCE_COMPUTED if report.get("dataset") else EVIDENCE_UNAVAILABLE,
        "artifact_manifest": EVIDENCE_COMPUTED,
        "progress_journal": progress,
        "rollout_seed_window": progress,
        "alpha_calibration": calibration_status,
        "stop_budget": EVIDENCE_OBSERVED,
        # The retention verdict is a function of the recorded snapshots.
        "retention": EVIDENCE_DERIVED if report.get("retention") else EVIDENCE_UNAVAILABLE,
    }


def evaluate_report_completeness(report: dict[str, Any], *, arm: str) -> dict[str, Any]:
    """Machine-readable verdict on whether the REPORT is complete.

    This is a statement about reporting, never about training.  A run whose
    model finished cleanly but whose provenance has a hole stays a completed
    run; it simply must not be described as a complete report.
    """
    results = []
    for field in REQUIRED_REPORT_FIELDS:
        value = _lookup(report, field)
        results.append(
            {
                "field": field,
                "present": value is not None,
                "provenance": EVIDENCE_OBSERVED if value is not None else EVIDENCE_UNAVAILABLE,
            }
        )
    missing = sorted(result["field"] for result in results if not result["present"])
    return {
        "schema_version": REPORT_COMPLETENESS_SCHEMA_VERSION,
        "arm": arm,
        "complete": not missing,
        "results": results,
        "missing": missing,
        "required_count": len(REQUIRED_REPORT_FIELDS),
        "present_count": len(REQUIRED_REPORT_FIELDS) - len(missing),
        # An incomplete report never retracts a finished training run.
        "model_completion_affected": False,
    }
