"""Strict, architecture-keyed checkpoint packing/loading for production Players.

Only the ranked top-k architecture (:mod:`qrokkun_env.agents.player_ranked_topk`)
is production-compatible here. Loading is *strict by construction*:

* the checkpoint must declare a known ``architecture`` -- legacy PlayerV4 /
  PlayerV1 checkpoints (which carry no architecture metadata) are refused, so
  a ranked top-k checkpoint can never be loaded accidentally as PlayerV4 and
  vice versa;
* the schema version, action list, observation dimensions, and the recorded
  state-dict SHA-256 must all match the running code exactly;
* weights are restored with ``load_state_dict(..., strict=True)``.

Nothing in this module mutates or re-interprets PlayerV4 checkpoints.
"""

from __future__ import annotations

import hashlib
import io
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4
from qrokkun_env.agents.player_ranked_topk import ARCHITECTURE as RANKED_TOP_K_ARCHITECTURE
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_env.env import ACTIONS

CKPT_SCHEMA_VERSION = 1
CKPT_ROLE_PLAYER = "player"

# Architectures this loader is allowed to build. PlayerV4 is intentionally NOT
# here: its checkpoints keep their own legacy loading path untouched.
SUPPORTED_ARCHITECTURES: dict[str, type[nn.Module]] = {
    RANKED_TOP_K_ARCHITECTURE: PlayerRankedTopK,
}


class CheckpointError(Exception):
    """Base error for strict checkpoint validation."""


class CheckpointArchitectureError(CheckpointError):
    """Checkpoint declares a missing/unknown/unexpected architecture."""


class CheckpointSchemaError(CheckpointError):
    """Checkpoint architecture is known but its schema/contents are invalid."""


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Stable content hash of a state dict (order-independent, value-exact)."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        h.update(key.encode("utf-8"))
        tensor = value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit(repo_root: Path | str | None = None) -> str | None:
    """Best-effort ``git rev-parse HEAD`` for checkpoint provenance."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def build_player_from_metadata(meta: Mapping[str, Any]) -> nn.Module:
    """Construct a Player network from architecture metadata only."""
    arch = meta.get("architecture")
    if arch not in SUPPORTED_ARCHITECTURES:
        raise CheckpointArchitectureError(
            f"unsupported/missing architecture {arch!r}; supported: {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    top_k = meta.get("top_k")
    hidden = meta.get("hidden")
    if not isinstance(top_k, int) or not isinstance(hidden, int) or top_k <= 0 or hidden <= 0:
        raise CheckpointSchemaError(f"invalid top_k/hidden in metadata: top_k={top_k!r} hidden={hidden!r}")
    return SUPPORTED_ARCHITECTURES[arch](top_k=top_k, hidden=hidden)


def pack_player_checkpoint(
    net: PlayerRankedTopK,
    *,
    state_dict: Mapping[str, Any] | None = None,
    source_commit: str | None = None,
    source_tool: str | None = None,
    extra: Mapping[str, Any] | None = None,
    experimental: bool = False,
    production_compatible: bool = True,
) -> dict[str, Any]:
    """Pack a checkpoint for the ranked top-k Player.

    By default the checkpoint is marked production-compatible
    (``experimental=False``/``production_compatible=True``), matching every
    existing caller. Callers that produce non-production artifacts (e.g. a
    PPO retention control snapshot) must pass ``experimental=True,
    production_compatible=False`` explicitly -- this never changes the
    strict loading/validation contract, only the declared provenance flags.
    """
    if not isinstance(net, PlayerRankedTopK):
        raise CheckpointArchitectureError(f"pack_player_checkpoint only supports PlayerRankedTopK, got {type(net)!r}")
    net_device = next(net.parameters()).device
    sd = {k: v.detach().cpu().clone() for k, v in (state_dict if state_dict is not None else net.state_dict()).items()}
    meta = net.architecture_metadata()
    ckpt: dict[str, Any] = {
        "schema_version": CKPT_SCHEMA_VERSION,
        "ckpt_role": CKPT_ROLE_PLAYER,
        "architecture": meta["architecture"],
        "architecture_version": meta["architecture_version"],
        "production_compatible": bool(production_compatible),
        "experimental": bool(experimental),
        "top_k": meta["top_k"],
        "hidden": meta["hidden"],
        "actions": list(ACTIONS),
        "n_actions": len(ACTIONS),
        "observation": {
            "player_feat": PLAYER_FEAT_V4,
            "bullet_feat": BULLET_FEAT_V4,
            "max_bullets": MAX_BULLETS_V4,
            "in_dim": meta["in_dim"],
        },
        "param_count": meta["param_count"],
        "state_dict": sd,
        "state_dict_sha256": state_dict_sha256(sd),
        "source": {
            "commit": source_commit if source_commit is not None else current_git_commit(),
            "tool": source_tool,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "device": str(net_device),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    if extra:
        ckpt["extra"] = dict(extra)
    return ckpt


def save_player_checkpoint(
    net: PlayerRankedTopK,
    path: Path | str,
    *,
    state_dict: Mapping[str, Any] | None = None,
    source_commit: str | None = None,
    source_tool: str | None = None,
    extra: Mapping[str, Any] | None = None,
    experimental: bool = False,
    production_compatible: bool = True,
) -> dict[str, Any]:
    """Pack and write a checkpoint; returns the packed dict (without weights copy)."""
    ckpt = pack_player_checkpoint(
        net,
        state_dict=state_dict,
        source_commit=source_commit,
        source_tool=source_tool,
        extra=extra,
        experimental=experimental,
        production_compatible=production_compatible,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return ckpt


def validate_player_checkpoint(ckpt: Any, *, expected_architecture: str | None = None) -> dict[str, Any]:
    """Strictly validate checkpoint metadata; returns the metadata (no weights)."""
    if not isinstance(ckpt, Mapping):
        raise CheckpointArchitectureError(f"checkpoint is not a mapping: {type(ckpt)!r}")
    arch = ckpt.get("architecture")
    if arch not in SUPPORTED_ARCHITECTURES:
        raise CheckpointArchitectureError(
            f"refusing to load checkpoint with architecture {arch!r}; "
            f"supported: {sorted(SUPPORTED_ARCHITECTURES)} (legacy PlayerV1/PlayerV4 checkpoints are never accepted here)"
        )
    if expected_architecture is not None and arch != expected_architecture:
        raise CheckpointArchitectureError(
            f"architecture mismatch: checkpoint is {arch!r}, caller expected {expected_architecture!r}"
        )
    if ckpt.get("schema_version") != CKPT_SCHEMA_VERSION:
        raise CheckpointSchemaError(
            f"schema_version {ckpt.get('schema_version')!r} != supported {CKPT_SCHEMA_VERSION}"
        )
    if ckpt.get("ckpt_role") != CKPT_ROLE_PLAYER:
        raise CheckpointSchemaError(f"ckpt_role {ckpt.get('ckpt_role')!r} != {CKPT_ROLE_PLAYER!r}")
    if list(ckpt.get("actions") or []) != list(ACTIONS):
        raise CheckpointSchemaError(f"action schema mismatch: {ckpt.get('actions')!r} != {list(ACTIONS)}")
    if ckpt.get("n_actions") != len(ACTIONS):
        raise CheckpointSchemaError(f"n_actions {ckpt.get('n_actions')!r} != {len(ACTIONS)}")
    obs = ckpt.get("observation")
    if not isinstance(obs, Mapping):
        raise CheckpointSchemaError("checkpoint is missing an 'observation' mapping")
    expected_obs = {
        "player_feat": PLAYER_FEAT_V4,
        "bullet_feat": BULLET_FEAT_V4,
        "max_bullets": MAX_BULLETS_V4,
    }
    for key, want in expected_obs.items():
        if obs.get(key) != want:
            raise CheckpointSchemaError(f"observation.{key} {obs.get(key)!r} != {want}")
    top_k, hidden = ckpt.get("top_k"), ckpt.get("hidden")
    if not isinstance(top_k, int) or not isinstance(hidden, int) or top_k <= 0 or hidden <= 0:
        raise CheckpointSchemaError(f"invalid top_k/hidden: top_k={top_k!r} hidden={hidden!r}")
    expected_in_dim = PLAYER_FEAT_V4 + top_k * BULLET_FEAT_V4 + top_k
    if obs.get("in_dim") != expected_in_dim:
        raise CheckpointSchemaError(f"observation.in_dim {obs.get('in_dim')!r} != {expected_in_dim}")
    sd = ckpt.get("state_dict")
    if not isinstance(sd, Mapping) or not sd:
        raise CheckpointSchemaError("checkpoint is missing a non-empty 'state_dict'")
    recorded_hash = ckpt.get("state_dict_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise CheckpointSchemaError(f"invalid state_dict_sha256: {recorded_hash!r}")
    actual_hash = state_dict_sha256(sd)
    if actual_hash != recorded_hash:
        raise CheckpointSchemaError(f"state_dict hash mismatch: {actual_hash} != recorded {recorded_hash}")
    return {k: v for k, v in ckpt.items() if k != "state_dict"}


def load_player_checkpoint(
    path_or_ckpt: Path | str | Mapping[str, Any],
    device: torch.device | str = "cpu",
    *,
    expected_architecture: str | None = None,
    eval_mode: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load a production Player checkpoint keyed by its architecture."""
    if isinstance(path_or_ckpt, (str, Path, io.IOBase)):
        ckpt = torch.load(path_or_ckpt, map_location="cpu", weights_only=False)
    else:
        ckpt = path_or_ckpt
    meta = validate_player_checkpoint(ckpt, expected_architecture=expected_architecture)
    net = build_player_from_metadata(meta)
    try:
        net.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:  # shape/key mismatch with the declared metadata
        raise CheckpointSchemaError(f"state_dict does not match declared architecture metadata: {exc}") from exc
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if meta.get("param_count") != param_count:
        raise CheckpointSchemaError(f"param_count {meta.get('param_count')!r} != rebuilt {param_count}")
    net.to(device)
    if eval_mode:
        net.eval()
    return net, meta


def load_ranked_top_k_checkpoint(
    path_or_ckpt: Path | str | Mapping[str, Any],
    device: torch.device | str = "cpu",
    *,
    eval_mode: bool = True,
) -> tuple[PlayerRankedTopK, dict[str, Any]]:
    """Load a checkpoint that MUST be the ranked top-k architecture."""
    net, meta = load_player_checkpoint(
        path_or_ckpt,
        device,
        expected_architecture=RANKED_TOP_K_ARCHITECTURE,
        eval_mode=eval_mode,
    )
    assert isinstance(net, PlayerRankedTopK)
    return net, meta
