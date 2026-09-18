#!/usr/bin/env python3
"""v4.6 checkpoint paths + separate Player/Spawner selection objectives.

Selection (do NOT use new×new_det_det as sole selector):
  - best_player: primarily newP×scripted (prefer det_stoch key if present)
  - best_spawner: primarily flee×newS_det_stoch
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

# Preferred keys first. new×new* intentionally absent.
PLAYER_SELECTION_KEYS: tuple[str, ...] = (
    "newP_vs_scripted_det_stoch",
    "newP|scripted_det_stoch",
    "newP_vs_scripted",
    "newP|scripted",
)

SPAWNER_SELECTION_KEYS: tuple[str, ...] = (
    "flee_vs_newS_det_stoch",
    "flee|newS_det_stoch",
)

# Documented forbidden sole selector for tests / reviewers.
FORBIDDEN_SOLE_SELECTOR_KEYS: frozenset[str] = frozenset(
    {
        "new_vs_new_det_det",
        "new|new",
        "new_vs_new",
        "new|new_det_det",
    }
)


def _first_present(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, float]:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return k, float(metrics[k])
    raise KeyError(f"none of {keys} present in metrics keys={list(metrics)}")


def player_selection_score(metrics: Mapping[str, Any]) -> float:
    """Score for best_player. Ignores new×new even if huge."""
    _k, v = _first_present(metrics, PLAYER_SELECTION_KEYS)
    return v


def spawner_selection_score(metrics: Mapping[str, Any]) -> float:
    """Score for best_spawner (lower survival vs flee is better? No — we track
    the flee survival time as eval signal; historically status stores the mean
    survival of flee vs newS. Lower = stronger spawner.

    Training today maximizes nothing on this key for the spawner net directly;
    best ckpt historically stored eval_flee_vs_newS and treated *improvement*
    as whatever the loop compared. Spec: key primarily on flee×newS_det_stoch.

    Convention in this codebase: we keep the raw flee survival seconds and treat
    a *decrease* as better for the spawner (stronger attacker). CheckpointManager
    uses minimize=True for spawner.
    """
    _k, v = _first_present(metrics, SPAWNER_SELECTION_KEYS)
    return v


def player_selection_key(metrics: Mapping[str, Any]) -> str:
    k, _ = _first_present(metrics, PLAYER_SELECTION_KEYS)
    return k


def spawner_selection_key(metrics: Mapping[str, Any]) -> str:
    k, _ = _first_present(metrics, SPAWNER_SELECTION_KEYS)
    return k


def default_best_path(latest: Path) -> Path:
    return latest.with_name(f"{latest.stem}_best{latest.suffix}")


def pack_player_ckpt(
    net,
    *,
    d_model: int,
    hidden: int,
    update: int,
    eval_metrics: Mapping[str, Any] | None = None,
    selection_score_value: float | None = None,
    selection_key: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_dict": net.state_dict(),
        "d_model": d_model,
        "hidden": hidden,
        "algo": "both-v4-player",
        "update": update,
        "ckpt_role": "player",
    }
    if eval_metrics:
        payload["eval"] = dict(eval_metrics)
        if "newP_vs_scripted" in eval_metrics:
            payload["eval_vs_scripted"] = eval_metrics["newP_vs_scripted"]
        elif "newP|scripted" in eval_metrics:
            payload["eval_vs_scripted"] = eval_metrics["newP|scripted"]
    if selection_score_value is not None:
        payload["selection_score"] = selection_score_value
    if selection_key is not None:
        payload["selection_key"] = selection_key
    return payload


def pack_spawner_ckpt(
    net,
    *,
    d_model: int,
    hidden: int,
    update: int,
    aim_scale: float,
    eval_metrics: Mapping[str, Any] | None = None,
    selection_score_value: float | None = None,
    selection_key: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_dict": net.state_dict(),
        "d_model": d_model,
        "hidden": hidden,
        "algo": "both-v4-spawner",
        "aim_scale": aim_scale,
        "update": update,
        "ckpt_role": "spawner",
    }
    if eval_metrics:
        payload["eval"] = dict(eval_metrics)
        for k in ("flee_vs_newS_det_stoch", "flee|newS_det_stoch", "flee_vs_newS"):
            if k in eval_metrics:
                payload["eval_flee_vs_newS"] = eval_metrics[k]
                break
    if selection_score_value is not None:
        payload["selection_score"] = selection_score_value
    if selection_key is not None:
        payload["selection_key"] = selection_key
    return payload


@dataclass
class CheckpointManager:
    """latest always overwritten; best_player / best_spawner on metric improve."""

    latest_player: Path
    latest_spawner: Path
    best_player: Path
    best_spawner: Path
    snapshot_dir: Path | None = None
    snapshot_every: int = 0  # 0 = disabled
    best_player_score: float = field(default=-math.inf)
    best_spawner_score: float = field(default=math.inf)  # minimize flee survival
    # Spawner: lower flee×newS is better (stronger S). Player: higher newP×scripted.

    def __post_init__(self) -> None:
        self.latest_player.parent.mkdir(parents=True, exist_ok=True)
        self.latest_spawner.parent.mkdir(parents=True, exist_ok=True)
        self.best_player.parent.mkdir(parents=True, exist_ok=True)
        self.best_spawner.parent.mkdir(parents=True, exist_ok=True)
        if self.snapshot_dir is not None and self.snapshot_every > 0:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_latest(self, player_payload: dict, spawner_payload: dict) -> None:
        torch.save(player_payload, self.latest_player)
        torch.save(spawner_payload, self.latest_spawner)

    def maybe_save_best_player(self, score: float, payload: dict) -> bool:
        """Higher score wins (newP×scripted survival)."""
        if score > self.best_player_score:
            self.best_player_score = score
            out = {**payload, "ckpt_kind": "best_player", "selection_score": score}
            torch.save(out, self.best_player)
            return True
        return False

    def maybe_save_best_spawner(self, score: float, payload: dict) -> bool:
        """Lower score wins (flee×newS_det_stoch survival — stronger spawner)."""
        if score < self.best_spawner_score:
            self.best_spawner_score = score
            out = {**payload, "ckpt_kind": "best_spawner", "selection_score": score}
            torch.save(out, self.best_spawner)
            return True
        return False

    def maybe_snapshot(self, update: int, player_payload: dict, spawner_payload: dict) -> bool:
        if self.snapshot_dir is None or self.snapshot_every <= 0:
            return False
        if update % self.snapshot_every != 0:
            return False
        p = self.snapshot_dir / f"player_update_{update}.pt"
        s = self.snapshot_dir / f"spawner_update_{update}.pt"
        torch.save({**player_payload, "ckpt_kind": "snapshot"}, p)
        torch.save({**spawner_payload, "ckpt_kind": "snapshot"}, s)
        return True
