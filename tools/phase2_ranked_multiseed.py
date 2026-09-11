#!/usr/bin/env python3
"""Phase 2 multi-seed ranked top-k runner.

Collects the canonical V1 teacher dataset exactly once (one train/held
episode split) and reuses the exact same train/held ``Frame`` objects for
every train seed in ``TRAIN_SEEDS``. Each seed trains an independent
``PlayerRankedTopK`` (hidden width auto-matched to a target trainable
parameter count) with a hybrid objective (0.5 hard teacher-argmax
cross-entropy + 0.5 soft cross-entropy at temperature 1.0), sharing an
identical optimizer-step / batch-order schedule across seeds so only the
model initialization differs seed-to-seed.

Checkpoint selection is frame-level only (held agreement, then lower held
teacher->student KL, then latest), reusing
``tools.phase2_capacity_controls.should_replace_selected`` verbatim.
Checkpoints are packed through the strict production provenance path in
``qrokkun_env.agents.player_checkpoints`` (never the loose experimental
dict packing used by the other phase2 harnesses).

A seed's held agreement must clear ``HELD_AGREEMENT_GATE_MIN`` before its
closed loop (deterministic argmax vs the built-in scripted environment only,
seeds 3000..3029 inclusive, up to 4200 steps) is ever attempted. The
aggregate gate then requires: every seed's held agreement to clear the
threshold, every such seed's closed loop to have actually run, every ran
seed's own closed-loop mean/median to clear 25s, and the mean of the
per-seed train-seed means to clear 30s.

This harness never trains with reinforcement learning, never uses a learned
spawn controller, never mixes in a random-fraction control arm, never writes
architecture-search artifacts, and never promotes anything to production.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4
from qrokkun_env.agents.player_checkpoints import file_sha256, save_player_checkpoint
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK
from qrokkun_env.env import ACTIONS, Qrokkun26Env

from tools.phase2_capacity_controls import (  # noqa: E402
    compute_metrics,
    live_counts_from_pad,
    should_replace_selected,
    uniform_batch_indices,
)
from tools.phase2_distill_v1_to_v4 import (  # noqa: E402
    Frame,
    collect_dataset,
    frames_to_tensors,
    load_teacher,
)

# --------------------------------------------------------------------------- #
# Pre-registered constants
# --------------------------------------------------------------------------- #
TRAIN_SEEDS: tuple[int, ...] = (4, 5, 6)

HYBRID_HARD_WEIGHT = 0.5
HYBRID_SOFT_WEIGHT = 0.5
HYBRID_TEMPERATURE = 1.0

HELD_AGREEMENT_GATE_MIN = 0.65

CLOSED_LOOP_SEED_START = 3000
CLOSED_LOOP_NUM_SEEDS = 30
CLOSED_LOOP_MAX_STEPS = 4200

AGGREGATE_CLOSED_LOOP_MEAN_MIN = 25.0
AGGREGATE_CLOSED_LOOP_MEDIAN_MIN = 25.0
AGGREGATE_TRAIN_SEED_MEAN_MIN = 30.0

TARGET_HIDDEN_PARAM_COUNT = 269050
DEFAULT_TOP_K = 8

# Batch order is intentionally decoupled from the per-seed model init seed so
# every train seed shares an identical optimizer-step / batch-order schedule.
BATCH_ORDER_SEED = 0


def closed_loop_seed_list() -> list[int]:
    return list(range(CLOSED_LOOP_SEED_START, CLOSED_LOOP_SEED_START + CLOSED_LOOP_NUM_SEEDS))


# --------------------------------------------------------------------------- #
# Hybrid objective
# --------------------------------------------------------------------------- #
def hybrid_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    hard = F.cross_entropy(student_logits, teacher_logits.argmax(-1))
    teacher_probs = F.softmax(teacher_logits / HYBRID_TEMPERATURE, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    soft = -(teacher_probs * student_logp).sum(-1).mean()
    return HYBRID_HARD_WEIGHT * hard + HYBRID_SOFT_WEIGHT * soft


# --------------------------------------------------------------------------- #
# Hidden width auto-match near the target parameter count
# --------------------------------------------------------------------------- #
def count_trainable_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _ranked_topk_param_count(hidden: int, top_k: int) -> int:
    return count_trainable_params(PlayerRankedTopK(top_k=top_k, hidden=hidden))


def _choose_hidden(target: int, top_k: int, lo: int = 1, hi: int = 4096) -> dict[str, Any]:
    hi_count = _ranked_topk_param_count(hi, top_k)
    while hi_count < target and hi < 1_000_000:
        hi *= 2
        hi_count = _ranked_topk_param_count(hi, top_k)

    best_hidden = lo
    best_count = _ranked_topk_param_count(lo, top_k)
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        count = _ranked_topk_param_count(mid, top_k)
        if abs(count - target) < abs(best_count - target):
            best_hidden, best_count = mid, count
        if count < target:
            a = mid + 1
        elif count > target:
            b = mid - 1
        else:
            best_hidden, best_count = mid, count
            break
    rel_error = abs(best_count - target) / target if target else 0.0
    return {
        "hidden": best_hidden,
        "param_count": best_count,
        "target": target,
        "rel_error": rel_error,
        "within_2pct": rel_error <= 0.02,
    }


def resolve_hidden(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Auto-choose (and report) the PlayerRankedTopK hidden width nearest
    TARGET_HIDDEN_PARAM_COUNT -- an explicit --hidden always overrides the
    search instead of being silently replaced."""
    target = TARGET_HIDDEN_PARAM_COUNT
    if getattr(args, "hidden", None) is not None:
        chosen = int(args.hidden)
        count = _ranked_topk_param_count(chosen, args.top_k)
        rel_error = abs(count - target) / target if target else 0.0
        info = {
            "hidden": chosen,
            "param_count": count,
            "target": target,
            "rel_error": rel_error,
            "within_2pct": rel_error <= 0.02,
            "source": "explicit_override",
        }
        return chosen, info
    info = _choose_hidden(target=target, top_k=args.top_k)
    info["source"] = "auto_matched"
    return info["hidden"], info


# --------------------------------------------------------------------------- #
# CLI / mode defaults
# --------------------------------------------------------------------------- #
_FULL_DEFAULTS = {
    "data_seed": 4,
    "collect_episodes": 240,
    "collect_seed_start": 20000,
    "max_steps": CLOSED_LOOP_MAX_STEPS,
    "frames_per_episode_cap": 700,
    "held_out_frac": 0.12,
    "epochs": 80,
    "batch_size": 1024,
    "lr": 3e-4,
    "held_agreement_min": HELD_AGREEMENT_GATE_MIN,
    "closed_loop_seed_start": CLOSED_LOOP_SEED_START,
    "closed_loop_num_seeds": CLOSED_LOOP_NUM_SEEDS,
    "closed_loop_max_steps": CLOSED_LOOP_MAX_STEPS,
}
_QUICK_DEFAULTS = {
    "data_seed": 4,
    "collect_episodes": 6,
    "collect_seed_start": 20000,
    "max_steps": 40,
    "frames_per_episode_cap": 20,
    "held_out_frac": 0.25,
    "epochs": 2,
    "batch_size": 8,
    "lr": 3e-4,
    "held_agreement_min": HELD_AGREEMENT_GATE_MIN,
    "closed_loop_seed_start": 9000,
    "closed_loop_num_seeds": 2,
    "closed_loop_max_steps": 20,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", type=Path, required=True, help="PlayerV1 ckpt, e.g. player_gpu.pt")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--quick", action="store_true", help="shrink defaults for a fast smoke run")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=list(TRAIN_SEEDS),
        help="train seeds to run (default: pre-registered 4 5 6)",
    )
    ap.add_argument("--collect-episodes", type=int, default=None)
    ap.add_argument("--collect-seed-start", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--frames-per-episode-cap", type=int, default=None)
    ap.add_argument("--held-out-frac", type=float, default=None)

    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--hidden", type=int, default=None, help="override the auto-matched hidden width")

    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)

    ap.add_argument("--held-agreement-min", type=float, default=None)
    ap.add_argument("--closed-loop-seed-start", type=int, default=None)
    ap.add_argument("--closed-loop-num-seeds", type=int, default=None)
    ap.add_argument("--closed-loop-max-steps", type=int, default=None)
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = _QUICK_DEFAULTS if args.quick else _FULL_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.train_seeds = tuple(args.train_seeds)
    return args


# --------------------------------------------------------------------------- #
# Shared dataset: collected exactly once, reused verbatim per train seed
# --------------------------------------------------------------------------- #
def dataset_identity(train_frames: list[Frame], held_frames: list[Frame]) -> dict[str, Any]:
    h = hashlib.sha256()
    for frames in (train_frames, held_frames):
        for f in frames:
            h.update(f.player.tobytes())
            h.update(f.bullets.tobytes())
            h.update(f.pad.tobytes())
            h.update(f.teacher_logits.tobytes())
            h.update(str(f.elapsed).encode("utf-8"))
            h.update(str(f.episode).encode("utf-8"))
    return {
        "hash": h.hexdigest(),
        "n_train": len(train_frames),
        "n_held": len(held_frames),
    }


def collect_shared_dataset(
    teacher: Any, device: torch.device, args: argparse.Namespace
) -> tuple[list[Frame], list[Frame], dict[str, Any]]:
    """Collect the canonical dataset exactly once for the whole multi-seed run."""
    import random

    py_rng = random.Random(args.data_seed)
    train_frames, held_frames = collect_dataset(
        teacher,
        device,
        n_episodes=args.collect_episodes,
        seed_start=args.collect_seed_start,
        max_steps=args.max_steps,
        frames_cap=args.frames_per_episode_cap,
        held_out_frac=args.held_out_frac,
        rng=py_rng,
    )
    identity = dataset_identity(train_frames, held_frames)
    return train_frames, held_frames, identity


def frames_for_seed(
    seed: int, train_frames: list[Frame], held_frames: list[Frame]
) -> tuple[list[Frame], list[Frame]]:
    """Every train seed reuses the exact SAME Frame objects (never re-collected)."""
    del seed
    return train_frames, held_frames


# --------------------------------------------------------------------------- #
# Independent deterministic model init; identical batch-order policy per seed
# --------------------------------------------------------------------------- #
def init_model(seed: int, top_k: int, hidden: int) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def make_batch_order(
    n: int, batch_size: int, args: argparse.Namespace | None, seed_for_model: int, epoch: int = 0
) -> list[torch.Tensor]:
    """Batch order is identical across train seeds -- it never depends on
    seed_for_model, only on a fixed schedule seed plus the epoch index."""
    del args, seed_for_model
    gen = torch.Generator().manual_seed(BATCH_ORDER_SEED + epoch)
    return uniform_batch_indices(n, batch_size, gen)


# --------------------------------------------------------------------------- #
# Per-seed training
# --------------------------------------------------------------------------- #
def _pack_extra(
    seed: int,
    args: argparse.Namespace,
    *,
    teacher_sha256: str | None = None,
    dataset_identity_hash: str | None = None,
    selected_epoch: int | None = None,
    selected_held_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "seed": seed,
        "tool": "tools/phase2_ranked_multiseed.py",
        "top_k": args.top_k,
    }
    if teacher_sha256 is not None:
        extra["teacher_sha256"] = teacher_sha256
    if dataset_identity_hash is not None:
        extra["dataset_identity_hash"] = dataset_identity_hash
    if selected_epoch is not None:
        extra["selected_epoch"] = selected_epoch
    if selected_held_metrics is not None:
        extra["selected_held_metrics"] = dict(selected_held_metrics)
    return extra


def train_one_seed(
    seed: int,
    args: argparse.Namespace,
    train_frames: list[Frame],
    held_frames: list[Frame],
    device: torch.device,
    out_dir: Path,
    *,
    hidden: int | None = None,
    hidden_info: dict[str, Any] | None = None,
    teacher_sha256: str | None = None,
    dataset_identity_hash: str | None = None,
) -> dict[str, Any]:
    if not held_frames:
        raise ValueError("train_one_seed requires non-empty held_frames -- cannot select/gate on empty held data")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hidden is None or hidden_info is None:
        hidden, hidden_info = resolve_hidden(args)
    net = init_model(seed=seed, top_k=args.top_k, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_tensors = frames_to_tensors(train_frames, device)
    held_tensors = frames_to_tensors(held_frames, device)
    n = int(train_tensors["player"].shape[0])

    best_selected: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    latest_state: dict[str, Any] | None = None
    held_metrics: dict[str, Any] | None = None
    optimizer_steps = 0

    epochs_path = out_dir / "epochs.jsonl"
    epoch_records: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        for idx in make_batch_order(n, args.batch_size, args, seed, epoch=epoch):
            dist, _v = net(train_tensors["player"][idx], train_tensors["bullets"][idx], train_tensors["pad"][idx])
            loss = hybrid_loss(dist.logits, train_tensors["teacher_logits"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            optimizer_steps += 1

        held_metrics = _held_frame_metrics(net, held_tensors, args.top_k)
        latest_state = copy.deepcopy(net.state_dict())
        candidate = {
            "agreement": held_metrics["agreement"],
            "teacher_to_student_kl": held_metrics["teacher_to_student_kl"],
        }
        selected = bool(should_replace_selected(candidate, best_selected))
        if selected:
            best_selected = candidate
            best_epoch = epoch
            best_state = latest_state
            best_metrics = copy.deepcopy(held_metrics)
        epoch_records.append(
            {
                "epoch": epoch,
                "held_metrics": held_metrics,
                "candidate": candidate,
                "selected": selected,
            }
        )

    with epochs_path.open("w") as f:
        for rec in epoch_records:
            f.write(json.dumps(rec) + "\n")

    assert best_state is not None and latest_state is not None  # args.epochs >= 1

    net.load_state_dict(best_state)
    net.eval()

    ckpt_path = out_dir / f"seed{seed}.pt"
    save_player_checkpoint(
        net,
        ckpt_path,
        source_tool="tools/phase2_ranked_multiseed.py",
        extra=_pack_extra(
            seed,
            args,
            teacher_sha256=teacher_sha256,
            dataset_identity_hash=dataset_identity_hash,
            selected_epoch=best_epoch,
            selected_held_metrics=best_metrics,
        ),
    )

    return {
        "seed": seed,
        "epochs_run": args.epochs,
        "optimizer_steps": optimizer_steps,
        "held_metrics": best_metrics if best_metrics is not None else held_metrics,
        "checkpoint": str(ckpt_path),
        "hidden": hidden,
        "hidden_info": hidden_info,
        "selected_epoch": best_epoch,
        "net": net,
    }


@torch.no_grad()
def _held_frame_metrics(net: PlayerRankedTopK, tensors: dict[str, torch.Tensor], top_k: int) -> dict[str, Any]:
    net.eval()
    n = int(tensors["player"].shape[0])
    batch_size = 4096
    outs = []
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        dist, _v = net(tensors["player"][sl], tensors["bullets"][sl], tensors["pad"][sl])
        outs.append(dist.logits)
    student_logits = torch.cat(outs, dim=0) if outs else torch.zeros(0, len(ACTIONS))
    live_counts = live_counts_from_pad(tensors["pad"])
    return compute_metrics(student_logits, tensors["teacher_logits"], tensors["elapsed"], live_counts)


# --------------------------------------------------------------------------- #
# Per-seed closed-loop gate + rollout
# --------------------------------------------------------------------------- #
def seed_passes_closed_loop_gate(held_metrics: dict[str, Any], *, threshold: float = HELD_AGREEMENT_GATE_MIN) -> bool:
    return held_metrics["agreement"] >= threshold


@torch.no_grad()
def closed_loop_for_seed(
    net: PlayerRankedTopK, device: torch.device, seeds: list[int], max_steps: int
) -> list[dict[str, Any]]:
    """Deterministic argmax rollout vs the built-in scripted environment only.

    Each result reports ``elapsed`` and ``censored``: ``censored`` is True
    when the episode did not finish (timed out) within ``max_steps`` -- this
    is a right-censored observation, never conflated with an in-episode
    death, and must never be silently dropped from the reported elapsed time.
    """
    from qrokkun_env.agents.player_ranked_topk import argmax_action

    net.eval()
    results: list[dict[str, Any]] = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        for _ in range(max_steps):
            action = argmax_action(net, env, device)
            _obs, _r, done, info = env.step(action)
            if done:
                results.append({"elapsed": float(info.get("elapsed", env.elapsed)), "censored": False})
                break
        else:
            results.append({"elapsed": float(env.elapsed), "censored": True})
    return results


def _normalize_closed_loop_results(raw: list[Any]) -> tuple[list[float], list[bool]]:
    """Accept either the current dict-shaped results ({elapsed, censored})
    or plain floats (legacy stubs / callers), always returning elapsed times
    for stats and a same-length censored flag list."""
    elapsed: list[float] = []
    censored: list[bool] = []
    for item in raw:
        if isinstance(item, dict):
            elapsed.append(float(item["elapsed"]))
            censored.append(bool(item.get("censored", False)))
        else:
            elapsed.append(float(item))
            censored.append(False)
    return elapsed, censored


def run_closed_loop_stage(
    seed_reports: dict[int, dict[str, Any]], args: argparse.Namespace, device: torch.device
) -> dict[int, dict[str, Any]]:
    seeds = list(range(args.closed_loop_seed_start, args.closed_loop_seed_start + args.closed_loop_num_seeds))
    max_steps = args.closed_loop_max_steps
    threshold = getattr(args, "held_agreement_min", HELD_AGREEMENT_GATE_MIN)

    out: dict[int, dict[str, Any]] = {}
    for seed, rep in seed_reports.items():
        rep = dict(rep)
        if seed_passes_closed_loop_gate(rep["held_metrics"], threshold=threshold):
            raw = closed_loop_for_seed(rep["net"], device, seeds, max_steps)
            times, censored_flags = _normalize_closed_loop_results(raw)
            per_seed = {str(s): t for s, t in zip(seeds, times)}
            per_seed_censored = {str(s): c for s, c in zip(seeds, censored_flags)}
            censor_count = sum(1 for c in censored_flags if c)
            rep["closed_loop_ran"] = True
            rep["closed_loop"] = {
                "mean": statistics.mean(times) if times else 0.0,
                "median": float(statistics.median(times)) if times else 0.0,
                "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
                "min": min(times) if times else 0.0,
                "max": max(times) if times else 0.0,
                "n": len(times),
                "per_seed": per_seed,
                "per_seed_censored": per_seed_censored,
                "censor_count": censor_count,
                "censor_rate": (censor_count / len(censored_flags)) if censored_flags else 0.0,
            }
        else:
            rep["closed_loop_ran"] = False
            rep["closed_loop"] = None
        out[seed] = rep
    return out


# --------------------------------------------------------------------------- #
# Aggregate gate: explicit pass/fail detail
# --------------------------------------------------------------------------- #
def _closed_loop_covers_canonical_window(closed_loop: dict[str, Any] | None) -> bool:
    if not closed_loop:
        return False
    keys = set(closed_loop.get("per_seed", {}) or {})
    canonical_keys = {str(s) for s in closed_loop_seed_list()}
    return keys == canonical_keys


def evaluate_aggregate_gate(seed_reports: dict[int, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    threshold = getattr(args, "held_agreement_min", HELD_AGREEMENT_GATE_MIN)
    configured_num_seeds = getattr(args, "closed_loop_num_seeds", CLOSED_LOOP_NUM_SEEDS)
    configured_seed_start = getattr(args, "closed_loop_seed_start", CLOSED_LOOP_SEED_START)

    per_seed: dict[str, dict[str, Any]] = {}
    all_held_pass = True
    all_closed_loop_ran = True
    seed_means: list[float] = []
    seed_medians: list[float] = []
    any_seed_covers_canonical_window = False

    for seed, rep in seed_reports.items():
        agreement = rep["held_metrics"]["agreement"]
        held_pass = agreement >= threshold
        ran = bool(rep.get("closed_loop_ran", False))
        cl = rep.get("closed_loop")
        n_covered = int(cl.get("n", len(cl.get("per_seed", {}) or {}))) if cl else 0
        coverage_complete = ran and n_covered == configured_num_seeds
        entry: dict[str, Any] = {
            "held_agreement": agreement,
            "held_pass": held_pass,
            "closed_loop_ran": ran,
            "closed_loop_coverage_complete": coverage_complete,
        }
        if not held_pass:
            all_held_pass = False
        elif not coverage_complete:
            all_closed_loop_ran = False
        else:
            entry["closed_loop_mean_for_seed"] = cl["mean"]
            entry["closed_loop_median_for_seed"] = cl["median"]
            seed_means.append(cl["mean"])
            seed_medians.append(cl["median"])
            if _closed_loop_covers_canonical_window(cl):
                any_seed_covers_canonical_window = True
        per_seed[str(seed)] = entry

    min_seed_closed_loop_mean = min(seed_means) if seed_means else 0.0
    min_seed_closed_loop_median = min(seed_medians) if seed_medians else 0.0
    closed_loop_mean_pass = min_seed_closed_loop_mean >= AGGREGATE_CLOSED_LOOP_MEAN_MIN
    closed_loop_median_pass = min_seed_closed_loop_median >= AGGREGATE_CLOSED_LOOP_MEDIAN_MIN

    train_seed_mean_of_means = statistics.mean(seed_means) if seed_means else 0.0
    train_seed_mean_of_means_pass = train_seed_mean_of_means >= AGGREGATE_TRAIN_SEED_MEAN_MIN

    gate_pass = (
        all_held_pass
        and all_closed_loop_ran
        and closed_loop_mean_pass
        and closed_loop_median_pass
        and train_seed_mean_of_means_pass
    )

    # The production reproducibility claim additionally requires the exact
    # canonical seed window (start/count *and* the actual reported seed keys
    # for every counted seed) -- a quick/ad-hoc window can pass its own
    # configured gate without ever being canonical.
    is_canonical_seed_window = (
        configured_seed_start == CLOSED_LOOP_SEED_START
        and configured_num_seeds == CLOSED_LOOP_NUM_SEEDS
        and any_seed_covers_canonical_window
    )
    production_reproducibility_gate = bool(gate_pass and is_canonical_seed_window)

    failed: list[str] = []
    for seed, entry in per_seed.items():
        if not entry["held_pass"]:
            failed.append(f"seed {seed} held agreement {entry['held_agreement']:.3f} below {threshold}")
    if not all_closed_loop_ran:
        failed.append("closed loop did not run (or did not fully cover the configured seed count) for every held-passing seed")
    if not closed_loop_mean_pass:
        failed.append(f"closed loop mean {min_seed_closed_loop_mean:.2f}s below {AGGREGATE_CLOSED_LOOP_MEAN_MIN}s")
    if not closed_loop_median_pass:
        failed.append(f"closed loop median {min_seed_closed_loop_median:.2f}s below {AGGREGATE_CLOSED_LOOP_MEDIAN_MIN}s")
    if not train_seed_mean_of_means_pass:
        failed.append(
            f"mean of train-seed means {train_seed_mean_of_means:.2f}s below {AGGREGATE_TRAIN_SEED_MEAN_MIN}s"
        )
    reason = "gate passed: all thresholds met" if gate_pass else "gate failed: " + "; ".join(failed)

    return {
        "gate_pass": gate_pass,
        "all_held_pass": all_held_pass,
        "all_closed_loop_ran": all_closed_loop_ran,
        "min_seed_closed_loop_mean": min_seed_closed_loop_mean,
        "min_seed_closed_loop_median": min_seed_closed_loop_median,
        "closed_loop_mean_pass": closed_loop_mean_pass,
        "closed_loop_median_pass": closed_loop_median_pass,
        "train_seed_mean_of_means": train_seed_mean_of_means,
        "train_seed_mean_of_means_pass": train_seed_mean_of_means_pass,
        "is_canonical_seed_window": is_canonical_seed_window,
        "production_reproducibility_gate": production_reproducibility_gate,
        "per_seed": per_seed,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def current_git_commit() -> str | None:
    root = Path(__file__).resolve().parents[1]
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


# --------------------------------------------------------------------------- #
# Full multi-seed run
# --------------------------------------------------------------------------- #
def run_multiseed(
    args: argparse.Namespace, train_frames: list[Frame], held_frames: list[Frame], device: torch.device
) -> dict[str, Any]:
    if not held_frames:
        raise ValueError("run_multiseed requires non-empty held_frames -- cannot select/gate on empty held data")

    start_time = time.monotonic()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    identity = dataset_identity(train_frames, held_frames)
    hidden, hidden_info = resolve_hidden(args)
    teacher_sha256 = file_sha256(args.teacher)

    seed_reports: dict[int, dict[str, Any]] = {}
    for seed in args.train_seeds:
        t_frames, h_frames = frames_for_seed(seed, train_frames, held_frames)
        seed_reports[seed] = train_one_seed(
            seed,
            args,
            t_frames,
            h_frames,
            device,
            out_dir / f"seed{seed}",
            hidden=hidden,
            hidden_info=hidden_info,
            teacher_sha256=teacher_sha256,
            dataset_identity_hash=identity["hash"],
        )

    seed_reports = run_closed_loop_stage(seed_reports, args, device)
    gate = evaluate_aggregate_gate(seed_reports, args)

    runtime_seconds = time.monotonic() - start_time

    report = {
        "seeds": seed_reports,
        "dataset_identity": identity,
        "teacher_sha256": teacher_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "device": str(device),
        "actions": list(ACTIONS),
        "observation": {
            "player_feat": PLAYER_FEAT_V4,
            "bullet_feat": BULLET_FEAT_V4,
            "max_bullets": MAX_BULLETS_V4,
        },
        "params": {
            "data_seed": args.data_seed,
            "collect_episodes": args.collect_episodes,
            "collect_seed_start": args.collect_seed_start,
            "max_steps": args.max_steps,
            "frames_per_episode_cap": args.frames_per_episode_cap,
            "held_out_frac": args.held_out_frac,
            "top_k": args.top_k,
            "hidden": hidden,
            "hidden_info": hidden_info,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "held_agreement_min": args.held_agreement_min,
            "closed_loop_seed_start": args.closed_loop_seed_start,
            "closed_loop_num_seeds": args.closed_loop_num_seeds,
            "closed_loop_max_steps": args.closed_loop_max_steps,
        },
        "runtime_seconds": runtime_seconds,
        "git_commit": current_git_commit(),
        "torch_version": torch.__version__,
        "teacher": str(args.teacher),
        "gate": gate,
    }
    return report


def main() -> None:
    args = apply_mode_defaults(build_parser().parse_args())
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    teacher = load_teacher(args.teacher, device)
    train_frames, held_frames, _identity = collect_shared_dataset(teacher, device, args)
    if not train_frames:
        raise RuntimeError("no training frames collected -- increase --collect-episodes")

    report = run_multiseed(args, train_frames, held_frames, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    import json

    serializable = {k: v for k, v in report.items() if k != "seeds"}
    serializable["seeds"] = {
        str(seed): {k: v for k, v in rep.items() if k != "net"} for seed, rep in report["seeds"].items()
    }
    report_path.write_text(json.dumps(serializable, indent=2) + "\n")

    gate = report["gate"]
    print(f"wrote {report_path}", flush=True)
    print(f"[gate] gate_pass={gate['gate_pass']} reason={gate['reason']}", flush=True)


if __name__ == "__main__":
    main()
