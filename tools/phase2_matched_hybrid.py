#!/usr/bin/env python3
"""Phase 2 matched architecture/objective experiment with a gated closed-loop eval.

Builds on tools/phase2_capacity_controls.py (kept untouched/reproducible).
This harness pre-registers exactly six arms on ONE episode-level split and a
uniform frame set:

  attn64_hard      - PlayerV4 (full 64-slot attention), hard (teacher-argmax) labels.
  attn64_soft_t1   - PlayerV4, soft teacher-softmax labels at temperature 1.0.
  attn64_hybrid    - PlayerV4, hybrid loss (0.5 hard CE + 0.5 soft CE at T=1).
  attn8_hybrid     - PlayerV4 with bullet slots beyond the nearest 8 masked to
                     pad (same architecture, capacity-limited input), hybrid loss.
  flat8_hard       - FlatTop8MLP (non-attention), hard labels. Hidden width is
                     auto-chosen so trainable parameter count matches PlayerV4
                     within 2% (never silently reuses the global --hidden).
  flat8_hybrid     - FlatTop8MLP, same matched hidden width, hybrid loss.

All non-tiny arms (every arm here) share identical epochs, batch size, seed,
optimizer step count, and train/held frame sets -- the only things that vary
are model architecture and training objective.

Checkpoint selection remains frame-level only (held agreement, then lower
held teacher->student KL, then latest tie) -- reused verbatim from
tools.phase2_capacity_controls.should_replace_selected. It NEVER uses
closed-loop / episode-outcome signals.

A gate is pre-registered and evaluated purely from held-out frame metrics
BEFORE any closed-loop evaluation is attempted:
  1. highest-held-agreement arm's held agreement >= --gate-agreement-min
  2. that arm improves over attn64_soft_t1 by >= --gate-improvement-min (abs)
  3. every nonzero bullet-density bucket (1-7, 8-19, 20+) for that arm has
     held agreement >= --gate-bullet-bucket-min

If (and only if) the gate passes, the selected architecture (including
flat8) is reloaded from its checkpoint and evaluated closed-loop: deterministic
argmax vs the built-in scripted env only, seeds 3000..3029 inclusive, up to
4200 frames (70s) per seed. No PPO, no learned Spawner, no NAS writes, no
promotion. If the gate fails, no env is ever constructed or stepped.

Checkpoints produced by this script are experimental / not production
compatible (see _pack_ckpt).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qrokkun_env.agents.obs_v4 import encode_obs
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.env import ACTIONS, Qrokkun26Env

# Reuse frame plumbing, the FlatTop8MLP positive-control architecture, the
# top-k bullet mask, and the frame-level-only selection rule from the first
# harness instead of re-implementing them.
from tools.phase2_capacity_controls import (  # noqa: E402
    FlatTop8MLP,
    apply_top_k_mask,
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

TOP_K = 8

# --------------------------------------------------------------------------- #
# Arm catalogue
# --------------------------------------------------------------------------- #
MODEL_ATTN64 = "attn64"
MODEL_FLAT8 = "flat8"

OBJ_HARD = "hard"
OBJ_SOFT_T1 = "soft_t1"
OBJ_HYBRID = "hybrid"

ARM_ATTN64_HARD = "attn64_hard"
ARM_ATTN64_SOFT_T1 = "attn64_soft_t1"
ARM_ATTN64_HYBRID = "attn64_hybrid"
ARM_ATTN8_HYBRID = "attn8_hybrid"
ARM_FLAT8_HARD = "flat8_hard"
ARM_FLAT8_HYBRID = "flat8_hybrid"

HYBRID_HARD_WEIGHT = 0.5
HYBRID_SOFT_WEIGHT = 0.5
HYBRID_TEMPERATURE = 1.0
SOFT_T1_TEMPERATURE = 1.0

BULLET_BUCKETS_NONZERO: tuple[str, ...] = ("1-7", "8-19", "20+")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    model: str
    objective: str
    top_k_mask: bool = False
    description: str = ""


ARM_SPECS: dict[str, ArmSpec] = {
    ARM_ATTN64_HARD: ArmSpec(
        name=ARM_ATTN64_HARD,
        model=MODEL_ATTN64,
        objective=OBJ_HARD,
        description="PlayerV4 (full 64-slot attention), hard teacher-argmax labels.",
    ),
    ARM_ATTN64_SOFT_T1: ArmSpec(
        name=ARM_ATTN64_SOFT_T1,
        model=MODEL_ATTN64,
        objective=OBJ_SOFT_T1,
        description="PlayerV4, soft teacher-softmax labels at temperature 1.0.",
    ),
    ARM_ATTN64_HYBRID: ArmSpec(
        name=ARM_ATTN64_HYBRID,
        model=MODEL_ATTN64,
        objective=OBJ_HYBRID,
        description="PlayerV4, hybrid loss (0.5 hard CE + 0.5 soft CE at T=1).",
    ),
    ARM_ATTN8_HYBRID: ArmSpec(
        name=ARM_ATTN8_HYBRID,
        model=MODEL_ATTN64,
        objective=OBJ_HYBRID,
        top_k_mask=True,
        description=(
            "PlayerV4 with bullet slots beyond the nearest 8 masked to pad "
            "(same architecture, capacity-limited input), hybrid loss."
        ),
    ),
    ARM_FLAT8_HARD: ArmSpec(
        name=ARM_FLAT8_HARD,
        model=MODEL_FLAT8,
        objective=OBJ_HARD,
        description=(
            "FlatTop8MLP (non-attention), hard labels. Hidden width is "
            "auto-chosen to match PlayerV4 trainable parameter count within 2%."
        ),
    ),
    ARM_FLAT8_HYBRID: ArmSpec(
        name=ARM_FLAT8_HYBRID,
        model=MODEL_FLAT8,
        objective=OBJ_HYBRID,
        description="FlatTop8MLP, same matched hidden width, hybrid loss.",
    ),
}
ALL_ARMS: tuple[str, ...] = (
    ARM_ATTN64_HARD,
    ARM_ATTN64_SOFT_T1,
    ARM_ATTN64_HYBRID,
    ARM_ATTN8_HYBRID,
    ARM_FLAT8_HARD,
    ARM_FLAT8_HYBRID,
)


def resolve_arms(csv: str) -> list[str]:
    if csv == "all":
        return list(ALL_ARMS)
    names = [x.strip() for x in csv.split(",") if x.strip()]
    unknown = [n for n in names if n not in ARM_SPECS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; choose from {list(ALL_ARMS)} or 'all'")
    return names


# --------------------------------------------------------------------------- #
# Objective (hard / soft-T1 / hybrid) loss
# --------------------------------------------------------------------------- #
def _soft_ce(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    return -(teacher_probs * student_logp).sum(-1).mean()


def arm_loss(spec: ArmSpec, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    if spec.objective == OBJ_HARD:
        return F.cross_entropy(student_logits, teacher_logits.argmax(-1))
    if spec.objective == OBJ_SOFT_T1:
        return _soft_ce(student_logits, teacher_logits, SOFT_T1_TEMPERATURE)
    if spec.objective == OBJ_HYBRID:
        hard = F.cross_entropy(student_logits, teacher_logits.argmax(-1))
        soft = _soft_ce(student_logits, teacher_logits, HYBRID_TEMPERATURE)
        return HYBRID_HARD_WEIGHT * hard + HYBRID_SOFT_WEIGHT * soft
    raise ValueError(f"unknown objective: {spec.objective}")


# --------------------------------------------------------------------------- #
# Parameter counting + flat8 hidden auto-match (never silently reuse --hidden)
# --------------------------------------------------------------------------- #
def count_trainable_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def flat8_param_count(hidden: int, top_k: int) -> int:
    return count_trainable_params(FlatTop8MLP(top_k=top_k, hidden=hidden))


def choose_flat_hidden(target: int, top_k: int, lo: int = 1, hi: int = 4096) -> dict[str, Any]:
    """Binary-search the FlatTop8MLP hidden width whose trainable parameter
    count is closest to `target` (flat8_param_count is strictly increasing in
    hidden, so bisection finds the closest achievable match)."""
    hi_count = flat8_param_count(hi, top_k)
    while hi_count < target and hi < 1_000_000:
        hi *= 2
        hi_count = flat8_param_count(hi, top_k)

    best_hidden = lo
    best_count = flat8_param_count(lo, top_k)
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        count = flat8_param_count(mid, top_k)
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


def resolve_flat8_hidden(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Auto-choose (and report) the FlatTop8MLP hidden width matching
    PlayerV4(d_model=args.d_model, hidden=args.hidden) trainable parameter
    count within 2% -- explicit --flat-hidden overrides the search."""
    target = count_trainable_params(PlayerV4(d_model=args.d_model, hidden=args.hidden))
    if getattr(args, "flat_hidden", None) is not None:
        chosen = int(args.flat_hidden)
        count = flat8_param_count(chosen, args.top_k)
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
    info = choose_flat_hidden(target=target, top_k=args.top_k)
    info["source"] = "auto_matched"
    return info["hidden"], info


# --------------------------------------------------------------------------- #
# CLI / mode defaults
# --------------------------------------------------------------------------- #
# Real-run defaults per the pre-registered experiment spec.
_FULL_DEFAULTS = {
    "seed": 4,
    "collect_episodes": 240,
    "collect_seed_start": 20000,
    "held_out_frac": 0.12,
    "frames_per_episode_cap": 700,
    "epochs": 80,
    "batch_size": 1024,
    "lr": 3e-4,
    "d_model": 128,
    "hidden": 256,
    "max_steps": 60 * 70,
    "closed_loop_seed_start": 3000,
    "closed_loop_num_seeds": 30,
    "closed_loop_max_steps": 60 * 70,
}
_QUICK_DEFAULTS = {
    "seed": 4,
    "collect_episodes": 6,
    "collect_seed_start": 20000,
    "held_out_frac": 0.25,
    "frames_per_episode_cap": 40,
    "epochs": 2,
    "batch_size": 8,
    "lr": 3e-4,
    "d_model": 16,
    "hidden": 16,
    "max_steps": 60,
    "closed_loop_seed_start": 9000,
    "closed_loop_num_seeds": 2,
    "closed_loop_max_steps": 30,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", type=Path, required=True, help="PlayerV1 ckpt, e.g. player_gpu.pt")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--arms", type=str, default="all", help="'all' or a comma-separated list of arm names")
    ap.add_argument("--quick", action="store_true", help="shrink defaults for a fast smoke run")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--collect-episodes", type=int, default=None)
    ap.add_argument("--collect-seed-start", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--frames-per-episode-cap", type=int, default=None)
    ap.add_argument("--held-out-frac", type=float, default=None)

    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--flat-hidden", type=int, default=None, help="override the auto-matched flat8 hidden width")

    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)

    ap.add_argument("--gate-agreement-min", type=float, default=0.50)
    ap.add_argument("--gate-improvement-min", type=float, default=0.03)
    ap.add_argument("--gate-bullet-bucket-min", type=float, default=0.40)

    ap.add_argument("--closed-loop-seed-start", type=int, default=None)
    ap.add_argument("--closed-loop-num-seeds", type=int, default=None)
    ap.add_argument("--closed-loop-max-steps", type=int, default=None)
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = _QUICK_DEFAULTS if args.quick else _FULL_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# Model dispatch / top-k mask forward (reuses apply_top_k_mask verbatim)
# --------------------------------------------------------------------------- #
def make_model(spec: ArmSpec, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if spec.model == MODEL_FLAT8:
        flat_hidden = getattr(args, "flat_hidden", None)
        if flat_hidden is None:
            flat_hidden, _info = resolve_flat8_hidden(args)
            args.flat_hidden = flat_hidden
        net: nn.Module = FlatTop8MLP(top_k=args.top_k, hidden=flat_hidden)
    else:
        net = PlayerV4(d_model=args.d_model, hidden=args.hidden)
    return net.to(device)


def transform_batch(spec: ArmSpec, bullets: torch.Tensor, pad: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.top_k_mask:
        return apply_top_k_mask(bullets, pad, k)
    return bullets, pad


def forward_arm(
    net: nn.Module, spec: ArmSpec, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor, top_k: int
) -> tuple[Any, torch.Tensor]:
    tb, tp = transform_batch(spec, bullets, pad, top_k)
    return net(player, tb, tp)


@torch.no_grad()
def model_metrics(
    net: nn.Module, spec: ArmSpec, tensors: dict[str, torch.Tensor], top_k: int, batch_size: int = 4096
) -> dict[str, Any]:
    net.eval()
    n = int(tensors["player"].shape[0])
    outs = []
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        dist, _v = forward_arm(net, spec, tensors["player"][sl], tensors["bullets"][sl], tensors["pad"][sl], top_k)
        outs.append(dist.logits)
    student_logits = torch.cat(outs, dim=0) if outs else torch.zeros(0, len(ACTIONS))
    # Bullet-density buckets always use the *raw*, unmasked pad -- a masked
    # arm must still be reported against the true bullet density it faced.
    live_counts = live_counts_from_pad(tensors["pad"])
    return compute_metrics(student_logits, tensors["teacher_logits"], tensors["elapsed"], live_counts)


# --------------------------------------------------------------------------- #
# Checkpoint packing -- marked experimental / not production compatible
# --------------------------------------------------------------------------- #
def _pack_ckpt(spec: ArmSpec, args: argparse.Namespace, state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dict": state_dict,
        "arm": spec.name,
        "ckpt_role": "player",
        "objective": spec.objective,
        "model": spec.model,
        "top_k_mask": spec.top_k_mask,
        "top_k": args.top_k,
        "d_model": getattr(args, "d_model", None),
        "hidden": getattr(args, "hidden", None),
        "flat_hidden": getattr(args, "flat_hidden", None),
        "actions": list(ACTIONS),
        "experimental": True,
        "production_compatible": False,
        "provenance": {
            "tool": "tools/phase2_matched_hybrid.py",
            "seed": args.seed,
        },
    }


def train_arm(
    spec: ArmSpec,
    args: argparse.Namespace,
    train_frames: list[Frame],
    held_frames: list[Frame],
    device: torch.device,
    out_dir: Path,
    log_path: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    train_tensors = frames_to_tensors(train_frames, device)
    held_tensors = frames_to_tensors(held_frames, device) if held_frames else None

    net = make_model(spec, args, device)
    param_count = count_trainable_params(net)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    n = int(train_tensors["player"].shape[0])
    # Same seed + same n + same batch_size for every arm -> identical batch
    # order/count across arms; only the model/objective differ.
    gen = torch.Generator().manual_seed(args.seed)

    epoch_history: list[dict[str, Any]] = []
    best_selected: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    latest_state: dict[str, Any] | None = None
    train_metrics: dict[str, Any] = {}
    held_metrics: dict[str, Any] | None = None
    optimizer_steps = 0

    for epoch in range(1, args.epochs + 1):
        net.train()
        for idx in uniform_batch_indices(n, args.batch_size, gen):
            dist, _v = forward_arm(
                net, spec, train_tensors["player"][idx], train_tensors["bullets"][idx], train_tensors["pad"][idx], args.top_k
            )
            loss = arm_loss(spec, dist.logits, train_tensors["teacher_logits"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            optimizer_steps += 1

        train_metrics = model_metrics(net, spec, train_tensors, args.top_k, batch_size=max(args.batch_size, 256))
        held_metrics = (
            model_metrics(net, spec, held_tensors, args.top_k, batch_size=max(args.batch_size, 256))
            if held_tensors is not None
            else None
        )
        epoch_history.append({"arm": spec.name, "epoch": epoch, "train": train_metrics, "held": held_metrics})
        if log_path is not None:
            with Path(log_path).open("a") as f:
                f.write(json.dumps(epoch_history[-1]) + "\n")

        latest_state = copy.deepcopy(net.state_dict())
        if held_metrics is not None:
            # Frame-level-only selection: held agreement, then lower held
            # teacher->student KL, then latest tie -- reused verbatim.
            candidate = {
                "agreement": held_metrics["agreement"],
                "teacher_to_student_kl": held_metrics["teacher_to_student_kl"],
            }
            if should_replace_selected(candidate, best_selected):
                best_selected = candidate
                best_epoch = epoch
                best_state = latest_state
                best_metrics = copy.deepcopy(held_metrics)
        else:
            best_epoch = epoch
            best_state = latest_state

    assert best_state is not None and latest_state is not None  # args.epochs >= 1 in all call sites

    ckpt_path = out_dir / f"{spec.name}.best.pt"
    latest_path = out_dir / f"{spec.name}.latest.pt"
    torch.save(_pack_ckpt(spec, args, best_state), ckpt_path)
    torch.save(_pack_ckpt(spec, args, latest_state), latest_path)

    return {
        "epochs_run": len(epoch_history),
        "epoch_history": epoch_history,
        "final": {"train": train_metrics, "held": held_metrics},
        "selected": {"epoch": best_epoch},
        "selected_metrics": best_metrics,
        "checkpoint": str(ckpt_path),
        "latest_checkpoint": str(latest_path),
        "param_count": param_count,
        "optimizer_steps": optimizer_steps,
    }


def run_experiment(
    args: argparse.Namespace, train_frames: list[Frame], held_frames: list[Frame], device: torch.device
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flat8 hidden is matched exactly once so both flat8 arms share it.
    flat_hidden, flat_match_info = resolve_flat8_hidden(args)
    args.flat_hidden = flat_hidden

    arms_result: dict[str, Any] = {}
    for name in resolve_arms(args.arms):
        spec = ARM_SPECS[name]
        arm_dir = out_dir / name
        res = train_arm(spec, args, train_frames, held_frames, device, arm_dir, log_path=arm_dir / "epochs.jsonl")
        arms_result[name] = {
            "spec": {
                "name": spec.name,
                "model": spec.model,
                "objective": spec.objective,
                "top_k_mask": spec.top_k_mask,
                "description": spec.description,
            },
            **res,
        }

    report_path = out_dir / "report.json"
    report = {
        "seed": args.seed,
        "arms": arms_result,
        "selection": {"criteria": ["held_agreement", "held_kl", "latest"]},
        "dataset": {"train_frames": len(train_frames), "held_frames": len(held_frames)},
        "flat8_match": flat_match_info,
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report

# closed-loop stage begins here
def evaluate_gate(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the pre-registered gate purely from held-out frame metrics.

    Selects the arm with the highest held agreement (frame-level only --
    never touches closed-loop/episode signals), then checks:
      1. that arm's held agreement >= args.gate_agreement_min
      2. its improvement over attn64_soft_t1's held agreement >=
         args.gate_improvement_min (absolute)
      3. every nonzero bullet-density bucket for that arm has held
         agreement >= args.gate_bullet_bucket_min

    Returns a report dict with per-threshold observed/threshold/pass detail,
    the chosen arm, overall gate_pass, a human-readable reason, and
    closed_loop_ran=False (this function never runs the closed loop; it only
    decides whether the caller *may*).
    """
    arms = report["arms"]
    # attn64_soft_t1 is the comparison baseline, not a selection candidate --
    # the gate picks the best of the remaining (non-baseline) arms.
    def selected_held(arm: dict[str, Any]) -> dict[str, Any]:
        # Older/synthetic reports may not carry selected_metrics; retaining
        # this fallback keeps the pure gate helper backwards-compatible.
        return arm.get("selected_metrics") or arm["final"]["held"]

    candidates = [name for name in arms if name != ARM_ATTN64_SOFT_T1]
    chosen_arm = max(candidates, key=lambda name: selected_held(arms[name])["agreement"])
    chosen_held = selected_held(arms[chosen_arm])
    chosen_agreement = chosen_held["agreement"]

    baseline_agreement = selected_held(arms[ARM_ATTN64_SOFT_T1])["agreement"]
    improvement = chosen_agreement - baseline_agreement

    agreement_threshold = args.gate_agreement_min
    agreement_pass = chosen_agreement >= agreement_threshold

    improvement_threshold = args.gate_improvement_min
    improvement_pass = improvement >= improvement_threshold

    bucket_threshold = args.gate_bullet_bucket_min
    bucket_results: dict[str, Any] = {}
    for bucket_name in BULLET_BUCKETS_NONZERO:
        bucket = chosen_held["by_bullet_bucket"].get(bucket_name)
        if bucket is None or not bucket.get("n"):
            continue
        bucket_agreement = bucket["agreement"]
        bucket_results[bucket_name] = {
            "observed": bucket_agreement,
            "threshold": bucket_threshold,
            "pass": bucket_agreement >= bucket_threshold,
            "n": bucket["n"],
        }
    bucket_pass = all(b["pass"] for b in bucket_results.values())

    thresholds = {
        "held_agreement_min": {
            "observed": chosen_agreement,
            "threshold": agreement_threshold,
            "pass": agreement_pass,
        },
        "improvement_over_attn64_soft_t1_min": {
            "observed": improvement,
            "threshold": improvement_threshold,
            "pass": improvement_pass,
        },
        "bullet_bucket_min_agreement": {
            "threshold": bucket_threshold,
            "pass": bucket_pass,
            "buckets": bucket_results,
        },
    }

    gate_pass = agreement_pass and improvement_pass and bucket_pass
    if gate_pass:
        reason = f"gate passed: {chosen_arm} met all thresholds"
    else:
        failed = [k for k, v in thresholds.items() if not v["pass"]]
        reason = f"gate failed: {chosen_arm} did not meet threshold(s): {', '.join(failed)}"

    return {
        "chosen_arm": chosen_arm,
        "gate_pass": gate_pass,
        "thresholds": thresholds,
        "reason": reason,
        "closed_loop_ran": False,
    }


@torch.no_grad()
def closed_loop_arm_times(
    net: nn.Module, spec: ArmSpec, device: torch.device, seeds: list[int], max_steps: int, top_k: int
) -> list[float]:
    """Deterministic argmax rollout vs the built-in scripted env only (no PPO,
    no learned Spawner, no NAS writes) -- mirrors
    tools.phase2_distill_v1_to_v4.closed_loop_times but dispatches through
    forward_arm so it also works for the flat8/top-k-masked arms."""
    net.eval()
    times: list[float] = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        for _ in range(max_steps):
            p, b, m = encode_obs(env)
            pt = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
            bt = torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
            mt = torch.tensor(m, dtype=torch.bool, device=device).unsqueeze(0)
            dist, _v = forward_arm(net, spec, pt, bt, mt, top_k)
            action = int(dist.probs.argmax(dim=-1).item())
            _obs, _r, done, info = env.step(action)
            if done:
                times.append(float(info.get("elapsed", env.elapsed)))
                break
        else:
            times.append(float(env.elapsed))
    return times


def run_gated_closed_loop(report: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Evaluate the pre-registered gate; only if it passes, reload the chosen
    arm's checkpoint and run the deterministic scripted closed loop over
    seeds [args.closed_loop_seed_start, args.closed_loop_seed_start +
    args.closed_loop_num_seeds). If the gate fails, no env is ever
    constructed or stepped."""
    gate = evaluate_gate(report, args)
    if not gate["gate_pass"]:
        return gate

    chosen_arm = gate["chosen_arm"]
    spec = ARM_SPECS[chosen_arm]
    ckpt_path = Path(report["arms"][chosen_arm]["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = make_model(spec, args, device)
    net.load_state_dict(ckpt["state_dict"])

    seeds = list(range(args.closed_loop_seed_start, args.closed_loop_seed_start + args.closed_loop_num_seeds))
    times = closed_loop_arm_times(net, spec, device, seeds, args.closed_loop_max_steps, args.top_k)

    gate["closed_loop_ran"] = True
    gate["closed_loop_seeds"] = seeds
    per_seed = {str(seed): elapsed for seed, elapsed in zip(seeds, times)}
    gate["closed_loop"] = {
        "mean": sum(times) / len(times) if times else 0.0,
        "median": float(statistics.median(times)) if times else 0.0,
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "min": min(times) if times else 0.0,
        "max": max(times) if times else 0.0,
        "n": len(times),
        "per_seed": per_seed,
    }
    return gate


def main() -> None:
    args = apply_mode_defaults(build_parser().parse_args())
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    teacher = load_teacher(args.teacher, device)
    py_rng = random.Random(args.seed)
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
    if not train_frames:
        raise RuntimeError("no training frames collected -- increase --collect-episodes")

    report = run_experiment(args, train_frames, held_frames, device)
    gate = run_gated_closed_loop(report, args, device)
    report["gate"] = gate
    report_path = Path(report["report_path"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {report['report_path']}", flush=True)
    print(f"[gate] chosen_arm={gate['chosen_arm']} gate_pass={gate['gate_pass']} reason={gate['reason']}", flush=True)
    if gate["closed_loop_ran"]:
        print(f"[closed_loop] seeds={gate['closed_loop_seeds'][0]}..{gate['closed_loop_seeds'][-1]} "
              f"mean={gate['closed_loop']['mean']:.2f}s median={gate['closed_loop']['median']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
