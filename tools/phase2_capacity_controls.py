#!/usr/bin/env python3
"""Phase 2 capacity-control experiment harness (offline, frame-level only).

This tool answers a narrower question than tools/phase2_distill_v1_to_v4.py:
*why* does PlayerV4 disagree with the PlayerV1 teacher on held-out frames?
Is it (a) an architecture/capacity problem in the 64-slot attention encoder,
(b) a labeling/objective problem (hard vs soft targets, temperature), or
(c) neither (the student just needs more data/epochs)?

It never touches the NAS, never runs an adversary network, and never gates a
checkpoint on episode outcomes. Every arm is judged purely on frame-level
agreement / cross-entropy / KL against the *same* frozen teacher-logit
dataset (collected once via tools.phase2_distill_v1_to_v4.collect_dataset),
split by episode into train/held sets. Selection between epochs uses only
held-out frame agreement (then KL, then latest-wins on exact ties) — see
should_replace_selected(). This keeps the experiment fast, deterministic and
decoupled from the real trainer/encoder, which are both left untouched.

Arms (see ARM_SPECS):
  tiny_overfit    - PlayerV4, hard labels, trained on a tiny frame slice with
                    no held-out set. Sanity/positive-control: if the model
                    can't memorize a handful of frames, something is broken
                    upstream of capacity (data pipeline, loss, optimizer).
  top8_mask       - PlayerV4, hard labels, bullets beyond the nearest 8 are
                    replaced with pad slots before *both* train and eval
                    forward passes (apply_top_k_mask). Tests whether the
                    64-slot attention set is the bottleneck: if agreement
                    barely drops vs the unmasked hard-label arm, the extra
                    56 slots are not carrying much signal.
  flat_top8_mlp   - A deliberately different, non-attention architecture
                    (FlatTop8MLP) that flattens player + top-8 bullet
                    features + a live/pad mask into one vector and feeds a
                    plain MLP. Positive control: an architecture that is
                    *not* the attention encoder, to check whether attention
                    itself (vs. capacity/objective) is the limiting factor.
  obj_hard        - PlayerV4, hard labels (teacher argmax), the objective
                    baseline the other objective arms are compared against.
  obj_soft_t1     - PlayerV4, soft labels at temperature 1.0 (full teacher
                    softmax distribution).
  obj_soft_lowt   - PlayerV4, soft labels at a low temperature (--low-
                    temperature, sharper than 1.0, approaching hard labels).

Sampling is uniform over frames for every arm (SAMPLING_POLICY == "uniform")
-- no elapsed-bucket reweighting like the main distill script uses, so the
capacity/objective comparison isn't confounded by a second knob.

IMPORTANT: this script reports agreement/CE/KL numbers; it does not itself
claim any conclusion about Phase 2 capacity. See
docs/phase2_capacity_controls_experiment.md for the experiment contract and
the explicit "no results yet" statement.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from qrokkun_env.agents.obs_v4 import (
    BULLET_FEAT_V4,
    MAX_BULLETS_V4,
    PAD_RADIUS,
    PLAYER_FEAT_V4,
)
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.env import ACTIONS

# Reuse the teacher-collection / frame plumbing from the main distill script
# instead of re-implementing it. Deliberately do NOT import episode-outcome or
# adversary-network helpers; this harness only ever looks at frame-level
# agreement/CE/KL against the frozen teacher logits.
from tools.phase2_distill_v1_to_v4 import (  # noqa: E402
    Frame,
    bucket_of,
    collect_dataset,
    frames_to_tensors,
    load_teacher,
)

# --------------------------------------------------------------------------- #
# Arm catalogue
# --------------------------------------------------------------------------- #
MODEL_PLAYER_V4 = "player_v4"
MODEL_FLAT_TOP8_MLP = "flat_top8_mlp"

LABEL_HARD = "hard"
LABEL_SOFT = "soft"

ARM_TINY_OVERFIT = "tiny_overfit"
ARM_TOP8_MASK = "top8_mask"
ARM_FLAT_TOP8_MLP = "flat_top8_mlp"
ARM_OBJ_HARD = "obj_hard"
ARM_OBJ_SOFT_T1 = "obj_soft_t1"
ARM_OBJ_SOFT_LOWT = "obj_soft_lowt"

SAMPLING_POLICY = "uniform"
TOP_K = 8


@dataclass(frozen=True)
class ArmSpec:
    name: str
    model: str
    label_mode: str
    description: str
    top_k_mask: bool = False
    temperature: float | None = None


ARM_SPECS: dict[str, ArmSpec] = {
    ARM_TINY_OVERFIT: ArmSpec(
        name=ARM_TINY_OVERFIT,
        model=MODEL_PLAYER_V4,
        label_mode=LABEL_HARD,
        description=(
            "Positive control: PlayerV4 trained on a tiny hard-label frame "
            "slice with no held-out set; should reach near-perfect train "
            "agreement if the data/loss/optimizer plumbing is sound."
        ),
    ),
    ARM_TOP8_MASK: ArmSpec(
        name=ARM_TOP8_MASK,
        model=MODEL_PLAYER_V4,
        label_mode=LABEL_HARD,
        description=(
            "Capacity control: PlayerV4 with bullet slots beyond the "
            "nearest 8 masked to pad for both train and eval, isolating "
            "whether the 64-slot bullet set is the bottleneck."
        ),
        top_k_mask=True,
    ),
    ARM_FLAT_TOP8_MLP: ArmSpec(
        name=ARM_FLAT_TOP8_MLP,
        model=MODEL_FLAT_TOP8_MLP,
        label_mode=LABEL_HARD,
        description=(
            "Positive control with a non-attention architecture: flattens "
            "player + top-8 bullet features + live mask into a plain MLP, "
            "to separate 'attention encoder' from 'capacity/objective'."
        ),
    ),
    ARM_OBJ_HARD: ArmSpec(
        name=ARM_OBJ_HARD,
        model=MODEL_PLAYER_V4,
        label_mode=LABEL_HARD,
        description="Objective baseline: PlayerV4 trained on teacher-argmax hard labels.",
    ),
    ARM_OBJ_SOFT_T1: ArmSpec(
        name=ARM_OBJ_SOFT_T1,
        model=MODEL_PLAYER_V4,
        label_mode=LABEL_SOFT,
        description="Objective control: PlayerV4 trained on teacher softmax at temperature 1.0.",
        temperature=1.0,
    ),
    ARM_OBJ_SOFT_LOWT: ArmSpec(
        name=ARM_OBJ_SOFT_LOWT,
        model=MODEL_PLAYER_V4,
        label_mode=LABEL_SOFT,
        description=(
            "Objective control: PlayerV4 trained on teacher softmax at a low "
            "temperature (--low-temperature), approaching hard labels while "
            "keeping the loss soft."
        ),
        temperature=None,
    ),
}
ALL_ARMS: tuple[str, ...] = (
    ARM_TINY_OVERFIT,
    ARM_TOP8_MASK,
    ARM_FLAT_TOP8_MLP,
    ARM_OBJ_HARD,
    ARM_OBJ_SOFT_T1,
    ARM_OBJ_SOFT_LOWT,
)

CORE_METRIC_KEYS: tuple[str, ...] = (
    "n",
    "agreement",
    "hard_ce",
    "soft_ce",
    "teacher_entropy",
    "student_entropy",
    "teacher_to_student_kl",
)
ELAPSED_BUCKETS: tuple[str, ...] = ("early", "mid", "late")
BULLET_BUCKETS: tuple[str, ...] = ("0", "1-7", "8-19", "20+")


def resolve_temperature(spec: ArmSpec, args: argparse.Namespace) -> float | None:
    if spec.temperature is not None:
        return spec.temperature
    if spec.label_mode == LABEL_SOFT:
        return args.low_temperature
    return None


def resolve_arms(csv: str) -> list[str]:
    if csv == "all":
        return list(ALL_ARMS)
    names = [x.strip() for x in csv.split(",") if x.strip()]
    unknown = [n for n in names if n not in ARM_SPECS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; choose from {list(ALL_ARMS)} or 'all'")
    return names


# --------------------------------------------------------------------------- #
# Sampling (uniform for every arm -- no bucket reweighting)
# --------------------------------------------------------------------------- #
def sampling_weights(spec: ArmSpec, frames: list[Frame], args: argparse.Namespace) -> None:
    """Every arm samples uniformly; this always returns None (no per-frame weights)."""
    del spec, frames, args
    return None


def uniform_batch_indices(n: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[start : start + batch_size] for start in range(0, n, batch_size)]


def split_frames_by_episode(frames: list[Frame], held_frac: float, seed: int) -> tuple[list[Frame], list[Frame]]:
    episodes = sorted({f.episode for f in frames})
    rng = random.Random(seed)
    order = list(episodes)
    rng.shuffle(order)
    n_held = max(1, round(len(order) * held_frac)) if len(order) > 1 else 0
    held_eps = set(order[:n_held])
    train = [f for f in frames if f.episode not in held_eps]
    held = [f for f in frames if f.episode in held_eps]
    return train, held


def resolve_arm_data(
    spec: ArmSpec, train_frames: list[Frame], held_frames: list[Frame], args: argparse.Namespace
) -> tuple[list[Frame], list[Frame]]:
    if spec.name == ARM_TINY_OVERFIT:
        return list(train_frames[: args.tiny_frames]), []
    return train_frames, held_frames


# --------------------------------------------------------------------------- #
# Top-k bullet mask (capacity control) and forward dispatch
# --------------------------------------------------------------------------- #
def apply_top_k_mask(bullets: torch.Tensor, pad: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return copies with slots >= k forced to the pad-bullet convention."""
    mb = bullets.clone()
    mp = pad.clone()
    mb[:, k:, :] = 0.0
    mb[:, k:, 4] = PAD_RADIUS
    mp[:, k:] = True
    return mb, mp


def transform_batch(spec: ArmSpec, bullets: torch.Tensor, pad: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.top_k_mask:
        return apply_top_k_mask(bullets, pad, k)
    return bullets, pad


def forward_arm(
    net: nn.Module, spec: ArmSpec, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor, top_k: int
) -> tuple[Categorical, torch.Tensor]:
    tb, tp = transform_batch(spec, bullets, pad, top_k)
    return net(player, tb, tp)


# --------------------------------------------------------------------------- #
# Flat top-8 MLP positive control (non-attention architecture)
# --------------------------------------------------------------------------- #
class FlatTop8MLP(nn.Module):
    def __init__(self, top_k: int = TOP_K, hidden: int = 256) -> None:
        super().__init__()
        self.top_k = top_k
        self.in_dim = PLAYER_FEAT_V4 + top_k * BULLET_FEAT_V4 + top_k
        self.body = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, len(ACTIONS))
        self.value = nn.Linear(hidden, 1)

    def features(self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        b = bullets[:, : self.top_k]
        p = pad[:, : self.top_k]
        live = (~p).float()
        b = b * live.unsqueeze(-1)
        flat_b = b.reshape(b.shape[0], -1)
        return torch.cat([player, flat_b, live], dim=-1)

    def forward(
        self, player: torch.Tensor, bullets: torch.Tensor, pad: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        h = self.body(self.features(player, bullets, pad))
        return Categorical(logits=self.policy(h)), self.value(h).squeeze(-1)


def make_model(spec: ArmSpec, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if spec.model == MODEL_FLAT_TOP8_MLP:
        net: nn.Module = FlatTop8MLP(top_k=args.top_k, hidden=args.hidden)
    else:
        net = PlayerV4(d_model=args.d_model, hidden=args.hidden)
    return net.to(device)


# --------------------------------------------------------------------------- #
# Objective controls (hard vs soft-label loss)
# --------------------------------------------------------------------------- #
def arm_loss(
    spec: ArmSpec, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    if spec.label_mode == LABEL_HARD:
        return F.cross_entropy(student_logits, teacher_logits.argmax(-1))
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    return -(teacher_probs * student_logp).sum(-1).mean()


# --------------------------------------------------------------------------- #
# Metrics: agreement / CE / entropy / KL, plus elapsed and bullet-density buckets
# --------------------------------------------------------------------------- #
def _core_metrics(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> dict[str, Any]:
    n = int(student_logits.shape[0])
    if n == 0:
        empty: dict[str, Any] = {k: None for k in CORE_METRIC_KEYS}
        empty["n"] = 0
        return empty
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    hard_target = teacher_logits.argmax(-1)

    hard_ce = F.cross_entropy(student_logits, hard_target).item()
    soft_ce = -(teacher_probs * student_logp).sum(-1).mean().item()
    teacher_entropy = -(teacher_probs * teacher_logp).sum(-1).mean().item()
    student_entropy = -(student_probs * student_logp).sum(-1).mean().item()
    agreement = (student_logits.argmax(-1) == hard_target).float().mean().item()
    kl = soft_ce - teacher_entropy
    return {
        "n": n,
        "agreement": agreement,
        "hard_ce": hard_ce,
        "soft_ce": soft_ce,
        "teacher_entropy": teacher_entropy,
        "student_entropy": student_entropy,
        "teacher_to_student_kl": kl,
    }


def live_bullet_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count < 8:
        return "1-7"
    if count < 20:
        return "8-19"
    return "20+"


def live_counts_from_pad(pad: torch.Tensor) -> torch.Tensor:
    return (~pad).sum(dim=-1)


def compute_metrics(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, elapsed: torch.Tensor, live_counts: torch.Tensor
) -> dict[str, Any]:
    base = _core_metrics(student_logits, teacher_logits)
    n = int(student_logits.shape[0])
    by_elapsed: dict[str, Any] = {}
    by_bullet: dict[str, Any] = {}
    if n > 0:
        elapsed_names = [bucket_of(float(e)) for e in elapsed.tolist()]
        bullet_names = [live_bullet_bucket(int(c)) for c in live_counts.tolist()]
        for name in ELAPSED_BUCKETS:
            idx = [i for i, nm in enumerate(elapsed_names) if nm == name]
            if idx:
                sel = torch.tensor(idx, dtype=torch.long)
                by_elapsed[name] = _core_metrics(student_logits[sel], teacher_logits[sel])
        for name in BULLET_BUCKETS:
            idx = [i for i, nm in enumerate(bullet_names) if nm == name]
            if idx:
                sel = torch.tensor(idx, dtype=torch.long)
                by_bullet[name] = _core_metrics(student_logits[sel], teacher_logits[sel])
    base["by_elapsed_bucket"] = by_elapsed
    base["by_bullet_bucket"] = by_bullet
    return base


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
# Selection (frame-level only) and acceptance reporting
# --------------------------------------------------------------------------- #
def should_replace_selected(candidate: dict[str, Any] | None, current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    if candidate is None:
        return False
    if candidate["agreement"] != current["agreement"]:
        return candidate["agreement"] > current["agreement"]
    # Equal agreement: prefer lower KL; on an exact tie, keep the latest
    # candidate (>= rather than >).
    return candidate["teacher_to_student_kl"] <= current["teacher_to_student_kl"]


def acceptance_report(observed: float | None, target: float) -> dict[str, Any]:
    if observed is None:
        return {
            "metric": "train_agreement",
            "target": target,
            "observed": None,
            "met": None,
            "verdict": "not_evaluated",
            "note": "no observation available (e.g. zero training epochs run)",
        }
    met = observed >= target
    return {
        "metric": "train_agreement",
        "target": target,
        "observed": observed,
        "met": met,
        "verdict": "met" if met else "below_target",
        "note": (
            "tiny-overfit positive control passed"
            if met
            else (
                "tiny-overfit positive control missed target -- report only, "
                "do not treat as proof of a broken architecture; investigate "
                "data/loss/optimizer plumbing before drawing conclusions"
            )
        ),
    }


# --------------------------------------------------------------------------- #
# CLI / experiment orchestration
# --------------------------------------------------------------------------- #
_FULL_DEFAULTS = {
    "tiny_frames": 2000,
    "epochs": 40,
    "batch_size": 256,
    "collect_episodes": 240,
    "max_steps": 60 * 70,
    "d_model": 128,
    "hidden": 256,
}
_QUICK_DEFAULTS = {
    "tiny_frames": 200,
    "epochs": 5,
    "batch_size": 64,
    "collect_episodes": 8,
    "max_steps": 300,
    "d_model": 16,
    "hidden": 16,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", type=Path, required=True, help="PlayerV1 ckpt, e.g. player_gpu.pt")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--arms", type=str, default="all", help="'all' or a comma-separated list of arm names")
    ap.add_argument("--quick", action="store_true", help="shrink defaults for a fast smoke run")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--collect-episodes", type=int, default=None)
    ap.add_argument("--collect-seed-start", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--frames-per-episode-cap", type=int, default=700)
    ap.add_argument("--held-out-frac", type=float, default=0.12)

    ap.add_argument("--tiny-frames", type=int, default=None)
    ap.add_argument("--tiny-target-agreement", type=float, default=0.95)
    ap.add_argument("--low-temperature", type=float, default=0.1)
    ap.add_argument("--top-k", type=int, default=TOP_K)

    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    return ap


def apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = _QUICK_DEFAULTS if args.quick else _FULL_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _pack_ckpt(spec: ArmSpec, args: argparse.Namespace, state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dict": state_dict,
        "arm": spec.name,
        "ckpt_role": "player",
        "label_mode": spec.label_mode,
        "model": spec.model,
        "top_k": args.top_k,
        "d_model": getattr(args, "d_model", None),
        "hidden": getattr(args, "hidden", None),
        "actions": list(ACTIONS),
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
    train_f, held_f = resolve_arm_data(spec, train_frames, held_frames, args)
    train_tensors = frames_to_tensors(train_f, device)
    held_tensors = frames_to_tensors(held_f, device) if held_f else None

    net = make_model(spec, args, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    temperature = resolve_temperature(spec, args) or 1.0

    n = int(train_tensors["player"].shape[0])
    gen = torch.Generator().manual_seed(args.seed)

    epoch_history: list[dict[str, Any]] = []
    best_selected: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    best_epoch: int | None = None
    latest_state: dict[str, Any] | None = None
    train_metrics: dict[str, Any] = {}
    held_metrics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        net.train()
        for idx in uniform_batch_indices(n, args.batch_size, gen):
            dist, _v = forward_arm(
                net, spec, train_tensors["player"][idx], train_tensors["bullets"][idx], train_tensors["pad"][idx], args.top_k
            )
            loss = arm_loss(spec, dist.logits, train_tensors["teacher_logits"][idx], temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_metrics = model_metrics(net, spec, train_tensors, args.top_k, batch_size=max(args.batch_size, 256))
        held_metrics = (
            model_metrics(net, spec, held_tensors, args.top_k, batch_size=max(args.batch_size, 256))
            if held_tensors is not None
            else None
        )
        epoch_history.append(
            {"arm": spec.name, "epoch": epoch, "train": train_metrics, "held": held_metrics}
        )
        if log_path is not None:
            with Path(log_path).open("a") as f:
                f.write(json.dumps(epoch_history[-1]) + "\n")

        latest_state = copy.deepcopy(net.state_dict())
        if held_metrics is not None:
            candidate = {
                "agreement": held_metrics["agreement"],
                "teacher_to_student_kl": held_metrics["teacher_to_student_kl"],
            }
            if should_replace_selected(candidate, best_selected):
                best_selected = candidate
                best_epoch = epoch
                best_state = latest_state
        else:
            # No held set (tiny arm): there is nothing to select against, so
            # the latest epoch is always the reported one.
            best_epoch = epoch
            best_state = latest_state

    assert best_state is not None and latest_state is not None  # args.epochs >= 1 in all call sites

    ckpt_path = out_dir / f"{spec.name}.best.pt"
    latest_path = out_dir / f"{spec.name}.latest.pt"
    torch.save(_pack_ckpt(spec, args, best_state), ckpt_path)
    torch.save(_pack_ckpt(spec, args, latest_state), latest_path)

    result: dict[str, Any] = {
        "epochs_run": len(epoch_history),
        "epoch_history": epoch_history,
        "final": {"train": train_metrics, "held": held_metrics},
        "selected": {"epoch": best_epoch},
        "checkpoint": str(ckpt_path),
        "latest_checkpoint": str(latest_path),
    }
    if spec.name == ARM_TINY_OVERFIT:
        result["acceptance"] = acceptance_report(train_metrics.get("agreement"), args.tiny_target_agreement)
    else:
        result["acceptance"] = None
    return result


def run_experiment(
    args: argparse.Namespace, train_frames: list[Frame], held_frames: list[Frame], device: torch.device
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms_result: dict[str, Any] = {}
    for name in resolve_arms(args.arms):
        spec = ARM_SPECS[name]
        arm_dir = out_dir / name
        res = train_arm(spec, args, train_frames, held_frames, device, arm_dir, log_path=arm_dir / "epochs.jsonl")
        arms_result[name] = {
            "spec": {
                "name": spec.name,
                "model": spec.model,
                "label_mode": spec.label_mode,
                "top_k_mask": spec.top_k_mask,
                "temperature": spec.temperature,
                "description": spec.description,
            },
            **res,
        }

    report_path = out_dir / "report.json"
    report = {
        "seed": args.seed,
        "sampling": SAMPLING_POLICY,
        "arms": arms_result,
        "selection": {"criteria": ["held_agreement", "held_kl", "latest"]},
        "dataset": {"train_frames": len(train_frames), "held_frames": len(held_frames)},
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


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
    print(f"wrote {report['report_path']}", flush=True)
    for name, arm in report["arms"].items():
        held = arm["final"]["held"]
        agree = held["agreement"] if held else arm["final"]["train"]["agreement"]
        print(f"[{name}] selected_epoch={arm['selected']['epoch']} agreement={agree:.3f}", flush=True)


if __name__ == "__main__":
    main()
