#!/usr/bin/env python3
"""Phase 2: distill PlayerV1 (teacher) -> PlayerV4 (student) via behavior cloning.

Teacher: PlayerV1 + qrokkun_env.obs.vectorize, loaded from --teacher (e.g.
artifacts/nas_tmp_runs/player_gpu.pt). Rolls argmax vs the scripted env only
(Qrokkun26Env.step() already runs the built-in scripted spawner — no learned
Spawner is used anywhere in this script).

Student: PlayerV4 (agents.player_v4 + agents.obs_v4.encode_obs). Trained by
supervised distillation (soft-label cross-entropy / KL) from teacher logits.
Never load V1 weights into V4 — the two nets have unrelated architectures and
obs contracts; the student is always randomly initialized.

No PPO, no learned Spawner, no --random-fraction, no gamma/entropy A/B: this
is pure offline behavior cloning against the same env used by both_v4.py
(left untouched).

Pipeline:
  1. Collect: for --collect-episodes seeds, roll the teacher (argmax) vs the
     scripted env for up to --max-steps frames. At every frame, record BOTH
     observations of the same env state: obs.vectorize(env) -> teacher logits,
     and obs_v4.encode_obs(env) -> (player, bullets, pad) for the student. If
     an episode has more than --frames-per-episode-cap frames, subsample with
     an evenly spaced stride so opening/mid/late game are all represented
     (rather than truncating to the first N frames).
  2. Split by *episode* (not frame) into train / held-out so held-out frame
     metrics never leak collection episodes into training.
  3. Train: soft-label CE from teacher softmax(logits / temperature) to
     student log-softmax, with a per-frame sampling weight that balances the
     early/mid/late elapsed buckets so long, late-game-heavy episodes do not
     drown in short opening-only ones (and vice versa).
  4. Periodically evaluate the student closed-loop (argmax policy, scripted
     env, deterministic) on a held-out *seed* set (SELECTION_SEEDS, disjoint
     from both collection and the final report seeds) and save the ckpt with
     the best mean survival there. Beating the flee baseline is NOT required
     to save a ckpt.
  5. Final report: held-out frame agreement/cross-entropy, closed-loop
     mean/median survival on REPORT_SEEDS (3000-3029, 4200 steps == 70s), and
     student policy entropy.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from qrokkun_env.agents.obs_v4 import encode_obs
from qrokkun_env.agents.player_v1 import PlayerV1
from qrokkun_env.agents.player_v4 import PlayerV4
from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.policies import FleeNearestBullet

# Fixed, disjoint seed ranges: collection never touches these, so both the
# selection signal and the final report are honest held-out closed-loop evals.
SELECTION_SEEDS: tuple[int, ...] = tuple(range(8000, 8012))
REPORT_SEEDS: tuple[int, ...] = tuple(range(3000, 3030))
REPORT_MAX_STEPS = 60 * 70  # 4200 == 70s @ 60Hz, matches both_v4 PAIRED_EVAL_SEEDS cap
DEFAULT_MAX_STEPS = 60 * 70

EARLY_S = 20.0
MID_S = 45.0


def bucket_of(elapsed: float) -> str:
    if elapsed < EARLY_S:
        return "early"
    if elapsed < MID_S:
        return "mid"
    return "late"


@dataclass
class Frame:
    player: np.ndarray
    bullets: np.ndarray
    pad: np.ndarray
    teacher_logits: np.ndarray
    elapsed: float
    episode: int


# --------------------------------------------------------------------------- #
# Teacher
# --------------------------------------------------------------------------- #
def load_teacher(path: Path, device: torch.device) -> PlayerV1:
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck.get("obs_dim", OBS_DIM) == OBS_DIM, ck.get("obs_dim")
    assert list(ck["actions"]) == list(ACTIONS), ck["actions"]
    net = PlayerV1(hidden=int(ck.get("hidden", 256)))
    net.load_state_dict(ck["state_dict"])  # NEVER load V1 weights into PlayerV4.
    net.to(device)
    net.eval()
    return net


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_episode(
    teacher: PlayerV1, device: torch.device, seed: int, max_steps: int, frames_cap: int, episode_idx: int
) -> list[Frame]:
    env = Qrokkun26Env(seed=seed)
    env.reset(seed=seed)
    raw: list[Frame] = []
    for _ in range(max_steps):
        x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
        dist, _v = teacher(x)
        logits = dist.logits.detach().cpu().numpy().astype(np.float32)
        action = int(logits.argmax())
        p4, b4, m4 = encode_obs(env)
        raw.append(Frame(p4, b4, m4, logits, float(env.elapsed), episode_idx))
        _obs, _r, done, _info = env.step(action)
        if done:
            break
    if frames_cap > 0 and len(raw) > frames_cap:
        # Evenly spaced stride over the *whole* episode (not the first N frames)
        # so opening/mid/late elapsed are all represented even after capping.
        idx = sorted({int(round(i)) for i in np.linspace(0, len(raw) - 1, num=frames_cap)})
        raw = [raw[i] for i in idx]
    return raw


def collect_dataset(
    teacher: PlayerV1,
    device: torch.device,
    *,
    n_episodes: int,
    seed_start: int,
    max_steps: int,
    frames_cap: int,
    held_out_frac: float,
    rng: random.Random,
) -> tuple[list[Frame], list[Frame]]:
    order = list(range(n_episodes))
    rng.shuffle(order)
    n_held_episodes = max(1, int(round(n_episodes * held_out_frac))) if n_episodes > 1 else 0
    held_episode_ids = set(order[:n_held_episodes])

    train_frames: list[Frame] = []
    held_frames: list[Frame] = []
    ep_lengths: list[int] = []
    for ep_idx in range(n_episodes):
        seed = seed_start + ep_idx
        frames = collect_episode(teacher, device, seed, max_steps, frames_cap, ep_idx)
        ep_lengths.append(len(frames))
        if ep_idx in held_episode_ids:
            held_frames.extend(frames)
        else:
            train_frames.extend(frames)
    print(
        f"collected {n_episodes} episodes ({n_episodes - len(held_episode_ids)} train / "
        f"{len(held_episode_ids)} held-out) — frames train={len(train_frames)} "
        f"held={len(held_frames)} mean_ep_len={sum(ep_lengths) / max(len(ep_lengths), 1):.1f}",
        flush=True,
    )
    return train_frames, held_frames


def frames_to_tensors(frames: list[Frame], device: torch.device) -> dict[str, torch.Tensor]:
    player = np.stack([f.player for f in frames]).astype(np.float32)
    bullets = np.stack([f.bullets for f in frames]).astype(np.float32)
    pad = np.stack([f.pad for f in frames]).astype(bool)
    logits = np.stack([f.teacher_logits for f in frames]).astype(np.float32)
    elapsed = np.asarray([f.elapsed for f in frames], dtype=np.float32)
    return {
        "player": torch.from_numpy(player).to(device),
        "bullets": torch.from_numpy(bullets).to(device),
        "pad": torch.from_numpy(pad).to(device),
        "teacher_logits": torch.from_numpy(logits).to(device),
        "elapsed": torch.from_numpy(elapsed).to(device),
    }


def bucket_weights(frames: list[Frame]) -> torch.Tensor:
    """Inverse-frequency weight per frame over {early, mid, late} elapsed buckets."""
    counts = {"early": 0, "mid": 0, "late": 0}
    buckets = [bucket_of(f.elapsed) for f in frames]
    for b in buckets:
        counts[b] += 1
    weights = torch.tensor(
        [1.0 / max(counts[b], 1) for b in buckets], dtype=torch.float32
    )
    return weights


# --------------------------------------------------------------------------- #
# Distillation loss / frame-level eval
# --------------------------------------------------------------------------- #
def soft_ce_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_logp = F.log_softmax(student_logits, dim=-1)
    return -(teacher_probs * student_logp).sum(-1).mean()


@torch.no_grad()
def frame_level_eval(
    student: PlayerV4, tensors: dict[str, torch.Tensor], batch_size: int = 4096
) -> dict[str, float]:
    student.eval()
    n = tensors["player"].shape[0]
    if n == 0:
        return {"agreement": float("nan"), "cross_entropy": float("nan"), "entropy": float("nan"), "n": 0}
    agree = 0
    ce_sum = 0.0
    ent_sum = 0.0
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        dist, _v = student(tensors["player"][sl], tensors["bullets"][sl], tensors["pad"][sl])
        student_logits = dist.logits
        teacher_logits = tensors["teacher_logits"][sl]
        bsz = student_logits.shape[0]
        ce_sum += float(soft_ce_loss(student_logits, teacher_logits, 1.0).item()) * bsz
        agree += int((student_logits.argmax(-1) == teacher_logits.argmax(-1)).sum().item())
        ent_sum += float(dist.entropy().sum().item())
    return {
        "agreement": agree / n,
        "cross_entropy": ce_sum / n,
        "entropy": ent_sum / n,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Closed-loop eval (student vs scripted env only — no Spawner net anywhere)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def closed_loop_times(student: PlayerV4, device: torch.device, seeds, max_steps: int) -> list[float]:
    student.eval()
    times: list[float] = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        for _ in range(max_steps):
            p, b, m = encode_obs(env)
            pt = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
            bt = torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
            mt = torch.tensor(m, dtype=torch.bool, device=device).unsqueeze(0)
            dist, _v = student(pt, bt, mt)
            action = int(dist.probs.argmax(dim=-1).item())
            _obs, _r, done, info = env.step(action)
            if done:
                times.append(float(info.get("elapsed", env.elapsed)))
                break
        else:
            times.append(float(env.elapsed))
    return times


def flee_closed_loop_times(seeds, max_steps: int) -> list[float]:
    """Informational-only baseline (never gates ckpt selection)."""
    flee = FleeNearestBullet()
    times: list[float] = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        for _ in range(max_steps):
            action = ACTIONS.index(flee.act(env))
            _obs, _r, done, info = env.step(action)
            if done:
                times.append(float(info.get("elapsed", env.elapsed)))
                break
        else:
            times.append(float(env.elapsed))
    return times


def summarize(times: list[float]) -> dict[str, float]:
    n = len(times)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": sum(times) / n,
        "median": float(statistics.median(times)),
        "std": float(statistics.pstdev(times)) if n > 1 else 0.0,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Checkpoint packing (mirrors qrokkun_env.train.checkpoints_v4 conventions)
# --------------------------------------------------------------------------- #
def pack_student_ckpt(
    net: PlayerV4, *, d_model: int, hidden: int, epoch: int, eval_metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "state_dict": net.state_dict(),
        "d_model": d_model,
        "hidden": hidden,
        "algo": "phase2-distill-player-v1-to-v4",
        "ckpt_role": "player",
        "actions": list(ACTIONS),
        "epoch": epoch,
        "eval": dict(eval_metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", type=Path, required=True, help="PlayerV1 ckpt, e.g. artifacts/nas_tmp_runs/player_gpu.pt")
    ap.add_argument(
        "--out", type=Path, default=Path("artifacts/nas_tmp_runs/phase2_distill/player_v4_distill.pt")
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--collect-episodes", type=int, default=240)
    ap.add_argument("--collect-seed-start", type=int, default=20000, help="offset kept clear of eval seed ranges")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="cap per collection episode (60*70=4200)")
    ap.add_argument("--frames-per-episode-cap", type=int, default=700, help="0 disables subsampling")
    ap.add_argument("--held-out-episode-frac", type=float, default=0.12)

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0, help="teacher softmax temperature for the training loss")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hours", type=float, default=0.0, help="if > 0, train by wall-clock budget instead of --epochs")
    ap.add_argument("--eval-every", type=int, default=2, help="epochs between selection/report evals")
    ap.add_argument("--selection-max-steps", type=int, default=REPORT_MAX_STEPS)
    ap.add_argument("--log", type=Path, default=None, help="jsonl log; defaults to <out>.jsonl")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_rng = random.Random(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log or args.out.with_suffix(".jsonl")
    best_path = args.out
    report_path = args.out.with_suffix(".report.json")

    teacher = load_teacher(args.teacher, device)

    # Informational only — beating flee is never required to save a ckpt.
    flee_times = flee_closed_loop_times(REPORT_SEEDS, REPORT_MAX_STEPS)
    flee_stats = summarize(flee_times)
    print(f"[baseline] flee|scripted mean={flee_stats['mean']:.2f}s median={flee_stats['median']:.2f}s", flush=True)

    train_frames, held_frames = collect_dataset(
        teacher,
        device,
        n_episodes=args.collect_episodes,
        seed_start=args.collect_seed_start,
        max_steps=args.max_steps,
        frames_cap=args.frames_per_episode_cap,
        held_out_frac=args.held_out_episode_frac,
        rng=py_rng,
    )
    if not train_frames:
        raise RuntimeError("no training frames collected — increase --collect-episodes")

    train_tensors = frames_to_tensors(train_frames, device)
    held_tensors = frames_to_tensors(held_frames, device) if held_frames else None
    train_weights = bucket_weights(train_frames)

    student = PlayerV4(d_model=args.d_model, hidden=args.hidden).to(device)

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    n_train = train_tensors["player"].shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)

    best_selection_score = -float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1

    t0 = time.time()
    deadline = t0 + args.hours * 3600 if args.hours > 0 else None

    def should_continue(epoch: int) -> bool:
        if deadline is not None:
            return time.time() < deadline
        return epoch < args.epochs

    with log_path.open("w") as logf:
        epoch = 0
        while should_continue(epoch):
            student.train()
            sampler = torch.utils.data.WeightedRandomSampler(
                train_weights, num_samples=n_train, replacement=True
            )
            batch_iter = torch.utils.data.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx in batch_iter:
                idx = torch.tensor(list(batch_idx), dtype=torch.long, device=device)
                dist, _v = student(
                    train_tensors["player"][idx], train_tensors["bullets"][idx], train_tensors["pad"][idx]
                )
                loss = soft_ce_loss(dist.logits, train_tensors["teacher_logits"][idx], args.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
                if n_batches >= steps_per_epoch:
                    break
            epoch += 1

            do_eval = (epoch % args.eval_every == 0) or not should_continue(epoch) or epoch == 1
            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "elapsed_s": time.time() - t0,
            }
            if do_eval:
                held_metrics = (
                    frame_level_eval(student, held_tensors) if held_tensors is not None else {"agreement": float("nan")}
                )
                sel_times = closed_loop_times(student, device, SELECTION_SEEDS, args.selection_max_steps)
                sel_stats = summarize(sel_times)
                record.update(
                    {
                        "held_agreement": held_metrics.get("agreement"),
                        "held_cross_entropy": held_metrics.get("cross_entropy"),
                        "held_entropy": held_metrics.get("entropy"),
                        "selection_mean_s": sel_stats["mean"],
                        "selection_median_s": sel_stats["median"],
                    }
                )
                print(
                    f"epoch={epoch} loss={record['train_loss']:.4f} "
                    f"held_agree={held_metrics.get('agreement', float('nan')):.3f} "
                    f"held_ce={held_metrics.get('cross_entropy', float('nan')):.4f} "
                    f"sel(8000-8011)|scripted mean={sel_stats['mean']:.2f}s median={sel_stats['median']:.2f}s",
                    flush=True,
                )
                if sel_stats["mean"] > best_selection_score:
                    best_selection_score = sel_stats["mean"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(student.state_dict())
                    ckpt = pack_student_ckpt(
                        student,
                        d_model=args.d_model,
                        hidden=args.hidden,
                        epoch=epoch,
                        eval_metrics={
                            "held_agreement": held_metrics.get("agreement"),
                            "held_cross_entropy": held_metrics.get("cross_entropy"),
                            "selection_seeds": list(SELECTION_SEEDS),
                            "selection_max_steps": args.selection_max_steps,
                            "selection_mean_s": sel_stats["mean"],
                            "selection_median_s": sel_stats["median"],
                            "teacher_ckpt": str(args.teacher),
                        },
                    )
                    torch.save(ckpt, best_path)
                    print(f"  -> new best (selection mean={best_selection_score:.2f}s), saved {best_path}", flush=True)
            logf.write(json.dumps(record) + "\n")

    if best_state is None:
        # No eval ever ran (e.g. --epochs 0) — fall back to saving current weights.
        best_state = student.state_dict()
        best_epoch = epoch

    student.load_state_dict(best_state)
    student.eval()

    # --- Final report -------------------------------------------------------
    held_metrics = frame_level_eval(student, held_tensors) if held_tensors is not None else {}
    report_times = closed_loop_times(student, device, REPORT_SEEDS, REPORT_MAX_STEPS)
    report_stats = summarize(report_times)

    print("=== phase2 distillation report (best ckpt) ===", flush=True)
    print(f"best_epoch={best_epoch} selection_mean_s={best_selection_score:.2f}", flush=True)
    print(
        f"held-out frames: agreement={held_metrics.get('agreement', float('nan')):.3f} "
        f"cross_entropy={held_metrics.get('cross_entropy', float('nan')):.4f} "
        f"n={held_metrics.get('n', 0)}",
        flush=True,
    )
    print(
        f"closed-loop P|scripted seeds=3000-3029 max_steps={REPORT_MAX_STEPS} (70s): "
        f"mean={report_stats['mean']:.2f}s median={report_stats['median']:.2f}s std={report_stats['std']:.2f}",
        flush=True,
    )
    print(f"student policy entropy (held-out frames): {held_metrics.get('entropy', float('nan')):.4f}", flush=True)
    print(f"flee|scripted baseline (informational only): mean={flee_stats['mean']:.2f}s", flush=True)
    print(f"best ckpt written to {best_path}", flush=True)

    report = {
        "teacher_ckpt": str(args.teacher),
        "student_ckpt": str(best_path),
        "best_epoch": best_epoch,
        "device": str(device),
        "collect_episodes": args.collect_episodes,
        "held_out_frames": held_metrics,
        "closed_loop_report_3000_3029": report_stats,
        "closed_loop_selection_8000_8011_best": best_selection_score,
        "flee_baseline_3000_3029": flee_stats,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
