"""CLI arg parsing for train/both_v4.py, split out so it can be unit-tested
without importing torch (argparse only; no tensors/nets touched here).

v4.7 adds --random-fraction (default 0.0 — mechanism-only; see reset_modes.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--out-player", type=Path, default=Path("runs/both_v4_player.pt"))
    ap.add_argument("--out-spawner", type=Path, default=Path("runs/both_v4_spawner.pt"))
    ap.add_argument("--status", type=Path, default=Path("runs/both_v4_status.json"))
    ap.add_argument("--compare", type=Path, default=Path("runs/both_v4_compare.json"))
    ap.add_argument("--log", type=Path, default=Path("runs/both_v4.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument(
        "--temp-p",
        type=float,
        default=1.0,
        help="Player action temperature (default 1.0). Exploration uses sampling+entropy, not temp!=1. Non-1.0 is rejected.",
    )
    ap.add_argument(
        "--temp-s",
        type=float,
        default=1.0,
        help="Spawner action temperature (default 1.0). Exploration uses sampling+entropy, not temp!=1. Non-1.0 is rejected.",
    )
    ap.add_argument("--entropy-p", type=float, default=0.04)
    ap.add_argument("--entropy-s", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99, help="Per-frame discount; GAE uses gamma**delta_frames (v4.5).")
    ap.add_argument("--lam", type=float, default=0.95, help="Per-frame GAE lambda; uses lam**delta_frames (v4.5).")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=60 * 70)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument(
        "--corner-probe",
        type=Path,
        nargs="?",
        const=Path("runs/both_v4_corner_probe.json"),
        default=None,
        help="If set, run locked-P corner probe at end and write JSON (default path if flag alone).",
    )
    ap.add_argument(
        "--out-player-best",
        type=Path,
        default=None,
        help="Best player ckpt (default: <out-player>_best.pt). Selection: newP×scripted.",
    )
    ap.add_argument(
        "--out-spawner-best",
        type=Path,
        default=None,
        help="Best spawner ckpt (default: <out-spawner>_best.pt). Selection: flee×newS_det_stoch.",
    )
    ap.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="If >0, copy latest weights to snapshot-dir every N updates (0=off).",
    )
    ap.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("runs/snapshots"),
        help="Directory for periodic snapshot copies (v4.6).",
    )
    ap.add_argument(
        "--random-fraction",
        type=float,
        default=0.0,
        help=(
            "v4.7 mechanism-only flag: fraction of eligible training rollouts "
            "(self / p_vs_scripted) that use a near-center random-Player, "
            "empty-field reset instead of env.reset(). Default 0.0 preserves "
            "the current normal-reset-only training mix. s_vs_flee is ALWAYS "
            "normal reset regardless of this flag. Official eval/compare/best-ckpt "
            "selection always uses normal reset. Enabling fraction>0 as a new "
            "baseline is out of scope for v4.7 (see v4.8)."
        ),
    )
    return ap


def reject_non_unit_temp(args: argparse.Namespace) -> None:
    """v4.2: keep CLI names but refuse any temperature other than 1.0."""
    if float(args.temp_p) != 1.0 or float(args.temp_s) != 1.0:
        raise SystemExit(
            "error: non-1.0 temperature is not supported "
            f"(got --temp-p={args.temp_p}, --temp-s={args.temp_s}). "
            "Exploration uses sampling+entropy, not temp!=1."
        )
