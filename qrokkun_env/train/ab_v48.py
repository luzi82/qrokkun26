"""v4.8 — fraction-only A/B experiment config + manifest layer.

This is a *diagnostic* controlled A/B layer on top of the v4.7 reset-mode
mechanism (see docs/suggestions/1788798436_grokbot.txt, v4.7 section, and
docs/suggestions/1789085919_v47_landed.txt). It does **not** change the
``--random-fraction`` default (still ``0.0``, see
``qrokkun_env/train/both_v4_args.py``) and does **not** modify PPO, rewards,
gamma/lambda, aim/birth geometry, checkpoint selection, dynamics burn-in, or
any default training mode. It only builds/validates a paired
baseline (fraction=0) vs treatment (fraction>0) CLI config for
``qrokkun_env.train.both_v4`` and persists a machine-readable manifest next
to each arm's own run outputs.

This module intentionally only touches ``argparse``/``json``/``subprocess``
(via ``qrokkun_env.train.both_v4_args.build_parser``, which itself does not
import torch), so it — and its tests — do not require torch to be installed.

Promotion is explicitly **forbidden** by this tool: it never selects a new
default baseline and never overwrites/renames a "best" checkpoint. See the
manifest's ``safety_gates`` for the full list of gates a human must check
before any baseline change is even considered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from qrokkun_env.train.both_v4_args import build_parser

ARM_BASELINE = "baseline"
ARM_TREATMENT = "treatment"
ARMS = (ARM_BASELINE, ARM_TREATMENT)

# Knobs that MUST be identical between the baseline and treatment arm for the
# comparison to be a fair fraction-only A/B (everything else held constant).
PAIRED_KNOB_FIELDS: tuple[str, ...] = (
    "seed",
    "hours",
    "rollouts",
    "temp_p",
    "temp_s",
    "entropy_p",
    "entropy_s",
    "lr",
    "gamma",
    "lam",
    "clip",
    "ppo_epochs",
    "minibatch",
    "max_steps",
    "d_model",
    "hidden",
    "device",
    "snapshot_every",
)

# Output-path fields that MUST be distinct (when set) between the two arms so
# one run cannot silently clobber the other's artifacts.
OUTPUT_PATH_FIELDS: tuple[str, ...] = (
    "out_player",
    "out_spawner",
    "status",
    "compare",
    "log",
    "out_player_best",
    "out_spawner_best",
    "snapshot_dir",
    "corner_probe",
)

# Safety gates a human must review before EVER treating a random_fraction>0
# run as a candidate new baseline. This tool cannot and does not check these
# automatically; it only records the requirement in the manifest.
SAFETY_GATES: tuple[str, ...] = (
    "compare normal-reset unseen-seed Player mean+median survival for the "
    "treatment arm against the baseline (fraction=0) arm; treatment must not "
    "regress either statistic",
    "corner probe (9 locked-Player sites) must not worsen at any single site "
    "for the treatment arm vs the baseline arm",
    "monitor Player PPO diagnostics (approx_kl, clipfrac, explained_variance) "
    "for the treatment arm; treatment must not show materially worse KL/"
    "clipfrac/EV than baseline",
    "monitor effective Spawner sample count (n_s_normal / n_s_total) for the "
    "treatment arm; the normal-reset-only Spawner PPO batch must not shrink "
    "to the point of starving Spawner training",
    "new x new (self-play) survival remains observation only and must never "
    "be used as the sole or primary signal to accept the treatment arm",
    "promotion (changing the default --random-fraction, or renaming a "
    "treatment checkpoint into a 'best'/default artifact) is forbidden by "
    "this script/tool; any baseline change requires a separate, explicit, "
    "human-reviewed decision",
)


def _validate_treatment_fraction(fraction: float) -> None:
    f = float(fraction)
    if not (0.0 < f <= 1.0):
        raise ValueError(
            f"v4.8 treatment random_fraction must satisfy 0 < f <= 1.0, got {f!r}"
        )


def _validate_baseline_fraction(fraction: float) -> None:
    f = float(fraction)
    if f != 0.0:
        raise ValueError(
            f"v4.8 baseline arm random_fraction must be exactly 0.0, got {f!r}"
        )


def validate_paired_args(baseline_args: argparse.Namespace, treatment_args: argparse.Namespace) -> None:
    """Validate a (baseline, treatment) argparse.Namespace pair produced by
    ``build_parser().parse_args(...)`` (or ``build_paired_configs``).

    Raises ``ValueError`` on:
    - baseline random_fraction != 0.0
    - treatment random_fraction outside (0, 1]
    - any paired training/eval knob differing between the two arms
    - any output path field set to the same value in both arms
    """
    _validate_baseline_fraction(baseline_args.random_fraction)
    _validate_treatment_fraction(treatment_args.random_fraction)

    mismatches = []
    for field in PAIRED_KNOB_FIELDS:
        bv = getattr(baseline_args, field)
        tv = getattr(treatment_args, field)
        if bv != tv:
            mismatches.append((field, bv, tv))
    if mismatches:
        raise ValueError(
            "v4.8 incompatible paired config: the following knobs differ "
            f"between baseline and treatment arms (must be identical for a "
            f"fair fraction-only A/B): {mismatches!r}"
        )

    duplicates = []
    for field in OUTPUT_PATH_FIELDS:
        bv = getattr(baseline_args, field, None)
        tv = getattr(treatment_args, field, None)
        if bv is not None and tv is not None and str(bv) == str(tv):
            duplicates.append((field, str(bv)))
    if duplicates:
        raise ValueError(
            "v4.8 duplicate output path(s) between baseline and treatment "
            f"arms (each arm must write to its own artifacts): {duplicates!r}"
        )


def _arm_argv(
    arm: str,
    *,
    seed: int,
    hours: float,
    fraction: float,
    out_dir: Path,
    extra_argv: Sequence[str] = (),
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")
    d = Path(out_dir) / arm
    argv = [
        "--seed", str(seed),
        "--hours", str(hours),
        "--random-fraction", str(fraction),
        "--out-player", str(d / "both_v4_player.pt"),
        "--out-spawner", str(d / "both_v4_spawner.pt"),
        "--status", str(d / "both_v4_status.json"),
        "--compare", str(d / "both_v4_compare.json"),
        "--log", str(d / "both_v4.jsonl"),
        "--snapshot-dir", str(d / "snapshots"),
    ]
    argv.extend(extra_argv)
    return argv


def build_paired_configs(
    *,
    seed: int,
    hours: float,
    treatment_fraction: float,
    out_dir: Path,
    extra_argv: Sequence[str] = (),
) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Build a validated (baseline_args, treatment_args) pair for
    ``qrokkun_env.train.both_v4``: identical seed/hours/training knobs
    (``extra_argv`` is applied verbatim to BOTH arms), baseline forced to
    ``random_fraction=0.0``, treatment set to ``treatment_fraction``, and
    distinct output paths under ``out_dir/baseline`` / ``out_dir/treatment``.

    Raises ``ValueError`` for an invalid ``treatment_fraction`` (must satisfy
    0 < f <= 1.0) before anything else is attempted.
    """
    _validate_treatment_fraction(treatment_fraction)
    parser = build_parser()
    baseline_args = parser.parse_args(
        _arm_argv(ARM_BASELINE, seed=seed, hours=hours, fraction=0.0, out_dir=out_dir, extra_argv=extra_argv)
    )
    treatment_args = parser.parse_args(
        _arm_argv(ARM_TREATMENT, seed=seed, hours=hours, fraction=treatment_fraction, out_dir=out_dir, extra_argv=extra_argv)
    )
    validate_paired_args(baseline_args, treatment_args)
    return baseline_args, treatment_args


def git_head(cwd: str | Path | None = None) -> str:
    """Best-effort ``git rev-parse HEAD``; returns "unknown" if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        head = result.stdout.strip()
        return head if result.returncode == 0 and head else "unknown"
    except Exception:
        return "unknown"


def build_manifest(args: argparse.Namespace, arm: str, *, git_head_value: str | None = None) -> dict[str, Any]:
    """Build the machine-readable v4.8 experiment manifest for one arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}; expected one of {ARMS!r}")
    return {
        "version": "v4.8",
        "arm": arm,
        "random_fraction": float(args.random_fraction),
        "seed": args.seed,
        "knobs": {field: getattr(args, field) for field in PAIRED_KNOB_FIELDS},
        "output_paths": {
            field: (str(getattr(args, field)) if getattr(args, field, None) is not None else None)
            for field in OUTPUT_PATH_FIELDS
        },
        "git_head": git_head_value if git_head_value is not None else git_head(),
        "eval_reset_mode": "normal",
        "promotion": "forbidden_by_this_tool",
        "safety_gates": list(SAFETY_GATES),
        "notes": (
            "v4.8 is a diagnostic fraction-only A/B layer on top of the v4.7 "
            "reset-mode mechanism (qrokkun_env/reset_modes.py). It does not "
            "modify PPO, rewards, gamma/lambda, aim/birth geometry, checkpoint "
            "selector, dynamics burn-in, or default training modes, and does "
            "not change the --random-fraction default (0.0). Official eval "
            "(eval_pair/eval_pair_stats/compare/best-ckpt selection) remains "
            "normal-reset only for both arms."
        ),
    }


def manifest_path_for(args: argparse.Namespace) -> Path:
    """Where this arm's manifest is persisted: next to its own status file."""
    status = Path(args.status)
    return status.with_name(status.stem + "_ab_manifest_v48.json")


def write_manifest(args: argparse.Namespace, arm: str, *, git_head_value: str | None = None) -> Path:
    manifest = build_manifest(args, arm, git_head_value=git_head_value)
    path = manifest_path_for(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


def write_paired_manifests(
    baseline_args: argparse.Namespace, treatment_args: argparse.Namespace
) -> tuple[Path, Path]:
    """Validate the paired config, then persist a manifest for EACH arm next
    to that arm's own run outputs. Raises before writing anything if the
    pair is invalid (fraction, mismatched knobs, or duplicate paths)."""
    validate_paired_args(baseline_args, treatment_args)
    head = git_head()
    b_path = write_manifest(baseline_args, ARM_BASELINE, git_head_value=head)
    t_path = write_manifest(treatment_args, ARM_TREATMENT, git_head_value=head)
    return b_path, t_path


def build_ab_meta_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="qrokkun_env.train.ab_v48",
        description=(
            "v4.8 diagnostic fraction-only A/B config/manifest generator. "
            "Builds a validated paired baseline (--random-fraction 0.0) vs "
            "treatment (--random-fraction TREATMENT_FRACTION) config for "
            "qrokkun_env.train.both_v4, writes a manifest for each arm, and "
            "prints the exact reproducible command for each arm. Does NOT "
            "run training itself and NEVER promotes a checkpoint or default."
        ),
    )
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument(
        "--treatment-fraction",
        type=float,
        required=True,
        help="Treatment arm --random-fraction; must satisfy 0 < f <= 1.0.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/ab_v48"),
        help="Parent dir; arms write under out-dir/baseline and out-dir/treatment.",
    )
    ap.add_argument(
        "--extra-argv",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Remaining both_v4 CLI flags applied identically to BOTH arms "
            "(e.g. --extra-argv --lr 1e-4 --gamma 0.98). Must be last."
        ),
    )
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    ap = build_ab_meta_parser()
    ns = ap.parse_args(argv)
    baseline_args, treatment_args = build_paired_configs(
        seed=ns.seed,
        hours=ns.hours,
        treatment_fraction=ns.treatment_fraction,
        out_dir=ns.out_dir,
        extra_argv=ns.extra_argv,
    )
    b_path, t_path = write_paired_manifests(baseline_args, treatment_args)
    for arm, fraction, manifest_path in (
        (ARM_BASELINE, 0.0, b_path),
        (ARM_TREATMENT, ns.treatment_fraction, t_path),
    ):
        argv_list = _arm_argv(
            arm, seed=ns.seed, hours=ns.hours, fraction=fraction, out_dir=ns.out_dir, extra_argv=ns.extra_argv
        )
        cmd = ["python", "-m", "qrokkun_env.train.both_v4"] + argv_list
        print(f"# {arm} manifest -> {manifest_path}")
        print(" ".join(cmd))
    print(
        "# NOTE: v4.8 is diagnostic only. This tool does not run training and "
        "does not promote any checkpoint or change the default baseline."
    )


if __name__ == "__main__":
    main()
