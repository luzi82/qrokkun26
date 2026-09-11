"""v4.8: fraction-only A/B experiment config/manifest layer, built on top of
the v4.7 --random-fraction mechanism (qrokkun_env/reset_modes.py).

v4.8 is diagnostic only: it must NEVER change the --random-fraction default
(0.0) or normal v4.7/v4.6 CLI behavior. All of this is pure argparse/json/
subprocess-level plumbing (qrokkun_env.train.ab_v48) so these tests do not
require torch to be installed.
"""

from __future__ import annotations

import json

import pytest

from qrokkun_env.train.ab_v48 import (
    ARM_BASELINE,
    ARM_TREATMENT,
    OUTPUT_PATH_FIELDS,
    PAIRED_KNOB_FIELDS,
    build_manifest,
    build_paired_configs,
    manifest_path_for,
    validate_paired_args,
    write_paired_manifests,
)
from qrokkun_env.train.both_v4_args import build_parser


def test_default_random_fraction_still_zero() -> None:
    """v4.8 must never change the v4.7 default."""
    ap = build_parser()
    args = ap.parse_args([])
    assert args.random_fraction == 0.0


def test_build_paired_configs_baseline_is_fraction_zero(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    assert baseline_args.random_fraction == 0.0
    assert treatment_args.random_fraction == 0.3


def test_build_paired_configs_shares_seed_and_hours(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=7, hours=2.5, treatment_fraction=0.1, out_dir=tmp_path
    )
    assert baseline_args.seed == treatment_args.seed == 7
    assert baseline_args.hours == treatment_args.hours == 2.5


def test_build_paired_configs_distinct_output_paths(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    assert str(baseline_args.out_player) != str(treatment_args.out_player)
    assert str(baseline_args.status) != str(treatment_args.status)


@pytest.mark.parametrize("bad_fraction", [0.0, -0.1, 1.5, 2.0])
def test_invalid_treatment_fraction_rejected(tmp_path, bad_fraction) -> None:
    with pytest.raises(ValueError):
        build_paired_configs(
            seed=4, hours=1.0, treatment_fraction=bad_fraction, out_dir=tmp_path
        )


def test_valid_treatment_fraction_boundary_one_accepted(tmp_path) -> None:
    # f == 1.0 is inclusive per spec (0 < f <= 1).
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=1.0, out_dir=tmp_path
    )
    assert treatment_args.random_fraction == 1.0


def test_baseline_fraction_tampering_rejected(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    baseline_args.random_fraction = 0.2
    with pytest.raises(ValueError):
        validate_paired_args(baseline_args, treatment_args)


def test_duplicate_output_paths_rejected(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    treatment_args.out_player = baseline_args.out_player
    with pytest.raises(ValueError):
        validate_paired_args(baseline_args, treatment_args)


def test_incompatible_knob_mismatch_rejected(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    treatment_args.gamma = 0.5  # diverges from baseline -> not a fair paired A/B
    with pytest.raises(ValueError):
        validate_paired_args(baseline_args, treatment_args)


def test_manifest_contains_required_fields(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    manifest = build_manifest(treatment_args, ARM_TREATMENT, git_head_value="deadbeef")
    assert manifest["arm"] == ARM_TREATMENT
    assert manifest["random_fraction"] == 0.3
    assert manifest["seed"] == 4
    assert manifest["git_head"] == "deadbeef"
    assert manifest["knobs"]["hours"] == 1.0
    assert manifest["knobs"]["gamma"] == baseline_args.gamma
    assert manifest["eval_reset_mode"] == "normal"
    assert manifest["promotion"] == "forbidden_by_this_tool"
    gates = " ".join(manifest["safety_gates"]).lower()
    assert "corner probe" in gates
    assert "new" in gates and "observation only" in gates
    assert "clipfrac" in gates or "kl" in gates
    assert "spawner" in gates


def test_manifest_rejects_unknown_arm(tmp_path) -> None:
    baseline_args, _ = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    with pytest.raises(ValueError):
        build_manifest(baseline_args, "not_an_arm")


def test_write_paired_manifests_persisted_next_to_each_run(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    b_path, t_path = write_paired_manifests(baseline_args, treatment_args)

    assert b_path.exists() and t_path.exists()
    # Manifests live next to each arm's own status output.
    assert b_path.parent == baseline_args.status.parent
    assert t_path.parent == treatment_args.status.parent

    b_manifest = json.loads(b_path.read_text())
    t_manifest = json.loads(t_path.read_text())
    assert b_manifest["arm"] == ARM_BASELINE
    assert b_manifest["random_fraction"] == 0.0
    assert t_manifest["arm"] == ARM_TREATMENT
    assert t_manifest["random_fraction"] == 0.3
    assert b_manifest["seed"] == t_manifest["seed"] == 4


def test_snapshot_every_mismatch_rejected(tmp_path) -> None:
    """snapshot_every is a training knob, not an output path or
    random_fraction, so it MUST be checked for paired equality just like
    gamma/lr/etc. A baseline/treatment mismatch must be rejected."""
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    treatment_args.snapshot_every = baseline_args.snapshot_every + 5
    with pytest.raises(ValueError):
        validate_paired_args(baseline_args, treatment_args)


def test_snapshot_every_is_reported_in_manifest_knobs(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    manifest = build_manifest(treatment_args, ARM_TREATMENT, git_head_value="deadbeef")
    assert manifest["knobs"]["snapshot_every"] == treatment_args.snapshot_every


def test_every_both_v4_parser_destination_is_accounted_for() -> None:
    """Durable parser-coverage regression: every argparse ``dest`` produced
    by ``both_v4_args.build_parser()`` must be covered by exactly one of:
    - a PAIRED_KNOB_FIELDS entry (checked for equality between arms), or
    - an OUTPUT_PATH_FIELDS entry (permitted/required to differ between
      arms), or
    - ``random_fraction`` (the one deliberately-differing experiment knob).

    This guards against future both_v4_args.py flags silently falling
    through the cracks of v4.8's paired-config validation (e.g. the
    snapshot_every gap this test was added to catch).
    """
    parser = build_parser()
    all_dests = {
        action.dest
        for action in parser._actions
        if action.dest != "help"
    }

    accounted = set(PAIRED_KNOB_FIELDS) | set(OUTPUT_PATH_FIELDS) | {"random_fraction"}

    missing = all_dests - accounted
    assert not missing, (
        f"both_v4 parser dest(s) {sorted(missing)!r} are not accounted for "
        "as a paired knob, an output path, or random_fraction in ab_v48.py"
    )

    extra = accounted - all_dests
    assert not extra, (
        f"ab_v48.py references dest(s) {sorted(extra)!r} that no longer "
        "exist in both_v4_args.build_parser()"
    )

    # Each dest must be classified exactly once (no double-counting/overlap).
    overlap = set(PAIRED_KNOB_FIELDS) & set(OUTPUT_PATH_FIELDS)
    assert not overlap, f"fields double-classified as knob AND path: {overlap!r}"
    assert "random_fraction" not in PAIRED_KNOB_FIELDS
    assert "random_fraction" not in OUTPUT_PATH_FIELDS


def test_write_paired_manifests_raises_before_writing_on_invalid_config(tmp_path) -> None:
    baseline_args, treatment_args = build_paired_configs(
        seed=4, hours=1.0, treatment_fraction=0.3, out_dir=tmp_path
    )
    treatment_args.out_player = baseline_args.out_player
    with pytest.raises(ValueError):
        write_paired_manifests(baseline_args, treatment_args)
    # Nothing should have been written when validation fails up front.
    assert not manifest_path_for(baseline_args).exists()
    assert not manifest_path_for(treatment_args).exists()
