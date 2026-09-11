"""v4.7: --random-fraction CLI plumbing. Default must be 0.0 (mechanism-only;
opening fraction>0 as a baseline is v4.8, out of scope here).

Imports only argparse-level helpers (qrokkun_env.train.both_v4_args) so this
test does not require torch to be installed.
"""

from __future__ import annotations

from qrokkun_env.train.both_v4_args import build_parser


def test_random_fraction_default_is_zero() -> None:
    ap = build_parser()
    args = ap.parse_args([])
    assert args.random_fraction == 0.0


def test_random_fraction_is_settable_float() -> None:
    ap = build_parser()
    args = ap.parse_args(["--random-fraction", "0.25"])
    assert isinstance(args.random_fraction, float)
    assert args.random_fraction == 0.25


def test_random_fraction_zero_is_falsy_default_path() -> None:
    ap = build_parser()
    args = ap.parse_args([])
    assert not args.random_fraction


def test_temp_rejection_unaffected_by_random_fraction() -> None:
    from qrokkun_env.train.both_v4_args import reject_non_unit_temp

    ap = build_parser()
    args = ap.parse_args(["--random-fraction", "0.5"])
    # Should not raise: temp defaults are still 1.0.
    reject_non_unit_temp(args)
