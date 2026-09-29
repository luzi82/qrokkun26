"""Narrow CLI facade for the preregistered gamma/lambda horizon arms."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention_aux as aux


def build_parser() -> argparse.ArgumentParser:
    parser = aux.build_parser()
    parser.add_argument("--horizon-arm", choices=("control", "long"), required=True)
    return parser


def main() -> None:
    args = aux.apply_mode_defaults(build_parser().parse_args())
    args.argv = list(sys.argv)
    report = aux.run_experiment(args, torch.device(args.device))
    print(json.dumps({"status": report["status"], "arm_ran": report["arm_ran"]}))


if __name__ == "__main__":
    main()
