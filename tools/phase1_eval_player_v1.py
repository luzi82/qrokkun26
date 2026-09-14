#!/usr/bin/env python3
"""Phase 1: re-eval historical PlayerV1 checkpoint under two eval contracts.

Does not train. Load PlayerV1 + obs.vectorize only — never PlayerV4.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from qrokkun_env.env import ACTIONS, Qrokkun26Env
from qrokkun_env.obs import OBS_DIM, vectorize
from qrokkun_env.agents.player_v1 import PlayerV1
from qrokkun_env.policies import FleeNearestBullet, IdlePolicy


def summarize(times: list[float], max_seconds: float) -> dict:
    n = len(times)
    mean = sum(times) / n
    med = statistics.median(times)
    std = statistics.pstdev(times) if n > 1 else 0.0
    timeouts = sum(1 for t in times if t >= max_seconds - 1e-6)
    return {
        "n": n,
        "mean": mean,
        "median": med,
        "std": std,
        "min": min(times),
        "max": max(times),
        "timeout_rate": timeouts / n,
        "times": times,
    }


@torch.no_grad()
def eval_policy(*, kind: str, net, device, seeds: list[int], max_steps: int) -> list[float]:
    flee = FleeNearestBullet()
    idle = IdlePolicy()
    times: list[float] = []
    dt = None
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        env.reset(seed=seed)
        if dt is None:
            dt = float(env.dt)
        rng = random.Random(seed + 17)
        if net is not None:
            net.eval()
        for _ in range(max_steps):
            if kind == "player":
                x = torch.tensor(vectorize(env), dtype=torch.float32, device=device)
                dist, _v = net(x)
                a = int(dist.probs.argmax().item())
            elif kind == "flee":
                a = ACTIONS.index(flee.act(env))
            elif kind == "idle":
                a = ACTIONS.index(idle.act(env))
            elif kind == "random":
                a = rng.randrange(len(ACTIONS))
            else:
                raise ValueError(kind)
            _o, _r, done, info = env.step(a)
            if done:
                times.append(float(info.get("elapsed", env.elapsed)))
                break
        else:
            times.append(float(env.elapsed))
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    assert ck.get("obs_dim", OBS_DIM) == OBS_DIM, ck.get("obs_dim")
    assert list(ck["actions"]) == list(ACTIONS), ck["actions"]
    net = PlayerV1(hidden=int(ck.get("hidden", 256)))
    net.load_state_dict(ck["state_dict"])
    net.to(device)
    net.eval()

    contracts = {
        "historical_90s_seeds300": {
            "seeds": list(range(300, 330)),
            "max_steps": 60 * 90,
            "max_seconds": 90.0,
            "note": "player_v1 train eval window (range(300,330), 90s cap)",
        },
        "current_70s_paired3000": {
            "seeds": list(range(3000, 3030)),
            "max_steps": 60 * 70,
            "max_seconds": 70.0,
            "note": "current both_v4 PAIRED_EVAL_SEEDS prefix, 70s cap",
        },
    }

    report = {
        "ckpt": str(args.ckpt),
        "ckpt_eval_mean_s": ck.get("eval_mean_s"),
        "ckpt_update": ck.get("update"),
        "ckpt_algo": ck.get("algo"),
        "obs_dim": ck.get("obs_dim"),
        "hidden": ck.get("hidden"),
        "actions": list(ck["actions"]),
        "device": str(device),
        "contracts": {},
    }
    for cname, spec in contracts.items():
        block = {"spec": {k: spec[k] for k in ("seeds", "max_steps", "max_seconds", "note")}}
        for kind in ("player", "flee", "idle", "random"):
            times = eval_policy(
                kind=kind,
                net=net if kind == "player" else None,
                device=device,
                seeds=spec["seeds"],
                max_steps=spec["max_steps"],
            )
            block[kind] = summarize(times, spec["max_seconds"])
            print(
                f"{cname} {kind}: mean={block[kind]['mean']:.2f}s "
                f"median={block[kind]['median']:.2f}s std={block[kind]['std']:.2f} "
                f"timeout={block[kind]['timeout_rate']:.2f}",
                flush=True,
            )
        report["contracts"][cname] = block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
