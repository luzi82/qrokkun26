"""v4.3 corner / edge probe: fixed Player geometry diagnostics (locked P).

Independent of new×new mean eval. Player is idle and forcibly re-pinned each
frame so position never drifts (Code reviewer: LOCK player during probe).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from qrokkun_env import constants as C
from qrokkun_env.agents.spawner_v4 import AIM_SCALE, SpawnerV4, spawn_continuous
from qrokkun_env.env import ACTIONS, Qrokkun26Env, _spawn_interval
from qrokkun_env.train.both_v4 import act_spawner, apply_player_action


def probe_positions() -> dict[str, tuple[float, float]]:
    """9 fixed points: center, 4 corners (incl bottom-right), 4 edge midpoints."""
    x0 = C.FIELD_X + C.PLAYER_MARGIN
    y0 = C.FIELD_Y + C.PLAYER_MARGIN
    x1 = C.FIELD_X + C.FIELD_W - C.PLAYER_MARGIN
    y1 = C.FIELD_Y + C.FIELD_H - C.PLAYER_MARGIN
    cx = C.FIELD_X + C.FIELD_W * 0.5
    cy = C.FIELD_Y + C.FIELD_H * 0.5
    return {
        "center": (cx, cy),
        "top_left": (x0, y0),
        "top_right": (x1, y0),
        "bottom_left": (x0, y1),
        "bottom_right": (x1, y1),
        "top_mid": (cx, y0),
        "bottom_mid": (cx, y1),
        "left_mid": (x0, cy),
        "right_mid": (x1, cy),
    }


def _lock_player(env: Qrokkun26Env, px: float, py: float) -> None:
    """Force player to stay at (px, py) with zero velocity."""
    env.px = float(px)
    env.py = float(py)
    env.pvx = 0.0
    env.pvy = 0.0


@torch.no_grad()
def run_locked_probe_episode(
    env: Qrokkun26Env,
    spawner: SpawnerV4 | None,
    device: torch.device,
    px: float,
    py: float,
    max_steps: int,
    *,
    sample_policy: bool = False,
    rng_jitter: bool = False,
    episode_seed: int | None = None,
) -> dict[str, Any]:
    """One probe episode with Player locked. Spawner may be learned or scripted (None → env.step idle)."""
    env.reset()
    _lock_player(env, px, py)

    closest = float("inf")
    first_hit: float | None = None
    birth_dirs: list[tuple[float, float]] = []
    aim_raws: list[tuple[float, float]] = []
    aim_offsets: list[tuple[float, float]] = []  # tanh*scale relative to player
    positions_seen: list[tuple[float, float]] = [(env.px, env.py)]
    idle_idx = ACTIONS.index("idle")

    def _maybe_fork_act():
        if sample_policy and episode_seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(int(episode_seed) + int(env.elapsed * 1000))
                return act_spawner(spawner, env, device, True, 1.0)
        return act_spawner(spawner, env, device, False, 1.0)

    for _ in range(max_steps):
        if spawner is None:
            # Scripted path via env.step; still lock after.
            _o, _r, done, _ = env.step(idle_idx)
            _lock_player(env, px, py)
        else:
            env.elapsed += env.dt
            env.spawn_acc += env.dt
            interval = _spawn_interval(env.elapsed)
            spawns = 0
            while env.spawn_acc >= interval:
                env.spawn_acc -= interval
                act, _lp, _v, _p, _b, _m = _maybe_fork_act()
                spawn_continuous(
                    env, act["birth"], act["aim"], act["kind"], rng_jitter=rng_jitter
                )
                birth_dirs.append(tuple(act["birth"]))
                aim_raws.append(tuple(act["aim"]))
                ox = math.tanh(act["aim"][0]) * AIM_SCALE
                oy = math.tanh(act["aim"][1]) * AIM_SCALE
                aim_offsets.append((ox, oy))
                spawns += 1
                thr, p_double = 8.0, 0.0
                if env.elapsed > 18.0:
                    p_double = 0.20
                elif env.elapsed > thr:
                    p_double = 0.10
                if spawns == 1 and p_double > 0 and env.rng.randf() < p_double:
                    env.spawn_acc += interval
                interval = _spawn_interval(env.elapsed)

            apply_player_action(env, idle_idx)
            # CRITICAL: re-pin after any physics so P never moves.
            _lock_player(env, px, py)
            env._integrate_bullets()
            done = env._check_hit()
            if done:
                env.dead = True

        positions_seen.append((float(env.px), float(env.py)))
        for b in env.bullets:
            d = math.hypot(b.x - px, b.y - py)
            if d < closest:
                closest = d

        if done and first_hit is None:
            first_hit = float(env.elapsed)
            break

    # Assert lock held for this episode (also checked in unit tests).
    for ax, ay in positions_seen:
        if abs(ax - px) > 1e-6 or abs(ay - py) > 1e-6:
            raise RuntimeError(f"player moved during probe: saw {(ax, ay)} want {(px, py)}")

    hit = first_hit is not None
    return {
        "px": px,
        "py": py,
        "first_hit_time": first_hit,
        "hit": hit,
        "timeout": not hit,
        "elapsed": float(env.elapsed),
        "closest_approach": None if closest == float("inf") else float(closest),
        "n_spawns": len(birth_dirs),
        "birth_dirs": birth_dirs,
        "aim_raws": aim_raws,
        "aim_offsets": aim_offsets,
        "player_locked": True,
        "positions_unique": sorted({(round(a, 6), round(b, 6)) for a, b in positions_seen}),
    }


def _agg_birth_aim(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    births = [xy for ep in episodes for xy in ep["birth_dirs"]]
    aims = [xy for ep in episodes for xy in ep["aim_offsets"]]
    def mean2(xs: list[tuple[float, float]]) -> list[float] | None:
        if not xs:
            return None
        return [sum(a for a, _ in xs) / len(xs), sum(b for _, b in xs) / len(xs)]
    return {
        "n_spawn_actions": len(births),
        "birth_dir_mean": mean2(births),
        "aim_offset_mean": mean2(aims),
    }


def probe_position(
    spawner: SpawnerV4 | None,
    device: torch.device,
    name: str,
    xy: tuple[float, float],
    seeds: list[int],
    max_steps: int,
    *,
    sample_policy: bool = False,
    rng_jitter: bool = False,
) -> dict[str, Any]:
    px, py = xy
    episodes = []
    for seed in seeds:
        env = Qrokkun26Env(seed=seed)
        ep = run_locked_probe_episode(
            env,
            spawner,
            device,
            px,
            py,
            max_steps,
            sample_policy=sample_policy,
            rng_jitter=rng_jitter,
            episode_seed=seed,
        )
        episodes.append(ep)
    hits = [e for e in episodes if e["hit"]]
    first_hits = [e["first_hit_time"] for e in hits]
    closest = [e["closest_approach"] for e in episodes if e["closest_approach"] is not None]
    return {
        "name": name,
        "px": px,
        "py": py,
        "n_trials": len(episodes),
        "hit_rate": len(hits) / max(len(episodes), 1),
        "first_hit_mean": (sum(first_hits) / len(first_hits)) if first_hits else None,
        "closest_approach_mean": (sum(closest) / len(closest)) if closest else None,
        "closest_approach_min": min(closest) if closest else None,
        "player_locked": all(e["player_locked"] for e in episodes),
        "birth_aim_stats": _agg_birth_aim(episodes),
        "trials": [
            {
                "seed": seeds[i],
                "first_hit_time": episodes[i]["first_hit_time"],
                "hit": episodes[i]["hit"],
                "closest_approach": episodes[i]["closest_approach"],
                "elapsed": episodes[i]["elapsed"],
                "n_spawns": episodes[i]["n_spawns"],
            }
            for i in range(len(episodes))
        ],
    }


def run_corner_probe(
    spawner: SpawnerV4 | None,
    device: torch.device,
    *,
    seeds: list[int] | None = None,
    max_steps: int = 60 * 40,
    sample_policy: bool = False,
    rng_jitter: bool = False,
) -> dict[str, Any]:
    seeds = list(seeds) if seeds is not None else list(range(4000, 4008))
    positions = probe_positions()
    results = []
    for name, xy in positions.items():
        results.append(
            probe_position(
                spawner,
                device,
                name,
                xy,
                seeds,
                max_steps,
                sample_policy=sample_policy,
                rng_jitter=rng_jitter,
            )
        )
    by_name = {r["name"]: r for r in results}
    return {
        "schema": "both_v4_corner_probe.v1",
        "n_positions": len(results),
        "position_names": list(positions.keys()),
        "seeds": seeds,
        "max_steps": max_steps,
        "sample_policy": sample_policy,
        "rng_jitter": rng_jitter,
        "aim_scale": AIM_SCALE,
        "note": "Independent diagnostic; do NOT mix into new×new mean.",
        "positions": results,
        "bottom_right": by_name.get("bottom_right"),
    }


def load_spawner(path: Path, device: torch.device) -> SpawnerV4:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = SpawnerV4(d_model=int(ck.get("d_model", 128)), hidden=int(ck.get("hidden", 256)))
    net.load_state_dict(ck["state_dict"])
    net.to(device)
    net.eval()
    return net


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="v4.3 locked-P corner / edge probe")
    ap.add_argument("--spawner", type=Path, default=Path("runs/both_v4_spawner.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/both_v4_corner_probe.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=60 * 40)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=4000)
    ap.add_argument("--rng-jitter", action="store_true", help="stoch env jitter (default off)")
    ap.add_argument("--sample-policy", action="store_true", help="sample spawner (default det)")
    ap.add_argument("--scripted", action="store_true", help="use scripted spawner instead of ckpt")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    spawner = None if args.scripted else load_spawner(args.spawner, device)
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    report = run_corner_probe(
        spawner,
        device,
        seeds=seeds,
        max_steps=args.max_steps,
        sample_policy=args.sample_policy,
        rng_jitter=args.rng_jitter,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} positions={report['n_positions']} bottom_right hit_rate="
          f"{report['bottom_right']['hit_rate'] if report['bottom_right'] else 'n/a'}", flush=True)


if __name__ == "__main__":
    main()
