"""v4.3 evaluation modes: det/stoch policy × det/stoch env, paired seeds, metrics.

Modes (sample_policy, rng_jitter):
  det_det   — False, False  (debug / deterministic collapse)
  det_stoch — False, True   (robustness / primary ckpt signal for learned S)
  stoch_stoch — True, True  (deploy-distribution sanity; fork_rng per episode)

Invalid: (True, False) — sample_policy without rng_jitter is rejected (ValueError).
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Iterable, Sequence

import torch

# Fixed paired seed table shared across modes and matchups (compare / reports).
PAIRED_EVAL_SEEDS: tuple[int, ...] = tuple(range(3000, 3030))

# Named modes: (sample_policy, rng_jitter)
MODE_DET_DET = (False, False)
MODE_DET_STOCH = (False, True)
MODE_STOCH_STOCH = (True, True)


def validate_eval_mode(sample_policy: bool, rng_jitter: bool) -> None:
    """Reject unsupported (sample_policy=True, rng_jitter=False)."""
    if bool(sample_policy) and not bool(rng_jitter):
        raise ValueError(
            "Invalid eval mode (sample_policy=True, rng_jitter=False): "
            "stoch_policy requires stoch_env; use stoch_stoch (rng_jitter=True) "
            "or disable sample_policy."
        )


MODE_NAME = {
    MODE_DET_DET: "det_det",
    MODE_DET_STOCH: "det_stoch",
    MODE_STOCH_STOCH: "stoch_stoch",
}


def mode_tag(sample_policy: bool, rng_jitter: bool) -> str:
    validate_eval_mode(sample_policy, rng_jitter)
    return MODE_NAME[(bool(sample_policy), bool(rng_jitter))]


def metric_key(base: str, sample_policy: bool, rng_jitter: bool) -> str:
    """Distinct status/compare field, e.g. new_vs_new_det_stoch."""
    return f"{base}_{mode_tag(sample_policy, rng_jitter)}"


def summarize_times(times: Sequence[float]) -> dict[str, Any]:
    xs = [float(t) for t in times]
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n": 0, "times": []}
    mean = sum(xs) / n
    med = float(statistics.median(xs))
    std = float(statistics.pstdev(xs)) if n > 1 else 0.0
    return {"mean": mean, "median": med, "std": std, "n": n, "times": xs}


def run_eval_episode(
    run_episode_fn: Callable[..., tuple],
    env,
    player,
    spawner,
    device,
    max_steps: int,
    *,
    sample_policy: bool,
    rng_jitter: bool,
    episode_seed: int,
) -> tuple:
    """Run one eval episode; when sample_policy, fork Torch RNG so train RNG is untouched."""
    validate_eval_mode(sample_policy, rng_jitter)
    kwargs = dict(
        sample=bool(sample_policy),
        train_player=False,
        train_spawner=False,
        temp_p=1.0,
        temp_s=1.0,
        rng_jitter=bool(rng_jitter),
    )
    if sample_policy:
        with torch.random.fork_rng():
            torch.manual_seed(int(episode_seed))
            return run_episode_fn(env, player, spawner, device, max_steps, **kwargs)
    return run_episode_fn(env, player, spawner, device, max_steps, **kwargs)


def eval_survival_times(
    run_episode_fn: Callable[..., tuple],
    env_factory: Callable[[int], Any],
    player,
    spawner,
    device,
    seeds: Iterable[int],
    max_steps: int,
    *,
    sample_policy: bool = False,
    rng_jitter: bool = False,
) -> list[float]:
    validate_eval_mode(sample_policy, rng_jitter)
    times: list[float] = []
    for seed in seeds:
        env = env_factory(int(seed))
        _p, _s, t = run_eval_episode(
            run_episode_fn,
            env,
            player,
            spawner,
            device,
            max_steps,
            sample_policy=sample_policy,
            rng_jitter=rng_jitter,
            episode_seed=int(seed),
        )
        times.append(float(t))
    return times


def quantiles(times: Sequence[float], qs: Sequence[float] = (0.25, 0.75)) -> dict[str, float]:
    if not times:
        return {f"q{int(q * 100)}": 0.0 for q in qs}
    xs = sorted(float(t) for t in times)
    n = len(xs)
    out: dict[str, float] = {}
    for q in qs:
        # nearest-rank style
        idx = min(max(int(math.ceil(q * n) - 1), 0), n - 1)
        out[f"q{int(q * 100)}"] = xs[idx]
    return out
