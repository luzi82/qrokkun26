"""Real eval wiring for the frozen learned-Spawner PlayerV5 comparison
(:doc:`docs/journal/1789263159_player_v5_frozen_spawner_eval_preregistration`).

This module is the missing "actually run it" piece that
:mod:`qrokkun_env.frozen_spawner_seed_package` (immutable manifest) and
:mod:`qrokkun_env.frozen_spawner_eval_runner` (pure statistics) deliberately
left out. It provides:

* fail-closed, architecture-dispatched checkpoint loading
  (:func:`load_checkpoint_for_role`) that validates a checkpoint's on-disk
  SHA-256 -- and, if recorded, its state-dict SHA-256 -- *before* any weight
  is constructed or restored, using the correct, distinct loader/observation
  contract for legacy ``PlayerV1`` (flat 45-dim obs, dict checkpoint) versus
  ``PlayerRankedTopK`` (masked player/bullets/pad obs, strict metadata
  checkpoint via :mod:`qrokkun_env.agents.player_checkpoints`), and the
  existing ``SpawnerV4`` checkpoint-dict contract (``d_model``/``hidden``/
  ``state_dict``);
* a single-episode runner (:func:`run_frozen_spawner_episode`) that evaluates
  one player arm against the frozen learned Spawner with *independent*
  player/Spawner sampling flags -- the player is always evaluated with a
  preselected, deterministic action mode (argmax), while the Spawner may be
  deterministic or stochastic, matching the preregistered CRN pairing
  contract (this independence is exactly what
  ``qrokkun_env.train.both_v4.run_episode`` cannot do, since it ties both
  roles to a single ``sample`` flag);
* a CRN wrapper (:func:`run_frozen_spawner_episode_crn`) that forks the Torch
  RNG on the per-slot seed only when Spawner sampling actually consumes it,
  mirroring ``qrokkun_env.eval_modes.run_eval_episode``'s reproducibility
  contract;
* :func:`run_frozen_spawner_comparison`, which runs exactly the
  preregistered comparison pairs from a :class:`FrozenSpawnerSeedPackage` and
  produces a JSON-serializable report with per-seed elapsed times, restricted
  survival statistics per arm, and the paired-difference statistics for each
  preregistered pair -- nothing else;
* a small CLI (:func:`main`) that takes checkpoint paths + pinned SHA-256
  values, builds the seed package, runs the comparison, and writes the
  manifest and report JSON files.

Structurally eval-only: this module never constructs a ``torch.optim``
optimizer, never calls a PPO update, never launches self-play, and never
selects/writes a "promoted" checkpoint. The report's ``promotion``/
``training``/``self_play`` fields are hardcoded ``False`` literals, not
values threaded from any computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qrokkun_env import constants as C
from qrokkun_env.agents.obs_v4 import encode_obs
from qrokkun_env.agents.player_checkpoints import (
    file_sha256,
    load_ranked_top_k_checkpoint,
    state_dict_sha256,
)
from qrokkun_env.agents.player_ranked_topk import ARCHITECTURE as RANKED_TOP_K_ARCHITECTURE
from qrokkun_env.agents.player_v1 import PlayerV1
from qrokkun_env.agents.spawner_v4 import SpawnerV4, spawn_continuous
from qrokkun_env.env import Qrokkun26Env, _spawn_interval
from qrokkun_env.eval_modes import MODE_DET_DET, MODE_DET_STOCH, MODE_STOCH_STOCH
from qrokkun_env.train.both_v4 import apply_player_action as _canonical_apply_player_action
from qrokkun_env.frozen_spawner_eval_runner import (
    paired_difference_stats,
    restricted_survival_stats,
)
from qrokkun_env.frozen_spawner_seed_package import (
    CheckpointRef,
    CONTRACT_ID,
    DIAGNOSTIC_CONTRACT_ID,
    FrozenSpawnerManifestError,
    FrozenSpawnerSeedPackage,
    PREREGISTERED_COMPARISON_PAIRS,
    PREREGISTRATION_PATH,
    ROLE_LEARNED_SPAWNER,
    ROLE_V1_TEACHER,
    ROLE_V5_AUX_U200,
    ROLE_V5_BC_SEED6,
    build_frozen_spawner_seed_package,
    is_canonical_formal_package,
    SUPPORTED_SCHEMA_VERSION,
    validate_serializable_frozen_spawner_seed_package,
)
from qrokkun_env.obs import vectorize as vectorize_v1

# --- Architecture tags -------------------------------------------------------

ARCHITECTURE_PLAYER_V1 = "player_v1"
ARCHITECTURE_PLAYER_RANKED_TOP_K = RANKED_TOP_K_ARCHITECTURE
ARCHITECTURE_SPAWNER_V4 = "spawner_v4"

# Fixed player role -> architecture (the roles a comparison pair may name).
PLAYER_ROLE_ARCHITECTURE = {
    ROLE_V1_TEACHER: ARCHITECTURE_PLAYER_V1,
    ROLE_V5_BC_SEED6: ARCHITECTURE_PLAYER_RANKED_TOP_K,
    ROLE_V5_AUX_U200: ARCHITECTURE_PLAYER_RANKED_TOP_K,
}

# The exact two preregistered comparison pairs (see the preregistration
# journal entry); this runner computes these and nothing else. Re-exported
# from (and identical to) qrokkun_env.frozen_spawner_seed_package's own
# locked constant, so there is a single source of truth for this ordering.
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = PREREGISTERED_COMPARISON_PAIRS

EARLY_DEATH_THRESHOLD_SECONDS = 5.0


class FrozenSpawnerEvalError(RuntimeError):
    """Base error for the frozen-Spawner eval CLI/runner."""


class ChecksumMismatchError(FrozenSpawnerEvalError):
    """A checkpoint's on-disk/state-dict SHA-256 did not match the recorded
    value. Raised *before* any weight is loaded (fail closed)."""


class UnsupportedArchitectureError(FrozenSpawnerEvalError):
    """A checkpoint role declared an architecture this runner cannot load."""


def _verify_file_sha256(path: Path | str, expected_sha256: str) -> str:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"file sha256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


# --- Loaders: one per architecture, distinct loader/observation contract ----


def load_player_v1_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[PlayerV1, dict]:
    """Legacy ``PlayerV1`` loader: dict checkpoint (``hidden``/``state_dict``),
    flat 45-dim observation (:func:`qrokkun_env.obs.vectorize`). No
    architecture metadata is recorded in legacy checkpoints, so this loader
    is only ever reached via the explicit ``player_v1`` architecture tag in a
    :class:`CheckpointRef`, never by inspecting the checkpoint itself.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    hidden = int(ck.get("hidden", 256))
    net = PlayerV1(hidden=hidden).to(device)
    net.load_state_dict(ck["state_dict"], strict=True)
    net.eval()
    meta = {
        "architecture": ARCHITECTURE_PLAYER_V1,
        "hidden": hidden,
        "state_dict_sha256": state_dict_sha256(net.state_dict()),
    }
    return net, meta


def load_player_ranked_topk_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[Any, dict]:
    """Strict ``PlayerRankedTopK`` loader (masked player/bullets/pad obs);
    delegates to :func:`qrokkun_env.agents.player_checkpoints.load_ranked_top_k_checkpoint`,
    which itself enforces schema version, action list, observation dims, and
    the checkpoint's own recorded ``state_dict_sha256``.
    """
    net, meta = load_ranked_top_k_checkpoint(path, device, eval_mode=True)
    out_meta = dict(meta)
    out_meta.setdefault("architecture", ARCHITECTURE_PLAYER_RANKED_TOP_K)
    return net, out_meta


def load_frozen_spawner_v4_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[SpawnerV4, dict]:
    """Existing ``SpawnerV4`` checkpoint-dict contract (``d_model``/
    ``hidden``/``state_dict``), matching ``corner_probe.load_spawner`` /
    ``render_v4_demo.py``. This introduces no new Spawner-checkpoint schema.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    d_model = int(ck.get("d_model", 128))
    hidden = int(ck.get("hidden", 256))
    net = SpawnerV4(d_model=d_model, hidden=hidden).to(device)
    net.load_state_dict(ck["state_dict"], strict=True)
    net.eval()
    meta = {
        "architecture": ARCHITECTURE_SPAWNER_V4,
        "d_model": d_model,
        "hidden": hidden,
        "state_dict_sha256": state_dict_sha256(net.state_dict()),
    }
    return net, meta


_ARCHITECTURE_LOADERS = {
    ARCHITECTURE_PLAYER_V1: load_player_v1_checkpoint,
    ARCHITECTURE_PLAYER_RANKED_TOP_K: load_player_ranked_topk_checkpoint,
    ARCHITECTURE_SPAWNER_V4: load_frozen_spawner_v4_checkpoint,
}


def load_checkpoint_for_role(ref: CheckpointRef, device: torch.device | str = "cpu") -> tuple[Any, dict]:
    """Fail-closed, architecture-dispatched checkpoint load.

    Validates the on-disk file SHA-256 against ``ref.sha256`` *before*
    calling ``torch.load`` at all. After loading, if ``ref.state_dict_sha256``
    is recorded, it is compared against the loader's own computed state-dict
    hash; a mismatch raises without ever handing the net back to the caller.
    """
    _verify_file_sha256(ref.path, ref.sha256)
    loader = _ARCHITECTURE_LOADERS.get(ref.architecture)
    if loader is None:
        raise UnsupportedArchitectureError(
            f"no eval loader for architecture {ref.architecture!r} (role {ref.role!r}); "
            f"supported: {sorted(_ARCHITECTURE_LOADERS)}"
        )
    net, meta = loader(ref.path, device)
    if ref.state_dict_sha256 is not None and meta.get("state_dict_sha256") != ref.state_dict_sha256:
        raise ChecksumMismatchError(
            f"state_dict sha256 mismatch for role {ref.role!r}: "
            f"expected {ref.state_dict_sha256}, got {meta.get('state_dict_sha256')}"
        )
    # Bind the returned object to the manifest identity.  Consumers of an
    # already-loaded checkpoint must be able to verify this provenance again
    # immediately before evaluation, without touching the file system.
    meta = dict(meta)
    meta.update(
        {
            "role": ref.role,
            "architecture": ref.architecture,
            "file_sha256": ref.sha256,
            "state_dict_sha256": meta.get("state_dict_sha256"),
        }
    )
    return net, meta


# --- Action selection (distinct per architecture) ---------------------------


@torch.no_grad()
def _act_player_v1(net: PlayerV1, env: Qrokkun26Env, device: torch.device | str, sample: bool) -> int:
    x = torch.tensor(vectorize_v1(env), dtype=torch.float32, device=device).unsqueeze(0)
    dist, _value = net(x)
    a = dist.sample() if sample else dist.probs.argmax(dim=-1)
    return int(a.item())


@torch.no_grad()
def _act_player_ranked_topk(net: Any, env: Qrokkun26Env, device: torch.device | str, sample: bool) -> int:
    p, b, m = encode_obs(env)
    pt = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
    bt = torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
    mt = torch.tensor(m, dtype=torch.bool, device=device).unsqueeze(0)
    dist, _value = net(pt, bt, mt)
    a = dist.sample() if sample else dist.probs.argmax(dim=-1)
    return int(a.item())


_PLAYER_ACTORS = {
    ARCHITECTURE_PLAYER_V1: _act_player_v1,
    ARCHITECTURE_PLAYER_RANKED_TOP_K: _act_player_ranked_topk,
}


@torch.no_grad()
def _act_spawner_v4(
    net: SpawnerV4, env: Qrokkun26Env, device: torch.device | str, sample: bool
) -> tuple[tuple[float, float], tuple[float, float], int]:
    p, b, m = encode_obs(env)
    pt = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
    bt = torch.tensor(b, dtype=torch.float32, device=device).unsqueeze(0)
    mt = torch.tensor(m, dtype=torch.bool, device=device).unsqueeze(0)
    birth, aim, kind, _value = net(pt, bt, mt)
    if sample:
        bv = birth.sample()
        av = aim.sample()
        kv = kind.sample()
    else:
        bv = birth.mean
        av = aim.mean
        kv = kind.probs.argmax(dim=-1)
    return (float(bv[0, 0]), float(bv[0, 1])), (float(av[0, 0]), float(av[0, 1])), int(kv.item())


def apply_player_action(env: Qrokkun26Env, a: int) -> None:
    """Physics-only player-move integration for one frame.

    This delegates directly to the single canonical implementation,
    ``qrokkun_env.train.both_v4.apply_player_action``, instead of keeping a
    second, independently-maintained copy: a duplicated implementation would
    risk silently diverging (e.g. dropping the float32 rounding both_v4
    applies at every step) from the physics this eval module is required to
    reproduce exactly. Reusing the training module's pure physics helper
    here does not import or invoke any optimizer/PPO/self-play/promotion
    code -- ``apply_player_action`` is a stateless, gradient-free function of
    ``(env, action)`` only.
    """
    _canonical_apply_player_action(env, a)


# --- Episode runner -----------------------------------------------------------


@torch.no_grad()
def run_frozen_spawner_episode(
    player_architecture: str,
    player_net: Any,
    spawner_net: SpawnerV4,
    device: torch.device | str,
    seed: int,
    cap_frames: int,
    *,
    player_sample: bool,
    spawner_sample: bool,
    rng_jitter: bool,
) -> Mapping[str, Any]:
    """Run exactly one eval-only episode of ``player_net`` vs the frozen
    ``spawner_net``, dispatching to the architecture-correct observation
    contract for the player, with *independent* player/Spawner sampling
    flags. No trajectory is recorded, no gradient is computed, and no
    checkpoint is written.

    Returns a structured outcome mapping with ``elapsed`` (seconds),
    ``frames`` (integer frame count actually simulated), ``hit`` (bool),
    ``censored`` (bool, ``not hit``), and ``termination_reason`` (``"hit"``
    or ``"cap"``). A plain float cannot distinguish a hit that lands exactly
    on the frame cap from a same-elapsed censor, so both the elapsed time
    and the explicit hit/censor classification are always returned together.

    Mirrors the physics loop of ``qrokkun_env.train.both_v4.run_episode``'s
    learned-Spawner branch exactly (spawn scheduling, action application,
    bullet integration, hit check), the only behavioral difference being
    that player and Spawner sampling are governed by separate flags.
    """
    if player_architecture not in _PLAYER_ACTORS:
        raise UnsupportedArchitectureError(
            f"no eval actor for player architecture {player_architecture!r}; "
            f"supported: {sorted(_PLAYER_ACTORS)}"
        )
    act_player = _PLAYER_ACTORS[player_architecture]

    env = Qrokkun26Env(seed=int(seed))
    env.reset(seed=int(seed))

    frames = 0
    for _ in range(int(cap_frames)):
        frames += 1
        env.elapsed += env.dt
        env.spawn_acc += env.dt
        interval = _spawn_interval(env.elapsed)
        spawns = 0
        while env.spawn_acc >= interval:
            env.spawn_acc -= interval
            birth, aim, kind = _act_spawner_v4(spawner_net, env, device, spawner_sample)
            spawn_continuous(env, birth, aim, kind, rng_jitter=rng_jitter)
            spawns += 1
            p_double = 0.0
            if env.elapsed > 18.0:
                p_double = 0.20
            elif env.elapsed > 8.0:
                p_double = 0.10
            if spawns == 1 and p_double > 0 and env.rng.randf() < p_double:
                env.spawn_acc += interval
            interval = _spawn_interval(env.elapsed)

        a = act_player(player_net, env, device, player_sample)
        apply_player_action(env, a)
        env._integrate_bullets()
        if env._check_hit():
            env.dead = True
            return {
                "elapsed": float(env.elapsed),
                "frames": frames,
                "hit": True,
                "censored": False,
                "termination_reason": "hit",
            }

    return {
        "elapsed": float(env.elapsed),
        "frames": frames,
        "hit": False,
        "censored": True,
        "termination_reason": "cap",
    }


def run_frozen_spawner_episode_crn(
    player_architecture: str,
    player_net: Any,
    spawner_net: SpawnerV4,
    device: torch.device | str,
    seed: int,
    cap_frames: int,
    *,
    player_sample: bool,
    spawner_sample: bool,
    rng_jitter: bool,
) -> Mapping[str, Any]:
    """CRN wrapper: forks the Torch RNG, seeded on ``seed``, only when
    sampling actually consumes it (``player_sample or spawner_sample``),
    mirroring ``qrokkun_env.eval_modes.run_eval_episode``. This is what makes
    two different player arms, evaluated at the same seed slot against the
    same stochastic Spawner, draw identical Spawner actions -- the shared
    nuisance factor CRN pairing is designed to cancel. Returns the same
    structured outcome mapping as :func:`run_frozen_spawner_episode`.
    """
    if player_sample or spawner_sample:
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            return run_frozen_spawner_episode(
                player_architecture,
                player_net,
                spawner_net,
                device,
                seed,
                cap_frames,
                player_sample=player_sample,
                spawner_sample=spawner_sample,
                rng_jitter=rng_jitter,
            )
    return run_frozen_spawner_episode(
        player_architecture,
        player_net,
        spawner_net,
        device,
        seed,
        cap_frames,
        player_sample=player_sample,
        spawner_sample=spawner_sample,
        rng_jitter=rng_jitter,
    )


def evaluate_role_outcomes(
    player_architecture: str,
    player_net: Any,
    spawner_net: SpawnerV4,
    device: torch.device | str,
    env_seeds: Sequence[int],
    cap_frames: int,
    *,
    spawner_sample: bool,
    rng_jitter: bool,
) -> list[Mapping[str, Any]]:
    """Per-seed structured outcomes for one player arm, player always
    deterministic. Each entry is the ``run_frozen_spawner_episode`` outcome
    mapping (``elapsed``/``frames``/``hit``/``censored``/``termination_reason``).
    """
    return [
        run_frozen_spawner_episode_crn(
            player_architecture,
            player_net,
            spawner_net,
            device,
            seed,
            cap_frames,
            player_sample=False,
            spawner_sample=spawner_sample,
            rng_jitter=rng_jitter,
        )
        for seed in env_seeds
    ]


# --- Comparison / report ------------------------------------------------------


def _preflight_report_package(package: FrozenSpawnerSeedPackage) -> None:
    """Validate raw representation and full package semantics before access."""
    # Repeat the package's complete construction-time validation at this
    # execution boundary.  A frozen dataclass can be altered by a malformed
    # deserializer (or object.__setattr__), so its original __post_init__ is
    # not sufficient before checkpoint access or an episode.  This covers
    # the eval-only flags as well as structural and identity invariants for
    # both formal and diagnostic reports.
    try:
        validate_serializable_frozen_spawner_seed_package(package)
    except FrozenSpawnerManifestError as exc:
        raise FrozenSpawnerEvalError(
            f"noncanonical frozen-Spawner package before report: {exc}"
        ) from exc

    # This check deliberately precedes checkpoint lookup and every episode:
    # no report is emitted from a package with an unsupported schema, even if
    # a malformed/deserialized instance bypassed construction validation.
    if (
        type(package.schema_version) is not int
        or package.schema_version != SUPPORTED_SCHEMA_VERSION
    ):
        raise FrozenSpawnerEvalError("noncanonical schema_version before frozen-Spawner report")
    # A formal schema is never emitted from a merely formal-labelled package.
    if package.contract_id == CONTRACT_ID and not is_canonical_formal_package(package):
        raise FrozenSpawnerEvalError("formal report requested for a noncanonical formal package")
    if package.contract_id not in (CONTRACT_ID, DIAGNOSTIC_CONTRACT_ID):
        raise FrozenSpawnerEvalError(f"unknown frozen-Spawner contract identity {package.contract_id!r}")
    if (
        package.contract_id == DIAGNOSTIC_CONTRACT_ID
        and package.preregistration_path is not None
    ):
        raise FrozenSpawnerEvalError("diagnostic report requested with a non-null preregistration path")
    if package.player_action_mode[0]:
        raise FrozenSpawnerEvalError(
            "player_action_mode must be deterministic (sample_policy=False) for this runner; "
            f"got {package.player_action_mode!r}"
        )


def _validate_loaded_checkpoint_identities(
    package: FrozenSpawnerSeedPackage,
    checkpoints: Mapping[str, tuple[Any, dict]],
) -> None:
    """Ensure loaded objects are bound to the package before any episode."""
    if not isinstance(checkpoints, Mapping):
        raise FrozenSpawnerEvalError("loaded checkpoint identities must be a mapping")
    for role, ref in package.checkpoints.items():
        try:
            net, meta = checkpoints[role]
        except (KeyError, TypeError, ValueError) as exc:
            raise FrozenSpawnerEvalError(
                f"missing or malformed loaded checkpoint identity for role {role!r}"
            ) from exc
        if not isinstance(meta, Mapping):
            raise FrozenSpawnerEvalError(f"malformed loaded checkpoint identity metadata for role {role!r}")
        expected = {
            "role": role,
            "architecture": ref.architecture,
            "file_sha256": ref.sha256,
        }
        if ref.state_dict_sha256 is not None:
            expected["state_dict_sha256"] = ref.state_dict_sha256
        for field, value in expected.items():
            if meta.get(field) != value:
                raise FrozenSpawnerEvalError(
                    f"loaded checkpoint identity mismatch for role {role!r}: "
                    f"{field} expected {value!r}, got {meta.get(field)!r}"
                )
        # Metadata is provenance asserted by the caller, not evidence that
        # this particular in-memory module still holds the pinned weights.
        # Bind the supplied object itself to each available pin immediately
        # before evaluation.  Formal reports require pins for all roles;
        # diagnostic packages may opt into the same check without acquiring
        # formal schema identity.
        if ref.state_dict_sha256 is not None:
            try:
                actual_state_dict_sha256 = state_dict_sha256(net.state_dict())
            except Exception as exc:
                raise FrozenSpawnerEvalError(
                    f"loaded checkpoint identity cannot verify supplied net state_dict for role {role!r}"
                ) from exc
            if actual_state_dict_sha256 != ref.state_dict_sha256:
                raise FrozenSpawnerEvalError(
                    f"loaded checkpoint identity mismatch for role {role!r}: "
                    f"state_dict_sha256 expected {ref.state_dict_sha256!r}, "
                    f"got {actual_state_dict_sha256!r} from supplied net"
                )


def run_frozen_spawner_comparison(
    package: FrozenSpawnerSeedPackage,
    checkpoints: Mapping[str, tuple[Any, dict]],
    device: torch.device | str = "cpu",
) -> dict:
    """Run the preregistered comparison for ``package`` given already-loaded
    ``(net, meta)`` pairs per role in ``checkpoints`` (see
    :func:`load_checkpoint_for_role`).

    Structurally: this function only runs eval episodes and computes
    statistics. It never constructs an optimizer, never calls a PPO update,
    never launches self-play, and never writes or selects a "promoted"
    checkpoint -- the returned report hardcodes
    ``promotion``/``training``/``self_play`` to ``False``.
    """
    _preflight_report_package(package)
    _validate_loaded_checkpoint_identities(package, checkpoints)

    spawner_net, _spawner_meta = checkpoints[ROLE_LEARNED_SPAWNER]
    spawner_sample = bool(package.spawner_action_mode[0])
    rng_jitter = bool(package.spawner_action_mode[1])
    cap_frames = package.episode_cap_frames

    player_roles = sorted({role for pair in package.comparison_pairs for role in pair})
    per_role_outcomes: dict[str, list[Mapping[str, Any]]] = {}
    for role in player_roles:
        net, meta = checkpoints[role]
        architecture = PLAYER_ROLE_ARCHITECTURE.get(role, meta.get("architecture"))
        per_role_outcomes[role] = evaluate_role_outcomes(
            architecture,
            net,
            spawner_net,
            device,
            package.env_seeds,
            cap_frames,
            spawner_sample=spawner_sample,
            rng_jitter=rng_jitter,
        )

    per_role_elapsed: dict[str, list[float]] = {
        role: [float(o["elapsed"]) for o in outcomes] for role, outcomes in per_role_outcomes.items()
    }

    cap_seconds = cap_frames * C.DT
    include_termination_reason = "termination_reason" in package.metrics
    restricted_stats = {}
    for role, outcomes in per_role_outcomes.items():
        stats = restricted_survival_stats(
            outcomes, cap=cap_seconds, early_death_threshold=EARLY_DEATH_THRESHOLD_SECONDS
        )
        if not include_termination_reason:
            stats.pop("termination_reasons", None)
        restricted_stats[role] = stats
    paired_comparisons = [
        {"pair": [a, b], **paired_difference_stats(per_role_elapsed[a], per_role_elapsed[b])}
        for a, b in package.comparison_pairs
    ]

    return {
        "schema": (
            "frozen_spawner_eval_report.v1"
            if package.contract_id == CONTRACT_ID
            else "frozen_spawner_diagnostic_report.v1"
        ),
        "manifest": package.to_json_dict(),
        "env_seeds": list(package.env_seeds),
        "per_seed_elapsed": {role: list(vals) for role, vals in per_role_elapsed.items()},
        "restricted_stats": restricted_stats,
        "paired_comparisons": paired_comparisons,
        "promotion": False,
        "training": False,
        "self_play": False,
    }


# --- CLI ----------------------------------------------------------------------

_NAMED_MODES = {"det_det": MODE_DET_DET, "det_stoch": MODE_DET_STOCH, "stoch_stoch": MODE_STOCH_STOCH}

# Hard-locked formal-report *parameter* identity for this one frozen
# preregistration: the player arm is always evaluated deterministically
# (det_det) and the frozen Spawner is always evaluated at its preregistered
# stochastic mode (stoch_stoch), against exactly the CRN seed window
# `qrokkun_env.eval_modes.PAIRED_EVAL_SEEDS` (3000..3029, 30 seeds) and the
# preregistered 4,200-frame episode cap. `build_parser()`'s argparse
# `choices` below reject *any* other value for these five fields (rather
# than silently coercing/clamping), so a caller cannot drive this CLI's
# `main()` entry point to a non-canonical parameter combination. Tests that
# need a smaller/faster synthetic run (e.g. to keep unit tests fast) must
# bypass this formal parser entirely and call the lower-level, still
# independently-configurable helpers (`run_frozen_spawner_episode`,
# `build_frozen_spawner_seed_package`) directly -- those internal helpers
# accept arbitrary seeds/caps/modes for testing and cannot themselves go
# through this locked CLI surface. Generic packages instead use the
# diagnostic identity (`frozen_spawner_diagnostic_v1`), a null preregistration
# path, and the diagnostic report schema. A formal package/report identity is
# available only when every formal invariant is present: the formal contract
# and preregistration path, schema version 1, modes, CRN seed lists, frame
# cap, and comparison pairs. The package validator and report boundary
# enforce those invariants independently of this CLI parser.
CANONICAL_PLAYER_ACTION_MODE = "det_det"
CANONICAL_SPAWNER_ACTION_MODE = "stoch_stoch"
CANONICAL_ENV_SEED_START = 3000
CANONICAL_ENV_SEED_COUNT = 30
CANONICAL_EPISODE_CAP_FRAMES = 4200


def build_prepare_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Eval-only frozen learned-Spawner PlayerV5 comparison runner. "
            "Never launches training/PPO/self-play/promotion. The player "
            "action mode, Spawner action mode, CRN seed window, and episode "
            "cap are all hard-locked to this preregistration's formal "
            "identity and cannot be overridden."
        )
    )
    ap.add_argument("--v1-teacher-path", required=True, type=Path)
    ap.add_argument("--v1-teacher-sha256", required=True)
    ap.add_argument("--v1-teacher-state-dict-sha256", default=None)
    ap.add_argument("--v5-bc-seed6-path", required=True, type=Path)
    ap.add_argument("--v5-bc-seed6-sha256", required=True)
    ap.add_argument("--v5-bc-seed6-state-dict-sha256", default=None)
    ap.add_argument("--v5-aux-u200-path", required=True, type=Path)
    ap.add_argument("--v5-aux-u200-sha256", required=True)
    ap.add_argument("--v5-aux-u200-state-dict-sha256", default=None)
    ap.add_argument("--learned-spawner-path", required=True, type=Path)
    ap.add_argument("--learned-spawner-sha256", required=True)
    ap.add_argument("--learned-spawner-state-dict-sha256", default=None)
    # Hard-locked: the player arm is always det_det and the frozen Spawner is
    # always stoch_stoch under this preregistration's formal identity; any
    # other value is rejected by argparse (SystemExit), not silently ignored.
    ap.add_argument(
        "--player-action-mode",
        default=CANONICAL_PLAYER_ACTION_MODE,
        choices=(CANONICAL_PLAYER_ACTION_MODE,),
    )
    ap.add_argument(
        "--spawner-action-mode",
        default=CANONICAL_SPAWNER_ACTION_MODE,
        choices=(CANONICAL_SPAWNER_ACTION_MODE,),
    )
    ap.add_argument(
        "--env-seed-start", type=int, default=CANONICAL_ENV_SEED_START, choices=(CANONICAL_ENV_SEED_START,)
    )
    ap.add_argument(
        "--env-seed-count", type=int, default=CANONICAL_ENV_SEED_COUNT, choices=(CANONICAL_ENV_SEED_COUNT,)
    )
    ap.add_argument(
        "--episode-cap-frames",
        type=int,
        default=CANONICAL_EPISODE_CAP_FRAMES,
        choices=(CANONICAL_EPISODE_CAP_FRAMES,),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-manifest", required=True, type=Path)
    return ap


def build_parser() -> argparse.ArgumentParser:
    """Parser for formal execution only.

    Artifact selection is exclusively a preparation concern.  Keeping every
    role path/hash option out of this parser makes it impossible for one
    formal invocation to select, write, and execute an artifact manifest.
    """
    ap = argparse.ArgumentParser(
        description="Execute only an already prepared locked frozen-Spawner manifest."
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    return ap


def build_seed_package_from_args(args: argparse.Namespace) -> FrozenSpawnerSeedPackage:
    checkpoints = {
        ROLE_V1_TEACHER: CheckpointRef(
            role=ROLE_V1_TEACHER,
            path=str(args.v1_teacher_path),
            sha256=args.v1_teacher_sha256,
            architecture=ARCHITECTURE_PLAYER_V1,
            state_dict_sha256=args.v1_teacher_state_dict_sha256,
        ),
        ROLE_V5_BC_SEED6: CheckpointRef(
            role=ROLE_V5_BC_SEED6,
            path=str(args.v5_bc_seed6_path),
            sha256=args.v5_bc_seed6_sha256,
            architecture=ARCHITECTURE_PLAYER_RANKED_TOP_K,
            state_dict_sha256=args.v5_bc_seed6_state_dict_sha256,
        ),
        ROLE_V5_AUX_U200: CheckpointRef(
            role=ROLE_V5_AUX_U200,
            path=str(args.v5_aux_u200_path),
            sha256=args.v5_aux_u200_sha256,
            architecture=ARCHITECTURE_PLAYER_RANKED_TOP_K,
            state_dict_sha256=args.v5_aux_u200_state_dict_sha256,
        ),
        ROLE_LEARNED_SPAWNER: CheckpointRef(
            role=ROLE_LEARNED_SPAWNER,
            path=str(args.learned_spawner_path),
            sha256=args.learned_spawner_sha256,
            architecture=ARCHITECTURE_SPAWNER_V4,
            state_dict_sha256=args.learned_spawner_state_dict_sha256,
        ),
    }
    player_action_mode = _NAMED_MODES[args.player_action_mode]
    spawner_action_mode = _NAMED_MODES[args.spawner_action_mode]
    env_seeds = tuple(range(args.env_seed_start, args.env_seed_start + args.env_seed_count))
    spawner_seeds = env_seeds if spawner_action_mode[0] else ()
    return build_frozen_spawner_seed_package(
        checkpoints=checkpoints,
        player_action_mode=player_action_mode,
        spawner_action_mode=spawner_action_mode,
        env_seeds=env_seeds,
        spawner_seeds=spawner_seeds,
        episode_cap_frames=int(args.episode_cap_frames),
        comparison_pairs=COMPARISON_PAIRS,
    )


def prepare_main(argv: list[str] | None = None) -> FrozenSpawnerSeedPackage:
    """Write a concrete locked manifest without opening a checkpoint or episode.

    This is intentionally the only CLI surface that accepts artifact
    selectors. It records caller-supplied, reviewable identity pins but does
    not bless or load them; formal execution independently verifies them.
    """
    args = build_prepare_parser().parse_args(argv)
    package = build_seed_package_from_args(args)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(package.to_json_dict(), indent=2))
    return package


def load_locked_manifest(path: Path | str) -> FrozenSpawnerSeedPackage:
    """Deserialize an existing manifest through the same strict constructor.

    No permissive defaults are applied: missing, extra, or non-JSON manifest
    fields fail before any checkpoint is opened.
    """
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenSpawnerEvalError(f"cannot read locked manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FrozenSpawnerEvalError("locked manifest must be a JSON object")
    expected = {
        "schema_version", "checkpoints", "player_action_mode", "spawner_action_mode",
        "env_seeds", "spawner_seeds", "episode_cap_frames", "comparison_pairs", "metrics",
        "ci_method", "promotion", "training", "self_play", "contract_id", "preregistration_path",
    }
    if set(raw) != expected:
        raise FrozenSpawnerEvalError("locked manifest has missing or unexpected fields")
    try:
        refs = {
            role: CheckpointRef(
                role=value["role"], path=value["path"], sha256=value["sha256"],
                architecture=value["architecture"], state_dict_sha256=value["state_dict_sha256"],
            )
            for role, value in raw["checkpoints"].items()
        }
        return FrozenSpawnerSeedPackage(
            checkpoints=refs, player_action_mode=tuple(raw["player_action_mode"]),
            spawner_action_mode=tuple(raw["spawner_action_mode"]), env_seeds=tuple(raw["env_seeds"]),
            spawner_seeds=tuple(raw["spawner_seeds"]), episode_cap_frames=raw["episode_cap_frames"],
            comparison_pairs=tuple(tuple(pair) for pair in raw["comparison_pairs"]),
            metrics=tuple(raw["metrics"]), ci_method=raw["ci_method"], schema_version=raw["schema_version"],
            promotion=raw["promotion"], training=raw["training"], self_play=raw["self_play"],
            contract_id=raw["contract_id"], preregistration_path=raw["preregistration_path"],
        )
    except (KeyError, TypeError, FrozenSpawnerManifestError) as exc:
        raise FrozenSpawnerEvalError(f"invalid locked manifest: {exc}") from exc


def main(argv: list[str] | None = None) -> dict:
    """Execute only a pre-existing formal manifest, fail closed before episodes."""
    args = build_parser().parse_args(argv)
    package = load_locked_manifest(args.manifest)
    _preflight_report_package(package)
    device = args.device
    checkpoints = {role: load_checkpoint_for_role(ref, device) for role, ref in package.checkpoints.items()}
    # Repeat both concrete file and in-memory net identity checks after all
    # loads and immediately before the first possible episode.
    for ref in package.checkpoints.values():
        _verify_file_sha256(ref.path, ref.sha256)
    _validate_loaded_checkpoint_identities(package, checkpoints)
    report = run_frozen_spawner_comparison(package, checkpoints, device=device)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
