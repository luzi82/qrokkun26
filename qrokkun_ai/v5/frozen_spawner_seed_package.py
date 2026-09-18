"""Eval-only, immutable "seed package" / manifest for the frozen learned-Spawner
PlayerV5 diagnostic comparison.

This module has nothing to do with training. It exists to make the inputs of
a future eval-only comparison (V1 teacher vs seed-6 BC vs aux-u200 PlayerV5,
all evaluated against one fixed, pre-selected learned Spawner) explicit and
machine-checked, per
docs/journal/<...>_player_v5_frozen_spawner_eval_preregistration.txt:

* four fixed checkpoint roles, identified by path + SHA-256, with an
  explicit architecture tag each;
* player and Spawner action modes, restricted to the three named modes
  already defined by :mod:`qrokkun_ai.v4.eval_modes`;
* common-random-number (CRN) environment seeds and Spawner sampling seeds;
* a locked episode cap;
* a restricted metric allow-list (no ad hoc "gate" metrics);
* pre-specified comparison pairs (each pair of checkpoint roles to be
  compared, paired on CRN seed slot).

The resulting :class:`FrozenSpawnerSeedPackage` is an immutable value object:
``promotion``, ``training``, and ``self_play`` are always ``False`` and
cannot be requested as ``True`` (the builder raises ``TypeError`` if a
caller tries), and the object cannot be mutated after construction. It is
JSON-serializable via :meth:`FrozenSpawnerSeedPackage.to_json_dict`.

Nothing here requires a real checkpoint file to exist on disk: paths/SHA-256
values are just recorded strings, validated for shape only. Unit tests use
synthetic placeholder identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from qrokkun_ai.v4.eval_modes import (
    MODE_DET_DET,
    MODE_DET_STOCH,
    MODE_STOCH_STOCH,
    validate_eval_mode,
)

ROLE_V1_TEACHER = "v1_teacher"
ROLE_V5_BC_SEED6 = "v5_bc_seed6"
ROLE_V5_AUX_U200 = "v5_aux_u200"
ROLE_LEARNED_SPAWNER = "learned_spawner"

REQUIRED_ROLES: tuple[str, ...] = (
    ROLE_V1_TEACHER,
    ROLE_V5_BC_SEED6,
    ROLE_V5_AUX_U200,
    ROLE_LEARNED_SPAWNER,
)

# Fixed role -> architecture mapping, locked here (not left to callers) so
# that a checkpoint role naming the wrong architecture is rejected by the
# seed package itself, before any checkpoint is ever loaded.
ROLE_ARCHITECTURE: Mapping[str, str] = {
    ROLE_V1_TEACHER: "player_v1",
    ROLE_V5_BC_SEED6: "player_ranked_topk",
    ROLE_V5_AUX_U200: "player_ranked_topk",
    ROLE_LEARNED_SPAWNER: "spawner_v4",
}

# The exact two preregistered comparison pairs, in their preregistered
# order (see docs/journal/<...>_player_v5_frozen_spawner_eval_preregistration.txt):
# V1-vs-BC, then BC-vs-aux. The seed package accepts only this exact
# sequence -- no reordering, no dropping, no substituting, no extra pairs.
PREREGISTERED_COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6),
    (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200),
)

_NAMED_ACTION_MODES = {MODE_DET_DET, MODE_DET_STOCH, MODE_STOCH_STOCH}

_ALLOWED_METRICS: tuple[str, ...] = (
    "restricted_mean",
    "median",
    "std",
    "min",
    "max",
    "censor_count",
    "per_seed_values",
    "paired_diff_mean",
    "paired_diff_median",
    "paired_diff_ci95",
    "hit_rate",
    "termination_reason",
    "early_death_rate",
)

DEFAULT_METRICS: tuple[str, ...] = _ALLOWED_METRICS

# Stable, immutable identities for the preregistered formal protocol and
# explicitly non-preregistered synthetic/diagnostic runs.
CONTRACT_ID = "player_v5_frozen_spawner_eval_v1"
PREREGISTRATION_PATH = "docs/journal/1789263159_player_v5_frozen_spawner_eval_preregistration.txt"
DIAGNOSTIC_CONTRACT_ID = "frozen_spawner_diagnostic_v1"
FORMAL_PLAYER_ACTION_MODE = MODE_DET_DET
FORMAL_SPAWNER_ACTION_MODE = MODE_STOCH_STOCH
FORMAL_ENV_SEEDS = tuple(range(3000, 3030))
FORMAL_EPISODE_CAP_FRAMES = 4200
SUPPORTED_SCHEMA_VERSION = 1


class FrozenSpawnerManifestError(ValueError):
    """Raised when a frozen-Spawner seed package is malformed or conflicting."""


_ALLOWED_CI_METHODS = ("paired_t95",)


def _validate_contract_identity(
    *, contract_id: str, preregistration_path: str | None,
    checkpoints: Mapping[str, "CheckpointRef"],
    player_action_mode: tuple[bool, bool], spawner_action_mode: tuple[bool, bool],
    env_seeds: tuple[int, ...], spawner_seeds: tuple[int, ...],
    episode_cap_frames: int, comparison_pairs: tuple[tuple[str, ...], ...],
    metrics: tuple[str, ...],
) -> None:
    if contract_id == CONTRACT_ID:
        canonical = (
            player_action_mode == FORMAL_PLAYER_ACTION_MODE
            and spawner_action_mode == FORMAL_SPAWNER_ACTION_MODE
            and env_seeds == FORMAL_ENV_SEEDS
            and spawner_seeds == FORMAL_ENV_SEEDS
            and episode_cap_frames == FORMAL_EPISODE_CAP_FRAMES
            and comparison_pairs == PREREGISTERED_COMPARISON_PAIRS
            and metrics == DEFAULT_METRICS
        )
        if preregistration_path != PREREGISTRATION_PATH or not canonical:
            raise FrozenSpawnerManifestError(
                "formal contract identity requires the fully canonical formal modes, seeds, "
                "episode cap, comparison pairs, metrics, and preregistration path"
            )
        missing_state_dict_pins = [
            role for role in REQUIRED_ROLES
            if checkpoints[role].state_dict_sha256 is None
        ]
        if missing_state_dict_pins:
            raise FrozenSpawnerManifestError(
                "formal contract identity requires state_dict_sha256 pins for every checkpoint role; "
                f"missing {missing_state_dict_pins}"
            )
    elif contract_id == DIAGNOSTIC_CONTRACT_ID:
        if preregistration_path is not None:
            raise FrozenSpawnerManifestError("diagnostic contract identity requires preregistration_path=None")
    else:
        raise FrozenSpawnerManifestError(f"unknown contract_id {contract_id!r}")


def is_canonical_formal_package(package: "FrozenSpawnerSeedPackage") -> bool:
    """Whether a package has every field required for the formal report."""
    return (
        type(package.schema_version) is int
        and package.schema_version == SUPPORTED_SCHEMA_VERSION
        and package.contract_id == CONTRACT_ID
        and package.preregistration_path == PREREGISTRATION_PATH
        and package.player_action_mode == FORMAL_PLAYER_ACTION_MODE
        and package.spawner_action_mode == FORMAL_SPAWNER_ACTION_MODE
        and package.env_seeds == FORMAL_ENV_SEEDS
        and package.spawner_seeds == FORMAL_ENV_SEEDS
        and package.episode_cap_frames == FORMAL_EPISODE_CAP_FRAMES
        and package.comparison_pairs == PREREGISTERED_COMPARISON_PAIRS
        and package.metrics == DEFAULT_METRICS
        and all(
            package.checkpoints[role].state_dict_sha256 is not None
            for role in REQUIRED_ROLES
        )
    )


def _validate_seed_package_inputs(
    *,
    checkpoints: Mapping[str, "CheckpointRef"],
    player_action_mode: tuple[bool, bool],
    spawner_action_mode: tuple[bool, bool],
    env_seeds: tuple[int, ...],
    spawner_seeds: tuple[int, ...],
    episode_cap_frames: int,
    comparison_pairs: tuple[tuple[str, ...], ...],
    metrics: tuple[str, ...],
    ci_method: str,
) -> None:
    """Single, shared structural validator applied to *every* construction
    path (builder and direct dataclass construction alike), so direct
    construction cannot bypass, or diverge from, the builder's validation of
    checkpoint roles, action modes, CRN seeds, episode cap, comparison pairs,
    metrics, and ci_method.
    """
    # --- checkpoints ---
    if not isinstance(checkpoints, Mapping):
        raise FrozenSpawnerManifestError(f"checkpoints must be a mapping, got {type(checkpoints)!r}")
    missing = [role for role in REQUIRED_ROLES if role not in checkpoints]
    if missing:
        raise FrozenSpawnerManifestError(f"missing required checkpoint role(s): {missing}")
    unknown_roles = [role for role in checkpoints if role not in REQUIRED_ROLES]
    if unknown_roles:
        raise FrozenSpawnerManifestError(f"unknown checkpoint role(s) in checkpoints: {unknown_roles}")
    for role in REQUIRED_ROLES:
        ref = checkpoints[role]
        if not isinstance(ref, CheckpointRef):
            raise FrozenSpawnerManifestError(f"checkpoints[{role!r}] must be a CheckpointRef, got {type(ref)!r}")
        # Check the actual fields again: a frozen dataclass can still be
        # altered by a deserializer or object.__setattr__ before it is handed
        # to the public builder.
        try:
            CheckpointRef(
                role=ref.role,
                path=ref.path,
                sha256=ref.sha256,
                architecture=ref.architecture,
                state_dict_sha256=ref.state_dict_sha256,
            )
        except FrozenSpawnerManifestError as exc:
            raise FrozenSpawnerManifestError(
                f"checkpoints[{role!r}] has invalid checkpoint identity: {exc}"
            ) from exc
        if ref.role != role:
            raise FrozenSpawnerManifestError(f"checkpoints[{role!r}].role {ref.role!r} != key {role!r}")
        if not ref.path:
            raise FrozenSpawnerManifestError(f"checkpoints[{role!r}].path must be non-empty")
        if not _is_sha256_hex(ref.sha256):
            raise FrozenSpawnerManifestError(f"checkpoints[{role!r}].sha256 is not a 64-hex-char sha256: {ref.sha256!r}")
        if ref.state_dict_sha256 is not None and not _is_sha256_hex(ref.state_dict_sha256):
            raise FrozenSpawnerManifestError(
                f"checkpoints[{role!r}].state_dict_sha256 is not a 64-hex-char sha256: {ref.state_dict_sha256!r}"
            )
        if not ref.architecture:
            raise FrozenSpawnerManifestError(f"checkpoints[{role!r}].architecture must be non-empty")
        expected_architecture = ROLE_ARCHITECTURE[role]
        if ref.architecture != expected_architecture:
            raise FrozenSpawnerManifestError(
                f"checkpoints[{role!r}].architecture {ref.architecture!r} != required "
                f"architecture {expected_architecture!r} for role {role!r}"
            )

    # --- action modes (shape-checked *before* any indexing into them) ---
    for field_name, mode in (("player_action_mode", player_action_mode), ("spawner_action_mode", spawner_action_mode)):
        if len(mode) != 2:
            raise FrozenSpawnerManifestError(
                f"{field_name} must be a 2-tuple of (sample_policy, rng_jitter) bools, got {mode!r}"
            )
        try:
            validate_eval_mode(*mode)
        except ValueError as exc:
            raise FrozenSpawnerManifestError(f"{field_name} invalid: {exc}") from exc
        if mode not in _NAMED_ACTION_MODES:
            raise FrozenSpawnerManifestError(f"{field_name} {mode!r} is not a named eval mode")

    # --- env seeds ---
    if not env_seeds:
        raise FrozenSpawnerManifestError("env_seeds must be non-empty")
    if len(set(env_seeds)) != len(env_seeds):
        raise FrozenSpawnerManifestError("env_seeds contains duplicate seed(s)")

    # --- spawner seeds / CRN semantics ---
    # eval_modes.run_eval_episode is keyed by a single episode_seed used for
    # both the env seed and (when sample_policy is on) the Torch RNG fork
    # seed. There is no separate per-episode Spawner sampling seed threaded
    # through that shared runner, so the only currently-compatible way to
    # record a "Spawner sampling seed sequence" for a stochastic Spawner is
    # for it to equal the env-seed sequence exactly (same values, same
    # order/slot).
    spawner_is_stochastic = bool(spawner_action_mode[0])
    if spawner_is_stochastic:
        if not spawner_seeds:
            raise FrozenSpawnerManifestError(
                "spawner_action_mode is stochastic but spawner_seeds is empty; "
                "CRN pairing requires an explicit per-seed Spawner sampling seed"
            )
        if len(spawner_seeds) != len(env_seeds):
            raise FrozenSpawnerManifestError(
                f"spawner_seeds length {len(spawner_seeds)} != env_seeds length {len(env_seeds)}; "
                "CRN pairing requires one Spawner sampling seed per environment seed slot"
            )
        if len(set(spawner_seeds)) != len(spawner_seeds):
            raise FrozenSpawnerManifestError("spawner_seeds contains duplicate seed(s)")
        if tuple(spawner_seeds) != tuple(env_seeds):
            raise FrozenSpawnerManifestError(
                "spawner_seeds must equal env_seeds exactly when spawner_action_mode is stochastic: "
                "eval_modes.run_eval_episode reuses a single episode_seed for both the environment "
                "and the Spawner-sampling RNG fork, so CRN pairing requires spawner_seeds == env_seeds"
            )
    elif spawner_seeds and len(spawner_seeds) != len(env_seeds):
        raise FrozenSpawnerManifestError(
            f"spawner_seeds length {len(spawner_seeds)} != env_seeds length {len(env_seeds)}"
        )

    # --- episode cap ---
    if episode_cap_frames <= 0:
        raise FrozenSpawnerManifestError(f"episode_cap_frames must be positive, got {episode_cap_frames}")

    # --- comparison pairs ---
    if not comparison_pairs:
        raise FrozenSpawnerManifestError("comparison_pairs must be non-empty")
    for pair in comparison_pairs:
        if len(pair) != 2:
            raise FrozenSpawnerManifestError(f"comparison pair must have exactly 2 roles: {pair!r}")
        a, b = pair
        for role in (a, b):
            if role not in checkpoints:
                raise FrozenSpawnerManifestError(f"comparison pair references unknown checkpoint role: {role!r}")
        if a == b:
            raise FrozenSpawnerManifestError(f"comparison pair references the same role twice: {pair!r}")

    if tuple(tuple(pair) for pair in comparison_pairs) != PREREGISTERED_COMPARISON_PAIRS:
        raise FrozenSpawnerManifestError(
            f"comparison_pairs must be exactly the preregistered pairs in order "
            f"{PREREGISTERED_COMPARISON_PAIRS!r}, got {tuple(tuple(p) for p in comparison_pairs)!r}"
        )

    # --- metrics ---
    if not metrics:
        raise FrozenSpawnerManifestError("metrics must be non-empty")
    unknown_metrics = [m for m in metrics if m not in _ALLOWED_METRICS]
    if unknown_metrics:
        raise FrozenSpawnerManifestError(
            f"metrics contains unapproved name(s) {unknown_metrics}; allowed: {sorted(_ALLOWED_METRICS)}"
        )

    # --- ci_method ---
    if ci_method not in _ALLOWED_CI_METHODS:
        raise FrozenSpawnerManifestError(
            f"ci_method {ci_method!r} is not permitted; the preregistered CI method "
            f"is locked to {_ALLOWED_CI_METHODS!r}"
        )


def _is_sha256_hex(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class CheckpointRef:
    """One fixed, pre-selected checkpoint identity (path + SHA-256 record)."""

    role: str
    path: str
    sha256: str
    architecture: str
    state_dict_sha256: str | None = None

    def __post_init__(self) -> None:
        """Keep every identity field directly serializable and unambiguous."""
        if type(self.role) is not str or not self.role:
            raise FrozenSpawnerManifestError(f"checkpoint role must be a non-empty str, got {self.role!r}")
        if type(self.path) is not str or not self.path:
            raise FrozenSpawnerManifestError(f"checkpoint path must be a non-empty str, got {self.path!r}")
        if not _is_sha256_hex(self.sha256):
            raise FrozenSpawnerManifestError(
                f"checkpoint sha256 is not a 64-hex-char sha256: {self.sha256!r}"
            )
        if type(self.architecture) is not str or not self.architecture:
            raise FrozenSpawnerManifestError(
                f"checkpoint architecture must be a non-empty str, got {self.architecture!r}"
            )
        if self.state_dict_sha256 is not None and not _is_sha256_hex(self.state_dict_sha256):
            raise FrozenSpawnerManifestError(
                "checkpoint state_dict_sha256 is not None or a 64-hex-char sha256: "
                f"{self.state_dict_sha256!r}"
            )


@dataclass(frozen=True)
class FrozenSpawnerSeedPackage:
    """Immutable, validated eval-only manifest. Construct via
    :func:`build_frozen_spawner_seed_package`; do not instantiate directly.
    """

    checkpoints: Mapping[str, CheckpointRef]
    player_action_mode: tuple[bool, bool]
    spawner_action_mode: tuple[bool, bool]
    env_seeds: tuple[int, ...]
    spawner_seeds: tuple[int, ...]
    episode_cap_frames: int
    comparison_pairs: tuple[tuple[str, str], ...]
    metrics: tuple[str, ...] = DEFAULT_METRICS
    ci_method: str = "paired_t95"
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    promotion: bool = False
    training: bool = False
    self_play: bool = False
    contract_id: str = CONTRACT_ID
    preregistration_path: str | None = PREREGISTRATION_PATH

    ALLOWED_METRICS = _ALLOWED_METRICS

    def __post_init__(self) -> None:
        # Structural eval-only guarantee: even direct construction of this
        # public dataclass (bypassing the builder) must not be able to
        # represent promotion/training/self-play, and its checkpoint mapping
        # must not be mutable in place.
        for name, value in (
            ("promotion", self.promotion),
            ("training", self.training),
            ("self_play", self.self_play),
        ):
            if value is not False:
                raise TypeError(
                    f"{name}=True is not permitted: this is an eval-only frozen-Spawner "
                    f"seed package and cannot request {name}"
                )

        if type(self.schema_version) is not int or self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise FrozenSpawnerManifestError(
                f"schema_version must be the supported integer {SUPPORTED_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )

        # Defensively canonicalize mutable sequence inputs to immutable
        # tuples so later mutation of a caller-owned list cannot alter this
        # (nominally frozen) package.
        object.__setattr__(self, "player_action_mode", tuple(self.player_action_mode))
        object.__setattr__(self, "spawner_action_mode", tuple(self.spawner_action_mode))
        object.__setattr__(self, "env_seeds", tuple(int(s) for s in self.env_seeds))
        object.__setattr__(self, "spawner_seeds", tuple(int(s) for s in self.spawner_seeds))
        object.__setattr__(self, "episode_cap_frames", int(self.episode_cap_frames))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(
            self, "comparison_pairs", tuple(tuple(pair) for pair in self.comparison_pairs)
        )

        # Checkpoints must be a Mapping (not e.g. a plain list of (role, ref)
        # pairs) *before* being detached/copied: dict() would otherwise
        # silently accept a list of 2-tuples, diverging from the builder's
        # explicit Mapping check. Always detach/copy the checkpoint mapping
        # before wrapping it, even when the input is already a
        # MappingProxyType: a proxy is just a read-only *view* onto its
        # backing dict, so wrapping it in place would let a caller-held
        # reference to that backing dict mutate this (nominally frozen)
        # package after construction.
        if not isinstance(self.checkpoints, Mapping):
            raise FrozenSpawnerManifestError(
                f"checkpoints must be a mapping, got {type(self.checkpoints)!r}"
            )
        object.__setattr__(self, "checkpoints", MappingProxyType(dict(self.checkpoints)))

        _validate_seed_package_inputs(
            checkpoints=self.checkpoints,
            player_action_mode=self.player_action_mode,
            spawner_action_mode=self.spawner_action_mode,
            env_seeds=self.env_seeds,
            spawner_seeds=self.spawner_seeds,
            episode_cap_frames=self.episode_cap_frames,
            comparison_pairs=self.comparison_pairs,
            metrics=self.metrics,
            ci_method=self.ci_method,
        )
        _validate_contract_identity(
            contract_id=self.contract_id,
            preregistration_path=self.preregistration_path,
            checkpoints=self.checkpoints,
            player_action_mode=self.player_action_mode,
            spawner_action_mode=self.spawner_action_mode,
            env_seeds=self.env_seeds,
            spawner_seeds=self.spawner_seeds,
            episode_cap_frames=self.episode_cap_frames,
            comparison_pairs=self.comparison_pairs,
            metrics=self.metrics,
        )

    def to_json_dict(self) -> dict:
        """A plain, JSON-serializable dict of this validated manifest."""
        validate_serializable_frozen_spawner_seed_package(self)
        return {
            "schema_version": self.schema_version,
            "checkpoints": {
                role: {
                    "role": ref.role,
                    "path": ref.path,
                    "sha256": ref.sha256,
                    "architecture": ref.architecture,
                    "state_dict_sha256": ref.state_dict_sha256,
                }
                for role, ref in self.checkpoints.items()
            },
            "player_action_mode": list(self.player_action_mode),
            "spawner_action_mode": list(self.spawner_action_mode),
            "env_seeds": list(self.env_seeds),
            "spawner_seeds": list(self.spawner_seeds),
            "episode_cap_frames": self.episode_cap_frames,
            "comparison_pairs": [list(pair) for pair in self.comparison_pairs],
            "metrics": list(self.metrics),
            "ci_method": self.ci_method,
            "promotion": self.promotion,
            "training": self.training,
            "self_play": self.self_play,
            "contract_id": self.contract_id,
            "preregistration_path": self.preregistration_path,
        }


def validate_serializable_frozen_spawner_seed_package(
    package: FrozenSpawnerSeedPackage,
) -> None:
    """Validate the raw representation and semantics of a serializable package.

    Frozen dataclasses may be changed with ``object.__setattr__`` after
    construction.  Serialization must validate those actual values, rather
    than reconstructing/canonicalizing them and accidentally concealing a
    malformed manifest.  This helper is deliberately independent of the CLI
    so both serialization and report execution share one preflight.
    """
    if type(package) is not FrozenSpawnerSeedPackage:
        raise FrozenSpawnerManifestError("package must be an exact FrozenSpawnerSeedPackage")
    if type(package.checkpoints) is not MappingProxyType:
        raise FrozenSpawnerManifestError("checkpoints must be an immutable mapping")
    for role, ref in package.checkpoints.items():
        if type(role) is not str or type(ref) is not CheckpointRef:
            raise FrozenSpawnerManifestError("checkpoints keys and refs must have exact manifest types")
        try:
            CheckpointRef(
                role=ref.role,
                path=ref.path,
                sha256=ref.sha256,
                architecture=ref.architecture,
                state_dict_sha256=ref.state_dict_sha256,
            )
        except FrozenSpawnerManifestError as exc:
            raise FrozenSpawnerManifestError(
                f"checkpoints[{role!r}] has invalid checkpoint identity: {exc}"
            ) from exc
    for field_name in ("player_action_mode", "spawner_action_mode"):
        mode = getattr(package, field_name)
        if type(mode) is not tuple or len(mode) != 2 or any(type(value) is not bool for value in mode):
            raise FrozenSpawnerManifestError(
                f"{field_name} must be a 2-tuple of exact bools for serialization"
            )
    for field_name in ("env_seeds", "spawner_seeds"):
        seeds = getattr(package, field_name)
        if type(seeds) is not tuple or any(type(seed) is not int for seed in seeds):
            raise FrozenSpawnerManifestError(
                f"{field_name} must be a tuple of exact ints for serialization"
            )
    if type(package.episode_cap_frames) is not int:
        raise FrozenSpawnerManifestError("episode_cap_frames must be an exact int for serialization")
    if (
        type(package.comparison_pairs) is not tuple
        or any(
            type(pair) is not tuple or len(pair) != 2 or any(type(role) is not str for role in pair)
            for pair in package.comparison_pairs
        )
    ):
        raise FrozenSpawnerManifestError("comparison_pairs must be 2-tuples of exact strs for serialization")
    if type(package.metrics) is not tuple or any(type(metric) is not str for metric in package.metrics):
        raise FrozenSpawnerManifestError("metrics must be a tuple of exact strs for serialization")
    if type(package.ci_method) is not str:
        raise FrozenSpawnerManifestError("ci_method must be an exact str for serialization")
    if type(package.schema_version) is not int:
        raise FrozenSpawnerManifestError("schema_version must be an exact int for serialization")
    for field in _LOCKED_FALSE_FIELDS:
        if getattr(package, field) is not False:
            raise FrozenSpawnerManifestError(f"{field} must be exactly False for serialization")
    if type(package.contract_id) is not str:
        raise FrozenSpawnerManifestError("contract_id must be an exact str for serialization")
    if package.preregistration_path is not None and type(package.preregistration_path) is not str:
        raise FrozenSpawnerManifestError("preregistration_path must be None or an exact str for serialization")

    if package.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise FrozenSpawnerManifestError(
            f"schema_version must be the supported integer {SUPPORTED_SCHEMA_VERSION}, "
            f"got {package.schema_version!r}"
        )
    _validate_seed_package_inputs(
        checkpoints=package.checkpoints,
        player_action_mode=package.player_action_mode,
        spawner_action_mode=package.spawner_action_mode,
        env_seeds=package.env_seeds,
        spawner_seeds=package.spawner_seeds,
        episode_cap_frames=package.episode_cap_frames,
        comparison_pairs=package.comparison_pairs,
        metrics=package.metrics,
        ci_method=package.ci_method,
    )
    _validate_contract_identity(
        contract_id=package.contract_id,
        preregistration_path=package.preregistration_path,
        checkpoints=package.checkpoints,
        player_action_mode=package.player_action_mode,
        spawner_action_mode=package.spawner_action_mode,
        env_seeds=package.env_seeds,
        spawner_seeds=package.spawner_seeds,
        episode_cap_frames=package.episode_cap_frames,
        comparison_pairs=package.comparison_pairs,
        metrics=package.metrics,
    )


# Field names on FrozenSpawnerSeedPackage that a caller must never be able to
# request non-default (False) values for via the public builder.
_LOCKED_FALSE_FIELDS = ("promotion", "training", "self_play")


def build_frozen_spawner_seed_package(
    *,
    checkpoints: Mapping[str, CheckpointRef],
    player_action_mode: tuple[bool, bool],
    spawner_action_mode: tuple[bool, bool],
    env_seeds: tuple[int, ...],
    spawner_seeds: tuple[int, ...],
    episode_cap_frames: int,
    comparison_pairs: tuple[tuple[str, str], ...],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    ci_method: str = "paired_t95",
    schema_version: int = SUPPORTED_SCHEMA_VERSION,
    contract_id: str = CONTRACT_ID,
    preregistration_path: str | None = PREREGISTRATION_PATH,
    promotion: bool = False,
    training: bool = False,
    self_play: bool = False,
) -> FrozenSpawnerSeedPackage:
    """Validate inputs and build an immutable frozen-Spawner seed package.

    Raises :class:`FrozenSpawnerManifestError` for malformed/conflicting
    inputs, and ``TypeError`` if the caller attempts to request
    ``promotion``/``training``/``self_play`` as ``True`` (this eval-only
    package structurally cannot represent any of those).
    """
    for name, value in (("promotion", promotion), ("training", training), ("self_play", self_play)):
        if value is not False:
            raise TypeError(
                f"{name}=True is not permitted: this is an eval-only frozen-Spawner "
                f"seed package and cannot request {name}"
            )

    if type(schema_version) is not int or schema_version != SUPPORTED_SCHEMA_VERSION:
        raise FrozenSpawnerManifestError(
            f"schema_version must be the supported integer {SUPPORTED_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )

    # Canonicalize sequence-ish inputs up front so the shared validator sees
    # the same shapes the dataclass itself will canonicalize to; actual
    # rejection logic lives solely in _validate_seed_package_inputs so the
    # builder and direct dataclass construction cannot drift apart.
    if not isinstance(checkpoints, Mapping):
        raise FrozenSpawnerManifestError(f"checkpoints must be a mapping, got {type(checkpoints)!r}")
    player_action_mode = tuple(player_action_mode)  # type: ignore[assignment]
    spawner_action_mode = tuple(spawner_action_mode)  # type: ignore[assignment]
    env_seeds = tuple(int(s) for s in env_seeds)
    spawner_seeds = tuple(int(s) for s in spawner_seeds)
    episode_cap_frames = int(episode_cap_frames)
    comparison_pairs = tuple(tuple(pair) for pair in comparison_pairs)  # type: ignore[assignment]
    metrics = tuple(metrics)

    _validate_seed_package_inputs(
        checkpoints=checkpoints,
        player_action_mode=player_action_mode,  # type: ignore[arg-type]
        spawner_action_mode=spawner_action_mode,  # type: ignore[arg-type]
        env_seeds=env_seeds,
        spawner_seeds=spawner_seeds,
        episode_cap_frames=episode_cap_frames,
        comparison_pairs=comparison_pairs,
        metrics=metrics,
        ci_method=ci_method,
    )
    _validate_contract_identity(
        contract_id=contract_id,
        preregistration_path=preregistration_path,
        checkpoints=checkpoints,
        player_action_mode=player_action_mode,  # type: ignore[arg-type]
        spawner_action_mode=spawner_action_mode,  # type: ignore[arg-type]
        env_seeds=env_seeds,
        spawner_seeds=spawner_seeds,
        episode_cap_frames=episode_cap_frames,
        comparison_pairs=comparison_pairs,  # type: ignore[arg-type]
        metrics=metrics,
    )

    return FrozenSpawnerSeedPackage(
        checkpoints=dict(checkpoints),
        player_action_mode=player_action_mode,  # type: ignore[arg-type]
        spawner_action_mode=spawner_action_mode,  # type: ignore[arg-type]
        env_seeds=env_seeds,
        spawner_seeds=spawner_seeds,
        episode_cap_frames=episode_cap_frames,
        comparison_pairs=comparison_pairs,  # type: ignore[arg-type]
        metrics=metrics,
        ci_method=ci_method,
        schema_version=schema_version,
        contract_id=contract_id,
        preregistration_path=preregistration_path,
        promotion=False,
        training=False,
        self_play=False,
    )
