"""Tests for the eval-only frozen learned-Spawner seed package / manifest
(:mod:`qrokkun_ai.v5.frozen_spawner_seed_package`).

This is diagnostic/eval-only tooling: no test here trains anything, launches
PPO, or touches a real checkpoint file. All checkpoint identity in these
tests is synthetic (placeholder SHA-256-shaped strings); tiny/no neural nets
are used for the runner-primitive tests.
"""

from __future__ import annotations

import json

import pytest

from qrokkun_ai.v5.frozen_spawner_seed_package import (
    CheckpointRef,
    CONTRACT_ID,
    DEFAULT_METRICS,
    DIAGNOSTIC_CONTRACT_ID,
    FrozenSpawnerManifestError,
    FrozenSpawnerSeedPackage,
    PREREGISTERED_COMPARISON_PAIRS,
    PREREGISTRATION_PATH,
    ROLE_ARCHITECTURE,
    ROLE_LEARNED_SPAWNER,
    ROLE_V1_TEACHER,
    ROLE_V5_AUX_U200,
    ROLE_V5_BC_SEED6,
    build_frozen_spawner_seed_package,
)

# 64 lowercase hex chars, i.e. shaped like a real sha256 hexdigest.
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64

CAP_FRAMES = 4200
ENV_SEEDS = tuple(range(3000, 3030))


def _make_checkpoints(**overrides):
    checkpoints = {
        ROLE_V1_TEACHER: CheckpointRef(
            role=ROLE_V1_TEACHER, path="/exec-host/v1_teacher.pt", sha256=_SHA_A, architecture="player_v1",
            state_dict_sha256=_SHA_A,
        ),
        ROLE_V5_BC_SEED6: CheckpointRef(
            role=ROLE_V5_BC_SEED6, path="/exec-host/v5_bc_seed6.pt", sha256=_SHA_B, architecture="player_ranked_topk",
            state_dict_sha256=_SHA_B,
        ),
        ROLE_V5_AUX_U200: CheckpointRef(
            role=ROLE_V5_AUX_U200, path="/exec-host/v5_aux_u200.pt", sha256=_SHA_C, architecture="player_ranked_topk",
            state_dict_sha256=_SHA_C,
        ),
        ROLE_LEARNED_SPAWNER: CheckpointRef(
            role=ROLE_LEARNED_SPAWNER, path="/exec-host/learned_spawner.pt", sha256=_SHA_D, architecture="spawner_v4",
            state_dict_sha256=_SHA_D,
        ),
    }
    checkpoints.update(overrides)
    return checkpoints


def _build(**kwargs):
    defaults = dict(
        checkpoints=_make_checkpoints(),
        player_action_mode=(False, True),  # det_stoch
        spawner_action_mode=(False, True),  # det_stoch
        env_seeds=ENV_SEEDS,
        spawner_seeds=ENV_SEEDS,
        episode_cap_frames=CAP_FRAMES,
        comparison_pairs=((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6), (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200)),
        contract_id=DIAGNOSTIC_CONTRACT_ID,
        preregistration_path=None,
    )
    defaults.update(kwargs)
    return build_frozen_spawner_seed_package(**defaults)


def test_build_valid_package_records_all_inputs():
    pkg = _build()
    assert pkg.checkpoints[ROLE_V1_TEACHER].sha256 == _SHA_A
    assert pkg.player_action_mode == (False, True)
    assert pkg.spawner_action_mode == (False, True)
    assert pkg.env_seeds == ENV_SEEDS
    assert pkg.spawner_seeds == ENV_SEEDS
    assert pkg.episode_cap_frames == CAP_FRAMES
    assert pkg.comparison_pairs == ((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6), (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200))
    # Structural: no way to have requested promotion/training/self-play.
    assert pkg.promotion is False
    assert pkg.training is False
    assert pkg.self_play is False


def test_package_is_immutable():
    pkg = _build()
    with pytest.raises(Exception):
        pkg.promotion = True  # type: ignore[misc]
    with pytest.raises(Exception):
        pkg.episode_cap_frames = 1  # type: ignore[misc]


def test_missing_checkpoint_role_rejected():
    checkpoints = _make_checkpoints()
    del checkpoints[ROLE_LEARNED_SPAWNER]
    with pytest.raises(FrozenSpawnerManifestError, match="learned_spawner"):
        _build(checkpoints=checkpoints)


def test_malformed_sha256_rejected():
    checkpoints = _make_checkpoints()
    with pytest.raises(FrozenSpawnerManifestError, match="sha256"):
        CheckpointRef(
            role=ROLE_V1_TEACHER, path="/exec-host/v1_teacher.pt", sha256="not-a-sha", architecture="player_v1"
        )
    # The builder also rejects a raw reference whose constructor validation
    # was bypassed by a deserializer or object.__setattr__.
    object.__setattr__(checkpoints[ROLE_V1_TEACHER], "sha256", "not-a-sha")
    with pytest.raises(FrozenSpawnerManifestError, match="sha256"):
        _build(checkpoints=checkpoints)


def test_invalid_action_mode_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="sample_policy=True, rng_jitter=False"):
        _build(player_action_mode=(True, False))
    with pytest.raises(FrozenSpawnerManifestError, match="sample_policy=True, rng_jitter=False"):
        _build(spawner_action_mode=(True, False))


def test_stochastic_spawner_without_spawner_seeds_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="spawner_seeds"):
        _build(spawner_action_mode=(True, True), spawner_seeds=())


def test_spawner_seeds_length_mismatch_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="spawner_seeds"):
        _build(spawner_action_mode=(True, True), spawner_seeds=tuple(range(3000, 3010)))


def test_empty_env_seeds_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="env_seeds"):
        _build(env_seeds=())


def test_non_positive_episode_cap_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="episode_cap_frames"):
        _build(episode_cap_frames=0)


def test_comparison_pair_unknown_role_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="unknown checkpoint role"):
        _build(comparison_pairs=((ROLE_V1_TEACHER, "not_a_role"),))


def test_comparison_pair_self_pair_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="same role"):
        _build(comparison_pairs=((ROLE_V1_TEACHER, ROLE_V1_TEACHER),))


def test_no_comparison_pairs_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        _build(comparison_pairs=())


def test_promotion_training_self_play_cannot_be_requested_true():
    with pytest.raises(TypeError):
        _build(promotion=True)
    with pytest.raises(TypeError):
        _build(training=True)
    with pytest.raises(TypeError):
        _build(self_play=True)


def test_duplicate_env_seeds_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="duplicate"):
        _build(env_seeds=(3000, 3000, 3001))


def test_to_json_dict_round_trips_through_json():
    pkg = _build()
    manifest = pkg.to_json_dict()
    blob = json.dumps(manifest)  # must not raise: pure JSON-serializable types
    restored = json.loads(blob)
    assert restored["player_action_mode"] == [False, True]
    assert restored["spawner_action_mode"] == [False, True]
    assert restored["env_seeds"] == list(ENV_SEEDS)
    assert restored["episode_cap_frames"] == CAP_FRAMES
    assert restored["promotion"] is False
    assert restored["training"] is False
    assert restored["self_play"] is False
    assert restored["checkpoints"][ROLE_V1_TEACHER]["sha256"] == _SHA_A
    assert restored["comparison_pairs"] == [
        [ROLE_V1_TEACHER, ROLE_V5_BC_SEED6],
        [ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200],
    ]
    assert restored["metrics"]


def test_metrics_default_to_restricted_allowlist():
    pkg = _build()
    for name in pkg.metrics:
        assert name in pkg.ALLOWED_METRICS  # type: ignore[attr-defined]


def test_unknown_metric_name_rejected():
    with pytest.raises(FrozenSpawnerManifestError, match="metrics"):
        _build(metrics=("mean", "some_unapproved_gate_metric"))


def _direct_kwargs(**overrides):
    """kwargs for constructing FrozenSpawnerSeedPackage directly (bypassing
    the builder), to check the dataclass's own structural guarantees."""
    defaults = dict(
        checkpoints=_make_checkpoints(),
        player_action_mode=(False, True),
        spawner_action_mode=(False, True),
        env_seeds=ENV_SEEDS,
        spawner_seeds=ENV_SEEDS,
        episode_cap_frames=CAP_FRAMES,
        comparison_pairs=((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6), (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200)),
        contract_id=DIAGNOSTIC_CONTRACT_ID,
        preregistration_path=None,
    )
    defaults.update(overrides)
    return defaults


def test_direct_construction_rejects_promotion_true():
    with pytest.raises(TypeError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(promotion=True))


def test_direct_construction_rejects_training_true():
    with pytest.raises(TypeError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(training=True))


def test_direct_construction_rejects_self_play_true():
    with pytest.raises(TypeError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(self_play=True))


def test_direct_construction_checkpoints_mapping_is_immutable():
    pkg = FrozenSpawnerSeedPackage(**_direct_kwargs())
    with pytest.raises(Exception):
        pkg.checkpoints[ROLE_V1_TEACHER] = None  # type: ignore[index]


def test_builder_checkpoints_mapping_is_also_immutable():
    pkg = _build()
    with pytest.raises(Exception):
        pkg.checkpoints[ROLE_V1_TEACHER] = None  # type: ignore[index]


# --- Review finding 1: checkpoint mapping must be detached/copied, never a
# live view onto a caller-owned backing dict, including MappingProxyType
# input. ---


def test_direct_construction_detaches_dict_backing_mutation_does_not_leak():
    backing = _make_checkpoints()
    pkg = FrozenSpawnerSeedPackage(**_direct_kwargs(checkpoints=backing))
    backing[ROLE_V1_TEACHER] = CheckpointRef(
        role=ROLE_V1_TEACHER, path="/tampered.pt", sha256=_SHA_D, architecture="tampered"
    )
    assert pkg.checkpoints[ROLE_V1_TEACHER].path == "/exec-host/v1_teacher.pt"
    assert pkg.checkpoints[ROLE_V1_TEACHER].sha256 == _SHA_A


def test_direct_construction_detaches_mappingproxytype_input_from_backing_dict():
    from types import MappingProxyType

    backing = _make_checkpoints()
    proxy = MappingProxyType(backing)
    pkg = FrozenSpawnerSeedPackage(**_direct_kwargs(checkpoints=proxy))
    # Mutate the dict backing the proxy (not the proxy itself, which is
    # read-only) after construction.
    backing[ROLE_V1_TEACHER] = CheckpointRef(
        role=ROLE_V1_TEACHER, path="/tampered.pt", sha256=_SHA_D, architecture="tampered"
    )
    assert pkg.checkpoints[ROLE_V1_TEACHER].path == "/exec-host/v1_teacher.pt"
    assert pkg.checkpoints[ROLE_V1_TEACHER].sha256 == _SHA_A


# --- Review finding 2: direct public construction must defensively
# canonicalize mutable sequence inputs (lists) to immutable tuples. ---


def test_direct_construction_canonicalizes_mutable_sequences_to_tuples():
    kwargs = _direct_kwargs(
        player_action_mode=[False, True],
        spawner_action_mode=[False, True],
        env_seeds=list(ENV_SEEDS),
        spawner_seeds=list(ENV_SEEDS),
        metrics=list(DEFAULT_METRICS),
        comparison_pairs=[[ROLE_V1_TEACHER, ROLE_V5_BC_SEED6], [ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200]],
    )
    pkg = FrozenSpawnerSeedPackage(**kwargs)
    assert isinstance(pkg.player_action_mode, tuple)
    assert isinstance(pkg.spawner_action_mode, tuple)
    assert isinstance(pkg.env_seeds, tuple)
    assert isinstance(pkg.spawner_seeds, tuple)
    assert isinstance(pkg.metrics, tuple)
    assert isinstance(pkg.comparison_pairs, tuple)
    assert all(isinstance(pair, tuple) for pair in pkg.comparison_pairs)


def test_direct_construction_input_list_mutation_does_not_alter_package():
    env_seeds_list = list(ENV_SEEDS)
    spawner_seeds_list = list(ENV_SEEDS)
    metrics_list = list(DEFAULT_METRICS)
    comparison_pairs_list = [[ROLE_V1_TEACHER, ROLE_V5_BC_SEED6], [ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200]]
    player_mode_list = [False, True]
    spawner_mode_list = [False, True]
    pkg = FrozenSpawnerSeedPackage(
        **_direct_kwargs(
            player_action_mode=player_mode_list,
            spawner_action_mode=spawner_mode_list,
            env_seeds=env_seeds_list,
            spawner_seeds=spawner_seeds_list,
            metrics=metrics_list,
            comparison_pairs=comparison_pairs_list,
        )
    )
    env_seeds_list.append(999999)
    spawner_seeds_list.append(999999)
    metrics_list.append("some_unapproved_gate_metric")
    comparison_pairs_list.append([ROLE_V1_TEACHER, ROLE_V5_AUX_U200])
    comparison_pairs_list[0][0] = "tampered"
    player_mode_list[0] = True
    spawner_mode_list[0] = True

    assert pkg.env_seeds == ENV_SEEDS
    assert pkg.spawner_seeds == ENV_SEEDS
    assert pkg.metrics == DEFAULT_METRICS
    assert pkg.comparison_pairs == ((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6), (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200))
    assert pkg.player_action_mode == (False, True)
    assert pkg.spawner_action_mode == (False, True)

    # JSON output stays stable / unaffected by post-construction mutation.
    manifest = pkg.to_json_dict()
    assert manifest["env_seeds"] == list(ENV_SEEDS)
    assert manifest["spawner_seeds"] == list(ENV_SEEDS)
    assert manifest["metrics"] == list(DEFAULT_METRICS)
    assert manifest["comparison_pairs"] == [
        [ROLE_V1_TEACHER, ROLE_V5_BC_SEED6],
        [ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200],
    ]


# --- Review finding 3: ci_method is locked to the preregistered
# "paired_t95"; CRN semantics (single episode_seed key reused by
# eval_modes.run_eval_episode) require stochastic Spawner sampling seeds to
# equal the env-seed sequence. ---


def test_ci_method_locked_rejects_other_value_via_builder():
    with pytest.raises(FrozenSpawnerManifestError, match="ci_method"):
        _build(ci_method="bootstrap95")


def test_direct_construction_rejects_non_paired_t95_ci_method():
    with pytest.raises(FrozenSpawnerManifestError, match="ci_method"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(ci_method="bootstrap95"))


def test_stochastic_spawner_seeds_must_equal_env_seeds_via_builder():
    mismatched = tuple(s + 1 for s in ENV_SEEDS)
    with pytest.raises(FrozenSpawnerManifestError, match="spawner_seeds"):
        _build(spawner_action_mode=(True, True), spawner_seeds=mismatched)


def test_direct_construction_stochastic_spawner_seeds_must_equal_env_seeds():
    mismatched = tuple(s + 1 for s in ENV_SEEDS)
    with pytest.raises(FrozenSpawnerManifestError, match="spawner_seeds"):
        FrozenSpawnerSeedPackage(
            **_direct_kwargs(
                spawner_action_mode=(True, True),
                spawner_seeds=mismatched,
            )
        )


# --- Review finding 4: direct construction must not trivially bypass
# checkpoint-roles/metrics validation. ---


def test_direct_construction_rejects_missing_checkpoint_role():
    checkpoints = _make_checkpoints()
    del checkpoints[ROLE_LEARNED_SPAWNER]
    with pytest.raises(FrozenSpawnerManifestError, match="learned_spawner"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(checkpoints=checkpoints))


def test_direct_construction_rejects_unknown_checkpoint_role():
    checkpoints = _make_checkpoints()
    checkpoints["not_a_role"] = checkpoints[ROLE_V1_TEACHER]
    with pytest.raises(FrozenSpawnerManifestError, match="unknown checkpoint role"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(checkpoints=checkpoints))


def test_direct_construction_rejects_unknown_metric():
    with pytest.raises(FrozenSpawnerManifestError, match="metrics"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(metrics=("mean", "some_unapproved_gate_metric")))


# --- Final review: direct construction must reject exactly what the builder
# rejects. Each gap below is currently only enforced by
# build_frozen_spawner_seed_package, and direct dataclass construction
# either silently accepts the bad input or (worse) raises an incidental
# IndexError instead of FrozenSpawnerManifestError. ---


def test_direct_construction_rejects_invalid_named_action_mode():
    with pytest.raises(FrozenSpawnerManifestError, match="sample_policy=True, rng_jitter=False"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(player_action_mode=(True, False)))
    with pytest.raises(FrozenSpawnerManifestError, match="sample_policy=True, rng_jitter=False"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(spawner_action_mode=(True, False)))


def test_direct_construction_rejects_malformed_action_mode_without_indexerror():
    # Too-short tuple: must not raise a bare IndexError from indexing into
    # spawner_action_mode[0] before shape validation runs.
    with pytest.raises(FrozenSpawnerManifestError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(spawner_action_mode=(True,)))
    with pytest.raises(FrozenSpawnerManifestError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(player_action_mode=(True,)))


def test_direct_construction_rejects_empty_action_mode_without_indexerror():
    with pytest.raises(FrozenSpawnerManifestError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(spawner_action_mode=()))
    with pytest.raises(FrozenSpawnerManifestError):
        FrozenSpawnerSeedPackage(**_direct_kwargs(player_action_mode=()))


def test_direct_construction_rejects_nonpositive_episode_cap():
    with pytest.raises(FrozenSpawnerManifestError, match="episode_cap_frames"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(episode_cap_frames=0))
    with pytest.raises(FrozenSpawnerManifestError, match="episode_cap_frames"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(episode_cap_frames=-1))


def test_direct_construction_rejects_empty_comparison_pairs():
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(comparison_pairs=()))


def test_direct_construction_rejects_self_pair():
    with pytest.raises(FrozenSpawnerManifestError, match="same role"):
        FrozenSpawnerSeedPackage(
            **_direct_kwargs(comparison_pairs=((ROLE_V1_TEACHER, ROLE_V1_TEACHER),))
        )


def test_direct_construction_rejects_unknown_role_pair():
    with pytest.raises(FrozenSpawnerManifestError, match="unknown checkpoint role"):
        FrozenSpawnerSeedPackage(
            **_direct_kwargs(comparison_pairs=((ROLE_V1_TEACHER, "not_a_role"),))
        )


def test_direct_construction_rejects_wrong_pair_arity():
    with pytest.raises(FrozenSpawnerManifestError, match="exactly 2 roles"):
        FrozenSpawnerSeedPackage(
            **_direct_kwargs(
                comparison_pairs=((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200),)
            )
        )
    with pytest.raises(FrozenSpawnerManifestError, match="exactly 2 roles"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(comparison_pairs=((ROLE_V1_TEACHER,),)))


def test_direct_construction_rejects_empty_env_seeds():
    with pytest.raises(FrozenSpawnerManifestError, match="env_seeds"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(env_seeds=(), spawner_seeds=()))


def test_direct_construction_rejects_duplicate_env_seeds():
    with pytest.raises(FrozenSpawnerManifestError, match="duplicate"):
        FrozenSpawnerSeedPackage(
            **_direct_kwargs(env_seeds=(3000, 3000, 3001), spawner_seeds=(3000, 3000, 3001))
        )


# --- Final review: remaining constructor/builder divergences. Direct
# dataclass construction must reject non-Mapping checkpoints (e.g. a plain
# list of (role, ref) pairs) exactly like the builder does, instead of
# silently accepting it via dict(list_of_pairs), and must canonicalize
# episode_cap_frames to int exactly like the builder does. ---


def test_builder_rejects_non_mapping_checkpoints():
    checkpoints = _make_checkpoints()
    with pytest.raises(FrozenSpawnerManifestError, match="mapping"):
        _build(checkpoints=list(checkpoints.items()))


def test_direct_construction_rejects_non_mapping_checkpoints():
    checkpoints = _make_checkpoints()
    with pytest.raises(FrozenSpawnerManifestError, match="mapping"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(checkpoints=list(checkpoints.items())))


def test_builder_canonicalizes_episode_cap_frames_to_int():
    pkg = _build(episode_cap_frames=4200.9)
    assert pkg.episode_cap_frames == 4200
    assert isinstance(pkg.episode_cap_frames, int)

    pkg_bool = _build(episode_cap_frames=True)
    assert pkg_bool.episode_cap_frames == 1
    assert isinstance(pkg_bool.episode_cap_frames, int)
    assert not isinstance(pkg_bool.episode_cap_frames, bool)


def test_direct_construction_canonicalizes_episode_cap_frames_to_int():
    pkg = FrozenSpawnerSeedPackage(**_direct_kwargs(episode_cap_frames=4200.9))
    assert pkg.episode_cap_frames == 4200
    assert isinstance(pkg.episode_cap_frames, int)

    pkg_bool = FrozenSpawnerSeedPackage(**_direct_kwargs(episode_cap_frames=True))
    assert pkg_bool.episode_cap_frames == 1
    assert isinstance(pkg_bool.episode_cap_frames, int)
    assert not isinstance(pkg_bool.episode_cap_frames, bool)


# --- Strict-review finding: exact role -> architecture mapping must be
# enforced by the seed package itself (not merely by callers who happen to
# pass the right architecture string), and must be rejected *before* any
# checkpoint load is ever attempted. ---


def test_role_architecture_mapping_is_fixed_and_exhaustive():
    assert ROLE_ARCHITECTURE[ROLE_V1_TEACHER] == "player_v1"
    assert ROLE_ARCHITECTURE[ROLE_V5_BC_SEED6] == "player_ranked_topk"
    assert ROLE_ARCHITECTURE[ROLE_V5_AUX_U200] == "player_ranked_topk"
    assert ROLE_ARCHITECTURE[ROLE_LEARNED_SPAWNER] == "spawner_v4"


def test_builder_rejects_wrong_architecture_for_v1_teacher():
    checkpoints = _make_checkpoints()
    checkpoints[ROLE_V1_TEACHER] = CheckpointRef(
        role=ROLE_V1_TEACHER, path="/exec-host/v1_teacher.pt", sha256=_SHA_A, architecture="player_ranked_topk"
    )
    with pytest.raises(FrozenSpawnerManifestError, match="architecture"):
        _build(checkpoints=checkpoints)


def test_builder_rejects_wrong_architecture_for_v5_bc_seed6():
    checkpoints = _make_checkpoints()
    checkpoints[ROLE_V5_BC_SEED6] = CheckpointRef(
        role=ROLE_V5_BC_SEED6, path="/exec-host/v5_bc_seed6.pt", sha256=_SHA_B, architecture="player_v1"
    )
    with pytest.raises(FrozenSpawnerManifestError, match="architecture"):
        _build(checkpoints=checkpoints)


def test_builder_rejects_wrong_architecture_for_learned_spawner():
    checkpoints = _make_checkpoints()
    checkpoints[ROLE_LEARNED_SPAWNER] = CheckpointRef(
        role=ROLE_LEARNED_SPAWNER,
        path="/exec-host/learned_spawner.pt",
        sha256=_SHA_D,
        architecture="spawner_v2",
    )
    with pytest.raises(FrozenSpawnerManifestError, match="architecture"):
        _build(checkpoints=checkpoints)


def test_direct_construction_rejects_wrong_role_architecture():
    checkpoints = _make_checkpoints()
    checkpoints[ROLE_V5_AUX_U200] = CheckpointRef(
        role=ROLE_V5_AUX_U200, path="/exec-host/v5_aux_u200.pt", sha256=_SHA_C, architecture="player_v1"
    )
    with pytest.raises(FrozenSpawnerManifestError, match="architecture"):
        FrozenSpawnerSeedPackage(**_direct_kwargs(checkpoints=checkpoints))


# --- Strict-review finding: comparison_pairs must be exactly the two
# preregistered pairs, in their preregistered order -- not merely valid
# role pairs. Reordering, dropping, or adding a pair must be rejected. ---


def test_preregistered_comparison_pairs_constant_is_locked():
    assert PREREGISTERED_COMPARISON_PAIRS == (
        (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6),
        (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200),
    )


def test_builder_rejects_reordered_comparison_pairs():
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        _build(comparison_pairs=((ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200), (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6)))


def test_builder_rejects_extra_comparison_pair():
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        _build(
            comparison_pairs=(
                (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6),
                (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200),
                (ROLE_V1_TEACHER, ROLE_V5_AUX_U200),
            )
        )


def test_builder_rejects_reversed_pair_direction():
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        _build(comparison_pairs=((ROLE_V5_BC_SEED6, ROLE_V1_TEACHER), (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200)))


def test_builder_rejects_substituted_valid_pair():
    # Both pairs individually reference known roles and are not self-pairs,
    # but this is not the preregistered pair set (v1_teacher vs v5_aux_u200
    # directly is explicitly out of scope as a primary/CI-bearing pair).
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        _build(comparison_pairs=((ROLE_V1_TEACHER, ROLE_V5_AUX_U200),))


def test_direct_construction_rejects_non_preregistered_comparison_pairs():
    with pytest.raises(FrozenSpawnerManifestError, match="comparison_pairs"):
        FrozenSpawnerSeedPackage(
            **_direct_kwargs(comparison_pairs=((ROLE_V1_TEACHER, ROLE_V5_AUX_U200),))
        )


def test_only_the_locked_pairs_pass_validation():
    pkg = _build(comparison_pairs=PREREGISTERED_COMPARISON_PAIRS)
    assert pkg.comparison_pairs == PREREGISTERED_COMPARISON_PAIRS


def _formal_package() -> FrozenSpawnerSeedPackage:
    return _build(
        player_action_mode=(False, False),
        spawner_action_mode=(True, True),
        contract_id=CONTRACT_ID,
        preregistration_path=PREREGISTRATION_PATH,
    )


@pytest.mark.parametrize(
    ("package_kind", "mutation", "match"),
    [
        ("formal", lambda pkg: object.__setattr__(pkg, "env_seeds", tuple(float(x) for x in ENV_SEEDS)), "env_seeds"),
        ("diagnostic", lambda pkg: object.__setattr__(pkg, "episode_cap_frames", float(CAP_FRAMES)), "episode_cap_frames"),
        ("formal", lambda pkg: object.__setattr__(pkg, "player_action_mode", [False, False]), "player_action_mode"),
        ("diagnostic", lambda pkg: object.__setattr__(pkg, "comparison_pairs", [list(pair) for pair in pkg.comparison_pairs]), "comparison_pairs"),
        ("formal", lambda pkg: object.__setattr__(pkg, "metrics", list(pkg.metrics)), "metrics"),
        ("diagnostic", lambda pkg: object.__setattr__(pkg, "ci_method", ["paired_t95"]), "ci_method"),
        ("formal", lambda pkg: object.__setattr__(pkg, "checkpoints", dict(pkg.checkpoints)), "checkpoints"),
        ("diagnostic", lambda pkg: object.__setattr__(pkg.checkpoints[ROLE_V1_TEACHER], "path", object()), "checkpoint"),
        ("formal", lambda pkg: object.__setattr__(pkg, "schema_version", 1.0), "schema_version"),
        ("diagnostic", lambda pkg: object.__setattr__(pkg, "promotion", True), "promotion"),
        # These are semantic, raw-canonical tamper cases: serialization must
        # validate contract identity too, rather than only JSON-safe shapes.
        ("formal", lambda pkg: object.__setattr__(pkg, "preregistration_path", None), "formal contract"),
        ("diagnostic", lambda pkg: object.__setattr__(pkg, "preregistration_path", PREREGISTRATION_PATH), "diagnostic contract"),
    ],
)
def test_to_json_dict_rejects_bypass_mutations_at_every_serialized_boundary(
    package_kind: str, mutation, match: str,
) -> None:
    """Serialization validates actual raw fields before producing a manifest."""
    package = _formal_package() if package_kind == "formal" else _build()
    mutation(package)

    with pytest.raises(FrozenSpawnerManifestError, match=match):
        package.to_json_dict()


def test_to_json_dict_valid_formal_and_diagnostic_controls() -> None:
    assert _formal_package().to_json_dict()["contract_id"] == CONTRACT_ID
    assert _build().to_json_dict()["contract_id"] == DIAGNOSTIC_CONTRACT_ID
