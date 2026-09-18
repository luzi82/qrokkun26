"""RED regressions for the frozen-Spawner formal-evaluation contract.

These tests deliberately use parser-only checks and tiny synthetic environments;
they never load checkpoints or run training/evaluation jobs.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import qrokkun_ai.v5.frozen_spawner_eval_cli as cli
from qrokkun_ai.v5.agents.player_checkpoints import state_dict_sha256
from qrokkun_env.env import Qrokkun26Env
from qrokkun_ai.v4.eval_modes import MODE_DET_DET, MODE_STOCH_STOCH
from qrokkun_ai.v5.frozen_spawner_seed_package import (
    CheckpointRef,
    FrozenSpawnerManifestError,
    FrozenSpawnerSeedPackage,
    PREREGISTERED_COMPARISON_PAIRS,
    ROLE_ARCHITECTURE,
    ROLE_LEARNED_SPAWNER,
    ROLE_V1_TEACHER,
    ROLE_V5_AUX_U200,
    ROLE_V5_BC_SEED6,
    build_frozen_spawner_seed_package,
)
from qrokkun_ai.v5.frozen_spawner_eval_runner import restricted_survival_stats
from qrokkun_ai.v4.train.both_v4 import apply_player_action as train_apply_player_action


class _StringSubclass(str):
    """Deliberately non-exact JSON string representation for raw-state tests."""


def _formal_cli_args(*extra: str) -> list[str]:
    """Required identity arguments only; their paths are never dereferenced."""
    return [
        "--v1-teacher-path", "/synthetic/v1.pt",
        "--v1-teacher-sha256", "1" * 64,
        "--v1-teacher-state-dict-sha256", "a" * 64,
        "--v5-bc-seed6-path", "/synthetic/bc.pt",
        "--v5-bc-seed6-sha256", "2" * 64,
        "--v5-bc-seed6-state-dict-sha256", "b" * 64,
        "--v5-aux-u200-path", "/synthetic/aux.pt",
        "--v5-aux-u200-sha256", "3" * 64,
        "--v5-aux-u200-state-dict-sha256", "c" * 64,
        "--learned-spawner-path", "/synthetic/spawner.pt",
        "--learned-spawner-sha256", "4" * 64,
        "--learned-spawner-state-dict-sha256", "d" * 64,
        "--out-manifest", "/synthetic/manifest.json",
        *extra,
    ]


def _prepare_cli_args(manifest: Path, *extra: str) -> list[str]:
    """Synthetic identities for preparation only; files are never opened."""
    return [
        "--v1-teacher-path", "/synthetic/v1.pt", "--v1-teacher-sha256", "1" * 64,
        "--v1-teacher-state-dict-sha256", "a" * 64,
        "--v5-bc-seed6-path", "/synthetic/bc.pt", "--v5-bc-seed6-sha256", "2" * 64,
        "--v5-bc-seed6-state-dict-sha256", "b" * 64,
        "--v5-aux-u200-path", "/synthetic/aux.pt", "--v5-aux-u200-sha256", "3" * 64,
        "--v5-aux-u200-state-dict-sha256", "c" * 64,
        "--learned-spawner-path", "/synthetic/spawner.pt", "--learned-spawner-sha256", "4" * 64,
        "--learned-spawner-state-dict-sha256", "d" * 64,
        "--out-manifest", str(manifest), *extra,
    ]


def test_formal_parser_accepts_only_a_locked_manifest_not_artifact_selectors() -> None:
    parser = cli.build_parser()
    parser.parse_args(["--manifest", "/locked/manifest.json", "--out-report", "/out/report.json"])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--manifest", "/locked/manifest.json", "--out-report", "/out/report.json",
            "--v1-teacher-path", "/replacement.pt",
        ])


def test_prepare_writes_manifest_without_loading_or_episodes(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "locked.json"
    monkeypatch.setattr(cli, "load_checkpoint_for_role", lambda *_a, **_k: pytest.fail("prepare loaded a model"))
    monkeypatch.setattr(cli, "run_frozen_spawner_comparison", lambda *_a, **_k: pytest.fail("prepare ran episodes"))
    package = cli.prepare_main(_prepare_cli_args(manifest))
    assert manifest.is_file()
    assert json.loads(manifest.read_text()) == package.to_json_dict()


def test_formal_execution_reads_locked_manifest_and_rejects_mutated_artifact_before_episode(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "locked.json"
    cli.prepare_main(_prepare_cli_args(manifest))
    raw = json.loads(manifest.read_text())
    raw["checkpoints"][ROLE_V5_BC_SEED6]["path"] = "/synthetic/replaced-bc.pt"
    manifest.write_text(json.dumps(raw))
    def reject_only_mutated_ref(ref, *_args, **_kwargs):
        if "replaced-bc" in ref.path:
            raise cli.ChecksumMismatchError(ref.path)
        return _synthetic_identity_net(ref.role), {
            "role": ref.role, "architecture": ref.architecture,
            "file_sha256": ref.sha256, "state_dict_sha256": ref.state_dict_sha256,
        }
    monkeypatch.setattr(cli, "load_checkpoint_for_role", reject_only_mutated_ref)
    monkeypatch.setattr(cli, "run_frozen_spawner_comparison", lambda *_a, **_k: pytest.fail("episode ran"))
    with pytest.raises(cli.ChecksumMismatchError, match="replaced-bc"):
        cli.main(["--manifest", str(manifest), "--out-report", str(tmp_path / "report.json")])


def test_formal_command_cannot_generate_and_execute_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(_prepare_cli_args(tmp_path / "same-command.json") + ["--out-report", str(tmp_path / "report.json")])


def _synthetic_identity_net(role: str) -> torch.nn.Module:
    net = torch.nn.Linear(1, 1)
    with torch.no_grad():
        value = float((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200, ROLE_LEARNED_SPAWNER).index(role) + 1)
        net.weight.fill_(value)
        net.bias.fill_(-value)
    return net


def _verified_loaded_checkpoints(package: FrozenSpawnerSeedPackage) -> dict[str, tuple[object, dict]]:
    """Synthetic loaded values with the provenance required by the runner."""
    return {
        role: (
            _synthetic_identity_net(role),
            {
                "role": role,
                "architecture": ref.architecture,
                "file_sha256": ref.sha256,
                "state_dict_sha256": ref.state_dict_sha256,
            },
        )
        for role, ref in package.checkpoints.items()
    }


def test_formal_cli_hard_locks_preregistered_modes_and_rejects_overrides() -> None:
    """The formal identity has one mode tuple, not caller-selectable modes."""
    parser = cli.build_prepare_parser()
    args = parser.parse_args(_formal_cli_args())
    assert (args.player_action_mode, args.spawner_action_mode) == ("det_det", "stoch_stoch")

    for override in (
        ("--player-action-mode", "det_stoch"),
        ("--player-action-mode", "stoch_stoch"),
        ("--spawner-action-mode", "det_det"),
        ("--spawner-action-mode", "det_stoch"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(_formal_cli_args(*override))


def test_formal_cli_rejects_noncanonical_seed_windows_and_frame_caps() -> None:
    """Only seeds 3000..3029 and cap 4200 belong to this preregistration."""
    parser = cli.build_prepare_parser()
    for override in (
        ("--env-seed-start", "2999"),
        ("--env-seed-start", "3001"),
        ("--env-seed-count", "29"),
        ("--env-seed-count", "31"),
        ("--episode-cap-frames", "4199"),
        ("--episode-cap-frames", "4201"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(_formal_cli_args(*override))


def test_formal_manifest_exposes_stable_contract_and_preregistration_identity() -> None:
    """Downstream reports need a durable identifier for the frozen protocol."""
    args = cli.build_prepare_parser().parse_args(
        _formal_cli_args(
            "--player-action-mode", "det_det",
            "--spawner-action-mode", "stoch_stoch",
        )
    )
    manifest_a = cli.build_seed_package_from_args(args).to_json_dict()
    manifest_b = cli.build_seed_package_from_args(args).to_json_dict()
    assert manifest_a["contract_id"] == manifest_b["contract_id"]
    assert manifest_a["preregistration_path"] == (
        "docs/journal/1789263159_player_v5_frozen_spawner_eval_preregistration.txt"
    )


def test_eval_apply_player_action_matches_both_v4_float32_physics_long_sequence() -> None:
    """The eval fork must preserve every float32 player state transition."""
    eval_env = Qrokkun26Env(seed=3000)
    train_env = Qrokkun26Env(seed=3000)
    eval_env.reset(seed=3000)
    train_env.reset(seed=3000)

    actions = (2, 4, 6, 8, 1, 3, 5, 7, 0) * 64
    for frame, action in enumerate(actions, start=1):
        cli.apply_player_action(eval_env, action)
        train_apply_player_action(train_env, action)
        assert (eval_env.px, eval_env.py, eval_env.pvx, eval_env.pvy) == (
            train_env.px,
            train_env.py,
            train_env.pvx,
            train_env.pvy,
        ), f"float32 physics diverged at frame {frame}"


class _SyntheticEpisodeEnv:
    """Minimal loop-compatible env: it hits exactly on the cap frame."""

    def __init__(self, seed: int) -> None:
        self.dt = 1.0
        self.elapsed = 0.0
        self.spawn_acc = -100.0
        self.dead = False
        self.frames = 0

    def reset(self, seed: int) -> None:
        return None

    def _integrate_bullets(self) -> None:
        self.frames += 1

    def _check_hit(self) -> bool:
        return self.frames == 3


def test_episode_outcome_explicitly_distinguishes_final_frame_hit_from_censoring(monkeypatch) -> None:
    """A hit on frame cap is terminal, despite sharing elapsed time with a timeout."""
    monkeypatch.setattr(cli, "Qrokkun26Env", _SyntheticEpisodeEnv)
    monkeypatch.setitem(cli._PLAYER_ACTORS, "synthetic", lambda *_args: 0)
    monkeypatch.setattr(cli, "apply_player_action", lambda *_args: None)

    outcome = cli.run_frozen_spawner_episode(
        "synthetic",
        object(),
        object(),
        "cpu",
        seed=3000,
        cap_frames=3,
        player_sample=False,
        spawner_sample=False,
        rng_jitter=False,
    )

    assert isinstance(outcome, Mapping), (
        "run_frozen_spawner_episode must return a structured outcome mapping "
        "with elapsed, frames, hit, censored, and termination_reason; a float "
        "cannot distinguish a final-frame hit from censoring"
    )
    assert set(outcome) >= {"elapsed", "frames", "hit", "censored", "termination_reason"}
    assert outcome["elapsed"] == 3.0
    assert outcome["frames"] == 3
    assert outcome["hit"] is True
    assert outcome["censored"] is False
    assert outcome["termination_reason"] == "hit"


def test_restricted_stats_use_explicit_termination_flags_at_the_frame_cap() -> None:
    """A cap-frame hit is a hit; elapsed alone must not turn it into a censor."""
    outcomes = [
        {
            "elapsed": 3.0,
            "frames": 3,
            "hit": True,
            "censored": False,
            "termination_reason": "hit",
        },
        {
            "elapsed": 3.0,
            "frames": 3,
            "hit": False,
            "censored": True,
            "termination_reason": "cap",
        },
    ]

    stats = restricted_survival_stats(outcomes, cap=3.0)

    assert stats["hit_flags"] == [True, False]
    assert stats["censor_count"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


# --- Provenance identity blocker: formal and diagnostic packages must not
# impersonate one another.  All checkpoint references below are synthetic. ---


def _identity_checkpoints() -> dict[str, CheckpointRef]:
    return {
        role: CheckpointRef(
            role, f"/synthetic/{role}.pt", char * 64, ROLE_ARCHITECTURE[role],
            state_dict_sha256=state_dict_sha256(_synthetic_identity_net(role).state_dict()),
        )
        for role, char in zip(
            (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200, ROLE_LEARNED_SPAWNER),
            "abcd",
            strict=True,
        )
    }


def _formal_package_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "checkpoints": _identity_checkpoints(),
        "player_action_mode": MODE_DET_DET,
        "spawner_action_mode": MODE_STOCH_STOCH,
        "env_seeds": tuple(range(3000, 3030)),
        "spawner_seeds": tuple(range(3000, 3030)),
        "episode_cap_frames": 4200,
        "comparison_pairs": PREREGISTERED_COMPARISON_PAIRS,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "identity_overrides",
    [
        {},
        {
            "player_action_mode": (False, True),
            "spawner_action_mode": (False, True),
            "env_seeds": (7, 8),
            "spawner_seeds": (7, 8),
            "episode_cap_frames": 12,
            "contract_id": "frozen_spawner_diagnostic_v1",
            "preregistration_path": None,
        },
    ],
    ids=("formal", "diagnostic"),
)
def test_schema_version_is_locked_for_construction_builder_and_formal_report_boundary(
    monkeypatch, identity_overrides: dict[str, object]
) -> None:
    """Schema 1 is required before manifest serialization or any formal episode."""
    kwargs = _formal_package_kwargs(**identity_overrides)

    with pytest.raises(FrozenSpawnerManifestError, match="schema_version"):
        FrozenSpawnerSeedPackage(**kwargs, schema_version=999)  # type: ignore[arg-type]
    with pytest.raises(FrozenSpawnerManifestError, match="schema_version"):
        build_frozen_spawner_seed_package(**kwargs, schema_version=999)  # type: ignore[arg-type]

    direct = FrozenSpawnerSeedPackage(**kwargs, schema_version=1)  # type: ignore[arg-type]
    built = build_frozen_spawner_seed_package(**kwargs, schema_version=1)  # type: ignore[arg-type]
    assert direct.to_json_dict()["schema_version"] == 1
    assert built.to_json_dict()["schema_version"] == 1

    # Simulate a malformed/deserialized package that bypassed __post_init__.
    # The report boundary must reject it before checkpoint lookup or an episode.
    object.__setattr__(direct, "schema_version", 999)
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )
    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(direct, {})


@pytest.mark.parametrize(
    ("changed_field", "value"),
    [
        ("player_action_mode", (False, True)),
        ("spawner_action_mode", MODE_DET_DET),
        ("env_seeds", (3000, 3001)),
        ("episode_cap_frames", 12),
    ],
)
def test_formal_identity_rejects_each_noncanonical_field_before_serialization(
    changed_field: str, value: object
) -> None:
    kwargs = _formal_package_kwargs(**{changed_field: value})
    if changed_field == "env_seeds":
        kwargs["spawner_seeds"] = value
    with pytest.raises(FrozenSpawnerManifestError, match="formal"):
        build_frozen_spawner_seed_package(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(FrozenSpawnerManifestError, match="formal"):
        FrozenSpawnerSeedPackage(**kwargs)  # type: ignore[arg-type]


def test_generic_package_cannot_serialize_formal_contract_identity() -> None:
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            player_action_mode=(False, True),
            spawner_action_mode=(False, True),
            env_seeds=(7, 8),
            spawner_seeds=(),
            episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1",
            preregistration_path=None,
        )
    )
    manifest = package.to_json_dict()
    assert manifest["contract_id"] != "player_v5_frozen_spawner_eval_v1"
    assert manifest["preregistration_path"] != (
        "docs/journal/1789263159_player_v5_frozen_spawner_eval_preregistration.txt"
    )


def test_diagnostic_comparison_never_emits_formal_schema(monkeypatch) -> None:
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            env_seeds=(7, 8), spawner_seeds=(7, 8), episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1", preregistration_path=None,
        )
    )
    calls = []

    def fake_outcomes(*_args, **_kwargs):
        calls.append(True)
        return [{"elapsed": 1.0, "frames": 1, "hit": True, "censored": False, "termination_reason": "hit"}] * 2

    monkeypatch.setattr(cli, "evaluate_role_outcomes", fake_outcomes)
    report = cli.run_frozen_spawner_comparison(package, _verified_loaded_checkpoints(package))
    assert calls
    assert report["schema"] == "frozen_spawner_diagnostic_report.v1"
    assert report["schema"] != "frozen_spawner_eval_report.v1"


def test_formal_report_rejects_noncanonical_before_any_episode(monkeypatch) -> None:
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            env_seeds=(7, 8), spawner_seeds=(7, 8), episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1", preregistration_path=None,
        )
    )
    # Simulate a malicious/deserialized mutation after constructor validation;
    # the runner remains the final barrier before its first episode.
    object.__setattr__(package, "contract_id", "player_v5_frozen_spawner_eval_v1")
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )
    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(package, {})


def test_diagnostic_report_rejects_forged_preregistration_path_before_any_episode(monkeypatch) -> None:
    """A diagnostic label cannot serialize a package as preregistered."""
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            env_seeds=(7, 8), spawner_seeds=(7, 8), episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1", preregistration_path=None,
        )
    )
    object.__setattr__(
        package,
        "preregistration_path",
        "docs/journal/1789263159_player_v5_frozen_spawner_eval_preregistration.txt",
    )
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )
    with pytest.raises(cli.FrozenSpawnerEvalError, match="diagnostic"):
        cli.run_frozen_spawner_comparison(package, {})


def test_canonical_formal_package_can_emit_formal_schema(monkeypatch) -> None:
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs())

    def fake_outcomes(*_args, **_kwargs):
        return [{"elapsed": 1.0, "frames": 1, "hit": True, "censored": False, "termination_reason": "hit"}] * 30

    monkeypatch.setattr(cli, "evaluate_role_outcomes", fake_outcomes)
    report = cli.run_frozen_spawner_comparison(package, _verified_loaded_checkpoints(package))
    assert report["schema"] == "frozen_spawner_eval_report.v1"
    assert report["manifest"]["contract_id"] == "player_v5_frozen_spawner_eval_v1"


@pytest.mark.parametrize("flag", ("promotion", "training", "self_play"))
def test_report_preflight_rejects_bypass_mutated_eval_only_flags_before_lookup_or_episode(
    monkeypatch, flag: str
) -> None:
    """Deserialized/bypassed packages cannot turn this report path into training."""
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs())
    object.__setattr__(package, flag, True)

    monkeypatch.setattr(
        cli,
        "load_checkpoint_for_role",
        lambda *_args, **_kwargs: pytest.fail("checkpoint lookup must not run"),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_role_outcomes",
        lambda *_args, **_kwargs: pytest.fail("episode must not run"),
    )

    with pytest.raises(cli.FrozenSpawnerEvalError, match=flag):
        cli.run_frozen_spawner_comparison(package, {})


def test_report_preflight_rejects_bypass_mutated_formal_numeric_representations(
    monkeypatch,
) -> None:
    """Formal identity must validate the raw package, before any execution.

    A bypassed/deserialized package must not become reportable merely because
    reconstruction happens to coerce numerically equal floats back to ints.
    """
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs())
    object.__setattr__(package, "env_seeds", tuple(float(x) for x in range(3000, 3030)))
    object.__setattr__(package, "episode_cap_frames", 4200.0)

    monkeypatch.setattr(
        cli,
        "load_checkpoint_for_role",
        lambda *_args, **_kwargs: pytest.fail("checkpoint lookup must not run"),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_role_outcomes",
        lambda *_args, **_kwargs: pytest.fail("episode must not run"),
    )

    checkpoints = {
        role: (object(), {"architecture": architecture})
        for role, architecture in ROLE_ARCHITECTURE.items()
    }
    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(package, checkpoints)


def test_report_preflight_rejects_bypass_mutated_diagnostic_numeric_representations(
    monkeypatch,
) -> None:
    """Diagnostic reports have the same raw numeric representation rules."""
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            env_seeds=(7, 8),
            spawner_seeds=(7, 8),
            episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1",
            preregistration_path=None,
        )
    )
    object.__setattr__(package, "env_seeds", (7.0, 8.0))
    object.__setattr__(package, "spawner_seeds", (7.0, 8.0))
    object.__setattr__(package, "episode_cap_frames", 12.0)

    monkeypatch.setattr(
        cli,
        "load_checkpoint_for_role",
        lambda *_args, **_kwargs: pytest.fail("checkpoint lookup must not run"),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_role_outcomes",
        lambda *_args, **_kwargs: pytest.fail("episode must not run"),
    )

    checkpoints = {
        role: (object(), {"architecture": architecture})
        for role, architecture in ROLE_ARCHITECTURE.items()
    }
    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(package, checkpoints)


def test_main_preflight_rejects_bypass_mutated_formal_numeric_representations_before_loading(
    monkeypatch,
) -> None:
    """The CLI path must reject the raw object before its first checkpoint lookup."""
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs())
    object.__setattr__(package, "env_seeds", tuple(float(x) for x in range(3000, 3030)))
    object.__setattr__(package, "episode_cap_frames", 4200.0)

    monkeypatch.setattr(cli, "load_locked_manifest", lambda _path: package)
    monkeypatch.setattr(
        cli,
        "load_checkpoint_for_role",
        lambda *_args, **_kwargs: pytest.fail("checkpoint lookup must not run"),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_role_outcomes",
        lambda *_args, **_kwargs: pytest.fail("episode must not run"),
    )

    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.main(["--manifest", "/synthetic/locked.json", "--out-report", "/synthetic/report.json"])


def test_main_preflight_rejects_raw_canonical_semantic_checkpoint_tamper_before_loading(
    monkeypatch,
) -> None:
    """CLI loading is behind the same complete semantic preflight as reports."""
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs())
    # Every representation remains canonical; only the role/architecture
    # semantic contract is tampered after construction.
    object.__setattr__(
        package.checkpoints[ROLE_V1_TEACHER], "architecture", ROLE_ARCHITECTURE[ROLE_LEARNED_SPAWNER]
    )
    monkeypatch.setattr(cli, "load_locked_manifest", lambda _path: package)
    monkeypatch.setattr(
        cli, "load_checkpoint_for_role", lambda *_args, **_kwargs: pytest.fail("checkpoint load must not run")
    )
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.main(["--manifest", "/synthetic/locked.json", "--out-report", "/synthetic/report.json"])


def test_report_preflight_rejects_bypass_mutated_diagnostic_manifest_lists_before_access(
    monkeypatch,
) -> None:
    """Raw list representations must not be normalized into a reportable package."""
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            env_seeds=(7, 8), spawner_seeds=(7, 8), episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1", preregistration_path=None,
        )
    )
    object.__setattr__(package, "player_action_mode", [False, True])
    object.__setattr__(package, "comparison_pairs", [list(pair) for pair in PREREGISTERED_COMPARISON_PAIRS])
    object.__setattr__(package, "metrics", list(package.metrics))

    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    class NoCheckpointAccess(Mapping):
        def __getitem__(self, key):
            pytest.fail(f"checkpoint access must not run: {key}")

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(package, NoCheckpointAccess())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", 7),
        ("path", object()),
        ("sha256", 7),
        ("sha256", _StringSubclass("a" * 64)),
        ("architecture", ["player_v1"]),
        ("state_dict_sha256", 7),
        ("state_dict_sha256", "not-a-sha256"),
        ("state_dict_sha256", _StringSubclass("a" * 64)),
    ],
)
def test_checkpoint_ref_constructor_and_builder_reject_invalid_raw_identity_fields(
    field: str, value: object,
) -> None:
    """Checkpoint identity fields have an exact, JSON-manifest-safe contract."""
    values: dict[str, object] = {
        "role": ROLE_V1_TEACHER,
        "path": "/synthetic/v1.pt",
        "sha256": "a" * 64,
        "architecture": ROLE_ARCHITECTURE[ROLE_V1_TEACHER],
        "state_dict_sha256": None,
    }
    values[field] = value
    with pytest.raises(FrozenSpawnerManifestError):
        CheckpointRef(**values)  # type: ignore[arg-type]

    checkpoints = _identity_checkpoints()
    checkpoints[ROLE_V1_TEACHER] = CheckpointRef(
        ROLE_V1_TEACHER, "/synthetic/v1.pt", "a" * 64, ROLE_ARCHITECTURE[ROLE_V1_TEACHER]
    )
    object.__setattr__(checkpoints[ROLE_V1_TEACHER], field, value)
    with pytest.raises(FrozenSpawnerManifestError):
        build_frozen_spawner_seed_package(**_formal_package_kwargs(checkpoints=checkpoints))


def test_checkpoint_ref_valid_identity_control_is_reportable(monkeypatch) -> None:
    """Strict identity typing preserves the valid diagnostic control path."""
    ref = CheckpointRef(
        ROLE_V1_TEACHER, "/synthetic/v1.pt", "a" * 64, ROLE_ARCHITECTURE[ROLE_V1_TEACHER], "b" * 64
    )
    assert ref.state_dict_sha256 == "b" * 64
    package = build_frozen_spawner_seed_package(
        **_formal_package_kwargs(
            env_seeds=(7, 8), spawner_seeds=(7, 8), episode_cap_frames=12,
            contract_id="frozen_spawner_diagnostic_v1", preregistration_path=None,
        )
    )
    monkeypatch.setattr(
        cli,
        "evaluate_role_outcomes",
        lambda *_args, **_kwargs: [
            {"elapsed": 1.0, "frames": 1, "hit": True, "censored": False, "termination_reason": "hit"}
        ] * 2,
    )
    report = cli.run_frozen_spawner_comparison(package, _verified_loaded_checkpoints(package))
    assert report["schema"] == "frozen_spawner_diagnostic_report.v1"


@pytest.mark.parametrize("contract", ("formal", "diagnostic"))
def test_report_preflight_rejects_bypass_mutated_checkpoint_identity_before_access(
    monkeypatch, contract: str,
) -> None:
    """Malformed raw checkpoint references cannot reach checkpoint lookup or episodes."""
    overrides: dict[str, object] = {}
    if contract == "diagnostic":
        overrides = {
            "env_seeds": (7, 8), "spawner_seeds": (7, 8), "episode_cap_frames": 12,
            "contract_id": "frozen_spawner_diagnostic_v1", "preregistration_path": None,
        }
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs(**overrides))
    object.__setattr__(package.checkpoints[ROLE_V1_TEACHER], "path", object())
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    class NoCheckpointAccess(Mapping):
        def __getitem__(self, key):
            pytest.fail(f"checkpoint access must not run: {key}")

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(package, NoCheckpointAccess())


@pytest.mark.parametrize("contract", ("formal", "diagnostic"))
@pytest.mark.parametrize("field", ("sha256", "state_dict_sha256"))
def test_report_preflight_rejects_string_subclass_checkpoint_hash_before_access(
    monkeypatch, contract: str, field: str,
) -> None:
    """Hash fields must be exact strings, including after a bypass mutation."""
    overrides: dict[str, object] = {}
    if contract == "diagnostic":
        overrides = {
            "env_seeds": (7, 8), "spawner_seeds": (7, 8), "episode_cap_frames": 12,
            "contract_id": "frozen_spawner_diagnostic_v1", "preregistration_path": None,
        }
    package = build_frozen_spawner_seed_package(**_formal_package_kwargs(**overrides))
    object.__setattr__(package.checkpoints[ROLE_V1_TEACHER], field, _StringSubclass("a" * 64))
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    class NoCheckpointAccess(Mapping):
        def __getitem__(self, key):
            pytest.fail(f"checkpoint access must not run: {key}")

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    with pytest.raises(cli.FrozenSpawnerEvalError, match="noncanonical"):
        cli.run_frozen_spawner_comparison(package, NoCheckpointAccess())
