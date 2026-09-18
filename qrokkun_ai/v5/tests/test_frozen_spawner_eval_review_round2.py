"""RED review evidence for the formal frozen-Spawner metric contract."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import qrokkun_ai.v5.frozen_spawner_eval_cli as cli
from qrokkun_ai.v5.agents.player_checkpoints import state_dict_sha256
from qrokkun_ai.v5.frozen_spawner_seed_package import (
    CONTRACT_ID,
    DEFAULT_METRICS,
    FORMAL_ENV_SEEDS,
    FORMAL_EPISODE_CAP_FRAMES,
    FORMAL_PLAYER_ACTION_MODE,
    FORMAL_SPAWNER_ACTION_MODE,
    PREREGISTERED_COMPARISON_PAIRS,
    PREREGISTRATION_PATH,
    CheckpointRef,
    FrozenSpawnerManifestError,
    FrozenSpawnerSeedPackage,
    ROLE_ARCHITECTURE,
    ROLE_LEARNED_SPAWNER,
    ROLE_V1_TEACHER,
    ROLE_V5_AUX_U200,
    ROLE_V5_BC_SEED6,
    build_frozen_spawner_seed_package,
)


def _synthetic_net(role: str) -> torch.nn.Module:
    """A tiny, role-distinguishable module with a stable state dict."""
    net = torch.nn.Linear(1, 1)
    with torch.no_grad():
        value = float((ROLE_V1_TEACHER, ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200, ROLE_LEARNED_SPAWNER).index(role) + 1)
        net.weight.fill_(value)
        net.bias.fill_(-value)
    return net


def _synthetic_formal_kwargs(**overrides: object) -> dict[str, object]:
    """Canonical formal inputs with synthetic-only checkpoint identities."""
    values: dict[str, object] = {
        "checkpoints": {
            role: CheckpointRef(
                role,
                f"/synthetic/{role}.pt",
                digest * 64,
                ROLE_ARCHITECTURE[role],
                state_dict_sha256=state_dict_sha256(_synthetic_net(role).state_dict()),
            )
            for role, digest in zip(
                (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200, ROLE_LEARNED_SPAWNER),
                "abcd",
                strict=True,
            )
        },
        "player_action_mode": FORMAL_PLAYER_ACTION_MODE,
        "spawner_action_mode": FORMAL_SPAWNER_ACTION_MODE,
        "env_seeds": FORMAL_ENV_SEEDS,
        "spawner_seeds": FORMAL_ENV_SEEDS,
        "episode_cap_frames": FORMAL_EPISODE_CAP_FRAMES,
        "comparison_pairs": PREREGISTERED_COMPARISON_PAIRS,
        "contract_id": CONTRACT_ID,
        "preregistration_path": PREREGISTRATION_PATH,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "construct",
    (FrozenSpawnerSeedPackage, build_frozen_spawner_seed_package),
    ids=("dataclass", "builder"),
)
def test_formal_contract_defaults_to_the_complete_preregistered_metrics(
    construct,
) -> None:
    package = construct(**_synthetic_formal_kwargs())

    assert package.metrics == DEFAULT_METRICS


@pytest.mark.parametrize(
    "construct",
    (FrozenSpawnerSeedPackage, build_frozen_spawner_seed_package),
    ids=("dataclass", "builder"),
)
def test_formal_contract_rejects_metric_subsets_before_serialization_or_report_execution(
    construct,
) -> None:
    with pytest.raises(FrozenSpawnerManifestError, match="metrics"):
        construct(**_synthetic_formal_kwargs(metrics=("restricted_mean",)))


def _loaded_synthetic_identities(package: FrozenSpawnerSeedPackage) -> dict[str, tuple[object, dict]]:
    return {
        role: (
            _synthetic_net(role),
            {
                "role": role,
                "architecture": ref.architecture,
                "file_sha256": ref.sha256,
                "state_dict_sha256": ref.state_dict_sha256,
            },
        )
        for role, ref in package.checkpoints.items()
    }


def test_comparison_rejects_swapped_loaded_bc_aux_identities_before_episode(monkeypatch) -> None:
    package = FrozenSpawnerSeedPackage(**_synthetic_formal_kwargs())
    checkpoints = _loaded_synthetic_identities(package)
    checkpoints[ROLE_V5_BC_SEED6], checkpoints[ROLE_V5_AUX_U200] = (
        checkpoints[ROLE_V5_AUX_U200],
        checkpoints[ROLE_V5_BC_SEED6],
    )
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    with pytest.raises(cli.FrozenSpawnerEvalError, match="identity"):
        cli.run_frozen_spawner_comparison(package, checkpoints)


def test_comparison_rejects_bc_aux_net_only_swap_despite_matching_metadata_before_episode(monkeypatch) -> None:
    """Metadata provenance cannot stand in for the supplied module's weights."""
    package = FrozenSpawnerSeedPackage(**_synthetic_formal_kwargs())
    checkpoints = _loaded_synthetic_identities(package)
    bc_net = checkpoints[ROLE_V5_BC_SEED6][0]
    aux_net = checkpoints[ROLE_V5_AUX_U200][0]
    checkpoints[ROLE_V5_BC_SEED6] = (aux_net, checkpoints[ROLE_V5_BC_SEED6][1])
    checkpoints[ROLE_V5_AUX_U200] = (bc_net, checkpoints[ROLE_V5_AUX_U200][1])
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    with pytest.raises(cli.FrozenSpawnerEvalError, match="identity"):
        cli.run_frozen_spawner_comparison(package, checkpoints)


def test_comparison_rejects_missing_loaded_identity_metadata_before_episode(monkeypatch) -> None:
    package = FrozenSpawnerSeedPackage(**_synthetic_formal_kwargs())
    checkpoints = _loaded_synthetic_identities(package)
    del checkpoints[ROLE_V1_TEACHER][1]["file_sha256"]
    monkeypatch.setattr(
        cli, "evaluate_role_outcomes", lambda *_args, **_kwargs: pytest.fail("episode must not run")
    )

    with pytest.raises(cli.FrozenSpawnerEvalError, match="identity"):
        cli.run_frozen_spawner_comparison(package, checkpoints)


def test_comparison_accepts_matching_loaded_synthetic_identities(monkeypatch) -> None:
    package = FrozenSpawnerSeedPackage(**_synthetic_formal_kwargs())
    calls: list[str] = []

    def fake_outcomes(architecture: str, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        calls.append(architecture)
        return [
            {"elapsed": 1.0, "hit": True, "censored": False, "termination_reason": "hit"}
            for _ in package.env_seeds
        ]

    monkeypatch.setattr(cli, "evaluate_role_outcomes", fake_outcomes)
    report = cli.run_frozen_spawner_comparison(package, _loaded_synthetic_identities(package))

    assert calls == ["player_v1", "player_ranked_topk", "player_ranked_topk"]
    assert report["schema"] == "frozen_spawner_eval_report.v1"
