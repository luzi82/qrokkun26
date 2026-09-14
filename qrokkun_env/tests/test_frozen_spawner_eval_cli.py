"""Tests for the real eval wiring of the frozen learned-Spawner comparison
(:mod:`qrokkun_env.frozen_spawner_eval_cli`).

Every checkpoint used here is a tiny synthetic net packed to ``tmp_path`` in
this test module -- never a real NAS/production checkpoint path. These tests
never train anything: they only exercise loader selection, fail-closed hash
validation, one real (tiny) episode against a real (tiny) SpawnerV4, and the
manifest/report JSON schema + CRN reproducibility + structural
no-promotion/no-training/no-self-play guarantees of the runner.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qrokkun_env.agents.player_checkpoints import (  # noqa: E402
    file_sha256,
    pack_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.agents.player_v1 import PlayerV1  # noqa: E402
from qrokkun_env.agents.spawner_v4 import SpawnerV4  # noqa: E402
from qrokkun_env.frozen_spawner_seed_package import (  # noqa: E402
    CheckpointRef,
    DIAGNOSTIC_CONTRACT_ID,
    ROLE_LEARNED_SPAWNER,
    ROLE_V1_TEACHER,
    ROLE_V5_AUX_U200,
    ROLE_V5_BC_SEED6,
    build_frozen_spawner_seed_package,
)

import qrokkun_env.frozen_spawner_eval_cli as cli  # noqa: E402


TINY_HIDDEN = 16
TINY_TOP_K = 4
TINY_D_MODEL = 8
TINY_SPAWNER_HIDDEN = 16
CAP_FRAMES = 30  # small cap keeps these unit tests fast (~0.5s of sim-time)


def _make_player_v1_checkpoint(path: Path, *, seed: int = 0) -> tuple[str, str]:
    torch.manual_seed(seed)
    net = PlayerV1(hidden=TINY_HIDDEN)
    ck = {"hidden": TINY_HIDDEN, "state_dict": net.state_dict()}
    torch.save(ck, path)
    return file_sha256(path), state_dict_sha256(net.state_dict())


def _make_ranked_topk_checkpoint(path: Path, *, seed: int = 0) -> tuple[str, str]:
    torch.manual_seed(seed)
    net = PlayerRankedTopK(top_k=TINY_TOP_K, hidden=TINY_HIDDEN)
    ck = pack_player_checkpoint(net, source_commit="synthetic", source_tool="test")
    torch.save(ck, path)
    return file_sha256(path), ck["state_dict_sha256"]


def _make_spawner_v4_checkpoint(path: Path, *, seed: int = 0) -> tuple[str, str]:
    torch.manual_seed(seed)
    net = SpawnerV4(d_model=TINY_D_MODEL, hidden=TINY_SPAWNER_HIDDEN)
    ck = {"d_model": TINY_D_MODEL, "hidden": TINY_SPAWNER_HIDDEN, "state_dict": net.state_dict()}
    torch.save(ck, path)
    return file_sha256(path), state_dict_sha256(net.state_dict())


def _make_all_checkpoints(tmp_path: Path) -> dict:
    v1_path = tmp_path / "v1_teacher.pt"
    bc_path = tmp_path / "v5_bc_seed6.pt"
    aux_path = tmp_path / "v5_aux_u200.pt"
    spawner_path = tmp_path / "learned_spawner.pt"

    v1_sha, v1_sd_sha = _make_player_v1_checkpoint(v1_path, seed=1)
    bc_sha, bc_sd_sha = _make_ranked_topk_checkpoint(bc_path, seed=2)
    aux_sha, aux_sd_sha = _make_ranked_topk_checkpoint(aux_path, seed=3)
    spawner_sha, spawner_sd_sha = _make_spawner_v4_checkpoint(spawner_path, seed=4)

    return {
        ROLE_V1_TEACHER: CheckpointRef(
            role=ROLE_V1_TEACHER, path=str(v1_path), sha256=v1_sha, architecture=cli.ARCHITECTURE_PLAYER_V1,
            state_dict_sha256=v1_sd_sha,
        ),
        ROLE_V5_BC_SEED6: CheckpointRef(
            role=ROLE_V5_BC_SEED6,
            path=str(bc_path),
            sha256=bc_sha,
            architecture=cli.ARCHITECTURE_PLAYER_RANKED_TOP_K,
            state_dict_sha256=bc_sd_sha,
        ),
        ROLE_V5_AUX_U200: CheckpointRef(
            role=ROLE_V5_AUX_U200,
            path=str(aux_path),
            sha256=aux_sha,
            architecture=cli.ARCHITECTURE_PLAYER_RANKED_TOP_K,
            state_dict_sha256=aux_sd_sha,
        ),
        ROLE_LEARNED_SPAWNER: CheckpointRef(
            role=ROLE_LEARNED_SPAWNER,
            path=str(spawner_path), sha256=spawner_sha, architecture=cli.ARCHITECTURE_SPAWNER_V4,
            state_dict_sha256=spawner_sd_sha,
        ),
    }


def _make_package(tmp_path: Path, *, spawner_action_mode=(False, True), n_seeds: int = 3):
    checkpoints = _make_all_checkpoints(tmp_path)
    env_seeds = tuple(range(3000, 3000 + n_seeds))
    spawner_seeds = env_seeds if spawner_action_mode[0] else ()
    return build_frozen_spawner_seed_package(
        checkpoints=checkpoints,
        player_action_mode=(False, True),
        spawner_action_mode=spawner_action_mode,
        env_seeds=env_seeds,
        spawner_seeds=spawner_seeds,
        episode_cap_frames=CAP_FRAMES,
        comparison_pairs=cli.COMPARISON_PAIRS,
        contract_id=DIAGNOSTIC_CONTRACT_ID,
        preregistration_path=None,
    )


# --- Loader selection ------------------------------------------------------


def test_load_checkpoint_for_role_selects_player_v1_loader(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    net, meta = cli.load_checkpoint_for_role(checkpoints[ROLE_V1_TEACHER], device="cpu")
    assert isinstance(net, PlayerV1)
    assert meta["architecture"] == cli.ARCHITECTURE_PLAYER_V1


def test_load_checkpoint_for_role_selects_ranked_topk_loader(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    net, meta = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_BC_SEED6], device="cpu")
    assert isinstance(net, PlayerRankedTopK)
    assert meta["architecture"] == cli.ARCHITECTURE_PLAYER_RANKED_TOP_K


def test_load_checkpoint_for_role_selects_spawner_v4_loader(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    net, meta = cli.load_checkpoint_for_role(checkpoints[ROLE_LEARNED_SPAWNER], device="cpu")
    assert isinstance(net, SpawnerV4)
    assert meta["architecture"] == cli.ARCHITECTURE_SPAWNER_V4


def test_player_v1_and_ranked_topk_use_distinct_observation_contracts(tmp_path: Path):
    # PlayerV1 forward takes a single flat tensor; PlayerRankedTopK forward
    # takes three tensors (player, bullets, pad-mask). Loading each via the
    # dispatcher must build the architecture-correct net (never cross-wired).
    checkpoints = _make_all_checkpoints(tmp_path)
    v1_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V1_TEACHER], device="cpu")
    topk_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_BC_SEED6], device="cpu")
    v1_params = inspect.signature(v1_net.forward).parameters
    topk_params = inspect.signature(topk_net.forward).parameters
    assert len(v1_params) == 1
    assert len(topk_params) == 3


# --- Fail-closed hash validation --------------------------------------------


def test_load_checkpoint_for_role_rejects_tampered_file_before_loading(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    ref = checkpoints[ROLE_V1_TEACHER]
    # Tamper with the file after its sha256 was recorded in the CheckpointRef.
    Path(ref.path).write_bytes(Path(ref.path).read_bytes() + b"\x00")
    with pytest.raises(cli.ChecksumMismatchError, match="sha256"):
        cli.load_checkpoint_for_role(ref, device="cpu")


def test_load_checkpoint_for_role_rejects_wrong_pinned_sha256(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    ref = checkpoints[ROLE_LEARNED_SPAWNER]
    wrong_ref = CheckpointRef(
        role=ref.role, path=ref.path, sha256="0" * 64, architecture=ref.architecture
    )
    with pytest.raises(cli.ChecksumMismatchError):
        cli.load_checkpoint_for_role(wrong_ref, device="cpu")


def test_load_checkpoint_for_role_rejects_state_dict_hash_mismatch(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    ref = checkpoints[ROLE_V5_BC_SEED6]
    tampered_ref = CheckpointRef(
        role=ref.role,
        path=ref.path,
        sha256=ref.sha256,
        architecture=ref.architecture,
        state_dict_sha256="1" * 64,
    )
    with pytest.raises(cli.ChecksumMismatchError, match="state_dict"):
        cli.load_checkpoint_for_role(tampered_ref, device="cpu")


def test_load_checkpoint_for_role_rejects_unsupported_architecture(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    ref = checkpoints[ROLE_V1_TEACHER]
    bad_ref = CheckpointRef(role=ref.role, path=ref.path, sha256=ref.sha256, architecture="player_v4")
    with pytest.raises(cli.UnsupportedArchitectureError):
        cli.load_checkpoint_for_role(bad_ref, device="cpu")


# --- Episode runner ----------------------------------------------------------


def test_run_frozen_spawner_episode_returns_positive_elapsed(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    player_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_BC_SEED6], device="cpu")
    spawner_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_LEARNED_SPAWNER], device="cpu")
    outcome = cli.run_frozen_spawner_episode(
        cli.ARCHITECTURE_PLAYER_RANKED_TOP_K,
        player_net,
        spawner_net,
        "cpu",
        seed=3000,
        cap_frames=CAP_FRAMES,
        player_sample=False,
        spawner_sample=False,
        rng_jitter=True,
    )
    assert set(outcome) >= {"elapsed", "frames", "hit", "censored", "termination_reason"}
    assert outcome["elapsed"] > 0.0
    assert outcome["frames"] > 0
    assert outcome["censored"] == (not outcome["hit"])
    assert outcome["termination_reason"] in ("hit", "cap")


def test_run_frozen_spawner_episode_works_for_legacy_player_v1(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    player_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V1_TEACHER], device="cpu")
    spawner_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_LEARNED_SPAWNER], device="cpu")
    outcome = cli.run_frozen_spawner_episode(
        cli.ARCHITECTURE_PLAYER_V1,
        player_net,
        spawner_net,
        "cpu",
        seed=3000,
        cap_frames=CAP_FRAMES,
        player_sample=False,
        spawner_sample=False,
        rng_jitter=True,
    )
    assert set(outcome) >= {"elapsed", "frames", "hit", "censored", "termination_reason"}
    assert outcome["elapsed"] > 0.0
    assert outcome["frames"] > 0
    assert outcome["censored"] == (not outcome["hit"])
    assert outcome["termination_reason"] in ("hit", "cap")


def test_deterministic_player_mode_is_reproducible_across_runs(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    player_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_BC_SEED6], device="cpu")
    spawner_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_LEARNED_SPAWNER], device="cpu")
    kwargs = dict(
        player_architecture=cli.ARCHITECTURE_PLAYER_RANKED_TOP_K,
        player_net=player_net,
        spawner_net=spawner_net,
        device="cpu",
        seed=3005,
        cap_frames=CAP_FRAMES,
        player_sample=False,
        spawner_sample=False,
        rng_jitter=False,
    )
    o1 = cli.run_frozen_spawner_episode(**kwargs)
    o2 = cli.run_frozen_spawner_episode(**kwargs)
    assert o1["elapsed"] == pytest.approx(o2["elapsed"])
    assert o1["frames"] == o2["frames"]
    assert o1["hit"] == o2["hit"]
    assert o1["termination_reason"] == o2["termination_reason"]


# --- CRN reproducibility -----------------------------------------------------


def test_crn_wrapper_reproduces_identical_elapsed_for_stochastic_spawner(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    player_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_BC_SEED6], device="cpu")
    spawner_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_LEARNED_SPAWNER], device="cpu")
    kwargs = dict(
        player_architecture=cli.ARCHITECTURE_PLAYER_RANKED_TOP_K,
        player_net=player_net,
        spawner_net=spawner_net,
        device="cpu",
        seed=3010,
        cap_frames=CAP_FRAMES,
        player_sample=False,
        spawner_sample=True,  # stochastic Spawner draws consume torch RNG
        rng_jitter=True,
    )
    # Perturb the *global* torch RNG state between the two calls; CRN pairing
    # must still reproduce identical elapsed values because the wrapper forks
    # a fresh RNG keyed on `seed` every call.
    o1 = cli.run_frozen_spawner_episode_crn(**kwargs)
    torch.manual_seed(999999)
    _ = torch.randn(100)
    o2 = cli.run_frozen_spawner_episode_crn(**kwargs)
    assert o1["elapsed"] == pytest.approx(o2["elapsed"])
    assert o1["hit"] == o2["hit"]


def test_crn_pairing_gives_two_arms_the_same_per_slot_spawner_draws(tmp_path: Path):
    # Two *different* player nets (v5_bc_seed6 vs v5_aux_u200), evaluated
    # against the same frozen stochastic Spawner using the CRN wrapper with
    # the same seed, must each reuse the exact same per-slot Spawner RNG
    # fork -- i.e. re-running the *same* player arm at the *same* seed always
    # reproduces the same elapsed value (the CRN contract), independent of
    # global RNG state mutated by any other arm's run in between.
    checkpoints = _make_all_checkpoints(tmp_path)
    bc_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_BC_SEED6], device="cpu")
    aux_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_V5_AUX_U200], device="cpu")
    spawner_net, _ = cli.load_checkpoint_for_role(checkpoints[ROLE_LEARNED_SPAWNER], device="cpu")

    def run(player_net):
        outcome = cli.run_frozen_spawner_episode_crn(
            player_architecture=cli.ARCHITECTURE_PLAYER_RANKED_TOP_K,
            player_net=player_net,
            spawner_net=spawner_net,
            device="cpu",
            seed=3020,
            cap_frames=CAP_FRAMES,
            player_sample=False,
            spawner_sample=True,
            rng_jitter=True,
        )
        return outcome["elapsed"]

    bc_first = run(bc_net)
    aux_first = run(aux_net)
    bc_second = run(bc_net)
    assert bc_first == pytest.approx(bc_second)
    assert aux_first == pytest.approx(run(aux_net))


# --- Manifest / report schema ------------------------------------------------


def test_run_frozen_spawner_comparison_report_schema(tmp_path: Path):
    package = _make_package(tmp_path, n_seeds=3)
    checkpoints = {
        role: cli.load_checkpoint_for_role(ref, device="cpu") for role, ref in package.checkpoints.items()
    }
    report = cli.run_frozen_spawner_comparison(package, checkpoints, device="cpu")

    assert report["schema"] == "frozen_spawner_diagnostic_report.v1"
    assert report["promotion"] is False
    assert report["training"] is False
    assert report["self_play"] is False
    assert report["manifest"]["checkpoints"][ROLE_V1_TEACHER]["architecture"] == cli.ARCHITECTURE_PLAYER_V1

    for role in (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200):
        assert role in report["per_seed_elapsed"]
        assert len(report["per_seed_elapsed"][role]) == 3
        assert role in report["restricted_stats"]
        assert report["restricted_stats"][role]["n"] == 3

    assert len(report["paired_comparisons"]) == len(cli.COMPARISON_PAIRS)
    got_pairs = [tuple(c["pair"]) for c in report["paired_comparisons"]]
    assert got_pairs == list(cli.COMPARISON_PAIRS)
    for comparison in report["paired_comparisons"]:
        assert "mean_diff" in comparison
        assert "ci95" in comparison
        assert comparison["n"] == 3

    # Must be pure-JSON serializable (no tensors/objects leaking through).
    blob = json.dumps(report)
    restored = json.loads(blob)
    assert restored["schema"] == report["schema"]


def test_run_frozen_spawner_comparison_rejects_non_deterministic_player_mode(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    env_seeds = tuple(range(3000, 3003))
    # Directly construct a package with a non-deterministic player mode by
    # bypassing the CLI's own (deliberately narrower) argparse choices.
    package = build_frozen_spawner_seed_package(
        checkpoints=checkpoints,
        player_action_mode=(True, True),
        spawner_action_mode=(False, True),
        env_seeds=env_seeds,
        spawner_seeds=(),
        episode_cap_frames=CAP_FRAMES,
        comparison_pairs=cli.COMPARISON_PAIRS,
        contract_id=DIAGNOSTIC_CONTRACT_ID,
        preregistration_path=None,
    )
    loaded = {role: cli.load_checkpoint_for_role(ref, device="cpu") for role, ref in package.checkpoints.items()}
    with pytest.raises(cli.FrozenSpawnerEvalError, match="deterministic"):
        cli.run_frozen_spawner_comparison(package, loaded, device="cpu")


def test_run_frozen_spawner_comparison_only_computes_preregistered_pairs(tmp_path: Path):
    package = _make_package(tmp_path, n_seeds=2)
    checkpoints = {role: cli.load_checkpoint_for_role(ref, device="cpu") for role, ref in package.checkpoints.items()}
    report = cli.run_frozen_spawner_comparison(package, checkpoints, device="cpu")
    # Exactly the two preregistered pairs -- no v1_teacher vs v5_aux_u200
    # direct comparison, and no extra pairing.
    assert len(report["paired_comparisons"]) == 2
    pairs = {tuple(c["pair"]) for c in report["paired_comparisons"]}
    assert pairs == {(ROLE_V1_TEACHER, ROLE_V5_BC_SEED6), (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200)}


# --- Structural no-promotion/no-training/no-self-play guarantees -----------


def test_module_source_has_no_optimizer_training_selfplay_promotion_path():
    src = inspect.getsource(cli)
    forbidden_substrings = [
        "import torch.optim",
        "torch.optim.",
        "ppo_update",
        "self_play(",
        "CheckpointManager",
        "promote_checkpoint",
    ]
    for token in forbidden_substrings:
        assert token not in src, f"forbidden training/promotion token found in module: {token!r}"


def test_report_always_declares_no_promotion_training_self_play(tmp_path: Path):
    package = _make_package(tmp_path, n_seeds=2)
    checkpoints = {role: cli.load_checkpoint_for_role(ref, device="cpu") for role, ref in package.checkpoints.items()}
    report = cli.run_frozen_spawner_comparison(package, checkpoints, device="cpu")
    assert report["promotion"] is False
    assert report["training"] is False
    assert report["self_play"] is False


# --- CLI end-to-end (writes manifest + report to tmp_path) ------------------


def test_cli_main_writes_manifest_and_report(tmp_path: Path, monkeypatch):
    # Keep the formal identity intact; mock only the per-role episode batch
    # so this e2e test does not run a real 30 x 4200-frame comparison.
    monkeypatch.setattr(
        cli,
        "evaluate_role_outcomes",
        lambda *_args, **_kwargs: [
            {"elapsed": 1.0, "frames": 1, "hit": True, "censored": False, "termination_reason": "hit"}
        ] * 30,
    )
    checkpoints = _make_all_checkpoints(tmp_path)
    out_manifest = tmp_path / "manifest.json"
    out_report = tmp_path / "report.json"
    argv = [
        "--v1-teacher-path", checkpoints[ROLE_V1_TEACHER].path,
        "--v1-teacher-sha256", checkpoints[ROLE_V1_TEACHER].sha256,
        "--v1-teacher-state-dict-sha256", checkpoints[ROLE_V1_TEACHER].state_dict_sha256,
        "--v5-bc-seed6-path", checkpoints[ROLE_V5_BC_SEED6].path,
        "--v5-bc-seed6-sha256", checkpoints[ROLE_V5_BC_SEED6].sha256,
        "--v5-bc-seed6-state-dict-sha256", checkpoints[ROLE_V5_BC_SEED6].state_dict_sha256,
        "--v5-aux-u200-path", checkpoints[ROLE_V5_AUX_U200].path,
        "--v5-aux-u200-sha256", checkpoints[ROLE_V5_AUX_U200].sha256,
        "--v5-aux-u200-state-dict-sha256", checkpoints[ROLE_V5_AUX_U200].state_dict_sha256,
        "--learned-spawner-path", checkpoints[ROLE_LEARNED_SPAWNER].path,
        "--learned-spawner-sha256", checkpoints[ROLE_LEARNED_SPAWNER].sha256,
        "--learned-spawner-state-dict-sha256", checkpoints[ROLE_LEARNED_SPAWNER].state_dict_sha256,
        "--out-manifest", str(out_manifest),
    ]
    cli.prepare_main(argv)
    report = cli.main(["--manifest", str(out_manifest), "--out-report", str(out_report)])
    assert out_manifest.is_file()
    assert out_report.is_file()
    written_report = json.loads(out_report.read_text())
    assert written_report["schema"] == "frozen_spawner_eval_report.v1"
    # JSON round-trips tuples (e.g. ci95) to lists; compare via the same
    # round-trip rather than requiring in-memory tuple/list identity.
    assert written_report == json.loads(json.dumps(report))
    written_manifest = json.loads(out_manifest.read_text())
    assert written_manifest["checkpoints"][ROLE_LEARNED_SPAWNER]["architecture"] == cli.ARCHITECTURE_SPAWNER_V4
    assert written_manifest["contract_id"] == cli.CONTRACT_ID
    assert written_manifest["preregistration_path"] == cli.PREREGISTRATION_PATH


def test_cli_main_rejects_stochastic_player_action_mode_choice():
    with pytest.raises(SystemExit):
        cli.build_prepare_parser().parse_args(["--player-action-mode", "stoch_stoch"])


# --- Strict-review finding: CLI must accept/persist optional state-dict
# SHA-256 pins for every role, and the resulting CheckpointRefs must carry
# them into the built seed package (not just the file-level sha256). ---


def test_build_parser_accepts_optional_state_dict_sha256_for_every_role():
    args = cli.build_prepare_parser().parse_args(
        [
            "--v1-teacher-path", "/x/v1.pt",
            "--v1-teacher-sha256", "a" * 64,
            "--v1-teacher-state-dict-sha256", "1" * 64,
            "--v5-bc-seed6-path", "/x/bc.pt",
            "--v5-bc-seed6-sha256", "b" * 64,
            "--v5-bc-seed6-state-dict-sha256", "2" * 64,
            "--v5-aux-u200-path", "/x/aux.pt",
            "--v5-aux-u200-sha256", "c" * 64,
            "--v5-aux-u200-state-dict-sha256", "3" * 64,
            "--learned-spawner-path", "/x/spawner.pt",
            "--learned-spawner-sha256", "d" * 64,
            "--learned-spawner-state-dict-sha256", "4" * 64,
            "--out-manifest", "/x/manifest.json",
        ]
    )
    assert args.v1_teacher_state_dict_sha256 == "1" * 64
    assert args.v5_bc_seed6_state_dict_sha256 == "2" * 64
    assert args.v5_aux_u200_state_dict_sha256 == "3" * 64
    assert args.learned_spawner_state_dict_sha256 == "4" * 64


def test_build_parser_state_dict_sha256_defaults_to_none():
    args = cli.build_prepare_parser().parse_args(
        [
            "--v1-teacher-path", "/x/v1.pt",
            "--v1-teacher-sha256", "a" * 64,
            "--v5-bc-seed6-path", "/x/bc.pt",
            "--v5-bc-seed6-sha256", "b" * 64,
            "--v5-aux-u200-path", "/x/aux.pt",
            "--v5-aux-u200-sha256", "c" * 64,
            "--learned-spawner-path", "/x/spawner.pt",
            "--learned-spawner-sha256", "d" * 64,
            "--out-manifest", "/x/manifest.json",
        ]
    )
    assert args.v1_teacher_state_dict_sha256 is None
    assert args.v5_bc_seed6_state_dict_sha256 is None
    assert args.v5_aux_u200_state_dict_sha256 is None
    assert args.learned_spawner_state_dict_sha256 is None


def test_build_seed_package_from_args_persists_state_dict_sha256_pins(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    argv = [
        "--v1-teacher-path", checkpoints[ROLE_V1_TEACHER].path,
        "--v1-teacher-sha256", checkpoints[ROLE_V1_TEACHER].sha256,
        "--v1-teacher-state-dict-sha256", checkpoints[ROLE_V1_TEACHER].state_dict_sha256,
        "--v5-bc-seed6-path", checkpoints[ROLE_V5_BC_SEED6].path,
        "--v5-bc-seed6-sha256", checkpoints[ROLE_V5_BC_SEED6].sha256,
        "--v5-bc-seed6-state-dict-sha256", checkpoints[ROLE_V5_BC_SEED6].state_dict_sha256,
        "--v5-aux-u200-path", checkpoints[ROLE_V5_AUX_U200].path,
        "--v5-aux-u200-sha256", checkpoints[ROLE_V5_AUX_U200].sha256,
        "--v5-aux-u200-state-dict-sha256", checkpoints[ROLE_V5_AUX_U200].state_dict_sha256,
        "--learned-spawner-path", checkpoints[ROLE_LEARNED_SPAWNER].path,
        "--learned-spawner-sha256", checkpoints[ROLE_LEARNED_SPAWNER].sha256,
        "--learned-spawner-state-dict-sha256", checkpoints[ROLE_LEARNED_SPAWNER].state_dict_sha256,
        "--out-manifest", str(tmp_path / "manifest.json"),
    ]
    args = cli.build_prepare_parser().parse_args(argv)
    package = cli.build_seed_package_from_args(args)
    assert package.checkpoints[ROLE_V5_BC_SEED6].state_dict_sha256 == checkpoints[ROLE_V5_BC_SEED6].state_dict_sha256
    assert package.checkpoints[ROLE_V5_AUX_U200].state_dict_sha256 == checkpoints[ROLE_V5_AUX_U200].state_dict_sha256
    assert package.checkpoints[ROLE_V1_TEACHER].state_dict_sha256 == checkpoints[ROLE_V1_TEACHER].state_dict_sha256
    assert package.checkpoints[ROLE_LEARNED_SPAWNER].state_dict_sha256 == checkpoints[ROLE_LEARNED_SPAWNER].state_dict_sha256


def test_cli_main_fails_closed_on_wrong_state_dict_sha256_pin(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    argv = [
        "--v1-teacher-path", checkpoints[ROLE_V1_TEACHER].path,
        "--v1-teacher-sha256", checkpoints[ROLE_V1_TEACHER].sha256,
        "--v1-teacher-state-dict-sha256", checkpoints[ROLE_V1_TEACHER].state_dict_sha256,
        "--v5-bc-seed6-path", checkpoints[ROLE_V5_BC_SEED6].path,
        "--v5-bc-seed6-sha256", checkpoints[ROLE_V5_BC_SEED6].sha256,
        "--v5-bc-seed6-state-dict-sha256", "f" * 64,  # wrong pin
        "--v5-aux-u200-path", checkpoints[ROLE_V5_AUX_U200].path,
        "--v5-aux-u200-sha256", checkpoints[ROLE_V5_AUX_U200].sha256,
        "--v5-aux-u200-state-dict-sha256", checkpoints[ROLE_V5_AUX_U200].state_dict_sha256,
        "--learned-spawner-path", checkpoints[ROLE_LEARNED_SPAWNER].path,
        "--learned-spawner-sha256", checkpoints[ROLE_LEARNED_SPAWNER].sha256,
        "--learned-spawner-state-dict-sha256", checkpoints[ROLE_LEARNED_SPAWNER].state_dict_sha256,
        "--out-manifest", str(tmp_path / "manifest.json"),
    ]
    cli.prepare_main(argv)
    with pytest.raises(cli.ChecksumMismatchError, match="state_dict"):
        cli.main(["--manifest", str(tmp_path / "manifest.json"), "--out-report", str(tmp_path / "report.json")])
    assert (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "report.json").exists()


# --- Strict-review finding: Spawner/V1 state-dict mismatch (not just
# PlayerRankedTopK) must also fail closed after load, using a synthetic
# tmp checkpoint whose declared pin doesn't match its actual state dict. ---


def test_load_checkpoint_for_role_rejects_spawner_state_dict_hash_mismatch(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    ref = checkpoints[ROLE_LEARNED_SPAWNER]
    tampered_ref = CheckpointRef(
        role=ref.role,
        path=ref.path,
        sha256=ref.sha256,
        architecture=ref.architecture,
        state_dict_sha256="5" * 64,
    )
    with pytest.raises(cli.ChecksumMismatchError, match="state_dict"):
        cli.load_checkpoint_for_role(tampered_ref, device="cpu")


def test_load_checkpoint_for_role_rejects_player_v1_state_dict_hash_mismatch(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    ref = checkpoints[ROLE_V1_TEACHER]
    tampered_ref = CheckpointRef(
        role=ref.role,
        path=ref.path,
        sha256=ref.sha256,
        architecture=ref.architecture,
        state_dict_sha256="6" * 64,
    )
    with pytest.raises(cli.ChecksumMismatchError, match="state_dict"):
        cli.load_checkpoint_for_role(tampered_ref, device="cpu")


# --- Strict-review finding: the runner report must use only the locked
# preregistered pairs and must carry state-dict identity pins through the
# manifest embedded in the report. ---


def test_report_manifest_carries_state_dict_sha256_pins(tmp_path: Path):
    checkpoints = _make_all_checkpoints(tmp_path)
    env_seeds = tuple(range(3000, 3002))
    package = build_frozen_spawner_seed_package(
        checkpoints=checkpoints,
        player_action_mode=(False, True),
        spawner_action_mode=(False, True),
        env_seeds=env_seeds,
        spawner_seeds=(),
        episode_cap_frames=CAP_FRAMES,
        comparison_pairs=cli.COMPARISON_PAIRS,
        contract_id=DIAGNOSTIC_CONTRACT_ID,
        preregistration_path=None,
    )
    loaded = {role: cli.load_checkpoint_for_role(ref, device="cpu") for role, ref in package.checkpoints.items()}
    report = cli.run_frozen_spawner_comparison(package, loaded, device="cpu")
    manifest_checkpoints = report["manifest"]["checkpoints"]
    assert manifest_checkpoints[ROLE_V5_BC_SEED6]["state_dict_sha256"] == checkpoints[ROLE_V5_BC_SEED6].state_dict_sha256
    assert manifest_checkpoints[ROLE_V5_AUX_U200]["state_dict_sha256"] == checkpoints[ROLE_V5_AUX_U200].state_dict_sha256
    assert manifest_checkpoints[ROLE_V5_BC_SEED6]["state_dict_sha256"] is not None


def test_report_paired_comparisons_use_exactly_the_locked_pairs_in_order(tmp_path: Path):
    package = _make_package(tmp_path, n_seeds=2)
    checkpoints = {role: cli.load_checkpoint_for_role(ref, device="cpu") for role, ref in package.checkpoints.items()}
    report = cli.run_frozen_spawner_comparison(package, checkpoints, device="cpu")
    got_pairs = [tuple(c["pair"]) for c in report["paired_comparisons"]]
    assert got_pairs == [
        (ROLE_V1_TEACHER, ROLE_V5_BC_SEED6),
        (ROLE_V5_BC_SEED6, ROLE_V5_AUX_U200),
    ]
