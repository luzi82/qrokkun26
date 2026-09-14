"""Regression tests for review blockers B1-B4 in
tools/phase3_ranked_ppo_retention.py and qrokkun_env/agents/player_checkpoints.py.

Each test below is a targeted RED-then-GREEN regression for one blocker:

* B1: ``collect_canonical_dataset`` must reuse the exact Phase2
  ``collect_dataset``/``collect_episode`` contract (full episode to
  ``max_steps`` then even-stride cap subsampling, episode-level held split
  with the shared data seed) -- never truncate to the first
  ``frames_cap`` frames.
* B2: dataset identity must hash actual Frame/tensor content via
  ``tools.phase2_ranked_multiseed.dataset_identity`` (player, bullets, pad,
  teacher logits, elapsed, episode) and bind the teacher file SHA-256;
  episode counts and frame counts must be reported under unambiguous names.
* B3: the reused ``player_v1.gae`` always appends a terminal value of zero,
  so this control never bootstraps values past a rollout boundary and
  truncated (non-``done``) rollouts are treated exactly like terminated
  ones. The knobs dict and every per-update rollout log line must say so
  truthfully, and rollout censoring must be logged every update.
* B4: every PPO snapshot/final checkpoint must be packed as
  ``experimental=True`` / ``production_compatible=False`` with an ``extra``
  block recording update, parent (initial) state-dict/file hash, dataset
  hash, eval summary, every PPO knob, and the rollout/eval seed windows.

Plus caveats: knobs must reflect the *actual* quick/full max_frames /
updates / episodes / eval seeds / eval max steps / gates used by a run
(never only the production defaults); rollout seeds must be produced by a
single declared helper (``rollout_seed_schedule``), not reimplemented
inline; a missing initial held teacher agreement must fail closed rather
than silently defaulting to 0.0; and the module must document its
``PYTHONPATH`` invocation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qrokkun_env.agents.player_checkpoints import (  # noqa: E402
    file_sha256,
    load_ranked_top_k_checkpoint,
    pack_player_checkpoint,
    save_player_checkpoint,
    state_dict_sha256,
)
from qrokkun_env.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_env.agents.player_v1 import OBS_DIM_V1, PlayerV1  # noqa: E402
from qrokkun_env.env import ACTIONS  # noqa: E402

from tools import phase2_ranked_multiseed as rm  # noqa: E402
from tools import phase3_ranked_ppo_retention as ret  # noqa: E402
from tools.phase2_distill_v1_to_v4 import collect_dataset  # noqa: E402

TEACHER_CKPT = _REPO_ROOT / "artifacts" / "nas_tmp_runs" / "player_gpu.pt"


def _tiny_net(seed: int = 0, hidden: int = 16, top_k: int = 8) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def _tiny_teacher(seed: int = 0, hidden: int = 8) -> PlayerV1:
    torch.manual_seed(seed)
    return PlayerV1(hidden=hidden)


def _write_ranked_ckpt(tmp_path: Path, name: str = "init.pt", **kwargs: Any) -> tuple[Path, PlayerRankedTopK]:
    net = _tiny_net(**kwargs)
    path = tmp_path / name
    save_player_checkpoint(net, path, source_tool="tests")
    return path, net


def _write_tiny_teacher_ckpt(tmp_path: Path, name: str = "teacher.pt", *, hidden: int = 8) -> Path:
    net = _tiny_teacher(hidden=hidden)
    path = tmp_path / name
    torch.save({"hidden": hidden, "state_dict": net.state_dict(), "actions": list(range(6))}, path)
    return path


@pytest.mark.parametrize(
    ("production_compatible", "experimental"),
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
def test_load_initial_checkpoint_rejects_every_non_production_flag_combination(
    tmp_path: Path, production_compatible: bool, experimental: bool
) -> None:
    path, _net = _write_ranked_ckpt(tmp_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["production_compatible"] = production_compatible
    ckpt["experimental"] = experimental
    torch.save(ckpt, path)

    with pytest.raises(ret.CheckpointError, match="production-compatible"):
        ret.load_initial_checkpoint(path, torch.device("cpu"))


def test_load_initial_checkpoint_preserves_valid_production_load(tmp_path: Path) -> None:
    path, source = _write_ranked_ckpt(tmp_path)

    loaded, meta = ret.load_initial_checkpoint(path, torch.device("cpu"))

    assert meta["production_compatible"] is True
    assert meta["experimental"] is False
    assert state_dict_sha256(loaded.state_dict()) == state_dict_sha256(source.state_dict())


def test_load_teacher_uses_safe_deserialization_and_strict_legacy_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_teacher(seed=3, hidden=8)
    path = tmp_path / "canonical_legacy_teacher.pt"
    torch.save(
        {
            "obs_dim": OBS_DIM_V1,
            "actions": list(ACTIONS),
            "hidden": 8,
            "state_dict": source.state_dict(),
        },
        path,
    )

    real_torch_load = torch.load
    load_kwargs: list[dict[str, Any]] = []

    def recording_torch_load(*args: Any, **kwargs: Any) -> Any:
        load_kwargs.append(kwargs.copy())
        return real_torch_load(*args, **kwargs)

    real_load_state_dict = PlayerV1.load_state_dict
    strict_values: list[bool] = []

    def recording_load_state_dict(self: PlayerV1, state_dict: Any, strict: bool = True) -> Any:
        strict_values.append(strict)
        return real_load_state_dict(self, state_dict, strict=strict)

    monkeypatch.setattr(ret.torch, "load", recording_torch_load)
    monkeypatch.setattr(PlayerV1, "load_state_dict", recording_load_state_dict)

    loaded = ret.load_teacher(path, torch.device("cpu"))

    assert load_kwargs == [{"map_location": "cpu", "weights_only": True}]
    assert strict_values == [True]
    assert loaded.training is False
    assert loaded.body[0].out_features == 8
    for key, expected in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)


# --------------------------------------------------------------------------- #
# B1: canonical dataset collection reuses the exact Phase2 contract
# --------------------------------------------------------------------------- #
def test_collect_canonical_dataset_matches_phase2_collect_dataset_hash() -> None:
    """After collecting with identical parameters/data seed, the dataset hash
    must equal ``dataset_identity`` applied to Phase2's own
    ``collect_dataset`` output -- i.e. this harness must not reimplement
    collection, it must call the exact same helper."""
    import random

    device = torch.device("cpu")
    teacher = _tiny_teacher(hidden=8)
    episodes, seed_start, max_steps, frames_cap, held_frac = 6, 20000, 25, 8, 0.34

    dataset = ret.collect_canonical_dataset(
        teacher,
        device,
        episodes=episodes,
        seed_start=seed_start,
        max_steps=max_steps,
        frames_cap=frames_cap,
        held_frac=held_frac,
    )

    rng = random.Random(ret.DATA_SEED)
    train_frames, held_frames = collect_dataset(
        teacher,
        device,
        n_episodes=episodes,
        seed_start=seed_start,
        max_steps=max_steps,
        frames_cap=frames_cap,
        held_out_frac=held_frac,
        rng=rng,
    )
    identity = rm.dataset_identity(train_frames, held_frames)

    assert dataset["hash"] == identity["hash"]
    assert dataset["n_held_frames"] == identity["n_held"]
    assert dataset["n_train_frames"] == identity["n_train"]


def test_collect_canonical_dataset_held_elapsed_extends_beyond_first_cap_frames() -> None:
    """The cap must be an even-stride subsample of the WHOLE episode (up to
    max_steps), never a truncation to the first ``frames_cap`` frames -- so
    held elapsed values must reach past ``frames_cap / 60`` seconds whenever
    an episode actually survives close to ``max_steps``."""
    device = torch.device("cpu")
    teacher = _tiny_teacher(hidden=8)
    max_steps = 120  # 2.0s @ 60Hz
    frames_cap = 10  # "first-N" bug ceiling would be 10/60 =~ 0.167s

    dataset = ret.collect_canonical_dataset(
        teacher,
        device,
        episodes=8,
        seed_start=20000,
        max_steps=max_steps,
        frames_cap=frames_cap,
        held_frac=0.5,
    )
    held_elapsed = dataset["held_tensors"]["elapsed"]
    naive_first_cap_ceiling = frames_cap / 60.0
    assert float(held_elapsed.max()) > naive_first_cap_ceiling


# --------------------------------------------------------------------------- #
# B2: identity hashes real content and binds the teacher file SHA
# --------------------------------------------------------------------------- #
def test_dataset_identity_binds_teacher_file_sha_and_reports_unambiguous_counts(tmp_path: Path) -> None:
    teacher_path = _write_tiny_teacher_ckpt(tmp_path)
    teacher = _tiny_teacher(hidden=8)
    device = torch.device("cpu")

    dataset = ret.collect_canonical_dataset(
        teacher,
        device,
        episodes=6,
        seed_start=20000,
        max_steps=20,
        frames_cap=6,
        held_frac=0.34,
        teacher_path=teacher_path,
    )

    assert dataset["teacher_file_sha256"] == file_sha256(teacher_path)
    # episode-level and frame-level counts must be unambiguously distinct.
    assert dataset["n_episodes"] == 6
    assert dataset["n_held_episodes"] + dataset["n_train_episodes"] == 6
    assert dataset["n_held_frames"] > 0
    assert dataset["n_train_frames"] > 0
    assert dataset["n_held_frames"] != dataset["n_held_episodes"]


# --------------------------------------------------------------------------- #
# B3: honest value-bootstrap/truncation knobs + per-update censor logging
# --------------------------------------------------------------------------- #
def test_ppo_hyperparameters_report_no_value_bootstrap_and_truncation_as_terminal() -> None:
    knobs = ret.ppo_hyperparameters()
    assert knobs["value_bootstrap"] is False
    assert knobs["truncation_treated_as_terminal"] is True


def _rollout(rewards: list[float], dones: list[bool], values: list[float]) -> Any:
    from qrokkun_env.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4

    n = len(rewards)
    return ret.Rollout(
        seed=0,
        player=[np.zeros(PLAYER_FEAT_V4, dtype=np.float32) for _ in range(n)],
        bullets=[np.zeros((MAX_BULLETS_V4, BULLET_FEAT_V4), dtype=np.float32) for _ in range(n)],
        pad=[np.ones(MAX_BULLETS_V4, dtype=np.bool_) for _ in range(n)],
        actions=[0] * n,
        log_probs=[0.0] * n,
        values=list(values),
        rewards=list(rewards),
        dones=list(dones),
        elapsed=float(n) / 60.0,
        censored=not dones[-1] if dones else True,
    )


def test_ppo_update_row_logs_rollout_censor_count_and_rate() -> None:
    """Every ``ppo_updates.jsonl`` row must report how many of that update's
    rollouts were censored (timed out without termination)."""
    device = torch.device("cpu")
    net = _tiny_net(seed=1)

    censored_rollout = _rollout(rewards=[0.1] * 5, dones=[False] * 5, values=[0.0] * 5)
    terminal_rollout = _rollout(rewards=[0.1] * 5, dones=[False, False, False, False, True], values=[0.0] * 5)
    rollouts = [censored_rollout, terminal_rollout]

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    metrics = ret.ppo_update(net, opt, rollouts, device)
    row = ret.rollout_censor_stats(rollouts)

    assert row["censor_count"] == 1
    assert row["censor_rate"] == pytest.approx(0.5)
    assert "optimizer_steps" in metrics  # sanity: update still ran


# --------------------------------------------------------------------------- #
# B4: snapshot/final checkpoints are marked experimental/non-production, with
#     full provenance extras
# --------------------------------------------------------------------------- #
def test_pack_player_checkpoint_can_mark_experimental_non_production_round_trip(tmp_path: Path) -> None:
    net = _tiny_net(seed=2)
    ckpt = pack_player_checkpoint(
        net,
        source_tool="tests",
        experimental=True,
        production_compatible=False,
        extra={"probe": 1},
    )
    assert ckpt["experimental"] is True
    assert ckpt["production_compatible"] is False
    assert ckpt["extra"]["probe"] == 1

    path = tmp_path / "exp.pt"
    torch.save(ckpt, path)
    loaded, meta = load_ranked_top_k_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, PlayerRankedTopK)
    assert meta["experimental"] is True
    assert meta["production_compatible"] is False
    assert state_dict_sha256(loaded.state_dict()) == state_dict_sha256(net.state_dict())


@pytest.mark.skipif(not TEACHER_CKPT.exists(), reason="teacher checkpoint not available locally")
def test_ppo_snapshots_are_experimental_non_production_with_full_extras(tmp_path: Path) -> None:
    ckpt, _net = _write_ranked_ckpt(tmp_path)
    out_dir = tmp_path / "out_b4"
    args = ret.apply_mode_defaults(
        ret.build_parser().parse_args(
            [
                "--init-checkpoint",
                str(ckpt),
                "--teacher",
                str(TEACHER_CKPT),
                "--out-dir",
                str(out_dir),
                "--quick",
            ]
        )
    )
    report = ret.run_experiment(args, torch.device("cpu"))
    assert report["control_ran"] is True

    for snap in report["ppo_arm"]["snapshots"]:
        raw = torch.load(snap["checkpoint"], map_location="cpu", weights_only=False)
        assert raw["experimental"] is True
        assert raw["production_compatible"] is False
        extra = raw["extra"]
        assert extra["update"] == snap["update"]
        assert extra["parent_state_dict_sha256"] == report["provenance"]["init_checkpoint"]["state_dict_sha256"]
        assert extra["parent_file_sha256"] == report["provenance"]["init_checkpoint"]["file_sha256"]
        assert extra["dataset_hash"] == report["dataset"]["hash"]
        assert extra["eval_summary"]["mean"] == snap["evaluation"]["mean"]
        assert extra["ppo_knobs"] == report["knobs"]
        assert extra["rollout_seed_window"]
        assert extra["eval_seed_window"] == args.eval_seeds

    raw_final = torch.load(report["ppo_arm"]["final_checkpoint"], map_location="cpu", weights_only=False)
    assert raw_final["experimental"] is True
    assert raw_final["production_compatible"] is False
    assert raw_final["extra"]["update"] == args.updates


# --------------------------------------------------------------------------- #
# caveats
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not TEACHER_CKPT.exists(), reason="teacher checkpoint not available locally")
def test_report_knobs_reflect_actual_quick_run_not_only_production_defaults(tmp_path: Path) -> None:
    ckpt, _net = _write_ranked_ckpt(tmp_path)
    out_dir = tmp_path / "out_knobs"
    args = ret.apply_mode_defaults(
        ret.build_parser().parse_args(
            [
                "--init-checkpoint",
                str(ckpt),
                "--teacher",
                str(TEACHER_CKPT),
                "--out-dir",
                str(out_dir),
                "--quick",
            ]
        )
    )
    report = ret.run_experiment(args, torch.device("cpu"))
    knobs = report["knobs"]
    assert knobs["max_frames_per_episode"] == args.max_frames
    assert knobs["updates"] == args.updates
    assert knobs["episodes_per_update"] == args.episodes_per_update
    assert knobs["eval_seeds"] == args.eval_seeds
    assert knobs["eval_max_steps"] == args.eval_max_steps
    assert knobs["initial_gate_mean_min"] == args.initial_gate_mean_min
    assert knobs["initial_gate_median_min"] == args.initial_gate_median_min
    assert knobs["data_episodes"] == args.data_episodes
    assert knobs["data_max_steps"] == args.data_max_steps
    assert knobs["data_frames_cap"] == args.data_frames_cap
    # these must genuinely differ from the production defaults in quick mode
    assert knobs["updates"] != ret.PPO_UPDATES
    assert knobs["max_frames_per_episode"] != ret.PPO_MAX_FRAMES


def test_rollout_seed_schedule_is_the_single_declared_helper_used_by_run_ppo_arm() -> None:
    """``run_ppo_arm`` must call ``rollout_seed_schedule`` (or pass its args
    through it) rather than reimplementing the seed arithmetic inline, and
    the helper itself must honor a caller-supplied episodes-per-update."""
    assert ret.rollout_seed_schedule(0, episodes_per_update=2) == [ret.PPO_ROLLOUT_SEED_START, ret.PPO_ROLLOUT_SEED_START + 1]
    assert ret.rollout_seed_schedule(1, episodes_per_update=2) == [
        ret.PPO_ROLLOUT_SEED_START + 2,
        ret.PPO_ROLLOUT_SEED_START + 3,
    ]
    src = Path(ret.__file__).read_text()
    run_ppo_arm_src = src[src.index("def run_ppo_arm") : src.index("def run_ppo_arm") + 3000]
    assert "rollout_seed_schedule(" in run_ppo_arm_src


def test_missing_initial_held_agreement_fails_closed_not_zero() -> None:
    """A missing/None held teacher agreement on the reference snapshot must
    raise rather than silently substituting 0.0 (which would corrupt the
    retention agreement-floor math)."""
    initial_snapshot = {
        "update": 0,
        "evaluation": {"mean": 30.0, "median": 30.0},
        "teacher_diagnostics": {"agreement": None},
    }
    assert hasattr(ret, "MissingHeldAgreementError")
    with pytest.raises(ret.MissingHeldAgreementError):
        ret.build_retention_reference(initial_snapshot)


def test_retention_fails_closed_when_intermediate_snapshot_omits_held_agreement() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.70}
    snapshots = [
        {"update": 0, "evaluation": {"mean": 40.0, "median": 40.0}, "teacher_diagnostics": {"agreement": 0.70}},
        {"update": 10, "evaluation": {"mean": 38.0, "median": 38.0}, "teacher_diagnostics": {}},
        {"update": 20, "evaluation": {"mean": 36.0, "median": 36.0}, "teacher_diagnostics": {"agreement": 0.68}},
    ]

    report = ret.evaluate_retention(reference, snapshots)

    assert report["retention_pass"] is False
    assert report["agreement_pass"] is False
    assert report["first_breaking_snapshot"] == 10
    assert "missing held teacher agreement" in report["reason"]


def test_retention_fails_closed_when_final_snapshot_has_none_held_agreement() -> None:
    reference = {"mean": 40.0, "median": 40.0, "held_agreement": 0.70}
    snapshots = [
        {"update": 0, "evaluation": {"mean": 40.0, "median": 40.0}, "teacher_diagnostics": {"agreement": 0.70}},
        {"update": 10, "evaluation": {"mean": 36.0, "median": 36.0}, "teacher_diagnostics": {"agreement": None}},
    ]

    report = ret.evaluate_retention(reference, snapshots)

    assert report["retention_pass"] is False
    assert report["agreement_pass"] is False
    assert report["first_breaking_snapshot"] == 10
    assert report["final_agreement_drop"] is None
    assert "missing held teacher agreement" in report["reason"]


def test_module_documents_pythonpath_invocation() -> None:
    doc = ret.__doc__ or ""
    assert "PYTHONPATH" in doc
