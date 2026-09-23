"""Contracts for the shared Phase 3 launch/provenance reporting module.

Both Phase 3 arms (control and aux) must record the SAME launch, input
identity, artifact and completeness evidence, produced by exactly one shared
implementation so the two arms cannot drift apart.  Everything here is
CPU-only and touches no real training, GPU or NAS path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qrokkun_ai.v1.agents.player_v1 import PlayerV1  # noqa: E402
from qrokkun_ai.v5.agents.player_checkpoints import save_player_checkpoint  # noqa: E402
from qrokkun_ai.v5.agents.player_ranked_topk import PlayerRankedTopK  # noqa: E402
from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention as control  # noqa: E402
from qrokkun_ai.v5.tools import phase3_ranked_ppo_retention_aux as aux  # noqa: E402
from qrokkun_ai.v5.tools import phase3_run_provenance as prov  # noqa: E402

# Both arms must be driven through exactly the same provenance expectations.
ARMS = [pytest.param(control, "control", id="control"), pytest.param(aux, "aux", id="aux")]


def _tiny_net(seed: int = 0, hidden: int = 8, top_k: int = 4) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def _quick_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny production-compatible init checkpoint and a tiny V1 teacher.

    Written once per ``tmp_path`` and then reused: a packed checkpoint
    embeds its creation timestamp, so rewriting it would change its file
    SHA-256 and make a resume look like a different experiment.
    """
    init_path = tmp_path / "init.pt"
    teacher_path = tmp_path / "teacher.pt"
    if not init_path.is_file():
        torch.manual_seed(0)
        save_player_checkpoint(_tiny_net(seed=21), init_path, source_tool="tests")
    if not teacher_path.is_file():
        torch.save({"hidden": 16, "state_dict": PlayerV1(hidden=16).state_dict()}, teacher_path)
    return init_path, teacher_path


def _quick_args(module: Any, tmp_path: Path, out_dir: Path, **overrides: Any) -> argparse.Namespace:
    init_path, teacher_path = _quick_inputs(tmp_path)
    return module.apply_mode_defaults(
        argparse.Namespace(
            init_checkpoint=init_path,
            teacher=teacher_path,
            out_dir=out_dir,
            device="cpu",
            quick=True,
            **overrides,
        )
    )


def _run_quick(module: Any, tmp_path: Path, out_dir: Path, **overrides: Any) -> dict[str, Any]:
    args = _quick_args(module, tmp_path, out_dir, **overrides)
    return module.run_experiment(args, torch.device("cpu"))


# --------------------------------------------------------------------------- #
# 1. argv sanitization: the full command line is recorded, secrets never are
# --------------------------------------------------------------------------- #
def test_sanitize_argv_preserves_ordinary_experiment_flags() -> None:
    argv = [
        "-m", "qrokkun_ai.v5.tools.phase3_ranked_ppo_retention_aux",
        "--init-checkpoint", "/mnt/ro/assets/seed5/seed5.pt",
        "--teacher", "/mnt/ro/assets/teacher.pt",
        "--out-dir", "/mnt/rw/runs/20260920-1724/aux",
        "--max-updates", "10000",
        "--seed", "11",
        "--rollout-seed-start", "130000",
        "--terminal-teacher-diagnostics",
        "--device", "cuda",
    ]
    assert prov.sanitize_argv(argv) == argv


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--api-key", "sk-live-abcdef"], ["--api-key", prov.REDACTED]),
        (["--token=ghp_secretvalue"], [f"--token={prov.REDACTED}"]),
        (["--password", "hunter2"], ["--password", prov.REDACTED]),
        (["--hf-secret", "xyz"], ["--hf-secret", prov.REDACTED]),
        (["--auth-credential=abc"], [f"--auth-credential={prov.REDACTED}"]),
        (["AWS_SECRET_ACCESS_KEY=abc"], [f"AWS_SECRET_ACCESS_KEY={prov.REDACTED}"]),
        (
            ["--endpoint", "https://user:pa55w0rd@example.test/x"],
            ["--endpoint", f"https://user:{prov.REDACTED}@example.test/x"],
        ),
    ],
)
def test_sanitize_argv_redacts_secret_bearing_values(argv: list[str], expected: list[str]) -> None:
    assert prov.sanitize_argv(argv) == expected


def test_sanitize_argv_redaction_is_value_only_and_keeps_flag_position() -> None:
    argv = ["--seed", "11", "--token", "ghp_x", "--device", "cpu"]
    assert prov.sanitize_argv(argv) == ["--seed", "11", "--token", prov.REDACTED, "--device", "cpu"]


def test_sanitize_argv_never_treats_a_following_flag_as_a_secret_value() -> None:
    """A valueless secret-named flag must not swallow the next flag."""
    assert prov.sanitize_argv(["--token", "--device", "cpu"]) == ["--token", "--device", "cpu"]


def test_launch_snapshot_never_contains_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QROKKUN_TEST_SECRET", "super-secret-value")
    snapshot = prov.runtime_snapshot()
    assert "super-secret-value" not in prov.dumps_canonical(snapshot)
    for key in ("env", "environ", "environment", "environment_variables"):
        assert key not in snapshot


# --------------------------------------------------------------------------- #
# 2. runtime/hardware snapshot: complete on CUDA, explicit on a CPU-only host
# --------------------------------------------------------------------------- #
def test_runtime_snapshot_records_interpreter_and_library_identity() -> None:
    import numpy as np
    import platform

    snapshot = prov.runtime_snapshot(device_request="cpu")
    assert snapshot["python_version"] == platform.python_version()
    assert snapshot["numpy_version"] == np.__version__
    assert snapshot["torch_version"] == str(torch.__version__)
    assert snapshot["host"] == platform.node() or isinstance(snapshot["host"], str)
    assert snapshot["device_request"] == "cpu"
    assert snapshot["platform"]["system"] == platform.system()


def test_runtime_snapshot_on_cpu_only_host_reports_explicit_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU/no-CUDA host must produce explicit unavailability, never a crash
    and never a silently missing key."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    snapshot = prov.runtime_snapshot(device_request="cpu")

    cuda = snapshot["cuda"]
    assert cuda["available"] is False
    assert cuda["runtime_version"] is None
    assert cuda["driver_version"] is None
    assert cuda["status"] == prov.EVIDENCE_UNAVAILABLE
    # The build-time CUDA tag is a property of the installed torch wheel and is
    # knowable even with no device present.
    assert "build_version" in cuda

    gpu = snapshot["gpu"]
    assert gpu["count"] == 0
    assert gpu["devices"] == []
    assert gpu["status"] == prov.EVIDENCE_UNAVAILABLE


def test_runtime_snapshot_never_initializes_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CUDA was probed on a host that reports no CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", explode)
    monkeypatch.setattr(torch.cuda, "get_device_properties", explode)
    monkeypatch.setattr(torch.cuda, "get_device_name", explode)

    snapshot = prov.runtime_snapshot(device_request="cuda")
    assert snapshot["device_request"] == "cuda"
    assert snapshot["cuda"]["available"] is False


def test_runtime_snapshot_records_gpu_model_count_and_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CUDA is present the model/count/properties are recorded as observed."""

    class _Props:
        name = "NVIDIA GB10"
        total_memory = 1234567890
        major = 12
        minor = 1
        multi_processor_count = 96

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"NVIDIA GB10 #{index}")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _Props())

    snapshot = prov.runtime_snapshot(device_request="cuda")
    gpu = snapshot["gpu"]
    assert gpu["status"] == prov.EVIDENCE_OBSERVED
    assert gpu["count"] == 2
    assert [d["index"] for d in gpu["devices"]] == [0, 1]
    assert gpu["devices"][1]["name"] == "NVIDIA GB10 #1"
    assert gpu["devices"][0]["total_memory_bytes"] == 1234567890
    assert gpu["devices"][0]["capability"] == "12.1"
    assert gpu["devices"][0]["multi_processor_count"] == 96
    assert snapshot["cuda"]["available"] is True


def test_runtime_snapshot_survives_a_probe_that_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driver query that raises must degrade to explicit unavailability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no NVIDIA driver on this host")

    monkeypatch.setattr(torch.cuda, "device_count", explode)
    snapshot = prov.runtime_snapshot(device_request="cuda")
    assert snapshot["gpu"]["count"] == 0
    assert snapshot["gpu"]["status"] == prov.EVIDENCE_UNAVAILABLE


# --------------------------------------------------------------------------- #
# 3. launch.json: written by BOTH arms before any evaluation/dataset/rollout
# --------------------------------------------------------------------------- #
def test_write_launch_record_captures_argv_cwd_clock_and_runtime(tmp_path: Path) -> None:
    record = prov.write_launch_record(
        tmp_path,
        arm="control",
        tool="phase3_ranked_ppo_retention",
        argv=["-m", "tool", "--token", "ghp_x", "--device", "cpu"],
        device_request="cpu",
        lineage=prov.launch_lineage(resume=False, resume_from_update=None),
        git_commit="abc123",
        git_dirty=False,
    )

    on_disk = json.loads((tmp_path / "launch.json").read_text())
    assert on_disk["schema_version"] == prov.LAUNCH_SCHEMA_VERSION
    assert on_disk["launches"] == [record]

    assert record["argv"] == ["-m", "tool", "--token", prov.REDACTED, "--device", "cpu"]
    assert record["cwd"] == str(Path.cwd().resolve())
    assert Path(record["cwd"]).is_absolute()
    assert record["started_hkt"].endswith("+08:00")
    assert record["started_utc"].endswith("+00:00")
    assert record["runtime"]["device_request"] == "cpu"
    assert record["runtime"]["torch_version"] == str(torch.__version__)
    assert record["git"] == {"commit": "abc123", "dirty": False}
    assert record["lineage"]["mode"] == "fresh"
    assert record["sequence"] == 1
    assert record["executable"] == sys.executable


def test_write_launch_record_appends_lineage_across_launches(tmp_path: Path) -> None:
    prov.write_launch_record(
        tmp_path, arm="aux", tool="t", argv=[], device_request="cpu",
        lineage=prov.launch_lineage(resume=False, resume_from_update=None),
        git_commit=None, git_dirty=None,
    )
    second = prov.write_launch_record(
        tmp_path, arm="aux", tool="t", argv=[], device_request="cpu",
        lineage=prov.launch_lineage(resume=True, resume_from_update=None),
        git_commit=None, git_dirty=None,
    )
    third = prov.write_launch_record(
        tmp_path, arm="aux", tool="t", argv=[], device_request="cpu",
        lineage=prov.launch_lineage(resume=True, resume_from_update=50),
        git_commit=None, git_dirty=None,
    )

    launches = json.loads((tmp_path / "launch.json").read_text())["launches"]
    assert [entry["sequence"] for entry in launches] == [1, 2, 3]
    assert [entry["lineage"]["mode"] for entry in launches] == ["fresh", "resume", "rewind"]
    assert second["lineage"]["resume_from_update"] is None
    assert third["lineage"]["resume_from_update"] == 50
    assert third["lineage"]["previous_launch_sequence"] == 2


def test_launch_record_never_serializes_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QROKKUN_LAUNCH_SECRET", "do-not-persist-me")
    prov.write_launch_record(
        tmp_path, arm="control", tool="t", argv=["--api-key", "do-not-persist-me"],
        device_request="cpu",
        lineage=prov.launch_lineage(resume=False, resume_from_update=None),
        git_commit=None, git_dirty=None,
    )
    text = (tmp_path / "launch.json").read_text()
    assert "do-not-persist-me" not in text
    assert "QROKKUN_LAUNCH_SECRET" not in text


@pytest.mark.parametrize("module, arm", ARMS)
def test_arm_writes_launch_json_before_any_evaluation_dataset_or_rollout(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    seen: list[str] = []

    def _record(name: str, result: Any):
        def _hook(*_args: Any, **_kwargs: Any) -> Any:
            seen.append(name)
            assert (out_dir / "launch.json").is_file(), (
                f"{name} ran before launch.json was written"
            )
            raise _Stop()

        return _hook

    class _Stop(RuntimeError):
        pass

    monkeypatch.setattr(module, "evaluate_deterministic", _record("evaluate", None))
    monkeypatch.setattr(module, "collect_rollout", _record("rollout", None))

    args = _quick_args(module, tmp_path, out_dir)
    with pytest.raises(_Stop):
        module.run_experiment(args, torch.device("cpu"))
    assert seen, "no gated stage ran at all"

    launches = json.loads((out_dir / "launch.json").read_text())["launches"]
    assert len(launches) == 1
    assert launches[0]["arm"] == arm
    assert launches[0]["lineage"]["mode"] == "fresh"


@pytest.mark.parametrize("module, arm", ARMS)
def test_arm_launch_json_records_sanitized_argv_and_resolved_cwd(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    argv = ["-m", f"tool.{arm}", "--device", "cpu", "--quick", "--api-key", "sk-secret"]
    _run_quick(module, tmp_path, out_dir, argv=argv)

    entry = json.loads((out_dir / "launch.json").read_text())["launches"][0]
    assert entry["argv"] == ["-m", f"tool.{arm}", "--device", "cpu", "--quick",
                             "--api-key", prov.REDACTED]
    assert entry["cwd"] == str(Path.cwd().resolve())
    assert entry["tool"] == module.__name__.rsplit(".", 1)[-1]
    assert entry["runtime"]["cuda"]["available"] in (True, False)
    assert entry["runtime"]["gpu"]["status"] in prov.EVIDENCE_STATUSES
    assert "sk-secret" not in (out_dir / "launch.json").read_text()


# --------------------------------------------------------------------------- #
# 4. input_manifest.json: resolved paths + real identity, or fail closed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module, arm", ARMS)
def test_input_manifest_records_resolved_paths_and_actual_identity(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256, state_dict_sha256

    out_dir = tmp_path / "run"
    report = _run_quick(module, tmp_path, out_dir)

    manifest = json.loads((out_dir / "input_manifest.json").read_text())
    assert manifest["schema_version"] == prov.INPUT_MANIFEST_SCHEMA_VERSION
    assert manifest["arm"] == arm
    assert manifest["complete"] is True
    assert manifest["missing"] == []

    init = manifest["inputs"]["init_checkpoint"]
    init_path = tmp_path / "init.pt"
    assert init["path"] == str(init_path.resolve())
    assert Path(init["path"]).is_absolute()
    assert init["file_sha256"] == file_sha256(init_path)
    assert len(init["state_dict_sha256"]) == 64
    assert init["architecture"] == "player_ranked_topk"
    assert init["architecture_version"] == 1
    assert init["checkpoint_schema_version"] == 1
    assert init["production_compatible"] is True
    assert init["provenance"] == prov.EVIDENCE_COMPUTED

    teacher_path = tmp_path / "teacher.pt"
    teacher = manifest["inputs"]["teacher"]
    assert teacher["path"] == str(teacher_path.resolve())
    assert teacher["file_sha256"] == file_sha256(teacher_path)
    raw = torch.load(teacher_path, map_location="cpu", weights_only=True)
    assert teacher["state_dict_sha256"] == state_dict_sha256(raw["state_dict"])
    assert teacher["architecture"] == "player_v1"
    assert teacher["hidden"] == 16

    dataset = manifest["dataset"]
    assert dataset["hash"] == report["dataset"]["hash"]
    assert dataset["n_held_frames"] == report["dataset"]["n_held_frames"]
    assert dataset["provenance"] == prov.EVIDENCE_COMPUTED


@pytest.mark.parametrize("module, arm", ARMS)
def test_input_manifest_resolves_a_relative_input_path_to_an_absolute_one(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    args = _quick_args(module, tmp_path, out_dir)
    monkeypatch.chdir(tmp_path)
    args.init_checkpoint = Path("init.pt")
    args.teacher = Path("teacher.pt")
    module.run_experiment(args, torch.device("cpu"))

    manifest = json.loads((out_dir / "input_manifest.json").read_text())
    for key in ("init_checkpoint", "teacher"):
        recorded = Path(manifest["inputs"][key]["path"])
        assert recorded.is_absolute()
        assert recorded == (tmp_path / f"{'init' if key == 'init_checkpoint' else 'teacher'}.pt").resolve()


@pytest.mark.parametrize("module, arm", ARMS)
def test_input_manifest_is_written_before_the_initial_gate_evaluation(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"

    class _Stop(RuntimeError):
        pass

    def _hook(*_args: Any, **_kwargs: Any) -> Any:
        assert (out_dir / "input_manifest.json").is_file(), (
            "the initial gate evaluation ran before input identity was recorded"
        )
        raise _Stop()

    monkeypatch.setattr(module, "evaluate_deterministic", _hook)
    args = _quick_args(module, tmp_path, out_dir)
    with pytest.raises(_Stop):
        module.run_experiment(args, torch.device("cpu"))


@pytest.mark.parametrize("module, arm", ARMS)
def test_arm_fails_closed_before_any_experiment_work_when_identity_is_unavailable(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required input identity that cannot be produced stops the run before
    a single environment step, and says exactly which field is missing."""
    out_dir = tmp_path / "run"
    args = _quick_args(module, tmp_path, out_dir)
    (tmp_path / "teacher.pt").unlink()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran despite unavailable input identity")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)

    report = module.run_experiment(args, torch.device("cpu"))
    assert report["status"] == "failed_closed"

    manifest = json.loads((out_dir / "input_manifest.json").read_text())
    assert manifest["complete"] is False
    assert "inputs.teacher.file_sha256" in manifest["missing"]
    assert manifest["inputs"]["teacher"]["file_sha256"] is None
    assert manifest["inputs"]["teacher"]["provenance"] == prov.EVIDENCE_UNAVAILABLE
    assert report["input_manifest"]["complete"] is False


@pytest.mark.parametrize("module, arm", ARMS)
def test_run_contract_binds_teacher_state_dict_identity_at_the_bumped_schema(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import state_dict_sha256

    assert control.CURRENT_RUN_SCHEMA_VERSION == 5

    out_dir = tmp_path / "run"
    _run_quick(module, tmp_path, out_dir)
    contract = json.loads((out_dir / "run.json").read_text())
    assert contract["schema_version"] == 5

    raw = torch.load(tmp_path / "teacher.pt", map_location="cpu", weights_only=True)
    assert contract["inputs"]["teacher"]["state_dict_sha256"] == state_dict_sha256(raw["state_dict"])
    assert contract["inputs"]["teacher"]["architecture"] == "player_v1"


@pytest.mark.parametrize("module, arm", ARMS)
def test_resume_refuses_a_previous_schema_version_run_json(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    """No legacy hand-edit compatibility: a schema-4 run.json is refused."""
    out_dir = tmp_path / "run"
    _run_quick(module, tmp_path, out_dir)

    contract = json.loads((out_dir / "run.json").read_text())
    contract["schema_version"] = 4
    (out_dir / "run.json").write_text(json.dumps(contract, indent=2, sort_keys=True))

    args = _quick_args(module, tmp_path, out_dir)
    args.resume = True
    with pytest.raises(control.RunStateError, match="schema_version"):
        module.run_experiment(args, torch.device("cpu"))


# --------------------------------------------------------------------------- #
# 5. every per-update progress row carries the seeds that update really used
# --------------------------------------------------------------------------- #
def test_rollout_seed_record_describes_the_seeds_actually_consumed() -> None:
    record = prov.rollout_seed_record([130000, 130001, 130002, 130003])
    assert record["rollout_seeds"] == [130000, 130001, 130002, 130003]
    assert record["rollout_seed_window"] == {
        "start": 130000, "end_inclusive": 130003, "count": 4,
    }


def test_rollout_seed_record_is_explicit_about_an_empty_window() -> None:
    record = prov.rollout_seed_record([])
    assert record["rollout_seeds"] == []
    assert record["rollout_seed_window"] == {"start": None, "end_inclusive": None, "count": 0}


@pytest.mark.parametrize(
    "module, arm, progress_name",
    [
        pytest.param(control, "control", "ppo_updates.jsonl", id="control"),
        pytest.param(aux, "aux", "ppo_aux_updates.jsonl", id="aux"),
    ],
)
def test_every_progress_row_records_actual_rollout_seeds(
    module: Any, arm: str, progress_name: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    args = _quick_args(module, tmp_path, out_dir, rollout_seed_start=130000)
    module.run_experiment(args, torch.device("cpu"))

    rows = [
        json.loads(line)
        for line in (out_dir / progress_name).read_text().splitlines()
        if line.strip()
    ]
    assert rows, "no progress rows were written"
    for row in rows:
        expected = module.rollout_seed_schedule(
            row["update"] - 1,
            episodes_per_update=args.episodes_per_update,
            rollout_seed_start=130000,
        )
        assert row["rollout_seeds"] == expected
        assert row["rollout_seed_window"] == {
            "start": expected[0], "end_inclusive": expected[-1], "count": len(expected),
        }


@pytest.mark.parametrize("module, arm", ARMS)
def test_progress_rollout_seeds_are_observed_not_reconstructed(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row must report the seeds the rollouts really carry, so a collector
    that deviates from the schedule is visible rather than papered over."""
    real_collect = module.collect_rollout

    def shifted(net: Any, device: Any, seed: int, max_frames: int) -> Any:
        rollout = real_collect(net, device, seed, max_frames=max_frames)
        rollout.seed = int(seed) + 7
        return rollout

    monkeypatch.setattr(module, "collect_rollout", shifted)
    out_dir = tmp_path / "run"
    args = _quick_args(module, tmp_path, out_dir)
    module.run_experiment(args, torch.device("cpu"))

    progress = "ppo_aux_updates.jsonl" if arm == "aux" else "ppo_updates.jsonl"
    rows = [
        json.loads(line)
        for line in (out_dir / progress).read_text().splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        scheduled = module.rollout_seed_schedule(
            row["update"] - 1,
            episodes_per_update=args.episodes_per_update,
            rollout_seed_start=args.rollout_seed_start,
        )
        assert row["rollout_seeds"] == [seed + 7 for seed in scheduled]


# --------------------------------------------------------------------------- #
# 6. alpha calibration: every matched pair persisted, not just the aggregates
# --------------------------------------------------------------------------- #
def _calibration_inputs(n_frames: int = 6, n_teacher: int = 6):
    from qrokkun_ai.v5.agents.obs_v4 import BULLET_FEAT_V4, MAX_BULLETS_V4, PLAYER_FEAT_V4
    from qrokkun_env.env import ACTIONS

    net = _tiny_net(seed=5)
    rollouts = [
        control.Rollout(
            seed=50000,
            player=[torch.randn(PLAYER_FEAT_V4).numpy() for _ in range(n_frames)],
            bullets=[torch.randn(MAX_BULLETS_V4, BULLET_FEAT_V4).numpy() for _ in range(n_frames)],
            pad=[torch.zeros(MAX_BULLETS_V4, dtype=torch.bool).numpy() for _ in range(n_frames)],
            actions=[0] * n_frames,
            log_probs=[-1.0] * n_frames,
            values=[0.0] * n_frames,
            rewards=[1.0] * n_frames,
            dones=[False] * (n_frames - 1) + [True],
            elapsed=1.0,
            censored=False,
        )
    ]
    g = torch.Generator().manual_seed(7)
    train = {
        "player": torch.randn(n_teacher, PLAYER_FEAT_V4, generator=g),
        "bullets": torch.randn(n_teacher, MAX_BULLETS_V4, BULLET_FEAT_V4, generator=g),
        "pad": torch.zeros(n_teacher, MAX_BULLETS_V4, dtype=torch.bool),
        "teacher_logits": torch.randn(n_teacher, len(ACTIONS), generator=g),
    }
    return net, rollouts, train


def test_calibration_persists_every_matched_pair_not_only_aggregates() -> None:
    import statistics as stats

    net, rollouts, train = _calibration_inputs()
    result = aux.calibrate_alpha(
        net, rollouts, train, torch.device("cpu"), minibatch=3, n_minibatches=4,
    )

    pairs = result["pairs"]
    assert len(pairs) == 4
    assert [pair["index"] for pair in pairs] == [0, 1, 2, 3]
    for pair in pairs:
        for key in (
            "g_ppo_norm", "g_ret_norm", "g_ret_weighted_norm",
            "cosine_similarity", "realized_ratio",
            "n_ppo_samples", "n_retention_samples",
        ):
            assert key in pair, key
        assert pair["g_ret_weighted_norm"] == pytest.approx(result["alpha"] * pair["g_ret_norm"])
        assert pair["realized_ratio"] == pytest.approx(
            pair["g_ret_weighted_norm"] / (pair["g_ppo_norm"] + aux._GRAD_EPS)
        )
        assert pair["n_ppo_samples"] == 3

    # The aggregates stay exactly the means of the per-pair values they summarize.
    assert result["g_ppo_norm"] == pytest.approx(stats.fmean(p["g_ppo_norm"] for p in pairs))
    assert result["g_ret_norm"] == pytest.approx(stats.fmean(p["g_ret_norm"] for p in pairs))
    assert result["cosine_similarity"] == pytest.approx(
        stats.fmean(p["cosine_similarity"] for p in pairs)
    )
    assert result["pairs_provenance"] == prov.EVIDENCE_COMPUTED


def test_calibration_per_pair_values_are_not_all_identical() -> None:
    """Distinct matched draws must be reported distinctly, never one value
    repeated ``n`` times to look like a per-pair record."""
    net, rollouts, train = _calibration_inputs(n_frames=12, n_teacher=12)
    result = aux.calibrate_alpha(
        net, rollouts, train, torch.device("cpu"), minibatch=3, n_minibatches=5,
    )
    assert len({round(pair["g_ret_norm"], 10) for pair in result["pairs"]}) > 1


def test_aux_writes_calibration_json_with_full_detail(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    report = _run_quick(aux, tmp_path, out_dir)

    on_disk = json.loads((out_dir / "calibration.json").read_text())
    assert on_disk["arm"] == "aux"
    assert on_disk["status"] == "calibrated"
    # The report carries the same shared wrapper record the file does.
    assert report["alpha_calibration"] == on_disk
    assert on_disk["calibration"]["alpha"] == report["alpha_calibration"]["calibration"]["alpha"]
    assert len(on_disk["calibration"]["pairs"]) == aux.CALIBRATION_GRAD_SAMPLES
    assert report["alpha_calibration"]["calibration"]["pairs"] == on_disk["calibration"]["pairs"]


def test_control_records_calibration_as_not_applicable(tmp_path: Path) -> None:
    """The control has no auxiliary term; it says so explicitly rather than
    leaving the field absent and inviting an inference."""
    out_dir = tmp_path / "run"
    report = _run_quick(control, tmp_path, out_dir)

    on_disk = json.loads((out_dir / "calibration.json").read_text())
    assert on_disk["arm"] == "control"
    assert on_disk["status"] == prov.STATUS_NOT_APPLICABLE
    assert on_disk["calibration"] is None
    assert on_disk["reason"]
    assert report["alpha_calibration"]["status"] == prov.STATUS_NOT_APPLICABLE


# --------------------------------------------------------------------------- #
# 7. every checkpoint/recovery pack states its own kind
# --------------------------------------------------------------------------- #
def test_checkpoint_kind_vocabulary_is_shared_and_complete() -> None:
    assert prov.CHECKPOINT_KINDS == (
        "initial_snapshot",
        "diagnostic_full_snapshot",
        "periodic_model_only",
        "terminal_full_snapshot",
        "final_alias",
        "recovery_current",
        "recovery_archive",
    )


@pytest.mark.parametrize(
    "module, arm, prefix, final_name",
    [
        pytest.param(control, "control", "ppo_update_", "ppo_final.pt", id="control"),
        pytest.param(aux, "aux", "ppo_aux_update_", "ppo_aux_final.pt", id="aux"),
    ],
)
def test_every_written_checkpoint_declares_its_kind_explicitly(
    module: Any, arm: str, prefix: str, final_name: str, tmp_path: Path,
) -> None:
    """No kind is ever inferred from the absence of a field."""
    out_dir = tmp_path / "run"
    _run_quick(module, tmp_path, out_dir)

    kinds: dict[int, str] = {}
    for path in sorted(out_dir.glob(f"{prefix}*.pt")):
        packed = torch.load(path, map_location="cpu", weights_only=True)
        update = int(path.stem[len(prefix):])
        kind = packed["extra"]["checkpoint_kind"]
        assert kind in prov.CHECKPOINT_KINDS
        kinds[update] = kind

    assert kinds, "no update checkpoints were written"
    assert kinds[0] == "initial_snapshot"
    assert all(kind != "initial_snapshot" for update, kind in kinds.items() if update)

    final = torch.load(out_dir / final_name, map_location="cpu", weights_only=True)
    assert final["extra"]["checkpoint_kind"] == "final_alias"


@pytest.mark.parametrize("module, arm", ARMS)
def test_terminal_unscheduled_snapshot_is_labelled_terminal_not_diagnostic(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    # Quick mode schedules diagnostics at {0, 2}; running to update 3 makes
    # the final full snapshot an unscheduled terminal one, which is the shape
    # a long production run has at its terminal update.
    args = _quick_args(
        module, tmp_path, out_dir, max_updates=3, terminal_teacher_diagnostics=True,
    )
    module.run_experiment(args, torch.device("cpu"))

    prefix = "ppo_aux_update_" if arm == "aux" else "ppo_update_"
    packed = torch.load(out_dir / f"{prefix}3.pt", map_location="cpu", weights_only=True)
    assert packed["extra"]["checkpoint_kind"] == "terminal_full_snapshot"
    scheduled = torch.load(out_dir / f"{prefix}2.pt", map_location="cpu", weights_only=True)
    assert scheduled["extra"]["checkpoint_kind"] == "diagnostic_full_snapshot"


@pytest.mark.parametrize("module, arm", ARMS)
def test_recovery_packs_declare_current_versus_archive(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    _run_quick(module, tmp_path, out_dir)

    latest = torch.load(out_dir / "recovery.pt", map_location="cpu", weights_only=False)
    assert latest["checkpoint_kind"] == "recovery_current"

    archive = out_dir / "recovery_archives" / "update_0.pt"
    assert archive.is_file()
    archived = torch.load(archive, map_location="cpu", weights_only=False)
    assert archived["checkpoint_kind"] == "recovery_archive"


def test_rewind_relabels_a_restored_archive_as_the_current_recovery(tmp_path: Path) -> None:
    """A rewind promotes archive N to latest recovery; the promoted copy must
    say it is the current recovery, and the archive must stay an archive."""
    state = {
        "format": 1, "arm": "control", "model": {}, "optimizer": {},
        "completed_update": 0, "total_frames": 0, "optimizer_steps": 0,
        "snapshots": [], "rng": control.capture_rng_state(),
    }
    control.atomic_save_recovery(
        control.archived_recovery_path(tmp_path, 0), state, kind="recovery_archive",
    )
    control.rewind_run_to_archived_recovery(
        tmp_path, 0, progress_filename="ppo_updates.jsonl", device=torch.device("cpu"),
    )
    promoted = torch.load(tmp_path / "recovery.pt", map_location="cpu", weights_only=False)
    assert promoted["checkpoint_kind"] == "recovery_current"
    archived = torch.load(
        control.archived_recovery_path(tmp_path, 0), map_location="cpu", weights_only=False,
    )
    assert archived["checkpoint_kind"] == "recovery_archive"


# --------------------------------------------------------------------------- #
# 8. artifact_manifest.json at finalization
# --------------------------------------------------------------------------- #
def test_artifact_role_classification_is_shared_between_arms() -> None:
    assert prov.artifact_role("report.json") == "report"
    assert prov.artifact_role("launch.json") == "launch"
    assert prov.artifact_role("input_manifest.json") == "input_manifest"
    assert prov.artifact_role("calibration.json") == "calibration"
    assert prov.artifact_role("run.json") == "run_contract"
    assert prov.artifact_role("status.jsonl") == "status_journal"
    assert prov.artifact_role("ppo_updates.jsonl") == "progress_journal"
    assert prov.artifact_role("ppo_aux_updates.jsonl") == "progress_journal"
    assert prov.artifact_role("stop_budget_amendments.jsonl") == "stop_budget_amendments"
    assert prov.artifact_role("ppo_update_200.pt") == "update_checkpoint"
    assert prov.artifact_role("ppo_aux_update_200.pt") == "update_checkpoint"
    assert prov.artifact_role("ppo_final.pt") == "final_checkpoint"
    assert prov.artifact_role("ppo_aux_final.pt") == "final_checkpoint"
    assert prov.artifact_role("recovery.pt") == "recovery_current"
    assert prov.artifact_role("recovery_archives/update_50.pt") == "recovery_archive"


def test_artifact_update_index_is_parsed_only_from_update_indexed_names() -> None:
    assert prov.artifact_update("ppo_update_200.pt") == 200
    assert prov.artifact_update("ppo_aux_update_0.pt") == 0
    assert prov.artifact_update("recovery_archives/update_50.pt") == 50
    assert prov.artifact_update("ppo_final.pt") is None
    assert prov.artifact_update("report.json") is None


def test_artifact_manifest_hashes_small_files_and_skips_recovery_archives(
    tmp_path: Path,
) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    (tmp_path / "report.json").write_text("{}")
    (tmp_path / "ppo_update_0.pt").write_bytes(b"x" * 64)
    archives = tmp_path / "recovery_archives"
    archives.mkdir()
    (archives / "update_0.pt").write_bytes(b"y" * 100)
    (archives / "update_50.pt").write_bytes(b"z" * 200)

    manifest = prov.build_artifact_manifest(tmp_path, arm="control")
    by_path = {entry["path"]: entry for entry in manifest["artifacts"]}

    assert by_path["report.json"]["sha256"] == file_sha256(tmp_path / "report.json")
    assert by_path["report.json"]["sha256_status"] == prov.EVIDENCE_COMPUTED
    assert by_path["report.json"]["role"] == "report"
    assert by_path["ppo_update_0.pt"]["size_bytes"] == 64
    assert by_path["ppo_update_0.pt"]["update"] == 0

    archive_entry = by_path["recovery_archives/update_50.pt"]
    assert archive_entry["sha256"] is None
    assert archive_entry["sha256_status"] == prov.NOT_COMPUTED
    assert archive_entry["not_computed_reason"] == "recovery_archive_hashing_disabled"
    assert archive_entry["size_bytes"] == 200
    assert archive_entry["update"] == 50

    # The archives are still counted and sized exactly, never estimated.
    summary = manifest["recovery_archives"]
    assert summary["count"] == 2
    assert summary["total_size_bytes"] == 300
    assert summary["updates"] == [0, 50]
    assert summary["hashed"] is False


def test_artifact_manifest_can_be_asked_to_hash_recovery_archives(tmp_path: Path) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    archives = tmp_path / "recovery_archives"
    archives.mkdir()
    (archives / "update_0.pt").write_bytes(b"y" * 100)

    manifest = prov.build_artifact_manifest(tmp_path, arm="aux", hash_recovery_archives=True)
    entry = next(e for e in manifest["artifacts"] if e["path"] == "recovery_archives/update_0.pt")
    assert entry["sha256"] == file_sha256(archives / "update_0.pt")
    assert entry["sha256_status"] == prov.EVIDENCE_COMPUTED
    assert manifest["recovery_archives"]["hashed"] is True


def test_artifact_manifest_declines_to_hash_a_file_over_the_size_limit(tmp_path: Path) -> None:
    (tmp_path / "ppo_update_0.pt").write_bytes(b"x" * 4096)
    manifest = prov.build_artifact_manifest(tmp_path, arm="control", max_hash_bytes=1024)
    entry = next(e for e in manifest["artifacts"] if e["path"] == "ppo_update_0.pt")
    assert entry["sha256"] is None
    assert entry["sha256_status"] == prov.NOT_COMPUTED
    assert entry["not_computed_reason"] == "exceeds_max_hash_bytes"
    assert entry["size_bytes"] == 4096


def test_artifact_manifest_paths_are_relative_to_the_run_directory(tmp_path: Path) -> None:
    (tmp_path / "recovery_archives").mkdir()
    (tmp_path / "recovery_archives" / "update_0.pt").write_bytes(b"a")
    (tmp_path / "report.json").write_text("{}")
    manifest = prov.build_artifact_manifest(tmp_path, arm="control")
    for entry in manifest["artifacts"]:
        assert not Path(entry["path"]).is_absolute()
        assert (tmp_path / entry["path"]).is_file()


@pytest.mark.parametrize("module, arm", ARMS)
def test_arm_writes_artifact_manifest_on_normal_finalization(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    out_dir = tmp_path / "run"
    _run_quick(module, tmp_path, out_dir)

    manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
    assert manifest["schema_version"] == prov.ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert manifest["arm"] == arm

    by_path = {entry["path"]: entry for entry in manifest["artifacts"]}
    progress = "ppo_aux_updates.jsonl" if arm == "aux" else "ppo_updates.jsonl"
    for expected in ("launch.json", "input_manifest.json", "run.json",
                     "calibration.json", "status.jsonl", progress):
        assert expected in by_path, expected
        assert by_path[expected]["sha256"] == file_sha256(out_dir / expected)

    final_name = "ppo_aux_final.pt" if arm == "aux" else "ppo_final.pt"
    assert by_path[final_name]["role"] == "final_checkpoint"
    assert by_path["recovery_archives/update_0.pt"]["sha256_status"] == prov.NOT_COMPUTED
    assert manifest["recovery_archives"]["count"] >= 1
    assert manifest["recovery_archives"]["total_size_bytes"] > 0
    # The manifest cannot hash itself; it says so rather than omitting itself.
    assert "artifact_manifest.json" not in by_path


# --------------------------------------------------------------------------- #
# 9. evidence provenance status on every generated reporting record
# --------------------------------------------------------------------------- #
def test_evidence_helper_rejects_a_status_outside_the_vocabulary() -> None:
    assert prov.evidence("x", prov.EVIDENCE_OBSERVED) == {
        "value": "x", "provenance": prov.EVIDENCE_OBSERVED, "note": None,
    }
    assert prov.evidence(None, prov.EVIDENCE_UNAVAILABLE, note="no driver")["note"] == "no driver"
    with pytest.raises(ValueError, match="evidence provenance"):
        prov.evidence("x", "probably")


def test_evidence_unavailable_must_carry_a_null_value() -> None:
    """An 'unavailable' record that still carries a value would let a guess
    masquerade as a measurement."""
    with pytest.raises(ValueError, match="unavailable"):
        prov.evidence(1.23, prov.EVIDENCE_UNAVAILABLE)


@pytest.mark.parametrize("module, arm", ARMS)
def test_report_carries_evidence_provenance_for_its_generated_records(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    report = _run_quick(module, tmp_path, out_dir)

    evidence = report["evidence_provenance"]
    assert set(evidence).issuperset(
        {"launch", "input_manifest", "dataset", "artifact_manifest",
         "progress_journal", "alpha_calibration", "rollout_seed_window"}
    )
    assert set(evidence.values()) <= set(prov.EVIDENCE_STATUSES) | {prov.STATUS_NOT_APPLICABLE}
    assert evidence["launch"] == prov.EVIDENCE_OBSERVED
    assert evidence["input_manifest"] == prov.EVIDENCE_COMPUTED
    assert evidence["artifact_manifest"] == prov.EVIDENCE_COMPUTED
    # Seeds are recorded per update from the rollouts themselves, not rebuilt.
    assert evidence["rollout_seed_window"] == prov.EVIDENCE_OBSERVED
    assert evidence["alpha_calibration"] == (
        prov.EVIDENCE_COMPUTED if arm == "aux" else prov.STATUS_NOT_APPLICABLE
    )

    on_disk = json.loads((out_dir / "report.json").read_text())
    assert on_disk["evidence_provenance"] == evidence
    assert evidence["progress_journal"] == prov.EVIDENCE_OBSERVED
    progress = out_dir / ("ppo_aux_updates.jsonl" if arm == "aux" else "ppo_updates.jsonl")
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert rows
    assert all(row.get("rollout_seeds") for row in rows)


def _assert_progress_evidence_unavailable(report: dict[str, Any]) -> None:
    evidence = report["evidence_provenance"]
    assert evidence["progress_journal"] == prov.EVIDENCE_UNAVAILABLE
    assert evidence["rollout_seed_window"] == prov.EVIDENCE_UNAVAILABLE


def test_evidence_provenance_hides_progress_without_per_update_rows() -> None:
    """A report may name an arm and still have no journal. Update 0, a zero
    completed-update count, and a missing arm are not per-update rows, and
    they are not a seed window either."""
    unavailable_reports = (
        {},
        {"status": "failed_closed", "ppo_arm": None, "aux_arm": None},
        {"ppo_arm": {"completed_updates": 0, "snapshots": [{"update": 0}]}},
        {"aux_arm": {"completed_updates": 0, "snapshots": [{"update": 0}]}},
        {"ppo_arm": {"completed_updates": True}},
        {"aux_arm": {"completed_updates": True}},
    )
    for report in unavailable_reports:
        evidence = prov.report_evidence_provenance(report)
        assert evidence["progress_journal"] == prov.EVIDENCE_UNAVAILABLE
        assert evidence["rollout_seed_window"] == prov.EVIDENCE_UNAVAILABLE

    # A completed-update counter is not the retained journal. Without the
    # file (or an explicit journal to measure), there is nothing to observe.
    for key in ("ppo_arm", "aux_arm"):
        evidence = prov.report_evidence_provenance({key: {"completed_updates": 2}})
        assert evidence["progress_journal"] == prov.EVIDENCE_UNAVAILABLE
        assert evidence["rollout_seed_window"] == prov.EVIDENCE_UNAVAILABLE


@pytest.mark.parametrize("module, arm", ARMS)
def test_failed_identity_and_gate_reports_do_not_claim_progress_evidence(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "identity"
    init_path, teacher_path = _quick_inputs(tmp_path)
    teacher_path.unlink()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran despite unidentifiable inputs")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)
    identity = module.run_experiment(
        module.apply_mode_defaults(
            argparse.Namespace(
                init_checkpoint=init_path, teacher=teacher_path, out_dir=out_dir,
                device="cpu", quick=True,
            )
        ),
        torch.device("cpu"),
    )
    assert identity["status"] == "failed_closed"
    _assert_progress_evidence_unavailable(identity)

    monkeypatch.undo()
    gate_dir = tmp_path / "gate"
    args = _quick_args(module, tmp_path, gate_dir)
    args.initial_gate_mean_min = 1.0
    args.initial_gate_median_min = 1.0
    monkeypatch.setattr(
        module, "evaluate_deterministic",
        lambda *_a, **_k: [{"seed": 0, "elapsed": 0.0, "censored": False}],
    )
    gate = module.run_experiment(args, torch.device("cpu"))
    assert gate["status"] == "failed_closed"
    assert gate["initial_gate"]["gate_pass"] is False
    _assert_progress_evidence_unavailable(gate)


@pytest.mark.parametrize("module, arm", ARMS)
def test_zero_update_report_does_not_claim_progress_evidence(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    report = _run_quick(module, tmp_path, out_dir, max_updates=0)
    arm_key = "aux_arm" if arm == "aux" else "ppo_arm"
    assert report[arm_key]["completed_updates"] == 0
    progress = out_dir / ("ppo_aux_updates.jsonl" if arm == "aux" else "ppo_updates.jsonl")
    rows = progress.read_text().splitlines() if progress.is_file() else []
    assert not any(line.strip() for line in rows)
    _assert_progress_evidence_unavailable(report)


def _progress_filename(arm: str) -> str:
    return "ppo_aux_updates.jsonl" if arm == "aux" else "ppo_updates.jsonl"


def _force_gate_closed(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """Fail the pre-registered gate without touching the immutable knobs."""
    monkeypatch.setattr(
        module,
        "evaluate_initial_gate",
        lambda summary, **kwargs: {
            "gate_pass": False,
            "mean_pass": False,
            "median_pass": False,
            "mean_min": kwargs.get("mean_min", 0.0),
            "median_min": kwargs.get("median_min", 0.0),
            "reason": "forced closed before the arm",
        },
    )


def test_progress_evidence_reads_a_retained_journal_not_the_arm_counter(tmp_path: Path) -> None:
    """A zero completed-update counter must not hide a journal that still
    holds a real update row, and the explicit journal path is measured too."""
    journal = tmp_path / "ppo_updates.jsonl"
    journal.write_text(json.dumps({"update": 1, "rollout_seeds": [10, 11]}) + "\n")
    report = {"ppo_arm": {"completed_updates": 0}, "aux_arm": None}
    evidence = prov.report_evidence_provenance(report, run_dir=tmp_path, arm="control")
    assert evidence["progress_journal"] == prov.EVIDENCE_OBSERVED
    assert evidence["rollout_seed_window"] == prov.EVIDENCE_OBSERVED

    other = tmp_path / "custom.jsonl"
    other.write_text(json.dumps({"update": 3, "rollout_seeds": [4]}) + "\n")
    explicit = prov.report_evidence_provenance({}, progress_journal=other)
    assert explicit["progress_journal"] == prov.EVIDENCE_OBSERVED
    assert explicit["rollout_seed_window"] == prov.EVIDENCE_OBSERVED


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "\n",
        "not-json\n",
        "{}\n",
        json.dumps({"update": True, "rollout_seeds": [1]}) + "\n",
        json.dumps({"update": 0, "rollout_seeds": [1]}) + "\n",
        json.dumps({"update": 1, "rollout_seeds": []}) + "\n",
        json.dumps({"update": 1, "rollout_seeds": [True]}) + "\n",
        json.dumps({"update": 1, "rollout_seeds": [1.5]}) + "\n",
        json.dumps({"update": 1, "rollout_seeds": ["1"]}) + "\n",
        json.dumps({"update": 1}) + "\n",
    ],
)
def test_malformed_or_empty_journal_is_not_observed(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "ppo_updates.jsonl"
    path.write_text(payload)
    evidence = prov.report_evidence_provenance({}, run_dir=tmp_path, arm="control")
    assert evidence["progress_journal"] == prov.EVIDENCE_UNAVAILABLE
    assert evidence["rollout_seed_window"] == prov.EVIDENCE_UNAVAILABLE


def test_one_valid_journal_row_is_observed_beside_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "ppo_aux_updates.jsonl"
    path.write_text(
        "not-json\n"
        + json.dumps({"update": True, "rollout_seeds": [1]}) + "\n"
        + json.dumps({"update": 2, "rollout_seeds": [7, 8]}) + "\n"
    )
    evidence = prov.report_evidence_provenance({}, progress_journal=path)
    assert evidence["progress_journal"] == prov.EVIDENCE_OBSERVED
    assert evidence["rollout_seed_window"] == prov.EVIDENCE_OBSERVED


def test_missing_journal_is_unavailable_for_either_arm(tmp_path: Path) -> None:
    for arm in ("control", "aux"):
        evidence = prov.report_evidence_provenance({}, run_dir=tmp_path, arm=arm)
        assert evidence["progress_journal"] == prov.EVIDENCE_UNAVAILABLE
        assert evidence["rollout_seed_window"] == prov.EVIDENCE_UNAVAILABLE


def test_finalize_hashes_the_same_journal_it_reports(tmp_path: Path) -> None:
    """Evidence and the artifact manifest must describe one retained file."""
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    journal = tmp_path / "ppo_updates.jsonl"
    journal.write_text(json.dumps({"update": 1, "rollout_seeds": [10, 11]}) + "\n")
    report: dict[str, Any] = {"status": "failed_closed", "ppo_arm": None}

    def write_report(run_dir: Path, payload: dict[str, Any]) -> None:
        (run_dir / "report.json").write_text(json.dumps(payload))

    prov.finalize_run_reporting(tmp_path, arm="control", report=report, write_report=write_report)
    assert report["evidence_provenance"]["progress_journal"] == prov.EVIDENCE_OBSERVED
    assert report["evidence_provenance"]["rollout_seed_window"] == prov.EVIDENCE_OBSERVED
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    entry = next(item for item in manifest["artifacts"] if item["path"] == "ppo_updates.jsonl")
    assert entry["role"] == "progress_journal"
    assert entry["sha256"] == file_sha256(journal)


@pytest.mark.parametrize("module, arm", ARMS)
def test_failed_resume_reports_the_retained_journal(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume that dies before the arm still reports the journal already on disk."""
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    out_dir = tmp_path / "run"
    module.run_experiment(
        _quick_args(module, tmp_path, out_dir, max_updates=1), torch.device("cpu"),
    )
    progress = out_dir / _progress_filename(arm)
    retained = progress.read_text()
    assert any(
        json.loads(line).get("update", 0) >= 1
        for line in retained.splitlines() if line.strip()
    )

    _force_gate_closed(monkeypatch, module)
    resumed = _quick_args(module, tmp_path, out_dir, max_updates=2)
    resumed.resume = True
    report = module.run_experiment(resumed, torch.device("cpu"))

    assert report["status"] == "failed_closed"
    arm_key = "aux_arm" if arm == "aux" else "ppo_arm"
    assert report[arm_key] is None
    assert progress.read_text() == retained
    assert report["evidence_provenance"]["progress_journal"] == prov.EVIDENCE_OBSERVED
    assert report["evidence_provenance"]["rollout_seed_window"] == prov.EVIDENCE_OBSERVED
    manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
    entry = next(item for item in manifest["artifacts"] if item["path"] == progress.name)
    assert entry["sha256"] == file_sha256(progress)
    assert entry["role"] == "progress_journal"


@pytest.mark.parametrize("module, arm", ARMS)
def test_failed_rewind_to_zero_and_fresh_failure_leave_progress_unavailable(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewinding away every update>=1 row, or failing a fresh run, is not a seed window."""
    out_dir = tmp_path / "run"
    module.run_experiment(
        _quick_args(module, tmp_path, out_dir, max_updates=1), torch.device("cpu"),
    )
    _force_gate_closed(monkeypatch, module)
    rewound = _quick_args(module, tmp_path, out_dir, max_updates=2, resume_from_update=0)
    report = module.run_experiment(rewound, torch.device("cpu"))
    assert report["status"] == "failed_closed"
    progress = out_dir / _progress_filename(arm)
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert all(not isinstance(row.get("update"), int) or row["update"] < 1 for row in rows)
    _assert_progress_evidence_unavailable(report)
    manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
    entry = next(item for item in manifest["artifacts"] if item["path"] == progress.name)
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256
    assert entry["sha256"] == file_sha256(progress)

    fresh_dir = tmp_path / "fresh"
    fresh = module.run_experiment(
        _quick_args(module, tmp_path, fresh_dir), torch.device("cpu"),
    )
    assert fresh["status"] == "failed_closed"
    fresh_progress = fresh_dir / _progress_filename(arm)
    fresh_rows = (
        fresh_progress.read_text().splitlines() if fresh_progress.is_file() else []
    )
    assert not any(line.strip() for line in fresh_rows)
    _assert_progress_evidence_unavailable(fresh)


@pytest.mark.parametrize("failure", ["identity", "gate"])
@pytest.mark.parametrize("module, arm", ARMS)
def test_control_calibration_is_not_applicable_on_early_failure_and_aux_stays_unavailable(
    failure: str, module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control has no alpha on every path. Aux does not claim one until it calibrates."""
    out_dir = tmp_path / "run"
    if failure == "identity":
        init_path, teacher_path = _quick_inputs(tmp_path)
        teacher_path.unlink()

        def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("experiment work ran despite unidentifiable inputs")

        monkeypatch.setattr(module, "evaluate_deterministic", _boom)
        monkeypatch.setattr(module, "collect_rollout", _boom)
        report = module.run_experiment(
            module.apply_mode_defaults(
                argparse.Namespace(
                    init_checkpoint=init_path, teacher=teacher_path, out_dir=out_dir,
                    device="cpu", quick=True,
                )
            ),
            torch.device("cpu"),
        )
    else:
        args = _quick_args(module, tmp_path, out_dir)
        args.initial_gate_mean_min = 1.0
        args.initial_gate_median_min = 1.0
        monkeypatch.setattr(
            module, "evaluate_deterministic",
            lambda *_a, **_k: [{"seed": 0, "elapsed": 0.0, "censored": False}],
        )
        report = module.run_experiment(args, torch.device("cpu"))

    assert report["status"] == "failed_closed"
    calibration_path = out_dir / prov.CALIBRATION_FILENAME
    if arm == "control":
        expected = prov.calibration_record(arm="control")
        assert report["alpha_calibration"] == expected
        assert json.loads(calibration_path.read_text()) == expected
        assert report["evidence_provenance"]["alpha_calibration"] == prov.STATUS_NOT_APPLICABLE
        manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
        entry = next(
            item for item in manifest["artifacts"] if item["path"] == prov.CALIBRATION_FILENAME
        )
        from qrokkun_ai.v5.agents.player_checkpoints import file_sha256
        assert entry["sha256"] == file_sha256(calibration_path)
    else:
        assert report["alpha_calibration"] is None
        assert report["evidence_provenance"]["alpha_calibration"] == prov.EVIDENCE_UNAVAILABLE
        assert not calibration_path.exists()
        manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
        assert prov.CALIBRATION_FILENAME not in {item["path"] for item in manifest["artifacts"]}


def test_retained_calibration_is_absent_only_when_the_file_is_absent(tmp_path: Path) -> None:
    assert prov.load_retained_calibration_record(tmp_path, arm="aux") is None


def test_retained_calibration_loads_a_valid_wrapper_without_rewriting(tmp_path: Path) -> None:
    record = prov.calibration_record(
        arm="aux", calibration={"alpha": 0.25, "pairs": [{"g": 1}]},
    )
    prov.write_calibration_record(tmp_path, record)
    path = tmp_path / prov.CALIBRATION_FILENAME
    before = path.read_bytes()
    loaded = prov.load_retained_calibration_record(tmp_path, arm="aux")
    assert loaded == record
    assert loaded == json.loads(before)
    assert path.read_bytes() == before


_MALFORMED_CALIBRATION = [
    pytest.param("{not json", id="invalid-json"),
    pytest.param("", id="empty-file"),
    pytest.param("[]", id="top-level-list"),
    pytest.param(
        json.dumps({"status": prov.STATUS_CALIBRATED, "calibration": {"alpha": 0.5}}),
        id="missing-keys",
    ),
    pytest.param(
        json.dumps({
            **prov.calibration_record(arm="aux", calibration={"alpha": 0.5}),
            "extra": True,
        }),
        id="extra-key",
    ),
    pytest.param(
        json.dumps(prov.calibration_record(arm="control")),
        id="arm-mismatch",
    ),
    pytest.param(
        json.dumps({
            "arm": "aux",
            "status": prov.STATUS_CALIBRATED,
            "calibration": {"pairs": []},
            "reason": None,
            "provenance": prov.EVIDENCE_COMPUTED,
        }),
        id="calibrated-without-finite-alpha",
    ),
    pytest.param(
        json.dumps({
            "arm": "aux",
            "status": prov.STATUS_CALIBRATED,
            "calibration": {"alpha": True},
            "reason": None,
            "provenance": prov.EVIDENCE_COMPUTED,
        }),
        id="boolean-alpha",
    ),
]


@pytest.mark.parametrize("payload", _MALFORMED_CALIBRATION)
def test_retained_calibration_refuses_a_malformed_or_mismatched_record(
    payload: str, tmp_path: Path,
) -> None:
    """A damaged or other-arm calibration.json is not 'no calibration'. The
    loader fails and leaves the only copy of the file byte-for-byte intact."""
    path = tmp_path / prov.CALIBRATION_FILENAME
    path.write_text(payload)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="calibration.json"):
        prov.load_retained_calibration_record(tmp_path, arm="aux")
    assert path.read_bytes() == before


def _assert_calibration_report_matches_file(out_dir: Path, report: dict[str, Any]) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    path = out_dir / prov.CALIBRATION_FILENAME
    on_disk = json.loads(path.read_text())
    assert report["alpha_calibration"] == on_disk
    assert json.loads((out_dir / "report.json").read_text())["alpha_calibration"] == on_disk
    assert report["evidence_provenance"]["alpha_calibration"] == prov.EVIDENCE_COMPUTED
    artifacts = json.loads((out_dir / "artifact_manifest.json").read_text())
    entry = next(item for item in artifacts["artifacts"] if item["path"] == prov.CALIBRATION_FILENAME)
    assert entry["sha256"] == file_sha256(path)
    assert entry["sha256_status"] == prov.EVIDENCE_COMPUTED


_REAL_BUILD_INPUT_MANIFEST = prov.build_input_manifest


def _incomplete_input_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    manifest = _REAL_BUILD_INPUT_MANIFEST(*args, **kwargs)
    manifest["inputs"]["teacher"]["architecture"] = None
    missing = list(manifest.get("missing") or [])
    if "inputs.teacher.architecture" not in missing:
        missing.append("inputs.teacher.architecture")
    manifest["missing"] = missing
    manifest["complete"] = False
    return manifest


@pytest.mark.parametrize(
    "resume_overrides",
    [{"resume": True}, {"max_updates": 1, "resume_from_update": 0}],
    ids=["resume", "rewind"],
)
@pytest.mark.parametrize("failure", ["identity", "gate"])
def test_aux_early_finalize_reconciles_a_retained_calibration(
    failure: str, resume_overrides: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed resume or rewind must report the calibration.json that the
    earlier launch already wrote, and must not rewrite that file."""
    source = tmp_path / "source"
    _run_quick(aux, tmp_path, source, max_updates=0)
    out_dir = tmp_path / "run"
    shutil.copytree(source, out_dir)
    calibration_path = out_dir / prov.CALIBRATION_FILENAME
    before = calibration_path.read_bytes()
    assert json.loads(before)["status"] == prov.STATUS_CALIBRATED

    if failure == "identity":
        monkeypatch.setattr(prov, "build_input_manifest", _incomplete_input_manifest)
    else:
        _force_gate_closed(monkeypatch, aux)
    report = aux.run_experiment(
        _quick_args(aux, tmp_path, out_dir, **resume_overrides), torch.device("cpu"),
    )
    assert report["status"] == "failed_closed"
    assert calibration_path.read_bytes() == before
    _assert_calibration_report_matches_file(out_dir, report)


@pytest.mark.parametrize(
    "resume_overrides",
    [{"resume": True}, {"max_updates": 1, "resume_from_update": 0}],
    ids=["resume", "rewind"],
)
@pytest.mark.parametrize("failure", ["identity", "gate"])
def test_aux_early_finalize_refuses_a_malformed_retained_calibration(
    failure: str, resume_overrides: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _run_quick(aux, tmp_path, source, max_updates=0)
    out_dir = tmp_path / "run"
    shutil.copytree(source, out_dir)
    calibration_path = out_dir / prov.CALIBRATION_FILENAME
    report_path = out_dir / "report.json"
    manifest_path = out_dir / "input_manifest.json"
    calibration_path.write_text(json.dumps(prov.calibration_record(arm="control")))
    before_calibration = calibration_path.read_bytes()
    before_report = report_path.read_bytes()
    before_manifest = manifest_path.read_bytes()

    if failure == "identity":
        monkeypatch.setattr(prov, "build_input_manifest", _incomplete_input_manifest)
    else:
        _force_gate_closed(monkeypatch, aux)
    with pytest.raises(ValueError, match="calibration.json"):
        aux.run_experiment(
            _quick_args(aux, tmp_path, out_dir, **resume_overrides), torch.device("cpu"),
        )
    assert calibration_path.read_bytes() == before_calibration
    assert report_path.read_bytes() == before_report
    assert manifest_path.read_bytes() == before_manifest


def _dataset_manifest(digest: str, provenance: str) -> dict[str, Any]:
    return {
        "inputs": {
            "init_checkpoint": {
                "file_sha256": "init-file",
                "state_dict_sha256": "init-state",
                "architecture": "player_v1",
            },
            "teacher": {
                "file_sha256": "teacher-file",
                "state_dict_sha256": "teacher-state",
                "architecture": "player_v1",
            },
        },
        "dataset": {"hash": digest, "provenance": provenance, "n_episodes": 1},
    }


@pytest.mark.parametrize("provenance", [prov.EVIDENCE_COMPUTED, prov.EVIDENCE_OBSERVED])
def test_attach_rejects_a_drifted_staged_dataset_hash_before_writing(
    provenance: str, tmp_path: Path,
) -> None:
    prior = _dataset_manifest("abc", provenance)
    prov.write_input_manifest(tmp_path, prior)
    path = tmp_path / "input_manifest.json"
    before = path.read_bytes()
    staged = prov.stage_input_manifest(
        tmp_path,
        {"inputs": prior["inputs"], "dataset": {"status": prov.EVIDENCE_UNAVAILABLE}},
        resume=True,
    )
    assert staged["dataset"]["hash"] == "abc"
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="dataset hash"):
        prov.attach_dataset_identity(tmp_path, staged, {"hash": "drifted", "n_episodes": 1})
    assert path.read_bytes() == before
    assert staged["dataset"]["hash"] == "abc"


@pytest.mark.parametrize("provenance", [prov.EVIDENCE_COMPUTED, prov.EVIDENCE_OBSERVED])
def test_attach_records_an_equal_recollected_dataset_hash(
    provenance: str, tmp_path: Path,
) -> None:
    prior = _dataset_manifest("abc", provenance)
    prov.write_input_manifest(tmp_path, prior)
    staged = prov.stage_input_manifest(
        tmp_path,
        {"inputs": prior["inputs"], "dataset": {"status": prov.EVIDENCE_UNAVAILABLE}},
        resume=True,
    )
    written = prov.attach_dataset_identity(
        tmp_path, staged, {"hash": "abc", "n_episodes": 2},
    )
    assert written["dataset"]["hash"] == "abc"
    assert written["dataset"]["n_episodes"] == 2
    assert written["dataset"]["provenance"] == prov.EVIDENCE_COMPUTED
    assert json.loads((tmp_path / "input_manifest.json").read_text())["dataset"]["hash"] == "abc"


def test_attach_still_replaces_a_dataset_hash_that_was_not_observed_or_computed(
    tmp_path: Path,
) -> None:
    prior = _dataset_manifest("abc", prov.EVIDENCE_UNAVAILABLE)
    prov.write_input_manifest(tmp_path, prior)
    staged = prov.stage_input_manifest(
        tmp_path,
        {"inputs": prior["inputs"], "dataset": {"status": prov.EVIDENCE_UNAVAILABLE}},
        resume=True,
    )
    written = prov.attach_dataset_identity(tmp_path, staged, {"hash": "replacement"})
    assert written["dataset"]["hash"] == "replacement"


@pytest.mark.parametrize(
    "alpha",
    [float("nan"), float("inf"), float("-inf"), True, False, "0.5", None, [0.5]],
)
def test_non_finite_or_nonnumeric_alpha_is_not_a_calibration(alpha: Any) -> None:
    """A bad alpha must not be stored as calibrated, and must not be reported computed."""
    record = prov.calibration_record(arm="aux", calibration={"alpha": alpha, "pairs": [{"g": 1}]})
    assert record["status"] != prov.STATUS_CALIBRATED
    assert record["status"] == prov.EVIDENCE_UNAVAILABLE
    assert record["calibration"] is None
    assert record["reason"]
    assert record["provenance"] == prov.EVIDENCE_UNAVAILABLE
    evidence = prov.report_evidence_provenance({"alpha_calibration": record})
    assert evidence["alpha_calibration"] == prov.EVIDENCE_UNAVAILABLE

    forged = {
        "arm": "aux",
        "status": prov.STATUS_CALIBRATED,
        "calibration": {"alpha": alpha},
        "reason": None,
        "provenance": prov.EVIDENCE_COMPUTED,
    }
    forged_evidence = prov.report_evidence_provenance({"alpha_calibration": forged})
    assert forged_evidence["alpha_calibration"] == prov.EVIDENCE_UNAVAILABLE


def test_finite_numeric_alpha_stays_calibrated() -> None:
    for alpha in (0, 0.0, 1, 0.5):
        record = prov.calibration_record(arm="aux", calibration={"alpha": alpha})
        assert record["status"] == prov.STATUS_CALIBRATED
        assert record["calibration"]["alpha"] == alpha
        assert record["provenance"] == prov.EVIDENCE_COMPUTED
        evidence = prov.report_evidence_provenance({"alpha_calibration": record})
        assert evidence["alpha_calibration"] == prov.EVIDENCE_COMPUTED


# --------------------------------------------------------------------------- #
# 10. report_completeness: machine readable, and never overclaims
# --------------------------------------------------------------------------- #
def test_report_completeness_lists_every_required_field_with_a_verdict() -> None:
    verdict = prov.evaluate_report_completeness({}, arm="control")
    assert verdict["complete"] is False
    assert {result["field"] for result in verdict["results"]} == set(prov.REQUIRED_REPORT_FIELDS)
    assert all(result["present"] is False for result in verdict["results"])
    assert verdict["missing"] == sorted(prov.REQUIRED_REPORT_FIELDS)


def test_report_completeness_treats_an_explicit_null_as_absent() -> None:
    verdict = prov.evaluate_report_completeness({"provenance": None}, arm="control")
    result = next(r for r in verdict["results"] if r["field"] == "provenance")
    assert result["present"] is False


@pytest.mark.parametrize("module, arm", ARMS)
def test_completed_run_reports_itself_complete(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    report = _run_quick(module, tmp_path, out_dir)

    completeness = report["report_completeness"]
    assert completeness["schema_version"] == prov.REPORT_COMPLETENESS_SCHEMA_VERSION
    assert completeness["missing"] == []
    assert completeness["complete"] is True
    assert all(result["present"] for result in completeness["results"])


@pytest.mark.parametrize("module, arm", ARMS)
def test_missing_provenance_makes_the_report_incomplete_without_invalidating_the_model(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provenance gap must never be reported as a complete report, and must
    never be reported as a failed training run either."""
    out_dir = tmp_path / "run"

    real = prov.runtime_snapshot
    monkeypatch.setattr(
        prov, "runtime_snapshot",
        lambda **kwargs: {**real(**kwargs), "host": None, "python_version": None},
    )
    report = _run_quick(module, tmp_path, out_dir)

    completeness = report["report_completeness"]
    assert completeness["complete"] is False
    assert "launch.runtime.host" in completeness["missing"]
    assert "launch.runtime.python_version" in completeness["missing"]

    # The model side of the run is untouched: it completed, and says so.
    assert report["status"] == "completed"
    assert report[("arm_ran" if arm == "aux" else "control_ran")] is True
    assert completeness["model_completion_affected"] is False
    assert json.loads((out_dir / "report.json").read_text())["report_completeness"]["complete"] is False


# --------------------------------------------------------------------------- #
# 11. relaunch lineage: resume and rewind keep the full launch history
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module, arm", ARMS)
def test_resume_appends_a_second_launch_record_without_losing_the_first(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    first = _quick_args(module, tmp_path, out_dir, max_updates=1, argv=["-m", "tool", "--first"])
    module.run_experiment(first, torch.device("cpu"))
    first_launch = json.loads((out_dir / "launch.json").read_text())["launches"][0]

    second = _quick_args(
        module, tmp_path, out_dir, max_updates=2, argv=["-m", "tool", "--second"],
    )
    second.resume = True
    module.run_experiment(second, torch.device("cpu"))

    launches = json.loads((out_dir / "launch.json").read_text())["launches"]
    assert [entry["lineage"]["mode"] for entry in launches] == ["fresh", "resume"]
    assert launches[0] == first_launch
    assert launches[1]["argv"] == ["-m", "tool", "--second"]
    assert launches[1]["lineage"]["previous_launch_sequence"] == 1


@pytest.mark.parametrize("module, arm", ARMS)
def test_rewind_records_a_rewind_lineage_with_its_selected_update(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    module.run_experiment(
        _quick_args(module, tmp_path, out_dir, max_updates=1), torch.device("cpu"),
    )

    rewound = _quick_args(
        module, tmp_path, out_dir, max_updates=2, resume_from_update=0,
    )
    module.run_experiment(rewound, torch.device("cpu"))

    launches = json.loads((out_dir / "launch.json").read_text())["launches"]
    assert [entry["lineage"]["mode"] for entry in launches] == ["fresh", "rewind"]
    assert launches[1]["lineage"]["resume_from_update"] == 0


@pytest.mark.parametrize("module, arm", ARMS)
def test_a_resumed_run_still_finalizes_a_complete_report_and_manifest(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    module.run_experiment(
        _quick_args(module, tmp_path, out_dir, max_updates=1), torch.device("cpu"),
    )
    resumed = _quick_args(module, tmp_path, out_dir, max_updates=2)
    resumed.resume = True
    report = module.run_experiment(resumed, torch.device("cpu"))

    assert report["report_completeness"]["complete"] is True
    manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
    by_path = {entry["path"]: entry for entry in manifest["artifacts"]}
    assert "launch.json" in by_path and "calibration.json" in by_path
    # The calibration record survives a resume in both arms.
    calibration = json.loads((out_dir / "calibration.json").read_text())
    assert calibration["arm"] == arm
    assert calibration["status"] in (prov.STATUS_NOT_APPLICABLE, prov.STATUS_CALIBRATED)
