"""Independent-review blockers for the shared Phase 3 reporting/provenance path.

Everything here is CPU-only, touches no real training budget, no GPU and no
NAS path.  The contracts asserted are the ones an independent reader of a
finished run depends on:

* a FRESH run of either arm creates no run directory state before its inputs
  are identified and its pre-registered initial gate has passed, while a
  resume/rewind still validates its immutable contract first;
* an existing ``launch.json`` that cannot be read as a launch history fails
  closed without a single byte of it being rewritten;
* the teacher identity recorded as ``computed`` really loaded into
  ``PlayerV1``;
* both arms report their alpha calibration in the SAME shape;
* a completed report is only called complete when it carries its gate and
  retention verdicts;
* a progress row names the seed range it consumed unambiguously.
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

# Both arms must satisfy every contract here: the point of the shared module is
# that the control and the auxiliary arm cannot drift apart.
ARMS = [pytest.param(control, "control", id="control"), pytest.param(aux, "aux", id="aux")]

TEACHER_HIDDEN = 16


class _Stop(RuntimeError):
    """Raised by a probe to stop a run at the exact stage under test."""


def _tiny_net(seed: int = 0, hidden: int = 8, top_k: int = 4) -> PlayerRankedTopK:
    torch.manual_seed(seed)
    return PlayerRankedTopK(top_k=top_k, hidden=hidden)


def _write_teacher(path: Path, *, hidden: int = TEACHER_HIDDEN) -> Path:
    torch.save({"hidden": hidden, "state_dict": PlayerV1(hidden=hidden).state_dict()}, path)
    return path


def _quick_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny production-compatible init checkpoint and a tiny V1 teacher.

    Written once per ``tmp_path``: a packed checkpoint embeds its creation
    timestamp, so rewriting it would change its file SHA-256 and make a
    resume look like a different experiment.
    """
    init_path, teacher_path = tmp_path / "init.pt", tmp_path / "teacher.pt"
    if not init_path.is_file():
        save_player_checkpoint(_tiny_net(seed=21), init_path, source_tool="tests")
    if not teacher_path.is_file():
        _write_teacher(teacher_path)
    return init_path, teacher_path


def _quick_args(
    module: Any, tmp_path: Path, out_dir: Path, **overrides: Any
) -> argparse.Namespace:
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
    return module.run_experiment(
        _quick_args(module, tmp_path, out_dir, **overrides), torch.device("cpu")
    )


# --------------------------------------------------------------------------- #
# 1. a fresh run creates no run state before identity and the initial gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module, arm", ARMS)
def test_fresh_run_creates_no_run_json_or_status_before_the_initial_gate(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run.json`` locks experiment identity and ``status.jsonl`` claims a
    started run.  Neither may exist before the inputs have been identified and
    the pre-registered gate has passed, or a run that never legitimately began
    leaves a locked directory and a started history behind."""
    out_dir = tmp_path / "run"

    def _gate(*_args: Any, **_kwargs: Any) -> Any:
        assert (out_dir / "input_manifest.json").is_file(), (
            "the initial gate ran before input identity was recorded"
        )
        assert not (out_dir / "run.json").exists(), (
            "run.json was created before the initial gate had passed"
        )
        assert not (out_dir / "status.jsonl").exists(), (
            "status.jsonl claimed a started run before the initial gate had passed"
        )
        raise _Stop()

    monkeypatch.setattr(module, "evaluate_deterministic", _gate)
    with pytest.raises(_Stop):
        module.run_experiment(_quick_args(module, tmp_path, out_dir), torch.device("cpu"))


@pytest.mark.parametrize("module, arm", ARMS)
def test_fresh_run_that_fails_its_initial_gate_leaves_no_run_json(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    args = _quick_args(module, tmp_path, out_dir)
    # Quick mode relaxes the gate to 0.0; a strictly positive floor with a
    # zero-survival evaluation fails it without touching the gate math.
    args.initial_gate_mean_min = 1.0
    args.initial_gate_median_min = 1.0
    monkeypatch.setattr(
        module, "evaluate_deterministic",
        lambda *_a, **_k: [{"seed": 0, "elapsed": 0.0, "censored": False}],
    )

    report = module.run_experiment(args, torch.device("cpu"))
    assert report["status"] == "failed_closed"
    assert report["initial_gate"]["gate_pass"] is False
    assert not (out_dir / "run.json").exists()
    assert not (out_dir / "status.jsonl").exists()


@pytest.mark.parametrize("module, arm", ARMS)
def test_missing_teacher_leaves_no_run_json_and_a_retry_in_the_same_dir_succeeds(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression a locked-too-early run directory causes: a first launch
    whose teacher is absent fails closed on identity, and the operator's retry
    in the SAME directory must still be a fresh run rather than being refused
    for an already-present run.json."""
    out_dir = tmp_path / "run"
    init_path, teacher_path = _quick_inputs(tmp_path)
    teacher_path.unlink()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran despite unidentifiable inputs")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)
    first = module.run_experiment(
        module.apply_mode_defaults(
            argparse.Namespace(
                init_checkpoint=init_path, teacher=teacher_path, out_dir=out_dir,
                device="cpu", quick=True,
            )
        ),
        torch.device("cpu"),
    )
    assert first["status"] == "failed_closed"
    assert first["input_manifest"]["complete"] is False
    assert not (out_dir / "run.json").exists()
    assert not (out_dir / "status.jsonl").exists()

    monkeypatch.undo()
    _write_teacher(teacher_path)
    second = _run_quick(module, tmp_path, out_dir)
    assert second["status"] == "completed"
    assert second[("arm_ran" if arm == "aux" else "control_ran")] is True
    assert (out_dir / "run.json").is_file()
    # Both launches are still on record; only the second one locked identity.
    launches = json.loads((out_dir / "launch.json").read_text())["launches"]
    assert [entry["lineage"]["mode"] for entry in launches] == ["fresh", "fresh"]


_REPORTING_FILES = ("input_manifest.json", "report.json", "artifact_manifest.json")


@pytest.mark.parametrize("module, arm", ARMS)
@pytest.mark.parametrize("missing_teacher", [False, True], ids=["valid", "missing-teacher"])
def test_fresh_reentry_with_existing_run_json_leaves_reporting_files_byte_identical(
    module: Any, arm: str, missing_teacher: bool, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-resume, non-rewind launch into a directory that already locked
    ``run.json`` must fail before it rewrites the input manifest, the report,
    or the artifact manifest.  ``launch.json`` may still append.  A first
    failed-closed launch that never wrote ``run.json`` stays retryable and is
    covered separately."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "run.json").write_bytes(b'{"sentinel":"run"}\n')
    before = {
        name: f"sentinel-{arm}-{name}\n".encode() for name in _REPORTING_FILES
    }
    for name, payload in before.items():
        (out_dir / name).write_bytes(payload)

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran against an already locked run directory")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)

    args = _quick_args(module, tmp_path, out_dir)
    if missing_teacher:
        Path(args.teacher).unlink()
    with pytest.raises(control.RunStateError, match="run.json"):
        module.run_experiment(args, torch.device("cpu"))
    for name, payload in before.items():
        assert (out_dir / name).read_bytes() == payload


@pytest.mark.parametrize("module, arm", ARMS)
def test_resume_validates_its_immutable_contract_before_any_experiment_work(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume is tied to an existing contract, so it is validated FIRST --
    the fresh-run ordering must not weaken that."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran before the resume contract was validated")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)

    args = _quick_args(module, tmp_path, out_dir)
    args.resume = True
    with pytest.raises(control.RunStateError, match="run.json is missing"):
        module.run_experiment(args, torch.device("cpu"))


@pytest.mark.parametrize("module, arm", ARMS)
def test_rewind_validates_its_immutable_contract_before_any_experiment_work(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran before the rewind contract was validated")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)

    args = _quick_args(module, tmp_path, out_dir, max_updates=50, resume_from_update=0)
    with pytest.raises(control.RunStateError, match="run.json is missing"):
        module.run_experiment(args, torch.device("cpu"))


def _dataset_hash(report: dict[str, Any]) -> str:
    digest = (report.get("dataset") or {}).get("hash")
    assert isinstance(digest, str) and digest
    return digest


def _assert_reporting_files_agree(out_dir: Path, *, dataset_hash: str) -> None:
    from qrokkun_ai.v5.agents.player_checkpoints import file_sha256

    report = json.loads((out_dir / "report.json").read_text())
    manifest = json.loads((out_dir / "input_manifest.json").read_text())
    artifacts = json.loads((out_dir / "artifact_manifest.json").read_text())
    assert report["input_manifest"] == manifest
    assert report["dataset"]["hash"] == dataset_hash
    assert manifest["dataset"]["hash"] == dataset_hash
    by_path = {entry["path"]: entry for entry in artifacts["artifacts"]}
    for name in ("report.json", "input_manifest.json"):
        assert by_path[name]["sha256"] == file_sha256(out_dir / name)
        assert by_path[name]["sha256_status"] == prov.EVIDENCE_COMPUTED


@pytest.mark.parametrize("module, arm", ARMS)
@pytest.mark.parametrize(
    "resume_overrides",
    [{"resume": True}, {"max_updates": 2, "resume_from_update": 0}],
    ids=["resume", "rewind"],
)
def test_resume_evaluation_exception_preserves_finished_input_manifest(
    module: Any, arm: str, resume_overrides: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished run's dataset identity stays on disk if resume or rewind
    dies while the initial gate is evaluating. Contract checks and that
    exception must not rewrite ``input_manifest.json`` to dataset unavailable."""
    out_dir = tmp_path / "run"
    finished = _run_quick(module, tmp_path, out_dir)
    dataset_hash = _dataset_hash(finished)
    _assert_reporting_files_agree(out_dir, dataset_hash=dataset_hash)
    before = {name: (out_dir / name).read_bytes() for name in _REPORTING_FILES}

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("evaluate_deterministic failed")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    args = _quick_args(module, tmp_path, out_dir, **resume_overrides)
    with pytest.raises(RuntimeError, match="evaluate_deterministic failed"):
        module.run_experiment(args, torch.device("cpu"))
    for name, payload in before.items():
        assert (out_dir / name).read_bytes() == payload


@pytest.mark.parametrize("module, arm", ARMS)
def test_resume_gate_failure_finalizes_the_carried_dataset(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intentional failed resume report may refresh the manifest, and the
    dataset it already knew must still agree across the three reporting files."""
    out_dir = tmp_path / "run"
    finished = _run_quick(module, tmp_path, out_dir)
    dataset_hash = _dataset_hash(finished)

    # The locked contract pins the gate floors, so the failure has to come
    # from the verdict itself. Changing the floors would be a different run.
    real_gate = module.evaluate_initial_gate

    def _fail_gate(summary: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        verdict = real_gate(summary, **kwargs)
        verdict["gate_pass"] = False
        verdict["reason"] = "initial gate failed: forced"
        return verdict

    args = _quick_args(module, tmp_path, out_dir, resume=True)
    monkeypatch.setattr(module, "evaluate_initial_gate", _fail_gate)
    report = module.run_experiment(args, torch.device("cpu"))
    assert report["status"] == "failed_closed"
    assert report["initial_gate"]["gate_pass"] is False
    _assert_reporting_files_agree(out_dir, dataset_hash=dataset_hash)


@pytest.mark.parametrize("module, arm", ARMS)
def test_successful_resume_refreshes_a_complete_consistent_manifest(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    out_dir = tmp_path / "run"
    finished = _run_quick(module, tmp_path, out_dir)
    dataset_hash = _dataset_hash(finished)

    resumed = _run_quick(module, tmp_path, out_dir, resume=True)
    assert resumed["report_completeness"]["complete"] is True
    assert resumed["input_manifest"]["complete"] is True
    assert _dataset_hash(resumed) == dataset_hash
    _assert_reporting_files_agree(out_dir, dataset_hash=dataset_hash)
    on_disk = json.loads((out_dir / "report.json").read_text())
    assert on_disk["evidence_provenance"] == resumed["evidence_provenance"]
    assert on_disk["report_completeness"]["complete"] is True


def _recollected_dataset(digest: str) -> dict[str, Any]:
    return {
        "hash": digest,
        "n_episodes": 1,
        "n_held_episodes": 1,
        "n_train_episodes": 1,
        "n_held_frames": 1,
        "n_train_frames": 1,
        "teacher_file_sha256": "teacher",
        "held_tensors": {},
        "train_tensors": {},
    }


def _model_checkpoint_bytes(out_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(out_dir).as_posix(): path.read_bytes()
        for path in out_dir.rglob("*.pt")
        if path.name != "recovery.pt" and "recovery_archives" not in path.parts
    }


@pytest.mark.parametrize("module, arm", ARMS)
@pytest.mark.parametrize(
    "resume_overrides",
    [{"resume": True}, {"max_updates": 1, "resume_from_update": 0}],
    ids=["resume", "rewind"],
)
def test_resume_and_rewind_accept_an_equal_recollected_dataset_hash(
    module: Any, arm: str, resume_overrides: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recollecting the staged hash is the resume path. Training may start,
    and the manifest keeps that hash."""
    source = tmp_path / "source"
    finished = _run_quick(module, tmp_path, source, max_updates=0)
    digest = _dataset_hash(finished)
    out_dir = tmp_path / "run"
    shutil.copytree(source, out_dir)
    collect_name = "collect_aux_dataset" if arm == "aux" else "collect_canonical_dataset"
    train_name = "run_aux_arm" if arm == "aux" else "run_frozen_arm"
    entered = {"train": False}

    def _collect(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _recollected_dataset(digest)

    def _train(*_args: Any, **_kwargs: Any) -> Any:
        entered["train"] = True
        raise _Stop()

    monkeypatch.setattr(module, collect_name, _collect)
    monkeypatch.setattr(module, train_name, _train)
    with pytest.raises(_Stop):
        module.run_experiment(
            _quick_args(module, tmp_path, out_dir, **resume_overrides), torch.device("cpu"),
        )
    assert entered["train"] is True
    manifest = json.loads((out_dir / "input_manifest.json").read_text())
    assert manifest["dataset"]["hash"] == digest
    assert manifest["dataset"]["provenance"] == prov.EVIDENCE_COMPUTED


@pytest.mark.parametrize("module, arm", ARMS)
@pytest.mark.parametrize(
    "resume_overrides",
    [{"resume": True}, {"max_updates": 1, "resume_from_update": 0}],
    ids=["resume", "rewind"],
)
def test_resume_and_rewind_refuse_a_drifted_dataset_hash_before_training(
    module: Any, arm: str, resume_overrides: dict[str, Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recollected hash that is not the staged one must not refresh
    input_manifest.json and must not start training or write checkpoints."""
    source = tmp_path / "source"
    finished = _run_quick(module, tmp_path, source, max_updates=0)
    digest = _dataset_hash(finished)
    out_dir = tmp_path / "run"
    shutil.copytree(source, out_dir)
    manifest_path = out_dir / "input_manifest.json"
    before_manifest = manifest_path.read_bytes()
    before_checkpoints = _model_checkpoint_bytes(out_dir)
    collect_name = "collect_aux_dataset" if arm == "aux" else "collect_canonical_dataset"

    def _collect(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return _recollected_dataset(digest + "-drift")

    def _forbid(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("training started after a dataset hash drift")

    monkeypatch.setattr(module, collect_name, _collect)
    for name in ("run_aux_arm", "run_frozen_arm", "run_ppo_arm"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _forbid)
    with pytest.raises(ValueError, match="dataset hash"):
        module.run_experiment(
            _quick_args(module, tmp_path, out_dir, **resume_overrides), torch.device("cpu"),
        )
    assert manifest_path.read_bytes() == before_manifest
    assert _model_checkpoint_bytes(out_dir) == before_checkpoints
    assert json.loads(before_manifest)["dataset"]["hash"] == digest


# --------------------------------------------------------------------------- #
# 2. an unreadable launch history fails closed, and its bytes survive intact
# --------------------------------------------------------------------------- #
def _valid_launch_payload() -> dict[str, Any]:
    return {
        "schema_version": prov.LAUNCH_SCHEMA_VERSION,
        "launches": [{"sequence": 1, "arm": "control", "tool": "t"}],
    }


# Every shape here is a launch.json that cannot be read as a launch history.
# Silently treating any of them as "no launches yet" would renumber the next
# record to sequence 1 and overwrite the only copy of the real history.
MALFORMED_LAUNCH_PAYLOADS = [
    pytest.param("{not json", id="invalid-json"),
    pytest.param("", id="empty-file"),
    pytest.param(json.dumps([{"sequence": 1}]), id="top-level-list"),
    pytest.param(json.dumps("launches"), id="top-level-string"),
    pytest.param(json.dumps({"launches": [{"sequence": 1}]}), id="missing-schema-version"),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION + 1,
                    "launches": [{"sequence": 1}]}),
        id="unsupported-schema-version",
    ),
    pytest.param(
        json.dumps({"schema_version": True, "launches": [{"sequence": 1}]}),
        id="schema-version-bool",
    ),
    pytest.param(
        json.dumps({"schema_version": 1.0, "launches": [{"sequence": 1}]}),
        id="schema-version-float",
    ),
    pytest.param(json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION}), id="no-launches-key"),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION, "launches": {"1": {}}}),
        id="launches-not-a-list",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION, "launches": ["first"]}),
        id="entry-not-an-object",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION, "launches": [{"arm": "control"}]}),
        id="entry-without-sequence",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION, "launches": [{"sequence": "1"}]}),
        id="entry-sequence-not-an-int",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION, "launches": [{"sequence": True}]}),
        id="entry-sequence-is-a-bool",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION, "launches": [{"sequence": 0}]}),
        id="entry-sequence-below-one",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION,
                    "launches": [{"sequence": 1}, {"sequence": 3}]}),
        id="noncontiguous-sequence",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION,
                    "launches": [{"sequence": 1}, {"sequence": 1}]}),
        id="duplicate-sequence",
    ),
    pytest.param(
        json.dumps({"schema_version": prov.LAUNCH_SCHEMA_VERSION,
                    "launches": [{"sequence": 2}, {"sequence": 1}]}),
        id="out-of-order-sequence",
    ),
]


@pytest.mark.parametrize("payload", MALFORMED_LAUNCH_PAYLOADS)
def test_reading_a_malformed_launch_history_fails_closed(payload: str, tmp_path: Path) -> None:
    (tmp_path / prov.LAUNCH_FILENAME).write_text(payload)
    with pytest.raises(prov.LaunchRecordError):
        prov.read_launch_records(tmp_path)


@pytest.mark.parametrize("payload", MALFORMED_LAUNCH_PAYLOADS)
def test_writing_beside_a_malformed_launch_history_changes_no_byte(
    payload: str, tmp_path: Path,
) -> None:
    """The existing file is the only copy of how earlier launches happened, so
    an unreadable one is never rewritten, truncated or appended to."""
    path = tmp_path / prov.LAUNCH_FILENAME
    path.write_text(payload)
    before = path.read_bytes()

    with pytest.raises(prov.LaunchRecordError):
        prov.write_launch_record(
            tmp_path, arm="control", tool="t", argv=[], device_request="cpu",
            lineage=prov.launch_lineage(resume=False, resume_from_update=None),
            git_commit=None, git_dirty=None,
        )
    assert path.read_bytes() == before


def test_reading_launch_records_returns_empty_only_when_there_is_no_file(
    tmp_path: Path,
) -> None:
    assert prov.read_launch_records(tmp_path) == []
    prov.write_launch_record(
        tmp_path, arm="aux", tool="t", argv=[], device_request="cpu",
        lineage=prov.launch_lineage(resume=False, resume_from_update=None),
        git_commit=None, git_dirty=None,
    )
    records = prov.read_launch_records(tmp_path)
    assert [entry["sequence"] for entry in records] == [1]


@pytest.mark.parametrize("module, arm", ARMS)
def test_arm_refuses_to_launch_beside_a_malformed_launch_history(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    path = out_dir / prov.LAUNCH_FILENAME
    path.write_text('{"schema_version": 1, "launches": [{"sequence": 7}]}')
    before = path.read_bytes()

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran beside an unreadable launch history")

    monkeypatch.setattr(module, "evaluate_deterministic", _boom)
    monkeypatch.setattr(module, "collect_rollout", _boom)

    with pytest.raises(prov.LaunchRecordError):
        module.run_experiment(_quick_args(module, tmp_path, out_dir), torch.device("cpu"))
    assert path.read_bytes() == before
    assert not (out_dir / "run.json").exists()


# --------------------------------------------------------------------------- #
# 3. both arms report alpha calibration in the SAME shape
# --------------------------------------------------------------------------- #
CALIBRATION_RECORD_KEYS = {"arm", "status", "calibration", "reason", "provenance"}


@pytest.mark.parametrize("module, arm", ARMS)
def test_report_alpha_calibration_uses_the_shared_wrapper_shape(
    module: Any, arm: str, tmp_path: Path,
) -> None:
    """A reader comparing the two arms' reports must not have to know that one
    arm nests its calibration under a status wrapper while the other inlines
    the raw calibration dict at the same key."""
    out_dir = tmp_path / "run"
    report = _run_quick(module, tmp_path, out_dir)

    record = report["alpha_calibration"]
    assert set(record) == CALIBRATION_RECORD_KEYS
    assert record["arm"] == arm
    # The report carries exactly what was written beside the run.
    assert record == json.loads((out_dir / prov.CALIBRATION_FILENAME).read_text())

    if arm == "aux":
        assert record["status"] == prov.STATUS_CALIBRATED
        assert record["provenance"] == prov.EVIDENCE_COMPUTED
        assert record["reason"] is None
        assert record["calibration"]["alpha"] > 0.0
        # The aux arm's full per-pair calibration evidence is preserved.
        assert len(record["calibration"]["pairs"]) == aux.CALIBRATION_GRAD_SAMPLES
        assert record["calibration"] == report["aux_arm"]["alpha_calibration"]
    else:
        assert record["status"] == prov.STATUS_NOT_APPLICABLE
        assert record["calibration"] is None
        assert record["reason"]


def test_aux_cli_still_prints_the_frozen_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Normalizing the report shape must not cost the CLI its alpha output."""
    init_path, teacher_path = _quick_inputs(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv",
        ["phase3_ranked_ppo_retention_aux", "--init-checkpoint", str(init_path),
         "--teacher", str(teacher_path), "--out-dir", str(out_dir), "--quick", "--device", "cpu"],
    )
    aux.main()

    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["arm_ran"] is True
    on_disk = json.loads((out_dir / "report.json").read_text())
    assert printed["alpha"] == on_disk["alpha_calibration"]["calibration"]["alpha"]
    assert printed["alpha"] > 0.0


def test_evidence_provenance_reads_alpha_through_the_wrapper() -> None:
    computed = prov.report_evidence_provenance(
        {"alpha_calibration": prov.calibration_record(arm="aux", calibration={"alpha": 0.5})}
    )
    assert computed["alpha_calibration"] == prov.EVIDENCE_COMPUTED

    not_applicable = prov.report_evidence_provenance(
        {"alpha_calibration": prov.calibration_record(arm="control")}
    )
    assert not_applicable["alpha_calibration"] == prov.STATUS_NOT_APPLICABLE

    # A wrapper that claims "calibrated" while carrying no alpha is not
    # evidence of a calibration; it must never be reported as computed.
    hollow = prov.report_evidence_provenance(
        {"alpha_calibration": prov.calibration_record(arm="aux", calibration={"pairs": []})}
    )
    assert hollow["alpha_calibration"] == prov.EVIDENCE_UNAVAILABLE
    assert prov.report_evidence_provenance({})["alpha_calibration"] == prov.EVIDENCE_UNAVAILABLE


# --------------------------------------------------------------------------- #
# 4. "player_v1" is computed only after a strict PlayerV1 load
# --------------------------------------------------------------------------- #
def _fill_report(dotted_fields: tuple[str, ...], value: Any = 1) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for dotted in dotted_fields:
        node = report
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return report


def test_teacher_identity_strict_loads_into_player_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``architecture: player_v1`` is a claim that these weights loaded into
    ``PlayerV1``.  Key presence alone is not that claim."""
    path = _write_teacher(tmp_path / "teacher.pt", hidden=8)
    seen: dict[str, Any] = {}
    real_load = PlayerV1.load_state_dict

    def _recording(self: PlayerV1, state_dict: Any, strict: bool = True) -> Any:
        seen["called"] = True
        seen["strict"] = strict
        seen["hidden"] = self.body[0].out_features
        return real_load(self, state_dict, strict=strict)

    monkeypatch.setattr(PlayerV1, "load_state_dict", _recording)
    identity = prov.teacher_identity(path)

    assert seen.get("called") is True
    assert seen.get("strict") is True
    assert seen.get("hidden") == 8
    assert identity["architecture"] == "player_v1"
    assert identity["hidden"] == 8
    assert identity["state_dict_sha256"]


def test_teacher_identity_refuses_a_state_dict_that_is_not_player_v1(tmp_path: Path) -> None:
    """A file that merely has ``hidden`` and ``state_dict`` keys is not a V1
    teacher.  It must not be tagged computed, and its unbound weights must
    not be hashed into the identity a contract would pin."""
    path = tmp_path / "not_v1.pt"
    torch.save({"hidden": 16, "state_dict": {"policy.weight": torch.zeros(2, 2)}}, path)

    identity = prov.teacher_identity(path)
    assert identity["file_sha256"]
    assert identity["architecture"] is None
    assert identity["hidden"] is None
    assert identity["state_dict_sha256"] is None

    manifest = prov.build_input_manifest(
        arm="control", tool="t",
        init_path=tmp_path / "absent-init.pt", init_meta={}, init_provenance={},
        teacher_path=path,
    )
    assert manifest["inputs"]["teacher"]["provenance"] == prov.EVIDENCE_UNAVAILABLE
    assert "inputs.teacher.architecture" in manifest["missing"]
    assert "inputs.teacher.state_dict_sha256" in manifest["missing"]


def test_teacher_identity_refuses_a_player_v1_dict_at_the_wrong_hidden_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong_hidden.pt"
    torch.save({"hidden": 8, "state_dict": PlayerV1(hidden=16).state_dict()}, path)
    identity = prov.teacher_identity(path)
    assert identity["architecture"] is None
    assert identity["state_dict_sha256"] is None


@pytest.mark.parametrize("module, arm", ARMS)
def test_a_non_player_v1_teacher_fails_closed_with_no_run_contract(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    init_path, teacher_path = _quick_inputs(tmp_path)
    torch.save({"hidden": 16, "state_dict": {"policy.weight": torch.zeros(2, 2)}}, teacher_path)

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("experiment work ran against a teacher that is not PlayerV1")

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
    assert report["status"] == "failed_closed"
    assert report["input_manifest"]["inputs"]["teacher"]["architecture"] is None
    assert report["input_manifest"]["inputs"]["teacher"]["provenance"] == prov.EVIDENCE_UNAVAILABLE
    assert not (out_dir / "run.json").exists()
    assert not (out_dir / "status.jsonl").exists()


# --------------------------------------------------------------------------- #
# 5. a complete report carries the gate and the retention verdict
# --------------------------------------------------------------------------- #
GATE_VERDICT = "initial_gate.gate_pass"
RETENTION_VERDICT = "retention.retention_pass"


def test_report_completeness_requires_teacher_architecture() -> None:
    """Teacher architecture is part of the input identity. A report that fills
    every other required field and leaves it null is not complete."""
    report = _fill_report(prov.REQUIRED_REPORT_FIELDS)
    teacher = report.setdefault("input_manifest", {}).setdefault("inputs", {}).setdefault("teacher", {})
    teacher["architecture"] = None
    verdict = prov.evaluate_report_completeness(report, arm="control")
    assert verdict["complete"] is False
    assert "input_manifest.inputs.teacher.architecture" in verdict["missing"]


def test_report_completeness_requires_the_gate_and_retention_verdicts() -> None:
    """Every other required field can be filled and the report is still not
    complete until it carries the pre-registered gate and the retention
    outcome.  A hollow object at those keys does not count."""
    report = _fill_report(prov.REQUIRED_REPORT_FIELDS)
    report.pop("initial_gate", None)
    report.pop("retention", None)
    report["initial_gate"] = {}
    report["retention"] = {"promotion": False}

    verdict = prov.evaluate_report_completeness(report, arm="control")
    assert verdict["complete"] is False
    assert GATE_VERDICT in verdict["missing"]
    assert RETENTION_VERDICT in verdict["missing"]


@pytest.mark.parametrize("module, arm", ARMS)
def test_identity_failure_report_is_honestly_incomplete(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
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
    assert report["status"] == "failed_closed"
    completeness = report["report_completeness"]
    assert completeness["complete"] is False
    assert GATE_VERDICT in completeness["missing"]
    assert RETENTION_VERDICT in completeness["missing"]


@pytest.mark.parametrize("module, arm", ARMS)
def test_gate_failure_report_keeps_the_gate_and_stays_incomplete_without_retention(
    module: Any, arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "run"
    args = _quick_args(module, tmp_path, out_dir)
    args.initial_gate_mean_min = 1.0
    args.initial_gate_median_min = 1.0
    monkeypatch.setattr(
        module, "evaluate_deterministic",
        lambda *_a, **_k: [{"seed": 0, "elapsed": 0.0, "censored": False}],
    )
    report = module.run_experiment(args, torch.device("cpu"))

    assert report["status"] == "failed_closed"
    assert report["initial_gate"]["gate_pass"] is False
    completeness = report["report_completeness"]
    assert completeness["complete"] is False
    assert GATE_VERDICT not in completeness["missing"]
    assert RETENTION_VERDICT in completeness["missing"]
