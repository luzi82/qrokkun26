"""Actual engine traces; no seed equivalence assumed."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import pytest
from qrokkun_env.gv20 import Gv20Env

ROOT = Path(__file__).resolve().parents[1]
def godot_binary():
    binary = os.environ.get("GODOT_BIN") or shutil.which("godot") or shutil.which("godot4")
    if not binary:
        pytest.skip("Install Godot 4 or set GODOT_BIN to its executable")
    return binary


def engine_env(tmp_path):
    """Isolate persistence and discard caller controls before explicit test settings."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("QROKKUN_", "GV20_"))}
    for key, directory in (("HOME", "home"), ("XDG_DATA_HOME", "data"),
                           ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache")):
        path = tmp_path / "profile" / directory
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


@pytest.fixture(scope="session", autouse=True)
def import_project(tmp_path_factory):
    result = subprocess.run([godot_binary(), "--headless", "--path", str(ROOT),
                             "--editor", "--import"], env=engine_env(tmp_path_factory.mktemp("import")),
                             capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCRIPT ERROR" not in result.stderr and "Parse Error" not in result.stderr


def trace(seed, tmp_path, long=False, **options):
    path = tmp_path / f"trace-{seed}.jsonl"
    env = dict(engine_env(tmp_path), QROKKUN_SEED=str(seed), GV20_TRACE=str(path),
               GV20_LONG="1" if long else "0")
    env.update(options)
    result = subprocess.run([godot_binary(), "--headless", "--path", str(ROOT),
                            "--script", "tools/gv20_probe.gd", "--fixed-fps", "60"],
                           env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCRIPT ERROR" not in result.stderr
    return [json.loads(line) for line in path.read_text().splitlines()]

class RecordedRandom:
    def __init__(self, events, field):
        self.values = iter(value for e in events if e[1] == field
                           for value in ([e[2]] if field else [e[2],e[3],e[4],e[5]]))
    def uniform(self, *_): return next(self.values)
    def gauss(self, *_): return next(self.values)

class Replay(Gv20Env):
    def _spawn_one(self):
        # reset installs RNGs immediately before its first spawn.
        if self.frame == 0:
            self.s_rng = RecordedRandom(self.recorded, False)
            self.field_rng = RecordedRandom(self.recorded, True)
        super()._spawn_one()

@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("seed", [3,17,42])
def test_engine_parity(seed, moving, tmp_path):
    rows = trace(seed, tmp_path, GV20_MOVE="1" if moving else "0")
    env = Replay(seed=seed)
    env.recorded = [e for row in rows for e in row["events"]]
    env.reset()
    for row in rows:
        env.step(row["action"])
        assert env.frame == row["frame"]
        assert [env.px,env.py,env.pvx,env.pvy] == pytest.approx(row["player"], abs=0.0001)
        assert len(env.bullets) == len(row["bullets"])
        for b, actual in zip(env.bullets,row["bullets"]):
            assert [b.x,b.y,b.vx,b.vy,b.radius] == pytest.approx(actual, abs=0.003)
        assert env.dead == row["dead"]
    assert env.dead

def test_long_schedule(tmp_path):
    from qrokkun_env.gv20 import spawn_frames
    rows = trace(42, tmp_path, True)
    assert len(rows) == 7200
    events = [e for row in rows for e in row["events"]]
    assert [e[0] for e in events if not e[1]] == list(spawn_frames(7201))
    for e in events:
        if e[1]:
            speeds = [s[5] for s in events if not s[1] and e[0]-60 <= s[0] < e[0]]
            assert e[5] == pytest.approx(sum(speeds)/len(speeds), abs=1e-8)
        else:
            assert 30 <= e[5] <= 200


def test_stream_independence_and_replay(tmp_path):
    baseline = trace(17, tmp_path, True)
    disturbed = trace(17, tmp_path, True, GV20_PERTURB="1")
    assert baseline == disturbed
    events = [e for row in baseline for e in row["events"]]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))
    replay = trace(999, tmp_path, True, GV20_REPLAY=str(path))
    assert replay == baseline

def test_moving_and_restart(tmp_path):
    rows = trace(3, tmp_path, GV20_MOVE="1", GV20_RESTART="1")
    assert sum(row["frame"] == 1 for row in rows) == 2
    assert rows[-1]["dead"]


@pytest.mark.parametrize("distance,death", [(6.59,True),(6.61,False)])
def test_collision_geometry(distance, death, tmp_path):
    rows = trace(42, tmp_path, GV20_BOUNDARY=str(distance))
    assert rows[0]["dead"] == death
    assert rows[0]["bullets"][0][4] == pytest.approx(2.6, abs=1e-6)


def test_historical_reset_delay():
    env = Gv20Env(seed=42)
    env.reset()
    bullet = env.bullets[0]
    before = (bullet.x, bullet.y)
    assert bullet.moved is False
    env.step(0)
    assert (bullet.x, bullet.y) == before
    env.step(0)
    assert (bullet.x, bullet.y) != before


def test_input_start_restart_trajectory(tmp_path):
    rows = trace(3, tmp_path, GV20_INPUT="1")
    runs = []
    for row in rows:
        if row["frame"] == 0:
            runs.append([])
        runs[-1].append(row)
    assert len(runs) == 2
    for run in runs:
        env = Replay(seed=3)
        env.recorded = [e for row in run for e in row["events"]]
        env.reset()
        for row in run:
            if row["frame"]:
                env.step(3)
            else:
                assert len(row["bullets"]) == 1
                assert row["elapsed"] == 0
                assert row["playing_ui"]
                assert (row["flash_t"], row["flash_alpha"]) == (0, 0)
            if row["dead"]:
                assert row["gameover_ui"]
                assert row["hit_animation"] > 0
            assert [env.px, env.py, env.pvx, env.pvy] == pytest.approx(row["player"], abs=0.0001)
            for bullet, actual in zip(env.bullets, row["bullets"]):
                assert [bullet.x, bullet.y] == pytest.approx(actual[:2], abs=0.003)
            assert len(env.bullets) == len(row["bullets"])
            assert env.dead == row["dead"]
        assert run[-1]["dead"]


def test_production_memory_bounds(tmp_path):
    rows = trace(42, tmp_path, True, GV20_MEMORY="1")
    assert len(rows) == 7200
    assert all(not row["events"] for row in rows)
    assert max(row["history_size"] for row in rows) <= 61


def test_godot_discovery(monkeypatch):
    monkeypatch.setenv("GODOT_BIN", "/chosen/godot")
    assert godot_binary() == "/chosen/godot"
    monkeypatch.delenv("GODOT_BIN")
    monkeypatch.setattr("shutil.which", lambda name: "/path/godot4" if name == "godot4" else None)
    assert godot_binary() == "/path/godot4"
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="GODOT_BIN"):
        godot_binary()


def test_fresh_import(tmp_path):
    project = tmp_path / "fresh"
    project.mkdir()
    for name in ("scripts", "scenes", "assets"):
        shutil.copytree(ROOT / name, project / name)
    shutil.copy(ROOT / "project.godot", project)
    result = subprocess.run([godot_binary(), "--headless", "--path", str(project),
                             "--editor", "--import"], env=engine_env(tmp_path),
                             capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCRIPT ERROR" not in result.stderr
    assert "Parse Error" not in result.stderr


def test_score_profile_isolation(tmp_path, monkeypatch):
    # A synthetic caller profile makes the RED safe: never touch a real profile.
    caller = tmp_path / "caller"
    for key, value in engine_env(caller).items():
        if key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
            monkeypatch.setenv(key, value)
    sentinel = caller / "profile/data/godot/app_userdata/Qrokkun/qrokkun.cfg"
    sentinel.parent.mkdir(parents=True)
    original = '[score]\nbest=0.0\n'
    sentinel.write_text(original)
    isolated = tmp_path / "engine"
    isolated.mkdir()
    rows = trace(3, isolated)
    assert rows[-1]["dead"] and rows[-1]["elapsed"] > 0
    assert sentinel.read_text() == original
    saves = list((isolated / "profile").rglob("qrokkun.cfg"))
    assert len(saves) == 1
    saved = saves[0].read_bytes()
    assert b"best=" in saved and b"best=0.0" not in saved
    # The same isolated profile loads its score in a second real engine process.
    assert trace(3, isolated)[0]["best_time"] == pytest.approx(rows[-1]["elapsed"])
    assert saves[0].read_bytes() == saved


def test_hostile_inherited_controls(tmp_path, monkeypatch):
    # Only synthetic paths are hostile: a RED can overwrite this sentinel safely.
    sentinel = tmp_path / "caller-recording.jsonl"
    sentinel.write_text("synthetic recording must survive\n")
    hostile = {"QROKKUN_MONKEY": "1", "QROKKUN_OUT": str(sentinel),
               "QROKKUN_FRAMES": "1", "QROKKUN_SEED": "999",
               "GV20_INPUT": "1", "GV20_CAPTURE": str(tmp_path / "capture.png"),
               "QROKKUN_FUTURE_CONTROL": "hostile", "GV20_FUTURE_CONTROL": "hostile"}
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GODOT_BIN", godot_binary())
    monkeypatch.setenv("GV20_RUNTIME_SENTINEL", "hostile")
    monkeypatch.setenv("NORMAL_RUNTIME_SENTINEL", "preserved")
    env = engine_env(tmp_path / "isolated")
    # Run even in RED to demonstrate the real engine consequence safely.
    result = subprocess.run([godot_binary(), "--headless", "--path", str(ROOT),
                             "--quit-after", "5"], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text() == "synthetic recording must survive\n"
    assert not any(k.startswith(("QROKKUN_", "GV20_")) for k in env)
    assert env["GODOT_BIN"] == os.environ["GODOT_BIN"]
    assert env["NORMAL_RUNTIME_SENTINEL"] == "preserved"
    assert trace(3, tmp_path / "explicit", GV20_INPUT="1")[-1]["dead"]


@pytest.mark.parametrize("key,action,start", [
    ("UP", 1, "ENTER"), ("W", 1, "SPACE"),
    ("RIGHT", 3, "Z"), ("D", 3, "KP_ENTER"),
    ("DOWN", 5, "ENTER"), ("S", 5, "SPACE"),
    ("LEFT", 7, "Z"), ("A", 7, "KP_ENTER"),
])
def test_physical_keys_hold_restart_release(tmp_path, key, action, start):
    rows = trace(3, tmp_path, GV20_INPUT="1", GV20_MOVE_KEY=key,
                 GV20_START_KEY=start, GV20_RELEASE="1")
    runs = []
    for row in rows:
        if row["frame"] == 0:
            runs.append([])
        runs[-1].append(row)
    assert len(runs) == 2
    for run in runs:
        env = Replay(seed=3)
        env.recorded = [e for row in run for e in row["events"]]
        env.reset()
        assert run[-1]["frame"] > 17
        for row in run:
            if row["frame"]:
                expected = 0 if 13 <= row["frame"] <= 16 else action
                env.step(expected)
                assert row["action"] == expected
            assert row["player"] == pytest.approx([env.px, env.py, env.pvx, env.pvy], abs=0.0001)
            assert row["dead"] == env.dead
        assert run[-1]["dead"]
