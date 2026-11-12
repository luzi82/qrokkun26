"""Consolidated CLI boundary, claim ownership, and cooperative signal tests."""
from contextlib import closing
import importlib
import json
import signal
import sqlite3
import subprocess
import sys
import zlib

import pytest

from qrokkun_ai.v26.cli import boundary
from qrokkun_ai.v26.trace import digest, json_write
from .test_validation_contracts import reseal, tiny_pair


def _arm_args(out, trace):
    return [
        '--arm', 'Pv26c',
        '--out-dir', str(out),
        '--trace-dir', str(trace),
        '--source-commit', 'a' * 40,
        '--nonformal',
        '--synthetic',
        '--device', 'cpu',
        '--updates', '1',
        '--episodes', '2',
        '--cap', '2',
        '--validation', '1',
        '--hidden', '8',
    ]


def test_cli_boundary_in_process(monkeypatch):
    for mod_name in ('run', 'supervise', 'compare'):
        m = importlib.import_module('qrokkun_ai.v26.' + mod_name)
        for exc, code in [
            (OSError('disk'), 3),
            (ValueError('config'), 3),
            (InterruptedError('stop'), 130),
            (KeyboardInterrupt(), 130),
        ]:
            monkeypatch.setattr(m, 'main', lambda argv=None, e=exc: (_ for _ in ()).throw(e))
            assert m.cli([]) == code

    for mod_name in ('run', 'supervise', 'compare'):
        m = importlib.import_module('qrokkun_ai.v26.' + mod_name)
        monkeypatch.undo()

    for mod_name in ('run', 'supervise', 'compare'):
        m = importlib.import_module('qrokkun_ai.v26.' + mod_name)
        assert m.cli(['--unknown-flag']) == 3
        with pytest.raises(SystemExit) as help_exit:
            m.cli(['--help'])
        assert help_exit.value.code == 0

    for code in (0, 1, 2, 7, None, 'control'):
        with pytest.raises(SystemExit) as caught:
            boundary(lambda argv=None, c=code: (_ for _ in ()).throw(SystemExit(c)))
        assert caught.value.code == code


def test_claim_loser_preserves_existing_claim(tmp_path):
    import qrokkun_ai.v26.run as run_mod
    import qrokkun_ai.v26.supervise as sup_mod

    out_run = tmp_path / 'out-run'
    trace_run = tmp_path / 'trace-run'
    out_run.mkdir()
    (out_run / 'claim.json').write_text('winner')
    assert run_mod.cli(_arm_args(out_run, trace_run)) == 3
    assert (out_run / 'claim.json').read_text() == 'winner'
    assert sorted(p.name for p in out_run.iterdir()) == ['claim.json']
    assert not trace_run.exists()

    out_sup = tmp_path / 'out-sup'
    trace_sup = tmp_path / 'trace-sup'
    out_sup.mkdir()
    (out_sup / 'pair-claim.json').write_text('winner')
    assert sup_mod.cli(_arm_args(out_sup, trace_sup)[2:]) == 3
    assert (out_sup / 'pair-claim.json').read_text() == 'winner'
    assert not trace_sup.exists()


def test_supervisor_launch_windows_mask_and_cleanup(tmp_path, monkeypatch):
    import qrokkun_ai.v26.supervise as s

    for window in ('status', 'publication'):
        sub = tmp_path / f'win-{window}'
        sub.mkdir()
        original_write = s.json_write
        original_popen = s.subprocess.Popen
        children = []
        fired = []

        def fire(fired=fired):
            if not fired:
                fired.append(True)
                signal.raise_signal(signal.SIGTERM)

        def write(path, data, window=window, fire=fire, original_write=original_write):
            original_write(path, data)
            if path.name == 'pair-status.json' and window == 'status':
                fire()

        def popen(*a, window=window, fire=fire, children=children, original_popen=original_popen, **kw):
            child = original_popen(*a, **kw)
            children.append(child)
            if window == 'publication':
                fire()
            return child

        monkeypatch.setattr(s, 'json_write', write)
        monkeypatch.setattr(s.subprocess, 'Popen', popen)
        before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        cmds = {k: [sys.executable, '-c', 'import time;time.sleep(30)'] for k in ('Pv26c', 'Pv26t', 'compare')}
        result = s.supervise(sub, cmds)
        assert result['status'] == 'stopped'
        assert len(children) == (1 if window == 'publication' else 0)
        assert all(c.poll() is not None for c in children)
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before
        monkeypatch.undo()

    mask_dir = tmp_path / 'mask'
    mask_dir.mkdir()
    mask_file = mask_dir / 'mask.json'
    cmd = [
        sys.executable, '-c',
        f'import json,signal;open({str(mask_file)!r},"w").write(json.dumps([int(x) for x in signal.pthread_sigmask(signal.SIG_BLOCK,[])]));signal.raise_signal(signal.SIGTERM)',
    ]
    res = s.supervise(mask_dir, {k: cmd for k in ('Pv26c', 'Pv26t', 'compare')})
    assert signal.SIGTERM not in json.loads(mask_file.read_text())
    assert res['exitcodes'] == {'Pv26c': -signal.SIGTERM}

    wait_dir = tmp_path / 'wait-exc'
    wait_dir.mkdir()
    orig_popen = s.subprocess.Popen
    spawned = []

    def spawn_failing_wait(*a, **kw):
        child = orig_popen(*a, **kw)
        spawned.append(child)
        real_wait = child.wait
        calls = []

        def failing_wait(*wa, **wkw):
            if not calls:
                calls.append(True)
                raise RuntimeError('injected wait exception')
            return real_wait(*wa, **wkw)

        child.wait = failing_wait
        return child

    monkeypatch.setattr(s.subprocess, 'Popen', spawn_failing_wait)
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    with pytest.raises(RuntimeError, match='injected wait'):
        s.supervise(wait_dir, {k: [sys.executable, '-c', 'import time;time.sleep(30)'] for k in ('Pv26c', 'Pv26t', 'compare')})
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before
    monkeypatch.undo()

    for code, expected_status in [(130, 'stopped'), (3, 'runtime_or_incomplete')]:
        exit_dir = tmp_path / f'exit-{code}'
        exit_dir.mkdir()
        res = s.supervise(exit_dir, {k: [sys.executable, '-c', f'raise SystemExit({code})'] for k in ('Pv26c', 'Pv26t', 'compare')})
        assert res['status'] == expected_status and res['exitcodes'] == {'Pv26c': code}
        assert not (exit_dir / 'Pv26t.log').exists()


def test_direct_arm_midrun_sigterm_leaves_trace_unclosed(tmp_path):
    out, trace = tmp_path / 'out', tmp_path / 'trace'
    script = f'''import os, signal
import qrokkun_ai.v26.run as m
def stop(*a, **kw):
    os.kill(os.getpid(), signal.SIGTERM)
    kw.get('check', a[-1])()
m.collect = stop
raise SystemExit(m.cli({_arm_args(out, trace)!r}))
'''
    child = subprocess.run([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert child.returncode == 130, child.stdout
    assert json.loads((out / 'status.json').read_text())['status'] == 'stopped'
    assert not (out / 'closed.json').exists()
    assert not (trace / 'closed.json').exists()


def test_supervised_finalize_sigterm_stops_pair(tmp_path):
    from qrokkun_ai.v26.supervise import supervise

    out, trace = tmp_path / 'Pv26c', tmp_path / 'trace'
    script = f'''import os, signal
import qrokkun_ai.v26.run as m
signal.alarm(30)
fired = False
write = m.json_write
def writing(path, data):
    global fired
    res = write(path, data)
    if path.name == 'closed.json' and not fired:
        fired = True
        os.kill(os.getpid(), signal.SIGTERM)
    return res
m.json_write = writing
raise SystemExit(m.cli({_arm_args(out, trace)!r}))
'''
    commands = {'Pv26c': [sys.executable, '-c', script]}
    for label in ('Pv26t', 'compare'):
        commands[label] = [sys.executable, '-c', f'open({str(tmp_path / "NEXT")!r}, "w").write("{label}")']
    result = supervise(tmp_path, commands)
    assert result['exitcodes'] == {'Pv26c': 130}
    assert result['status'] == 'stopped'
    assert json.loads((out / 'status.json').read_text())['status'] == 'stopped'
    assert not (tmp_path / 'NEXT').exists()


def test_compare_cli_subprocess_weight_diff(tiny_pair, tmp_path):
    def flip_weight(rows):
        for row in rows:
            if row['kind'] == 'checkpoint' and row['key'] == [1]:
                model = row['data']['model']
                ref = dict(model['policy_head.bias'])
                with closing(sqlite3.connect(tmp_path / 'Pv26t-trace' / 'blobs.sqlite')) as db:
                    with db:
                        raw = bytearray(zlib.decompress(
                            db.execute('SELECT raw FROM blobs WHERE sha=?', (ref['blob'],)).fetchone()[0]
                        ))
                        raw[0] ^= 1
                        ref['blob'] = digest(raw)
                        db.execute('INSERT INTO blobs VALUES (?,?,?)', (ref['blob'], len(raw), zlib.compress(raw)))
                model['policy_head.bias'] = ref

    right = reseal(tiny_pair, tmp_path, 'Pv26t', flip_weight)
    report = tmp_path / 'report.json'
    child = subprocess.run(
        [sys.executable, '-m', 'qrokkun_ai.v26.compare', str(tiny_pair / 'Pv26c'), str(right), '--out-report', str(report)],
        capture_output=True, text=True, timeout=30,
    )
    assert child.returncode == 2, child.stdout + child.stderr
    assert json.loads(report.read_text())['status'] == 'complete_different'
