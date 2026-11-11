"""Run the unchanged production provenance resolver in an actual source snapshot."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from qrokkun_ai.v25 import run

SOURCE_SHA = '0123456789abcdef0123456789abcdef01234567'


@pytest.fixture
def archive(tmp_path):
    root = Path(__file__).resolve().parents[3]
    snapshot = tmp_path / 'source'
    snapshot.mkdir()
    # Copy imported runtime dependencies and package initializers, plus v25 tests
    # and reference data. This includes untracked v25 files, unlike git archive.
    paths = {Path(m.__file__).resolve() for name, m in list(sys.modules.items())
             if name.startswith(('qrokkun_ai', 'qrokkun_env'))
             and getattr(m, '__file__', None) and m.__file__.endswith('.py')}
    paths.update(p for p in (root / 'qrokkun_ai/v25').rglob('*')
                 if p.suffix in ('.py', '.json'))
    for source in paths:
        target = snapshot / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(snapshot), TMPDIR=str(scratch),
               GIT_CEILING_DIRECTORIES=str(tmp_path), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in list(env):
        if key.startswith('GIT_') and key != 'GIT_CEILING_DIRECTORIES':
            env.pop(key)
    env.pop('PV25_TEST_REAL_INPUTS', None)
    assert not list(snapshot.rglob('.git'))
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=snapshot, env=env,
                         capture_output=True, text=True)
    assert git.returncode != 0, git.stdout
    return snapshot, env


def invoke(archive, args):
    snapshot, env = archive
    command = [sys.executable, *args]
    result = subprocess.run(command, cwd=snapshot, env=env,
                            capture_output=True, text=True, timeout=120)
    (snapshot.parent / 'subprocess.stdout').write_text(result.stdout)
    (snapshot.parent / 'subprocess.stderr').write_text(result.stderr)
    (snapshot.parent / 'subprocess.json').write_text(json.dumps(
        dict(command=command, cwd=str(snapshot), returncode=result.returncode), indent=2))
    return result


@pytest.mark.parametrize('opt_in', [False, True])
def test_real_metadata_integration_skips_without_artifacts(archive, opt_in):
    snapshot, env = archive
    env.update(PV25_TEST_PLAYER_CHECKPOINT=str(snapshot/'absent-p.pt'),
               PV25_TEST_SPAWNER_CHECKPOINT=str(snapshot/'absent-s.pt'))
    if opt_in:
        env['PV25_TEST_REAL_INPUTS'] = '1'
    result = invoke(archive, ['-m', 'pytest', '-q', '-rs',
        'qrokkun_ai/v25/tests/test_run.py::test_real_locked_input_metadata_only'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 skipped' in result.stdout, result.stdout


def test_archive_focused_suite(archive):
    result = invoke(archive, ['-m', 'pytest', '-q', '-rs', *[
        f'qrokkun_ai/v25/tests/test_{name}.py'
        for name in ('contract', 'run', 'training', 'supervise')]])
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 skipped' in result.stdout, result.stdout


@pytest.mark.parametrize('explicit', [True, False])
def test_archive_source_cli_provenance(archive, explicit):
    # Exercise the real parser/resolver in a fresh process, without loading any
    # checkpoint or running episodes. Missing provenance must remain an error.
    script = '''
import sys
from pathlib import Path
from qrokkun_ai.v25 import run
assert Path(run.__file__).resolve().is_relative_to(Path.cwd())
a = run.parser().parse_args(sys.argv[1:])
run.validate_args(a)
print(run.v24.resolve_source_commit(a.source_commit))
'''
    args = ['-c', script, 'train', '--out-dir', 'unused']
    if explicit:
        args += ['--source-commit', SOURCE_SHA]
    result = invoke(archive, args)
    if explicit:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == SOURCE_SHA
    else:
        assert result.returncode != 0
        assert 'Cannot discover source commit; supply --source-commit' in result.stderr
