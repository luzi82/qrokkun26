"""Exercise deployed source bundling without checkout metadata."""
import importlib.util
import shutil
import tarfile
from pathlib import Path

from qrokkun_ai.v37 import fingerprint, protocol


def test_archive_source_without_git(tmp_path, monkeypatch):
    root = Path(fingerprint.__file__).resolve().parents[2]
    deployed = tmp_path / 'deployed'
    # Only the production dependency closure, never a checkout or test tree.
    paths = fingerprint.source_files()
    for path in paths:
        target = deployed / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    monkeypatch.chdir(deployed)
    spec = importlib.util.spec_from_file_location(
        'qrokkun_ai.v37.deployed_protocol', deployed / 'qrokkun_ai/v37/protocol.py')
    bundled_protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bundled_protocol)
    monkeypatch.setattr(fingerprint, '__file__', str(deployed / 'qrokkun_ai/v37/fingerprint.py'))
    destination = tmp_path / 'source.tar'
    commit = 'd' * 40
    bundled_protocol.archive_source(commit, destination)
    with tarfile.open(destination) as archive:
        assert archive.pax_headers['qrokkun.source_commit'] == commit
        assert 'git-unverified' in archive.pax_headers['qrokkun.source_identity']
        expected = {str(path.relative_to(root)) for path in paths}
        assert set(archive.getnames()) == expected
        assert {'qrokkun_ai/v37/contract.json', 'qrokkun_ai/v25/reference.json',
                'qrokkun_env/gv36.py', 'qrokkun_ai/v38/formal-contract.json',
                'qrokkun_ai/v38/run.py', 'qrokkun_ai/v38/final.py'} <= expected
        for name in expected:
            assert archive.extractfile(name).read() == (deployed / name).read_bytes()
