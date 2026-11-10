"""Archive provenance and preregistered formal seed regressions."""
import pytest
from qrokkun_ai.v23 import run as r

SHA = '0123456789abcdef0123456789abcdef01234567'

def test_archive_explicit_source_without_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / '.git').exists()
    def forbidden(*args, **kwargs):
        pytest.fail('explicit provenance must not invoke Git')
    monkeypatch.setattr(r.subprocess, 'check_output', forbidden)
    a = r.parser().parse_args(['train', '--out-dir', 'out', '--source-commit', SHA])
    r.validate_args(a)
    assert r.resolve_source_commit(a.source_commit) == SHA
    assert vars(a)['source_commit'] == SHA

@pytest.mark.parametrize('value', ['HEAD', 'a'*39, 'a'*41, 'g'*40, ' '+SHA, SHA+'\n'])
def test_source_literal_rejected(value):
    with pytest.raises(SystemExit):
        r.parser().parse_args(['train', '--out-dir', 'out', '--source-commit', value])

def test_archive_missing_source_actionable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='--source-commit'):
        r.resolve_source_commit(None)

def test_formal_seed18_rejected():
    a = r.parser().parse_args(['train', '--out-dir', 'out', '--seed', '18'])
    with pytest.raises(ValueError, match='seed.*17'):
        r.validate_args(a)
    for flag in ('quick', 'nonformal'):
        setattr(a, flag, True)
        r.validate_args(a)
        setattr(a, flag, False)


def test_supervisor_forwards_source(tmp_path, monkeypatch):
    from qrokkun_ai.v23 import supervise as s
    a = r.parser().parse_args(['train', '--out-dir', str(tmp_path/'run'), '--source-commit', SHA])
    calls = []
    class Child:
        pid = 123
        def __init__(self, args, **kwargs): calls.append(args)
        def poll(self): return 7
        def wait(self): return 7
    monkeypatch.setattr(s.subprocess, 'Popen', Child)
    assert s.execute(a)['stage'] == 'failed'
    assert calls[0][calls[0].index('--source-commit')+1] == SHA
