"""In-memory worker cleanup diagnostics, without spawning processes."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from qrokkun_ai.v37 import adapter


@pytest.mark.parametrize('fail', [False, True])
def test_worker_stats_and_all_resource_closes(monkeypatch, capsys, fail):
    calls, messages = [], []

    class Resource:
        buf = bytearray(b'buffer')
        def __init__(self, name): self.name = name
        def close(self):
            calls.append(self.name)
            if fail:
                raise OSError('full close failure: ' + self.name)

    class Connection(Resource):
        def send(self, message): messages.append(message)
        def recv(self): return None

    class Model:
        def __init__(self, **kwargs): pass
        def eval(self): return self
        def requires_grad_(self, value): pass
        def load_state_dict(self, state): pass
        def state_dict(self): return {}

    def open_shm(name): return Resource(name)
    def bind(shm, specs):
        assert shm.buf == bytearray(b'buffer')
        return {}

    monkeypatch.setattr(adapter, 'runtime', lambda *args: nullcontext())
    monkeypatch.setattr(adapter.core, '_open_child_shm', open_shm)
    monkeypatch.setattr(adapter.core, '_bind_traj_views', bind)
    monkeypatch.setattr(adapter.core, 'PlayerV21', Model)
    monkeypatch.setattr(adapter.core, 'Spawner', Model)
    monkeypatch.setattr(adapter.core, 'state_dict_sha256', lambda state: 'same')
    args = (Connection('conn'), 4, {}, 'weights', [], 'trajectory', [], None, None)
    if fail:
        with pytest.raises(BaseExceptionGroup) as caught:
            adapter.worker(*args, search=False, root=None)
        for name in ('weights', 'trajectory', 'conn'):
            assert 'full close failure: ' + name in repr(caught.value)
        stderr = capsys.readouterr().err
        assert all('full close failure: ' + name in stderr
                   for name in ('weights', 'trajectory', 'conn'))
    else:
        assert adapter.worker(*args, search=False, root=None) is None
    assert calls == ['weights', 'trajectory', 'conn']
    assert [message[0] for message in messages] == ['ready', 'stats']
    assert messages[-1][1]['frozen_S_verified'] is True
    assert adapter.core._open_child_shm is open_shm


def test_worker_preserves_escaping_primary(monkeypatch):
    primary = ValueError('primary failure')
    original_open = adapter.core._open_child_shm
    def original(conn):
        try:
            raise primary
        finally:
            conn.close()
    def close(): raise OSError('connection failure')
    monkeypatch.setattr(adapter, 'runtime', lambda *args: nullcontext())
    monkeypatch.setattr(adapter, '_ORIGINAL_WORKER', original)
    with pytest.raises(BaseExceptionGroup) as caught:
        adapter.worker(SimpleNamespace(close=close), search=False, root=None)
    assert caught.value.exceptions[0] is primary
    assert 'connection failure' in repr(caught.value)
    assert adapter.core._open_child_shm is original_open


def test_parent_rejects_nonzero_exit_after_join(monkeypatch):
    calls = []
    class Process:
        exitcode = None
        def join(self, timeout):
            calls.append('join')
            self.exitcode = 1
        def is_alive(self): return False
        def close(self): calls.append('process.close')
    pool = object.__new__(adapter.EpisodePool)
    proc = Process()
    pool.children = [(SimpleNamespace(close=lambda: calls.append('conn.close')), proc)]
    pool.stop_event = SimpleNamespace(set=lambda: calls.append('stop'))
    pool._closed = False
    pool.root = 'in-memory'
    monkeypatch.setattr(adapter.Path, 'glob', lambda *args: [])
    with pytest.raises(BaseExceptionGroup, match='episode pool cleanup failed') as caught:
        pool.close()
    assert 'exited with code 1' in repr(caught.value)
    assert calls == ['stop', 'conn.close', 'join', 'process.close']
    assert pool.children[0][1] is proc
