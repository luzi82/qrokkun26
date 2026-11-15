"""Memory-only regression for the actual v32 pool dependency."""
from types import SimpleNamespace
import pytest
from qrokkun_ai.v32 import run


def test_pool_unlink_failure_is_not_success():
    class Shm:
        def close(self): pass
        def unlink(self): raise OSError('observed unlink failure')
    pool = run.EpisodePool.__new__(run.EpisodePool) if hasattr(run, 'EpisodePool') else run.core.EpisodePool.__new__(run.core.EpisodePool)
    pool._closed = False
    pool.stop_event = SimpleNamespace(set=lambda: None)
    pool.children = []
    pool.w_shm, pool.t_shm = Shm(), None
    with pytest.raises(BaseException, match='observed unlink failure'):
        pool.shutdown_and_stats()
    assert not pool._closed


def test_cleanup_attempts_all_resources_and_retries_without_losing_evidence():
    calls = []
    class Shm:
        def close(self): calls.append('shm close')
        def unlink(self):
            calls.append('unlink')
            if calls.count('unlink') == 1: raise OSError('first unlink')
    class Proc:
        live = True
        def join(self, timeout): calls.append(('join', timeout))
        def is_alive(self): return self.live
        def terminate(self):
            calls.append('term')
            raise OSError('term failure')
        def kill(self): calls.append('kill'); self.live = False
        def close(self): calls.append('proc close')
    pool = run.EpisodePool.__new__(run.EpisodePool)
    pool._closed = False
    pool.stop_event = SimpleNamespace(set=lambda: calls.append('stop'))
    pool.children = [(SimpleNamespace(close=lambda: calls.append('conn close')), Proc())]
    pool.w_shm, pool.t_shm = Shm(), Shm()
    with pytest.raises(BaseExceptionGroup) as first: pool.close()
    assert 'term failure' in repr(first.value) and 'first unlink' in repr(first.value)
    assert calls == ['stop', 'conn close', ('join', 1.0), 'term', ('join', 2.0), 'kill', ('join', 2.0), 'proc close', 'shm close', 'unlink', 'shm close', 'unlink']
    with pytest.raises(BaseExceptionGroup) as retry: pool.close()
    assert 'first unlink' in repr(retry.value) and not pool._closed
    assert calls[-1] == 'unlink' and calls.count('proc close') == 1
    assert calls.count('shm close') == 2


def test_constructor_cleanup_keeps_primary_context(monkeypatch):
    from qrokkun_ai.v32.pool import core
    def partial(self):
        self._closed = False
        self.stop_event = SimpleNamespace(set=lambda: None)
        self.children = []
        self.w_shm = SimpleNamespace(close=lambda: None, unlink=lambda: (_ for _ in ()).throw(OSError('unlink failure')))
        self.t_shm = None
        try:
            raise ValueError('primary initialization failure')
        except BaseException:
            self.close()
            raise
    monkeypatch.setattr(core.EpisodePool, '__init__', partial)
    with pytest.raises(BaseExceptionGroup) as caught: run.EpisodePool()
    assert 'primary initialization failure' in repr(caught.value)
    assert 'unlink failure' in repr(caught.value)
