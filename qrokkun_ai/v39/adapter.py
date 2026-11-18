"""Reversible hooks at the aliases actually used by rollout and search."""
from contextlib import contextmanager, ExitStack
from functools import partial
from qrokkun_ai.v31 import core
from qrokkun_ai.v35.adapter import encode_obs as original_encode
from qrokkun_ai.v37 import adapter as old, episode, teaching
from .world import View, LearnedEnv, CaptureEnv, snapshot, restore

_ORIGINAL_EPISODE = episode.run_episode
_ORIGINAL_WORKER = old.worker

def encode_obs(env, max_bullets=64):
    return original_encode(View(env, 'P'), max_bullets)

@contextmanager
def bindings(module, **values):
    saved = {k:getattr(module,k) for k in values}
    try:
        for k,v in values.items(): setattr(module,k,v)
        yield
    finally:
        for k,v in saved.items(): setattr(module,k,v)

def run_episode(*args, **kwargs):
    mode = 0 if kwargs.get('validation', False) else 40
    with bindings(episode, CaptureEnv=partial(CaptureEnv, difficulty_offset=mode)):
        return _ORIGINAL_EPISODE(*args, **kwargs)

@contextmanager
def runtime(search=False, root=None):
    from .fingerprint import source_digest
    with old.runtime(search, root), ExitStack() as stack:
        stack.enter_context(bindings(core, source_digest=source_digest,
            LearnedEnv=LearnedEnv, run_episode=run_episode, encode_obs_fast=encode_obs))
        stack.enter_context(bindings(episode, encode_obs_fast=encode_obs, CaptureEnv=CaptureEnv))
        stack.enter_context(bindings(teaching, LearnedEnv=LearnedEnv,
            snapshot=snapshot, restore=restore, encode_obs=encode_obs))
        yield

def worker(conn, *args, search, root):
    # old.worker enters old.runtime again. Hook that runtime before delegation.
    original = old.runtime
    @contextmanager
    def nested(*a, **kw):
        old.runtime = original
        try:
            with runtime(*a, **kw): yield
        finally:
            old.runtime = nested
    with bindings(old, runtime=nested):
        return _ORIGINAL_WORKER(conn, *args, search=search, root=root)

class EpisodePool(old.EpisodePool):
    def __init__(self, *args, stats_path=None, **kwargs):
        from pathlib import Path
        self.stop_check = kwargs.get('stop_check', lambda:False)
        self.stats_path = Path(stats_path) if stats_path is not None else Path(kwargs['root']).parent/'workers.json'
        with bindings(old, worker=worker):
            super().__init__(*args, **kwargs)
    def collect(self, *args, **kwargs):
        # Bypass old.collect's extra runtime; lifecycle/shm stays unchanged.
        from qrokkun_ai.v32.pool import EpisodePool as Lifecycle
        with runtime(self.search, self.root):
            return Lifecycle.collect(self, *args, **kwargs)
    def shutdown_and_stats(self):
        # Snapshot requests: a successful reply may already be queued after exit.
        from qrokkun_ai.v37.run import json_write
        stats=[]; errors=[]; expected=[]
        try:
            for conn,proc in self.children:
                try:
                    if not proc.is_alive():
                        raise RuntimeError(f'worker pid={proc.pid} terminated before shutdown request')
                    conn.send(None)
                    expected.append((conn,proc))
                except BaseException as exc: errors.append(exc)
            for conn,proc in expected:
                try:
                    kind,rest=self._recv_one(conn,proc,self.stop_check)
                    if kind!='stats' or rest[0]['pid']!=proc.pid:
                        raise RuntimeError(f'worker pid={proc.pid} invalid shutdown stats')
                    stats.append(rest[0])
                except BaseException as exc:
                    errors.append(exc)
                    if isinstance(exc, InterruptedError): break
            if not errors:
                validate_worker_coverage(stats, self.workers_count,
                    [info['pid'] for info in self.worker_init_info])
            json_write(self.stats_path,stats)
        except BaseException as exc: errors.append(exc)
        try: self.close()
        except BaseException as exc: errors.append(exc)
        if errors: raise BaseExceptionGroup('Pv39 shutdown failed',errors)
        return stats


def validate_worker_coverage(stats, count, pids):
    actual=[row.get('pid') for row in stats]
    if (len(pids)!=count or len(set(pids))!=count
        or len(actual)!=count or len(set(actual))!=count or set(actual)!=set(pids)):
        raise ValueError('training worker stats coverage')
