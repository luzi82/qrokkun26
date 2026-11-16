"""Explicit spawn configuration and reversible core hooks."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import json
import sys
import numpy as np
from qrokkun_ai.v31 import core
from qrokkun_ai.v32.pool import EpisodePool as LifecyclePool
from qrokkun_ai.v35.adapter import runtime as model_runtime
from .fingerprint import source_digest
from .world import LearnedEnv
from . import episode

_ORIGINAL_WORKER = core._pool_worker
_ORIGINAL_SLICE = core._slice_trajectory

def slice_trajectory(meta, buf):
    ep = _ORIGINAL_SLICE(meta, buf)
    ep.demos = []
    ep.teaching = {}
    if 'teaching' in meta:
        path = Path(episode.ROOT) / meta['teaching']
        try:
            with np.load(path, allow_pickle=False) as data:
                ep.teaching = json.loads(str(data['ledger']))
                if 'actions' in data:
                    ep.demos = list(zip(*(data[k].copy() for k in ('player','bullets','pad','actions'))))
        finally:
            path.unlink(missing_ok=True)
    return ep

@contextmanager
def runtime(search=False, root=None):
    saved = (episode.SEARCH, episode.ROOT)
    episode.SEARCH, episode.ROOT = search, root
    with model_runtime():
        replacements = dict(source_digest=source_digest, LearnedEnv=LearnedEnv, run_episode=episode.run_episode,
                            _slice_trajectory=slice_trajectory)
        previous = {k:getattr(core,k) for k in replacements}
        try:
            for k,v in replacements.items(): setattr(core,k,v)
            yield
        finally:
            for k,v in previous.items(): setattr(core,k,v)
            episode.SEARCH, episode.ROOT = saved

class _CloseCapture:
    def __init__(self, resource, errors):
        self.resource, self.errors = resource, errors

    def __getattr__(self, name):
        return getattr(self.resource, name)

    def close(self):
        try:
            return self.resource.close()
        except BaseException as exc:
            self.errors.append(exc)


def worker(conn, *args, search, root):
    errors = []
    with runtime(search, root):
        previous = core._open_child_shm
        try:
            core._open_child_shm = lambda *a, **kw: _CloseCapture(previous(*a, **kw), errors)
            result = _ORIGINAL_WORKER(_CloseCapture(conn, errors), *args)
        except BaseException as primary:
            if errors:
                failure = BaseExceptionGroup('v37 worker cleanup failed', [primary, *errors])
                print(repr(failure), file=sys.stderr)
                raise failure from primary
            raise
        finally:
            core._open_child_shm = previous
    if errors:
        failure = BaseExceptionGroup('v37 worker cleanup failed', errors)
        print(repr(failure), file=sys.stderr)
        raise failure
    return result


class _ExitCapture:
    """Read exitcode after v32's join, before Process.close invalidates it."""
    def __init__(self, process, errors):
        self.process, self.errors = process, errors

    def __getattr__(self, name):
        return getattr(self.process, name)

    def close(self):
        code = self.process.exitcode
        if code not in (None, 0):
            self.errors.append(RuntimeError(f'worker exited with code {code}'))
        return self.process.close()

class EpisodePool(LifecyclePool):
    def __init__(self, *args, search=False, root=None, **kwargs):
        self.search, self.root = search, str(root)
        Path(self.root).mkdir(parents=True, exist_ok=False)
        previous = core._pool_worker
        try:
            core._pool_worker = partial(worker, search=search, root=self.root)
            super().__init__(*args, **kwargs)
        finally:
            core._pool_worker = previous
    def collect(self,*args,**kwargs):
        with runtime(self.search, self.root):
            return super().collect(*args,**kwargs)
    def close(self):
        errors = []
        result = None
        try:
            children = self.children
            try:
                self.children = [(conn, _ExitCapture(proc, errors)) for conn, proc in children]
                result = super().close()
            finally:
                self.children = children
        except BaseException as exc:
            errors.append(exc)
        try:
            for p in Path(self.root).glob('*.npz'):
                try:
                    p.unlink()
                except BaseException as exc:
                    errors.append(exc)
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise BaseExceptionGroup('episode pool cleanup failed', errors)
        return result
