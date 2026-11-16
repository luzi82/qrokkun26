"""Explicit spawn configuration and reversible core hooks."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import json
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

def worker(*args, search, root):
    with runtime(search, root):
        return _ORIGINAL_WORKER(*args)

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
            result = super().close()
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
