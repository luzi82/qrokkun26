"""Scoped v31 hooks. Spawn target is importable and installs its own hooks."""
from contextlib import contextmanager
from types import SimpleNamespace
from qrokkun_ai.v31 import core
from qrokkun_ai.v32.pool import EpisodePool as LifecyclePool
from .filter import harmless

_ORIGINAL_ENCODER = core.encode_obs_fast
_ORIGINAL_WORKER = core._pool_worker
_ORIGINAL_EPISODE = core.run_episode
_ORIGINAL_SLICE = core._slice_trajectory
_ORIGINAL_DIGEST = core.source_digest
_STATS = dict(considered=0, discarded=0, retained=0)


def encode_obs(env, max_bullets=64):
    candidates=[]
    for b in env.bullets:
        _STATS['considered'] += 1
        if harmless(env,b):
            _STATS['discarded'] += 1
        else:
            candidates.append(b)
            _STATS['retained'] += 1
    # Forward attributes without copying or mutating environment state.
    class View:
        def __getattr__(self, key):
            return getattr(env,key)
    view=View()
    view.bullets=candidates
    return _ORIGINAL_ENCODER(view,max_bullets)


def run_episode(*args, **kwargs):
    before=dict(_STATS)
    meta,buf=_ORIGINAL_EPISODE(*args,**kwargs)
    meta['filter_stats']={k:_STATS[k]-before[k] for k in before}
    return meta,buf


def slice_trajectory(meta,buf):
    ep=_ORIGINAL_SLICE(meta,buf)
    stats=meta.get('filter_stats',{})
    original=ep.summary
    ep.summary=lambda: {**original(), 'filter_stats':stats}
    return ep


from .fingerprint import source_digest


@contextmanager
def runtime():
    replacements=dict(encode_obs_fast=encode_obs, run_episode=run_episode,
                      _slice_trajectory=slice_trajectory,source_digest=source_digest)
    saved={k:getattr(core,k) for k in replacements}
    try:
        for k,v in replacements.items():
            setattr(core,k,v)
        yield
    finally:
        for k,v in saved.items():
            setattr(core,k,v)


def worker(*args):
    with runtime():
        return _ORIGINAL_WORKER(*args)


class EpisodePool(LifecyclePool):
    def __init__(self,*args,**kwargs):
        previous=core._pool_worker
        try:
            core._pool_worker=worker
            super().__init__(*args,**kwargs)
        finally:
            core._pool_worker=previous

    def collect(self,*args,**kwargs):
        with runtime():
            return super().collect(*args,**kwargs)
