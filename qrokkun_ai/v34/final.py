"""Single claimed, offline paired final evaluation after formal closure."""
import json
import signal
from pathlib import Path
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v25.run import claim
from qrokkun_ai.v31 import core
from .adapter import EpisodePool, runtime
from .initialization import random_player, locked_spawner
from .reference import load_reference
from .protocol import FINAL_SEEDS, closure_valid, paired


def _preflight(a, stop=lambda: False):
    from .run import validate, ckpt_name
    validate(a)
    out = Path(a.out_dir)
    m, c, status = [json.loads((out/name).read_text()) for name in ('manifest.json','closed.json','status.json')]
    if not closure_valid(m,c,status):
        raise ValueError('complete formal 1000 training and terminal validation required')
    s = m['semantic']
    if s['source_commit'] != a.source_commit or s['source_tree_sha'] != core.source_digest():
        raise ValueError('source provenance mismatch')
    if c['manifest_sha'] != file_hash(out/'manifest.json'):
        raise ValueError('manifest changed')
    for key in ('updates','episodes','cap','seed','validation','hidden'):
        if s[key] != getattr(a,key):
            raise ValueError('config mismatch: '+key)
    expected = {ckpt_name(u) for u in (0,c['best_update'],1000)}
    if set(c['checkpoint_hashes']) != expected:
        raise ValueError('closure checkpoints incomplete')
    for name, sha in c['checkpoint_hashes'].items():
        if stop():
            raise InterruptedError('owner stop in preflight')
        if file_hash(out/name) != sha:
            raise ValueError('checkpoint hash mismatch')
    if stop():
        raise InterruptedError('owner stop in preflight')
    if c['saved_checkpoints'] != list(range(0,1001,10)):
        raise ValueError('101 checkpoints required')
    totals=[]
    for u in range(0,1001,10):
        e=json.loads((out/f'evaluation-{u:06d}.json').read_text())
        if e['update']!=u or [r['seed'] for r in e['episodes']]!=list(range(600000,600016)) or e['total_frames']!=sum(r['frames'] for r in e['episodes']):
            raise ValueError('validation records incomplete')
        totals.append((u,e['total_frames']))
    if max(totals,key=lambda x:x[1])[0]!=c['best_update']:
        raise ValueError('best identity mismatch')
    for label,u in [('best',c['best_update']),('terminal',1000),('latest',1000),('initial',0)]:
        pointer=json.loads((out/(label+'.json')).read_text())
        if pointer!={'update':u,'checkpoint':ckpt_name(u)}:
            raise ValueError('pointer identity mismatch')
    return m,c


def preflight(a, stop=lambda: False):
    """Validate closure and provenance in the scoped v34 source context."""
    with runtime():
        return _preflight(a, stop)


def _execute(a):
    from .run import json_write, ckpt_name
    stopped = []
    handlers = {}
    pool = None
    owned = False
    errors = []
    out = Path(a.out_dir)
    threads = torch.get_num_threads()
    def stop(sig, *_):
        stopped.append(sig)
        if pool is not None:
            pool.stop_event.set()
    def check():
        if stopped:
            raise InterruptedError('owner stop')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,stop)
        m,c = preflight(a,lambda:bool(stopped))
        check()
        claim(out/'final-claimed.json',dict(source_commit=a.source_commit))
        owned = True
        core.configure_runtime('cpu')
        original,s = random_player(),locked_spawner(a.spawner_checkpoint)
        warm = load_reference(Path(a.control_reference_dir), lambda:bool(stopped))
        if warm['identity'] != m['semantic']['control_reference']:
            raise ValueError('control reference changed')
        if m['semantic']['inputs'] != {'spawner':file_hash(a.spawner_checkpoint)}:
            raise ValueError('input mismatch')
        check()
        pool = EpisodePool(original,s,16,7200,10,'on',lambda:bool(stopped))
        results = {}
        arms = [('best',c['best_update']),('terminal',1000)]
        for version,(name,u) in enumerate(arms):
            check()
            if name == 'terminal' and u == c['best_update']:
                results[name] = dict(reference='best', **results['best'])
                continue
            path = out/ckpt_name(u)
            payload = torch.load(path,map_location='cpu',weights_only=False)
            if payload['update'] != u or payload['schema'] != 'pv34_checkpoint_v1':
                raise ValueError('checkpoint identity mismatch')
            sd = payload['model']
            identity = dict(file_sha=file_hash(path),checkpoint=path.name,update=u)
            check()
            pool.sync_weights(sd,policy_version=version,stop_check=lambda:bool(stopped))
            rows = []
            for start in range(0,64,16):
                check()
                trajs = pool.collect(FINAL_SEEDS[start:start+16],7200,validation=True,stop_check=lambda:bool(stopped))
                rows.extend({**ep.summary(),'cap_frames':7200, **identity, 'state_sha':state_dict_sha256(sd)} for ep in trajs)
            if [r['seed'] for r in rows] != FINAL_SEEDS:
                raise ValueError('incomplete final')
            results[name] = dict(**identity,state_sha=state_dict_sha256(sd),player_mode='argmax',spawner_mode='stochastic',episodes=rows)
            json_write(out/('final-'+name+'.json'),results[name])
        check()
        pool.shutdown_and_stats()
        pool = None
        check()
        json_write(out/'learning-comparison.json', dict(control=warm['curve'], filtered=[{k:v for k,v in json.loads((out/f'evaluation-{u:06d}.json').read_text()).items() if k != 'episodes'} for u in range(0,1001,10)], control_timings=warm['timings'], filtered_timings=c['timings'], budget='same u1000/E20 from matched random u0; actual frames and wall time differ'))
        json_write(out/'final.json',dict(status='complete',window='previously used Pv25 common benchmark; NOT fresh heldout',
                    results=results,control_reference=warm['identity'], primary=paired(results['best']['episodes'],warm['results']['best']['episodes']),
                    secondary=paired(results['terminal']['episodes'],warm['results']['terminal']['episodes'])))
    except BaseException as exc:
        errors.append(repr(exc))
        raise
    finally:
        for cleanup in (lambda: pool.close() if pool is not None else None,lambda:torch.set_num_threads(threads)):
            try:
                cleanup()
            except BaseException as exc:
                errors.append('cleanup: '+repr(exc))
        if owned:
            json_write(out/'final-status.json',dict(status='stopped' if stopped else ('failed' if errors else 'complete'),errors=errors,owner_stop=stopped))
        for sig,h in handlers.items():
            signal.signal(sig,h)
        if (errors or stopped) and not __import__('sys').exc_info()[0]:
            raise RuntimeError(str(errors or stopped))
    return 0


def execute(a):
    with runtime():
        return _execute(a)
