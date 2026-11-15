"""Single claimed, offline paired final evaluation after formal closure."""
import json
import signal
from pathlib import Path
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v25.run import inputs, claim
from qrokkun_ai.v31 import core
from .pool import EpisodePool
from .protocol import FINAL_SEEDS, closure_valid, paired


def preflight(a, stop=lambda: False):
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
    return m,c


def execute(a):
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
        original,s,_ = inputs(a.player_checkpoint,a.spawner_checkpoint,'cpu')
        if m['semantic']['inputs'] != {'player':file_hash(a.player_checkpoint),'spawner':file_hash(a.spawner_checkpoint)}:
            raise ValueError('input mismatch')
        check()
        pool = EpisodePool(original,s,16,7200,10,'on',lambda:bool(stopped))
        results = {}
        arms = [('original',None),('best',c['best_update']),('terminal',1000)]
        for version,(name,u) in enumerate(arms):
            check()
            if name == 'terminal' and u == c['best_update']:
                results[name] = dict(reference='best', **results['best'])
                continue
            if u is None:
                sd = original.state_dict()
                identity = dict(file_sha=file_hash(a.player_checkpoint),checkpoint=a.player_checkpoint,update=400)
            else:
                path = out/ckpt_name(u)
                payload = torch.load(path,map_location='cpu',weights_only=False)
                if payload['update'] != u or payload['schema'] != 'pv32_checkpoint_v1':
                    raise ValueError('checkpoint identity mismatch')
                sd = payload['model']
                identity = dict(file_sha=file_hash(path),checkpoint=path.name,update=u)
            check()
            pool.sync_weights(sd,policy_version=version,stop_check=lambda:bool(stopped))
            rows = []
            for start in range(0,64,16):
                check()
                trajs = pool.collect(FINAL_SEEDS[start:start+16],7200,validation=True,stop_check=lambda:bool(stopped))
                rows.extend({**ep.summary(),'cap_frames':7200} for ep in trajs)
            if [r['seed'] for r in rows] != FINAL_SEEDS:
                raise ValueError('incomplete final')
            results[name] = dict(**identity,state_sha=state_dict_sha256(sd),player_mode='argmax',spawner_mode='stochastic',episodes=rows)
            json_write(out/('final-'+name+'.json'),results[name])
        check()
        pool.shutdown_and_stats()
        pool = None
        check()
        json_write(out/'final.json',dict(status='complete',window='previously used Pv25 common benchmark; NOT fresh heldout',
                    results=results,primary=paired(results['best']['episodes'],results['original']['episodes']),
                    secondary=paired(results['terminal']['episodes'],results['original']['episodes'])))
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
