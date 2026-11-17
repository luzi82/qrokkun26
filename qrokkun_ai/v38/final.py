"""Single claimed, offline paired final evaluation after formal closure."""
import json
import signal
from pathlib import Path
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v25.run import claim
from qrokkun_ai.v31 import core
from qrokkun_ai.v37.adapter import EpisodePool, runtime
from qrokkun_ai.v35.initialization import random_player, locked_spawner
from qrokkun_ai.v35.model import PlayerV35
from .protocol import FINAL_SEEDS, paired, control_pin, read, validate_record


def _preflight(a, stop=lambda: False):
    from qrokkun_ai.v37.run import validate
    from .protocol import closure
    validate(a)
    m,c=closure(Path(a.out_dir),stop)
    if m['semantic']['source_commit']!=a.source_commit or m['semantic']['role']!=a.role:
        raise ValueError('source/role mismatch')
    return m,c

def preflight(a, stop=lambda: False):
    with runtime():
        return _preflight(a,stop)


def _execute(a):
    from qrokkun_ai.v37.run import json_write, ckpt_name
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
        warm = None
        if a.role == 'search':
            pin = control_pin(Path(a.control_dir),lambda:bool(stopped))
            if pin != m['semantic']['control_pin']:
                raise ValueError('control pin changed')
            warm = read(Path(a.control_dir),'final.json')
        if m['semantic']['inputs'] != {'spawner':file_hash(a.spawner_checkpoint)}:
            raise ValueError('input mismatch')
        check()
        pool = EpisodePool(p_template=original,s_cpu=s,max_slots=16,max_cap=7200,workers=10,pin_cores='on',stop_check=lambda:bool(stopped),search=False,root=out/'final-cache')
        results = {}
        arms = [('best',c['best_update']),('terminal',1000)]
        for version,(name,u) in enumerate(arms):
            check()
            if name == 'terminal' and u == c['best_update']:
                results[name] = dict(reference='best', **results['best'])
                json_write(out/('final-'+name+'.json'),results[name])
                continue
            path = out/ckpt_name(u)
            payload = torch.load(path,map_location='cpu',weights_only=False)
            if payload['update'] != u or payload['schema'] != ('pv37_checkpoint_v2' if a.role=='control' else 'pv38_checkpoint_v1') or payload['arm'] != m['arm'] or payload['source_tree_sha'] != m['semantic']['source_tree_sha']:
                raise ValueError('checkpoint identity mismatch')
            loaded = PlayerV35.from_metadata(payload['architecture'])
            loaded.load_state_dict(payload['model'],strict=True)
            sd = payload['model']
            identity = dict(file_sha=file_hash(path),checkpoint=path.name,update=u)
            check()
            pool.sync_weights(sd,policy_version=version,stop_check=lambda:bool(stopped))
            rows = []
            for start in range(0,64,16):
                check()
                trajs = pool.collect(FINAL_SEEDS[start:start+16],7200,validation=True,stop_check=lambda:bool(stopped))
                rows.extend({**ep.summary(),'cap_frames':7200, **identity, 'state_sha':state_dict_sha256(sd)} for ep in trajs)
            check()
            for row in rows:
                validate_record(row)
            if [r['seed'] for r in rows] != FINAL_SEEDS:
                raise ValueError('incomplete final')
            results[name] = dict(**identity,state_sha=state_dict_sha256(sd),player_mode='argmax',spawner_mode='stochastic',episodes=rows,mean_frames=sum(r["frames"] for r in rows)/64,mean_seconds=sum(r["frames"] for r in rows)/3840)
            json_write(out/('final-'+name+'.json'),results[name])
        check()
        pool.shutdown_and_stats()
        pool = None
        check()
        comparison = {} if warm is None else dict(primary=paired(results['best']['episodes'],warm['results']['best']['episodes']), secondary=paired(results['terminal']['episodes'],warm['results']['terminal']['episodes']))
        json_write(out/'final.json',dict(status='complete',window='previously seen common reference; NOT fresh heldout',spawner_offset=core.S_VAL_OFFSET, results=results, **comparison))
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
