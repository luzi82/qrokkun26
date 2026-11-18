"""Once-only normal-clock paired evaluation after version-local closure."""
import argparse
import signal
from pathlib import Path
import torch
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v25.run import claim
from qrokkun_ai.v35.initialization import random_player, locked_spawner
from qrokkun_ai.v35.model import PlayerV35
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
from qrokkun_ai.v37.run import json_write, ckpt_name
from .protocol import read, read_pinned_json, consume_control, SCHEDULE, FINAL_SEEDS, validate_record, paired
from .adapter import EpisodePool, runtime
from .fingerprint import source_digest

def closure(out, stop=lambda:False):
    m,c,t=[read(out,n) for n in ('manifest.json','closed.json','status.json')]
    s=m['semantic']
    if (s['schema']!='pv39_manifest_v1' or s['nonformal'] or not c['formal']
        or c['completed_updates']!=1000 or c['latest_validated_update']!=1000
        or t['status']!='complete' or t['errors'] or t['owner_stop']
        or c['manifest_sha']!=file_hash(Path(out)/'manifest.json')
        or s['source_tree_sha']!=source_digest() or c['source_tree_sha']!=source_digest()
        or s['checkpoint_boundaries']!=SCHEDULE or s['validation_boundaries']!=SCHEDULE):
        raise ValueError('complete clean Pv39 formal closure required')
    if c['workers_stats']!={'path':'workers.json','sha256':file_hash(Path(out)/'workers.json')}:
        raise ValueError('training worker stats changed')
    workers=read(out,'workers.json')
    from .adapter import validate_worker_coverage
    validate_worker_coverage(workers, s['workers'], c['worker_pids'])
    expected={ckpt_name(u) for u in SCHEDULE}
    if set(c['checkpoint_hashes'])!=expected: raise ValueError('checkpoint coverage')
    totals=[]
    for u in SCHEDULE:
        if stop(): raise InterruptedError('owner stop')
        if file_hash(Path(out)/ckpt_name(u))!=c['checkpoint_hashes'][ckpt_name(u)]: raise ValueError('checkpoint bytes')
        e=read(out,f'evaluation-{u:06d}.json')
        if [r['seed'] for r in e['episodes']]!=list(range(600000,600016)): raise ValueError('validation seeds')
        for r in e['episodes']: validate_record(r)
        total=sum(r['frames'] for r in e['episodes'])
        if total!=e['total_frames']: raise ValueError('validation total')
        totals.append((u,total))
    if max(totals,key=lambda x:x[1])[0]!=c['best_update']: raise ValueError('earliest best')
    for label,u in [('initial',0),('best',c['best_update']),('latest',1000),('terminal',1000)]:
        if read(out,label+'.json')!={'update':u,'checkpoint':ckpt_name(u)}:
            raise ValueError('checkpoint role pointer')
    if stop(): raise InterruptedError('owner stop')
    updates=[__import__('json').loads(line) for line in (Path(out)/'updates.jsonl').read_text().splitlines()]
    if [row['update'] for row in updates]!=list(range(1,1001)): raise ValueError('update coverage')
    from .protocol import training_seeds
    for row in updates:
        seeds=training_seeds(row['update'])
        if (row['seeds']!=seeds or [ep['seed'] for ep in row['episodes']]!=seeds
            or row['attempted_episodes']!=40 or row['completed_episodes']!=40
            or row['frames']!=sum(ep['frames'] for ep in row['episodes'])
            or row['metrics']['n_samples']!=row['frames']): raise ValueError('update episode accounting')
        for ep in row['episodes']: validate_record(ep)
    return m,c

def execute(a):
    from .run import validate_inputs
    scratch=validate_inputs(a)
    out=Path(a.out_dir); errors=[]; stopped=[]; handlers={}; pool=None; owned=False
    def stop(sig,*_):
        stopped.append(sig)
        if pool is not None: pool.stop_event.set()
    def cancelled():
        if __import__('time').time()>=a.deadline_epoch and 'deadline' not in stopped: stopped.append('deadline')
        if Path(a.stop_path).exists() and 'owner STOP' not in stopped: stopped.append('owner STOP')
        if stopped and pool is not None: pool.stop_event.set()
        return bool(stopped)
    def check():
        if cancelled(): raise InterruptedError('owner stop')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,stop)
        m,c=closure(out,cancelled)
        if m['semantic']['source_commit']!=a.source_commit: raise ValueError('source commit')
        if m['semantic'].get('control_pins_sha',file_hash(a.control_pins))!=file_hash(a.control_pins): raise ValueError('control pin file changed')
        if consume_control(a.control_dir,a.control_pins,cancelled)!=m['semantic']['control_pin']: raise ValueError('control changed')
        old=read_pinned_json(a.control_dir,'final.json',m['semantic']['control_pin'],cancelled)['results']
        check(); claim(out/'final-claimed.json',dict(source_commit=a.source_commit)); owned=True
        from qrokkun_ai.v31 import core
        core.configure_runtime('cpu')
        p=random_player()
        import shutil
        local_s=scratch/('pv39-spawner-'+file_hash(a.spawner_checkpoint)+'.pt')
        shutil.copyfile(a.spawner_checkpoint,local_s)
        if file_hash(local_s)!=m['semantic']['inputs']['spawner']: raise ValueError('local S copy hash')
        s=locked_spawner(local_s)
        if file_hash(a.spawner_checkpoint)!=m['semantic']['inputs']['spawner']: raise ValueError('S input changed')
        pool=EpisodePool(p_template=p,s_cpu=s,max_slots=16,max_cap=7200,workers=10,pin_cores='on',search=False,root=out/'final-cache',stats_path=out/'final-workers.json',stop_check=cancelled)
        results={}
        for version,(label,u) in enumerate([('best',c['best_update']),('terminal',1000)]):
            check()
            if label=='terminal' and u==c['best_update']:
                results[label]=dict(reference='best',**results['best'])
            else:
                import shutil
                local=scratch/('pv39-final-'+c['checkpoint_hashes'][ckpt_name(u)]+'.pt')
                shutil.copyfile(out/ckpt_name(u),local)
                if file_hash(local)!=c['checkpoint_hashes'][ckpt_name(u)]: raise ValueError('local candidate hash')
                check()
                payload=torch.load(local,map_location='cpu',weights_only=False)
                if payload['schema']!='pv39_checkpoint_v1' or payload['update']!=u or payload['source_tree_sha']!=source_digest(): raise ValueError('checkpoint identity')
                net=PlayerV35.from_metadata(payload['architecture']); net.load_state_dict(payload['model'],strict=True)
                sha=state_dict_sha256(net.state_dict())
                pool.sync_weights(net.state_dict(),policy_version=version,stop_check=cancelled)
                rows=[]
                for start in range(0,64,16):
                    check()
                    trajs=pool.collect(FINAL_SEEDS[start:start+16],7200,validation=True,stop_check=cancelled)
                    rows.extend(dict(**ep.summary(),cap_frames=7200,state_sha=sha) for ep in trajs)
                for row in rows: validate_record(row)
                if [r['seed'] for r in rows]!=FINAL_SEEDS: raise ValueError('final coverage')
                results[label]=dict(update=u,state_sha=sha,file_sha=file_hash(out/ckpt_name(u)),episodes=rows,
                    player_mode='argmax',spawner_mode='stochastic',mean_seconds=sum(r['frames'] for r in rows)/3840)
            json_write(out/f'final-{label}.json',results[label])
            if read(out,f'final-{label}.json')!=results[label]: raise ValueError('final readback')
        pool.shutdown_and_stats(); pool=None; check()
        if file_hash(a.control_pins)!=m['semantic'].get('control_pins_sha',file_hash(a.control_pins)): raise ValueError('control pin file changed')
        if consume_control(a.control_dir,a.control_pins,cancelled)!=m['semantic']['control_pin']: raise ValueError('control changed')
        check()
        json_write(out/'final.json',dict(status='complete',results=results,spawner_offset=24000000,
            primary=paired(results['best']['episodes'],old['best']['episodes']),
            secondary=paired(results['terminal']['episodes'],old['terminal']['episodes']),
            window='previously seen reference; descriptive update-matched whole-package comparison'))
    except BaseException as exc:
        errors.append(repr(exc)); raise
    finally:
        if pool is not None:
            try: pool.close()
            except BaseException as exc: errors.append('cleanup: '+repr(exc))
        if owned: json_write(out/'final-status.json',dict(status='stopped' if stopped else 'failed' if errors else 'complete',errors=errors,owner_stop=stopped))
        for sig,h in handlers.items(): signal.signal(sig,h)
        if (errors or stopped) and not __import__('sys').exc_info()[0]: raise RuntimeError(str(errors or stopped))
    return 0

def main(argv=None):
    from .run import parser
    a=parser().parse_args(argv)
    if a.smoke: raise ValueError('smoke has no formal final')
    with runtime(): return execute(a)

if __name__=='__main__': raise SystemExit(main())
