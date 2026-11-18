"""Version-local orchestration around unchanged Pv31 PPO and Pv37 teaching."""
import argparse
import json
import random
import os
import shutil
import signal
import time
from pathlib import Path
import numpy as np
import torch
from qrokkun_ai.v31 import core
from qrokkun_ai.v35.initialization import random_player, locked_spawner
from qrokkun_ai.v37.teaching import auxiliary
from qrokkun_ai.v37.run import json_write, ckpt_name, sync_cuda
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
from .adapter import EpisodePool, runtime
from .fingerprint import source_digest
from .protocol import archive_source, consume_control, training_seeds

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir','source-commit','spawner-checkpoint','control-dir','control-pins','scratch','stop-path'): p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-epoch',required=True,type=float)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',default='cuda')
    return p

def validate_inputs(a):
    import math
    scratch=Path(a.scratch).resolve()
    if not scratch.is_dir() or str(scratch).startswith('/mnt/'):
        raise ValueError('scratch must be an existing local directory')
    if not math.isfinite(a.deadline_epoch) or not 0<a.deadline_epoch-time.time()<=57600:
        raise ValueError('finite future deadline within16h required')
    if Path(a.stop_path).exists(): raise InterruptedError('owner STOP')
    if not a.smoke and a.device!='cuda': raise ValueError('formal requires CUDA')
    return scratch

def adam_steps(optimizer):
    return max((int(x['step']) for x in optimizer.state.values()),default=0)

def execute(a):
    scratch=validate_inputs(a)
    total_start=time.perf_counter()
    timing=dict(checkpoint_s=0.,validation_s=0.)
    ordinary_actor=ordinary_critic=aux_actor=aux_critic=0

    updates,episodes,cap,workers,nval,cadence=(2,3,180,2,2,1) if a.smoke else (1000,40,7200,10,16,10)
    if not a.smoke and a.device!='cuda': raise ValueError('formal requires CUDA')
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=False)
    stop=[]; errors=[]; pool=None; completed=0; handlers={}
    last_stop_poll=0.
    original_threads=torch.get_num_threads()
    original_affinity=os.sched_getaffinity(0)
    def stopped():
        nonlocal last_stop_poll
        if time.time()>=a.deadline_epoch and 'deadline' not in stop: stop.append('deadline')
        now=time.monotonic()
        if now-last_stop_poll>=.25:
            last_stop_poll=now
            if Path(a.stop_path).exists() and 'owner STOP' not in stop: stop.append('owner STOP')
        if stop and pool is not None: pool.stop_event.set()
        return bool(stop)
    def check():
        if stopped(): raise InterruptedError('owner stop')
    def on_stop(sig,*_):
        stop.append(sig)
        if pool is not None: pool.stop_event.set()
    def status(state):
        json_write(out/'status.json',dict(status=state,completed_updates=completed,target_updates=updates,errors=errors,owner_stop=stop))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,on_stop)
        archive_source(a.source_commit,out/'source.tar')
        pins=consume_control(a.control_dir,a.control_pins,stopped); check()
        timing['preflight_s']=time.perf_counter()-total_start
        startup_start=time.perf_counter()
        core.configure_runtime(a.device)
        with torch.random.fork_rng(devices=[]): p=random_player().to(a.device)
        from qrokkun_ai.v24.core import file_hash
        local_s=scratch/('pv39-spawner-'+file_hash(a.spawner_checkpoint)+'.pt')
        if not local_s.exists(): shutil.copyfile(a.spawner_checkpoint,local_s)
        if file_hash(local_s)!=file_hash(a.spawner_checkpoint): raise ValueError('local S copy hash')
        s=locked_spawner(local_s)
        initial=state_dict_sha256(p.state_dict()); frozen=state_dict_sha256(s.state_dict())
        from .protocol import read
        contract=read(Path(__file__).parent,'contract.json')
        if initial!=contract['initial_state_sha']: raise ValueError('untrained u0 full state mismatch')
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        oa,oc=core.optimizers(p); gen=torch.Generator(device=a.device).manual_seed(core.PERM_BASE+17)
        boundaries=sorted({0,*range(cadence,updates+1,cadence),updates})
        from qrokkun_ai.v26.run import HYPER
        from qrokkun_ai.v24.core import file_hash
        semantic=dict(hyperparameters=HYPER,inputs={'spawner':file_hash(a.spawner_checkpoint)},architecture=p.architecture_metadata(),fresh_adam=True,schema='pv39_smoke_v1' if a.smoke else 'pv39_manifest_v1',nonformal=a.smoke,
            source_commit=a.source_commit,source_tree_sha=source_digest(),updates=updates,episodes=episodes,cap=cap,workers=workers,
            seed=17,hidden=472,validation=nval,checkpoint_boundaries=boundaries,validation_boundaries=boundaries,
            initial_player_state_sha=initial,spawner_state_sha=frozen,control_pin=pins,
            control_pins_path=str(Path(a.control_pins).resolve()),control_pins_sha=file_hash(a.control_pins),difficulty_train=40,difficulty_validation=0,budget='update matched; E40 vs old E20 seed windows differ')
        json_write(out/'manifest.json',dict(arm='Pv39',semantic=semantic,source_archive_sha=file_hash(out/'source.tar')))
        pool=EpisodePool(search=True,root=out/'cache',stats_path=out/'workers.json',p_template=p,s_cpu=s,max_slots=max(episodes,nval),
            max_cap=cap,workers=workers,pin_cores='off' if a.smoke else 'on',stop_check=stopped)
        learner_cpu=core.select_learner_cpu('off' if a.smoke else 'on',strict=not a.smoke)
        if learner_cpu is not None: os.sched_setaffinity(0,{int(learner_cpu)})
        best,best_total,latest=0,-1,-1
        def boundary(u):
            nonlocal best,best_total,latest
            check()
            needed=core.estimate_bytes(472,sum(v>=u for v in boundaries))
            if shutil.disk_usage(out).free<needed: raise OSError('checkpoint disk space guard')
            checkpoint_start=time.perf_counter()
            core.finite(p.state_dict())
            torch.save(dict(schema='pv39_checkpoint_v1',arm='Pv39',update=u,architecture=p.architecture_metadata(),
                source_tree_sha=semantic['source_tree_sha'],model={k:v.detach().cpu() for k,v in p.state_dict().items()},
                actor=oa.state_dict(),critic=oc.state_dict(),training_rng=gen.get_state(),
                rng_metadata=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                    torch_cuda=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [])),out/ckpt_name(u))
            timing['checkpoint_s']+=time.perf_counter()-checkpoint_start
            validation_start=time.perf_counter()
            rows=pool.collect(list(range(600000,600000+nval)),cap,validation=True,stop_check=stopped)
            if [ep.summary()['seed'] for ep in rows]!=list(range(600000,600000+nval)): raise ValueError('exact validation coverage')
            total=sum(ep.frames for ep in rows)
            if total>best_total: best,best_total=u,total
            if u==0 and not a.smoke:
                from .protocol import read
                old=read(a.control_dir,'evaluation-000000.json')['episodes']
                observed=[dict(**ep.summary(),cap_frames=cap) for ep in rows]
                if len(old)!=16 or len(observed)!=16: raise ValueError('exact normal C0 coverage')
                for candidate,control in zip(observed,old):
                    if any(candidate[k]!=control[k] for k in ('seed','frames','death','censored')):
                        raise ValueError('normal u0 differs from historical control before optimization')
            timing['validation_s']+=time.perf_counter()-validation_start
            latest=u
            json_write(out/f'evaluation-{u:06d}.json',dict(update=u,total_frames=total,episodes=[dict(**ep.summary(),cap_frames=cap) for ep in rows]))
            for label,v in [('initial',0),('best',best),('latest',u)]+([('terminal',u)] if u==updates else []):
                json_write(out/(label+'.json'),dict(update=v,checkpoint=ckpt_name(v)))
        timing['startup_s']=time.perf_counter()-startup_start
        pool.sync_weights(p.state_dict(),policy_version=0,stop_check=stopped); boundary(0)
        for u in range(1,updates+1):
            check(); start=time.perf_counter()
            seeds=training_seeds(u) if not a.smoke else list(range(100000+(u-1)*episodes,100000+u*episodes))
            json_write(out/'attempt.json',dict(update=u,seeds=seeds,attempted_episodes=len(seeds),completed_episodes=None,status='collecting'))
            trajs=pool.collect(seeds,cap,validation=False,stop_check=stopped)
            json_write(out/'attempt.json',dict(update=u,seeds=seeds,attempted_episodes=len(seeds),completed_episodes=len(trajs),status='collected'))
            if [ep.summary()['seed'] for ep in trajs]!=seeds: raise ValueError('exact training episode coverage')
            rollout_s=time.perf_counter()-start
            batch_start=time.perf_counter()
            batch=core.assemble_batch(trajs,p,a.device)
            batch_s=time.perf_counter()-batch_start
            before=(adam_steps(oa),adam_steps(oc)); ppo_start=time.perf_counter()
            metrics=core.train(p,oa,oc,batch,generator=gen,stop_check=stopped); check()
            sync_cuda(a.device)
            after_ppo=(adam_steps(oa),adam_steps(oc)); ppo_s=time.perf_counter()-ppo_start
            aux_start=time.perf_counter()
            metrics['auxiliary']=auxiliary(p,oa,[r for ep in trajs for r in ep.demos],u)
            sync_cuda(a.device)
            aux_s=time.perf_counter()-aux_start
            after_aux=(adam_steps(oa),adam_steps(oc))
            ordinary_actor+=after_ppo[0]-before[0]; ordinary_critic+=after_ppo[1]-before[1]
            aux_actor+=after_aux[0]-after_ppo[0]; aux_critic+=after_aux[1]-after_ppo[1]
            if aux_actor>u or after_aux[1]!=after_ppo[1]: raise ValueError('aux optimizer accounting')
            expected_steps=core.v21.PPO_EPOCHS*((metrics['n_samples']+core.v21.PPO_MINIBATCH-1)//core.v21.PPO_MINIBATCH)
            if (after_ppo[0]-before[0],after_ppo[1]-before[1])!=(expected_steps,expected_steps): raise ValueError('ordinary optimizer step accounting')
            if metrics['n_samples']!=sum(ep.frames for ep in trajs): raise ValueError('sample/frame accounting')
            if state_dict_sha256(s.state_dict())!=frozen: raise RuntimeError('frozen S mutated')
            pool.sync_weights({k:v.detach().cpu() for k,v in p.state_dict().items()},policy_version=u,stop_check=stopped)
            search_timing={key:sum(ep.teaching[key] for ep in trajs if key in ep.teaching) for key in ('search_s','replay_s','validate_s')}
            for key,value in dict(rollout_s=rollout_s,batch_s=batch_s,ppo_s=ppo_s,aux_s=aux_s,**search_timing).items(): timing[key]=timing.get(key,0.)+value
            with (out/'updates.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=u,frames=sum(ep.frames for ep in trajs),wall_s=time.perf_counter()-start,
                    seeds=seeds,episodes=[dict(**ep.summary(),cap_frames=cap) for ep in trajs],attempted_episodes=len(seeds),completed_episodes=len(trajs),
                    epochs=core.v21.PPO_EPOCHS,minibatch=core.v21.PPO_MINIBATCH,expected_ordinary_steps=core.v21.PPO_EPOCHS*((metrics['n_samples']+core.v21.PPO_MINIBATCH-1)//core.v21.PPO_MINIBATCH),
                    timings=dict(rollout_s=rollout_s,batch_s=batch_s,ppo_s=ppo_s,aux_s=aux_s,**search_timing),
                    ordinary_actor_delta=after_ppo[0]-before[0],ordinary_critic_delta=after_ppo[1]-before[1],
                    auxiliary_actor_delta=after_aux[0]-after_ppo[0],auxiliary_critic_delta=after_aux[1]-after_ppo[1],
                    ordinary_actor_cumulative=ordinary_actor,ordinary_critic_cumulative=ordinary_critic,auxiliary_actor_cumulative=aux_actor,auxiliary_critic_cumulative=aux_critic,actor_total=after_aux[0],critic_total=after_aux[1],metrics=metrics,teaching=[ep.teaching for ep in trajs]),allow_nan=False)+'\n')
            completed=u
            if u in boundaries: boundary(u)
            status('running')
        check(); cleanup_start=time.perf_counter()
        stats=pool.shutdown_and_stats(); json_write(out/'workers.json',stats); pool=None; check()
        timing['cleanup_s']=time.perf_counter()-cleanup_start
        timing['total_s']=time.perf_counter()-total_start
        json_write(out/'timing.json',timing)
        if source_digest()!=semantic['source_tree_sha']: raise RuntimeError('production source changed')
        from qrokkun_ai.v24.core import file_hash
        if file_hash(a.control_pins)!=semantic['control_pins_sha']: raise ValueError('control pin file changed')
        if consume_control(a.control_dir,a.control_pins,stopped)!=pins: raise ValueError('control changed')
        check()
        json_write(out/'closed.json',dict(formal=not a.smoke,completed_updates=completed,latest_validated_update=latest,
            workers_stats=dict(path='workers.json',sha256=file_hash(out/'workers.json')),
            worker_pids=[row['pid'] for row in stats],
            best_update=best,source_tree_sha=semantic['source_tree_sha'],manifest_sha=file_hash(out/'manifest.json'),
            checkpoint_hashes={ckpt_name(u):file_hash(out/ckpt_name(u)) for u in boundaries}))
        status('complete')
    except BaseException as exc:
        errors.append(repr(exc)); raise
    finally:
        for cleanup in (lambda:pool.close() if pool is not None else None,lambda:torch.set_num_threads(original_threads),lambda:os.sched_setaffinity(0,original_affinity)):
            try: cleanup()
            except BaseException as exc: errors.append('cleanup: '+repr(exc))
        if errors or stop: status('stopped' if stop else 'failed')
        for sig,h in handlers.items(): signal.signal(sig,h)
        if (errors or stop) and not __import__('sys').exc_info()[0]: raise RuntimeError(str(errors or stop))
    return 0

def main(argv=None):
    a=parser().parse_args(argv)
    with runtime(search=True): return execute(a)

if __name__=='__main__': raise SystemExit(main())
