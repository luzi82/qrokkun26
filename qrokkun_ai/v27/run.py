"""Pv27 bounded weights-only serial / pinned spawn experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import platform
import re
import shutil
import signal
import time
import numpy as np
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21,state_dict_sha256
from qrokkun_ai.v24.core import Spawner,file_hash
from qrokkun_ai.v25.run import inputs
from qrokkun_ai.v25.core import P_PATH,S_PATH,VALIDATION
from qrokkun_ai.v26.run import runtime,HYPER,S_STATE
from qrokkun_ai.v26.cli import boundary
from . import core


def json_write(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['Pv27c','Pv27w'],required=True)
    p.add_argument('--out-dir',required=True); p.add_argument('--source-commit',required=True)
    p.add_argument('--player-checkpoint',default=P_PATH); p.add_argument('--spawner-checkpoint',default=S_PATH)
    p.add_argument('--device',default='cuda')
    for k,v in dict(updates=5,episodes=12,cap=7200,seed=17,validation=16,hidden=472).items(): p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--nonformal',action='store_true'); p.add_argument('--synthetic',action='store_true')
    return p


def validate(a):
    if not re.fullmatch('[0-9a-fA-F]{40}',a.source_commit): raise ValueError('explicit source commit required')
    for k,v in dict(updates=5,episodes=12,cap=7200,seed=17,validation=16,hidden=472).items():
        n=getattr(a,k)
        if n<=0 or n>v or (not a.nonformal and n!=v): raise ValueError('invalid '+k)
    if a.synthetic and not a.nonformal: raise ValueError('synthetic requires nonformal')
    if not a.nonformal and a.device!='cuda': raise ValueError('formal requires cuda')
    if not a.synthetic and a.hidden!=472: raise ValueError('locked architecture')


def summary(ros):
    return [dict(seed=r.seed,frames=len(r.actions),death=r.terminated,censored=r.censored,bootstrap=r.bootstrap_value,elapsed=r.elapsed) for r in ros]


def main(argv=None, expected=None):
    a=parser().parse_args(argv); validate(a)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); stopped=[False]; handlers={}; completed=0
    def stop(sig,frame): stopped[0]=True
    def check():
        if stopped[0]: raise InterruptedError('owner stop')
        if shutil.disk_usage(out).free<core.estimate(a.hidden,a.updates): raise OSError('checkpoint free space guard')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,stop)
        check(); torch.set_num_threads(1)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if a.device.startswith('cuda'): torch.cuda.manual_seed_all(a.seed)
        if a.synthetic: p=PlayerV21(hidden=a.hidden).to(a.device); s=Spawner().requires_grad_(False).eval()
        else:
            p,s,_=inputs(a.player_checkpoint,a.spawner_checkpoint,a.device)
            if state_dict_sha256(s.state_dict())!=S_STATE: raise ValueError('locked S state')
        frozen=state_dict_sha256(s.state_dict()); oa,oc=core.optimizers(p)
        g=torch.Generator(device=a.device).manual_seed(27000000+a.seed)
        semantic={k:getattr(a,k) for k in ('source_commit','updates','episodes','cap','seed','validation','hidden','synthetic','nonformal')}
        runtime_info=runtime(a)
        runtime_info.update(python=platform.python_version(),platform=platform.platform(),cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None)
        semantic.update(source_tree_sha=core.source_digest(),runtime=runtime_info,hyperparameters=HYPER,
                        player_state_sha=state_dict_sha256(p.state_dict()),spawner_state_sha=frozen,
                        inputs={} if a.synthetic else dict(player=file_hash(a.player_checkpoint),spawner=file_hash(a.spawner_checkpoint)),
                        fresh_adam=True,validation_seeds=VALIDATION[:a.validation],validation_boundaries=[0,a.updates],
                        terminal_validation_adaptation='Pv26 terminal u20 -> Pv27 terminal u5',P_offset=26000000,S_train_offset=25000000,S_validation_offset=24000000,permutation_seed=27000000+a.seed)
        manifest=dict(arm=a.arm,semantic=semantic,cpu_budget=core.cpu_budget(),workers=min(a.episodes,core.cpu_budget()[0]) if a.arm=='Pv27w' else 0)
        if expected is not None and semantic!=expected: raise ValueError('source/input/runtime mismatch before episodes')
        json_write(out/'manifest.json',manifest)
        def save(u):
            check(); core.finite(p.state_dict())
            tmp=out/'checkpoint.tmp'; torch.save(dict(model=p.state_dict(),actor=oa.state_dict(),critic=oc.state_dict(),training_rng=g.get_state()),tmp); tmp.replace(out/f'checkpoint-{u:02d}.pt')
        def evaluate(u):
            t=time.perf_counter()
            devices=list(range(torch.cuda.device_count())) if a.device.startswith('cuda') else []
            with torch.random.fork_rng(devices=devices):
                ros,workers=core.collect(p,s,a.device,VALIDATION[:a.validation],a.cap,a.arm=='Pv27w',True,check)
            json_write(out/f'evaluation-{u:02d}.json',dict(episodes=summary(ros),wall_s=time.perf_counter()-t,workers=workers))
        save(0); evaluate(0)
        totals=dict(collector_s=0.,ppo_s=0.,frames=0)
        for u in range(1,a.updates+1):
            check(); t=time.perf_counter()
            seeds=list(range(100000+(u-1)*a.episodes,100000+u*a.episodes))
            ros,workers=core.collect(p,s,a.device,seeds,a.cap,a.arm=='Pv27w',check=check)
            batch=core.v21.rollouts_to_batch_v21(ros,.9995,.999,torch.device(a.device),reward_scale=.05)
            collector=time.perf_counter()-t; t=time.perf_counter()
            metrics=core.train(p,oa,oc,batch,generator=g,check=check)
            ppo=time.perf_counter()-t
            if state_dict_sha256(s.state_dict())!=frozen: raise RuntimeError('S mutated')
            save(u); completed=u
            frames=sum(len(r.actions) for r in ros)
            totals['collector_s']+=collector; totals['ppo_s']+=ppo; totals['frames']+=frames
            json_write(out/f'update-{u:02d}.json',dict(episodes=summary(ros),metrics=metrics,collector_s=collector,ppo_s=ppo,frames_per_collector_s=frames/collector,frames_per_ppo_s=frames/ppo,workers=workers))
            json_write(out/'status.json',dict(status='running',completed_updates=u))
        evaluate(a.updates); check()
        if core.source_digest()!=semantic['source_tree_sha']: raise RuntimeError('source changed')
        totals.update(total_s=time.perf_counter()-started,evaluation_s=sum(json.loads(f.read_text())['wall_s'] for f in out.glob('evaluation-*.json')))
        totals['frames_per_total_s']=totals['frames']/totals['total_s']
        json_write(out/'closed.json',dict(completed_updates=completed,timings=totals))
        json_write(out/'status.json',dict(status='complete',completed_updates=completed))
        check(); return 0
    except BaseException as e:
        if stopped[0]:
            json_write(out/'status.json',dict(status='stopped',error=repr(e),completed_updates=completed))
            raise InterruptedError('owner stop') from e
        json_write(out/'status.json',dict(status='stopped' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'failed',error=repr(e),completed_updates=completed)); raise
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)


def cli(argv=None): return boundary(main,argv)
if __name__=='__main__': raise SystemExit(cli())
