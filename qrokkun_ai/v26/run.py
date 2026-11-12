"""Pv26 execution-equivalence arm runner (fresh outputs only)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import time
import numpy as np
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21, state_dict_sha256
from qrokkun_ai.v21.train import train_player_v21 as v21
from qrokkun_ai.v24.core import Spawner
from qrokkun_ai.v25.run import inputs
from qrokkun_ai.v25.core import P_PATH, S_PATH, S_SHA, VALIDATION
from qrokkun_ai.v24.core import PLAYER_SHA, STATE_SHA
from .kernel import global_rng
from .training import collect, train, snapshot
from .trace import Writer, json_write, digest

S_STATE = '3db671ed476886b8e9640beff0e0724a66b1a72263b55be2e988078061e69b63'
HYPER = dict(gamma=.9995,lam=.999,reward_scale=.05,actor_lr=3e-4,critic_lr=1e-3,
             adam_eps=1e-8,epochs=4,minibatch=512,clip=.2,max_grad=1.,dagger=1.,ppo=.3,entropy=.005,hard=.65,soft=.35)


def estimate(hidden=472, episodes=12, cap=7200, updates=20, validation=16):
    # Architecture arithmetic only: two decoupled MLPs, input 84, actor9/value1.
    params = 2*((84+1)*hidden+(hidden+1)*hidden)+(hidden+1)*10
    weight_bytes = params*4
    frames = updates*episodes*cap+3*validation*cap
    steps = updates*4*math.ceil(episodes*cap/512)
    # Exact CAS reuse of unchanged named tensors: per step at most new actor+critic
    # weights, two Adam moments, pre/postclip gradients, plus generous metadata.
    optimizer = steps*(6*weight_bytes+1024**2)
    # A straight bullet of speed>=30 must cross the cull rectangle in at most
    # diagonal/30 seconds (one extra unmoved birth frame). S interval >=.042s,
    # so at most ceil(lifetime/.042)+2 S and ceil(lifetime)+2 field bullets live.
    lifetime = math.hypot(368,288)/30 + 2/60
    live = min(2*cap+1, math.ceil(lifetime/.042)+math.ceil(lifetime)+4)
    # Structured raw bullet rows 50 B, both G boundaries; histories up to all births.
    births = math.ceil(cap/60/.042)+math.ceil(cap/60)+2
    per_frame = 2*live*64 + 2*births*16 + 160000
    physics = frames*per_frame
    batch = updates*episodes*cap*5000
    return dict(estimated_bytes=optimizer+physics+batch+1024**3,model_bytes=weight_bytes,
                max_frames=frames,max_optimizer_steps=steps,optimizer_bytes=optimizer,
                physics_bytes=physics,batch_bytes=batch,compression_credit=0,
                warning='Conservative raw+metadata bound; known unchanged optimizer tensors CAS-deduplicated, no compression credit.')


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=['Pv26c','Pv26t'],required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--trace-dir',required=True)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--player-checkpoint',default=P_PATH)
    ap.add_argument('--spawner-checkpoint',default=S_PATH)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--workers',type=int)
    for k,v in [('updates',20),('episodes',12),('cap',7200),('seed',17),('validation',16),('hidden',472)]:
        ap.add_argument('--'+k,type=int,default=v)
    ap.add_argument('--nonformal',action='store_true')
    ap.add_argument('--synthetic',action='store_true')
    ap.add_argument('--deterministic',action='store_true')
    ap.add_argument('--estimate',action='store_true')
    return ap


def validate(a):
    if not re.fullmatch('[0-9a-fA-F]{40}',a.source_commit):
        raise ValueError('explicit 40-hex source SHA required')
    if not a.nonformal and torch.device(a.device).type!='cuda':
        raise ValueError('formal execution requires CUDA')
    limits = dict(updates=20,episodes=12,cap=7200,seed=17,validation=16,hidden=472)
    for k,v in limits.items():
        n = getattr(a,k)
        if n<=0 or n>v or (not a.nonformal and n!=v):
            raise ValueError('invalid formal/bounded smoke configuration '+k)
    if a.synthetic and not a.nonformal:
        raise ValueError('synthetic is nonformal only')
    workers = a.workers if a.workers is not None else (1 if a.arm=='Pv26c' else 4)
    if (a.arm=='Pv26c' and workers!=1) or workers not in (1,2,4) or (a.arm=='Pv26t' and workers==1):
        raise ValueError('invalid arm/worker mapping')
    if not a.nonformal and workers != (1 if a.arm=='Pv26c' else 4):
        raise ValueError('formal worker mapping')
    return workers


def source_digest():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    paths = sorted([*root.glob('qrokkun_ai/**/*.py'), *root.glob('qrokkun_env/**/*.py')])
    for path in paths:
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def runtime(a):
    return dict(torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
                device=a.device,deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'))


def main(argv=None):
    a = parser().parse_args(argv)
    workers = validate(a)
    storage = estimate(a.hidden,a.episodes,a.cap,a.updates,a.validation)
    if a.estimate:
        print(json.dumps(storage,indent=2)); return 0
    out, trace = Path(a.out_dir).resolve(),Path(a.trace_dir).resolve()
    if out==trace or out in trace.parents or trace in out.parents:
        raise ValueError('output and trace must be separate directories')
    if out.exists() or trace.exists():
        raise FileExistsError('occupied output/trace; no resume or overwrite')
    trace.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(trace.parent).free < storage['estimated_bytes']:
        raise OSError('insufficient disk for conservative full-trace estimate: '+str(storage))
    out.mkdir(parents=True,exist_ok=False)
    json_write(out/'claim.json',dict(arm=a.arm,pid=os.getpid(),source=a.source_commit))
    torch.set_num_threads(1)
    if a.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device.startswith('cuda'):
        torch.cuda.manual_seed_all(a.seed)
    w = Writer(trace,reserve=1024**3)
    handlers = {}
    stopped = [False]
    def stop(sig,frame): stopped[0]=True
    def restore_handlers():
        for sig in list(handlers):
            signal.signal(sig,handlers[sig])
            del handlers[sig]
    def check():
        if stopped[0]: raise InterruptedError('owner stop')
        w.watermark()
    completed = 0
    start = time.perf_counter()
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,stop)
        w.emit('global_init',(0,),global_rng())
        if a.synthetic:
            p = PlayerV21(hidden=a.hidden).to(a.device)
            s = Spawner().requires_grad_(False).eval()
        else:
            p,s,_ = inputs(a.player_checkpoint,a.spawner_checkpoint,a.device)
            if state_dict_sha256(s.state_dict())!=S_STATE:
                raise ValueError('locked S state mismatch')
        oa = torch.optim.Adam(p.actor_parameters(),lr=3e-4,eps=1e-8)
        oc = torch.optim.Adam(p.critic_parameters(),lr=1e-3,eps=1e-8)
        generator = torch.Generator(device=a.device).manual_seed(27000000+a.seed)
        semantic = {k:getattr(a,k) for k in ('source_commit','updates','episodes','cap','seed','validation','nonformal','synthetic','hidden','deterministic')}
        semantic.update(hyperparameters=HYPER,architecture=p.architecture_metadata(),runtime=runtime(a),
                        player_sha=PLAYER_SHA if not a.synthetic else 'synthetic',player_state_sha=STATE_SHA if not a.synthetic else state_dict_sha256(p.state_dict()),
                        spawner_sha=S_SHA if not a.synthetic else 'synthetic',spawner_state_sha=state_dict_sha256(s.state_dict()),
                        validation_seeds=VALIDATION[:a.validation],P_seed_offset=26000000,S_train_offset=25000000,S_val_offset=24000000,
                        permutation_seed=27000000+a.seed,source_tree_sha=source_digest(),resume=False,heldout=[])
        manifest = dict(schema=2,arm=a.arm,workers=workers,semantic=semantic,trace_dir=str(trace),storage=storage,
                        trace_instrumented=True,uninstrumented_speedup_claim=False,trace_durable_bundle=False)
        json_write(out/'manifest.json',manifest)
        w.emit('opponent',(0,),dict(model=s.state_dict(),frozen=True,device='cpu'))
        def save(u):
            check()
            state = snapshot(p,oa,oc,generator)
            w.emit('checkpoint',(u,),state)
            json_write(out/f'checkpoint-{u:02d}.json',w.encode(state))
        def evaluate(u):
            # Preserve all torch globals, including CUDA. No heldout accessor exists.
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                collect(p,s,a.device,VALIDATION[:a.validation],a.cap,workers,w,u,True,check)
        save(0); evaluate(0)
        for u in range(1,a.updates+1):
            check()
            t = time.perf_counter()
            seeds=list(range(100000+(u-1)*a.episodes,100000+u*a.episodes))
            ros=collect(p,s,a.device,seeds,a.cap,workers,w,u,False,check)
            w.emit('rollouts',(u,),dict(flattened_order=[(r.seed,len(r.actions)) for r in ros],rollouts=[{k:np.asarray(v) if isinstance(v,list) else v for k,v in vars(r).items()} for r in ros]))
            batch=v21.rollouts_to_batch_v21(ros,.9995,.999,torch.device(a.device),reward_scale=.05)
            rollout_wall=time.perf_counter()-t
            t=time.perf_counter()
            metrics=train(p,oa,oc,batch,w,u,generator,check)
            ppo_wall=time.perf_counter()-t
            completed=u
            if not a.synthetic and state_dict_sha256(s.state_dict())!=S_STATE:
                raise RuntimeError('S mutated')
            save(u)
            if u in (10,20) or (a.nonformal and u==a.updates): evaluate(u)
            w.emit('update_end',(u,),dict(metrics=metrics,frames=sum(len(r.actions) for r in ros)),
                   dict(rollout_wall_s=rollout_wall,ppo_wall_s=ppo_wall))
            json_write(out/'status.json',dict(status='running',completed_updates=completed))
        w.emit('global_final',(a.updates,),global_rng())
        if source_digest()!=semantic['source_tree_sha']:raise RuntimeError('execution source changed during run')
        check(); w.close()
        trace_bytes=sum(p.stat().st_size for p in trace.iterdir() if p.is_file())
        json_write(out/'closed.json',dict(status='complete',completed_updates=completed,
                   manifest_sha=digest((out/'manifest.json').read_bytes()),trace_closure_sha=digest((trace/'closed.json').read_bytes()),
                   checkpoints={p.name:digest(p.read_bytes()) for p in sorted(out.glob('checkpoint-*.json'))},
                   trace_bytes=trace_bytes,wall_s=time.perf_counter()-start))
        json_write(out/'status.json',dict(status='complete',completed_updates=completed))
        # Completion is provisional until our handlers have been handed back.
        # Stops observed during close/publication/restoration still own this arm.
        restore_handlers()
        if stopped[0]: raise InterruptedError('owner stop during finalization')
        return 0
    except BaseException as e:
        if not w.closed: w.interrupt()
        json_write(out/'status.json',dict(status='stopped' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'failed',
                                        completed_updates=completed,error=repr(e)))
        raise
    finally:
        restore_handlers()


def cli(argv=None):
    from .cli import boundary
    return boundary(main,argv)


if __name__=='__main__':
    raise SystemExit(cli())
