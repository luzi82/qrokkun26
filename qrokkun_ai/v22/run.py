"""Local Sv22 PPO runner. See README for formal protocol and recovery limits."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .core import (Spawner, LearnedEnv, load_player, PLAYER_PATH, PLAYER_SHA,
                   STATE_SHA, timed_returns, file_hash)
from qrokkun_env.gv20 import Gv20Env
from qrokkun_ai.v21.obs_v21 import encode_obs_v21

HARD=1794322800
TRAIN_HARD=1794320100
class Stopped(Exception): pass
class Guard:
    def __init__(self,deadline):
        if not math.isfinite(deadline): raise ValueError("finite deadline required")
        self.deadline=min(deadline,HARD); self.reason=None
    def check(self):
        now=time.time()
        if now>=HARD: raise Stopped('deadline')
        if self.reason: raise Stopped(self.reason)
        if not math.isfinite(self.deadline) or now>=self.deadline: raise Stopped('deadline')
    def signal(self,s,*_): self.reason=f'signal-{s}'

def atomic(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)

def json_write(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(obj,indent=2)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

@torch.no_grad()
def player_actions(player,envs,device):
    obs=[encode_obs_v21(e) for e in envs]
    p,b,m=[torch.as_tensor(np.stack(x),device=device) for x in zip(*obs)]
    return player.forward_policy(p,b,m).logits.argmax(-1).cpu().tolist()

def episodes(net,player,seeds,cap,device,guard,mode='stochastic',train_rng=None,on_complete=None):
    """P batches across live episodes each tick; S runs CPU only at spawn events."""
    traces=[[] for _ in seeds]; envs=[]
    for i,seed in enumerate(seeds):
        guard.check()
        rng=train_rng if train_rng is not None else torch.Generator().manual_seed(seed+7000000)
        def callback(obs,i=i,rng=rng):
            guard.check()
            with torch.no_grad():
                x=torch.from_numpy(obs).unsqueeze(0); d=net.distribution(x)
                a=d.mean if mode=='deterministic' else d.mean+d.stddev*torch.randn(d.mean.shape,generator=rng)
                lp=d.log_prob(a).sum(-1).item()
            traces[i].append((envs[i].frame if i<len(envs) else 0,obs,a[0].numpy(),lp))
            return a[0].numpy()
        e=Gv20Env() if mode=='scripted' else LearnedEnv(callback)
        envs.append(e); e.reset(seed)
    active=list(range(len(envs)))
    while active:
        guard.check()
        actions=player_actions(player,[envs[i] for i in active],device)
        for i,a in zip(active,actions):
            envs[i].step(a)
            e=envs[i]
            if (e.dead or e.frame>=cap) and on_complete is not None:
                on_complete(dict(seed=seeds[i],frames=e.frame,seconds=e.frame/60,
                                 death=e.dead,censored=not e.dead,cap_frames=cap))
        active=[i for i in active if not envs[i].dead and envs[i].frame<cap]
    records=[dict(seed=s,frames=e.frame,seconds=e.frame/60,death=e.dead,
                  censored=not e.dead,cap_frames=cap) for s,e in zip(seeds,envs)]
    data=[]
    for trace,e in zip(traces,envs):
        if trace:
            _,returns=timed_returns([t[0] for t in trace],e.frame)
            data.extend((t[1],t[2],t[3],g) for t,g in zip(trace,returns))
    return records,data

def ppo(net,opt,data,guard,epochs,entropy):
    x,a,old,ret=[torch.tensor(np.array(v),dtype=torch.float32) for v in zip(*data)]
    with torch.no_grad(): adv=ret-net.value(x); adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    before=torch.cat([p.detach().flatten().clone() for p in net.parameters()])
    metrics={}
    for _ in range(epochs):
        for ids in torch.randperm(len(x)).split(256):
            guard.check(); d=net.distribution(x[ids]); ratio=(d.log_prob(a[ids]).sum(-1)-old[ids]).exp()
            policy=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
            value=(net.value(x[ids])-ret[ids]).square().mean()
            ent=d.entropy().sum(-1).mean(); loss=policy+.5*value-entropy*ent
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            opt.zero_grad(); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(net.parameters(),.5,error_if_nonfinite=True); opt.step()
            metrics=dict(policy_loss=policy.item(),value_loss=value.item(),entropy=ent.item(),grad_norm=grad.item())
    after=torch.cat([p.detach().flatten() for p in net.parameters()])
    metrics['parameter_delta']=(after-before).norm().item()
    return metrics

def summary(records):
    return dict(mean_seconds=float(np.mean([r['seconds'] for r in records])),
                deaths=sum(r['death'] for r in records),censors=sum(r['censored'] for r in records),per_seed=records)

def compare(a,b):
    assert [r['seed'] for r in a]==[r['seed'] for r in b]
    diff=np.array([x['seconds']-y['seconds'] for x,y in zip(a,b)])
    rng=np.random.default_rng(17); means=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    return dict(mean_difference=float(diff.mean()),ci95=[float(lo),float(hi)],interpretation='evidence of learning' if len(diff)>1 and hi<0 else 'inconclusive')

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','validate','final'])
    p.add_argument('--player-checkpoint',default=PLAYER_PATH); p.add_argument('--expected-player-sha256',default=PLAYER_SHA)
    p.add_argument('--out-dir',required=True); p.add_argument('--device',default='cpu'); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--updates',type=int,default=100000); p.add_argument('--episodes-per-update',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=10); p.add_argument('--eval-seeds',type=int,default=16)
    p.add_argument('--final-seeds',type=int,default=64); p.add_argument('--quick',action='store_true',help='NONFORMAL smoke: 120 frames only')
    p.add_argument('--nonformal',action='store_true',help='explicit nonformal reduced-count run')
    p.add_argument('--deadline-unix','--end-time',type=float,default=HARD,help='absolute Unix hard deadline, capped at 1794322800')
    p.add_argument('--train-deadline-unix',type=float,default=1794320100)
    p.add_argument('--resume-from-run',help='read-only frozen input snapshot (train only)')
    p.add_argument('--threads',type=int,default=1); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--entropy',type=float,default=.001); p.add_argument('--epochs',type=int,default=4)
    return p

def main():
    a=parser().parse_args()
    if not all(math.isfinite(x) for x in (a.deadline_unix,a.train_deadline_unix)): raise ValueError('finite deadlines required')
    if not (a.quick or a.nonformal) and (a.eval_seeds!=16 or a.final_seeds!=64): raise ValueError('formal requires 16 validation and 64 final seeds')
    if not 1<=a.eval_seeds<=16 or not 1<=a.final_seeds<=64: raise ValueError('seed counts exceed preregistered windows')
    if a.updates>100000: raise ValueError('update safety maximum 100000')
    if not a.train_deadline_unix<a.deadline_unix<=HARD or a.train_deadline_unix>TRAIN_HARD: raise ValueError('invalid deadlines')
    if a.resume_from_run and a.command!='train': raise ValueError('resume is train only')
    if min(a.episodes_per_update,a.eval_every,a.epochs,a.updates)<=0: raise ValueError('positive counts required')
    torch.set_num_threads(a.threads)
    guard=Guard(a.deadline_unix)
    for s in (signal.SIGINT,signal.SIGTERM): signal.signal(s,guard.signal)
    try: guard.check()
    except Stopped as e: print(json.dumps(dict(status='not_started',reason=str(e)))); return
    out=Path(a.out_dir); cap=120 if a.quick else 7200
    snapshot=None
    if a.command=='train':
        if a.resume_from_run:
            from .resume import read_snapshot, validate_output
            validate_output(a.resume_from_run,out)
            snapshot=read_snapshot(a.resume_from_run,vars(a),cap)
            if out.resolve().is_relative_to(snapshot['root']): raise ValueError('input must remain read only')
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()): raise ValueError('new empty directory required; resume unsupported, use explicit new variant')
        json_write(out/'manifest.json',dict(config=vars(a),cap_frames=cap,formal=not (a.quick or a.nonformal),player_state=STATE_SHA,protocol=1,**({'lineage':snapshot['lineage']} if snapshot else {})))
    else:
        manifest=json.loads((out/'manifest.json').read_text())
        original=manifest['config']
        formal=manifest.get('formal',False) and manifest['cap_frames']==7200 and original['eval_seeds']==16 and original['final_seeds']==64 and not original.get('quick') and not original.get('nonformal')
        cap=manifest['cap_frames']
        for k in ('quick','nonformal','seed','eval_seeds','final_seeds','player_checkpoint','expected_player_sha256'):
            if getattr(a,k)!=original.get(k,False if k=='nonformal' else None): raise ValueError(f'manifest mismatch: {k}')
    if a.command!='train' and manifest.get('lineage'):
        for k,v in original.items():
            if k not in ('command','out_dir','resume_from_run') and getattr(a,k)!=v:
                raise ValueError(f'continuation manifest mismatch: {k}')
    player,_=load_player(a.player_checkpoint,a.expected_player_sha256,a.device)
    torch.manual_seed(a.seed); net=Spawner(); initial=copy.deepcopy(net).eval()
    opt=torch.optim.Adam(net.parameters(),lr=a.lr); rng=torch.Generator().manual_seed(a.seed)
    started=time.time(); attempt=0
    update=0; best=float('inf'); metrics={}; status='initialized'
    def report(state,**extra):
        target='status.json' if a.command=='train' else a.command+'-status.json'
        json_write(out/target,dict(status=state,update=update,attempted_update=attempt,
            wallclock=time.time(),elapsed=time.time()-started,metrics=metrics,**extra,formal=(not (a.quick or a.nonformal) if a.command=='train' else formal)))
    def progress():
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(wallclock=time.time(),elapsed=time.time()-started,update=update,inherited_updates=inherited,locally_completed_updates=update-inherited,
                loss={k:v for k,v in metrics.items() if k not in ('rollout','validation')},
                rollout=metrics['rollout'],validation=metrics.get('validation')))+'\n')
            f.flush(); os.fsync(f.fileno())
    def save(name):
        atomic(out/name,dict(model=net.state_dict(),optimizer=opt.state_dict(),update=update,
            attempted_update=attempt,partial_update=attempt>update,
            train_rng=rng.get_state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            manifest=json.loads((out/'manifest.json').read_text()),status=status,metrics=metrics,
            player_sha=PLAYER_SHA,player_state=STATE_SHA,resume_supported=False))
    inherited=0
    if snapshot:
        from .resume import restore, copy_controls
        restore(snapshot,net,initial,opt,rng)
        copy_controls(snapshot,out)
        json_write(out/'parent-lineage.json',snapshot['lineage'])
        latest=snapshot['checkpoints']['latest.pt']
        update=attempt=inherited=latest['update']; best=snapshot['best']
        metrics=latest['metrics']; status='resumed'
    evaluation_index=0
    def evaluate(model,seeds,mode='stochastic',label=None):
        nonlocal evaluation_index
        evaluation_index+=1
        label=label or f'{a.command}-{evaluation_index}-{mode}'
        seeds=list(seeds); completed=[]
        def metadata(state,reason=None):
            done={r['seed'] for r in completed}
            json_write(out/f'{label}-completion.json',dict(status=state,mode=mode,
                completed_seeds=[r['seed'] for r in completed],
                unfinished_seeds=[s for s in seeds if s not in done],reason=reason))
        def persist(record):
            with (out/f'{label}-episodes.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
            completed.append(record)
            metadata('running')
        metadata('running')
        try:
            records=episodes(model,player,seeds,cap,a.device,guard,mode,on_complete=persist)[0]
            metadata('complete')
            return records
        except Stopped as e:
            metadata('incomplete',str(e))
            raise
    if a.command=='train':
        if not snapshot: save('initial.pt')
        save('latest.pt')
        try:
            guard.deadline=min(a.train_deadline_unix,a.deadline_unix,HARD)
            report('baseline')
            controls=snapshot['controls'] if snapshot else {mode:summary(evaluate(initial,range(400000,400000+a.eval_seeds),mode)) for mode in ('scripted','stochastic')}
            if not snapshot: json_write(out/'validation-controls.json',controls)
            for attempt in range(update+1,a.updates+1):
                report('rollout')
                guard.check(); records,data=episodes(net,player,range(1000000+(attempt-1)*a.episodes_per_update,1000000+attempt*a.episodes_per_update),cap,a.device,guard,train_rng=rng)
                report('ppo')
                metrics=ppo(net,opt,data,guard,a.epochs,a.entropy)
                update=attempt
                metrics['rollout']=summary(records)
                metrics['frozen_player']=dict(file_sha=PLAYER_SHA,state_sha=STATE_SHA,update=400,no_grad=all(p.grad is None and not p.requires_grad for p in player.parameters()))
                metrics['full_cap']=not a.quick
                status='trained'; save('latest.pt')
                try:
                    if update%a.eval_every==0 or update==1:
                        report('validation')
                        val=summary(evaluate(net,range(400000,400000+a.eval_seeds)))
                        metrics['validation']=val
                        if val['mean_seconds']<best:
                            best=val['mean_seconds']; save('best.pt')
                        save('latest.pt'); save(f'update-{update}.pt')
                finally: progress()
                report('trained')
                print(json.dumps(dict(update=update,latest='latest.pt',best_mean=best,formal=not (a.quick or a.nonformal),**metrics)),flush=True)
            status='updates_complete'
        except Stopped as e: status='stopped:'+str(e)
        finally:
            save('latest.pt')
            if (out/'best.pt').exists():
                json_write(out/'training-closed.json',dict(best_sha=file_hash(out/'best.pt'),status=status))
            report(status)
        print(status)
    else:
        if a.command=='final':
            closed=json.loads((out/'training-closed.json').read_text())
            if file_hash(out/'best.pt')!=closed['best_sha']: raise ValueError('locked best changed')
        # Atomic rename by trainer means this open reads one immutable snapshot.
        with (out/'best.pt').open('rb') as snapshot:
            best_snapshot=torch.load(snapshot,weights_only=False)
        update=best_snapshot['update']
        net.load_state_dict(best_snapshot['model']); net.eval()
        initial.load_state_dict(torch.load(out/'initial.pt',weights_only=False)['model'])
        if a.command=='validate':
            report('validation')
            try:
                result=summary(evaluate(net,range(400000,400000+a.eval_seeds)))
                report('validation_complete')
                print(json.dumps(result))
            except Stopped as e:
                report('validation_incomplete:'+str(e))
                print('validation incomplete: '+str(e))
        else:
            # Exclusive claim BEFORE seed access. Interrupted final is never rerun/overwritten.
            with (out/'final-claimed.json').open('x') as f: json.dump(closed,f)
            try:
                seeds=range(500000,500000+a.final_seeds)
                results={}
                for pending in ('scripted','initial','learned'):
                    json_write(out/f'final-{pending}-completion.json',dict(status='not_started',
                        mode='scripted' if pending=='scripted' else 'stochastic',
                        completed_seeds=[],unfinished_seeds=list(seeds),reason=None))
                for label,model,mode in [('scripted',initial,'scripted'),('initial',initial,'stochastic'),('learned',net,'stochastic')]:
                    report('final_'+label)
                    results[label]=summary(evaluate(model,seeds,mode,label="final-"+label)); json_write(out/f'final-{label}.json',results[label])
                results['comparisons']={label:compare(results['learned']['per_seed'],results[label]['per_seed']) for label in ('scripted','initial')}
                if not formal:
                    for comparison in results['comparisons'].values(): comparison['interpretation']='inconclusive'
                results['conclusion']='evidence of learning against both controls' if formal and all(c['ci95'][1]<0 for c in results['comparisons'].values()) else 'inconclusive'
                results['warning']='Deterministic trajectories may repeat; seeds only matter through actual randomness. Quick runs nonformal.'
                results['formal']=formal
                json_write(out/'final.json',results)
            except Stopped as e:
                json_write(out/'final-incomplete.json',dict(mode=label,reason=str(e)))
                report('primary_incomplete',mode=label,reason=str(e))
                return
            report('primary_complete_diagnostic_running')
            try:
                diagnostic=summary(evaluate(net,seeds,'deterministic'))
                json_write(out/'final-deterministic_diagnostic.json',diagnostic)
                report('primary_complete_diagnostic_complete')
            except Stopped as e:
                json_write(out/'final-diagnostic-incomplete.json',dict(reason=str(e)))
                report('primary_complete_diagnostic_incomplete')

if __name__=='__main__': main()
