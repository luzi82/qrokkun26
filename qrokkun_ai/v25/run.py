"""Owner-authorized Pv25 C experiment; fresh lineage, no resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import uuid
import numpy as np
import torch
from qrokkun_ai.v24 import run as v24
from qrokkun_ai.v24.core import load_player, load_spawner, file_hash, PLAYER_SHA, STATE_SHA, ARCHITECTURE
from qrokkun_ai.v21.agents.player_v21 import load_v21_checkpoint, state_dict_sha256
from qrokkun_ai.v21.train import train_player_v21 as v21
from .core import S_SHA, S_PATH, P_PATH, VALIDATION, HELDOUT, Selection, historical_gate, REFERENCE, REFERENCE_SHA
from .training import collect, optimizers, evaluate, check_frozen

Guard, Stopped = v24.Guard, v24.Stopped
def durable_directory(path):
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, obj):
    v24.atomic(path, obj)
    durable_directory(path)


def json_write(path, obj):
    v24.json_write(path, obj)
    durable_directory(path)


def append(path, obj):
    v24.append(path, obj)
    durable_directory(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'final'])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--source-commit', type=v24.source_commit_literal)
    p.add_argument('--player-checkpoint', default=P_PATH)
    p.add_argument('--spawner-checkpoint', default=S_PATH)
    p.add_argument('--device', default='cpu')
    for name, default in [('seed',17),('updates',1000),('episodes-per-update',12),('max-steps',7200),('eval-seeds',16),('final-seeds',64)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--nonformal', action='store_true')
    return p


def validate_args(a):
    for k, v in dict(seed=17, updates=1000, episodes_per_update=12, max_steps=7200, eval_seeds=16, final_seeds=64).items():
        n = getattr(a, k)
        if n <= 0 or (not a.nonformal and n != v):
            raise ValueError('invalid configuration: '+k)
    if a.max_steps > 7200 or a.eval_seeds > 16 or a.final_seeds > 64:
        raise ValueError('seed/cap bounds')
    if a.command == 'final' and a.nonformal:
        raise ValueError('nonformal has no final accessor')
    Guard(a.deadline_unix)
    if a.source_commit is not None:
        v24.source_commit_literal(a.source_commit)


def inputs(player_path, spawner_path, device):
    if file_hash(player_path) != PLAYER_SHA or file_hash(spawner_path) != S_SHA:
        raise ValueError('locked input file hash mismatch')
    p, pm = load_player(player_path, device=device)
    s, sm = load_spawner(spawner_path)
    if (sm.get('completed_updates') != 980 or sm.get('attempted_updates') != 980 or sm.get('partial_update') is not False or
        sm.get('player_sha') != PLAYER_SHA or sm.get('player_state_sha') != STATE_SHA or
        sm.get('architecture') != ARCHITECTURE):
        raise ValueError('Sv24 metadata/source Pv21 mismatch')
    for k,v in p.architecture_metadata().items():
        if pm.get(k) != v and k != 'param_count':
            raise ValueError('inherited architecture mismatch: '+k)
    p.requires_grad_(True)
    s.requires_grad_(False).eval()
    return p, s, pm


def config(a):
    return {k:v for k,v in vars(a).items() if k != 'command'}


def claim(path, value):
    # Exclusive durable claim is never removed, including failures.
    with Path(path).open('x') as f:
        json.dump(value, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    durable_directory(path)


def preflight_final(a):
    validate_args(a)
    out = Path(a.out_dir)
    m = json.loads((out/'manifest.json').read_text())
    c = json.loads((out/'training-closed.json').read_text())
    training_status = json.loads((out/'status.json').read_text())
    if training_status.get('status') != 'complete':
        raise ValueError('training status is not complete')
    if (not m['formal'] or c['status'] != 'complete' or c['completed_updates'] != 1000 or
        c['last_complete_validation'] != 1000 or c['partial_update'] or
        c['manifest_sha'] != file_hash(out/'manifest.json') or m['config'] != config(a) or
        m['source_commit'] != v24.resolve_source_commit(a.source_commit)):
        raise ValueError('final closure/config/source mismatch')
    if (set(c['checkpoint_hashes']) != {'best.pt','latest.pt','initial.pt','update500.pt','update1000.pt'} or
        c.get('u0_reference_sha') != REFERENCE_SHA or m.get('player_sha') != PLAYER_SHA or
        m.get('player_state_sha') != STATE_SHA or m.get('spawner_sha') != S_SHA or m.get('heldout') != HELDOUT):
        raise ValueError('final identity/closure mismatch')
    train_claim = json.loads((out/'train-claimed.json').read_text())
    if not (train_claim['id'] == m.get('run_id') == c.get('run_id')):
        raise ValueError('training claim identity mismatch')
    if (out/'final-claimed.json').exists():
        raise ValueError('final already claimed; no repeat')
    allowed = {'train-claimed.json','manifest.json','status.json','progress.jsonl','training-closed.json',
               'update500-summary.json','update1000-summary.json',*c['checkpoint_hashes']}
    for u in range(0,1001,10):
        allowed.update({f'validation-{u}.json',f'validation-{u}-completion.json',f'validation-{u}-episodes.jsonl'})
    if any(p.name not in allowed or not p.is_file() for p in out.iterdir()):
        raise ValueError('occupied output contains foreign/final artifacts')
    for name, digest in c['checkpoint_hashes'].items():
        if file_hash(out/name) != digest:
            raise ValueError('closed checkpoint hash mismatch')
    if (out/'final-claimed.json').exists():
        raise ValueError('final already claimed; no repeat')
    return m, c


def validation_mean(result):
    return sum(r['frames'] for r in result['per_seed']) / (60 * len(result['per_seed']))


def paired(a,b):
    if [r['seed'] for r in a] != HELDOUT or [r['seed'] for r in b] != HELDOUT:
        raise ValueError('exact 64 unique paired seeds required')
    d = np.array([x['seconds']-y['seconds'] for x,y in zip(a,b)])
    rng = np.random.default_rng(17)
    ci = np.quantile(d[rng.integers(64,size=(10000,64))].mean(1),[.025,.975])
    return dict(mean_difference=float(d.mean()), ci95=ci.tolist(), bootstrap_seed=17,
                interpretation='indistinguishable' if ci[0] <= 0 <= ci[1] else ('Pv25 longer survival' if ci[0]>0 else 'Pv25 shorter survival'))


def main(a=None):
    a = a or parser().parse_args()
    validate_args(a)
    torch.set_num_threads(1)
    guard = Guard(a.deadline_unix)
    out = Path(a.out_dir)
    handlers = {}
    final_claimed = False
    if a.command == 'final':
        preflight_final(a)
    p, s, pm = inputs(a.player_checkpoint, a.spawner_checkpoint, a.device)
    sdigest = state_dict_sha256(s.state_dict())
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    oa, oc = optimizers(p, guard)
    selection = Selection()
    completed = attempted = 0
    lastval = None
    partial = False
    phase = 'initializing'
    def status(state, error=None):
        if state in ('running','complete','nonformal_complete'):
            guard.check()
        json_write(out/('status.json' if a.command=='train' else 'final-status.json'),
                   dict(status=state, phase=phase, completed_updates=completed, attempted_updates=attempted,
                        last_complete_validation=lastval, validation_pending=lastval is None or completed>lastval,
                        partial_update=partial, formal=not a.nonformal, error=error, **selection.report()))
    def save(name):
        guard_state = dict(completed_updates=completed, attempted_updates=attempted,
                           last_complete_validation=lastval, partial_update=partial)
        state = {k:v.detach().cpu() for k,v in p.state_dict().items()}
        atomic(out/name, dict(**p.architecture_metadata(), checkpoint_type='pv25_warm_start_v1',formal=not a.nonformal,
               state_dict=state, state_dict_sha256=state_dict_sha256(state), extra=dict(update=completed),
               lineage=dict(init_file_sha=PLAYER_SHA,init_state_sha=STATE_SHA,parent_update=400),
               **guard_state, selection=selection.report(), spawner_sha=S_SHA, spawner_state_sha=sdigest,
               optimizer_actor=oa.state_dict(),optimizer_critic=oc.state_dict(),
               torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               numpy_rng=np.random.get_state(),resume_supported=False))
    def ev(model,seeds,label):
        done=[]
        def seal(state,error=None):
            guard.check() if state=='complete' else None
            json_write(out/(label+'-completion.json'),dict(status=state,completed_seeds=[r['seed'] for r in done],
                       unfinished_seeds=[x for x in seeds if x not in {r['seed'] for r in done}],error=error))
        def persist(r):
            guard.check()
            append(out/(label+'-episodes.jsonl'),r)
            done.append(r)
            seal('running')
        try:
            result=evaluate(model,s,seeds,a.max_steps,a.device,guard,persist)
            check_frozen(s,sdigest)
            if len(done)!=len(seeds) or len({r['seed'] for r in done})!=len(seeds):
                raise ValueError('incomplete/duplicate episodes')
            guard.check()
            json_write(out/(label+'.json'),result)
            seal('complete')
            return result
        except BaseException as e:
            seal('incomplete',str(e))
            raise
    if a.command=='train':
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()):
            raise ValueError('empty fresh output required; no resume')
        run_id=uuid.uuid4().hex
        claim(out/'train-claimed.json',dict(id=run_id,pid=os.getpid()))
        json_write(out/'manifest.json',dict(config=config(a),formal=not a.nonformal,run_id=run_id,
                   source_commit=v24.resolve_source_commit(a.source_commit),player_sha=PLAYER_SHA,player_state_sha=STATE_SHA,
                   spawner_sha=S_SHA, reference=REFERENCE, reference_sha=REFERENCE_SHA, inherited_architecture=p.architecture_metadata(),
                   train_rng_offset=25000000,eval_rng_offset=24000000,heldout=HELDOUT,
                   hyperparameters=dict(gamma=.9995,lam=.999,reward_scale=.05,actor_lr=.0003,critic_lr=.001,
                    ppo_epochs=4,clip=.2,minibatch=512,norm=1,entropy=.005,dagger=1,ppo=.3), resume_supported=False))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,guard.signal)
        guard.check()
        if a.command=='final':
            m,c=preflight_final(a)
            claim(out/'final-claimed.json',dict(manifest_sha=file_hash(out/'manifest.json'),closure_sha=file_hash(out/'training-closed.json'),seeds=HELDOUT))
            final_claimed = True
            guard.check()
            learned, ck=load_v21_checkpoint(out/'best.pt',a.device)
            if ck.get('formal') is not True or ck.get('partial_update') or ck.get('checkpoint_type')!='pv25_warm_start_v1':
                raise ValueError('invalid selected checkpoint')
            completed=attempted=lastval=1000
            selection=Selection(**ck['selection'])
            results={}
            for label,model in [('learned',learned),('original',p)]:
                phase='final_'+label
                status('running')
                results[label]=ev(model,HELDOUT,'final-'+label)
            results['comparison']=paired(results['learned']['per_seed'],results['original']['per_seed'])
            guard.check()
            json_write(out/'final.json',results)
            status('complete')
            return
        phase='u0_validation'
        status('running')
        val=ev(p,VALIDATION[:a.eval_seeds],'validation-0')
        if not a.nonformal: historical_gate(val['per_seed'])
        # Canonical integer frame mean avoids a floating-point false crossing at u0.
        mean=validation_mean(val)
        lastval=0
        selection.observe(0,mean)
        save('initial.pt'); save('best.pt'); save('latest.pt')
        for u in range(1,a.updates+1):
            attempted=u
            phase='rollout'; status('running'); guard.check()
            seeds=list(range(100000+(u-1)*a.episodes_per_update,100000+u*a.episodes_per_update))
            ro=collect(p,s,torch.device(a.device),seeds,a.max_steps,guard)
            batch=v21.rollouts_to_batch_v21(ro,.9995,.999,torch.device(a.device),reward_scale=.05)
            phase='ppo'; partial=True; status('running'); guard.check()
            metrics=v21.train_step_v21(p,oa,oc,batch)
            metrics['rollout']=[dict(seed=r.seed,elapsed=r.elapsed,terminated=r.terminated,truncated=r.truncated,censored=r.censored) for r in ro]
            completed=u; partial=False
            check_frozen(s,sdigest)
            save('latest.pt')
            if u%10==0 or u==a.updates:
                phase='validation'; status('running')
                val=ev(p,VALIDATION[:a.eval_seeds],f'validation-{u}')
                lastval=u
                mean=validation_mean(val)
                if selection.observe(u,mean): save('best.pt')
                metrics['validation']=val
                save('latest.pt')
            if not a.nonformal and u in (500,1000):
                save(f'update{u}.pt')
                json_write(out/f'update{u}-summary.json',dict(**selection.report(),latest_update=u,terminal_validation=val,best_sha=file_hash(out/'best.pt'),latest_sha=file_hash(out/'latest.pt')))
            append(out/'progress.jsonl',dict(completed_updates=completed,attempted_updates=attempted,metrics=metrics,
                   frozen_s=dict(no_grad=True,state_sha=sdigest),**selection.report()))
        guard.check()
        state='nonformal_complete' if a.nonformal else 'complete'
        names=['best.pt','latest.pt','initial.pt']+([] if a.nonformal else ['update500.pt','update1000.pt'])
        json_write(out/'training-closed.json',dict(status=state,run_id=run_id,completed_updates=completed,last_complete_validation=lastval,
                   partial_update=False,u0_reference_sha=REFERENCE_SHA if not a.nonformal else None,manifest_sha=file_hash(out/'manifest.json'),checkpoint_hashes={n:file_hash(out/n) for n in names}))
        status(state)
    except BaseException as e:
        if a.command=='train': save('interrupted.pt')
        if a.command=='train' or final_claimed:
            status('stopped' if isinstance(e,Stopped) else 'failed',f'{type(e).__name__}: {e}')
        raise
    finally:
        for sig,h in handlers.items(): signal.signal(sig,h)


if __name__=='__main__': main()
