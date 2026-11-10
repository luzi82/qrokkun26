"""Explicit read-only frozen checkpoint fork; no discovery or fallback."""
import json
from pathlib import Path
import shutil
import torch
from .core import file_hash, PLAYER_SHA, STATE_SHA

FILES=('initial.pt','latest.pt','best.pt','validation-controls.json','manifest.json','progress.jsonl')

def read_snapshot(path, config, cap):
    root=Path(path).resolve()
    if any(p.name.startswith('final') for p in root.iterdir()):
        raise ValueError('parent final claim/access forbidden')
    provenance=json.loads((root/'provenance.json').read_text())
    original=Path(provenance.get('run_path',''))
    if not original.is_absolute() or str(original.resolve())!=provenance.get('run_path'):
        raise ValueError('provenance.run_path must be absolute and resolved')
    if not provenance.get('run_path') or not str(provenance.get('source_commit','')).startswith('c4e322b'):
        raise ValueError('expected original immutable c4e322b source lineage')
    if provenance.get('heldout_access') is not False or provenance.get('frozen') is not True:
        raise ValueError('frozen snapshot and no heldout access attestation required')
    if provenance.get('core_sha256')!=file_hash(Path(__file__).with_name('core.py')):
        raise ValueError('architecture/mapping source changed')
    hashes={name:file_hash(root/name) for name in FILES}
    if provenance.get('hashes')!=hashes: raise ValueError('snapshot hash mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    ignored={'command','out_dir','deadline_unix','train_deadline_unix','resume_from_run'}
    old=manifest['config']
    if {k:v for k,v in old.items() if k not in ignored}!={k:v for k,v in config.items() if k not in ignored}:
        raise ValueError('incompatible continuation config')
    if config['seed']!=17 or manifest['cap_frames']!=cap or manifest['player_state']!=STATE_SHA or manifest['protocol']!=1:
        raise ValueError('incompatible protocol')
    if not old['deadline_unix']<=config['deadline_unix'] or not old['train_deadline_unix']<=config['train_deadline_unix']:
        raise ValueError('only deadline extension allowed')
    def no_heldout(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if 'final' in k.lower() and k not in ('final_seeds',):
                    if v: raise ValueError('heldout claim in snapshot')
                no_heldout(v)
        elif isinstance(obj,(list,tuple)):
            for v in obj: no_heldout(v)
        elif isinstance(obj,int) and 500000<=obj<500064:
            raise ValueError('heldout access in snapshot')
    controls=json.loads((root/'validation-controls.json').read_text())
    no_heldout(controls)
    for line in (root/'progress.jsonl').read_text().splitlines(): no_heldout(json.loads(line))
    checkpoints={name:torch.load(root/name,map_location='cpu',weights_only=False) for name in ('initial.pt','latest.pt','best.pt')}
    latest=checkpoints['latest.pt']
    if latest.get('partial_update') is not False or latest['update']!=latest['attempted_update']:
        raise ValueError('partial update snapshot forbidden')
    if not 1<=latest['update']<=config['updates']<=100000: raise ValueError('invalid update bounds')
    if latest['update']==1 or latest['update']%config['eval_every']==0:
        import math
        val=latest.get('metrics',{}).get('validation') or {}
        if ([r.get('seed') for r in val.get('per_seed',[])]!=list(range(400000,400000+config['eval_seeds']))
                or not isinstance(val.get('mean_seconds'),(int,float)) or not math.isfinite(val['mean_seconds'])):
            raise ValueError('scheduled validation pending/incomplete; capture complete validation or nonboundary update')
    for name,checkpoint in checkpoints.items():
        if checkpoint['player_sha']!=PLAYER_SHA or checkpoint['player_state']!=STATE_SHA or checkpoint['manifest']!=manifest:
            raise ValueError('wrong P or checkpoint manifest')
        no_heldout(checkpoint.get('metrics',{}))
        if checkpoint['update']>latest['update']: raise ValueError('inconsistent snapshot update')
    if checkpoints['initial.pt']['update']!=0: raise ValueError('initial must be update zero')
    best=checkpoints['best.pt']['metrics']['validation']['mean_seconds']
    for mode in ('scripted','stochastic'):
        if [r['seed'] for r in controls[mode]['per_seed']]!=list(range(400000,400000+config['eval_seeds'])):
            raise ValueError('invalid validation baselines')
    return dict(root=root,checkpoints=checkpoints,best=best,controls=controls,
                lineage=dict(parent_snapshot=str(root),**provenance,resume_update=latest['update']))

def restore(snapshot,net,initial,opt,rng):
    cp=snapshot['checkpoints']; latest=cp['latest.pt']
    net.load_state_dict(latest['model'],strict=True)
    initial.load_state_dict(cp['initial.pt']['model'],strict=True)
    # Validate the inherited selected model against the same CPU architecture.
    import copy
    copy.deepcopy(net).load_state_dict(cp['best.pt']['model'],strict=True)
    expected=opt.state_dict()['param_groups']
    actual=latest['optimizer']['param_groups']
    if len(actual)!=len(expected) or any({k:v for k,v in x.items() if k!='params'}!={k:v for k,v in y.items() if k!='params'} for x,y in zip(actual,expected)):
        raise ValueError('incompatible Adam settings')
    ids=[i for group in actual for i in group['params']]
    states=latest['optimizer']['state']
    if set(states)!=set(ids): raise ValueError('missing Adam state; reset forbidden')
    for i,param in zip(ids,net.parameters()):
        state=states[i]
        if not {'step','exp_avg','exp_avg_sq'}<=state.keys(): raise ValueError('incomplete Adam state')
        if any(state[k].shape!=param.shape or not torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('invalid Adam moments')
        if not torch.isfinite(state['step']).all() or state['step'].item()<=0: raise ValueError('invalid Adam step')
    opt.load_state_dict(latest['optimizer'])
    rng.set_state(latest['train_rng'])
    import numpy as np
    np.random.set_state(latest['numpy_rng'])
    # LAST: all constructors (including validation copies) precede RNG restoration.
    torch.set_rng_state(latest['torch_rng'])

def copy_controls(snapshot,out):
    for name in ('initial.pt','best.pt','validation-controls.json'):
        shutil.copyfile(snapshot['root']/name,out/name)


def validate_output(path, out):
    """Read provenance only; reject both write roots before creating anything."""
    root=Path(path).resolve()
    provenance=json.loads((root/'provenance.json').read_text())
    original=Path(provenance.get('run_path',''))
    if not original.is_absolute() or str(original.resolve())!=provenance.get('run_path'):
        raise ValueError('provenance.run_path must be absolute and resolved')
    output=Path(out).resolve()
    launcher=output.with_name(output.name+'.launcher').resolve()
    if any(p.is_relative_to(r) for p in (output,launcher) for r in (root,original)):
        raise ValueError('output and launcher must be outside snapshot and original run')


def preflight(path, config):
    """CPU-only, read-only restoration rehearsal, preserving caller global RNG."""
    import copy
    import math
    import numpy as np
    from .core import Spawner
    from .run import HARD, TRAIN_HARD
    tr=torch.get_rng_state().clone(); nr=np.random.get_state()
    try:
        if not all(math.isfinite(config[k]) for k in ('deadline_unix','train_deadline_unix')) or not config['train_deadline_unix'] < config['deadline_unix'] <= HARD or config['train_deadline_unix']>TRAIN_HARD:
            raise ValueError('invalid deadlines')
        validate_output(path,config['out_dir'])
        snapshot=read_snapshot(path,config,120 if config['quick'] else 7200)
        net=Spawner(); initial=copy.deepcopy(net)
        opt=torch.optim.Adam(net.parameters(),lr=config['lr'])
        restore(snapshot,net,initial,opt,torch.Generator())
        return dict(update=snapshot['checkpoints']['latest.pt']['update'],best=snapshot['best'],
                    latest_sha256=file_hash(snapshot['root']/'latest.pt'),hashes=snapshot['lineage']['hashes'])
    finally:
        np.random.set_state(nr); torch.set_rng_state(tr)


def main():
    import argparse
    p=argparse.ArgumentParser(description='Read-only CPU continuation restoration preflight')
    p.add_argument('--snapshot',required=True)
    p.add_argument('--config-json',required=True,help='JSON file containing exact desired effective trainer vars')
    a=p.parse_args()
    print(json.dumps(preflight(a.snapshot,json.loads(Path(a.config_json).read_text())),sort_keys=True))

if __name__=='__main__': main()
